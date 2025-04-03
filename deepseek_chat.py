import os
import json
import numpy as np
import faiss
import datetime
import threading
import tiktoken

# ---------------------------
# Gemini Embedding Configuration
# ---------------------------
import google.generativeai as genai

# Hardcode the Gemini API key for testing purposes.
GEMINI_API_KEY = "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Configure the Gemini client globally.
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Chat API Client Configuration (DeepSeek)
# ---------------------------
import openai

# Hardcode DeepSeek API key for testing.
openai_api_key = "sk-5ec02f12083d4f97aad73e1adb3a6f48"
os.environ["DEEPSEEK_API_KEY"] = openai_api_key

openai.api_key = openai_api_key
client = openai.OpenAI(
    api_key=openai_api_key,
    base_url="https://api.deepseek.com"  # DeepSeek API endpoint
)

# ---------------------------
# Vector Database Setup
# ---------------------------
embedding_dim = 3072  # Adjust if Gemini returns a different dimension
index_file = "memory.index"
metadata_file = "memory_meta.json"
index_lock = threading.Lock()

# In-memory cache for embeddings to avoid repeated API calls.
embedding_cache = {}

if os.path.exists(index_file) and os.path.exists(metadata_file):
    try:
        with index_lock:
            index = faiss.read_index(index_file)
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        print(f"Loaded memory index with {len(metadata)} entries.")
    except Exception as e:
        print("Error loading index:", e)
        index = faiss.IndexFlatL2(embedding_dim)
        metadata = []
else:
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []
    print("Initialized new memory index.")

def save_index():
    try:
        with index_lock:
            faiss.write_index(index, index_file)
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        print(f"Saved memory index with {len(metadata)} entries.")
    except Exception as e:
        print("Error saving index:", e)

# ---------------------------
# Gemini Embedding Functions
# ---------------------------
def get_text_embedding(text):
    """
    Compute text embedding using Gemini's embedding model.
    Utilizes caching to avoid redundant API calls.
    """
    if text in embedding_cache:
        return embedding_cache[text]
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-exp-03-07",
            content=text
        )
        # Assume the embedding is returned under the key 'embedding'
        embedding = np.array(result['embedding'], dtype=np.float32)
        # Adjust embedding size if necessary.
        if embedding.shape[0] != embedding_dim:
            if embedding.shape[0] < embedding_dim:
                padded = np.zeros((embedding_dim,), dtype=np.float32)
                padded[:embedding.shape[0]] = embedding
                embedding = padded
            else:
                embedding = embedding[:embedding_dim]
        embedding_cache[text] = embedding
        return embedding
    except Exception as e:
        print("Error getting text embedding:", e)
        return np.zeros((embedding_dim,), dtype=np.float32)

def get_text_embeddings_batch(texts):
    """Process a batch of texts to obtain embeddings in one API call."""
    new_texts = [txt for txt in texts if txt not in embedding_cache]
    embeddings = {}
    if new_texts:
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-exp-03-07",
                content=new_texts
            )
            for i, txt in enumerate(new_texts):
                embeddings[txt] = np.array(result['embedding'][i], dtype=np.float32)
                embedding_cache[txt] = embeddings[txt]
        except Exception as e:
            print("Error during batch embedding:", e)
            for txt in new_texts:
                embeddings[txt] = np.zeros((embedding_dim,), dtype=np.float32)
    return [embedding_cache[txt] for txt in texts]

# ---------------------------
# Memory Management Functions with Additional Metadata
# ---------------------------
def add_to_memory(text, source="unknown", user_id=None, tags=None):
    """
    Add an entry to the persistent memory.
    Optional parameters:
      - user_id: to associate memory with a specific user.
      - tags: a list of tags (e.g., ["preference", "laptop"]).
    Only records if memory_enabled is True.
    """
    if not memory_enabled:
        return
    embedding = get_text_embedding(text)
    vec = embedding.reshape(1, -1)
    with index_lock:
        index.add(vec)
    entry = {
        "text": text,
        "source": source,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_id": user_id,
        "tags": tags or []
    }
    metadata.append(entry)
    save_index()
    print("Added memory:", entry)

def is_personal_preference_query(query):
    """
    Determine if the query indicates a personal preference inquiry.
    """
    keywords = ["fav", "favorite", "like", "prefer", "my preference"]
    return any(kw in query.lower() for kw in keywords)

def retrieve_from_memory(query, top_k=3):
    query_embedding = get_text_embedding(query).reshape(1, -1)
    with index_lock:
        distances, indices = index.search(query_embedding, top_k)
    results = []
    # Collect results along with recency and, if applicable, preference match bonus.
    for idx in indices[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            time_obj = datetime.datetime.fromisoformat(entry["timestamp"])
            age_seconds = (datetime.datetime.now(datetime.timezone.utc) - time_obj).total_seconds()
            # Base weight: newer entries get higher weight.
            weight = 1 / (1 + age_seconds/3600)
            # If the query seems to be about personal preferences and the entry has a "preference" tag, boost its weight.
            if is_personal_preference_query(query) and "preference" in entry.get("tags", []):
                weight *= 1.5
            results.append((entry, weight))
    results.sort(key=lambda x: x[1], reverse=True)
    return [entry for entry, _ in results]

# ---------------------------
# Accurate Token Counting with tiktoken
# ---------------------------
def count_tokens(text, model="text-davinci-003"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print("Error counting tokens:", e)
        return len(text) // 4

def approximate_history_token_count(messages, model="text-davinci-003"):
    return sum(count_tokens(msg["content"], model) for msg in messages)

# ---------------------------
# Advanced Prompt Management: Adaptive Summarization
# ---------------------------
MAX_HISTORY_TOKENS = 3000

def summarize_text(text):
    """
    Summarize the provided text using OpenAI's completion API.
    In production, consider using a local summarization model to reduce latency.
    """
    try:
        prompt = "Summarize the following conversation concisely:\n\n" + text
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5,
            n=1,
            stop=None,
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        print("Error during summarization:", e)
        return text

def prune_history(messages, model="text-davinci-003"):
    token_count = approximate_history_token_count(messages, model)
    if token_count <= MAX_HISTORY_TOKENS:
        return messages

    system_message = messages[0]
    recent_messages = messages[1:]
    tokens_to_remove = token_count - MAX_HISTORY_TOKENS

    accumulated = 0
    cutoff_index = 0
    for i, msg in enumerate(recent_messages):
        msg_tokens = count_tokens(msg["content"], model)
        accumulated += msg_tokens
        if accumulated >= tokens_to_remove:
            cutoff_index = i
            break

    if cutoff_index > 0:
        concatenated = "\n".join(msg["content"] for msg in recent_messages[:cutoff_index])
        summary = summarize_text(concatenated)
        pruned_history = [system_message, {"role": "summary", "content": summary}]
        pruned_history.extend(recent_messages[cutoff_index:])
        if approximate_history_token_count(pruned_history, model) > MAX_HISTORY_TOKENS:
            print("Warning: History still exceeds token limit after summarization.")
        return pruned_history
    else:
        return messages

# ---------------------------
# Memory Recording Control Flag
# ---------------------------
memory_enabled = True  # Global flag for memory recording.

def toggle_memory(state: bool):
    global memory_enabled
    memory_enabled = state
    status = "enabled" if state else "disabled"
    print(f"Memory recording has been {status}.")

# ---------------------------
# Chat Wrapper with Enhanced Features
# ---------------------------
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Interactive DeepSeek Chat with Enhanced Memory & Prompt Management (type 'exit' to quit)")
print("Type 'stop recording memory' to disable memory logging, and 'resume recording memory' to enable it.")
print("------------------------------------------------------------")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("\nGoodbye!")
        break

    # Check for memory control commands.
    if user_input.lower() == "stop recording memory":
        toggle_memory(False)
        continue
    elif user_input.lower() == "resume recording memory":
        toggle_memory(True)
        continue

    messages.append({"role": "user", "content": user_input})
    add_to_memory("User: " + user_input, source="user", user_id="test_user", tags=["preference"] if "macbook" in user_input.lower() else [])

    # For personal preference queries, reformulate the query to be more explicit.
    if is_personal_preference_query(user_input):
        reformulated_query = user_input + " (based on my preferences)"
    else:
        reformulated_query = user_input

    retrieved_entries = retrieve_from_memory(reformulated_query, top_k=3)
    retrieved_context = " ".join(entry["text"] for entry in retrieved_entries)
    
    augmented_input = retrieved_context + " " + user_input
    messages[-1]["content"] = augmented_input

    messages = prune_history(messages)

    try:
        print("\nDeepSeek: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        
        assistant_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_chunk = chunk.choices[0].delta.content
                assistant_response += content_chunk
                print(content_chunk, end="", flush=True)
        print()
        
        messages.append({"role": "assistant", "content": assistant_response})
        add_to_memory("Assistant: " + assistant_response, source="assistant", user_id="test_user")
        
    except Exception as e:
        print(f"\nError during API call: {str(e)}")
