import os
import json
import numpy as np
import faiss
import datetime
import threading
import tiktoken
import re
from transformers import pipeline, BartTokenizer
from thefuzz import fuzz
import google.generativeai as genai
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Iterable, List, Optional, Union
from starlette.responses import StreamingResponse, JSONResponse
#from deepseek_chat import chat_with_memory
import time
import json
# For fuzzy tag matching
from thefuzz import fuzz
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ---------------------------
# Gemini Embedding Configuration
# ---------------------------
import google.generativeai as genai

# Example key; do not hardcode in production
GEMINI_API_KEY = "your key here"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Chat API Client Configuration (DeepSeek)
# ---------------------------
import openai

# Example key; do not hardcode in production
openai_api_key = "your key here"
os.environ["DEEPSEEK_API_KEY"] = openai_api_key

openai.api_key = openai_api_key
client = openai.OpenAI(
    api_key=openai_api_key,
    base_url="https://api.deepseek.com"  # Adjust if needed
)

# ---------------------------
# Vector Database Setup
# ---------------------------
embedding_dim = 3072  # For gemini-embedding-exp-03-07
index_file = "memory.index"
metadata_file = "memory_meta.json"
index_lock = threading.Lock()

# Cache to avoid re-embedding repeated text
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
    """Persist the Faiss index and metadata to disk."""
    try:
        with index_lock:
            faiss.write_index(index, index_file)
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        print(f"Saved memory index with {len(metadata)} entries.")
    except Exception as e:
        print("Error saving index:", e)


# ---------------------------
# Basic Tag Extraction
# ---------------------------
def extract_tags(text):
    """
    Trivial example: split on alphanumeric, keep unique tokens of length >= 4 for 'tags'.
    In production, you might use a more robust approach 
    (keyword extraction, spaCy-based NER, etc.).
    """
    text_lower = text.lower()
    tokens = re.findall(r"\w+", text_lower)
    tokens = [t for t in tokens if len(t) >= 4]
    return list(set(tokens))


# ---------------------------
# Gemini Embedding Functions
# ---------------------------
def get_text_embedding(text):
    """
    Compute text embedding using Gemini's embedding model,
    optionally L2-normalize for better semantic search with Faiss (IndexFlatL2).
    """
    if text in embedding_cache:
        return embedding_cache[text]

    try:
        result = genai.embed_content(
            model="models/gemini-embedding-exp-03-07",
            content=text
        )
        embedding_array = np.array(result["embedding"], dtype=np.float32)

        # Optional L2 normalization
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm

        embedding_cache[text] = embedding_array
        return embedding_array

    except Exception as e:
        print("Error getting text embedding:", e)
        return np.zeros((embedding_dim,), dtype=np.float32)


# ---------------------------
# Memory Management Functions
# ---------------------------
memory_enabled = True

def toggle_memory(state: bool):
    global memory_enabled
    memory_enabled = state
    status = "enabled" if state else "disabled"
    print(f"Memory recording has been {status}.")


def add_to_memory(text, source="unknown", user_id=None, tags=None):
    """
    Add an entry to the persistent memory (Faiss index + metadata).
    We also auto-extract tags from `text` if not provided.
    """
    if not memory_enabled:
        return

    embedding = get_text_embedding(text)
    vec = embedding.reshape(1, -1)
    with index_lock:
        index.add(vec)

    auto_tags = extract_tags(text)
    combined_tags = list(set(auto_tags + (tags or [])))

    entry = {
        "text": text,
        "source": source,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_id": user_id,
        "tags": combined_tags
    }
    metadata.append(entry)
    save_index()
    print("Added memory:", entry)


def is_personal_preference_query(query):
    """
    Detect a personal preference inquiry to apply a weighting boost.
    """
    keywords = ["fav", "favorite", "like", "prefer", "my preference"]
    q_lower = query.lower()
    return any(kw in q_lower for kw in keywords)


# ---------------------------
# Fuzzy Overlap Helper
# ---------------------------
def fuzzy_tag_overlap(entry_tags, query_tags, threshold=80):
    """
    Return how many 'fuzzy-matched' tags we find between two sets of tags.
    E.g. if entry has ["macbook"] and query has ["macbok"], 
    ratio might be >= 80 -> counts as match.
    """
    overlap_count = 0
    for qtag in query_tags:
        for etag in entry_tags:
            score = fuzz.ratio(qtag, etag)
            if score >= threshold:
                overlap_count += 1
                break
    return overlap_count


def retrieve_from_memory(query, top_k=None):
    """
    Perform a semantic search over ALL stored memory.
    If top_k is None, we retrieve everything from the index,
    then re-rank by distance, recency, preference, plus fuzzy tag overlap.
    """
    if len(metadata) == 0:
        return []

    query_embedding = get_text_embedding(query).reshape(1, -1)
    with index_lock:
        k = top_k if top_k is not None else index.ntotal
        distances, indices = index.search(query_embedding, k)

    query_tags = extract_tags(query)
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            distance = distances[0][i]
            # Convert L2 distance to a "similarity" style
            base_score = 1.0 / (1.0 + distance)

            time_obj = datetime.datetime.fromisoformat(entry["timestamp"])
            age_seconds = (datetime.datetime.now(datetime.timezone.utc) - time_obj).total_seconds()
            recency_weight = 1 / (1 + (age_seconds / 3600.0))

            preference_boost = 1.0
            if is_personal_preference_query(query) and ("preference" in entry["tags"]):
                preference_boost = 1.5

            overlap_count = fuzzy_tag_overlap(entry["tags"], query_tags, threshold=80)
            tag_boost = 1.0 + 0.1 * overlap_count

            final_score = base_score * recency_weight * preference_boost * tag_boost
            results.append((entry, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results]


# ---------------------------
# Token Counting & Summaries
# ---------------------------
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def count_tokens(text, model="facebook/bart-large-cnn"):
    try:
        if model == "facebook/bart-large-cnn":
            tokens = bart_tokenizer.tokenize(text)
            return len(tokens)
        else:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
    except Exception as e:
        print("Error counting tokens:", e)
        return len(text.split())


def approximate_history_token_count(messages, model="facebook/bart-large-cnn"):
    return sum(count_tokens(msg["content"], model) for msg in messages)


MAX_HISTORY_TOKENS = 3000

# Summarization pipeline on GPU (device=0).
bart_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer=bart_tokenizer,
    device=0,             # keep on GPU
    max_length=150,       # max output length for summary
    truncation=True
)

# ---------------------------
# CHUNKING-BASED SUMMARIZATION
# ---------------------------
def chunk_text(text, tokenizer, chunk_size=512): # 1024
    """
    Split text into multiple chunks of up to `chunk_size` tokens each
    so we don't overflow the model's input limit.
    """
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(all_ids), chunk_size):
        chunks.append(all_ids[i : i + chunk_size])
    return chunks


def summarize_chunk(token_ids, tokenizer, summarizer):
    """
    Summarize a single chunk of token IDs.
    """
    chunk_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    # Summarize the chunk
    out = summarizer(chunk_text, do_sample=False, min_length=40)
    return out[0]["summary_text"].strip()

def process_in_chunks(
    items,
    chunk_size,
    process_fn,
    on_error=None,
    max_retries=2,
    delay=1,
    verbose=True
):
    """
    General utility to process items in chunks with error handling.
    
    Args:
        items (list): List of items to process.
        chunk_size (int): Max chunk size.
        process_fn (callable): Function to apply to each chunk.
        on_error (callable): Fallback function on failure (e.g., lambda chunk: chunk).
        max_retries (int): Number of retries per chunk.
        delay (int): Seconds to wait between retries.
        verbose (bool): Print debug messages.
    
    Returns:
        List of processed results.
    """
    results = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        attempt = 0
        while attempt <= max_retries:
            try:
                result = process_fn(chunk)
                results.append(result)
                break
            except Exception as e:
                attempt += 1
                if verbose:
                    print(f"Error processing chunk {i // chunk_size + 1}: {e}")
                if attempt > max_retries:
                    fallback = on_error(chunk) if on_error else chunk
                    results.append(fallback)
                else:
                    time.sleep(delay * attempt)
    return results


def summarize_chunks(chunk_texts):
    return [bart_summarizer(text, do_sample=False, min_length=30, max_length=120)[0]["summary_text"].strip() for text in chunk_texts]

def chunked_summarize(text, chunk_size=512, pass_count=2):
    for _ in range(pass_count):
        token_ids = bart_tokenizer.encode(text, add_special_tokens=False)
        chunks = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size)]
        chunk_texts = [bart_tokenizer.decode(ids, skip_special_tokens=True) for ids in chunks]

        partial_summaries = process_in_chunks(
            chunk_texts,
            chunk_size=1,  # process each summary individually
            process_fn=summarize_chunks,
            on_error=lambda c: c,  # fallback: keep raw chunk text
            max_retries=1,
            verbose=True
        )

        text = "\n".join(flatten(partial_summaries))
        if len(chunks) == 1:
            break

    return text

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in (sublist if isinstance(sublist, list) else [sublist])]

def prune_history(messages, model="facebook/bart-large-cnn"):
    """
    If chat history is too large, summarize older chunks.
    Uses chunked_summarize() to handle very large text.
    The summary is inserted into the first user message after the cutoff to maintain role alternation.
    """
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
        # Summarize everything up to cutoff_index with chunked summarization
        concatenated = "\n".join(msg["content"] for msg in recent_messages[:cutoff_index])
        summary = chunked_summarize(concatenated)

        updated_messages = recent_messages[cutoff_index:]

        # Insert summary into the first user message after the cutoff
        for i, msg in enumerate(updated_messages):
            if msg["role"] == "user":
                msg["content"] = f"(Summary of earlier conversation)\n{summary}\n\n{msg['content']}"
                break
        else:
            # If no user message found, fallback: insert summary as assistant
            updated_messages.insert(0, {
                "role": "assistant",
                "content": f"(Summary of earlier conversation)\n{summary}"
            })

        pruned_history = [system_message] + updated_messages

        if approximate_history_token_count(pruned_history, model) > MAX_HISTORY_TOKENS:
            print("Warning: History still exceeds token limit after summarization.")

        return pruned_history

    else:
        return messages



from fastapi import FastAPI
from pydantic import BaseModel
from typing import Iterable, List, Optional, Union
from starlette.responses import StreamingResponse, JSONResponse
import time
import json

# -----------------------------
# OpenAI-compatible Request Format
# -----------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None


# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    messages = request.messages
    user_input = " ".join(m["content"] for m in messages if m["role"] == "user")
    
    
    custom_tags = []

    add_to_memory(user_input, source="user", user_id="test_user", tags=custom_tags)

    # If user is asking about personal preferences, reformulate slightly
    if is_personal_preference_query(user_input):
        reformulated_query = user_input + " (based on my preferences)"
    else:
        reformulated_query = user_input

    # Retrieve memory (all)
    all_entries = retrieve_from_memory(reformulated_query, top_k=None)
    top_entries = all_entries[:5]

    # Combine top matches into snippet
    retrieved_context = "\n".join(
        f"Relevant Memory: {entry['text']} [tags={entry['tags']}]"
        for entry in top_entries
    )

    # Place memory context + user message into the conversation
    augmented_input = f"{retrieved_context}\n\nUser's current message: {user_input}"
    for i in reversed(range(len(messages))):
        if messages[i]["role"] == "user":
            messages[i]["content"] = augmented_input
            break

    # Prune conversation with chunked summarization
    messages = prune_history(messages)
    print(messages)
    # Send to DeepSeek for streaming chat completion
    assistant_response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p
    )

    # Record assistant response in memory as well
    messages.append({"role": "assistant", "content": assistant_response.choices[0].message.content})
    add_to_memory(assistant_response.choices[0].message.content, source="assistant", user_id="test_user")
    

    return assistant_response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deepseek_chat:app", host="127.0.0.1", port=8000)
