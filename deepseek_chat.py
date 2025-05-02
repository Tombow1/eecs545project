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
import torch


# For fuzzy tag matching
from thefuzz import fuzz

# ---------------------------
# Gemini Embedding Configuration
# ---------------------------
import google.generativeai as genai
# ───────────────────────────────────────────────────────────────────────────────
# Multimodal Helpers  —  OLLAMA PYTHON CLIENT (llama3.2‑vision)
# ───────────────────────────────────────────────────────────────────────────────
import ollama                                                         # NEW

OLLAMA_MODEL = "llama3.2-vision:90b"      # ← EXACT tag shown by `ollama list`
                                      #   (add :Q4_K_M or :90b if that's what you have)

def describe_image_with_llama(path: str,
                              prompt: str = "Describe this image in detail.") -> str:
    """
    Call the vision‑capable Llama 3.2 model via ollama.chat() and return the text.
    """
    messages = [
        {
            "role":    "user",
            "content": prompt,
            "images":  [path]          # Ollama client will load & encode the file
        }
    ]
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        # `response` is a dict like: {'message': {'role': 'assistant', 'content': '...'}}
        return response.get("message", {}).get("content", str(response)).strip()
    except Exception as exc:
        return f"[ERROR contacting Ollama] {exc}"

def add_image_to_memory(path: str,
                        prompt: str = "Describe this image in detail.",
                        user_id: str = "gui_user") -> str:
    """
    Convenience: run vision model, add result to Faiss memory, return text.
    """
    description = describe_image_with_llama(path, prompt)
    add_to_memory(description, source="image", user_id=user_id, tags=["image_analysis"])
    return description

# ─────────────────────────────────────────────────────────────────────
# 7)  AUDIO  → text  (HF Whisper‑large‑v3)
# ─────────────────────────────────────────────────────────────────────
HF_MODEL_ID = "openai/whisper-large-v3"
_whisper_pipe = None


def _get_whisper_pipe():
    global _whisper_pipe
    if _whisper_pipe is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = pipeline(
            "automatic-speech-recognition",
            model=HF_MODEL_ID,
            torch_dtype=dtype,
            device=0 if torch.cuda.is_available() else -1
        )
        _whisper_pipe = model
    return _whisper_pipe


def transcribe_audio(path: str) -> str:
    try:
        pipe = _get_whisper_pipe()
        out = pipe(path, return_timestamps=True)
        return out["text"].strip()
    except Exception as e:
        return f"[ERROR transcribing audio] {e}"


def add_audio_to_memory(path: str):
    txt = transcribe_audio(path)
    add_to_memory(txt, source="audio", tags=["audio_transcript"])
    return txt


# ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "your api key here"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Chat API Client Configuration (DeepSeek)
# ---------------------------
import openai

openai_api_key = "your api key ehre"
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


def chunked_summarize(text, chunk_size=512, pass_count=2):
    # Each pass chunk-summarizes the text, then feeds the combined summary
    # into the next pass if there's more than 1 chunk.

    for _ in range(pass_count):
        token_ids = bart_tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            token_ids[i : i + chunk_size]
            for i in range(0, len(token_ids), chunk_size)
        ]
        partial_summaries = []
        for c in chunks:
            c_text = bart_tokenizer.decode(c, skip_special_tokens=True)
            try:
                out = bart_summarizer(c_text, do_sample=False, min_length=30, max_length=120)
                partial_summaries.append(out[0]["summary_text"].strip())
            except Exception as e:
                print("Error summarizing chunk:", e)
                partial_summaries.append(c_text)  # fallback: keep raw text if chunk fails

        text = "\n".join(partial_summaries)

        # if there's only one chunk, we've effectively done final summarization
        if len(chunks) == 1:
            break

    return text


def prune_history(messages, model="facebook/bart-large-cnn"):
    """
    If chat history is too large, summarize older chunks.
    Uses chunked_summarize() to handle very large text.
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

        # We store the summary as an assistant message so we remain valid for the model
        pruned_history = [
            system_message,
            {"role": "assistant", "content": summary}
        ]
        pruned_history.extend(recent_messages[cutoff_index:])

        if approximate_history_token_count(pruned_history, model) > MAX_HISTORY_TOKENS:
            print("Warning: History still exceeds token limit after summarization.")
        return pruned_history
    else:
        return messages


# ---------------------------
# Main Chat Loop
# ---------------------------
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant with access to a vector-memory system. "
            "Use the retrieved memory context if relevant."
        )
    }
]

def cli_loop():
    print("Interactive DeepSeek Chat with Hybrid Fuzzy Tag + Vector Search & Chunked Summaries (type 'exit' to quit)")
    print("Type 'stop recording memory' to disable memory logging, and 'resume recording memory' to enable it.")
    print("------------------------------------------------------------")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break

        if user_input.lower() == "stop recording memory":
            toggle_memory(False)
            continue
        elif user_input.lower() == "resume recording memory":
            toggle_memory(True)
            continue

        # Add the user's raw text to memory (with auto-tagging)
        custom_tags = []
        # Example: if "macbook" in user_input => custom_tags = ["preference"]
        if "macbook" in user_input.lower() or "macbooks" in user_input.lower():
            custom_tags.append("preference")

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
        messages.append({"role": "user", "content": augmented_input})

        # Prune conversation with chunked summarization
        messages = prune_history(messages)

        # Send to DeepSeek for streaming chat completion
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
                    piece = chunk.choices[0].delta.content
                    assistant_response += piece
                    print(piece, end="", flush=True)
            print()

            # Record assistant response in memory as well
            messages.append({"role": "assistant", "content": assistant_response})
            add_to_memory(assistant_response, source="assistant", user_id="test_user")

        except Exception as e:
            print(f"\nError during API call: {str(e)}")
        
# Only run the CLI chat loop if this file is executed directly
if __name__ == "__main__":
    cli_loop()