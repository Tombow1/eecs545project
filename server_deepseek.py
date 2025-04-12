from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
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
from openai import OpenAI

# ---------------------------
# Schema Definitions
# ---------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    
class MemoryEntry(BaseModel):
    text: str
    source: str = "unknown"
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None

class MemoryQuery(BaseModel):
    query: str
    top_k: Optional[int] = None
    
class MemoryToggle(BaseModel):
    enabled: bool

# ---------------------------
# Gemini Embedding Configuration
# ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA")
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Chat API Client Configuration (DeepSeek)
# ---------------------------
openai_api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-5ec02f12083d4f97aad73e1adb3a6f48")
deepseek_client = OpenAI(
    api_key=openai_api_key,
    base_url="https://api.deepseek.com"
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

# Initialize or load the vector index
def initialize_index():
    global index, metadata
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
    return index, metadata

index, metadata = initialize_index()

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
    Extract tags from text. Keeps unique tokens of length >= 4.
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
    with L2-normalization for better semantic search.
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
    return {"status": status}

def add_to_memory(entry: MemoryEntry, background_tasks: BackgroundTasks):
    """
    Add an entry to the persistent memory (Faiss index + metadata).
    We also auto-extract tags from `text` if not provided.
    """
    if not memory_enabled:
        return {"status": "skipped", "reason": "Memory recording is disabled"}
    
    def _add_memory_task(entry):
        text = entry.text
        embedding = get_text_embedding(text)
        vec = embedding.reshape(1, -1)
        with index_lock:
            index.add(vec)

        auto_tags = extract_tags(text)
        combined_tags = list(set(auto_tags + (entry.tags or [])))

        memory_entry = {
            "text": text,
            "source": entry.source,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_id": entry.user_id,
            "tags": combined_tags
        }
        metadata.append(memory_entry)
        save_index()
        print("Added memory:", memory_entry)
    
    # Add to background tasks to not block the API response
    background_tasks.add_task(_add_memory_task, entry)
    return {"status": "processing"}

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
    """
    overlap_count = 0
    for qtag in query_tags:
        for etag in entry_tags:
            score = fuzz.ratio(qtag, etag)
            if score >= threshold:
                overlap_count += 1
                break
    return overlap_count

def retrieve_from_memory(query: MemoryQuery):
    """
    Perform a semantic search over ALL stored memory.
    """
    if len(metadata) == 0:
        return []

    query_embedding = get_text_embedding(query.query).reshape(1, -1)
    with index_lock:
        k = query.top_k if query.top_k is not None else index.ntotal
        distances, indices = index.search(query_embedding, k)

    query_tags = extract_tags(query.query)
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
            if is_personal_preference_query(query.query) and ("preference" in entry["tags"]):
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
# Load BART tokenizer and summarizer
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
    return sum(count_tokens(msg.content, model) for msg in messages)

MAX_HISTORY_TOKENS = 3000

# Initialize BART summarizer
bart_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer=bart_tokenizer,
    device=0,  # GPU if available
    max_length=150,
    truncation=True
)

# ---------------------------
# CHUNKING-BASED SUMMARIZATION
# ---------------------------
def chunk_text(text, tokenizer, chunk_size=512):
    """
    Split text into multiple chunks of up to `chunk_size` tokens each.
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
    out = summarizer(chunk_text, do_sample=False, min_length=40)
    return out[0]["summary_text"].strip()

def chunked_summarize(text, chunk_size=512, pass_count=2):
    """
    Process text in chunks and summarize iteratively.
    """
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
        msg_tokens = count_tokens(msg.content, model)
        accumulated += msg_tokens
        if accumulated >= tokens_to_remove:
            cutoff_index = i
            break

    if cutoff_index > 0:
        # Summarize everything up to cutoff_index with chunked summarization
        concatenated = "\n".join(msg.content for msg in recent_messages[:cutoff_index])
        summary = chunked_summarize(concatenated)

        # We store the summary as an assistant message so we remain valid for the model
        pruned_history = [
            system_message,
            Message(role="assistant", content=summary)
        ]
        pruned_history.extend(recent_messages[cutoff_index:])

        if approximate_history_token_count(pruned_history, model) > MAX_HISTORY_TOKENS:
            print("Warning: History still exceeds token limit after summarization.")
        return pruned_history
    else:
        return messages

# ---------------------------
# FastAPI App Setup
# ---------------------------
app = FastAPI(title="Memory-Augmented AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# API Routes
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Memory-Augmented AI API is running"}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    background_tasks: BackgroundTasks
):
    """
    Proxy endpoint for chat completions, with memory augmentation.
    """
    # Convert Pydantic models to dicts for processing
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Extract user's message (the last one where role is 'user')
    user_messages = [msg for msg in messages_dict if msg["role"] == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in the request")
    
    user_input = user_messages[-1]["content"]
    
    # Add the user's raw text to memory
    custom_tags = []
    if "macbook" in user_input.lower() or "macbooks" in user_input.lower():
        custom_tags.append("preference")
    
    memory_entry = MemoryEntry(
        text=user_input,
        source="user",
        user_id="api_user",  # You might want to get this from auth
        tags=custom_tags
    )
    add_to_memory(memory_entry, background_tasks)
    
    # If user is asking about personal preferences, reformulate slightly
    if is_personal_preference_query(user_input):
        reformulated_query = user_input + " (based on my preferences)"
    else:
        reformulated_query = user_input
    
    # Retrieve memory
    memory_query = MemoryQuery(query=reformulated_query, top_k=5)
    top_entries = retrieve_from_memory(memory_query)
    
    # Combine top matches into context
    retrieved_context = "\n".join(
        f"Relevant Memory: {entry['text']} [tags={entry['tags']}]"
        for entry in top_entries
    )
    
    # Augment the last user message with memory context
    augmented_input = f"{retrieved_context}\n\nUser's current message: {user_input}"
    
    # Replace the last user message content
    for i in reversed(range(len(messages_dict))):
        if messages_dict[i]["role"] == "user":
            messages_dict[i]["content"] = augmented_input
            break
    
    # Convert to Message objects for pruning
    message_objects = [Message(**msg) for msg in messages_dict]
    
    # Prune conversation history if needed
    pruned_messages = prune_history(message_objects)
    
    # Convert back to dict for API call
    pruned_messages_dict = [{"role": msg.role, "content": msg.content} for msg in pruned_messages]
    
    # Call DeepSeek API
    try:
        completion = deepseek_client.chat.completions.create(
            model=request.model,
            messages=pruned_messages_dict,
            stream=request.stream,
        )
        
        # If streaming is enabled, we need to handle it differently
        if request.stream:
            # This would require StreamingResponse in FastAPI
            # For simplicity, disabling streaming in this example
            raise HTTPException(
                status_code=400, 
                detail="Streaming is not supported in this example"
            )
        
        # Record assistant response in memory
        assistant_response = completion.choices[0].message.content
        assistant_memory = MemoryEntry(
            text=assistant_response,
            source="assistant",
            user_id="system"
        )
        add_to_memory(assistant_memory, background_tasks)
        
        # Return response
        return {
            "id": "chatcmpl-memory-augmented",
            "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_response
                    },
                    "finish_reason": completion.choices[0].finish_reason,
                }
            ],
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during API call: {str(e)}")

@app.post("/memory/add")
async def add_memory_entry(entry: MemoryEntry, background_tasks: BackgroundTasks):
    """
    Add a new entry to the memory system.
    """
    result = add_to_memory(entry, background_tasks)
    return result

@app.post("/memory/retrieve")
async def retrieve_memory(query: MemoryQuery):
    """
    Retrieve relevant memories based on a query.
    """
    results = retrieve_from_memory(query)
    return {"results": results}

@app.post("/memory/toggle")
async def toggle_memory_recording(toggle: MemoryToggle):
    """
    Enable or disable memory recording.
    """
    return toggle_memory(toggle.enabled)

# ---------------------------
# Main Entrypoint
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
