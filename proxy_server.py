from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os
import json
import numpy as np
import faiss
import datetime
import threading
import tiktoken
import re
import time
from transformers import pipeline, BartTokenizer
from thefuzz import fuzz
import google.generativeai as genai
import openai
import asyncio
from starlette.responses import StreamingResponse

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
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class MemoryEntry(BaseModel):
    text: str
    source: str = "unknown"
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None

class MemorySearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    user_id: Optional[str] = None

class MemoryToggleRequest(BaseModel):
    enabled: bool

# ---------------------------
# Gemini Embedding Configuration
# ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA")
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Chat API Client Configuration (DeepSeek)
# ---------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-5ec02f12083d4f97aad73e1adb3a6f48")
deepseek_client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
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
    Extract unique tokens of length >= 4 from text for 'tags'.
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

        # L2 normalization
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

def add_to_memory(text, source="unknown", user_id=None, tags=None):
    """
    Add an entry to the persistent memory (Faiss index + metadata).
    Auto-extracts tags from `text` if not provided.
    """
    if not memory_enabled:
        return {"status": "skipped", "reason": "memory recording disabled"}

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
    
    return {"status": "success", "entry_id": len(metadata) - 1}

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

def retrieve_from_memory(query, top_k=None, user_id=None):
    """
    Perform a semantic search over stored memory.
    """
    if len(metadata) == 0:
        return []

    query_embedding = get_text_embedding(query).reshape(1, -1)
    with index_lock:
        k = top_k if top_k is not None else index.ntotal
        k = min(k, index.ntotal)  # Make sure k is not larger than the index size
        distances, indices = index.search(query_embedding, k)

    query_tags = extract_tags(query)
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            
            # Filter by user_id if provided
            if user_id is not None and entry.get("user_id") != user_id:
                continue
                
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
    return sum(count_tokens(msg.content, model) for msg in messages)

MAX_HISTORY_TOKENS = 3000

# Initialize summarization pipeline 
bart_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer=bart_tokenizer,
    device=-1,  # CPU
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
app = FastAPI(title="Memory-Enhanced LLM API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# API Routes
# ---------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completion endpoint that enhances requests with memory context.
    """
    # Convert Pydantic models to dict for API compatibility
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Get the user message (last one if multiple)
    user_messages = [msg for msg in messages_dict if msg["role"] == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in the request")
    
    user_input = user_messages[-1]["content"]
    
    # Extract tags from the user message and add to memory
    custom_tags = []
    if "macbook" in user_input.lower() or "macbooks" in user_input.lower():
        custom_tags.append("preference")
    
    add_to_memory(user_input, source="user", user_id="api_user", tags=custom_tags)
    
    # Retrieve relevant memory
    reformulated_query = user_input
    if is_personal_preference_query(user_input):
        reformulated_query += " (based on my preferences)"
    
    retrieved_entries = retrieve_from_memory(reformulated_query, top_k=5)
    
    # Format retrieved context
    memory_context = "\n".join(
        f"Relevant Memory: {entry['text']} [tags={entry['tags']}]"
        for entry in retrieved_entries
    )
    
    # Augment the user's last message with memory context
    if memory_context:
        messages_dict[-1]["content"] = f"{memory_context}\n\nUser's current message: {user_input}"
    
    # Prune history if needed
    pydantic_messages = [Message(**msg) for msg in messages_dict]
    pruned_messages = prune_history(pydantic_messages)
    pruned_dict = [{"role": msg.role, "content": msg.content} for msg in pruned_messages]
    
    # Handle streaming vs. non-streaming differently
    if request.stream:
        async def generate():
            stream = deepseek_client.chat.completions.create(
                model=request.model,
                messages=pruned_dict,
                stream=True,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            assistant_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    assistant_response += piece
                    
                    # Convert to the expected format
                    chunk_json = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    yield f"data: {json.dumps(chunk_json)}\n\n"
            
            # Record the full response in memory
            add_to_memory(assistant_response, source="assistant", user_id="api_user")
            
            # Send the completion message
            yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Non-streaming response
        completion = deepseek_client.chat.completions.create(
            model=request.model,
            messages=pruned_dict,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Record the assistant's response in memory
        assistant_response = completion.choices[0].message.content
        add_to_memory(assistant_response, source="assistant", user_id="api_user")
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
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

@app.post("/memory/add")
async def add_memory(entry: MemoryEntry):
    """Add a new entry to the memory store."""
    result = add_to_memory(
        text=entry.text,
        source=entry.source,
        user_id=entry.user_id,
        tags=entry.tags
    )
    return result

@app.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search the memory store for relevant entries."""
    results = retrieve_from_memory(
        query=request.query,
        top_k=request.top_k,
        user_id=request.user_id
    )
    return {"results": results}

@app.post("/memory/toggle")
async def memory_toggle(request: MemoryToggleRequest):
    """Enable or disable memory recording."""
    return toggle_memory(request.enabled)

@app.get("/memory/status")
async def memory_status():
    """Get the current status of the memory system."""
    return {
        "enabled": memory_enabled,
        "entries_count": len(metadata),
        "index_size": index.ntotal if hasattr(index, "ntotal") else 0
    }

# ---------------------------
# Startup and Shutdown Events
# ---------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("Memory-Enhanced LLM API is starting up...")
    # Initialize any additional resources here

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("Saving memory index before shutdown...")
    save_index()

# ---------------------------
# Server Start
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("proxy_server:app", host="0.0.0.0", port=8000, reload=True)

