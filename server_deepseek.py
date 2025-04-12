from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
from openai import OpenAI
import asyncio
import json
import os
import numpy as np
import faiss
import datetime
import threading
import tiktoken
import re
from transformers import pipeline, BartTokenizer
from thefuzz import fuzz
import google.generativeai as genai

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
    user: Optional[str] = None
    memory_enabled: Optional[bool] = True
    
# ---------------------------
# Gemini Embedding Configuration
# ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA")
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Chat API Client Configuration (DeepSeek)
# ---------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-5ec02f12083d4f97aad73e1adb3a6f48")
deepseek_client = OpenAI(
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
    Extract tags from text.
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
    Compute text embedding using Gemini's embedding model.
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
def is_personal_preference_query(query):
    """
    Detect a personal preference inquiry to apply a weighting boost.
    """
    keywords = ["fav", "favorite", "like", "prefer", "my preference"]
    q_lower = query.lower()
    return any(kw in q_lower for kw in keywords)


def add_to_memory(text, source="unknown", user_id=None, tags=None):
    """
    Add an entry to the persistent memory (Faiss index + metadata).
    We also auto-extract tags from `text` if not provided.
    """
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
    print(f"Added memory for user {user_id}: {entry['text'][:30]}...")


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


def retrieve_from_memory(query, user_id=None, top_k=5):
    """
    Perform a semantic search over stored memory.
    """
    if len(metadata) == 0:
        return []

    query_embedding = get_text_embedding(query).reshape(1, -1)
    with index_lock:
        k = min(top_k, index.ntotal) if top_k is not None else index.ntotal
        k = max(k, 1)  # Ensure k is at least 1
        distances, indices = index.search(query_embedding, k)

    query_tags = extract_tags(query)
    results = []
    
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            
            # Skip entries not belonging to this user if user_id is provided
            if user_id and entry.get("user_id") != user_id:
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

# Summarization pipeline
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
def chunk_text(text, tokenizer, chunk_size=512):
    """
    Split text into multiple chunks of up to `chunk_size` tokens each.
    """
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(all_ids), chunk_size):
        chunks.append(all_ids[i : i + chunk_size])
    return chunks


def chunked_summarize(text, chunk_size=512, pass_count=2):
    """
    Summarize text using a chunking approach for long texts.
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
# FastAPI App
# ---------------------------
app = FastAPI(title="Memory-Enhanced DeepSeek API")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Handle chat completions with memory enhancement.
    """
    # Extract the user's message (assume it's the last one)
    if not request.messages:
        return {"error": "No messages provided"}
    
    user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
    if not user_message:
        return {"error": "No user message found"}
    
    user_id = request.user or "default_user"
    memory_enabled = request.memory_enabled
    
    # Process with memory if enabled
    if memory_enabled:
        # Add user message to memory
        custom_tags = []
        if "macbook" in user_message.content.lower() or "macbooks" in user_message.content.lower():
            custom_tags.append("preference")
            
        add_to_memory(user_message.content, source="user", user_id=user_id, tags=custom_tags)
        
        # Retrieve relevant memories
        if is_personal_preference_query(user_message.content):
            reformulated_query = user_message.content + " (based on my preferences)"
        else:
            reformulated_query = user_message.content
            
        top_entries = retrieve_from_memory(reformulated_query, user_id=user_id, top_k=5)
        
        # Combine top matches into context
        if top_entries:
            retrieved_context = "\n".join(
                f"Relevant Memory: {entry['text']} [tags={entry['tags']}]"
                for entry in top_entries
            )
            
            # Augment the user's message with memory context
            augmented_content = f"{retrieved_context}\n\nUser's current message: {user_message.content}"
            
            # Replace the original user message with the augmented one
            for i, msg in enumerate(request.messages):
                if msg.role == "user" and msg.content == user_message.content:
                    request.messages[i].content = augmented_content
                    break
        
        # Prune history if needed
        request.messages = prune_history(request.messages)
    
    # Handle streaming response if requested
    if request.stream:
        return StreamingResponse(
            stream_completion(request),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    try:
        completion = deepseek_client.chat.completions.create(
            model=request.model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            stream=False
        )
        
        # Store assistant's response in memory if enabled
        if memory_enabled and completion.choices:
            assistant_response = completion.choices[0].message.content
            add_to_memory(assistant_response, source="assistant", user_id=user_id)
        
        return {
            "id": f"chatcmpl-memory-{datetime.datetime.now().timestamp()}",
            "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": completion.choices[0].message.role,
                        "content": completion.choices[0].message.content
                    },
                    "finish_reason": completion.choices[0].finish_reason,
                }
            ],
        }
    except Exception as e:
        return {"error": str(e)}

async def stream_completion(request: ChatCompletionRequest):
    """
    Stream the completion response.
    """
    user_id = request.user or "default_user"
    memory_enabled = request.memory_enabled
    
    try:
        stream = deepseek_client.chat.completions.create(
            model=request.model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            stream=True
        )
        
        assistant_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                assistant_response += piece
                
                # Format as SSE
                yield f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n"
        
        # Store complete response in memory
        if memory_enabled and assistant_response:
            add_to_memory(assistant_response, source="assistant", user_id=user_id)
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
@app.get("/memory/status")
async def memory_status():
    """
    Get the status of the memory system.
    """
    return {
        "status": "active",
        "entries_count": len(metadata),
        "index_size": index.ntotal if hasattr(index, "ntotal") else 0
    }

@app.post("/memory/clear/{user_id}")
async def clear_user_memory(user_id: str):
    """
    Clear memory for a specific user.
    """
    global metadata, index
    
    # Filter out entries for this user
    with index_lock:
        # First get indices of entries to keep
        keep_indices = []
        new_metadata = []
        
        for i, entry in enumerate(metadata):
            if entry.get("user_id") != user_id:
                keep_indices.append(i)
                new_metadata.append(entry)
        
        if len(keep_indices) < len(metadata):
            # We need to rebuild the index
            if keep_indices:
                # Get embeddings for entries we're keeping
                embeddings = []
                for idx in keep_indices:
                    text = metadata[idx]["text"]
                    embedding = get_text_embedding(text)
                    embeddings.append(embedding)
                
                # Build new index
                new_index = faiss.IndexFlatL2(embedding_dim)
                embeddings_array = np.vstack(embeddings)
                new_index.add(embeddings_array)
                
                # Replace old index and metadata
                index = new_index
                metadata = new_metadata
            else:
                # No entries left, initialize empty
                index = faiss.IndexFlatL2(embedding_dim)
                metadata = []
                
            # Save changes
            save_index()
            return {"status": "success", "message": f"Memory cleared for user {user_id}"}
        
        return {"status": "info", "message": f"No memory entries found for user {user_id}"}

if __name__ == "__main__":
    uvicorn.run("memory_api:app", host="127.0.0.1", port=8000, reload=True)