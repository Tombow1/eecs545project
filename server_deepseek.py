from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from openai import OpenAI
import json
import numpy as np
import faiss
import datetime
import os
import threading
import re
from thefuzz import fuzz
import google.generativeai as genai

# Schema Definitions
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

# Memory System Setup
embedding_dim = 3072  # For gemini-embedding-exp-03-07
index_file = "memory.index"
metadata_file = "memory_meta.json"
index_lock = threading.Lock()
embedding_cache = {}
memory_enabled = True

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Load or initialize memory index
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

# Create DeepSeek client
deepseek_client = OpenAI(
    api_key="sk-5ec02f12083d4f97aad73e1adb3a6f48",
    base_url="https://api.deepseek.com"
)

# Memory Management Functions
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

def extract_tags(text):
    """Extract simple tags from text."""
    text_lower = text.lower()
    tokens = re.findall(r"\w+", text_lower)
    tokens = [t for t in tokens if len(t) >= 4]
    return list(set(tokens))

def get_text_embedding(text):
    """Compute text embedding using Gemini's embedding model."""
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

def add_to_memory(text, source="unknown", user_id=None, tags=None):
    """Add an entry to the persistent memory."""
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
    """Detect a personal preference inquiry to apply a weighting boost."""
    keywords = ["fav", "favorite", "like", "prefer", "my preference"]
    q_lower = query.lower()
    return any(kw in q_lower for kw in keywords)

def fuzzy_tag_overlap(entry_tags, query_tags, threshold=80):
    """Return how many 'fuzzy-matched' tags we find between two sets of tags."""
    overlap_count = 0
    for qtag in query_tags:
        for etag in entry_tags:
            score = fuzz.ratio(qtag, etag)
            if score >= threshold:
                overlap_count += 1
                break
    return overlap_count

def retrieve_from_memory(query, top_k=None):
    """Perform a semantic search over stored memory."""
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

# API Routes
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Process messages to extract user query
    system_msg = next((m for m in request.messages if m.role == "system"), None)
    user_msgs = [m for m in request.messages if m.role == "user"]
    
    if not user_msgs:
        return {"error": "No user message found"}
    
    user_input = user_msgs[-1].content
    
    # Custom tags for memory
    custom_tags = []
    if "macbook" in user_input.lower() or "macbooks" in user_input.lower():
        custom_tags.append("preference")
    
    # Add the user's message to memory
    add_to_memory(user_input, source="user", user_id="api_user", tags=custom_tags)
    
    # Query memory for relevant context
    if is_personal_preference_query(user_input):
        reformulated_query = user_input + " (based on my preferences)"
    else:
        reformulated_query = user_input
    
    all_entries = retrieve_from_memory(reformulated_query, top_k=None)
    top_entries = all_entries[:5]
    
    # Combine top matches into context
    retrieved_context = "\n".join(
        f"Relevant Memory: {entry['text']} [tags={entry['tags']}]"
        for entry in top_entries
    )
    
    # Prepare augmented messages for DeepSeek
    augmented_messages = []
    
    # Add system message if it exists
    if system_msg:
        augmented_system_content = (
            f"{system_msg.content}\n\n"
            "You have access to the user's memory context. "
            "Use the retrieved memory if relevant."
        )
        augmented_messages.append({"role": "system", "content": augmented_system_content})
    
    # Add previous messages (except the last user message)
    for m in request.messages:
        if m.role != "system" and m != user_msgs[-1]:
            augmented_messages.append({"role": m.role, "content": m.content})
    
    # Add augmented user message with memory context
    augmented_input = f"{retrieved_context}\n\nUser's current message: {user_input}"
    augmented_messages.append({"role": "user", "content": augmented_input})
    
    # Create request dict for DeepSeek
    req_dict = request.dict(exclude_unset=True)
    req_dict["messages"] = [{"role": m["role"], "content": m["content"]} for m in augmented_messages]
    
    try:
        # Forward to DeepSeek
        completion = deepseek_client.chat.completions.create(**req_dict)
        
        # Record the assistant's response in memory
        assistant_response = completion.choices[0].message.content
        add_to_memory(assistant_response, source="assistant", user_id="api_user")
        
        # Return the response
        return {
            "id": "chatcmpl-proxy",
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

# Add a memory management endpoint
@app.post("/memory/toggle")
async def toggle_memory(enable: bool):
    global memory_enabled
    memory_enabled = enable
    return {"status": "Memory recording is now " + ("enabled" if enable else "disabled")}

if __name__ == "__main__":
    uvicorn.run("memory_proxy_server:app", host="127.0.0.1", port=8000)
