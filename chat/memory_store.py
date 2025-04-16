# memory_store.py
import os, json, threading, datetime, re
import numpy as np
import faiss
from thefuzz import fuzz
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA"
genai.configure(api_key=GEMINI_API_KEY)

embedding_dim = 3072
index_file = "memory.index"
metadata_file = "memory_meta.json"
index_lock = threading.Lock()
embedding_cache = {}

# Initialize FAISS
if os.path.exists(index_file) and os.path.exists(metadata_file):
    with index_lock:
        index = faiss.read_index(index_file)
        metadata = json.load(open(metadata_file))
else:
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []

memory_enabled = True

def toggle_memory(state: bool):
    global memory_enabled
    memory_enabled = state

def save_index():
    with index_lock:
        faiss.write_index(index, index_file)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

def extract_tags(text):
    return list(set(t for t in re.findall(r"\w+", text.lower()) if len(t) >= 4))

def get_text_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    try:
        result = genai.embed_content(model="models/gemini-embedding-exp-03-07", content=text)
        vec = np.array(result["embedding"], dtype=np.float32)
        vec /= np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1
        embedding_cache[text] = vec
        return vec
    except:
        return np.zeros((embedding_dim,), dtype=np.float32)

def add_to_memory(text, source="unknown", user_id=None, tags=None):
    if not memory_enabled:
        return
    embedding = get_text_embedding(text)
    with index_lock:
        index.add(embedding.reshape(1, -1))
    combined_tags = list(set(extract_tags(text) + (tags or [])))
    metadata.append({
        "text": text, "source": source,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_id": user_id, "tags": combined_tags
    })
    save_index()

def is_personal_preference_query(query):
    keywords = ["fav", "favorite", "like", "prefer", "my preference"]
    return any(kw in query.lower() for kw in keywords)

def fuzzy_tag_overlap(entry_tags, query_tags, threshold=80):
    return sum(1 for q in query_tags for e in entry_tags if fuzz.ratio(q, e) >= threshold)

def retrieve_from_memory(query, top_k=None):
    if not metadata:
        return []

    embedding = get_text_embedding(query).reshape(1, -1)
    with index_lock:
        k = top_k or index.ntotal
        distances, indices = index.search(embedding, k)

    results = []
    query_tags = extract_tags(query)
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            base_score = 1.0 / (1.0 + distances[0][i])
            recency_weight = 1 / (1 + (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(entry["timestamp"])).total_seconds() / 3600)
            preference_boost = 1.5 if is_personal_preference_query(query) and "preference" in entry["tags"] else 1.0
            tag_boost = 1.0 + 0.1 * fuzzy_tag_overlap(entry["tags"], query_tags)
            results.append((entry, base_score * recency_weight * preference_boost * tag_boost))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results]
