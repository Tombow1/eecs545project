# memory_store.py
import os, json, threading, datetime, re, logging
import numpy as np
import faiss
from thefuzz import fuzz
import google.generativeai as genai
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memory_store')

GEMINI_API_KEY = "AIzaSyDm3hL9ZMIjdz8gI0-Q0wkpuY9SdGYtpuA"
genai.configure(api_key=GEMINI_API_KEY)

embedding_dim = 3072
# Use absolute paths and ensure directory exists
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

index_file = os.path.join(data_dir, "memory.index")
metadata_file = os.path.join(data_dir, "memory_meta.json")

# Thread safety
index_lock = threading.Lock()
metadata_lock = threading.Lock()
cache_lock = threading.Lock()

# LRU Cache with size limit
MAX_CACHE_SIZE = 1000
class LRUCache(OrderedDict):
    def __init__(self, maxsize=MAX_CACHE_SIZE):
        self.maxsize = maxsize
        super().__init__()
    
    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
    
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

embedding_cache = LRUCache()

# Initialize FAISS with proper error handling
try:
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        with index_lock:
            index = faiss.read_index(index_file)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        metadata = []
    memory_enabled = True
except Exception as e:
    logger.error(f"Failed to initialize memory store: {str(e)}")
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []
    memory_enabled = False

def toggle_memory(state: bool):
    global memory_enabled
    memory_enabled = state
    logger.info(f"Memory store {'enabled' if state else 'disabled'}")

def save_index():
    try:
        with index_lock:
            faiss.write_index(index, index_file)
        with metadata_lock:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        logger.info("Memory index saved successfully")
    except Exception as e:
        logger.error(f"Failed to save memory index: {str(e)}")

def extract_tags(text):
    return list(set(t for t in re.findall(r"\w+", text.lower()) if len(t) >= 4))

def get_text_embedding(text):
    with cache_lock:
        if text in embedding_cache:
            return embedding_cache[text]
    
    try:
        result = genai.embed_content(model="models/gemini-embedding-exp-03-07", content=text)
        vec = np.array(result["embedding"], dtype=np.float32)
        vec /= np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1
        
        with cache_lock:
            embedding_cache[text] = vec
        return vec
    except Exception as e:
        logger.warning(f"Failed to generate embedding: {str(e)}")
        return np.zeros((embedding_dim,), dtype=np.float32)

def add_to_memory(text, source="unknown", user_id=None, tags=None):
    if not memory_enabled:
        return
    
    try:
        embedding = get_text_embedding(text)
        combined_tags = list(set(extract_tags(text) + (tags or [])))
        entry = {
            "text": text, 
            "source": source,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_id": user_id, 
            "tags": combined_tags
        }
        
        with index_lock:
            index.add(embedding.reshape(1, -1))
        
        with metadata_lock:
            metadata.append(entry)
        
        # Save periodically (could be optimized to save after batches)
        save_index()
    except Exception as e:
        logger.error(f"Failed to add to memory: {str(e)}")

def is_personal_preference_query(query):
    keywords = ["fav", "favorite", "like", "prefer", "my preference"]
    return any(kw in query.lower() for kw in keywords)

def fuzzy_tag_overlap(entry_tags, query_tags, threshold=80):
    return sum(1 for q in query_tags for e in entry_tags if fuzz.ratio(q, e) >= threshold)

def retrieve_from_memory(query, top_k=None):
    if not memory_enabled:
        return []
        
    try:
        with metadata_lock:
            if not metadata:
                return []
        
        embedding = get_text_embedding(query).reshape(1, -1)
        
        with index_lock:
            k = min(top_k or index.ntotal, index.ntotal)
            if k == 0:
                return []
            distances, indices = index.search(embedding, k)

        results = []
        query_tags = extract_tags(query)
        
        # Pre-calculate current time to avoid repeated calls
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        with metadata_lock:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(metadata):
                    entry = metadata[idx]
                    base_score = 1.0 / (1.0 + distances[0][i])
                    
                    # Parse timestamp once
                    entry_time = datetime.datetime.fromisoformat(entry["timestamp"])
                    time_diff_hours = (current_time - entry_time).total_seconds() / 3600
                    recency_weight = 1 / (1 + time_diff_hours)
                    
                    preference_boost = 1.5 if is_personal_preference_query(query) and "preference" in entry["tags"] else 1.0
                    tag_boost = 1.0 + 0.1 * fuzzy_tag_overlap(entry["tags"], query_tags)
                    
                    results.append((entry, base_score * recency_weight * preference_boost * tag_boost))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]
    except Exception as e:
        logger.error(f"Error retrieving from memory: {str(e)}")
        return []
