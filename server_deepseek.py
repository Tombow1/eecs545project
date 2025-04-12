import requests
import json

# Base URL for your memory server
BASE_URL = "http://127.0.0.1:8000"

# Example 1: Basic chat completion (without memory)
def simple_chat_completion():
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example 2: Chat completion (with memory automatically applied)
def memory_enhanced_chat():
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with memory."},
            {"role": "user", "content": "I really like MacBooks because they're reliable."},
            {"role": "assistant", "content": "That's great! MacBooks are known for their reliability."},
            {"role": "user", "content": "What's your recommendation for a laptop?"}
        ],
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example 3: Manually add entry to memory
def add_to_memory():
    url = f"{BASE_URL}/memory/add"
    payload = {
        "text": "I prefer Windows laptops for gaming",
        "source": "user_preference",
        "user_id": "test_user",
        "tags": ["preference", "windows", "gaming"]
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example 4: Retrieve from memory
def retrieve_from_memory():
    url = f"{BASE_URL}/memory/retrieve"
    payload = {
        "query": "What laptops do I like?",
        "top_k": 3
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example 5: Toggle memory recording
def toggle_memory(enable=True):
    url = f"{BASE_URL}/memory/toggle"
    payload = {
        "state": enable
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example 6: Get memory status
def get_memory_status():
    url = f"{BASE_URL}/memory/status"
    response = requests.get(url)
    return response.json()

# Run examples
if __name__ == "__main__":
    print("=== Simple Chat Completion ===")
    print(json.dumps(simple_chat_completion(), indent=2))
    print("\n")
    
    print("=== Memory Status ===")
    print(json.dumps(get_memory_status(), indent=2))
    print("\n")
    
    print("=== Add Entry to Memory ===")
    print(json.dumps(add_to_memory(), indent=2))
    print("\n")
    
    print("=== Retrieve from Memory ===")
    print(json.dumps(retrieve_from_memory(), indent=2))
    print("\n")
    
    print("=== Memory Enhanced Chat ===")
    print(json.dumps(memory_enhanced_chat(), indent=2))
