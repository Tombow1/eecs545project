# chat_engine.py
from memory_store import (
    add_to_memory, retrieve_from_memory, is_personal_preference_query
)
from summarizer import prune_history
from openai import OpenAI
import os

client = OpenAI(
    api_key="sk-5ec02f12083d4f97aad73e1adb3a6f48",
    base_url="https://api.deepseek.com"
)

global_messages = [
    {"role": "system", "content": "You are a helpful assistant with memory."}
]

def chat_with_memory(messages, model="deepseek-chat"):
    user_input = messages[-1]["content"]
    tags = ["preference"] if "macbook" in user_input.lower() else []
    add_to_memory(user_input, "user", "api_user", tags)

    if is_personal_preference_query(user_input):
        query = user_input + " (based on my preferences)"
    else:
        query = user_input

    top = retrieve_from_memory(query, top_k=5)
    context = "\n".join(f"Relevant Memory: {e['text']} [tags={e['tags']}]" for e in top)
    augmented = f"{context}\n\nUser's current message: {user_input}"
    global_messages.append({"role": "user", "content": augmented})
    pruned = prune_history(global_messages)
    response = client.chat.completions.create(model=model, messages=pruned, stream=False)
    content = response.choices[0].message.content
    global_messages.append({"role": "assistant", "content": content})
    add_to_memory(content, "assistant", "api_user")
    return content

def chat_with_memory_streaming(messages, model="deepseek-chat"):
    user_input = messages[-1]["content"]
    tags = ["preference"] if "macbook" in user_input.lower() else []
    add_to_memory(user_input, "user", "api_user", tags)

    query = user_input + " (based on my preferences)" if is_personal_preference_query(user_input) else user_input
    context = "\n".join(f"Relevant Memory: {e['text']} [tags={e['tags']}]" for e in retrieve_from_memory(query, top_k=5))
    augmented = f"{context}\n\nUser's current message: {user_input}"
    global_messages.append({"role": "user", "content": augmented})
    pruned = prune_history(global_messages)

    return client.chat.completions.create(model=model, messages=pruned, stream=True)
