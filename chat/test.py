import requests
response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi, what modal are you"}],
        "stream": False
    }
)
print(response.json())
