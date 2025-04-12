from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 100 + 2?"}
    ]
)

print(response.choices[0].message.content)
