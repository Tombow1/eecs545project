# sample_chat.py

from openai import OpenAI

client = OpenAI(
    api_key="sk-local",  # dummy key
    base_url="http://localhost:8000/v1"
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with memory. Answer based on memory context if available."
    }
]

print("Connected to local chat API (http://localhost:8000)")
print("Type 'exit' to quit.")
print("--------------------------------------------------")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("Exiting.")
        break

    messages.append({"role": "user", "content": user_input})

    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )

        print("Assistant: ", end="", flush=True)
        full_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_reply += content

        print()
        messages.append({"role": "assistant", "content": full_reply})

    except Exception as e:
        print(f"\n[Error] {e}")
