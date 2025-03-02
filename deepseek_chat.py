import os
import openai

# Get API key from environment variable
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    api_key = input("Please enter your DeepSeek API key: ")
    os.environ["DEEPSEEK_API_KEY"] = api_key

# Configure the client
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"  # DeepSeek API endpoint
)

# Initialize conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Interactive DeepSeek Chat (type 'exit' to quit)")
print("-----------------------------------------------")

while True:
    # Get user input - using the prompt directly in the input function
    user_input = input("\nYou: ")
    
    # Check if user wants to exit
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("\nGoodbye!")
        break
    
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    try:
        # Stream the response for a more interactive feel
        print("\nDeepSeek: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="deepseek-chat",  # or "deepseek-reasoner" for the reasoning model
            messages=messages,
            stream=True
        )
        
        assistant_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_chunk = chunk.choices[0].delta.content
                assistant_response += content_chunk
                print(content_chunk, end="", flush=True)
        print()  # Add newline after response
        
        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": assistant_response})
        
    except Exception as e:
        print(f"\nError: {str(e)}")
