from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from openai import OpenAI

# Schema Definitions
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

# Create DeepSeek client
deepseek_client = OpenAI(
    api_key="sk-5ec02f12083d4f97aad73e1adb3a6f48",
    base_url="https://api.deepseek.com"
)

# App setup
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Forward to DeepSeek
    completion = deepseek_client.chat.completions.create(
        model=request.model,
        messages=[m.dict() for m in request.messages],
        stream=request.stream,
    )

    return {
        "id": "chatcmpl-proxy",
        "object": "chat.completion",
        "created": 1234567890,  # You could use int(time.time())
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": completion.choices[0].message.dict(),
                "finish_reason": completion.choices[0].finish_reason,
            }
        ],
    }

if __name__ == "__main__":
    uvicorn.run("proxy_server:app", host="127.0.0.1", port=8000)
