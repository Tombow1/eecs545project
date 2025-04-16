# proxy_server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from starlette.responses import StreamingResponse, JSONResponse
from chat_engine import chat_with_memory, chat_with_memory_streaming
import time
import json

# -----------------------------
# OpenAI-compatible Request Format
# -----------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        def event_stream():
            stream = chat_with_memory_streaming(
                messages=[m.dict() for m in request.messages],
                model=request.model
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    payload = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "delta": {"role": "assistant", "content": delta.content},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            # Stream end
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---------- Non-streaming ----------
    content = chat_with_memory(
        messages=[m.dict() for m in request.messages],
        model=request.model
    )

    response = {
        "id": "chatcmpl-nostream",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": None  # Optional: add token counts if desired
    }
    return JSONResponse(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy_server:app", host="127.0.0.1", port=8000)
