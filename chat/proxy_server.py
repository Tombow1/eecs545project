# proxy_server.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from starlette.responses import StreamingResponse, JSONResponse
from chat_engine import chat_with_memory, chat_with_memory_streaming
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proxy_server")

# -----------------------------
# OpenAI-compatible Request Format
# -----------------------------
class Message(BaseModel):
    role: str
    content: Union[str, Dict[str, Any], None]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class FunctionCall(BaseModel):
    name: str
    arguments: str

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Log incoming request
        logger.info(f"Received request for model: {request.model}")
        
        # Convert Pydantic model to dict for all messages
        messages = []
        for m in request.messages:
            msg_dict = m.dict(exclude_unset=True)
            messages.append(msg_dict)
        
        # Handle streaming response
        if request.stream:
            def event_stream():
                try:
                    stream = chat_with_memory_streaming(
                        messages=messages,
                        model=request.model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        user=request.user
                    )
                    
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and hasattr(delta, 'content') and delta.content:
                            payload = {
                                "id": f"chatcmpl-{int(time.time())}",
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
                except Exception as e:
                    logger.error(f"Error in streaming: {str(e)}")
                    error_payload = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # ---------- Non-streaming ----------
        content = chat_with_memory(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user=request.user
        )

        response = {
            "id": f"chatcmpl-{int(time.time())}",
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
            "usage": {
                "prompt_tokens": 0,  # Placeholder
                "completion_tokens": 0,  # Placeholder
                "total_tokens": 0  # Placeholder
            }
        }
        return JSONResponse(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """Endpoint to list available models - required by some OpenAI clients"""
    models = [
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "organization-owner"
        },
        {
            "id": "gpt-4",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "organization-owner"
        }
    ]
    return {"object": "list", "data": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy_server:app", host="127.0.0.1", port=8000)
