"""
Very thin wrapper around Ollama’s /api/generate for the
vision‑enabled ‘llama3.2‑vision:90b’ model.
"""
import base64
import mimetypes
import pathlib
import requests


OLLAMA_HOST   = "http://localhost:11434"                 # change if your Ollama listens elsewhere
OLLAMA_MODEL  = "llama3.2-vision:90b"                    # vision‑capable model tag
OLLAMA_ROUTE  = f"{OLLAMA_HOST}/api/generate"            # official REST route


def _encode_image(image_path: str) -> str:
    """
    Read image and return a data‑URI suitable for Ollama’s `images` field.
    """
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    data = pathlib.Path(image_path).read_bytes()
    b64  = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def describe(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """
    Send the image to Ollama, get back the textual description (non‑stream).
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "images": [_encode_image(image_path)],
        "stream": False                        # one‑shot response
    }
    try:
        res = requests.post(OLLAMA_ROUTE, json=payload, timeout=300)
        res.raise_for_status()
        data = res.json()                     # {"response": "...", "done": true}
        return data.get("response", "").strip()
    except Exception as exc:
        return f"[ERROR contacting Ollama] {exc}"