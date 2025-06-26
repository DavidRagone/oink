""" Over-provisioned Inference ‘N’ Kit (OINK)

A simple OpenAI‑compatible server on MLX + FastAPI for use on ridiculously 
over-provisioned local hardware like the Mac M3 Ultra.

Run with:
    pip install "fastapi>=0.111" "uvicorn[standard]>=0.29" "pydantic>=2.7" mlx_lm
    python openai_compat_fastapi_mlx.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

Then point the OpenAI Python SDK at:
    openai.base_url = "http://localhost:8889/v1"
    openai.api_key = "sk-local"  # anything except empty string

Endpoints implemented:
  • GET /v1/models
  • POST /v1/chat/completions (supports stream=True)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import AsyncGenerator, List

import mlx_lm as mxlm
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# ─── CLI / CONFIG ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Return CLI args."""
    parser = argparse.ArgumentParser(
            description="Run a local OpenAI‑compatible server using MLX."
            )
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv(
            "MODEL_ID", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
            ),
        help="HF repo or local path of the model to load",
    )
    parser.add_argument("--host",
                        default="0.0.0.0",
                        help="Bind address for Uvicorn")
    parser.add_argument("--port",
                        type=int,
                        default=8889,
                        help="Port for Uvicorn")
    return parser.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# ─── MODEL INITIALISATION ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------
print(f"[server] Loading model '{args.model}' … (first run can take ~10 s)")
model, tokenizer = mxlm.load(args.model, compile=True)
model.eval()
# warm‑up JIT graph so the *next* request is instant
_ = model.generate("warmup", max_tokens=1, stream=False)
print("[server] Model ready → starting API")

# ---------------------------------------------------------------------------
# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
app = FastAPI(title="Local OpenAI‑compatible API (MLX)", version="0.1.0")
MODEL_NAME = args.model

# ---------------------------------------------------------------------------
# ─── SCHEMAS (subset of OpenAI spec) ────────────────────────────────────────
# ---------------------------------------------------------------------------


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx"


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int | None = 256
    temperature: float | None = 0.7
    stream: bool | None = False


# ---------------------------------------------------------------------------
# ─── UTILS ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = "{role}: {content}"


def build_prompt(messages: List[ChatMessage]) -> str:
    """Very simple prompt formatter – join messages line‑by‑line."""
    lines: list[str] = [
            PROMPT_TEMPLATE.format(role=m.role.capitalize(), content=m.content) 
            for m in messages
            ]
    lines.append("Assistant:")
    return "\n".join(lines)


def generate_tokens(prompt: str, *, max_tokens: int, temperature: float):
    """Wrapper around mlx_lm.generate so we can call it sync/async."""
    return model.generate(prompt,
                          max_tokens=max_tokens,
                          temperature=temperature,
                          stream=True)


# ---------------------------------------------------------------------------
# ─── ENDPOINTS ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    """Return a list containing the single loaded model."""
    return {"object": "list", "data": [ModelData(id=MODEL_NAME)]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.model != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Model not loaded")

    prompt = build_prompt(req.messages)
    comp_id = str(uuid.uuid4())
    created = int(time.time())

    if req.stream:
        async def event_stream() -> AsyncGenerator[str, None]:
            collected_tokens: list[str] = []
            for tok in generate_tokens(prompt,
                                       max_tokens=req.max_tokens,
                                       temperature=req.temperature):
                chunk = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "delta": {"content": tok},
                        "index": 0,
                        "finish_reason": None,
                    }],
                }
                collected_tokens.append(tok)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # final message signalling completion
            yield (
                "data: "
                + json.dumps({
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop",
                    }],
                })
                + "\n\n"
                "data: [DONE]\n\n"
            )

        return StreamingResponse(
                event_stream(),
                media_type="text/event-stream"
                )

    # ── non‑stream case ────────────────────────────────────────────────────
    answer = "".join(generate_tokens(prompt,
                                     max_tokens=req.max_tokens,
                                     temperature=req.temperature))
    return {
        "id": comp_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": None,
    }


# ---------------------------------------------------------------------------
# ─── ENTRYPOINT ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "openai_compat_fastapi_mlx:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
    )

