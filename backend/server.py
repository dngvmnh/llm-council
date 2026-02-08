"""
Local dev server: POST /debate with JSON {"messages": [{"role":"user","content":"..."}]}
Optional: POST /debate/stream for SSE streaming (OpenRouter or Groq only).
Run from backend dir: python server.py  or  uvicorn server:app --reload --port 8080
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env: project root first, then backend. Only set if not already set so root keys win when backend/.env is empty.
_backend_dir = Path(__file__).resolve().parent
_root_dir = _backend_dir.parent
load_dotenv(_root_dir / ".env")
load_dotenv(_backend_dir / ".env", override=False)
load_dotenv(override=False)  # cwd .env if server run from another directory

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core import run_debate_round, stream_debate_round, get_available_models
from handlers.shared import serialize_responses
from providers import Message

app = FastAPI(title="Multi-LLM Debate API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DebateRequest(BaseModel):
    messages: list[dict]
    model_ids: list[str] | None = None  # Optional: filter to specific models


@app.get("/")
async def root():
    """Health check; confirms backend is up."""
    return {"ok": True, "service": "multi-llm-debate"}


@app.get("/models")
async def list_models():
    """List all available model IDs."""
    return {"models": get_available_models()}


@app.post("/debate")
async def debate(req: DebateRequest):
    messages = [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in req.messages]
    responses = await run_debate_round(messages, selected_models=req.model_ids)
    return {"responses": serialize_responses(responses)}


async def _stream_events(messages: list[Message], selected_models: list[str] | None = None):
    async for ev in stream_debate_round(messages, selected_models=selected_models):
        yield f"data: {json.dumps(ev)}\n\n"


@app.post("/debate/stream")
async def debate_stream(req: DebateRequest):
    """Stream debate via SSE when using OPENROUTER_API_KEY or GROQ_API_KEY."""
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Streaming requires OPENROUTER_API_KEY or GROQ_API_KEY. Use non-streaming POST /debate for native provider keys.",
            },
        )
    messages = [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in req.messages]
    return StreamingResponse(
        _stream_events(messages, selected_models=req.model_ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
