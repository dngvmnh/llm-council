"""
Local dev server: POST /debate with JSON {"messages": [{"role":"user","content":"..."}]}
Optional: POST /debate/stream for SSE streaming (OpenAI).
Run from backend dir: python server.py  or  uvicorn server:app --reload --port 8080
"""
import json
import os
from pathlib import Path
import asyncio
import time

from dotenv import load_dotenv

# Load env files (local dev):
# - Prefer `a.env` for secrets (e.g. OPENAI_API_KEY), keep `.env` for non-secret config.
# - Project root wins over backend/ when both exist.
_backend_dir = Path(__file__).resolve().parent
_root_dir = _backend_dir.parent
load_dotenv(_root_dir / "a.env", override=False)
load_dotenv(_backend_dir / "a.env", override=False)
load_dotenv(_root_dir / ".env", override=False)
load_dotenv(_backend_dir / ".env", override=False)
load_dotenv(override=False)  # cwd .env if server run from another directory

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from core import (
    get_available_models,
    run_debate_pro_con_moderator_judge,
    run_debate_round,
    stream_debate_pro_con_moderator_judge,
    stream_debate_round,
)
from handlers.shared import serialize_responses
from providers import Message
from instagram import (
    iter_instagram_text_messages,
    send_instagram_text,
    verify_x_hub_signature_256,
)
from telegram_api import (
    get_updates as tg_get_updates,
    iter_telegram_text_messages,
    send_telegram_text,
    should_respond as tg_should_respond,
    verify_telegram_webhook_secret,
)

app = FastAPI(title="Multi-LLM Debate API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_IG_THREADS: dict[str, list[Message]] = {}
_IG_THREADS_LOCK = asyncio.Lock()
_TG_THREADS: dict[int, list[Message]] = {}
_TG_THREADS_LOCK = asyncio.Lock()
_TG_POLLING_TASK: asyncio.Task | None = None
_TG_LAST_RUN: dict[int, float] = {}


class DebateRequest(BaseModel):
    messages: list[dict]
    model_ids: list[str] | None = None  # Optional: filter to specific models
    pipeline: str | None = None  # Optional: "panel" | "debate"


@app.get("/")
async def root():
    """Health check; confirms backend is up."""
    return {"ok": True, "service": "multi-llm-debate"}


@app.get("/models")
async def list_models():
    """List all available model IDs."""
    return {"models": await get_available_models()}


@app.post("/debate")
async def debate(req: DebateRequest):
    messages = [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in req.messages]
    pipeline = (req.pipeline or os.getenv("DEFAULT_PIPELINE", "panel")).strip().lower()
    if pipeline == "debate":
        responses = await run_debate_pro_con_moderator_judge(messages, selected_models=req.model_ids)
    else:
        responses = await run_debate_round(messages, selected_models=req.model_ids)
    return {"responses": serialize_responses(responses)}


async def _stream_events(messages: list[Message], selected_models: list[str] | None = None, pipeline: str = "panel"):
    if pipeline == "debate":
        async for ev in stream_debate_pro_con_moderator_judge(messages, selected_models=selected_models):
            yield f"data: {json.dumps(ev)}\n\n"
        return

    async for ev in stream_debate_round(messages, selected_models=selected_models):
        yield f"data: {json.dumps(ev)}\n\n"


@app.post("/debate/stream")
async def debate_stream(req: DebateRequest):
    """Stream debate via SSE (OpenAI)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Streaming requires OPENAI_API_KEY.",
            },
        )
    messages = [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in req.messages]
    pipeline = (req.pipeline or os.getenv("DEFAULT_PIPELINE", "panel")).strip().lower()
    return StreamingResponse(
        _stream_events(messages, selected_models=req.model_ids, pipeline=pipeline),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _handle_instagram_text_event(ig_user_id: str, sender_id: str, text: str) -> None:
    """
    Handle an incoming Instagram DM text: run the structured debate pipeline and reply
    back as 4 persona messages (Moderator/Pro/Con/Judge).
    """
    # Use env override if set (helpful when webhook payload omits it).
    ig_id = (os.getenv("IG_USER_ID") or ig_user_id or "").strip()
    if not ig_id:
        return

    model_pool_raw = (os.getenv("IG_MODEL_POOL") or "").strip()
    model_pool = [m.strip() for m in model_pool_raw.split(",") if m.strip()] if model_pool_raw else None

    async with _IG_THREADS_LOCK:
        history = list(_IG_THREADS.get(sender_id) or [])

    llm_messages = history + [Message(role="user", content=text)]
    try:
        responses = await run_debate_pro_con_moderator_judge(llm_messages, selected_models=model_pool)
    except Exception as e:
        try:
            await send_instagram_text(ig_user_id=ig_id, recipient_id=sender_id, text=f"[System]\nError: {e}")
        except Exception:
            pass
        return

    # Update in-memory thread history (best-effort, capped).
    thread_max = int(os.getenv("IG_THREAD_MAX_MESSAGES", "60"))
    async with _IG_THREADS_LOCK:
        thread = _IG_THREADS.setdefault(sender_id, [])
        thread.append(Message(role="user", content=text))
        for r in responses:
            label = (r.provider_id or "assistant").strip()
            if r.error:
                thread.append(Message(role="assistant", content=f"[{label}]\nError: {r.error}"))
            else:
                thread.append(Message(role="assistant", content=f"[{label}]\n{(r.content or '').strip()}"))
        if thread_max > 0 and len(thread) > thread_max:
            _IG_THREADS[sender_id] = thread[-thread_max:]

    # Reply to the user as separate persona messages.
    for r in responses:
        label = (r.provider_id or "assistant").strip()
        if r.error:
            msg = f"[{label}]\nError: {r.error}"
        else:
            msg = f"[{label}]\n{(r.content or '').strip()}"
        try:
            await send_instagram_text(ig_user_id=ig_id, recipient_id=sender_id, text=msg)
        except Exception:
            # Don't crash the whole webhook handler if one send fails.
            pass


@app.get("/webhooks/instagram")
async def instagram_webhook_verify(request: Request):
    """
    Meta Webhooks verification endpoint for Instagram.
    Configure your callback URL to: /webhooks/instagram
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    verify = (os.getenv("IG_VERIFY_TOKEN") or "").strip()
    if mode == "subscribe" and verify and token == verify and challenge:
        return PlainTextResponse(challenge)
    return JSONResponse(status_code=403, content={"error": "Forbidden"})


@app.post("/webhooks/instagram")
async def instagram_webhook(request: Request):
    """
    Instagram Messaging API webhook receiver (text messages).

    This ACKs quickly and processes events in background tasks.
    """
    body = await request.body()
    app_secret = (os.getenv("IG_APP_SECRET") or "").strip()
    sig = request.headers.get("X-Hub-Signature-256")
    if app_secret and not verify_x_hub_signature_256(body, sig, app_secret):
        return JSONResponse(status_code=403, content={"error": "Bad signature"})

    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    events = list(iter_instagram_text_messages(payload))
    for ig_user_id, sender_id, text in events:
        asyncio.create_task(_handle_instagram_text_event(ig_user_id, sender_id, text))

    return PlainTextResponse("EVENT_RECEIVED")


def _parse_int_set(csv: str) -> set[int]:
    out: set[int] = set()
    for part in (csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


async def _handle_telegram_text_event(
    chat_id: int,
    message_id: int,
    chat_type: str,
    sender: str,
    text: str,
    msg_obj: dict,
) -> None:
    token = (os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        return

    allowed_raw = (os.getenv("TG_ALLOWED_CHAT_IDS") or "").strip()
    if allowed_raw:
        allowed = _parse_int_set(allowed_raw)
        if allowed and chat_id not in allowed:
            return

    bot_username = (os.getenv("TG_BOT_USERNAME") or "").strip()
    trigger_mode = (os.getenv("TG_TRIGGER") or "command_or_mention").strip().lower()
    should, cmd, prompt = tg_should_respond(
        text=text,
        chat_type=chat_type,
        message_obj=msg_obj,
        bot_username=bot_username,
        trigger_mode=trigger_mode,
    )
    if not should:
        return

    if cmd and cmd not in ("council", "debate", "panel", "start", "help"):
        # Ignore commands meant for other bots (or just unknown commands) to avoid accidental cost burn.
        return

    pipeline_default = (os.getenv("TG_DEFAULT_PIPELINE") or os.getenv("DEFAULT_PIPELINE") or "debate").strip().lower()
    pipeline = pipeline_default
    if cmd in ("panel",):
        pipeline = "panel"
    if cmd in ("debate", "council"):
        pipeline = "debate"
    if cmd in ("start", "help"):
        help_text = (
            "Use:\n"
            "- /council <topic>  (structured debate)\n"
            "- /panel <topic>    (panel of models)\n"
            "In groups, I only respond to commands/mentions by default."
        )
        try:
            await send_telegram_text(token=token, chat_id=chat_id, text=help_text, reply_to_message_id=message_id or None)
        except Exception:
            pass
        return

    prompt = (prompt or "").strip()
    if not prompt:
        try:
            await send_telegram_text(
                token=token,
                chat_id=chat_id,
                text="Please provide a topic. Example: /council Should AI be regulated?",
                reply_to_message_id=message_id or None,
            )
        except Exception:
            pass
        return

    cooldown = float((os.getenv("TG_COOLDOWN_SEC") or "0").strip() or 0)
    if cooldown > 0:
        now = time.time()
        async with _TG_THREADS_LOCK:
            last = float(_TG_LAST_RUN.get(chat_id) or 0.0)
            if now - last < cooldown:
                return
            _TG_LAST_RUN[chat_id] = now

    # In group chats, include the speaker so the models can track who said what.
    if (chat_type or "").lower() != "private":
        user_content = f"{sender}: {prompt}"
    else:
        user_content = prompt

    model_pool_raw = (os.getenv("TG_MODEL_POOL") or "").strip()
    model_pool = [m.strip() for m in model_pool_raw.split(",") if m.strip()] if model_pool_raw else None

    async with _TG_THREADS_LOCK:
        history = list(_TG_THREADS.get(chat_id) or [])
    llm_messages = history + [Message(role="user", content=user_content)]

    try:
        if pipeline == "panel":
            responses = await run_debate_round(llm_messages, selected_models=model_pool)
        else:
            responses = await run_debate_pro_con_moderator_judge(llm_messages, selected_models=model_pool)
    except Exception as e:
        try:
            await send_telegram_text(token=token, chat_id=chat_id, text=f"[System]\nError: {e}", reply_to_message_id=message_id or None)
        except Exception:
            pass
        return

    # Update in-memory chat history (best-effort, capped).
    thread_max = int(os.getenv("TG_THREAD_MAX_MESSAGES", "80"))
    async with _TG_THREADS_LOCK:
        thread = _TG_THREADS.setdefault(chat_id, [])
        thread.append(Message(role="user", content=user_content))
        for r in responses:
            label = (r.provider_id or "assistant").strip()
            if r.error:
                thread.append(Message(role="assistant", content=f"[{label}]\nError: {r.error}"))
            else:
                thread.append(Message(role="assistant", content=f"[{label}]\n{(r.content or '').strip()}"))
        if thread_max > 0 and len(thread) > thread_max:
            _TG_THREADS[chat_id] = thread[-thread_max:]

    # Reply policy:
    # - Debate mode: 4 persona messages (Moderator/Pro/Con/Judge) by default.
    # - Panel mode: combined into 1 message by default to avoid flooding groups.
    persona_delay_ms = int(os.getenv("TG_PERSONA_DELAY_MS", "250"))

    if pipeline == "panel" and (os.getenv("TG_PANEL_COMBINE", "1").strip() != "0"):
        blocks: list[str] = []
        for r in responses:
            label = (r.provider_id or "model").strip()
            if r.error:
                blocks.append(f"[{label}]\nError: {r.error}")
            else:
                blocks.append(f"[{label}]\n{(r.content or '').strip()}")
        combined = "\n\n".join(blocks).strip()
        try:
            await send_telegram_text(token=token, chat_id=chat_id, text=combined, reply_to_message_id=message_id or None)
        except Exception:
            pass
        return

    first = True
    for r in responses:
        label = (r.provider_id or "assistant").strip()
        if r.error:
            msg = f"[{label}]\nError: {r.error}"
        else:
            msg = f"[{label}]\n{(r.content or '').strip()}"
        try:
            await send_telegram_text(
                token=token,
                chat_id=chat_id,
                text=msg,
                reply_to_message_id=(message_id if first and message_id else None),
            )
        except Exception:
            pass
        first = False
        if persona_delay_ms > 0:
            await asyncio.sleep(persona_delay_ms / 1000.0)


@app.post("/webhooks/telegram")
async def telegram_webhook(request: Request):
    """
    Telegram webhook receiver.

    Configure webhook URL to: /webhooks/telegram
    Optionally set `TG_WEBHOOK_SECRET_TOKEN` and configure webhook with the same secret token.
    """
    secret = (os.getenv("TG_WEBHOOK_SECRET_TOKEN") or "").strip()
    if secret and not verify_telegram_webhook_secret(request.headers.get("X-Telegram-Bot-Api-Secret-Token"), secret):
        return JSONResponse(status_code=403, content={"error": "Bad secret token"})

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    events = list(iter_telegram_text_messages(payload if isinstance(payload, dict) else {}))
    for chat_id, message_id, chat_type, sender, text, msg_obj in events:
        asyncio.create_task(_handle_telegram_text_event(chat_id, message_id, chat_type, sender, text, msg_obj))
    return JSONResponse(content={"ok": True})


async def _telegram_polling_loop() -> None:
    token = (os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        return

    poll_timeout = int(os.getenv("TG_POLL_TIMEOUT_SEC", "30"))
    offset: int | None = None
    while True:
        try:
            data = await tg_get_updates(token=token, offset=offset, timeout=poll_timeout)
            # Advance offset first to reduce duplicate processing on slow handlers.
            updates = data.get("result") if isinstance(data, dict) else None
            if isinstance(updates, list):
                for upd in updates:
                    if isinstance(upd, dict):
                        uid = upd.get("update_id")
                        if isinstance(uid, int):
                            offset = (uid + 1) if offset is None else max(offset, uid + 1)

            for chat_id, message_id, chat_type, sender, text, msg_obj in iter_telegram_text_messages(data if isinstance(data, dict) else {}):
                asyncio.create_task(_handle_telegram_text_event(chat_id, message_id, chat_type, sender, text, msg_obj))
        except Exception:
            await asyncio.sleep(2.0)


@app.on_event("startup")
async def _startup() -> None:
    # Optional: enable long polling for local dev.
    # Note: Telegram can't use polling while a webhook is active.
    global _TG_POLLING_TASK
    if (os.getenv("TG_POLLING") or "").strip() == "1" and _TG_POLLING_TASK is None:
        _TG_POLLING_TASK = asyncio.create_task(_telegram_polling_loop())


if __name__ == "__main__":
    reload = (os.getenv("UVICORN_RELOAD") or "1").strip() == "1"
    # Avoid running multiple pollers when uvicorn reload restarts the worker process.
    if (os.getenv("TG_POLLING") or "").strip() == "1":
        reload = False
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=reload)
