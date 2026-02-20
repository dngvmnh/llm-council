"""
OpenAI multi-model provider: one API key, N OpenAI models.

This repo's UI lets the user select which models to run per message; the backend must
only call selected models to avoid burning credits.
"""
import asyncio
import os
import time
from typing import Iterable

from openai import AsyncOpenAI

from .base import Message, ProviderResponse


DEFAULT_OPENAI_MODELS = [
    # Keep defaults conservative and cheap-ish; user can override via OPENAI_DEBATE_MODELS.
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
]

_MODEL_CACHE: dict[str, object] = {"ts": 0.0, "models": []}
_MODEL_CACHE_TTL_SEC = int(os.getenv("OPENAI_MODEL_CACHE_TTL_SEC", "300"))

_CHAT_ALLOW_PREFIXES = ("gpt-", "o1", "o3", "chatgpt-")
# Keep this conservative: the UI uses /v1/chat/completions, so exclude families that
# require other endpoints (realtime, instruct, etc.).
_CHAT_DENY_SUBSTRINGS = (
    "realtime",
    "instruct",
    "transcribe",
    "transcription",
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "image",
    "audio",
    "moderation",
)


def get_debate_models() -> list[str]:
    raw = os.environ.get("OPENAI_DEBATE_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return DEFAULT_OPENAI_MODELS


def is_chat_model_id(model_id: str) -> bool:
    if not isinstance(model_id, str) or not model_id:
        return False
    s = model_id.lower()
    if any(d in s for d in _CHAT_DENY_SUBSTRINGS):
        return False
    return s.startswith(_CHAT_ALLOW_PREFIXES)


def uses_max_completion_tokens(model_id: str) -> bool:
    """
    Some OpenAI models don't support `max_tokens` on `/v1/chat/completions` and require
    `max_completion_tokens` instead.
    """
    s = (model_id or "").lower()
    return s.startswith(("o1", "o3", "gpt-5"))


def filter_chat_model_ids(model_ids: Iterable[str]) -> list[str]:
    """
    The OpenAI /models list can include non-chat models (embeddings, tts, whisper, etc.).
    This app is chat-only, so we filter by common chat model prefixes to prevent
    noisy failures in the UI model selector.
    """
    out: list[str] = []
    for mid in model_ids:
        if is_chat_model_id(mid):
            out.append(mid)
    return sorted(set(out))


async def list_available_openai_models(client: AsyncOpenAI) -> list[str]:
    """List OpenAI chat-capable models (cached)."""
    now = time.time()
    ts = float(_MODEL_CACHE.get("ts") or 0.0)
    cached = _MODEL_CACHE.get("models")
    if isinstance(cached, list) and now - ts < _MODEL_CACHE_TTL_SEC:
        return cached

    models = await client.models.list()
    ids = [m.id for m in getattr(models, "data", []) if getattr(m, "id", None)]
    filtered = filter_chat_model_ids(ids)
    _MODEL_CACHE["ts"] = now
    _MODEL_CACHE["models"] = filtered
    return filtered


async def get_openai_responses(
    messages: list[Message],
    system_prompt: str | None = None,
    selected_models: list[str] | None = None,
) -> list[ProviderResponse]:
    """Call OpenAI for each selected model in parallel."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []

    models = selected_models or get_debate_models()
    if not models:
        return []

    client = AsyncOpenAI(api_key=api_key)
    system = system_prompt or ""

    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))

    requested_models = list(models)
    call_models = [m for m in requested_models if is_chat_model_id(m)]

    async def one_model(model_id: str) -> ProviderResponse:
        try:
            params = {"model": model_id, "messages": openai_messages}
            if uses_max_completion_tokens(model_id):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            try:
                r = await client.chat.completions.create(**params)
            except Exception as e:
                # Fallback for models that switch token-limit parameter names.
                msg = str(e)
                if "Unsupported parameter: 'max_tokens'" in msg and "max_completion_tokens" in msg:
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = max_tokens
                    r = await client.chat.completions.create(**params)
                elif "Unsupported parameter: 'max_completion_tokens'" in msg and "max_tokens" in msg:
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = max_tokens
                    r = await client.chat.completions.create(**params)
                else:
                    raise
            content = (r.choices[0].message.content or "").strip()
            return ProviderResponse(provider_id=model_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=model_id, content="", error=str(e))

    results = await asyncio.gather(*[one_model(m) for m in call_models], return_exceptions=True)
    out: list[ProviderResponse] = []
    by_id: dict[str, ProviderResponse] = {}
    for i, r in enumerate(results):
        model_id = call_models[i]
        if isinstance(r, Exception):
            by_id[model_id] = ProviderResponse(provider_id=model_id, content="", error=str(r))
        else:
            by_id[model_id] = r

    for model_id in requested_models:
        if not is_chat_model_id(model_id):
            out.append(
                ProviderResponse(
                    provider_id=model_id,
                    content="",
                    error="Model is not supported by the /v1/chat/completions endpoint (not a chat model).",
                )
            )
            continue
        out.append(by_id.get(model_id) or ProviderResponse(provider_id=model_id, content="", error="Unknown error"))

    return out
