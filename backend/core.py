"""
Debate orchestration: fan-out to selected OpenAI models.
"""
import asyncio
import os
import time
import hashlib
import json
from typing import List

from providers import (
    Message,
    ProviderResponse,
)
from providers.openai_multi_provider import (
    filter_chat_model_ids,
    get_openai_responses,
    get_debate_models,
    list_available_openai_models,
    uses_max_completion_tokens,
)

DEFAULT_SYSTEM = (
    "You are participating in a shared discussion with the user and other AI assistants. "
    "You can see what other models have said in previous messages (they are labeled with their model names). "
    "Read their responses carefully, and feel free to agree, disagree, build upon, or challenge their viewpoints. "
    "Give a clear, concise response that engages with the conversation."
)

# Simple in-memory cache for recent debate requests (key -> (ts, responses_list))
_CACHE: dict = {}
_CACHE_LOCK = asyncio.Lock()
# TTL for cache entries (seconds)
_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL_SEC", "300"))

# Default max tokens (applies to provider requests when supported)
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))


def _has_any_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


async def get_available_models() -> list[str]:
    """Return OpenAI chat-capable model IDs for frontend selection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        models = await list_available_openai_models(client)
        # Ensure debate defaults are always present if the API listing filters them out.
        return filter_chat_model_ids(set(models).union(set(get_debate_models())))
    except Exception:
        # Fallback to configured debate models.
        return filter_chat_model_ids(get_debate_models())


async def run_debate_round(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
) -> List[ProviderResponse]:
    """Send conversation to selected OpenAI models in parallel."""
    if not _has_any_key():
        return [
            ProviderResponse(
                provider_id="system",
                content="",
                error="No LLM API keys configured. Set OPENAI_API_KEY.",
            )
        ]
    system = system_prompt or DEFAULT_SYSTEM

    # Guardrail: cap number of models per round to avoid accidental credit burn.
    max_models_per_round = int(os.getenv("MAX_MODELS_PER_ROUND", "100"))
    effective_selected_models: list[str] | None = None
    truncated_notice: str | None = None
    if selected_models:
        # Preserve caller order, drop duplicates.
        deduped = list(dict.fromkeys(selected_models))
        if max_models_per_round > 0 and len(deduped) > max_models_per_round:
            truncated = deduped[:max_models_per_round]
            truncated_notice = (
                f"Selected {len(deduped)} models but MAX_MODELS_PER_ROUND={max_models_per_round}; "
                f"running only: {', '.join(truncated)}"
            )
            deduped = truncated
        effective_selected_models = deduped

    # Compute cache key (messages + system + selected_models)
    try:
        key_obj = {
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "selected_models": effective_selected_models or [],
        }
        key_raw = json.dumps(key_obj, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    except Exception:
        key = None

    # Return cached responses if fresh
    if key is not None:
        async with _CACHE_LOCK:
            item = _CACHE.get(key)
            if item:
                ts, cached = item
                if time.time() - ts < _CACHE_TTL:
                    # return a shallow copy to avoid mutation
                    return [ProviderResponse(r.provider_id, r.content, r.error) for r in cached]
                else:
                    del _CACHE[key]

    tasks = []
    tasks.append(get_openai_responses(messages, system, selected_models=effective_selected_models))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[ProviderResponse] = []
    idx = 0
    r = results[idx]
    if isinstance(r, Exception):
        out.append(ProviderResponse(provider_id="openai", content="", error=str(r)))
    else:
        out.extend(r)

    if truncated_notice:
        out.insert(0, ProviderResponse(provider_id="system", content="", error=truncated_notice))

    # store in cache
    if key is not None:
        async with _CACHE_LOCK:
            try:
                _CACHE[key] = (time.time(), [ProviderResponse(r.provider_id, r.content, r.error) for r in out])
            except Exception:
                pass

    return out


async def stream_debate_round(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
):
    """
    Stream debate responses when using OpenAI.
    Yields dicts: {"model_id": str, "delta": str} | {"model_id": str, "done": True} | {"model_id": str, "error": str}.
    If OPENAI_API_KEY is not set, yields nothing.
    """
    system = system_prompt or DEFAULT_SYSTEM
    max_models_per_round = int(os.getenv("MAX_MODELS_PER_ROUND", "100"))
    effective_selected_models: list[str] | None = None
    truncated_notice: str | None = None
    if selected_models:
        deduped = list(dict.fromkeys(selected_models))
        if max_models_per_round > 0 and len(deduped) > max_models_per_round:
            truncated = deduped[:max_models_per_round]
            truncated_notice = (
                f"Selected {len(deduped)} models but MAX_MODELS_PER_ROUND={max_models_per_round}; "
                f"running only: {', '.join(truncated)}"
            )
            deduped = truncated
        effective_selected_models = deduped

    if truncated_notice:
        yield {"model_id": "system", "delta": truncated_notice}

    openai_messages = [{"role": "system", "content": system}] if system else []
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    queue: asyncio.Queue = asyncio.Queue()
    expected_sentinels = 0

    if os.environ.get("OPENAI_API_KEY"):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        models = effective_selected_models or get_debate_models()

        for model_id in models:
            expected_sentinels += 1

            async def stream_one(m_id: str):
                try:
                    params = {"model": m_id, "messages": openai_messages, "stream": True}
                    if uses_max_completion_tokens(m_id):
                        params["max_completion_tokens"] = DEFAULT_MAX_TOKENS
                    else:
                        params["max_tokens"] = DEFAULT_MAX_TOKENS

                    try:
                        stream = await client.chat.completions.create(**params)
                    except Exception as e:
                        msg = str(e)
                        if "Unsupported parameter: 'max_tokens'" in msg and "max_completion_tokens" in msg:
                            params.pop("max_tokens", None)
                            params["max_completion_tokens"] = DEFAULT_MAX_TOKENS
                            stream = await client.chat.completions.create(**params)
                        elif "Unsupported parameter: 'max_completion_tokens'" in msg and "max_tokens" in msg:
                            params.pop("max_completion_tokens", None)
                            params["max_tokens"] = DEFAULT_MAX_TOKENS
                            stream = await client.chat.completions.create(**params)
                        else:
                            raise
                    async for chunk in stream:
                        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                        if delta:
                            await queue.put({"model_id": m_id, "delta": delta})
                    await queue.put({"model_id": m_id, "done": True})
                except Exception as e:
                    await queue.put({"model_id": m_id, "error": str(e)})
                finally:
                    await queue.put(None)

            asyncio.create_task(stream_one(model_id))

    if expected_sentinels == 0:
        return

    sentinels = 0
    while sentinels < expected_sentinels:
        ev = await queue.get()
        if ev is None:
            sentinels += 1
            continue
        yield ev
