"""
Debate orchestration: fan-out to all configured LLMs (single-key via OpenRouter/Groq or per-provider keys).
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
    OpenAIProvider,
    GeminiProvider,
    GrokProvider,
    KimiProvider,
    ClaudeProvider,
)
from providers.openrouter_provider import get_openrouter_responses, _get_models as _get_openrouter_models, _slug as _openrouter_slug
from providers.groq_multi_provider import get_groq_responses, _get_models as _get_groq_models

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


def _get_native_providers():
    """Per-provider API keys (OpenAI, Gemini, xAI, Moonshot, Anthropic)."""
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
    if os.environ.get("GEMINI_API_KEY"):
        providers.append(GeminiProvider())
    if os.environ.get("XAI_API_KEY"):
        providers.append(GrokProvider())
    if os.environ.get("MOONSHOT_API_KEY"):
        providers.append(KimiProvider())
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(ClaudeProvider())
    return providers


def _has_any_key() -> bool:
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def get_available_models() -> list[str]:
    """Return list of all available model IDs (for frontend selection)."""
    models = []
    if os.environ.get("OPENROUTER_API_KEY"):
        for m in _get_openrouter_models():
            models.append(_openrouter_slug(m))
    if os.environ.get("GROQ_API_KEY"):
        models.extend(_get_groq_models())
    native = _get_native_providers()
    for p in native:
        models.append(p.provider_id)
    return sorted(models)


async def run_debate_round(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
) -> List[ProviderResponse]:
    """Send conversation to all configured LLMs in parallel. Uses OpenRouter/Groq (one key, N models) and/or native provider keys."""
    if not _has_any_key():
        return [
            ProviderResponse(
                provider_id="system",
                content="",
                error="No LLM API keys configured. Set OPENROUTER_API_KEY or GROQ_API_KEY (one key, multiple models), or any of: OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, MOONSHOT_API_KEY, ANTHROPIC_API_KEY",
            )
        ]
    system = system_prompt or DEFAULT_SYSTEM

    # Compute cache key (messages + system + selected_models)
    try:
        key_obj = {
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "selected_models": selected_models or [],
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

    async def openrouter_task():
        responses = await get_openrouter_responses(messages, system)
        if selected_models:
            return [r for r in responses if r.provider_id in selected_models]
        return responses

    async def groq_task():
        responses = await get_groq_responses(messages, system)
        if selected_models:
            return [r for r in responses if r.provider_id in selected_models]
        return responses

    native = _get_native_providers()
    if selected_models:
        native = [p for p in native if p.provider_id in selected_models]

    tasks = []
    if os.environ.get("OPENROUTER_API_KEY"):
        tasks.append(openrouter_task())
    if os.environ.get("GROQ_API_KEY"):
        tasks.append(groq_task())
    for p in native:
        tasks.append(p.chat(messages, system))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[ProviderResponse] = []
    idx = 0
    if os.environ.get("OPENROUTER_API_KEY"):
        r = results[idx]
        idx += 1
        if isinstance(r, Exception):
            out.append(ProviderResponse(provider_id="openrouter", content="", error=str(r)))
        else:
            out.extend(r)
    if os.environ.get("GROQ_API_KEY"):
        r = results[idx]
        idx += 1
        if isinstance(r, Exception):
            out.append(ProviderResponse(provider_id="groq", content="", error=str(r)))
        else:
            out.extend(r)
    for i, p in enumerate(native):
        r = results[idx + i]
        if isinstance(r, Exception):
            out.append(ProviderResponse(provider_id=p.provider_id, content="", error=str(r)))
        else:
            out.append(r)

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
    Stream debate responses when using OpenRouter or Groq (one key, N models).
    Yields dicts: {"model_id": str, "delta": str} | {"model_id": str, "done": True} | {"model_id": str, "error": str}.
    If only native providers are configured, yields nothing (use run_debate_round instead).
    """
    from providers.openrouter_provider import (
        OPENROUTER_BASE,
        _get_models as _openrouter_models,
        _slug as _openrouter_slug,
    )
    from providers.groq_multi_provider import (
        GROQ_BASE,
        _get_models as _groq_models,
    )

    system = system_prompt or DEFAULT_SYSTEM
    openai_messages = [{"role": "system", "content": system}] if system else []
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    queue: asyncio.Queue = asyncio.Queue()
    expected_sentinels = 0

    if os.environ.get("OPENROUTER_API_KEY"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE)
        openrouter_models = _openrouter_models()
        if selected_models:
            openrouter_models = [m for m in openrouter_models if _openrouter_slug(m) in selected_models]
        for model_id in openrouter_models:
            slug = _openrouter_slug(model_id)
            expected_sentinels += 1

            async def stream_one(m_id: str, s: str):
                try:
                    stream = await client.chat.completions.create(
                        model=m_id,
                        messages=openai_messages,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        stream=True,
                    )
                    async for chunk in stream:
                        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                        if delta:
                            await queue.put({"model_id": s, "delta": delta})
                    await queue.put({"model_id": s, "done": True})
                except Exception as e:
                    await queue.put({"model_id": s, "error": str(e)})
                finally:
                    await queue.put(None)

            asyncio.create_task(stream_one(model_id, slug))

    if os.environ.get("GROQ_API_KEY"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE)
        groq_models = _groq_models()
        if selected_models:
            groq_models = [m for m in groq_models if m in selected_models]
        for model_id in groq_models:
            expected_sentinels += 1

            async def stream_one_groq(m_id: str):
                try:
                    stream = await client.chat.completions.create(
                        model=m_id,
                        messages=openai_messages,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        stream=True,
                    )
                    async for chunk in stream:
                        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                        if delta:
                            await queue.put({"model_id": m_id, "delta": delta})
                    await queue.put({"model_id": m_id, "done": True})
                except Exception as e:
                    await queue.put({"model_id": m_id, "error": str(e)})
                finally:
                    await queue.put(None)

            asyncio.create_task(stream_one_groq(model_id))

    if expected_sentinels == 0:
        return

    sentinels = 0
    while sentinels < expected_sentinels:
        ev = await queue.get()
        if ev is None:
            sentinels += 1
            continue
        yield ev
