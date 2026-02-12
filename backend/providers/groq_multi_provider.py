"""
Groq: one API key, multiple models (Llama, Mixtral, etc.).
Set GROQ_API_KEY and optionally GROQ_DEBATE_MODELS (comma-separated).
"""
import os
from openai import AsyncOpenAI
from .base import Message, ProviderResponse

GROQ_BASE = "https://api.groq.com/openai/v1"

# Default: a few Groq-hosted models for variety
# Note: mixtral-8x7b-32768 was decommissioned; removed from defaults
DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def _get_models() -> list[str]:
    raw = os.environ.get("GROQ_DEBATE_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return DEFAULT_GROQ_MODELS


async def get_groq_responses(
    messages: list[Message],
    system_prompt: str | None = None,
) -> list[ProviderResponse]:
    """Call Groq for each configured model in parallel. One API key, N responses."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return []
    models = _get_models()
    if not models:
        return []
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE)
    system = system_prompt or ""

    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))

    async def one_model(model_id: str) -> ProviderResponse:
        try:
            r = await client.chat.completions.create(
                model=model_id,
                messages=openai_messages,
                max_tokens=max_tokens,
            )
            content = (r.choices[0].message.content or "").strip()
            return ProviderResponse(provider_id=model_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=model_id, content="", error=str(e))

    import asyncio
    results = await asyncio.gather(*[one_model(m) for m in models], return_exceptions=True)
    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append(ProviderResponse(provider_id=models[i], content="", error=str(r)))
        else:
            out.append(r)
    return out
