"""
OpenRouter: one API key, many models (OpenAI, Anthropic, Google, etc.).
Set OPENROUTER_API_KEY and OPENROUTER_DEBATE_MODELS (comma-separated model IDs).
"""
import os
from openai import AsyncOpenAI
from .base import Message, ProviderResponse

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Default: one cheap/fast model per "family" for debate diversity
DEFAULT_OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.5-flash",  # Valid OpenRouter model ID
    "x-ai/grok-3",
    "moonshotai/kimi-k2",
]


def _get_models() -> list[str]:
    raw = os.environ.get("OPENROUTER_DEBATE_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return DEFAULT_OPENROUTER_MODELS


def _slug(model_id: str) -> str:
    """Use last part of model id as display id (e.g. openai/gpt-4o-mini -> gpt-4o-mini)."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


async def get_openrouter_responses(
    messages: list[Message],
    system_prompt: str | None = None,
) -> list[ProviderResponse]:
    """Call OpenRouter for each configured model in parallel. One API key, N responses."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return []
    models = _get_models()
    if not models:
        return []
    client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE)
    system = system_prompt or ""

    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    async def one_model(model_id: str) -> ProviderResponse:
        try:
            r = await client.chat.completions.create(
                model=model_id,
                messages=openai_messages,
                max_tokens=2048,
            )
            content = (r.choices[0].message.content or "").strip()
            return ProviderResponse(provider_id=_slug(model_id), content=content)
        except Exception as e:
            return ProviderResponse(provider_id=_slug(model_id), content="", error=str(e))

    import asyncio
    results = await asyncio.gather(*[one_model(m) for m in models], return_exceptions=True)
    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append(ProviderResponse(provider_id=_slug(models[i]), content="", error=str(r)))
        else:
            out.append(r)
    return out
