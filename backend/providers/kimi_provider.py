import os
import httpx
from .base import LLMProvider, Message, ProviderResponse

MOONSHOT_BASE = "https://api.moonshot.cn/v1"


class KimiProvider(LLMProvider):
    provider_id = "kimi"

    def __init__(self):
        self._api_key = os.environ.get("MOONSHOT_API_KEY")
        self._model = "moonshot-v1-128k"

    async def chat(self, messages: list[Message], system_prompt: str | None = None) -> ProviderResponse:
        try:
            openai_messages = []
            if system_prompt:
                openai_messages.append({"role": "system", "content": system_prompt})
            for m in messages:
                openai_messages.append({"role": m.role, "content": m.content})
            async with httpx.AsyncClient(timeout=60.0) as client:
                max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))

                r = await client.post(
                    f"{MOONSHOT_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": openai_messages,
                        "max_tokens": max_tokens,
                    },
                )
                r.raise_for_status()
                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            return ProviderResponse(provider_id=self.provider_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=self.provider_id, content="", error=str(e))
