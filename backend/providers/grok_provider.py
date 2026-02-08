import os
from openai import AsyncOpenAI
from .base import LLMProvider, Message, ProviderResponse


class GrokProvider(LLMProvider):
    provider_id = "grok"

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

    async def chat(self, messages: list[Message], system_prompt: str | None = None) -> ProviderResponse:
        try:
            openai_messages = []
            if system_prompt:
                openai_messages.append({"role": "system", "content": system_prompt})
            for m in messages:
                openai_messages.append({"role": m.role, "content": m.content})
            model = "grok-3"
            r = await self._client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=2048,
            )
            content = r.choices[0].message.content or ""
            return ProviderResponse(provider_id=self.provider_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=self.provider_id, content="", error=str(e))
