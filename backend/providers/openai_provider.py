import os
from openai import AsyncOpenAI
from .base import LLMProvider, Message, ProviderResponse


class OpenAIProvider(LLMProvider):
    provider_id = "chatgpt"

    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def chat(self, messages: list[Message], system_prompt: str | None = None) -> ProviderResponse:
        try:
            openai_messages = []
            if system_prompt:
                openai_messages.append({"role": "system", "content": system_prompt})
            for m in messages:
                openai_messages.append({"role": m.role, "content": m.content})
            max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
            r = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                max_tokens=max_tokens,
            )
            content = r.choices[0].message.content or ""
            return ProviderResponse(provider_id=self.provider_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=self.provider_id, content="", error=str(e))
