import os
from anthropic import AsyncAnthropic
from .base import LLMProvider, Message, ProviderResponse


class ClaudeProvider(LLMProvider):
    provider_id = "claude"

    def __init__(self):
        self._client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def chat(self, messages: list[Message], system_prompt: str | None = None) -> ProviderResponse:
        try:
            system = system_prompt or ""
            anthropic_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
            r = await self._client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                system=system,
                messages=anthropic_messages,
            )
            content = (r.content[0].text if r.content else "").strip()
            return ProviderResponse(provider_id=self.provider_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=self.provider_id, content="", error=str(e))
