import asyncio
import os
from .base import LLMProvider, Message, ProviderResponse

# Use the supported google-genai package (not deprecated google.generativeai)
from google import genai
from google.genai import types


def _gemini_sync_generate(client: genai.Client, model_name: str, prompt: str) -> str:
    r = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    return (r.text or "").strip()


class GeminiProvider(LLMProvider):
    provider_id = "gemini"

    def __init__(self):
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._model = "gemini-1.5-flash"

    async def chat(self, messages: list[Message], system_prompt: str | None = None) -> ProviderResponse:
        try:
            parts = []
            if system_prompt:
                parts.append(system_prompt)
            for m in messages:
                parts.append(f"{m.role}: {m.content}")
            prompt = "\n\n".join(parts)
            content = await asyncio.to_thread(
                _gemini_sync_generate, self._client, self._model, prompt
            )
            return ProviderResponse(provider_id=self.provider_id, content=content)
        except Exception as e:
            return ProviderResponse(provider_id=self.provider_id, content="", error=str(e))
