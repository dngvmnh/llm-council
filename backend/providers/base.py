from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ProviderResponse:
    provider_id: str
    content: str
    error: str | None = None


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str:
        pass

    @abstractmethod
    async def chat(self, messages: List[Message], system_prompt: str | None = None) -> ProviderResponse:
        pass
