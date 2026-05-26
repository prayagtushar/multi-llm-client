from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..schema import Message


class LLMProvider(ABC):
    """Abstract base for all LLM Providers"""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Return a complete LLMResponse"""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive."""
        ...
