import time
from collections.abc import AsyncIterator
from typing import Any

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..schema import FinishReason, LLMResponse, Message, Provider, Usage
from .base import LLMProvider

_FINISH_MAP = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALL,
}


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    @staticmethod
    def _split_messages(
        messages: list[Message], system: str | None
    ) -> tuple[list[dict[str, str]], str | None]:
        """Anthropic takes the system prompt as a top-level arg, not a message."""
        user_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        if not system:
            system_messages = [m for m in messages if m.role == "system"]
            if system_messages:
                system = system_messages[-1].content
        return user_messages, system

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        user_messages, extracted_system = self._split_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": user_messages,
            "temperature": temperature,
        }
        if extracted_system:
            kwargs["system"] = extracted_system

        start = time.perf_counter()
        response = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        finish = _FINISH_MAP.get(response.stop_reason or "", FinishReason.UNKNOWN)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            content=text,
            provider=Provider.ANTHROPIC,
            model=response.model,
            finish_reason=finish,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            latency_ms=latency_ms,
            request_id=response.id,
        )

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        user_messages, extracted_system = self._split_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": user_messages,
            "temperature": temperature,
        }
        if extracted_system:
            kwargs["system"] = extracted_system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
