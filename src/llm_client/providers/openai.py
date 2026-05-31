import time
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..schema import FinishReason, LLMResponse, Message, Provider, Usage
from .base import LLMProvider

_FINISH_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALL,
    "function_call": FinishReason.TOOL_CALL,
    "content_filter": FinishReason.SAFETY,
}


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    def _build_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, str]]:
        oai_messages: list[dict[str, str]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend({"role": m.role, "content": m.content} for m in messages)
        return oai_messages

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
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
        oai_messages = self._build_messages(messages, system)

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, oai_messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        raw_finish = response.choices[0].finish_reason
        finish = _FINISH_MAP.get(raw_finish or "", FinishReason.UNKNOWN)

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            provider=Provider.OPEN_AI,
            model=response.model,
            finish_reason=finish,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
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
        oai_messages = self._build_messages(messages, system)

        stream = cast(
            Any,
            await self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, oai_messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ),
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
