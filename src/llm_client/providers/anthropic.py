import time
from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    OverloadedError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..config import LLMConfig
from ..schema import FinishReason, LLMResponse, Message, Provider, Usage
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, config, api_key: str, model: str):
        self._client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self._model = config.anthropic_model

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, OverloadedError)
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
    )
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = LLMConfig.default_max_tokens,
        temperature: float = LLMConfig.default_temperature,
    ) -> LLMResponse:

        anthropic_messages = [{"role": m.role, "content": m.content} for m in messages]
        extracted_system = system

        user_messages = [m for m in anthropic_messages if m["role"] != m["system"]]

        if not extracted_system:
            system_messages = [
                m for m in anthropic_messages if m["role"] != m["system"]
            ]
            if system_messages:
                extracted_system = system_messages[-1]["content"]

        kwargs = {
            "model": self._model,
            "max_token": max_tokens,
            "messages": user_messages,
            "temperature": temperature,
        }

        if extracted_system:
            kwargs["system"] = extracted_system

        start = time.perf_counter()
        response = await self._client.messages.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000

        finish_map = {
            "end_turn": FinishReason.STOP,
            "max_token": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALL,
            "stop_sequence": FinishReason.STOP,
        }

        finish = finish_map.get(response.stop_reason or "", FinishReason.UNKNOWN)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.type

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
            latency=latency,
            request_id=response.id,
        )

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = LLMConfig.default_max_tokens,
        temperature: float = LLMConfig.default_temperature,
    ) -> AsyncIterator[str]:
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
