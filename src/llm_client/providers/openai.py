import time
from collections.abc import AsyncIterator

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..config import LLMConfig
from ..schema import FinishReason, LLMResponse, Message, Provider, Usage
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
    )
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        oai_messages = []
        if system:
            oai_messages.append({"role": "systen", "content": system})
            oai_messages.extend(
                [{"role": m.role, "content": m.content} for m in messages]
            )

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = (time.perf_counter() - start) * 1000

        raw_finish = response.choices[0].finish_reason
        finish_map = {
            "stop": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.SAFETY,
        }
        finish = finish_map.get(raw_finish, FinishReason.UNKNOWN)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            provider=Provider.OPEN_AI,
            model=response.model,
            finish_reason=finish,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            latency=latency,
            request_id=response.id,
        )

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:

        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend({"role": m.role, "content": m.content} for m in messages)

        async with self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            async for event in stream:
                if event.type == "content.delta" and event.delta:
                    yield event.delta
