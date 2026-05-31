import time
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai.errors import APIError, ServerError
from google.genai.types import GenerateContentConfig
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ..schema import FinishReason, LLMResponse, Message, Provider, Usage
from .base import LLMProvider

_FINISH_MAP = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.SAFETY,
    "RECITATION": FinishReason.SAFETY,
    "OTHER": FinishReason.UNKNOWN,
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, APIError):
        return getattr(exc, "code", None) == 429
    return False


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _format_content(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert our Message list to Gemini's content format."""
        content: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            content.append({"role": role, "parts": [{"text": m.content}]})
        return content

    def _resolve_system(
        self, messages: list[Message], system: str | None
    ) -> str | None:
        if system:
            return system
        system_messages = [m for m in messages if m.role == "system"]
        if system_messages:
            return system_messages[-1].content
        return None

    @retry(
        retry=retry_if_exception(_is_retryable),
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
        contents = self._format_content(messages)
        system = self._resolve_system(messages, system)

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system

        start = time.perf_counter()
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=GenerateContentConfig(**config_kwargs),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        raw_finish = (
            response.candidates[0].finish_reason.name
            if response.candidates and response.candidates[0].finish_reason
            else "UNKNOWN"
        )
        finish = _FINISH_MAP.get(raw_finish, FinishReason.UNKNOWN)

        text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    text += part_text

        meta = response.usage_metadata
        return LLMResponse(
            content=text,
            provider=Provider.GEMINI,
            model=self._model,
            finish_reason=finish,
            usage=Usage(
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                total_tokens=getattr(meta, "total_token_count", 0) or 0,
            ),
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        contents = self._format_content(messages)
        system = self._resolve_system(messages, system)

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system

        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=GenerateContentConfig(**config_kwargs),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
