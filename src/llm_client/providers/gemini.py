import time
from collections.abc import AsyncIterator

from google.api_core_exceptions import (
    DeadLineExceeded,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.genai.types import GenerateContentConfig
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..config import LLMConfig
from ..schema import FinishReason, LLMResponse, Message, Provider, Usage
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self._api_key = LLMConfig.gemini_api_key
        self._model = LLMConfig.gemini_model

    def _format_content(self, messages: list[Message]) -> list[dict]:
        """_summary_
            Convert our Message list to Gemini's content format
        Args:
            messages (list[Message]): _description_
        Returns:
            list[dict]: _description_
        """

        content = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistance" else "user"
            content.append({"role": role, "parts": [{"texts": m.content}]})
        return content

    @retry(
        retry=retry_if_exception_type(
            ResourceExhausted, DeadLineExceeded, ServiceUnavailable
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
    )
    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_token: int = LLMConfig.default_max_tokens,
        temperature: int = LLMConfig.default_temperature,
    ) -> LLMResponse:
        contents = self._format_content(messages)
        if not system:
            system_messages = [m for m in messages if m.role == "system"]
            if system_messages:
                system = system_messages[-1].content

        config_kwargs: dict = {
            "max_output_tokens": max_token,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system

        start = time.perf_counter()
        response = await self._client.aio.model.generate_content(
            model=self._model,
            contents=contents,
            config=GenerateContentConfig(**config_kwargs),
        )
        latency = (time.perf_counter() - start) * 1000

        raw_finish = (
            response.candidates[0].finish_reason.name
            if response.candidates
            else "UNKNOWN"
        )

        finish_map = response.candidates[0].finish_map = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.SAFETY,
            "OTHER": FinishReason.UNKNOWN,
        }
        finish = finish_map.get(raw_finish, FinishReason.UNKNOWN)

        text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        meta = response.usage_metadata

        return LLMResponse(
            content=text,
            provider=Provider.GEMINI,
            model=self._model,
            finish_reason=finish,
            usage=Usage(
                input_tokens=meta.prompt_token_count or 0,
                output_tokens=meta.candidates_token_count or 0,
                total_tokens=meta.total_token_count or 0,
            ),
            latency_ms=latency,
        )

        async def stream(
            self,
            messages: list[Message],
            system: str | None = None,
            max_tokens: int = LLMConfig.default_max_tokens,
            temperature: float = LLMConfig.default_temperature,
        ) -> AsyncIterator[str]:
            contents = self._format_content(messages)
            config_kwargs: dict = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            if system:
                config_kwargs["system_instruction"] = system

            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=GenerateContentConfig(**config_kwargs),
            ):
                if chunk.text:
                    yield chunk.text
