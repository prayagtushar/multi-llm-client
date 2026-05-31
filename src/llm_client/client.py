import asyncio
from collections.abc import AsyncIterator

from .config import LLMConfig
from .exceptions import ProviderNotConfiguredError
from .providers.base import LLMProvider
from .providers.registry import build_providers
from .schema import LLMResponse, Message, Provider


class LLMClient:
    """Entry point for all LLM calls. Provider is swappable via config."""

    def __init__(self, config: LLMConfig | None = None):
        self._config = config or LLMConfig()
        self._providers: dict[Provider, LLMProvider] = build_providers(self._config)

    def _get_provider(self, provider: Provider | None = None) -> LLMProvider:
        p = provider or self._config.default_provider
        if p not in self._providers:
            raise ProviderNotConfiguredError(
                f"Provider {p} not configured. Set the API key in config/.env."
            )
        return self._providers[p]

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        provider: Provider | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return await self._get_provider(provider).complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens or self._config.default_max_tokens,
            temperature=temperature
            if temperature is not None
            else self._config.default_temperature,
        )

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        provider: Provider | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive."""
        async for chunk in self._get_provider(provider).stream(
            messages=messages,
            system=system,
            max_tokens=max_tokens or self._config.default_max_tokens,
            temperature=temperature
            if temperature is not None
            else self._config.default_temperature,
        ):
            yield chunk

    async def compare(
        self,
        messages: list[Message],
        providers: list[Provider] | None = None,
    ) -> dict[Provider, LLMResponse | BaseException]:
        """Call the selected providers concurrently and return results.

        A failing provider yields its ``Exception`` instead of an
        ``LLMResponse`` so one bad backend never sinks the whole comparison.
        """
        targets = providers or list(self._providers.keys())
        tasks = {p: self.complete(messages, provider=p) for p in targets}

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))
