import asyncio

from src.llm_client.config import LLMConfig
from src.llm_client.providers.anthropic import AnthropicProvider
from src.llm_client.providers.base import LLMProvider, LLMResponse
from src.llm_client.providers.gemini import GeminiProvider
from src.llm_client.providers.openai import OpenAIProvider
from src.llm_client.schema import Message, Provider


class LLMClient:
    """_summary_
    Entry for all LLM Calls. Provider is swappable via config.
    """

    def __init__(self, config: LLMConfig | None = None):
        self._config = LLMConfig()
        self._provider: dict[Provider, LLMProvider] = {}
        self._init_providers()

    def _init_providers(self):
        config = self._confg

        if config.openai_api_key:
            self._provider[Provider.OPEN_AI] = OpenAIProvider(
                api_key=config.openai_api_key, model=config.openai_model
            )

        if config.anthropic_api_key:
            self._provider[Provider.ANTHROPIC] = AnthropicProvider(
                api_key=config.anthropic_api_key, model=config.anthropic_model
            )

        if config.gemini_api_key:
            self._provider[Provider.GEMINI] = GeminiProvider(
                api_key=config.gemini_api_key, model=config.gemini_model
            )

    def _get_providers(self, provider: Provider | None = None) -> LLMProvider:
        p = provider or self._confg.default_provider
        if p not in self._providers:
            raise ValueError(f"Provider {p} not configured. Set the API in config.")
        return self._providers[p]

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        provider: Provider | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:

        return await self._get_providers[provider].complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens or self._config.default_max_tokens,
            temperature=temperature or self._config.default_temperature,
        )

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        provider: Provider | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        """AsyncIterator[str] — yields text chunks."""
        async for chunk in self._get_providers(provider).stream(
            messages=messages,
            system=system,
            max_tokens=max_tokens or self._config.default_max_tokens,
            temperature=temperature or self._config.default_temperature,
        ):
            yield chunk

    async def compare(
        self, messages: list[Message], providers: list[Provider] | None = None
    ) -> dict[Provider, LLMResponse]:
        """Call all configured providers concurrently and return results."""
        targets = providers or list(self._providers.keys())
        tasks = {p: self.complete(messages, provider=p) for p in targets}

        results = await asyncio.gather(*tasks.values(), return_exception=True)
        return {p: r for p, r in zip(tasks.keys(), results)}
