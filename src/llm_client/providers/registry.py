"""Provider lookup factory.

Builds the set of configured providers from an ``LLMConfig``. A provider is
only instantiated when its API key is present, so callers can run with any
subset of the three backends.
"""

from ..config import LLMConfig
from ..schema import Provider
from .anthropic import AnthropicProvider
from .base import LLMProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider


def build_providers(config: LLMConfig) -> dict[Provider, LLMProvider]:
    """Instantiate every provider that has an API key configured."""
    providers: dict[Provider, LLMProvider] = {}

    if config.openai_api_key:
        providers[Provider.OPEN_AI] = OpenAIProvider(
            api_key=config.openai_api_key, model=config.openai_model
        )
    if config.anthropic_api_key:
        providers[Provider.ANTHROPIC] = AnthropicProvider(
            api_key=config.anthropic_api_key, model=config.anthropic_model
        )
    if config.gemini_api_key:
        providers[Provider.GEMINI] = GeminiProvider(
            api_key=config.gemini_api_key, model=config.gemini_model
        )

    return providers
