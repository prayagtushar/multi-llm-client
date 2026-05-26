from pydantic_settings import BaseSettings
from schema import Provider


class LLMConfig(BaseSettings):
    default_provider: Provider = Provider.ANTHROPIC

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str = None

    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"

    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    max_retries: int = 5

    model_config = {"env_prefix": "", "env_file": ".env"}
