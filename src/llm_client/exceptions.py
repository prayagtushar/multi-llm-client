"""Error hierarchy for the llm_client package."""


class LLMClientError(Exception):
    """Base class for all llm_client errors."""


class ProviderNotConfiguredError(LLMClientError):
    """Raised when a requested provider has no API key configured."""


class ProviderError(LLMClientError):
    """Raised when an underlying provider call fails irrecoverably."""
