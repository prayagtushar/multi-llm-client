# llm-client

Unified async Python client for OpenAI, Anthropic, and Gemini.

Provides a single interface to interact with multiple LLM providers,
with support for both completion and streaming, consistent data models,
and pluggable provider architecture.

```text
src/llm_client/
├── __init__.py         # Package exports
├── api.py              # FastAPI REST API
├── cli.py              # Click command-line interface
├── client.py           # Unified client orchestrator
├── config.py           # pydantic-settings configuration
├── exceptions.py       # Error hierarchy
├── schema.py           # Shared data models
└── providers/
    ├── __init__.py     # Provider exports
    ├── base.py         # LLMProvider ABC
    ├── openai.py       # OpenAI provider
    ├── anthropic.py    # Anthropic provider
    ├── gemini.py       # Gemini provider
    └── registry.py     # Provider lookup factory
```
