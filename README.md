# multi llm client

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

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys you have
```

Configuration is read from environment variables / `.env` (see `config.py`):
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and optional
`DEFAULT_PROVIDER`. Only providers with a key set are activated.

## Python usage

```python
import asyncio
from llm_client import LLMClient, Message, Provider

client = LLMClient()

async def main():
    resp = await client.complete(
        [Message(role="user", content="Hello!")],
        provider=Provider.ANTHROPIC,
    )
    print(resp.content, resp.usage.total_tokens, resp.latency_ms)

    async for chunk in client.stream([Message(role="user", content="Stream this")]):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## CLI

```bash
llm ask "What is the capital of France?" --provider openai
llm ask "Tell me a story" --stream
llm compare "Explain recursion in one sentence"
```

## REST API

```bash
uvicorn llm_client.api:app --reload
# GET  /health   POST /complete   POST /stream (SSE)   POST /compare
```

## Development

```bash
pytest          # run the unit tests (fully mocked, no network)
ruff check .    # lint
ruff format .   # format
mypy src        # strict type-check
```

