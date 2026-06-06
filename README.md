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

## Highlights

- One async interface over **OpenAI, Anthropic, and Gemini** — normalizes messages, streaming, token usage, and errors across all three.
- Four surfaces from one core: Python **library**, **CLI** (`llm`), interactive **REPL** (`multi-llm`), and a **FastAPI** service.
- **Tenacity** exponential-backoff retries (5 attempts) with provider-specific error mapping and a custom exception hierarchy.
- Type-safe throughout: **Pydantic v2** models + `pydantic-settings` config + **mypy-strict**.
- `compare()` races a prompt across all configured providers **concurrently** (asyncio.gather) with per-call latency + request IDs.
- **40 tests** across client / provider adapters / API / REPL; Python 3.11+.

## Demo

> _Demo GIF/asciinema coming soon._

<!-- TODO: add a CLI/REPL asciinema or GIF at docs/demo.gif and embed here -->

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

## CLI (one-shot)

```bash
llm ask "What is the capital of France?" --provider openai
llm ask "Tell me a story" --stream
llm compare "Explain recursion in one sentence"
```

## Interactive shell

A persistent REPL with multi-turn conversation memory:

```bash
multi-llm
```

Type a prompt to chat; the conversation is remembered across turns. Manage the
session with slash commands:

```
/help                 show all commands
/provider openai      switch the active provider
/system <text>        set a system prompt   (/system clear to remove)
/temp 0.2  /max 256   set sampling params
/stream on|off        toggle streaming
/markdown on|off      toggle markdown rendering (on by default)
/compare <prompt>     ask every configured provider at once
/history  /clear      view or reset the conversation
/exit                 leave (also Ctrl-D / Ctrl-C)
```

Answers are rendered as formatted markdown (bold, headings, lists, syntax-
highlighted code) via [rich](https://github.com/Textualize/rich) — including
live-rendered while streaming. It also has arrow-key history (saved to
`~/.multi_llm_history`) and Tab-completion for commands and provider names.

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

