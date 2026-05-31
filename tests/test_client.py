import pytest

from src.llm_client.client import LLMClient
from src.llm_client.config import LLMConfig
from src.llm_client.exceptions import ProviderNotConfiguredError
from src.llm_client.providers.base import LLMProvider
from src.llm_client.schema import FinishReason, LLMResponse, Message, Provider, Usage


def _response(provider: Provider, content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        provider=provider,
        model="fake-model",
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        latency_ms=1.0,
    )


class FakeProvider(LLMProvider):
    def __init__(self, provider: Provider, *, fail: bool = False):
        self._provider = provider
        self._fail = fail
        self.calls = 0

    async def complete(self, messages, system=None, max_tokens=1024, temperature=0.7):
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return _response(self._provider, "ok")

    async def stream(self, messages, system=None, max_tokens=1024, temperature=0.7):
        for piece in ["a", "b"]:
            yield piece


@pytest.fixture
def empty_config():
    return LLMConfig(
        _env_file=None,
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
    )


def test_unconfigured_provider_raises(empty_config):
    client = LLMClient(empty_config)
    with pytest.raises(ProviderNotConfiguredError):
        client._get_provider(Provider.OPEN_AI)


async def test_complete_delegates_to_provider(empty_config):
    client = LLMClient(empty_config)
    fake = FakeProvider(Provider.OPEN_AI)
    client._providers[Provider.OPEN_AI] = fake

    result = await client.complete(
        [Message(role="user", content="hi")], provider=Provider.OPEN_AI
    )
    assert result.content == "ok"
    assert fake.calls == 1


async def test_default_provider_used_when_unspecified(empty_config):
    empty_config.default_provider = Provider.ANTHROPIC
    client = LLMClient(empty_config)
    client._providers[Provider.ANTHROPIC] = FakeProvider(Provider.ANTHROPIC)

    result = await client.complete([Message(role="user", content="hi")])
    assert result.provider == Provider.ANTHROPIC


async def test_stream_delegates_to_provider(empty_config):
    client = LLMClient(empty_config)
    client._providers[Provider.OPEN_AI] = FakeProvider(Provider.OPEN_AI)

    out = [
        c
        async for c in client.stream(
            [Message(role="user", content="hi")], provider=Provider.OPEN_AI
        )
    ]
    assert out == ["a", "b"]


async def test_compare_returns_results_and_captures_errors(empty_config):
    client = LLMClient(empty_config)
    client._providers[Provider.OPEN_AI] = FakeProvider(Provider.OPEN_AI)
    client._providers[Provider.ANTHROPIC] = FakeProvider(Provider.ANTHROPIC, fail=True)

    results = await client.compare([Message(role="user", content="hi")])

    assert isinstance(results[Provider.OPEN_AI], LLMResponse)
    assert isinstance(results[Provider.ANTHROPIC], Exception)
