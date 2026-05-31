from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import stop_after_attempt

from src.llm_client.providers.openai import OpenAIProvider
from src.llm_client.schema import FinishReason, LLMResponse, Message, Provider


@pytest.fixture
def mock_openai_response():
    """Build a mock that looks like openai.types.chat.ChatCompletion."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "The answer is GPT-5.5."
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 8
    response.usage.total_tokens = 18
    response.model = "gpt-4o-mini"
    response.id = "chatcmpl-test123"
    return response


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="secret-key", model="gpt-4o-mini")


async def test_openai_complete_returns_normalized_response(
    openai_provider, mock_openai_response
):
    """complete() should return a normalized LLMResponse."""
    with patch.object(
        openai_provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_openai_response),
    ):
        result = await openai_provider.complete(
            messages=[Message(role="user", content="What's the latest OpenAI model?")]
        )

    assert isinstance(result, LLMResponse)
    assert result.content == "The answer is GPT-5.5."
    assert result.provider == Provider.OPEN_AI
    assert result.model == "gpt-4o-mini"
    assert result.finish_reason == FinishReason.STOP
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 8
    assert result.usage.total_tokens == 18
    assert result.latency_ms >= 0
    assert result.request_id == "chatcmpl-test123"


async def test_openai_complete_prepends_system_message(
    openai_provider, mock_openai_response
):
    """A system prompt should be prepended as a system-role message."""
    captured = {}

    async def capture_call(**kwargs):
        captured["messages"] = kwargs["messages"]
        return mock_openai_response

    with patch.object(
        openai_provider._client.chat.completions, "create", new=capture_call
    ):
        await openai_provider.complete(
            messages=[Message(role="user", content="Hello")],
            system="You are a pirate.",
        )

    assert captured["messages"][0] == {"role": "system", "content": "You are a pirate."}
    assert captured["messages"][1] == {"role": "user", "content": "Hello"}


async def test_openai_stream_yields_text_chunks(openai_provider, make_async_iter):
    """stream() should yield the text deltas from each chunk."""

    def chunk(text):
        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].delta.content = text
        return c

    chunks = make_async_iter([chunk("Hel"), chunk("lo"), chunk(None)])

    with patch.object(
        openai_provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=chunks),
    ):
        out = [
            piece
            async for piece in openai_provider.stream(
                messages=[Message(role="user", content="hi")]
            )
        ]

    assert out == ["Hel", "lo"]


async def test_openai_rate_limit_propagates_after_retries(openai_provider):
    """RateLimitError should propagate once retries are exhausted."""
    from openai import RateLimitError

    http_response = MagicMock()
    http_response.status_code = 429

    openai_provider.complete.retry.stop = stop_after_attempt(1)

    with patch.object(
        openai_provider._client.chat.completions,
        "create",
        new=AsyncMock(
            side_effect=RateLimitError(
                "rate limited",
                response=http_response,
                body={"error": {"message": "rate limited"}},
            )
        ),
    ):
        with pytest.raises(RateLimitError):
            await openai_provider.complete(
                messages=[Message(role="user", content="test")]
            )
