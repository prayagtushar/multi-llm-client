from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_client.providers.anthropic import AnthropicProvider
from src.llm_client.schema import FinishReason, LLMResponse, Message, Provider


@pytest.fixture
def mock_anthropic_response():
    """Mock resembling anthropic.types.Message."""
    response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "Ahoy there!"
    response.content = [block]
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 5
    response.usage.output_tokens = 7
    response.model = "claude-haiku-4-5-20251001"
    response.id = "msg_test123"
    return response


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(api_key="secret-key", model="claude-haiku-4-5-20251001")


async def test_anthropic_complete_returns_normalized_response(
    anthropic_provider, mock_anthropic_response
):
    with patch.object(
        anthropic_provider._client.messages,
        "create",
        new=AsyncMock(return_value=mock_anthropic_response),
    ):
        result = await anthropic_provider.complete(
            messages=[Message(role="user", content="Hi")]
        )

    assert isinstance(result, LLMResponse)
    assert result.content == "Ahoy there!"
    assert result.provider == Provider.ANTHROPIC
    assert result.finish_reason == FinishReason.STOP
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 7
    assert result.usage.total_tokens == 12
    assert result.request_id == "msg_test123"


async def test_anthropic_extracts_system_and_filters_system_messages(
    anthropic_provider, mock_anthropic_response
):
    """System messages must be lifted into the top-level `system` arg."""
    captured = {}

    async def capture_call(**kwargs):
        captured.update(kwargs)
        return mock_anthropic_response

    with patch.object(anthropic_provider._client.messages, "create", new=capture_call):
        await anthropic_provider.complete(
            messages=[
                Message(role="system", content="Be terse."),
                Message(role="user", content="Hello"),
            ]
        )

    assert captured["system"] == "Be terse."
    assert captured["messages"] == [{"role": "user", "content": "Hello"}]


async def test_anthropic_explicit_system_takes_precedence(
    anthropic_provider, mock_anthropic_response
):
    captured = {}

    async def capture_call(**kwargs):
        captured.update(kwargs)
        return mock_anthropic_response

    with patch.object(anthropic_provider._client.messages, "create", new=capture_call):
        await anthropic_provider.complete(
            messages=[Message(role="user", content="Hello")],
            system="Explicit system.",
        )

    assert captured["system"] == "Explicit system."


async def test_anthropic_stream_yields_text(anthropic_provider, make_async_iter):
    @asynccontextmanager
    async def fake_stream(**kwargs):
        manager = MagicMock()
        manager.text_stream = make_async_iter(["Ah", "oy"])
        yield manager

    with patch.object(anthropic_provider._client.messages, "stream", new=fake_stream):
        out = [
            piece
            async for piece in anthropic_provider.stream(
                messages=[Message(role="user", content="hi")]
            )
        ]

    assert out == ["Ah", "oy"]
