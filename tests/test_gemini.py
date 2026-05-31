from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_client.providers.gemini import GeminiProvider
from src.llm_client.schema import FinishReason, LLMResponse, Message, Provider


@pytest.fixture
def mock_gemini_response():
    """Mock resembling google.genai GenerateContentResponse."""
    response = MagicMock()
    candidate = MagicMock()
    candidate.finish_reason.name = "STOP"
    part = MagicMock()
    part.text = "Hello from Gemini."
    candidate.content.parts = [part]
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = 3
    response.usage_metadata.candidates_token_count = 4
    response.usage_metadata.total_token_count = 7
    return response


@pytest.fixture
def gemini_provider():
    return GeminiProvider(api_key="secret-key", model="gemini-2.5-flash")


async def test_gemini_complete_returns_normalized_response(
    gemini_provider, mock_gemini_response
):
    with patch.object(
        gemini_provider._client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_gemini_response),
    ):
        result = await gemini_provider.complete(
            messages=[Message(role="user", content="Hi")]
        )

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from Gemini."
    assert result.provider == Provider.GEMINI
    assert result.model == "gemini-2.5-flash"
    assert result.finish_reason == FinishReason.STOP
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 7


async def test_gemini_format_content_maps_roles_and_skips_system(gemini_provider):
    formatted = gemini_provider._format_content(
        [
            Message(role="system", content="ignored"),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
    )
    assert formatted == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello"}]},
    ]


async def test_gemini_stream_yields_text(gemini_provider, make_async_iter):
    def chunk(text):
        c = MagicMock()
        c.text = text
        return c

    chunks = make_async_iter([chunk("Hel"), chunk("lo"), chunk("")])

    with patch.object(
        gemini_provider._client.aio.models,
        "generate_content_stream",
        new=AsyncMock(return_value=chunks),
    ):
        out = [
            piece
            async for piece in gemini_provider.stream(
                messages=[Message(role="user", content="hi")]
            )
        ]

    assert out == ["Hel", "lo"]
