from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.llm_client import api
from src.llm_client.exceptions import ProviderNotConfiguredError
from src.llm_client.schema import FinishReason, LLMResponse, Provider, Usage


@pytest.fixture
def client():
    return TestClient(api.app)


def _response() -> LLMResponse:
    return LLMResponse(
        content="hello",
        provider=Provider.OPEN_AI,
        model="gpt-4o-mini",
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        latency_ms=12.0,
    )


def test_health_reports_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["providers"], list)


def test_complete_returns_response(client):
    with patch.object(api.llm, "complete", new=AsyncMock(return_value=_response())):
        resp = client.post(
            "/complete",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "hello"
    assert body["provider"] == "openai"
    assert body["usage"]["total_tokens"] == 2


def test_complete_unconfigured_provider_returns_400(client):
    with patch.object(
        api.llm,
        "complete",
        new=AsyncMock(side_effect=ProviderNotConfiguredError("nope")),
    ):
        resp = client.post(
            "/complete",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "gemini",
            },
        )
    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]


def test_stream_emits_sse(client, make_async_iter):
    def fake_stream(**kwargs):
        return make_async_iter(["Hel", "lo"])

    with patch.object(api.llm, "stream", new=fake_stream):
        resp = client.post(
            "/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    text = resp.text
    assert "data: Hel" in text
    assert "data: lo" in text
    assert "data: [DONE]" in text


def test_compare_aggregates_providers(client):
    async def fake_compare(messages, providers=None):
        return {
            Provider.OPEN_AI: _response(),
            Provider.ANTHROPIC: RuntimeError("boom"),
        }

    with patch.object(api.llm, "compare", new=fake_compare):
        resp = client.post(
            "/compare",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["openai"]["content"] == "hello"
    assert "error" in body["anthropic"]
