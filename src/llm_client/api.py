from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .client import LLMClient
from .exceptions import ProviderNotConfiguredError
from .schema import LLMResponse, Message, Provider

app = FastAPI(title="Multi LLM Client", version="0.1.0")
llm = LLMClient()


class CompleteRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    provider: Provider | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class StreamRequest(CompleteRequest):
    pass


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check — reports which providers are configured."""
    return {"status": "ok", "providers": list(llm._providers.keys())}


@app.post("/complete", response_model=LLMResponse)
async def complete(request: CompleteRequest) -> LLMResponse:
    """Get a complete LLM response."""
    try:
        return await llm.complete(
            messages=request.messages,
            max_tokens=request.max_tokens,
            system=request.system,
            temperature=request.temperature,
            provider=request.provider,
        )
    except ProviderNotConfiguredError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/stream")
async def stream(request: StreamRequest) -> StreamingResponse:
    """Stream an LLM response as Server-Sent Events."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in llm.stream(
                messages=request.messages,
                max_tokens=request.max_tokens,
                system=request.system,
                temperature=request.temperature,
                provider=request.provider,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/compare")
async def compare(request: CompleteRequest) -> dict[str, Any]:
    """Call all configured providers and compare results."""
    result = await llm.compare(messages=request.messages)
    return {
        provider.value: (
            response.model_dump()
            if isinstance(response, LLMResponse)
            else {"error": str(response)}
        )
        for provider, response in result.items()
    }
