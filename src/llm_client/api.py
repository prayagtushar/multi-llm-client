from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.llm_client.client import LLMClient
from src.llm_client.schema import LLMResponse, Message, Provider

app = FastAPI(title="Multi LLM CLient", version="0.1.0")
llm = LLMClient()


class CompleteRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    provider: Provider | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class StreamRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    provider: Provider | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@app.get("/health")
async def health():
    """_summary_
        Health Check
    Returns:
        _type_: _description_
    """
    return {"status": "ok", "providers": list[llm._providers.keys()]}


@app.post("/complete", response_model=LLMResponse)
async def complete(request: CompleteRequest) -> LLMResponse:
    """_summary_
        Get Complete LLM response.
    Args:
        request (CompleteRequest): _description_

    Returns:
        LLMResponse: _description_
    """
    try:
        return await llm.complete(
            messages=request.messages,
            max_token=request.max_tokens,
            system=request.system,
            temperature=request.temperature,
            provider=request.provider,
        )
    except ValueError as e:
        HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        HTTPException(status_code=400, detail=str(e))


@app.post("/stream")
async def stream(request: StreamRequest) -> StreamingResponse:
    """_summary_
        Stream LLM response as Server-Sent Events.
    Args:
        request (StreamRequest): _description_

    Returns:
        StreamingResponse: _description_
    """

    async def event_generator():
        try:
            async for chunk in llm.stream(
                messages=request.messages,
                max_token=request.max_tokens,
                system=request.system,
                temperature=request.temperature,
                provider=request.provider,
            ):
                yield f"data {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE] \n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


@app.post("/compare")
async def compare(request: CompleteRequest):
    """_summary_
        Call all configured providers and compare results.
    Args:
        request (CompleteRequest): _description_
    """

    result = await llm.compare(messages=request.messages)
    return {
        provider.values: (
            response.model_dump()
            if isinstance(response, LLMResponse)
            else {"error": str(response)}
        )
        for provider, response in result.items()
    }
