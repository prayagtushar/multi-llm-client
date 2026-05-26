from enum import StrEnum

from pydantic import BaseModel


class Provider(StrEnum):
    OPEN_AI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class Usage(BaseModel):
    input_token: int
    output_token: int
    total_token: int


class Response(BaseModel):
    content: str
    provider: Provider
    model: str
    finish_reason = FinishReason
    usage: Usage
    latency: float
    request_id: str | None | None = None


class Message(BaseModel):
    role: str
    content: str
