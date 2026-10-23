"""API request and response models."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    tool_used: bool = False
    tool_name: str | None = None


class HealthResponse(BaseModel):
    status: str
    tracing_enabled: bool
    demo_faults_enabled: bool
