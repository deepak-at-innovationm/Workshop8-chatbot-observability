"""HTTP routes for the demo UI and chat API."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.chatbot import ChatService, ChatbotError
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.tracing import apply_runtime_env

logger = logging.getLogger(__name__)
router = APIRouter()
_STATIC = Path(__file__).resolve().parent / "static"
_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """Process-wide chatbot. Tests replace this dependency."""
    global _service
    if _service is None:
        settings = get_settings()
        apply_runtime_env(settings)
        _service = ChatService(settings)
    return _service


def reset_chat_service() -> None:
    """Drop the cached service. Used by tests."""
    global _service
    _service = None


@router.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@router.get("/health", response_model=HealthResponse)
def health(service: ChatService = Depends(get_chat_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        tracing_enabled=service.settings.langsmith_tracing,
        demo_faults_enabled=service.settings.allow_demo_faults,
    )


@router.post("/api/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    demo_fault = None
    if service.settings.allow_demo_faults:
        header = request.headers.get("x-demo-fault", "").strip().lower()
        if header:
            demo_fault = header
    try:
        result = service.answer(body.message, session_id=body.session_id, demo_fault=demo_fault)
    except ChatbotError as exc:
        logger.error("Chat request returned an error to the client")
        raise HTTPException(status_code=500, detail=exc.public_message) from exc
    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        request_id=result.request_id,
        tool_used=result.tool_used,
        tool_name=result.tool_name,
    )
