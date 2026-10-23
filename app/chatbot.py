"""Chat orchestration: history, one OpenAI tool-calling step, and a parent trace."""

import logging
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.config import Settings
from app.datetime_tool import build_datetime_tool
from app.history import SessionStore
from app.prompts import SYSTEM_PROMPT
from app.tracing import run_config, run_metadata

logger = logging.getLogger(__name__)

PUBLIC_ERROR_MESSAGE = "The assistant could not complete this request. Please try again."
DEMO_FAULT_MESSAGE = "Simulated model failure for the LangSmith demo."
MISSING_KEY_MESSAGE = "OPENAI_API_KEY is not configured. Add it to the .env file and restart the server."


class ChatbotError(Exception):
    """A failure that is safe to show to the user."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


@dataclass
class ChatResult:
    answer: str
    session_id: str
    request_id: str
    tool_used: bool
    tool_name: str | None


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Construct the OpenAI chat model.

    Temperature is left unset so the API applies the model's own default.
    """
    if not settings.openai_api_key:
        raise ChatbotError(MISSING_KEY_MESSAGE)
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        max_retries=2,
        timeout=60.0,
    )


def message_text(message: AIMessage) -> str:
    """Visible reply text, without provider thinking blocks."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") not in {"thinking", "reasoning"}:
                text = block.get("text") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _trace_inputs(inputs: dict) -> dict:
    """Record the user question and session, never the service object or credentials."""
    history = inputs.get("history") or []
    return {
        "question": inputs.get("question"),
        "session_id": inputs.get("session_id"),
        "request_id": inputs.get("request_id"),
        "demo_fault": inputs.get("demo_fault"),
        "history": history,
    }


class ChatService:
    """One in-process chatbot. Pass `model` in tests to avoid network calls."""

    def __init__(self, settings: Settings, model=None, tool=None, store: SessionStore | None = None) -> None:
        self.settings = settings
        self.model = model
        self.tool = tool or build_datetime_tool(settings.app_timezone)
        self.store = store or SessionStore(max_messages=settings.history_max_messages)

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        request_id: str | None = None,
        demo_fault: str | None = None,
    ) -> ChatResult:
        """Answer one user message and store the turn in that session only."""
        session_id = session_id or str(uuid.uuid4())
        request_id = request_id or str(uuid.uuid4())
        history = self.store.get(session_id)
        try:
            payload = self._chat_request(
                question=question,
                history=history,
                session_id=session_id,
                request_id=request_id,
                demo_fault=demo_fault,
            )
        except ChatbotError:
            raise
        except Exception as exc:
            logger.exception("Chat request failed")
            raise ChatbotError(PUBLIC_ERROR_MESSAGE) from exc
        self.store.append_turn(session_id, question, payload["answer"])
        return ChatResult(
            answer=payload["answer"],
            session_id=session_id,
            request_id=request_id,
            tool_used=payload["tool_used"],
            tool_name=payload["tool_name"],
        )

    @traceable(name="chat_request", run_type="chain", process_inputs=_trace_inputs)
    def _chat_request(
        self,
        question: str,
        history: list[dict[str, str]],
        session_id: str,
        request_id: str,
        demo_fault: str | None = None,
    ) -> dict:
        """Parent trace for one request. Model and tool calls nest underneath it."""
        self._annotate(session_id, request_id, "chat")
        if demo_fault == "model":
            raise RuntimeError(DEMO_FAULT_MESSAGE)

        model = self._require_model()
        messages = _conversation(history, question)
        decision = model.bind_tools([self.tool]).invoke(
            messages,
            config=run_config(
                self.settings,
                name="decide_tool_use",
                session_id=session_id,
                request_id=request_id,
                workflow_type="chat",
            ),
        )
        tool_call = _datetime_call(decision)
        if tool_call is None:
            answer = message_text(decision) or "The model returned an empty response. Please try again."
            return {"answer": answer, "tool_used": False, "tool_name": None}

        args = tool_call.get("args") or {}
        if isinstance(args, str):
            args = {}
        tool_output = self.tool.invoke(
            {"unused": args.get("unused", "")},
            config=run_config(
                self.settings,
                name="get_current_datetime",
                session_id=session_id,
                request_id=request_id,
                workflow_type="tool",
            ),
        )
        messages.append(decision)
        messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call.get("id") or "get_current_datetime",
            )
        )
        final = model.invoke(
            messages,
            config=run_config(
                self.settings,
                name="final_answer",
                session_id=session_id,
                request_id=request_id,
                workflow_type="tool",
            ),
        )
        self._annotate(session_id, request_id, "tool")
        answer = message_text(final) or "The model returned an empty response. Please try again."
        return {
            "answer": answer,
            "tool_used": True,
            "tool_name": "get_current_datetime",
        }

    def _require_model(self):
        if self.model is None:
            self.model = build_chat_model(self.settings)
        return self.model

    def _annotate(self, session_id: str, request_id: str, workflow_type: str) -> None:
        run = get_current_run_tree()
        if run is None:
            return
        run.metadata.update(
            run_metadata(
                self.settings,
                session_id=session_id,
                request_id=request_id,
                workflow_type=workflow_type,
            )
        )
        tags = list(run.tags or [])
        for tag in ("mlops-chatbot-demo", workflow_type):
            if tag not in tags:
                tags.append(tag)
        run.tags = tags


def _conversation(history: list[dict[str, str]], question: str) -> list:
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=question))
    return messages


def _datetime_call(message: AIMessage) -> dict | None:
    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") == "get_current_datetime":
            return call
    return None
