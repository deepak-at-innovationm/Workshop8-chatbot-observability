import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.chatbot import (
    DEMO_FAULT_MESSAGE,
    MISSING_KEY_MESSAGE,
    PUBLIC_ERROR_MESSAGE,
    ChatService,
    ChatbotError,
    message_text,
)
from app.config import Settings
from app.prompts import SYSTEM_PROMPT
from tests.fakes import ScriptedChatModel


def _settings(**overrides) -> Settings:
    data = {
        "openai_api_key": "",
        "langsmith_tracing": False,
        "langsmith_api_key": "",
        "allow_demo_faults": False,
        "history_max_messages": 10,
        "app_timezone": "Asia/Kolkata",
    }
    data.update(overrides)
    return Settings(**data)


def _service(replies, **settings):
    model = ScriptedChatModel(replies)
    service = ChatService(_settings(**settings), model=model)
    return service, model


def test_system_prompt_reserves_the_tool_for_clock_questions():
    assert "get_current_datetime" in SYSTEM_PROMPT
    assert "Do not call the tool" in SYSTEM_PROMPT


def test_message_text_skips_thinking_blocks():
    message = AIMessage(
        content=[
            {"type": "thinking", "text": "hidden"},
            {"type": "text", "text": "Visible answer"},
        ]
    )
    assert message_text(message) == "Visible answer"


def test_general_question_does_not_call_the_tool():
    service, model = _service([AIMessage(content="MLOps is the practice of running machine learning in production.")])
    result = service.answer("What is MLOps?", session_id="s1")
    assert result.tool_used is False
    assert result.tool_name is None
    assert "MLOps" in result.answer
    assert model.invocations[0]["tools"]
    assert model.invocations[0]["tools"][0].name == "get_current_datetime"
    assert len(model.invocations) == 1


def test_clock_question_does_not_call_the_tool_unless_the_model_requests_it():
    service, model = _service([AIMessage(content="I did not call a tool.")])
    result = service.answer("What time is it now?", session_id="s1")
    assert result.tool_used is False
    assert "did not call" in result.answer
    assert len(model.invocations) == 1


def test_model_tool_call_executes_the_real_clock():
    service, model = _service(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_datetime",
                        "args": {"unused": ""},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The clock result is shown above."),
        ]
    )
    result = service.answer("What is the current date and time?", session_id="clock")
    assert result.tool_used is True
    assert result.tool_name == "get_current_datetime"
    assert len(model.invocations) == 2
    assert model.invocations[1]["tools"] is None
    tool_messages = [message for message in model.invocations[1]["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0].content)
    assert payload["timezone"] == "Asia/Kolkata"
    assert payload["date"] == datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()


def test_sessions_do_not_share_history():
    service, model = _service(
        [
            AIMessage(content="first"),
            AIMessage(content="second"),
            AIMessage(content="third"),
        ]
    )
    service.answer("alpha", session_id="a")
    service.answer("beta", session_id="b")
    service.answer("again", session_id="a")

    def user_texts(invocation):
        return [message.content for message in invocation["messages"] if message.type == "human"]

    assert user_texts(model.invocations[1]) == ["beta"]
    assert user_texts(model.invocations[2]) == ["alpha", "again"]


def test_history_is_bounded():
    replies = [AIMessage(content=f"reply {index}") for index in range(3)]
    service, model = _service(replies, history_max_messages=2)
    service.answer("one", session_id="s")
    service.answer("two", session_id="s")
    service.answer("three", session_id="s")
    users = [message.content for message in model.invocations[2]["messages"] if message.type == "human"]
    assert users == ["two", "three"]


def test_demo_fault_is_a_safe_error():
    service, model = _service([AIMessage(content="should not be used")])
    with pytest.raises(ChatbotError) as caught:
        service.answer("Simulate a model failure.", session_id="err", demo_fault="model")
    assert caught.value.public_message == PUBLIC_ERROR_MESSAGE
    assert DEMO_FAULT_MESSAGE not in caught.value.public_message
    assert model.invocations == []


def test_missing_api_key_is_reported_without_calling_a_provider():
    service = ChatService(_settings())
    with pytest.raises(ChatbotError) as caught:
        service.answer("What is MLOps?")
    assert caught.value.public_message == MISSING_KEY_MESSAGE
