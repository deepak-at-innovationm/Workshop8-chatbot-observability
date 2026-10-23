"""Live OpenAI checks. Not part of the default offline run.

Run with:
    RUN_INTEGRATION_TESTS=1 pytest -m integration
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to call OpenAI. OPENAI_API_KEY is also required.",
)


@pytest.mark.integration
def test_live_model_chooses_the_datetime_tool_only_for_clock_questions():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    from app.chatbot import ChatService
    from app.config import get_settings
    from app.tracing import apply_runtime_env

    get_settings.cache_clear()
    settings = get_settings()
    apply_runtime_env(settings)
    service = ChatService(settings)

    knowledge = service.answer("What is MLOps?", session_id="integration-knowledge")
    clock = service.answer("What is the current date and time?", session_id="integration-clock")

    assert knowledge.tool_used is False
    assert knowledge.answer.strip()
    assert clock.tool_used is True
    assert clock.tool_name == "get_current_datetime"
    assert clock.answer.strip()
