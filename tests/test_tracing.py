from langchain_core.messages import AIMessage
from langsmith.utils import tracing_is_enabled

from app.chatbot import ChatService
from app.config import Settings, get_settings
from app.tracing import apply_runtime_env
from tests.fakes import ScriptedChatModel


def _service(monkeypatch, tracing: bool):
    monkeypatch.setenv("LANGSMITH_TRACING", "true" if tracing else "false")
    monkeypatch.setenv("LANGSMITH_PROJECT", "mlops-chatbot-demo")
    if tracing:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    else:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = Settings(
        openai_api_key="",
        langsmith_tracing=tracing,
        langsmith_api_key="test-key-not-real" if tracing else "",
        langsmith_project="mlops-chatbot-demo",
    )
    apply_runtime_env(settings)
    model = ScriptedChatModel([AIMessage(content="MLOps is how teams operate machine learning systems.")])
    return ChatService(settings, model=model), model


def test_tracing_disabled_works_without_credentials(monkeypatch):
    service, model = _service(monkeypatch, tracing=False)
    assert tracing_is_enabled() is False
    result = service.answer("What is MLOps?", session_id="off")
    assert "MLOps" in result.answer
    assert model.invocations[0]["run"] is None
    metadata = model.invocations[0]["config"]["metadata"]
    assert metadata["thread_id"] == "off"
    assert metadata["session_id"] == "off"
    assert metadata["model_provider"] == "openai"


def test_tracing_enabled_nests_the_model_call_and_survives_upload_failure(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("telemetry down")

    monkeypatch.setattr("langsmith.client.Client.create_run", explode)
    monkeypatch.setattr("langsmith.client.Client.update_run", explode)
    service, model = _service(monkeypatch, tracing=True)
    assert tracing_is_enabled() is True
    result = service.answer("What is MLOps?", session_id="thread-123")
    assert "MLOps" in result.answer
    run = model.invocations[0]["run"]
    assert run is not None
    assert run.name == "chat_request"
    assert run.metadata["thread_id"] == "thread-123"
    assert run.metadata["session_id"] == "thread-123"
    assert run.metadata["request_id"] == result.request_id
    assert "openai_api_key" not in run.metadata
