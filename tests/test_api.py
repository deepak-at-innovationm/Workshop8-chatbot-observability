from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.chatbot import PUBLIC_ERROR_MESSAGE, ChatService
from app.config import Settings
from app.main import create_app
from app.routes import get_chat_service
from tests.fakes import ScriptedChatModel


def _client(replies, **setting_overrides):
    settings = Settings(
        openai_api_key="",
        langsmith_tracing=False,
        langsmith_api_key="",
        allow_demo_faults=False,
        history_max_messages=10,
        **setting_overrides,
    )
    model = ScriptedChatModel(replies)
    service = ChatService(settings, model=model)
    application = create_app()
    application.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(application), service, model


def test_health_and_homepage():
    client, _service, _model = _client([])
    with client:
        health = client.get("/health")
        page = client.get("/")
        script = client.get("/static/app.js")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["tracing_enabled"] is False
    assert body["demo_faults_enabled"] is False
    assert "api_key" not in health.text.lower()
    assert page.status_code == 200
    assert "MLOps Chatbot" in page.text
    assert script.status_code == 200
    assert "/api/chat" in script.text


def test_chat_returns_answer_and_session():
    client, _service, _model = _client([AIMessage(content="MLOps keeps models running in production.")])
    with client:
        response = client.post("/api/chat", json={"message": "What is MLOps?"})
    assert response.status_code == 200
    payload = response.json()
    assert "MLOps" in payload["answer"]
    assert payload["session_id"]
    assert payload["request_id"]
    assert payload["tool_used"] is False


def test_empty_message_is_rejected():
    client, _service, _model = _client([])
    with client:
        response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_invalid_session_id_is_rejected():
    client, _service, _model = _client([])
    with client:
        response = client.post("/api/chat", json={"message": "Hello", "session_id": "bad id"})
    assert response.status_code == 422


def test_sessions_stay_isolated_through_the_api():
    client, _service, model = _client(
        [
            AIMessage(content="one"),
            AIMessage(content="two"),
        ]
    )
    with client:
        client.post("/api/chat", json={"message": "alpha", "session_id": "alpha-session"})
        client.post("/api/chat", json={"message": "beta", "session_id": "beta-session"})
    second_users = [message.content for message in model.invocations[1]["messages"] if message.type == "human"]
    assert second_users == ["beta"]


def test_demo_fault_header_is_ignored_unless_enabled():
    client, _service, model = _client([AIMessage(content="Normal answer.")])
    with client:
        response = client.post(
            "/api/chat",
            json={"message": "What is MLOps?"},
            headers={"X-Demo-Fault": "model"},
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "Normal answer."
    assert model.invocations


def test_demo_fault_returns_a_safe_error():
    client, service, model = _client([AIMessage(content="should not run")])
    service.settings.allow_demo_faults = True
    with client:
        response = client.post(
            "/api/chat",
            json={"message": "Simulate a model failure."},
            headers={"X-Demo-Fault": "model"},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == PUBLIC_ERROR_MESSAGE
    assert "Traceback" not in response.text
    assert "Simulated model failure" not in response.text
    assert "API_KEY" not in response.text
    assert model.invocations == []
