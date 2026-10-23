"""LangSmith environment setup and trace metadata.

LangChain and LangSmith read tracing flags from the process environment.
Settings are copied across at startup. When tracing is disabled, no API key is required.
"""

import os

from app.config import Settings

APPLICATION_NAME = "MLOps Chatbot Demo"


def apply_runtime_env(settings: Settings) -> None:
    """Publish tracing and model settings to the process environment.

    The official SDKs read these variables themselves. An empty API key is not
    written, so a disabled tracing configuration does not invent a credential.
    """
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    # LangSmith caches env lookups. Clear them so a later change, including
    # startup applying .env, is visible to the tracer.
    from langsmith.utils import get_env_var, get_tracer_project

    get_env_var.cache_clear()
    get_tracer_project.cache_clear()


def run_metadata(
    settings: Settings,
    *,
    session_id: str,
    request_id: str,
    workflow_type: str,
) -> dict:
    """Metadata attached to a chat trace and its child model and tool runs.

    `thread_id` is the LangSmith conversation key. It matches the API session id.
    """
    return {
        "application": APPLICATION_NAME,
        "environment": os.getenv("ENV", "local"),
        "model_provider": "openai",
        "model_name": settings.llm_model,
        "request_id": request_id,
        "session_id": session_id,
        "thread_id": session_id,
        "workflow_type": workflow_type,
    }


def run_config(
    settings: Settings,
    *,
    name: str,
    session_id: str,
    request_id: str,
    workflow_type: str,
) -> dict:
    """LangChain config for one child run. Passing metadata here tags that child."""
    return {
        "run_name": name,
        "tags": ["mlops-chatbot-demo", workflow_type],
        "metadata": run_metadata(
            settings,
            session_id=session_id,
            request_id=request_id,
            workflow_type=workflow_type,
        ),
    }
