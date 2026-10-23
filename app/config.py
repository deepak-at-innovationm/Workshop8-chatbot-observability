"""Environment configuration.

Values come from the process environment and, when present, a local `.env` file.
Existing environment variables win over the file. Secrets are never given defaults.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the demo chatbot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    llm_model: str = "gpt-5-nano"
    app_timezone: str = "Asia/Kolkata"
    history_max_messages: int = Field(default=10, ge=2)
    allow_demo_faults: bool = False

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "mlops-chatbot-demo"
    langsmith_endpoint: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return cached settings. Tests call `get_settings.cache_clear()` after env changes."""
    return Settings()
