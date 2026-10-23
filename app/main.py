"""FastAPI entrypoint for the MLOps chatbot demo."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import router
from app.tracing import apply_runtime_env

STATIC_DIR = str(Path(__file__).resolve().parent / "static")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    apply_runtime_env(settings)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info(
        "MLOps Chatbot Demo starting. tracing=%s project=%s model=%s",
        settings.langsmith_tracing,
        settings.langsmith_project,
        settings.llm_model,
    )
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="MLOps Chatbot Demo", lifespan=lifespan)
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


app = create_app()
