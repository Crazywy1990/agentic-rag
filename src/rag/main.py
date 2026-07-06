from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from rag.config import settings
from rag.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "service": settings.app_name}

    return app


app = create_app()
