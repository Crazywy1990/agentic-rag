from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from rag.config import settings
from rag.embed.openai_embedder import OpenAIEmbedder
from rag.generate.answer import Answer, answer_question
from rag.generate.gateway_client import GatewayClient
from rag.logging import configure_logging
from rag.store.lancedb_store import LanceDBStore


class AskRequest(BaseModel):
    question: str
    k: int = 5


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "service": settings.app_name}

    @app.post("/ask")
    async def ask(req: AskRequest) -> Answer:
        # Per-request construction: fine at dev scale. W4 hardening: hoist to
        # app-startup singletons + move blocking work off the event loop.
        embedder = OpenAIEmbedder()
        store = LanceDBStore(settings.data_dir / "lancedb", dimension=embedder.dimension)
        gateway = GatewayClient()
        return answer_question(req.question, embedder, store, gateway, k=req.k)

    return app


app = create_app()
