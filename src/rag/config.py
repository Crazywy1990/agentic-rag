from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    # Service
    app_name: str = "agentic-rag"
    log_level: str = "INFO"

    # Corpus
    data_dir: Path = Path("data")
    arxiv_query: str = 'cat:cs.CL AND abs:"retrieval"'
    corpus_size: int = 50

    # LLM gateway (Project 1)
    gateway_base_url: str = "https://llm-gateway-892437203383.us-central1.run.app"
    gateway_api_key: str = ""  # from env, never committed

    # Embeddings (direct, not via gateway — see README)
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"


settings = Settings()
