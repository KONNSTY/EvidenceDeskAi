from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIM = 1536


class Settings(BaseSettings):
    """Laufzeitkonfiguration, ueberschreibbar per Umgebungsvariable mit Prefix ``ED_``."""

    model_config = SettingsConfigDict(env_prefix="ED_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://evidencedesk:evidencedesk@localhost:5432/evidencedesk"
    openai_api_key: str | None = None

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = EMBEDDING_DIM

    top_k: int = 5
    score_threshold: float = 0.35
    max_context_chars: int = 6000
    chunk_size: int = 800
    chunk_overlap: int = 120

    log_level: str = "INFO"

    price_prompt_per_mtok: float = 0.15
    price_completion_per_mtok: float = 0.60


@lru_cache
def get_settings() -> Settings:
    """Liefert die gecachte Konfiguration."""
    return Settings()
