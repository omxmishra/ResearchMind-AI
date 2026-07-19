from pydantic_settings import BaseSettings
from functools import lru_cache
from app.core.constants import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BATCH_SIZE,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    API_PREFIX,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)


class Settings(BaseSettings):
    APP_NAME: str = "ResearchMind-AI"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = API_PREFIX

    ANTHROPIC_API_KEY: str = ""

    EMBEDDING_MODEL: str = EMBEDDING_MODEL
    EMBEDDING_DIM: int = EMBEDDING_DIM
    BATCH_SIZE: int = BATCH_SIZE

    FAISS_INDEX_PATH: str = str(FAISS_INDEX_PATH)
    METADATA_PATH: str = str(METADATA_PATH)

    TOP_K_DEFAULT: int = TOP_K_DEFAULT
    TOP_K_MAX: int = TOP_K_MAX

    CACHE_TTL: int = 3600
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379"

    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()