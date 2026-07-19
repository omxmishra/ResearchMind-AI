from functools import lru_cache
from app.core.config import get_settings
from app.core.constants import (
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    ARXIV_CSV_PATH,
    PAPERS_PKL_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDINGS_PATH,
    SUPPORTED_CATEGORIES,
)


@lru_cache()
def get_app_settings():
    config = get_settings()

    return {
        "app_name": config.APP_NAME,
        "debug": config.DEBUG,
        "api_prefix": config.API_PREFIX,
        "api_host": config.API_HOST,
        "api_port": config.API_PORT,
        "anthropic_api_key": config.ANTHROPIC_API_KEY,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "batch_size": config.BATCH_SIZE,
        "faiss_index_path": FAISS_INDEX_PATH,
        "metadata_path": METADATA_PATH,
        "papers_pkl_path": PAPERS_PKL_PATH,
        "embeddings_path": EMBEDDINGS_PATH,
        "arxiv_csv_path": ARXIV_CSV_PATH,
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "raw_data_dir": RAW_DATA_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "embeddings_dir": EMBEDDINGS_DIR,
        "top_k_default": config.TOP_K_DEFAULT,
        "top_k_max": config.TOP_K_MAX,
        "cache_ttl": config.CACHE_TTL,
        "use_redis": config.USE_REDIS,
        "redis_url": config.REDIS_URL,
        "allowed_origins": config.ALLOWED_ORIGINS,
        "supported_categories": SUPPORTED_CATEGORIES,
    }