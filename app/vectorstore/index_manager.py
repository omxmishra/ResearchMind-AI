import numpy as np
from app.core.logger import get_logger
from app.vectorstore.faiss_store import FAISSStore
from app.models.documents import Paper

logger = get_logger(__name__)

_store: FAISSStore | None = None


def get_store() -> FAISSStore:
    global _store
    if _store is None:
        raise RuntimeError("Index not loaded. Call load_index() first.")
    return _store


def load_index() -> FAISSStore:
    global _store
    if _store is not None:
        logger.info("Index already loaded — reusing existing instance")
        return _store
    logger.info("Loading FAISS index into memory")
    _store = FAISSStore()
    _store.load()
    logger.info(f"Index ready — {_store.size} vectors loaded")
    return _store


def is_loaded() -> bool:
    return _store is not None


def get_index_stats() -> dict:
    if _store is None:
        return {"loaded": False, "total_vectors": 0, "total_papers": 0}
    return {
        "loaded": True,
        "total_vectors": _store.size,
        "total_papers": len(_store.papers),
        "index_type": type(_store.index).__name__,
    }