import numpy as np
import pickle
import hashlib
from pathlib import Path
from app.core.logger import get_logger
from app.core.constants import EMBEDDINGS_DIR, EMBEDDINGS_PATH

logger = get_logger(__name__)


class EmbeddingCache:
    def __init__(self, cache_dir: Path = EMBEDDINGS_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = EMBEDDINGS_PATH
        self.meta_path = cache_dir / "cache_meta.pkl"

    def exists(self) -> bool:
        return self.embeddings_path.exists() and self.meta_path.exists()

    def save(self, embeddings: np.ndarray, model_name: str, num_papers: int):
        np.save(str(self.embeddings_path), embeddings)
        meta = {
            "model_name": model_name,
            "num_papers": num_papers,
            "embedding_dim": embeddings.shape[1],
            "shape": embeddings.shape,
        }
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Cached {num_papers} embeddings to {self.embeddings_path}")

    def load(self) -> tuple[np.ndarray, dict]:
        logger.info(f"Loading cached embeddings from {self.embeddings_path}")
        embeddings = np.load(str(self.embeddings_path))
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings, meta

    def get_meta(self) -> dict | None:
        if not self.meta_path.exists():
            return None
        with open(self.meta_path, "rb") as f:
            return pickle.load(f)

    def invalidate(self):
        if self.embeddings_path.exists():
            self.embeddings_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        logger.info("Embedding cache invalidated")