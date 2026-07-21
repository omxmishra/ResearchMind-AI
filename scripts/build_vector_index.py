import pickle
import numpy as np
from app.core.logger import get_logger
from app.core.constants import PAPERS_PKL_PATH
from app.services.embeddings.embedding_cache import EmbeddingCache
from app.vectorstore.faiss_store import FAISSStore

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting vector index build pipeline")

    logger.info(f"Loading papers from {PAPERS_PKL_PATH}")
    with open(PAPERS_PKL_PATH, "rb") as f:
        papers = pickle.load(f)
    logger.info(f"Loaded {len(papers)} papers")

    cache = EmbeddingCache()
    if not cache.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run: python scripts/generate_embeddings.py first"
        )

    embeddings, meta = cache.load()
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding model: {meta['model_name']}")

    store = FAISSStore()
    store.build(embeddings, papers)
    store.save()

    print(f"\nIndex Summary:")
    print(f"  Total vectors: {store.size}")
    print(f"  Total papers: {len(store.papers)}")
    print(f"  Index type: {type(store.index).__name__}")
    print(f"  Embedding dim: {store.dim}")

    logger.info("Vector index build complete")