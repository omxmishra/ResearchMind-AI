import faiss
import pickle
import numpy as np
from pathlib import Path
from app.core.logger import get_logger
from app.core.constants import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIM
from app.models.documents import Paper

logger = get_logger(__name__)


class FAISSStore:
    def __init__(self):
        self.index: faiss.Index = None
        self.papers: list[Paper] = []
        self.dim: int = EMBEDDING_DIM

    def build(self, embeddings: np.ndarray, papers: list[Paper]):
        n, d = embeddings.shape
        logger.info(f"Building FAISS index for {n} vectors of dim {d}")

        if n > 10_000:
            nlist = min(int(n ** 0.5), 256)
            quantiser = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = 32
        else:
            self.index = faiss.IndexFlatIP(d)

        self.index.add(embeddings)
        self.papers = papers
        logger.info(f"Index built with {self.index.ntotal} vectors")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[Paper, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Run build_vector_index.py first.")

        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.papers[idx], float(score)))
        return results

    def save(self):
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.papers, f)
        logger.info(f"Saved index to {FAISS_INDEX_PATH}")
        logger.info(f"Saved metadata to {METADATA_PATH}")

    def load(self):
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run: python scripts/build_vector_index.py"
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.papers = pickle.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        logger.info(f"Loaded {len(self.papers)} papers from metadata")

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0