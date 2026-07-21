import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.logger import get_logger
from app.core.constants import EMBEDDING_MODEL, BATCH_SIZE
from app.models.documents import Paper

logger = get_logger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded — embedding dim: {self.dim}")

    def embed_texts(self, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        logger.info(f"Embedding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)

    def embed_papers(self, papers: list[Paper]) -> np.ndarray:
        texts = [p.text_for_embedding for p in papers]
        logger.info(f"Embedding {len(papers)} papers using title+abstract")
        return self.embed_texts(texts)