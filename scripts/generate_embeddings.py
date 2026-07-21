import pickle
from app.core.logger import get_logger
from app.core.constants import PAPERS_PKL_PATH
from app.services.embeddings.embedding_generator import EmbeddingGenerator
from app.services.embeddings.embedding_cache import EmbeddingCache

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting embedding generation pipeline")

    logger.info(f"Loading papers from {PAPERS_PKL_PATH}")
    with open(PAPERS_PKL_PATH, "rb") as f:
        papers = pickle.load(f)
    logger.info(f"Loaded {len(papers)} papers")

    cache = EmbeddingCache()

    if cache.exists():
        meta = cache.get_meta()
        logger.info(f"Cache found — model: {meta['model_name']}, papers: {meta['num_papers']}")
        logger.info("Skipping embedding generation — delete cache to regenerate")
    else:
        logger.info("No cache found — generating embeddings")
        generator = EmbeddingGenerator()
        embeddings = generator.embed_papers(papers)

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        cache.save(
            embeddings=embeddings,
            model_name=generator.model_name,
            num_papers=len(papers),
        )
        logger.info("Embedding generation complete")

    embeddings, meta = cache.load()
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Model used: {meta['model_name']}")
    print(f"Papers embedded: {meta['num_papers']}")
    print(f"Embedding dim: {meta['embedding_dim']}")