import time
import numpy as np
from datetime import date
from app.core.logger import get_logger
from app.vectorstore.index_manager import get_store
from app.services.embeddings.embedding_generator import EmbeddingGenerator
from app.models.documents import Paper
from app.core.constants import EMBEDDING_MODEL, TOP_K_DEFAULT, TOP_K_MAX

logger = get_logger(__name__)

_generator: EmbeddingGenerator | None = None


def get_generator() -> EmbeddingGenerator:
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator(EMBEDDING_MODEL)
    return _generator


def retrieve(
    query: str,
    top_k: int = TOP_K_DEFAULT,
    category_filter: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[tuple[Paper, float]], float]:
    start = time.perf_counter()

    generator = get_generator()
    query_vec = generator.embed_query(query)

    fetch_k = min(top_k * 5 if (category_filter or date_from or date_to) else top_k, TOP_K_MAX * 5)

    store = get_store()
    raw_results = store.search(query_vec, top_k=fetch_k)

    filtered = _apply_filters(raw_results, category_filter, date_from, date_to)
    filtered = filtered[:top_k]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Retrieved {len(filtered)} results for query '{query[:50]}' in {elapsed_ms:.1f}ms")

    return filtered, elapsed_ms


def retrieve_by_paper_id(paper_id: str, top_k: int = TOP_K_DEFAULT) -> tuple[list[tuple[Paper, float]], float]:
    store = get_store()
    source_paper = None
    source_idx = None

    for i, paper in enumerate(store.papers):
        if paper.paper_id == paper_id:
            source_paper = paper
            source_idx = i
            break

    if source_paper is None:
        raise ValueError(f"Paper ID '{paper_id}' not found in index")

    paper_vec = np.array(store.index.reconstruct(source_idx)).reshape(1, -1)
    scores, indices = store.index.search(paper_vec, top_k + 1)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx == source_idx:
            continue
        results.append((store.papers[idx], float(score)))

    return results[:top_k], 0.0


def _apply_filters(
    results: list[tuple[Paper, float]],
    category_filter: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> list[tuple[Paper, float]]:
    if not any([category_filter, date_from, date_to]):
        return results

    filtered = []
    for paper, score in results:
        if category_filter:
            if not set(paper.categories) & set(category_filter):
                continue
        if date_from and paper.published_date:
            if paper.published_date < date.fromisoformat(date_from):
                continue
        if date_to and paper.published_date:
            if paper.published_date > date.fromisoformat(date_to):
                continue
        filtered.append((paper, score))

    return filtered