import re
from datetime import date
from app.core.logger import get_logger
from app.models.documents import Paper

logger = get_logger(__name__)


def rerank(
    query: str,
    results: list[tuple[Paper, float]],
    use_recency: bool = True,
) -> list[tuple[Paper, float]]:
    if not results:
        return results

    logger.info(f"Reranking {len(results)} results")
    query_terms = set(re.sub(r'[^a-z0-9\s]', ' ', query.lower()).split())

    scored = []
    for paper, base_score in results:
        final_score = base_score
        final_score += _title_overlap_boost(query_terms, paper.title)
        final_score += _abstract_density_boost(query_terms, paper.abstract)
        if use_recency:
            final_score += _recency_boost(paper.published_date)
        scored.append((paper, final_score))

    reranked = sorted(scored, key=lambda x: x[1], reverse=True)
    logger.info("Reranking complete")
    return reranked


def _title_overlap_boost(query_terms: set, title: str) -> float:
    title_terms = set(re.sub(r'[^a-z0-9\s]', ' ', title.lower()).split())
    overlap = len(query_terms & title_terms)
    return min(overlap * 0.02, 0.1)


def _abstract_density_boost(query_terms: set, abstract: str) -> float:
    abstract_lower = abstract.lower()
    count = sum(1 for term in query_terms if term in abstract_lower)
    return min(count * 0.005, 0.05)


def _recency_boost(published_date: date | None) -> float:
    if not published_date:
        return 0.0
    today = date.today()
    days_old = (today - published_date).days
    if days_old < 90:
        return 0.03
    elif days_old < 180:
        return 0.02
    elif days_old < 365:
        return 0.01
    return 0.0