from app.core.logger import get_logger
from app.models.documents import Paper

logger = get_logger(__name__)


def compute_similarity_score(
    source_paper: Paper,
    candidate_paper: Paper,
    base_score: float,
) -> float:
    score = base_score
    score += _category_boost(source_paper, candidate_paper)
    score += _author_boost(source_paper, candidate_paper)
    score += _recency_similarity_boost(source_paper, candidate_paper)
    return score


def _category_boost(source: Paper, candidate: Paper) -> float:
    if not source.primary_category or not candidate.primary_category:
        return 0.0
    if source.primary_category == candidate.primary_category:
        return 0.05
    source_cats = set(source.categories)
    candidate_cats = set(candidate.categories)
    overlap = len(source_cats & candidate_cats)
    return min(overlap * 0.01, 0.03)


def _author_boost(source: Paper, candidate: Paper) -> float:
    if not source.authors or not candidate.authors:
        return 0.0
    source_authors = set(a.lower() for a in source.authors)
    candidate_authors = set(a.lower() for a in candidate.authors)
    shared = len(source_authors & candidate_authors)
    return min(shared * 0.02, 0.04)


def _recency_similarity_boost(source: Paper, candidate: Paper) -> float:
    if not source.published_date or not candidate.published_date:
        return 0.0
    days_diff = abs((source.published_date - candidate.published_date).days)
    if days_diff < 30:
        return 0.02
    elif days_diff < 90:
        return 0.01
    return 0.0


def rerank_by_similarity(
    source_paper: Paper,
    results: list[tuple[Paper, float]],
) -> list[tuple[Paper, float]]:
    logger.info(f"Reranking {len(results)} recommendations by similarity")
    reranked = [
        (paper, compute_similarity_score(source_paper, paper, score))
        for paper, score in results
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked