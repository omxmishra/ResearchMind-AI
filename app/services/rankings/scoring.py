import numpy as np
from app.models.documents import Paper
from app.core.logger import get_logger

logger = get_logger(__name__)


def normalize_scores(results: list[tuple[Paper, float]]) -> list[tuple[Paper, float]]:
    if not results:
        return results

    scores = np.array([s for _, s in results])
    min_s, max_s = scores.min(), scores.max()

    if max_s == min_s:
        return [(p, 1.0) for p, _ in results]

    normalized = (scores - min_s) / (max_s - min_s)
    return [(paper, float(score)) for (paper, _), score in zip(results, normalized)]


def combine_scores(
    semantic_results: list[tuple[Paper, float]],
    keyword_results: list[tuple[Paper, float]],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[tuple[Paper, float]]:
    semantic_norm = normalize_scores(semantic_results)
    keyword_norm = normalize_scores(keyword_results)

    semantic_map = {p.paper_id: s for p, s in semantic_norm}
    keyword_map = {p.paper_id: s for p, s in keyword_norm}
    paper_map = {p.paper_id: p for p, _ in semantic_results + keyword_results}

    all_ids = set(semantic_map.keys()) | set(keyword_map.keys())

    combined = []
    for pid in all_ids:
        sem_score = semantic_map.get(pid, 0.0)
        kw_score = keyword_map.get(pid, 0.0)
        final_score = (semantic_weight * sem_score) + (keyword_weight * kw_score)
        combined.append((paper_map[pid], final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Combined {len(semantic_results)} semantic + {len(keyword_results)} keyword results into {len(combined)} unique results")
    return combined


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Paper, float]]],
    k: int = 60,
) -> list[tuple[Paper, float]]:
    rrf_scores: dict[str, float] = {}
    paper_map: dict[str, Paper] = {}

    for result_list in result_lists:
        for rank, (paper, _) in enumerate(result_list, start=1):
            pid = paper.paper_id
            paper_map[pid] = paper
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)

    combined = [(paper_map[pid], score) for pid, score in rrf_scores.items()]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined