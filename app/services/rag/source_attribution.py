import re
from app.models.documents import Paper
from app.core.logger import get_logger

logger = get_logger(__name__)


def attribute_sources(answer: str, papers: list[Paper]) -> list[Paper]:
    if not papers or not answer:
        return papers[:3]

    answer_lower = answer.lower()
    attributed = []
    unattributed = []

    for paper in papers:
        if _is_referenced(answer_lower, paper):
            attributed.append(paper)
        else:
            unattributed.append(paper)

    if len(attributed) < 2:
        needed = 2 - len(attributed)
        attributed.extend(unattributed[:needed])

    logger.info(f"Attributed {len(attributed)} sources from {len(papers)} candidates")
    return attributed[:5]


def _is_referenced(answer_lower: str, paper: Paper) -> bool:
    title_words = [
        w for w in paper.title.lower().split()
        if len(w) > 4 and w not in {"about", "using", "based", "with", "from", "that", "this", "their", "which"}
    ]

    significant_title_words = title_words[:5]
    matches = sum(1 for w in significant_title_words if w in answer_lower)

    if matches >= 2:
        return True

    if paper.primary_category and paper.primary_category.lower() in answer_lower:
        return True

    return False


def format_sources_for_display(papers: list[Paper]) -> list[dict]:
    return [
        {
            "title": paper.title,
            "authors": ", ".join(paper.authors[:3]),
            "url": paper.arxiv_url,
            "category": paper.primary_category,
            "date": str(paper.published_date),
        }
        for paper in papers
    ]