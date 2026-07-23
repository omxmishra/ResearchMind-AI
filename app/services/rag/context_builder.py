from app.models.documents import Paper
from app.core.logger import get_logger

logger = get_logger(__name__)

MAX_ABSTRACT_CHARS = 600
MAX_PAPERS_IN_CONTEXT = 5


def build_context(papers: list[Paper]) -> str:
    if not papers:
        return "No relevant papers found."

    papers = papers[:MAX_PAPERS_IN_CONTEXT]
    logger.info(f"Building context from {len(papers)} papers")

    blocks = []
    for i, paper in enumerate(papers, 1):
        abstract_snippet = (
            paper.abstract[:MAX_ABSTRACT_CHARS] + "..."
            if len(paper.abstract) > MAX_ABSTRACT_CHARS
            else paper.abstract
        )
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += f" et al."

        block = (
            f"[Paper {i}]\n"
            f"Title: {paper.title}\n"
            f"Authors: {authors_str}\n"
            f"Category: {paper.primary_category}\n"
            f"Date: {paper.published_date}\n"
            f"Abstract: {abstract_snippet}\n"
            f"URL: {paper.arxiv_url}"
        )
        blocks.append(block)

    context = "\n\n---\n\n".join(blocks)
    logger.info(f"Context built — {len(context)} characters")
    return context