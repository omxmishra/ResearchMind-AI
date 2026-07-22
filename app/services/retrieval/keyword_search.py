import re
from rank_bm25 import BM25Okapi
from app.core.logger import get_logger
from app.models.documents import Paper

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 2]


class KeywordSearchService:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.papers: list[Paper] = []

    def build_index(self, papers: list[Paper]):
        logger.info(f"Building BM25 index for {len(papers)} papers")
        self.papers = papers
        corpus = [tokenize(p.title + " " + p.abstract) for p in papers]
        self.bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built")

    def search(self, query: str, top_k: int = 10) -> list[tuple[Paper, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.papers[idx], float(scores[idx])))

        logger.info(f"BM25 search returned {len(results)} results for '{query[:50]}'")
        return results