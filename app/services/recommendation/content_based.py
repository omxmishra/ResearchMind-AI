from app.core.logger import get_logger
from app.vectorstore.retriever import retrieve_by_paper_id
from app.vectorstore.index_manager import get_store
from app.models.documents import Paper
from app.models.schemas import RecommendRequest, RecommendResponse, SearchResult

logger = get_logger(__name__)


class ContentBasedRecommender:
    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        logger.info(f"Finding recommendations for paper: {request.paper_id}")

        source_paper = self._get_paper(request.paper_id)
        if source_paper is None:
            raise ValueError(f"Paper ID '{request.paper_id}' not found")

        results, _ = retrieve_by_paper_id(
            paper_id=request.paper_id,
            top_k=request.top_k,
        )

        recommendations = [
            SearchResult(paper=paper, score=round(score, 4), rank=i + 1)
            for i, (paper, score) in enumerate(results)
        ]

        logger.info(f"Found {len(recommendations)} recommendations")

        return RecommendResponse(
            source_paper=source_paper,
            recommendations=recommendations,
            total_found=len(recommendations),
        )

    def _get_paper(self, paper_id: str) -> Paper | None:
        store = get_store()
        for paper in store.papers:
            if paper.paper_id == paper_id:
                return paper
        return None