import time
from app.core.logger import get_logger
from app.core.constants import TOP_K_DEFAULT
from app.vectorstore.retriever import retrieve
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.models.documents import Paper

logger = get_logger(__name__)


class SemanticSearchService:
    def search(self, request: SearchRequest) -> SearchResponse:
        logger.info(f"Semantic search: '{request.query[:60]}'")

        results, elapsed_ms = retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        search_results = [
            SearchResult(paper=paper, score=round(score, 4), rank=i + 1)
            for i, (paper, score) in enumerate(results)
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_time_ms=round(elapsed_ms, 2),
            model_used="all-MiniLM-L6-v2",
        )