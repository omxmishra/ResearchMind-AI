from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchRequest, SearchResponse
from app.services.retrieval.semantic_search import SemanticSearchService
from app.services.retrieval.query_expansion import expand_query, should_expand
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
search_service = SemanticSearchService()


@router.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest):
    try:
        if should_expand(request.query):
            request.query = expand_query(request.query)

        return search_service.search(request)
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))