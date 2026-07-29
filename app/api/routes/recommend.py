from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.recommendation.content_based import ContentBasedRecommender
from app.vectorstore.index_manager import get_store
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
recommender = ContentBasedRecommender()


@router.post("/recommend", response_model=RecommendResponse)
def recommend_papers(request: RecommendRequest):
    try:
        return recommender.recommend(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
def list_categories():
    try:
        store = get_store()
        categories = set()
        for paper in store.papers:
            if paper.primary_category:
                categories.add(paper.primary_category)
        return {"categories": sorted(categories), "total": len(categories)}
    except Exception as e:
        logger.error(f"Categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}")
def get_paper(paper_id: str):
    try:
        store = get_store()
        for paper in store.papers:
            if paper.paper_id == paper_id:
                return paper
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get paper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))