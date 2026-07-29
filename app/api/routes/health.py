from fastapi import APIRouter
from app.models.responses import HealthResponse
from app.vectorstore.index_manager import get_index_stats, is_loaded
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health_check():
    stats = get_index_stats()
    return HealthResponse(
        status="healthy" if is_loaded() else "degraded",
        app_name=settings.APP_NAME,
        version="1.0.0",
        index_loaded=stats["loaded"],
        total_papers=stats.get("total_papers", 0),
        embedding_model=settings.EMBEDDING_MODEL,
    )