from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag.pipeline import RAGPipeline
from app.core.config import get_settings
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()
rag_pipeline = RAGPipeline()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured. Add ANTHROPIC_API_KEY to .env file.",
        )
    try:
        return rag_pipeline.run(request)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))