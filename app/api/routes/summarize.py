from fastapi import APIRouter, HTTPException
from app.models.schemas import SummarizeRequest, SummarizeResponse
from app.services.summarization.abstractive import AbstractiveSummarizer
from app.vectorstore.index_manager import get_store
from app.core.config import get_settings
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()
summarizer = AbstractiveSummarizer()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_paper(request: SummarizeRequest):
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured. Add ANTHROPIC_API_KEY to .env file.",
        )
    try:
        store = get_store()
        paper = None
        for p in store.papers:
            if p.paper_id == request.paper_id:
                paper = p
                break

        if paper is None:
            raise HTTPException(
                status_code=404,
                detail=f"Paper '{request.paper_id}' not found"
            )

        return summarizer.summarize(paper, style=request.style)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))