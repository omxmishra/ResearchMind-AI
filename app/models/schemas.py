from pydantic import BaseModel, Field
from typing import Optional
from app.models.documents import Paper


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    category_filter: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    rerank: bool = False


class SearchResult(BaseModel):
    paper: Paper
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    search_time_ms: float
    model_used: str


class RecommendRequest(BaseModel):
    paper_id: str
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendResponse(BaseModel):
    source_paper: Paper
    recommendations: list[SearchResult]
    total_found: int


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    conversation_history: list[dict] = Field(default_factory=list)
    top_k_context: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Paper]
    follow_up_questions: list[str]
    tokens_used: Optional[int] = None


class SummarizeRequest(BaseModel):
    paper_id: str
    style: str = Field(default="tldr", pattern="^(tldr|detailed|beginner|linkedin)$")


class SummarizeResponse(BaseModel):
    paper_id: str
    title: str
    summary: str
    style: str