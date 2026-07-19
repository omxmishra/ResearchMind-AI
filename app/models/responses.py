from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str = datetime.utcnow().isoformat()


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    index_loaded: bool
    total_papers: int
    embedding_model: str
    timestamp: str = datetime.utcnow().isoformat()


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str = datetime.utcnow().isoformat()