from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.logger import get_logger
from app.vectorstore.index_manager import load_index
from app.api.middleware.logging import RequestLoggingMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.routes import health, search, chat, recommend, summarize

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME}")
    load_index()
    logger.info("FAISS index loaded — server ready")
    yield
    logger.info(f"{settings.APP_NAME} shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="Semantic search and RAG-powered Q&A over 7,700+ ArXiv AI/ML research papers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health"])
app.include_router(search.router, prefix=settings.API_PREFIX, tags=["Search"])
app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["Chat"])
app.include_router(recommend.router, prefix=settings.API_PREFIX, tags=["Recommend"])
app.include_router(summarize.router, prefix=settings.API_PREFIX, tags=["Summarize"])