import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        logger.info(f"→ {request.method} {request.url.path}")

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"← {request.method} {request.url.path} "
            f"| status={response.status_code} "
            f"| {elapsed_ms:.1f}ms"
        )

        return response