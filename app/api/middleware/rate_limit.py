import time
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

RATE_LIMIT = 60
WINDOW_SECONDS = 60

request_counts: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        now = time.time()
        window_start = now - WINDOW_SECONDS

        request_counts[ip] = [t for t in request_counts[ip] if t > window_start]

        if len(request_counts[ip]) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded",
                    "detail": f"Max {RATE_LIMIT} requests per minute allowed",
                },
            )

        request_counts[ip].append(now)
        return await call_next(request)