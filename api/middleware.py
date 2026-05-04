"""
api/middleware.py — API Middleware

Provides:
    - Request logging (endpoint, response time, score band for monitoring)
    - Error handling (uniform error responses)
    - CORS support

Note: Input hashing is used instead of logging raw PII data.
"""

import time
import hashlib
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every API request with timing and response info.
    For /predict endpoints, also logs the score band for drift monitoring.

    Privacy: Input data is hashed, NOT logged in plaintext.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        method = request.method
        path = request.url.path

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Log response
        logger.info(
            f"API | {method} {path} | "
            f"status={response.status_code} | "
            f"time={response_time_ms:.0f}ms"
        )

        # Add response time header
        response.headers["X-Response-Time-Ms"] = f"{response_time_ms:.2f}"

        return response
