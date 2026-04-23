import logging
import os
import time
import uuid

from fastapi import FastAPI, Request


def configure_logging(name: str) -> logging.Logger:
    level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        )
    root.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def register_request_logging(app: FastAPI, logger: logging.Logger) -> None:
    @app.middleware('http')
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get('X-Request-ID') or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                'request_failed method=%s path=%s request_id=%s duration_ms=%.2f',
                request.method,
                request.url.path,
                request_id,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            'request method=%s path=%s status=%s request_id=%s duration_ms=%.2f',
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            duration_ms,
        )
        response.headers['X-Request-ID'] = request_id
        return response
