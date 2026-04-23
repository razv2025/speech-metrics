import base64
import secrets
from typing import Callable, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import Response


def decode_basic_auth(authorization: str) -> Optional[Tuple[str, str]]:
    if not authorization.startswith('Basic '):
        return None
    try:
        decoded = base64.b64decode(authorization[6:]).decode('utf-8', errors='replace')
        username, password = decoded.split(':', 1)
        return username, password
    except Exception:
        return None


def sanitize_username(username: str) -> str:
    return ''.join(c for c in username if 32 <= ord(c) < 127)[:40].strip()


def request_username(request: Request) -> str:
    creds = decode_basic_auth(request.headers.get('Authorization', ''))
    return sanitize_username(creds[0]) if creds else ''


def check_auth(authorization: str, password: str) -> bool:
    creds = decode_basic_auth(authorization)
    if not creds:
        return False
    _, supplied_password = creds
    return secrets.compare_digest(supplied_password.encode('utf-8'), password.encode('utf-8'))


def register_basic_auth(app: FastAPI, get_password: Callable[[], str]) -> None:
    @app.middleware('http')
    async def basic_auth(request: Request, call_next):
        if request.url.path == '/health':
            return await call_next(request)
        password = get_password()
        if password and check_auth(request.headers.get('Authorization', ''), password):
            return await call_next(request)
        return Response(
            status_code=401,
            headers={'WWW-Authenticate': 'Basic realm="Speech Assessment"'},
            content='Unauthorized',
        )
