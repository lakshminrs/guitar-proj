"""Blob storage (local disk in v1; swap for S3/R2 presigned URLs later) + signed upload URLs."""
from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path

from .config import DATA_DIR, PUBLIC_API_BASE, SECRET

SESSIONS_DIR = DATA_DIR / "sessions"


def session_dir(session_id: str) -> Path:
    d = SESSIONS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sig(session_id: str, exp: int) -> str:
    return hmac.new(SECRET.encode(), f"{session_id}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def signed_upload_url(session_id: str, ttl_s: int = 3600) -> str:
    exp = int(time.time()) + ttl_s
    return f"{PUBLIC_API_BASE}/api/uploads/{session_id}?exp={exp}&sig={_sig(session_id, exp)}"


def verify_upload(session_id: str, exp: int, sig: str) -> bool:
    return exp >= time.time() and hmac.compare_digest(sig, _sig(session_id, exp))
