"""Job queue: Redis + RQ when REDIS_URL is set (docker compose), else an in-process worker
thread so local dev needs nothing but Python + ffmpeg."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .config import REDIS_URL

_local = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis") if not REDIS_URL else None


def enqueue_analysis(session_id: str) -> None:
    if REDIS_URL:
        from redis import Redis
        from rq import Queue
        Queue("analysis", connection=Redis.from_url(REDIS_URL)).enqueue(
            "app.jobs.analyze_session", session_id, job_timeout=900)
    else:
        from .jobs import analyze_session
        _local.submit(analyze_session, session_id)
