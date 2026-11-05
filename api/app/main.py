"""FastAPI app: exercises, sessions (signed upload, analyze), results, replay data."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from .analysis.geometry import homography_from_points, is_mirrored
from .config import CORS_ORIGINS, MAX_UPLOAD_MB
from .db import ChordShapeRow, EventRow, ExerciseRow, SessionLocal, SessionRow, init_db
from .queue import enqueue_analysis
from .storage import session_dir, signed_upload_url, verify_upload

logging.basicConfig(level=logging.INFO)

EXT = {"video/webm": "webm", "video/mp4": "mp4", "audio/webm": "webm", "audio/wav": "wav",
       "audio/x-wav": "wav", "audio/mp4": "m4a", "video/quicktime": "mov", "audio/ogg": "ogg"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Guitar Coach AR", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2048)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

class CreateSession(BaseModel):
    exercise_id: int


class Calibration(BaseModel):
    # normalized coords in the *recorded* (unmirrored) video frame
    points: dict[str, list[float]] = Field(default_factory=dict)
    mirrored_preview: bool = False


class AnalyzeBody(BaseModel):
    clicks_ms: list[float]
    count_in_beats: int = 4
    calibration: Calibration | None = None
    device: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/chords")
def chords():
    with SessionLocal() as db:
        return [{"id": c.id, "name": c.name, "frets": c.frets, "fingers": c.fingers, "midi": c.midi,
                 "barre": c.barre} for c in db.scalars(select(ChordShapeRow))]


@app.get("/api/exercises")
def exercises():
    with SessionLocal() as db:
        return [e.to_dict() for e in db.scalars(select(ExerciseRow).order_by(ExerciseRow.id))]


@app.get("/api/exercises/{exercise_id}")
def exercise(exercise_id: int):
    with SessionLocal() as db:
        e = db.get(ExerciseRow, exercise_id)
        if not e:
            raise HTTPException(404, "exercise not found")
        return e.to_dict()


@app.post("/api/sessions")
def create_session(body: CreateSession):
    with SessionLocal() as db:
        if not db.get(ExerciseRow, body.exercise_id):
            raise HTTPException(404, "exercise not found")
        s = SessionRow(exercise_id=body.exercise_id)
        db.add(s)
        db.commit()
        return {"id": s.id, "upload_url": signed_upload_url(s.id), "upload_method": "PUT"}


@app.put("/api/uploads/{session_id}")
async def upload(session_id: str, request: Request, exp: int = Query(...), sig: str = Query(...)):
    if not verify_upload(session_id, exp, sig):
        raise HTTPException(403, "bad or expired upload signature")
    ctype = (request.headers.get("content-type") or "video/webm").split(";")[0].strip()
    ext = EXT.get(ctype, "bin")
    path = session_dir(session_id) / f"take.{ext}"
    size = 0
    with open(path, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, "upload too large")
            f.write(chunk)
    if size == 0:
        raise HTTPException(400, "empty upload")
    with SessionLocal() as db:
        s = db.get(SessionRow, session_id)
        if not s:
            raise HTTPException(404, "session not found")
        s.upload_path, s.status = str(path), "uploaded"
        db.commit()
    return {"ok": True, "bytes": size}


@app.post("/api/sessions/{session_id}/analyze")
def analyze(session_id: str, body: AnalyzeBody):
    with SessionLocal() as db:
        s = db.get(SessionRow, session_id)
        if not s:
            raise HTTPException(404, "session not found")
        if not s.upload_path:
            raise HTTPException(409, "upload the take first")
        cal = None
        if body.calibration and body.calibration.points:
            H = homography_from_points({k: tuple(v) for k, v in body.calibration.points.items()})
            if H is None:
                raise HTTPException(422, "calibration points don't form a valid fretboard")
            cal = {**body.calibration.model_dump(), "homography": H.tolist(), "mirrored": is_mirrored(H)}
        s.clicks_ms, s.count_in_beats, s.calibration, s.device = body.clicks_ms, body.count_in_beats, cal, body.device
        s.status, s.error = "queued", None
        db.commit()
    enqueue_analysis(session_id)
    return {"id": session_id, "status": "queued"}


def _session_dict(s: SessionRow, full: bool = True) -> dict:
    summ = s.summary or {}
    d = {"id": s.id, "exercise_id": s.exercise_id, "status": s.status, "progress": s.progress,
         "error": s.error, "score": s.score, "created_at": s.created_at.isoformat(),
         "video_url": s.video_url, "audio_url": s.audio_url,
         "components": summ.get("components"), "top_errors": summ.get("top"),
         "coaching_note": summ.get("coaching_note")}
    if full:
        d.update({"calibration": s.calibration, "count_in_beats": s.count_in_beats,
                  "meta": summ.get("meta"), "aggregates": summ.get("aggregates"),
                  "errors": summ.get("errors")})
    return d


@app.get("/api/sessions")
def list_sessions(limit: int = 20):
    with SessionLocal() as db:
        rows = db.scalars(select(SessionRow).order_by(SessionRow.created_at.desc()).limit(limit))
        return [_session_dict(s, full=False) for s in rows]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    with SessionLocal() as db:
        s = db.get(SessionRow, session_id)
        if not s:
            raise HTTPException(404, "session not found")
        return _session_dict(s)


@app.get("/api/sessions/{session_id}/events")
def get_events(session_id: str):
    with SessionLocal() as db:
        rows = db.scalars(select(EventRow).where(EventRow.session_id == session_id).order_by(EventRow.t_ms))
        return [{"id": r.id, "t_ms": r.t_ms, "bar": r.bar, "beat": r.beat, "expected_chord": r.expected_chord,
                 "fingertips": r.fingertips_json, "notes": r.notes_json, "scores": r.scores_json,
                 "errors": r.errors_json, **(r.extra_json or {})} for r in rows]


@app.get("/api/sessions/{session_id}/track")
def get_track(session_id: str):
    p = session_dir(session_id) / "track.json"
    if not p.exists():
        raise HTTPException(404, "no vision track for this session")
    return Response(p.read_bytes(), media_type="application/json")


@app.get("/api/sessions/{session_id}/media/{kind}")
def media(session_id: str, kind: str):
    name = {"video": "video.mp4", "audio": "audio.wav"}.get(kind)
    if not name:
        raise HTTPException(404)
    p = session_dir(session_id) / name
    if not p.exists():
        raise HTTPException(404, "media not ready")
    return FileResponse(p, media_type="video/mp4" if kind == "video" else "audio/wav")
