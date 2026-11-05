"""SQLAlchemy models: chord_shape, exercise, session, event."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATABASE_URL
from .theory import CHORDS, EXERCISES

engine = create_engine(DATABASE_URL, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ChordShapeRow(Base):
    __tablename__ = "chord_shape"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(16), unique=True)
    frets: Mapped[list] = mapped_column(JSON)      # [6] low E -> high E, -1 = muted
    fingers: Mapped[list] = mapped_column(JSON)    # [6] 0 = open, 1-4 index..pinky
    midi: Mapped[list] = mapped_column(JSON)       # [6]
    barre: Mapped[int] = mapped_column(Integer, default=0)


class ExerciseRow(Base):
    __tablename__ = "exercise"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    bpm: Mapped[int] = mapped_column(Integer)
    time_sig: Mapped[str] = mapped_column(String(8), default="4/4")
    bars: Mapped[list] = mapped_column(JSON)       # [{chord, strum_pattern}]

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "description": self.description, "bpm": self.bpm,
                "time_sig": self.time_sig, "bars": self.bars}


def _uuid() -> str:
    return uuid.uuid4().hex


class SessionRow(Base):
    __tablename__ = "session"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    exercise_id: Mapped[int] = mapped_column(ForeignKey("exercise.id"))
    video_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    upload_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # points, homography, mirrored
    device: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    clicks_ms: Mapped[list | None] = mapped_column(JSON, nullable=True)
    count_in_beats: Mapped[int] = mapped_column(Integer, default=4)
    status: Mapped[str] = mapped_column(String(20), default="created")      # created|uploaded|queued|processing|done|failed
    progress: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)       # components, top errors, meta, note
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EventRow(Base):
    __tablename__ = "event"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("session.id", ondelete="CASCADE"), index=True)
    t_ms: Mapped[float] = mapped_column(Float)
    bar: Mapped[int] = mapped_column(Integer)
    beat: Mapped[int] = mapped_column(Integer)
    expected_chord: Mapped[str] = mapped_column(String(16))
    fingertips_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scores_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    errors_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    extra_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)    # timing, direction, beat label...


def init_db() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        for c in CHORDS.values():
            row = db.scalar(select(ChordShapeRow).where(ChordShapeRow.name == c.name))
            if row is None:
                db.add(ChordShapeRow(name=c.name, frets=list(c.frets), fingers=list(c.fingers),
                                     midi=c.midi, barre=c.barre))
        for e in EXERCISES:
            row = db.get(ExerciseRow, e["id"])
            if row is None:
                db.add(ExerciseRow(id=e["id"], title=e["title"], description=e.get("description", ""),
                                   bpm=e["bpm"], time_sig=e["time_sig"], bars=e["bars"]))
            else:  # keep seed exercises in sync with code
                row.title, row.bpm, row.time_sig, row.bars = e["title"], e["bpm"], e["time_sig"], e["bars"]
                row.description = e.get("description", "")
        db.commit()
