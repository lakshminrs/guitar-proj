"""The analysis job: demux -> audio + vision (independent) -> fusion -> DB rows."""
from __future__ import annotations

import json
import logging
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import delete

from .analysis.audio import analyze_audio
from .analysis.fusion import grade
from .analysis.vision import analyze_video
from .db import EventRow, ExerciseRow, SessionLocal, SessionRow
from .feedback import coaching_note
from .storage import session_dir

log = logging.getLogger(__name__)
VIDEO_FPS = 30


def _probe_streams(path: str) -> set[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return {s["codec_type"] for s in json.loads(out.stdout).get("streams", [])}


def demux(src: str, out_dir) -> tuple[str | None, str]:
    """-> (constant-frame-rate H.264 mp4 for analysis + replay, 44.1 kHz mono WAV)."""
    streams = _probe_streams(src)
    if "audio" not in streams:
        raise ValueError("upload has no audio track")
    wav = str(out_dir / "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-map", "0:a:0", "-ac", "1", "-ar", "44100",
                    "-c:a", "pcm_s16le", wav], check=True)
    mp4 = None
    if "video" in streams:
        mp4 = str(out_dir / "video.mp4")
        # CFR so frame index = t * fps holds; keep audio so replay plays the take
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-map", "0:v:0", "-map", "0:a:0",
                        "-vf", f"fps={VIDEO_FPS}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", mp4],
                       check=True)
    return mp4, wav


def _set(session_id: str, **kw) -> None:
    with SessionLocal() as db:
        s = db.get(SessionRow, session_id)
        for k, v in kw.items():
            setattr(s, k, v)
        db.commit()


def analyze_session(session_id: str) -> None:
    try:
        with SessionLocal() as db:
            s = db.get(SessionRow, session_id)
            ex = db.get(ExerciseRow, s.exercise_id).to_dict()
            upload, calibration = s.upload_path, s.calibration
            clicks, count_in = s.clicks_ms or [], s.count_in_beats
        _set(session_id, status="processing", progress="demuxing")
        out = session_dir(session_id)
        mp4, wav = demux(upload, out)
        _set(session_id, video_url=f"/api/sessions/{session_id}/media/video" if mp4 else None,
             audio_url=f"/api/sessions/{session_id}/media/audio", progress="analyzing audio + video")

        # vision and audio are independent until fusion: run them in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(analyze_audio, wav, [c / 1000 for c in clicks], True, count_in)
            fv = pool.submit(analyze_video, mp4, calibration) if mp4 else None
            audio = fa.result()
            vision = None
            if fv is not None:
                try:
                    vision = fv.result()
                except Exception:
                    log.exception("vision failed; returning audio-only grade")

        _set(session_id, progress="grading")
        res = grade(ex, audio, vision, clicks, count_in)
        if vision is not None:
            (out / "track.json").write_text(json.dumps(vision.to_json()))
        (out / "notes.json").write_text(json.dumps(audio.note_events))

        note = coaching_note(ex["title"], res.score, res.top)
        summary = {"components": res.components, "top": res.top, "aggregates": res.aggregates,
                   "meta": {**res.meta, "vision_stats": vision.stats if vision else None,
                            "vision_source": vision.source if vision else None,
                            "mirrored": vision.mirrored if vision else None},
                   "coaching_note": note, "n_errors": len(res.errors), "errors": res.errors}
        with SessionLocal() as db:
            db.execute(delete(EventRow).where(EventRow.session_id == session_id))
            for ev in res.events:
                extra = {k: ev.get(k) for k in ("idx", "expected_t_ms", "slot", "beat_label", "expected_direction",
                                                "observed_direction", "direction_confidence", "direction_sources",
                                                "timing_ms", "is_offbeat", "is_change", "frame", "missed",
                                                "sustain", "rel_sustain", "next_is_change")}
                db.add(EventRow(session_id=session_id, t_ms=ev["t_ms"], bar=ev["bar"], beat=ev["beat"],
                                expected_chord=ev["expected_chord"], fingertips_json=ev["fingertips"],
                                notes_json=ev["notes"], scores_json=ev["scores"], errors_json=ev["errors"],
                                extra_json=extra))
            s = db.get(SessionRow, session_id)
            s.status, s.progress, s.score, s.summary, s.error = "done", None, res.score, summary, None
            db.commit()
    except Exception as e:
        log.exception("analysis failed for %s", session_id)
        _set(session_id, status="failed", progress=None, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
