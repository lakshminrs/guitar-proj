# Guitar Coach AR — v1

Browser app that grades a recorded take against a known exercise (chords + strum pattern + BPM)
and replays it with an AR overlay. Spec: `Guitar Coach AR — v1 Spec.pdf`.

```
web/   Next.js 16 + React 19 + TS — capture, live MediaPipe hands + fret grid, replay UI
api/   FastAPI + analysis worker — ffmpeg demux, audio (librosa + basic-pitch), vision
       (MediaPipe + homography + optical flow), fusion & rules, SQLite/Postgres
```

## Run locally (no Docker needed)

Needs Python 3.12, Node 22, ffmpeg (`brew install ffmpeg`).

```bash
# backend  → http://localhost:8000
cd api
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install --no-deps basic-pitch==0.4.0
mkdir -p models && curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
.venv/bin/uvicorn app.main:app --port 8000 --reload --reload-dir app

# frontend → http://localhost:3000   (proxies /api/* to :8000)
cd web && npm install && npm run dev
```

Without `REDIS_URL` analysis runs in an in-process worker thread and data goes to `api/data/` (SQLite).
Set `DATABASE_URL` (Postgres) + `REDIS_URL` and run `rq worker analysis` for the full stack.
Phone over the LAN: the camera needs HTTPS, use `npx next dev --experimental-https -H 0.0.0.0`.

## Tests / eval

```bash
cd api && .venv/bin/python -m pytest -q          # geometry, audio grading, tracking, fusion, API e2e
.venv/bin/python -m scripts.eval --synthetic 30  # per-error-type precision/recall
```

## Fretboard model (milestone 3)

Not trained yet: there is no labelled data. Until then the grid comes from the 4-tap manual calibration
and is tracked with Lucas–Kanade optical flow. To train: label ~1,500 frames (see `api/data/fretboard.yaml`),
`pip install -r requirements-train.txt`, `python scripts/train_fretboard.py`, then set `FRETBOARD_MODEL_PATH`.

Optional coaching note: `FEEDBACK_LLM=1` + Anthropic credentials (uses `claude-opus-5-5`; rephrases the
top 3 errors, never adds any).
