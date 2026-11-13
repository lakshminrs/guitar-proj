"""End-to-end API flow on a synthetic take: create -> signed upload -> analyze -> results."""
import os
import time
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.delenv("REDIS_URL", raising=False)
    import importlib

    import app.config, app.db, app.storage, app.queue, app.jobs, app.main  # noqa: E401
    for m in (app.config, app.db, app.storage, app.queue, app.jobs, app.main):
        importlib.reload(m)
    with TestClient(app.main.app) as c:
        yield c


def test_full_flow(client, tmp_path):
    from app.synth import write_take
    from app.theory import expected_events, get_exercise

    exs = client.get("/api/exercises").json()
    assert len(exs) >= 5
    ex = get_exercise(1)
    d_down = next(e.idx for e in expected_events(ex) if e.chord == "D" and e.direction == "D")
    wav = tmp_path / "take.wav"
    clicks = write_take(str(wav), ex, mute_string={"event": d_down, "string": 1}, late_ms={4: 100}, seed=1)

    s = client.post("/api/sessions", json={"exercise_id": 1}).json()
    u = urlparse(s["upload_url"])
    bad = client.put(u.path + "?exp=1&sig=nope", content=b"x")
    assert bad.status_code == 403
    r = client.put(f"{u.path}?{u.query}", content=wav.read_bytes(), headers={"content-type": "audio/wav"})
    assert r.status_code == 200, r.text

    r = client.post(f"/api/sessions/{s['id']}/analyze", json={"clicks_ms": clicks, "count_in_beats": 4})
    assert r.status_code == 200
    for _ in range(120):
        st = client.get(f"/api/sessions/{s['id']}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] == "done", st.get("error")
    assert 0 < st["score"] <= 100
    types = {e["type"] for e in st["errors"]}
    assert {"muted_string", "late"} <= types
    assert st["components"]["fingering"] is None   # audio-only take

    evs = client.get(f"/api/sessions/{s['id']}/events").json()
    assert len(evs) == len(expected_events(ex))
    assert all("scores" in e and "errors" in e for e in evs)
    assert client.get(f"/api/sessions/{s['id']}/media/audio").status_code == 200
    assert client.get(f"/api/sessions/{s['id']}/track").status_code == 404


def _upload_take(client, tmp_path, name, collect, **mistakes):
    from app.synth import write_take
    from app.theory import get_exercise

    wav = tmp_path / f"{name}.wav"
    clicks = write_take(str(wav), get_exercise(1), seed=2, **mistakes)
    s = client.post("/api/sessions", json={"exercise_id": 1}).json()
    u = urlparse(s["upload_url"])
    client.put(f"{u.path}?{u.query}", content=wav.read_bytes(), headers={"content-type": "audio/wav"})
    r = client.post(f"/api/sessions/{s['id']}/analyze",
                    json={"clicks_ms": clicks, "count_in_beats": 4, "collect": collect})
    assert r.status_code == 200
    return s["id"]


def test_scripted_takes_export_as_eval_manifest(client, tmp_path):
    """Collect mode -> labels.json -> GET /api/collect/manifest -> scripts.eval can score it."""
    from app.evaluation import evaluate, manifest_cases
    from app.theory import expected_events, get_exercise

    late = next(e for e in expected_events(get_exercise(1)) if e.bar == 4 and e.slot == 0)
    tags = {"player": "ls", "guitar": "acoustic", "mic": "laptop", "room": "bedroom", "audio": "headphones"}
    good = _upload_take(client, tmp_path, "a", {
        "labels": [{"type": "late", "bar": 4, "beat": 1}], "usable": True, "tags": tags,
        "script": [{"bar": 4, "beat": "1", "text": "come in late", "done": True}]}, late_ms={late.idx: 110})
    _upload_take(client, tmp_path, "b", {"labels": [], "usable": False, "tags": tags, "script": []})
    _upload_take(client, tmp_path, "c", None)          # normal practice take: not in the manifest

    man = client.get("/api/collect/manifest").json()
    assert [m["name"] for m in man] == [good]
    m = man[0]
    assert m["labels"] == [{"type": "late", "bar": 4, "beat": 1}]
    assert m["tags"]["device"] == "laptop" and m["tags"]["guitar"] == "acoustic" and m["tags"]["clean"] == "False"
    assert len(client.get("/api/collect/manifest?include_unusable=true").json()) == 2

    res = evaluate(manifest_cases(man, tmp_path))
    assert res["overall"]["errors"]["late"]["tp"] == 1
