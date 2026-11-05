"""Milestone 2 'done when': a deliberately muted string and a late strum both get flagged.
Uses synthetic strummed takes (app/synth.py) with known, deliberate mistakes."""
import pytest

from app.analysis.audio import analyze_audio
from app.analysis.fusion import grade
from app.synth import write_take
from app.theory import expected_events, get_exercise


def _run(tmp_path, exercise_id=1, **mistakes):
    ex = get_exercise(exercise_id)
    p = str(tmp_path / "take.wav")
    clicks = write_take(p, ex, **mistakes)
    audio = analyze_audio(p, [c / 1000 for c in clicks])
    return ex, grade(ex, audio, None, clicks, 4)


def _types(res, **where):
    return [e for e in res.errors if all(e.get(k) == v for k, v in where.items())]


@pytest.fixture(scope="module")
def ex1_events():
    return expected_events(get_exercise(1))


def test_clean_take_scores_high_and_has_no_pitch_or_timing_errors(tmp_path):
    _, res = _run(tmp_path, seed=3)
    assert res.score > 92
    assert res.meta["n_matched"] == res.meta["n_expected"]
    bad = [e for e in res.errors if e["type"] in ("muted_string", "wrong_fret", "late", "early",
                                                    "missed_strum", "strum_too_wide", "extra_strum")]
    assert bad == []


def test_muted_string_and_late_strum_flagged(tmp_path, ex1_events):
    d_down = next(e.idx for e in ex1_events if e.chord == "D" and e.direction == "D")
    _, res = _run(tmp_path, mute_string={"event": d_down, "string": 1}, late_ms={4: 100}, seed=1)
    muted = _types(res, type="muted_string")
    assert any(e["string"] == 1 and e["chord"] == "D" for e in muted), res.errors
    late = _types(res, type="late")
    assert any(e["bar"] == 1 and e["beat"] == 4 and e["offset_ms"] > 60 for e in late), res.errors
    assert len(muted) == 1 and len(late) == 1


def test_wrong_fret_flagged_with_direction(tmp_path):
    _, res = _run(tmp_path, wrong_fret={"event": 0, "string": 1, "delta": -1}, seed=2)
    wf = _types(res, type="wrong_fret")
    assert len(wf) == 1
    e = wf[0]
    assert (e["string"], e["expected_fret"], e["observed_fret"], e["finger"]) == (1, 3, 2, 3)
    assert "flat" in e["message"]


def test_strum_too_wide_on_d(tmp_path, ex1_events):
    d_down = [e.idx for e in ex1_events if e.chord == "D" and e.direction == "D"]
    _, res = _run(tmp_path, extra_low={"event": d_down[1]}, seed=4)
    tw = _types(res, type="strum_too_wide")
    assert len(tw) == 1 and "skip the low E" in tw[0]["message"]


def test_click_latency_is_calibrated_from_bleed(tmp_path):
    _, res = _run(tmp_path, click_latency_ms=40, seed=5)
    assert abs(res.meta["click_offset_ms"] - 40) < 5
    assert not _types(res, type="late")


def test_late_on_changes_pattern(tmp_path, ex1_events):
    late = {e.idx: 90 for e in ex1_events if e.is_change}
    _, res = _run(tmp_path, late_ms=late, seed=6)
    assert any(a["type"] == "late_on_changes" for a in res.aggregates)
    assert res.top[0]["type"] == "late_on_changes"
