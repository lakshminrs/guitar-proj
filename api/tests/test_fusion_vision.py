"""Fusion with a synthetic vision track: fingertips placed on known cells, a strumming hand
moving through the pattern. Checks fingering score, vision-explained errors, video direction."""
import numpy as np

from app.analysis import geometry as G
from app.analysis.audio import analyze_audio
from app.analysis.fusion import grade
from app.analysis.vision import VisionTrack
from app.synth import write_take
from app.theory import CHORDS, expected_events, get_exercise

FPS = 30
H = G.homography_from_points({"nut_low": (0.1, 0.40), "nut_high": (0.1, 0.52),
                              "f12_low": (0.6, 0.39), "f12_high": (0.6, 0.53)})


def _fret_lm(chord, override=None):
    """21 landmarks: tips on the chord's cells (or overrides {finger: (string, fret)})."""
    lm = np.tile(G.apply_h(H, [[0.08, 1.6]])[0], (21, 1))
    placed = {}
    shape = CHORDS[chord]
    for i, (f, fg) in enumerate(zip(shape.frets, shape.fingers)):
        if f > 0 and fg > 0 and fg not in placed:
            placed[fg] = (6 - i, f)
    placed.update(override or {})
    for fg, (s, f) in placed.items():
        tip = G.apply_h(H, [G.cell_center(s, f)])[0]
        for k in ({1: (6, 7, 8), 2: (10, 11, 12), 3: (14, 15, 16), 4: (18, 19, 20)}[fg]):
            lm[k] = tip
    return lm


def _track(ex, t0_s, beat_s, n_frames, overrides=None, flip=()):
    evs = expected_events(ex)
    times = [t0_s + e.beat_pos * beat_s for e in evs]
    # strum hand canonical y(t): a stroke through the strings at each event
    knots_t, knots_y = [], []
    for i, (e, t) in enumerate(zip(evs, times)):
        d = e.direction if i not in flip else ("U" if e.direction == "D" else "D")
        a, b = (-0.4, 1.4) if d == "D" else (1.4, -0.4)
        knots_t += [t - 0.07, t + 0.07]
        knots_y += [a, b]
    ts = np.arange(n_frames) / FPS
    ys = np.interp(ts, knots_t, knots_y)
    fret_lm, strum_lm = [], []
    for f, t in enumerate(ts):
        i = int(np.clip(np.searchsorted(times, t + 0.2) - 1, 0, len(evs) - 1))
        fret_lm.append(_fret_lm(evs[i].chord, (overrides or {}).get(i)))
        p = G.apply_h(H, [[0.85, ys[f]]])[0]
        strum_lm.append(np.tile(p, (21, 1)))
    return VisionTrack(fps=FPS, width=1280, height=720, H=[H] * n_frames, fret_lm=fret_lm, strum_lm=strum_lm)


def _run(tmp_path, overrides=None, flip=(), **mistakes):
    ex = get_exercise(1)
    p = str(tmp_path / "t.wav")
    clicks = write_take(p, ex, **mistakes)
    audio = analyze_audio(p, [c / 1000 for c in clicks])
    beat_s = 60 / ex["bpm"]
    track = _track(ex, 0.5 + 4 * beat_s, beat_s, int(audio.duration * FPS) + 1, overrides, flip)
    return grade(ex, audio, track, clicks, 4)


def test_fingering_and_direction_from_video(tmp_path):
    res = _run(tmp_path, flip={20}, seed=7)
    assert res.components["fingering"] > 95
    wd = [e for e in res.errors if e["type"] == "wrong_direction"]
    assert [e["bar"] for e in wd] == [4], wd   # event 20 is in bar 4
    assert res.events[20]["direction_sources"]["video"] == "D"


def test_wrong_fret_confirmed_by_vision(tmp_path):
    # event 0 (G): ring finger sits on fret 2 of high E, and audio hears F# instead of G
    res = _run(tmp_path, overrides={0: {3: (1, 2)}}, wrong_fret={"event": 0, "string": 1, "delta": -1}, seed=8)
    wf = [e for e in res.errors if e["type"] == "wrong_fret"]
    assert len(wf) == 1
    assert wf[0]["message"].startswith("Ring finger is on fret 2 of high E, move it up to fret 3")
    assert wf[0]["confidence"] >= 0.8
    assert res.events[0]["scores"]["fingering"] < 1


def test_leaning_finger_explains_muted_string(tmp_path):
    ex = get_exercise(1)
    d_down = next(e.idx for e in expected_events(ex) if e.chord == "D" and e.direction == "D")
    # D chord: high E (string 1) muted; the ring finger (target string 2) lies on string 1's line
    res = _run(tmp_path, overrides={d_down: {3: (1, 3)}}, mute_string={"event": d_down, "string": 1}, seed=9)
    m = [e for e in res.errors if e["type"] == "muted_string"]
    assert len(m) == 1 and m[0]["cause"] == "leaning"
    assert "ring finger is leaning on the high E string" in m[0]["message"]
