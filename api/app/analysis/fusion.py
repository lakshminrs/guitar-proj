"""Fusion & grading: every expected strum becomes one graded event.

Audio is the judge of correctness (did it sound right, on time); vision explains the cause
(where were the fingers, which way did the hand move).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..theory import CHORDS, beats_per_bar, expected_events
from . import geometry as G
from .audio import AudioAnalysis, verify_chord
from .params import P
from .rules import aggregate_errors, event_errors, rank_top
from .vision import VisionTrack

WEIGHTS = {"fingering": 0.30, "pitch": 0.35, "timing": 0.25, "direction": 0.10}


@dataclass
class GradeResult:
    events: list[dict]
    errors: list[dict]
    aggregates: list[dict]
    top: list[dict]
    score: float
    components: dict
    meta: dict


class BeatClock:
    """Maps exercise beat positions to recording time (ms) using the logged clicks.
    Click k is beat k of the take, the first `count_in` clicks are the count-in."""

    def __init__(self, clicks_ms: list[float], count_in: int, bpm: float, offset_ms: float = 0.0):
        self.c = np.asarray(clicks_ms, dtype=float) + offset_ms
        self.count_in = count_in
        self.beat_ms = 60000.0 / bpm
        if len(self.c) >= 2:
            self.beat_ms = float(np.median(np.diff(self.c)))

    def __call__(self, beat_pos: float) -> float:
        k = beat_pos + self.count_in
        if len(self.c) == 0:
            return k * self.beat_ms
        i = int(np.floor(k))
        if i < 0:
            return float(self.c[0] + k * self.beat_ms)
        if i + 1 < len(self.c):
            return float(self.c[i] + (k - i) * (self.c[i + 1] - self.c[i]))
        return float(self.c[-1] + (k - (len(self.c) - 1)) * self.beat_ms)


def timing_score(dt_ms: float) -> float:
    a = abs(dt_ms)
    if a <= 20:
        return 1.0
    return float(max(0.0, 1 - (a - 20) / 100))


def _match(strums, exp_times: list[float], windows: list[float]) -> dict[int, int]:
    """Greedy one-to-one nearest matching of strums to expected events."""
    pairs = []
    for ei, te in enumerate(exp_times):
        for si, s in enumerate(strums):
            d = abs(s.t * 1000 - te)
            if d <= windows[ei]:
                pairs.append((d, ei, si))
    pairs.sort()
    used_e, used_s, m = set(), set(), {}
    for d, ei, si in pairs:
        if ei in used_e or si in used_s:
            continue
        used_e.add(ei)
        used_s.add(si)
        m[ei] = si
    return m


def _relative_sustain(strums, match: dict[int, int], n_exp: int) -> dict[int, float]:
    """Each matched strum's sustain relative to the player's typical sustain for a similar gap
    to the next strum. A chord's natural decay depends on the guitar and the gap, so a fixed
    threshold would flag every long gap on a dead-sounding guitar; comparing against the take
    itself flags only chords cut short. Only strums followed by a long enough gap are judged,
    and never the last one (nothing tells us when it should stop)."""
    cand = {ei: strums[si] for ei, si in match.items()
            if ei < n_exp - 1 and strums[si].gap >= P.choke_min_gap_s}
    out = {}
    for ei, st in cand.items():
        peers = [o.sustain for oj, o in cand.items() if oj != ei and 0.8 <= o.gap / st.gap <= 1.25]
        if len(peers) >= 3:
            out[ei] = round(st.sustain / (float(np.median(peers)) + 1e-9), 3)
    return out


def _beat_label(slot: int, subdiv: int) -> str:
    beat = slot // subdiv + 1
    r = slot % subdiv
    if r == 0:
        return str(beat)
    if subdiv == 2:
        return f"{beat}&"
    return f"{beat}.{r + 1}"


def grade(exercise: dict, audio: AudioAnalysis, vision: VisionTrack | None,
          clicks_ms: list[float], count_in: int) -> GradeResult:
    offset = audio.click_offset_ms or 0.0
    clock = BeatClock(clicks_ms, count_in, exercise["bpm"], offset)
    exp = expected_events(exercise)
    bpb = beats_per_bar(exercise["time_sig"])
    exp_t = [clock(e.beat_pos) for e in exp]
    windows = [min(0.5 * clock.beat_ms / e.subdiv, 180.0) for e in exp]

    # ignore count-in noise and anything far after the end
    t_first, t_last = exp_t[0] - windows[0], exp_t[-1] + 1500
    strums = [s for s in audio.strums if t_first <= s.t * 1000 <= t_last and s.energy >= P.ghost_energy]
    match = _match(strums, exp_t, windows)
    # score-informed recovery: search each unmatched expected window for a quieter strum
    recovered = 0
    for ei, te in enumerate(exp_t):
        if ei in match:
            continue
        st = audio.recover(te / 1000, windows[ei] / 1000)
        if st is not None:
            recovered += 1
    if recovered:
        strums = [s for s in audio.strums if t_first <= s.t * 1000 <= t_last and s.energy >= P.ghost_energy]
        match = _match(strums, exp_t, windows)

    rel_sustain = _relative_sustain(strums, match, len(exp))
    fps = vision.fps if vision else 30.0
    has_grid = bool(vision and any(h is not None for h in vision.H))
    events: list[dict] = []
    all_errors: list[dict] = []
    prev_chord_of_bar = {}
    for e in exp:
        prev_chord_of_bar.setdefault(e.bar, exercise["bars"][e.bar - 2]["chord"] if e.bar > 1 else None)

    for ei, e in enumerate(exp):
        te = exp_t[ei]
        si = match.get(ei)
        st = strums[si] if si is not None else None
        t_ms = st.t * 1000 if st else te
        ev = {"idx": ei, "t_ms": round(t_ms, 1), "expected_t_ms": round(te, 1), "bar": e.bar,
              "beat": e.beat, "slot": e.slot, "beat_label": _beat_label(e.slot, e.subdiv),
              "expected_chord": e.chord, "prev_chord": prev_chord_of_bar[e.bar],
              "expected_direction": e.direction, "is_offbeat": e.is_offbeat, "is_change": e.is_change,
              "frame": int(round(t_ms / 1000 * fps)), "missed": st is None,
              "sustain": None if st is None else round(st.sustain, 3),
              "rel_sustain": rel_sustain.get(ei),
              "next_is_change": bool(ei + 1 < len(exp) and exp[ei + 1].is_change)}
        scores: dict[str, float | None] = {k: None for k in WEIGHTS}
        verdict, fingers_canon, fingers_img = None, None, {}

        if st is not None:
            dt = t_ms - te
            ev["timing_ms"] = round(dt, 1)
            scores["timing"] = timing_score(dt)
            verdict = verify_chord(st, e.chord, e.direction)
            scores["pitch"] = verdict["pitch_score"]

            # direction: video first (wrist velocity), audio string-order as cross-check
            a_dir, a_conf = audio.direction(st, e.chord)
            v_dir, v_conf, speed = vision.strum_direction(st.t) if vision else (None, 0.0, 0.0)
            if v_dir and a_dir:
                obs, conf = (v_dir, min(1.0, 0.5 + v_conf / 2 + a_conf / 4)) if v_dir == a_dir else (v_dir, v_conf * 0.6)
            elif v_dir:
                obs, conf = v_dir, v_conf * 0.85
            else:
                obs, conf = a_dir, a_conf * 0.7
            ev["observed_direction"], ev["direction_confidence"] = obs, round(conf, 2)
            ev["direction_sources"] = {"video": v_dir, "audio": a_dir}
            if obs:
                scores["direction"] = 1.0 if obs == e.direction else 0.0

            if has_grid:
                fingers_canon = vision.fingertips_canon(st.t)
                fingers_img = vision.fingertips_image(st.t)
                targets = CHORDS[e.chord].fingering_targets()
                if fingers_canon and targets:
                    hits = 0
                    for tgt in targets:
                        xy = fingers_canon.get(tgt["finger"])
                        if xy is None:
                            continue
                        s_obs, f_obs = G.canon_to_cell(*xy)
                        hits += int(f_obs == tgt["fret"] and (tgt["barre"] or s_obs == tgt["string"]))
                    scores["fingering"] = hits / len(targets)
                elif fingers_canon and not targets:  # all-open chord: nothing to finger
                    scores["fingering"] = 1.0
        else:
            scores.update({"pitch": 0.0, "timing": 0.0, "direction": 0.0})

        avail = {k: v for k, v in scores.items() if v is not None}
        wsum = sum(WEIGHTS[k] for k in avail)
        scores["total"] = round(sum(WEIGHTS[k] * v for k, v in avail.items()) / wsum, 3) if wsum else 0.0
        ev["scores"] = {k: (round(v, 3) if v is not None else None) for k, v in scores.items()}
        ev["notes"] = ({"strings": {str(6 - i): s for i, s in verdict["strings"].items()},
                        "chroma_sim": verdict["chroma_sim"], "unexpected": verdict["unexpected"]}
                       if verdict else None)
        ev["fingertips"] = {
            "canon": {str(k): [round(v[0], 4), round(v[1], 4)] for k, v in (fingers_canon or {}).items()},
            "image": {str(k): [round(v[0], 4), round(v[1], 4)] for k, v in fingers_img.items()},
            "cells": {str(k): list(G.canon_to_cell(*v)) for k, v in (fingers_canon or {}).items()},
        }
        errs = event_errors(ev, verdict, fingers_canon)
        ev["errors"] = errs
        all_errors.extend(errs)
        events.append(ev)

    # extra strums (played where nothing was expected)
    used = set(match.values())
    matched_t = [strums[i].t for i in used]
    extras = [s for i, s in enumerate(strums) if i not in used and s.energy > 0.35
              and s.rise > 1.6 and s.strength > 0.5 and all(abs(s.t - mt) > 0.08 for mt in matched_t)]
    for s in extras:
        # attach to the nearest expected event's bar for context
        ei = int(np.argmin([abs(s.t * 1000 - t) for t in exp_t]))
        all_errors.append({"t_ms": round(s.t * 1000), "bar": exp[ei].bar, "beat": exp[ei].beat,
                           "chord": exp[ei].chord, "type": "extra_strum", "string": None, "finger": None,
                           "expected_fret": None, "observed_fret": None, "severity": 0.3, "confidence": 0.6,
                           "message": f"Extra strum in bar {exp[ei].bar} that isn't in the pattern.",
                           "overlay": {"frame": int(round(s.t * fps)), "draw": [{"kind": "beat", "style": "error"}]}})

    aggregates = aggregate_errors(events)
    top = rank_top(all_errors, aggregates)
    all_errors.sort(key=lambda x: x["t_ms"])

    comps = {}
    for k in WEIGHTS:
        vals = [ev["scores"][k] for ev in events if ev["scores"][k] is not None]
        comps[k] = round(100 * float(np.mean(vals)), 1) if vals else None
    score = round(100 * float(np.mean([ev["scores"]["total"] for ev in events])), 1) if events else 0.0
    meta = {"click_offset_ms": audio.click_offset_ms, "n_expected": len(exp),
            "n_detected_strums": len(strums), "n_matched": len(match), "n_recovered": recovered, "n_extra": len(extras),
            "beat_ms": clock.beat_ms, "vision": bool(vision), "grid": has_grid,
            "basic_pitch": audio.used_basic_pitch, "bars": len(exercise["bars"]), "beats_per_bar": bpb}
    return GradeResult(events=events, errors=all_errors, aggregates=aggregates, top=top,
                       score=score, components=comps, meta=meta)
