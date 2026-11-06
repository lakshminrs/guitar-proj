"""Evaluation core shared by scripts/eval.py and scripts/tune.py.

A *case* is one take plus whatever ground truth exists for it:
  labels  - error-level labels [{type, bar, beat, string?}]  (scripted takes, synthetic takes)
  truth   - per expected event [{idx, t_ms, direction, strings{1..6: state}, choked, skipped}]
            (synthetic: exact; real: from the collect tool + onset marking; any field may be None)
  finger_frames - [{t_ms, finger, string, fret}] hand-labelled fingertip cells (real takes)
  tags    - {"device": ..., "guitar": ..., "source": ...} for per-group breakdowns

Feature extraction (basic-pitch, CQT, MediaPipe) is the expensive part and is done once per case;
`regrade()` re-runs only strum detection + grading, so threshold searches are cheap.
"""
from __future__ import annotations

import math
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis.audio import AudioAnalysis, analyze_audio
from .analysis.fusion import GradeResult, grade
from .analysis.vision import VisionTrack
from .synth import write_take
from .theory import CHORDS, EXERCISES, expected_events, get_exercise

ERROR_TYPES = ["muted_string", "wrong_fret", "strum_too_wide", "wrong_chord", "late", "early",
               "missed_strum", "extra_strum", "wrong_direction", "choked_chord"]
STRING_CLASSES = ["missing", "wrong_fret", "muted_sounding"]
ONSET_PARAMS = {"global_delta", "merge_window_s", "strum_time_bias_ms"}


@dataclass
class Case:
    name: str
    exercise: dict
    audio: AudioAnalysis
    vision: VisionTrack | None
    clicks_ms: list[float]
    count_in: int
    labels: list[dict] = field(default_factory=list)
    truth: list[dict] | None = None
    finger_frames: list[dict] = field(default_factory=list)
    tags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# building cases
# ---------------------------------------------------------------------------

def _frets(chord: str):
    return CHORDS[chord].frets


def synthetic_cases(n: int, seed: int = 0, augment: bool = False, tmp: str | None = None) -> list[Case]:
    """Generated takes with 0-3 known mistakes each, exact ground truth, and a mix of
    speaker (audible click) and headphone recordings."""
    rng = random.Random(seed)
    out = []
    with tempfile.TemporaryDirectory() as d:
        d = tmp or d
        for i in range(n):
            ex = rng.choice(EXERCISES)
            evs = expected_events(ex)
            downs = [e for e in evs if e.direction == "D"]
            kw: dict = {"seed": seed * 1000 + i, "late_ms": {}}
            labels: list[dict] = []
            used: set[int] = set()

            def pick(cands):
                cands = [c for c in cands if (c[0] if isinstance(c, tuple) else c).idx not in used]
                if not cands:
                    return None
                c = rng.choice(cands)
                used.add((c[0] if isinstance(c, tuple) else c).idx)
                return c

            kinds = rng.sample(["mute", "wrong_fret", "too_wide", "late", "early", "skip", "choke", "flip"],
                               k=rng.choice([0, 1, 1, 2, 2, 3]))
            for kind in kinds:
                if kind == "mute":
                    c = pick([(e, s) for e in downs for s in range(1, 7) if _frets(e.chord)[6 - s] >= 0])
                    if c:
                        e, s = c
                        kw["mute_string"] = {"event": e.idx, "string": s}
                        labels.append({"type": "muted_string", "bar": e.bar, "beat": e.beat, "string": s})
                elif kind == "wrong_fret":
                    c = pick([(e, s) for e in downs for s in range(1, 7) if _frets(e.chord)[6 - s] > 0])
                    if c:
                        e, s = c
                        kw["wrong_fret"] = {"event": e.idx, "string": s, "delta": -1}
                        labels.append({"type": "wrong_fret", "bar": e.bar, "beat": e.beat, "string": s})
                elif kind == "too_wide":
                    c = pick([e for e in downs if _frets(e.chord)[0] < 0])
                    if c:
                        kw["extra_low"] = {"event": c.idx}
                        labels.append({"type": "strum_too_wide", "bar": c.bar, "beat": c.beat})
                elif kind in ("late", "early"):
                    c = pick(evs)
                    if c:
                        kw["late_ms"][c.idx] = rng.choice([70, 90, 110]) * (1 if kind == "late" else -1)
                        labels.append({"type": kind, "bar": c.bar, "beat": c.beat})
                elif kind == "skip":
                    c = pick(evs)
                    if c:
                        kw["skip"] = {c.idx}
                        labels.append({"type": "missed_strum", "bar": c.bar, "beat": c.beat})
                elif kind == "choke":
                    # only where there's room for the chord to (fail to) ring
                    c = pick([e for e in downs if e.subdiv == 1])
                    if c:
                        kw["choke"] = {"event": c.idx, "after_s": 0.12}
                        labels.append({"type": "choked_chord", "bar": c.bar, "beat": c.beat})
                elif kind == "flip":
                    c = pick(evs)
                    if c:
                        kw["flip_direction"] = {c.idx}
                        labels.append({"type": "wrong_direction", "bar": c.bar, "beat": c.beat})
            headphones = rng.random() < 0.3
            if headphones:
                kw["click_hz"] = None
            else:
                kw["click_latency_ms"] = rng.choice([0, 0, 20, 45])
            if augment:
                kw["augment"] = {"reverb": rng.uniform(0, 0.8), "tilt_db": rng.uniform(-6, 6),
                                 "snr_db": rng.uniform(20, 45)}
            truth: list[dict] = []
            path = str(Path(d) / f"take{i}.wav")
            clicks = write_take(path, ex, truth=truth, **kw)
            audio = analyze_audio(path, [c / 1000 for c in clicks], count_in=4)
            out.append(Case(name=f"synth{i}", exercise=ex, audio=audio, vision=None, clicks_ms=clicks,
                            count_in=4, labels=labels, truth=truth,
                            tags={"source": "synthetic", "device": "headphones" if headphones else "speaker",
                                  "augment": str(bool(augment))}))
    return out


def manifest_cases(items: list[dict], base: Path) -> list[Case]:
    from .analysis.vision import analyze_video
    from .jobs import demux
    out = []
    for k, item in enumerate(items):
        ex = get_exercise(item["exercise_id"])
        f = Path(item["file"])
        f = f if f.is_absolute() else base / f
        with tempfile.TemporaryDirectory() as d:
            mp4, wav = demux(str(f), Path(d))
            audio = analyze_audio(wav, [c / 1000 for c in item["clicks_ms"]],
                                  count_in=item.get("count_in_beats", 4))
            vision = analyze_video(mp4, item.get("calibration")) if mp4 else None
        out.append(Case(name=item.get("name") or f.stem or f"take{k}", exercise=ex, audio=audio, vision=vision,
                        clicks_ms=item["clicks_ms"], count_in=item.get("count_in_beats", 4),
                        labels=item.get("labels", []), truth=item.get("truth"),
                        finger_frames=item.get("finger_frames", []),
                        tags={"source": "real", **item.get("tags", {})}))
    return out


# ---------------------------------------------------------------------------
# grading + metrics
# ---------------------------------------------------------------------------

def regrade(case: Case, redetect: bool = False) -> GradeResult:
    if redetect:
        case.audio.detect()
    return grade(case.exercise, case.audio, case.vision, case.clicks_ms, case.count_in)


def _match_labels(dets: list[dict], labels: list[dict]) -> dict[str, int]:
    tp, used = defaultdict(int), set()
    for lab in labels:
        for j, d in enumerate(dets):
            if j in used or d["type"] != lab["type"] or d["bar"] != lab["bar"] or d["beat"] != lab["beat"]:
                continue
            if lab.get("string") and d.get("string") != lab["string"]:
                continue
            used.add(j)
            tp[lab["type"]] += 1
            break
    return tp


class Counts:
    """Accumulates raw counts across cases; `summary()` turns them into rates."""

    def __init__(self):
        self.err = {t: [0, 0, 0] for t in ERROR_TYPES}            # tp, fp, fn
        self.strings = {c: [0, 0, 0] for c in STRING_CLASSES}
        self.timing_abs: list[float] = []
        self.onset = [0, 0, 0]                                    # tp, n_ref, n_est
        self.direction = defaultdict(lambda: [0, 0])              # source -> [correct, judged]
        self.n_dir_events = 0
        self.finger = [0, 0]                                      # correct cells, labelled tips
        self.wrong_chord_fp = 0
        self.n_events = 0

    def add(self, case: Case, res: GradeResult) -> None:
        # error-level labels
        dets = [e for e in res.errors if e["type"] in ERROR_TYPES]
        tp = _match_labels(dets, case.labels)
        for t in ERROR_TYPES:
            n_lab = sum(1 for x in case.labels if x["type"] == t)
            n_det = sum(1 for x in dets if x["type"] == t)
            self.err[t][0] += tp[t]
            self.err[t][1] += n_det - tp[t]
            self.err[t][2] += n_lab - tp[t]

        if case.truth:
            truth = {t["idx"]: t for t in case.truth if t.get("idx") is not None}
            self.n_events += len(res.events)
            for ev in res.events:
                tr = truth.get(ev["idx"])
                if tr is None:
                    continue
                if any(e["type"] == "wrong_chord" for e in ev["errors"]):
                    self.wrong_chord_fp += 1
                if ev.get("missed") or tr.get("skipped"):
                    continue
                if tr.get("t_ms") is not None:
                    self.timing_abs.append(abs(ev["t_ms"] - tr["t_ms"]))
                if tr.get("direction"):
                    self.n_dir_events += 1
                    obs, srcs = ev.get("observed_direction"), ev.get("direction_sources") or {}
                    for name, d in (("fused", obs), ("video", srcs.get("video")), ("audio", srcs.get("audio"))):
                        if d:
                            self.direction[name][1] += 1
                            self.direction[name][0] += int(d == tr["direction"])
                pred = (ev.get("notes") or {}).get("strings") or {}
                for s_no, t_state in (tr.get("strings") or {}).items():
                    p_state = (pred.get(str(s_no)) or {}).get("status")
                    if t_state == "not_struck" or p_state in (None, "not_required"):
                        continue
                    for c in STRING_CLASSES:
                        if p_state == c and t_state == c:
                            self.strings[c][0] += 1
                        elif p_state == c:
                            self.strings[c][1] += 1
                        elif t_state == c:
                            self.strings[c][2] += 1
            # onset detection F-measure (50 ms window) over the played span
            ref = sorted(t["t_ms"] for t in case.truth if t.get("t_ms") is not None)
            if ref:
                lo, hi = ref[0] - 200, ref[-1] + 200
                est = sorted(s.t * 1000 for s in case.audio.strums if lo <= s.t * 1000 <= hi)
                self.onset[1] += len(ref)
                self.onset[2] += len(est)
                self.onset[0] += _onset_tp(ref, est, 50.0)

        if case.finger_frames and case.vision is not None:
            from .analysis import geometry as G
            for lab in case.finger_frames:
                tips = case.vision.fingertips_canon(lab["t_ms"] / 1000, window=0.05)
                xy = tips.get(lab["finger"])
                self.finger[1] += 1
                if xy is not None and G.canon_to_cell(*xy) == (lab["string"], lab["fret"]):
                    self.finger[0] += 1

    def summary(self) -> dict:
        def prf(tp, fp, fn):
            p = tp / (tp + fp) if tp + fp else None
            r = tp / (tp + fn) if tp + fn else None
            f = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) else None)
            return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}

        ta = np.array(self.timing_abs) if self.timing_abs else None
        otp, nref, nest = self.onset
        op = otp / nest if nest else None
        orr = otp / nref if nref else None
        return {
            "errors": {t: prf(*v) for t, v in self.err.items()},
            "strings": {c: prf(*v) for c, v in self.strings.items()},
            "timing": {"n": len(self.timing_abs),
                       "median_abs_ms": float(np.median(ta)) if ta is not None else None,
                       "p90_abs_ms": float(np.percentile(ta, 90)) if ta is not None else None},
            "onsets": {"precision": op, "recall": orr,
                       "f1": (2 * op * orr / (op + orr)) if op and orr else None},
            "direction": {k: {"accuracy": c / n if n else None, "coverage": n / self.n_dir_events if self.n_dir_events else None}
                          for k, (c, n) in self.direction.items()},
            "fingers": {"n": self.finger[1], "accuracy": self.finger[0] / self.finger[1] if self.finger[1] else None},
            "wrong_chord_false_alarm_rate": self.wrong_chord_fp / self.n_events if self.n_events else None,
        }


def _onset_tp(ref: list[float], est: list[float], win: float) -> int:
    used, tp = set(), 0
    for r in ref:
        best, bd = None, win
        for j, e in enumerate(est):
            if j not in used and abs(e - r) <= bd:
                best, bd = j, abs(e - r)
        if best is not None:
            used.add(best)
            tp += 1
    return tp


def evaluate(cases: list[Case], redetect: bool = False, by_tag: tuple[str, ...] = ()) -> dict:
    total, groups = Counts(), defaultdict(Counts)
    for c in cases:
        res = regrade(c, redetect)
        total.add(c, res)
        for tag in by_tag:
            if tag in c.tags:
                groups[f"{tag}={c.tags[tag]}"].add(c, res)
    out = {"n_cases": len(cases), "overall": total.summary()}
    if groups:
        out["groups"] = {k: v.summary() for k, v in sorted(groups.items())}
    out["objective"] = objective(out["overall"])
    return out


def objective(m: dict) -> float:
    """Single number for tuning (higher is better): mean F1 over error types and string classes
    that have any labels, plus onset F1 and direction accuracy, minus a timing-error penalty."""
    f1s = [v["f1"] for v in list(m["errors"].values()) + list(m["strings"].values())
           if v["f1"] is not None and (v["tp"] + v["fn"]) > 0]
    score = float(np.mean(f1s)) if f1s else 0.0
    if m["onsets"]["f1"] is not None:
        score += 0.5 * m["onsets"]["f1"]
    fused = m["direction"].get("fused", {}).get("accuracy")
    if fused is not None:
        score += 0.25 * fused
    med = m["timing"]["median_abs_ms"]
    if med is not None:
        score -= 0.01 * max(0.0, med - 5.0)
    wc = m["wrong_chord_false_alarm_rate"]
    if wc:
        score -= wc
    return round(score, 4)


def fmt_float(x, nd=2) -> str:
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
