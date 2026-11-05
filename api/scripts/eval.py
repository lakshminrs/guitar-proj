"""Per-error-type precision / recall. Run this every time you change a threshold.

  python -m scripts.eval --synthetic 30            # generated takes with known mistakes
  python -m scripts.eval --manifest evalset.json   # your labelled real recordings

Manifest format (real takes, recorded through the app or any A/V file):
  [{"file": "takes/01.webm", "exercise_id": 1, "clicks_ms": [...], "count_in_beats": 4,
    "calibration": {"points": {...}} | null,
    "labels": [{"type": "muted_string", "bar": 3, "beat": 1, "string": 1}, ...]}]
A detection matches a label with the same type and bar/beat (and string, if labelled).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.analysis.audio import analyze_audio  # noqa: E402
from app.analysis.fusion import grade  # noqa: E402
from app.synth import write_take  # noqa: E402
from app.theory import EXERCISES, expected_events, get_exercise  # noqa: E402

TYPES = ["muted_string", "wrong_fret", "strum_too_wide", "late", "early", "missed_strum"]


def synthetic_case(rng: random.Random, i: int, tmp: str) -> tuple[dict, list[dict], object]:
    ex = rng.choice(EXERCISES)
    evs = expected_events(ex)
    downs = [e for e in evs if e.direction == "D"]
    kw, labels = {"seed": i}, []
    kind = rng.choice(["mute", "wrong_fret", "too_wide", "late", "early", "clean"])
    if kind == "mute":
        e = rng.choice(downs)
        s = rng.choice([s for s in range(1, 7) if e.chord and get_frets(e.chord)[6 - s] >= 0])
        kw["mute_string"] = {"event": e.idx, "string": s}
        labels.append({"type": "muted_string", "bar": e.bar, "beat": e.beat, "string": s})
    elif kind == "wrong_fret":
        cands = [(e, s) for e in downs for s in range(1, 7) if get_frets(e.chord)[6 - s] > 0]
        e, s = rng.choice(cands)
        kw["wrong_fret"] = {"event": e.idx, "string": s, "delta": -1}
        labels.append({"type": "wrong_fret", "bar": e.bar, "beat": e.beat, "string": s})
    elif kind == "too_wide":
        cands = [e for e in downs if get_frets(e.chord)[0] < 0]
        if cands:
            e = rng.choice(cands)
            kw["extra_low"] = {"event": e.idx}
            labels.append({"type": "strum_too_wide", "bar": e.bar, "beat": e.beat})
    elif kind in ("late", "early"):
        e = rng.choice(evs)
        kw["late_ms"] = {e.idx: rng.choice([70, 90, 110]) * (1 if kind == "late" else -1)}
        labels.append({"type": kind, "bar": e.bar, "beat": e.beat})
    path = os.path.join(tmp, f"take{i}.wav")
    clicks = write_take(path, ex, **kw)
    audio = analyze_audio(path, [c / 1000 for c in clicks])
    return ex, labels, grade(ex, audio, None, clicks, 4)


def get_frets(chord: str):
    from app.theory import CHORDS
    return CHORDS[chord].frets


def manifest_case(item: dict):
    from app.analysis.vision import analyze_video
    from app.jobs import demux
    ex = get_exercise(item["exercise_id"])
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        mp4, wav = demux(item["file"], Path(d))
        audio = analyze_audio(wav, [c / 1000 for c in item["clicks_ms"]])
        vision = analyze_video(mp4, item.get("calibration")) if mp4 else None
        res = grade(ex, audio, vision, item["clicks_ms"], item.get("count_in_beats", 4))
    return ex, item["labels"], res


def match(dets: list[dict], labels: list[dict]):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--manifest")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tp, fp_n, fn_n = defaultdict(int), defaultdict(int), defaultdict(int)
    cases = []
    if args.synthetic:
        rng = random.Random(args.seed)
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(args.synthetic):
                cases.append(synthetic_case(rng, i, tmp))
                print(".", end="", flush=True)
        print()
    if args.manifest:
        cases += [manifest_case(it) for it in json.load(open(args.manifest))]
    for _ex, labels, res in cases:
        dets = [e for e in res.errors if e["type"] in TYPES]
        t = match(dets, labels)
        for typ in TYPES:
            n_lab = sum(1 for lab in labels if lab["type"] == typ)
            n_det = sum(1 for d in dets if d["type"] == typ)
            tp[typ] += t[typ]
            fn_n[typ] += n_lab - t[typ]
            fp_n[typ] += n_det - t[typ]
    print(f"{'type':<16}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'recall':>8}")
    for typ in TYPES:
        p = tp[typ] / (tp[typ] + fp_n[typ]) if tp[typ] + fp_n[typ] else float("nan")
        r = tp[typ] / (tp[typ] + fn_n[typ]) if tp[typ] + fn_n[typ] else float("nan")
        print(f"{typ:<16}{tp[typ]:>5}{fp_n[typ]:>5}{fn_n[typ]:>5}{p:>8.2f}{r:>8.2f}")


if __name__ == "__main__":
    main()
