"""Recognition accuracy report. Run this every time you change a threshold or a detector.

  python -m scripts.eval --synthetic 30                 # generated takes with exact ground truth
  python -m scripts.eval --synthetic 30 --augment       # + room reverb, mic tilt, noise
  python -m scripts.eval --manifest takes/manifest.json # labelled real recordings (see below)
  python -m scripts.eval --compare A.json B.json        # diff two saved runs

Every run is saved to eval_results/<timestamp>.json (or --out).

Manifest format (export from the app: GET /api/collect/manifest):
  [{"file": "takes/01.webm", "exercise_id": 1, "clicks_ms": [...], "count_in_beats": 4,
    "calibration": {"points": {...}} | null,
    "labels": [{"type": "muted_string", "bar": 3, "beat": 1, "string": 1}, ...],
    "truth": [{"idx": 0, "t_ms": 2513.0, "direction": "D", "strings": {"1": "ok", ...}}, ...],   # optional
    "finger_frames": [{"t_ms": 4100, "finger": 3, "string": 1, "fret": 3}, ...],               # optional
    "tags": {"device": "iphone", "guitar": "acoustic"}}]
A detection matches a label with the same type and bar/beat (and string, if labelled).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.evaluation import evaluate, fmt_float, manifest_cases, synthetic_cases  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "eval_results"


def print_report(r: dict, title: str = "overall") -> None:
    m = r if "errors" in r else r["overall"]
    print(f"\n=== {title}")
    print(f"{'error type':<18}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'recall':>8}{'F1':>7}")
    for t, v in m["errors"].items():
        if v["tp"] + v["fp"] + v["fn"] == 0:
            continue
        print(f"{t:<18}{v['tp']:>5}{v['fp']:>5}{v['fn']:>5}{fmt_float(v['precision']):>8}"
              f"{fmt_float(v['recall']):>8}{fmt_float(v['f1']):>7}")
    if any(v["tp"] + v["fp"] + v["fn"] for v in m["strings"].values()):
        print(f"{'string verdict':<18}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'recall':>8}{'F1':>7}")
        for c, v in m["strings"].items():
            print(f"{c:<18}{v['tp']:>5}{v['fp']:>5}{v['fn']:>5}{fmt_float(v['precision']):>8}"
                  f"{fmt_float(v['recall']):>8}{fmt_float(v['f1']):>7}")
    t, o = m["timing"], m["onsets"]
    if t["n"]:
        print(f"timing error      median {fmt_float(t['median_abs_ms'], 1)} ms   p90 {fmt_float(t['p90_abs_ms'], 1)} ms   (n={t['n']})")
    if o["f1"] is not None:
        print(f"onset detection   P {fmt_float(o['precision'])}  R {fmt_float(o['recall'])}  F1 {fmt_float(o['f1'])}  (±50 ms)")
    for src, v in m["direction"].items():
        print(f"direction {src:<7} accuracy {fmt_float(v['accuracy'])}  coverage {fmt_float(v['coverage'])}")
    if m["fingers"]["n"]:
        print(f"fingertip cells   accuracy {fmt_float(m['fingers']['accuracy'])}  (n={m['fingers']['n']})")
    if m["wrong_chord_false_alarm_rate"] is not None:
        print(f"wrong_chord false alarms per strum: {fmt_float(m['wrong_chord_false_alarm_rate'], 3)}")


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    fa, fb = _flat(a["overall"]), _flat(b["overall"])
    print(f"{'metric':<44}{'A':>10}{'B':>10}{'Δ':>10}")
    for k in sorted(set(fa) | set(fb)):
        if not any(k.endswith(s) for s in (".f1", ".precision", ".recall", "_ms", ".accuracy", "rate")):
            continue
        va, vb = fa.get(k), fb.get(k)
        d = (vb - va) if va is not None and vb is not None else None
        if d is not None and abs(d) < 1e-9:
            continue
        print(f"{k:<44}{fmt_float(va, 3):>10}{fmt_float(vb, 3):>10}{fmt_float(d, 3):>10}")
    print(f"{'objective':<44}{a.get('objective', 0):>10.3f}{b.get('objective', 0):>10.3f}"
          f"{b.get('objective', 0) - a.get('objective', 0):>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="room/mic/noise augmentation on synthetic takes")
    ap.add_argument("--manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by", default="device,guitar,source", help="tags to break results down by")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)

    cases = []
    if args.synthetic:
        cases += synthetic_cases(args.synthetic, seed=args.seed, augment=args.augment)
    if args.manifest:
        mp = Path(args.manifest)
        cases += manifest_cases(json.load(open(mp)), mp.parent)
    if not cases:
        ap.error("nothing to evaluate: pass --synthetic N and/or --manifest FILE")
    r = evaluate(cases, by_tag=tuple(t for t in args.by.split(",") if t))
    print_report(r)
    for g, m in (r.get("groups") or {}).items():
        print_report(m, g)
    print(f"\nobjective: {r['objective']:.4f}")

    from app.analysis.params import P
    from dataclasses import asdict
    r["params"] = asdict(P)
    r["args"] = vars(args)
    out = Path(args.out) if args.out else RESULTS / time.strftime("%Y%m%d-%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
