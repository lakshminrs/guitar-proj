"""Strum-pattern inference: what rhythm did the player actually strum, bar by bar?

Grading checks each *expected* strum. This goes the other way: every played strum is placed on
the bar's subdivision grid and given its observed direction, giving a played pattern in the same
notation as the chart ("D-DU-UDU"; "x" = a strum whose direction couldn't be judged). Comparing the
two shows *how* the rhythm differs (a missing up-strum on 2&, an extra down on 4), and the spread
of played patterns shows how consistent the strumming hand is.

Input positions are in beats from bar 1 beat 1, so the source of the beat grid (metronome clicks
today, a tempo tracker for free-tempo playing later) doesn't matter here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..theory import beats_per_bar


@dataclass
class PlayedStrum:
    beat_pos: float          # beats from bar 1 beat 1
    direction: str | None    # "D" | "U" | None (unknown)
    strength: float = 1.0


def _render(pat: str, res: int, sub: int) -> str:
    """Re-render a chart pattern written with `sub` slots/beat at `res` slots/beat."""
    k = res // sub
    return "".join(ch + "-" * (k - 1) for ch in pat)


def _label(slot: int, res: int) -> str:
    beat, r = slot // res + 1, slot % res
    if r == 0:
        return str(beat)
    if res == 2:
        return f"{beat}&"
    return f"{beat}{['', 'e', '&', 'a'][r]}" if res == 4 else f"{beat}.{r + 1}"


def infer_patterns(exercise: dict, played: list[PlayedStrum]) -> dict:
    bpb = beats_per_bar(exercise["time_sig"])
    pats = [b["strum_pattern"].replace(" ", "") for b in exercise["bars"]]
    subs = [len(p) // bpb for p in pats]
    # resolution: the chart's own, but at least eighths so extra off-beat strums are visible
    res = max(max(subs), 2)
    if any(res % s for s in subs):
        res = 4 if all(4 % s == 0 for s in subs) else max(subs)
    n_bars, per_bar = len(pats), bpb * res

    slots: dict[int, PlayedStrum] = {}
    for p in played:
        s = int(round(p.beat_pos * res))
        if not 0 <= s < n_bars * per_bar or abs(p.beat_pos * res - s) > 0.5:
            continue
        if s not in slots or p.strength > slots[s].strength:   # two strums on one slot: keep the louder
            slots[s] = p

    bars, played_strs = [], []
    for b, pat in enumerate(pats):
        exp = _render(pat, res, subs[b])
        got = []
        for k in range(per_bar):
            p = slots.get(b * per_bar + k)
            got.append("-" if p is None else (p.direction or "x"))
        got_s = "".join(got)
        diffs = []
        for k, (e, g) in enumerate(zip(exp, got_s)):
            if e == g or (g == "x" and e in "DU"):
                continue
            kind = ("missing" if g == "-" else "extra" if e == "-" else
                    "direction" if g in "DU" else None)
            if kind:
                diffs.append({"slot": k, "beat_label": _label(k, res), "expected": e, "played": g, "kind": kind})
        bars.append({"bar": b + 1, "chord": exercise["bars"][b]["chord"], "expected": exp, "played": got_s,
                     "matches": not diffs, "diffs": diffs})
        played_strs.append(got_s)

    # consistency: how often the player repeats their own most common rhythm (directions aside,
    # so an unjudged "x" doesn't count as a different pattern)
    rhythms = ["".join("-" if c == "-" else "*" for c in s) for s in played_strs]
    common, n_common = Counter(rhythms).most_common(1)[0] if rhythms else ("", 0)
    bars_match = sum(b["matches"] for b in bars)
    return {
        "resolution": res,
        "bars": bars,
        "bars_matching": bars_match,
        "match_rate": round(bars_match / n_bars, 3) if n_bars else None,
        "consistency": round(n_common / n_bars, 3) if n_bars else None,
        "most_common_rhythm": common,
        "diff_counts": dict(Counter(d["kind"] for b in bars for d in b["diffs"])),
    }
