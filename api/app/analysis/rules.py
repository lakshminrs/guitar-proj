"""Error -> explanation rules. Rules decide errors; an LLM may only rephrase them."""
from __future__ import annotations

from collections import defaultdict

from ..theory import CHORDS, FINGER_NAMES, STRING_NAMES, string_to_idx
from . import geometry as G

TIMING_ERR_MS = 40.0   # report early/late beyond this


def _err(ev: dict, type_: str, severity: float, confidence: float, message: str, draw: list,
         **extra) -> dict:
    e = {"t_ms": round(ev["t_ms"]), "bar": ev["bar"], "beat": ev["beat"], "chord": ev["expected_chord"],
         "type": type_, "string": None, "finger": None, "expected_fret": None, "observed_fret": None,
         "severity": round(float(severity), 2), "confidence": round(float(confidence), 2),
         "message": message, "overlay": {"frame": ev.get("frame"), "draw": draw}}
    e.update(extra)
    return e


def _sname(s: int) -> str:
    return STRING_NAMES[s]


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _art(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _fname(f: int) -> str:
    return FINGER_NAMES.get(f, "finger")


def event_errors(ev: dict, verdict: dict | None, fingers_canon: dict | None) -> list[dict]:
    """Per-strum errors from the audio verdict (the judge) + vision (the explanation)."""
    errs: list[dict] = []
    chord = ev["expected_chord"]
    shape = CHORDS[chord]

    if ev.get("missed"):
        errs.append(_err(ev, "missed_strum", 0.9, 0.8,
                         f"Bar {ev['bar']}, beat {ev['beat_label']}: expected {_art('down' if ev['expected_direction'] == 'D' else 'up')}-strum on {chord} but nothing sounded.",
                         [{"kind": "beat", "style": "error"}]))
        return errs

    # --- pitch / string errors -------------------------------------------------
    if verdict:
        observed_cells = {}
        if fingers_canon:
            observed_cells = {fg: G.canon_to_cell(*xy) for fg, xy in fingers_canon.items()}

        too_wide: list[tuple[int, float]] = []
        for idx, s in verdict["strings"].items():
            string = 6 - idx
            exp_fret = shape.frets[idx]
            finger = shape.fingers[idx] if shape.fingers[idx] > 0 else None
            status = s["status"]
            if status == "wrong_fret":
                delta = s["delta"]
                obs_fret = exp_fret + delta
                flat = delta < 0
                vis_confirms = finger is not None and observed_cells.get(finger) == (string, obs_fret)
                conf = min(0.95, s["confidence"] + (0.15 if vis_confirms else 0))
                if exp_fret == 0:
                    msg = (f"The open {_sname(string)} string should ring open on {chord} but came out "
                           f"{'sharp' if not flat else 'flat'} — a finger is pressing it at fret {max(obs_fret, 1)}.")
                elif finger and (vis_confirms or not fingers_canon):
                    msg = (f"{_cap(_fname(finger))} finger is on fret {obs_fret} of {_sname(string)}, "
                           f"move it {'up' if flat else 'back'} to fret {exp_fret} → note came out {'flat' if flat else 'sharp'}.")
                else:
                    msg = (f"{_cap(_sname(string))} string came out {'flat' if flat else 'sharp'} on {chord} — "
                           f"it should be fretted at {exp_fret}.")
                errs.append(_err(ev, "wrong_fret", 0.8, conf, msg,
                                 [{"kind": "cell", "string": string, "fret": exp_fret, "style": "target"},
                                  *([{"kind": "fingertip", "finger": finger, "style": "error"}] if finger else [])],
                                 string=string, finger=finger, expected_fret=exp_fret, observed_fret=obs_fret))
            elif status == "missing":
                cause, cause_finger, conf_boost = None, finger, 0.0
                if fingers_canon:
                    sy = G.STRING_Y[idx]
                    # someone else's finger lying on this string's line => leaning / muting it
                    for fg, (x, y) in fingers_canon.items():
                        tgt_idx = [k for k, ff in enumerate(shape.fingers) if ff == fg]
                        if idx in tgt_idx or not tgt_idx or x <= 0:
                            continue
                        if abs(y - sy) < 0.6 * G.STRING_SPACING:
                            cause, cause_finger, conf_boost = "leaning", fg, 0.1
                            break
                    if cause is None and finger and observed_cells.get(finger) == (string, exp_fret):
                        cause, conf_boost = "pressure", 0.05
                    elif cause is None and finger and finger in observed_cells:
                        cause = "misplaced"
                if cause == "leaning":
                    msg = (f"Your {_fname(cause_finger)} finger is leaning on the {_sname(string)} string "
                           f"and muting it — arch it more.")
                elif cause == "pressure":
                    msg = (f"{_cap(_sname(string))} string on {chord}: right spot, not pressed hard enough "
                           f"or too far from the fret wire.")
                elif cause == "misplaced":
                    o = observed_cells[finger]
                    msg = (f"{_cap(_sname(string))} string didn't sound — your {_fname(finger)} finger is at "
                           f"string {o[0]}, fret {o[1]}; it belongs on fret {exp_fret} of the {_sname(string)} string.")
                else:
                    msg = (f"{_cap(_sname(string))} string didn't ring on {chord}"
                           + (" — check nothing is touching it." if exp_fret == 0 else
                              f" — press firmly at fret {exp_fret}, just behind the wire."))
                draw = [{"kind": "string", "string": string, "style": "error"}]
                if exp_fret > 0:
                    draw.append({"kind": "cell", "string": string, "fret": exp_fret, "style": "target"})
                if cause_finger:
                    draw.append({"kind": "fingertip", "finger": cause_finger, "style": "error"})
                errs.append(_err(ev, "muted_string", 0.6, min(0.95, s["confidence"] + conf_boost), msg, draw,
                                 string=string, finger=cause_finger, expected_fret=exp_fret, cause=cause))
            elif status == "muted_sounding":
                too_wide.append((string, s["confidence"]))
        if too_wide:
            ss = sorted(x[0] for x in too_wide)
            conf = max(x[1] for x in too_wide)
            if min(ss) >= 5:
                skip = "low E" + (" and A" if shape.frets[string_to_idx(5)] < 0 else "")
                msg = f"Strum is starting too high on {chord} — skip the {skip}."
            else:
                msg = f"The {', '.join(_sname(x) for x in ss)} string should be muted on {chord}, but it rang."
            errs.append(_err(ev, "strum_too_wide", 0.5, conf, msg,
                             [{"kind": "string", "string": x, "style": "error"} for x in ss], string=max(ss)))

        # wrong chord altogether (chroma backup check)
        best = max(verdict["chroma_all"], key=verdict["chroma_all"].get)
        if best != chord and verdict["chroma_all"][best] - verdict["chroma_sim"] > 0.12 and verdict["pitch_score"] < 0.5:
            errs.append(_err(ev, "wrong_chord", 0.9, 0.6,
                             f"Bar {ev['bar']}: that sounded like {best}, not {chord}.",
                             [{"kind": "chord", "expected": chord, "observed": best, "style": "error"}],
                             observed_chord=best))

    # --- vision-only fingering mismatch (audio OK): low severity -----------------
    if fingers_canon and verdict:
        bad_strings = {e["string"] for e in errs if e.get("string")}
        for tgt in shape.fingering_targets():
            fg = tgt["finger"]
            if fg not in fingers_canon:
                continue
            obs_s, obs_f = G.canon_to_cell(*fingers_canon[fg])
            ok = obs_f == tgt["fret"] and (tgt["barre"] or obs_s == tgt["string"])
            if not ok and tgt["string"] not in bad_strings:
                errs.append(_err(ev, "finger_position", 0.25, 0.4,
                                 f"{_cap(_fname(fg))} finger looks like it's at string {obs_s}, fret {obs_f}"
                                 f" — target is {'a barre at fret ' + str(tgt['fret']) if tgt['barre'] else 'fret ' + str(tgt['fret']) + ' of the ' + _sname(tgt['string']) + ' string'}.",
                                 [{"kind": "cell", "string": tgt["string"] or 1, "fret": tgt["fret"], "style": "target"},
                                  {"kind": "fingertip", "finger": fg, "style": "error"}],
                                 string=tgt["string"], finger=fg, expected_fret=tgt["fret"], observed_fret=obs_f))

    # --- timing ------------------------------------------------------------------
    dt = ev.get("timing_ms")
    if dt is not None and abs(dt) > TIMING_ERR_MS:
        sev = min(1.0, 0.3 + (abs(dt) - TIMING_ERR_MS) / 120)
        errs.append(_err(ev, "late" if dt > 0 else "early", sev, 0.85,
                         f"Bar {ev['bar']}, beat {ev['beat_label']}: {abs(dt):.0f} ms {'late' if dt > 0 else 'early'}.",
                         [{"kind": "timing", "offset_ms": round(dt)}], offset_ms=round(dt)))

    # --- strum direction ---------------------------------------------------------
    obs = ev.get("observed_direction")
    if obs and obs != ev["expected_direction"] and ev.get("direction_confidence", 0) >= 0.45:
        word = {"D": "down", "U": "up"}
        errs.append(_err(ev, "wrong_direction", 0.4, ev["direction_confidence"],
                         f"Bar {ev['bar']}, beat {ev['beat_label']}: expected {_art(word[ev['expected_direction']])}-strum, "
                         f"played {_art(word[obs])}-strum.",
                         [{"kind": "strum", "expected": ev["expected_direction"], "observed": obs, "style": "error"}]))
    return errs


def aggregate_errors(events: list[dict]) -> list[dict]:
    """Take-level patterns that single events can't show."""
    out = []
    played = [e for e in events if not e.get("missed") and e.get("timing_ms") is not None]
    # late on changes only
    changes = [e for e in played if e["is_change"]]
    others = [e for e in played if not e["is_change"]]
    if len(changes) >= 2 and others:
        mc = sum(e["timing_ms"] for e in changes) / len(changes)
        mo = sum(e["timing_ms"] for e in others) / len(others)
        if mc > 45 and mc - mo > 30:
            pairs = defaultdict(int)
            for e in changes:
                pairs[(e["prev_chord"], e["expected_chord"])] += 1
            (a, b), _ = max(pairs.items(), key=lambda kv: kv[1])
            out.append({"type": "late_on_changes", "severity": min(1.0, 0.5 + mc / 200), "confidence": 0.8,
                        "count": len(changes), "chord": b,
                        "message": f"You're late on changes (avg {mc:.0f} ms) — practice the switch from {a} to {b} slowly.",
                        "t_ms": changes[0]["t_ms"], "bar": changes[0]["bar"], "beat": 1})
    # up-strums turning into down-strums on the offbeats
    off = [e for e in played if e["is_offbeat"] and e["expected_direction"] == "U" and e.get("observed_direction")
           and e.get("direction_confidence", 0) >= 0.45]
    if len(off) >= 3:
        bad = [e for e in off if e["observed_direction"] == "D"]
        if len(bad) / len(off) >= 0.4:
            out.append({"type": "offbeat_direction", "severity": 0.6, "confidence": 0.7, "count": len(bad),
                        "message": "Up-strums are turning into down-strums on the offbeats.",
                        "t_ms": bad[0]["t_ms"], "bar": bad[0]["bar"], "beat": bad[0]["beat"], "chord": None})
    # overall rushing / dragging
    if len(played) >= 6:
        mean = sum(e["timing_ms"] for e in played) / len(played)
        if abs(mean) > 35:
            out.append({"type": "dragging" if mean > 0 else "rushing", "severity": min(1.0, abs(mean) / 100),
                        "confidence": 0.8, "count": len(played), "chord": None,
                        "message": f"You're {'behind' if mean > 0 else 'ahead of'} the click overall by {abs(mean):.0f} ms — "
                                   f"{'relax and lock into the click' if mean > 0 else 'wait for the click, don’t rush'}.",
                        "t_ms": played[0]["t_ms"], "bar": played[0]["bar"], "beat": played[0]["beat"]})
    return out


def rank_top(errors: list[dict], aggregates: list[dict], k: int = 3) -> list[dict]:
    """Rank by frequency x severity; return the top k as 'fix these first'."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in errors:
        if e["type"] == "finger_position":
            key = (e["type"], e["chord"], e["finger"])
        elif e["type"] in ("early", "late", "missed_strum", "wrong_direction"):
            key = (e["type"],)
        else:
            key = (e["type"], e["chord"], e.get("string"))
        groups[key].append(e)
    ranked = []
    for key, es in groups.items():
        sev = sum(x["severity"] for x in es) / len(es)
        conf = sum(x["confidence"] for x in es) / len(es)
        rep = max(es, key=lambda x: x["severity"] * x["confidence"])
        msg = rep["message"]
        if key[0] in ("early", "late"):
            avg = sum(abs(x.get("offset_ms", 0)) for x in es) / len(es)
            n = len(es)
            msg = f"{n} strum{'s' if n > 1 else ''} {key[0]} by more than {TIMING_ERR_MS:.0f} ms (avg {avg:.0f} ms)."
        elif key[0] == "wrong_direction":
            msg = (f"{len(es)} strums went the wrong direction — e.g. {rep['message']}" if len(es) > 1 else rep["message"])
        elif key[0] == "missed_strum":
            msg = (f"{len(es)} expected strums were missing — e.g. {rep['message']}" if len(es) > 1 else rep["message"])
        ranked.append({"type": key[0], "chord": rep.get("chord"), "string": rep.get("string"),
                       "finger": rep.get("finger"), "count": len(es), "severity": round(sev, 2),
                       "confidence": round(conf, 2), "priority": round(len(es) * sev * (0.5 + conf / 2), 3),
                       "message": msg + (f" ({len(es)}×)" if len(es) > 1 and key[0] not in ("early", "late", "wrong_direction", "missed_strum") else ""),
                       "example": {"t_ms": rep["t_ms"], "bar": rep["bar"], "beat": rep["beat"]}})
    for a in aggregates:
        # patterns get a boost: they explain many events at once
        ranked.append({**{k: a.get(k) for k in ("type", "chord", "count", "severity", "confidence", "message")},
                       "string": None, "finger": None,
                       "priority": round(1.5 * a["count"] * a["severity"], 3),
                       "example": {"t_ms": a["t_ms"], "bar": a["bar"], "beat": a["beat"]}})
    ranked.sort(key=lambda r: -r["priority"])
    return ranked[:k]
