"""Every tunable detection threshold in one place.

Defaults are the hand-set v1 values. `scripts/tune.py` searches them against a labelled eval set
and writes the winners to a JSON file; set PARAMS_PATH to load it (default: app/analysis/params.json
if present). Modules read `P.<name>` at call time, so `override()` takes effect immediately.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

log = logging.getLogger(__name__)


@dataclass
class Params:
    # --- audio: strum onsets -------------------------------------------------------------
    global_delta: float = 0.30          # onset threshold (env normalized to the 95th pct of its peaks)
    recover_min: float = 0.12           # minimum peak for a score-informed (recovered) strum
    merge_window_s: float = 0.050       # onsets this close belong to one strum
    strum_time_bias_ms: float = float(os.environ.get("STRUM_TIME_BIAS_MS", "13"))
    ghost_energy: float = 0.12          # strums quieter than this (rel. to typical) are ghosts / mutes
    min_rise: float = 1.0               # RMS after/before the onset; below = decay flux, not an attack
    # --- audio: per-string verdicts ------------------------------------------------------
    present_thr: float = 0.30           # basic-pitch activation => "sounded"
    absent_thr: float = 0.15            # below this => clearly absent
    missing_cqt_db: float = -18.0       # CQT energy below this (rel. frame max) supports "missing"
    wrong_fret_margin: float = 0.15     # neighbour semitone must beat the target by this much
    muted_sounding_presence: float = 0.40
    muted_sounding_cqt_db: float = -20.0
    unexpected_presence: float = 0.50
    # --- audio: chord sustain ------------------------------------------------------------
    choke_ratio: float = 0.25           # energy before the next strum / energy after attack
    choke_min_gap_s: float = 0.45       # only judge sustain when the next strum is this far away
    # --- direction -----------------------------------------------------------------------
    video_min_speed: float = 0.6        # neck-widths/s; slower = ghost / tracking loss
    direction_min_conf: float = 0.45    # report wrong_direction above this confidence
    # --- rules ---------------------------------------------------------------------------
    timing_err_ms: float = 40.0         # report early/late beyond this
    wrong_chord_margin: float = 0.12
    wrong_chord_max_pitch: float = 0.5

    @property
    def strum_time_bias_s(self) -> float:
        return self.strum_time_bias_ms / 1000


P = Params()
_DEFAULTS = asdict(P)


def names() -> list[str]:
    return [f.name for f in fields(Params)]


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data = data.get("params", data)
    unknown = set(data) - set(names())
    if unknown:
        log.warning("ignoring unknown params in %s: %s", path, sorted(unknown))
    for k, v in data.items():
        if k in _DEFAULTS:
            setattr(P, k, type(_DEFAULTS[k])(v))
    return data


def reset() -> None:
    for k, v in _DEFAULTS.items():
        setattr(P, k, v)


@contextmanager
def override(**kw):
    old = {k: getattr(P, k) for k in kw}
    for k, v in kw.items():
        if k not in _DEFAULTS:
            raise KeyError(k)
        setattr(P, k, v)
    try:
        yield P
    finally:
        for k, v in old.items():
            setattr(P, k, v)


_path = os.environ.get("PARAMS_PATH") or os.path.join(os.path.dirname(__file__), "params.json")
if os.path.exists(_path):
    load(_path)
    log.info("loaded tuned params from %s", _path)
