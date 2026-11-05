"""Synthetic strummed-guitar takes with deliberate, known mistakes.

Used by the tests and the eval script: lets us check the audio grader end-to-end
("a deliberately muted string and a late strum both get flagged") without a guitar.
Additive synthesis of plucked strings (inharmonic partials, exponential decay, pick noise).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from .theory import CHORDS, OPEN_MIDI, beats_per_bar, expected_events

SR = 44100


def _pluck(midi: float, dur: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    out = np.zeros(n)
    B = 1e-4  # string inharmonicity
    for k in range(1, 16):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > SR / 2.2:
            break
        a = (1.0 / k ** 1.1) * (1.0 + 0.3 * rng.standard_normal() * 0.2)
        decay = 1.4 + 0.5 * k + f0 / 300.0
        out += a * np.exp(-decay * t) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
    att = np.minimum(1.0, t / 0.002)
    noise = rng.standard_normal(n) * np.exp(-t / 0.006) * 0.15
    return amp * (out * att + noise)


def render_take(
    exercise: dict,
    *,
    count_in_beats: int = 4,
    lead_in_s: float = 0.5,
    mute_string: dict | None = None,       # {"event": i, "string": 1..6}
    wrong_fret: dict | None = None,        # {"event": i, "string": 1..6, "delta": -1}
    extra_low: dict | None = None,         # {"event": i}  -> also strike muted low strings
    late_ms: dict | None = None,           # {event_idx: ms}
    flip_direction: set[int] | None = None,
    jitter_ms: float = 4.0,
    click_hz: float | None = 3000.0,       # audible click bleed; None = headphones
    click_latency_ms: float = 0.0,         # click arrives this much after logged time
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Returns (audio, clicks_ms). clicks_ms are 'logged' click times (every beat,
    including the count-in) relative to recording start."""
    rng = np.random.default_rng(seed)
    beat_s = 60.0 / exercise["bpm"]
    bpb = beats_per_bar(exercise["time_sig"])
    n_beats = count_in_beats + len(exercise["bars"]) * bpb
    start = lead_in_s + count_in_beats * beat_s
    total = lead_in_s + (n_beats + 2) * beat_s
    y = np.zeros(int(total * SR))
    clicks_ms = [(lead_in_s + k * beat_s) * 1000.0 for k in range(n_beats)]

    if click_hz:
        n = int(0.03 * SR)
        tt = np.arange(n) / SR
        blip = 0.25 * np.sin(2 * np.pi * click_hz * tt) * np.exp(-tt / 0.008) * np.minimum(1, tt / 0.002)
        for c in clicks_ms:
            a = int((c + click_latency_ms) / 1000 * SR)
            y[a:a + n] += blip[: len(y) - a]

    events = expected_events(exercise)
    late_ms = late_ms or {}
    flip_direction = flip_direction or set()
    # Track when each string was last struck to damp it on re-strike.
    for i, ev in enumerate(events):
        # the player locks to the click they *hear*, i.e. logged time + output latency
        t0 = (start + ev.beat_pos * beat_s + (late_ms.get(i, 0.0) + click_latency_ms) / 1000
              + rng.normal(0, jitter_ms / 1000))
        nxt = events[i + 1] if i + 1 < len(events) else None
        dur = ((start + nxt.beat_pos * beat_s) - t0 + 0.05) if nxt else 1.5
        shape = CHORDS[ev.chord]
        midis = list(shape.midi)
        if extra_low and extra_low.get("event") == i:
            for k in shape.muted_idx:
                midis[k] = OPEN_MIDI[k]
        if mute_string and mute_string.get("event") == i:
            midis[6 - mute_string["string"]] = -1
        if wrong_fret and wrong_fret.get("event") == i:
            k = 6 - wrong_fret["string"]
            midis[k] = midis[k] + wrong_fret.get("delta", -1)
        direction = ev.direction
        if i in flip_direction:
            direction = "U" if direction == "D" else "D"
        order = list(range(6)) if direction == "D" else list(range(5, -1, -1))
        if direction == "U":  # up-strums usually catch only the top 3-4 strings
            order = order[:4]
        spread = 0.008
        amp = 0.18 if direction == "D" else 0.13
        for j, k in enumerate(order):
            m = midis[k]
            if m < 0:
                continue
            ts = t0 + j * spread
            seg = _pluck(m, max(dur - j * spread, 0.05), amp, rng)
            a = int(ts * SR)
            seg = seg[: max(0, len(y) - a)]
            # short release so the previous ring doesn't smear into the next strum
            r = min(len(seg), int(0.02 * SR))
            if r:
                seg[-r:] *= np.linspace(1, 0, r)
            y[a:a + len(seg)] += seg
    y += rng.standard_normal(len(y)) * 0.0015
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
    return y.astype(np.float32), clicks_ms


def write_take(path: str, exercise: dict, **kw) -> list[float]:
    y, clicks = render_take(exercise, **kw)
    sf.write(path, y, SR)
    return clicks
