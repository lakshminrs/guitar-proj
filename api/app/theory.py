"""Music theory: tuning, chord shapes, exercises, and the expected-event grid.

Conventions used everywhere in the backend:
  * Arrays of 6 per-string values are ordered LOW E -> HIGH E (index 0 = string 6).
  * "string" numbers in errors / UI follow guitar convention: 1 = high E ... 6 = low E.
  * frets: -1 = muted, 0 = open.   fingers: 0 = open/none, 1-4 = index..pinky, -1 = muted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
STRING_NAMES = {6: "low E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "high E"}
FINGER_NAMES = {1: "index", 2: "middle", 3: "ring", 4: "pinky", 0: "open"}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def idx_to_string(i: int) -> int:
    """array index (0 = low E) -> guitar string number (6 = low E)."""
    return 6 - i


def string_to_idx(s: int) -> int:
    return 6 - s


def midi_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


@dataclass(frozen=True)
class ChordShape:
    name: str
    frets: tuple[int, ...]    # low E -> high E
    fingers: tuple[int, ...]  # low E -> high E
    barre: int = 0            # fret of index-finger barre, 0 = none

    @property
    def midi(self) -> list[int]:
        return [OPEN_MIDI[i] + f if f >= 0 else -1 for i, f in enumerate(self.frets)]

    @property
    def sounding_idx(self) -> list[int]:
        return [i for i, f in enumerate(self.frets) if f >= 0]

    @property
    def muted_idx(self) -> list[int]:
        return [i for i, f in enumerate(self.frets) if f < 0]

    def fingering_targets(self) -> list[dict]:
        """(finger, string, fret) targets. A barre contributes one target for the index
        finger that only checks the fret (it covers several strings)."""
        out = []
        barre_done = False
        for i, (f, fg) in enumerate(zip(self.frets, self.fingers)):
            if f <= 0 or fg <= 0:
                continue
            if self.barre and fg == 1 and f == self.barre:
                if barre_done:
                    continue
                barre_done = True
                out.append({"finger": 1, "string": None, "fret": f, "barre": True})
                continue
            out.append({"finger": fg, "string": idx_to_string(i), "fret": f, "barre": False})
        return out

    def to_dict(self) -> dict:
        return {"name": self.name, "frets": list(self.frets), "fingers": list(self.fingers),
                "midi": self.midi, "barre": self.barre}


# Open chords + simple barre chords (v1 scope).
CHORDS: dict[str, ChordShape] = {c.name: c for c in [
    ChordShape("G",  (3, 2, 0, 0, 0, 3),    (2, 1, 0, 0, 0, 3)),
    ChordShape("C",  (-1, 3, 2, 0, 1, 0),   (-1, 3, 2, 0, 1, 0)),
    ChordShape("D",  (-1, -1, 0, 2, 3, 2),  (-1, -1, 0, 1, 3, 2)),
    ChordShape("Em", (0, 2, 2, 0, 0, 0),    (0, 2, 3, 0, 0, 0)),
    ChordShape("Am", (-1, 0, 2, 2, 1, 0),   (-1, 0, 2, 3, 1, 0)),
    ChordShape("E",  (0, 2, 2, 1, 0, 0),    (0, 2, 3, 1, 0, 0)),
    ChordShape("A",  (-1, 0, 2, 2, 2, 0),   (-1, 0, 1, 2, 3, 0)),
    ChordShape("Dm", (-1, -1, 0, 2, 3, 1),  (-1, -1, 0, 2, 3, 1)),
    ChordShape("F",  (1, 3, 3, 2, 1, 1),    (1, 3, 4, 2, 1, 1), barre=1),
    ChordShape("Bm", (-1, 2, 4, 4, 3, 2),   (-1, 1, 3, 4, 2, 1), barre=2),
]}


def chord_chroma(name: str) -> list[float]:
    v = [0.0] * 12
    for m in CHORDS[name].midi:
        if m >= 0:
            v[m % 12] += 1.0
    return v


# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------

def _bars(chords: list[str], pattern: str, repeat: int = 1) -> list[dict]:
    return [{"chord": c, "strum_pattern": pattern} for c in chords] * repeat


EXERCISES: list[dict] = [
    {"id": 1, "title": "G–C–D–Em · D-DU-UDU", "bpm": 80, "time_sig": "4/4",
     "bars": _bars(["G", "C", "D", "Em"], "D-DU-UDU", 2),
     "description": "The classic campfire progression with the 'old faithful' strum."},
    {"id": 2, "title": "Em–C–G–D · quarter downs", "bpm": 70, "time_sig": "4/4",
     "bars": _bars(["Em", "C", "G", "D"], "DDDD", 2),
     "description": "Four down-strums per bar. Focus on clean changes."},
    {"id": 3, "title": "G ↔ C change drill", "bpm": 60, "time_sig": "4/4",
     "bars": _bars(["G", "C"], "DDDD", 4),
     "description": "Slow G-to-C switches until they are automatic."},
    {"id": 4, "title": "A–D–E · D-DU", "bpm": 90, "time_sig": "4/4",
     "bars": _bars(["A", "D", "E", "A"], "D-D-D-DU", 2),
     "description": "Three-chord rock with an up-strum pickup at the end of each bar."},
    {"id": 5, "title": "Am–Dm–E–Am · D-DU-UDU", "bpm": 72, "time_sig": "4/4",
     "bars": _bars(["Am", "Dm", "E", "Am"], "D-DU-UDU", 2),
     "description": "Minor-key progression; watch the Dm voicing (don't hit the low strings)."},
    {"id": 6, "title": "F barre intro · quarters", "bpm": 60, "time_sig": "4/4",
     "bars": _bars(["C", "F", "Am", "G"], "DDDD", 2),
     "description": "First steps into the F barre chord."},
    {"id": 7, "title": "Bm barre · D-DU-UDU", "bpm": 70, "time_sig": "4/4",
     "bars": _bars(["G", "D", "Bm", "A"], "D-DU-UDU", 2),
     "description": "Second-fret barre in context."},
]


def get_exercise(exercise_id: int) -> dict:
    for e in EXERCISES:
        if e["id"] == exercise_id:
            return e
    raise KeyError(exercise_id)


def beats_per_bar(time_sig: str) -> int:
    return int(time_sig.split("/")[0])


@dataclass
class ExpectedEvent:
    idx: int
    bar: int          # 1-based
    beat: int         # 1-based beat in bar
    slot: int         # 0-based subdivision slot in bar
    subdiv: int       # slots per beat
    beat_pos: float   # absolute position in beats from exercise start (0 = bar 1 beat 1)
    chord: str
    direction: str    # "D" | "U"
    is_offbeat: bool
    is_change: bool   # first strum of a bar whose chord differs from previous bar
    t_ms: float = 0.0
    meta: dict = field(default_factory=dict)


def expected_events(ex: dict) -> list[ExpectedEvent]:
    bpb = beats_per_bar(ex["time_sig"])
    out: list[ExpectedEvent] = []
    prev_chord = None
    for b, bar in enumerate(ex["bars"]):
        pat = bar["strum_pattern"].replace(" ", "")
        if len(pat) % bpb:
            raise ValueError(f"strum pattern {pat!r} doesn't divide into {bpb} beats")
        sub = len(pat) // bpb
        first = True
        for s, ch in enumerate(pat):
            if ch not in "DU":
                continue
            out.append(ExpectedEvent(
                idx=len(out), bar=b + 1, beat=s // sub + 1, slot=s, subdiv=sub,
                beat_pos=b * bpb + s / sub, chord=bar["chord"], direction=ch,
                is_offbeat=(s % sub) != 0,
                is_change=first and prev_chord is not None and prev_chord != bar["chord"],
            ))
            first = False
        prev_chord = bar["chord"]
    return out


def exercise_duration_beats(ex: dict) -> int:
    return len(ex["bars"]) * beats_per_bar(ex["time_sig"])
