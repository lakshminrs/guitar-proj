"""Strum-pattern inference: the played rhythm per bar, compared with the chart."""
from app.analysis.audio import analyze_audio
from app.analysis.fusion import BeatClock, grade
from app.analysis.patterns import PlayedStrum, infer_patterns
from app.synth import write_take
from app.theory import expected_events, get_exercise

EX = {"time_sig": "4/4", "bars": [{"chord": "G", "strum_pattern": "D-DU-UDU"},
                                  {"chord": "C", "strum_pattern": "D-DU-UDU"}]}


def _played(pattern: str, bar: int = 0, jitter: float = 0.0, unknown: bool = False):
    return [PlayedStrum(bar * 4 + k / 2 + jitter, None if unknown else ch)
            for k, ch in enumerate(pattern) if ch in "DU"]


def test_clean_playing_matches_every_bar():
    out = infer_patterns(EX, _played("D-DU-UDU", 0, 0.08) + _played("D-DU-UDU", 1, -0.08))
    assert [b["played"] for b in out["bars"]] == ["D-DU-UDU", "D-DU-UDU"]
    assert out["match_rate"] == 1.0 and out["consistency"] == 1.0


def test_differences_are_named_by_beat():
    # bar 1: up-strum on 2& missing; bar 2: extra down on 1& and a down where 4& should be up
    out = infer_patterns(EX, _played("D-D--UDU", 0) + _played("DDDU-UDD", 1))
    b1, b2 = out["bars"]
    assert b1["played"] == "D-D--UDU"
    assert b1["diffs"] == [{"slot": 3, "beat_label": "2&", "expected": "U", "played": "-", "kind": "missing"}]
    assert {(d["beat_label"], d["kind"]) for d in b2["diffs"]} == {("1&", "extra"), ("4&", "direction")}
    assert out["diff_counts"] == {"missing": 1, "extra": 1, "direction": 1}
    assert out["match_rate"] == 0.0


def test_unknown_direction_is_not_a_mistake():
    out = infer_patterns(EX, _played("D-DU-UDU", 0, unknown=True))
    assert out["bars"][0]["played"] == "x-xx-xxx" and out["bars"][0]["matches"]


def test_quarter_note_chart_still_shows_offbeat_extras():
    ex = {"time_sig": "4/4", "bars": [{"chord": "G", "strum_pattern": "DDDD"}]}
    out = infer_patterns(ex, [PlayedStrum(p, "D") for p in (0, 1, 1.5, 2, 3)])
    assert out["resolution"] == 2
    assert out["bars"][0]["expected"] == "D-D-D-D-" and out["bars"][0]["played"] == "D-DDD-D-"
    assert out["bars"][0]["diffs"][0]["beat_label"] == "2&"


def test_beat_clock_inverse_roundtrips():
    clk = BeatClock([0, 750, 1500, 2250, 3010, 3760], count_in=4, bpm=80)
    for pos in (-4, -1.5, 0, 0.5, 1.25, 3.0):
        assert abs(clk.beat_pos(clk(pos)) - pos) < 1e-6


def test_skipped_strum_shows_in_played_pattern(tmp_path):
    ex = get_exercise(1)
    evs = expected_events(ex)
    skip = next(e for e in evs if e.bar == 3 and e.slot == 3)   # bar 3, 2& up-strum
    p = str(tmp_path / "take.wav")
    clicks = write_take(p, ex, seed=5, skip={skip.idx})
    res = grade(ex, analyze_audio(p, [c / 1000 for c in clicks]), None, clicks, 4)
    bar3 = res.patterns["bars"][2]
    assert bar3["played"][3] == "-"
    assert [d["kind"] for d in bar3["diffs"]] == ["missing"]
    others = [b for b in res.patterns["bars"] if b["bar"] != 3]
    assert all(b["matches"] for b in others)
