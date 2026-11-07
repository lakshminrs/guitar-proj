import subprocess

import numpy as np
import pytest

from app import jobs
from app.analysis.vision import VisionTrack


def _track(ys: np.ndarray, fps: float) -> VisionTrack:
    vt = VisionTrack(fps=fps, width=640, height=480, H=[None] * len(ys), fret_lm=[None] * len(ys),
                     strum_lm=[None] * len(ys))
    vt._strum_cache = ys
    return vt


@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize("sign,expect", [(1, "D"), (-1, "U")])
def test_strum_direction_from_velocity(fps, sign, expect):
    t = np.arange(int(2 * fps)) / fps
    ys = 0.5 + sign * 2.0 * (t - 1.0)            # 2 neck-widths/s through t = 1 s
    d, conf, speed = _track(ys, fps).strum_direction(1.0)
    assert d == expect and speed == pytest.approx(2.0, rel=0.05) and conf > 0.5


def test_strum_direction_tolerates_one_dropped_frame():
    fps = 60.0
    t = np.arange(120) / fps
    ys = 0.5 + 2.0 * (t - 1.0)
    ys[59] = np.nan
    assert _track(ys, fps).strum_direction(1.0)[0] == "D"


def test_strum_direction_gives_up_on_mostly_missing_window():
    fps = 60.0
    ys = np.full(120, np.nan)
    ys[60] = 0.5
    ys[61] = 0.6
    assert _track(ys, fps).strum_direction(1.0)[0] is None


def test_fps_choice():
    assert jobs.target_fps(59.94) == 60
    assert jobs.target_fps(30.0) == 30
    assert jobs.target_fps(0.0) == 30            # unknown -> safe default
    assert jobs._rate("1000/1") == 1000.0 and jobs._rate("0/0") == 0.0 and jobs._rate(None) == 0.0


def test_source_fps_falls_back_to_counting_frames(monkeypatch):
    """Browser WebM reports a 1000/1 timebase as its rate; count frames over the duration instead."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = ('{"streams":[{"avg_frame_rate":"1000/1"}],"format":{"duration":"4.0"}}'
               if "-count_packets" not in cmd else "240\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    assert jobs.source_fps("x.webm") == pytest.approx(60.0)
    assert len(calls) == 2
