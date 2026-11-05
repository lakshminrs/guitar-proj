"""Optical-flow tracking of the calibrated neck on a synthetic video: a textured fretboard
drifts across the frame; the grid must stay locked (< 1 fret of drift)."""
import cv2
import numpy as np

from app.analysis import geometry as G
from app.analysis.vision import analyze_video

W, H, FPS, N = 640, 360, 30, 90


def _neck_texture(rng):
    tex = np.full((120, 900, 3), (40, 70, 110), np.uint8)
    tex = cv2.add(tex, rng.integers(0, 40, tex.shape, dtype=np.uint8))
    for n in range(0, 16):  # fret wires at true positions
        x = int(G.fret_x(n) * 1400)
        if x < 900:
            cv2.line(tex, (x, 0), (x, 119), (200, 200, 200), 3)
    for x in (int(G.fret_x(n + 0.5) * 1400) for n in (3, 5, 7, 9)):
        cv2.circle(tex, (x, 60), 7, (230, 230, 230), -1)
    for s in range(6):
        y = int(G.string_y(s) * 119)
        cv2.line(tex, (0, y), (899, y), (180, 180, 160), 1)
    return tex


def _write_video(path):
    rng = np.random.default_rng(0)
    tex = _neck_texture(rng)
    bg = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    corners_by_frame = []
    for f in range(N):
        dx, dy = 40 * np.sin(f / 25), 15 * np.sin(f / 17)
        # neck quad: nut at x=60, 12th fret (x_can=0.5 -> 700 px in texture) ...
        src = np.float32([[0, 0], [0, 119], [899, 0], [899, 119]])
        dst = np.float32([[60 + dx, 140 + dy], [60 + dx, 210 + dy], [560 + dx, 130 + dy], [560 + dx, 220 + dy]])
        M = cv2.getPerspectiveTransform(src, dst)
        frame = bg.copy()
        warped = cv2.warpPerspective(tex, M, (W, H))
        mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), M, (W, H))
        frame[mask > 0] = warped[mask > 0]
        out.write(frame)
        # true image positions of nut/12th-fret edges (canonical x = px / 1400)
        pts = np.float32([[0, 0], [0, 119], [700, 0], [700, 119]]).reshape(-1, 1, 2)
        corners_by_frame.append(cv2.perspectiveTransform(pts, M).reshape(-1, 2) / [W, H])
    out.release()
    return corners_by_frame


def test_grid_stays_locked(tmp_path):
    path = tmp_path / "neck.mp4"
    truth = _write_video(path)
    c0 = truth[0]
    cal = {"points": {"nut_low": c0[0].tolist(), "nut_high": c0[1].tolist(),
                      "f12_low": c0[2].tolist(), "f12_high": c0[3].tolist()}}
    track = analyze_video(str(path), cal)
    assert track is not None and track.n == N
    worst = 0.0
    for f in range(N):
        Ht = G.homography_from_points({"nut_low": truth[f][0], "nut_high": truth[f][1],
                                       "f12_low": truth[f][2], "f12_high": truth[f][3]})
        # where does the tracked grid put the 7th-fret cell centre, in the true fretboard?
        x, y = G.cell_center(3, 7)
        img = G.apply_h(track.H[f], [[x, y]])
        xc, yc = G.image_to_canon(Ht, img)[0]
        drift_frets = abs(xc - x) / (G.FRET_X[7] - G.FRET_X[6])
        worst = max(worst, drift_frets)
        assert G.canon_to_cell(xc, yc)[0] == 3
    print(f"worst drift: {worst:.3f} frets")
    assert worst < 1.0, f"grid drifted {worst:.2f} frets"
