"""Fretboard geometry.

Canonical fretboard space ("can"):
  x = distance from the nut in units of scale length L  (nut = 0, 12th fret = 0.5)
  y = position across the neck, 0 = low-E edge, 1 = high-E edge
Image space: normalized video coordinates (0..1, unmirrored / as recorded).

Fret n sits at d_n = L (1 - 2^(-n/12)); since x is in units of L, x_n = 1 - 2^(-n/12).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

N_FRETS = 15
STRING_MARGIN = 0.08  # outer strings sit slightly inside the neck edge


def fret_x(n: float) -> float:
    return 1.0 - 2.0 ** (-n / 12.0)


FRET_X = np.array([fret_x(n) for n in range(N_FRETS + 1)])


def string_y(idx: int) -> float:
    """idx 0 = low E ... 5 = high E."""
    return STRING_MARGIN + idx * (1 - 2 * STRING_MARGIN) / 5


STRING_Y = np.array([string_y(i) for i in range(6)])
STRING_SPACING = STRING_Y[1] - STRING_Y[0]

# Calibration keypoint names -> canonical coordinates. Used both for the manual 4-tap
# calibration and for YOLO keypoints (which add frets 3/5/7/9 ends).
KEYPOINT_CANON = {
    "nut_low": (0.0, 0.0), "nut_high": (0.0, 1.0),
    **{f"f{n}_low": (fret_x(n), 0.0) for n in (3, 5, 7, 9, 12)},
    **{f"f{n}_high": (fret_x(n), 1.0) for n in (3, 5, 7, 9, 12)},
}


def homography_from_points(points: dict[str, tuple[float, float]]) -> np.ndarray | None:
    """Fit canonical->image homography from named keypoints (image coords normalized)."""
    names = [n for n in points if n in KEYPOINT_CANON and points[n] is not None]
    if len(names) < 4:
        return None
    src = np.array([KEYPOINT_CANON[n] for n in names], dtype=np.float64)
    dst = np.array([points[n] for n in names], dtype=np.float64)
    if len(names) == 4:
        H, _ = cv2.findHomography(src, dst, 0)
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.01)
    if H is None or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def image_to_canon(H_c2i: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return apply_h(np.linalg.inv(H_c2i), pts)


def canon_to_cell(x: float, y: float) -> tuple[int, int]:
    """Canonical point -> (guitar string number 1..6, fret number).
    fret 0 = behind the nut / open; fret n = the cell between wire n-1 and n."""
    if x <= 0:
        fret = 0
    else:
        fret = int(np.searchsorted(FRET_X, x, side="left"))
        fret = min(max(fret, 1), N_FRETS + 1)
    idx = int(np.argmin(np.abs(STRING_Y - y)))
    return 6 - idx, fret


def cell_center(string: int, fret: int) -> tuple[float, float]:
    """Where a fingertip ideally sits: just behind the fret wire (75% across the cell)."""
    idx = 6 - string
    if fret <= 0:
        x = -0.02
    else:
        x = FRET_X[fret - 1] + 0.75 * (FRET_X[fret] - FRET_X[fret - 1])
    return float(x), float(STRING_Y[idx])


def quad_corners(H: np.ndarray, x_max: float | None = None) -> np.ndarray:
    x_max = FRET_X[12] if x_max is None else x_max
    return apply_h(H, np.array([[0, 0], [0, 1], [x_max, 0], [x_max, 1]]))


def homography_from_corners(corners: np.ndarray, x_max: float | None = None) -> np.ndarray:
    x_max = FRET_X[12] if x_max is None else x_max
    src = np.array([[0, 0], [0, 1], [x_max, 0], [x_max, 1]], dtype=np.float64)
    H, _ = cv2.findHomography(src, corners.astype(np.float64), 0)
    return H / H[2, 2]


def is_mirrored(H: np.ndarray) -> bool:
    """True if the nut appears to the RIGHT of the 12th fret in the image (e.g. a mirrored
    front-camera recording or a left-handed player)."""
    c = quad_corners(H)
    return bool(c[0:2, 0].mean() > c[2:4, 0].mean())


class OneEuro:
    """1-Euro filter (Casiez et al. 2012) for jittery keypoints; vectorized over arrays."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat
