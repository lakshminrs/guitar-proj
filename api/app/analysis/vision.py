"""Vision pipeline (offline, full quality on the recorded video).

For every frame: locate the fretboard (homography canonical -> image), run MediaPipe hands,
decide which hand frets and which strums (by position relative to the neck -- NOT MediaPipe's
handedness, since front cameras mirror), and keep fingertip positions in fretboard space.

Fretboard localization:
  * If a fine-tuned YOLO-pose ONNX model is configured (FRETBOARD_MODEL_PATH), detect neck
    keypoints every DETECT_EVERY frames and track in between with Lucas-Kanade optical flow.
  * Otherwise start from the session's manual calibration (4 taps: nut + 12th fret, both
    edges) and track with optical flow from there.
  * Quad corners are smoothed with a 1-Euro filter.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import savgol_filter

from . import geometry as G
from .fretboard_model import FretboardDetector
from .params import P

log = logging.getLogger(__name__)

HAND_MODEL = os.environ.get(
    "HAND_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "hand_landmarker.task"))
DETECT_EVERY = 5
TRACK_WIDTH = 640
TIP_IDX = {1: 8, 2: 12, 3: 16, 4: 20}           # index, middle, ring, pinky tips
DIP_IDX = {1: 7, 2: 11, 3: 15, 4: 19}
PIP_IDX = {1: 6, 2: 10, 3: 14, 4: 18}
TIP_EXTRAPOLATION = 0.3   # blend tip with DIP+bone-direction estimate (occlusion mitigation)


@dataclass
class VisionTrack:
    fps: float
    width: int
    height: int
    H: list[np.ndarray | None]                 # per frame canonical->image (normalized)
    fret_lm: list[np.ndarray | None]           # per frame 21x2 normalized image coords
    strum_lm: list[np.ndarray | None]
    source: str = "calibration"               # calibration | model
    mirrored: bool = False
    stats: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.H)

    # -- queries used by fusion ------------------------------------------------
    def fingertips_canon(self, t: float, window: float = 0.150) -> dict[int, tuple[float, float]]:
        """Median fingertip positions (canonical space) over frames in [t - window, t]."""
        f1 = min(int(t * self.fps), self.n - 1)
        f0 = max(0, int((t - window) * self.fps))
        acc: dict[int, list] = {k: [] for k in TIP_IDX}
        for f in range(f0, f1 + 1):
            H, lm = self.H[f], self.fret_lm[f]
            if H is None or lm is None:
                continue
            Hinv = np.linalg.inv(H)
            for finger, ti in TIP_IDX.items():
                tip = lm[ti]
                ext = lm[DIP_IDX[finger]] + 0.9 * (lm[DIP_IDX[finger]] - lm[PIP_IDX[finger]])
                p = (1 - TIP_EXTRAPOLATION) * tip + TIP_EXTRAPOLATION * ext
                acc[finger].append(G.apply_h(Hinv, p)[0])
        return {k: tuple(np.median(np.array(v), axis=0)) for k, v in acc.items() if v}

    def fingertips_image(self, t: float) -> dict[int, tuple[float, float]]:
        f = min(int(t * self.fps), self.n - 1)
        lm = self.fret_lm[f] if f >= 0 else None
        if lm is None:
            return {}
        return {k: (float(lm[i][0]), float(lm[i][1])) for k, i in TIP_IDX.items()}

    def strum_direction(self, t: float) -> tuple[str | None, float, float]:
        """Direction from the strumming hand's velocity across the strings at time t.
        Returns (D|U|None, confidence, speed in neck-widths/s)."""
        ys = self._strum_y()
        if ys is None:
            return None, 0.0, 0.0
        f = int(t * self.fps)
        a, b = max(0, f - int(0.06 * self.fps)), min(len(ys) - 1, f + int(0.02 * self.fps) + 1)
        seg = ys[a:b + 1]
        if len(seg) < 2 or np.isnan(seg).any():
            return None, 0.0, 0.0
        v = (seg[-1] - seg[0]) / ((len(seg) - 1) / self.fps)
        speed = abs(v)
        if speed < P.video_min_speed:  # too slow: likely a ghost/mute or tracking loss
            return None, 0.0, float(speed)
        # canonical y grows from low E to high E: a down-strum moves toward high E
        return ("D" if v > 0 else "U"), float(min(1.0, speed / 3.0)), float(speed)

    _strum_cache: np.ndarray | None = field(default=None, repr=False)

    def _strum_y(self) -> np.ndarray | None:
        if self._strum_cache is not None:
            return self._strum_cache
        ys = np.full(self.n, np.nan)
        for f in range(self.n):
            H, lm = self.H[f], self.strum_lm[f]
            if H is None or lm is None:
                continue
            p = (lm[0] + lm[5]) / 2  # wrist + index MCP
            ys[f] = G.apply_h(np.linalg.inv(H), p)[0, 1]
        ok = ~np.isnan(ys)
        if ok.sum() < 8:
            return None
        idx = np.arange(self.n)
        ys = np.interp(idx, idx[ok], ys[ok])
        win = 7 if self.n >= 7 else (self.n // 2) * 2 - 1
        if win >= 3:
            ys = savgol_filter(ys, win, 2)  # low-pass the trajectory
        near = np.convolve(ok.astype(float), np.ones(5), "same") > 0
        ys[~near] = np.nan  # don't invent motion across long tracking gaps
        self._strum_cache = ys
        return ys

    def to_json(self, every: int = 1) -> dict:
        def r(a, nd=4):
            return None if a is None else np.round(a, nd).reshape(-1).tolist()
        return {
            "fps": self.fps, "width": self.width, "height": self.height, "source": self.source,
            "mirrored": self.mirrored, "every": every, "stats": self.stats,
            "fret_x": G.FRET_X.tolist(), "string_y": G.STRING_Y.tolist(),
            "frames": [{"H": r(self.H[f], 6), "fret": r(self.fret_lm[f]), "strum": r(self.strum_lm[f])}
                       for f in range(0, self.n, every)],
        }


def _hand_landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4)
    return vision.HandLandmarker.create_from_options(opts)


def _assign_hands(hands: list[np.ndarray], H: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fretting hand = the hand nearer the nut along the neck axis."""
    if not hands:
        return None, None
    if H is None:
        return None, None
    Hinv = np.linalg.inv(H)
    xs = [float(G.apply_h(Hinv, h[[0, 9]].mean(axis=0))[0, 0]) for h in hands]
    order = np.argsort(xs)
    if len(hands) == 1:
        return (hands[0], None) if xs[0] < 0.62 else (None, hands[0])
    return hands[order[0]], hands[order[-1]]


def _neck_mask(H: np.ndarray, w: int, h: int, hands: list[np.ndarray]) -> np.ndarray:
    corners = G.apply_h(H, np.array([[-0.02, -0.15], [-0.02, 1.15], [G.FRET_X[15], 1.15], [G.FRET_X[15], -0.15]]))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, (corners * [w, h]).astype(np.int32), 255)
    for hand in hands:  # don't track features on the moving hands
        hull = cv2.convexHull((hand * [w, h]).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 0)
        cv2.polylines(mask, [hull], True, 0, thickness=max(4, w // 40))
    return mask


def analyze_video(video_path: str, calibration: dict | None, max_frames: int | None = None) -> VisionTrack | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("cannot open video %s", video_path)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = TRACK_WIDTH / W if W > TRACK_WIDTH else 1.0
    tw, th = int(W * scale), int(Hh * scale)

    detector = FretboardDetector.load()
    H_cal = None
    if calibration and calibration.get("points"):
        H_cal = G.homography_from_points({k: tuple(v) for k, v in calibration["points"].items()})
    if H_cal is None and detector is None:
        log.info("no fretboard calibration or model: vision limited to hands only")

    import mediapipe as mp
    landmarker = _hand_landmarker()
    euro = G.OneEuro(min_cutoff=1.5, beta=20.0)  # normalized coords: beta large => little lag on motion

    Hs, fret_lms, strum_lms = [], [], []
    prev_gray, pts = None, None
    H_cur = H_cal
    n_detect = n_track_fail = 0
    f = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and f >= max_frames):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int(f * 1000 / fps))
            hands = [np.array([[p.x, p.y] for p in hl], dtype=np.float64) for hl in res.hand_landmarks]
            small = cv2.resize(frame, (tw, th)) if scale != 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # 1) model detection every DETECT_EVERY frames (if a model is configured)
            detected = None
            if detector is not None and f % DETECT_EVERY == 0:
                kps = detector.detect(frame)
                if kps:
                    detected = G.homography_from_points(kps)
                    if detected is not None:
                        n_detect += 1

            # 2) optical-flow tracking between detections / from calibration
            if detected is not None:
                H_cur = detected
            elif H_cur is not None and prev_gray is not None and pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
                good0, good1 = pts[st[:, 0] == 1], nxt[st[:, 0] == 1]
                if len(good0) >= 8:
                    M, inl = cv2.findHomography(good0, good1, cv2.RANSAC, 2.5)
                    if M is not None and inl is not None and inl.sum() >= 8:
                        S = np.diag([tw, th, 1.0])
                        H_cur = np.linalg.inv(S) @ M @ S @ H_cur
                        H_cur /= H_cur[2, 2]
                        pts = good1[inl[:, 0] == 1].reshape(-1, 1, 2)
                    else:
                        n_track_fail += 1
                else:
                    n_track_fail += 1

            H_out = None
            if H_cur is not None:
                # smooth only what we output; tracking keeps propagating the raw estimate
                c = euro(G.quad_corners(H_cur).reshape(-1), f / fps).reshape(4, 2)
                H_out = G.homography_from_corners(c)
                if pts is None or len(pts) < 40 or f % 10 == 0:
                    mask = _neck_mask(H_cur, tw, th, hands)
                    pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                                  minDistance=6, mask=mask)
                    if pts is not None:
                        pts = pts.astype(np.float32)
            fret_hand, strum_hand = _assign_hands(hands, H_out)
            Hs.append(H_out)
            fret_lms.append(fret_hand)
            strum_lms.append(strum_hand)
            prev_gray = gray
            f += 1
    finally:
        cap.release()
        landmarker.close()

    if f == 0:
        return None
    H_first = next((h for h in Hs if h is not None), None)
    track = VisionTrack(
        fps=fps, width=W, height=Hh, H=Hs, fret_lm=fret_lms, strum_lm=strum_lms,
        source="model" if n_detect else "calibration",
        mirrored=bool(H_first is not None and G.is_mirrored(H_first)),
        stats={"frames": f, "frames_with_grid": sum(h is not None for h in Hs),
               "frames_with_fret_hand": sum(h is not None for h in fret_lms),
               "frames_with_strum_hand": sum(h is not None for h in strum_lms),
               "detections": n_detect, "track_failures": n_track_fail})
    return track
