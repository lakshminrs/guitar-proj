"""YOLO-pose fretboard keypoint detector (ONNX Runtime inference).

Train with `scripts/train_fretboard.py` (Ultralytics YOLO11-pose), export to ONNX, then set
FRETBOARD_MODEL_PATH=/path/to/fretboard.onnx. Without a model the pipeline falls back to the
manual calibration + optical-flow tracking, and still returns an audio grade.

Keypoint order (12) must match data/fretboard.yaml:
  nut_low, nut_high, f3_low, f3_high, f5_low, f5_high, f7_low, f7_high,
  f9_low, f9_high, f12_low, f12_high
("low" = low-E edge of the neck, "high" = high-E edge.)
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)

KEYPOINTS = ["nut_low", "nut_high", "f3_low", "f3_high", "f5_low", "f5_high",
             "f7_low", "f7_high", "f9_low", "f9_high", "f12_low", "f12_high"]
IMG = 640
CONF_THR = 0.35
KP_CONF_THR = 0.5


class FretboardDetector:
    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=ort.get_available_providers())
        self.inp = self.sess.get_inputs()[0].name

    @classmethod
    def load(cls) -> "FretboardDetector | None":
        path = os.environ.get("FRETBOARD_MODEL_PATH")
        if not path or not os.path.exists(path):
            return None
        try:
            return cls(path)
        except Exception:
            log.exception("failed to load fretboard model %s", path)
            return None

    def detect(self, bgr: np.ndarray) -> dict[str, tuple[float, float]] | None:
        """Returns keypoints in normalized image coords for the best neck detection."""
        h, w = bgr.shape[:2]
        r = IMG / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad = np.full((IMG, IMG, 3), 114, np.uint8)
        top, left = (IMG - nh) // 2, (IMG - nw) // 2
        pad[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
        x = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]   # (4 + 1 + K*3, N)
        conf = out[4]
        best = int(np.argmax(conf))
        if conf[best] < CONF_THR:
            return None
        kp = out[5:, best].reshape(-1, 3)
        res = {}
        for name, (kx, ky, kc) in zip(KEYPOINTS, kp):
            if kc < KP_CONF_THR:
                continue
            res[name] = (float((kx - left) / r / w), float((ky - top) / r / h))
        return res if len(res) >= 4 else None
