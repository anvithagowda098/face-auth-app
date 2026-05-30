"""
align.py — 5-point face alignment to the ArcFace template.

Why this exists: the recognizer is trained on faces warped so that the eyes,
nose tip, and mouth corners land at fixed positions in a 112x112 image. Feeding
it un-aligned crops costs a lot of accuracy. Two ways to use this:

  1. norm_crop(img, landmarks)  -- you already have 5 landmarks (e.g. from
     BlazeFace on-device, or MediaPipe), warp the face to the template.
  2. InsightFaceAligner          -- raw image in, aligned 112x112 crop out.
     Uses insightface's detector, which produces the SAME 5-point alignment the
     ArcFace weights were trained with, so benchmark numbers stay consistent.

For pre-aligned academic eval sets (aligned LFW etc.) you don't need this at
benchmark time — the warp was done when the set was prepared. You need it for
raw images and for the live/on-device pipeline.
"""

import numpy as np
import cv2

# Canonical ArcFace destination landmarks for a 112x112 crop:
# [left eye, right eye, nose tip, left mouth corner, right mouth corner]
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def estimate_transform(landmarks, image_size=112):
    """Similarity transform (rotation+scale+translation) from src 5pts -> template."""
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_DST.copy()
    if image_size != 112:
        dst = dst * (image_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)
    return M


def norm_crop(img_bgr, landmarks, image_size=112):
    """Warp a face to the aligned 112x112 template given its 5 landmarks."""
    M = estimate_transform(landmarks, image_size)
    return cv2.warpAffine(img_bgr, M, (image_size, image_size), borderValue=0.0)


class InsightFaceAligner:
    """
    Raw image -> aligned 112x112 crop, using a LOCAL insightface detector ONNX
    for the 5 landmarks (no auto-download — works on locked-down boxes).
    Requires: pip install insightface onnxruntime
    Fetch the detector first (get_model_hf.py grabs det_500m.onnx into weights/).
    """
    def __init__(self, det_onnx="weights/det_500m.onnx", det_size=640):
        from insightface.model_zoo import model_zoo
        import os
        if not os.path.exists(det_onnx):
            raise FileNotFoundError(
                f"{det_onnx} not found — run get_model_hf.py to fetch the detector")
        self.det = model_zoo.get_model(det_onnx)
        self.det.prepare(ctx_id=-1, input_size=(det_size, det_size))  # ctx_id=-1 = CPU

    def align(self, img_bgr, image_size=112):
        bboxes, kpss = self.det.detect(img_bgr, max_num=0, metric="default")
        if kpss is None or len(kpss) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return norm_crop(img_bgr, kpss[int(areas.argmax())], image_size)


def build_aligner(kind, det_onnx="weights/det_500m.onnx"):
    """kind: 'none' (pre-aligned data) or 'insightface' (raw images)."""
    if kind in (None, "none"):
        return None
    if kind == "insightface":
        return InsightFaceAligner(det_onnx=det_onnx)
    raise ValueError(f"unknown aligner '{kind}'")
