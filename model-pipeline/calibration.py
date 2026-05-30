"""
calibration.py — representative dataset for int8_full / int16x8.

int8 quantisation is only meaningful when calibrated on the real input
distribution. Point this at a folder of aligned face crops (the same
preprocessing you use at inference) — NOT random noise.
"""

from pathlib import Path
import numpy as np
import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(bgr, size=112, mean=127.5, std=128.0, to_rgb=True):
    """
    Standard ArcFace/MobileFaceNet preprocessing -> NHWC float32.
    IMPORTANT: must exactly match how the pretrained weights were trained.
    The common recipe is RGB, (x - 127.5) / 128.0 -> ~[-1, 1].
    """
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - mean) / std
    return img[np.newaxis, ...]  # (1, H, W, C)


def representative_dataset_factory(image_dir: str,
                                   size: int = 112,
                                   max_samples: int = 300,
                                   mean: float = 127.5,
                                   std: float = 128.0,
                                   to_rgb: bool = True):
    """Return a zero-arg generator suitable for TFLiteConverter.representative_dataset."""
    paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMG_EXT]
    if not paths:
        raise FileNotFoundError(f"no images found under {image_dir}")
    paths = paths[:max_samples]
    print(f"[calibration] {len(paths)} crops from {image_dir}")

    def gen():
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            yield [preprocess(bgr, size, mean, std, to_rgb)]

    return gen
