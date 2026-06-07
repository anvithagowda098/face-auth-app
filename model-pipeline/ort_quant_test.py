#!/usr/bin/env python3
"""
ort_quant_test.py — decisive isolation test.

Quantize the ORIGINAL ONNX to int8 with onnxruntime (per-channel weights),
then benchmark fp32-onnx vs int8-onnx with onnxruntime — completely bypassing
onnx2tf/TFLite. This answers: can THIS model be int8-quantized at all, or is
onnx2tf the broken link?

  fp32 onnx  ~= 99.7%  AND  int8 onnx ~= 99%   -> model is fine; onnx2tf is the bug
  int8 onnx also collapses                      -> model needs mixed precision
"""

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

import cv2
from pathlib import Path
from benchmark import read_lfw_pairs, tar_at_far, equal_error_rate

FP32 = "weights/w600k_mbf.onnx"
INT8 = "weights/w600k_mbf_ort_int8.onnx"
PAIRS, DATA = "data/pairs.txt", "data/lfw_aligned"


def load_nchw(path, mean=127.5, std=127.5):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    img = cv2.cvtColor(cv2.resize(bgr, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW (1,3,112,112)


class OrtEmbedder:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def embed(self, x):
        e = self.sess.run(None, {self.iname: x.astype(np.float32)})[0][0]
        n = np.linalg.norm(e)
        return e / n if n > 0 else e


def evaluate(emb, pairs):
    cache, scores, labels, skip = {}, [], [], 0
    def get(p):
        sp = str(p)
        if sp not in cache:
            cache[sp] = load_nchw(p)
        return cache[sp]
    for a, b, same in pairs:
        if not (Path(a).exists() and Path(b).exists()):
            skip += 1; continue
        ca, cb = get(a), get(b)
        if ca is None or cb is None:
            skip += 1; continue
        scores.append(float(np.dot(emb.embed(ca), emb.embed(cb))))
        labels.append(int(same))
    scores, labels = np.array(scores), np.array(labels)
    thrs = np.unique(scores)
    best_acc, best_thr = max(((((scores >= t) == labels).mean(), t) for t in thrs), key=lambda z: z[0])
    out = {
        "n_pairs": len(scores), "n_skipped": skip,
        "best_accuracy": round(float(best_acc), 4), "best_threshold": round(float(best_thr), 4),
        "EER": equal_error_rate(scores, labels),
        "genuine_mean": round(float(scores[labels == 1].mean()), 4),
        "impostor_mean": round(float(scores[labels == 0].mean()), 4),
    }
    out.update(tar_at_far(scores, labels))
    return out


def main():
    print("[1] quantize_dynamic (per-channel int8 weights) ...")
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8, per_channel=True)
    sz = Path(INT8).stat().st_size / 1e6
    print(f"    {INT8}  ({sz:.2f} MB)")

    pairs = read_lfw_pairs(PAIRS, DATA, ext="jpg")
    print(f"[2] benchmarking on {len(pairs)} pre-aligned pairs (onnxruntime) ...")

    print("\n--- fp32 onnx ---")
    print(evaluate(OrtEmbedder(FP32), pairs))
    print("\n--- int8 onnx (onnxruntime per-channel) ---")
    print(evaluate(OrtEmbedder(INT8), pairs))


if __name__ == "__main__":
    main()
