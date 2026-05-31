#!/usr/bin/env python3
"""
finalize_int8.py — produce + benchmark the deployable int8 recognizers.

onnx2tf's quantiser collapses this MobileFaceNet (per-tensor weight scaling
kills the depthwise channels). onnxruntime's per-CHANNEL int8 keeps 99.7%.
This script produces the two deployable int8 ONNX variants and benchmarks
accuracy + latency against fp32:

  dynamic int8  : weights int8 / activations float, no calibration. Smallest,
                  most robust, ~fp32 accuracy. Latency ~ fp32 (size is the win).
  static  int8  : weights + activations int8 (QDQ, per-channel), calibrated on
                  aligned crops. Integer kernels -> faster on ARM CPU/NNAPI.

Outputs:
  weights/w600k_mbf_int8_dynamic.onnx
  weights/w600k_mbf_int8_static.onnx
  out/int8_final_results.json
"""

import json
import time
from pathlib import Path

import numpy as np
import cv2
import onnx
from onnx import version_converter
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from benchmark import read_lfw_pairs, tar_at_far, equal_error_rate
from ort_quant_test import load_nchw, OrtEmbedder, evaluate

FP32 = "weights/w600k_mbf.onnx"
FP32_PREP = "weights/w600k_mbf_op13_prep.onnx"   # opset-13 + shape-inferred (for static QDQ)
DYN = "weights/w600k_mbf_int8_dynamic.onnx"
STAT = "weights/w600k_mbf_int8_static.onnx"
CALIB_DIR = "data/calib_aligned"
PAIRS, DATA = "data/pairs.txt", "data/lfw_aligned"


class CalibReader(CalibrationDataReader):
    def __init__(self, input_name, max_n=200):
        self.input_name = input_name
        paths = sorted(Path(CALIB_DIR).glob("*.jpg"))[:max_n]
        self.data = (load_nchw(p) for p in paths)

    def get_next(self):
        x = next(self.data, None)
        return None if x is None else {self.input_name: x.astype(np.float32)}


def latency(path, runs=50, warmup=5):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 112, 112).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {iname: x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {iname: x})
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": round(float(ts.mean()), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}


def main():
    iname = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    print("[1] dynamic int8 (per-channel weights) ...")
    quantize_dynamic(FP32, DYN, weight_type=QuantType.QInt8, per_channel=True)

    print("[2] static int8 (QDQ, per-channel, calibrated on aligned crops) ...")
    variants = {"fp32": FP32, "int8_dynamic": DYN}
    try:
        # per-channel QDQ needs opset>=13 (DequantizeLinear 'axis'); upstream model
        # is older -> upgrade opset, then run ORT's recommended pre-processing.
        m13 = version_converter.convert_version(onnx.load(FP32), 13)
        onnx.save(m13, "weights/_tmp_op13.onnx")
        quant_pre_process("weights/_tmp_op13.onnx", FP32_PREP, skip_symbolic_shape=True)
        quantize_static(FP32_PREP, STAT, CalibReader(iname), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8,
                        activation_type=QuantType.QInt8)
        variants["int8_static"] = STAT
    except Exception as e:
        print(f"    [static int8 skipped] {type(e).__name__}: {str(e).splitlines()[-1][:160]}")

    pairs = read_lfw_pairs(PAIRS, DATA, ext="jpg")
    print(f"[3] benchmarking on {len(pairs)} pre-aligned pairs ...\n")

    results = {}
    for name, path in variants.items():
        try:
            acc = evaluate(OrtEmbedder(path), pairs)
            lat = latency(path)
            size_mb = round(Path(path).stat().st_size / 1e6, 2)
            results[name] = {"path": path, "size_mb": size_mb, "latency": lat, "accuracy": acc}
            print(f"  {name:<14} {size_mb:>6} MB  lat {lat['mean_ms']:>6} ms  "
                  f"acc {acc['best_accuracy']}  EER {acc['EER']}  "
                  f"TAR@1e-3 {acc['TAR@FAR=1e-03']}  gen {acc['genuine_mean']} imp {acc['impostor_mean']}")
        except Exception as e:
            print(f"  {name:<14} FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:160]}")

    Path("out/int8_final_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> out/int8_final_results.json")


if __name__ == "__main__":
    main()
