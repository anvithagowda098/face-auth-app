#!/usr/bin/env python3
"""
quant_sweep.py — produce EVERY tflite variant onnx2tf can emit (keeping the
output folder), then benchmark each on the pre-aligned eval set.

Why: full-integer int8 collapses this ArcFace MobileFaceNet (activation int8
crushes the embedding -> genuine~=impostor). We need the variant that keeps
activations in float (dynamic-range / weights-only int8), which export.py was
discarding due to a tag-name mismatch. This script keeps ALL of them so we can
see which precision actually holds >95%.
"""

import json
from pathlib import Path

import numpy as np
import onnx2tf

from export import _build_calib_npy, _onnx_input_name
from benchmark import run_one, read_lfw_pairs

ONNX = "weights/w600k_mbf.onnx"
CALIB_DIR = "data/calib_aligned"
OUT = Path("out/onnx2tf_all")
PAIRS, DATA = "data/pairs.txt", "data/lfw_aligned"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    workdir = Path("out")
    npy = _build_calib_npy(CALIB_DIR, 112, workdir)
    in_name = _onnx_input_name(ONNX)

    onnx2tf.convert(
        input_onnx_file_path=ONNX,
        output_folder_path=str(OUT),
        copy_onnx_input_output_names_to_tflite=True,
        output_integer_quantized_tflite=True,                 # integer_quant + full_integer_quant
        output_dynamic_range_quantized_tflite=True,           # weights-only int8 (activations float)
        custom_input_op_name_np_data_path=[[in_name, str(npy), 0.0, 1.0]],
    )

    produced = sorted(OUT.glob("*.tflite"))
    print("\n=== produced tflite variants ===")
    for f in produced:
        print(f"  {f.name}  ({f.stat().st_size/1e6:.2f} MB)")

    pairs = read_lfw_pairs(PAIRS, DATA, ext="jpg")
    print(f"\n=== benchmarking each on {len(pairs)} pre-aligned pairs ===")
    results = {}
    for f in produced:
        r = run_one(str(f), pairs, do_latency=True, aligner=None)
        a = r.get("accuracy", {})
        results[f.name] = r
        print(f"  {f.name:<55} size={r['size_mb']:>6}MB "
              f"acc={a.get('best_accuracy','-')!s:<7} EER={a.get('EER','-')!s:<7} "
              f"gen={a.get('genuine_mean','-')!s:<8} imp={a.get('impostor_mean','-')!s:<8} "
              f"{'ERR:'+r['error'] if r.get('error') else ''}")

    Path("out/quant_sweep_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> out/quant_sweep_results.json")


if __name__ == "__main__":
    main()
