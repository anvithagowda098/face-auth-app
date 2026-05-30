#!/usr/bin/env python3
"""
run.py — build/load -> export (per precision) -> benchmark -> summary.

Model source (pick one):
  --onnx PATH    pretrained ONNX (e.g. insightface w600k_mbf.onnx) — REAL weights
  --weights W    our MobileFaceNet checkpoint (omit both -> random init, size/latency only)

Tiers:
  + --calib DIR            adds int8_dynamic / int8_full
  + --pairs + --data       adds accuracy (TAR@FAR / EER / best-acc)
  + --align insightface    re-align raw eval images to the ArcFace template
                           (needs: pip install insightface onnxruntime)

Examples
--------
# pretrained MobileFaceNet -> all precisions -> real accuracy on LFW
python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops \
    --pairs data/pairs.txt --data data/lfw-deepfunneled
"""

import argparse
import json
from pathlib import Path

from export import export_precisions, PRECISION_TAG, NEEDS_CALIB
from benchmark import run_one, read_lfw_pairs


def main():
    ap = argparse.ArgumentParser(description="EdgeFace recognizer: load -> export -> benchmark")
    ap.add_argument("--onnx", help="pretrained ONNX (real weights, e.g. w600k_mbf.onnx)")
    ap.add_argument("--weights", help="our-model checkpoint (random if omitted)")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--calib", help="folder of crops -> enables int8_dynamic/int8_full")
    ap.add_argument("--pairs", help="LFW-format pairs.txt -> enables accuracy")
    ap.add_argument("--data", help="image root -> enables accuracy")
    ap.add_argument("--align", default="none", choices=["none", "insightface"],
                    help="re-align raw eval images to the ArcFace template")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--precisions", nargs="*",
                    help="subset of fp32 fp16 int8_dynamic int8_full (default: auto)")
    args = ap.parse_args()

    if args.precisions:
        precisions = args.precisions
    elif args.calib:
        precisions = ["fp32", "fp16", "int8_dynamic", "int8_full", "int16x8"]
    else:
        precisions = ["fp32", "fp16"]
    if any(p in NEEDS_CALIB for p in precisions) and not args.calib:
        raise SystemExit("int8_dynamic/int8_full selected but --calib <crops folder> not given")

    real_weights = bool(args.onnx)   # only the pretrained ONNX path has real weights here

    print("=" * 66)
    print("  EdgeFace Sentinel — recognizer runner")
    print("=" * 66)
    print(f"  source     : {'ONNX ' + args.onnx if args.onnx else 'our MobileFaceNet'}")
    print(f"  precisions : {precisions}")
    if not real_weights:
        print("  [!] no pretrained weights: size + latency are REAL; accuracy is random noise.")

    # 1) export each precision (single onnx2tf pass)
    print("\n  [1] export (onnx2tf) ...")
    if args.onnx:
        tflites = export_precisions(precisions, args.outdir, onnx_path=args.onnx,
                                    calib_dir=args.calib, size=args.size)
    else:
        from models.recognizer import build_recognizer
        model = build_recognizer(embedding_dim=args.embedding_dim,
                                 weights=args.weights, normalize_output=True)
        print(f"      our model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        tflites = export_precisions(precisions, args.outdir, model=model,
                                    input_shape=(1, 3, args.size, args.size),
                                    calib_dir=args.calib, size=args.size)
    if not tflites:
        raise SystemExit("no tflite files produced — see onnx2tf output above")

    # 2) benchmark
    pairs, aligner = None, None
    do_acc = bool(args.pairs and args.data)
    if do_acc:
        pairs = read_lfw_pairs(args.pairs, args.data, ext=args.ext)
        if args.align != "none":
            from align import build_aligner   # only needed when re-aligning
            aligner = build_aligner(args.align)
        align_note = f" (aligner={args.align})" if args.align != "none" else " (no re-align; assumes pre-aligned)"
        print(f"\n  [2] benchmarking: latency + accuracy on {len(pairs)} pairs{align_note}")
    else:
        print("\n  [2] benchmarking: latency only (add --pairs + --data for accuracy)")

    base = tflites.get("fp32", next(iter(tflites.values())))[1]
    results = {}
    for prec, (path, sz) in tflites.items():
        r = run_one(path, pairs, aligner=aligner)
        r["size_vs_fp32"] = round(base / sz, 2) if sz else None
        results[prec] = r

    # 3) summary
    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    header = f"  {'precision':<13}{'size(MB)':<10}{'lat_mean':<12}"
    if do_acc:
        header += f"{'best_acc':<10}{'EER':<8}{'TAR@1e-3':<9}"
    print(header)
    print("  " + "-" * len(header))
    for prec, r in results.items():
        lat = r.get("latency", {}).get("mean_ms", "-")
        row = f"  {prec:<13}{r['size_mb']:<10}{str(lat) + ' ms':<12}"
        if do_acc:
            a = r.get("accuracy", {})
            row += (f"{a.get('best_accuracy', '-'):<10}"
                    f"{a.get('EER', '-'):<8}"
                    f"{a.get('TAR@FAR=1e-03', '-'):<9}")
        print(row)
    for prec, r in results.items():
        if r.get("note"):
            print(f"  note ({prec}): {r['note']}")
        acc = r.get("accuracy")
        if isinstance(acc, dict) and acc.get("error"):
            print(f"  accuracy error ({prec}): {acc['error']}")
        elif isinstance(acc, dict) and acc.get("n_skipped"):
            print(f"  accuracy ({prec}): {acc['n_skipped']} pairs skipped (no face detected)")

    Path(args.outdir, "run_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  full results -> {Path(args.outdir, 'run_results.json')}")
    print("  reminder: latency is desktop/CPU; int8 speedup shows on ARM/NPU, not x86.")
    if do_acc and not real_weights:
        print("  reminder: accuracy is RANDOM-weight noise — use --onnx for real weights.")


if __name__ == "__main__":
    main()
