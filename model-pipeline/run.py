#!/usr/bin/env python3
"""
run.py — one command: build -> export (all precisions) -> benchmark -> summary.

Runs in tiers based on what you provide:

  no args            architecture + size + latency  (REAL; no weights needed)
  + --calib DIR      also exports int8_full / int16x8
  + --weights W      loads real weights (required for meaningful accuracy)
  + --pairs + --data also runs accuracy: TAR@FAR / EER / best-acc per precision

Examples
--------
# Tier 1 — validate the architecture + size fix, nothing to download:
python run.py

# Tier 2 — full ablation with real numbers:
python run.py --weights weights/mobilefacenet.pt \
    --calib data/calib_crops \
    --pairs data/lfw_pairs.txt --data data/lfw_aligned
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from precision import NEEDS_CALIBRATION, convert_saved_model, fmt_size
from calibration import representative_dataset_factory
from models.recognizer import build_recognizer
from export import torch_to_saved_model
from benchmark import run_one, read_lfw_pairs


def main():
    ap = argparse.ArgumentParser(description="EdgeFace recognizer: build -> export -> benchmark")
    ap.add_argument("--weights", help="checkpoint (.pt/.pth); omit for architecture/size/latency only")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--calib", help="folder of aligned crops -> enables int8_full / int16x8")
    ap.add_argument("--pairs", help="LFW-format pairs.txt -> enables accuracy")
    ap.add_argument("--data", help="aligned image root -> enables accuracy")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--precisions", nargs="*",
                    help="subset of fp32 fp16 int8_dynamic int8_full int16x8 (default: auto)")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # choose precisions
    if args.precisions:
        precisions = args.precisions
    else:
        precisions = ["fp32", "fp16", "int8_dynamic"]
        if args.calib:
            precisions += ["int8_full", "int16x8"]
    need_calib = any(p in NEEDS_CALIBRATION for p in precisions)
    if need_calib and not args.calib:
        raise SystemExit("int8_full/int16x8 selected but --calib <crops folder> not given")

    print("=" * 66)
    print("  EdgeFace Sentinel — recognizer runner")
    print("=" * 66)
    if not args.weights:
        print("  [!] no --weights: size + latency + architecture are REAL.")
        print("      Accuracy (if --pairs/--data given) is random-weight noise")
        print("      until you supply real --weights.")
    print(f"  precisions : {precisions}")
    print(f"  outdir     : {out}")

    # 1) build + architecture check
    model = build_recognizer(embedding_dim=args.embedding_dim,
                             weights=args.weights, normalize_output=True)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  [1] model: {nparams:.2f}M params | embedding dim {args.embedding_dim}")

    # 2) torch -> SavedModel once, then 3) export each precision
    workdir = Path(tempfile.mkdtemp(prefix="run_"))
    tflites = {}
    try:
        print("  [2] torch -> ONNX -> TF SavedModel (onnx2tf) ...")
        sm = torch_to_saved_model(model, (1, 3, args.size, args.size), workdir)

        rep_ds = representative_dataset_factory(args.calib, size=args.size) if need_calib else None

        print("  [3] exporting:")
        for prec in precisions:
            path = out / f"recognizer_{prec}.tflite"
            sz = convert_saved_model(
                str(sm), prec, str(path),
                representative_dataset=rep_ds if prec in NEEDS_CALIBRATION else None,
            )
            tflites[prec] = (str(path), sz)
            print(f"      {prec:<13} {fmt_size(sz)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # 4) benchmark each (latency always; accuracy if pairs+data)
    pairs = None
    do_acc = bool(args.pairs and args.data)
    if do_acc:
        pairs = read_lfw_pairs(args.pairs, args.data, ext=args.ext)
        print(f"\n  [4] benchmarking: latency + accuracy on {len(pairs)} pairs")
    else:
        print("\n  [4] benchmarking: latency only (add --pairs + --data for accuracy)")

    base = tflites.get("fp32", next(iter(tflites.values())))[1]
    results = {}
    for prec, (path, sz) in tflites.items():
        r = run_one(path, pairs)
        r["size_vs_fp32"] = round(base / sz, 2) if sz else None
        results[prec] = r

    # 5) summary
    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    header = f"  {'precision':<13}{'size(MB)':<10}{'lat_mean':<12}"
    if do_acc:
        header += f"{'best_acc':<10}{'EER':<8}{'TAR@1e-3':<9}"
    print(header)
    print("  " + "-" * (len(header)))
    for prec, r in results.items():
        lat = r.get("latency", {}).get("mean_ms", "-")
        row = f"  {prec:<13}{r['size_mb']:<10}{str(lat) + ' ms':<12}"
        if do_acc:
            a = r.get("accuracy", {})
            row += (f"{a.get('best_accuracy', '-'):<10}"
                    f"{a.get('EER', '-'):<8}"
                    f"{a.get('TAR@FAR=1e-03', '-'):<9}")
        print(row)

    (out / "run_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  full results -> {out / 'run_results.json'}")
    print("  reminder: latency is desktop/CPU, not a phone number.")
    if do_acc and not args.weights:
        print("  reminder: accuracy above is RANDOM-weight noise — supply --weights.")


if __name__ == "__main__":
    main()
