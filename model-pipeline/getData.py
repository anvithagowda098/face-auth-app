#!/usr/bin/env python3
"""
get_data_hf.py — pull face crops for int8 calibration via HuggingFace.

Your box resolves PyPI/HuggingFace but not arbitrary university hosts, so we
STREAM a face dataset from the HF hub (only the ~200 images we need, not the
whole thing) instead of wget-ing umass.

Setup:
  pip install datasets pillow

Run:
  python get_data_hf.py                                   # -> data/calib_crops
  python get_data_hf.py --dataset bitmind/lfw --n 200
  python get_data_hf.py --dataset sammyboi1801/lfw-face-transformer-dataset

Then:
  python run.py --calib data/calib_crops

Note: these crops are for int8 CALIBRATION only — representative face pixels so
the quantizer can set activation ranges. They don't need to be ArcFace-aligned,
and accuracy still needs real --weights. (Verification pairs come later.)
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitmind/lfw",
                    help="HF dataset slug (face images). Try sammyboi1801/lfw-face-transformer-dataset if this 404s")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/calib_crops")
    args = ap.parse_args()

    try:
        from datasets import load_dataset, get_dataset_split_names
        from PIL import Image
    except ImportError:
        raise SystemExit("pip install datasets pillow   # then re-run")

    try:
        splits = get_dataset_split_names(args.dataset)
    except Exception as e:
        raise SystemExit(
            f"could not reach HF dataset '{args.dataset}': {e}\n"
            f"If HuggingFace is also blocked on this box, use the scp fallback."
        )
    split = "train" if "train" in splits else splits[0]
    print(f"  streaming {args.dataset} [{split}] ...")
    ds = load_dataset(args.dataset, split=split, streaming=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for ex in ds:
        img = next((v for v in ex.values() if isinstance(v, Image.Image)), None)
        if img is None:
            continue
        img.convert("RGB").save(out / f"calib_{saved:04d}.jpg", quality=95)
        saved += 1
        if saved >= args.n:
            break

    if saved == 0:
        raise SystemExit("no images found in dataset examples — try a different --dataset")
    print(f"  saved {saved} crops -> {out}")
    print(f"\n  next: python run.py --calib {args.out}")


if __name__ == "__main__":
    main()
