#!/usr/bin/env python3
"""
make_eval_hf.py — build a verification eval set + pairs.txt from a HF dataset.

Streams an identity-labeled face dataset and writes:
  data/lfw/<name>/<name>_0001.jpg ...        (benchmark's expected layout)
  data/pairs.txt                              (genuine + impostor, LFW format)

HF-only — no university hosts. Works for any image-classification-style dataset
where the label is the person's identity (auto-detects the image + label columns).

Setup: pip install datasets pillow
Run:   python make_eval_hf.py
Then:  python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops \
           --pairs data/pairs.txt --data data/lfw

If the default dataset has no identity labels, try:
       python make_eval_hf.py --dataset vilsonrodrigues/lfw
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

PREFERRED = ["name", "identity", "person", "label", "id", "target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="0xmoose/lfw-deepfunneled")
    ap.add_argument("--out", default="data/lfw")
    ap.add_argument("--pairs", default="data/pairs.txt")
    ap.add_argument("--max-images", type=int, default=4000)
    ap.add_argument("--genuine", type=int, default=500)
    ap.add_argument("--impostor", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset, get_dataset_split_names
        from PIL import Image as PILImage
    except ImportError:
        raise SystemExit("pip install datasets pillow")

    try:
        splits = get_dataset_split_names(args.dataset)
    except Exception as e:
        raise SystemExit(f"could not reach HF dataset '{args.dataset}': {e}")
    split = "train" if "train" in splits else splits[0]
    ds = load_dataset(args.dataset, split=split, streaming=True)
    feats = getattr(ds, "features", None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    img_key = label_key = None
    total = 0

    def to_name(v):
        try:
            return feats[label_key].int2str(v)
        except Exception:
            return str(v)

    print(f"  streaming {args.dataset} [{split}] ...")
    for ex in ds:
        if img_key is None:
            img_key = next((k for k, v in ex.items() if isinstance(v, PILImage.Image)), None)
            scal = [k for k, v in ex.items() if k != img_key and isinstance(v, (int, str))]
            label_key = next((k for k in PREFERRED if k in scal), scal[0] if scal else None)
            if img_key is None or label_key is None:
                raise SystemExit("dataset lacks image+identity columns; try --dataset vilsonrodrigues/lfw")

        img = ex.get(img_key)
        if not isinstance(img, PILImage.Image):
            continue
        name = "".join(c if c.isalnum() else "_" for c in str(to_name(ex.get(label_key)))).strip("_") or "unknown"
        counts[name] += 1
        d = out / name
        d.mkdir(exist_ok=True)
        img.convert("RGB").save(d / f"{name}_{counts[name]:04d}.jpg", quality=95)
        total += 1
        if total >= args.max_images:
            break

    multi = [n for n, c in counts.items() if c >= 2]
    if not multi:
        raise SystemExit("no identity has >=2 images in the sample; raise --max-images or change --dataset")

    rng = random.Random(args.seed)
    all_names = list(counts)
    genuine, impostor = [], []
    while len(genuine) < args.genuine:
        n = rng.choice(multi)
        i, j = rng.sample(range(1, counts[n] + 1), 2)
        genuine.append((n, i, j))
    while len(impostor) < args.impostor and len(all_names) >= 2:
        a, b = rng.sample(all_names, 2)
        impostor.append((a, rng.randint(1, counts[a]), b, rng.randint(1, counts[b])))

    lines = [str(len(genuine))]  # header line (benchmark skips non 3/4-token lines)
    lines += [f"{n}\t{i}\t{j}" for n, i, j in genuine]
    lines += [f"{a}\t{i}\t{b}\t{j}" for a, i, b, j in impostor]
    Path(args.pairs).write_text("\n".join(lines) + "\n")

    print(f"  saved {total} images across {len(counts)} identities -> {out}")
    print(f"  pairs: {len(genuine)} genuine + {len(impostor)} impostor -> {args.pairs}")
    print(f"\n  next: python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops "
          f"--pairs {args.pairs} --data {args.out}")
    sys.stdout.flush()
    os._exit(0)   # skip datasets/aiohttp teardown (avoids the finalize crash)


if __name__ == "__main__":
    main()
