#!/usr/bin/env python3
"""
make_eval_sklearn.py — build LFW eval set + pairs.txt via scikit-learn.

The HF LFW mirrors dropped per-person identity (label = ingestion/recovery, 2
classes), so they can't form verification pairs. sklearn's fetch_lfw_people
keeps real person identities. Writes the layout benchmark.py already reads:
  data/lfw/<name>/<name>_0001.jpg ...
  data/pairs.txt   (genuine + impostor)

Setup: pip install scikit-learn pillow
Run:   python make_eval_sklearn.py
Then:  python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops \
           --pairs data/pairs.txt --data data/lfw
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-faces", type=int, default=3, help="keep identities with >= this many images")
    ap.add_argument("--out", default="data/lfw")
    ap.add_argument("--pairs", default="data/pairs.txt")
    ap.add_argument("--genuine", type=int, default=500)
    ap.add_argument("--impostor", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from sklearn.datasets import fetch_lfw_people
        from PIL import Image
    except ImportError:
        raise SystemExit("pip install scikit-learn pillow")

    print("  fetching LFW via scikit-learn (first run downloads ~200MB) ...")
    try:
        # slice_=None keeps the full funneled image (face + surrounding context),
        # which the face detector needs for --align insightface to work.
        lfw = fetch_lfw_people(min_faces_per_person=args.min_faces, color=True,
                               resize=1.0, funneled=True, slice_=None)
    except Exception as e:
        raise SystemExit(
            f"sklearn could not download LFW: {e}\n"
            f"If this is a DNS/Name-resolution error, the box can't reach sklearn's "
            f"host either (only PyPI + HF are open) — use the scp fallback."
        )

    images, targets, names = lfw.images, lfw.target, lfw.target_names
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    per_person = defaultdict(list)
    for img, t in zip(images, targets):
        name = "".join(c if c.isalnum() else "_" for c in str(names[t])).strip("_") or "unknown"
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        counts[name] += 1
        d = out / name
        d.mkdir(exist_ok=True)
        Image.fromarray(arr).save(d / f"{name}_{counts[name]:04d}.jpg", quality=95)
        per_person[name].append(counts[name])

    multi = [n for n, idxs in per_person.items() if len(idxs) >= 2]
    all_names = list(per_person)
    if not multi:
        raise SystemExit("no identity has >=2 images; lower --min-faces")

    rng = random.Random(args.seed)
    genuine, impostor = [], []
    while len(genuine) < args.genuine:
        n = rng.choice(multi)
        i, j = rng.sample(per_person[n], 2)
        genuine.append((n, i, j))
    while len(impostor) < args.impostor and len(all_names) >= 2:
        a, b = rng.sample(all_names, 2)
        impostor.append((a, rng.choice(per_person[a]), b, rng.choice(per_person[b])))

    lines = [str(len(genuine))]
    lines += [f"{n}\t{i}\t{j}" for n, i, j in genuine]
    lines += [f"{a}\t{i}\t{b}\t{j}" for a, i, b, j in impostor]
    Path(args.pairs).write_text("\n".join(lines) + "\n")

    print(f"  {sum(counts.values())} images across {len(counts)} identities -> {out}")
    print(f"  pairs: {len(genuine)} genuine + {len(impostor)} impostor -> {args.pairs}")
    print(f"\n  next: python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops "
          f"--pairs {args.pairs} --data {args.out}")


if __name__ == "__main__":
    main()
