#!/usr/bin/env python3
"""
make_eval_hf.py — build a verification eval set + pairs.txt from a HF dataset.

Writes the layout benchmark.py reads:
  data/lfw/<name>/<name>_0001.jpg ...
  data/pairs.txt   (genuine + impostor)

Works for any identity-labeled face dataset (auto-detects the image + label
columns). Tries streaming load_dataset first; if that fails (e.g. AutoTrain
'save_to_disk' repos whose dataset-viewer is broken), falls back to
snapshot_download + load_from_disk and merges all splits.

Default: amitpuri/bollywood-celebs (100 Indian celebrities, ~8.6k images).
  NOTE: those are 64x64 localized face crops — low-res and already cropped, so
  run the benchmark on this WITHOUT --align (the detector won't find a face in a
  tight 64px crop). Numbers are a relative demographic signal, not comparable to
  the aligned-LFW 99.7%.

Setup: pip install datasets huggingface_hub pillow
Run:   python make_eval_hf.py
Then:  python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops \
           --pairs data/pairs.txt --data data/lfw
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PREFERRED = ["name", "identity", "person", "label", "id", "target"]


def load_any(dataset):
    """Return (iterable_examples, features) — streaming if possible, else from disk."""
    from datasets import load_dataset
    try:
        from datasets import get_dataset_split_names
        splits = get_dataset_split_names(dataset)
        split = "train" if "train" in splits else splits[0]
        ds = load_dataset(dataset, split=split, streaming=True)
        print(f"  streaming {dataset} [{split}] ...")
        return ds, ds.features
    except Exception as e1:
        print(f"  streaming failed ({type(e1).__name__}); snapshot_download + load_from_disk ...")
        from huggingface_hub import snapshot_download
        from datasets import load_from_disk, concatenate_datasets, DatasetDict
        local = snapshot_download(dataset, repo_type="dataset")
        obj = load_from_disk(local)
        if isinstance(obj, DatasetDict):
            ds = concatenate_datasets([obj[s] for s in obj])
            print(f"  loaded from disk, merged splits: {list(obj)} -> {len(ds)} rows")
        else:
            ds = obj
            print(f"  loaded from disk: {len(ds)} rows")
        return ds, ds.features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="amitpuri/bollywood-celebs")
    ap.add_argument("--out", default="data/lfw")
    ap.add_argument("--pairs", default="data/pairs.txt")
    ap.add_argument("--max-images", type=int, default=8000)
    ap.add_argument("--genuine", type=int, default=500)
    ap.add_argument("--impostor", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from PIL import Image as PILImage
    except ImportError:
        raise SystemExit("pip install datasets huggingface_hub pillow")

    ds, feats = load_any(args.dataset)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)          # don't mix with a previous (e.g. LFW) build
    out.mkdir(parents=True)

    counts = defaultdict(int)
    img_key = label_key = None
    total = 0

    def to_name(v):
        try:
            return feats[label_key].int2str(v)
        except Exception:
            return str(v)

    for ex in ds:
        if img_key is None:
            img_key = next((k for k, v in ex.items() if isinstance(v, PILImage.Image)), None)
            scal = [k for k, v in ex.items() if k != img_key and isinstance(v, (int, str))]
            label_key = next((k for k in PREFERRED if k in scal), scal[0] if scal else None)
            if img_key is None or label_key is None:
                raise SystemExit(f"no image+identity columns found (cols: {list(ex)})")
            print(f"  image col='{img_key}'  identity col='{label_key}'")

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
        raise SystemExit("no identity has >=2 images; raise --max-images or change --dataset")

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

    lines = [str(len(genuine))]
    lines += [f"{n}\t{i}\t{j}" for n, i, j in genuine]
    lines += [f"{a}\t{i}\t{b}\t{j}" for a, i, b, j in impostor]
    Path(args.pairs).write_text("\n".join(lines) + "\n")

    print(f"  saved {total} images across {len(counts)} identities -> {out}")
    print(f"  pairs: {len(genuine)} genuine + {len(impostor)} impostor -> {args.pairs}")
    print(f"\n  next (no --align for low-res pre-cropped faces):")
    print(f"  python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops "
          f"--pairs {args.pairs} --data {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
