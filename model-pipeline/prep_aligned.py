#!/usr/bin/env python3
"""
prep_aligned.py — align the LFW eval set ONCE to the ArcFace template, and
build a CORRECT int8 calibration set from those same aligned crops.

Why: int8_full/int16x8 collapsed to ~50% because calibration used raw,
unaligned 250x250 LFW images while inference uses tightly-cropped, ArcFace-
aligned 112x112 faces. The quantiser set activation ranges from the wrong
distribution. Calibration MUST match the inference distribution.

Outputs:
  data/lfw_aligned/<name>/<name>_NNNN.jpg   (every image used by pairs.txt)
  data/calib_aligned/*.jpg                   (~200 aligned crops for int8)

Then benchmark WITHOUT --align (data is already aligned -> fast, deterministic).
"""

import argparse
import shutil
from pathlib import Path

import cv2

from align import build_aligner
from benchmark import read_lfw_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.txt")
    ap.add_argument("--src", default="data/lfw")
    ap.add_argument("--dst", default="data/lfw_aligned")
    ap.add_argument("--calib", default="data/calib_aligned")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--size", type=int, default=112)
    args = ap.parse_args()

    aligner = build_aligner("insightface")
    pairs = read_lfw_pairs(args.pairs, args.src, ext=args.ext)

    # unique image paths referenced by the pair list
    uniq = []
    seen = set()
    for a, b, _ in pairs:
        for p in (a, b):
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                uniq.append(p)
    print(f"[prep] {len(pairs)} pairs -> {len(uniq)} unique images")

    dst_root = Path(args.dst)
    calib_dir = Path(args.calib)
    if calib_dir.exists():
        shutil.rmtree(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)

    aligned, skipped, calib_saved = 0, 0, 0
    for p in uniq:
        if not Path(p).exists():
            skipped += 1
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            skipped += 1
            continue
        crop = aligner.align(bgr, image_size=args.size)
        if crop is None:
            skipped += 1
            continue
        # mirror the original <name>/<name>_NNNN.jpg layout under dst
        rel = Path(p).relative_to(args.src)
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop)
        aligned += 1
        if calib_saved < args.n_calib:
            cv2.imwrite(str(calib_dir / f"calib_{calib_saved:04d}.jpg"), crop)
            calib_saved += 1

    print(f"[prep] aligned {aligned}, skipped {skipped} (no face detected)")
    print(f"[prep] aligned eval -> {dst_root}")
    print(f"[prep] calibration  -> {calib_dir} ({calib_saved} crops)")


if __name__ == "__main__":
    main()
