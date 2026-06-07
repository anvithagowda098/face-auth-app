#!/usr/bin/env python3
"""
validate_indian.py — demographic validation of the DEPLOYABLE int8 artifact on
Indian faces (amitpuri/bollywood-celebs), reusing the ORT benchmark harness.

The hackathon requires Indian-demographic performance; all prior numbers are on
Western-skewed LFW. This scores the chosen static-int8 ONNX (and fp32 + dynamic
int8 for reference) on bollywood-celebs pairs built DIRECTLY from the downloaded
image-folder layout (the HF datasets API is broken for this repo).

NOTE: bollywood-celebs are 64x64 pre-localized crops -> NO alignment (load_nchw
just resizes 64->112). Numbers are a RELATIVE demographic signal, NOT comparable
to the aligned-LFW 99.7%. What matters: does int8_static keep genuine/impostor
separation on Indian faces, and does it track fp32?
"""

import json
import random
from pathlib import Path
from glob import glob

from ort_quant_test import OrtEmbedder, evaluate

# resolve the snapshot dir without hardcoding the revision hash
_BASE = Path.home() / ".cache/huggingface/hub/datasets--amitpuri--bollywood-celebs/snapshots"
_HITS = glob(str(_BASE / "*" / "raw/image_folders/*/Bollywood_celeb_face_localized"))
IMG_ROOT = Path(_HITS[0]) if _HITS else None

N_GENUINE = 500
N_IMPOSTOR = 500
SEED = 42

VARIANTS = {
    "fp32": "weights/w600k_mbf.onnx",
    "int8_static": "weights/w600k_mbf_int8_static.onnx",
    "int8_dynamic": "weights/w600k_mbf_int8_dynamic.onnx",
}


def build_pairs():
    ids = {}
    for d in sorted(p for p in IMG_ROOT.iterdir() if p.is_dir()):
        imgs = sorted(str(f) for f in d.glob("*.jpg"))
        if len(imgs) >= 2:
            ids[d.name] = imgs
    names = list(ids)
    rng = random.Random(SEED)

    genuine = []
    while len(genuine) < N_GENUINE:
        n = rng.choice(names)
        a, b = rng.sample(ids[n], 2)
        genuine.append((a, b, 1))

    impostor = []
    while len(impostor) < N_IMPOSTOR:
        x, y = rng.sample(names, 2)
        impostor.append((rng.choice(ids[x]), rng.choice(ids[y]), 0))

    print(f"[validate] {len(names)} identities, "
          f"{N_GENUINE} genuine + {N_IMPOSTOR} impostor pairs "
          f"(64->112 resize, NO alignment)\n")
    return genuine + impostor


def main():
    if IMG_ROOT is None or not IMG_ROOT.exists():
        raise SystemExit("bollywood image-folder not found in HF cache; "
                         "run the snapshot_download first")
    pairs = build_pairs()

    results = {}
    for name, path in VARIANTS.items():
        if not Path(path).exists():
            print(f"  {name:<14} SKIP (missing {path})")
            continue
        acc = evaluate(OrtEmbedder(path), pairs)
        size_mb = round(Path(path).stat().st_size / 1e6, 2)
        results[name] = {"path": path, "size_mb": size_mb, "accuracy": acc}
        print(f"  {name:<14} {size_mb:>6} MB  acc {acc['best_accuracy']}  "
              f"EER {acc['EER']}  TAR@1e-3 {acc['TAR@FAR=1e-03']}  "
              f"gen {acc['genuine_mean']}  imp {acc['impostor_mean']}  "
              f"(n={acc['n_pairs']}, skip={acc['n_skipped']})")

    Path("out/indian_validation_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> out/indian_validation_results.json")


if __name__ == "__main__":
    main()
