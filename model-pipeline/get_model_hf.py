#!/usr/bin/env python3
"""
get_model_hf.py — fetch pretrained insightface ONNX models from HuggingFace.

  weights/w600k_mbf.onnx   MobileFaceNet recognizer (ArcFace, ~99.7% aligned LFW)
  weights/det_500m.onnx    SCRFD face detector (for --align insightface)

Setup: pip install huggingface_hub
Run:   python get_model_hf.py
Then:  python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops \
           --pairs data/pairs.txt --data data/lfw --align insightface
"""

import os
import shutil
import sys
from pathlib import Path

# filename -> ordered (repo_id, path_in_repo) candidates
TARGETS = {
    "w600k_mbf.onnx": [
        ("WePrompt/buffalo_sc", "w600k_mbf.onnx"),
        ("deepghs/insightface", "buffalo_s/w600k_mbf.onnx"),
    ],
    "det_500m.onnx": [
        ("WePrompt/buffalo_sc", "det_500m.onnx"),
        ("deepghs/insightface", "buffalo_s/det_500m.onnx"),
    ],
}


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub   # then re-run")

    Path("weights").mkdir(exist_ok=True)
    for out_name, sources in TARGETS.items():
        dst = Path("weights") / out_name
        if dst.exists():
            print(f"  [skip] {dst} already present")
            continue
        for repo_id, filename in sources:
            try:
                print(f"  {out_name}: trying {repo_id} :: {filename} ...")
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                shutil.copy(path, dst)
                print(f"     saved -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
                break
            except Exception as e:
                print(f"     failed: {e}")
        else:
            raise SystemExit(f"could not fetch {out_name} from any source")

    print("\n  next: python run.py --onnx weights/w600k_mbf.onnx --calib data/calib_crops \\")
    print("            --pairs data/pairs.txt --data data/lfw --align insightface")
    sys.stdout.flush()
    os._exit(0)   # skip HF/xet thread teardown (avoids the finalize crash)


if __name__ == "__main__":
    main()
