"""
benchmark.py — REAL recognizer benchmark.

Every number here comes from running the actual .tflite model on real data.
There are no synthetic score distributions and no rng.normal anywhere.

  Latency   : timed interpreter.invoke() calls (warmup + N runs).
  Accuracy  : verification on a real pair list -> TAR@FAR, EER, accuracy.

Latency note: numbers here are for THIS machine's CPU (or whatever delegate
LiteRT picks). They are a valid *relative* comparison across precisions, but
they are NOT phone numbers. For device latency, run the model through the
React Native / Android harness on the target handset.

Examples
--------
python benchmark.py --tflite out/recognizer_fp16.tflite \
    --pairs data/lfw_pairs.txt --data data/lfw_aligned

# Sweep every precision produced by `export.py --precision all`
python benchmark.py --sweep out/recognizer --pairs data/lfw_pairs.txt --data data/lfw_aligned
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import cv2

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback


# ----------------------------------------------------------------------
# preprocessing (must match training/calibration)
# ----------------------------------------------------------------------

def load_crop(path, size=112, mean=127.5, std=127.5, to_rgb=True, aligner=None):
    """
    Returns NHWC float32. If aligner is None the image is assumed PRE-ALIGNED
    and simply resized (correct for aligned LFW etc.). If an aligner is given,
    the raw image is detected + warped to the ArcFace template first.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    if aligner is not None:
        crop = aligner.align(bgr, image_size=size)
        if crop is None:
            return None  # no face detected -> caller skips this image
        img = crop
    else:
        img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return ((img.astype(np.float32) - mean) / std)[np.newaxis, ...]


class TFLiteEmbedder:
    def __init__(self, tflite_path, num_threads=4):
        self.itp = Interpreter(model_path=str(tflite_path), num_threads=num_threads)
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        self.out = self.itp.get_output_details()[0]
        self.size = int(self.inp["shape"][1])

    def _set_input(self, x):
        # handle int8-IO models (full-integer with int8 interface)
        if self.inp["dtype"] == np.int8:
            scale, zero = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        else:
            x = x.astype(self.inp["dtype"])
        self.itp.set_tensor(self.inp["index"], x)

    def _get_output(self):
        y = self.itp.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y.astype(np.float32)

    def embed(self, x):
        self._set_input(x)
        self.itp.invoke()
        emb = self._get_output()[0]
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb


# ----------------------------------------------------------------------
# 1. latency
# ----------------------------------------------------------------------

def benchmark_latency(embedder, warmup=10, runs=100):
    x = np.random.rand(1, embedder.size, embedder.size, 3).astype(np.float32)
    for _ in range(warmup):
        embedder._set_input(x); embedder.itp.invoke()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        embedder._set_input(x); embedder.itp.invoke(); embedder._get_output()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {
        "mean_ms": round(float(ts.mean()), 2),
        "p50_ms": round(float(np.percentile(ts, 50)), 2),
        "p95_ms": round(float(np.percentile(ts, 95)), 2),
        "p99_ms": round(float(np.percentile(ts, 99)), 2),
        "runs": runs,
        "note": "desktop/CPU timing; not a phone number",
    }


# ----------------------------------------------------------------------
# 2. verification accuracy (real cosine scores)
# ----------------------------------------------------------------------

def read_lfw_pairs(pairs_file, data_dir, ext="jpg"):
    """
    Standard LFW pairs.txt format.
      genuine : <name> <i> <j>
      impostor: <name1> <i> <name2> <j>
    Returns list of (pathA, pathB, is_same).
    """
    def img(name, idx):
        return Path(data_dir) / name / f"{name}_{int(idx):04d}.{ext}"

    pairs = []
    for line in Path(pairs_file).read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            pairs.append((img(parts[0], parts[1]), img(parts[0], parts[2]), 1))
        elif len(parts) == 4:
            pairs.append((img(parts[0], parts[1]), img(parts[2], parts[3]), 0))
    return pairs


def tar_at_far(scores, labels, far_targets=(1e-2, 1e-3, 1e-4)):
    scores, labels = np.asarray(scores), np.asarray(labels)
    impostor = np.sort(scores[labels == 0])[::-1]
    genuine = scores[labels == 1]
    out = {}
    for far in far_targets:
        k = max(1, int(round(far * len(impostor))))
        thr = impostor[min(k - 1, len(impostor) - 1)]
        tar = float((genuine >= thr).mean())
        out[f"TAR@FAR={far:.0e}"] = round(tar, 4)
        out[f"thr@FAR={far:.0e}"] = round(float(thr), 4)
    return out


def equal_error_rate(scores, labels):
    """EER = the rate where FAR and FRR are closest to equal."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    imp = scores[labels == 0]
    gen = scores[labels == 1]
    best_gap, best_eer = 1.0, 0.5
    for t in np.unique(scores):
        far = float((imp >= t).mean())
        frr = float((gen < t).mean())
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer = gap, (far + frr) / 2.0
    return round(float(best_eer), 4)


def benchmark_accuracy(embedder, pairs, aligner=None):
    cache, scores, labels = {}, [], []
    def get(p):
        p = str(p)
        if p not in cache:
            cache[p] = load_crop(p, size=embedder.size, aligner=aligner)
        return cache[p]

    skipped = 0
    for a, b, same in pairs:
        if not (Path(a).exists() and Path(b).exists()):
            skipped += 1
            continue
        ca, cb = get(a), get(b)
        if ca is None or cb is None:   # no face detected during alignment
            skipped += 1
            continue
        ea, eb = embedder.embed(ca), embedder.embed(cb)
        scores.append(float(np.dot(ea, eb)))   # cosine (both unit-norm)
        labels.append(int(same))

    if not scores:
        return {"error": "no valid pairs — check --data layout, --pairs, or alignment"}

    scores, labels = np.array(scores), np.array(labels)
    # best-accuracy threshold
    thrs = np.unique(scores)
    accs = [( ((scores >= t) == labels).mean(), t) for t in thrs]
    best_acc, best_thr = max(accs, key=lambda z: z[0])

    res = {
        "n_pairs": int(len(scores)),
        "n_skipped": int(skipped),
        "n_genuine": int((labels == 1).sum()),
        "n_impostor": int((labels == 0).sum()),
        "best_accuracy": round(float(best_acc), 4),
        "best_threshold": round(float(best_thr), 4),
        "EER": equal_error_rate(scores, labels),
        "genuine_mean": round(float(scores[labels == 1].mean()), 4),
        "impostor_mean": round(float(scores[labels == 0].mean()), 4),
    }
    res.update(tar_at_far(scores, labels))
    return res


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def run_one(tflite_path, pairs, do_latency=True, aligner=None):
    size_mb = round(Path(tflite_path).stat().st_size / 1e6, 2)
    out = {"tflite": str(tflite_path), "size_mb": size_mb}
    try:
        emb = TFLiteEmbedder(tflite_path)
    except Exception as e:
        last = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
        out["error"] = last
        out["note"] = ("could not load on CPU — likely a true-fp16 graph needing a "
                       "GPU/NNAPI delegate; file size is valid, CPU runtime skipped")
        return out
    if do_latency:
        out["latency"] = benchmark_latency(emb)
    if pairs:
        out["accuracy"] = benchmark_accuracy(emb, pairs, aligner=aligner)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", help="single .tflite to benchmark")
    ap.add_argument("--sweep", help="prefix from export.py --precision all (benchmarks every variant)")
    ap.add_argument("--pairs", help="LFW-format pairs.txt")
    ap.add_argument("--data", help="aligned image root (data/<name>/<name>_NNNN.jpg)")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    pairs = None
    if args.pairs and args.data:
        pairs = read_lfw_pairs(args.pairs, args.data, ext=args.ext)
        print(f"[accuracy] {len(pairs)} pairs from {args.pairs}")

    results = {}
    if args.sweep:
        for path in sorted(glob.glob(f"{args.sweep}_*.tflite")):
            prec = Path(path).stem.replace(Path(args.sweep).name + "_", "")
            print(f"\n=== {prec} ===")
            results[prec] = run_one(path, pairs)
            print(json.dumps(results[prec], indent=2))
    elif args.tflite:
        results = run_one(args.tflite, pairs)
        print(json.dumps(results, indent=2))
    else:
        raise SystemExit("pass --tflite <file> or --sweep <prefix>")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
