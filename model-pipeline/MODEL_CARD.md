# Face Recognition Model Card — EdgeFace Sentinel recognizer

Lightweight, offline face-embedding model for the Hackathon 7.0 brief
(secure offline face recognition + liveness on mid-range Android/iOS, React Native).

## TL;DR

| Variant | File | Size | Accuracy (LFW) | EER | Latency¹ | Use |
|---|---|---|---|---|---|---|
| **int8 static** ✅ | `weights/w600k_mbf_int8_static.onnx` | **3.68 MB** | **99.7%** | 0.006 | 28.6 ms | **Primary deployable** — real integer kernels, fast on ARM/NNAPI |
| int8 dynamic | `weights/w600k_mbf_int8_dynamic.onnx` | 3.52 MB | 99.7% | 0.006 | 164 ms¹ | Fallback — simplest, weights-only, no calibration |
| fp32 reference | `weights/w600k_mbf.onnx` | 13.62 MB | 99.7% | 0.006 | 13.5 ms | Accuracy ground truth |

¹ Desktop x86 CPU (onnxruntime, single image). **Relative** only — not phone numbers.
Dynamic is slow on x86 (ConvInteger dequant overhead); static uses integer kernels.
All variants clear the brief's **<1 s** budget by a wide margin even at a 20–30× mobile slowdown.

**Every variant beats every hard requirement: ≤20 MB (3.68 MB), >95% (99.7%), <1 s.**

## Model

- **Architecture:** MobileFaceNet (InsightFace `buffalo_sc` recognizer, `w600k_mbf`), ArcFace-trained.
- **Input:** 1×3×112×112, **RGB**, NCHW, normalized `(x − 127.5) / 127.5` → [−1, 1].
- **Output:** 512-d embedding; L2-normalize, then compare with **cosine similarity**.
- **Decision threshold (this eval):** ~0.22–0.26 best-accuracy; use a tuned operating
  point per the FAR target (TAR@FAR=1e-3 = 0.994 at thr ≈ 0.19–0.20).
- **Face alignment is mandatory.** The recognizer expects a 5-point ArcFace-aligned
  crop. On-device: detect + 5 landmarks (BlazeFace / ML Kit / MediaPipe) → similarity
  warp to the ArcFace template (`align.py: norm_crop`) → 112×112. Feeding unaligned
  crops costs large accuracy.

## Evaluation

- **Set:** LFW, 1000 pairs (500 genuine / 500 impostor), pre-aligned to the ArcFace
  template (`data/lfw_aligned/`, 1602 unique images, 8 undetected → skipped).
- **Metric:** cosine of L2-normalized embeddings; best-accuracy threshold, EER, TAR@FAR.
- Numbers are on a public-faces benchmark. The brief additionally requires **Indian-
  demographic** robustness and **harsh outdoor lighting** — see *Limitations*.

## The quantization finding (important)

Naive int8 export via **onnx2tf collapses this model to ~50% (random)** — every face
maps to a near-identical embedding (genuine ≈ impostor cosine). This was reproduced
across *all* onnx2tf variants (`dynamic_range`, `integer`, `full_integer`, `int16x8`),
see `out/quant_sweep_results.json`. Root cause is **onnx2tf's quantizer** (per-tensor
weight scaling that zeroes MobileFaceNet's depthwise channels), **not** the data and
**not** int8 itself:

- Regenerating *correctly aligned* calibration crops did **not** fix it.
- Even weights-only `dynamic_range` (float activations) collapsed → not an activation issue.
- fp32 via onnx2tf is fine (99.7%) → graph topology is correct; only quant is broken.

**Fix:** quantize the ONNX with **onnxruntime per-*channel* int8** instead. This is
lossless here (99.7% preserved at 3.5–3.7 MB). Per-channel weight scaling is the key —
it preserves the depthwise layers that per-tensor scaling destroys.

## Reproduce

```bash
# venv has: onnxruntime, insightface, onnx, ai_edge_litert, tensorflow, onnx2tf
PY=../.gitignore/ve/bin/python

# 1) align eval set once + build aligned calibration crops
$PY prep_aligned.py

# 2) produce + benchmark the deployable int8 models (dynamic + static) vs fp32
$PY finalize_int8.py            # -> weights/*_int8_*.onnx, out/int8_final_results.json

# (evidence) show every onnx2tf int8 variant collapses:
$PY quant_sweep.py              # -> out/quant_sweep_results.json
# (isolation) onnxruntime per-channel int8 keeps 99.7%:
$PY ort_quant_test.py
```

## Deployment (next phase — not yet wired)

- Target: **onnxruntime-react-native** (MIT, single ONNX artifact for Android + iOS,
  runs the exact validated model). Alternative: convert to TFLite via the *native*
  `tf.lite` per-channel converter (not onnx2tf) for `react-native-fast-tflite` +
  NNAPI/Core ML delegates.
- Full on-device path: camera frame → face detect + 5 landmarks → ArcFace align (112²)
  → this recognizer → L2-norm → cosine vs enrolled gallery → threshold.
- Liveness (blink/smile/turn) and SQLite gallery + AWS sync/purge already scaffolded in
  `react-native-app/` (currently simulated; to be wired to the real engine).

## Limitations / open items

- Accuracy measured on LFW (Western-skewed). The brief requires **Indian demographics +
  outdoor lighting**; fine-tune/validate on a representative set before claiming field
  numbers. The current model is a strong, honest baseline, not a demographic claim.
- Latency here is desktop x86 and **relative only**; measure on a real mid-range handset
  (the brief's bar is <1 s end-to-end including detect + liveness).
- Detector `det_500m.onnx` (SCRFD, 2.52 MB) is used for offline alignment; budget it in
  the on-device footprint (still well under 20 MB total with the 3.68 MB recognizer).
