# EdgeFace Sentinel — recognizer pipeline (real build)

This replaces the mock pipeline. Every number comes from running the actual
model on real data. The recognizer is the doc's MobileFaceNet rebuilt with a
GDConv head (~1.0M params, not the 7.3M FC-bloated version).

## What's here (this batch)

```
models/recognizer.py   MobileFaceNet (correct GDConv head) + ArcFaceHead
precision.py           uniform precision switch + TFLite recipes + simulated fp8
calibration.py         representative dataset from real face crops (for int8)
export.py              torch -> ONNX -> TF SavedModel -> TFLite at any precision
benchmark.py           real latency + verification (TAR@FAR, EER) — no rng
requirements.txt
```

Coming next (same `precision.py` drives all of them): BlazeFace detector +
5-point alignment, MiniFASNet liveness, end-to-end `pipeline.py`, RN module.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Weights (you supply these — needs internet)

A model with random weights cannot produce real accuracy. Use one of:

- **MobileFaceNet (recommended, clean TFLite path).** Pure-conv, quantises
  cleanly. Grab a pretrained ArcFace MobileFaceNet checkpoint (e.g. from an
  insightface-compatible repo) and adapt the `state_dict` keys to
  `models/recognizer.py`. Confirm the preprocessing (RGB, `(x-127.5)/128`) matches.
- **EdgeFace (higher accuracy).** From the official `edgeface` repo / its
  HuggingFace weights. Note: its hybrid transformer (STDA attention, LayerNorm)
  may need TF fallback ops in TFLite and won't quantise as cleanly to int8 —
  good for the fp16 row, trickier for full-int8.
- **Train your own** with `ArcFaceHead` on MS1M-V3 (multi-GPU, days). Only worth
  it if you specifically need the "custom distilled model" story.

> The architecture in `recognizer.py` is correct, but no two pretrained repos
> name their layers identically. `build_recognizer(..., weights=...)` loads with
> `strict=False` and prints missing/unexpected keys — if that count is large,
> the checkpoint's layer names need remapping before the weights actually apply.

## Evaluation data

Use an aligned verification set in LFW layout:

```
data/lfw_aligned/<name>/<name>_0001.jpg
data/lfw_pairs.txt        # standard 3-col (genuine) / 4-col (impostor) format
data/calib_crops/*.jpg    # ~300 aligned crops for int8 calibration
```

Aligned LFW (already cropped to 112×112) and the official `pairs.txt` are the
standard choice. Real accuracy depends on alignment matching training.

## Run the precision ablation

```bash
# one tflite per deployable precision + a size table
python export.py --weights weights/mobilefacenet.pt --precision all \
    --calib data/calib_crops --out out/recognizer

# benchmark every variant: real latency + real TAR@FAR / EER
python benchmark.py --sweep out/recognizer \
    --pairs data/lfw_pairs.txt --data data/lfw_aligned
```

`benchmark_results.json` will have, per precision: file size, measured latency
(mean/p50/p95/p99), and accuracy (best-acc + threshold, EER, TAR@FAR=1e-2/1e-3/1e-4,
genuine/impostor score means). These are the numbers that go in the deck.

## On fp8 (important)

TFLite/LiteRT has **no fp8** kernel on CPU, GPU delegate, or NNAPI — you cannot
deploy an fp8 model on the target phones. The deployable precisions are
`fp32 / fp16 / int8_dynamic / int8_full / int16x8`. `precision.simulate_fp8_accuracy()`
fake-quantises weights to e4m3 in PyTorch so you can report an **accuracy-only**
fp8 row in the ablation; it yields no `.tflite` and no size/latency benefit.
Lead with fp16 (headline) and int8_full (smallest deployable).

## On latency numbers

`benchmark.py` timings are this machine's CPU. They are a fair *relative*
comparison across precisions, but they are **not** phone numbers. The sub-1s /
~600ms device claim must be measured on the actual handset via the Android/RN
harness (next batch) — don't put desktop timings in the deck as device latency.
