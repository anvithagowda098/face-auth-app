"""
precision.py — uniform precision switch for the whole pipeline.

One enum drives every model's export, so detector / recognizer / liveness all
quantise the same way and the ablation is a single flag.

Deployable precisions (real TFLite/LiteRT recipes):
    fp32          baseline, no quantisation
    fp16          weights fp16, ~2x smaller, GPU-delegate native, ~0 acc loss
    int8_dynamic  weights int8 / activations float, ~4x smaller, no calib set
    int8_full     weights+activations int8, needs a real calibration set,
                  best for NNAPI / Hexagon / EdgeTPU
    int16x8       int16 activations + int8 weights, higher acc than pure int8

NOT deployable on this target:
    fp8           TFLite/LiteRT has no fp8 (e4m3/e5m2) kernel on CPU, GPU
                  delegate, or NNAPI. `simulate_fp8_accuracy()` below gives an
                  ACCURACY-ONLY number via PyTorch fake-quant for the ablation
                  table; it produces no deployable artifact and no size/latency
                  benefit. Treat it as a research data point, not a build target.
"""

from pathlib import Path
from enum import Enum

DEPLOYABLE = ["fp32", "fp16", "int8_dynamic", "int8_full", "int16x8"]
NEEDS_CALIBRATION = {"int8_full", "int16x8"}


def convert_saved_model(saved_model_dir: str,
                        precision: str,
                        out_path: str,
                        representative_dataset=None,
                        io_int8: bool = False) -> int:
    """
    Convert a TF SavedModel to a .tflite at the requested precision.
    Returns the file size in bytes.

    representative_dataset: required for int8_full / int16x8. A zero-arg
    callable that yields lists like [np.ndarray(shape=(1,H,W,C), float32)].
    io_int8: for int8_full, expose int8 input/output (max NNAPI accel) instead
    of the default float32 interface (simpler; embeds stay float for cosine).
    """
    import tensorflow as tf

    if precision not in DEPLOYABLE:
        raise ValueError(f"unknown precision '{precision}', expected one of {DEPLOYABLE}")
    if precision in NEEDS_CALIBRATION and representative_dataset is None:
        raise ValueError(f"precision '{precision}' requires a representative_dataset")

    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if precision == "fp32":
        pass

    elif precision == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]

    elif precision == "int8_dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

    elif precision == "int8_full":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if io_int8:
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        # else: default float32 IO, int8 internals

    elif precision == "int16x8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]

    buf = conv.convert()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(buf)
    return len(buf)


def simulate_fp8_accuracy(torch_model, dtype="e4m3"):
    """
    ACCURACY-ONLY fp8 simulation (no deployable artifact).

    Returns a *copy* of the model with conv/linear weights fake-quantised to
    fp8 via per-tensor scaling -> cast to torch.float8 -> cast back. Run your
    verification benchmark on this copy to fill an fp8 row in the ablation
    table. Requires PyTorch >= 2.1 (torch.float8_e4m3fn / e5m2).
    """
    import copy
    import torch
    import torch.nn as nn

    fp8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[dtype]
    amax = {"e4m3": 448.0, "e5m2": 57344.0}[dtype]

    def fake_quant(t):
        scale = amax / t.abs().max().clamp(min=1e-8)
        return (t * scale).to(fp8).to(torch.float32) / scale

    m = copy.deepcopy(torch_model)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.copy_(fake_quant(mod.weight.data))
    return m.eval()


def fmt_size(nbytes: int) -> str:
    return f"{nbytes / 1e6:.2f} MB"
