"""
export.py — (torch model OR prebuilt .onnx) -> TFLite, via onnx2tf direct outputs.

Two modes:
  --weights ...   build our MobileFaceNet, export it (architecture/size only;
                  random weights unless you trained it).
  --onnx PATH     feed a prebuilt ONNX straight through (e.g. insightface
                  w600k_mbf.onnx — a real pretrained MobileFaceNet). No torch
                  model, no key-remap: the pretrained graph is converted as-is.

onnx2tf ('flatbuffer_direct') writes the .tflite files directly:
  no calibration: float32 + float16
  with --calib:   also dynamic_range_quant + full_integer_quant
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

PRECISION_TAG = {
    "fp32": "float32",
    "fp16": "float16",
    "int8_dynamic": "dynamic_range_quant",
    "int8_full": "full_integer_quant",
}
NEEDS_CALIB = {"int8_dynamic", "int8_full"}

# insightface recognition preprocessing: RGB, (x-127.5)/127.5 -> [-1, 1]
MEAN, STD = 127.5, 127.5


def _build_calib_npy(calib_dir, size, workdir, max_samples=200):
    """Stack real crops into (N,H,W,C) float32, pre-normalised to [-1,1]."""
    import cv2
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(calib_dir).rglob("*") if p.suffix.lower() in exts][:max_samples]
    if not paths:
        raise FileNotFoundError(f"no images under {calib_dir}")
    arr = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        img = cv2.resize(bgr, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        arr.append((img - MEAN) / STD)
    data = np.stack(arr).astype(np.float32)
    npy = workdir / "calib.npy"
    np.save(npy, data)
    print(f"  [calib] {data.shape[0]} crops -> {npy.name}")
    return npy


def _onnx_input_name(onnx_path):
    """Read the graph's first input name (needed for int8 calibration mapping)."""
    import onnx
    m = onnx.load(str(onnx_path))
    return m.graph.input[0].name


def export_precisions(precisions, out_dir, *, model=None, input_shape=None,
                      onnx_path=None, calib_dir=None, size=112):
    """
    Export to .tflite at each precision. Provide EITHER model+input_shape OR onnx_path.
    Returns {precision: (path, size_bytes)}.
    """
    import onnx2tf

    if (model is None) == (onnx_path is None):
        raise ValueError("provide exactly one of model=... or onnx_path=...")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want_calib = any(p in NEEDS_CALIB for p in precisions)
    if want_calib and not calib_dir:
        raise SystemExit("int8_dynamic/int8_full need --calib <folder of crops>")

    workdir = Path(tempfile.mkdtemp(prefix="export_"))
    try:
        if onnx_path is None:
            import torch
            onnx_path = workdir / "model.onnx"
            torch.onnx.export(
                model.eval(), torch.randn(*input_shape), str(onnx_path),
                input_names=["input"], output_names=["embedding"], opset_version=18,
            )
        in_name = _onnx_input_name(onnx_path)
        tfl_dir = workdir / "tflite"

        kwargs = dict(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tfl_dir),
            copy_onnx_input_output_names_to_tflite=True,
        )
        if want_calib:
            npy = _build_calib_npy(calib_dir, size, workdir)
            kwargs["output_integer_quantized_tflite"] = True
            # data already normalised -> mean=0, std=1 here. Input op name auto-detected.
            kwargs["custom_input_op_name_np_data_path"] = [[in_name, str(npy), 0.0, 1.0]]

        onnx2tf.convert(**kwargs)

        produced = sorted(tfl_dir.glob("*.tflite"))
        print("  onnx2tf produced:")
        for f in produced:
            print(f"      {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")

        result = {}
        for prec in precisions:
            tag = PRECISION_TAG[prec]
            match = next((f for f in produced if tag in f.name), None)
            if match is None:
                print(f"  [skip] {prec}: no '{tag}' file emitted on this onnx2tf version")
                continue
            dst = out_dir / f"recognizer_{prec}.tflite"
            shutil.copy(match, dst)
            result[prec] = (str(dst), dst.stat().st_size)
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", help="prebuilt ONNX (e.g. weights/w600k_mbf.onnx)")
    ap.add_argument("--weights", help="our-model checkpoint (ignored if --onnx given)")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--precision", default="fp16", help="fp32|fp16|int8_dynamic|int8_full|all")
    ap.add_argument("--calib", help="folder of crops (for int8_*)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    precisions = list(PRECISION_TAG) if args.precision == "all" else [args.precision]
    if args.onnx:
        res = export_precisions(precisions, args.out, onnx_path=args.onnx,
                                calib_dir=args.calib, size=args.size)
    else:
        from models.recognizer import build_recognizer
        model = build_recognizer(embedding_dim=args.embedding_dim,
                                 weights=args.weights, normalize_output=True)
        res = export_precisions(precisions, args.out, model=model,
                                input_shape=(1, 3, args.size, args.size),
                                calib_dir=args.calib, size=args.size)
    for prec, (path, sz) in res.items():
        print(f"  {prec:<13} -> {path}  ({sz / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
