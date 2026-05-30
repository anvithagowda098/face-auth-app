"""
export.py — PyTorch -> ONNX -> TF SavedModel -> TFLite at a chosen precision.

Path: torch.onnx.export -> onnx2tf (handles NCHW->NHWC) -> tf.lite converter
(via precision.py). onnx2tf is the most reliable PyTorch->TFLite route for
conv models; ai-edge-torch is a valid alternative (see README).

Examples
--------
# Single precision
python export.py --weights weights/mobilefacenet.pt --precision fp16 \
    --out out/recognizer_fp16.tflite

# Full ablation sweep (one tflite per deployable precision + size table)
python export.py --weights weights/mobilefacenet.pt --precision all \
    --calib data/calib_crops --out out/recognizer
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from precision import convert_saved_model, DEPLOYABLE, NEEDS_CALIBRATION, fmt_size
from calibration import representative_dataset_factory
from models.recognizer import build_recognizer


def torch_to_saved_model(model, input_shape, workdir: Path) -> Path:
    """Export torch model to ONNX, then to a TF SavedModel (NHWC) via onnx2tf."""
    import onnx2tf

    onnx_path = workdir / "model.onnx"
    saved_model_dir = workdir / "saved_model"

    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=13,
    )
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    return saved_model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="recognizer checkpoint (.pt/.pth). Omit to export random init.")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--precision", default="fp16",
                    help="one of fp32|fp16|int8_dynamic|int8_full|int16x8|all")
    ap.add_argument("--calib", help="folder of aligned face crops (needed for int8_full/int16x8)")
    ap.add_argument("--out", required=True, help="output .tflite path, or prefix when --precision all")
    ap.add_argument("--io-int8", action="store_true", help="int8_full only: int8 input/output")
    args = ap.parse_args()

    model = build_recognizer(
        embedding_dim=args.embedding_dim,
        weights=args.weights,
        normalize_output=True,   # emit unit-norm embeddings in-graph
    )
    input_shape = (1, 3, args.size, args.size)

    rep_ds = None
    targets = DEPLOYABLE if args.precision == "all" else [args.precision]
    if any(p in NEEDS_CALIBRATION for p in targets):
        if not args.calib:
            raise SystemExit("int8_full/int16x8 need --calib <folder of face crops>")
        rep_ds = representative_dataset_factory(args.calib, size=args.size)

    workdir = Path(tempfile.mkdtemp(prefix="export_"))
    try:
        saved_model_dir = torch_to_saved_model(model, input_shape, workdir)

        rows = []
        for prec in targets:
            if args.precision == "all":
                out_path = f"{args.out}_{prec}.tflite"
            else:
                out_path = args.out
            size = convert_saved_model(
                str(saved_model_dir), prec, out_path,
                representative_dataset=rep_ds if prec in NEEDS_CALIBRATION else None,
                io_int8=args.io_int8,
            )
            rows.append((prec, out_path, size))
            print(f"  {prec:<13} -> {out_path}  ({fmt_size(size)})")

        if len(rows) > 1:
            base = next((s for p, _, s in rows if p == "fp32"), rows[0][2])
            print("\n  precision     size        vs fp32")
            print("  " + "-" * 42)
            for prec, _, size in rows:
                ratio = base / size if size else 0
                print(f"  {prec:<13} {fmt_size(size):<11} {ratio:5.2f}x smaller")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
