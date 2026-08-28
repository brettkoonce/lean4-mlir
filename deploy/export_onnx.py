#!/usr/bin/env python3
"""Lean checkpoint -> ONNX, for TensorRT on a Jetson Orin.

## Why this exists

IREE compiles this graph for sm_87 fine, and on an Orin it runs at ~0.5 fps.
That is a compiler gap, not a hardware one: IREE's CUDA backend generates its own
convolution kernels, where TensorRT dispatches to cuDNN and tensor cores and does
fp16 natively. Same reason the repo's own notes say "gfx1100 is MIOpen-conv-weak"
overstates it — IREE is simply worse at convolutions.

## Why via PyTorch rather than StableHLO

TensorRT ingests ONNX. StableHLO -> ONNX is not a trodden path, but this repo
already keeps a faithful PyTorch replica of this exact detector
(`demos/visdrone/bespoke/`), together with a loader for Lean's flat checkpoints
and a differ that proves the two agree. So the shortest correct route is to load
the trained Lean weights into the replica and export that.

⚠ That makes the replica load-bearing for deployment, not just for validation.
`--verify` exists for exactly that reason: it re-scores the exported graph
against a Lean logits dump and refuses on disagreement. Run it. A silent
architecture drift between the replica and the Lean spec would ship a detector
that is subtly not the one that was measured.

## Usage

    # anywhere torch is available (NOT this box — no torch here)
    python3 export_onnx.py \
        --ckpt ../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_params.bin \
        --bn   ../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_bn_stats.bin \
        --out  build/detector.onnx \
        --verify ../runs/fpn_ctrl12_final/logits.bin

    # then on the Orin
    trtexec --onnx=detector.onnx --saveEngine=detector.plan --fp16
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "demos" / "visdrone"))

IMG_PX = 448
NTOT = 185220


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Lean *_params.bin")
    ap.add_argument("--bn", required=True, help="Lean *_bn_stats.bin")
    ap.add_argument("--out", default="build/detector.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1,
                    help="1 for a camera; the training graph is 8")
    ap.add_argument("--verify", default=None,
                    help="Lean logits.bin to check the export against")
    ap.add_argument("--val-bin", default=str(REPO / "data/visdrone_fpn/val.bin"),
                    help="source of the images --verify compares on")
    ap.add_argument("--verify-n", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("torch required — run this where torch exists "
                         "(the Orin, or any box with it installed)")

    from bespoke.model import FpnDetector
    from bespoke.lean_ckpt import load_lean_params
    from bespoke.bn_stats import load_bn_stats

    # These four arguments are NOT defaults — they are what `bespoke/diff_lean.py`
    # uses to make the replica agree with the Lean spec. `pool="lean"` in
    # particular matches `.maxPool 2 2` rather than torchvision's padded 3x3 stem.
    model = FpnDetector(backbone="r34", tower=0, norm=None,
                        pretrained=False, pool="lean")
    load_lean_params(model, args.ckpt)
    load_bn_stats(model, args.bn)
    model.eval()

    dummy = torch.zeros(args.batch, 3, IMG_PX, IMG_PX)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=None,          # fixed batch: TensorRT prefers a static shape
    )
    print(f"wrote {args.out}  (batch {args.batch}, opset {args.opset})")

    if not args.verify:
        print("⚠ exported WITHOUT --verify. The replica is load-bearing here; "
              "run with --verify before trusting this on device.")
        return

    # ---- the gate: does the exported graph reproduce the Lean logits? ----
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("--verify needs onnxruntime (pip install onnxruntime)")

    lean = np.fromfile(args.verify, dtype=np.float32).reshape(-1, NTOT)
    n = min(args.verify_n, lean.shape[0])
    rec = 3 * IMG_PX * IMG_PX + NTOT * 4
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    istd = (1.0 / np.array([0.229, 0.224, 0.225], np.float32)).reshape(3, 1, 1)

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    worst = 0.0
    with open(args.val_bin, "rb") as f:
        for i in range(n):
            f.seek(4 + i * rec)
            img = np.frombuffer(f.read(3 * IMG_PX * IMG_PX), dtype=np.uint8)
            x = img.reshape(3, IMG_PX, IMG_PX).astype(np.float32) / 255.0
            x = ((x - mean) * istd)[None]
            got = sess.run(None, {"image": x})[0].reshape(-1)[:NTOT]
            d = float(np.abs(got - lean[i]).max())
            worst = max(worst, d)
            print(f"  record {i}: max abs diff {d:.3e}")
    print(f"worst {worst:.3e} over {n} records (tol {args.tol})")
    if worst > args.tol:
        raise SystemExit(
            "⛔ EXPORT DOES NOT MATCH THE LEAN MODEL. Do not deploy this. The "
            "replica has drifted from the spec, or the checkpoint/pool/pad "
            "settings are wrong.")
    print("✅ export matches the Lean model")


if __name__ == "__main__":
    main()
