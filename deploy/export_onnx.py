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
(`demos/visdrone/bespoke/`) together with a loader for Lean's flat checkpoints.
So the shortest correct route is to load the trained Lean weights into the
replica and export that.

⚠⚠ That makes the replica load-bearing for deployment, so this script is only
as good as its gate. The first Orin run produced 279 detections and top score
0.7656 on the reference frame where the Lean stack produces 238 and 0.7109 —
a different function, shipped under the same name.

**Root cause, settled 2026-08-28: this script built the replica without
`pad="lean"`.** Lean's convolutions use `MlirCodegen.samePad`, TF-style
ASYMMETRIC SAME (the odd pixel goes on the high side); torchvision pads
symmetrically. Output shapes, parameter counts and `iree-compile` are all
identical either way, so nothing structural could catch it, but the sampling
grid shifts half an output pixel at every stride-2 convolution and the shift
compounds through the 3 / 4 / 5 downsampling stages feeding C3 / C4 / C5. That
is why the objectness correlation fell 0.90 / 0.80 / 0.62 with tap depth, and
why the BN/eps/pool/permutation probes that "ruled out" everything else came
back empty — every one of them built the replica the same wrong way.

    pool=lean pad=lean          max|Δ| 8.0e-03   obj r 1.0000/1.0000/1.0000
    pool=lean pad=torchvision   max|Δ| 1.6e+01   obj r 0.9031/0.8022/0.6219

With `pad="lean"` the replica decodes the same 238 detections at top 0.7108
against the Lean stack's 238 at 0.7109.

**Run `--verify-frame` anyway, every time.** It is self-contained, needs nothing
from the training box, and is the only thing standing between a plausible-looking
export and shipping a different model twice.

## Usage

    # Anywhere torch is available. ⚠ On the training box that means a THROWAWAY
    # CPU-only venv, never the pinned .venv — installing torch there pulls its own
    # CUDA wheels over the pinned cuDNN and kills every JAX/XLA convolution:
    #   python3 -m venv /tmp/venv-torch && /tmp/venv-torch/bin/pip install \
    #       torch torchvision --index-url https://download.pytorch.org/whl/cpu
    #   /tmp/venv-torch/bin/pip install onnxruntime pillow
    python3 export_onnx.py \
        --ckpt ../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_params.bin \
        --bn   ../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_bn_stats.bin \
        --out  build/detector.onnx \
        --verify-frame

    # then on the Orin
    trtexec --onnx=detector.onnx --saveEngine=detector.plan --fp16
"""
import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "demos" / "visdrone"))

IMG_PX = 448
NTOT = 185220



# Per-scale layout of the flat output, in the codegen's concat order [P3|P4|P5].
FPN_GRIDS = (56, 28, 14)
A, SLOTS = 3, 15
# Correlation floor for the per-scale objectness channels. This is the statistic
# that separated the two regimes most sharply: the wrong-padding export scored
# 0.903 / 0.802 / 0.622 while a correct one scores 1.0000 at every scale. It is
# scale-free, so unlike an absolute tolerance it cannot be defeated by a model
# that happens to output small numbers.
MIN_OBJ_R = 0.999


def _per_scale_obj_r(a, b):
    """Objectness-channel correlation at P3 / P4 / P5.

    Objectness because it is what detections rank on, and per-scale because a
    geometric misalignment compounds with depth — C3/C4/C5 sit behind 3/4/5
    stride-2 stages, so a padding or resampling difference shows up as a
    correlation that FALLS from P3 to P5 rather than as uniform noise.
    """
    out, off = [], 0
    for g in FPN_GRIDS:
        n = A * SLOTS * g * g
        oa = a[off:off + n].reshape(A, SLOTS, g, g)[:, 4].ravel()
        ob = b[off:off + n].reshape(A, SLOTS, g, g)[:, 4].ravel()
        out.append(float(np.corrcoef(oa, ob)[0, 1]))
        off += n
    return out


def verify_frame(onnx_path, tol, fold="none"):
    """Self-contained gate: testdata/frame.png vs testdata/frame_logits.bin.

    Both ship in the repo, so this needs nothing from the training box. The
    logits are the Lean stack's own output for that frame under the eval graph
    (BN in inference mode, running stats) — reproducibly val record 374 of an
    `infer` dump, byte for byte.

    ⚠ The tolerance is RELATIVE (max abs difference over max abs logit) and it
    is not a knob. Relative because logit magnitude varies by two orders of
    magnitude across records — this frame spans +-16, val record 40 spans
    -976 .. +1287 — so an absolute threshold is either flaky or vacuous
    depending on which record it was tuned on. Both endpoints are measured: a
    correct export sits at 5.0e-4 (max abs 8.0e-3; the reference dump runs
    XLA's TF32 convolutions, and forcing `NVIDIA_TF32_OVERRIDE=0` on the Lean
    side drops it to 2.6e-3, the Lean graph disagreeing with ITSELF by 7.8e-3
    across that switch), while the export that shipped a different function sat
    at ~1.0. There is no marginal case in between, so a failure here means a
    real structural difference and widening `--tol` only hides it.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("--verify-frame needs onnxruntime")
    from PIL import Image

    ref = np.fromfile(HERE / "testdata" / "frame_logits.bin", dtype=np.float32)
    if ref.size != NTOT:
        raise SystemExit(f"reference logits are {ref.size} floats, want {NTOT}")
    raw = np.asarray(Image.open(HERE / "testdata" / "frame.png").convert("RGB"))
    if fold == "u8":
        # exactly the bytes a camera path would hand over — no host arithmetic
        x = raw.astype(np.uint8)[None]
    else:
        img = raw.astype(np.float32).transpose(2, 0, 1) / 255.0
        if fold == "f32":
            x = img[None]                      # the graph normalizes
        else:
            mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
            istd = (1.0 / np.array([0.229, 0.224, 0.225], np.float32)).reshape(3, 1, 1)
            x = ((img - mean) * istd)[None]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x})[0].reshape(-1)[:NTOT]
    d = np.abs(got - ref)
    rel = float(d.max() / max(float(np.abs(ref).max()), 1e-6))
    r3, r4, r5 = _per_scale_obj_r(ref, got)
    print(f"  max rel diff {rel:.3e}  (max abs {d.max():.3e})  (tol {tol})")
    print(f"  ref  range {ref.min():+.3f} .. {ref.max():+.3f}")
    print(f"  onnx range {got.min():+.3f} .. {got.max():+.3f}")
    print(f"  objectness r  P3 {r3:.4f}  P4 {r4:.4f}  P5 {r5:.4f}"
          f"  (floor {MIN_OBJ_R})")
    worst_r = min(r3, r4, r5)
    if rel > tol or worst_r < MIN_OBJ_R:
        falling = r5 < r4 < r3
        raise SystemExit(
            "⛔ THE EXPORT DOES NOT MATCH THE LEAN MODEL.\n"
            "   Do not deploy, and do not widen the tolerance — report the "
            "numbers above.\n"
            + ("   The correlation FALLS from P3 to P5, which is the signature "
               "of a geometric\n   misalignment compounding through the "
               "backbone's stride-2 stages. Check\n   `pad=` (Lean is TF-style "
               "ASYMMETRIC SAME, torchvision is symmetric) and `pool=`\n"
               "   (Lean is `.maxPool 2 2`, torchvision is a padded 3x3) before "
               "anything else.\n"
               if falling else
               "   The error is spread evenly across scales, so it is NOT a "
               "geometric shift.\n   Check the BN running statistics and the "
               "checkpoint parameter order.\n"))
    print("✅ export matches the Lean model on the reference frame")


def _wrap_preprocess(model, mode):
    """Move the host-side preprocessing INTO the graph.

    Measured on an Orin Nano: preprocess was 11.9 ms against a 6.3 ms forward,
    and the PIL decode+resize is only 0.5 ms of it. The other 8.5 ms is numpy
    elementwise work — `transpose`, `/255`, `(x-mean)*istd` — each allocating a
    fresh 600 K-float array on a CPU that is much worse at this than the GPU
    already sitting idle behind it.

      mode="f32"  input stays [N,3,448,448] float32 in [0,1]; only the normalize
                  moves into the graph. Safe everywhere, saves the arithmetic.
      mode="u8"   input becomes [N,448,448,3] UINT8 — exactly what
                  `np.asarray(pil_image)` already returns, so the host does no
                  arithmetic and no transpose at all, and the host-to-device copy
                  drops 4x (602 KB against 2.4 MB). The permute, the /255 and the
                  normalize all run on the GPU.

    ⚠ `u8` needs a TensorRT that accepts a UINT8 network input (10.x does; older
    ones may not). If trtexec rejects it, fall back to `f32`, which still removes
    the 4.0 ms normalize and needs nothing special from the runtime.
    """
    import torch
    import torch.nn as nn

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    istd = 1.0 / torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    class Wrapped(nn.Module):
        def __init__(self, inner, mode):
            super().__init__()
            self.inner = inner
            self.mode = mode
            self.register_buffer("mean", mean)
            self.register_buffer("istd", istd)

        def forward(self, x):
            if self.mode == "u8":
                x = x.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            return self.inner((x - self.mean) * self.istd)

    return Wrapped(model, mode).eval()


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
    ap.add_argument("--fold-preprocess", choices=["none", "f32", "u8"],
                    default="none",
                    help="move normalization (and for u8, the /255 and the "
                         "HWC->CHW permute) into the graph. u8 takes a "
                         "[N,448,448,3] uint8 input — what np.asarray(pil) "
                         "already returns — so the host does no arithmetic and "
                         "the H2D copy drops 4x. See _wrap_preprocess.")
    ap.add_argument("--verify-frame", action="store_true",
                    help="self-contained gate: compare against testdata/frame.png "
                         "+ testdata/frame_logits.bin, which ship in the repo. "
                         "Needs no val.bin and no dump from the training box.")
    ap.add_argument("--val-bin", default=str(REPO / "data/visdrone_fpn/val.bin"),
                    help="source of the images --verify compares on")
    ap.add_argument("--verify-n", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="RELATIVE logit tolerance (max abs difference over max "
                         "abs logit). Both endpoints are measured and they are 3 "
                         "orders of magnitude apart: a correct export sits at "
                         "5.0e-4 (XLA runs the reference graph with TF32 convs — "
                         "turning TF32 off drops it to 1.6e-4), the wrong-padding "
                         "export sat at ~1.0. Widening this does not buy a "
                         "marginal case; there isn't one.")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("torch required — run this where torch exists "
                         "(the Orin, or any box with it installed)")

    from bespoke.model import FpnDetector
    from bespoke.lean_ckpt import load_lean_params
    from bespoke.bn_stats import load_bn_stats

    # NONE of these are defaults, and every one is load-bearing.
    #   pool="lean"  matches `.maxPool 2 2`, not torchvision's padded 3x3 stem.
    #   pad="lean"   matches `MlirCodegen.samePad`, TF-style ASYMMETRIC SAME, not
    #                torchvision's symmetric `padding=`. ⚠ THIS ONE was omitted
    #                here and in every probe that "ruled out" the other suspects,
    #                and it alone was the 16.5 mismatch: it shifts the sampling
    #                grid half an output pixel per stride-2 conv, compounding over
    #                the 3/4/5 downsamples above C3/C4/C5 — which is exactly why
    #                the objectness correlation fell 0.90/0.80/0.62 with depth.
    #                With it, the replica reproduces the Lean eval graph to 8e-3
    #                and decodes the same 238 detections at the same top score.
    model = FpnDetector(backbone="r34", tower=0, norm=None,
                        pretrained=False, pool="lean", pad="lean")
    load_lean_params(model, args.ckpt)
    load_bn_stats(model, args.bn)
    model.eval()

    if args.fold_preprocess != "none":
        model = _wrap_preprocess(model, args.fold_preprocess)
    if args.fold_preprocess == "u8":
        dummy = torch.zeros(args.batch, IMG_PX, IMG_PX, 3, dtype=torch.uint8)
    else:
        dummy = torch.zeros(args.batch, 3, IMG_PX, IMG_PX)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=None,          # fixed batch: TensorRT prefers a static shape
    )
    # ⚠ `--opset` is a REQUEST, not a guarantee: torch exports at its own opset
    # and then tries to down-convert, and that conversion can fail silently
    # (onnx has no Pad adapter down to 17, and the lean asymmetric padding
    # introduces a Pad before the optimizer folds it into the Conv `pads`
    # attribute). Read the number back out of the file rather than reprinting
    # the request — shipping an artifact whose properties differ from the ones
    # reported is the exact failure this script already made once.
    written = args.opset
    try:
        import onnx
        m = onnx.load(args.out)
        # torch may park the weights in a sibling `<name>.onnx.data` and leave a
        # 200 KB stub behind. That is a deployment trap: scp'ing "the onnx" to the
        # device then yields a model with no weights, and the failure surfaces at
        # trtexec as something unrelated. Fold them back in so the artifact is one
        # self-contained file — 86 MB is far below the 2 GB protobuf ceiling.
        sidecar = pathlib.Path(args.out + ".data")
        if sidecar.exists():
            onnx.load_external_data_for_model(m, str(sidecar.parent))
            for init in m.graph.initializer:
                init.ClearField("data_location")
                del init.external_data[:]
            onnx.save(m, args.out, save_as_external_data=False)
            sidecar.unlink()
            m = onnx.load(args.out)
            print(f"  folded {sidecar.name} back into the model (one file to ship)")
        written = next((o.version for o in m.opset_import if o.domain == ""),
                       args.opset)
        n_asym = sum(1 for n in m.graph.node if n.op_type == "Conv"
                     for a in n.attribute if a.name == "pads"
                     and list(a.ints)[:len(a.ints) // 2] != list(a.ints)[len(a.ints) // 2:])
        print(f"wrote {args.out}  (batch {args.batch}, opset {written}, "
              f"{n_asym} asymmetric-pad convs)")
        if n_asym != 4:
            print(f"⚠ expected 4 asymmetric-pad convs (stem 7x7/s2 + "
                  f"layer2/3/4[0].conv1 3x3/s2), found {n_asym} — the lean "
                  f"padding may not have survived the export")
    except ImportError:
        print(f"wrote {args.out}  (batch {args.batch}, opset {args.opset} requested; "
              f"install onnx to read back what was actually written)")
    if written != args.opset:
        print(f"⚠ requested opset {args.opset} but the file is opset {written} — "
              f"torch's down-conversion did not take. Harmless if the consumer "
              f"accepts {written}; pass --opset {written} to stop being surprised.")

    if args.verify_frame:
        verify_frame(args.out, args.tol, args.fold_preprocess)
        return

    if not args.verify:
        print("⚠ exported WITHOUT a verify gate. The replica is load-bearing "
              "here and its agreement with Lean has only ever been tested in "
              "TRAINING mode (batch stats); this exports EVAL mode (running "
              "stats). Run --verify-frame.")
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
