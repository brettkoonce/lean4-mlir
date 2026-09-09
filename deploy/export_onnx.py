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

    # Anywhere torch is available. ⚠ On the training box that is the CPU-only
    # `.venv-timm` (torch + onnx + onnxruntime + onnxscript live there), never the
    # pinned .venv — installing torch there pulls its own CUDA wheels over the
    # pinned cuDNN and kills every JAX/XLA convolution.
    P=../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__aff30e28

    # 0. once per shipped checkpoint: the frame golden is that checkpoint's own
    #    Lean logits, so cut it out of a fresh `infer` dump first
    ../.venv-timm/bin/python export_onnx.py --regen-golden ../runs/<dump>/logits.bin

    # 1. export + the self-contained gate
    ../.venv-timm/bin/python export_onnx.py --ckpt ${P}_params.bin --bn ${P}_bn_stats.bin \
        --out build/detector_aff30e28.onnx --opset 18 --verify-frame

    # 2. optional, the speed lever: the whole preprocess moves into the graph and
    #    the input becomes [1,448,448,3] uint8
    ../.venv-timm/bin/python export_onnx.py --ckpt ${P}_params.bin --bn ${P}_bn_stats.bin \
        --out build/detector_aff30e28_u8.onnx --opset 18 --fold-preprocess u8 --verify-frame

    # then on the Orin
    trtexec --onnx=detector_aff30e28.onnx --saveEngine=detector_aff30e28.plan --fp16
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
# testdata/frame.png is this row of data/visdrone_fpn/val.bin, and the golden
# next to it is the Lean stack's logits for that row under ONE checkpoint.
FRAME_VAL_RECORD = 374
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
ISTD = (1.0 / np.array([0.229, 0.224, 0.225], np.float32)).reshape(3, 1, 1)

# The input tensor's NAME carries the preprocessing contract. A plain graph and
# an f32-folded one have the identical [1,3,448,448] float32 signature, and
# feeding normalized pixels to a graph that normalizes again yields a detector
# that runs and is quietly wrong. The name survives into the TensorRT engine,
# so `orin_detect.py` reads the mode back off the engine instead of being told.
INPUT_NAMES = {"none": "image", "f32": "image_01", "u8": "image_u8"}


def input_mode_of(name, is_uint8):
    """Preprocessing mode implied by an input tensor — the inverse of INPUT_NAMES."""
    if is_uint8:
        return "u8"
    return "f32" if name.endswith("_01") else "none"


def host_input(raw_hwc_u8, mode):
    """What the host hands the graph for `mode`, from a [448,448,3] uint8 frame."""
    if mode == "u8":
        return np.ascontiguousarray(raw_hwc_u8.astype(np.uint8))[None]
    img = raw_hwc_u8.astype(np.float32).transpose(2, 0, 1) / 255.0
    if mode == "none":
        img = (img - MEAN) * ISTD
    return np.ascontiguousarray(img[None])


def _per_scale_obj_r(a, b):
    """Objectness-channel correlation at P3 / P4 / P5, worst record of [n, NTOT].

    Objectness because it is what detections rank on, and per-scale because a
    geometric misalignment compounds with depth — C3/C4/C5 sit behind 3/4/5
    stride-2 stages, so a padding or resampling difference shows up as a
    correlation that FALLS from P3 to P5 rather than as uniform noise.
    """
    a, b = a.reshape(-1, NTOT), b.reshape(-1, NTOT)
    out, off = [], 0
    for g in FPN_GRIDS:
        n = A * SLOTS * g * g
        oa = a[:, off:off + n].reshape(-1, A, SLOTS, g, g)[:, :, 4].reshape(len(a), -1)
        ob = b[:, off:off + n].reshape(-1, A, SLOTS, g, g)[:, :, 4].reshape(len(b), -1)
        out.append(min(float(np.corrcoef(oa[i], ob[i])[0, 1]) for i in range(len(a))))
        off += n
    return out


def check_against_lean(ref, got, tol, what):
    """The gate both verify paths share: a RELATIVE per-record tolerance plus
    the per-scale objectness floor. `ref`/`got` are [n, NTOT] (or one flat row).
    Raises with the diagnosis; returns the worst relative difference."""
    ref, got = ref.reshape(-1, NTOT), got.reshape(-1, NTOT)
    d = np.abs(got - ref)
    scale = np.maximum(np.abs(ref).max(axis=1), 1e-6)
    per_rec = d.max(axis=1) / scale
    rel = float(per_rec.max())
    r3, r4, r5 = _per_scale_obj_r(ref, got)
    print(f"  {what}: max rel diff {rel:.3e}  (max abs {d.max():.3e}, worst "
          f"record {int(per_rec.argmax())}, |logit| up to {scale.max():.1f})  "
          f"(tol {tol})")
    print(f"  ref  range {ref.min():+.3f} .. {ref.max():+.3f}")
    print(f"  onnx range {got.min():+.3f} .. {got.max():+.3f}")
    print(f"  objectness r  P3 {r3:.4f}  P4 {r4:.4f}  P5 {r5:.4f}"
          f"  (floor {MIN_OBJ_R})")
    if rel > tol or min(r3, r4, r5) < MIN_OBJ_R:
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
               "geometric shift.\n   Check the BN running statistics, the "
               "checkpoint parameter order — and whether\n   the golden is for "
               "THIS checkpoint (--regen-golden), which is the usual cause.\n"))
    return rel


def regen_golden(logits_path, val_bin, out=None):
    """Cut the reference frame's row out of a Lean `infer` dump.

    The golden is CHECKPOINT-SPECIFIC: it is the Lean stack's logits for val
    record FRAME_VAL_RECORD under one set of weights. Ship a new arm without
    regenerating it and --verify-frame fails with the padding-bug signature
    (rel ≈ 1, objectness r ≪ 0.999) while nothing says why.
    """
    out = Path(out) if out else HERE / "testdata" / "frame_logits.bin"
    lean = np.fromfile(logits_path, dtype=np.float32)
    if lean.size % NTOT:
        raise SystemExit(f"{logits_path}: {lean.size} floats is not a multiple of {NTOT}")
    lean = lean.reshape(-1, NTOT)
    if lean.shape[0] <= FRAME_VAL_RECORD:
        raise SystemExit(f"{logits_path}: only {lean.shape[0]} rows, need row "
                         f"{FRAME_VAL_RECORD} — is this a full val dump?")
    val_bin = Path(val_bin)
    if val_bin.exists():
        from PIL import Image
        rec = 3 * IMG_PX * IMG_PX + NTOT * 4
        with open(val_bin, "rb") as f:
            f.seek(4 + FRAME_VAL_RECORD * rec)
            raw = np.frombuffer(f.read(3 * IMG_PX * IMG_PX), dtype=np.uint8)
        png = np.asarray(Image.open(HERE / "testdata" / "frame.png").convert("RGB"))
        if not np.array_equal(raw.reshape(3, IMG_PX, IMG_PX).transpose(1, 2, 0), png):
            raise SystemExit(f"val record {FRAME_VAL_RECORD} is not testdata/frame.png "
                             f"— the dump's row order does not match the frame")
    else:
        print(f"  ⚠ {val_bin} not found; cannot confirm row {FRAME_VAL_RECORD} is frame.png")
    lean[FRAME_VAL_RECORD].tofile(out)
    print(f"wrote {out} from row {FRAME_VAL_RECORD} of {logits_path}  "
          f"(range {lean[FRAME_VAL_RECORD].min():+.3f} .. "
          f"{lean[FRAME_VAL_RECORD].max():+.3f})")


def fold_pads_into_convs(m):
    """Fold every constant zero `Pad` that feeds a `Conv` into that Conv's `pads`.

    The replica spells Lean's asymmetric SAME padding as an explicit `F.pad`
    ahead of a padding-0 conv (`bespoke.model._pre_pad`). Older torch exporters
    folded that into the Conv `pads` attribute; torch 2.13's TorchScript
    exporter leaves the Pad op in place, its `pads` input spelled as a small
    ConstantOfShape / Concat / Slice subgraph. TensorRT would fold that, but the
    shipped artifact should not lean on it: fold here, so the graph has the
    shape that was gated and measured — four asymmetric-pad convs and no Pad
    ops — and the census in main() keeps meaning what it says.
    """
    import onnx
    import onnxruntime as ort
    pads = [n for n in m.graph.node if n.op_type == "Pad"]
    if not pads:
        return 0
    # The pads (and constant_value) inputs are constant but COMPUTED; evaluate
    # them once by asking for them as extra graph outputs.
    probe = onnx.ModelProto()
    probe.CopyFrom(m)
    del probe.graph.output[:]
    want = {}
    for n in pads:
        want[n.input[1]] = onnx.TensorProto.INT64
        if len(n.input) > 2 and n.input[2]:
            want[n.input[2]] = onnx.TensorProto.FLOAT
    for nm, ty in want.items():
        probe.graph.output.append(onnx.helper.make_tensor_value_info(nm, ty, None))
    sess = ort.InferenceSession(probe.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape],
                 np.uint8 if "uint8" in inp.type else np.float32)
    vals = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp.name: x})))

    consumers = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    folded = 0
    for p in pads:
        mode = next((a.s for a in p.attribute if a.name == "mode"), b"constant")
        pv = np.asarray(vals[p.input[1]]).astype(np.int64).ravel()
        cval = (float(np.asarray(vals[p.input[2]]).ravel()[0])
                if len(p.input) > 2 and p.input[2] else 0.0)
        users = consumers.get(p.output[0], [])
        if not (mode == b"constant" and cval == 0.0 and pv.size == 8
                and not pv[[0, 1, 4, 5]].any() and len(users) == 1
                and users[0].op_type == "Conv"
                and not any(a.name == "pads" and any(a.ints) for a in users[0].attribute)):
            continue
        conv = users[0]
        for a in [a for a in conv.attribute if a.name == "pads"]:
            conv.attribute.remove(a)
        # ONNX Conv pads: [H_begin, W_begin, H_end, W_end]
        conv.attribute.append(onnx.helper.make_attribute(
            "pads", [int(v) for v in pv[[2, 3, 6, 7]]]))
        conv.input[0] = p.input[0]
        m.graph.node.remove(p)
        folded += 1
    # Sweep the orphaned pads subgraphs and any initializer nothing reads.
    outputs = {o.name for o in m.graph.output}
    while True:
        used = {i for n in m.graph.node for i in n.input} | outputs
        dead = [n for n in m.graph.node if n.output and not any(o in used for o in n.output)]
        if not dead:
            break
        for n in dead:
            m.graph.node.remove(n)
    used = {i for n in m.graph.node for i in n.input}
    for init in [i for i in m.graph.initializer if i.name not in used]:
        m.graph.initializer.remove(init)
    return folded


def verify_frame(onnx_path, tol, ref_path=None):
    """Self-contained gate: testdata/frame.png vs testdata/frame_logits.bin.

    Both ship in the repo, so this needs nothing from the training box. The
    logits are the Lean stack's own output for that frame under the eval graph
    (BN in inference mode, running stats) — reproducibly val record 374 of an
    `infer` dump, byte for byte. ⚠ Under ONE checkpoint: the golden must be
    regenerated for every arm that ships (--regen-golden), or this gate fails
    on a correct export with the same signature as a wrong one.

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

    ref_path = Path(ref_path) if ref_path else HERE / "testdata" / "frame_logits.bin"
    ref = np.fromfile(ref_path, dtype=np.float32)
    if ref.size != NTOT:
        raise SystemExit(f"reference logits are {ref.size} floats, want {NTOT}")
    raw = np.asarray(Image.open(HERE / "testdata" / "frame.png").convert("RGB"))

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    mode = input_mode_of(inp.name, "uint8" in inp.type)
    print(f"  input {inp.name} {inp.type} {inp.shape} -> host preprocessing "
          f"mode '{mode}'; golden {ref_path.name}")
    got = sess.run(None, {inp.name: host_input(raw, mode)})[0].reshape(-1)[:NTOT]
    check_against_lean(ref, got, tol, "testdata/frame.png")
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

    ⚠ TensorRT (10.3 measured) accepts UINT8 only as a NETWORK I/O tensor, never
    as an intermediate: the op that consumes `image_u8` must be the Cast. The
    first u8 export did `permute` THEN `to(float32)`, which is a uint8->uint8
    Transpose at node 0, and trtexec died parsing it ("legalUINT8: TensorRT does
    not support UINT8 types for intermediate tensors", 2026-09-09 on the Orin).
    Cast first; the Transpose then runs on f32 and TensorRT folds it into the
    convolution's format anyway, so nothing of the u8 win is lost. `f32` is the
    fallback only for a runtime older than 10 that has no UINT8 I/O at all.
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
                # cast FIRST — see the docstring; permute-then-cast is a uint8
                # intermediate and TensorRT refuses the graph at node 0
                x = x.to(torch.float32).permute(0, 3, 1, 2) / 255.0
            return self.inner((x - self.mean) * self.istd)

    return Wrapped(model, mode).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="Lean *_params.bin")
    ap.add_argument("--bn", help="Lean *_bn_stats.bin")
    ap.add_argument("--regen-golden", default=None, metavar="LOGITS_BIN",
                    help="rewrite testdata/frame_logits.bin from row "
                         f"{FRAME_VAL_RECORD} of a Lean `infer` dump, then exit. "
                         "Once per shipped checkpoint, BEFORE --verify-frame: "
                         "the golden is that checkpoint's own logits.")
    ap.add_argument("--ref-logits", default=None,
                    help="golden for --verify-frame (default "
                         "testdata/frame_logits.bin)")
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

    if args.regen_golden:
        regen_golden(args.regen_golden, args.val_bin)
        return
    if not (args.ckpt and args.bn):
        ap.error("--ckpt and --bn are required (or --regen-golden)")

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
    # torch >= 2.9 defaults to the dynamo exporter, which needs onnxscript and
    # lays the graph out differently. The artifact that was gated and measured
    # came from the TorchScript exporter, so pin it where the kwarg exists.
    import inspect
    legacy = ({"dynamo": False}
              if "dynamo" in inspect.signature(torch.onnx.export).parameters else {})
    torch.onnx.export(
        model, dummy, args.out,
        input_names=[INPUT_NAMES[args.fold_preprocess]], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=None,          # fixed batch: TensorRT prefers a static shape
        **legacy,
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
        n_folded = fold_pads_into_convs(m)
        if n_folded:
            onnx.checker.check_model(m)
            onnx.save(m, args.out)
            m = onnx.load(args.out)
            print(f"  folded {n_folded} explicit Pad ops into their convs' `pads`")
        written = next((o.version for o in m.opset_import if o.domain == ""),
                       args.opset)
        n_asym = sum(1 for n in m.graph.node if n.op_type == "Conv"
                     for a in n.attribute if a.name == "pads"
                     and list(a.ints)[:len(a.ints) // 2] != list(a.ints)[len(a.ints) // 2:])
        n_pad = sum(1 for n in m.graph.node if n.op_type == "Pad")
        print(f"wrote {args.out}  (batch {args.batch}, opset {written}, "
              f"{n_asym} asymmetric-pad convs, {n_pad} Pad ops, "
              f"input '{m.graph.input[0].name}')")
        if n_asym != 4 or n_pad:
            print(f"⚠ expected 4 asymmetric-pad convs (stem 7x7/s2 + "
                  f"layer2/3/4[0].conv1 3x3/s2) and no Pad ops, found "
                  f"{n_asym} / {n_pad} — the lean padding may not have "
                  f"survived the export")
    except ImportError:
        print(f"wrote {args.out}  (batch {args.batch}, opset {args.opset} requested; "
              f"install onnx to read back what was actually written)")
    if written != args.opset:
        print(f"⚠ requested opset {args.opset} but the file is opset {written} — "
              f"torch's down-conversion did not take. Harmless if the consumer "
              f"accepts {written}; pass --opset {written} to stop being surprised.")

    if args.verify_frame:
        verify_frame(args.out, args.tol, args.ref_logits)
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

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    mode = input_mode_of(inp.name, "uint8" in inp.type)
    got = np.empty((n, NTOT), np.float32)
    with open(args.val_bin, "rb") as f:
        for i in range(n):
            f.seek(4 + i * rec)
            raw = np.frombuffer(f.read(3 * IMG_PX * IMG_PX), dtype=np.uint8)
            raw = raw.reshape(3, IMG_PX, IMG_PX).transpose(1, 2, 0)
            got[i] = sess.run(None, {inp.name: host_input(raw, mode)})[0].reshape(-1)[:NTOT]
    check_against_lean(lean[:n], got, args.tol, f"{n} val records")
    print("✅ export matches the Lean model")


if __name__ == "__main__":
    main()
