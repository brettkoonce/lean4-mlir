"""Does the Lean forward agree with the twin on identical weights?

⚠⚠ READ THIS BEFORE TRUSTING A GREEN RUN. Without `--lean-logits` this script
compares SCALARS -- a loss and an objectness mean/std -- against numbers hard
coded in the print statements below. That is not a verification. It passed
while the twin was computing a materially different function (2026-08-28: max
elementwise difference 16.5 on logits spanning +-16), and the deployment path
was built on the strength of it. Two architectures with similar loss and similar
logit statistics agree on every number this prints.

**The elementwise comparison is the one that means something.** Pass
`--lean-logits`, a Lean dump whose row i is record i of `--data`:

    # Lean side: dump val logits (row order matches data/visdrone_fpn/val.bin)
    FPN_TAG=<tag> ./.lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn runs/<out>

    # twin side: same weights, same records, elementwise
    python3 -m bespoke.diff_lean --ckpt ..._params.bin --bn-stats ..._bn_stats.bin \\
        --data data/visdrone_fpn/val.bin --lean-logits runs/<out>/logits.bin \\
        --eval-mode --n 64

`--eval-mode` matters: BN running statistics are the deployment path, and were
never checked against Lean until the export shipped the wrong model.
`deploy/export_onnx.py --verify-frame` is the same gate on one committed frame,
needing no dump.

The scalar mode is still useful for LOCALIZING a failure once one is known:

  Lean of8long @ step 2000 (8-image overfit, train mode, batch stats):
      total 36.0   box 6.16   obj 27.55 (pos 14.8 / neg 12.8)   cls 1.72

If the twin reproduces ~36 on those weights, the emitted FORWARD is plausible
and the defect is more likely in the backward or the update path. If the twin
reports something far lower, the forward that produced those weights is not
computing what the architecture says -- and every FD probe would still have
passed, because they check the emitted gradient against the SAME emitted
forward. But "plausible" is the strongest word that belongs here.

    .venv/bin/python3 -m bespoke.diff_lean \\
        --ckpt ../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__of8long_params_e2000.bin
"""
import argparse
import os

import numpy as np
import torch

from bespoke.bn_stats import load_bn_stats
from bespoke.data import FpnBinDataset
from bespoke.lean_ckpt import load_lean_params
from bespoke.loss import CLS_WEIGHTS, fpn_loss
from bespoke.model import FPN_SCALES, FpnDetector

NTOT = sum(len(a) * 15 * g * g for g, a in FPN_SCALES)
# Correlation floor for the per-scale objectness channels, and the reason it is
# checked separately from the absolute difference: a geometric misalignment
# between the two implementations compounds with tap depth, so it shows up as a
# correlation FALLING from P3 to P5 (0.90 / 0.80 / 0.62 was the padding bug)
# rather than as uniform noise. Absolute tolerance alone does not tell you which.
MIN_OBJ_R = 0.999


def _per_scale_obj_r(a, b):
    """Objectness-channel correlation at P3 / P4 / P5, on flat [NTOT] arrays."""
    out, off = [], 0
    for g, anc in FPN_SCALES:
        n = len(anc) * 15 * g * g
        oa = a[off:off + n].reshape(len(anc), 15, g, g)[:, 4].ravel()
        ob = b[off:off + n].reshape(len(anc), 15, g, g)[:, 4].ravel()
        out.append(float(np.corrcoef(oa, ob)[0, 1]))
        off += n
    return out


def elementwise(model, ds, n, lean_logits, dev, tol):
    """The comparison that actually verifies the twin: logit for logit.

    `lean_logits` is a Lean `infer` dump, [N, NTOT] float32, row-aligned to
    `ds` by construction.

    ⚠ The tolerance is RELATIVE, and that is not fussiness. Logit magnitude
    varies enormously across val records -- the reference frame spans +-16 but
    record 40 spans -976 .. +1287 -- so any absolute threshold either fails on
    the loud records or is meaningless on the quiet ones. In relative terms the
    two regimes are stable and three orders of magnitude apart: a correct twin
    sits at 5e-4 either way (XLA runs the reference dump with TF32
    convolutions), and the wrong-padding twin sat at ~1.0 on both.
    """
    lean = np.fromfile(lean_logits, dtype=np.float32)
    if lean.size % NTOT:
        raise SystemExit(f"{lean_logits}: {lean.size} floats is not a multiple "
                         f"of NTOT={NTOT}")
    lean = lean.reshape(-1, NTOT)
    n = min(n, lean.shape[0], len(ds))
    imgs = torch.stack([ds[i][0] for i in range(n)]).to(dev)
    with torch.no_grad():
        got = model(imgs).cpu().numpy()[:, :NTOT]
    ref = lean[:n]

    d = np.abs(got - ref)
    scale = np.abs(ref).reshape(n, -1).max(axis=1)          # per-record magnitude
    per_rec = d.reshape(n, -1).max(axis=1) / np.maximum(scale, 1e-6)
    rel = float(per_rec.max())
    rs = _per_scale_obj_r(ref.ravel(), got.ravel())
    print(f"\n  elementwise over {n} records (relative tol {tol})")
    print(f"    max rel diff   {rel:.3e}   (max abs {d.max():.3e})")
    print(f"    worst record   {int(per_rec.argmax())} at {rel:.3e} rel, "
          f"|logit| up to {scale.max():.1f}")
    print(f"    lean range     {ref.min():+.3f} .. {ref.max():+.3f}")
    print(f"    twin range     {got.min():+.3f} .. {got.max():+.3f}")
    print(f"    objectness r   P3 {rs[0]:.4f}  P4 {rs[1]:.4f}  P5 {rs[2]:.4f}"
          f"   (floor {MIN_OBJ_R})")
    if rel > tol or min(rs) < MIN_OBJ_R:
        falling = rs[2] < rs[1] < rs[0]
        raise SystemExit(
            "\n⛔ THE TWIN IS NOT COMPUTING THE LEAN FUNCTION.\n"
            "   Report these numbers; do not widen the tolerance.\n"
            + ("   Correlation falls from P3 to P5 — a geometric misalignment "
               "compounding through\n   the stride-2 stages. Check --pad "
               "(Lean is TF-style ASYMMETRIC SAME) and the\n   stem pool "
               "(Lean is `.maxPool 2 2`) first.\n" if falling else
               "   Error is even across scales, so it is not a geometric "
               "shift. Check the BN\n   running statistics, --eval-mode, and "
               "the checkpoint parameter order.\n"))
    print("\n  ✅ the twin reproduces the Lean logits elementwise")

# demos/visdrone/bespoke/ -> repo root is THREE levels up, not two. With two
# this resolved to demos/, so every `--data` default pointed at a path that
# has never existed and the documented invocations all needed an explicit one.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(REPO, "data/visdrone_fpn/train.bin"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--eval-mode", action="store_true",
                    help="BN in eval mode with running stats (needs --bn-stats). "
                         "This is the DEPLOYMENT path and the one that shipped a "
                         "wrong model unnoticed; the Lean *training* loss uses "
                         "batch stats, so leave it off to compare losses and turn "
                         "it on to compare against an `infer` dump")
    ap.add_argument("--bn-stats", default=None,
                    help="Lean *_bn_stats.bin; required by --eval-mode, since "
                         "torchvision's untrained (0,1) buffers otherwise make "
                         "this a comparison against noise")
    ap.add_argument("--pad", default="lean", choices=["lean", "torchvision"],
                    help="stride-2 conv padding convention; 'lean' is the "
                         "faithful one (MlirCodegen.samePad is asymmetric). ⚠ "
                         "This script hardcoded 'torchvision' until 2026-08-28 "
                         "and that alone was a 16.5 elementwise disagreement")
    ap.add_argument("--lean-logits", default=None,
                    help="Lean `infer` dump, [N, NTOT] f32, row-aligned to --data. "
                         "Turns this into an ELEMENTWISE verification instead of a "
                         "scalar sanity check — see the module docstring")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="RELATIVE logit tolerance for --lean-logits (max abs "
                         "difference over max abs logit, per record). Measured "
                         "endpoints: a correct twin sits at 5e-4 because the Lean "
                         "dump uses TF32 convs, the padding bug sat at ~1.0")
    args = ap.parse_args()

    if args.eval_mode and not args.bn_stats:
        raise SystemExit("--eval-mode needs --bn-stats, or BN runs on "
                         "torchvision's untrained (0,1) buffers and the "
                         "comparison is meaningless")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FpnDetector(backbone="r34", tower=0, norm=None, pretrained=False,
                        pool="lean", pad=args.pad).to(dev)
    load_lean_params(model, args.ckpt)
    if args.bn_stats:
        load_bn_stats(model, args.bn_stats)
    model.train(not args.eval_mode)

    ds = FpnBinDataset(args.data)

    if args.lean_logits:
        elementwise(model, ds, args.n, args.lean_logits, dev, args.tol)
        return

    imgs = torch.stack([ds[i][0] for i in range(args.n)]).to(dev)
    tgts = torch.stack([ds[i][1] for i in range(args.n)]).to(dev)

    bd = dict.fromkeys(["box", "obj", "cls", "obj_pos", "obj_neg", "npos"], 0.0)
    with torch.no_grad():
        logits = model(imgs)
        loss = fpn_loss(logits, tgts, FPN_SCALES, args.gamma, CLS_WEIGHTS, breakdown=bd)

    obj_logits = torch.cat([
        logits[:, o:o + len(a) * 15 * g * g].view(args.n, len(a) * 15, g, g)[:, 4::15]
        .reshape(args.n, -1)
        for o, (g, a) in zip(
            [0, 3 * 15 * 56 * 56, 3 * 15 * 56 * 56 + 3 * 15 * 28 * 28], FPN_SCALES)
    ], dim=1)

    print(f"\n{args.n} records, BN {'eval' if args.eval_mode else 'train (batch stats)'}")
    print(f"  twin total     {float(loss):10.4f}")
    print(f"  twin box       {bd['box']:10.4f}")
    print(f"  twin obj       {bd['obj']:10.4f}  (pos {bd['obj_pos']:.3f} / "
          f"neg {bd['obj_neg']:.3f})")
    print(f"  twin cls       {bd['cls']:10.4f}")
    print(f"  npos/img       {bd['npos'] / args.n:10.1f}")
    print(f"\n  Lean of8long @2000:  total 36.0  box 6.16  obj 27.55 "
          f"(pos 14.8 / neg 12.8)  cls 1.72")
    print(f"\n  objectness logits: mean {obj_logits.mean():.4f} "
          f"std {obj_logits.std():.4f} min {obj_logits.min():.4f} "
          f"max {obj_logits.max():.4f}")
    print("  (Lean measured mean -1.80, std ~0.30 on the production arm)")
    print("\n⚠ Those two reference lines are HARDCODED, from runs that predate "
          "the shuffle fix.\n  Agreeing with them is not evidence the twin "
          "computes the Lean function — a twin\n  that differed elementwise by "
          "16.5 agreed with them. Pass --lean-logits.")


if __name__ == "__main__":
    main()
