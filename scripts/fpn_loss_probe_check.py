#!/usr/bin/env python3
"""Compile + numerically validate the Lean-emitted FPN multi-scale-loss block
(detection-infra brick #3, planning/yolo_fpn.md bites 4+6).

  1. lake build fpn-loss-probe
  2. .lake/build/bin/fpn-loss-probe B A g3 g4 g5 <out.mlir>
  3. this script: iree-compile (CPU) + run, then
     (a) emitted forward  vs numpy Σ-of-per-scale-anchor-loss,
     (b) emitted grad     vs numpy per-scale-grad re-concatenated, and
     (c) emitted grad     vs f64 finite differences through the CONCAT (box+cls
         channels; obj is a detached focal weight, checked analytically in (b)).

This is the conv-free glue of the FPN detector: split the [B,Ntot] concat back
per scale, run the FD-verified anchor loss on each, sum, re-concat the grads. The
per-scale anchor loss itself is already FD-verified (anchor_loss_probe_check.py);
this probe pins the split/sum/concat plumbing that adapts 3 scales into the
single-output train step. Conv heads feeding this are verified convBn (ROCm).

Run:  <jax-venv>/bin/python scripts/fpn_loss_probe_check.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from anchor_loss_probe_check import np_forward, np_grad, make_data, anchors_for, P  # noqa: E402


# T1b probe weights: deterministic, must match demos/MainFpnLossProbe.lean --clsw
CLSW = [0.5 + 0.25 * c for c in range(10)]

PROBE = ".lake/build/bin/fpn-loss-probe"
IREE_COMPILE = ".venv/bin/iree-compile"


def make_runner(B, A, grids, clsw=False):
    td = tempfile.mkdtemp()
    mlir = os.path.join(td, "fpn_loss_gen.mlir")
    cmd = [PROBE, str(B), str(A), *[str(g) for g in grids], mlir]
    if clsw:
        cmd.append("--clsw")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit("probe emit failed")
    vmfb = os.path.join(td, "fl.vmfb")
    r = subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                        "-o", vmfb], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:3000]); sys.exit("iree-compile failed")
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    with open(vmfb, "rb") as f:
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
    fn = ctx.modules.fpn_loss_probe["main"]

    def run(logits, tgts):
        out = fn(logits.astype(np.float32), *[t.astype(np.float32) for t in tgts])
        return np.asarray(out[0]).item(), np.asarray(out[1]).astype(np.float64)
    return run


def ms_forward_from_logits(logits, tgts, masks, grids, anchors, B, clsw=None):
    """Split [B,Ntot] back per scale and sum the anchor losses — the exact numpy
    mirror of emitMultiScaleYoloLoss's forward."""
    A = len(anchors)
    off = 0
    total = 0.0
    for s, g in enumerate(grids):
        ln = A * P * g * g
        pred = logits[:, off:off + ln].reshape(B, A * P, g, g)
        total += np_forward(pred, tgts[s], masks[s], g, g, anchors, clsw)
        off += ln
    return total


def ms_grad_concat(preds, tgts, masks, grids, anchors, B, clsw=None):
    parts = []
    for s, g in enumerate(grids):
        parts.append(np_grad(preds[s], tgts[s], masks[s], g, g, anchors, clsw).reshape(B, -1))
    return np.concatenate(parts, axis=1)


def check(B=2, A=3, grids=(8, 4, 2), seed=0, fd_samples=400, clsw=None):
    anchors = anchors_for(A)
    preds, tgts, masks = [], [], []
    for s, g in enumerate(grids):
        pr, tg, mk = make_data(B, g, g, A, seed * 10 + s)   # distinct per scale
        preds.append(pr); tgts.append(tg); masks.append(mk)
    logits = np.concatenate([preds[s].reshape(B, -1) for s in range(len(grids))], axis=1)
    Ntot = logits.shape[1]

    run = make_runner(B, A, list(grids), clsw=clsw is not None)
    loss, grad = run(logits, tgts)

    ref = ms_forward_from_logits(logits, tgts, masks, grids, anchors, B, clsw)
    frel = abs(loss - ref) / max(abs(ref), 1e-9)
    npg = ms_grad_concat(preds, tgts, masks, grids, anchors, B, clsw)
    gmax = np.abs(grad - npg).max()

    # (c) FD through the concat on box+cls positions (skip obj channel base+4).
    #     Map each flat position -> (scale, channel) to decide box/cls vs obj.
    offs, spans = [], []
    off = 0
    for g in grids:
        offs.append(off); spans.append(A * P * g * g); off += A * P * g * g
    rng = np.random.RandomState(1234)
    boxcls = []
    for b in range(B):
        for s, g in enumerate(grids):
            for q in range(spans[s]):
                c = q // (g * g)
                if c % P == 4:           # obj channel (detached weight): skip
                    continue
                boxcls.append((b, offs[s] + q))
    sample = [boxcls[i] for i in rng.choice(len(boxcls), min(fd_samples, len(boxcls)), replace=False)]
    hf = 1e-6
    fd_err = 0.0
    for (b, p) in sample:
        lp = logits.copy(); lp[b, p] += hf
        lm = logits.copy(); lm[b, p] -= hf
        fdv = (ms_forward_from_logits(lp, tgts, masks, grids, anchors, B, clsw)
               - ms_forward_from_logits(lm, tgts, masks, grids, anchors, B, clsw)) / (2 * hf)
        fd_err = max(fd_err, abs(grad[b, p] - fdv))

    ok = frel < 1e-4 and gmax < 1e-3 and fd_err < 1e-3
    tagw = " clsw=ON" if clsw is not None else ""
    print(f"fpn-loss probe  B={B} A={A} grids={tuple(grids)} Ntot={Ntot} seed={seed}{tagw}")
    print(f"  emitted forward  vs numpy Σ-loss     : rel={frel:.2e}  {'PASS' if frel<1e-4 else 'FAIL'}")
    print(f"  emitted grad     vs numpy concat-grad: max={gmax:.2e}  {'PASS' if gmax<1e-3 else 'FAIL'}")
    print(f"  emitted grad     vs f64 FD (box+cls) : max={fd_err:.2e}  ({len(sample)} samples) "
          f"{'PASS' if fd_err<1e-3 else 'FAIL'}")
    return ok


def main():
    ok = True
    for seed in (0, 1):
        ok &= check(B=2, A=3, grids=(8, 4, 2), seed=seed)
    ok &= check(B=1, A=6, grids=(6, 4, 2), seed=2)   # A=6, the near-real config
    # T1b: the class-weighted path (weights are target-only ⇒ still exactly FD-able)
    for seed in (0, 3):
        ok &= check(B=2, A=3, grids=(8, 4, 2), seed=seed, clsw=CLSW)
    ok &= check(B=1, A=6, grids=(6, 4, 2), seed=4, clsw=CLSW)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
