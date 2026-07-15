#!/usr/bin/env python
"""Compile + numerically validate the Lean-emitted segmentation loss blocks
(planning/brats_demo.md Workstream B).

  1. lake build seg-loss-probe
  2. .lake/build/bin/seg-loss-probe <kind> <B NC H W> <out.mlir>
  3. this script: iree-compile (CPU) + iree.runtime run, then
       (a) compare `loss` to an independent numpy implementation, and
       (b) compare `d_logits` to CENTRAL FINITE DIFFERENCES of `loss`.

Why (b) matters most: per-pixel CE's backward seed is `(softmax - onehot)/N`
because the softmax Jacobian cancels against the log — verifiable by eye.
Dice gets no such cancellation and carries an explicit Jacobian-vector product
`dz_i = p_i(g_i - Σ_j g_j p_j)`. Finite differences don't care what we believe
the derivative is, which is the point.

Run:  <jax-venv>/bin/python scripts/seg_loss_probe_check.py
Needs iree.compiler + iree.runtime. GPU not required — CPU is the whole point.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402

PROBE = ".lake/build/bin/seg-loss-probe"
IREE_COMPILE = ".venv/bin/iree-compile"
SMOOTH = 1.0  # must match emitSegDiceBlock's default `smooth`


# ---------------------------------------------------------------- numpy model
def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def onehot(y, NC):
    # y: [B,H,W] int -> [B,NC,H,W] float
    B, H, W = y.shape
    o = np.zeros((B, NC, H, W), dtype=np.float64)
    for c in range(NC):
        o[:, c, :, :] = (y == c).astype(np.float64)
    return o


def ce_loss(z, y, NC, ls=0.0):
    """Per-pixel softmax CE, mean over B*H*W. Mirrors emitPerPixelCEBlock."""
    B, _, H, W = z.shape
    zs = z - z.max(axis=1, keepdims=True)
    logp = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True))
    yh = onehot(y, NC)
    if ls > 0.0:
        on = 1.0 - ls + ls / NC
        off = ls / NC
        yh = yh * on + (1.0 - yh) * off
    return -(logp * yh).sum() / (B * H * W)


def dice_loss(z, y, NC, smooth=SMOOTH):
    """Batch soft Dice, meaned over classes. Mirrors emitSegDiceBlock."""
    p = softmax(z, axis=1)
    yh = onehot(y, NC)
    I = (p * yh).sum(axis=(0, 2, 3))       # [NC]
    P = p.sum(axis=(0, 2, 3))
    Y = yh.sum(axis=(0, 2, 3))
    D = (2.0 * I + smooth) / (P + Y + smooth)
    return 1.0 - D.mean()


def total_loss(kind, z, y, NC, ls=0.0):
    if kind == "ce":
        return ce_loss(z, y, NC, ls)
    if kind == "dice":
        return dice_loss(z, y, NC)
    return ce_loss(z, y, NC, ls) + dice_loss(z, y, NC)


# ------------------------------------------------------------------- harness
def build_and_run(kind, B, NC, H, W, z, y):
    with tempfile.TemporaryDirectory() as td:
        mlir = os.path.join(td, "seg_loss_gen.mlir")
        r = subprocess.run([PROBE, kind, str(B), str(NC), str(H), str(W), mlir],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            sys.exit(f"probe emit failed for {kind}")
        vmfb = os.path.join(td, "seg_loss.vmfb")
        r = subprocess.run([IREE_COMPILE, mlir,
                            "--iree-hal-target-backends=llvm-cpu",
                            "-o", vmfb], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[:3000])
            sys.exit(f"iree-compile failed for {kind}")
        cfg = rt.Config("local-task")
        ctx = rt.SystemContext(config=cfg)
        with open(vmfb, "rb") as f:
            vm = rt.VmModule.copy_buffer(ctx.instance, f.read())
        ctx.add_vm_module(vm)
        out = ctx.modules.seg_loss_probe["main"](
            z.astype(np.float32), y.astype(np.int32))
        loss = np.asarray(out[0]).item()
        dz = np.asarray(out[1]).astype(np.float64)
        return loss, dz


def check(kind, B=2, NC=4, H=3, W=3, seed=0):
    rng = np.random.RandomState(seed)
    # Logit scale ~1 keeps the softmax off its saturated tails, where f32 FD
    # would be dominated by cancellation rather than by any real error.
    z = rng.randn(B, NC, H, W).astype(np.float64)
    y = rng.randint(0, NC, size=(B, H, W))

    loss_mlir, dz_mlir = build_and_run(kind, B, NC, H, W, z, y)
    loss_np = total_loss(kind, z, y, NC)

    # (a) forward agreement
    ferr = abs(loss_mlir - loss_np)

    # (b) central finite differences of the numpy loss (the ground truth the
    #     emitted gradient must match). h chosen for f64 central differences.
    h = 1e-5
    dz_fd = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        zp = z.copy(); zp[i] += h
        zm = z.copy(); zm[i] -= h
        dz_fd[i] = (total_loss(kind, zp, y, NC) - total_loss(kind, zm, y, NC)) / (2 * h)
        it.iternext()

    denom = np.maximum(np.abs(dz_fd).max(), 1e-12)
    gerr = np.abs(dz_mlir - dz_fd).max() / denom

    ok = ferr < 1e-6 and gerr < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] {kind:6s} "
          f"loss mlir={loss_mlir:+.8f} numpy={loss_np:+.8f} |Δ|={ferr:.2e} | "
          f"grad vs FD rel={gerr:.2e}")
    if not ok:
        bad = np.unravel_index(np.abs(dz_mlir - dz_fd).argmax(), dz_fd.shape)
        print(f"         worst @ {bad}: mlir={dz_mlir[bad]:+.8f} fd={dz_fd[bad]:+.8f}")
    return ok


def main():
    if not os.path.exists(PROBE):
        sys.exit(f"{PROBE} missing — run: lake build seg-loss-probe")
    print("seg-loss probe: emitted loss + d_logits vs numpy + central finite differences")
    results = []
    for kind in ("ce", "dice", "dicece"):
        results.append(check(kind))
    # A shape where some class is entirely absent from the batch — the case the
    # Dice smoothing term exists for (0/0 without it), and the case that
    # actually occurs on BraTS, where enhancing tumour is missing from many
    # slices.
    print("  -- class-absent case (Dice's 0/0 guard) --")
    for kind in ("dice", "dicece"):
        rng = np.random.RandomState(7)
        B, NC, H, W = 2, 4, 3, 3
        z = rng.randn(B, NC, H, W)
        y = np.zeros((B, H, W), dtype=int)  # only class 0 present
        loss_mlir, dz_mlir = build_and_run(kind, B, NC, H, W, z, y)
        loss_np = total_loss(kind, z, y, NC)
        ferr = abs(loss_mlir - loss_np)
        ok = ferr < 1e-6 and np.isfinite(dz_mlir).all()
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {kind:6s} all-background: "
              f"loss mlir={loss_mlir:+.8f} numpy={loss_np:+.8f} |Δ|={ferr:.2e} "
              f"grad finite={bool(np.isfinite(dz_mlir).all())}")
    print()
    if all(results):
        print(f"All {len(results)} seg-loss checks PASS.")
        return 0
    print(f"{sum(1 for r in results if not r)}/{len(results)} FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
