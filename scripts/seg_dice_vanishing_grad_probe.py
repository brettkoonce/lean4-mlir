#!/usr/bin/env python
"""Does the Dice gradient vanish at a collapsed softmax? (planning/brats_demo.md)

MEASURED 2026-07-15: yes, exactly linearly in p_i. This is why the matched
1-epoch BraTS ablation gave dicece == ce to four decimals — see Gate B.

Hypothesis: dz_i = p_i(g_i - Σ_j g_j p_j) carries a p_i factor, so once the
net predicts p_c ~ 0 for a rare class, Dice can no longer push it back up.
CE's dz_i = (p_i - y_i)/N has no such factor.

Test: drive class 3's logit down (simulating progressive collapse) and watch
the magnitude of the gradient on the class-3 channel AT PIXELS WHERE CLASS 3
IS THE TRUE LABEL — i.e. the signal that is supposed to rescue it.

Uses the real emitted MLIR, not the numpy model.
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


def run(kind, B, NC, H, W, z, y):
    with tempfile.TemporaryDirectory() as td:
        mlir = os.path.join(td, "g.mlir")
        subprocess.run([PROBE, kind, str(B), str(NC), str(H), str(W), mlir],
                       capture_output=True, check=True)
        vmfb = os.path.join(td, "g.vmfb")
        subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                        "-o", vmfb], capture_output=True, check=True)
        ctx = rt.SystemContext(config=rt.Config("local-task"))
        with open(vmfb, "rb") as f:
            ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
        out = ctx.modules.seg_loss_probe["main"](z.astype(np.float32), y.astype(np.int32))
        return np.asarray(out[0]).item(), np.asarray(out[1]).astype(np.float64)


def main():
    B, NC, H, W = 2, 4, 8, 8
    rng = np.random.RandomState(0)

    # A realistic BraTS-ish imbalance: mostly class 0, a few class-3 pixels.
    y = np.zeros((B, H, W), dtype=int)
    y[0, 3:5, 3:5] = 3          # a small "enhancing tumour"
    frac = (y == 3).mean()
    print(f"class-3 pixels: {(y == 3).sum()}/{y.size} = {100*frac:.2f}%\n")

    print("Sweep: push class-3 logit down (collapse), watch the gradient that")
    print("should rescue it (mean |dz| on channel 3 at true-class-3 pixels).\n")
    print(f"  {'logit_3':>8} {'p_3':>10} | {'dice |dz3|':>12} {'ce |dz3|':>12} {'ratio d/ce':>11}")
    print("  " + "-" * 60)

    tum = (y == 3)
    for bias in (0.0, -2.0, -4.0, -6.0, -8.0, -10.0):
        z = rng.randn(B, NC, H, W) * 0.5
        z[:, 3, :, :] += bias           # suppress class 3
        # p_3 at the tumour pixels
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p3 = p[:, 3, :, :][tum].mean()

        _, dz_d = run("dice", B, NC, H, W, z, y)
        _, dz_c = run("ce", B, NC, H, W, z, y)
        gd = np.abs(dz_d[:, 3, :, :][tum]).mean()
        gc = np.abs(dz_c[:, 3, :, :][tum]).mean()
        print(f"  {bias:8.1f} {p3:10.2e} | {gd:12.3e} {gc:12.3e} {gd/gc:11.2e}")

    print()
    print("Reading: if the dice column collapses toward 0 while ce stays flat,")
    print("the p_i factor in the softmax Jacobian is the trap — Dice cannot")
    print("recover a class it has already zeroed, and .diceCE is then the worst")
    print("of both (CE drives the collapse, Dice can't undo it).")


if __name__ == "__main__":
    sys.exit(main())
