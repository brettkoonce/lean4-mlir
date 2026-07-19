#!/usr/bin/env python3
"""Compile + f64-FD-validate the Lean-emitted WHOLE FPN detector (neck + 1x1 heads
+ concat + multi-scale loss + full DAG backward) — the bite-7 de-risk (planning/
yolo_fpn.md). This is the one part of bite 7 that can't be FD-checked once the conv
backbone is attached (conv doesn't CPU-compile), so we pin it in isolation here.

Emitted at focal gamma=0, so the objectness focal weight (1-pt)^gamma = 1 is a
genuine constant -> the whole loss is exactly differentiable and EVERY input and
param gradient (C3/C4/C5 + the 6 neck/head weights) is finite-differenceable. The
detached-obj subtlety that forces channel-skipping at gamma=2 is gone.

  1. lake build fpn-detect-probe
  2. .lake/build/bin/fpn-detect-probe B oc c3 c4 c5 g5 A <out.mlir>
  3. this script: iree-compile (CPU) + run, numpy detector forward (neck+heads+
     concat+loss), then compare emitted forward vs numpy and emitted grads vs f64
     central FD (subsampled per tensor).

Run:  <jax-venv>/bin/python scripts/fpn_detect_probe_check.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fpn_neck_check import fpn_forward, conv1x1  # noqa: E402
import anchor_loss_probe_check as al  # noqa: E402

al.GAMMA = 0.0                 # <-- makes the objectness weight a true constant
P = 15                         # per-anchor channels
PROBE = ".lake/build/bin/fpn-detect-probe"
IREE_COMPILE = ".venv/bin/iree-compile"


def make_runner(B, oc, c3, c4, c5, g5, A, tower=0):
    td = tempfile.mkdtemp()
    mlir = os.path.join(td, "fpn_detect_gen.mlir")
    r = subprocess.run([PROBE, str(B), str(oc), str(c3), str(c4), str(c5),
                        str(g5), str(A), str(tower), mlir],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit("probe emit failed")
    vmfb = os.path.join(td, "fd.vmfb")
    r = subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                        "-o", vmfb], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:3000]); sys.exit("iree-compile failed")
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    with open(vmfb, "rb") as f:
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
    fn = ctx.modules.fpn_detect_probe["main"]

    def run(arrs):
        out = fn(*[a.astype(np.float32) for a in arrs])
        return [np.asarray(o).astype(np.float64) for o in out]
    return run


def conv3x3(x, W):
    """Same-pad stride-1 NCHW convolution matching stablehlo.convolution.

    out[b,o,y,x] = sum_{i,ky,kx} xpad[b,i,y+ky,x+kx] * W[o,i,ky,kx]
    """
    k = W.shape[2]
    pad = (k - 1) // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    B, _, H, Wd = x.shape
    out = np.zeros((B, W.shape[0], H, Wd), dtype=np.float64)
    for ky in range(k):
        for kx in range(k):
            out += np.einsum("bihw,oi->bohw", xp[:, :, ky:ky + H, kx:kx + Wd],
                             W[:, :, ky, kx])
    return out


def tower_forward(Pn, Ws, bs):
    """depth x (3x3 conv -> +bias -> ReLU), channel-preserving."""
    cur = Pn
    for W, b in zip(Ws, bs):
        cur = np.maximum(conv3x3(cur, W) + b[None, :, None, None], 0.0)
    return cur


def detector_forward(C3, C4, C5, Wn, Wh, bh, tgts, masks, grids, anchors, B,
                     tW=None, tb=None):
    """numpy mirror: neck -> tower -> 1x1 heads (+bias) -> concat -> Sigma loss (g=0)."""
    P3, P4, P5 = fpn_forward(C3, C4, C5, Wn[0], Wn[1], Wn[2])
    feats = [P3, P4, P5]
    if tW:
        feats = [tower_forward(Pn, tW[i], tb[i]) for i, Pn in enumerate(feats)]
    heads = [conv1x1(Pn, Wh[i]) + bh[i][None, :, None, None]
             for i, Pn in enumerate(feats)]
    logits = np.concatenate([h.reshape(B, -1) for h in heads], axis=1)
    A = len(anchors)
    off = 0
    total = 0.0
    for s, g in enumerate(grids):
        ln = A * P * g * g
        pred = logits[:, off:off + ln].reshape(B, A * P, g, g)
        total += al.np_forward(pred, tgts[s], masks[s], g, g, anchors)
        off += ln
    return total


def fd_sample(fwd, arr, emitted, k, rng, hf=1e-6):
    """Central FD at k random entries of `arr`; return max|emitted-FD|."""
    idxs = [tuple(rng.randint(0, s) for s in arr.shape)
            for _ in range(min(k, arr.size))]
    err = 0.0
    for idx in idxs:
        ap = arr.copy(); ap[idx] += hf
        am = arr.copy(); am[idx] -= hf
        fd = (fwd(ap) - fwd(am)) / (2 * hf)
        err = max(err, abs(emitted[idx] - fd))
    return err


def check(B=2, oc=8, c3=6, c4=10, c5=12, g5=2, A=3, seed=0, k=70, tower=0):
    rng = np.random.RandomState(seed)
    g4, g3 = 2 * g5, 4 * g5
    grids = [g3, g4, g5]
    anchors = al.anchors_for(A)
    C3 = rng.randn(B, c3, g3, g3)
    C4 = rng.randn(B, c4, g4, g4)
    C5 = rng.randn(B, c5, g5, g5)
    Wn = [rng.randn(oc, c) * 0.3 for c in (c3, c4, c5)]
    Wh = [rng.randn(A * P, oc) * 0.1 for _ in range(3)]
    # Deliberately NONZERO: a zero bias would make the emitted add a no-op and
    # the FD check vacuous on exactly the param this change adds.
    bh = [rng.randn(A * P) * 0.5 for _ in range(3)]
    tgts, masks = [], []
    for s, g in enumerate(grids):
        _pr, tg, mk = al.make_data(B, g, g, A, seed * 10 + s)
        tgts.append(tg); masks.append(mk)

    # Per-level tower params. He-ish scale so ReLU keeps a healthy live fraction:
    # an all-dead tower would make every tower gradient trivially 0 and the FD
    # check vacuous, the same trap as a zero bias.
    tW = [[rng.randn(oc, oc, 3, 3) * (2.0 / (9 * oc)) ** 0.5 for _ in range(tower)]
          for _ in range(3)]
    tb = [[rng.randn(oc) * 0.1 for _ in range(tower)] for _ in range(3)]

    run = make_runner(B, oc, c3, c4, c5, g5, A, tower)
    towerArgs = []
    for i in range(3):
        for j in range(tower):
            towerArgs += [tW[i][j], tb[i][j]]
    out = run([C3, C4, C5, Wn[0], Wn[1], Wn[2]] + towerArgs +
              [Wh[0], Wh[1], Wh[2], bh[0], bh[1], bh[2],
               tgts[0], tgts[1], tgts[2]])
    loss = out[0].item()
    dC3, dC4, dC5 = out[1:4]
    pg = out[4:]                       # param grads, fpnDetectParamShapes order
    dWn3, dWn4, dWn5 = pg[0], pg[1], pg[2]
    nT = 6 * tower
    dTower = pg[3:3 + nT]
    dWh3, dWh4, dWh5 = pg[3 + nT], pg[4 + nT], pg[5 + nT]
    dbh3, dbh4, dbh5 = pg[6 + nT], pg[7 + nT], pg[8 + nT]

    ref = detector_forward(C3, C4, C5, Wn, Wh, bh, tgts, masks, grids, anchors, B,
                           tW, tb)
    frel = abs(loss - ref) / max(abs(ref), 1e-9)

    def F(which):
        def f(a):
            args = dict(C3=C3, C4=C4, C5=C5, Wn=list(Wn), Wh=list(Wh), bh=list(bh),
                        tW=[list(x) for x in tW], tb=[list(x) for x in tb])
            if which in ("C3", "C4", "C5"):
                args[which] = a
            elif which[0] == "t":          # tW<lvl>_<layer> / tb<lvl>_<layer>
                grp, rest = which[:2], which[2:]
                lvl, lay = (int(v) for v in rest.split("_"))
                args[grp][lvl][lay] = a
            else:
                grp, i = which[:2], int(which[2])
                args[grp][i] = a
            return detector_forward(args["C3"], args["C4"], args["C5"],
                                    args["Wn"], args["Wh"], args["bh"],
                                    tgts, masks, grids, anchors, B,
                                    args["tW"], args["tb"])
        return f

    errs = {
        "dC3": fd_sample(F("C3"), C3, dC3, k, rng),
        "dC4": fd_sample(F("C4"), C4, dC4, k, rng),
        "dC5": fd_sample(F("C5"), C5, dC5, k, rng),
        "dWn3": fd_sample(F("Wn0"), Wn[0], dWn3, k, rng),
        "dWn4": fd_sample(F("Wn1"), Wn[1], dWn4, k, rng),
        "dWn5": fd_sample(F("Wn2"), Wn[2], dWn5, k, rng),
        "dWh3": fd_sample(F("Wh0"), Wh[0], dWh3, k, rng),
        "dWh4": fd_sample(F("Wh1"), Wh[1], dWh4, k, rng),
        "dWh5": fd_sample(F("Wh2"), Wh[2], dWh5, k, rng),
        "dbh3": fd_sample(F("bh0"), bh[0], dbh3, k, rng),
        "dbh4": fd_sample(F("bh1"), bh[1], dbh4, k, rng),
        "dbh5": fd_sample(F("bh2"), bh[2], dbh5, k, rng),
    }
    for i in range(3):
        for j in range(tower):
            idx = (i * tower + j) * 2
            errs[f"dWt{i}_{j}"] = fd_sample(F(f"tW{i}_{j}"), tW[i][j], dTower[idx], k, rng)
            errs[f"dbt{i}_{j}"] = fd_sample(F(f"tb{i}_{j}"), tb[i][j], dTower[idx + 1], k, rng)
    gmax = max(errs.values())
    ok = frel < 1e-4 and gmax < 2e-3
    print(f"fpn-detect probe  B={B} oc={oc} c=({c3},{c4},{c5}) g5={g5} A={A} "
          f"tower={tower} seed={seed}")
    print(f"  emitted forward vs numpy : rel={frel:.2e}  {'PASS' if frel<1e-4 else 'FAIL'}")
    print(f"  emitted grads vs f64 FD  : max={gmax:.2e}  {'PASS' if gmax<2e-3 else 'FAIL'}  "
          f"({k}/tensor)")
    for kname, v in errs.items():
        print(f"      {kname:5}: {v:.2e}")
    return ok


def main():
    ok = True
    # tower = 0: the minimal head (the in-flight T2-bias arm's geometry).
    ok &= check(seed=0)
    ok &= check(seed=1)
    ok &= check(B=1, oc=6, c3=4, c4=5, c5=7, g5=3, A=2, seed=2)
    # tower > 0: T2a. Exercises the 3x3 conv fwd/VJP and the ReLU mask, and
    # checks that the tower's dW/db land in the fpnDetectParamShapes slots.
    ok &= check(seed=3, tower=1)
    ok &= check(seed=4, tower=2)
    ok &= check(B=1, oc=6, c3=4, c4=5, c5=7, g5=3, A=2, seed=5, tower=3)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
