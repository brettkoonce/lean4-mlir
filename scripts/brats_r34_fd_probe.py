#!/usr/bin/env python3
"""Finite-difference gate for the encoder-tap gradient seam.

The skip-equipped `r34UnetBrats` routes gradient from the decoder back into the
R34 encoder through `fpnTapGrad` (residual stages) and `addSkipGrad` (the stem's
maxPool). That join is the exact seam class that silently broke the detector's
`fpnTapGrad`, and its failure mode is not a crash — a missing or double-counted
skip gradient still trains, still descends, and merely learns a slightly worse
model. Nothing downstream would flag it.

So: check the analytic gradient against finite differences of the loss.

With `fdprobe` the optimizer is plain SGD with no decay, warmup, clipping or
weight decay, so one step is exactly

    θ′ = θ − η·g        ⟹        g = (θ − θ′) / η

and the directional-derivative identity must hold for any direction v:

    ⟨g, v⟩  ==  d/ds L(θ + s·v) |_{s=0}  ≈  (L(θ+εv) − L(θ−εv)) / 2ε

Directions tested:
  * **v = ĝ** — the best-conditioned choice; the finite difference is maximal,
    so it is not swamped by the 6-decimal precision of the printed loss.
  * **v = ĝ restricted to encoder parameters** — targets precisely what the
    skip connections change. If the tap gradient never reached the encoder,
    the full-gradient test could still pass on the decoder's contribution
    while this one fails.

Usage:
    python3 scripts/brats_r34_fd_probe.py [--eps 1e-3] [--lr 10]
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

EXE = "./.lake/build/bin/unet-brats-r34"
NOSKIP = False
TAGSUF = ""
DATA = "data/brats224_overfit"
SCRATCH = "/tmp/claude-1000/-home-skoonce-lean-proof-verify-demo-verify-v2/fd"
# Encoder = 4-ch stem + the R34 body. Everything after is decoder/head.
ENCODER_FLOATS = (64 * 4 * 49 + 64 + 64) + (21284672 - (64 * 3 * 49 + 64 + 64))


def run(tag, lr, init_load=None, init_dump=None):
    env = dict(os.environ)
    env["IREE_BACKEND"] = "rocm"
    env["HIP_VISIBLE_DEVICES"] = "0"
    if init_load:
        env["LEAN_MLIR_INIT_LOAD"] = init_load
    else:
        env.pop("LEAN_MLIR_INIT_LOAD", None)
    if init_dump:
        env["LEAN_MLIR_INIT_DUMP"] = init_dump
    else:
        env.pop("LEAN_MLIR_INIT_DUMP", None)
    # One shared tag for every run: the spec is identical across them, so a
    # per-run tag would only force iree-compile to rebuild the same 2 MB train
    # step four extra times. theta is preserved out-of-band in SCRATCH, and
    # theta' is read immediately after the step that writes it.
    cmd = [EXE, DATA, "1", "scratch", "fdprobe", f"lr={lr}", f"tag=fd{TAGSUF}"]
    if NOSKIP:
        cmd.append("noskip")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out[-3000:])
        sys.exit(f"run failed (tag={tag}, lr={lr})")
    m = re.findall(r"step \d+/\d+: loss=([\d.eE+-]+)", out)
    if not m:
        print(out[-3000:])
        sys.exit("no loss line found")
    pfx = re.search(r"artifacts: (\S+)_\*", out)
    return float(m[0]), pfx.group(1) if pfx else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=int, default=10, help="n x 1e-4")
    ap.add_argument("--noskip", action="store_true",
                    help="CONTROL: probe the skipless net, which uses none of the\n"
                         "new tap wiring. If it fails by the same margin, the\n"
                         "discrepancy is pre-existing, not caused by the skips.")
    args = ap.parse_args()
    global NOSKIP, TAGSUF
    NOSKIP = args.noskip
    TAGSUF = "ns" if args.noskip else ""
    eta = args.lr * 1e-4
    os.makedirs(SCRATCH, exist_ok=True)
    theta_path = f"{SCRATCH}/theta.bin"

    print(f"=== FD gate ({'NO-SKIP CONTROL' if NOSKIP else 'skip net'}): "
          f"eta={eta}, eps={args.eps} ===")

    # 1. dump theta and get L(theta)
    print("[1/5] baseline: lr=0, dumping theta ...")
    l0, pfx = run("fd0", 0, init_dump=theta_path)
    theta = np.fromfile(theta_path, dtype=np.float32)
    print(f"      L(theta) = {l0:.6f}   |theta| = {theta.size} floats")

    # 2. one SGD step -> theta', recover g
    print(f"[2/5] one SGD step at lr={eta} to recover g ...")
    _, pfx2 = run("fdg", args.lr, init_load=theta_path)
    theta_p = np.fromfile(f"{pfx2}_params.bin", dtype=np.float32)
    if theta_p.size != theta.size:
        sys.exit(f"size mismatch: {theta_p.size} vs {theta.size}")
    g = (theta.astype(np.float64) - theta_p.astype(np.float64)) / eta
    gn = np.linalg.norm(g)
    print(f"      |g| = {gn:.6f}   nonzero = {(g != 0).sum()}/{g.size}")
    if gn == 0:
        sys.exit("FAIL: gradient is identically zero")

    # Sweep eps. A central difference has O(eps^2) truncation error, so a
    # CORRECT gradient shows rel-err falling ~4x per halving of eps until
    # float32 noise takes over. A WRONG gradient shows rel-err flat in eps —
    # that is the discriminator, not any single eps value.
    epses = [args.eps * f for f in (1.0, 0.3, 0.1, 0.03)]
    ok = True
    for name, mask in [("full", None), ("encoder-only", "enc")]:
        v = g.copy()
        if mask == "enc":
            v[ENCODER_FLOATS:] = 0.0
            if np.linalg.norm(v) == 0:
                print(f"[{name}] FAIL: encoder gradient is identically zero")
                print("       → the skip/tap gradient never reached the encoder")
                ok = False
                continue
        v = v / np.linalg.norm(v)
        lhs = float(np.dot(g, v))
        print(f"\n[{name}] analytic <g,v> = {lhs:.6f}")
        print(f"   {'eps':>9}  {'finite diff':>12}  {'rel err':>9}")
        best = None
        for e in epses:
            tp, tm = f"{SCRATCH}/tp.bin", f"{SCRATCH}/tm.bin"
            (theta + e * v).astype(np.float32).tofile(tp)
            (theta - e * v).astype(np.float32).tofile(tm)
            lp, _ = run("fdp", 0, init_load=tp)
            lm, _ = run("fdm", 0, init_load=tm)
            rhs = (lp - lm) / (2 * e)
            rel = abs(lhs - rhs) / (abs(rhs) + 1e-12)
            print(f"   {e:>9.2e}  {rhs:>12.6f}  {rel:>9.4f}")
            best = rel if best is None else min(best, rel)
        if best is not None and best < 0.05:
            print(f"   [{name}] PASS (best rel err {best:.4f})")
        else:
            print(f"   [{name}] FAIL — rel err never converged (best {best})")
            ok = False

    print()
    print("FD GATE OK" if ok else "FD GATE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
