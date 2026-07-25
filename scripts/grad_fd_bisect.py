#!/usr/bin/env python3
"""Bisect the analytic-vs-finite-difference gradient gap by layer type.

Same test as scripts/brats_r34_fd_probe.py, run up a ladder of MNIST-scale nets
that each add one ingredient. The rung where the gap first appears names the
culprit.

Read `mlp` first and separately: it is dense-only, the most exercised backward
in the repo. If `mlp` shows the gap, the harness is wrong and no other row on
the table means anything.

Usage:
    python3 scripts/grad_fd_bisect.py [--rungs mlp,conv,...] [--eps 1e-3]
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

EXE = "./.lake/build/bin/grad-fd-probe"
SCRATCH = "/tmp/claude-1000/-home-skoonce-lean-proof-verify-demo-verify-v2/fdb"


def run(rung, lr, init_load=None, init_dump=None):
    env = dict(os.environ)
    env["IREE_BACKEND"] = "rocm"
    env["HIP_VISIBLE_DEVICES"] = "0"
    for k, v in (("LEAN_MLIR_INIT_LOAD", init_load), ("LEAN_MLIR_INIT_DUMP", init_dump)):
        if v:
            env[k] = v
        else:
            env.pop(k, None)
    r = subprocess.run([EXE, rung, str(lr), "data/mnist16"], env=env,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out[-2500:])
        sys.exit(f"run failed (rung={rung}, lr={lr})")
    m = re.findall(r"step \d+/\d+: loss=([\d.eE+-]+)", out)
    if not m:
        print(out[-2500:])
        sys.exit(f"no loss line (rung={rung})")
    pfx = re.search(r"artifacts: (\S+)_\*", out)
    return float(m[0]), (pfx.group(1) if pfx else None)


def probe(rung, eps_list, lr):
    eta = lr * 1e-4
    theta_path = f"{SCRATCH}/{rung}_theta.bin"
    _, _ = run(rung, 0, init_dump=theta_path)
    theta = np.fromfile(theta_path, dtype=np.float32)
    _, pfx = run(rung, lr, init_load=theta_path)
    theta_p = np.fromfile(f"{pfx}_params.bin", dtype=np.float32)
    if theta_p.size != theta.size:
        sys.exit(f"{rung}: size mismatch {theta_p.size} vs {theta.size}")
    g = (theta.astype(np.float64) - theta_p.astype(np.float64)) / eta
    if np.linalg.norm(g) == 0:
        return theta.size, None, [("--", float("nan"), float("nan"))]
    v = g / np.linalg.norm(g)
    lhs = float(np.dot(g, v))
    rows = []
    for e in eps_list:
        tp, tm = f"{SCRATCH}/tp.bin", f"{SCRATCH}/tm.bin"
        (theta + e * v).astype(np.float32).tofile(tp)
        (theta - e * v).astype(np.float32).tofile(tm)
        lp, _ = run(rung, 0, init_load=tp)
        lm, _ = run(rung, 0, init_load=tm)
        rhs = (lp - lm) / (2 * e)
        rows.append((e, rhs, abs(lhs - rhs) / (abs(rhs) + 1e-12)))
    return theta.size, lhs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="mlp,conv,pool,convbn,res,gap")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    eps_list = [args.eps, args.eps * 0.1]

    print(f"{'rung':>8} {'params':>9} {'analytic':>11} {'fd@1e-3':>11} "
          f"{'rel':>7} {'fd@1e-4':>11} {'rel':>7}")
    print("-" * 72)
    results = {}
    for rung in args.rungs.split(","):
        n, lhs, rows = probe(rung, eps_list, args.lr)
        if lhs is None:
            print(f"{rung:>8} {n:>9} {'ZERO GRAD':>11}")
            continue
        r0, r1 = rows[0], rows[1]
        print(f"{rung:>8} {n:>9} {lhs:>11.5f} {r0[1]:>11.5f} {r0[2]:>7.4f} "
              f"{r1[1]:>11.5f} {r1[2]:>7.4f}")
        results[rung] = min(r0[2], r1[2])

    print()
    if "mlp" in results:
        if results["mlp"] > 0.05:
            print(f"HARNESS SUSPECT: mlp rel err {results['mlp']:.4f} — the dense-only")
            print("  backward is the repo's most exercised path. Distrust every other")
            print("  row until this is explained.")
        else:
            print(f"harness OK (mlp rel err {results['mlp']:.4f})")
    clean = [k for k, v in results.items() if v < 0.05]
    dirty = [k for k, v in results.items() if v >= 0.05]
    print(f"  agrees with FD : {clean}")
    print(f"  disagrees      : {dirty}")


if __name__ == "__main__":
    main()
