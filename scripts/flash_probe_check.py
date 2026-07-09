#!/usr/bin/env python
"""Compile + numerically validate the Lean-emitted FlashAttention forward
against dense attention (planning/flash_attention.md rung 2-3).

  1. lake build flash-probe
  2. .lake/build/bin/flash-probe <b heads n dh bk> [causal] <out.mlir>
  3. this script: iree-compile (CPU) + iree.runtime run + compare to numpy dense.

Run:  <lean4-jax-venv>/bin/python scripts/flash_probe_check.py
Needs iree.compiler + iree.runtime (the lean4-jax venv). GPU not required —
the whole point is that the flash pattern is validatable offline on CPU.
"""
import os, subprocess, sys, tempfile
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt

PROBE = ".lake/build/bin/flash-probe"
IREE_COMPILE = ".venv/bin/iree-compile"


def dense(Q, K, V, causal):
    B, H, T, d = Q.shape
    s = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(d)
    if causal:
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s); p /= p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, V)


def dense_bwd(Q, K, V, dO, causal):
    B, H, T, d = Q.shape
    sc = 1.0 / np.sqrt(d)
    S = np.einsum("bhqd,bhkd->bhqk", Q, K) * sc
    if causal:
        S = np.where(np.tril(np.ones((T, T), bool)), S, -np.inf)
    S = S - S.max(-1, keepdims=True); E = np.exp(S); P = E / E.sum(-1, keepdims=True)
    dV = np.einsum("bhqk,bhqd->bhkd", P, dO)
    dP = np.einsum("bhqd,bhkd->bhqk", dO, V)
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    dQ = np.einsum("bhqk,bhkd->bhqd", dS, K) * sc
    dK = np.einsum("bhqk,bhqd->bhkd", dS, Q) * sc
    return dQ, dK, dV


def run_vmfb(vmfb_path):
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    with open(vmfb_path, "rb") as f:
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
    return ctx.modules.flash_probe["main"]


def emit_and_compile(tmp, mode_args, b, h, n, d, bk, causal):
    mlir = os.path.join(tmp, "f.mlir"); vmfb = os.path.join(tmp, "f.vmfb")
    args = [PROBE, str(b), str(h), str(n), str(d), str(bk)] + mode_args
    if causal: args.append("causal")
    args.append(mlir)
    subprocess.run(args, check=True, capture_output=True)
    subprocess.run([IREE_COMPILE, "--iree-hal-target-backends=llvm-cpu", mlir, "-o", vmfb],
                   check=True, capture_output=True)
    return run_vmfb(vmfb)


def check(b, h, n, d, bk, causal):
    tag = "causal" if causal else "full"
    rng = np.random.default_rng(hash((b, h, n, d, bk, causal)) & 0xFFFF)
    Q, K, V, dO = (rng.standard_normal((b, h, n, d)).astype(np.float32) for _ in range(4))
    with tempfile.TemporaryDirectory() as tmp:
        fwd = emit_and_compile(tmp, [], b, h, n, d, bk, causal)
        ferr = np.abs(np.asarray(fwd(Q, K, V)) - dense(Q, K, V, causal)).max()
    with tempfile.TemporaryDirectory() as tmp:
        bwd = emit_and_compile(tmp, ["bwd"], b, h, n, d, bk, causal)
        dQf, dKf, dVf = (np.asarray(x) for x in bwd(Q, K, V, dO))
        dQd, dKd, dVd = dense_bwd(Q, K, V, dO, causal)
        berr = max(np.abs(dQf - dQd).max(), np.abs(dKf - dKd).max(), np.abs(dVf - dVd).max())
    ok = ferr < 1e-4 and berr < 1e-4
    print(f"  [{'OK ' if ok else 'FAIL'}] b={b} h={h} n={n} d={d} bk={bk} {tag:6s}"
          f"  fwd Δ={ferr:.2e}  bwd Δ={berr:.2e}")
    return ok


if __name__ == "__main__":
    ok = True
    for causal in (False, True):
        ok &= check(1, 2, 16, 8, 4, causal)    # 4 blocks
        ok &= check(1, 2, 64, 8, 8, causal)    # 8 blocks
        ok &= check(2, 4, 128, 16, 32, causal) # 4 blocks, bigger
    print("done — flash forward + backward emitters validated vs dense." if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
