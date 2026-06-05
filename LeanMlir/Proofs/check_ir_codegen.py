#!/usr/bin/env python3
"""Execution-side check for the denoted-IR bridge (planning/verified_codegen.md, Loop A).

Lean side (proven): ⟦emitMlpBack⟧ = mlp_has_vjp_at.backward  (IR.mlp_whole_bridge),
and the per-op bridges (dense_at_bridge, relu_at_bridge, …).

This script closes the loop on the *execution* side: it (re)generates the
StableHLO that `IRPrint.lean` renders from those same IR graphs, compiles it
with IREE, runs it, and checks it computes the VJP against an independent
numpy reference. Compiles on IREE's CPU backend (`llvm-cpu`) — correctness
needs no GPU; ROCm/CUDA only change the backend, not the numerics.

Run:  .venv/bin/python LeanMlir/Proofs/check_ir_codegen.py
Deps: iree-base-compiler, iree-base-runtime, numpy (pip); lake (to regenerate).
"""
import os, subprocess, sys
import numpy as np
import iree.compiler as ic
import iree.runtime as rt

# Regenerate the modules from IRPrint (the single source of truth).
subprocess.run(["lake", "env", "lean", "LeanMlir/Proofs/IRPrint.lean"],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_fn(mlir_path: str, fname: str, args):
    mlir = open(mlir_path).read()
    vmfb = ic.compile_str(mlir, target_backends=["llvm-cpu"], input_type="stablehlo")
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, vmfb))
    return np.asarray(ctx.modules.m[fname](*args).to_host()), len(vmfb)

def relu_mask(p):  # 1[p > 0]
    return np.where(p > 0, 1.0, 0.0).astype(np.float32)

ok = True
TOL = 1e-4
rng = np.random.default_rng(0)

# ── linear: dx = dy · W0ᵀ ──
dy = rng.standard_normal((2, 3)).astype(np.float32)
W0 = rng.standard_normal((4, 3)).astype(np.float32)
out, nb = run_fn("/tmp/linear_back.mlir", "linear_back", [dy, W0])
ref = dy @ W0.T
e = float(np.abs(out - ref).max()); ok &= e < TOL
print(f"linear_back  ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}")

# ── mlp 4→3→3→2: dx = ((((dy·W2ᵀ)⊙1[p1>0])·W1ᵀ)⊙1[p0>0])·W0ᵀ ──
B, d0, d1, d2, d3 = 2, 4, 3, 3, 2
dy = rng.standard_normal((B, d3)).astype(np.float32)
W0 = rng.standard_normal((d0, d1)).astype(np.float32)
W1 = rng.standard_normal((d1, d2)).astype(np.float32)
W2 = rng.standard_normal((d2, d3)).astype(np.float32)
p0 = rng.standard_normal((B, d1)).astype(np.float32)
p1 = rng.standard_normal((B, d2)).astype(np.float32)
out, nb = run_fn("/tmp/mlp_back.mlir", "mlp_back", [dy, W0, W1, W2, p0, p1])
ref = (((dy @ W2.T) * relu_mask(p1)) @ W1.T * relu_mask(p0)) @ W0.T
e = float(np.abs(out - ref).max()); ok &= e < TOL
print(f"mlp_back     ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}")

print("ALL PASS (cpu)" if ok else "FAILURES (cpu)")

# ── best-effort GPU check: same module, ROCm/HIP backend, real device ──
# CPU above is the gate (correctness is backend-independent); this just
# confirms it also runs on the GPU if one is present.
def gpu_arch():
    import glob, re
    for rocminfo in ["rocminfo", *glob.glob("/opt/rocm*/bin/rocminfo")]:
        try:
            out = subprocess.run([rocminfo], capture_output=True, text=True).stdout
            m = re.search(r"gfx[0-9a-f]+", out)
            if m:
                return m.group(0)
        except Exception:
            pass
    return None

try:
    if "hip" in rt.query_available_drivers() and (arch := gpu_arch()):
        vmfb = ic.compile_str(open("/tmp/mlp_back.mlir").read(), target_backends=["rocm"],
                              input_type="stablehlo", extra_args=[f"--iree-hip-target={arch}"])
        ctx = rt.SystemContext(config=rt.Config("hip"))
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, vmfb))
        outg = np.asarray(ctx.modules.m["mlp_back"](dy, W0, W1, W2, p0, p1).to_host())
        eg = float(np.abs(outg - ref).max())
        print(f"mlp_back     (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        ok &= eg < TOL
    else:
        print("gpu: no hip device — skipped (cpu check is the gate)")
except Exception as e:
    print(f"gpu: skipped ({type(e).__name__}: {e})")

# Hard-exit past IREE's nanobind teardown, whose at-exit "leaked instances"
# report is harmless noise that would otherwise bury the PASS lines.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if ok else 1)
