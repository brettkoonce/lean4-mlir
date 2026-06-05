#!/usr/bin/env python3
"""Execution-side check for the denoted-IR bridge (planning/verified_codegen.md, Loop A).

Lean side (proven): ⟦emitMlpBack⟧ = mlp_has_vjp_at.backward  (IR.mlp_whole_bridge),
the per-op bridges (dense_at_bridge, relu_at_bridge, …), and the parameter
gradients weight_grad_bridge / bias_grad_bridge (dW = outer(x, dy), db = Σ dy).

This script closes the loop on the *execution* side: it (re)generates the
StableHLO that `IRPrint.lean` renders from those same IR graphs, compiles it
with IREE, runs it, and checks it against an independent numpy reference:

  • linear_back / mlp_back   — the input-gradient (VJP) chain (dx);
  • mlp_fwd                   — the forward map (⟦emitMlpFwd⟧ = mlpForward);
  • loss_cot                  — softmax-CE gradient (⟦emitLossCot⟧ = ∂L/∂logits);
  • mlp_train_step           — a full SGD step: proof-backed forward → loss
                               cotangent → backward (dx + dW/db) → trusted SGD;
                               checks the six updated parameters against numpy.
  • conv_fwd / conv_back     — CNN (Phase 3): conv2d as stablehlo.convolution,
                               and the proven conv input-VJP (transpose+reverse
                               +conv = ⟦convBackDenote⟧).

Compiles on IREE's CPU backend (`llvm-cpu`) — correctness needs no GPU; the
ROCm/HIP leg only changes the backend, not the numerics.

Run:  .venv/bin/python LeanMlir/Proofs/check_ir_codegen.py
Deps: iree-base-compiler, iree-base-runtime, numpy (pip); lake (to regenerate).
"""
import os, subprocess, sys, glob, re
import numpy as np
import iree.compiler as ic
import iree.runtime as rt

# Regenerate the modules from IRPrint (the single source of truth).
subprocess.run(["lake", "env", "lean", "LeanMlir/Proofs/IRPrint.lean"],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _np(r):  # IREE result → numpy (single result or one element of a tuple)
    return np.asarray(r.to_host() if hasattr(r, "to_host") else r)

def compile_run(mlir_path, fname, args, backend="llvm-cpu", config="local-task", extra=None):
    """Compile `mlir_path` for `backend`, run `fname(*args)`, return (outputs, nbytes).
    `outputs` is a list of numpy arrays (one per result)."""
    vmfb = ic.compile_str(open(mlir_path).read(), target_backends=[backend],
                          input_type="stablehlo", extra_args=extra or [])
    ctx = rt.SystemContext(config=rt.Config(config))
    ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, vmfb))
    res = ctx.modules.m[fname](*args)
    outs = [_np(r) for r in res] if isinstance(res, (list, tuple)) else [_np(res)]
    return outs, len(vmfb)

def relu_mask(p):  # 1[p > 0]
    return np.where(p > 0, 1.0, 0.0).astype(np.float32)

def max_err(outs, refs):
    return max(float(np.abs(o - r).max()) for o, r in zip(outs, refs))

ok = True
TOL = 1e-4
rng = np.random.default_rng(0)

# ════════════════════════════════════════════════════════════════
# Reference computations (independent numpy)
# ════════════════════════════════════════════════════════════════
def softmax_np(z):
    e = np.exp(z); return e / e.sum(1, keepdims=True)

def mlp_fwd_ref(x, W0, b0, W1, b1, W2, b2):
    """Forward map mlpForward: relu(relu(x·W0+b0)·W1+b1)·W2+b2."""
    a0 = np.maximum(x @ W0 + b0, 0.0)
    a1 = np.maximum(a0 @ W1 + b1, 0.0)
    return a1 @ W2 + b2

def loss_cot_ref(logits, onehot):
    """softmax-CE gradient ∂L/∂logits = softmax(logits) − onehot."""
    return softmax_np(logits) - onehot

def conv2d_ref(x, W, b):
    """SAME-pad, stride-1 cross-correlation (the repo's conv2d). NCHW / OIHW."""
    B, IC, H, Wd = x.shape; OC, _, kH, kW = W.shape; pH, pW = (kH-1)//2, (kW-1)//2
    y = np.zeros((B, OC, H, Wd), np.float32)
    for o in range(OC):
        y[:, o] += b[o]
        for c in range(IC):
            for kh in range(kH):
                for kw in range(kW):
                    hs, ws = kh - pH, kw - pW         # shift of this tap
                    for hi in range(H):
                        for wi in range(Wd):
                            r, s = hi + hs, wi + ws
                            if 0 <= r < H and 0 <= s < Wd:
                                y[:, o, hi, wi] += W[o, c, kh, kw] * x[:, c, r, s]
    return y

def conv_input_vjp_ref(W, dy, IC, H, Wd):
    """Adjoint of conv2d wrt the input (independent of the transpose/reverse trick)."""
    OC, _, kH, kW = W.shape; pH, pW = (kH-1)//2, (kW-1)//2; B = dy.shape[0]
    dx = np.zeros((B, IC, H, Wd), np.float32)
    for o in range(OC):
        for c in range(IC):
            for kh in range(kH):
                for kw in range(kW):
                    for hi in range(H):
                        for wi in range(Wd):
                            r, s = hi + kh - pH, wi + kw - pW
                            if 0 <= r < H and 0 <= s < Wd:
                                dx[:, c, r, s] += W[o, c, kh, kw] * dy[:, o, hi, wi]
    return dx

def mlp_back_ref(dy, W0, W1, W2, p0, p1):
    """Input-gradient (VJP) chain dx."""
    return (((dy @ W2.T) * relu_mask(p1)) @ W1.T * relu_mask(p0)) @ W0.T

def mlp_sgd_ref(x, W0, b0, W1, b1, W2, b2, onehot, lr):
    """Forward → loss cotangent → backward (dx + param grads) → SGD; 6 updated params."""
    h0 = x @ W0 + b0; a0 = np.maximum(h0, 0.0)
    h1 = a0 @ W1 + b1; a1 = np.maximum(h1, 0.0)
    dy = softmax_np(a1 @ W2 + b2) - onehot         # loss cotangent (computed)
    dW2 = a1.T @ dy;            db2 = dy.sum(0);  dx2 = dy @ W2.T
    dy1 = (h1 > 0) * dx2;       dW1 = a0.T @ dy1; db1 = dy1.sum(0); dx1 = dy1 @ W1.T
    dy0 = (h0 > 0) * dx1;       dW0 = x.T @ dy0;  db0 = dy0.sum(0)
    return [W0 - lr*dW0, b0 - lr*db0, W1 - lr*dW1, b1 - lr*db1, W2 - lr*dW2, b2 - lr*db2]

# ════════════════════════════════════════════════════════════════
# § CPU checks (the correctness gate — backend-independent numerics)
# ════════════════════════════════════════════════════════════════
# ── linear: dx = dy · W0ᵀ ──
dy = rng.standard_normal((2, 3)).astype(np.float32)
W0 = rng.standard_normal((4, 3)).astype(np.float32)
outs, nb = compile_run("/tmp/linear_back.mlir", "linear_back", [dy, W0])
e = max_err(outs, [dy @ W0.T]); ok &= e < TOL
print(f"linear_back     ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}")

# ── mlp 4→3→3→2 input-gradient ──
B, d0, d1, d2, d3 = 2, 4, 3, 3, 2
dy = rng.standard_normal((B, d3)).astype(np.float32)
W0 = rng.standard_normal((d0, d1)).astype(np.float32)
W1 = rng.standard_normal((d1, d2)).astype(np.float32)
W2 = rng.standard_normal((d2, d3)).astype(np.float32)
p0 = rng.standard_normal((B, d1)).astype(np.float32)
p1 = rng.standard_normal((B, d2)).astype(np.float32)
mlp_back_args = [dy, W0, W1, W2, p0, p1]
outs, nb = compile_run("/tmp/mlp_back.mlir", "mlp_back", mlp_back_args)
e = max_err(outs, [mlp_back_ref(dy, W0, W1, W2, p0, p1)]); ok &= e < TOL
print(f"mlp_back        ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}")

# ── mlp forward (proof-backed: ⟦emitMlpFwd⟧ = mlpForward) ──
LR = 0.1
x  = rng.standard_normal((B, d0)).astype(np.float32)
b0 = rng.standard_normal((d1,)).astype(np.float32)
b1 = rng.standard_normal((d2,)).astype(np.float32)
b2 = rng.standard_normal((d3,)).astype(np.float32)
fwd_args = [x, W0, b0, W1, b1, W2, b2]
logits = mlp_fwd_ref(x, W0, b0, W1, b1, W2, b2)
outs, nb = compile_run("/tmp/mlp_fwd.mlir", "mlp_fwd", fwd_args)
e = max_err(outs, [logits]); ok &= e < TOL
print(f"mlp_fwd         ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}")

# ── softmax-CE loss cotangent dy = softmax(logits) − onehot (proof-backed) ──
labels = rng.integers(0, d3, size=B)
onehot = np.eye(d3, dtype=np.float32)[labels]
outs, nb = compile_run("/tmp/loss_cot.mlir", "loss_cot", [logits, onehot])
e = max_err(outs, [loss_cot_ref(logits, onehot)]); ok &= e < TOL
print(f"loss_cot        ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}")

# ── mlp full SGD train step: forward → loss → backward → SGD; 6 updated params ──
ts_args = [x, W0, b0, W1, b1, W2, b2, onehot]
ts_ref = mlp_sgd_ref(x, W0, b0, W1, b1, W2, b2, onehot, LR)
outs, nb = compile_run("/tmp/mlp_train_step.mlir", "mlp_train_step", ts_args)
e = max_err(outs, ts_ref); ok &= e < TOL
print(f"mlp_train_step  ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  "
      f"(6 updated params vs numpy, loss cotangent computed in-module)")

# ── CNN conv (Phase 3): SAME 3×3, 1→2 channels, 4×4 — forward + backward ──
cic, coc, cH, cW, ckH, ckW = 1, 2, 4, 4, 3, 3
cx = rng.standard_normal((1, cic, cH, cW)).astype(np.float32)
cW_ = rng.standard_normal((coc, cic, ckH, ckW)).astype(np.float32)
cb = rng.standard_normal((coc,)).astype(np.float32)
outs, nb = compile_run("/tmp/conv_fwd.mlir", "conv_fwd", [cx, cW_, cb])
e = max_err(outs, [conv2d_ref(cx, cW_, cb)]); ok &= e < TOL
print(f"conv_fwd        ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (stablehlo.convolution = conv2d)")
cdy = rng.standard_normal((1, coc, cH, cW)).astype(np.float32)
outs, nb = compile_run("/tmp/conv_back.mlir", "conv_back", [cdy, cW_])
e = max_err(outs, [conv_input_vjp_ref(cW_, cdy, cic, cH, cW)]); ok &= e < TOL
print(f"conv_back       ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (transpose+reverse+conv = proven conv VJP)")

print("ALL PASS (cpu)" if ok else "FAILURES (cpu)")

# ════════════════════════════════════════════════════════════════
# § Best-effort GPU check: same modules, ROCm/HIP backend, real device
# CPU above is the gate (correctness is backend-independent); this just
# confirms the proof-backed backward + train step also run on the GPU.
# ════════════════════════════════════════════════════════════════
def gpu_arch():
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
        extra = [f"--iree-hip-target={arch}"]
        outs, _ = compile_run("/tmp/mlp_back.mlir", "mlp_back", mlp_back_args,
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, [mlp_back_ref(dy, W0, W1, W2, p0, p1)]); ok &= eg < TOL
        print(f"mlp_back        (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/mlp_train_step.mlir", "mlp_train_step", ts_args,
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, ts_ref); ok &= eg < TOL
        print(f"mlp_train_step  (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/conv_back.mlir", "conv_back", [cdy, cW_],
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, [conv_input_vjp_ref(cW_, cdy, cic, cH, cW)]); ok &= eg < TOL
        print(f"conv_back       (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
    else:
        print("gpu: no hip device — skipped (cpu check is the gate)")
except Exception as e:
    print(f"gpu: skipped ({type(e).__name__}: {e})")

# Hard-exit past IREE's nanobind teardown, whose at-exit "leaked instances"
# report is harmless noise that would otherwise bury the PASS lines.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if ok else 1)
