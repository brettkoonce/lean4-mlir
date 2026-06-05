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
  • maxpool_fwd / _back       — CNN (Phase 3): maxPool2 as reduce_window(max),
                               and the proven maxpool VJP (select_and_scatter
                               = ⟦maxPoolBackDenote⟧, route dy to argmax).
  • cnn_back                  — CNN capstone: conv→relu→maxpool→flatten→dense
                               forward + the full input-gradient backward chain,
                               every op a rendered proof-backed bridge.
  • cnn_train_step            — full CNN SGD step: 4 updated params; conv weight
                               gradient via the transpose trick (same conv op).
  • bn_fwd / bn_back          — BatchNorm/LayerNorm: reduce/broadcast/rsqrt
                               forward + the proven 3-term rank-1 backward.
  • softmax_fwd / _back       — softmax + the proven rank-1 backward
                               (p⊙(dy−⟨p,dy⟩)); the attention building block.
  • sdpa_fwd / sdpa_back      — scaled dot-product attention softmax(QKᵀ/√d)·V
                               + the proven dQ/dK/dV (the ViT apex).
  • {sigmoid,swish,relu6,gelu} — pointwise activations: forward + the proven
                               diagonal backward dy⊙act'(x).
  • residual / se             — the fan-in chapters: residual add (I + dense),
                               squeeze-excite gate-multiply (se_back_bridge).

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

def maxpool_fwd_ref(x):
    """2×2 stride-2 max pool (the repo's maxPool2), NCHW."""
    B, C, H, W = x.shape
    return x.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))

def maxpool_back_ref(x, dy):
    """Route dy to each 2×2 window's argmax cell (the proven maxPoolBackDenote)."""
    B, C, H, W = x.shape; dx = np.zeros_like(x)
    for b in range(B):
        for c in range(C):
            for i in range(H // 2):
                for j in range(W // 2):
                    win = x[b, c, 2*i:2*i+2, 2*j:2*j+2]
                    di, dj = np.unravel_index(int(np.argmax(win)), (2, 2))
                    dx[b, c, 2*i+di, 2*j+dj] = dy[b, c, i, j]
    return dx

def conv_weight_grad_ref(x, dy, wshape):
    """Conv weight VJP dW[o,c,kh,kw] = Σ_{h,w} x[c,h+kh-p,w+kw-p]·dy[o,h,w] (the
    transpose-trick formula, computed directly — independent of the rendered op)."""
    OC, IC, kH, kW = wshape; pH, pW = (kH-1)//2, (kW-1)//2; H, W = x.shape[2], x.shape[3]
    dW = np.zeros(wshape, np.float32)
    for o in range(OC):
        for c in range(IC):
            for kh in range(kH):
                for kw in range(kW):
                    for h in range(H):
                        for w in range(W):
                            r, s = h + kh - pH, w + kw - pW
                            if 0 <= r < H and 0 <= s < W:
                                dW[o, c, kh, kw] += x[0, c, r, s] * dy[0, o, h, w]
    return dW

def cnn_sgd_ref(x, Wc, bc, Wd, bd, onehot, lr, ic, H, W):
    """Full CNN SGD step: forward → loss → backward → 4 param grads → SGD."""
    hconv = conv2d_ref(x, Wc, bc); a = np.maximum(hconv, 0.0)
    p = maxpool_fwd_ref(a); oc = Wc.shape[0]
    flat = p.reshape(1, -1)
    dlog = softmax_np(flat @ Wd + bd) - onehot
    dWd = flat.T @ dlog; dbd = dlog.sum(0)
    dp = (dlog @ Wd.T).reshape(p.shape)
    da = maxpool_back_ref(a, dp); dhconv = (hconv > 0) * da
    dWc = conv_weight_grad_ref(x, dhconv, Wc.shape); dbc = dhconv.sum(axis=(0, 2, 3))
    return [Wc - lr*dWc, bc - lr*dbc, Wd - lr*dWd, bd - lr*dbd]

def cnn_back_ref(x, Wc, bc, Wd, dy, ic, H, W):
    """CNN input-gradient: conv → relu → maxpool → flatten → dense, then the
    adjoint chain (dense → reshape → maxpool → relu → conv). Independent of the
    rendered ops (uses the loop refs above), so it checks the whole composition."""
    hconv = conv2d_ref(x, Wc, bc)                  # [1, oc, H, W]
    a = np.maximum(hconv, 0.0)                      # relu
    oc = Wc.shape[0]; H2, W2 = H // 2, W // 2
    dflat = dy @ Wd.T                               # [1, oc*H2*W2]
    dp = dflat.reshape(1, oc, H2, W2)               # C-order (= Tensor3.flatten)
    da = maxpool_back_ref(a, dp)                    # [1, oc, H, W]  (operand = a)
    dhconv = (hconv > 0) * da                       # relu backward
    return conv_input_vjp_ref(Wc, dhconv, ic, H, W) # [1, ic, H, W]

def bn_fwd_ref(x, g, b, eps):
    """bnForward over the feature axis: y = γ·(x−μ)/√(σ²+ε) + β (population var)."""
    mu = x.mean(1, keepdims=True); var = ((x - mu) ** 2).mean(1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def bn_back_ref(x, g, dy, eps):
    """Proven 3-term BN backward: dx = (istd/N)·(N·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂)), dx̂=γ·dy."""
    n = x.shape[1]; mu = x.mean(1, keepdims=True); var = ((x - mu) ** 2).mean(1, keepdims=True)
    istd = 1.0 / np.sqrt(var + eps); xhat = (x - mu) * istd
    dxhat = g * dy
    sdx = dxhat.sum(1, keepdims=True); sxdx = (xhat * dxhat).sum(1, keepdims=True)
    return (istd / n) * (n * dxhat - sdx - xhat * sxdx)

def softmax_back_ref(z, dy):
    """Rank-1 softmax backward: dz = p ⊙ (dy − ⟨p,dy⟩)."""
    p = softmax_np(z); return p * (dy - (p * dy).sum(1, keepdims=True))

def sdpa_fwd_ref(Q, K, V, d):
    """sdpa = softmax(QKᵀ/√d)·V (row softmax)."""
    return softmax_np(Q @ K.T / np.sqrt(d)) @ V

def sdpa_back_ref(Q, K, V, dOut, d):
    """Proven SDPA input grads (sdpa_back_Q/K/V)."""
    scale = 1.0 / np.sqrt(d)
    w = softmax_np(Q @ K.T * scale)
    dW = dOut @ V.T
    dV = w.T @ dOut
    dScaled = w * (dW - (w * dW).sum(1, keepdims=True))   # row-softmax VJP
    dScores = dScaled * scale
    return [dScores @ K, dScores.T @ Q, dV]               # dQ, dK, dV

def _sig(x): return 1.0 / (1.0 + np.exp(-x))
def _gelu_parts(x):
    c = np.sqrt(2.0 / np.pi); t = np.tanh(c * (x + 0.044715 * x**3)); return c, t
# (forward, backward dy⊙act'(x)) per activation — derivatives match *ScalarDeriv.
ACT_REF = {
    "sigmoid": (lambda x: _sig(x),
                lambda x, dy: dy * _sig(x) * (1 - _sig(x))),
    "swish":   (lambda x: x * _sig(x),
                lambda x, dy: dy * (_sig(x) * (1 + x * (1 - _sig(x))))),
    "relu6":   (lambda x: np.minimum(np.maximum(x, 0.0), 6.0),
                lambda x, dy: dy * ((x > 0) & (x < 6)).astype(np.float32)),
    "gelu":    (lambda x: 0.5 * x * (1 + _gelu_parts(x)[1]),
                lambda x, dy: dy * (0.5 * (1 + _gelu_parts(x)[1])
                    + 0.5 * x * (1 - _gelu_parts(x)[1]**2) * _gelu_parts(x)[0]
                      * (1 + 3 * 0.044715 * x**2))),
}

def residual_fwd_ref(x, W, b): return x + (x @ W + b)
def residual_back_ref(dy, W): return dy + dy @ W.T            # add fan-in: I + dense
def se_fwd_ref(x): return x * _sig(x)
def se_back_ref(x, dy):
    s = _sig(x); return s * dy + (x * dy) * (s * (1 - s))     # gate⊙dy + gate_back(x⊙dy)

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

# ── CNN maxpool (Phase 3): 2×2 stride-2, 2 ch, 4×4 → 2×2 — forward + backward ──
mpx = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
mpdy = rng.standard_normal((1, 2, 2, 2)).astype(np.float32)
outs, nb = compile_run("/tmp/maxpool_fwd.mlir", "maxpool_fwd", [mpx])
e = max_err(outs, [maxpool_fwd_ref(mpx)]); ok &= e < TOL
print(f"maxpool_fwd     ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (reduce_window max)")
outs, nb = compile_run("/tmp/maxpool_back.mlir", "maxpool_back", [mpx, mpdy])
e = max_err(outs, [maxpool_back_ref(mpx, mpdy)]); ok &= e < TOL
print(f"maxpool_back    ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (select_and_scatter = route dy to argmax)")

# ── CNN capstone (Phase 3): conv→relu→maxpool→flatten→dense, fwd + full dx chain ──
nic, noc, nH, nW, nkH, nkW, ncls = 1, 2, 4, 4, 3, 3, 3
nflat = noc * (nH // 2) * (nW // 2)
nx  = rng.standard_normal((1, nic, nH, nW)).astype(np.float32)
nWc = rng.standard_normal((noc, nic, nkH, nkW)).astype(np.float32)
nbc = rng.standard_normal((noc,)).astype(np.float32)
nWd = rng.standard_normal((nflat, ncls)).astype(np.float32)
ndy = rng.standard_normal((1, ncls)).astype(np.float32)
cnn_args = [nx, nWc, nbc, nWd, ndy]
cnn_ref = cnn_back_ref(nx, nWc, nbc, nWd, ndy, nic, nH, nW)
outs, nb = compile_run("/tmp/cnn_back.mlir", "cnn_back", cnn_args)
e = max_err(outs, [cnn_ref]); ok &= e < TOL
print(f"cnn_back        ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  "
      f"(conv→relu→maxpool→flatten→dense backward, every op proof-backed)")

# ── CNN full SGD train step: 4 updated params (conv dW via transpose trick) ──
nbd = rng.standard_normal((ncls,)).astype(np.float32)
nlabels = rng.integers(0, ncls, size=1)
nonehot = np.eye(ncls, dtype=np.float32)[nlabels]
cts_args = [nx, nWc, nbc, nWd, nbd, nonehot]
cts_ref = cnn_sgd_ref(nx, nWc, nbc, nWd, nbd, nonehot, LR, nic, nH, nW)
outs, nb = compile_run("/tmp/cnn_train_step.mlir", "cnn_train_step", cts_args)
e = max_err(outs, cts_ref); ok &= e < TOL
print(f"cnn_train_step  ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  "
      f"(4 updated params vs numpy; conv dW = transpose trick)")

# ── BatchNorm/LayerNorm (Phase 3 sweep): forward + proven 3-term backward ──
EPS = 1e-5
bnB, bnN = 2, 4
bx = rng.standard_normal((bnB, bnN)).astype(np.float32)
bg = np.float32(rng.standard_normal()); bb = np.float32(rng.standard_normal())
bdy = rng.standard_normal((bnB, bnN)).astype(np.float32)
gsc = np.array(bg, np.float32); bsc = np.array(bb, np.float32)
outs, nb = compile_run("/tmp/bn_fwd.mlir", "bn_fwd", [bx, gsc, bsc])
e = max_err(outs, [bn_fwd_ref(bx, bg, bb, EPS)]); ok &= e < TOL
print(f"bn_fwd          ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (reduce/broadcast/rsqrt = bnForward)")
outs, nb = compile_run("/tmp/bn_back.mlir", "bn_back", [bx, gsc, bdy])
e = max_err(outs, [bn_back_ref(bx, bg, bdy, EPS)]); ok &= e < TOL
print(f"bn_back         ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (proven 3-term rank-1 backward)")

# ── Softmax (Phase 3 sweep): forward + proven rank-1 backward ──
sz = rng.standard_normal((2, 4)).astype(np.float32)
sdy = rng.standard_normal((2, 4)).astype(np.float32)
outs, nb = compile_run("/tmp/softmax_fwd.mlir", "softmax_fwd", [sz])
e = max_err(outs, [softmax_np(sz)]); ok &= e < TOL
print(f"softmax_fwd     ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (exp/reduce/divide = softmax)")
outs, nb = compile_run("/tmp/softmax_back.mlir", "softmax_back", [sz, sdy])
e = max_err(outs, [softmax_back_ref(sz, sdy)]); ok &= e < TOL
print(f"softmax_back    ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (rank-1: p⊙(dy−⟨p,dy⟩))")

# ── Scaled dot-product attention (Phase 3 apex): forward + proven dQ/dK/dV ──
an, ad = 3, 4
aQ = rng.standard_normal((an, ad)).astype(np.float32)
aK = rng.standard_normal((an, ad)).astype(np.float32)
aV = rng.standard_normal((an, ad)).astype(np.float32)
aD = rng.standard_normal((an, ad)).astype(np.float32)
outs, nb = compile_run("/tmp/sdpa_fwd.mlir", "sdpa_fwd", [aQ, aK, aV])
e = max_err(outs, [sdpa_fwd_ref(aQ, aK, aV, ad)]); ok &= e < TOL
print(f"sdpa_fwd        ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (softmax(QKᵀ/√d)·V)")
outs, nb = compile_run("/tmp/sdpa_back.mlir", "sdpa_back", [aQ, aK, aV, aD])
e = max_err(outs, sdpa_back_ref(aQ, aK, aV, aD, ad)); ok &= e < TOL
print(f"sdpa_back       ({nb}B): max_err={e:.2e}  {'PASS' if e < TOL else 'FAIL'}  (dQ,dK,dV vs proven sdpa_back_Q/K/V)")

# ── Pointwise activations (Phase 3 sweep): fwd + proven dy⊙act'(x) backward ──
ax = rng.standard_normal((8,)).astype(np.float32)
ady = rng.standard_normal((8,)).astype(np.float32)
for nm, (fref, bref) in ACT_REF.items():
    outs, nb = compile_run(f"/tmp/{nm}_fwd.mlir", f"{nm}_fwd", [ax])
    e = max_err(outs, [fref(ax).astype(np.float32)]); ok &= e < TOL
    outs2, nb2 = compile_run(f"/tmp/{nm}_back.mlir", f"{nm}_back", [ax, ady])
    e2 = max_err(outs2, [bref(ax, ady).astype(np.float32)]); ok &= e2 < TOL
    print(f"{nm+'_fwd/back':15s} ({nb}/{nb2}B): fwd={e:.2e} back={e2:.2e}  "
          f"{'PASS' if max(e, e2) < TOL else 'FAIL'}  (dy⊙{nm}'(x))")

# ── Residual + SE (Phase 3 sweep): the fan-in chapters ──
rW = rng.standard_normal((4, 4)).astype(np.float32); rb = rng.standard_normal((4,)).astype(np.float32)
rx = rng.standard_normal((2, 4)).astype(np.float32); rdy = rng.standard_normal((2, 4)).astype(np.float32)
outs, nb = compile_run("/tmp/residual_fwd.mlir", "residual_fwd", [rx, rW, rb])
e = max_err(outs, [residual_fwd_ref(rx, rW, rb)]); ok &= e < TOL
outs2, nb2 = compile_run("/tmp/residual_back.mlir", "residual_back", [rdy, rW])
e2 = max_err(outs2, [residual_back_ref(rdy, rW)]); ok &= e2 < TOL
print(f"residual_fwd/back ({nb}/{nb2}B): fwd={e:.2e} back={e2:.2e}  {'PASS' if max(e,e2)<TOL else 'FAIL'}  (add fan-in)")
outs, nb = compile_run("/tmp/se_fwd.mlir", "se_fwd", [ax])
e = max_err(outs, [se_fwd_ref(ax)]); ok &= e < TOL
outs2, nb2 = compile_run("/tmp/se_back.mlir", "se_back", [ax, ady])
e2 = max_err(outs2, [se_back_ref(ax, ady)]); ok &= e2 < TOL
print(f"se_fwd/back     ({nb}/{nb2}B): fwd={e:.2e} back={e2:.2e}  {'PASS' if max(e,e2)<TOL else 'FAIL'}  (gate-multiply fan-in)")

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
        outs, _ = compile_run("/tmp/maxpool_back.mlir", "maxpool_back", [mpx, mpdy],
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, [maxpool_back_ref(mpx, mpdy)]); ok &= eg < TOL
        print(f"maxpool_back    (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/cnn_back.mlir", "cnn_back", cnn_args,
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, [cnn_ref]); ok &= eg < TOL
        print(f"cnn_back        (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/cnn_train_step.mlir", "cnn_train_step", cts_args,
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, cts_ref); ok &= eg < TOL
        print(f"cnn_train_step  (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/bn_back.mlir", "bn_back", [bx, gsc, bdy],
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, [bn_back_ref(bx, bg, bdy, EPS)]); ok &= eg < TOL
        print(f"bn_back         (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/softmax_back.mlir", "softmax_back", [sz, sdy],
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, [softmax_back_ref(sz, sdy)]); ok &= eg < TOL
        print(f"softmax_back    (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
        outs, _ = compile_run("/tmp/sdpa_back.mlir", "sdpa_back", [aQ, aK, aV, aD],
                              backend="rocm", config="hip", extra=extra)
        eg = max_err(outs, sdpa_back_ref(aQ, aK, aV, aD, ad)); ok &= eg < TOL
        print(f"sdpa_back       (hip/{arch}): max_err={eg:.2e}  {'PASS' if eg < TOL else 'FAIL'}")
    else:
        print("gpu: no hip device — skipped (cpu check is the gate)")
except Exception as e:
    print(f"gpu: skipped ({type(e).__name__}: {e})")

# Hard-exit past IREE's nanobind teardown, whose at-exit "leaked instances"
# report is harmless noise that would otherwise bury the PASS lines.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if ok else 1)
