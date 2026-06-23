#!/usr/bin/env python3
"""E4M3 (fp8) MNIST-linear TRAINING demo — the Tier-1 precursor.

Companion to `mnist_e4m3_demo.py` (which does fp8 *inference* on an fp32-trained
model). Here we TRAIN in fp8, in the deployed mixed shape:
  - fp32 MASTER weights `W` (only quantized to E4M3 for the matmuls);
  - each step both matmuls run E4M3-leaf / fp32-accumulate:
      forward     z      = (x_q @ W_q) + b
      weight-grad gW     = x_q.T @ g_q       (g = softmax−onehot cotangent)
  - LOSS SCALING `S`: scale the cotangent by S before E4M3-quantizing it, then
    unscale the gradient by S — lifts small gradients out of E4M3's narrow range
    (subnormals flush below 2^-9). This is the trick the relative-only FloatModel
    in FloatBridge.lean CANNOT see (Obstruction 2); here it bites empirically.
  - fp32 master update.

INSTRUMENTATION — the exact precondition of `linear_float_sgd_descends`:
  the theorem's binding "dominance" hypothesis is, with uniform gradient error η,
      η · Σ|grad|  ≤  Σgrad² / 4                                (call the LHS/RHS ratio ρ)
  where η = max_idx |grad_fp8 − grad_fp32| (the uniform accuracy the theorem
  assumes via `linear_grad_close`). ρ ≤ 1 ⟺ the proven descent step applies.
  Tracking η, Σ|grad|, Σgrad² and ρ per epoch shows WHEN fp8 descent is
  provable and WHEN the ~6% leaf roundoff overwhelms the gradient signal
  (the SNR wall — Obstruction 1). This is the number the Tier-1 proof consumes.

Usage: scripts/mnist_e4m3_train_demo.py [mnist-idx-data-dir]
No deps beyond numpy.
"""
import struct, sys
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skoonce/lean/claude_max/mnist-lean4/data"


def load_idx_images(path):
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(n * r * c), dtype=np.uint8).reshape(n, r * c)


def load_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(n), dtype=np.uint8)


Xtr = (load_idx_images(f"{DATA}/train-images-idx3-ubyte") / 255.0).astype(np.float32)
ytr = load_idx_labels(f"{DATA}/train-labels-idx1-ubyte").astype(np.int64)
Xte = (load_idx_images(f"{DATA}/t10k-images-idx3-ubyte") / 255.0).astype(np.float32)
yte = load_idx_labels(f"{DATA}/t10k-labels-idx1-ubyte").astype(np.int64)

D0, D3 = 784, 10
Ytr1h = np.eye(D3, dtype=np.float32)[ytr]


# ── E4M3 fake-quant (identical grid to the inference demo) ───────────────────
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9   # below this, round-to-nearest flushes to 0


def to_e4m3(x):
    x = x.astype(np.float64)
    sign = np.sign(x)
    a = np.minimum(np.abs(x), E4M3_MAX)
    nz = a > 0
    e = np.where(nz, np.floor(np.log2(np.where(nz, a, 1.0))), -7.0)
    e = np.clip(e, -6.0, 8.0)
    step = np.exp2(e - 3.0)
    q = np.round(a / step) * step
    q = np.minimum(q, E4M3_MAX)
    return (sign * q).astype(np.float32)


def quant_per_row(W):
    s = np.abs(W).max(axis=0, keepdims=True) / E4M3_MAX
    s = np.where(s > 0, s, 1.0)
    return to_e4m3(W / s) * s


def quant_per_tensor(x):
    s = np.abs(x).max() / E4M3_MAX
    s = s if s > 0 else 1.0
    return to_e4m3(x / s) * s


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ── gradients: fp32 reference vs E4M3-leaf/fp32-accumulate (+ loss scale) ─────
def fp32_grad(W, b, X, Y1h):
    s = softmax(X @ W + b)
    g = (s - Y1h) / len(X)
    return X.T @ g, g.sum(0)


# Static per-tensor activation scale ⇒ quantize activations ONCE (real fp8
# inference uses a calibrated static scale; this also keeps the demo fast).
_SX = np.abs(Xtr).max() / E4M3_MAX
Xtr_q = (to_e4m3(Xtr / _SX) * _SX).astype(np.float32)
Xte_q = (to_e4m3(Xte / _SX) * _SX).astype(np.float32)


def fp8_grad(W, b, Xq, Y1h, S):
    """`Xq` is the pre-E4M3-quantized activation tensor (static act scale)."""
    W_q = quant_per_row(W)
    s = softmax(Xq @ W_q + b)           # fp32 accumulate, fp32 softmax
    g = (s - Y1h) / len(Xq)             # cotangent, fp32, in [-1,1]/N
    g_q = quant_per_tensor(g * S)       # scale → quantize (lift out of subnormals)
    gW = (Xq.T @ g_q) / S               # fp32 accumulate, then unscale
    return gW, g.sum(0)


def evaluate(W, b, fp8_infer):
    z = (Xte_q @ quant_per_row(W) + b) if fp8_infer else (Xte @ W + b)
    return (z.argmax(1) == yte).mean()


def train(fp8, S=1024.0, epochs=20, lr=0.1, B=128, seed=0, instrument=False):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((D0, D3)) * np.sqrt(1.0 / D0)).astype(np.float32)
    b = np.zeros(D3, np.float32)
    log = []
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - B + 1, B):
            idx = perm[i:i + B]
            Y1h = Ytr1h[idx]
            if fp8:
                gW, gb = fp8_grad(W, b, Xtr_q[idx], Y1h, S)
            else:
                gW, gb = fp32_grad(W, b, Xtr[idx], Y1h)
            W -= lr * gW
            b -= lr * gb
        if instrument:
            # full-training-set gradient at current W: the descent precondition here
            gW32, _ = fp32_grad(W, b, Xtr, Ytr1h)
            gW8, _ = fp8_grad(W, b, Xtr_q, Ytr1h, S)
            eta = np.abs(gW8 - gW32).max()
            sum_abs = np.abs(gW32).sum()
            sum_sq = (gW32 ** 2).sum()
            rho = (eta * sum_abs) / (sum_sq / 4.0) if sum_sq > 0 else float("inf")
            # The tighter, L2 inexact-gradient quantities (the descent condition
            # that should actually be non-vacuous):
            #   rel_l2  = ‖e‖₂ / ‖g‖₂          (need < 1 for a descent direction)
            #   align   = ⟨g8, g32⟩ / ‖g32‖²   (the inexact-grad descent coeff; >0 ⟹ descends)
            err = gW8 - gW32
            rel_l2 = np.sqrt((err ** 2).sum() / sum_sq) if sum_sq > 0 else float("inf")
            align = (gW8 * gW32).sum() / sum_sq if sum_sq > 0 else 0.0
            log.append((ep + 1, evaluate(W, b, fp8), eta, sum_sq, rho, rel_l2, align))
    return W, b, log


# ── 1. fp32-trained vs fp8-trained, final accuracy ───────────────────────────
print("=== Training MNIST-linear: fp32 vs E4M3 (fp32 master + loss scale) ===")
W32, b32, _ = train(fp8=False)
W8, b8, log = train(fp8=True, S=1024.0, instrument=True)
print(f"fp32-trained        test acc (fp32 infer): {evaluate(W32, b32, False)*100:.2f}%")
print(f"fp8-trained         test acc (fp32 infer): {evaluate(W8,  b8,  False)*100:.2f}%")
print(f"fp8-trained         test acc (fp8  infer): {evaluate(W8,  b8,  True )*100:.2f}%")

# ── 2. per-epoch descent precondition (ρ = η·Σ|grad| / (Σgrad²/4)) ────────────
print("\n=== fp8 training: descent conditions, loose (ρ, L1) vs tight (L2 inexact-grad) ===")
print(" ep   fp8acc   η=max|Δg|    Σg²       ρ(L1)   ‖e‖₂/‖g‖₂   align=⟨g8,g32⟩/‖g32‖²")
for ep, acc, eta, ss, rho, rel_l2, align in log:
    print(f"{ep:3d}   {acc*100:5.2f}%  {eta:.2e}  {ss:.2e}  {rho:6.2f}    {rel_l2:7.4f}      {align:7.4f}")
print(" (descent holds for small lr whenever align>0 AND ‖e‖₂/‖g‖₂<1 — the L2 condition;")
print("  ρ≤1 is the *current* theorem's far stricter, and here near-always-violated, hypothesis)")

# ── 3. loss-scaling sweep (Obstruction 2: relative-only model can't see this) ─
print("\n=== loss-scale sweep — final fp8 acc + final-epoch η (underflow effect) ===")
print(f"  (E4M3 smallest subnormal = 2^-9 = {E4M3_MIN_SUBNORMAL:.2e}; gradients below it flush to 0)")
for S in (1.0, 16.0, 256.0, 4096.0, 65536.0):
    Ws, bs, lg = train(fp8=True, S=S, instrument=True)
    _, _, eta_f, _, rho_f, rel_f, _ = lg[-1]
    print(f"  S = {S:8.0f} : fp8-trained acc {evaluate(Ws, bs, False)*100:5.2f}%   "
          f"final-epoch η {eta_f:.3e}   ‖e‖₂/‖g‖₂ {rel_f:.4f}")
print("  (identical across S: per-tensor *dynamic* gradient scaling makes the loss scale")
print("   cancel — quant_per_tensor(S·g) = S·quant_per_tensor(g). Loss scaling only bites")
print("   under a STATIC gradient scale; that underflow regime is the Tier-2/Obstruction-2 study.)")
