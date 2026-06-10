#!/usr/bin/env python3
"""Empirical probe of the FloatBridge/SgdDescent hypotheses on a real
MNIST-MLP training run (784-512-512-10, ReLU, softmax-CE, SGD lr=0.1,
batch 128 -- the MainMnistMlpVerified config).

Per-step coupled measurement (the theorems are single-step): the f32
trajectory is the trainer; at sampled steps we recompute forward+backward
in f64 at the SAME params/batch and measure exactly the quantities the
theorems hypothesize or budget:

  - logit drift  delta = max|z32 - z64|        (vs mnist_cot_budget's 1/100
                                                and the worst-case 3/4)
  - cotangent    max|g32 - g64|                (vs 21/1000)
  - SGD step     max|step32 - step64|          (vs the W2 budget 1/300)
  - ReLU margins: min |p64| per layer, and DIRECT count of sign
    disagreements between p32 and p64 (mask flips -- the quantitative
    margin hypothesis E < |p_i| is exactly "no flips")
  - weight magnitudes max|W| (vs the magnitude hypotheses)

Usage: scripts/margin_probe.py [mnist-idx-data-dir]

Reference run (2026-06-10, seed 0, alongside the verified GPU trainer at
97.76%): twin acc 97.88%; logit drift max 1.6e-5 (vs delta=1e-2 hyp,
worst-case 3/4); cotangent dev 2.2e-6 (vs 21/1000); W2 step dev 7.5e-9;
ReLU flips 0/29.5M per layer; trained max|W| = 0.52 (the numeric capstones
use |W| <= 3/5 to cover it).
"""
import struct, sys
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skoonce/lean/claude_max/mnist-lean4/data"

def load_idx_images(path):
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(n*r*c), dtype=np.uint8).reshape(n, r*c)

def load_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(n), dtype=np.uint8)

Xtr = (load_idx_images(f"{DATA}/train-images-idx3-ubyte") / 255.0).astype(np.float32)
ytr = load_idx_labels(f"{DATA}/train-labels-idx1-ubyte").astype(np.int64)
Xte = (load_idx_images(f"{DATA}/t10k-images-idx3-ubyte") / 255.0).astype(np.float32)
yte = load_idx_labels(f"{DATA}/t10k-labels-idx1-ubyte").astype(np.int64)

rng = np.random.default_rng(0)
D0, D1, D2, D3 = 784, 512, 512, 10
LR, B, EPOCHS = np.float32(0.1), 128, 12
PROBE_EVERY = 25  # steps

def he(fan_in, shape):
    return (rng.standard_normal(shape) * np.sqrt(2.0/fan_in)).astype(np.float32)

W0, b0 = he(D0, (D0, D1)), np.zeros(D1, np.float32)
W1, b1 = he(D1, (D1, D2)), np.zeros(D2, np.float32)
W2, b2 = he(D2, (D2, D3)), np.zeros(D3, np.float32)

def fwdback(X, y, W0, b0, W1, b1, W2, b2, dt):
    """One batched forward+backward in dtype dt. Returns preacts, logits,
    cotangent, and the param-gradient dict (mean over batch)."""
    X = X.astype(dt); W0, b0 = W0.astype(dt), b0.astype(dt)
    W1, b1 = W1.astype(dt), b1.astype(dt); W2, b2 = W2.astype(dt), b2.astype(dt)
    p0 = X @ W0 + b0;  a1 = np.maximum(p0, dt(0))
    p1 = a1 @ W1 + b1; a2 = np.maximum(p1, dt(0))
    z  = a2 @ W2 + b2
    zs = z - z.max(axis=1, keepdims=True)
    e  = np.exp(zs); s = e / e.sum(axis=1, keepdims=True)
    g  = s.copy(); g[np.arange(len(y)), y] -= dt(1)   # softmax - onehot
    gB = g / dt(len(y))
    dW2 = a2.T @ gB;            db2 = gB.sum(0)
    c1  = (gB @ W2.T) * (p1 > 0)
    dW1 = a1.T @ c1;            db1 = c1.sum(0)
    c0  = (c1 @ W1.T) * (p0 > 0)
    dW0 = X.T @ c0;             db0 = c0.sum(0)
    return dict(p0=p0, p1=p1, z=z, g=g,
                dW0=dW0, db0=db0, dW1=dW1, db1=db1, dW2=dW2, db2=db2)

stats = dict(delta_z=[], delta_g=[], delta_stepW2=[], flips0=[], flips1=[],
             margin0=[], margin1=[], wmax=[], near0=[], near1=[])
step = 0
for ep in range(EPOCHS):
    perm = rng.permutation(len(Xtr))
    for i in range(0, len(Xtr) - B + 1, B):
        idx = perm[i:i+B]; X, y = Xtr[idx], ytr[idx]
        r32 = fwdback(X, y, W0, b0, W1, b1, W2, b2, np.float32)
        if step % PROBE_EVERY == 0:
            r64 = fwdback(X, y, W0, b0, W1, b1, W2, b2, np.float64)
            stats["delta_z"].append(np.abs(r32["z"].astype(np.float64) - r64["z"]).max())
            stats["delta_g"].append(np.abs(r32["g"].astype(np.float64) - r64["g"]).max())
            stats["delta_stepW2"].append(float(LR) * np.abs(
                r32["dW2"].astype(np.float64) - r64["dW2"]).max())
            f0 = int(((r32["p0"] > 0) != (r64["p0"] > 0)).sum())
            f1 = int(((r32["p1"] > 0) != (r64["p1"] > 0)).sum())
            stats["flips0"].append(f0); stats["flips1"].append(f1)
            stats["margin0"].append(np.abs(r64["p0"]).min())
            stats["margin1"].append(np.abs(r64["p1"]).min())
            dz0 = np.abs(r32["p0"].astype(np.float64) - r64["p0"]).max()
            dz1 = np.abs(r32["p1"].astype(np.float64) - r64["p1"]).max()
            stats["near0"].append(int((np.abs(r64["p0"]) < dz0).sum()))
            stats["near1"].append(int((np.abs(r64["p1"]) < dz1).sum()))
            stats["wmax"].append(max(np.abs(W0).max(), np.abs(W1).max(),
                                     np.abs(W2).max()))
        for (P, gname) in ((W0,"dW0"),(b0,"db0"),(W1,"dW1"),(b1,"db1"),
                           (W2,"dW2"),(b2,"db2")):
            P -= LR * r32[gname]
        step += 1
    # epoch eval (f32)
    zte = np.maximum(np.maximum(Xte @ W0 + b0, 0) @ W1 + b1, 0) @ W2 + b2
    acc = (zte.argmax(1) == yte).mean()
    print(f"epoch {ep+1:2d}  test acc {acc*100:.2f}%", flush=True)

n_probes = len(stats["delta_z"])
total_preacts0 = n_probes * B * D1
total_preacts1 = n_probes * B * D2
print("\n=== probe summary over", n_probes, "sampled steps ===")
print(f"logit drift  max|z32-z64|   : max {max(stats['delta_z']):.3e}   "
      f"median {np.median(stats['delta_z']):.3e}   (budget hyp delta=1e-2, worst-case 0.75)")
print(f"cotangent    max|g32-g64|   : max {max(stats['delta_g']):.3e}   "
      f"(budget 21/1000 = 2.1e-2)")
print(f"W2 SGD step  max|d32-d64|   : max {max(stats['delta_stepW2']):.3e}   "
      f"(budget 1/300 = 3.3e-3)")
print(f"ReLU flips   layer0: {sum(stats['flips0'])} / {total_preacts0}   "
      f"layer1: {sum(stats['flips1'])} / {total_preacts1}")
print(f"preacts with |p64| < actual-float-error:  layer0 {sum(stats['near0'])}, "
      f"layer1 {sum(stats['near1'])}  (margin-hypothesis at-risk set)")
print(f"min margin   layer0 {min(stats['margin0']):.3e}   layer1 {min(stats['margin1']):.3e}")
print(f"max|W| over training: {max(stats['wmax']):.3f}   (vs 1/32 = 0.03125 hypothesis)")
