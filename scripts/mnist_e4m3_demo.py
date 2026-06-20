#!/usr/bin/env python3
"""E4M3 (fp8) MNIST-linear demo — planning/floatbridge_quantization.md §3a.

MNIST-linear (784->10) is DEPTH-1: a single matmul, so the per-matmul fp8
leaf bound IS a non-vacuous end-to-end accuracy bound. This is the one
realistic fp8 case where end-to-end accuracy is honestly provable, and the
empirical "it works" headline for the FloatBridge two-roundoff story
(`dot_close_mixed`): bf16/fp8 leaf compute + fp32 accumulate.

What it does, mirroring the deployed mixed-precision shape:
  1. Train an fp32 MNIST-linear baseline (plain SGD, softmax-CE).
  2. Fake-quantize the trained weights AND the test activations to E4M3:
       - per-ROW weight scale, per-TENSOR activation scale (block scaling),
       - fp32 ACCUMULATE (x_q @ W_q), fp32 softmax.
     This is exactly the `dotMixed` model: u_leaf = E4M3, u_acc = fp32.
  3. Measure: top-1 (fp32 vs E4M3), prediction agreement, and the fp32
     logit-margin distribution -- the input to the §3c argmax-preservation
     bound ("same prediction whenever the fp32 margin exceeds 2B").

E4M3 = 1-4-3 (sign, 4 exp, 3 mantissa), bias 7: max finite 448, normals down
to 2^-6, subnormals down to 2^-9. Relative rounding u_leaf ~ 2^-4 = 6.25%.

Usage: scripts/mnist_e4m3_demo.py [mnist-idx-data-dir]
No deps beyond numpy. Reference baseline: fp32 linear ~92% test acc.
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


# ── E4M3 fake-quant ────────────────────────────────────────────────────────
E4M3_MAX = 448.0  # S.1111.110 = 2^8 * 1.75


def to_e4m3(x):
    """Round to the nearest E4M3-representable value (round-to-nearest).
    Per-binade mantissa step 2^(e-3); subnormals share the e=-6 grid
    (step 2^-9); clamp magnitude to 448. Saturating, no inf/nan."""
    x = x.astype(np.float64)
    sign = np.sign(x)
    a = np.minimum(np.abs(x), E4M3_MAX)
    nz = a > 0
    e = np.where(nz, np.floor(np.log2(np.where(nz, a, 1.0))), -7.0)
    e = np.clip(e, -6.0, 8.0)          # normal exponent range (subnormals -> e=-6)
    step = np.exp2(e - 3.0)            # 3-bit mantissa LSB
    q = np.round(a / step) * step
    q = np.minimum(q, E4M3_MAX)
    return (sign * q).astype(np.float32)


def quant_per_row(W):
    """Per-output-column (row of Wt) E4M3 weight quant: scale each column so its
    max magnitude maps to E4M3_MAX, then fake-quant. Returns dequantized W."""
    s = np.abs(W).max(axis=0, keepdims=True) / E4M3_MAX     # (1, n_out)
    s = np.where(s > 0, s, 1.0)
    return to_e4m3(W / s) * s


def quant_per_tensor(x):
    s = np.abs(x).max() / E4M3_MAX
    s = s if s > 0 else 1.0
    return to_e4m3(x / s) * s


# ── 1. fp32 MNIST-linear baseline ──────────────────────────────────────────
rng = np.random.default_rng(0)
D0, D3 = 784, 10
LR, B, EPOCHS = np.float32(0.1), 128, 20
W = (rng.standard_normal((D0, D3)) * np.sqrt(1.0 / D0)).astype(np.float32)
b = np.zeros(D3, np.float32)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


for ep in range(EPOCHS):
    perm = rng.permutation(len(Xtr))
    for i in range(0, len(Xtr) - B + 1, B):
        idx = perm[i:i + B]
        X, y = Xtr[idx], ytr[idx]
        s = softmax(X @ W + b)
        g = s.copy()
        g[np.arange(len(y)), y] -= 1.0
        g /= len(y)
        W -= LR * (X.T @ g)
        b -= LR * g.sum(0)
    acc = ((Xte @ W + b).argmax(1) == yte).mean()
    print(f"epoch {ep + 1:2d}  fp32 test acc {acc * 100:.2f}%", flush=True)


# ── 2. E4M3 inference (per-row W scale, per-tensor act scale, fp32 accum) ────
W_q = quant_per_row(W)
Xte_q = quant_per_tensor(Xte)

z_fp32 = Xte @ W + b
z_e4m3 = Xte_q @ W_q + b                  # fp32 accumulate, fp32 bias

pred_fp32 = z_fp32.argmax(1)
pred_e4m3 = z_e4m3.argmax(1)
acc_fp32 = (pred_fp32 == yte).mean()
acc_e4m3 = (pred_e4m3 == yte).mean()
agree = (pred_fp32 == pred_e4m3).mean()

print("\n=== E4M3 (fp8) vs fp32, MNIST-linear (depth-1) ===")
print(f"fp32  test acc          : {acc_fp32 * 100:.2f}%")
print(f"E4M3  test acc          : {acc_e4m3 * 100:.2f}%")
print(f"drop                    : {(acc_fp32 - acc_e4m3) * 100:+.2f} pts")
print(f"pred agreement fp32==E4M3: {agree * 100:.2f}%")

# logit perturbation: the empirical per-logit |z_e4m3 - z_fp32| (the "B" of §3c)
dz = np.abs(z_e4m3 - z_fp32)
print(f"\nlogit drift |z_e4m3 - z_fp32|: mean {dz.mean():.4f}  max {dz.max():.4f}  "
      f"p99 {np.percentile(dz, 99):.4f}")

# ── 3. argmax-preservation (§3c): same prediction when fp32 margin > 2B ─────
zs = np.sort(z_fp32, axis=1)
margin = zs[:, -1] - zs[:, -2]            # top1 - top2 of the fp32 logits
Bemp = dz.max()                           # a-posteriori per-logit bound
print(f"\nfp32 logit margin: mean {margin.mean():.3f}  median "
      f"{np.median(margin):.3f}  min {margin.min():.4f}")
print(f"empirical per-logit bound B = max|dz| = {Bemp:.4f}  (2B = {2 * Bemp:.4f})")
for frac_name, thr in (("2B (a-posteriori)", 2 * Bemp), ("0.5", 0.5), ("1.0", 1.0)):
    safe = (margin > thr).mean()
    # on margin>2B inputs the prediction is provably unchanged
    print(f"  test fraction with margin > {frac_name:>18}: {safe * 100:5.2f}%")

# sanity: every margin>2B input indeed kept its prediction
safe_mask = margin > 2 * Bemp
if safe_mask.any():
    kept = (pred_fp32[safe_mask] == pred_e4m3[safe_mask]).mean()
    print(f"\nverified-region check: of the {safe_mask.mean()*100:.1f}% with "
          f"margin>2B, prediction kept on {kept*100:.2f}% (expect 100.00%)")
