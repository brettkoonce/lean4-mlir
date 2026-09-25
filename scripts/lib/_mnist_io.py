"""MNIST for the scripts: the idx readers, 4x4 pooled integer features, and the seed-0 ReLU MLP.

Several generators regenerate committed Lean from these, so everything here is bit-for-bit what
their private copies did: the same reads, the same integer pooling, and the same SGD loop in the
same operation order under the same RNG — a reordered float op changes the trained weights.
"""
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
_FILES = {"train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
          "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")}


def images(path, flat=False):
    """An idx3 image file as uint8 `[n, rows, cols]`, or `[n, rows*cols]` with `flat`."""
    with open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), dtype=np.uint8)
    return x.reshape(n, r * c) if flat else x.reshape(n, r, c)


def labels(path):
    """An idx1 label file as uint8 `[n]`."""
    with open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def mnist(split, flat=False, data=DATA):
    """(images, labels) of the "train" or "test" split under `data`."""
    img, lab = _FILES[split]
    return images(Path(data) / img, flat), labels(Path(data) / lab)


def pool_sums(x):
    """28x28 uint8 -> the 49 integer 4x4 block sums (0..4080) per image, exact."""
    return x.reshape(-1, 7, 4, 7, 4).astype(np.int64).sum(axis=(2, 4)).reshape(-1, 49)


def train_mlp(X, y, H, K, *, epochs=12, lr=0.15, bs=64, cap=None, seed=0):
    """The Lipschitz corpus' two-layer ReLU MLP (`X @ W1.T` -> relu -> `@ W2.T`, no biases):
    He init then minibatch SGD on softmax cross-entropy, one `default_rng(seed)` for the init and
    the per-epoch permutations. With `cap`, each step rescales a weight whose spectral norm
    exceeds it (the capped nets). Returns float64 `(W1 [H, dim], W2 [K, H])`."""
    rng = np.random.default_rng(seed)
    dim = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / dim), (H, dim))
    W2 = rng.normal(0, np.sqrt(2.0 / H), (K, H))
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for b in range(0, len(X), bs):
            xb = X[idx[b:b + bs]]; yb = y[idx[b:b + bs]]
            h = xb @ W1.T; hr = np.maximum(h, 0); z = hr @ W2.T
            z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            g = p.copy(); g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            W1 -= lr * ((g @ W2) * (h > 0)).T @ xb; W2 -= lr * g.T @ hr
            if cap is not None:
                s1 = np.linalg.svd(W1, compute_uv=False)[0]
                s2 = np.linalg.svd(W2, compute_uv=False)[0]
                if s1 > cap: W1 *= cap / s1
                if s2 > cap: W2 *= cap / s2
    return W1, W2
