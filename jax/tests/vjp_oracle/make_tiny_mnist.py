#!/usr/bin/env python3
"""Synthesize a tiny MNIST-format dataset for CI smoke runs.

The real VJP oracle (tests/vjp_oracle/run.sh) diffs phase 3 (Lean -> MLIR ->
IREE, hand-derived VJPs) against phase 2 (Lean -> JAX, autodiff). Phase 3
needs IREE and a GPU, so it cannot run on a stock CI runner. What CAN run
there is phase 2 alone, which exercises the whole Lean -> JAX emitter: shapes,
layer wiring, optimizer, loss. This writes just enough data for that.

Format is plain IDX, matching what Jax/Codegen.lean's load_mnist_* emits:
  images: >4I header (magic 2051, n, rows, cols) then n*rows*cols uint8
  labels: >2I header (magic 2049, n) then n uint8

Deterministic on purpose (fixed seed): a smoke run that flaps is worse than
no smoke run. The pixels are noise — nothing here is learning anything, we
only care that the emitted graph builds, runs, and produces finite numbers.

SIZING, and it is not arbitrary. The emitted `evaluate()` defaults to
batch_size=512 and loops `range(0, len(images) - batch_size + 1, batch_size)`,
so a test split under 512 makes that loop body never execute, leaving
total == 0 and the script dying on `acc1 = correct1 / total` with a
ZeroDivisionError. Hence n_test defaults to 512, not something cuter. The
train split only needs to exceed the net's batchSize (4 in every oracle
case), so 64 gives 16 steps.

Usage: make_tiny_mnist.py <out_dir> [n_train] [n_test]
"""
import os
import struct
import sys

import numpy as np


def write_images(path: str, arr: np.ndarray) -> None:
    n, rows, cols = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack(">4I", 2051, n, rows, cols))
        f.write(arr.astype(np.uint8).tobytes())


def write_labels(path: str, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(">2I", 2049, len(arr)))
        f.write(arr.astype(np.uint8).tobytes())


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "data-tiny"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("t10k", n_test)):
        imgs = rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)
        lbls = rng.integers(0, 10, size=n, dtype=np.uint8)
        write_images(os.path.join(out, f"{split}-images-idx3-ubyte"), imgs)
        write_labels(os.path.join(out, f"{split}-labels-idx1-ubyte"), lbls)
    print(f"tiny mnist: {n_train} train / {n_test} test -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
