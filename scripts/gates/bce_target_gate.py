#!/usr/bin/env python3
"""bce_target_gate.py — known-answer gate for the shim's BCE target transform (RSB-A2 / A1).

The verified BCE renders take `%onehot` as given, so a recipe's label smoothing and timm's
`--bce-target-thresh` are applied by the shim (`Jax/Codegen.lean`'s `_emit`), after mixing. This
drives each transforming shim and the untransformed `default` R50 shim at one seed and checks:

    images      bit-identical to the default shim's (the transform touches targets only)
    targets     == ((t * (1 - ls) + ls / K) > thresh)   with t the default shim's mixed target

A1's Mixup α differs from the default shim's, so the default shim is driven at A1's α
(`SHIM_MIXUP_ALPHA=0.2`) to draw the same λ stream. `--break` checks the control: the untransformed
target must NOT equal the thresholded one (otherwise the batch had no mixed class to test).

    .venv/bin/python scripts/gates/bce_target_gate.py [--batch 8] [--batches 4] [--break]
"""
import argparse, os, subprocess, sys
import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument("--python", default=os.environ.get("SHIM_PY", ".venv/bin/python3"))
AP.add_argument("--batch", type=int, default=8)
AP.add_argument("--batches", type=int, default=4)
AP.add_argument("--seed", type=int, default=7)
AP.add_argument("--break", dest="brk", action="store_true")
A = AP.parse_args()

K = 1000
FLAT = 3 * 224 * 224
B = "jax/.lake/build/generated_resnet50_imagenet_{}shim.py"
# (shim, label smoothing, threshold, env for the default shim's matching λ stream)
CASES = [("a2accum_", 0.0, 0.2, {}),
         ("a1_", 0.1, 0.2, {"SHIM_MIXUP_ALPHA": "0.2"})]


def stream(script, extra):
    env = dict(os.environ, SHIM_DETERMINISM="1", SHIM_BATCH=str(A.batch), SHIM_SEED=str(A.seed),
               SHIM_NCLASSES=str(K), SHIM_MIX="both", CUDA_VISIBLE_DEVICES="", **extra)
    p = subprocess.Popen([A.python, script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         env=env)

    def rd(n):
        buf = b""
        while len(buf) < n:
            c = p.stdout.read(n - len(buf))
            if not c:
                raise SystemExit(f"{script}: shim closed early ({len(buf)}/{n} bytes)")
            buf += c
        return buf

    assert rd(4) == b"LMSH", "bad preamble magic"
    ver, batch, flat, nc = np.frombuffer(rd(16), dtype=np.int32)
    assert (ver, batch, flat, nc) == (4, A.batch, FLAT, K), f"preamble {(ver, batch, flat, nc)}"
    out = []
    for _ in range(A.batches):
        rows = int(np.frombuffer(rd(4), dtype=np.int32)[0])
        assert rows == batch
        t = np.frombuffer(rd(4 * batch * nc), dtype=np.float32).reshape(batch, nc).copy()
        x = np.frombuffer(rd(4 * batch * flat), dtype=np.float32).reshape(batch, flat).copy()
        out.append((t, x))
    p.kill()
    return out


bad = 0
for tag, ls, thr, env in CASES:
    ref = stream(B.format(""), env)
    got = stream(B.format(tag), {})
    for i, ((t0, x0), (t1, x1)) in enumerate(zip(ref, got)):
        want = ((t0 * np.float32(1.0 - ls) + np.float32(ls / K)) > np.float32(thr)).astype(np.float32)
        img_ok = np.array_equal(x0, x1)
        tgt_ok = np.array_equal(t1, want)
        mixed = int(((t0 > 0) & (t0 < 1)).any(axis=1).sum())
        ones = t1.sum(axis=1)
        print(f"  {tag[:-1]:8s} batch {i}: images {'=' if img_ok else '≠'}, targets "
              f"{'=' if tgt_ok else '≠'}, {mixed}/{A.batch} rows mixed, "
              f"ones/row {ones.min():.0f}–{ones.max():.0f}")
        bad += (not img_ok) + (not tgt_ok)
        if A.brk and np.array_equal(t0, want):
            print(f"  ✗ control: batch {i}'s untransformed target already equals the thresholded one")
            bad += 1

print("✅ BCE target transform matches its reference" if bad == 0 else f"⛔ {bad} check(s) failed")
sys.exit(1 if bad else 0)
