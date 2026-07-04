"""P6 — trained-weight gains for ViT-Tiny (LayerNorm ⇒ no frozen-stats step).

Loads the trained verified ViT checkpoint (.lake/build/vit_adam_ckpt.bin, first
nP of [params, adam_m, adam_v], params in the vit_fwd render arg order) and a
real Imagenette val batch (ImageNet-normalized), injects them into
`vit_probe_ops(weights=…)`, and re-measures the op-gran + tree-reduce budget at
the trained net's real operating point. ViT uses LayerNorm (computed per-input at
eval), so unlike r34 there is NO frozen-BN-stats reconstruction — just load and
measure. Gate: the trained forward must classify the val batch well above 10%.
"""
import os
import re
import numpy as np
import jax

import adjoint_chain_probe as P
from adjoint_chain_probe import run_iree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, ".lake/build/vit_adam_ckpt.bin")
VAL = "/home/skoonce/lean/claude_max/lean4-jax/data/imagenette/val.bin"
mlir = open(os.path.join(ROOT, "verified_mlir/vit_fwd.mlir")).read()
hdr = mlir[:mlir.index("{", mlir.index("func.func"))]
sig = [(m.group(1), tuple(int(d) for d in m.group(2).split("x")[:-1]))
       for m in re.finditer(r"%(\w+): tensor<([^>]+)>", hdr)]

# ── trained params → render arg names (order = ckpt param order) ────────────
param_args = [(n, sh) for n, sh in sig if n != "x"]
nP = sum(int(np.prod(sh)) for _, sh in param_args)
flat = np.fromfile(CKPT, dtype=np.float32, count=nP)
vals, off = {}, 0
for n, sh in param_args:
    k = int(np.prod(sh))
    vals[n] = flat[off:off + k].reshape(sh)
    off += k

# ── real Imagenette val batch (4-byte header, class-sorted, ImageNet-norm) ──
REC = 1 + 3 * 224 * 224
NREC = 3925
allb = np.fromfile(VAL, dtype=np.uint8, offset=4).reshape(NREC, REC)
idx = np.arange(0, NREC, NREC // 32)[:32]
recs = allb[idx]
labels = recs[:, 0].astype(np.int64)
mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
imgs = ((recs[:, 1:].reshape(-1, 3, 224, 224).astype(np.float32) / 255.0)
        - mean) / std
vals["x"] = imgs.reshape(32, 150528).astype(np.float32)

# ── gate: the trained render must classify the val batch ────────────────────
out = run_iree(mlir, [vals[n] for n, _ in sig], fn="vit_fwd", module_name="m")
acc = float((np.asarray(out).argmax(1) == labels).mean())
print(f"GATE: trained ViT top-1 on 32 val = {acc:.1%} (random = 10%)")
if acc < 0.25:
    raise SystemExit("gate failed — operating point wrong, aborting")

# ── measure the trained op-gran + tree-reduce budget ────────────────────────
P.vit_probe_ops(tree_reduce=True, weights=vals)
