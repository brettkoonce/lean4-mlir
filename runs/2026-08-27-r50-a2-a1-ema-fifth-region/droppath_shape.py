"""Is the regenerated reference's stochastic depth PER-EXAMPLE, and was it not before?

The two forms have the same expectation, so no mean-based check separates them. What separates
them is that a per-example mask drops DIFFERENT ROWS of one batch and a scalar one drops all or
none. This executes both against the same key and counts distinct row outcomes.

    .venv/bin/python runs/2026-08-27-r50-a2-a1-ema-fifth-region/droppath_shape.py
"""
import os, re, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp

REF = "jax/.lake/build/generated_resnet50_imagenet_a2accum.py"
src = open(REF).read().split("\n")
j0 = next(i for i, l in enumerate(src) if l.startswith("def _drop_branch"))
j1 = next(i for i in range(j0 + 1, len(src)) if src[i] and not src[i].startswith((" ", "\t")))
ns = {"jax": jax, "jnp": jnp}
exec("\n".join(src[j0:j1]), ns)
new = ns["_drop_branch"]


def old(branch, drop_key, keep_prob):
    """The form every convolutional emitter carried until 2026-08-27 — no shape argument."""
    if drop_key is None or keep_prob >= 1.0:
        return branch
    keep = jax.random.bernoulli(drop_key, keep_prob).astype(branch.dtype)
    return branch * keep / keep_prob


B, C, H = 64, 8, 4
x = jnp.ones((B, C, H, H), jnp.float32)
keep_prob = 0.95                              # RSB-A2's deepest block: 1 - 0.05

print("── stochastic-depth mask shape, regenerated reference vs the form it replaced ──")
print(f"  batch {B}, keep_prob {keep_prob}   (RSB-A2 sd 0.05 at the deepest block)")
for label, f in (("per-example (now)", new), ("per-block scalar (was)", old)):
    rows = set()
    dropped_per_step = []
    for step in range(200):
        y = f(x, jax.random.PRNGKey(step), keep_prob)
        r = np.asarray(y[:, 0, 0, 0])         # one scalar per example
        rows.add(tuple(np.unique(r)))
        dropped_per_step.append(int((r == 0.0).sum()))
    d = np.array(dropped_per_step)
    print(f"  {label:<22} distinct row-outcome sets over 200 steps: {len(rows)}")
    print(f"  {'':22}   examples dropped per step: min {d.min()}, max {d.max()}, "
          f"mean {d.mean():.2f}  (expect ~{B * (1 - keep_prob):.1f})")
    print(f"  {'':22}   steps where ALL {B} or NONE dropped: "
          f"{int(((d == 0) | (d == B)).sum())}/200")
print("\n⭐ timm 1.0.28 `drop_path`: shape = (x.shape[0],) + (1,) * (x.ndim - 1) — per sample.")
