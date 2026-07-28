"""Minimal repro: XLA/ROCm fails to enqueue a large-filter convolution.

Both convs below are taken verbatim from a ViT-Tiny patch-embed training graph
(16x16/s16 patchify, 224x224 input, batch 32). The first is the forward; the
second is its weight gradient, expressed the standard way -- contract over the
batch, with the (dilated) output cotangent acting as the filter. That makes the
filter 209x209, which is unusual but valid.
"""
import sys, jax, jax.numpy as jnp
from jax import lax
print("jax", jax.__version__, "devices", jax.devices())

def run(name, lhs_shape, rhs_shape):
    lhs = jnp.ones(lhs_shape, jnp.float32)
    rhs = jnp.ones(rhs_shape, jnp.float32)
    dn = lax.conv_dimension_numbers(lhs.shape, rhs.shape, ('NCHW', 'OIHW', 'NCHW'))
    f = jax.jit(lambda a, b: lax.conv_general_dilated(
        a, b, window_strides=(1, 1), padding=((0, 0), (0, 0)),
        lhs_dilation=(1, 1), rhs_dilation=(1, 1), dimension_numbers=dn))
    try:
        out = f(lhs, rhs)
        out.block_until_ready()
        print(f"  {name:22s} {lhs_shape} x {rhs_shape} -> {out.shape}  OK  sum={float(jnp.sum(out)):.4g}")
        return True
    except Exception as e:
        print(f"  {name:22s} {lhs_shape} x {rhs_shape}  FAILED\n    {type(e).__name__}: {str(e)[:400]}")
        return False

ok1 = run("forward patch-embed", (32, 3, 224, 224), (192, 3, 16, 16))
ok2 = run("patch-embed wgrad",   (3, 32, 224, 224), (192, 32, 209, 209))
print(f"\nforward OK: {ok1}   weight-grad OK: {ok2}")
sys.exit(0 if (ok1 and ok2) else 1)
