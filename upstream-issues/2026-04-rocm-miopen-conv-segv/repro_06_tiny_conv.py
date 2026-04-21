"""Minimal repro #6: tiniest possible conv."""
import jax, jax.numpy as jnp
@jax.jit
def step(x, w):
    return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
x = jnp.ones((1, 1, 4, 4))   # batch 1, 1 channel, 4x4
w = jnp.ones((1, 1, 3, 3))   # 1 output channel, 1 input channel, 3x3
y = step(x, w)
print("shape:", y.shape, "sum:", y.sum())
