"""Minimal repro #4: JIT-compile a conv2d (matches MNIST-CNN shape)."""
import jax, jax.numpy as jnp
@jax.jit
def step(x, w):
    return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
x = jnp.ones((128, 1, 28, 28))
w = jnp.ones((32, 1, 3, 3))
y = step(x, w)
print("shape:", y.shape, "sum:", y.sum())
