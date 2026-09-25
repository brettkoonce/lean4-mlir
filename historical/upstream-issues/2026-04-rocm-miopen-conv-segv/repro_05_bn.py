"""Minimal repro #5: conv2d + batch-norm reduction (matches failing case)."""
import jax, jax.numpy as jnp
@jax.jit
def step(x, w, gamma, beta):
    x = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var  = jnp.var(x,  axis=(0, 2, 3), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    return x * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)
x     = jnp.ones((128, 1, 28, 28))
w     = jnp.ones((32, 1, 3, 3))
gamma = jnp.ones((32,))
beta  = jnp.zeros((32,))
y = step(x, w, gamma, beta)
print("shape:", y.shape, "sum:", y.sum())
