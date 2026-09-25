"""Minimal repro #2: allocate a tiny GPU array."""
import jax.numpy as jnp
x = jnp.arange(4)
print(x)
print(x.device)
