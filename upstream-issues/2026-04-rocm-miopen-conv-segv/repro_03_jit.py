"""Minimal repro #3: JIT-compile and run a trivial op."""
import jax, jax.numpy as jnp
f = jax.jit(lambda x: x + 1)
y = f(jnp.arange(4))
print("result:", y)
print("device:", y.device)
