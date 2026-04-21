"""Minimal repro #1: just import JAX and list devices."""
import jax
print("jax:", jax.__version__)
print("devices:", jax.devices())
