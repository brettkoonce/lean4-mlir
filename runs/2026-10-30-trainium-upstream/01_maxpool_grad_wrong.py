"""The gradient of a 2x2/stride-2 max pool is wrong on Neuron (trn1). jax.vjp of lax.reduce_window(max),
which lowers to stablehlo.select_and_scatter, compared with JAX's CPU backend. At small shapes every
window's gradient lands on its top-left element instead of its max (the total is right); at larger
shapes it lands on more positions than there are windows and the total is wrong. NEURON_CC_FLAGS=--auto-cast=none.
Inputs are distinct values (a permutation), so the max of each window is unique."""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
cpu, neu = jax.devices('cpu')[0], jax.devices('neuron')[0]
pool = lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
grad = lambda x, g: jax.vjp(pool, x)[1](g)[0]
print(f'jax {jax.__version__}, {neu}')
for shape in [(8, 8, 8, 8), (32, 16, 16, 16), (128, 16, 32, 32)]:
    x = np.random.default_rng(1).permutation(np.prod(shape)).reshape(shape).astype(np.float32)
    g = np.random.default_rng(2).standard_normal(shape[:2] + (shape[2] // 2, shape[3] // 2)).astype(np.float32)
    want = np.asarray(jax.jit(grad, device=cpu)(x, g))
    got = np.asarray(jax.jit(grad, device=neu)(jax.device_put(x, neu), jax.device_put(g, neu)))
    e = np.linalg.norm(got - want) / np.linalg.norm(want)
    print(f'{"WRONG" if e > 1e-5 else "ok   "} {shape}  rel err {e:.2e}  nonzeros cpu {(want != 0).sum()} neuron {(got != 0).sum()}'
          f'  (windows {g.size})')
# The smallest case, one 2x2 window: the compiler fails (INTERNAL) on trn1 rather than miscompiling.
x = np.array([[[[3., 1.], [4., 2.]]]], np.float32); g = np.ones((1, 1, 1, 1), np.float32)
try:
    got = np.asarray(jax.jit(grad, device=neu)(jax.device_put(x, neu), jax.device_put(g, neu)))[0, 0].tolist()
except Exception as e:
    got = f'error: {str(e).splitlines()[0][:80]}'
print('one window x =', x[0, 0].tolist(), '| cpu grad', np.asarray(jax.jit(grad, device=cpu)(x, g))[0, 0].tolist(), '| neuron grad', got)
