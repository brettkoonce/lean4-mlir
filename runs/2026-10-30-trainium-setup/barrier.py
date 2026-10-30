import numpy as np, jax, jax.numpy as jnp
from jax import lax
cpu, neu = jax.devices('cpu')[0], jax.devices('neuron')[0]
cv = lambda l, r: lax.conv_general_dilated(l, r, (1, 1), [(1, 1), (1, 1)], dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
T = lambda x: jnp.transpose(x, (1, 0, 2, 3))
up = lambda g: jnp.repeat(jnp.repeat(g, 2, axis=2), 2, axis=3)
V = {
 'baseline': lambda x, g: cv(T(x), T(up(g))),
 'barrier after upsample': lambda x, g: cv(T(x), T(lax.optimization_barrier(up(g)))),
 'barrier after transpose': lambda x, g: cv(T(x), lax.optimization_barrier(T(up(g)))),
 'transpose then upsample': lambda x, g: cv(T(x), up(T(g))),
}
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 32, 4, 4)).astype(np.float32); g = rng.standard_normal((128, 32, 2, 2)).astype(np.float32)
for k, f in V.items():
    want = np.asarray(jax.jit(f, device=cpu)(x, g)); got = np.asarray(jax.jit(f, device=neu)(jax.device_put(x, neu), jax.device_put(g, neu)))
    e = np.linalg.norm(got - want) / np.linalg.norm(want)
    print(f'{"WRONG" if e > 1e-4 else "ok   "} {k:26s} {e:.2e}', flush=True)
