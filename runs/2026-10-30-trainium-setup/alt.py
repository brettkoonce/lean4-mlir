import numpy as np, jax, jax.numpy as jnp
from jax import lax
cpu, neu = jax.devices('cpu')[0], jax.devices('neuron')[0]
P = [(1, 1), (1, 1)]
cv = lambda l, r: lax.conv_general_dilated(l, r, (1, 1), P, dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
T = lambda x: jnp.transpose(x, (1, 0, 2, 3))
up = lambda g: jnp.repeat(jnp.repeat(g, 2, axis=2), 2, axis=3)
U = np.kron(np.eye(2, dtype=np.float32), np.ones((1, 2), np.float32)).reshape(2, 4)  # (h, H): U[h, H] = 1 iff H//2 == h
U = np.zeros((2, 4), np.float32); U[[0, 0, 1, 1], [0, 1, 2, 3]] = 1
up_dot = lambda g: jnp.einsum('nchw,hH,wW->ncHW', g, U, U)
def up_pads(g):
    z = jnp.zeros((), g.dtype); acc = 0
    for i in (0, 1):
        for j in (0, 1): acc = acc + lax.pad(g, z, ((0, 0, 0), (0, 0, 0), (i, 1 - i, 1), (j, 1 - j, 1)))
    return acc
V = {
 'baseline (T + conv)': lambda x, g: cv(T(x), T(up(g))),
 'dim numbers, no transposes': lambda x, g: lax.conv_general_dilated(x, up(g), (1, 1), P, dimension_numbers=('CNHW', 'IOHW', 'NCHW')),
 'upsample as dot_general': lambda x, g: cv(T(x), T(up_dot(g))),
 'upsample as 4 pads summed': lambda x, g: cv(T(x), T(up_pads(g))),
 'dim numbers + dot upsample': lambda x, g: lax.conv_general_dilated(x, up_dot(g), (1, 1), P, dimension_numbers=('CNHW', 'IOHW', 'NCHW')),
}
rng = np.random.default_rng(0)
for shp in [(128, 32, 4), (128, 16, 16)]:
    n, c, h = shp
    x = rng.standard_normal((n, c, h, h)).astype(np.float32); g = rng.standard_normal((n, c, h // 2, h // 2)).astype(np.float32)
    if h != 4:
        U = np.zeros((h // 2, h), np.float32); U[np.arange(h) // 2, np.arange(h)] = 1
    for k, f in V.items():
        want = np.asarray(jax.jit(f, device=cpu)(x, g))
        try: got = np.asarray(jax.jit(f, device=neu)(jax.device_put(x, neu), jax.device_put(g, neu)))
        except Exception as ex: print(f'ERR   {shp} {k}: {str(ex).splitlines()[0][:90]}', flush=True); continue
        e = np.linalg.norm(got - want) / np.linalg.norm(want)
        print(f'{"WRONG" if e > 1e-4 else "ok   "} {str(shp):14s} {k:28s} {e:.2e}', flush=True)
