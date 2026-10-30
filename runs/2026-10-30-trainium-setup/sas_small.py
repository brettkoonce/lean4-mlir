import numpy as np, jax, jax.numpy as jnp
from jax import lax
pool = lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
for shape in [(1, 1, 4, 4), (8, 8, 8, 8), (128, 16, 32, 32)]:
    x = np.random.default_rng(1).permutation(np.prod(shape)).reshape(shape).astype(np.float32)
    g = np.arange(1, np.prod(shape) // 4 + 1, dtype=np.float32).reshape(shape[:2] + (shape[2] // 2, shape[3] // 2))
    f = lambda x, g: jax.vjp(pool, x)[1](g)[0]
    c = np.asarray(jax.jit(f, device=jax.devices('cpu')[0])(x, g))
    try: n = np.asarray(jax.jit(f, device=jax.devices('neuron')[0])(x, g))
    except Exception as e: print(shape, 'neuron error', str(e).splitlines()[0]); continue
    print(shape, 'match' if np.allclose(c, n) else 'MISMATCH', '| sum cpu', c.sum(), 'neuron', n.sum(),
          '| nonzeros cpu', (c != 0).sum(), 'neuron', (n != 0).sum())
    if shape[0] in (1, 8): sl = (0, 0, slice(0, 4), slice(0, 4)); print('x', x[sl], 'cpu', c[sl], 'neuron', n[sl], sep='\n')
    if False: print('x\n', x[0, 0], '\ng\n', g[0, 0], '\ncpu\n', c[0, 0], '\nneuron\n', n[0, 0])
