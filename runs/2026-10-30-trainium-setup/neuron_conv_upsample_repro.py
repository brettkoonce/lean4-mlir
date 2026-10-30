"""Neuron miscompile repro (SDK 2.33.0: neuronx-cc 2.28.7631, libneuronxla 3.0.10135, jax-neuronx
0.10.0.1.0.14198; trn1.2xlarge). A nearest-neighbour 2x upsample feeding a convolution whose
batch/feature dims are swapped by transposes (the conv weight-gradient form) returns wrong values.
Each piece alone is correct. NEURON_CC_FLAGS=--auto-cast=none."""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
cpu, neu = jax.devices('cpu')[0], jax.devices('neuron')[0]

def f(x, g):
    # x: (N, C, H, W) activations; g: (N, C, H/2, W/2) cotangent, upsampled 2x (nearest)
    u = jnp.repeat(jnp.repeat(g, 2, axis=2), 2, axis=3)
    l = jnp.transpose(x, (1, 0, 2, 3))         # (C, N, H, W): batch as the feature dim
    r = jnp.transpose(u, (1, 0, 2, 3))
    return lax.conv_general_dilated(l, r, (1, 1), [(1, 1), (1, 1)],
                                    dimension_numbers=('NCHW', 'OIHW', 'NCHW'))   # (C, C, 3, 3)

def no_upsample(x, g):
    u = jnp.asarray(g)
    return f.__wrapped__(x, u) if hasattr(f, '__wrapped__') else None

rng = np.random.default_rng(0)
for n, c, h in [(128, 32, 4), (128, 32, 8), (128, 16, 16), (32, 32, 4), (8, 8, 4), (128, 32, 2)]:
    x = rng.standard_normal((n, c, h, h)).astype(np.float32)
    g = rng.standard_normal((n, c, h // 2, h // 2)).astype(np.float32)
    want = np.asarray(jax.jit(f, device=cpu)(x, g), np.float64)
    try:
        got = np.asarray(jax.jit(f, device=neu)(jax.device_put(x, neu), jax.device_put(g, neu)), np.float64)
    except Exception as e:
        print(f'N={n} C={c} H=W={h}: neuron error {str(e).splitlines()[0][:100]}', flush=True); continue
    e = np.linalg.norm(got - want) / np.linalg.norm(want)
    print(f'{"WRONG" if e > 1e-4 else "ok   "} N={n:3d} C={c:2d} H=W={h:2d}  rel err {e:.2e}', flush=True)
# control: the upsampled tensor computed on the host and passed in — correct
x = rng.standard_normal((128, 32, 4, 4)).astype(np.float32); g = rng.standard_normal((128, 32, 2, 2)).astype(np.float32)
u = np.repeat(np.repeat(g, 2, 2), 2, 3)
h = lambda x, u: lax.conv_general_dilated(jnp.transpose(x, (1, 0, 2, 3)), jnp.transpose(u, (1, 0, 2, 3)), (1, 1), [(1, 1), (1, 1)], dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
want = np.asarray(jax.jit(h, device=cpu)(x, u)); got = np.asarray(jax.jit(h, device=neu)(jax.device_put(x, neu), jax.device_put(u, neu)))
print(f'control (upsample on host): rel err {np.linalg.norm(got - want) / np.linalg.norm(want):.2e}')
