"""Issue 5 repro: a weight-gradient conv whose kernel is an interior-padded (zero-dilated) tensor
returns wrong values on trn1, and the same gradient written with rhs_dilation does not.

This is the weight gradient of a P×P/stride-P patch-embedding conv (ViT), written two ways:
  A: pad the output cotangent g with interior P-1 (so the kernel is (14P-15)^2, 99.5 % zeros),
     then a valid stride-1 conv of the image against it;
  B: the same conv with rhs_dilation = P on the unpadded g.
Both are compared with JAX's CPU backend. Plain JAX, random data.

  python 05_dilated_kernel_conv_wrong.py            # small shapes first, then ViT-Tiny's
  SIZES=small python 05_dilated_kernel_conv_wrong.py
The ViT-Tiny shape (32×3×224×224, 192 channels, P=16) is a 206 GMAC conv in form A; its compile
takes ~9 min on an 88-core box. Form B is 0.9 GMAC."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
from jax import lax

def wgrad_A(x, g, P):
    u = lax.pad(g, jnp.float32(0), [(0, 0, 0), (0, 0, 0), (0, 0, P - 1), (0, 0, P - 1)])  # interior P-1
    return lax.conv_general_dilated(x.transpose(1, 0, 2, 3), u.transpose(1, 0, 2, 3), (1, 1), [(0, 0), (0, 0)],
                                    dimension_numbers=('NCHW', 'OIHW', 'NCHW')).transpose(1, 0, 2, 3)

def wgrad_B(x, g, P):
    return lax.conv_general_dilated(x.transpose(1, 0, 2, 3), g.transpose(1, 0, 2, 3), (1, 1), [(0, 0), (0, 0)],
                                    rhs_dilation=(P, P), dimension_numbers=('NCHW', 'OIHW', 'NCHW')).transpose(1, 0, 2, 3)

def run(B, C, D, H, P):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, C, H, H), dtype=np.float32); g = rng.standard_normal((B, D, H // P, H // P), dtype=np.float32)
    cpu = jax.devices('cpu')[0]; dev = jax.devices()[0]
    ref = np.asarray(jax.jit(wgrad_B, static_argnums=2)(jax.device_put(x, cpu), jax.device_put(g, cpu), P))
    for name, f in (('A pad+conv', wgrad_A), ('B rhs_dilation', wgrad_B)):
        out = np.asarray(jax.jit(f, static_argnums=2)(jax.device_put(x, dev), jax.device_put(g, dev), P))
        err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
        print(f'{"WRONG" if err > 1e-3 else "ok   "} B={B} C={C} D={D} H={H} P={P} {name:16s} rel err {err:.2e}', flush=True)

if __name__ == '__main__':
    print('device:', jax.devices()[0], flush=True)
    for B, C, D, H, P in [(4, 3, 8, 64, 16), (8, 3, 16, 128, 16), (8, 3, 32, 224, 16)]:
        run(B, C, D, H, P)
    if os.environ.get('SIZES') != 'small':
        run(32, 3, 192, 224, 16)
