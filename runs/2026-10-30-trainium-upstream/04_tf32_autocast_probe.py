"""neuronx-cc internal errors under --auto-cast=matmult --auto-cast-type=tf32 (NCC_IBCG901 BIRCodeGenLoop
assertion, NCC_IXCG863 ISA check). Compiles small graphs one at a time and prints which crash the compiler.
Compile only: lower().compile(). Run with NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=tf32"."""
import os, re, sys, jax, jax.numpy as jnp
from jax import lax
S = jax.ShapeDtypeStruct; f32 = jnp.float32
conv = lambda x, w, s=1: lax.conv_general_dilated(x, w, (s, s), [((w.shape[-1] - 1) // 2,) * 2] * 2,
                                                  dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
cases = {
  'matmul 128x192 @ 192x768':           (lambda a, b: a @ b, [S((128, 192), f32), S((192, 768), f32)]),
  'matmul grad (dw) 6304x192, 6304x768': (lambda a, b: a.T @ b, [S((6304, 192), f32), S((6304, 768), f32)]),
  'attention scores 32x3x197x64':        (lambda q, k: jnp.einsum('bhqd,bhkd->bhqk', q, k), [S((32, 3, 197, 64), f32)] * 2),
  'patch conv 16x16/16, 3->192':         (lambda x, w: conv(x, w, 16), [S((32, 3, 224, 224), f32), S((192, 3, 16, 16), f32)]),
  'conv 3x3 64->64 @56':                 (conv, [S((16, 64, 56, 56), f32), S((64, 64, 3, 3), f32)]),
  'conv 3x3 grad (dx,dw) 64->64 @56':    (lambda x, w: jax.grad(lambda x, w: conv(x, w).sum(), (0, 1))(x, w),
                                          [S((16, 64, 56, 56), f32), S((64, 64, 3, 3), f32)]),
  'conv 7x7/2 grad 3->64 @224':          (lambda x, w: jax.grad(lambda w: conv(x, w, 2).sum())(w),
                                          [S((16, 3, 224, 224), f32), S((64, 3, 7, 7), f32)]),
  'conv 3x3/2 grad 64->128 @56':         (lambda x, w: jax.grad(lambda x, w: conv(x, w, 2).sum(), (0, 1))(x, w),
                                          [S((16, 64, 56, 56), f32), S((128, 64, 3, 3), f32)]),
}
only = sys.argv[1:]
print(f'jax {jax.__version__}, {jax.devices()[0]}, NEURON_CC_FLAGS={os.environ.get("NEURON_CC_FLAGS")}', flush=True)
for name, (fn, args) in cases.items():
    if only and not any(o in name for o in only): continue
    try:
        jax.jit(fn).lower(*args).compile(); r = 'ok'
    except Exception as e:
        m = re.search(r'NCC_[A-Z0-9]+', str(e))
        r = f'COMPILER ERROR {m.group(0)}' if m else ('ok (compiled; load failed on the fake device)' if 'NRT' in str(e) else f'error: {str(e)[:150]}')
    print(f'{name:40s} {r}', flush=True)
