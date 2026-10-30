"""Issue 6 repro: neuronx-cc fails (NCC_IBIR243, access pattern out of bounds) on the kernel gradient
of a stride-2 depthwise convolution — the conv XLA emits for it has batch_group_count = C and
rhs_dilation = 2. Plain JAX, compile only (lower().compile()); each shape in its own process.
  python 06_depthwise_strided_kernel_grad_compile.py N C H K    # e.g. 32 240 28 3 (fails), 32 96 112 3 (compiles)
EfficientNet-B0's four strided depthwise layers at batch 32: (96,112,3) compiles; (144,56,5), (240,28,3)
and (672,14,5) fail. The same gradient written with the cotangent zero-dilated by hand (lax.pad,
interior 1) compiles — that is the form we fall back to."""
import os, re, sys, jax, jax.numpy as jnp
from jax import lax
N, C, H, K = (int(a) for a in sys.argv[1:5]); p = (K - 1) // 2
def dw(x, w):   # stride-2 depthwise conv, SAME-style pad [p-1, p+1] as TF's for even inputs
    return lax.conv_general_dilated(x, w, (2, 2), [(p - 1, p + 1), (p - 1, p + 1)], feature_group_count=C,
                                    dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
def kgrad(x, w, g):  # the kernel gradient: batch_group_count = C, rhs_dilation = 2 in HLO
    return jax.vjp(lambda w_: dw(x, w_), w)[1](g)[0]
args = [jax.ShapeDtypeStruct((N, C, H, H), jnp.float32), jax.ShapeDtypeStruct((C, 1, K, K), jnp.float32),
        jax.ShapeDtypeStruct((N, C, H // 2, H // 2), jnp.float32)]
hlo = jax.jit(kgrad).lower(*args).as_text('hlo')
form = re.search(r'batch_group_count=\d+|rhs_dilate=\d+x\d+|window_dilation', hlo)
try:
    jax.jit(kgrad).lower(*args).compile(); r = 'compiled'
except Exception as e:
    m = re.search(r'NCC_[A-Z0-9]+', str(e))
    r = f'COMPILER ERROR {m.group(0)}' if m else ('compiled (load failed: fake device)' if 'NRT' in str(e) else f'error {str(e)[:200]}')
print(f'depthwise {K}x{K}/s2 kernel grad N={N} C={C} H=W={H}  flags="{os.environ.get("NEURON_CC_FLAGS")}"  {r}', flush=True)
