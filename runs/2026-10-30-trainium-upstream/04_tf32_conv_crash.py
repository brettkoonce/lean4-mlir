"""neuronx-cc crashes (NCC_IBCG901, BIRCodeGenLoop assertion) compiling a single forward 3x3 convolution
under --auto-cast=matmult --auto-cast-type=tf32. The same graph compiles under --auto-cast=none and under
--auto-cast-type=bf16. Compile only: lower().compile(); each shape is compiled in its own process.
  NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=tf32" python 04_tf32_conv_crash.py N C H"""
import os, re, sys, jax, jax.numpy as jnp
from jax import lax
N, C, H = (int(a) for a in sys.argv[1:4])
f = lambda x, w: lax.conv_general_dilated(x, w, (1, 1), [(1, 1), (1, 1)], dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
args = [jax.ShapeDtypeStruct((N, C, H, H), jnp.float32), jax.ShapeDtypeStruct((C, C, 3, 3), jnp.float32)]
try:
    jax.jit(f).lower(*args).compile(); r = 'compiled'
except Exception as e:
    m = re.search(r'NCC_[A-Z0-9]+', str(e))
    r = f'COMPILER CRASH {m.group(0)}' if m else ('compiled (load failed: fake device)' if 'NRT' in str(e) else f'error {str(e)[:200]}')
print(f'conv3x3 N={N} C={C} H=W={H}  flags="{os.environ.get("NEURON_CC_FLAGS")}"  {r}', flush=True)
