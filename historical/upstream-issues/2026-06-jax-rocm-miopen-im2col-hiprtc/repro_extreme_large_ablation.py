"""Is the EXTREME_LARGE im2col branch broken in general, or only for dilated convs?

Same tensor shapes and channel count as the failing case, varying ONLY the filter
dilation. If the dilation-1 variant also gets -DEXTREME_LARGE and also fails, the
branch is simply broken and the dilation is incidental.
"""
import jax, jax.numpy as jnp
from jax import lax
print("jax", jax.__version__, jax.devices()[0])

def go(tag, filt_hw, rhs_dil):
    lhs = jnp.ones((3, 32, 224, 224), jnp.float32)     # [ic=3, B=32, 224, 224]
    rhs = jnp.ones((192, 32, filt_hw, filt_hw), jnp.float32)
    dn = lax.conv_dimension_numbers(lhs.shape, rhs.shape, ('NCHW', 'OIHW', 'NCHW'))
    f = jax.jit(lambda a, b: lax.conv_general_dilated(
        a, b, (1, 1), ((0, 0), (0, 0)), (1, 1), (rhs_dil, rhs_dil), dn))
    print(f"--- {tag}: filter {filt_hw}x{filt_hw}, rhs_dilation {rhs_dil} ---", flush=True)
    try:
        o = f(lhs, rhs); o.block_until_ready()
        print(f"    OK  out {o.shape}", flush=True)
    except Exception as e:
        print(f"    FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

go("dilated (the failing case)", 14, 16)
go("NON-dilated, same 14x14",    14, 1)
go("NON-dilated, big 209x209",  209, 1)
