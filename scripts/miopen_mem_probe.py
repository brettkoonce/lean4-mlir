"""Does the ViT patch-embed weight-grad conv fail under memory pressure?

The conv runs fine in isolation, so the hypothesis is that in the full graph it
fails because MIOpen cannot get its workspace. Hold a ballast tensor, then run
the same conv, and walk the ballast up until something breaks.
"""
import jax, jax.numpy as jnp, gc
from jax import lax
print("jax", jax.__version__, jax.devices()[0])

lhs_s, rhs_s = (3, 32, 224, 224), (192, 32, 209, 209)
dn = lax.conv_dimension_numbers(lhs_s, rhs_s, ('NCHW', 'OIHW', 'NCHW'))
conv = jax.jit(lambda a, b: lax.conv_general_dilated(
    a, b, (1, 1), ((0, 0), (0, 0)), (1, 1), (1, 1), dn))

print(f"  inputs: lhs {jnp.prod(jnp.array(lhs_s))*4/2**30:.3f} GiB, "
      f"rhs {jnp.prod(jnp.array(rhs_s))*4/2**30:.3f} GiB")
for gib in [0, 4, 8, 12, 14, 16]:
    ballast = None
    try:
        if gib:
            ballast = jnp.ones((gib * (2**30) // 4,), jnp.float32).block_until_ready()
        out = conv(jnp.ones(lhs_s, jnp.float32), jnp.ones(rhs_s, jnp.float32))
        out.block_until_ready()
        print(f"  ballast {gib:>2} GiB -> conv OK   (out {out.shape})")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:180]
        print(f"  ballast {gib:>2} GiB -> {type(e).__name__}: {msg}")
    finally:
        del ballast; gc.collect()
