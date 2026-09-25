"""Does XLA fuse the interior pad into the convolution, and does that break MIOpen?

This is the exact two-op pattern the ViT patch-embed weight-gradient emits:
  pad(dy, interior=15) -> transpose -> convolution(x^T, .)
Run standalone (materialised filter) it works. Here pad and conv are in ONE jit,
so XLA may rewrite them into a single dilated convolution instead.
"""
import jax, jax.numpy as jnp, re
from jax import lax
print("jax", jax.__version__, jax.devices()[0])

B, ic, H, W, P, D, ph = 32, 3, 224, 224, 16, 192, 14

def wgrad(x, dy):                      # x [B,ic,H,W]   dy [B,D,ph,pw]
    u  = lax.pad(dy, 0.0, ((0,0,0),(0,0,0),(0,0,P-1),(0,0,P-1)))   # -> [B,D,209,209]
    xt = jnp.transpose(x,  (1,0,2,3))                              # -> [ic,B,H,W]
    dt = jnp.transpose(u,  (1,0,2,3))                              # -> [D,B,209,209]
    dn = lax.conv_dimension_numbers(xt.shape, dt.shape, ('NCHW','OIHW','NCHW'))
    return lax.conv_general_dilated(xt, dt, (1,1), ((0,0),(0,0)), (1,1), (1,1), dn)

x  = jnp.ones((B, ic, H, W), jnp.float32)
dy = jnp.ones((B, D, ph, ph), jnp.float32)
f  = jax.jit(wgrad)

txt = f.lower(x, dy).compile().as_text()
convs = [l.strip() for l in txt.splitlines() if "convolution(" in l]
print(f"  convolutions in OPTIMISED HLO: {len(convs)}")
for c in convs[:3]:
    print("   ", c[:230])
print(f"  pad ops remaining: {len(re.findall(r'= f32.* pad\(', txt))}")

try:
    out = f(x, dy); out.block_until_ready()
    print(f"  RUN OK  out {out.shape}  sum={float(jnp.sum(out)):.6g}")
except Exception as e:
    print(f"  RUN FAILED  {type(e).__name__}: {str(e)[:300]}")
