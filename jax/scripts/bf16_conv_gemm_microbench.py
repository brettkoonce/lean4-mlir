import time, numpy as np, jax, jax.numpy as jnp
from jax import random, jit
print("backend:", jax.default_backend(), jax.devices())
def timeit(f, *a, n=30):
    r=f(*a); jax.block_until_ready(r)
    t0=time.time()
    for _ in range(n): r=f(*a)
    jax.block_until_ready(r); return (time.time()-t0)/n
k=random.PRNGKey(0)
print("=== GEMM 8192^3 ===")
for dt in (jnp.float32, jnp.bfloat16):
    a=random.normal(k,(8192,8192),dtype=dt); b=random.normal(k,(8192,8192),dtype=dt)
    t=timeit(jit(lambda x,y:x@y), a, b)
    print(f"  {dt.__name__:9s}: {t*1000:7.2f} ms  {2*8192**3/t/1e12:6.1f} TFLOP/s")
print("=== conv N128 C64 56x56, 64x64x3x3 (ResNet stage-1 shape) ===")
for dt in (jnp.float32, jnp.bfloat16):
    x=random.normal(k,(128,64,56,56),dtype=dt); w=random.normal(k,(64,64,3,3),dtype=dt)
    cv=jit(lambda x,w: jax.lax.conv_general_dilated(x,w,(1,1),((1,1),(1,1)),
           dimension_numbers=("NCHW","OIHW","NCHW")))
    print(f"  {dt.__name__:9s}: {timeit(cv,x,w)*1000:7.2f} ms")

def dwconv(groups, pad):
    return jit(lambda x,w: jax.lax.conv_general_dilated(x,w,(1,1),((pad,pad),(pad,pad)),
           dimension_numbers=("NCHW","OIHW","NCHW"), feature_group_count=groups))

# Depthwise 3x3 — the MobileNetV2/EfficientNet workhorse (weight is C x 1 x k x k, groups=C).
print("=== depthwise 3x3  N128 C144 28x28 (MBConv expanded shape) ===")
for dt in (jnp.float32, jnp.bfloat16):
    x=random.normal(k,(128,144,28,28),dtype=dt); w=random.normal(k,(144,1,3,3),dtype=dt)
    print(f"  {dt.__name__:9s}: {timeit(dwconv(144,1),x,w)*1000:7.2f} ms")

# Depthwise 7x7 — the ConvNeXt token-mixer (stage-1 dim 96 at 56x56).
print("=== depthwise 7x7  N128 C96 56x56 (ConvNeXt block shape) ===")
for dt in (jnp.float32, jnp.bfloat16):
    x=random.normal(k,(128,96,56,56),dtype=dt); w=random.normal(k,(96,1,7,7),dtype=dt)
    print(f"  {dt.__name__:9s}: {timeit(dwconv(96,3),x,w)*1000:7.2f} ms")

# Full MBConv block: 1x1 expand (24->144) -> 3x3 depthwise -> 1x1 project (144->24), at 28x28.
# Depthwise alone undercounts the block — the 1x1s carry most of the channel traffic.
print("=== MBConv block  N128 24->144->24 28x28 ===")
for dt in (jnp.float32, jnp.bfloat16):
    x =random.normal(k,(128,24,28,28),dtype=dt)
    we=random.normal(k,(144,24,1,1),dtype=dt)   # expand 1x1
    wd=random.normal(k,(144,1,3,3),dtype=dt)    # depthwise 3x3
    wp=random.normal(k,(24,144,1,1),dtype=dt)   # project 1x1
    def mbconv(x,we,wd,wp):
        h=jax.lax.conv_general_dilated(x,we,(1,1),((0,0),(0,0)),dimension_numbers=("NCHW","OIHW","NCHW"))
        h=jax.lax.conv_general_dilated(h,wd,(1,1),((1,1),(1,1)),dimension_numbers=("NCHW","OIHW","NCHW"),feature_group_count=144)
        return jax.lax.conv_general_dilated(h,wp,(1,1),((0,0),(0,0)),dimension_numbers=("NCHW","OIHW","NCHW"))
    print(f"  {dt.__name__:9s}: {timeit(jit(mbconv),x,we,wd,wp)*1000:7.2f} ms")
