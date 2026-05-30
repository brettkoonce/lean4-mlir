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
