#!/usr/bin/env python3
"""Minimal reproducer: SIGSEGV in RocmCommandBuffer::LaunchGraph on gfx1100.

Pure JAX. No repo code, no Lean, no FFI shim. A 784-512-512-10 MLP trained
with SGD on random data, batch 128, 468 steps per epoch, 12 epochs, with a
jitted eval between epochs.

    HIP_VISIBLE_DEVICES=0 python repro.py

Dies with SIGSEGV (exit 139) at a nondeterministic epoch. Observed 11 of 11
runs on the environment in README.md, dying anywhere between epoch 1 and
epoch 11. Prints the epoch it reached, so the survival point is visible.

The data is random, so accuracy stays at chance. That is deliberate: the
crash has nothing to do with what is being learned, only with how many
device dispatches have gone by.
"""
import jax, jax.numpy as jnp, numpy as np, sys
k = jax.random.PRNGKey(0)
def init(k):
    ks = jax.random.split(k, 3)
    s = lambda kk,a,b: jax.random.normal(kk,(a,b),jnp.float32)*np.sqrt(2.0/a)
    return [s(ks[0],784,512), jnp.zeros((512,)), s(ks[1],512,512), jnp.zeros((512,)),
            s(ks[2],512,10),  jnp.zeros((10,))]
def fwd(p,x):
    h = jnp.maximum(x@p[0]+p[1],0); h = jnp.maximum(h@p[2]+p[3],0); return h@p[4]+p[5]
def loss(p,x,y):
    lg = fwd(p,x); return -jnp.mean(jnp.sum(y*jax.nn.log_softmax(lg),-1))
@jax.jit
def step(p,x,y):
    g = jax.grad(loss)(p,x,y); return [w-0.1*gi for w,gi in zip(p,g)]
@jax.jit
def acc(p,x,y): return jnp.mean(jnp.argmax(fwd(p,x),-1)==y)
p = init(k)
X = jnp.asarray(np.random.rand(60000,784), jnp.float32)
Y = jnp.asarray(np.eye(10, dtype=np.float32)[np.random.randint(0,10,60000)])
Xt = jnp.asarray(np.random.rand(10000,784), jnp.float32)
Yt = jnp.asarray(np.random.randint(0,10,10000))
for ep in range(12):
    for b in range(468):
        p = step(p, X[b*128:(b+1)*128], Y[b*128:(b+1)*128])
    a = float(acc(p, Xt, Yt)); print(f"  epoch {ep+1}: acc={a:.4f}", flush=True)
print("DONE 12/12")
