#!/usr/bin/env python3
"""SIGSEGV at the FIRST execution of a jitted train step on gfx1100, in XLA's
ROCm command-buffer (HIP graph) path. Pure JAX, no repo code.

  python repro.py [batch] [depth] [f32|bf16] [noconv] [noattn]

  python repro.py                                 -> SIGSEGV (ViT-Tiny, B=128)
  python repro.py 8 1 f32 noconv noattn           -> SIGSEGV (the MINIMUM: B=8,
                                                     one block, no convolution
                                                     and no attention at all)
  XLA_FLAGS=--xla_gpu_enable_command_buffer= python repro.py 8 1 f32 noconv noattn
                                                  -> survives, 3 of 3

Deterministic: 3/3 crash with command buffers on, 3/3 survive with them off,
and the same holds for the full net (conv + attention, depth 12, B=128).

⚠ The crash is NOT in the patch-embed convolution, even though MIOpen's log
puts that conv immediately before it — `noconv noattn` removes every
convolution and all attention and it still dies. `ruled_out.py` in this folder
issues that conv four ways and all four pass.

ViT-Tiny geometry: patch16, dim 192, depth 12, 3 heads, mlp 4x, 1000 classes.
"""
import sys
import jax, jax.numpy as jnp, numpy as np

B     = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DT    = jnp.bfloat16 if (len(sys.argv) > 3 and sys.argv[3] == "bf16") else jnp.float32

D, H, P, NCLS = 192, 3, 16, 1000
T = (224 // P) ** 2                      # 196 tokens, no cls token

def init(key):
    k = jax.random.split(key, 4 + DEPTH * 4)
    n = lambda kk, *s: (jax.random.normal(kk, s, jnp.float32) * 0.02)
    p = {"pw": n(k[0], D, 3, P, P), "pb": jnp.zeros((D,)),
         "pos": n(k[1], 1, T, D), "hw": n(k[2], D, NCLS), "hb": jnp.zeros((NCLS,))}
    for i in range(DEPTH):
        p[f"qkv{i}"] = n(k[4 + i * 4], D, 3 * D)
        p[f"prj{i}"] = n(k[5 + i * 4], D, D)
        p[f"f1_{i}"] = n(k[6 + i * 4], D, 4 * D)
        p[f"f2_{i}"] = n(k[7 + i * 4], 4 * D, D)
    return p

def ln(x):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-6)

NOCONV = "noconv" in sys.argv
NOATTN = "noattn" in sys.argv

def fwd(p, x):
    if NOCONV:
        # same patch embedding as a reshape + matmul: no convolution anywhere
        xp = x.reshape(B, 3, 14, P, 14, P).transpose(0, 2, 4, 1, 3, 5).reshape(B, T, 3 * P * P)
        y = (xp @ p["pw"].astype(DT).reshape(D, 3 * P * P).T).transpose(0, 2, 1)
    else:
        # patch embed: the conv MIOpen logs right before the crash
        y = jax.lax.conv_general_dilated(
            x, p["pw"].astype(DT), window_strides=(P, P), padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    y = y.reshape(B, D, T).transpose(0, 2, 1) + p["pb"].astype(DT)
    y = y + p["pos"].astype(DT)
    for i in range(DEPTH):
        h = ln(y)
        if NOATTN:
            o = (h @ p[f"qkv{i}"].astype(DT))[:, :, :D]     # same weights, no attention
        else:
            qkv = (h @ p[f"qkv{i}"].astype(DT)).reshape(B, T, 3, H, D // H)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            a = jnp.einsum('bthd,bshd->bhts', q, k) / np.sqrt(D // H)
            a = jax.nn.softmax(a, -1)
            o = jnp.einsum('bhts,bshd->bthd', a, v).reshape(B, T, D)
        y = y + o @ p[f"prj{i}"].astype(DT)
        h = ln(y)
        y = y + jax.nn.gelu(h @ p[f"f1_{i}"].astype(DT)) @ p[f"f2_{i}"].astype(DT)
    return ln(y).mean(1) @ p["hw"].astype(DT) + p["hb"].astype(DT)

def loss(p, x, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(fwd(p, x).astype(jnp.float32)), -1))

@jax.jit
def step(p, x, y):
    g = jax.grad(loss)(p, x, y)
    return {k: w - 0.001 * g[k] for k, w in p.items()}

p = init(jax.random.PRNGKey(0))
x = jnp.asarray(np.random.rand(B, 3, 224, 224), DT)
y = jnp.asarray(np.eye(NCLS, dtype=np.float32)[np.random.randint(0, NCLS, B)])
print(f"B={B} depth={DEPTH} dtype={DT.__name__} — starting", flush=True)
for i in range(20):
    p = step(p, x, y)
    if i == 0:
        jax.block_until_ready(p); print("  step 1 OK", flush=True)
jax.block_until_ready(p)
print(f"SURVIVED 20 steps  B={B} depth={DEPTH} {DT.__name__}")
