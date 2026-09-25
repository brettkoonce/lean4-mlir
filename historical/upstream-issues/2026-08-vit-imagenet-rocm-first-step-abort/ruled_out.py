#!/usr/bin/env python3
"""FOUR NEGATIVE CONTROLS for the ViT/ImageNet first-step abort on gfx1100.

Every mode here PASSES. That is the point: this file exists so the next person
does not spend an afternoon re-deriving that the patch-embed convolution, in
isolation, is fine. It is the last MIOpen call before the abort and it is not,
by itself, the trigger.

    python ruled_out.py bare | tokens | grad | flat

  bare    the conv alone, N=128/32/1                     -> passes
  tokens  conv + the [N,192,14,14] -> [N,196,192] reshape -> passes
  grad    weight-grad through the patch embed            -> passes
  flat    the driver's [N,150528] input, reshaped in-graph -> passes

Shapes are exactly the ones MIOpen logs before the real crash:
  wDesc {192, 3, 16, 16}  xDesc {N, 3, 224, 224}  stride 16  yDesc {N, 192, 14, 14}
  solver GemmFwdRest, workSpace 602112 bytes

Compare the sibling issue 2026-06-jax-rocm-miopen-im2col-hiprtc, where the
trigger also required the conv to be FUSED with something else and the
standalone conv likewise passed. That is the direction to look.
"""
import sys
import jax, jax.numpy as jnp, numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "bare"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128

x = jnp.asarray(np.random.rand(N, 3, 224, 224), jnp.float32)
w = jnp.asarray(np.random.rand(192, 3, 16, 16), jnp.float32)


def patch(x, w, tokens=True):
    y = jax.lax.conv_general_dilated(
        x, w, window_strides=(16, 16), padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    return y.reshape(N, 192, 196).transpose(0, 2, 1) if tokens else y


if MODE == "bare":
    out = jax.jit(lambda x, w: patch(x, w, tokens=False))(x, w)
elif MODE == "tokens":
    out = jax.jit(patch)(x, w)
elif MODE == "grad":
    out = jax.jit(jax.grad(lambda w, x: jnp.sum(patch(x, w) ** 2)))(w, x)
elif MODE == "flat":
    xf = x.reshape(N, -1)
    out = jax.jit(lambda xf, w: patch(xf.reshape(N, 3, 224, 224), w))(xf, w)
else:
    raise SystemExit(f"unknown mode {MODE!r}")

out.block_until_ready()
print(f"{MODE} N={N} PASSED  {out.shape}  sum={float(jnp.sum(out)):.3f}")
