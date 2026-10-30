# neuronx-cc 2.28 fails (NCC_IBIR243) on the kernel gradient of a stride-2 depthwise convolution — the conv `jax.grad` emits for it

## Summary

The kernel gradient of a stride-2 depthwise convolution does not compile for trn1. `jax.vjp` of
`lax.conv_general_dilated(..., window_strides=(2, 2), feature_group_count=C)` with respect to its
kernel lowers to a convolution with `batch_group_count = C` and `rhs_dilation = (2, 2)`; at batch 32,
`neuronx-cc` fails on that convolution with

```
[INTERNAL_ERROR] [NCC_IBIR243] Access pattern out of bounds. Pattern: [[47040,32],[240,14],[1,128]]
  (DMACopy: U-ld_flt_h6_wf0_r1_bgc1) Source Kernel "neuronxcc.private_nkl.conv.conv2d_column_packing_io10"
  File ".../neuronxcc/private_nkl/conv.py", line 3536
```

for three of EfficientNet-B0's four strided depthwise layers (C=144 at 56×56 with a 5×5 kernel,
C=240 at 28×28 with 3×3, C=672 at 14×14 with 5×5). The first one, C=96 at 112×112 with 3×3, compiles.
So an EfficientNet-B0 or MobileNetV2 training step written in plain JAX cannot be compiled for trn1.

The same gradient written by hand with the cotangent zero-dilated first (`lax.pad` with interior
padding 1, then a stride-1 conv with `batch_group_count = C`) compiles at all four shapes, which is
what we use as a workaround. The input gradient of the same layers (`feature_group_count = C` with
`lhs_dilation = 2`) compiles fine.

## Environment

- trn1.2xlarge, Deep Learning AMI Neuron JAX 0.10 (Ubuntu 24.04.5)
- neuronx-cc 2.28.7631.0+8f7b0cf5, libneuronxla 3.0.10135.0+7d133f20, jax-neuronx 0.10.0.1.0.14198+40d442c2
- jax / jaxlib 0.10.0
- aws-neuronx-runtime-lib 2.35.30.0, aws-neuronx-collectives 2.35.43.0, aws-neuronx-dkms 2.31.7.0
- `NEURON_CC_FLAGS=--auto-cast=none` (the failure does not depend on this flag)

## Reproduction

Plain JAX, random shapes, compile only (`lower().compile()`); each shape in its own process.

```python
"""python repro.py N C H K   — e.g. `32 240 28 3` fails, `32 96 112 3` compiles"""
import os, re, sys, jax, jax.numpy as jnp
from jax import lax
N, C, H, K = (int(a) for a in sys.argv[1:5]); p = (K - 1) // 2

def dw(x, w):   # stride-2 depthwise conv, TF-style SAME pad for an even input
    return lax.conv_general_dilated(x, w, (2, 2), [(p - 1, p + 1), (p - 1, p + 1)], feature_group_count=C,
                                    dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

def kgrad(x, w, g):  # the kernel gradient: batch_group_count = C, rhs_dilation = 2 in HLO
    return jax.vjp(lambda w_: dw(x, w_), w)[1](g)[0]

args = [jax.ShapeDtypeStruct((N, C, H, H), jnp.float32), jax.ShapeDtypeStruct((C, 1, K, K), jnp.float32),
        jax.ShapeDtypeStruct((N, C, H // 2, H // 2), jnp.float32)]
try:
    jax.jit(kgrad).lower(*args).compile(); r = 'compiled'
except Exception as e:
    m = re.search(r'NCC_[A-Z0-9]+', str(e)); r = f'COMPILER ERROR {m.group(0)}' if m else f'error {str(e)[:200]}'
print(f'depthwise {K}x{K}/s2 kernel grad N={N} C={C} H=W={H}  {r}', flush=True)
```

```
$ for s in "32 96 112 3" "32 144 56 5" "32 240 28 3" "32 672 14 5"; do python repro.py $s; done
depthwise 3x3/s2 kernel grad N=32 C=96 H=W=112   compiled
depthwise 5x5/s2 kernel grad N=32 C=144 H=W=56   COMPILER ERROR NCC_IBIR243
depthwise 3x3/s2 kernel grad N=32 C=240 H=W=28   COMPILER ERROR NCC_IBIR243
depthwise 5x5/s2 kernel grad N=32 C=672 H=W=14   COMPILER ERROR NCC_IBIR243
```

The HLO the compiler receives for the failing case is a single convolution:

```
f32[240,1,3,3] convolution(x, g), window={size=14x14 pad=0_1x0_1 rhs_dilate=2x2},
    dim_labels=fb01_io01->fb01, batch_group_count=240          # x: f32[32,240,28,28], g: f32[32,240,14,14]
```

(for C=144 at 56×56 with the 5×5 kernel: `window={size=28x28 pad=1_2x1_2 rhs_dilate=2x2}, batch_group_count=144`).

The same failure is reached when the whole EfficientNet-B0 training step is compiled, so it is not
an artifact of compiling the gradient alone.

## Expected

The convolution compiles, as its `lhs_dilation` sibling (the input gradient) and the
hand-dilated form do.

## Workaround

Zero-dilate the cotangent explicitly and run a stride-1 grouped conv:

```python
u = lax.pad(g, jnp.float32(0), [(0, 0, 0), (0, 0, 0), (0, 1, 1), (0, 1, 1)])   # interior 1, one trailing row
w_grad = lax.conv_general_dilated(x.transpose(1, 0, 2, 3), u.transpose(1, 0, 2, 3), (1, 1),
                                  [(p - 1, p + 1), (p - 1, p + 1)], batch_group_count=C,
                                  dimension_numbers=('NCHW', 'OIHW', 'NCHW')).reshape(C, 1, K, K)
```

This matches `jax.vjp` on the CPU backend to 1e-6 and compiles for trn1 at all four shapes. It is
4× the multiply-adds of the dilated form, affordable for depthwise layers, but it is not something
`jax.grad` will produce on its own.
