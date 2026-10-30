# Neuron upstream reports — drafts (not filed)

For `aws-neuron/aws-neuron-sdk` issues. Each repro is plain JAX on random data: no framework, no
custom plugin, nothing from this repo. Each compares against JAX's CPU backend or compiles only.
Versions are in `trn1/versions.txt`: Neuron SDK 2.33.0, neuronx-cc 2.28.7631, libneuronxla 3.0.10135,
jax / jaxlib 0.10.0, jax-neuronx 0.10.0.1.0.14198, Deep Learning AMI Neuron JAX 0.10 (Ubuntu 24.04),
trn1.2xlarge. `NEURON_CC_FLAGS=--auto-cast=none` unless stated.

On-device output is in `trn1/`. Compile-only output is in `atlas/`: an x86 box with no Trainium,
using the runtime's fake device (`fake_env.sh`). The compiler runs exactly as it does on the instance
and only the NEFF load fails.

| # | file | kind | one line |
|---|---|---|---|
| 1 | `01_maxpool_grad_wrong.py` | wrong result | the max-pool gradient (`select_and_scatter`) puts gradients on the wrong elements |
| 2 | `02_upsample_weight_grad_conv_wrong.py` | wrong result | a 2× nearest upsample feeding a weight-gradient-form conv returns wrong values |
| 3 | `03_resnet34_train_step_compile.py` | compiler error | a textbook ResNet-34 train step (f32, batch 32) fails `neuronx-cc` |
| 4 | `04_tf32_conv_crash.py` | compiler crash | a single 3×3 conv with ≥32 channels crashes the compiler under tf32 auto-cast |

---

## 1. Gradient of a 2×2 max pool is wrong on trn1 (`select_and_scatter`)

**Summary.** `jax.vjp` of `lax.reduce_window(max)` with a 2×2 window and stride 2 (lowered to
`stablehlo.select_and_scatter`) returns wrong gradients on a NeuronCore. At 8×8×8×8 each window's
gradient lands on the window's top-left element instead of its maximum, so the total is still
right. At 128×16×32×32 it lands on more positions than there are windows, and the total is wrong.
Inputs are a permutation, so every window has a unique max.

**Impact.** Every CNN with max pooling trains on wrong gradients. A small CIFAR-10 CNN stays at
chance (10.00%), and the same graph reaches 30.9% on XLA:CPU and GPU.

**Repro.** `python 01_maxpool_grad_wrong.py` on trn1. Output: `trn1/01_maxpool_grad_wrong.out`.

**Notes.** A 1×1×4×4 input fails to compile (`INTERNAL`). An equivalent backward written without
`select_and_scatter` (strided slices, a first-max mask, interior pads) is correct on Neuron in isolation.

## 2. 2× nearest upsample feeding a weight-gradient conv miscompiles

**Summary.** With `x: (N, C, H, W)` and `g: (N, C, H/2, W/2)`:

```python
u = jnp.repeat(jnp.repeat(g, 2, axis=2), 2, axis=3)          # nearest 2x upsample
w = lax.conv_general_dilated(x.transpose(1, 0, 2, 3), u.transpose(1, 0, 2, 3),
                             (1, 1), [(1, 1), (1, 1)], dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
```

This is the weight gradient of a 3×3 conv whose output cotangent comes through an upsample. It
returns relative error ~0.7 against CPU for N, C ≥ 8 and H ≥ 4. Each piece alone is correct. The
same conv with `u` computed on the host and passed in is correct (rel err 4e-7). At H = 2 it is
correct.

**Variations (all on trn1):**
- An `optimization_barrier` after the upsample or after the transposes: still wrong.
- Letting the conv's dimension numbers do the transposes (`CNHW`/`IOHW`): still wrong.
- The upsample as four interior pads summed: wrong at 4×4, right at 16×16.
- The upsample as a `dot_general` with a 0/1 matrix: right.

**Impact.** Combined with issue 1, the backward pass of a CNN with max pooling is wrong even after
working around `select_and_scatter`.

**Repro.** `python 02_upsample_weight_grad_conv_wrong.py` on trn1. Output: `trn1/02_upsample_weight_grad_conv_wrong.out`.

## 3. neuronx-cc fails on a ResNet-34 training step at batch 32, f32

**Summary.** A standard ResNet-34 training step compiles at batch 16, or at batch 32 with bf16
convolutions. At batch 32 in f32 it fails after ~9 min with
`[INTERNAL_ERROR] [NCC_IBIR243] Access pattern out of bounds`, on a tensor derived from the
max-pool's `maximum` (`maximum.17.copy…d2dtranspose…Remat…ShrinkedCopy_partial_ReloadPartial`). The step is: 224×224 input, conv
+ batch-norm (batch statistics) + ReLU, the 3-4-6-3 basic-block stack, 3×3/2 max pool, global
average pool, linear head, softmax cross-entropy, one SGD update. The script is plain JAX, about
70 lines, and compiles only. We see the same failure from two other, independently written
ResNet-34 steps, so it isn't specific to one way of writing the graph. `-O1` gets past it on one
of them, but that NEFF then exceeds the runtime's DMA descriptor limit when loaded.

**Repro.** `python 03_resnet34_train_step_compile.py 32 f32` (fails) against `… 16 f32` and
`… 32 bf16` (compile). Output: `atlas/03-*.log`.

## 4. tf32 auto-cast crashes the compiler on a single 3×3 convolution

**Summary.** With `NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=tf32"`, compiling one
forward 3×3 convolution (stride 1, pad 1, NCHW/OIHW) with 32 or more channels crashes neuronx-cc:
`[INTERNAL_ERROR] [NCC_IBCG901] BIRCodeGenLoop assertion error: … BirCodeGenLoop.py:978`. One
shape, 16×64×14×14, gives `NCC_IXCG863` "ISA check failed" instead. The smallest crash we found is
1×32×8×8 with a 32×32×3×3 kernel. 16 channels or fewer compiles. The same graph compiles under
`--auto-cast=none` and under `--auto-cast-type=bf16`. Matmuls, attention einsums and a 16×16/16
patch conv compile under tf32.

**Repro.** `NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=tf32" python 04_tf32_conv_crash.py 1 32 8`.
Output: `atlas/04-*.log`.
