# Lean 4 → JAX

Lean 4 as a specification language for neural networks. Declare architecture
and training config in Lean, generate idiomatic JAX Python, run training.

Replicating the models from [Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2) (Apress).

## Models

| Model | File | Architecture | Params | Accuracy | Time (CPU) |
|-------|------|-------------|--------|----------|------------|
| MNIST MLP (Ch.1) | `MainMlp.lean` | 784→512→512→10 | 670K | 97.9% | 7.5s |
| MNIST CNN (Ch.2) | `MainCnn.lean` | Conv²→Pool→Dense³ | 3.5M | 97.6% | 6.7 min |
| CIFAR-10 CNN (Ch.3) | `MainCifar.lean` | Conv²→Pool→Conv²→Pool→Dense³ | 2.4M | 63.3% | 31 min |

## Lean specs

```lean
-- MLP (S4TF book Ch. 1)
def mnistMlp : NetSpec where
  name := "MNIST MLP"
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- CNN (S4TF book Ch. 2)
def mnistCnn : NetSpec where
  name := "MNIST CNN"
  layers := [
    .conv2d  1 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 6272 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]

-- CIFAR-10 CNN (S4TF book Ch. 3)
def cifarCnn : NetSpec where
  name := "CIFAR-10 CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]
```

Lean generates a complete JAX training script and runs it. The generated
Python is readable and auditable at `.lake/build/generated_mnist_*.py`.

## Quick start

### 1. Install Lean 4

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### 2. Install JAX

```bash
python3 -m venv .venv
.venv/bin/pip install jax jaxlib
```

For GPU (ROCm):
```bash
.venv/bin/pip install jax[rocm]
```

### 3. Get MNIST data

```bash
cd ../mnist-lean4 && ./download_mnist.sh && cd ../lean4-jax
```

Or if you already have the raw IDX files (`train-images-idx3-ubyte`, etc.),
pass the directory as a CLI argument.

### 4. Build and run

```bash
lake build mnist-mlp mnist-cnn cifar-cnn

# MLP — ~8 seconds
.lake/build/bin/mnist-mlp

# MNIST CNN — ~7 minutes
.lake/build/bin/mnist-cnn

# CIFAR-10 CNN — longer (4 conv layers, 32×32 images)
.lake/build/bin/cifar-cnn

# Custom data dir
.lake/build/bin/mnist-mlp /path/to/data
.lake/build/bin/cifar-cnn /path/to/cifar-10
```

## Project structure

```
LeanJax.lean      Shared types (Layer, NetSpec, TrainConfig, DatasetKind) + JAX codegen + runner
MainMlp.lean      MNIST MLP spec + main  (5 lines)
MainCnn.lean      MNIST CNN spec + main  (10 lines)
MainCifar.lean    CIFAR-10 CNN spec + main  (15 lines)
lakefile.lean     Build config (3 executables, 1 library)
```

## How it works

1. Lean defines the network as a `NetSpec` — a list of `Layer` values
2. `JaxCodegen.generate` walks the layer list and emits idiomatic JAX Python
   - Conv layers → `jax.lax.conv_general_dilated`
   - Pool layers → `jax.lax.reduce_window`
   - Dense layers → `x @ w.T + b`
   - Activation, init, loss, training loop — all generated
3. `runJax` writes the script to `.lake/build/` and runs it via `python3`
4. JAX handles autodiff (`value_and_grad`), JIT compilation, XLA

## GPU status

The generated Python is GPU-ready — **zero code changes needed**. JAX
auto-dispatches to GPU when one is available. The only change is the backend:

```bash
# ROCm (AMD) — needs jaxlib built for your ROCm version
pip install jax-rocm60-plugin     # ROCm 6.0
# ROCm 7.x plugin not yet on PyPI — build jaxlib from source or wait for package

# CUDA (NVIDIA)
pip install jax[cuda12]
```

On a 7900 XTX the CNN would go from ~7 min → ~seconds. The Lean spec
and generated code are identical; only the pip install differs.

## Why Lean → JAX?

The [mnist-lean4](../mnist-lean4) project built neural nets from scratch in Lean 4
with C FFI → OpenBLAS → hipBLAS. That works, but requires hand-written gradients
and manual BLAS calls for every operation.

Lean → JAX gives you:
- **Automatic differentiation** — JAX's `grad` replaces 100s of lines of manual backward passes
- **JIT compilation** — XLA compiles the compute graph, no manual optimization
- **GPU for free** — swap `jax[rocm]` or `jax[cuda]` and it just works
- **Lean as the spec** — type-checked architecture definitions, eventually provable properties

Compare: the Lean CNN backward pass is ~100 lines of hand-written gradient code.
The JAX version: zero — `value_and_grad(loss_fn)` does it all.

## Lean version

Tested with Lean 4.29.0 / Lake 5.0.0, JAX 0.9.2.

## License

Public domain.
