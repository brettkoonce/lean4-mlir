# `jax/` — the JAX reference path

Lean 4 as a metaprogramming layer that emits JAX Python. The same `NetSpec` the verified path
renders to StableHLO is walked here into a readable training script, and JAX's `value_and_grad`
supplies the gradients. That makes this directory the project's independent second opinion:

- **The ImageNet references.** Every pair in the book's Appendix C is this path against the
  verified one, same recipe, same validation protocol. `lake run imagenet --jax` (from the repo
  root) plans or launches the reference half of each row through
  [`scripts/jobs/`](../scripts/jobs/)'s `*-jax-4gpu` confs.
- **The VJP oracle's ground truth.** [`tests/vjp_oracle/`](../tests/vjp_oracle/) trains one
  batch on both paths from the same init and diffs the loss after the first update; the
  `vjp-oracle-*` and `verified-oracle-*` exes here are the JAX side.

It is its own Lake package (`jax/lakefile.lean`, depending on the parent for `LeanMlir.Types`
and `LeanMlir.Spec`), so everything below is built from inside `jax/`.

## What's here

| | |
|---|---|
| `Jax/Codegen.lean`, `Jax/Runner.lean` | the generator: `NetSpec` → JAX Python, then find a Python, write `.lake/build/generated_*.py` and run it |
| `Main*.lean` | one trainer per net and scale: MNIST (`MainMlp`, `MainCnn`), CIFAR (`MainCifar`), the Imagenette nets, and the `*Imagenet` trainers behind the ImageNet references |
| `generated/` | committed copies of the generated ImageNet trainers |
| `scripts/` | full-50k evaluators (`eval_*_full50k.py`), smoke tests, benches and the supervisors |
| `runs/` | the reference runs' logs and results |
| `probe/` | the Imagenette step-time probes and their estimator ([README](probe/README.md)) |
| `tests/vjp_oracle/` | the oracle's JAX-side fixtures |
| `timm_eval_protocols.json` | the validation protocol each net is scored under |
| `requirements-{cuda,rocm}-lock.txt` | the pinned environments |

## Quick start

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh   # Lean 4
python3 -m venv .venv && .venv/bin/pip install -r jax/requirements-cuda-lock.txt        # or requirements-rocm-lock.txt
./scripts/datasets/download_mnist.sh                                                    # also download_cifar.sh, download_imagenette.sh
cd jax && lake build mnist-mlp resnet34
.lake/build/bin/mnist-mlp
.lake/build/bin/resnet34 /path/to/imagenette     # the data directory is the one argument
```

Each exe writes its script to `jax/.lake/build/generated_<net>.py` and runs it, so the Python is
there to read after a build. Both locks pin jax 0.11.0 (`jax-cuda12-*` or `jax-rocm7-*`), and jax
0.10 onward needs Python ≥ 3.11. To refresh a lock, `pip freeze > requirements-<backend>-lock.txt`
from the matching venv.

## How it works

1. Lean defines the network as a `NetSpec` — a list of `Layer` values
2. `Jax.Codegen.generate` walks the layer list and emits idiomatic JAX Python
   - Conv layers → `jax.lax.conv_general_dilated`
   - Residual blocks → `basic_block` / `bottleneck_block` with skip connections
   - Depthwise separable convs → `feature_group_count` in JAX
   - MBConv blocks → inverted residuals + squeeze-excitation + Swish/hard-swish
   - Fire modules → squeeze + parallel expand + channel concat
   - Transformer blocks → multi-head self-attention + layer norm + MLP
   - Patch embedding → conv projection + CLS token + positional embedding
   - Pool layers → `jax.lax.reduce_window`
   - Dense layers → `x @ w.T + b`
   - Instance normalization, layer normalization, activation, init, loss, training loop — all generated
3. `runJax` writes the script to `.lake/build/generated_*.py` and runs it via `python3`
4. JAX handles autodiff (`value_and_grad`), JIT compilation, XLA

The generated Python is readable and auditable — open `.lake/build/generated_resnet34.py` after a build.

## Multi-GPU

Data parallelism is in the generated script: a `jax.sharding` `Mesh` over every visible device,
parameters replicated, batches sharded. `CUDA_VISIBLE_DEVICES` picks the cards; nothing in the
Lean spec changes.

## History: phase 2

Everything below is the record of this path when it was the project's trainer, spring to summer
2026, before the verified renders existed. The numbers are that period's (six RTX 4060 Ti,
instance norm, a short Adam recipe) and are not the book's; the bring-up notes for each backend
are in [`../historical/CUDA.md`](../historical/CUDA.md) and [`../historical/ROCM.md`](../historical/ROCM.md).

### Why this existed, and why we moved on

[`mnist-lean4/`](../historical/mnist-lean4) (phase 1) built neural nets from scratch in Lean 4
with C FFI → OpenBLAS → hipBLAS. That works, but requires hand-written gradients
and manual BLAS calls for every operation.

The JAX path (phase 2) gives you:
- **Automatic differentiation** — JAX's `grad` replaces 100s of lines of manual backward passes
- **JIT compilation** — XLA compiles the compute graph, no manual optimization
- **GPU for free** — swap `jax[rocm]` or `jax[cuda]` and it just works
- **Lean as the spec** — type-checked architecture definitions

Compare: the Lean CNN backward pass in phase 1 is ~100 lines of hand-written gradient code.
The JAX version: zero — `value_and_grad(loss_fn)` does it all.

The verified path (project root) goes one step further: instead of
generating JAX *Python*, the codegen emits StableHLO MLIR directly and a
lowerer (XLA/PJRT, or IREE) compiles it. No Python
runtime, no autodiff library — the gradients are computed at codegen time
in Lean. See the project root README for that path.

### Models trained via this path

| Model | File | Params | Accuracy | GPU time | Optimizer |
|-------|------|--------|----------|---------|-----------|
| MNIST MLP | `MainMlp.lean` | 670K | 97.9% | 7.5s | SGD |
| MNIST CNN | `MainCnn.lean` | 3.5M | 97.6% | 23s | SGD |
| CIFAR-10 CNN | `MainCifar.lean` | 2.4M | 63.3% | 53s | SGD |
| SqueezeNet v1.1 | `MainSqueezeNet.lean` | 730K | 77.2% | 45 min | Adam |
| MobileNet v1 | `MainMobilenet.lean` | 3.2M | 79.5% | 47 min | Adam |
| MobileNet v2 | `MainMobilenetV2.lean` | 2.2M | 79.2% | 48 min | Adam |
| MobileNet v3-Large | `MainMobilenetV3.lean` | 3.0M | 81.1% | 46 min | Adam |
| MobileNet V4-Medium | `MainMobilenetV4.lean` | 4.1M | 77.3% | 46 min | Adam |
| EfficientNet-B0 | `MainEfficientNet.lean` | 7.2M | 82.5% | 48 min | Adam |
| EfficientNet V2-S | `MainEfficientNetV2.lean` | 38.2M | 83.1% | 58 min | Adam |
| **VGG-16-BN** | `MainVgg.lean` | **14.7M** | **88.3%** | **42 min** | Adam |
| ResNet-34 | `MainResnet.lean` | 21.3M | 84.9% | 50 min | Adam |
| ResNet-50 | `MainResnet50.lean` | 23.5M | 85.0% | 55 min | Adam |
| ViT-Tiny | `MainVit.lean` | 5.5M | 65.2% | 45 min | Adam |

All Imagenette models trained from scratch on 6× RTX 4060 Ti, no pretrained weights, random crop augmentation. The phase 3 numbers (which use a different training recipe and proper running BN stats) are in [`../historical/RESULTS.md`](../historical/RESULTS.md) and tend to be 5-10 points higher.

### Differences from the S4TF book

All architectures are **structurally identical** to the Swift for TensorFlow
implementations — same layer configs, same channel counts. The training recipes
differ due to multi-GPU batching and optimizer discoveries:

| Model | Book config | Our config | Why |
|-------|------------|------------|-----|
| MNIST MLP | SGD 0.1, bs=128, 12ep | Same | Exact match |
| MNIST CNN | SGD 0.1 | SGD 0.01 | LR tuned for stability |
| CIFAR-10 | SGD 0.1, 12ep | SGD 0.01, 25ep | LR tuned, more epochs |
| ResNet-34 | SGD 0.002, mom=0.9, bs=32 | Adam 0.001 + cosine/WD | Adam outperforms SGD with instance norm |
| ResNet-50 | SGD 0.002, mom=0.9, bs=32 | Adam 0.001 + cosine/WD + zero-init residual | Bottleneck blocks need adaptive LR |
| MobileNets | SGD 0.002, mom=0.9 | Adam 0.001 + cosine/WD | SGD can't converge without skip connections |
| EfficientNet | SGD 0.002, mom=0.9 | Adam 0.001 + cosine/WD | Same — depthwise convs need adaptive LR |
| SqueezeNet | SGD 0.0001, bs=128, 100ep | Adam 0.001, bs=192, 50ep | Adam converges in half the epochs |
| VGG-16 | SGD 0.002, bs=32, 10ep, no BN | Adam 0.001, bs=192, 50ep, +BN, GAP | Added BN + GAP (14.7M vs ~134M params) |

**Other differences:**
- **Normalization** — book uses batch norm with running stats (per-sample forward).
  This path uses instance norm (spatial stats only) — batch norm diverged with multi-GPU sharding.
  Phase 3 has proper batch norm with running stats, see [`../historical/RESULTS.md`](../historical/RESULTS.md).
- **Book processes one image at a time** with gradient accumulation over the batch.
  We do true batched forward/backward across 6 GPUs.
- **We add** cosine LR decay, linear warmup, weight decay, random horizontal flip —
  none of which are in the book.
- **Book adds** dropout (MobileNet v1) and running BN stats for eval — phase 2 doesn't, phase 3 does.
