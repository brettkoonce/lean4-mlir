# Lean 4 → MLIR → GPU

**The book: [Verified Deep Learning with Lean 4](https://lean.brettkoonce.com/blueprint/)**
([PDF](https://lean.brettkoonce.com/blueprint.pdf)) — the interactive proof blueprint
*is* the book: every theorem clickable, from the `pdiv` primitives to the whole-network backward
passes.

Lean 4 as a specification language for neural networks. Declare the architecture in Lean, render
one StableHLO graph — forward, loss, backward and optimizer fused — from proofs that the backward
is the Jacobian-transpose of the forward, hand the graph to a trusted lowerer, train end to end.
No Python at run time and no autograd library: the gradients are derived at codegen time, in
Lean, and machine-checked over the reals.

Companion code for *Verified Deep Learning with Lean 4*, forthcoming from Apress (Springer
Nature) as the follow-up to
[Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2) (2021).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20402133.svg)](https://doi.org/10.5281/zenodo.20402133)
[![Certify Corpus](https://github.com/brettkoonce/lean4-mlir/actions/workflows/certs.yml/badge.svg?branch=main)](https://github.com/brettkoonce/lean4-mlir/actions/workflows/certs.yml)
[![Verify Proofs](https://github.com/brettkoonce/lean4-mlir/actions/workflows/proofs.yml/badge.svg?branch=main)](https://github.com/brettkoonce/lean4-mlir/actions/workflows/proofs.yml)
[![Comparator](https://github.com/brettkoonce/lean4-mlir/actions/workflows/comparator.yml/badge.svg?branch=main)](https://github.com/brettkoonce/lean4-mlir/actions/workflows/comparator.yml)

**Current version: `v0.7.1`.** Release history in [CHANGELOG.md](CHANGELOG.md).

## Three ways in

| to | do this | needs |
|---|---|---|
| read it | the [book](https://lean.brettkoonce.com/blueprint/) ([PDF](https://lean.brettkoonce.com/blueprint.pdf)), or the [API docs](https://lean.brettkoonce.com/docs/) | a browser |
| check the proofs | `lake exe cache get && lake build ProofsMinimal` (~1 min), then `lake build Certs` for all of them; [The proofs](#the-proofs) below | Lean 4, no GPU |
| train a net | the four setup lines in [the tour](#the-tour), then `lake run mnist` (~1 min) | an NVIDIA or AMD GPU |

## The tour

Four commands, one per scale, in the order the book meets the nets, then the demos. The numbers
are the book's, from the verified XLA path on one RTX 4060 Ti unless the row says otherwise; the
two Imagenette side quests the book does not quote (ResNet-50, MobileNetV4) are medians of five
seeds. Setup is the book's
[Getting started](https://lean.brettkoonce.com/blueprint/app-getting_started.html),
one track per tier; the short form is

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh   # Lean 4
python3 -m venv .venv && . .venv/bin/activate && pip install jax-cuda12-pjrt          # the XLA plugin only (jax-rocm7-pjrt on AMD)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so                        # the shim; needs nothing but libc
lake exe cache get && ./download_mnist.sh && lake run mnist                            # Mathlib oleans, MNIST, tier 1
```

| tier | command | trains | the number | chapter |
|---|---|---|---|---|
| 1 | `lake run mnist` | linear, MLP and CNN on MNIST, 12 epochs each (~1 min) | 92.10 · 97.81 · 98.77 % | [1](https://lean.brettkoonce.com/blueprint/chap-tensor.html) · [2](https://lean.brettkoonce.com/blueprint/chap-mlp.html) · [3](https://lean.brettkoonce.com/blueprint/chap-cnn.html) |
| 2 | `lake run cifar` | the wide 8-conv net on CIFAR-10: SGD / momentum / AdamW × no-BN / BN, 40 epochs at a constant lr (~19 min) | 76.3 % — BN + momentum, median of five | [4](https://lean.brettkoonce.com/blueprint/chap-bn.html) |
| 3 | `lake run imagenette` | seven nets on Imagenette at 224², 80 epochs AdamW, book order (~9 h) | R34 89.50 · R50 89.71 · MNv2 89.25 · MNv4-Conv-M 86.24 · B0 89.96 · ConvNeXt-T 85.07 · ViT-Tiny 68.74 % | [5](https://lean.brettkoonce.com/blueprint/chap-residual.html) · [6](https://lean.brettkoonce.com/blueprint/chap-depthwise.html) · [7](https://lean.brettkoonce.com/blueprint/chap-se.html) · [8](https://lean.brettkoonce.com/blueprint/chap-layernorm.html) · [9](https://lean.brettkoonce.com/blueprint/chap-attention.html) |
| 4 | `lake run imagenet` | the same nets on ImageNet-1k at the paper recipes, 4× 3060 / 4060 Ti, weeks of wall-clock; bare it prints the plan and every row's estimate, `start` runs it | R34 74.06 · R50 (RSB-A3) 77.98 · MNv2 71.91 · B0 76.88 · ConvNeXt-T 81.30 · ViT-Tiny 72.35 % on the verified path, each within 0.3 of its JAX reference; MNv4-Conv-M's 75.48 is a pre-timm-parity reference (rerun queued), its verified run still to come | [Track 4](https://lean.brettkoonce.com/blueprint/app-getting_started.html) |

The demos ride on the chapter nets, in the order Chapter 10 meets them — recognition, then
beyond it; [demos/README.md](demos/README.md) has the command, the figure and the reasoning for
each.

| demo | command | the number | book |
|---|---|---|---|
| object detection | `lake exe yolov1-visdrone-fpn` | VisDrone mAP@0.5 0.2363 — ResNet-34 + FPN at 448 | [10.2.2](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_detection) |
| industrial inspection | `lake exe yolov1-neudet-fpn` (and `yolov1-neudet448`) | NEU-DET mAP@0.5 0.623 with the VisDrone detector unchanged; the single 14×14 grid gets 0.607 here and 0.0391 on the drones | [10.2.3](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_inspection) |
| people watching | `lake exe arasl-signs` | ArASL 98.62 % under the published random split, 77.94 % once each hand's frames stay together | [10.2.4](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_people) |
| agriculture | `lake exe plant-leaf` | PlantVillage 99.57 % on lab leaves, 17.80 % on PlantDoc's field leaves — the CAM and an exact Shapley value say why | [10.2.5](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_agriculture) |
| segmentation | `lake exe unet-brats-r34`, then `brats-predict` | BraTS mIoU 0.741 — a ResNet-34 encoder under a UNet | [10.2.6](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_segmentation) |
| reinforcement learning | `lake exe blackjack-dqn` (environments: `blackjack-env`, `pong-env`) | Double DQN agrees with the exact blackjack policy on 188 of 200 decision states | [10.3.1](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_rl) |
| language | `lake exe tinygpt-shakespeare` (also `bigram-shakespeare`, `tinystories`) | 2.28 bits/char held-out | [10.3.2](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_nlp) |
| diffusion | `lake exe mnist-ddpm-train`, then `mnist-ddpm-sample` | energy distance to MNIST 0.0067 at 50 network evaluations — ancestral DDIM, ahead of the deterministic sampler at 200 | [10.3.4](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_diffusion) |
| physics | `lake exe diffusion-2d muller_brown flow` | a flow-matching Boltzmann generator on Müller–Brown; reweighted by its own density it returns the exact well populations to the last digit | [10.3.5](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_boltzmann) |
| signal processing | `lake exe gw-detect` | LIGO O3a strain: the chapter-4 CNN reaches half detection at network SNR 6.91 (false-alarm rate 10⁻²), where the matched filter needs 10.32 | [10.3.6](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_signal) |
| beyond vision | `lake exe nqs-ising gpt` | the transverse-field Ising ground state at N = 12: a GPT wavefunction within 3.3 × 10⁻⁶ of the exact energy | [10.3.7](https://lean.brettkoonce.com/blueprint/chap-bestiary.html#sec:bestiary_beyond) |

Everything else in the repository is the lab — the ablation and robustness
exes, the tests, the Bestiary — the evidence behind these numbers, one level down. `lakefile.lean`
is grouped the same way: the tour first, then the lab by home directory.

## The proofs

Every layer's backward is proven to be the Jacobian-transpose of its forward over the exact reals
(Mathlib's `fderiv`), composed up to whole-network VJPs for ResNet-34, MobileNetV2,
EfficientNet-B0, ConvNeXt-T and ViT-Tiny, with zero project axioms. For every chapter net the
committed train-step render in `verified_mlir/` is tied to those proofs at the denotational level:
each emitted parameter-update node denotes the certified descent step, and the tiers train on
exactly those bytes. What stays trusted is the ℝ→Float32 numerics, the per-op text printing, and
the lowerer with its runtime. The book's
[On Verification](https://lean.brettkoonce.com/blueprint/app-verification.html)
appendix is the full argument, gap by gap; [LeanMlir/Proofs/README.md](LeanMlir/Proofs/README.md)
is the file-level map, and the [API docs](https://lean.brettkoonce.com/docs/) open on the same map
with every name linked.

Check them without a GPU:

```bash
lake exe cache get           # Mathlib oleans, ~30 s
lake build ProofsMinimal     # the smallest end-to-end tie, ~1 min
lake build Certs             # every certificate CI checks (the long one)
```

`tests/comparator/run.sh` re-runs Lean's kernel typechecker over the headline theorems
independently, and `tests/comparator/Challenge.lean` imports Mathlib and nothing else, so those can
be read and checked without reading a line of this project.

## Two lowerers, one graph

Training runs through XLA/PJRT, and every number above comes from it. IREE is the second trusted
lowerer: it is what the differential oracle (`tests/vjp_oracle/`) lowers the Lean side through, so
that agreement with the JAX reference is evidence from two independent compilers, and each `lake
run` tier has an `-iree` twin. Building it is [historical/IREE_BUILD.md](historical/IREE_BUILD.md).

## Where things are

Each directory has a README that says what is in it and where to start.

- [`LeanMlir/`](LeanMlir/) — the library; [`LeanMlir/Proofs/`](LeanMlir/Proofs/) — the proofs,
  chapter by chapter; [`verified_mlir/`](verified_mlir/) — the committed renders the tiers train on
- [`apps/`](apps/) — one `Main` per exe, by tier; [`demos/`](demos/) — Chapter 10's eleven demos,
  in its order; [`Bestiary/`](Bestiary/) — 45 read-only `NetSpec` catalogue entries, Part 2 of
  the book
- [`jax/`](jax/) — the JAX reference implementations the ImageNet path is ported from, and the
  oracle's ground truth
- [`tests/`](tests/) — the gates: render ties, data-parallel checks, the comparator and the VJP
  oracle; [`ffi/`](ffi/) — the C runtime the lowerers sit behind
- [`scripts/`](scripts/) — gates, figures and the ImageNet jobs (`scripts/jobs/`, `lake run <job>`
  runs one); [`runs/`](runs/) — the logs and READMEs behind the numbers
- [`historical/`](historical/) — the two earlier phases (`mnist-lean4/`, pure Lean 4 with a C BLAS;
  `mlir_poc/`, the Python exporters), the cross-backend traces, the old logs and the upstream bug
  reproducers, the survey this README used to be
  ([README_survey.md](historical/README_survey.md)), and the reference notes:
  [RESULTS.md](historical/RESULTS.md) (per-epoch histories), [BENCHMARK.md](historical/BENCHMARK.md),
  [CUDA.md](historical/CUDA.md), [ROCM.md](historical/ROCM.md), [IREE_BUILD.md](historical/IREE_BUILD.md)
- [`blueprint/`](blueprint/) — the book's source; [`planning/`](planning/) — working notes

## Citing this work

```bibtex
@software{koonce2026,
  author  = {Brett Koonce},
  title   = {Verified Deep Learning with Lean 4: Formal Backpropagation from MLP to Attention, via MLIR},
  url     = {https://github.com/brettkoonce/lean4-mlir},
  doi     = {10.5281/zenodo.20402133},
  version = {0.7.1},
  year    = {2026},
}
```
