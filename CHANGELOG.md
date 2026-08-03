# Changelog

Release history for *Lean 4 → MLIR → GPU*. The README keeps only the current
version; older entries live here.

## v0.6.3 — The PJRT backend

A second trusted lowerer. `ffi/pjrt_ffi.c` implements the same C surface as
the IREE shim — symbol-identical under `nm -D`, so nothing above the shim
changed and the backend is whichever `.so` a binary linked — and hands the
proven graph to XLA through the PJRT C API, with no Python at run time. What
executes is `pretty(provenGraph)`: the same rendered artifact the proofs
reason about, not a hand-written emitter. ResNet-34 on Imagenette goes from
IREE's 1702 ms/step to **162**, within 1.04× of hand-written JAX per step;
ViT lands at 128 ms/step (9.2× IREE) with an 80-epoch run at 71.31%. The
AdamW scorecard reaches **6 of 6** — every whole-net render certified, each
swap licensed by a numeric tie verified to fail — and with §2i's cifar8 port
every committed artifact is `Proofs/`-rendered with no carve-out. Data
parallelism is gated by the exact batch-decomposition identity, and
device-resident parameters stop the `[θ|m|v]` blob crossing PCIe, taking
4-GPU ResNet-34/ImageNet from 596 to 386 ms/step.

ImageNet becomes real. All five nets — ResNet-34, ViT-Tiny, ConvNeXt-T,
EfficientNet-B0, MobileNetV2 — have gated trainers matching their JAX
reference parameter counts exactly, and the recipe gap closes behind them:
RMSProp (TF-flavoured, ε inside the sqrt), the EMA weight shadow,
stochastic depth, mixup/cutmix over a soft-target wire that needed no render
change because the renders are affine in `%onehot`, global-norm gradient
clipping, `wdExcludeNormBias`, and classifier dropout. ConvNeXt and ViT move
to the batched index `N := B`, byte-identical to their committed artifacts.
ConvNeXt gets its **real** channel LayerNorm — parameter count exact at
28,587,592, the scalar-LN chain deleted, 180/180 bit-identical on the render
capstone. Five copies of one label-smoothing bug, hardcoded at K=10 across
four nets, were found and derived.

Two new demos and a sharper trust story. A 2D UNet segments brain tumours on
MSD Task01 (BraTS) with a validated soft-Dice loss, a focal-CE arm chosen by
gradient scorecard, and an honest WT-0.66 verdict; detection grows anchor-YOLO
with DIoU, an FD-verified FPN neck, and multi-scale target encoding that lifts
coverage 60.9% → 88.2%. `tests/comparator/Challenge.lean` now **imports Mathlib
and nothing else** — the 13 architecture-free calculus theorems, including the
`fderiv` pin, with their vocabulary inlined; the 39 statements about specific
networks live in `ChallengeArch.lean`, which says why importing them beats
copying every architecture into the challenge. 52 theorems, both pairs green.
1524 theorems close over the three core axioms, 134 artifacts carry one writer
each, 41 bestiary binaries match their golden parameter table, and the
toolchain moves to Lean 4.32.2 (past the v4.32.1 kernel soundness hotfix) with
zero source changes and a byte-identical artifact regen.

## v0.6.2 — Certified robustness

The robustness ladder becomes the release. PGD attacks run against every
verified net (linear → MLP → CNN → CIFAR-10+BN) through the same IREE
pipeline that trains them, and two certificate families answer back.
Lipschitz-margin certificates (Tsuzuku 2018) are formalized and pushed to
the full 784-dim input via an in-kernel rational dot-product engine, with
per-pair LipSDP closing the certificate↔PGD sandwich (93/100 at ε=0.1
matches PGD, kernel-verified). Randomized smoothing (Cohen 2019) is carried
end to end into a theorem: the Monte-Carlo tie, exact Clopper–Pearson
arithmetic, kernel-checked decimal quantile bounds, and the Gaussian ladder
(Neyman–Pearson in 1-D and n-D, quantile inversion, the Cohen radius with
every hypothesis discharged) — 279 driver-reported radii certified across
three scorecards, spinning off the first two Mathlib upstream drafts (cdf
continuity, `gaussianReal` facts). A dispatch-only certs-heavy CI tier
carries the generated certificate corpora.

Elsewhere the A3 FloatBridge matrix completes — all five Imagenette nets ×
{forward, backward}, ViT's MHSA backward included — and the committed-spec
SpecVJP ties now cover all five nets. Muon's Newton–Schulz iteration gets a
convergence story (the tuned quintic is band-landing, not convergent; a
principled convergent quintic joins it). FlashAttention forward/backward
emitters land with the O(T²)→O(T·b) memory payoff measured, RoPE brings
length extrapolation, and TinyStories runs at 8K context. Chapter 9's
ViT/attention proofs are rewritten in Lamport's structured style (33
theorems). On the JAX bridge, ConvNeXt-T reaches 78.13% ImageNet top-1
(80 epochs, 4-GPU). The toolchain moves to Lean 4.32.0 / mathlib v4.32.0,
and the three-axiom CI gate now accepts axiom-closure subsets.

## v0.6.1 — Verified training reaches low precision

A FloatBridge proof layer carries the MNIST chain into fp8 (E4M3) and
bf16-mixed: per-operation rounding budgets, a "one binary32 SGD step
decreases the loss" descent theorem, and argmax-preservation under
quantization (with an E4M3 MNIST-linear demo). Chapter 4 is recast as the
MNIST→ResNet bridge — the same 2×512 head on a deeper conv body — with a
controlled SGD / momentum / AdamW × BatchNorm optimizer ablation (momentum
wins; head width barely moves the result). The base toolchain moves to
Lean 4.31.0. On-ramp polish: a `ProofsMinimal` "hello world" build target,
refreshed ROCm/CUDA setup guides (`ROCM.md` / `CUDA.md`), and pinned
per-backend JAX comparator environments.

## v0.6.0 — Object detection

Object detection joined the framework — a YOLOv1 person detector on Pascal
VOC off Chapter 5's ResNet-34 backbone (1×1 convolutional detection head),
plus global-norm gradient clipping, env-var checkpoint resume
(`LEAN_MLIR_INIT_LOAD` / `LEAN_MLIR_START_STEP`), per-step LR warmup, and
demo-anchored blueprint intros for detection and diffusion.

## v0.5.7 — Audits closed

Two parallel-agent audits closed. The "canonical `correct := rfl`" pattern
at non-smooth operators (ReLU, the composed MLP, MaxPool2) now has
machine-checked smooth-point bridges: `relu_codegen_matches_canonical` and
`maxPool2_codegen_matches_canonical` prove the canonical-witness backward
equals the codegen formula wherever every coordinate avoids the kink. A
`HasVJPAt` pointwise framework provides smooth-input variants of the three
kinked-operator instances whose `correct` field is a real chain-rule proof
rather than `rfl`. The comparator suite extends from 38 → 41 theorems
independently kernel-rechecked against `[propext, Quot.sound,
Classical.choice]`. Blueprint gets a half-dozen flow improvements (GAP
defined at first material use in Ch 5, Diffusion split into its own Bestiary
subsection, ResNet entry expanded to the full standard family including
R-18, Tomáš Skřivan's *Scientific Computing in Lean* credited at the top of
the acknowledgments). Android bottom-cutoff bug fixed (issue #2); Umami
cookieless analytics replaces planned GA. First Zenodo deposit lands with
this release.

## v0.5.6 — ConvNeXt + data augmentation

Chapter 8 lands its ConvNeXt-T worked example (84.94% val on Imagenette,
paper-faithful recipe); Chapter 9 gets a Data Augmentation section with a
9-row ViT recipe ablation table — CutMix is the load-bearing knob at 9.5K
images, and stacking RandAugment + Random Erasing on top of it *hurts* val
accuracy. Bestiary gets paper-exact entries for VGG, ResNet-50/101/152, WRN,
and DenseNet, plus the "N new primitives" claim reframed around the Ch 1-9
reader's toolbox (what's free) rather than the codebase (what's already in
`Types.lean`). Found and fixed a long-standing eval-pipeline bug along the
way: `centerCrop` was running on already-224 val data, reading past
per-image bounds and making heavy-aug runs appear to collapse. New
`LEAN_MLIR_EVAL_ONLY=1` mode re-evals saved checkpoints in ~5 sec each.

## v0.5.5 — Swish/SiLU + the VJP oracle

Swish/SiLU as a first-class activation (forward + backward + proved
`swish_has_vjp_correct`) plus the independent-kernel comparator re-check
covering 38 theorems via public `*_has_vjp_correct` wrappers, and Ch 1's
"Why VJPs, not Jacobians?" bridge + canonical-pdiv witness explainer +
three-pillar TikZ spine diagram.

On top of that, a differential-test suite in
[`tests/vjp_oracle/`](tests/vjp_oracle/) uses JAX's `value_and_grad` as an
oracle for the hand-derived VJPs in `LeanMlir/Proofs/`. Nine test cases
cover every axiom family — dense, conv, BN, maxPool, residual (biPath),
depthwise, SE (elementwise product), attention, and the transformer block —
each verified to 1–2 ULP of JAX autodiff.

## v0.5.4 — Cross-backend ULP-floor agreement

ULP-floor cross-backend agreement (Lean→IREE→GPU vs Lean→JAX→XLA on both
NVIDIA and AMD); see
[`traces/CROSS_BACKEND_RESULTS.md`](traces/CROSS_BACKEND_RESULTS.md) for the
four-corner verification tables.
