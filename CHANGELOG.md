# Changelog

Release history for *Lean 4 → MLIR → GPU*. The README keeps only the current
version; older entries live here.

## v0.7.3 — Every pair on its paper recipe, five seeds a row

Every ImageNet pair trains the recipe its paper trained, on both paths, and
all seven have run: ResNet-34 on Bag of Tricks **74.14%** against the
reference's 73.98 (McNemar p = 0.29); ResNet-50 **76.99%** against 77.01 on
the 2018 recipe (p = 0.88) and **78.05%** against 77.58 on RSB-A3 at timm's
eval resize (p = 0.0002, the one pair that parts, with seeds per arm queued);
MobileNetV2 on TF-slim's 420 epochs **71.69%** against 71.84 (p = 0.30);
EfficientNet-B0 on TensorFlow's recipe **77.21%** against 77.17 (p = 0.77);
ConvNeXt-T on the official recipe **82.15%** against 81.99 (p = 0.18);
ViT-Tiny on DeiT's **71.61%** against 71.64; and MobileNetV4-Conv-M's
100-epoch tier **77.42%** against 77.74 (p = 0.014). The `[TODO]` markers are
gone, Appendix C carries the seven pairs in one table, and each ImageNet
section is written anchor-then-diff: the recipe stated once, every later one
as a diff from ResNet-34's. Below ImageNet every number is five seeds. The
Imagenette column reads median and range per row (EfficientNet-B0 89.96,
ResNet-34 89.78, ResNet-50 89.63, MobileNetV2 88.87, ConvNeXt-T 83.75,
ViT-Tiny 68.61) and its transcripts are the median seeds; Chapter 4's CIFAR
board sits on one date with batch-statistics BatchNorm as the headline
(momentum 77.4%), two precision claims that did not replicate at five seeds
are gone, and the one no-BN momentum seed that never left chance now carries
the point that a verified gradient is not trainability. Lean 4.35.0.

The proofs: the Chapter 1–3 train steps are tied at batch N; Chapter 4's
batch-statistics net has its whole-net half and its loss-gradient rung; every
optimizer the book runs has its statement (momentum, AdamW, LAMB, the global
clip, gradient accumulation, the EMA shadow, RMSProp); the MobileNetV4 chain
is generic in its feature side; and the ResNet stem pool's backward is nine
stride-2 slices with no `select_and_scatter`, which every IREE backend
lowers. `#audit_axioms` checks the three-axiom closure in Lean rather than by
grep, the comparator runs as four CI legs with seconds per checker, `lake
test` and `lake lint` drive the gates, every file is on the module system,
`lake shake` settled the imports, and a 279-file audit rewrote the docstrings
that said more than their code and fixed the driver and codegen bugs it
found, with the repeated proof and emitter blocks written once. The book is
Part 1, Part 2 and the appendices, with Getting started and a Foundations
page in front, every figure and table numbered and cited, a Recap closing
every Part 1 chapter, listings held to 88 columns and checked against the
tree in CI, and two passes that cut what the reader did not need: the
development history and the operational detail. Appendix B opens on the
theorem budget and closes on why the three kinds of checking overlap; the
ladder is named by its commands and the platforms table is generated from
the record.

Part 2 adds MuZero on tic-tac-toe against the solved game, and every compared
table carries three seeds: VisDrone, NEU-DET, BraTS, the Boltzmann generator,
Pong's state arms and the TinyGPT ladder against a Kneser-Ney 7-gram; Leduc
poker replaces blackjack. Platforms: the Intel Arc Pro B50 on oneAPI runs the
MNIST, CIFAR and Imagenette tiers, each net inside CUDA's range or two images
from it; the Jetson Orin Nano 8 GB runs the platform suite and all seven
Imagenette nets, ConvNeXt-T as four accumulated micro-batches of eight; IREE
3.13 runs on a box without a GPU; AWS Trainium is prepared off-device; and
the platforms table names each row by its card.

## v0.7.2 — The pairs read paired, and five platforms

The ImageNet pairs are read the way the field reads them. Every reference
number is its final checkpoint scored over all 50,000 validation images
through an fp32 forward, every verified run writes a per-image bitmap, and the
two paths are compared by McNemar on the same images rather than by two
confidence intervals. ResNet got the full treatment — the fp32 head, zero-γ,
BN decay 0.9, timm's BCE target threshold, torchvision's crop, then timm's own
schedule (the cosine stepped once an epoch from 1e-4 to a floor of 1e-6, Mixup
or CutMix drawn per batch, RandAugment's fill at the image mean), on both paths
at once: ResNet-34 **74.17%** against the reference's 74.14 (not separable),
ResNet-50 **77.09%** against 76.76 on the 2018 recipe (p = 0.015) and
**78.08%** against 77.61 on RSB-A3 (p = 0.0002, the verified checkpoint ahead
on both). MobileNetV2's 350-epoch pair lands **71.12%** against 71.63,
MobileNetV4-Conv-M **76.68%** against 76.57, and ViT-Tiny **72.42%** against
72.07 with a six-item ledger of what the two paths still do differently. The
other chapters were audited line by line against the code that trained the
papers' numbers — TF-slim's MobileNetV2 command, TF TPU EfficientNet's B0 with
`SAME` padding carried through the ties, MobileNetV4's own config, the
ConvNeXt and DeiT recipes — and wired on both paths behind gates and smokes;
each chapter carries a `[TODO]` marker that names exactly what its rerun will
close, and the runs are queued. Lean 4.35.0.

Every parameter-gradient node of the seven ImageNet nets IS the loss's
derivative, stated as a theorem per net; the ties reach the bf16 renders by
erasure and the training masks (stochastic depth, classifier dropout); the
exact GELU x·Φ(x) joins the tanh form through erf and erfc, with an exact twin
of every ViT and ConvNeXt artifact; the pooling condition is "each window's
max attained at one cell"; Appendix C scopes the proofs to the gradient and
the theorem statements take the Lamport form. Behind that, the rubric review,
the API-design pass and the placement cleanup: 71 modules into 11 group
directories, 939 same-file helpers made private, docstrings that state their
statements, `TRUST.md`, a README in every directory a first visit opens, and
the comparator re-run as `lake comparator --paranoid`.

Platforms are a suite and a matrix — `scripts/platform/check.sh`,
`PLATFORMS.md`, one profile — on CUDA 12 and 13, ROCm, the Jetson Orin Nano
(a runbook from fresh JetPack to `lake run imagenette-orin` at 8 GB) and,
new, Apple Silicon on Metal through jax-graft's PJRT plugin, the MNIST tier
matching CUDA on an M1 Air; device residency is the default and the shims
carry a uint8 wire. Chapter 10 grows to sixteen demos: remote sensing (which
Sentinel-2 bands travel from Europe to the Amazon and the Cerrado), a CASP16
distogram ResNet scored against the field, and a game-theory section with an
exact instrument for each game — DQN from pixels on a Pong written in Lean,
above its own state twin; Deep CFR on Leduc hold'em against the exact best
response; AlphaZero on tic-tac-toe against the solved game, up to 5×5 — while
BraTS carries its 3D verdict on two seeds a side. The book gains a rendered
preview diff (`scripts/book/blueprint_preview.py`) and its Platforms table.

## v0.7.1 — The verified path runs the paper recipes

The verified path runs the paper recipes to the end. Six of the seven chapter
nets now train on ImageNet-1k through the certified render, in bf16 on 4× RTX
3060, and land beside their JAX references: ResNet-34 **74.06%** (reference
74.16), ResNet-50 **77.98%** on RSB-A3 (78.26; and 77.07 on the 2018 recipe,
ahead of its 76.95), MobileNetV2 **71.91%** (71.90), EfficientNet-B0 **76.88%**
(77.15), ConvNeXt-T **81.30%** (81.51 — McNemar against the reference's raw arm,
p = 0.10, not separable) and ViT-Tiny **72.35%** (72.31). Every ImageNet chapter
prints the two paths side by side; MobileNetV4-Conv-M's 75.48% stays the
reference's until its run. Getting there found what only a paired run could —
the ConvNeXt pair's two arms did not share a weight init, and its layer scale
started at one where the paper says 10⁻⁶ — and rebuilt the trainer's plumbing:
ImageNet validation streamed per pass instead of held as 30 GB, evaluated on
every replica (101 → 29 s per ConvNeXt epoch) and every N epochs as the
reference does, BN running statistics checkpointed atomically beside the blob,
shim loaders replaced every N epochs, the host-memory "leak" traced to mimalloc
and the p90 tail to a stale shim, and every job's ms/step re-measured to find
the feed is the constraint. The data-parallel render is synchronised BatchNorm:
a sync-BN kit of four StableHLO ops with Chan's variance, and the proof that the
R-replica step IS the single-device step at the global batch, for ResNet-34 and
-50, MobileNetV2 and V4 and EfficientNet-B0, with the bf16 sharding gates beside
it. Every kinked net gets its non-degeneracy seal on the full-width batched net
it ships — ResNet-34, ResNet-50 at both shipped resolutions, MobileNetV2,
MobileNetV4 — and the two-channel proxies retire. Lean 4.34.0.

The proof corpus is 11% lighter and says the same things. A Mathlib-reuse
audit and a census of what the book cites took `LeanMlir/Proofs/` from 172k to
153k lines: fourteen Jacobians of affine maps fall to one `pdiv_of_affine`,
`fun_prop` closes what 31 hand differentiability proofs did, one generic
batched-lift lemma replaces 32 per-op ones, 182 tie clauses fold into per-node
Props and the BN γ/β pairs into one, the SgdDescentCnn slots and the CertLayer
block composites each collapse a family, and the superseded tiers — the
per-example ResNet-34 and MobileNetV2, the scalar-LN ViT, the audit-only
restatements, some three hundred pinned declarations — retire. Three findings
came out of the census rather than the sweep: the emitted `bnBatchBack` node and
the ties' `bnBatchLABack` denote one map, the two spellings of the stem-pool
scatter are one map, and the MLP train-step tie now states all six parameter
ops. Compile time followed — SmoothingCP's tail check 97 → 22 s, 57 dead
heartbeat bumps out, the IBP/CROWN scorecards built in parallel. The comparator
re-checks **87** theorems (was 52), the tie, faithfulness and certificate tier
included; `docstring-checkrefs` resolves private, `File.decl`, module and
namespace citations and demands that a cited file be a real link, and the three
test defects the audit turned up are fixed — `parseFloat` read `abc` as 5451.0,
so a mistyped-learning-rate guard never fired. The blueprint's ladder is 38
whole-net nodes across chapters 2–9 with its `\uses` generated from the Lean
dependency walk and gated, the dependency graph is drawn in chapter order with
portal nodes, and the Certs modules render into the API docs so every `\lean`
link resolves.

Chapter 10 gains seven demos, three with a theorem for a ceiling, and the front
door takes the book's shape. The VisDrone detector runs unchanged on NEU-DET
steel defects — mAP@0.5 **0.623**, where the single 14×14 grid gets 0.607
against its 0.0391 on the drones, which is when multi-scale detection pays;
chapter 4's CNN on Arabic sign-language letters scores **98.62%** under the
published random split and **77.94%** once each hand's frames stay together;
chapter 6's ResNet-34 reaches PlantVillage's **99.57%** in the lab and
**17.80%** on PlantDoc's field leaves, with the CAM, an exact two-player Shapley
value and the counterfactuals as the diagnosis; a Double DQN through the DDPM
MSE block agrees with blackjack's exact policy on **188 of 200** decision
states; a flow-matching Boltzmann generator on Müller–Brown, reweighted by its
own density, returns the quadrature's well populations to the last digit; the
chapter-4 CNN on LIGO O3a strain reaches half detection at network SNR **6.91**
where the matched filter, scored against its own closed form, needs 10.32 in
real noise; and a GPT wavefunction lands within **3.3 × 10⁻⁶** of the Ising
chain's exact ground-state energy. The chapter splits at DQN — recognition,
then beyond it. The book's shape is now two generated figures (the six
certificates by chapter, the chapter citation graph), 239 stress italics are
gone, and the site is at lean.brettkoonce.com with a 404 page, a sitemap and a
robots.txt that ended 48 GB of overnight crawler traffic. This release puts
that shape on the front door: the README and the home page quote the verified
path's numbers, their demo tables follow Chapter 10's order with a link into
each section, `demos/README.md` is reordered the same way under the chapter's
figures with the missing signal-processing entry written, and the API docs no
longer open on doc-gen4's bare index — `LeanMlir.lean`'s module docstring is
the proofs map, gated like every other docstring, and CI makes its page the
landing.

## v0.7.0 — The paper nets, tied at the artifact that trains

The verified path reaches ImageNet-1k at the paper recipes. ResNet-50 arrives
from nothing — three bottleneck VJPs, a 25,557,032-parameter train-step
renderer, LAMB, BCE-with-logits, gradient accumulation and the 160/224
resolution split, each certified — and trains RSB-A3 on the verified path to
**77.91%**, beside its JAX reference's 78.26. MobileNetV4-Conv-M gets the Universal
Inverted Bottleneck's backward, its fold, and a 4-GPU job (**75.48%**);
ConvNeXt-T re-runs 300 epochs on the current artifact at **81.53%**,
EfficientNet-B0 350 paper-faithful epochs at **77.15%**, and ResNet-34,
MobileNetV2 and ViT-Tiny land at 74.16, 71.90 and 72.31. ViT-S/B and
ConvNeXt-S/B exist by widening one renderer each. bf16 goes through the
verified renderer — 27 ops, all seven nets, **1.3–2.5×** per step on real
ImageNet once pointwise ops run at their 4-D shape — and fp8 enters the
graph. `timm` is pinned as the reference environment, its validation protocol
and resampler adopted, and the sweep it enabled found RandAugment's Solarize
breaking every ImageNet run. Data parallelism gets its all-reduce as an AST
node with a proof of which function a data-parallel run minimises, device
residency and per-net feed tuning, and one supervisor engine whose job
configs check the recipe their own name claims. Every accuracy carries a
Wilson interval; Chapter 4's optimizer ablation is re-run at a constant
learning rate over five seeds, with bf16 as a third lever.

The Proofs tier reaches every net at the BatchNorm its trainers actually
run. All seven — ResNet-34, MobileNetV2, ResNet-50, EfficientNet-B0,
ConvNeXt-T, ViT-Tiny, MobileNetV4-Conv-M — carry the typed forward graph, the
fold of every gradient node, and the §1a tie of every parameter update to the
certified chain, at batch BatchNorm and at the ImageNet head; the capstones
are re-pointed to the un-fused gradient and the smoothed loss, and every net
is one renderer chain (the per-example renderers retired, byte-identically
where the artifacts allowed and with new folds where they did not). Whole-net
backward ties land for ResNet-34, MobileNetV2, EfficientNet-B0, ConvNeXt-T and
ViT-Tiny; the nine bf16 gradient kinds fold once for all nets. The float tier
produced ImageNet-scale numbers — the deployed ResNet-34 inference forward
within 1.548e209 of its real forward, MobileNetV2 within 1.444e96 — and, those
being vacuous, was cut back to `FloatClose` and the ℝ chains, 44k lines
lighter. The proof tree is one home per net (`Proofs/Nets/<family>/`, 114
files moved) with `Fold` / `StepTie` names.

The demos and the front door. The VisDrone detector is rebuilt — focal loss,
an FD-verified FPN neck, box-aware scale augmentation — to mAP@0.5
**0.2363**, 54% over its PyTorch twin, and deploys through TensorRT at 35.7
fps on an Orin Nano; BraTS's data-pairing bug is found and plain
cross-entropy wins (mIoU **0.742**); DDPM on MNIST gets the Score-SDE
samplers and the η sweep where ancestral sampling at 50 evaluations beats the
deterministic sampler at 200; 2-D diffusion and TinyStories join. The book
gains the three demo figures, the ImageNet chapters reprinted from the runs,
and Chapter 5's ablation as a two-precision figure. The repository becomes a
tour: `lake run imagenet` is the fourth tier (plan-only bare, `start` to run,
one `lake run <job>` per config), the README is 118 lines with the book's
numbers, `lakefile.lean` says which of its 233 exes are the tour and which
the lab, and the two earlier phases and the reference notes live in
`historical/`. Along the way the box lost its two PCIe-faulting cards and
every device list followed.

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
[`historical/traces/CROSS_BACKEND_RESULTS.md`](historical/traces/CROSS_BACKEND_RESULTS.md) for the
four-corner verification tables.
