# Verified VJP Proofs

Machine-checked proofs that the backward pass (VJP) of every layer
matches its forward-pass Jacobian. Zero `sorry`s. If `lake build Certs`
succeeds, every theorem is correct.

> New here? Read the **Start here** section directly below before the rest of this
> file — it's a reference, not an on-ramp.

## Directory layout

The 242 proof files are filed into seven buckets
(split rationale: `planning/archive/proofs_directory_refactor.md`; the per-net
tree dates from 2026-09-08, `planning/cleanup_backlog.md` §10). The Lean
namespace is `Proofs.*` throughout — only module paths carry the bucket:

| directory | what lives there |
|---|---|
| [`Foundation/`](Foundation/) | the roots (`Tensor`: pdiv/HasVJP kit; `MLP`; `DataParallel`; `OpaquePrefix`; the batched conv-net kit (`BatchedStages` → `BatchedStageLayers` → `BatchedBackLinks`, `GradNodesB` for the batched gradient nodes, `SgdNodes` for the per-example fused-SGD ones, `HeadLayers` for the stem pool / GAP / dense, `IndexCast` for the `c·h·w ↔ c·(h·w)` relabelling, `DataParallelSyncKit` for sync-BN); `GramQ`, `ListDot`, `Muon*`, `MatBridge` — opt-in `Mat` ↔ Mathlib `Matrix` interop) plus cross-net kits that sit *above* some nets (`IR`, `CertifiedChain`, `HeadLayers`, `BackwardMaps`, the batched-VJP and data-parallel-sync calculus, `Bf16GradNodes`, the conv interval bounds). Directories are by content; no Foundation file imports a net or a certificate. |
| [`SpecVJP.lean`](SpecVJP.lean) | the apex, not a starting point: the executable `VerifiedNetSpec`s (`LeanMlir/VerifiedNetsCore.lean`) denote the functions the proofs are about, and each net's whole-model VJP is stated at the spec's denotation |
| [`Architectures/`](Architectures/) | generic ops: `Attention`, `CNN`, `BatchNorm`, `LayerNorm`, `Depthwise`, `SE`, `Residual`, `MaxPool3s2`, the channel-LN and depthwise backward ties, and the per-op parameter-gradient bridges (`ConvGrad`, `PerChannelBNGrad`, `TokenParamGrad`) |
| [`Nets/`](Nets/) | one directory per net family — `Small/` (MNIST linear/MLP/CNN, CIFAR), `ResNet/`, `MobileNet/`, `EfficientNet/`, `ConvNeXt/`, `ViT/`: each net's forward, VJP, folds, step ties and whole-net backward ties |
| [`Float/`](Float/) | the rounding model: `FloatBridge`, `Binary32Instance`, bf16/E4M3, the ResNet-34 float chain |
| [`Codegen/`](Codegen/) | `StableHLO` (the `SHlo` AST and its `den` semantics), `StableHLOPretty` (the printer), the per-net `*Render*` artifact writers, `IRPrint` (a scratch-only execution oracle) |
| [`Certificates/`](Certificates/) | Lipschitz + smoothing scorecards — **machine-emitted**, see its README |
| [`Training/`](Training/) | `SgdDescent*`, Jacobian seals, trained witnesses; `Optim/` — the ℝ optimizer specs the emitted optimizer ops denote (`AdamStep`, `SgdMomentumStep`, `RmsPropStep`, `Lamb`, `GradClip`); `DropPath` |

## Two populations, two build targets

The files here are not one homogeneous suite; they split along the seam
the lakefile's libs encode (rationale: `planning/archive/repo_shape_deletion_audit.md`):

* **The engine slice — `lake build Proofs`** (23 roots reaching 50 modules, the default target):
  the IR/render layer every demo exe's import cone actually reaches —
  `StableHLO`/`IR`/`Tensor`, the per-net op+VJP modules the proven renderers
  are built on (`Attention`, `CNN`, `MLP`, `BatchNorm`, `MobileNetV2`, …),
  and the renderers CI's drift guard re-elaborates (`*Render`). If you're
  here to understand how "verified trainer" works, this slice is the whole
  story.
* **The certificate corpus — `lake build Certs`** (187 roots reaching 234 proof modules, ~121k lines):
  research results *about* the engine that no demo imports — the float model (`FloatClose`),
  the §1a tie certificates (`*Fold`/`*StepTie`), trained-net seals, SGD-descent
  capstones, the Lipschitz/LipSDP robustness scorecards, Muon geometry, the
  binary32/E4M3 hardware models, the lexer/SSA syntactic line. Built and
  three-axiom-audited by `.github/workflows/certs.yml` (proof-path pushes +
  nightly cron), not by the per-push workflow.

## Start here — the minimum working set

The suite is large, but you don't need the big files to understand it. Two things are
proved for each net, and the **Linear classifier** shows both in ~650 lines total:

1. **Faithfulness** — the *emitted* StableHLO train-step denotes the *certified*
   forward + gradient + SGD math.
2. **Descent** — that SGD step provably *decreases the loss*.

Read these three, in order:

1. [`LinearTrainStep.lean`](Nets/Small/LinearTrainStep.lean) (~170 L) — the linear train-step spec + ops.
2. [`LinearFold.lean`](Nets/Small/LinearFold.lean) (~110 L) — **capstone**: emitted step = certified math.
3. [`SgdDescentLinear.lean`](Training/SgdDescentLinear.lean) (~370 L) — **capstone**: that step decreases the loss.

Build *just* this slice (Linear + the shared foundation it needs, nothing else):

```bash
lake build ProofsMinimal
```

**Foundation (read once; shared by every net — big because reusable, not per-net work):**
`Tensor.lean` (chain rule / `fderiv`), `StableHLO.lean` (the AST + `den` denotation),
`FloatBridge.lean`, `IR.lean` (the small-net backward IR; ResNet onward uses `StableHLO`'s `SHlo`).

### The per-net file chain

The four conv nets (ResNet-34, ResNet-50, MobileNetV2, MobileNetV4) share one chain; learn it on
one and the others read the same:

`*BackB0` → `*FullB` → `*FullBVJP` → `*FullBSeal` → `*StepTieB` → `*BackChains` +
`*WholeBackCertifiedTieB` → `*SyncB` / `*SyncStepTieB`

| tier | R34 | R50 | MNv2 | MNv4 | ENet-B0 | ConvNeXt-T | ViT-Tiny |
|---|---|---|---|---|---|---|---|
| block backward graphs | BackB0 | BackB0 | BackB0 | BackB0 | BackB0 + BackNet | BackB0 (per-example) | BackB0 + BackNet |
| forward + **T2** graph | FullB | FullB | FullB | FullB | FullB0 | FullT | DepthK (+ VecLN, MultiHead, FwdGraph) |
| **T1** VJP | FullBVJP | FullBVJP | FullBVJP | FullBVJP | in FullB0 | in FullT | in DepthK |
| non-degeneracy seal | FullBSeal | FullBSeal | FullBSeal | FullBSeal | — | — | — |
| **T3** fold (un-fused, batched) | FoldB | R34's | FoldPaperG | the others' | FoldG | FoldGB | FoldGB |
| **T3** train-step tie | StepTieB | StepTieB | StepTieB | StepTieB | StepTieG | StepTieGB | StepTieGB |
| **T6** whole-net backward | BackCertifiedTieB | WholeBackCertifiedTieB | WholeBackCertifiedTieB | WholeBackCertifiedTieB | FullWholeBackCertifiedTie | WholeBackCertifiedTieB | WholeBackCertifiedTieB |
| data-parallel (sync BN) | SyncB / SyncStepTieB | same | same | same | SyncB / SyncStepTieG | — | — |

**Tiers.** T1 = the whole-net VJP exists (`*_has_vjp_at`); T2 = the emitted forward graph denotes
the net's forward (`*FwdGraph*_faithful`); T3 = every emitted parameter-update node denotes the
certified gradient step (`*_net_tied*`); T6 = the emitted whole-net backward denotes the net's VJP.

**Suffixes** (what they mean today; several are historical):

| suffix | meaning |
|---|---|
| `B` | batched index (`Vec (N * …)`), and for conv nets batch-statistics BatchNorm |
| `G` | tied at the RAW gradient node (`*GradB`) that every optimizer tail consumes, not the fused `θ − lr·g` node |
| `GB` | both. ⚠ `B`, `G`, `GB` and `PaperG` all name the same T3 tier — each records which axis differed from that net's first version (R34's `FoldB` is also un-fused; ENet's `FoldG` is also batched) |
| `B0` | in `*BackB0`: "block backward", a name inherited from the EfficientNet-B0 spike. In `EfficientNetFullB0`: the B0 model |
| `PC` / `Eval` | per-channel per-example BN / frozen running statistics |
| `Full` / `Paper` / `T` | paper depth / MobileNetV2's [t,c,n,s] table / ConvNeXt-T |
| `V`, `MH`, `K` | vector-`[D]` LayerNorm / multi-head / depth-`k` (ViT) |
| `Xla` | XLA-`SAME` (asymmetric) padding |

Declaration suffixes: `_faithful` / `_den` / `_eq_vjp` all say "this graph denotes that math", at
graph, gradient-node and backward-chain granularity; `_certified` / `_tied*` are ties to a
certified step; `…Tied*` are the per-node clause `Prop`s a tie is a conjunction of.

**Namespaces do not follow file names**, for history: `GradNodesB` holds `ResNet34PoCB`,
`EnetPoCG`, `Mnv2PaperPoCG` and `CnxPoCGB` (the batched f32 gradient-node lemmas, named for the net
that first needed each op), `ResNet34StepTieB` → `ResNet34TieB`, `ConvNeXtStepTieGB` → `CnxTiePoCGB`, `ViTStepTieGB` →
`ViTTiePoCGB`, `EfficientNetStepTieG` → `EnetTiePoCG`, `MobileNetV4StepTieB` → `Mnv4TieB`. The
`PoC*` namespaces are the production tier. The batched stages and their VJPs are in
`Foundation/BatchedStages`, the batched backward graphs and cotangent steps (`reassocB`, `cInB`,
`reluMaskB`, …, in namespaces `EnetTiePoC` / `ResNet34TieB`) in `Foundation/BatchedBackLinks`, and the
sync-BN twin kit (namespaces `ResNet34SyncTieB` / `MBConvSyncTieB`) in `Foundation/DataParallelSyncKit`.

**Don't start with the big files:** `SgdDescentCnn.lean` (~6.8k), `Attention.lean` (~2.3k), the
`StableHLO.lean` denotation internals, or the per-net `*Render*` files (1–2k lines each of
string assembly, no theorems). Start from a `*FullB` file — the net's forward, stated once.

## Foundation: Mathlib's `fderiv`

Earlier drafts of this suite axiomatized the entire calculus
foundation — chain rule, sum rule, product rule, identity, reindex —
as eight opaque facts. The current foundation **flips that**:
`pdiv` is *defined* in terms of Mathlib's Fréchet derivative
`fderiv`, the structural rules are *theorems* proved from Mathlib's
API, and every downstream chapter threads a `Differentiable`
hypothesis through its compositions.

Many later-chapter axioms have been pruned the same way. Where
earlier drafts axiomatized `conv2d`, `maxPool2`, `depthwiseConv2d`,
or `geluScalar` as opaque functions with stated Jacobians, the
current version defines them concretely and proves their
gradient-related lemmas from the foundation.

The diff-threading branch closed out every remaining "provable but
deferred" Jacobian: `pdivMat_rowIndep`, `pdiv_softmax`,
`softmaxCE_grad`, `pdiv_gelu`, `pdiv_bnIstdBroadcast`, the BN
inverse-stddev smoothness, the row-wise softmax smoothness, and all
seven transformer-level composition chains.

The progression: **30 → 0 project axioms.** See `planning/archive/VJP.md` (foundation
flip and per-chapter migration) and `planning/archive/pdiv.md` (final 4-axiom retirement)
for the full elimination history.

## Dependency graph

```
Tensor.lean                    ← pdiv (def via fderiv) + VJP framework
  │
  │  pdiv_comp (chain rule)         ← theorem
  │  pdiv_add  (sum rule)           ← theorem
  │  pdiv_mul  (product rule)       ← theorem
  │  pdiv_id   (identity)           ← theorem
  │  vjp_comp  (VJP composition)    ← theorem
  │  biPath    (additive fan-in)    ← theorem
  │  elemwiseProduct                ← theorem
  │  pdivMat_rowIndep               ← theorem (was the last surviving Mat-axiom)
  │
  ├── MLP.lean                 dense (proved both sides) + ReLU (pdiv_relu proved,
  │                            relu/mlp _has_vjp = canonical-witness defs)
  │                            + relu6; softmax CE is proved in Softmax.lean
  │
  ├── CNN.lean                 conv2d (def) + maxPool (def) + weight/bias grads (theorems)
  │                            conv2d_has_vjp3 (theorem); maxPool2_has_vjp3
  │                            (canonical-witness def — codegen substitutes argmax)
  │
  ├── BatchNorm.lean           BN (every axiom proved from foundation)
  │
  ├── Residual.lean            skip connections (biPath; zero new axioms)
  │
  ├── Depthwise.lean           depthwise conv (def) + weight/bias grads (theorems)
  │                            depthwise_has_vjp3 (theorem)
  │
  ├── SE.lean                  squeeze-and-excitation (elemwiseProduct; zero new axioms)
  │
  ├── LayerNorm.lean           LayerNorm (proved) + GELU (gelu Jacobian proved)
  │
  └── Attention.lean           softmax (proved) + SDPA (proved) + MHSA (proved)
                               + ViT body chains (proved) + patchEmbed (proved)
```

## Whole-network VJPs

Two forms, set by the architecture's activations:

- **Unconditional** (ViT, ConvNeXt, EfficientNet) — only smooth ops
  (GELU/Swish/sigmoid, softmax, LayerNorm, convolution; no ReLU, no
  max-pool), so `vit_full_has_vjp` / `convnext_has_vjp` /
  `efficientnet_has_vjp` are global `HasVJP`: correct at *every* input, with
  the `0 < ε` LayerNorm/BatchNorm positivity as the only side condition.

- **Conditional + concretely instantiated** (MLP, MNIST-CNN, ResNet,
  MobileNetV2/V4) — ReLU/ReLU6/max-pool have genuine kinks, so the generic
  whole-network VJP is pointwise (`*_has_vjp_at`, under per-site
  off-the-kink hypotheses). Each is instantiated at a point with every
  hypothesis discharged (`MlpConcrete`, `TrainedCnn`, `CnnConcrete`,
  and for ResNet-34/50 and MobileNetV2/V4 the full-width batched nets
  themselves in `ResNet34FullBSeal`, `ResNet50FullBSeal`,
  `MobileNetV2FullBSeal` and `MobileNetV4FullBSeal`), proving the bundle is
  jointly satisfiable — not vacuous. Every kinked net in the book now has
  one, on the forward its artifacts run.

Conditionality is intrinsic to the math, not a formalization gap: it enters
exactly at the non-smooth operators and is *recovered* by the
smooth-activation nets. Most concrete witnesses are deliberately tiny; the
ResNet-34/50 and MobileNetV2/V4 ones are not — each is the 224×224 batch-BN
net at paper depth, on the forward its artifacts run.
`CnnConcrete` has an injective stem, and `MobileNetV2FullBSeal` keeps every
ReLU6 input inside `(0,6)` with a BatchNorm window that holds at *every*
input, so all 35 of that net's kink clauses are weight-only and its forward
is non-constant (`Mnv2FullBSeal.sealX_nonconstant`).
`MobileNetV4FullBSeal` is the same story one net over — but its fused stage
is **swish**, which is the identity on no window at all, so its witness is
grid-constant rather than a ramp: the two examples then straddle `β` at that
BatchNorm and their swish outputs differ by a function of their gap alone
(`BatchSeal.swishGap`), which is what lets the carrier through.

## Axioms (0 project)

Pure-Mathlib closure on every theorem. `#print axioms vit_full_has_vjp`
shows only `propext`, `Classical.choice`, `Quot.sound` (Lean core).

The earlier 4-axiom floor was retired in Phase 7 (Apr 2026):

- `relu_has_vjp`, `mlp_has_vjp`, `maxPool2_has_vjp3` — converted from
  `axiom` to `noncomputable def` with the canonical pdiv-derived
  witness. `HasVJP.correct` holds by `rfl` since `pdiv` is a `def`
  over `fderiv` (post-foundation-flip). At non-smooth points the
  canonical backward is `fderiv`'s junk default of `0`; the codegen
  substitutes the standard subgradient/argmax convention — see
  "Codegen trust boundary" below.
- `pdiv_relu` — proved via local-diagonal-CLM transport
  (~80 LOC). At a smooth point (`∀ k, x k ≠ 0`), ReLU agrees with the
  diagonal indicator CLM `Π k, (if x k > 0 then proj k else 0)` on
  `Metric.ball x (min |x k|)` (every coordinate keeps its sign).
  `HasFDerivAt.congr_of_eventuallyEq` transports the CLM's self-fderiv
  to ReLU; direct evaluation at `basisVec i` reads off the entry.

**Tensor.lean** — calculus foundation: **0 axioms.** `pdiv` is a
`noncomputable def` over `fderiv`; every structural rule is a
theorem. `pdivMat_rowIndep` (the last surviving Mat-level axiom in
prior drafts) is now a theorem proved via the row-projection
`ContinuousLinearMap` and the chain rule, given a `Differentiable`
hypothesis on the per-row function.

**MLP.lean** — dense layers: **0 axioms.**

> `pdiv_dense`, `pdiv_dense_W`, `dense_weight_grad_correct`,
> `dense_bias_grad_correct`, and `pdiv_relu` are theorems.
> `relu_has_vjp` and `mlp_has_vjp` are `def`s over the canonical
> pdiv-derived witness. `softmaxCE_grad` is a theorem (in `Softmax.lean`, next to
> `pdiv_softmax`).

**CNN.lean** — convolution and pooling: **0 axioms.**

> `conv2d` and `maxPool2` are concrete `def`s. The weight-grad and
> bias-grad VJPs (`conv2d_weight_grad_has_vjp`,
> `conv2d_bias_grad_has_vjp`) are theorems proved from foundation
> via `unfold + fun_prop`. `conv2d_has_vjp3` is a theorem (Phase 1,
> Apr 2026) — proved via `pdiv_finset_sum` × 3 +
> `pdiv_const_mul_pi_pad_eval` per-summand + Σ_(c, kh, kw) collapse.
> `maxPool2_has_vjp3` is a `def` over the canonical pdiv-derived
> witness.

**BatchNorm.lean** — the hard one: **0 axioms.**

> Every BN Jacobian is now a theorem. `pdiv_bnAffine` and
> `pdiv_bnCentered` were proved in Stage 1; `pdiv_bnIstdBroadcast`
> and the smoothness lemma `bnIstdBroadcast_diff` were the last to
> fall in the diff-threading branch — the centering CLM,
> `HasFDerivAt.sqrt` (under `bnVar + ε > 0`), and
> `(hasDerivAt_inv).comp_hasFDerivAt` close the chain. Every BN
> proof now carries a `(hε : 0 < ε)` hypothesis.

**Residual.lean** — skip connections: **0 axioms.** Pure composition
over `biPath_has_vjp` + `identity_has_vjp` from `Tensor.lean`.

**Depthwise.lean** — depthwise conv: **0 axioms.**

> `depthwiseConv2d` is now a concrete `def`, weight and bias
> gradients are theorems via `unfold + fun_prop`. `depthwise_has_vjp3`
> is now a theorem (Phase 2, Apr 2026) — same recipe as conv2d with
> one fewer Σ level (no cross-channel mixing in depthwise).

**SE.lean** — squeeze-and-excitation: **0 axioms.** Pure composition
over `elemwiseProduct_has_vjp` + `dense_has_vjp` + `identity_has_vjp`.

**LayerNorm.lean** — layer norm and GELU: **0 axioms.**

> `geluScalar` and `geluScalarDeriv` are now concrete `def`s using
> the standard `tanh`-approximation formula. `pdiv_gelu` is a
> theorem proved via `fderiv_apply` + chain rule with
> `geluScalar ∘ ContinuousLinearMap.proj j`, then
> `fderiv_eq_smul_deriv` to convert scalar `fderiv` ↔ `deriv`. A
> new `Proofs.differentiable_tanh` `@[fun_prop]` lemma (derived from
> `Real.tanh_eq_sinh_div_cosh` + `Real.cosh_pos`) carries the
> smoothness through. `layerNorm_has_vjp` reuses the BN proof
> template on a different axis.

**Attention.lean** — softmax, attention, ViT: **0 axioms.**

> `pdiv_softmax`, `softmaxCE_grad`, the three `sdpa_back_*_correct`
> theorems, `rowSoftmax_flat_diff`, and **every** transformer-level
> chain (`transformerMlp_has_vjp_mat`,
> `transformerAttnSublayer_has_vjp_mat`,
> `transformerMlpSublayer_has_vjp_mat`,
> `transformerBlock_has_vjp_mat`,
> `transformerTower_has_vjp_mat`, `vit_body_has_vjp_mat`,
> `mhsa_has_vjp_mat`, `mhsa_layer_flat_diff`,
> `classifier_flat_has_vjp`, `vit_full_has_vjp`) are theorems.
> `patchEmbed_flat`, `patchEmbed_flat_diff`, and
> `patchEmbed_flat_has_vjp` were the last to fall: Phase 6a (Apr 2026)
> de-opaqued the forward and proved Diff via `differentiableAt_pad_eval`;
> Phase 6b (Apr 2026) proved the closed-form input-VJP via the same
> recipe used for `conv2d_has_vjp3`/`depthwise_has_vjp3`, with one new
> wrinkle: split `Σ n : Fin (N+1)` into the n=0 (CLS row, zero img-grad
> contribution) and `Σ p : Fin N` (n = p.succ, conv projection).

Plus three Lean core axioms (`propext`, `Classical.choice`,
`Quot.sound`) present in every nontrivial Lean program.

**Total: 0 project axioms across all nine content modules.**

## Codegen trust boundary

`HasVJP.correct` certifies the *canonical* backward
`backward x dy i = ∑ j, pdiv f x i j * dy j`. Where `f` is everywhere
differentiable, this is the true Jacobian-vector product (and the
`_diff` theorems on each layer carry that hypothesis through).

For the two non-smooth ops — ReLU at `x i = 0`, MaxPool at argmax
ties — the canonical backward is `fderiv`'s junk default of `0`,
because Mathlib's `fderiv` returns `0` at non-differentiable points
by convention. The codegen (`MlirCodegen.lean`) does **not** emit the
canonical backward at the kinks. Instead it emits the standard ML-
framework subgradient conventions:

- ReLU: `if x > 0 then dy else 0` (the `relu'(0) := 0` convention).
- MaxPool: **tile-compare-select** — the gradient is tiled to the
  input shape, compared against the pooled output with `stablehlo.compare EQ`,
  and `stablehlo.select`-ed through the resulting mask. The codegen
  avoids `stablehlo.select_and_scatter` because IREE does not support
  it (see `MlirCodegen.lean` near the maxPool backward case). At
  argmax ties, the EQ-mask routes the gradient to *every* tied input
  cell, matching the PyTorch/JAX semantics.

These match the canonical Lean witness at smooth points and differ
only at the kinks. The verification gap is intrinsic to backward
passes through non-smooth ops — every ML framework lives with the
same gap. The numerical FD checks in `check_jacobians.py` and the
end-to-end oracles in `tests/vjp_oracle/` cover the codegen-emitted
formula at the kinks.

**Two emit paths.** The kink discussion above is about `MlirCodegen.lean`
(~10,400 lines, zero theorems) — the path the full-recipe `*-train` trainers behind
the headline accuracy numbers use. The `*-verified` trainers instead consume the
StableHLO-subset render (the `SHlo` AST + its `den : SHlo n → Vec n` denotation),
and there the proof↔emitted link is a **theorem**, not just a numerical check:
for all 12 chapter nets the §1a whole-net ties (`LinearFold`'s
`poc_train_step_tail_certified` up through `r34_net_tiedB`,
`mnv2_net_tiedB`, `efficientnet_net_tiedG`, `cnx_net_tiedGB`, `vit_net_tiedGB` —
the last three at the gradient nodes every shipped EfficientNet / ConvNeXt / ViT artifact emits,
with `efficientnet_net_tied` / `cnx_net_tied_certified` / `vit_net_tied_certified` their
SGD-inline forms)
prove every emitted parameter-update node's `den` equals the certified
`fderiv`-derived loss-descent step (or its gradient), with the cotangent threaded
through the **real** forward and the proven per-block VJP backward (residual
fan-in included — not a free `∀`-cotangent). All 3-axiom-clean in
`tests/AuditAxioms.lean`. The residuals on *that* path are narrower: (a) `den` is
the `ℝ` denotation, so the `den`→`Float32` rounding gap, the per-op `pretty`
lexing, `iree-compile`, the runtime, and the FFI stay trusted; (b) the same
ReLU/MaxPool/ReLU6 kink convention above; and (c) the CI drift guard (`proofs.yml`,
"Verified-render drift guard") re-elaborates the `Proofs/Codegen` renderers and
byte-checks their committed `verified_mlir/` files against them;
`scripts/check_render_coverage.py` holds the unguarded remainder (14 of 240 files)
at its baseline.

## The three rules

All of backpropagation:

```
vjp_comp              f ∘ g  →  back_f(x, back_g(f(x), dy))
biPath_has_vjp        f + g  →  back_f(x, dy) + back_g(x, dy)
elemwiseProduct_has_vjp  f * g  →  back_f(x, g·dy) + back_g(x, f·dy)
```

## The five Jacobian tricks

Every layer's backward pass is one of:

1. **Diagonal** — activations (ReLU, GELU): one multiply
2. **Sparse Toeplitz** — conv: reversed kernel convolution
3. **Binary selection** — max pool: route to argmax
4. **Rank-1 correction** — batch/layer norm, softmax: closed-form 3-term formula
5. **Outer product** — dense/matmul: input ⊗ grad

## Numerical gradient checks

`check_jacobians.py` runs 25 finite-difference checks. They cover the
codegen-emitted backward formulas (where the trust gap actually
lives — see "Codegen trust boundary" above), particularly at the
ReLU and MaxPool kinks where the codegen substitutes a subgradient
convention for the canonical Lean witness. Typical max-error is
~1e-11 in float64.

## Independent kernel re-check (comparator)

`tests/comparator/` runs
[leanprover/comparator](https://github.com/leanprover/comparator) on
73 theorems spanning the foundation rules, every chapter's headline
Jacobian, the public `*_has_vjp_correct` wrappers, and the five
whole-network VJPs (ViT, ResNet, MobileNetV2, ConvNeXt, EfficientNet).
comparator
re-runs Lean's kernel typechecker independently
of the elaborator and verifies the transitive axiom closure of each
proof. The configured allowlist is exactly
`{propext, Quot.sound, Classical.choice}` (Lean core); any project
axiom in the closure would fail the run. See
`tests/comparator/README.md` for the prereq + run instructions.

## Verify

```bash
lake build Proofs        # the engine slice (default target) — minutes
lake build Certs         # the full certificate corpus (subsumes Proofs)
lake env lean tests/AuditAxioms.lean   # the three-axiom closure audit
```

`Certs`'s roots subsume the `Proofs` slice, so `lake build Certs`
type-checks the entire suite; CI runs it (plus the audit) in
`.github/workflows/certs.yml` on proof-path pushes and nightly.

If it builds, it's correct. That's the point.
