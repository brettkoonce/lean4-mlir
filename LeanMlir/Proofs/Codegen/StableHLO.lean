import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Architectures.StridedConv
import LeanMlir.Proofs.Foundation.Batched
import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.LayerNorm
import LeanMlir.Proofs.Architectures.SE
import LeanMlir.Proofs.Architectures.Attention
-- The ℝ AdamW spec (`adamMNext`/`adamVNext`/`adamWParam`), so the optimizer ops can denote it.
-- AdamStep only imports Foundation.Tensor + Mathlib, so this adds no cycle.
import LeanMlir.Proofs.Training.Optim.AdamStep
-- The ℝ global-norm clip spec (`gradSumSq`/`clipFactor`/`clipScale`), so the four clip ops can
-- denote it. GradClip imports only AdamStep, so this adds no cycle either.
import LeanMlir.Proofs.Training.Optim.GradClip
-- LAMB (`lambDir`/`lambTrust`/`lambScale`), RSB-A3's optimizer. Imports GradClip for `scalarOf`
-- and `gradSumSq` — the per-leaf squared norm is SHARED with the clip rather than re-derived, so
-- the two features cannot drift on what a norm is.
import LeanMlir.Proofs.Training.Optim.Lamb
import LeanMlir.Proofs.Training.Optim.SgdMomentumStep
-- RmsPropStep imports only the two above, so this adds no cycle either.
import LeanMlir.Proofs.Training.Optim.RmsPropStep
-- DropPath imports only Architectures.LayerNorm (for `layerScale`, which this file already has in
-- scope), so it adds no cycle either. `planning/archive/stochastic_depth.md`.
import LeanMlir.Proofs.Training.DropPath
-- He et al.'s 3×3/s2 stem pool (`maxPool3s2Flat` + its VJP witness), so the stem-pool ops can
-- denote it. MaxPool3s2 imports only Architectures.CNN, which this file already has in scope
-- transitively, so it adds no cycle. `planning/archive/rsb_a3_r50_verified.md` §4b.
import LeanMlir.Proofs.Architectures.MaxPool3s2

/-! # StableHLO — the emitted-graph AST and its ℝ semantics

Every verified artifact in `verified_mlir/` is `pretty` of a term of one typed AST, `SHlo`. This
file holds that AST and its semantic reading; the syntactic one, `pretty`, is in
[`StableHLOPretty`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Codegen/StableHLOPretty.lean).

* **Semantic** — `den : SHlo n → Vec n`, the ℝ denotation in StableHLO-spec terms (explicit
  contraction / reduce / divide). The `*_faithful` / `*_den` theorems say `den (graph) = <proven
  math>`; every train-step tie in `Nets/` is stated about `den`.
* **Syntactic** (`StableHLOPretty`) — `pretty` renders the same term to StableHLO text. SSA names
  are annotations `den` ignores, so the rendered program and the denoted one are one object.

Layout, in file order:

| part | where |
|---|---|
| `BatchableOp` — the per-example ops `SHlo.batchOp` lifts by `batchMap` | top |
| `inductive SHlo` (≈215 constructors; suffixes: `F` forward/optimizer op, `B`/`Batched` batched index, `Grad`/`GradB` raw gradient node, `Sgd`/`SgdB` fused `θ − lr·g`, `Bf16`/`F8` reduced precision) | § StableHLO-subset AST |
| `den`, the `denStep`/`denStepApp` dsimprocs, the per-op `*_faithful` lemmas | after the AST |
| chapter graphs and their faithfulness (linear, MLP, CNN, CIFAR) and the optimizer/clip ops | § Chapter 1–3, § Param gradients, § Global-norm clipping |

**Trusted residue.** `den` is over ℝ, so the ℝ→Float32 gap stays trusted. Everything here closes
under `[propext, Classical.choice, Quot.sound]`
([`tests/AuditAxioms.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/AuditAxioms.lean)).
-/

open Finset BigOperators

namespace Proofs
namespace StableHLO

-- ⛔ Never name `den` in a `simp` set; name `denStep`/`denStepApp` (defined after `den`). `den`
-- there makes Lean build `den.eq_def`, a 233 s proof for the 215-arm match (measured 2026-09-23 with
-- `trace.profiler`), and every proof in this file waited on it. That cost, not the proofs, is what
-- the 1M → 4M file-wide heartbeat floor this header used to carry was paying for. With the
-- dsimprocs every proof here checks at the default budget and the module takes ~40 s (measured
-- 2026-09-24, after the printer moved to `StableHLOPretty`) instead of ~345 s.
--
-- The arm-count rule from §0.8 still holds for `rfl` through `den`: *parametric arms are
-- affordable; fixed-index arms (no `{n : Nat}` binder) are not.*

/-- **A batch-separable EfficientNet op**, shape-indexed by per-example in/out
    length. The descriptor carried by `SHlo.batchOp`; its `denOp` is the proven
    per-example forward, lifted by `batchMap`.

    **On the pointwise ops.** An earlier note here said swish/sigmoid/relu/addV
    "need no descriptor — the existing tokens already denote them block-diagonally
    at the batched index `N·(c·h·w)`". The *denotation* half of that is true and the
    *emit* half is false, and the difference is what pinned the batched renderers at
    `N := 1`. `SHlo.swishF`'s token carries only the SHlo index `n` and emits
    `tensor<B×n>`, i.e. it reads the index as a PER-EXAMPLE width; a descriptor-less
    pointwise node at the batched index `N·s` therefore emits `tensor<B×(N·s)>`,
    which does not even typecheck against its own operand. Giving the pointwise ops
    descriptors separates the two numbers — `N` (batch, denotation) from `n`
    (per-example width, emit) — which is what lets a whole graph sit at `N := B`
    where the batch-coupled `den`s (`bnBatchF`, the `*SgdB` family) are honest.
    The per-example renderers keep the descriptor-less tokens unchanged. -/
inductive BatchableOp : Nat → Nat → Type where
  | conv {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*h*w) (oc*h*w)
  | convStrided {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*(2*h)*(2*w)) (oc*h*w)
  -- ⭐ The **bf16** peers of `conv`/`convStrided` — the batched forward convs every ResNet
  -- render actually uses (`.batchOp (.conv …)`), as distinct from the per-example
  -- `flatConvFBf16`. Same emit discipline: bf16 operands, **bf16-typed result**, convert back,
  -- then the bias in f32. See `flatConvFBf16` for why the result type is load-bearing.
  | convBf16 {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*h*w) (oc*h*w)
  -- ⭐ **fp8 (E4M3) peer of `convBf16`** — identical shape, identical denotation, one different
  -- type string. Measured 2026-08-25 (`planning/archive/fp8_in_graph.md` §1) to lower at cifar8's own
  -- conv shapes: every layer reaches `__cudnn$convForwardGraph` with f8 values surviving into
  -- the optimized HLO. ⚠ f8 operands, **f8-TYPED result**, convert back — an f32 result is
  -- 1.17× where the f8 result is 3.43× (§2.3), the same result-type rule as bf16 at a third
  -- precision. ⚠⚠ UNSCALED: E4M3's max is 448, so this is only sound where the operands are
  -- known to fit. See §4 — scales are the next rung, and `e4m3_render_faithful` already covers
  -- the scaled form for any `q`/`sx`/`sW`.
  | convF8 {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*h*w) (oc*h*w)
  | convStridedBf16 {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*(2*h)*(2*w)) (oc*h*w)
  -- ⭐ The **XLA `'SAME'`** stride-2 conv — same shape as `convStrided`, different padding, and
  -- the two are NOT interchangeable. `convStrided` pads symmetrically `((k-1)/2` each side), which
  -- is He et al./torchvision and is what R34/R50/ConvNeXt's references do. This one pads
  -- `((k-2)/2, k/2)` — `(0,1)` at k=3 — which is what XLA `'SAME'` does at an EVEN input and what
  -- the TF-origin ports (MobileNetV2/V4, EfficientNet) mean by `padding='SAME'`.
  --
  -- ⚠⚠ **Both produce the same output size, so nothing structural can tell them apart.** Shapes,
  -- arity, op counts and every `#guard` in the repo pass either way; only a forward tie against
  -- the reference on shared weights separates them (`planning/archive/mnv4_verified.md` §3b/§3d measured
  -- 6.16e-2 on mnv4's stem and 2.9e-1 across mnv2's five sites). Pick by which reference the net
  -- has: TF-origin → this one; torchvision-origin → `convStrided`.
  --
  -- `den` is `flatConvStride2Xla` = `decimateOddFlat ∘ flatConv` — the SAME stride-1 conv, read at
  -- the odd phase. So this adds no proof obligation: the forward, input-VJP, weight-VJP and
  -- bias-VJP are all `vjpComp`s of results already proven (`Architectures/StridedConv.lean`).
  | convStridedXla {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*(2*h)*(2*w)) (oc*h*w)
  -- ⭐ bf16 peer of `convStridedXla` — MobileNetV2's stem. Same asymmetric `((k-2)/2, k/2)` pad;
  -- the bf16 twin must NOT be "tidied" to the symmetric one, which is a different net.
  | convStridedXlaBf16 {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*(2*h)*(2*w)) (oc*h*w)
  | depthwise {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*h*w) (c*h*w)
  -- ⭐⭐ The **bf16 depthwise** — the first GROUPED bf16 conv in the kit. Same emit discipline as
  -- `convBf16`: bf16 operands, **bf16-TYPED result**, convert back, `feature_group_count = c`
  -- untouched. ⚠ The f32-result shape folds here exactly as it does for an ordinary conv —
  -- measured on a real MNv2 layer (c=144, 56², 3×3) before these ops were written, so grouping
  -- buys no exemption from §9.2.
  | depthwiseBf16 {c h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*h*w) (c*h*w)
  -- ⭐ The XLA-`SAME` depthwise peer of `convStridedXla`, and the token MobileNetV2 and
  -- EfficientNet need: their `depthwise_conv` defaults to `padding='SAME'`
  -- (`depthwise_conv` in `jax/Jax/Codegen.lean`), so every strided depthwise in those references pads
  -- `((k-2)/2, k/2)`, not symmetrically. Same invisibility caveat as `convStridedXla` — identical
  -- shapes, identical counts, identical group widths; only a forward tie separates them.
  -- `den` is `depthwiseStride2FlatXla` = `decimateOddFlat ∘ depthwiseFlat`.
  | depthwiseStridedXla {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  -- ⭐ bf16 peer of the XLA-`SAME` strided depthwise. Keeps the asymmetric pad verbatim.
  | depthwiseStridedXlaBf16 {c h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  | depthwiseStrided {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  -- ⭐ bf16 peer — MobileNetV4's strided depthwise. ⚠ SYMMETRIC pad `[[p,p],[p,p]]`, unlike
  -- `depthwiseStridedXlaBf16`'s `[[p-1,p],…]`: MNv4's torchvision-origin blocks pad symmetrically
  -- where MNv2/EfficientNet's TF-origin ones do not. Identical shapes and counts either way, so
  -- only a forward tie separates them — do not "unify" the two.
  | depthwiseStridedBf16 {c h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  | dense {a c : Nat} (wName bName : String)
      (W : Mat a c) (bias : Vec c)                         : BatchableOp a c
  | gap {c h w : Nat}                                      : BatchableOp (c*h*w) c
  | seBlock {c h w r : Nat} (w1Name b1Name w2Name b2Name : String)
      (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) : BatchableOp (c*h*w) (c*h*w)
  -- INFERENCE per-channel BN at the batched index — the frozen-stats peer of the OWN-CTOR
  -- `bnBatchF`, and the batched peer of `bnPerChannelEvalF`. A **descriptor is legal here and is
  -- the point**: γ/β/μ/var are the driver's EMA'd running statistics, arriving as graph inputs
  -- shared by the whole batch, so they are batch-INVARIANT data of exactly the kind the note above
  -- permits (a shared weight, not a saved per-example activation). `den` is therefore
  -- `batchMap N (bnPerChannelEvalTensor3 …)` — every example normalised by the SAME frozen stats,
  -- independently — which is the formal content of "eval is class-batch-independent".
  --
  -- That it can be a descriptor at all is the whole difference from training BN: `bnBatchF` needs
  -- its own constructor because it REDUCES over the batch (§2b's second kind), and `batchMap N`
  -- cannot express a reduction across examples. Here there is no reduction.
  | bnEval {oc h w : Nat} (gName bName muName varName epsStr : String) (ε : ℝ)
      (γ β μ var : Vec oc)                                 : BatchableOp (oc*h*w) (oc*h*w)
  -- Pointwise activation, as a descriptor (see the note above). `n` is the
  -- per-example width the emit uses; `den` is `batchMap N (swish n)`, which for a
  -- pointwise map is the pointwise map itself — so `N` is denotationally free here
  -- and the descriptor exists purely to keep the emit width off the SHlo index.
  | swish {n : Nat}                                        : BatchableOp n n
  -- ReLU forward, the ResNet-34 peer of `swish`. Same story: pointwise, carries no
  -- data, so `batchMap N` of it is itself and `N` is denotationally free.
  | relu {n : Nat}                                         : BatchableOp n n
  -- ReLU6 forward, MobileNetV2's activation (§2f). Same story a third time: a pointwise clamp
  -- to [0,6] carrying no data, so `batchMap N` of it is itself and `N` is denotationally free.
  -- Its BACKWARD is `selectMidB`, deliberately NOT a descriptor — see the note below.
  | relu6 {n : Nat}                                        : BatchableOp n n
  -- 2×2 max-pool FORWARD (ResNet-34's stem). A descriptor: the forward carries no saved
  -- value, so `batchMap N maxPoolFlat` is exactly per-example pooling across the batch.
  -- Its BACKWARD is `maxPoolBackB`, not a descriptor — it routes `dy` to the saved input's
  -- window argmax, which is per-example data.
  | maxPool {c h w : Nat}                                  : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  -- ⭐ **3×3/s2 max-pool FORWARD** — He et al.'s ResNet stem pool (`planning/archive/rsb_a3_r50_verified.md`
  -- §4b). Same TYPE as `.maxPool` above (112→56 either way, since symmetric `(3−1)/2 = 1` padding
  -- makes the output width `h`), and a **different function**: the windows OVERLAP. That the two
  -- share a type is exactly why the deviation survived undocumented on every ResNet here — nothing
  -- ever failed to compile. A descriptor for `.maxPool`'s reason: the forward carries no saved
  -- value, so `batchMap N maxPool3s2Flat` is per-example pooling across the batch.
  -- ⚠ Its BACKWARD is `maxPool3s2BackB`, and it ACCUMULATES: an input can be the argmax of up to
  -- four windows, where `maxPool2`'s backward is a single lookup. See `MaxPool3s2.lean`.
  | maxPool3s2 {c h w : Nat}                               : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  -- NOTE: the pointwise activation VJPs (`swishBack`/`sigmoidBack`/`selectPos`) are
  -- deliberately NOT here. `BatchableOp` lifts a FIXED function across examples, and
  -- their backward depends on the saved pre-activation, which varies per example —
  -- `batchMap N` of it would denote "every example shares one saved activation", which
  -- is not what the emit computes. `swishBack`/`sigmoidBack` are `dy i * deriv (x i)`;
  -- `selectPos` is `if x i > 0 then dy i else 0`, the same shape. They get their own
  -- `SHlo` constructors (`swishBackB`/`sigmoidBackB`/`selectPosB`) carrying the
  -- WHOLE-BATCH `x`.
  -- Row ops: `m`/`rows` is the per-example ROW count (ViT tokens; 1 logit row for a
  -- classifier head), NOT the batch — it was always emitted as a real inner
  -- dimension. The descriptor form exists so the batch can move to `N`.
  | softmaxRow {m n : Nat}                                 : BatchableOp (m*n) (m*n)
  | denseRowBack {rows a c : Nat} (wName : String) (W : Mat a c) : BatchableOp (rows*c) (rows*a)
  -- ⭐ bf16 peer of `denseRowBack` — ViT's input-VJP through Q/K/V/O/fc1/fc2.
  -- ⚠⚠ **bf16 operands, bf16-TYPED RESULT, convert back — the CONV shape, and this is a CHANGE.**
  -- `planning/archive/bf16_renderer.md` §9.2 measured that `dot_general` reaches the tensor cores with
  -- EITHER result type and concluded the result type was "inert" for dot. That is true of
  -- CORRECTNESS and **false of SPEED**, which nobody had measured: on ViT's own MLP chain the
  -- f32-result shape is 1.18× over f32 and the bf16-result shape is **1.60×** (§20.1). The f32
  -- result makes the gemm write twice the bytes and takes a worse epilogue.
  -- ▶ Consequence for `den`: a bf16-typed result means the hardware DOES round the output, so
  -- there is an outer `rnd` here exactly as in `convBf16`. Omitting it would claim precision the
  -- hardware does not deliver — the unsound direction.
  | denseRowBackBf16 {rows a c : Nat} (rnd : ℝ → ℝ) (wName : String) (W : Mat a c)
      : BatchableOp (rows*c) (rows*a)
  -- ── ViT / ConvNeXt: the row-indexed and pointwise forward forms (§0.2 ▶2, the batched-index
  --    move). All five carry only batch-INVARIANT data — a scalar ε/γ/β, a shared per-feature
  --    vector, or nothing — which is exactly the descriptor precondition (`den` is
  --    `batchMap N (denOp op)`, ONE fixed function across the batch). The saved-activation
  --    backwards of the same layers (`lnRowBack`, `geluBack`, `softmaxRowBack`) can NOT be
  --    descriptors and take the `batchMapAux` shape as their own constructors.
  --
  --    ⚠ `m` is rows PER EXAMPLE — the token axis on ViT, the spatial axis on ConvNeXt's
  --    channel-LN — never the batch. That separation is the whole point of a descriptor: `N` is
  --    the denotation's batch, `m*n` the emit width. Reading `m` as the batch is the mistake the
  --    per-example renderers make structurally.
  | gelu {n : Nat}                                          : BatchableOp n n
  | transpose {m n : Nat}                                   : BatchableOp (m*n) (n*m)
  -- ── increment 3: ConvNeXt's stem conv, its LayerScale, and the loss-path softmax pair.
  --    ⚠ `expe` and `softmaxDiv` are descriptors for OPPOSITE halves of the §2b defect, and the
  --    contrast is worth keeping. `expe`'s `den` is already honest at the batched index
  --    (`Real.exp` pointwise IS its own batch-lift) and only its EMIT is wrong there — it reads
  --    the width off the SHlo index and would emit `tensor<B×(N·n)>`. `softmaxDiv` is the reverse:
  --    its emit already reduces over `dimensions = [1]`, i.e. per example, while its `den`
  --    (`v j / ∑ k, v k`) would divide by the sum over the WHOLE BATCH at index `N·n`. One of them
  --    is a typing bug and the other a silent wrong answer; the descriptor fixes both the same way.
  | convStride4 {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)
      : BatchableOp (ic*(2*(2*h))*(2*(2*w))) (oc*h*w)
  -- ⭐ **The bf16 stem** — ConvNeXt's 4×4/s4 patchify, and the only STRIDE-4 conv in the kit. Same
  -- emit discipline as every other conv here: bf16 operands, **bf16-TYPED result**, convert back,
  -- bias in f32.
  --
  -- ⚠ `convStride4`'s **pad-one-less** rule is preserved verbatim: the denotation reads the stride-1
  -- SAME conv at the offset positions `4i+1`, so the emitted pad is `(k-1)/2 − 1`, which at the 4×4
  -- stem is `[[0,0]]` — the paper's left-aligned window, and NOT the symmetric `(k-1)/2` every other
  -- forward conv emits. Copying `convBf16`'s pad here renders a different net at identical shapes.
  --
  -- ⚠ Measured on this exact shape (B=32, 3→96, 224²→56², 4×4/s4) BEFORE this op was written: the
  -- §9.2 fold fires at stride 4 exactly as at stride 1, stride 2 and grouped — a bf16-operand
  -- convolution with an f32-TYPED result compiles to pure f32. Stride buys no exemption either.
  | convStride4Bf16 {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)
      : BatchableOp (ic*(2*(2*h))*(2*(2*w))) (oc*h*w)
  | layerScaleCh {c h w : Nat} (γName : String) (γ : Vec c)  : BatchableOp (c*h*w) (c*h*w)
  | dotOut {m n : Nat} (wName : String) (W : Mat m n)        : BatchableOp n m
  | expe {n : Nat}                                          : BatchableOp n n
  | softmaxDiv {n : Nat}                                    : BatchableOp n n
  | lnRow {m n : Nat} (gName bName epsStr : String) (ε γ β : ℝ) : BatchableOp (m*n) (m*n)
  | rowScale {m n : Nat} (gName : String) (γ : Vec n)       : BatchableOp (m*n) (m*n)
  | rowBias {m n : Nat} (bName : String) (β : Vec n)        : BatchableOp (m*n) (m*n)
  -- ── ViT increment 1 (handoff §0.2 ▶3): the six forms whose data is batch-INVARIANT.
  --
  --    ⚠⚠ `N` HERE IS ViT's TOKEN COUNT, NOT THE BATCH — and on this net the two are far easier
  --    to conflate than on ConvNeXt, because the per-example renderer already spells the token
  --    axis `N`. A `BatchableOp a b` never sees the batch at all: `SHlo.batchOp`'s own `{N}` is
  --    the batch and these `N`s ride INSIDE `a` and `b`. Reading either as the other is the exact
  --    defect this whole thread removes, and it type-checks in both directions.
  --
  --    All six qualify as descriptors by §4's rule — the data each carries is a weight, a bias, a
  --    head INDEX or nothing, i.e. the same for every example. ViT's saved-activation backwards
  --    (`softmaxRowBack`) and its batch-contracting parameter gradients cannot be descriptors and
  --    take their own constructors, exactly as ConvNeXt's did.
  | denseRow {N a c : Nat} (wName bName : String) (W : Mat a c) (b : Vec c)
      : BatchableOp (N*a) (N*c)
  -- ⭐ bf16 peer of `denseRow` — the six per-block matmuls (Q/K/V/O/fc1/fc2) that are 90 % of a
  -- ViT step (§17.3). bf16 operands, **bf16-typed result**, convert back, then the bias in f32.
  -- ⚠ The outer `rnd` in `den` is the bf16 STORE and the bias is added AFTER it, at the accumulate
  -- precision, exactly as emitted — `convBf16`'s shape. See `denseRowBackBf16` for why the result
  -- type changed from f32 and what it was worth.
  | denseRowBf16 {N a c : Nat} (rnd : ℝ → ℝ) (wName bName : String) (W : Mat a c) (b : Vec c)
      : BatchableOp (N*a) (N*c)
  | patchEmbed {ic H W P N D : Nat} (wName bName clsName posName : String)
      (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N+1) D)
      : BatchableOp (ic*H*W) ((N+1)*D)
  -- ⭐⭐ bf16 peer of `patchEmbed` — the 16×16/s16 patchify stem, and the ONE ViT op that is a
  -- `convolution` rather than a `dot_general`. ⚠⚠ So it takes the CONV shape: bf16 operands,
  -- **bf16-TYPED result**, convert back. Measured standalone at ViT's own stem shape before this
  -- constructor was written (§17.2): the f32-result spelling FOLDS to pure f32 exactly as it does
  -- for stride 1/2/4 and for grouped convs. Stride 16 buys no exemption from §9.2 either.
  -- ▶ Hence the outer `rnd` in `den` (the bf16 store), which `denseRowBf16` does NOT have. The
  -- bias, the CLS token and the position embedding are all added AFTER, in f32, exactly as
  -- emitted — they are f32 parameters that never cross a tensor core.
  | patchEmbedBf16 {ic H W P N D : Nat} (rnd : ℝ → ℝ) (wName bName clsName posName : String)
      (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N+1) D)
      : BatchableOp (ic*H*W) ((N+1)*D)
  -- ⚠ `clsSlice`/`clsPad` and `headSlice`/`headPad` are VJP pairs, and each pair is
  -- shape-asymmetric — the slice contracts, the pad scatters back. That asymmetry is what makes
  -- them safe as descriptors despite looking like data movement: neither reads a value.
  | clsSlice {N D : Nat}                                    : BatchableOp ((N+1)*D) D
  | clsPad {N D : Nat}                                      : BatchableOp D ((N+1)*D)
  -- ⚠ `h : Fin heads` is an INDEX, not per-example data — head `h` is the same head for every
  -- example. A per-example head choice would be a different architecture.
  | headSlice {N heads d : Nat} (h : Fin heads)             : BatchableOp (N*(heads*d)) (N*d)
  | headPad {N heads d : Nat} (h : Fin heads)               : BatchableOp (N*d) (N*(heads*d))

-- ════════════════════════════════════════════════════════════════
-- § StableHLO-subset AST — denotable AND renderable
-- ════════════════════════════════════════════════════════════════

/-- A StableHLO-subset expression, shape-indexed by result length. Leaves carry
    both a value (for `den`) and an SSA name (for `pretty`); the name is
    denotation-irrelevant. One constructor per emitted op. -/
inductive SHlo : Nat → Type where
  | operand    {n : Nat} (name : String) (v : Vec n)            : SHlo n
  | dotIn      {m n : Nat} (wName : String) (W : Mat m n)       : SHlo m → SHlo n
  -- Mixed-precision matmul (planning/archive/bf16_renderer.md): BOTH operands rounded by `rnd`,
  -- accumulate exact. This is `dotIn` with the leaf casts pulled INSIDE the op, and it
  -- exists because the casts cannot live outside it: a separate round node emits a
  -- convert PAIR, which XLA deletes (`xla_allow_excess_precision`, measured — see the
  -- `convertF` comment). Bundling is also what lets the emit be a single bf16-operand /
  -- f32-result `dot_general`, i.e. the only form that reaches tensor cores.
  --
  -- Why it is a new constructor rather than a dtype index on `SHlo`: `SHlo n` is indexed
  -- by WIDTH only and has no element type, so "the value is bf16 here" is unsayable. The
  -- op keeps its result f32 (the accumulate), so the index stays honest — the same
  -- bundling `flatConvF` already uses for conv+bias.
  -- ⚠⚠ **ITS f32-TYPED RESULT IS A PoC ARTEFACT, NOT A RECOMMENDATION.** `dotInBf16` is the
  -- depth-1 dense proof-of-concept and is rendered by NO net. §9.2 measured that `dot_general`
  -- reaches the tensor cores with either result type and read that as "the result type is inert
  -- for dot"; that is true of CORRECTNESS and false of SPEED (§20.1 — an f32 result makes the gemm
  -- write twice the bytes, worth ~1.2× on a real chain). ▶ ViT's dot ops take the **bf16-typed
  -- result** shape for that reason. Do not copy this constructor's shape into a new op.
  | dotInBf16  {m n : Nat} (rnd : ℝ → ℝ) (wName : String) (W : Mat m n) : SHlo m → SHlo n
  | dotOut     {m n : Nat} (wName : String) (W : Mat m n)       : SHlo n → SHlo m
  | addBcast   {n : Nat} (bName : String) (b : Vec n)           : SHlo n → SHlo n
  | expe       {n : Nat}                                        : SHlo n → SHlo n
  | softmaxDiv {n : Nat}                                        : SHlo n → SHlo n
  | sub        {n : Nat}                                        : SHlo n → SHlo n → SHlo n
  -- Chapter-1 SGD tail (the linear train step, folded into the AST): the two
  -- fused parameter-update ops that take the loss cotangent and emit the
  -- weight/bias SGD step. `weightSgd`: `W − lr·(x⊗dy)` (`dot_general` batch-
  -- contract → const → multiply → subtract), `den` = the certified `sgdW` step
  -- at B=1. `biasSgd`: `b − lr·(Σ_batch dy)` (`reduce` → const → mul → sub).
  -- LinearFold proves both `den`s = the certified loss-descent step.
  | weightSgd  {m n : Nat} (xName wName lrStr : String) (x : Vec m) (W : Mat m n) (lr : ℝ) : SHlo n → SHlo (m*n)
  | biasSgd    {n : Nat} (bName lrStr : String) (b : Vec n) (lr : ℝ)                        : SHlo n → SHlo n
  -- Chapter 2 (MLP): ReLU forward (`maximum(·,0)`) and its backward mask
  -- (`select(x>0,·,0)`); `xName`/`x` is the saved pre-activation.
  | reluF      {n : Nat}                                        : SHlo n → SHlo n
  | selectPos  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 6 (MobileNetV2): ReLU6 forward (`clamp(·,0,6) = min(max(·,0),6)`) and
  -- its backward mask (`select(0<x<6,·,0)` — the TWO-SIDED kink, smooth iff
  -- `x≠0 ∧ x≠6`). `selectMid`'s `xName`/`x` is the saved pre-activation.
  | relu6F     {n : Nat}                                        : SHlo n → SHlo n
  | selectMid  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Mixed precision (planning/archive/bf16_renderer.md): the in-graph ROUND node. `den` is
  -- literally `rnd ∘ den e`, so it is `den`-faithful for ANY rounding — bf16
  -- round-to-nearest being the instance we emit. This is the op
  -- `Proofs/Float/Bf16Fold.lean` names as the depth > 1 ingredient
  -- (`den (convertF rnd e) = rnd ∘ den e`), and it is ALSO what depth 1 needs on the
  -- emitter side: the PoC folds the leaf cast into the operand *value*, which is
  -- right for the proof but would leave the emitted graph pure `f32` and therefore
  -- exactly as fast as fp32. One op serves both.
  --
  -- It emits a convert ROUND TRIP (`f32 → bf16 → f32`), which is the honest reading
  -- of a `ℝ → ℝ` rounding: the value stays an f32 tensor and only its precision is
  -- degraded. Feeding a bf16-typed `dot_general` directly is a separate, later change
  -- (rung 2+), because that one changes the TYPE of the value and so cannot be a
  -- `SHlo n → SHlo n` node.
  | convertF   {n : Nat} (rnd : ℝ → ℝ)                          : SHlo n → SHlo n
  -- Chapter 3 (CNN): flattened conv forward (`stablehlo.convolution`) and
  -- 2×2 max-pool forward (`reduce_window`). Vec-indexed via the proofs'
  -- flattened forms `flatConv`/`maxPoolFlat`.
  | flatConvF  {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)                    : SHlo (ic*h*w) → SHlo (oc*h*w)
  -- ⭐ The **bf16** peer of `flatConvF`: bf16 conv operands, f32 bias add.
  --
  -- ⚠⚠ Its emit is NOT `dotInBf16`'s shape and must not be "made consistent" with it.
  -- Measured on ares 2026-08-24 (jax 0.11.0 and 0.10.2 alike, and NOT rescued by
  -- `xla_allow_excess_precision=false`): a `convolution` with bf16 operands and an
  -- **f32-typed result** has its converts DELETED — cuDNN receives f32 parameters and the
  -- optimized HLO contains no convert at all. That is `convertF`'s round-trip fold, one op
  -- over. `dot_general` is genuinely unaffected, which is why `dotInBf16` may keep an f32
  -- result and this may not. The shape that survives is a **bf16-TYPED result** followed by
  -- a separate convert back — what `jax/Jax/Codegen.lean`'s `conv2d` already emits, and why
  -- the JAX lowerer gets bf16 on ImageNet and the verified path does not.
  --
  -- ▶ So the value is rounded TWICE and `den` says so: once per operand (bf16 in) and once
  -- on the accumulated sum (bf16 store; the MAC itself accumulates in f32). `dotInBf16`'s
  -- `den` carries no outer rounding because its result really does stay f32 — copying it
  -- here would claim MORE precision than the hardware delivers, which is the unsound
  -- direction for an accuracy bound.
  -- ▶ The bias is added after the convert back, in f32, exactly as the emit orders it.
  | flatConvFBf16 {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)                    : SHlo (ic*h*w) → SHlo (oc*h*w)
  | maxPoolF   {c h w : Nat}                                    : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  -- ⭐ The **3×3/s2** peer of `maxPoolF` — He et al.'s ResNet stem pool, at the PER-EXAMPLE index
  -- (`ResNet34Render`'s world; the batched peer is the `BatchableOp.maxPool3s2` descriptor). Same
  -- type, overlapping windows, different function — see the note on that descriptor.
  | maxPool3s2F {c h w : Nat}                                   : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  -- Conv input-VJP backward (reversed-kernel `stablehlo.convolution`); `v` is
  -- the saved conv input. Conv is linear, so this is a global VJP.
  | convBack   {ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*h*w)) : SHlo (oc*h*w) → SHlo (ic*h*w)
  -- Max-pool backward (`select_and_scatter`, route dy to the window argmax);
  -- `x` is the saved pre-pool input. Conditional (no-ties) like the ReLU kink.
  | maxPoolBack {c h w : Nat} (xName : String) (x : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- ⭐ The **3×3/s2** peer of `maxPoolBack`. Same `select_and_scatter`, wider window, symmetric
  -- padding — ⚠ and **nothing else changes**, because `select_and_scatter` already scatters with an
  -- **add** reduction, which is exactly the accumulation overlapping windows need. The emitter was
  -- always general enough; only the window attributes move.
  | maxPool3s2Back {c h w : Nat} (xName : String) (x : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- Chapter 3 (CNN) param-SGD tail (the conv train step, folded into the AST):
  -- the fused conv kernel/bias update ops — the conv analogue of `weightSgd`/`biasSgd`.
  -- `convWeightSgd`: `W − lr·(conv2dWeightGrad(b,x)·dy)` via the transpose-trick conv
  -- (transpose→transpose→convolution→transpose, then const→multiply→subtract), `den`
  -- = `cnn_render_convW_certified`. `convBiasSgd`: `b − lr·(conv2dBiasGrad(W,x)·dy)`
  -- (reduce over batch+spatial [0,2,3], then SGD). `xName`/`wName`/`bName` are the saved
  -- activation/kernel/bias SSA names; `W,x,b,lr` carry the den. CnnFold proves
  -- both `den`s = the certified loss-descent step (via the conv VJP bridges).
  | convWeightSgd {ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convBiasSgd   {ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- Chapter 4 (per-channel BatchNorm) param-SGD tail (the BN train step, folded into
  -- the AST): the fused per-channel γ/β update ops. `bnGammaSgd`: `γ − lr·dγ`,
  -- `dγ_c = Σ_{b,h,w} dy·x̂` (x̂ recomputed from the saved BN input `v` = conv output,
  -- `den` = `cifar_bn_render_gamma_certified` via `reassocFwd`); `bnBetaSgd`: `β − lr·dβ`,
  -- `dβ_c = Σ_{b,h,w} dy`. `gName`/`bName`/`vName` are the γ/β/conv-output SSA names;
  -- `epsStr` the ε literal. `SgdNodes` proves both `den`s = the certified step.
  | bnGammaSgd {oc h w : Nat} (gName vName epsStr lrStr : String) (ε : ℝ) (γ : Vec oc)
      (v : Vec (oc*h*w)) (lr : ℝ)                          : SHlo (oc*h*w) → SHlo oc
  | bnBetaSgd  {oc h w : Nat} (bName lrStr : String) (β : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- Chapter 4 (BatchNorm): per-example normalization over the whole feature
  -- vec (reduce mean/var over axis [1], scalar γ/β). `gName,bName` are the γ,β
  -- scalar SSA inputs, `epsStr` the rendered ε literal; ε,γ,β carry the den.
  | bnF        {n : Nat} (gName bName epsStr : String) (ε γ β : ℝ)   : SHlo n → SHlo n
  -- BN input-VJP — the consolidated O(N) three-term gradient (`bnGradInput`),
  -- recomputing x̂/istd from the saved BN input `x` (`xName`). Total in `x`;
  -- faithful (= pdiv-Jacobian) under `0 < ε` (`bn_input_grad_correct`).
  | bnBack     {n : Nat} (gName xName epsStr : String) (ε γ : ℝ) (x : Vec n) : SHlo n → SHlo n
  -- Chapter 5 (ResNet): residual add (`stablehlo.add`) and global-average-pool.
  -- `addV` is binary (mirrors `.sub`); the residual skip reuses the block-input
  -- subtree in BOTH operands, so the graph stays a tree. `gapF` reduces the
  -- spatial axes (`reduce add over [2,3]`, ÷h·w), `Vec (c*h*w) → Vec c`.
  | addV       {n : Nat}                                        : SHlo n → SHlo n → SHlo n
  -- The BATCHED peers of `addV`/`sub`: same pointwise `den`, but the per-example
  -- emit width `n` is separated from the batch `N` so the node can sit in a graph
  -- indexed at `N·n` (where the batch-coupled `den`s are honest). The unbatched
  -- ctors above stay exactly as they were for the per-example renderers.
  | addVB      {N n : Nat}                                      : SHlo (N*n) → SHlo (N*n) → SHlo (N*n)
  | subB       {N n : Nat}                                      : SHlo (N*n) → SHlo (N*n) → SHlo (N*n)
  | gapF       {c h w : Nat}                                    : SHlo (c*h*w) → SHlo c
  -- GAP backward (VJP): per-channel cotangent broadcast over H×W, /(h·w).
  | gapBack    {c h w : Nat}                                    : SHlo c → SHlo (c*h*w)
  -- Broadcast backward (VJP = sum-over-spatial): the adjoint of `broadcastFlat`.
  | broadcastBack {c h w : Nat}                                 : SHlo (c*h*w) → SHlo c
  -- Chapter 5 Milestone B (ResNet-34 downsampling): stride-2 SAME conv forward
  -- (`stablehlo.convolution` with `window_strides=[2,2]`) and its input-VJP
  -- (zero-upsample the cotangent — `lhs_dilation` — then the reversed-kernel
  -- conv). `den` via the proven `flatConvStride2` / `flatConvStride2HasVJP`.
  | flatConvStridedF {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)              : SHlo (ic*(2*h)*(2*w)) → SHlo (oc*h*w)
  -- The XLA-`SAME` peer, for the TF-origin nets' per-example chains (`planning/archive/mnv4_verified.md`
  -- §3h). Same type, `pad` differs by one — see `BatchableOp.convStridedXla` for the full note.
  | flatConvStridedXlaF {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)              : SHlo (ic*(2*h)*(2*w)) → SHlo (oc*h*w)
  | convStridedBack  {ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*(2*h)*(2*w))) : SHlo (oc*h*w) → SHlo (ic*(2*h)*(2*w))
  -- Chapter 5 Milestone B (ResNet-34 downsampling) param-SGD tail: the strided conv
  -- kernel/bias update ops — the stride-2 analogues of `convWeightSgd`/`convBiasSgd`.
  -- `convStridedWeightSgd`: `W − lr·(flatConvStride2_weight_grad(b,x)·dy)` — zero-upsample
  -- the cotangent (the decimate-backward) then the SAME transpose-trick stride-1 weight-grad
  -- conv on the 2h×2w grid; `den` = the generic strided weight bridge (covers the 3×3
  -- downsample/projection AND the 7×7 stem, kH/kW-generic). `convStridedBiasSgd`: the bias
  -- grad is stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so it emits the SAME `reduce` text
  -- as `convBiasSgd` (its `skel` aliases that op's Raw); only its `den` differs (the strided
  -- VJP). ResNet34Fold proves both `den`s = the certified loss-descent step.
  | convStridedWeightSgd {ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (ic*(2*h)*(2*w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convStridedBiasSgd   {ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w))) (b : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- The XLA-`SAME` per-example peers, for MobileNetV2's per-example SGD train step
  -- (`MobileNetV2Render.lean`, one MobileNetV2 program since 2026-09-05). `den` is the
  -- `flatConvStride2Xla` weight / bias VJP. The weight op emits `convStridedWeightSgd`'s text with
  -- the correlation pad shifted one position (`[p-1, p+1]`, exactly `convStridedXlaWeightSgdB`);
  -- the bias grad is stride- and phase-independent, so `convStridedXlaBiasSgd` emits the same
  -- `reduce` as `convBiasSgd` (its `skel` aliases that Raw) and only its `den` differs.
  | convStridedXlaWeightSgd {ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (ic*(2*h)*(2*w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convStridedXlaBiasSgd   {ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w))) (b : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- MobileNetV2 (inverted-residual) param-SGD tail: the depthwise kernel/bias update ops,
  -- the depthwise analogues of `convWeightSgd`/`convBiasSgd`. `depthwiseWeightSgd` (stride-1,
  -- blocks b2/b4): `W − lr·(depthwise_weight_grad(b,x)·dy)` via the per-channel transpose-trick
  -- conv (`batch_group_count = c`, output [1,c,kH,kW]→[c,1,kH,kW]); `den` =
  -- `Mnv2PoC.depthwiseW_den`. `depthwiseStridedWeightSgd` (stride-2, blocks b1/b3/b5/b6):
  -- zero-upsample dy (interior=1 → 2h×2w) then the SAME per-channel weight-grad on the 2h×2w grid;
  -- `den` = `W − lr·` `depthwiseStride2WeightGradHasVJP`'s backward. The depthwise bias grad is stride-INDEPENDENT
  -- (`Σ_{batch,spatial} dy`), so both bias ops emit the SAME `reduce` text as `convBiasSgd` (their
  -- `skel` aliases that op's Raw); only their `den` differs. MobileNetV2Fold proves all four
  -- `den`s = the certified loss-descent step.
  | depthwiseWeightSgd {c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo (c*kH*kW)
  | depthwiseBiasSgd   {c h w kH kW : Nat} (bName lrStr : String)
      (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo c
  | depthwiseStridedWeightSgd {c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (c*(2*h)*(2*w))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo (c*kH*kW)
  | depthwiseStridedBiasSgd   {c h w kH kW : Nat} (bName lrStr : String)
      (W : DepthwiseKernel c kH kW) (x : Vec (c*(2*h)*(2*w))) (b : Vec c) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo c
  -- The XLA-`SAME` per-example peers (MobileNetV2's four strided depthwises in its SGD train
  -- step). `den` = `depthwiseStridedXla{Weight,Bias}SgdDen` (Depthwise.lean), the
  -- `depthwiseStride2FlatXla` VJPs. The weight op's per-channel correlation pad shifts to
  -- `[p-1, p+1]` (as `depthwiseStridedXlaWeightGradB`); the bias op aliases `convBiasSgd`'s Raw.
  | depthwiseStridedXlaWeightSgd {c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (c*(2*h)*(2*w))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo (c*kH*kW)
  | depthwiseStridedXlaBiasSgd   {c h w kH kW : Nat} (bName lrStr : String)
      (W : DepthwiseKernel c kH kW) (x : Vec (c*(2*h)*(2*w))) (b : Vec c) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo c
  -- Chapter 8 (ConvNeXt-T) param-SGD tail. `layerScaleChGammaSgd`: the PER-CHANNEL layer-scale γ
  -- update `γ_c − lr·dγ_c`, `dγ_c = Σ_{b,h,w} x⊙dy` (the saved layer input `x` ⊙ the cotangent,
  -- reduced over batch+spatial per channel — `lsGradCh`). `γ : Vec c`, broadcast over spatial via
  -- `chanIdx` by the `layerScaleChF` forward; `den` = ConvNeXtFold's `cnx_render_lsgammaCh`.
  | layerScaleChGammaSgd {c h w : Nat} (gName xName lrStr : String)
      (x : Vec (c*h*w)) (γ : Vec c) (lr : ℝ)               : SHlo (c*h*w) → SHlo c
  -- `lnGammaSgd`/`lnBetaSgd`: the SCALAR LayerNorm γ/β updates (the `bnF` sites — scalar LN over the
  -- whole `n = c·h·w`, `γ β : Vec 1` ≅ `tensor<f32>`). `dγ = Σ_{b,k} dy·x̂` (x̂ recomputed from the
  -- saved LN input `x`, `lnParamGrad`'s dγ half), `dβ = Σ_{b,k} dy`. `den` = the certified scalar-LN
  -- grad (`cnx_render_ln{gamma,beta}_certified`, the Vec-1 embedding). Output `SHlo 1`.
  | lnGammaSgd {n : Nat} (gName xName epsStr lrStr : String) (ε : ℝ) (x : Vec n) (γ : Vec 1) (lr : ℝ)
                                                           : SHlo n → SHlo 1
  | lnBetaSgd  {n : Nat} (bName lrStr : String) (β : Vec 1) (lr : ℝ)
                                                           : SHlo n → SHlo 1
  -- `veclnGammaSgd`: the Chapter-9 ViT VECTOR-[D] LayerNorm γ update. Per-token normalize over the
  -- `D` feature axis (x̂ = `layerNormForward D ε 1 0`), then per-channel affine `γ⊙x̂+β`; the γ grad
  -- `dγ_k = Σ_rows dy·x̂` reduces over the N=tokens row axis but KEEPS `D` (output `SHlo D` ≅
  -- `tensor<Dxf32>`, vs `lnGammaSgd`'s scalar `SHlo 1`). `den` = the per-channel certified grad
  -- (`vit_render_veclngamma_certified`). The rowwise dense W/b + vecln β reuse the enet batched
  -- `denseWeightSgdB`/`denseBiasSgdB` (their N-axis sum = vit's `rowDense_*_grad`).
  | veclnGammaSgd {N D : Nat} (gName xName epsStr lrStr : String) (ε : ℝ) (x : Vec (N*D)) (γ : Vec D) (lr : ℝ)
                                                           : SHlo (N*D) → SHlo D
  -- `patchEmbedWeightSgd`: the Chapter-9 ViT patch-embed (16×16/s16 non-overlapping patchify) conv
  -- WEIGHT update. The embed-output cotangent `SHlo ((N+1)*D)` (CLS token at row 0, excluded) drives
  -- the strided patchifyWGrad (dilate the patch-token grad interior P-1, valid conv with the saved
  -- image) → `dW : Kernel4 D ic P P`. `den` = the certified patch-weight grad
  -- (`vit_render_patchW_certified`, via the local `patchEmbedWeightGradFlat`). Output `SHlo (D*ic*P*P)`.
  -- The ViT analogue of ConvNeXt's stem 4×4/s4 weight — but here a VJP-cert EXISTS, so it is tied
  -- (vit has no even-kernel weight gap). Patch bias + cls + pos reuse the batched `denseBiasSgdB`.
  | patchEmbedWeightSgd {ic H W P N D : Nat} (wName xName lrStr : String)
      (x : Vec (ic*H*W)) (Wp : Kernel4 D ic P P) (lr : ℝ) : SHlo ((N+1)*D) → SHlo (D*ic*P*P)
  -- ViT patch-embed BIAS update: the conv bias only touches the N patch tokens (CLS row 0 excluded),
  -- so `db = Σ_{patches,batch} cot` (slice [1..N], reduce[0,1]). `den` = the certified `vit_render_patchb`.
  | patchEmbedBiasSgd {N c : Nat} (bName lrStr : String) (b : Vec c) (lr : ℝ) : SHlo ((N+1)*c) → SHlo c
  -- ViT positional-embedding update: `pos : Mat (N+1) D` is added to EVERY token (broadcast over
  -- batch), so its Jacobian is the identity ⇒ `dPos = dy` (the embed cotangent, KEEPING all N+1
  -- tokens; only the emit batch is summed). Unlike `patchEmbedBiasSgd`/`denseBiasSgdB` (which reduce
  -- to `[c]`), pos KEEPS the `(N+1)` token axis, so its update is the 2D `tensor<(N+1)xDxf32>` — a
  -- flat `denseBiasSgd` would mismatch the `%pos: tensor<197x192xf32>` arg. `den` = `vit_render_pos_certified`.
  | posEmbedSgd {N D : Nat} (pName lrStr : String) (pos : Mat (N+1) D) (lr : ℝ)
                                                           : SHlo ((N+1)*D) → SHlo ((N+1)*D)
  -- Chapter 8 scaling pass (full ConvNeXt-T): stride-4 SAME conv forward — the
  -- 4×4/s4 patchify stem (`stablehlo.convolution` with `window_strides=[4,4]`).
  -- `den` via the proven `flatConvStride4` (= decimate ∘ decimate ∘ stride-1 conv).
  | flatConvStride4F {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) : SHlo (ic*(2*(2*h))*(2*(2*w))) → SHlo (oc*h*w)
  -- Chapter 5 Milestone B8 (real-ResNet PER-CHANNEL BatchNorm): normalize each
  -- channel-slice over its h·w spatial cells with its OWN `(γ_c, β_c)`, γ/β : `Vec oc`
  -- (rank-1, `broadcast dims=[1]` — vs `bnF`'s rank-0 scalars). `den` via the proven
  -- `bnPerChannelTensor3` (the Mat-split block-diagonal BN bridged into the `(oc*h)*w`
  -- activation layout) / its renderable backward `bnPerChannelTensor3GradInput`.
  | bnPerChannelF    {oc h w : Nat} (gName bName epsStr : String) (ε : ℝ) (γ β : Vec oc)
                                                           : SHlo (oc*h*w) → SHlo (oc*h*w)
  | bnPerChannelBack {oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (oc*h*w))                                   : SHlo (oc*h*w) → SHlo (oc*h*w)
  -- INFERENCE per-channel BN: the same affine map with the statistics FROZEN — μ/var arrive as
  -- graph inputs (`muName`/`varName`, the driver's EMA'd running stats) instead of being reduced
  -- out of the activation. No reduction ⇒ pointwise ⇒ an example's logits do not depend on which
  -- other examples share its batch. Denotes `bnPerChannelEvalTensor3`. Forward-only: eval has no
  -- backward, so there is deliberately no `bnPerChannelEvalBack`.
  | bnPerChannelEvalF {oc h w : Nat} (gName bName muName varName epsStr : String) (ε : ℝ)
      (γ β μ var : Vec oc)                                 : SHlo (oc*h*w) → SHlo (oc*h*w)
  -- Chapter 6 (MobileNetV2): depthwise conv forward (`stablehlo.convolution` with
  -- `feature_group_count = c` and a `[c, 1, kH, kW]` kernel — one filter per channel,
  -- no cross-channel mixing) and its input-VJP (the SAME-pad reversed-kernel depthwise
  -- conv — spatial flip only, since the per-channel groups are 1×1; same
  -- `feature_group_count`). `den` via the proven `depthwiseFlat` / `depthwiseFlatHasVJP`.
  | depthwiseF    {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*h*w) → SHlo (c*h*w)
  | depthwiseBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*h*w)) : SHlo (c*h*w) → SHlo (c*h*w)
  -- Chapter 6 C3: STRIDE-2 depthwise conv forward (`window_strides=[2,2]`,
  -- `feature_group_count = c`, `[c,1,kH,kW]` kernel — halves spatial, the MNv2
  -- downsampling op) and its input-VJP (zero-upsample the cotangent via
  -- `stablehlo.pad` interior=1 then the reversed-kernel stride-1 depthwise — the
  -- `convStridedBack` shape, per-channel). `den` via the proven `depthwiseStride2Flat`
  -- / `depthwiseStride2FlatHasVJP` (= decimate ∘ depthwise).
  | depthwiseStridedF    {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  | depthwiseStridedXlaF {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  | depthwiseStridedBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- The XLA-`SAME` per-example input-VJP (MobileNetV2's SGD train step). ⚠ Its transposed-conv
  -- pad is `[p+1, p-1]` — the OPPOSITE shift from the two weight grads, because the kernel is
  -- reversed here; the batched `depthwiseStridedXlaBackBatched` carries the full note and
  -- `scripts/gates/xla_pad_op_check.py` checks both. `den` is `depthwiseStride2FlatXlaHasVJP`.
  | depthwiseStridedXlaBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- Chapter 7 (EfficientNet): swish forward (`x · σ(x)`, σ = `stablehlo.logistic`)
  -- and its input-VJP (`dy · swish'(x)`, closed form `σ(x)·(1 + x·(1−σ(x)))`).
  -- Swish is SMOOTH everywhere (no kink, NO smoothness hyp — unlike relu6); the
  -- VJP is the GLOBAL `swishHasVJP` (no `_at`). `swishBack`'s `xName`/`x` is the
  -- saved pre-activation. `den` via the proven `swish` / `swishHasVJP` (LayerNorm.lean).
  | swishF     {n : Nat}                                        : SHlo n → SHlo n
  | swishBack  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 7 (EfficientNet): sigmoid forward (`σ(x) = stablehlo.logistic`, the SE
  -- gate's output nonlinearity) and its input-VJP (`dy · σ(x)·(1−σ(x))`). Like swish,
  -- SMOOTH everywhere (no kink, NO smoothness hyp — GLOBAL `sigmoidHasVJP`, not `_at`).
  -- `sigmoidBack`'s `xName`/`x` is the saved pre-activation. `den` via the proven
  -- `sigmoid` / `sigmoidHasVJP` (SE.lean).
  | sigmoidF     {n : Nat}                                      : SHlo n → SHlo n
  | sigmoidBack  {n : Nat} (xName : String) (x : Vec n)         : SHlo n → SHlo n
  -- The BATCHED peers of `swishBack`/`sigmoidBack`. Identical `den` — the SAME
  -- pointwise VJP over the whole batch — but the index is split into the batch `N`
  -- and the per-example emit width `n`, so these can sit in a graph indexed at `N·n`
  -- (where the batch-coupled `den`s are honest) while still emitting `tensor<B×n>`.
  -- `x` is the WHOLE-BATCH saved pre-activation, `Vec (N*n)`: it is what the emitted
  -- `xName` holds at runtime, and it is why these are not `BatchableOp` descriptors
  -- (`batchMap N` lifts a fixed function, so it would share one example's saved
  -- activation across the batch — a different, wrong, function).
  -- The BATCHED peer of `selectPos` (ResNet-34's ReLU backward mask). Same `den`, but
  -- `x` is the WHOLE-BATCH saved pre-activation, for the reason spelled out on
  -- `BatchableOp`: the mask is per-example data, so this cannot be a descriptor.
  -- Batched 2×2 max-pool BACKWARD. `x` is the WHOLE-BATCH saved pre-pool input; `den` is
  -- `batchMapAux`, which hands example `n` its OWN slice of `x` (a `batchMap` descriptor would
  -- hand every example one example's input). Conditional (no window ties), like the unbatched op.
  | maxPoolBackB {N c h w : Nat} (xName : String) (x : Vec (N*(c*(2*h)*(2*w)))) :
      SHlo (N*(c*h*w)) → SHlo (N*(c*(2*h)*(2*w)))
  -- ⭐ Batched **3×3/s2** max-pool backward — `maxPoolBackB`'s peer at the paper's stem pool.
  -- Same `batchMapAux` story (example `n` gets its OWN slice of `x`), same reason it is not a
  -- descriptor. What differs from the 2×2 peer is only inside `maxPool3s2BackFlat`: the windows
  -- overlap, so the backward SUMS over every output that selected this input rather than looking
  -- one up. The emitted `select_and_scatter` needed no change for that — it already reduces with
  -- `add`. `planning/archive/rsb_a3_r50_verified.md` §4b.
  | maxPool3s2BackB {N c h w : Nat} (xName : String) (x : Vec (N*(c*(2*h)*(2*w)))) :
      SHlo (N*(c*h*w)) → SHlo (N*(c*(2*h)*(2*w)))
  -- Batched conv BIAS param-SGD, the peers of `conv{,Strided}WeightSgdB`: `b − lr·Σ_n dβ_n`,
  -- the same shared-parameter batch sum the rest of the `*SgdB` family takes. The bias grad is
  -- stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so both `skel` to ONE Raw — the emitted text is
  -- identical and only `den` differs, exactly as `convStridedBiasSgd` aliases `convBiasSgd`.
  | convBiasSgdB {N ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * h * w))) (b : Vec oc) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo oc
  | convStridedBiasSgdB {N ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2*h) * (2*w)))) (b : Vec oc) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo oc
  | selectPosB   {N n : Nat} (xName : String) (x : Vec (N*n))   : SHlo (N*n) → SHlo (N*n)
  -- MobileNetV2's ReLU6 backward mask (§2f), the batched peer of `selectMid` and the exact
  -- `selectPosB` shape one kink up: `if 0 < x i ∧ x i < 6 then dy i else 0` reads the saved
  -- pre-activation, which is PER-EXAMPLE data — so this is an own constructor holding the
  -- WHOLE-BATCH `x`, not a `BatchableOp` descriptor beside `relu6`. A descriptor here would
  -- denote "every example shares example 0's mask", which is not what the emit computes.
  | selectMidB   {N n : Nat} (xName : String) (x : Vec (N*n))   : SHlo (N*n) → SHlo (N*n)
  -- ▶ STOCHASTIC DEPTH (`planning/archive/stochastic_depth.md`): `branch * keep / keep_prob`, the
  -- per-SAMPLE branch scale. `mName` is a graph INPUT of type `tensor<Nxf32>` — the mask is drawn
  -- on the HOST, never by `stablehlo.rng`, because every numeric gate in this repo is a
  -- bit-exactness or known-answer argument over a deterministic graph (§2, that doc).
  --
  -- ⚠ It is an own constructor for `selectMidB`'s reason, one axis over: a `BatchableOp` descriptor
  -- may carry only batch-INVARIANT data (§4), and this mask is per-EXAMPLE. A descriptor would
  -- denote "every example shares example 0's mask" — which is exactly what stochastic depth is not.
  -- ⚠ `invKeep` is 1/keep_prob, the reference's INVERTED form, so eval at a ones mask is the exact
  -- identity (`Proofs.dropPath_ones_id`) and the forward render can emit the sites too — which is
  -- what keeps the `forward ⊂ train-step` prefix audit alive (§3, that doc).
  -- ⚠ THE BACKWARD IS THIS SAME OP at the same mask (`Proofs.dropPath_vjp_is_self`): a diagonal
  -- linear map is its own transpose, so there is no `*Grad` peer to build or to keep in step.
  | dropPathB    {N n : Nat} (mName : String) (s : Vec N)        : SHlo (N*n) → SHlo (N*n)
  -- ▶ CLASSIFIER DROPOUT (`recipe_gaps.md` gap C): the per-ELEMENT inverted mask the reference
  -- applies immediately before the classifier dense (`emitForward`'s classifier dropout in `jax/Jax/Codegen.lean`). `mName` is a
  -- graph INPUT of type `tensor<N×n×f32>`, drawn on the HOST for `dropPathB`'s reasons exactly.
  --
  -- ⚠⚠ IT IS `dropPathB` AT A MASK OF THE VALUE'S OWN TYPE, AND THAT IS THE ONLY DIFFERENCE.
  -- `Proofs.dropout_of_dropScale` proves the containment (`dropPath` is this op at a lifted mask);
  -- `Proofs.dropPath_scales_uniformly` proves the gap. In the emitted text the whole distinction is
  -- one line: `dropPathP` broadcasts `tensor<B>` over `dims = [0]`, this multiplies directly. Each
  -- of the two is what the OTHER's comments have been warning about — "emitting a `tensor<B×n>`
  -- scale is per-element dropout, a different regulariser" is now a live op, so the confusion runs
  -- both ways and both directions are pinned in `tests/TestBatchedEmitTie.lean`.
  --
  -- ⚠ Own constructor for `dropPathB`'s reason inverted: a `BatchableOp` descriptor's `den` is
  -- `batchMap N (denOp op)`, ONE fixed function, so it could not carry a mask that differs across
  -- examples any more than it could carry a saved per-example activation.
  -- ⚠ NO BAKED `1/keep` — the driver folds the inversion into the supplied mask, which is what
  -- makes the ones-mask forward the exact identity (`Proofs.dropout_ones_id`) and lets the op be
  -- emitted in the forward at all, keeping the `forward ⊂ train-step` prefix audit alive.
  -- ⚠ THE BACKWARD IS THIS SAME OP at the same mask (`Proofs.dropout_vjp_is_self`) — but see that
  -- theorem's note: the classifier WEIGHT gradient reads the dense's input, which is the DROPPED
  -- activation, and no ones-mask gate can see that being wrong.
  | dropoutB     {N n : Nat} (mName : String) (mask : Vec (N*n)) : SHlo (N*n) → SHlo (N*n)
  | swishBackB   {N n : Nat} (xName : String) (x : Vec (N*n))   : SHlo (N*n) → SHlo (N*n)
  -- ── ViT / ConvNeXt's two saved-activation BACKWARDS (§0.2 ▶2, increment 2). These cannot be
  --    `BatchableOp` descriptors and the reason is the descriptor rule itself: a descriptor's
  --    `den` is `batchMap N (denOp op)`, ONE fixed function, which would hand example 0's saved
  --    activation to all `N`. They take the whole-batch `x` instead — `geluBackB` pointwise (so
  --    the VJP at width `N*n` already IS the batch-lift, `swishBackB`'s exact shape), `lnRowBackB`
  --    via `batchMapAux` (so example `n` gets `batchSlice n x`).
  | geluBackB    {N n : Nat} (xName : String) (x : Vec (N*n))   : SHlo (N*n) → SHlo (N*n)
  -- ── increment 3: the batch-contracting PARAMETER gradients (`Σ_n` over the batch, the shape
  --    every `*GradB` takes). Two of them contract TWO levels — the batch AND the row axis —
  --    which no existing `*GradB` does, because `denseBiasGradB` sits on a net where each example
  --    is one row. ⚠ That is why the row count `R` is an explicit index here: reading `rowDense`'s
  --    own `N` as the batch is precisely the confusion the batched index exists to prevent.
  | convStride4WeightGradB {N ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Vec (N * (ic*(2*(2*h))*(2*(2*w))))) (W : Kernel4 oc ic kH kW)
      : SHlo (N * (oc*h*w)) → SHlo (oc*ic*kH*kW)
  -- ⭐ Its **bf16** peer, and ConvNeXt's second and last new op. As with `convWeightGradBBf16`, the
  -- wgrad is the transpose-trick convolution (the batch IS the contraction dim), so the whole `Σ_n`
  -- is ONE emitted convolution and therefore ONE bf16 store — which is why the outer `rnd` sits
  -- outside the sum, not inside it.
  -- ⭐ **There is no `convStride4BackBatchedBf16` and there must not be one.** `convStride4` is the
  -- patchify STEM, so its input is `%x` and there is no input gradient to compute. Two new ops for
  -- this net, not three.
  | convStride4WeightGradBBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec oc) (x : Vec (N * (ic*(2*(2*h))*(2*(2*w))))) (W : Kernel4 oc ic kH kW)
      : SHlo (N * (oc*h*w)) → SHlo (oc*ic*kH*kW)
  | layerScaleChGammaGradB {N c h w : Nat} (xName : String) (x : Vec (N*(c*h*w)))
      : SHlo (N*(c*h*w)) → SHlo c
  | veclnGammaGradB {N R D : Nat} (xName epsStr : String) (ε : ℝ) (x : Vec (N*(R*D)))
      : SHlo (N*(R*D)) → SHlo D
  | rowDenseBiasGradB {N R c : Nat}                             : SHlo (N*(R*c)) → SHlo c
  -- ⚠ The HEAD's dense grads. `denseWeightGradB`/`denseBiasGradB` already exist and denote the
  -- same thing — but they carry their OWN `Raw` tags ("denseWeightGrad"/"denseBiasGrad"), where
  -- ConvNeXt's and ViT's head emits `weightGrad`/`biasGrad`. Reusing them would change the
  -- emitted text, so these two alias the per-example Raw instead. A reminder that two ops
  -- denoting the same function are still two RENDERS, and the byte tie is what notices.
  | weightGradB {N m n : Nat} (xName : String) (x : Vec (N*m))  : SHlo (N*n) → SHlo (m*n)
  | biasGradB   {N n : Nat}                                     : SHlo (N*n) → SHlo (N*n)
  | lnRowBackB   {N m n : Nat} (gName xName epsStr : String) (ε γ : ℝ) (x : Vec (N*(m*n)))
      : SHlo (N*(m*n)) → SHlo (N*(m*n))
  -- ⭐ The BATCHED forward sigmoid, for **BCE-with-logits** (RSB-A2/A3's loss). `sigmoidF` already
  --    denotes `Proofs.sigmoid` and already carries a global hypothesis-free `sigmoidHasVJP`, but
  --    it is indexed PER EXAMPLE and emits at `ty [B, n]`; the loss cotangent lives at `SHlo (N*n)`
  --    with the `*B` family (`subB`, `divConstB`). So this is the same function at the batched
  --    index — one constructor, and `planning/archive/next_session_pipeline_then_r50.md` §4 estimated
  --    "~1 descriptor" for exactly this.
  -- ⚠ It needs NO new `Raw`/`Tok`/parse constructor: `skel` maps it onto the generic
  --    `.batched "sigmoidP"` node the whole batched-pointwise family shares.
  | sigmoidB     {N n : Nat}                                    : SHlo (N*n) → SHlo (N*n)
  | sigmoidBackB {N n : Nat} (xName : String) (x : Vec (N*n))   : SHlo (N*n) → SHlo (N*n)
  -- ── ViT increment 2 (handoff §0.2 ▶3): the six forms that CANNOT be descriptors, because each
  --    either reads a per-example saved activation or contracts the batch away.
  --
  --    ⚠⚠ `matmulFB` WAS BILLED AS THE SCHEDULE RISK AND IT IS NOT ONE. §0.2's cost note says
  --    attention needs "a `batchMap2`-shaped combinator, its own VJP and a `dot_general` carrying a
  --    batching dimension", because both its operands are per-example where every batched binary in
  --    the kit is pointwise-same-shape. Measured instead of estimated, all three evaporate:
  --      · **`batchMapAux` IS `batchMap2`.** Its body is `f (batchSlice aux k) (batchSlice x k)` —
  --        fully symmetric in the two arguments. Nothing in the definition requires the first to be
  --        a saved activation rather than a second graph operand; only the docstring did.
  --      · **the `dot_general` already carries `batching_dims = [0] x [0]`.** `matmulF`'s emit was
  --        written per-example from the start, so it needs no change — the `expe`/`softmaxDiv`
  --        situation (§0.2 increment 3) with only the `den` half wrong.
  --      · **`.batched2` already exists** (`addVB`/`subB` use it), and `matmulF`'s own `Raw` tag is
  --        binary, so this aliases it and needs no new skeleton shape.
  --    What is left is a constructor, a `den`, a `rfl` and one `skel` line. **Four sites.**
  | matmulFB {N m k n : Nat} : SHlo (N*(m*k)) → SHlo (N*(k*n)) → SHlo (N*(m*n))
  -- ⭐⭐ bf16 peer of `matmulFB` — **the first ACTIVATION × ACTIVATION bf16 op in the kit.** Every
  -- other bf16 op here rounds a constant weight against a running value; SDPA's `QKᵀ` and `P·V`
  -- round two activations. §10.3 called this the genuinely new KIND and it was right.
  -- ▶ The accuracy side comes free anyway: `dot_close_mixed` rounds BOTH operands and never asks
  -- which one is a weight, so this needs no theorem the dense case did not already have.
  -- ⚠ bf16 operands, **bf16-typed result**, convert back — see `denseRowBackBf16` for why that is
  -- not `dotInBf16`'s shape any more. Re-checked at ViT's own batched `[32,197,64] × [32,64,197]`
  -- in §17.2: batching dims buy no exemption in either direction.
  | matmulFBBf16 {N m k n : Nat} (rnd : ℝ → ℝ)
      : SHlo (N*(m*k)) → SHlo (N*(k*n)) → SHlo (N*(m*n))
  -- ⚠ A saved-activation backward, so it takes the WHOLE-batch `preAct` and hands example `k` its
  -- own slice via `batchMapAux`. A descriptor would give every example example 0's scores — same
  -- types, same emitted bytes, different function. `lnRowBackB`'s situation exactly.
  | softmaxRowBackB {N m n : Nat} (xName : String) (preAct : Vec (N*(m*n)))
      : SHlo (N*(m*n)) → SHlo (N*(m*n))
  -- ── the four batch-contracting parameter gradients. ⚠ Every one of their per-example emits
  --    ALREADY reduces over the batch axis (`dimensions = [0, 1]`, `[0]`,
  --    `contracting_dims = [0, 1] x [0, 1]`) — values flow as `tensor<B, …>` and `B` is `pretty`'s,
  --    never the SHlo index — so all four alias their per-example `Raw` and emit the same text BY
  --    CONSTRUCTION rather than by a copied body. It was only ever the `den` that was per-example.
  --    ⚠⚠ And the two-level contraction is INVISIBLE AT `N = 1`: a render that dropped the batch
  --    sum type-checks, emits the same bytes and agrees on a one-example batch. Any gate must run
  --    at `N > 1` (`den_rowDenseBiasGradB_at_one` states the same thing one op over).
  | rowDenseWeightGradB {N tk a c : Nat} (xName : String) (x : Vec (N*(tk*a)))
      : SHlo (N*(tk*c)) → SHlo (a*c)
  -- ⭐ bf16 peer — the weight gradient of the six per-block denses. `dot_general` contracting BOTH
  -- the batch and the token axis (`[0,1] x [0,1]`), so the batch sum happens inside one dot.
  -- ⚠⚠ **THE ONE ViT DOT THAT KEEPS ITS f32 RESULT, deliberately.** §20.1's win comes from a
  -- gemm writing half the bytes, and this gemm's result is the WEIGHT `[a,c]` — 147K elements
  -- against the activations' 4.8M — so there is no bandwidth to save. What a bf16 store would buy
  -- is nothing and what it would cost is precision on the optimizer's input. ▶ Hence no outer
  -- rounding in `den`: operands rounded, f32 accumulate, f32 store.
  -- ⚠ The BIAS gradient beside it (`rowDenseBiasGradB`) stays f32 in every net, for the reason it
  -- does in all six: `Σ dy` is a reduction, not a contraction, and there is no tensor core in it.
  | rowDenseWeightGradBBf16 {N tk a c : Nat} (rnd : ℝ → ℝ) (xName : String) (x : Vec (N*(tk*a)))
      : SHlo (N*(tk*c)) → SHlo (a*c)
  | posEmbedGradB {N tk D : Nat}                : SHlo (N*((tk+1)*D)) → SHlo ((tk+1)*D)
  | patchEmbedWeightGradB {N ic H W P tk D : Nat} (xName : String) (x : Vec (N*(ic*H*W)))
      : SHlo (N*((tk+1)*D)) → SHlo (D*ic*P*P)
  -- ⭐ bf16 peer of the stem's weight grad — the second half of ViT's `convolution` pair, and it
  -- takes the CONV shape (bf16 operands, **bf16-typed result**, convert back) for the same
  -- measured reason `patchEmbedBf16` does. ⚠⚠ Its convolution contracts the BATCH axis, so the
  -- `Σ_b` sits INSIDE the bf16 store — which is why `den`'s outer `rnd` wraps the whole batch sum
  -- and not each summand. Writing it per-summand would claim a rounding the hardware never does.
  -- ⭐ There is no `patchEmbedBackBf16` and there is no `patchEmbedBack` in ViT's traversal at all:
  -- the stem's input is `%x`, so it has no input gradient — ConvNeXt's `convStride4` exactly
  -- (§16.5), and the reason this net needs six ops rather than seven.
  | patchEmbedWeightGradBBf16 {N ic H W P tk D : Nat} (rnd : ℝ → ℝ) (xName : String)
      (x : Vec (N*(ic*H*W))) : SHlo (N*((tk+1)*D)) → SHlo (D*ic*P*P)
  -- ⚠ Sums tokens 1…tk and SKIPS the CLS row, which is what `p.succ` says. A batched peer that
  -- summed all `tk+1` rows would fold the CLS token's cotangent into the patch bias — it compiles,
  -- trains and descends, and the emitted `slice [.., 1:tk+1, ..]` is the only place it shows.
  | patchEmbedBiasGradB {N tk c : Nat}          : SHlo (N*((tk+1)*c)) → SHlo c
  -- Chapter 8 (ConvNeXt): GELU forward (tanh approximation,
  -- `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, via `stablehlo.tanh`) and its
  -- input-VJP (`dy · gelu'(x)`, closed form from the tanh-approx derivative).
  -- Like swish/sigmoid, SMOOTH everywhere (no kink, NO smoothness hyp — the VJP is
  -- the GLOBAL `geluHasVJP`, not `_at`). `geluBack`'s `xName`/`x` is the saved
  -- pre-activation. `den` via the proven `gelu` / `geluHasVJP` (LayerNorm.lean).
  | geluF      {n : Nat}                                        : SHlo n → SHlo n
  | geluBack   {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 8 (ConvNeXt): per-element layer-scale `γ ⊙ x` (diagonal linear, `γ : Vec n`
  -- over the flattened `c·h·w` map). `den` via the proven `layerScale` (LayerNorm.lean).
  | layerScaleF {n : Nat} (γName : String) (γ : Vec n)          : SHlo n → SHlo n
  -- Per-CHANNEL layer-scale (the paper's form, the committed full-T render's
  -- `tensor<c>` γ): `den` = the proven `layerScale` at the channel-expanded
  -- vector `γ ∘ chanIdx` (a constant reindex of the parameter).
  | layerScaleChF {c h w : Nat} (γName : String) (γ : Vec c)    : SHlo (c*h*w) → SHlo (c*h*w)
  -- Chapter 9 (ViT): ROW-softmax forward — each of the `m` rows of an `[m,n]`
  -- matrix (flattened to `Vec (m*n)`, row-major) gets the 1-D `softmax` over its
  -- `n` columns (`reduce add` over the LAST axis, broadcast, divide — NO max-shift,
  -- matching the proven plain exp/sum `softmax`). `den` via `rowSoftmaxFlat` (=
  -- `Mat.flatten ∘ rowSoftmax ∘ Mat.unflatten`, the proven `rowSoftmax`).
  | softmaxRowF    {m n : Nat}                                  : SHlo (m*n) → SHlo (m*n)
  -- ROW-softmax input-VJP — per row the proven closed form `pᵢ⊙(dyᵢ − ⟨pᵢ,dyᵢ⟩)`
  -- with `p = softmax(preActᵢ)` recomputed from the saved pre-softmax scores
  -- (`xName`/`preAct`). SMOOTH everywhere (softmax has no kink). `den` via
  -- `rowSoftmaxBackFlat` (= `Mat.flatten ∘ rowSoftmaxHasVJPMat.backward ∘ Mat.unflatten`).
  | softmaxRowBack {m n : Nat} (xName : String) (preAct : Vec (m*n)) : SHlo (m*n) → SHlo (m*n)
  -- Chapter 9 (ViT): matrix multiply `C = A·B` on row-major flattened operands
  -- (reshape both to rank-3, `stablehlo.dot_general` batching dim 0, contract A's
  -- last axis with B's middle, reshape back). Binary like `.sub`/`.addV`. `den` via
  -- `matMulFlat` (= flatten ∘ `Mat.mul` ∘ unflatten). The attention BACKWARDS reuse
  -- this same token — matmul's VJP IS matmul (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`).
  | matmulF    {m k n : Nat}                                    : SHlo (m*k) → SHlo (k*n) → SHlo (m*n)
  -- Matrix transpose on the row-major flat layout (`stablehlo.transpose
  -- dims=[0,2,1]` at rank 3). `den` via `transposeFlat` (= flatten ∘ `Mat.transpose`
  -- ∘ unflatten). Pairs with `matmulF` to spell the attention backward matmuls.
  | transposeF {m n : Nat}                                      : SHlo (m*n) → SHlo (n*m)
  -- Scalar multiply `s · x` (`stablehlo.multiply` against a splat constant) — the
  -- `1/√d` of SDPA. `sStr` is the rendered literal (denotation-irrelevant); `s`
  -- carries the den. Linear, so it is its own VJP.
  | scaleF     {n : Nat} (sStr : String) (s : ℝ)                : SHlo n → SHlo n
  -- ROW-wise LayerNorm forward over an `[m,n]` row-major flat: each token row gets
  -- `bnF`'s normalize/affine graph with μ/var reduced over the LAST axis (scalar
  -- γ/β — LayerNorm IS per-example BN, `layerNormForward := bnForward` defeq).
  -- `den` via `rowLNFlat` (rowwise `bnForward`).
  | lnRowF     {m n : Nat} (gName bName epsStr : String) (ε γ β : ℝ) : SHlo (m*n) → SHlo (m*n)
  -- ROW-wise LayerNorm input-VJP — per row `bnBack`'s consolidated three-term
  -- gradient, recomputing x̂/istd from the saved flat pre-LN input `x` (`xName`).
  -- Total in `x`; faithful (= pdiv-Jacobian per row) under `0 < ε`.
  | lnRowBack  {m n : Nat} (gName xName epsStr : String) (ε γ : ℝ) (x : Vec (m*n)) : SHlo (m*n) → SHlo (m*n)
  -- PER-TOKEN dense forward: every row of the `[N,a]` flat through the same
  -- `W:[a,c]` + bias (`dot_general` contracting the feature axis `[2] x [0]`,
  -- bias broadcast `dims=[2]`). `den` via `rowDenseFlat` (rowwise `dense`).
  | denseRowF  {N a c : Nat} (wName bName : String) (W : Mat a c) (b : Vec c) : SHlo (N*a) → SHlo (N*c)
  -- PER-TOKEN dense input-VJP `dX = dY·Wᵀ` (`dot_general` contracting dy's feature
  -- axis with W's OUTPUT axis `[2] x [1]`). `den` via `rowDenseBackFlat` (rowwise
  -- `Mat.mulVec W` = the proven `denseHasVJP` backward). Linear — global VJP.
  | denseRowBack {N a c : Nat} (wName : String) (W : Mat a c)   : SHlo (N*c) → SHlo (N*a)
  -- ViT patch embedding (one coarse token, like `seBlock`): stride-P VALID conv
  -- (kernel `[D,ic,P,P]`, the non-overlapping patch projection) + bias, channels-
  -- last transpose + flatten to `[N,D]` tokens, prepend the CLS token, add the
  -- position embedding. `den` via `patchEmbedFlat` (= the proven `patchEmbedFlat`,
  -- Attention.lean).
  | patchEmbedF {ic H W P N D : Nat} (wName bName clsName posName : String)
      (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N+1) D) :
      SHlo (ic*H*W) → SHlo ((N+1)*D)
  -- ViT patch-embedding input-VJP: the strided-P patchify conv's input gradient
  -- (reversed-kernel `conv_transpose` on the patch-token rows of the `[N+1,D]`
  -- cotangent; the CLS row and position-add contribute nothing — input-VJP = id
  -- on a +constant). `den` via `patchEmbedBackFlat` (= the proven
  -- `patchEmbedInputGradFormula` = `patchEmbedFlatHasVJP.backward`). Linear
  -- in the cotangent — activation-independent, so
  -- it routes through the generic `batched` Raw/Tok tag (like the strided-conv
  -- backward batched ops) rather than a bespoke top-level Raw/Tok constructor.
  | patchEmbedBack {ic H W P N D : Nat} (wName : String)
      (Wc : Kernel4 D ic P P) :
      SHlo ((N+1)*D) → SHlo (ic*H*W)
  -- CLS-token gather: row 0 of the `[N+1,D]` flat (`stablehlo.slice` after
  -- reshape) — the classifier head's input. `den` via `clsSliceFlat` (= the
  -- proven `clsTokenFlat`, Attention.lean).
  | clsSliceF  {N D : Nat}                                      : SHlo ((N+1)*D) → SHlo D
  -- CLS-slice VJP: scatter `dy` to row 0, zeros elsewhere (`stablehlo.pad` with
  -- `high = [0, N, 0]`). `den` via `clsPadFlat` (= the proven
  -- `clsTokenFlatHasVJP.backward`). Linear — global VJP.
  | clsPadF    {N D : Nat}                                      : SHlo D → SHlo ((N+1)*D)
  -- Multi-head (ch10 scaling pass): per-head column slice — head `h`'s `[N,d]`
  -- block of the `[N,heads·d]` flat (columns `[h·d,(h+1)·d)` are contiguous in the
  -- row-major layout: `stablehlo.slice` on the feature axis after reshape).
  -- `den` via `headSliceFlat` (= `mhsaLayer`'s `finProdFinEquiv (h, ·)` column
  -- gather). Linear reindex.
  | headSliceF {N heads d : Nat} (h : Fin heads)                : SHlo (N*(heads*d)) → SHlo (N*d)
  -- Multi-head: per-head column scatter — pad an `[N,d]` head block into head `h`'s
  -- columns of a zero `[N,heads·d]` (`stablehlo.pad` on the feature axis). Both the
  -- slice's VJP AND the forward concat (`concat = Σ_h headPadF h ∘ head h` — every
  -- column hits exactly one head, and the sum stays at the ONE index `N·(heads·d)`,
  -- dodging the `(N·a)+(N·b)` Nat-cast trap a binary concat token would hit). Linear.
  | headPadF   {N heads d : Nat} (h : Fin heads)                : SHlo (N*d) → SHlo (N*(heads*d))
  -- ViT vector-LN affine (the ch10 scaling pass): per-token broadcast scale — every
  -- row of an `[m,n]` flat elementwise-scaled by the SHARED `γ : [n]` (broadcast over
  -- the row axis; contrast `layerScaleF`, which has a distinct γ per position).
  -- Diagonal-linear, so it is its own input-VJP (the layer-scale trick, row-lifted).
  -- `den` via `rowScaleFlat`.
  | rowScaleF  {m n : Nat} (gName : String) (γ : Vec n)         : SHlo (m*n) → SHlo (m*n)
  -- Per-token broadcast bias `+ β` (`β : [n]` shared across rows). Translation —
  -- the input-VJP is the identity (cotangent passthrough). `den` via `rowBiasFlat`.
  | rowBiasF   {m n : Nat} (bName : String) (β : Vec n)         : SHlo (m*n) → SHlo (m*n)
  -- Chapter 7 (EfficientNet, BATCHED): a batch-separable op (conv/depthwise/dense/
  -- GAP/SE) lifted to `N` examples by `batchMap`; `den` is `batchMap N (denOp op)`.
  -- The whole EfficientNet forward graph lives at the batched index `N·(c·h·w)`;
  -- pointwise swish/sigmoid/relu/addV reuse their existing tokens at that index.
  | batchOp {N a b : Nat} (op : BatchableOp a b)               : SHlo (N * a) → SHlo (N * b)
  -- Chapter 7 (EfficientNet, BATCHED): TRUE batch-norm — reduce μ/var over the
  -- batch+spatial axes [0,2,3] per channel (NOT per-example). The one op that
  -- couples the batch; `den` is `bnBatchLA` (= the proven `bnBatchTensor4`,
  -- conjugated to the network's left-assoc `N·(oc·h·w)` flat index).
  | bnBatchF {N oc h w : Nat} (gName bName epsStr : String) (ε : ℝ) (γ β : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (oc * h * w))
  -- True batch-norm BACKWARD (VJP), `[N,C,H,W]` layout: the renderable three-term
  -- `bnBatchTensor4GradInput` (reduce over [0,2,3] per channel). `den` is the
  -- proven `bnBatchTensor4` VJP backward (batch-coupled). Routes through the
  -- generic `batched` Raw/Tok tag like the forward batched ops.
  | bnBatchBack {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (N * (oc * (h * w)))) :
      SHlo (N * (oc * (h * w))) → SHlo (N * (oc * (h * w)))
  -- Batched conv input-VJP: `batchMap N` of the proven per-example conv
  -- input-grad (activation-independent — conv is linear). Routes through the
  -- generic `batched` tag like the forward batched ops.
  | convBackBatched {N ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * h * w))
  -- Batched STRIDE-2 conv input-VJP: `batchMap N` of the proven per-example
  -- strided-conv input-grad (`flatConvStride2HasVJP` — activation-independent,
  -- strided conv = `decimate ∘ conv` is linear). The downsample basic-block's
  -- stride-2 conv1 backward; halves spatial vs `convBackBatched`. Routes through
  -- the generic `batched` tag like the stride-1 batched ops.
  | convStridedBackBatched {N ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * (2 * h) * (2 * w)))
  -- ⭐ The **bf16** input-VJP peers. These are where the money is: the backward is ~60% of the
  -- conv step (measured on R34's own layer shapes, `planning/archive/bf16_renderer.md`), and unlike JAX
  -- — which autodiffs the backward FROM the cast forward and so inherits bf16 for free — every
  -- hand-written VJP here needs its own bf16 twin. dgrad is itself a convolution, so it takes
  -- the same emit shape and the same `den` discipline as the forward.
  | convBackBatchedBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * h * w))
  -- fp8 (E4M3) peer. Same denotation shape as the bf16 one — `rnd` is the whole difference.
  | convBackBatchedF8 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * h * w))
  | convStridedBackBatchedBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * (2 * h) * (2 * w)))
  -- Batched depthwise input-VJP: `batchMap N` of the proven per-example
  -- depthwise input-grad (activation-independent — depthwise conv is linear).
  | depthwiseBackBatched {N c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * h * w))
  -- ⭐ bf16 depthwise input-VJP. dgrad is itself a (grouped) convolution, so it takes the same
  -- emit shape and the same outer `rnd` (the bf16 store) as the forward.
  | depthwiseBackBatchedBf16 {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * h * w))
  -- Batched STRIDE-2 depthwise input-VJP: `batchMap N` of the proven per-example
  -- strided-depthwise input-grad (`depthwiseStride2FlatHasVJP` — activation-
  -- independent, strided depthwise = `decimate ∘ depthwise` is linear). The
  -- EfficientNet downsample MBConv's stride-2 depthwise backward; halves spatial
  -- vs `depthwiseBackBatched` (the depthwise analog of `convStridedBackBatched`).
  -- Routes through the generic `batched` tag like the stride-1 batched ops.
  | depthwiseStridedBackBatched {N c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w)))
  -- ⭐ Its bf16 peer. ⚠ SYMMETRIC pad, unlike the `Xla` twin's `[p+1, p-1]`.
  | depthwiseStridedBackBatchedBf16 {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w)))
  -- The XLA-`SAME` peer (`planning/archive/mnv4_verified.md` §3e/§3g). ⚠ The backward must place the SAME
  -- asymmetry the forward did — its `den` scatters onto the ODD positions, so the emitted
  -- transposed-conv padding shifts by one. Pairing an `Xla` forward with the SYMMETRIC backward
  -- above type-checks, trains and descends, and computes a gradient for a different net.
  | depthwiseStridedXlaBackBatched {N c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w)))
  -- ⭐ Its bf16 peer. ⚠⚠ Keeps the `[p+1, p-1]` pad — the OPPOSITE shift from the weight grads,
  -- because the kernel is reversed here. `scripts/gates/xla_pad_op_check.py` caught that once already;
  -- the bf16 twin inherits the answer rather than re-deriving it.
  | depthwiseStridedXlaBackBatchedBf16 {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w)))
  -- True batch-norm backward on the NETWORK layout `N·(oc·h·w)` (what
  -- renderBody's `bnBatch` emits): the `bnBatchTensor4` backward reindex-
  -- conjugated to the left-assoc index (`bnBatchLA_eq_comp`).
  | bnBatchLABack {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (N * (oc * h * w))) :
      SHlo (N * (oc * h * w)) → SHlo (N * (oc * h * w))
  -- Batched squeeze-excite backward: rowwise application of the proven per-example
  -- `seBlockFull` VJP. SE is non-linear, so the backward uses each example's forward
  -- activation `v` (unlike the linear conv/depthwise). `den` references the proven
  -- witness rowwise; renderable emission (batchMap-of-SE-subgraph) is deferred.
  | seBackBatched {N c h w r : Nat} (w1Name b1Name w2Name b2Name vName : String)
      (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
      (v : Vec (N * (c * h * w))) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * h * w))
  -- Batched SE GATE COTANGENT: `dgate[n,c] = Σ_{h,w} x[n,c,h,w]·dy[n,c,h,w]` — the
  -- broadcast-adjoint of the Hadamard `x ⊙ dy`, i.e. the FIRST step of the SE gate
  -- backward (the cotangent at the gate's sigmoid output). The un-fused-SE param-grad
  -- ENTRY POINT: feeds `sigmoidBack → denseWeightSgdB(W₂)/denseBiasSgdB + denseRowBack(W₂)
  -- → swishBack → denseWeightSgdB(W₁)/denseBiasSgdB`, exposing the SE dense param grads the
  -- fused `seBackBatched` (input-cotangent only) cannot. `x` = the SE input (saved by name),
  -- `e` = the SE-output cotangent. `den` = batched `broadcastFlatHasVJP.backward (x⊙dy)`.
  | seReduceB {N c h w : Nat} (xName : String) (x : Vec (N * (c * h * w))) :
      SHlo (N * (c * h * w)) → SHlo (N * c)
  -- Batched GLOBAL-AVERAGE-POOL backward (VJP): `dx[n,c,h,w] = dgap[n,c]/(h·w)` — the
  -- per-example `globalAvgPoolFlatHasVJP` backward (broadcast over spatial, ÷h·w),
  -- lifted by `batchMap N`. The head's GAP backward (`gapBack` is per-example, not a
  -- `BatchableOp`, so it needs its own batched ctor). `den` = `batchMap N (gap-adjoint)`.
  | gapBackBatched {N c h w : Nat} : SHlo (N * c) → SHlo (N * (c * h * w))
  -- Chapter 7 (EfficientNet, BATCHED) param-SGD tail: the fused per-channel BN
  -- γ/β updates over the network layout `N·(oc·(h·w))`. `den` is the per-channel BN
  -- grad at the merged batch+spatial axis `m = N·(h·w)` (via `bnchwFwd`, the
  -- network→oc-major reindex), so it is *exactly* `enet_render_bn{gamma,beta}_certified`'s
  -- LHS — the §1 fold is a one-line delegation. Emit recomputes x̂ from the saved BN
  -- input `vName` then `reduce[0,2,3]` (the dγ/dβ in `bnBatchBack`). Output is `Vec oc`.
  | bnGammaSgdB {N oc h w : Nat} (gName vName epsStr lrStr : String) (ε : ℝ) (γ : Vec oc)
      (v : Vec (N * (oc * (h * w)))) (lr : ℝ)             : SHlo (N * (oc * (h * w))) → SHlo oc
  | bnBetaSgdB  {N oc h w : Nat} (bName lrStr : String) (β : Vec oc) (lr : ℝ)
                                                          : SHlo (N * (oc * (h * w))) → SHlo oc
  -- Batched dense weight/bias SGD (SE squeeze/excite convs as `dot_general`, head dense).
  -- `den` = θ − lr·(Σ_n per-example grad on `batchSlice n`); the shared-weight batch sum.
  -- Emit reuses the `weightSgd`/`biasSgd` text (already batch-contracts over `B`).
  | denseWeightSgdB {N a c : Nat} (xName wName lrStr : String) (x : Vec (N * a)) (W : Mat a c) (lr : ℝ)
                                                          : SHlo (N * c) → SHlo (a * c)
  | denseBiasSgdB   {N c : Nat} (bName lrStr : String) (b : Vec c) (lr : ℝ)
                                                          : SHlo (N * c) → SHlo c
  -- ══ The `*SgdB` family with the SGD tail cut off — the BATCHED peers of §2a's eight
  --    per-example `*Grad` ops. Same reason: every `*SgdB` computes a gradient and immediately
  --    spends it on `θ − lr·g`, and AdamW needs the gradient itself three times over (θ', m',
  --    v'). `den (xSgdB …) = θ − lr · den (xGradB …)` is `rfl` — the `*SgdB_eq_grad` theorems.
  --    Output is PARAM-shaped (unbatched), exactly like the fused ops: the Σ over the batch
  --    lives in the emitter, and each emit is a byte-PREFIX of its `*SgdB` peer's. ══
  | convWeightGradB {N ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  | convStridedWeightGradB {N ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- ⭐ The **bf16** weight-grad peers. wgrad is the transpose-trick convolution (batch as the
  -- contraction dim), so the whole `Σ_n` is ONE emitted convolution and therefore ONE bf16
  -- store — which is why the outer `rnd` sits outside the sum, not inside it.
  | convWeightGradBBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- fp8 (E4M3) peer of the batched conv weight-gradient.
  | convWeightGradBF8 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  | convStridedWeightGradBBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- Stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so both bias grads `skel` to ONE Raw and share
  -- an emit case — the same aliasing `convStridedBiasSgd`/`convBiasSgd` already use.
  | convBiasGradB {N ic oc h w kH kW : Nat}
      (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * h * w))) (b : Vec oc)
                                                          : SHlo (N * (oc * h * w)) → SHlo oc
  | convStridedBiasGradB {N ic oc h w kH kW : Nat}
      (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2*h) * (2*w)))) (b : Vec oc)
                                                          : SHlo (N * (oc * h * w)) → SHlo oc
  -- The XLA-`SAME` weight/bias peers. ⭐ The BIAS one needs no emitter and no new Raw: `∂y/∂b = 1`
  -- at every output position regardless of which input taps fed it, so the bias gradient is
  -- `Σ_{batch,spatial} dy` — padding-independent for exactly the reason it is already
  -- stride-independent. Only `den` changes. The WEIGHT one does need its own emit: the weight
  -- gradient contracts dy against the input windows, and those windows moved.
  | convStridedXlaWeightGradB {N ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- ⭐ bf16 peer — MobileNetV2's stem weight-grad.
  | convStridedXlaWeightGradBBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  | convStridedXlaBiasGradB {N ic oc h w kH kW : Nat}
      (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2*h) * (2*w)))) (b : Vec oc)
                                                          : SHlo (N * (oc * h * w)) → SHlo oc
  | bnGammaGradB {N oc h w : Nat} (vName epsStr : String) (ε : ℝ)
      (v : Vec (N * (oc * (h * w))))                      : SHlo (N * (oc * (h * w))) → SHlo oc
  | bnBetaGradB  {N oc h w : Nat}                         : SHlo (N * (oc * (h * w))) → SHlo oc
  | denseWeightGradB {N a c : Nat} (xName : String) (x : Vec (N * a))
                                                          : SHlo (N * c) → SHlo (a * c)
  | denseBiasGradB   {N c : Nat}                          : SHlo (N * c) → SHlo c
  -- ══ BN running statistics: the batch μ and var a batch-BN train step must hand back so the
  --    host can EMA them into the eval forward's frozen stats. `bnBatchF` is ONE node and does
  --    not surface its internal μ/var — a hand-written emitter can reach into its own fragment
  --    for them (that is what `tests/TestResnet34Train.lean` does with `%{p}smr`/`%{p}vsr`), but
  --    `pretty`'s intermediates are counter-named and not addressable. So they are their own ops,
  --    self-contained recomputes from the BN input, like every batched backward here.
  --    `den` is the SAME `bnMean`/`bnVar` that `bnBatchTensor4` normalises by — via `bnchwFwd`,
  --    the `[N,C,H,W] → [C, N·H·W]` reindex — so the returned stats are by construction the
  --    statistics the forward used, not a separately-derived approximation of them. ══
  | bnBatchMeanB {N oc h w : Nat}                         : SHlo (N * (oc * (h * w))) → SHlo oc
  | bnBatchVarB  {N oc h w : Nat}                         : SHlo (N * (oc * (h * w))) → SHlo oc
  -- ══ ⭐⭐ The SYNC-BN statistics, exchanged in TWO rounds — Chan's parallel variance.
  --    Round 1: `bnBatchMeanB` (the replica's μ_r) → `allReduceMeanF` → the global μ.
  --    Round 2: `bnBatchVarAtB x μ` = `σ²_r + (μ_r − μ)²` — the replica's TWO-PASS variance about
  --    its own mean, plus its mean's squared offset from the global one → `allReduceMeanF` →
  --    the global σ² EXACTLY (`bnVar_shard_chan`: Σ_{n∈r}(x−μ)² = Σ(x−μ_r)² + N(μ_r−μ)²).
  --    `bnPackB μ σ²` then packs the two `[oc]` values as one `[oc+oc]`, because `SHlo`'s
  --    skeleton language stops at `.batched2` and every consumer below has one operand slot
  --    left — see `planning/global_bn_verified.md` §2b.
  --    ⛔ The first cut (2026-09-21, morning) exchanged `[μ ‖ E[x²]]` in ONE round and formed
  --    `σ² = E[x²] − μ²` on every consumer. In f32 that costs `ε·E[x²]/σ²` per layer — up to
  --    ~30× rounding at R34's activation scales — compounding to 2e-4 over 36 layers, and the
  --    gradient at random init amplifies forward drift ~1000× (`resnet34-syncbn-check`'s
  --    sensitivity probe). One extra `[oc]` collective per BN layer buys a two-pass-quality σ².
  --    ⚠ Indexed `oc+oc`, NOT `2*oc`, so the packing is Mathlib's `Fin.append` and the two
  --    projections are `Fin.append_left`/`Fin.append_right` rather than hand-rolled. ══
  | bnBatchVarAtB {N oc h w : Nat}                        : SHlo (N * (oc * (h * w))) → SHlo oc → SHlo oc
  | bnPackB {oc : Nat}                                    : SHlo oc → SHlo oc → SHlo (oc + oc)
  -- ══ ⭐⭐ SYNCHRONISED BN FORWARD — normalise with the statistics HANDED IN.
  --    `bnBatchF` reduces its own input for μ/σ²; this one reads them off its second operand,
  --    the packed `[μ ‖ σ²]` above. That is how a replica normalises over a global batch it
  --    cannot see, and it is the whole of the sync-BN forward — no replica-family BN node, just
  --    the collective composed. `den` feeds the ℝ-level `bnSyncTensor4` (stated at μ and the
  --    second moment) `m2 := σ² + μ²`, so its `m2 − μ²` is σ² in ℝ; the emit uses σ² directly.
  --    `den` = `bnSyncTensor4`, whose `R = 1` anchor (`bnSyncTensor4_at_own_stats`) says that
  --    handed the batch's OWN statistics it IS `bnBatchTensor4` — so the single-device render
  --    denotes the function the existing tiers are already tied to. ══
  | bnSyncF {N oc h w : Nat} (gName bName epsStr : String) (ε : ℝ) (γ β : Vec oc) :
      SHlo (N * (oc * (h * w))) → SHlo (oc + oc) → SHlo (N * (oc * (h * w)))
  -- ══ ⭐⭐ The sync BACKWARD's statistic pair, and the backward itself.
  --    `bnSyncDyStatsB` returns `[μ ‖ σ² ‖ mdy ‖ mdyx]`: it PASSES ITS OPERAND THROUGH into the
  --    low half and appends the two dy-reductions. Re-averaging μ/σ² over replicas is the
  --    identity (they are already global), so that pass-through is free — and it buys the
  --    second collective: ONE `allReduceMeanF` then carries everything `bnSyncBack` needs.
  --    ⚠ Both reductions are MEANS, not sums. `bnGradInput` is
  --    `istd·(dx̂ − mean(dx̂) − x̂·mean(x̂·dx̂))`, and a mean over equal shards is the mean of the
  --    shards' means (`bnMean_shard`) — which is exactly why a plain mean-collective suffices.
  --    ⚠ `x` rides as a host literal (`xName` + `Vec`), as in `bnBatchBack`: it is the saved
  --    forward activation, not a graph value, so it costs no operand. ══
  | bnSyncDyStatsB {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (N * (oc * (h * w)))) :
      SHlo (N * (oc * (h * w))) → SHlo (oc + oc) → SHlo (oc + oc + (oc + oc))
  | bnSyncBack {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (N * (oc * (h * w)))) :
      SHlo (N * (oc * (h * w))) → SHlo (oc + oc + (oc + oc)) → SHlo (N * (oc * (h * w)))
  -- ══ ⭐⭐ The sync γ GRADIENT — `bnGammaGradB` with `x̂` at the HANDED-IN statistics.
  --    `bnGammaGradB` recomputes μ/σ² from its own operand (`reduce … [0,2,3]` over `B·h·w`),
  --    so under sync-BN it would build `x̂` from the SHARD's statistics while the forward used
  --    the global ones — a different function, and the wrong gradient. This one slices μ/σ² out
  --    of the same packed `[oc+oc]` operand the forward read, so its `x̂` is the forward's.
  --    ⚠ β's gradient is `Σ dy`, reads no statistic, and `bnBetaGradB` stays as it is.
  --    `den` is `bnSyncPerChannelGradGamma`; its `R = 1` anchor
  --    (`bnSyncPerChannelGradGamma_at_own_stats`) is `bnPerChannelGradGamma`. ══
  | bnSyncGammaGradB {N oc h w : Nat} (xName epsStr : String) (ε : ℝ)
      (x : Vec (N * (oc * (h * w)))) :
      SHlo (N * (oc * (h * w))) → SHlo (oc + oc) → SHlo oc
  -- ══ The running statistics a SYNC render hands back: μ and σ² read off the packed `[μ ‖ σ²]`
  --    vector, so that under DP the host EMAs the GLOBAL batch statistics rather than replica
  --    0's shard's (`bnBatchMeanB`/`bnBatchVarB` reduce the replica's own input). At `R = 1`
  --    both are the existing `bnBatchMeanB`/`bnBatchVarB` (`den_bnStatsMeanB_allReduce_R1`,
  --    `…VarB…`). ══
  | bnStatsMeanB {oc : Nat}                               : SHlo (oc + oc) → SHlo oc
  | bnStatsVarB  {oc : Nat}                               : SHlo (oc + oc) → SHlo oc
  -- ══ Pointwise affine-by-a-LITERAL at the batched index — the pieces a label-smoothed
  --    softmax-CE cotangent is composed from. `scaleB` is `scaleF`'s batched peer; `shiftB`
  --    and `divConstB` had no per-example peer at all.
  --    `divConstB` emits a real `divide` rather than `scaleB (1/c)` ON PURPOSE: the caller
  --    divides by the batch, and `1/B` is only exact in binary32 when `B` is a power of two.
  --    At the bs192/bs256 renders §2d wants, `x * (1/192) ≠ x / 192`. ══
  | scaleB    {N n : Nat} (sStr : String) (s : ℝ)         : SHlo (N*n) → SHlo (N*n)
  | shiftB    {N n : Nat} (sStr : String) (s : ℝ)         : SHlo (N*n) → SHlo (N*n)
  | divConstB {N n : Nat} (sStr : String) (s : ℝ)         : SHlo (N*n) → SHlo (N*n)
  -- ⭐⭐ **4d piece 2 — the cross-replica gradient MEAN as an AST node.** `R` graphs of ONE
  --    skeleton — the same program on `R` replicas, each with its own values, which is what SPMD
  --    data parallelism IS — reduced by `all_reduce(add)` and divided by `R`. `den` is
  --    `(1/R) Σ_r den (g r)` (`DataParallel.dpMean` of the per-replica denotations); `skel` and
  --    therefore `pretty` read replica 0, which is honest because `skel` erases the values the ops
  --    carry (`DataParallel.skel_allReduceMeanF_of_spmd`). Until 2026-09-07 this was
  --    `ViTRender.emitGradAllReduce`, emitted TEXT outside the AST and a declared carve-out in
  --    every train-step tie; the token's emit is that function's text verbatim, and the
  --    `%arsum{t}` / `%armean{t}` names come from `t` rather than `fresh`, so every committed
  --    `*dp*` artifact re-renders byte-identically off the node. `ds` is the parameter's shape,
  --    which is what the text types the tensor as (the optimizer-tail ops carry `ds` the same way,
  --    unlinked to the index). ⚠ `hR`: `skel` needs replica 0 to exist, and a mean over zero
  --    replicas is not a thing. At `R = 1` the emit is empty and the operand's name is threaded
  --    through, exactly as the text function did.
  | allReduceMeanF {n : Nat} (R : Nat) (hR : 0 < R) (t : String) (ds : List Nat)
      (g : Fin R → SHlo n) : SHlo n
  -- ViT per-token (rowwise) dense W/b SGD — the `denseRowF` partners. SAME `den` as
  -- `denseWeightSgdB`/`denseBiasSgdB` (Σ over the N rows), but a 3D `[B,N,·]` token-matrix EMIT
  -- (the weight grad contracts batch×tokens `[0,1]x[0,1]`; the bias reduces `[0,1]`), vs the enet
  -- ops' 2D `[B,·]` batch-only emit. Used for attn Wq/Wk/Wv/Wo + MLP Wfc1/Wfc2 + classifier Wc + the
  -- per-block biases + the vector-LN β (and patch bias). Emit via the generic `.batched` path.
  | rowDenseWeightSgd {N a c : Nat} (xName wName lrStr : String) (x : Vec (N * a)) (W : Mat a c) (lr : ℝ)
                                                          : SHlo (N * c) → SHlo (a * c)
  | rowDenseBiasSgd   {N c : Nat} (bName lrStr : String) (b : Vec c) (lr : ℝ)
                                                          : SHlo (N * c) → SHlo c
  -- Batched conv weight SGD (1×1 expand/project/head; the transpose-trick wgrad).
  -- `den` = flatten W − lr·(Σ_n per-example `conv2dWeightGrad` on `batchSlice n`).
  | convWeightSgdB {N ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- Batched STEM 3×3-strided conv weight + DEPTHWISE weight (stride 1/2) SGD. Same
  -- Σ_n shared-weight batch sum; the depthwise grad is HasVJP3 (flatten-bridged).
  | convStridedWeightSgdB {N ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- The XLA-`SAME` peer, for EfficientNet's SGD path (its Adam path uses `*WeightGradB`).
  | convStridedXlaWeightSgdB {N ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  | depthwiseWeightSgdB {N c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                          : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  | depthwiseStridedWeightSgdB {N c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                          : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  -- ══ PARAM GRADIENTS, un-fused from the update ══
  -- Every `*Sgd` op above computes a gradient and immediately spends it on `θ − lr·g`. That
  -- fusion is why the optimizer could never leave the trusted string emitter: Adam needs the
  -- gradient itself, three times over (θ', m', v'). These are the same gradients with the SGD
  -- tail cut off — `den (xSgd …) = θ − lr · den (xGrad …)` is `rfl`, see the `_sgd_eq` theorems.
  -- Output is PARAM-shaped (unbatched), like the `*Sgd` ops: the batch sum lives in the emitter.
  | weightGrad {m n : Nat} (xName : String) (x : Vec m)         : SHlo n → SHlo (m*n)
  | biasGrad   {n : Nat}                                        : SHlo n → SHlo n
  | convWeightGrad {ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW)
                                                     : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convBiasGrad   {ic oc h w kH kW : Nat}
      (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) : SHlo (oc*h*w) → SHlo oc
  -- The strided + BN peers, same trimming. `convStridedBiasGrad`'s `skel` aliases
  -- `convBiasGrad`'s Raw for the same reason `convStridedBiasSgd` aliases `convBiasSgd`'s: the
  -- bias grad is stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so the emitted text is identical
  -- and only `den` differs.
  | convStridedWeightGrad {ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Vec (ic*(2*h)*(2*w))) (W : Kernel4 oc ic kH kW)
                                                     : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  -- The STRIDE-4 weight gradient — ConvNeXt's 4×4/s4 patchify stem (`psW`), the last
  -- hand-written weight grad in that render. `flatConvStride4` (forward) and
  -- `flatConvStride4HasVJP` (input) were already proven; this op's `den` is the matching
  -- `flatConvStride4WeightGradHasVJP`, which is two `vjpComp` steps over the stride-1
  -- weight-VJP and the two decimations. Exercised only at 4×4 (nothing else in the kit is
  -- stride-4) and gated numerically by `convnext-adam-tie`, not by an emit-prefix case — there is
  -- no fused `convStride4WeightSgd` peer to be a prefix OF.
  | convStride4WeightGrad {ic oc h w kH kW : Nat} (xName : String)
      (b : Vec oc) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (W : Kernel4 oc ic kH kW)
                                                     : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convStridedBiasGrad   {ic oc h w kH kW : Nat}
      (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w))) (b : Vec oc)
                                                     : SHlo (oc*h*w) → SHlo oc
  | bnGammaGrad {oc h w : Nat} (vName epsStr : String) (ε : ℝ) (v : Vec (oc*h*w))
                                                     : SHlo (oc*h*w) → SHlo oc
  | bnBetaGrad  {oc h w : Nat}                       : SHlo (oc*h*w) → SHlo oc
  -- The TRANSFORMER family, un-fused the same way. §2a did the CNN ops above, which is why the
  -- AdamW scorecard stalled at cifar8 + resnet34: ViT's backward spends every gradient inside a
  -- `*Sgd` op, so there was nothing to hand `adamWParamF`. Each of these is its `*Sgd` peer with
  -- the const-lr / multiply / subtract tail cut off, so the emitted text is a byte PREFIX of the
  -- fused op's (checked in `tests/TestBatchedEmitTie.lean`) and `den` differs by exactly
  -- `θ − lr · ·` (`rfl`, the `*Sgd_eq_grad` theorems).
  | rowDenseWeightGrad {N a c : Nat} (xName : String) (x : Vec (N * a))
                                                     : SHlo (N * c) → SHlo (a * c)
  | rowDenseBiasGrad   {N c : Nat}                   : SHlo (N * c) → SHlo c
  | veclnGammaGrad {N D : Nat} (xName epsStr : String) (ε : ℝ) (x : Vec (N*D))
                                                     : SHlo (N*D) → SHlo D
  | patchEmbedWeightGrad {ic H W P N D : Nat} (xName : String) (x : Vec (ic*H*W))
                                                     : SHlo ((N+1)*D) → SHlo (D*ic*P*P)
  | patchEmbedBiasGrad {N c : Nat}                   : SHlo ((N+1)*c) → SHlo c
  -- pos KEEPS the (N+1) token axis (its Jacobian is the identity), so unlike the bias grads this
  -- one is shape-preserving — the same reason `posEmbedSgd` is 2D where the bias updates are 1D.
  | posEmbedGrad {N D : Nat}                         : SHlo ((N+1)*D) → SHlo ((N+1)*D)
  -- The DEPTHWISE weight gradients — the last thing between EfficientNet and a certified AdamW
  -- render (its depthwise convs are followed by BN, so the bias is folded and needs no peer).
  | depthwiseWeightGradB {N c h w kH kW : Nat} (xName : String)
      (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  -- ⭐ bf16 depthwise weight-grad. As with `convWeightGradBBf16`, the batch is the emitted
  -- convolution's contraction dim, so the whole `Σ_n` is ONE convolution and therefore ONE bf16
  -- store — the outer `rnd` sits OUTSIDE the sum, not inside it.
  | depthwiseWeightGradBBf16 {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  | depthwiseStridedWeightGradB {N c h w kH kW : Nat} (xName : String)
      (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  -- ⭐ Its bf16 peer. ⚠ SYMMETRIC pad, unlike the `Xla` twin's `[p-1, p+1]`.
  | depthwiseStridedWeightGradBBf16 {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  -- The depthwise BIAS gradients (§2f) — MobileNetV2's, not EfficientNet's: enet's depthwise convs
  -- are followed by BN so the bias is folded, mnv2's are not. Like every bias grad in the kit the
  -- emitted text is `Σ_{batch,spatial} dy` and therefore STRIDE-INDEPENDENT, so both `skel` to the
  -- SAME Raw as ConvNeXt's per-example `depthwiseBiasGrad` — character-identical to `convBiasGrad`
  -- and `bnBetaGrad` too. NO new emitter, no new Raw/Tok/parse case; only `den` differs, exactly as
  -- `convStridedBiasGradB` aliases `convBiasGradB`.
  | depthwiseBiasGradB {N c h w kH kW : Nat}
      (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * h * w))) (b : Vec c)
                                                     : SHlo (N * (c * h * w)) → SHlo c
  | depthwiseStridedBiasGradB {N c h w kH kW : Nat}
      (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * (2 * h) * (2 * w)))) (b : Vec c)
                                                     : SHlo (N * (c * h * w)) → SHlo c
  -- The XLA-`SAME` depthwise weight/bias peers — same split as the conv pair above: the weight
  -- grad needs its own emit (the windows moved), the bias grad does not (`Σ dy`).
  | depthwiseStridedXlaWeightGradB {N c h w kH kW : Nat} (xName : String)
      (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  -- ⭐ Its bf16 peer. ⚠ Keeps the `[p-1, p+1]` weight-grad pad — the opposite direction from the
  -- dgrad above, and that asymmetry is the whole content of the `Xla` variant.
  | depthwiseStridedXlaWeightGradBBf16 {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xName : String)
      (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  | depthwiseStridedXlaBiasGradB {N c h w kH kW : Nat}
      (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * (2 * h) * (2 * w)))) (b : Vec c)
                                                     : SHlo (N * (c * h * w)) → SHlo c
  -- ══ The CONVNEXT family, un-fused the same way — the last five between ConvNeXt-T and a
  --    certified AdamW render (§2f). Three of them are plain reductions; the two depthwise ones
  --    are the PER-EXAMPLE peers of the `*GradB` pair above (ConvNeXt renders at the per-example
  --    index, not `N := B`). `layerScaleChGammaGrad` is the per-channel layer-scale γ, which no
  --    other net in the kit has. ══
  | depthwiseWeightGrad {c h w kH kW : Nat} (xName : String)
      (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW)
                                                     : SHlo (c*h*w) → SHlo (c*kH*kW)
  | depthwiseBiasGrad   {c h w kH kW : Nat}
      (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c) : SHlo (c*h*w) → SHlo c
  -- Scalar LayerNorm γ/β (the `bnF` sites: scalar LN over the whole `n = c·h·w`, `γ β : Vec 1`).
  | lnGammaGrad {n : Nat} (xName epsStr : String) (ε : ℝ) (x : Vec n) : SHlo n → SHlo 1
  | lnBetaGrad  {n : Nat}                                            : SHlo n → SHlo 1
  | layerScaleChGammaGrad {c h w : Nat} (xName : String) (x : Vec (c*h*w)) : SHlo (c*h*w) → SHlo c
  -- ══ ADAM / ADAMW, shape-generic ══
  -- The child expression is the GRADIENT; θ/m/v ride as name+value fields exactly as the
  -- `*Sgd` ops carry their param. Three ops because `SHlo` is single-result while one AdamW
  -- step produces `(θ', m', v')` — the triple `Proofs.adamWStep` returns. `ds` is the param
  -- shape used only to type the emitted ops; its product must be `n` (a render-level
  -- obligation, like every `xName`/`x` pairing here). Scalar hyperparameters arrive as
  -- `tensor<f32>` function args, so the graph is re-usable across a schedule without re-render.
  | adamMNextF {n : Nat} (mName b1Name ob1Name : String) (ds : List Nat)
      (β₁ : ℝ) (m : Vec n)                                      : SHlo n → SHlo n
  | adamVNextF {n : Nat} (vName b2Name ob2Name : String) (ds : List Nat)
      (β₂ : ℝ) (v : Vec n)                                      : SHlo n → SHlo n
  | adamWParamF {n : Nat}
      (θName mName vName b1Name ob1Name b2Name ob2Name bc1Name bc2Name
        lrName epsName wdName : String) (ds : List Nat)
      (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n)               : SHlo n → SHlo n
  -- ── The SGD / Nesterov peers (§2i). Same shape as the AdamW triple: each carries the emitted
  --    NAME of every runtime `tensor<f32>` argument alongside the ℝ value `den` uses, which is how
  --    a SCHEDULED optimizer works at all — the fused `*Sgd` family bakes `lr` as a literal, and
  --    that fusion (not the optimizer) was the blocker, exactly as §2a found for Adam.
  | sgdParamF {n : Nat} (θName lrName : String) (ds : List Nat)
      (lr : ℝ) (θ : Vec n)                                      : SHlo n → SHlo n
  | momVNextF {n : Nat} (vName muName : String) (ds : List Nat)
      (μ : ℝ) (v : Vec n)                                       : SHlo n → SHlo n
  | momParamF {n : Nat} (θName vName muName lrName : String) (ds : List Nat)
      (μ lr : ℝ) (θ v : Vec n)                                  : SHlo n → SHlo n
  -- ── RMSProp with momentum (`RmsPropStep.lean`), the optimizer the MobileNetV2 and
  --    EfficientNet ImageNet references use. Only ONE op is new: the mean-square slot is
  --    `adamVNextF` at `β₂ := ρ` (`rmsSqNext_eq_adamVNext`, by `rfl`), the coupled-L2 gradient is
  --    `momVNextF` at `(μ := wd, v := θ)` (`momVNext_as_coupled_l2`), and the parameter update is
  --    `sgdParamF` applied to this op's output. ⚠ TENSORFLOW's placement — ε goes INSIDE the
  --    square root; the textbook `g/(√s' + ε)` is a DIFFERENT optimizer (see the ε-placement
  --    theorem). `sqName`/`bufName` ride as name+value like every other optimizer op here.
  | rmsBufNextF {n : Nat} (sqName bufName rhoName orhoName muName epsName : String)
      (ds : List Nat) (ρ μ ε : ℝ) (sq buf : Vec n)              : SHlo n → SHlo n
  -- ── ▶ GLOBAL-NORM GRADIENT CLIPPING (`GradClip.lean`, `planning/archive/grad_clip.md`), the ViT /
  --    ConvNeXt recipe's `gradClipNorm`. FOUR ops, all in this `ds : List Nat` parameter-shape
  --    family rather than the `n : Nat` batched-activation one — the distinction matters, because
  --    `addV` at `n = 1` emits `tensor<Bx1xf32>` and cannot fold a rank-0 scalar.
  --
  --    ⚠⚠ THE NORM IS GLOBAL: one scalar folded from every parameter's gradient and consumed by
  --    every site. That reads like a shared DAG node where `SHlo` is a tree, and it is not one:
  --    `SHlo` is single-OUTPUT, not single-INPUT (`sub`/`addV`/`matmulF` are already binary), and
  --    every gradient the fold consumes is ALREADY an `.operand` leaf, so the 200-way fold is an
  --    ordinary tree with 200 leaves and nothing is recomputed. No carve-out is needed.
  --
  --    ⚠ `clipScaleF` takes the factor as a CHILD, not as a `facName`+ℝ field pair (the `%lr`
  --    shape). As a child its `den` is exactly `factor · g` with no ℝ of its own to disagree with
  --    the norm. It costs nothing in the emit because the renderer hands it an `.operand` leaf at
  --    the norm tree's SSA name — `pretty` prints nothing for a leaf. ⚠ Handing it the norm
  --    SUBTREE instead would emit ~80,000 lines: `pretty` has no CSE (§4 of the handoff), so the
  --    tree would be duplicated at all 200 sites. Emit once, thread the name.
  --
  --    ⚠⚠ TWO ops, and the earlier four-op split (a separate `addScalarF : SHlo 1 → SHlo 1 →
  --    SHlo 1` and `gradClipFacF : SHlo 1 → SHlo 1`) was RETRACTED for a reason worth knowing
  --    before adding any op here: **a constructor with NO `{n : Nat}` binder is a shape this AST
  --    does not otherwise have**, and adding two of them made NINE unrelated `simp only [… den …]`
  --    proofs elsewhere in this file die with a `whnf` timeout — `den` is a ~200-case dependent
  --    match, and fully-index-fixed arms make unfolding it markedly more expensive. **4× the
  --    heartbeat budget did not fix it.** Both ops below are parametric in `n`, like every other
  --    constructor here. See `planning/archive/grad_clip.md` §3.
  | gradSumSqAccF {n : Nat} (ds : List Nat)                     : SHlo 1 → SHlo n → SHlo 1
  | clipScaleF   {n : Nat} (clipStr epsStr : String) (c ε : ℝ)
      (ds : List Nat)                                           : SHlo 1 → SHlo n → SHlo n
  -- ══ LAMB (`Proofs.Lamb`), RSB-A3's optimizer. TWO ops, and `gradSumSqAccF` above is the third
  --    it needs — already here for the clip, and deliberately reused: the per-leaf squared norm is
  --    one quantity and writing a second one is the double-writer failure.
  --
  --    ⚠ BOTH mirror a shape this AST already has. `lambDirF` is `adamWParamF`'s signature minus
  --    `%lr` (same fields, same single tensor child); `lambScaleF` is `clipScaleF`'s exactly
  --    (`SHlo 1 → SHlo n → SHlo n`). That is not tidiness — the retraction note above records that
  --    introducing an unfamiliar constructor SHAPE killed nine unrelated `simp only [… den …]`
  --    proofs with a `whnf` timeout that 4× the heartbeat budget did not fix.
  --
  --    ⚠⚠ `lambScaleF` takes only `‖θ‖²` as its scalar child and recomputes `‖r‖²` from its own
  --    tensor child. Taking both norms as children would make it the kit's first TERNARY
  --    constructor; `r` is already there, so the recomputation is free and the shape stays known.
  | lambDirF   {n : Nat}
      (θName mName vName b1Name ob1Name b2Name ob2Name bc1Name bc2Name
        epsName wdName : String) (ds : List Nat)
      (β₁ β₂ ε wd bc₁ bc₂ : ℝ) (θ m v : Vec n)                  : SHlo n → SHlo n
  | lambScaleF {n : Nat} (ds : List Nat)                        : SHlo 1 → SHlo n → SHlo n

-- Total argmax-routing max-pool backward (the `select_and_scatter` formula),
-- matching `maxPool2HasVJPAt3.backward` lifted through the flatten bridge.
-- Total in the saved input `xv` (the no-ties proof lives only in `.correct`).
noncomputable def maxPoolBackFlat (c h w : Nat)
    (xv : Vec (c*(2*h)*(2*w))) (dyv : Vec (c*h*w)) : Vec (c*(2*h)*(2*w)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    if MaxPool2IsArgmax (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w)) q.1 q.2 p.2
    then (Tensor3.unflatten dyv : Tensor3 c h w) q.1 (winRow q.2) (winCol p.2) else 0

/-- **3×3/s2 max-pool backward (flattened)** — the peer of `maxPoolBackFlat` at He et al.'s stem
    pool, matching `maxPool3s2HasVJPAt3.backward` lifted through `HasVJPAt3.toHasVJPAt`. Total
    in the saved input `xv` (the no-ties proof lives only in `.correct`).

    ⚠⚠ **This is a SUM where the 2×2 peer is a lookup, and that is the whole difference between the
    two pools.** `maxPool2`'s windows tile, so each input is the argmax of at most one output and
    the backward can name it directly. 3×3/s2 windows OVERLAP, so an input can be the argmax of up
    to four outputs (`win3Row_mem_le_two` squared) and the cotangent must ACCUMULATE. Nothing in
    `HasVJPAt3.correct` had to change for that — it already states the backward as a sum over all
    outputs, and `maxPool2`'s peer merely *collapses* it using disjointness. -/
noncomputable def maxPool3s2BackFlat (c h w : Nat)
    (xv : Vec (c*(2*h)*(2*w))) (dyv : Vec (c*h*w)) : Vec (c*(2*h)*(2*w)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      (if maxPool3s2LocalReindex (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w))
              (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = finProdFinEquiv (finProdFinEquiv (q.1, q.2), p.2)
        then (1 : ℝ) else 0) * (Tensor3.unflatten dyv : Tensor3 c h w) co ho wo

/-- **Row-softmax (flattened)** — apply the 1-D `softmax` (MLP.lean) to each of
    the `m` rows of the row-major `Vec (m*n)`. Definitionally equal to
    `Mat.flatten ∘ rowSoftmax ∘ Mat.unflatten` (Attention.lean's `rowSoftmax`; the
    tie is `rowSoftmaxFlat_flat` in ViTFwdGraph). -/
noncomputable def rowSoftmaxFlat (m n : Nat) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => softmax n ((Mat.unflatten v) i))

/-- **Row-softmax backward (flattened)** — per row, the proven closed form
    `pᵢ⊙(dyᵢ − ⟨pᵢ,dyᵢ⟩)` with `pᵢ = softmax(preActᵢ)`. Definitionally equal to
    `Mat.flatten ∘ rowSoftmaxHasVJPMat.backward (Mat.unflatten preAct) ∘ Mat.unflatten`
    (since `softmaxHasVJP.backward z dy i = let p := softmax z; p i·(dy i − ⟨p,dy⟩)`). -/
noncomputable def rowSoftmaxBackFlat (m n : Nat) (preAct dy : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i =>
    let p := softmax n ((Mat.unflatten preAct) i)
    let dyi := (Mat.unflatten dy) i
    let s := ∑ j, p j * dyi j
    fun c => p c * (dyi c - s))

-- ── Chapter 9 (ViT) den helpers — flattened matrix/row-wise forms (the `rfl` ties
--    to the Attention.lean forms live in ViTFwdGraph). ──

/-- **Flattened matrix multiply** `C = A·B` on row-major flat operands.
    Definitionally `Mat.flatten ∘ Mat.mul ∘ Mat.unflatten²`. -/
noncomputable def matMulFlat (m k n : Nat) (a : Vec (m*k)) (b : Vec (k*n)) : Vec (m*n) :=
  Mat.flatten (Mat.mul (Mat.unflatten a) (Mat.unflatten b))

/-- **Flattened transpose** — `Mat.transpose` conjugated by row-major flattening. -/
noncomputable def transposeFlat (m n : Nat) (v : Vec (m*n)) : Vec (n*m) :=
  Mat.flatten (Mat.transpose (Mat.unflatten v))

/-- **Row-wise LayerNorm (flattened)** — each of the `m` token rows gets the 1-D
    `bnForward` over its `n` features (LayerNorm IS per-example BN:
    `layerNormForward := bnForward` definitionally, LayerNorm.lean). -/
noncomputable def rowLNFlat (m n : Nat) (ε γ β : ℝ) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => bnForward n ε γ β ((Mat.unflatten v) i))

/-- **Row-wise LayerNorm input-VJP (flattened)** — per row the consolidated
    three-term `bnGradInput`, recomputing x̂/istd from the saved pre-LN input. -/
noncomputable def rowLNBackFlat (m n : Nat) (ε γ : ℝ) (x dy : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => bnGradInput n ε γ ((Mat.unflatten x) i) ((Mat.unflatten dy) i))

/-- **Per-token dense (flattened)** — every row of the `[N,a]` flat through the
    same `dense W b`. -/
noncomputable def rowDenseFlat (N a c : Nat) (W : Mat a c) (b : Vec c) (v : Vec (N*a)) :
    Vec (N*c) :=
  Mat.flatten (fun i => dense W b ((Mat.unflatten v) i))

/-- **Per-token dense input-VJP (flattened)** — per row `dX = W·dy` (=
    `(denseHasVJP W b).backward`'s `Mat.mulVec W`, MLP.lean). -/
noncomputable def rowDenseBackFlat (N a c : Nat) (W : Mat a c) (dy : Vec (N*c)) :
    Vec (N*a) :=
  Mat.flatten (fun i => Mat.mulVec W ((Mat.unflatten dy) i))

/-- **ViT patch-embedding input-VJP (flattened)** — the proven `patchEmbedInputGradFormula`
    (Attention.lean), i.e. `patchEmbedFlatHasVJP.backward`: the strided patchify conv's input-VJP
    on the patch-token rows of the cotangent. The CLS row and the position-add (a +constant)
    contribute nothing. -/
noncomputable abbrev patchEmbedBackFlat := @patchEmbedInputGradFormula

/-- **ViT patch-embedding weight-grad (flattened)** — `TokenParamGrad`'s `patchEmbedWeightGrad`,
    flattened (that file is downstream of this one; the tie is the §1-fold
    `vit_render_patchW_certified`). The non-overlapping 16×16/s16 patchify conv's weight-VJP:
    `dW_(d,c,kh,kw) = Σ_patches (patch pixel read)·dy_(patch.succ, d)` — token 0 is the CLS row
    (excluded); the pixel read mirrors `patchEmbedFlat`'s, and `dy (finProdFinEquiv (p.succ, d))`
    mirrors `patchEmbedBackFlat`. -/
noncomputable def patchEmbedWeightGradFlat
    (ic H W patchSize N D : Nat)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) :
    Vec (D * ic * patchSize * patchSize) :=
  Kernel4.flatten (fun (d : Fin D) (c : Fin ic) (kh kw : Fin patchSize) =>
    ∑ p : Fin N,
      (let W' := W / patchSize
       let h' := p.val / W'
       let w' := p.val % W'
       let hh := h' * patchSize + kh.val
       let ww := w' * patchSize + kw.val
       if hpad : hh < H ∧ ww < W then
         img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
       else 0)
      * dy (finProdFinEquiv (p.succ, d)))

/-- **ViT patch embedding at bf16 operands (flattened)** — `patchEmbedFlat`'s body with the three
    roundings the emit actually performs, and nothing else.

    ⚠⚠ **The placement of each `rnd` is the whole content of this definition**, so read it against
    the emitted text rather than against the f32 peer:

    * `rnd (W_conv …)` and `rnd (img …)` are the two **operand casts** — the `stablehlo.convert`s
      that make the convolution's inputs `bf16`.
    * the **outer** `rnd` on the patch sum is the **bf16 STORE**: the convolution is emitted with a
      `bf16`-TYPED result, so the hardware accumulates the MAC in f32 and rounds on the way out.
      Dropping it would claim more precision than the hardware delivers — the unsound direction,
      and the trap `planning/archive/bf16_renderer.md` §9.2 exists to name.
    * `b_conv`, `cls_token` and `pos_embed` are added **outside** every rounding, because the emit
      adds them after the convert-back, in f32. They are f32 parameters that never reach a tensor
      core.

    ▶ The CLS row (`n = 0`) carries no convolution at all, so no rounding touches it — which is why
    the `if` is INSIDE the roundings' scope rather than outside it. -/
noncomputable def patchEmbedFlatBf16
    (rnd : ℝ → ℝ)
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Vec (ic * H * W) → Vec ((N + 1) * D) :=
  fun img =>
    fun idx_out =>
      let n := (finProdFinEquiv.symm idx_out).1
      let d := (finProdFinEquiv.symm idx_out).2
      pos_embed n d +
        (if n.val = 0 then
          cls_token d
         else
          b_conv d +
          rnd (∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            rnd (W_conv d c kh kw) *
              rnd (let W' := W / patchSize
                   let p := n.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + kh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : hh < H ∧ ww < W then
                     img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
                   else 0)))

/-- **CLS slice (flattened)** — gather row 0 of the `[N+1,D]` flat (= the proven
    `clsTokenFlat`, Attention.lean; tie is `rfl` in ViTFwdGraph). -/
noncomputable def clsSliceFlat (N D : Nat) (v : Vec ((N+1)*D)) : Vec D :=
  fun k => v (finProdFinEquiv ((0 : Fin (N + 1)), k))

/-- **CLS pad (flattened)** — scatter `dy` to row 0, zeros elsewhere (= the proven
    `clsTokenFlatHasVJP.backward`; tie is `rfl` in ViTFwdGraph). -/
noncomputable def clsPadFlat (N D : Nat) (dy : Vec D) : Vec ((N+1)*D) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    if p.1 = (0 : Fin (N + 1)) then dy p.2 else 0

/-- **Per-head column slice (flattened)** — head `h`'s `[N,d]` block of the
    `[N,heads·d]` flat: the `finProdFinEquiv (h, ·)` column gather `mhsaLayer`
    uses to feed each head's SDPA. -/
noncomputable def headSliceFlat (N heads d : Nat) (h : Fin heads)
    (v : Vec (N*(heads*d))) : Vec (N*d) :=
  Mat.flatten (fun (r : Fin N) (j : Fin d) =>
    (Mat.unflatten v) r (finProdFinEquiv (h, j)))

/-- **Per-head column pad (flattened)** — scatter an `[N,d]` head block into head
    `h`'s columns of a zero `[N,heads·d]`. `mhsaLayer`'s concat is the sum of
    these over heads; it is also `headSliceFlat`'s VJP. -/
noncomputable def headPadFlat (N heads d : Nat) (h : Fin heads)
    (v : Vec (N*d)) : Vec (N*(heads*d)) :=
  Mat.flatten (fun (r : Fin N) (hj : Fin (heads*d)) =>
    let p := finProdFinEquiv.symm hj
    if p.1 = h then (Mat.unflatten v) r p.2 else 0)

/-- **Row-broadcast scale (flattened)** — every token row elementwise-scaled by the
    shared `γ : Vec n` (= rowwise `layerScale γ`). -/
noncomputable def rowScaleFlat (m n : Nat) (γ : Vec n) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun r => layerScale γ ((Mat.unflatten v) r))

/-- **Row-broadcast bias (flattened)** — `+ β` on every token row. -/
noncomputable def rowBiasFlat (m n : Nat) (β : Vec n) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun r k => (Mat.unflatten v) r k + β k)

/-- Channel index of a flat `c·h·w` position (the repo's left-assoc
    `finProdFinEquiv` convention: `k ↔ ((chan, row), col)`). Used to expand a
    per-channel parameter (`Vec c`) to the flat per-element map. -/
def chanIdx (c h w : Nat) (k : Fin (c * h * w)) : Fin c :=
  (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1

/-- **The proven per-example forward of a `BatchableOp`** — exactly the existing
    batch-1 op (`flatConv`/`depthwiseFlat`/`dense`/`globalAvgPoolFlat`/`seBlockFull`/…).
    `SHlo.batchOp`'s `den` is `batchMap N (denOp op)`. -/
noncomputable def denOp : {a b : Nat} → BatchableOp a b → (Vec a → Vec b)
  | _, _, .conv _ _ W bias => flatConv W bias
  | _, _, .convStrided _ _ W bias => flatConvStride2 W bias
  | _, _, .convBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (flatConv (fun o c a d => rnd (W o c a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun o _ _ => bias o) i
  -- ⭐ Byte-identical to `convBf16`'s denotation above, and that is the point: the meaning of a
  -- low-precision op is "round the operands, round the result", which is already parametric in
  -- `rnd`. Passing E4M3 rounding instead of bf16 rounding is the whole semantic difference, so
  -- the accuracy bounds INSTANTIATE rather than needing restatement (`fp8E4M3 : FloatModel` at
  -- u = 2⁻⁴ is already in `Binary32Instance.lean`).
  | _, _, .convF8 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (flatConv (fun o c a d => rnd (W o c a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun o _ _ => bias o) i
  | _, _, .convStridedBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (flatConvStride2 (fun o c a d => rnd (W o c a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun o _ _ => bias o) i
  -- ⚠ The XLA-`SAME` stride-2 conv is `flatConvStride2Xla`, NOT `flatConvStride2`: the two tokens
  -- have identical types and emitted shapes, so this arm is the only place they differ. Getting it
  -- wrong would make the render provably compute one net while emitting the other.
  | _, _, .convStridedXla _ _ W bias => flatConvStride2Xla W bias
  | _, _, .convStridedXlaBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (flatConvStride2Xla (fun o c a d => rnd (W o c a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun o _ _ => bias o) i
  | _, _, .depthwise _ _ W bias => depthwiseFlat W bias
  -- ⚠ The outer `rnd` is the bf16 STORE (the bf16-typed conv result); the inner two are the
  -- operand casts. The bias is added AFTER, at the accumulate precision, exactly as emitted.
  | _, _, .depthwiseBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (depthwiseFlat (fun cc a d => rnd (W cc a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun cc _ _ => bias cc) i
  | _, _, .depthwiseStrided _ _ W bias => depthwiseStride2Flat W bias
  | _, _, .depthwiseStridedBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (depthwiseStride2Flat (fun cc a d => rnd (W cc a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun cc _ _ => bias cc) i
  -- ⚠ `depthwiseStride2FlatXla`, NOT `depthwiseStride2Flat`: same caveat as `.convStridedXla`.
  | _, _, .depthwiseStridedXla _ _ W bias => depthwiseStride2FlatXla W bias
  | _, _, .depthwiseStridedXlaBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (depthwiseStride2FlatXla (fun cc a d => rnd (W cc a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun cc _ _ => bias cc) i
  | _, _, .dense _ _ W bias => dense W bias
  | _, _, .gap (c := c) (h := h) (w := w) => globalAvgPoolFlat c h w
  | _, _, .seBlock (h := h) (w := w) _ _ _ _ W₁ b₁ W₂ b₂ => seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂
  -- Inference BN: each example independently, with the same frozen statistics. Under `batchMap N`
  -- no `N` appears except as the number of applications, so an example's logits cannot depend on
  -- which others share its batch (unlike `bnBatchF`, whose `bnBatchLA` couples the batch).
  | _, _, .bnEval (oc := oc) (h := h) (w := w) _ _ _ _ _ ε γ β μ var =>
      bnPerChannelEvalTensor3 oc h w ε γ β μ var
  | _, _, .swish (n := n) => swish n
  | _, _, .relu (n := n) => relu n
  | _, _, .relu6 (n := n) => relu6 n
  | _, _, .maxPool (c := c) (h := h) (w := w) => maxPoolFlat c h w
  -- ⚠ Same type as `.maxPool`, different function (He et al.'s 3×3/s2 window). The pair share an
  -- arity, op counts, the prefix audit and the emitted shape, which is how the deviation went
  -- undocumented on every ResNet here; this arm and the emitted text are what separate them.
  | _, _, .maxPool3s2 (c := c) (h := h) (w := w) => maxPool3s2Flat c h w
  | _, _, .softmaxRow (m := m) (n := n) => rowSoftmaxFlat m n
  | _, _, .denseRowBack (rows := rows) (a := a) (c := c) _ W => rowDenseBackFlat rows a c W
  -- ⚠ THREE roundings: the two operand casts and the **bf16 STORE**. The emit gives this
  -- `dot_general` a bf16-TYPED result (§20.1 — it is worth 1.18× → 1.60× and §9.2 had only ever
  -- checked that shape for correctness), so the hardware rounds the output too.
  | _, _, .denseRowBackBf16 (rows := rows) (a := a) (c := c) rnd _ W =>
      fun dy i => rnd (rowDenseBackFlat rows a c (fun p q => rnd (W p q)) (fun j => rnd (dy j)) i)
  -- The five ViT/ConvNeXt row/pointwise forms — each denotes the SAME per-example function its
  -- descriptor-less peer does (`.geluF`, `.transposeF`, `.lnRowF`, `.rowScaleF`, `.rowBiasF`),
  -- which is what makes the batched node a `batchMap` of a proven map rather than a new function.
  | _, _, .gelu (n := n) => gelu n
  | _, _, .transpose (m := m) (n := n) => transposeFlat m n
  | _, _, .convStride4 _ _ W bias => flatConvStride4 W bias
  -- ⚠ `convBf16`'s exact shape at the stride-4 forward: the outer `rnd` is the bf16 STORE (the
  -- bf16-typed conv result), the inner two are the operand casts, and the bias is added AFTER at
  -- the accumulate precision — exactly as emitted.
  | _, _, .convStride4Bf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (flatConvStride4 (fun o c a d => rnd (W o c a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun o _ _ => bias o) i
  | _, _, .layerScaleCh (c := c) (h := h) (w := w) _ γ =>
      fun v => layerScale (fun k => γ (chanIdx c h w k)) v
  -- Per-EXAMPLE softmax: the denominator is that example's own sum, which is the whole point of
  -- giving `softmaxDiv` a descriptor (see the constructor's note).
  | _, _, .dotOut _ W => fun v i => ∑ j, W i j * v j
  | _, _, .expe => fun v j => Real.exp (v j)
  | _, _, .softmaxDiv => fun v j => v j / ∑ k, v k
  | _, _, .lnRow (m := m) (n := n) _ _ _ ε γ β => rowLNFlat m n ε γ β
  | _, _, .rowScale (m := m) (n := n) _ γ => rowScaleFlat m n γ
  | _, _, .rowBias (m := m) (n := n) _ β => rowBiasFlat m n β
  -- ViT increment 1. Each denotes the SAME per-example function its descriptor-less peer does
  -- (`.denseRowF`, `.patchEmbedF`, `.clsSliceF`, `.clsPadF`, `.headSliceF`, `.headPadF`), which is
  -- what makes the batched node a `batchMap` of a proven map rather than a new function.
  | _, _, .denseRow (N := N) (a := a) (c := c) _ _ W b => rowDenseFlat N a c W b
  -- ⚠ `convBf16`'s exact shape, one op class over: the outer `rnd` is the bf16 STORE of the
  -- `dot_general`'s bf16-typed result, the inner two are the operand casts, and the BIAS is added
  -- AFTER — outside the rounding, at the accumulate precision — because the emit adds it after the
  -- convert-back. Rounding the bias here, or folding it inside via `rowDenseFlat`'s own `b`
  -- argument, would describe a different graph.
  | _, _, .denseRowBf16 (N := N) (a := a) (c := c) rnd _ _ W b =>
      fun x => rowBiasFlat N c b
        (fun i => rnd (rowDenseFlat N a c (fun p q => rnd (W p q)) (fun _ => 0)
                        (fun j => rnd (x j)) i))
  | _, _, .patchEmbed (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ _ _ Wc bc cls pos =>
      patchEmbedFlat ic H W P N D Wc bc cls pos
  -- ⚠⚠ The rounding placement lives in `patchEmbedFlatBf16`, next to `patchEmbedFlat`, because it
  -- is the one ViT op whose `den` differs from its f32 peer by more than a wrapper — read the
  -- docstring there before trusting this line.
  | _, _, .patchEmbedBf16 (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D)
        rnd _ _ _ _ Wc bc cls pos =>
      patchEmbedFlatBf16 rnd ic H W P N D Wc bc cls pos
  | _, _, .clsSlice (N := N) (D := D) => clsSliceFlat N D
  | _, _, .clsPad (N := N) (D := D) => clsPadFlat N D
  | _, _, .headSlice (N := N) (heads := heads) (d := d) h => headSliceFlat N heads d h
  | _, _, .headPad (N := N) (heads := heads) (d := d) h => headPadFlat N heads d h

/-- Which BatchNorm a batched forward chain emits, for the renders whose one traversal produces
    both the training forward and its frozen-stats eval partner (EfficientNet, MobileNetV4).

    The distinction is not cosmetic and the §2a bug is what it exists to prevent: a `.train` chain
    reduces its statistics out of the activation (`bnBatchF`, which couples the batch), a `.eval`
    chain consumes frozen per-channel running stats as graph inputs (the `bnEval` descriptor, which
    does not). A net trained on one and *scored* with the other is evaluating a different function
    — which is exactly what `resnet34_fwd` did until 2026-07-27, at rel 1.13 on real logits. -/
inductive BnMode where
  /-- **Training**: batch statistics reduced out of the activation (`bnBatchF`, reduce `[0,2,3]`,
      n = B·H·W). What the train step differentiates. -/
  | train
  /-- **Inference**: frozen per-channel running stats arriving as graph inputs `%{p}mu`/`%{p}var`
      (the `bnEval` descriptor). Class-batch-independent. -/
  | eval
deriving DecidableEq, Repr

/-- **AST denotation `⟦·⟧ₐ`** — our reading of each StableHLO op's spec, over
    `ℝ`, per-example, in primitive terms — independent of `dense`/`Mat.mulVec`.
    SSA names are ignored. -/
noncomputable def den : {n : Nat} → SHlo n → Vec n
  | _, .operand _ v    => v
  | _, .dotIn _ W e    => fun j => ∑ i, den e i * W i j
  | _, .dotInBf16 rnd _ W e => fun j => ∑ i, rnd (den e i) * rnd (W i j)
  | _, .dotOut _ W e   => fun i => ∑ j, W i j * den e j
  | _, .addBcast _ b e => fun j => den e j + b j
  | _, .expe e         => fun j => Real.exp (den e j)
  | _, .softmaxDiv e   => fun j => den e j / ∑ k, den e k
  | _, .sub a b        => fun j => den a j - den b j
  | _, .weightSgd _ _ _ x W lr e => Mat.flatten (fun i j => W i j - lr * (x i * den e j))
  | _, .biasSgd _ _ b lr e       => fun j => b j - lr * den e j
  | _, .convWeightSgd _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx
        - lr * (conv2dWeightGradHasVJP b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (conv2dBiasGradHasVJP W x).backward b (den e) o
  -- Param gradients, un-fused (the `*Sgd` bodies above with `θ − lr·` stripped off).
  | _, .weightGrad _ x e     => Mat.flatten (fun i j => x i * den e j)
  | _, .biasGrad e           => den e
  | _, .convWeightGrad _ b x W e =>
      (conv2dWeightGradHasVJP b x).backward (Kernel4.flatten W) (den e)
  | _, .convBiasGrad W x b e => (conv2dBiasGradHasVJP W x).backward b (den e)
  | _, .convStridedWeightGrad _ b x W e =>
      (flatConvStride2WeightGradHasVJP b x).backward (Kernel4.flatten W) (den e)
  | _, .convStride4WeightGrad _ b x W e =>
      (flatConvStride4WeightGradHasVJP b x).backward (Kernel4.flatten W) (den e)
  | _, .convStridedBiasGrad W x b e => (flatConvStride2BiasGradHasVJP W x).backward b (den e)
  | _, .bnGammaGrad (oc := oc) (h := h) (w := w) _ _ ε v e =>
      bnPerChannelGradGamma oc (h*w) ε (reassocFwd oc h w v) (reassocFwd oc h w (den e))
  | _, .bnBetaGrad (oc := oc) (h := h) (w := w) e =>
      bnPerChannelGradBeta oc (h*w) (reassocFwd oc h w (den e))
  -- AdamW: the proven ℝ optimizer (AdamStep.lean) applied to the child's gradient.
  | _, .adamMNextF _ _ _ _ β₁ m e => adamMNext β₁ m (den e)
  | _, .adamVNextF _ _ _ _ β₂ v e => adamVNext β₂ v (den e)
  | _, .adamWParamF _ _ _ _ _ _ _ _ _ _ _ _ _ β₁ β₂ ε lr wd bc₁ bc₂ θ m v e =>
      adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v (den e)
  -- SGD / Nesterov: the proven ℝ optimizers (SgdMomentumStep.lean) on the child's gradient.
  | _, .sgdParamF _ _ _ lr θ e => sgdParam lr θ (den e)
  | _, .momVNextF _ _ _ μ v e => momVNext μ v (den e)
  | _, .momParamF _ _ _ _ _ μ lr θ v e => momParam μ lr θ v (den e)
  -- RMSProp: the proven ℝ optimizer (RmsPropStep.lean) on the child's gradient.
  | _, .rmsBufNextF _ _ _ _ _ _ _ ρ μ ε sq buf e => rmsBufNext ρ μ ε sq buf (den e)
  -- Global-norm gradient clipping (GradClip.lean). `gradSumSqF` collapses one parameter's gradient
  -- to its ∑g² as a rank-0 scalar (`SHlo 1`, the `lnBetaGrad` reading); `addScalarF` folds those
  -- across parameters; `gradClipFacF` roots the total and forms `min(1, c/(√s+ε))`; `clipScaleF`
  -- multiplies a gradient by that factor, which it takes as its FIRST CHILD — so `den` is exactly
  -- `factor · g` and there is no ℝ field here whose agreement with the norm has to be assumed.
  -- ⚠ `scalarOf` rather than `den acc 0`: `den` must never APPLY a recursive call to an index —
  -- every other arm of this match passes `den e` along whole. See `Proofs.scalarOf`.
  | _, .gradSumSqAccF _ acc e      => fun _ => scalarOf (den acc) + gradSumSq (den e)
  | _, .clipScaleF _ _ c ε _ s e   => clipScale (clipFactor c ε (scalarOf (den s))) (den e)
  -- LAMB: the direction from the INCOMING moments and this step's gradient, then the per-tensor
  -- trust scaling. `θ'` itself is `sgdParamF θ lr (lambScaleF …)` — an op that already exists.
  | _, .lambDirF _ _ _ _ _ _ _ _ _ _ _ _ β₁ β₂ ε wd bc₁ bc₂ θ m v e =>
      lambDir β₁ β₂ ε wd bc₁ bc₂ θ m v (den e)
  | _, .lambScaleF _ s e           => lambScale (scalarOf (den s)) (den e)
  | _, .bnGammaSgd (oc := oc) (h := h) (w := w) _ _ _ _ ε γ v lr e =>
      fun c => γ c - lr *
        bnPerChannelGradGamma oc (h*w) ε (reassocFwd oc h w v) (reassocFwd oc h w (den e)) c
  | _, .bnBetaSgd (oc := oc) (h := h) (w := w) _ _ β lr e =>
      fun c => β c - lr * bnPerChannelGradBeta oc (h*w) (reassocFwd oc h w (den e)) c
  | _, .layerScaleChGammaSgd (c := c) (h := h) (w := w) _ _ _ x γ lr e =>
      fun cc => γ cc - lr * ∑ k : Fin (c*h*w), (if chanIdx c h w k = cc then x k * den e k else 0)
  | _, .lnGammaSgd (n := n) _ _ _ _ ε x γ lr e =>
      fun _ => γ 0 - lr * bnGradGamma n ε x (den e)
  | _, .lnBetaSgd (n := n) _ _ β lr e =>
      fun _ => β 0 - lr * bnGradBeta n (den e)
  | _, .veclnGammaSgd (N := N) (D := D) _ _ _ _ ε x γ lr e =>
      fun k => γ k - lr * ∑ r : Fin N,
        Mat.unflatten (den e) r k * layerNormForward D ε 1 0 (Mat.unflatten x r) k
  | _, .patchEmbedWeightSgd (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ _ x Wp lr e =>
      fun idx => Kernel4.flatten Wp idx - lr * patchEmbedWeightGradFlat ic H W P N D x (den e) idx
  | _, .patchEmbedBiasSgd (N := N) (c := c) _ _ b lr e =>
      fun i => b i - lr * ∑ p : Fin N, batchSlice (N+1) c (den e) p.succ i
  | _, .posEmbedSgd (N := N) (D := D) _ _ pos lr e =>
      fun i => Mat.flatten pos i - lr * (den e) i
  | _, .veclnGammaGrad (N := N) (D := D) _ _ ε x e =>
      fun k => ∑ r : Fin N,
        Mat.unflatten (den e) r k * layerNormForward D ε 1 0 (Mat.unflatten x r) k
  | _, .patchEmbedWeightGrad (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ x e =>
      fun idx => patchEmbedWeightGradFlat ic H W P N D x (den e) idx
  | _, .patchEmbedBiasGrad (N := N) (c := c) e =>
      fun i => ∑ p : Fin N, batchSlice (N+1) c (den e) p.succ i
  | _, .posEmbedGrad e => fun i => (den e) i
  | _, .bnGammaSgdB (N := N) (oc := oc) (h := h) (w := w) _ _ _ _ ε γ v lr e =>
      fun c => γ c - lr *
        bnPerChannelGradGamma oc (N*(h*w)) ε (bnchwFwd N oc h w v) (bnchwFwd N oc h w (den e)) c
  | _, .bnBetaSgdB (N := N) (oc := oc) (h := h) (w := w) _ _ β lr e =>
      fun c => β c - lr * bnPerChannelGradBeta oc (N*(h*w)) (bnchwFwd N oc h w (den e)) c
  | _, .denseWeightSgdB (N := N) (a := a) (c := c) _ _ _ x W lr e =>
      Mat.flatten (fun i j => W i j - lr * ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .denseBiasSgdB (N := N) (c := c) _ _ b lr e =>
      fun j => b j - lr * ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .convWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (conv2dWeightGradHasVJP b (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (flatConvStride2WeightGradHasVJP b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (conv2dWeightGradHasVJP b
          (Tensor3.unflatten (fun j => rnd (batchSlice N (ic*h*w) x n j)))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convWeightGradBF8 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (conv2dWeightGradHasVJP b
          (Tensor3.unflatten (fun j => rnd (batchSlice N (ic*h*w) x n j)))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convStridedWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (flatConvStride2WeightGradHasVJP b
          (fun j => rnd (batchSlice N (ic*(2*h)*(2*w)) x n j))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convBiasGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (conv2dBiasGradHasVJP W (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .convStridedBiasGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (flatConvStride2BiasGradHasVJP W (batchSlice N (ic*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  -- The XLA-`SAME` peers. ⚠ Only the CERT changes (`…Xla…`); the shape of the batch sum is
  -- identical, which is precisely why this is easy to get wrong by copy-paste.
  | _, .convStridedXlaWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (flatConvStride2XlaWeightGradHasVJP b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedXlaWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (flatConvStride2XlaWeightGradHasVJP b
          (fun j => rnd (batchSlice N (ic*(2*h)*(2*w)) x n j))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convStridedXlaBiasGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (flatConvStride2XlaBiasGradHasVJP W (batchSlice N (ic*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .bnGammaGradB (N := N) (oc := oc) (h := h) (w := w) _ _ ε v e =>
      fun c =>
        bnPerChannelGradGamma oc (N*(h*w)) ε (bnchwFwd N oc h w v) (bnchwFwd N oc h w (den e)) c
  | _, .bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) e =>
      fun c => bnPerChannelGradBeta oc (N*(h*w)) (bnchwFwd N oc h w (den e)) c
  | _, .denseWeightGradB (N := N) (a := a) (c := c) _ x e =>
      Mat.flatten (fun i j => ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .denseBiasGradB (N := N) (c := c) e =>
      fun j => ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .bnBatchMeanB (N := N) (oc := oc) (h := h) (w := w) e =>
      fun c => bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c)
  | _, .bnBatchVarB (N := N) (oc := oc) (h := h) (w := w) e =>
      fun c => bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c)
  | _, .bnBatchVarAtB (N := N) (oc := oc) (h := h) (w := w) e mu =>
      fun c => bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c)
             + (bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) - den mu c)
             * (bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) - den mu c)
  | _, .bnPackB a b => Fin.append (den a) (den b)
  | _, .bnSyncDyStatsB (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x dy st =>
      Fin.append (den st)
        (Fin.append
          (fun c => bnMean (N*(h*w)) (fun k =>
              γ c * Mat.unflatten (bnchwFwd N oc h w (den dy)) c k))
          (fun c => bnMean (N*(h*w)) (fun k =>
              bnSyncXhat (N*(h*w)) ε (den st (Fin.castAdd oc c))
                (den st (Fin.natAdd oc c) + den st (Fin.castAdd oc c) * den st (Fin.castAdd oc c))
                (Mat.unflatten (bnchwFwd N oc h w x) c) k
              * (γ c * Mat.unflatten (bnchwFwd N oc h w (den dy)) c k))))
  | _, .bnSyncBack (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x dy ds =>
      bnSyncTensor4GradInput N oc h w ε γ
        (fun c => den ds (Fin.castAdd (oc+oc) (Fin.castAdd oc c)))
        (fun c => den ds (Fin.castAdd (oc+oc) (Fin.natAdd  oc c))
                  + den ds (Fin.castAdd (oc+oc) (Fin.castAdd oc c))
                    * den ds (Fin.castAdd (oc+oc) (Fin.castAdd oc c)))
        (fun c => den ds (Fin.natAdd  (oc+oc) (Fin.castAdd oc c)))
        (fun c => den ds (Fin.natAdd  (oc+oc) (Fin.natAdd  oc c)))
        x (den dy)
  | _, .bnSyncF (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ β x st =>
      -- the packed operand's halves, by `Fin.castAdd`/`Fin.natAdd` — the index-level peers of
      -- `Fin.append_left`/`Fin.append_right`, which is what makes this readable back off a
      -- `bnPackB` of two `allReduceMeanF`s. The second half is σ²; `bnSyncTensor4` is stated at
      -- the second moment, so it is handed `σ² + μ²` and its `m2 − μ²` is σ² in ℝ.
      bnSyncTensor4 N oc h w ε γ β
        (fun c => den st (Fin.castAdd oc c))
        (fun c => den st (Fin.natAdd  oc c) + den st (Fin.castAdd oc c) * den st (Fin.castAdd oc c))
        (den x)
  | _, .bnSyncGammaGradB (N := N) (oc := oc) (h := h) (w := w) _ _ ε x dy st =>
      bnSyncPerChannelGradGamma oc (N*(h*w)) ε
        (fun c => den st (Fin.castAdd oc c))
        (fun c => den st (Fin.natAdd  oc c) + den st (Fin.castAdd oc c) * den st (Fin.castAdd oc c))
        (bnchwFwd N oc h w x) (bnchwFwd N oc h w (den dy))
  | _, .bnStatsMeanB (oc := oc) e => fun c => den e (Fin.castAdd oc c)
  | _, .bnStatsVarB  (oc := oc) e => fun c => den e (Fin.natAdd oc c)
  | _, .scaleB _ s e    => fun i => den e i * s
  | _, .shiftB _ s e    => fun i => den e i + s
  | _, .divConstB _ s e => fun i => den e i / s
  | _, .allReduceMeanF R _ _ _ g => fun i => (1 / (R : ℝ)) * ∑ r : Fin R, den (g r) i
  | _, .rowDenseWeightSgd (N := N) (a := a) (c := c) _ _ _ x W lr e =>
      Mat.flatten (fun i j => W i j - lr * ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .rowDenseBiasSgd (N := N) (c := c) _ _ b lr e =>
      fun j => b j - lr * ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .rowDenseWeightGrad (N := N) (a := a) (c := c) _ x e =>
      Mat.flatten (fun i j => ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .rowDenseBiasGrad (N := N) (c := c) e =>
      fun j => ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .convWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (conv2dWeightGradHasVJP b (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (flatConvStride2WeightGradHasVJP b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedXlaWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (flatConvStride2XlaWeightGradHasVJP b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .depthwiseWeightSgdB (N := N) (c := c) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Tensor3.flatten W idx - lr * ∑ n : Fin N,
        Tensor3.flatten ((depthwiseWeightGradHasVJP3 b (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward
          W (Tensor3.unflatten (batchSlice N (c*h*w) (den e) n))) idx
  | _, .depthwiseStridedWeightSgdB (N := N) (c := c) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Tensor3.flatten W idx - lr * ∑ n : Fin N,
        (depthwiseStride2WeightGradHasVJP b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
  | _, .depthwiseWeightGradB (N := N) (c := c) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        Tensor3.flatten ((depthwiseWeightGradHasVJP3 b (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward
          W (Tensor3.unflatten (batchSlice N (c*h*w) (den e) n))) idx
  -- ⚠ `rnd` OUTSIDE the `Σ_n`: the emit makes the batch the convolution's contraction dim, so the
  -- whole sum is ONE convolution and therefore ONE bf16 store. Inside would model N stores.
  | _, .depthwiseWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        Tensor3.flatten ((depthwiseWeightGradHasVJP3 b
          (Tensor3.unflatten (fun j => rnd (batchSlice N (c*h*w) x n j)))).backward
          W (Tensor3.unflatten (fun j => rnd (batchSlice N (c*h*w) (den e) n j)))) idx)
  -- The ConvNeXt five. Each is exactly the gradient half of its `*Sgd` peer's `den`, so the
  -- `*Sgd_eq_grad` theorems below are `rfl`.
  | _, .depthwiseWeightGrad _ b x W e =>
      fun idx => Tensor3.flatten
        ((depthwiseWeightGradHasVJP3 b x).backward W (Tensor3.unflatten (den e))) idx
  | _, .depthwiseBiasGrad W x b e => fun o => (depthwiseBiasGradHasVJP W x).backward b (den e) o
  | _, .lnGammaGrad (n := n) _ _ ε x e => fun _ => bnGradGamma n ε x (den e)
  | _, .lnBetaGrad (n := n) e => fun _ => bnGradBeta n (den e)
  | _, .layerScaleChGammaGrad (c := c) (h := h) (w := w) _ x e =>
      fun cc => ∑ k : Fin (c*h*w), (if chanIdx c h w k = cc then x k * den e k else 0)
  | _, .depthwiseStridedWeightGradB (N := N) (c := c) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (depthwiseStride2WeightGradHasVJP b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
  | _, .depthwiseStridedWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (depthwiseStride2WeightGradHasVJP b
          (fun j => rnd (batchSlice N (c*(2*h)*(2*w)) x n j))).backward
          (Tensor3.flatten W) (fun j => rnd (batchSlice N (c*h*w) (den e) n j)) idx)
  -- The depthwise bias grads: the shared-parameter batch sum every `*GradB` takes, `Σ_n dβ_n`.
  -- Same shape as `convBiasGradB`/`convStridedBiasGradB` one row up, with the depthwise VJP certs.
  | _, .depthwiseBiasGradB (N := N) (c := c) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (depthwiseBiasGradHasVJP W (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward b
          (batchSlice N (c*h*w) (den e) n) o
  | _, .depthwiseStridedBiasGradB (N := N) (c := c) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (depthwiseStride2BiasGradHasVJP W (batchSlice N (c*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (c*h*w) (den e) n) o
  | _, .depthwiseStridedXlaWeightGradB (N := N) (c := c) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (depthwiseStride2XlaWeightGradHasVJP b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
  | _, .depthwiseStridedXlaWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (depthwiseStride2XlaWeightGradHasVJP b
          (fun j => rnd (batchSlice N (c*(2*h)*(2*w)) x n j))).backward
          (Tensor3.flatten W) (fun j => rnd (batchSlice N (c*h*w) (den e) n j)) idx)
  | _, .depthwiseStridedXlaBiasGradB (N := N) (c := c) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (depthwiseStride2XlaBiasGradHasVJP W (batchSlice N (c*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (c*h*w) (den e) n) o
  | _, .reluF e        => fun i => max (den e i) 0
  | _, .selectPos _ x e => fun i => if x i > 0 then den e i else 0
  | _, .relu6F e       => fun i => min (max (den e i) 0) 6
  | _, .selectMid _ x e => fun i => if 0 < x i ∧ x i < 6 then den e i else 0
  | _, .convertF rnd e => fun i => rnd (den e i)
  | _, .flatConvF _ _ W b e => flatConv W b (den e)
  -- Operands rounded, the accumulated sum rounded (bf16 store), bias added after in f32.
  | _, .flatConvFBf16 rnd _ _ W b e =>
      fun i => rnd (flatConv (fun o c kh kw => rnd (W o c kh kw)) 0 (fun j => rnd (den e j)) i)
               + Tensor3.flatten (fun o _ _ => b o) i
  | _, .maxPoolF (c := c) (h := h) (w := w) e => maxPoolFlat c h w (den e)
  | _, .maxPool3s2F (c := c) (h := h) (w := w) e => maxPool3s2Flat c h w (den e)
  | _, .convBack _ W b v e => (HasVJP3.toHasVJP (conv2dHasVJP3 W b)).backward v (den e)
  | _, .maxPoolBack (c := c) (h := h) (w := w) _ x e => maxPoolBackFlat c h w x (den e)
  | _, .maxPool3s2Back (c := c) (h := h) (w := w) _ x e => maxPool3s2BackFlat c h w x (den e)
  | _, .bnF (n := n) _ _ _ ε γ β e => bnForward n ε γ β (den e)
  | _, .bnBack (n := n) _ _ _ ε γ x e => bnGradInput n ε γ x (den e)
  | _, .addV a b       => fun j => den a j + den b j
  | _, .addVB a b      => fun j => den a j + den b j
  | _, .subB a b       => fun j => den a j - den b j
  | _, .gapF (c := c) (h := h) (w := w) e => globalAvgPoolFlat c h w (den e)
  | _, .gapBack (c := c) (h := h) (w := w) e =>
      (globalAvgPoolFlatHasVJP c h w).backward (fun _ => 0) (den e)
  | _, .broadcastBack (c := c) (h := h) (w := w) e =>
      fun k => ∑ idx : Fin (c * h * w), if flatChannel c h w idx = k then den e idx else 0
  | _, .flatConvStridedF _ _ W b e => flatConvStride2 W b (den e)
  | _, .flatConvStridedXlaF _ _ W b e => flatConvStride2Xla W b (den e)
  | _, .flatConvStride4F _ _ W b e => flatConvStride4 W b (den e)
  | _, .convStridedBack _ W b v e => (flatConvStride2HasVJP W b).backward v (den e)
  | _, .convStridedWeightSgd _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx
        - lr * (flatConvStride2WeightGradHasVJP b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convStridedBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (flatConvStride2BiasGradHasVJP W x).backward b (den e) o
  | _, .convStridedXlaWeightSgd _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx
        - lr * (flatConvStride2XlaWeightGradHasVJP b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convStridedXlaBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (flatConvStride2XlaBiasGradHasVJP W x).backward b (den e) o
  | _, .depthwiseWeightSgd _ _ _ b x W lr e => depthwiseWeightSgdDen b x W lr (den e)
  | _, .depthwiseBiasSgd _ _ W x b lr e => depthwiseBiasSgdDen W x b lr (den e)
  | _, .depthwiseStridedWeightSgd _ _ _ b x W lr e => depthwiseStridedWeightSgdDen b x W lr (den e)
  | _, .depthwiseStridedBiasSgd _ _ W x b lr e => depthwiseStridedBiasSgdDen W x b lr (den e)
  | _, .depthwiseStridedXlaWeightSgd _ _ _ b x W lr e => depthwiseStridedXlaWeightSgdDen b x W lr (den e)
  | _, .depthwiseStridedXlaBiasSgd _ _ W x b lr e => depthwiseStridedXlaBiasSgdDen W x b lr (den e)
  | _, .bnPerChannelF (oc := oc) (h := h) (w := w) _ _ _ ε γ β e =>
      bnPerChannelTensor3 oc h w ε γ β (den e)
  | _, .bnPerChannelBack (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      bnPerChannelTensor3GradInput oc h w ε γ x (den e)
  | _, .bnPerChannelEvalF (oc := oc) (h := h) (w := w) _ _ _ _ _ ε γ β μ var e =>
      bnPerChannelEvalTensor3 oc h w ε γ β μ var (den e)
  | _, .depthwiseF _ _ W b e => depthwiseFlat W b (den e)
  | _, .depthwiseBack _ W b v e => (depthwiseFlatHasVJP W b).backward v (den e)
  | _, .depthwiseStridedF _ _ W b e => depthwiseStride2Flat W b (den e)
  | _, .depthwiseStridedXlaF _ _ W b e => depthwiseStride2FlatXla W b (den e)
  | _, .depthwiseStridedBack _ W b v e => (depthwiseStride2FlatHasVJP W b).backward v (den e)
  | _, .depthwiseStridedXlaBack _ W b v e => (depthwiseStride2FlatXlaHasVJP W b).backward v (den e)
  | _, .swishF (n := n) e => swish n (den e)
  | _, .swishBack (n := n) _ x e => (swishHasVJP n).backward x (den e)
  | _, .sigmoidF (n := n) e => sigmoid n (den e)
  | _, .sigmoidBack (n := n) _ x e => (sigmoidHasVJP n).backward x (den e)
  | _, .maxPoolBackB (N := N) (c := c) (h := h) (w := w) _ x e =>
      batchMapAux N (maxPoolBackFlat c h w) x (den e)
  | _, .maxPool3s2BackB (N := N) (c := c) (h := h) (w := w) _ x e =>
      batchMapAux N (maxPool3s2BackFlat c h w) x (den e)
  | _, .convBiasSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ W x b lr e =>
      fun o => b o - lr * ∑ n : Fin N,
        (conv2dBiasGradHasVJP W (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .convStridedBiasSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ W x b lr e =>
      fun o => b o - lr * ∑ n : Fin N,
        (flatConvStride2BiasGradHasVJP W (batchSlice N (ic*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .selectPosB _ x e => fun i => if x i > 0 then den e i else 0
  | _, .selectMidB _ x e => fun i => if 0 < x i ∧ x i < 6 then den e i else 0
  | _, .dropPathB (N := N) (n := n) _ s e => Proofs.dropPath N n s (den e)
  | _, .dropoutB (N := N) (n := n) _ mask e => Proofs.dropout N n mask (den e)
  | _, .swishBackB (N := N) (n := n) _ x e => (swishHasVJP (N*n)).backward x (den e)
  -- `gelu` is POINTWISE, so its VJP at the batched width `N*n` is already the batch-lift of the
  -- per-example one — the same argument `swishBackB` rests on, and why neither needs `batchMapAux`.
  | _, .geluBackB (N := N) (n := n) _ x e => (geluHasVJP (N*n)).backward x (den e)
  -- The `Σ_n` shape, verbatim from `convWeightGradB`: a shared parameter's batched gradient is the
  -- sum over examples of the per-example gradient on `batchSlice n`.
  | _, .convStride4WeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (flatConvStride4WeightGradHasVJP b
            (batchSlice N (ic*(2*(2*h))*(2*(2*w))) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  -- The bf16 peer. ⚠ `rnd` OUTSIDE the `Σ_n`: the emit contracts the batch inside one convolution
  -- and stores its bf16 result once, so a rounding per summand would claim a coarser computation
  -- than the hardware performs — the same reason `convWeightGradBBf16` is written this way.
  | _, .convStride4WeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (flatConvStride4WeightGradHasVJP b
            (fun j => rnd (batchSlice N (ic*(2*(2*h))*(2*(2*w))) x n j))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) _ x e =>
      fun cc => ∑ n : Fin N, ∑ k : Fin (c*h*w),
        (if chanIdx c h w k = cc
         then batchSlice N (c*h*w) x n k * batchSlice N (c*h*w) (den e) n k else 0)
  -- ⚠ TWO-LEVEL: the outer `Σ_n` is the batch, the inner `Σ_r` the rows within one example. The
  -- per-example peer has only the inner one, so a naive copy would silently drop the batch sum —
  -- and the shapes would still check, because both spellings land in `Vec D`.
  | _, .veclnGammaGradB (N := N) (R := R) (D := D) _ _ ε x e =>
      fun k => ∑ n : Fin N, ∑ r : Fin R,
        Mat.unflatten (batchSlice N (R*D) (den e) n) r k
          * layerNormForward D ε 1 0 (Mat.unflatten (batchSlice N (R*D) x n) r) k
  | _, .rowDenseBiasGradB (N := N) (R := R) (c := c) e =>
      fun j => ∑ n : Fin N, ∑ r : Fin R, batchSlice R c (batchSlice N (R*c) (den e) n) r j
  -- ── ViT increment 2. The first two are `batchMapAux` (per-example, no contraction); the last
  --    four are `Σ_b` over the batch of the per-example gradient — the `*GradB` shape.
  --    ⚠ `batchMapAux` used SYMMETRICALLY here for the first time: `matmulFB`'s "aux" is the left
  --    operand of a binary op, not a saved activation. Its body never cared.
  | _, .matmulFB (N := N) (m := m) (k := k) (n := n) a b =>
      batchMapAux N (matMulFlat m k n) (den a) (den b)
  -- ⚠ BOTH operands rounded AND an outer bf16 store — the bf16-typed-result shape (§20.1). This is
  -- the activation × activation case, so all three roundings land on running values rather than on
  -- a weight; `batchMapAux` is unchanged because it never cared which operand was which.
  | _, .matmulFBBf16 (N := N) (m := m) (k := k) (n := n) rnd a b =>
      batchMapAux N (fun u v i => rnd (matMulFlat m k n (fun j => rnd (u j)) (fun j => rnd (v j)) i))
        (den a) (den b)
  | _, .softmaxRowBackB (N := N) (m := m) (n := n) _ preAct e =>
      batchMapAux N (rowSoftmaxBackFlat m n) preAct (den e)
  | _, .rowDenseWeightGradB (N := N) (tk := tk) (a := a) (c := c) _ x e =>
      fun idx => ∑ b : Fin N,
        Mat.flatten (fun i j => ∑ t : Fin tk,
          batchSlice tk a (batchSlice N (tk*a) x b) t i
            * batchSlice tk c (batchSlice N (tk*c) (den e) b) t j) idx
  -- ⚠ The emitted `dot_general` contracts `[0,1] x [0,1]` — batch AND token in one op — and keeps
  -- its f32-typed result deliberately (see the constructor), so both `∑`s ride the f32 accumulate,
  -- only the two leaf reads round, and there is NO outer store rounding. ⭐ It is now the only bf16
  -- dot in the kit shaped this way, which is exactly why the constructor says why.
  | _, .rowDenseWeightGradBBf16 (N := N) (tk := tk) (a := a) (c := c) rnd _ x e =>
      fun idx => ∑ b : Fin N,
        Mat.flatten (fun i j => ∑ t : Fin tk,
          rnd (batchSlice tk a (batchSlice N (tk*a) x b) t i)
            * rnd (batchSlice tk c (batchSlice N (tk*c) (den e) b) t j)) idx
  | _, .posEmbedGradB (N := N) (tk := tk) (D := D) e =>
      fun i => ∑ b : Fin N, batchSlice N ((tk+1)*D) (den e) b i
  | _, .patchEmbedWeightGradB (N := N) (ic := ic) (H := H) (W := W) (P := P) (tk := tk) (D := D)
        _ x e =>
      fun idx => ∑ b : Fin N,
        patchEmbedWeightGradFlat ic H W P tk D
          (batchSlice N (ic*H*W) x b) (batchSlice N ((tk+1)*D) (den e) b) idx
  -- ⚠⚠ **The outer `rnd` wraps the WHOLE batch sum**, and that placement is the measurement, not a
  -- style choice: the emit contracts the batch axis INSIDE a single `convolution` whose result is
  -- bf16-typed, so the hardware rounds once, after `Σ_b`. Rounding each summand instead would
  -- describe `N` stores where the graph performs one — and it is the direction that UNDERSTATES
  -- the error, which is the unsound one for a bound.
  | _, .patchEmbedWeightGradBBf16 (N := N) (ic := ic) (H := H) (W := W) (P := P) (tk := tk) (D := D)
        rnd _ x e =>
      fun idx => rnd (∑ b : Fin N,
        patchEmbedWeightGradFlat ic H W P tk D
          (fun j => rnd (batchSlice N (ic*H*W) x b j))
          (fun j => rnd (batchSlice N ((tk+1)*D) (den e) b j)) idx)
  -- ⚠ `p.succ` skips the CLS row, exactly as the per-example peer does; the batch sum is the outer
  -- one. Two levels, and the inner one is the one the emitted `slice [.., 1:tk+1, ..]` encodes.
  | _, .patchEmbedBiasGradB (N := N) (tk := tk) (c := c) e =>
      fun i => ∑ b : Fin N, ∑ p : Fin tk,
        batchSlice (tk+1) c (batchSlice N ((tk+1)*c) (den e) b) p.succ i
  -- `Σ_n` over the batch of the per-example outer product / cotangent, the `*GradB` shape.
  | _, .weightGradB (N := N) (m := m) (n := n) _ x e =>
      fun idx => ∑ k : Fin N,
        (Mat.flatten (fun i j => batchSlice N m x k i * batchSlice N n (den e) k j)) idx
  -- ⚠ `biasGrad` is the IDENTITY on its operand (the per-example peer returns `SHlo n`, not
  -- `SHlo` of the bias width) — the channel sum happens in the emitted reduce, outside the AST.
  -- Carried over verbatim so the batched form is the same carve-out, not a new one.
  | _, .biasGradB e => den e
  -- LayerNorm's backward is NOT pointwise (it reduces within a row), so this one genuinely needs
  -- the auxiliary lift: example `k` is handed `batchSlice k x`, never the whole `x`.
  | _, .lnRowBackB (N := N) (m := m) (n := n) _ _ _ ε γ x e =>
      batchMapAux N (rowLNBackFlat m n ε γ) x (den e)
  | _, .sigmoidB (N := N) (n := n) e => sigmoid (N*n) (den e)
  | _, .sigmoidBackB (N := N) (n := n) _ x e => (sigmoidHasVJP (N*n)).backward x (den e)
  | _, .geluF (n := n) e => gelu n (den e)
  | _, .geluBack (n := n) _ x e => (geluHasVJP n).backward x (den e)
  | _, .layerScaleF (n := n) _ γ e => layerScale γ (den e)
  | _, .layerScaleChF (c := c) (h := h) (w := w) _ γ e =>
      layerScale (fun k => γ (chanIdx c h w k)) (den e)
  | _, .softmaxRowF (m := m) (n := n) e => rowSoftmaxFlat m n (den e)
  | _, .softmaxRowBack (m := m) (n := n) _ preAct e => rowSoftmaxBackFlat m n preAct (den e)
  | _, .matmulF (m := m) (k := k) (n := n) a b => matMulFlat m k n (den a) (den b)
  | _, .transposeF (m := m) (n := n) e => transposeFlat m n (den e)
  | _, .scaleF _ s e => fun i => s * den e i
  | _, .lnRowF (m := m) (n := n) _ _ _ ε γ β e => rowLNFlat m n ε γ β (den e)
  | _, .lnRowBack (m := m) (n := n) _ _ _ ε γ x e => rowLNBackFlat m n ε γ x (den e)
  | _, .denseRowF (N := N) (a := a) (c := c) _ _ W b e => rowDenseFlat N a c W b (den e)
  | _, .denseRowBack (N := N) (a := a) (c := c) _ W e => rowDenseBackFlat N a c W (den e)
  | _, .patchEmbedF (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ _ _ Wc bc cls pos e =>
      patchEmbedFlat ic H W P N D Wc bc cls pos (den e)
  | _, .patchEmbedBack (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ Wc e =>
      patchEmbedBackFlat ic H W P N D Wc (den e)
  | _, .clsSliceF (N := N) (D := D) e => clsSliceFlat N D (den e)
  | _, .clsPadF (N := N) (D := D) e => clsPadFlat N D (den e)
  | _, .headSliceF (N := N) (heads := heads) (d := d) h e => headSliceFlat N heads d h (den e)
  | _, .headPadF (N := N) (heads := heads) (d := d) h e => headPadFlat N heads d h (den e)
  | _, .rowScaleF (m := m) (n := n) _ γ e => rowScaleFlat m n γ (den e)
  | _, .rowBiasF (m := m) (n := n) _ β e => rowBiasFlat m n β (den e)
  | _, .batchOp (N := N) op e => batchMap N (denOp op) (den e)
  | _, .bnBatchF (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ β e =>
      bnBatchLA N oc h w ε γ β (den e)
  | _, .bnBatchBack (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      bnBatchTensor4GradInput N oc h w ε γ x (den e)
  | _, .convBackBatched (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (HasVJP3.toHasVJP (conv2dHasVJP3 W b)).backward (fun _ => 0) dy) (den e)
  | _, .convStridedBackBatched (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (flatConvStride2HasVJP W b).backward (fun _ => 0) dy) (den e)
  | _, .convBackBatchedBf16 (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((HasVJP3.toHasVJP
        (conv2dHasVJP3 (fun o c a d => rnd (W o c a d)) b)).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .convBackBatchedF8 (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((HasVJP3.toHasVJP
        (conv2dHasVJP3 (fun o c a d => rnd (W o c a d)) b)).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .convStridedBackBatchedBf16 (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((flatConvStride2HasVJP
        (fun o c a d => rnd (W o c a d)) b).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .depthwiseBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (HasVJP3.toHasVJP (depthwiseHasVJP3 W b)).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((HasVJP3.toHasVJP
        (depthwiseHasVJP3 (fun cc a d => rnd (W cc a d)) b)).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .depthwiseStridedBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (depthwiseStride2FlatHasVJP W b).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseStridedBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((depthwiseStride2FlatHasVJP
        (fun cc a d => rnd (W cc a d)) b).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .depthwiseStridedXlaBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (depthwiseStride2FlatXlaHasVJP W b).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseStridedXlaBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((depthwiseStride2FlatXlaHasVJP
        (fun cc a d => rnd (W cc a d)) b).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      fun i => ∑ k, if i = (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) k then
        bnBatchTensor4GradInput N oc h w ε γ
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
          (fun i' => ∑ k', if i' = (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w))) k'
                           then den e k' else 0) k
        else 0
  | _, .seBackBatched (h := h) (w := w) _ _ _ _ _ W₁ b₁ W₂ b₂ v e =>
      fun idx =>
        (seBlockFullHasVJP (h := h) (w := w) W₁ b₁ W₂ b₂).backward
          (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (Mat.unflatten (den e) (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2
  | _, .seReduceB (N := N) (c := c) (h := h) (w := w) _ x e =>
      -- the SE gate cotangent: per example, the broadcast-adjoint of `x ⊙ dy`
      -- (`broadcastFlatHasVJP.backward` = sum each channel's spatial Hadamard).
      fun idx =>
        ∑ q : Fin (c * h * w),
          if flatChannel c h w q = (finProdFinEquiv.symm idx).2 then
            batchSlice N (c * h * w) x (finProdFinEquiv.symm idx).1 q
              * batchSlice N (c * h * w) (den e) (finProdFinEquiv.symm idx).1 q
          else 0
  | _, .gapBackBatched (N := N) (c := c) (h := h) (w := w) e =>
      batchMap N (fun dgap => (globalAvgPoolFlatHasVJP c h w).backward (fun _ => 0) dgap) (den e)

open Lean Meta in
/-- `den e` (and `den e i`) one constructor deep, by smart unfolding at default transparency: the
    match reduces the way the `den_*` lemmas' `rfl` does, and nothing asks for `den.eq_def`. -/
def denUnfold? (e : Expr) : MetaM (Option Expr) := do
  let args := e.getAppArgs
  if args.size < 2 then return none
  let some h ← withDefault <| unfoldDefinition? (mkAppN e.getAppFn args[:2]) | return none
  return some (mkAppN h args[2:]).headBeta

/-- What `simp only` should name instead of `den`. Naming `den` itself makes Lean build
    `den.eq_def` — about four minutes for the 215-arm match, on the critical path of this module
    and anything that first asks for it. `denStepApp` is the same step where `den e` is applied
    to an index, which `simp` does not visit as `den e`. -/
dsimproc denStep (den _) := fun e => do
  let some e' ← denUnfold? e | return .continue
  return .visit e'

/-- `denStep` where `den e` is applied to an index. -/
dsimproc denStepApp (den _ _) := fun e => do
  let some e' ← denUnfold? e | return .continue
  return .visit e'

@[simp] theorem den_operand {n : Nat} (s : String) (v : Vec n) :
    den (.operand s v) = v := rfl
@[simp] theorem den_dotIn {m n : Nat} (s : String) (W : Mat m n) (e : SHlo m) :
    den (.dotIn s W e) = fun j => ∑ i, den e i * W i j := rfl
/-- The mixed-precision matmul denotes the EXACT sum over ROUNDED operands. The fp32
    accumulate is why the `∑` carries no rounding of its own — the whole deviation from
    `dotIn` sits in the two `rnd`s, which is the structural reason bf16 is the easy twin
    of fp8 (no block scale to factor through the sum). -/
@[simp] theorem den_dotInBf16 {m n : Nat} (rnd : ℝ → ℝ) (s : String) (W : Mat m n)
    (e : SHlo m) :
    den (.dotInBf16 rnd s W e) = fun j => ∑ i, rnd (den e i) * rnd (W i j) := rfl
/-- **The bundling is inert.** `dotInBf16` on raw operands denotes exactly what `dotIn`
    denotes on PRE-rounded ones — so every tie already proven in the `dotIn` vocabulary
    (e.g. `Bf16PoC.bf16_render_faithful`) transfers to the emittable node by rewriting
    with this, rather than being reproved. -/
theorem dotInBf16_eq_dotIn_rounded {m n : Nat} (rnd : ℝ → ℝ) (s : String) (W : Mat m n)
    (e : SHlo m) :
    den (.dotInBf16 rnd s W e)
      = den (.dotIn s (fun i j => rnd (W i j)) (.convertF rnd e)) := rfl
@[simp] theorem den_dotOut {m n : Nat} (s : String) (W : Mat m n) (e : SHlo n) :
    den (.dotOut s W e) = fun i => ∑ j, W i j * den e j := rfl
@[simp] theorem den_addBcast {n : Nat} (s : String) (b : Vec n) (e : SHlo n) :
    den (.addBcast s b e) = fun j => den e j + b j := rfl
@[simp] theorem den_expe {n : Nat} (e : SHlo n) :
    den (.expe e) = fun j => Real.exp (den e j) := rfl
@[simp] theorem den_softmaxDiv {n : Nat} (e : SHlo n) :
    den (.softmaxDiv e) = fun j => den e j / ∑ k, den e k := rfl
@[simp] theorem den_addV {n : Nat} (a b : SHlo n) :
    den (.addV a b) = fun j => den a j + den b j := rfl
@[simp] theorem den_reluF {n : Nat} (e : SHlo n) :
    den (.reluF e) = fun i => max (den e i) 0 := rfl
@[simp] theorem den_selectPos {n : Nat} (s : String) (x : Vec n) (e : SHlo n) :
    den (.selectPos s x e) = fun i => if x i > 0 then den e i else 0 := rfl
/-- **The round node is `den`-faithful for any rounding.** This is the equation
    [`Proofs/Float/Bf16Fold.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Float/Bf16Fold.lean) asks for by name to lift its depth-1 tie to
    depth > 1: rounding an *intermediate* activation is now an in-graph op whose
    denotation is exactly post-composition with `rnd`. No bf16 specifics appear here —
    bf16 round-to-nearest is one instance, and the accuracy half is supplied separately
    by `dense_close_mixed` at `u_leaf = 2⁻⁸`. -/
@[simp] theorem den_convertF {n : Nat} (rnd : ℝ → ℝ) (e : SHlo n) :
    den (.convertF rnd e) = fun i => rnd (den e i) := rfl
/-- The round node composed with `den` as a function, the form the tie proofs want. -/
theorem convertF_faithful {n : Nat} (rnd : ℝ → ℝ) (e : SHlo n) :
    den (.convertF rnd e) = rnd ∘ den e := rfl
@[simp] theorem den_relu6F {n : Nat} (e : SHlo n) :
    den (.relu6F e) = fun i => min (max (den e i) 0) 6 := rfl
@[simp] theorem den_selectMid {n : Nat} (s : String) (x : Vec n) (e : SHlo n) :
    den (.selectMid s x e) = fun i => if 0 < x i ∧ x i < 6 then den e i else 0 := rfl

/-- **A batched token denotes its per-example op, lifted.** `den (.batchOp op e)` is `batchMap N` of
    `denOp op`, the proven per-example map, by `rfl`; `simp only [den_batchOp, denOp]` reads a
    batched graph's denotation off `denOp`'s arms. `skel` erases values, so a descriptor with the
    wrong `denOp` emits identical bytes: this equation is the half the emit ties cannot see. The
    true-batch-norm token is not a descriptor; it denotes `bnBatchLA` (`den_bnBatchF`). -/
@[simp] theorem den_batchOp {N a b : Nat} (op : BatchableOp a b) (e : SHlo (N * a)) :
    den (.batchOp (N := N) op e) = batchMap N (denOp op) (den e) := rfl

attribute [simp] denOp

/-- The descriptor form of swish denotes exactly what the descriptor-less `swishF` denoted at the
    same index — the batched graph computes the same function, only the emit width now travels
    separately from the batch. -/
theorem den_batchOp_swish_eq_swishF {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.swish (n := n)) e) = den (.swishF e) :=
  batchMap_pointwise swishScalar (den e)

/-- **The softmax denominator is PER EXAMPLE — the property this descriptor exists for.** Example
    `k`'s output divides by example `k`'s own sum, not by the sum over the whole batch.

    ⚠ This is the half the emit tie structurally cannot see. `.softmaxDiv`'s emitted MLIR was
    *already* per-example (it reduces over `dimensions = [1]` of `tensor<B,n>`), so the batched and
    per-example forms render byte-for-byte identically and always would — while the descriptor-less
    `den` at index `N·n` reads `v j / ∑ k, v k` over ALL `N·n` coordinates, i.e. it divides by the
    batch's total. Same bytes, different function, and only this statement separates them. -/
theorem den_batchOp_softmaxDiv_per_example {N n : Nat} (e : SHlo (N * n))
    (k : Fin N) (j : Fin n) :
    den (.batchOp (N := N) (.softmaxDiv (n := n)) e) (finProdFinEquiv (k, j))
      = batchSlice N n (den e) k j / ∑ i, batchSlice N n (den e) k i := by
  simp only [den_batchOp, batchMap, denOp, Equiv.symm_apply_apply, batchSlice]

/-- ⭐ **THE CLS SLICE IS THE ONE PLACE THE BATCH AND THE TOKEN AXIS COULD SWAP SILENTLY.**
    `clsSlice` takes `(tk+1)*D` to `D` — it CONTRACTS — and `batchMap N` of it takes `N*((tk+1)*D)`
    to `N*D`. A render that read the batch as the token axis would take `(N+1)*D` to `D`, i.e. drop
    every example but one and still type-check at `N = tk`. Stated so the two indices are pinned
    apart by a theorem rather than by a naming convention. -/
theorem den_batchOp_clsSlice_per_example {N tk D : Nat} (e : SHlo (N * ((tk+1)*D)))
    (k : Fin N) (i : Fin D) :
    den (.batchOp (N := N) (.clsSlice (N := tk) (D := D)) e) (finProdFinEquiv (k, i))
      = clsSliceFlat tk D (batchSlice N ((tk+1)*D) (den e) k) i := by
  simp only [den_batchOp, denOp, batchMap, Equiv.symm_apply_apply]
  rfl

@[simp] theorem den_scaleB {N n : Nat} (sS : String) (s : ℝ) (e : SHlo (N*n)) :
    den (.scaleB sS s e) = fun i => den e i * s := rfl
@[simp] theorem den_shiftB {N n : Nat} (sS : String) (s : ℝ) (e : SHlo (N*n)) :
    den (.shiftB sS s e) = fun i => den e i + s := rfl
@[simp] theorem den_divConstB {N n : Nat} (sS : String) (s : ℝ) (e : SHlo (N*n)) :
    den (.divConstB sS s e) = fun i => den e i / s := rfl

/-- **The all-reduce node denotes the replica mean of its operands.** The `DataParallel.dpMean`
    spelling, stated here so a tie can read it without importing that file. -/
@[simp] theorem den_allReduceMeanF {n : Nat} (R : Nat) (hR : 0 < R) (t : String) (ds : List Nat)
    (g : Fin R → SHlo n) :
    den (.allReduceMeanF R hR t ds g) = fun i => (1 / (R : ℝ)) * ∑ r : Fin R, den (g r) i := rfl
@[simp] theorem den_bnBatchMeanB {N oc h w : Nat} (e : SHlo (N * (oc * (h * w)))) :
    den (.bnBatchMeanB e)
      = fun c => bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) := rfl
@[simp] theorem den_bnBatchVarB {N oc h w : Nat} (e : SHlo (N * (oc * (h * w)))) :
    den (.bnBatchVarB e)
      = fun c => bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) := rfl
@[simp] theorem den_bnBatchVarAtB {N oc h w : Nat} (e : SHlo (N * (oc * (h * w))))
    (mu : SHlo oc) :
    den (.bnBatchVarAtB e mu)
      = fun c => bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c)
             + (bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) - den mu c)
             * (bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) - den mu c) := rfl
@[simp] theorem den_bnPackB {oc : Nat} (a b : SHlo oc) :
    den (.bnPackB a b) = Fin.append (den a) (den b) := rfl
@[simp] theorem den_bnSyncF {N oc h w : Nat} (gName bName epsStr : String) (ε : ℝ)
    (γ β : Vec oc) (x : SHlo (N * (oc * (h * w)))) (st : SHlo (oc + oc)) :
    den (.bnSyncF gName bName epsStr ε γ β x st)
      = bnSyncTensor4 N oc h w ε γ β
          (fun c => den st (Fin.castAdd oc c))
          (fun c => den st (Fin.natAdd  oc c) + den st (Fin.castAdd oc c) * den st (Fin.castAdd oc c))
          (den x) := rfl
@[simp] theorem den_bnSyncDyStatsB {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ)
    (γ : Vec oc) (x : Vec (N * (oc * (h * w)))) (dy : SHlo (N * (oc * (h * w))))
    (st : SHlo (oc + oc)) :
    den (.bnSyncDyStatsB gName xName epsStr ε γ x dy st)
      = Fin.append (den st)
          (Fin.append
            (fun c => bnMean (N*(h*w)) (fun k =>
                γ c * Mat.unflatten (bnchwFwd N oc h w (den dy)) c k))
            (fun c => bnMean (N*(h*w)) (fun k =>
                bnSyncXhat (N*(h*w)) ε (den st (Fin.castAdd oc c))
                  (den st (Fin.natAdd oc c) + den st (Fin.castAdd oc c) * den st (Fin.castAdd oc c))
                  (Mat.unflatten (bnchwFwd N oc h w x) c) k
                * (γ c * Mat.unflatten (bnchwFwd N oc h w (den dy)) c k)))) := rfl
@[simp] theorem den_bnSyncBack {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ)
    (γ : Vec oc) (x : Vec (N * (oc * (h * w)))) (dy : SHlo (N * (oc * (h * w))))
    (ds : SHlo (oc + oc + (oc + oc))) :
    den (.bnSyncBack gName xName epsStr ε γ x dy ds)
      = bnSyncTensor4GradInput N oc h w ε γ
          (fun c => den ds (Fin.castAdd (oc+oc) (Fin.castAdd oc c)))
          (fun c => den ds (Fin.castAdd (oc+oc) (Fin.natAdd  oc c))
                    + den ds (Fin.castAdd (oc+oc) (Fin.castAdd oc c))
                      * den ds (Fin.castAdd (oc+oc) (Fin.castAdd oc c)))
          (fun c => den ds (Fin.natAdd  (oc+oc) (Fin.castAdd oc c)))
          (fun c => den ds (Fin.natAdd  (oc+oc) (Fin.natAdd  oc c)))
          x (den dy) := rfl
@[simp] theorem den_bnSyncGammaGradB {N oc h w : Nat} (xName epsStr : String) (ε : ℝ)
    (x : Vec (N * (oc * (h * w)))) (dy : SHlo (N * (oc * (h * w)))) (st : SHlo (oc + oc)) :
    den (.bnSyncGammaGradB xName epsStr ε x dy st)
      = bnSyncPerChannelGradGamma oc (N*(h*w)) ε
          (fun c => den st (Fin.castAdd oc c))
          (fun c => den st (Fin.natAdd  oc c) + den st (Fin.castAdd oc c) * den st (Fin.castAdd oc c))
          (bnchwFwd N oc h w x) (bnchwFwd N oc h w (den dy)) := rfl
@[simp] theorem den_bnStatsMeanB {oc : Nat} (e : SHlo (oc + oc)) :
    den (.bnStatsMeanB e) = fun c => den e (Fin.castAdd oc c) := rfl
@[simp] theorem den_bnStatsVarB {oc : Nat} (e : SHlo (oc + oc)) :
    den (.bnStatsVarB e) = fun c => den e (Fin.natAdd oc c) := rfl

/-- The one-replica collective threads its operand. -/
theorem den_allReduceMeanF_one {n : Nat} (t : String) (ds : List Nat) (g : SHlo n) (i : Fin n) :
    den (SHlo.allReduceMeanF 1 Nat.one_pos t ds (fun _ => g)) i = den g i := by
  simp only [den_allReduceMeanF, Nat.cast_one, ne_eq, one_ne_zero, not_false_eq_true, div_self, univ_unique,
    Fin.default_eq_zero, Fin.isValue, sum_const, card_singleton, one_smul, one_mul]

/-- **At `R = 1` the two-round statistics subgraph is `[μ ‖ σ²]` of the batch itself**: the
    replica's own mean, and its own two-pass variance with a zero offset. -/
theorem den_syncStats_R1 {N oc h w : Nat} (t t' : String) (ds ds' : List Nat)
    (x : SHlo (N * (oc * (h * w)))) (c : Fin oc) :
    den (SHlo.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))
          (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB x
            (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))))) (Fin.castAdd oc c)
        = bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den x)) c)
    ∧ den (SHlo.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))
          (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB x
            (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))))) (Fin.natAdd oc c)
        = bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den x)) c) := by
  constructor
  · simp only [den_bnPackB, Fin.append_left, den_allReduceMeanF_one, den_bnBatchMeanB]
  · simp only [den_bnPackB, Fin.append_right, den_allReduceMeanF_one, den_bnBatchVarAtB,
               den_bnBatchMeanB, sub_self, mul_zero, add_zero]

/-- ⭐⭐ **THE DROP-IN, on actual graph nodes: at `R = 1` the sync-BN subgraph denotes
    `bnBatchTensor4`.**

    The Foundation anchors say the sync forward at its own statistics is the batch forward; this
    says the GRAPH a sync render emits — `bnSyncF` fed by `bnPackB` of the two one-replica
    collectives (μ, then σ² at μ) — is that, once `R = 1` collapses every collective to its
    single operand.

    So a single-device sync render computes exactly what today's `bnBatchF` render computes, and
    the `R = 1` artifacts need not move. The `R > 1` case is then purely a question about how
    shard statistics compose (`bnMean_shard` / `bnVar_shard_chan`), with BatchNorm itself already
    accounted for here. `planning/global_bn_verified.md` §2b/§2c. -/
theorem den_bnSyncF_allReduce_R1 {N oc h w : Nat} (gN bN es t t' : String) (ds ds' : List Nat)
    (ε : ℝ) (γ β : Vec oc) (hm : N * (h * w) ≠ 0) (x : SHlo (N * (oc * (h * w)))) :
    den (.bnSyncF gN bN es ε γ β x
          (.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))
            (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB x
              (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))))))
      = bnBatchTensor4 N oc h w ε γ β (den x) := by
  rw [den_bnSyncF]
  simp only [(den_syncStats_R1 t t' ds ds' x _).1, (den_syncStats_R1 t t' ds ds' x _).2,
    bnVar_add_mean_mul_mean _ hm]
  exact bnSyncTensor4_at_own_stats N oc h w hm ε γ β (den x)

/-- ⭐⭐ **THE DROP-IN, backward half: at `R = 1` the sync-BN backward subgraph denotes
    `bnBatchTensor4GradInput`.**

    The peer of `den_bnSyncF_allReduce_R1`. The graph is the one a sync render emits — an outer
    `allReduceMeanF` over `bnSyncDyStatsB`, itself fed by the packed forward statistics — and at
    `R = 1` every collective collapses to its single operand, leaving the committed three-term
    backward. `hx` ties the saved host activation to the graph value it came from, which is the
    renderer's own invariant. -/
theorem den_bnSyncBack_allReduce_R1 {N oc h w : Nat} (gN xN es t t' t'' : String)
    (ds ds' ds'' : List Nat) (ε : ℝ) (γ : Vec oc) (hm : N * (h * w) ≠ 0)
    (xg : SHlo (N * (oc * (h * w)))) (x : Vec (N * (oc * (h * w)))) (hx : den xg = x)
    (dy : SHlo (N * (oc * (h * w)))) :
    den (.bnSyncBack gN xN es ε γ x dy
          (.allReduceMeanF 1 Nat.one_pos t'' ds''
            (fun _ => .bnSyncDyStatsB gN xN es ε γ x dy
              (.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB xg))
                (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB xg
                  (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB xg))))))))
      = bnBatchTensor4GradInput N oc h w ε γ x (den dy) := by
  subst hx
  rw [den_bnSyncBack]
  simp only [den_allReduceMeanF_one, den_bnSyncDyStatsB, Fin.append_left, Fin.append_right,
             (den_syncStats_R1 t t' ds ds' xg _).1, (den_syncStats_R1 t t' ds ds' xg _).2,
             bnVar_add_mean_mul_mean _ hm, bnSyncXhat_at_own_stats _ hm]
  exact bnSyncTensor4GradInput_at_own_stats N oc h w hm ε γ _ (den dy)

/-- ⭐⭐ **THE DROP-IN, γ half: at `R = 1` the sync γ-gradient node denotes `bnGammaGradB`.**
    The third anchor beside `den_bnSyncF_allReduce_R1` / `den_bnSyncBack_allReduce_R1`: fed the
    collapsed collectives, the sync γ node reads the batch's own statistics and is the committed
    γ gradient. -/
theorem den_bnSyncGammaGradB_allReduce_R1 {N oc h w : Nat} (xN es t t' : String)
    (ds ds' : List Nat) (ε : ℝ) (hm : N * (h * w) ≠ 0)
    (xg : SHlo (N * (oc * (h * w)))) (x : Vec (N * (oc * (h * w)))) (hx : den xg = x)
    (dy : SHlo (N * (oc * (h * w)))) :
    den (.bnSyncGammaGradB xN es ε x dy
          (.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB xg))
            (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB xg
              (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB xg))))))
      = den (.bnGammaGradB xN es ε x dy) := by
  subst hx
  rw [den_bnSyncGammaGradB]
  simp only [(den_syncStats_R1 t t' ds ds' xg _).1, (den_syncStats_R1 t t' ds ds' xg _).2,
    bnVar_add_mean_mul_mean _ hm]
  exact bnSyncPerChannelGradGamma_at_own_stats oc (N*(h*w)) hm ε _ _

/-- **`R = 1`: the handed-back sync mean IS `bnBatchMeanB`.** -/
theorem den_bnStatsMeanB_allReduce_R1 {N oc h w : Nat} (t t' : String) (ds ds' : List Nat)
    (x : SHlo (N * (oc * (h * w)))) :
    den (.bnStatsMeanB
          (.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))
            (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB x
              (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))))))
      = den (.bnBatchMeanB x) := by
  funext c
  simp only [den_bnStatsMeanB, (den_syncStats_R1 t t' ds ds' x c).1, den_bnBatchMeanB]

/-- **`R = 1`: the handed-back sync variance IS `bnBatchVarB`** — the batch's own two-pass
    variance, offset zero. -/
theorem den_bnStatsVarB_allReduce_R1 {N oc h w : Nat} (t t' : String) (ds ds' : List Nat)
    (x : SHlo (N * (oc * (h * w)))) :
    den (.bnStatsVarB
          (.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))
            (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB x
              (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))))))
      = den (.bnBatchVarB x) := by
  funext c
  simp only [den_bnStatsVarB, (den_syncStats_R1 t t' ds ds' x c).2, den_bnBatchVarB]

@[simp] theorem den_maxPoolBackB {N c h w : Nat} (xN : String) (x : Vec (N*(c*(2*h)*(2*w))))
    (e : SHlo (N*(c*h*w))) :
    den (.maxPoolBackB xN x e) = batchMapAux N (maxPoolBackFlat c h w) x (den e) := rfl
@[simp] theorem den_maxPool3s2BackB {N c h w : Nat} (xN : String) (x : Vec (N*(c*(2*h)*(2*w))))
    (e : SHlo (N*(c*h*w))) :
    den (.maxPool3s2BackB xN x e) = batchMapAux N (maxPool3s2BackFlat c h w) x (den e) := rfl
-- ⚠ **What separates the two pools is NOT stated here, deliberately.** A `≠` between the two
-- constructors would be content-free — Lean makes distinct constructors distinct — and the pair
-- share a type, an arity, an op count and a `pretty` shape, so nothing structural tells them
-- apart. The claim worth pinning is that they **emit different text**, and that lives in
-- `tests/TestBatchedEmitTie.lean` beside the `dropoutB ≠ dropPathB` assertions (§0.12) for the
-- same reason: two poolings differing only in a window are exactly the pair a reader ticks off as
-- "present" without checking *which*, and only the bytes settle it.
@[simp] theorem den_selectPosB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.selectPosB xN x e) = fun i => if x i > 0 then den e i else 0 := rfl
@[simp] theorem den_selectMidB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.selectMidB xN x e) = fun i => if 0 < x i ∧ x i < 6 then den e i else 0 := rfl

@[simp] theorem den_dropPathB {N n : Nat} (mN : String) (s : Vec N) (e : SHlo (N*n)) :
    den (.dropPathB mN s e) = Proofs.dropPath N n s (den e) := rfl
@[simp] theorem den_dropoutB {N n : Nat} (mN : String) (mask : Vec (N*n)) (e : SHlo (N*n)) :
    den (.dropoutB mN mask e) = Proofs.dropout N n mask (den e) := rfl
@[simp] theorem den_swishBackB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.swishBackB xN x e) = (swishHasVJP (N*n)).backward x (den e) := rfl
@[simp] theorem den_geluBackB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.geluBackB xN x e) = (geluHasVJP (N*n)).backward x (den e) := rfl
@[simp] theorem den_rowDenseBiasGradB {N R c : Nat} (e : SHlo (N*(R*c))) :
    den (.rowDenseBiasGradB (N := N) (R := R) (c := c) e)
      = fun j => ∑ n : Fin N, ∑ r : Fin R, batchSlice R c (batchSlice N (R*c) (den e) n) r j := rfl

/-- **The two-level contraction is real, and this is what would have been silently lost.** At
    `N = 1` the batched bias gradient collapses to its per-example peer — so a render that dropped
    the batch sum type-checks, emits the same bytes and agrees on a one-example batch. The gate
    that catches it has to run at `N > 1`, which is why the emit tie alone is not enough here. -/
theorem den_rowDenseBiasGradB_at_one {R c : Nat} (e : SHlo (1*(R*c))) (j : Fin c) :
    den (.rowDenseBiasGradB (N := 1) (R := R) (c := c) e) j
      = ∑ r : Fin R, batchSlice R c (batchSlice 1 (R*c) (den e) 0) r j := by
  simp only [den_rowDenseBiasGradB, univ_unique, Fin.default_eq_zero, Fin.isValue, sum_fin_eq_sum_range,
    sum_singleton]
@[simp] theorem den_lnRowBackB {N m n : Nat} (gN xN es : String) (ε γ : ℝ)
    (x : Vec (N*(m*n))) (e : SHlo (N*(m*n))) :
    den (.lnRowBackB gN xN es ε γ x e) = batchMapAux N (rowLNBackFlat m n ε γ) x (den e) := rfl

/-- **`lnRowBackB` hands each example its OWN saved activation** — the property that forced it to be
    a constructor rather than a descriptor, stated so it can be cited instead of re-argued. Example
    `k`'s output block is the per-example backward applied to `batchSlice k x`, never to the whole
    `x` and never to example 0's. A descriptor would give the latter, silently: same types, same
    emitted bytes, different function. -/
theorem den_lnRowBackB_per_example {N m n : Nat} (gN xN es : String) (ε γ : ℝ)
    (x : Vec (N*(m*n))) (e : SHlo (N*(m*n))) (k : Fin N) (i : Fin (m*n)) :
    den (.lnRowBackB gN xN es ε γ x e) (finProdFinEquiv (k, i))
      = rowLNBackFlat m n ε γ (batchSlice N (m*n) x k) (batchSlice N (m*n) (den e) k) i := by
  simp only [den_lnRowBackB, batchMapAux, Equiv.symm_apply_apply]
@[simp] theorem den_sigmoidBackB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.sigmoidBackB xN x e) = (sigmoidHasVJP (N*n)).backward x (den e) := rfl

/-- **`sigmoidB` denotes `Proofs.sigmoid` at the batched index** — `rfl`, the same function
    `sigmoidF_faithful` states one index down. This is BCE-with-logits' only new op. -/
@[simp] theorem sigmoidB_faithful {N n : Nat} (e : SHlo (N*n)) :
    den (.sigmoidB (N := N) (n := n) e) = sigmoid (N*n) (den e) := rfl

/-! ### ViT increment 2 — the six forms that cannot be descriptors -/

@[simp] theorem den_matmulFB {N m k n : Nat} (a : SHlo (N*(m*k))) (b : SHlo (N*(k*n))) :
    den (.matmulFB a b) = batchMapAux N (matMulFlat m k n) (den a) (den b) := rfl

/-- ⭐ **ATTENTION'S MATMUL IS PER-EXAMPLE IN *BOTH* OPERANDS**, which is the property the whole
    `matmulF` scoping worry was about. Example `k`'s output is `Qₖ·Kₖᵀ` — its own `Q` against its
    own `K` — never `Q₀` against `Kₖ`, and never the whole batch flattened into one big matrix.

    ⚠ **All three of those type-check.** At the batched index `N*(m*k)`, a `den` that read the
    index as one matrix would compute `matMulFlat` at the wrong `m` and still be a `Vec`; a
    descriptor would hand every example operand 0's left factor. What separates them is this
    statement, and the emit tie cannot make it — the emitted `dot_general` carries
    `batching_dims = [0] x [0]` in every one of those worlds. -/
theorem den_matmulFB_per_example {N m k n : Nat} (a : SHlo (N*(m*k))) (b : SHlo (N*(k*n)))
    (t : Fin N) (i : Fin (m*n)) :
    den (.matmulFB a b) (finProdFinEquiv (t, i))
      = matMulFlat m k n (batchSlice N (m*k) (den a) t) (batchSlice N (k*n) (den b) t) i := by
  simp only [den_matmulFB, batchMapAux, Equiv.symm_apply_apply]

@[simp] theorem den_softmaxRowBackB {N m n : Nat} (xN : String) (preAct : Vec (N*(m*n)))
    (e : SHlo (N*(m*n))) :
    den (.softmaxRowBackB xN preAct e) = batchMapAux N (rowSoftmaxBackFlat m n) preAct (den e) :=
  rfl

/-- **Each example's softmax backward recomputes from ITS OWN saved scores.** The descriptor
    version would hand all `N` example 0's — same types, same bytes, different function.
    `lnRowBackB`'s statement, on attention. -/
theorem den_softmaxRowBackB_per_example {N m n : Nat} (xN : String) (preAct : Vec (N*(m*n)))
    (e : SHlo (N*(m*n))) (k : Fin N) (i : Fin (m*n)) :
    den (.softmaxRowBackB xN preAct e) (finProdFinEquiv (k, i))
      = rowSoftmaxBackFlat m n (batchSlice N (m*n) preAct k)
          (batchSlice N (m*n) (den e) k) i := by
  simp only [den_softmaxRowBackB, batchMapAux, Equiv.symm_apply_apply]

@[simp] theorem den_posEmbedGradB {N tk D : Nat} (e : SHlo (N*((tk+1)*D))) :
    den (.posEmbedGradB (tk := tk) (D := D) e)
      = fun i => ∑ b : Fin N, batchSlice N ((tk+1)*D) (den e) b i := rfl

@[simp] theorem den_patchEmbedBiasGradB {N tk c : Nat} (e : SHlo (N*((tk+1)*c))) :
    den (.patchEmbedBiasGradB (tk := tk) (c := c) e)
      = fun i => ∑ b : Fin N, ∑ p : Fin tk,
          batchSlice (tk+1) c (batchSlice N ((tk+1)*c) (den e) b) p.succ i := rfl

/-- ⚠⚠ **THE BATCH SUM IS INVISIBLE AT `N = 1`.** At one example the outer `∑ b` has a single term,
    so a render that dropped it type-checks, emits the same bytes and agrees exactly — which is why
    any gate on these four must run at `N > 1`. `den_rowDenseBiasGradB_at_one` says the same thing
    for ConvNeXt's bias gradient; this is ViT's positional embedding, where the shared parameter is
    the whole `(tk+1) × D` table. -/
theorem den_posEmbedGradB_at_one {tk D : Nat} (e : SHlo (1*((tk+1)*D))) (i : Fin ((tk+1)*D)) :
    den (.posEmbedGradB (N := 1) (tk := tk) (D := D) e) i
      = batchSlice 1 ((tk+1)*D) (den e) 0 i := by
  simp only [den_posEmbedGradB, univ_unique, Fin.default_eq_zero, Fin.isValue, sum_singleton]
@[simp] theorem den_addVB {N n : Nat} (a b : SHlo (N*n)) :
    den (.addVB a b) = fun j => den a j + den b j := rfl
@[simp] theorem den_subB {N n : Nat} (a b : SHlo (N*n)) :
    den (.subB a b) = fun j => den a j - den b j := rfl
@[simp] theorem den_bnBatchF {N oc h w : Nat} (gN bN es : String) (ε : ℝ) (γ β : Vec oc)
    (e : SHlo (N * (oc*h*w))) :
    den (.bnBatchF gN bN es ε γ β e) = bnBatchLA N oc h w ε γ β (den e) := rfl

-- ════════════════════════════════════════════════════════════════
-- § `emit`: the linear (Chapter-1) train-step graphs
-- ════════════════════════════════════════════════════════════════

variable {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)

/-- Forward logits graph `@linear_fwd`: `broadcast(b) + dot_general(x, W)`. -/
def fwdGraph : SHlo n := .addBcast "%b0" b (.dotIn "%W0" W (.operand "%x" x))

/-- Dense input-VJP graph (`@linear_back`): `dot_general(dy, W)`. -/
def backGraph (dy : Vec n) : SHlo m := .dotOut "%W0" W (.operand "%dy" dy)

/-- Softmax-CE loss-cotangent graph `softmax(logits) − onehot`. The one-hot is
    a parameter (a graph input `%onehot`); `den` reads it, `pretty` ignores it. -/
def lossCotGraph (oh : Vec n) : SHlo n :=
  .sub (.softmaxDiv (.expe (fwdGraph W b x))) (.operand "%onehot" oh)

-- ════════════════════════════════════════════════════════════════
-- § Semantic half: each emitted graph denotes the proven math
-- ════════════════════════════════════════════════════════════════

/-- **Forward faithfulness.** The forward graph denotes `mnistLinear W b`. -/
theorem fwdGraph_faithful : den (fwdGraph W b x) = mnistLinear W b x := by
  funext j; simp only [fwdGraph, denStepApp, mnistLinear, dense]

/-- **Dense input-VJP faithfulness.** The backward graph denotes the proven
    dense VJP backward `(denseHasVJP W b).backward x = Mat.mulVec W`. -/
theorem backGraph_faithful (dy : Vec n) :
    den (backGraph W dy) = (denseHasVJP W b).backward x dy := by
  funext i; simp only [backGraph, denStepApp, denseHasVJP, Mat.mulVec]

/-- The softmax sub-graph denotes the proven `softmax`. -/
theorem softmaxDiv_expe_faithful (z : Vec n) :
    den (.softmaxDiv (.expe (.operand "%logits" z))) = softmax n z := by
  funext j; simp only [denStepApp, softmax]

/-- **Loss-cotangent faithfulness (spec level).** -/
theorem lossCotGraph_faithful (label : Fin n) :
    den (lossCotGraph W b x (oneHot n label)) = IR.emitLossCot n (mnistLinear W b x) label := by
  funext j
  simp only [lossCotGraph, IR.emitLossCot, denStepApp, oneHot, softmax, fwdGraph_faithful,
             mnistLinear, dense]

/-- **Loss-cotangent faithfulness (to the proven gradient).** Via
    `IR.lossCot_bridge`: the cotangent graph denotes `∂(crossEntropy)/∂logits`
    at the linear logits. -/
theorem lossCotGraph_isCEgrad (label : Fin n) (j : Fin n) :
    den (lossCotGraph W b x (oneHot n label)) j
      = pdiv (fun (z : Vec n) (_ : Fin 1) => crossEntropy n z label)
             (mnistLinear W b x) j 0 := by
  rw [lossCotGraph_faithful]; exact IR.lossCot_bridge n (mnistLinear W b x) label j

-- ── Parameter gradients (per-example; the batch `dot_general`/`reduce`
--    reduce, per the D1 shortcut, to the outer product / the cotangent). ──

/-- Weight-gradient (per-example): the batch-contracting `dot_general`, i.e.
    the outer product `x ⊗ dy`. -/
def wGrad (x : Vec m) (dy : Vec n) : Mat m n := Mat.outer x dy

/-- Bias-gradient (per-example): the batch `reduce`-add is the cotangent. -/
def bGrad (dy : Vec n) : Vec n := dy

theorem wGrad_faithful (dy : Vec n) :
    wGrad x dy = IR.emitWeightGrad x .cotangent dy := rfl

/-- **Weight-grad faithfulness** to the certified ∂/∂W Jacobian. -/
theorem wGrad_isWeightJacobian (dy : Vec n) (i : Fin m) (j : Fin n) :
    wGrad x dy i j
      = ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k :=
  IR.weight_grad_bridge W b x .cotangent dy i j

theorem bGrad_faithful (dy : Vec n) : bGrad dy = IR.emitBiasGrad (.cotangent) dy := rfl

/-- **Bias-grad faithfulness** to the certified ∂/∂b Jacobian. -/
theorem bGrad_isBiasJacobian (dy : Vec n) (i : Fin n) :
    bGrad dy i = ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j :=
  IR.bias_grad_bridge W b x .cotangent dy i

-- ════════════════════════════════════════════════════════════════
-- § SGD update — proven (not trusted) for plain SGD on the linear net
-- ════════════════════════════════════════════════════════════════

/-- The emitted **weight** SGD update `W − lr·(x⊗dy)`, with `dy` the proven
    softmax-CE cotangent. -/
noncomputable def sgdW (lr : ℝ) (label : Fin n) : Mat m n :=
  fun i j => W i j - lr * wGrad x (den (lossCotGraph W b x (oneHot n label))) i j

/-- The emitted **bias** SGD update `b − lr·dy`. -/
noncomputable def sgdB (lr : ℝ) (label : Fin n) : Vec n :=
  fun j => b j - lr * bGrad (den (lossCotGraph W b x (oneHot n label))) j

/-- **SGD weight-step faithfulness.** The emitted update subtracts `lr` times
    the *certified* ∂/∂W Jacobian contracted with the proven loss cotangent —
    plain-SGD optimizer promoted from trusted to proven. -/
theorem sgdW_isCertifiedGradStep (lr : ℝ) (label : Fin n) (i : Fin m) (j : Fin n) :
    sgdW W b x lr label i j
      = W i j - lr * ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k
            * den (lossCotGraph W b x (oneHot n label)) k := by
  unfold sgdW
  rw [wGrad_isWeightJacobian W b x (den (lossCotGraph W b x (oneHot n label))) i j]

/-- **SGD bias-step faithfulness.** Likewise for `b`. -/
theorem sgdB_isCertifiedGradStep (lr : ℝ) (label : Fin n) (j : Fin n) :
    sgdB W b x lr label j
      = b j - lr * ∑ i : Fin n,
          pdiv (fun b' : Vec n => dense W b' x) b j i
            * den (lossCotGraph W b x (oneHot n label)) i := by
  unfold sgdB
  rw [bGrad_isBiasJacobian W b x (den (lossCotGraph W b x (oneHot n label))) j]

-- ════════════════════════════════════════════════════════════════
-- § Chapter 2 — MLP: ReLU + multi-layer composition (semantic)
--
-- The forward adds ReLU (`maximum(·,0)`); the backward chains the proven
-- per-layer VJPs through `select(x>0,·,0)` ReLU masks. ReLU has a kink, so the
-- whole-MLP VJP is *conditional* (`mlpHasVJPAt`, off the kink) — exactly the
-- regime the codegen's subgradient (`relu'(0)=0`) targets. The parameter grads
-- and SGD update reuse the layer-agnostic `wGrad`/`bGrad`/`sgd*` theorems above.
-- ════════════════════════════════════════════════════════════════

/-- `maximum(a,0)` equals ReLU's pointwise `if a>0 then a else 0`. -/
private theorem max_zero_eq (a : ℝ) : max a 0 = if a > 0 then a else 0 := by
  by_cases h : (0 : ℝ) < a
  · rw [ite_eq_left h, max_eq_left h.le]
  · rw [ite_eq_right h, max_eq_right (not_lt.1 h)]

/-- **ReLU forward faithfulness.** `maximum(·,0)` denotes the proven `relu`. -/
theorem reluF_faithful {k : Nat} (e : SHlo k) : den (.reluF e) = relu k (den e) := by
  funext i; simp only [denStepApp, relu]; exact max_zero_eq _

/-- **ReLU backward faithfulness (smooth point).** `select(x>0,·,0)` denotes the
    proven `reluHasVJPAt` backward — the codegen's `relu'(0)=0` convention. -/
theorem selectPos_faithful {k : Nat} (s : String) (x : Vec k) (hx : ∀ i, x i ≠ 0)
    (e : SHlo k) :
    den (.selectPos s x e) = (reluHasVJPAt k x hx).backward (den e) := rfl

/-- The `relu` descriptor denotes exactly what the descriptor-less `reluF` denoted at the same
    index: the batched graph computes the same function, only the emit width now travels
    separately from the batch. The ResNet-34 peer of `den_batchOp_swish_eq_swishF`. -/
theorem den_batchOp_relu_eq_reluF {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.relu (n := n)) e) = den (.reluF e) := by
  rw [reluF_faithful]
  exact batchMap_pointwise (fun y => if y > 0 then y else 0) (den e)

/-- **Batched ReLU backward faithfulness.** `selectPosB` denotes the same proven
    `reluHasVJPAt` backward as `selectPos`, now over the whole batch — which is what the
    emitted `xName` holds. This is the statement that would be FALSE had `selectPos` been made
    a `BatchableOp` descriptor (that `den` would apply one example's mask to all `N`). -/
theorem selectPosB_faithful {N n : Nat} (s : String) (x : Vec (N*n)) (hx : ∀ i, x i ≠ 0)
    (e : SHlo (N*n)) :
    den (.selectPosB s x e) = (reluHasVJPAt (N*n) x hx).backward (den e) := rfl

/-- **ReLU6 forward faithfulness.** `min(max(·,0),6)` denotes the proven `relu6`
    (MLP.lean). (`rfl` — `relu6` is defined as exactly this clamp.) -/
@[simp] theorem relu6F_faithful {k : Nat} (e : SHlo k) :
    den (.relu6F e) = relu6 k (den e) := rfl

/-- **ReLU6 backward faithfulness (smooth point).** `select(0<x<6,·,0)` denotes the
    proven `relu6HasVJPAt` backward — the two-sided kink's mask, smooth iff
    `x≠0 ∧ x≠6` (both bounds, unlike ReLU's one-sided `x≠0`). -/
theorem selectMid_faithful {k : Nat} (s : String) (x : Vec k)
    (h_smooth : ∀ i, x i ≠ 0 ∧ x i ≠ 6) (e : SHlo k) :
    den (.selectMid s x e) = (relu6HasVJPAt k x h_smooth).backward (den e) := rfl

/-- **Batched ReLU6 forward faithfulness (§2f).** The `relu6` descriptor at the batched index
    denotes exactly `relu6F`'s per-example clamp applied across the batch — the MobileNetV2 peer
    of `den_batchOp_relu_eq_reluF`. This is the statement that keeps the emit width off the SHlo
    index: at `N := B` the descriptor emits `tensor<B×n>`, not `tensor<B×(N·n)>`. -/
theorem den_batchOp_relu6_eq_relu6F {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.relu6 (n := n)) e) = den (.relu6F e) := by
  rw [relu6F_faithful]
  exact batchMap_pointwise (fun y => min (max y 0) 6) (den e)

/-- **Batched ReLU6 backward faithfulness.** `selectMidB` denotes the same proven
    `relu6HasVJPAt` backward as `selectMid`, now over the whole batch — which is what the
    emitted `xName` holds. FALSE had `selectMid` been made a `BatchableOp` descriptor beside
    `relu6` (that `den` would apply one example's two-sided mask to all `N`). Note the smoothness
    hypothesis is TWO-sided (`x ≠ 0 ∧ x ≠ 6`), unlike `selectPosB_faithful`'s `x ≠ 0`. -/
theorem selectMidB_faithful {N n : Nat} (s : String) (x : Vec (N*n))
    (h_smooth : ∀ i, x i ≠ 0 ∧ x i ≠ 6) (e : SHlo (N*n)) :
    den (.selectMidB s x e) = (relu6HasVJPAt (N*n) x h_smooth).backward (den e) := rfl

/-- **Stochastic-depth forward faithfulness.** `dropPathB` denotes `Proofs.dropPath`, the per-sample
    residual-branch scale. `rfl`, because `dropPath` is `layerScale` at a per-example-broadcast
    scale and this op is that multiply. -/
theorem dropPathB_faithful {N n : Nat} (mN : String) (s : Vec N) (e : SHlo (N*n)) :
    den (.dropPathB mN s e) = Proofs.dropPath N n s (den e) := rfl

/-- ⭐ **Stochastic-depth BACKWARD faithfulness — and it is the SAME constructor.** A diagonal
    linear map is its own transpose, so the renderer emits `dropPathB` on the cotangent at the same
    scale, and that IS the certified VJP. No `*Grad` peer exists to drift out of step with this one,
    which is the whole reason this feature costs one op rather than two. -/
theorem dropPathB_back_faithful {N n : Nat} (mN : String) (s : Vec N)
    (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.dropPathB mN s e) = (Proofs.dropPathHasVJP N n s).backward x (den e) := rfl

@[simp] theorem den_dropPathB_ones {N n : Nat} (mN : String) (e : SHlo (N*n)) :
    den (.dropPathB mN (fun _ => 1) e) = den e := by
  simp only [den_dropPathB, dropPath_ones_id]

/-- **Classifier-dropout forward faithfulness.** `dropoutB` denotes `Proofs.dropout`, the
    per-ELEMENT inverted mask. `rfl`, because dropout is `layerScale` at a mask of the value's own
    type — no lift, which is what makes it cheaper than `dropPathB` rather than dearer. -/
theorem dropoutB_faithful {N n : Nat} (mN : String) (mask : Vec (N*n)) (e : SHlo (N*n)) :
    den (.dropoutB mN mask e) = Proofs.dropout N n mask (den e) := rfl

/-- ⭐ **Classifier-dropout BACKWARD faithfulness — the SAME constructor**, `dropPathB_back_faithful`
    one mask rank up. ⚠ This covers the cotangent flowing THROUGH the site and nothing else; see
    `Proofs.dropout_vjp_is_self` on the classifier weight gradient, which reads the dense's input
    and must therefore read the DROPPED activation. -/
theorem dropoutB_back_faithful {N n : Nat} (mN : String) (mask : Vec (N*n))
    (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.dropoutB mN mask e) = (Proofs.dropoutHasVJP N n mask).backward x (den e) := rfl

/-- ⭐ **The ones-mask identity on the AST**, which is what licenses emitting the dropout site in
    the FORWARD artifact: `@efficientnet_do_fwd` and `@efficientnet_adamdo_train_step` are then one
    graph differing only in the mask the driver supplies, and the prefix audit survives. -/
@[simp] theorem den_dropoutB_ones {N n : Nat} (mN : String) (e : SHlo (N*n)) :
    den (.dropoutB mN (fun _ => 1) e) = den e := by
  simp only [den_dropoutB, dropout_ones_id]

/-- ⭐⭐ **THE TWO OPS AGREE EXACTLY WHEN THE MASK IS LIFTED, AND THE AST SAYS SO.**
    `Proofs.dropout_of_dropScale` at the node level: a `dropoutB` carrying `dropScale N n s` denotes
    what the `dropPathB` at `s` denotes. This is the substitution that would be a silent regulariser
    swap if it were made in the *other* direction on an unlifted mask, and it is stated here so that
    the containment is checkable rather than argued. -/
theorem den_dropoutB_of_dropScale {N n : Nat} (mN dN : String) (s : Vec N) (e : SHlo (N*n)) :
    den (.dropoutB mN (Proofs.dropScale N n s) e) = den (.dropPathB dN s e) := rfl

/-- A dense forward layer graph: `broadcast(bias) + dot_general(·, W)`. -/
def denseF {a c : Nat} (wN bN : String) (W : Mat a c) (bias : Vec c) (e : SHlo a) : SHlo c :=
  .addBcast bN bias (.dotIn wN W e)

theorem denseF_faithful {a c : Nat} (wN bN : String) (W : Mat a c) (bias : Vec c) (e : SHlo a) :
    den (denseF wN bN W bias e) = dense W bias (den e) := by
  funext j; simp only [denseF, denStepApp, dense]

variable {e₀ e₁ e₂ e₃ : Nat}

/-- Whole-MLP **forward** graph `dense W₂ ∘ relu ∘ dense W₁ ∘ relu ∘ dense W₀`. -/
def mlpFwdGraph (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀) : SHlo e₃ :=
  denseF "%W2" "%b2" W₂ b₂ (.reluF (denseF "%W1" "%b1" W₁ b₁
    (.reluF (denseF "%W0" "%b0" W₀ b₀ (.operand "%x" x)))))

/-- **MLP forward faithfulness.** The forward graph denotes `mlpForward`. -/
theorem mlpFwdGraph_faithful (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀) :
    den (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x) = mlpForward W₀ b₀ W₁ b₁ W₂ b₂ x := by
  simp only [mlpFwdGraph, mlpForward, Function.comp_apply, denseF_faithful, reluF_faithful,
             den_operand]

/-- Whole-MLP **backward** (input-VJP) graph: `dotOut W₀ ∘ select(p₀) ∘
    dotOut W₁ ∘ select(p₁) ∘ dotOut W₂`, `pᵢ` the ReLU pre-activations. -/
def mlpBackGraph (W₀ : Mat e₀ e₁) (W₁ : Mat e₁ e₂) (W₂ : Mat e₂ e₃)
    (p₀ : Vec e₁) (p₁ : Vec e₂) (dy : Vec e₃) : SHlo e₀ :=
  .dotOut "%W0" W₀ (.selectPos "%h0" p₀ (.dotOut "%W1" W₁
    (.selectPos "%h1" p₁ (.dotOut "%W2" W₂ (.operand "%dy" dy)))))

/-- **MLP backward faithfulness (smooth point).** The backward graph denotes
    the proven `mlpHasVJPAt.backward` — the per-op `dot_general`/`select`
    ops assembled into the proven whole-network VJP (cf. `IR.mlp_whole_bridge`). -/
theorem mlpBackGraph_faithful (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu e₁ (dense W₀ b₀ x)) k ≠ 0) (dy : Vec e₃) :
    den (mlpBackGraph W₀ W₁ W₂ (dense W₀ b₀ x)
          (dense W₁ b₁ (relu e₁ (dense W₀ b₀ x))) dy)
      = (mlpHasVJPAt W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1).backward dy := by
  simp only [mlpBackGraph, denStep, denStepApp, mlpHasVJPAt, denseHasVJP, reluHasVJPAt,
             HasVJP.toHasVJPAt, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Chapter 3 — CNN: conv + maxpool (forward, semantic)
--
-- The conv/maxpool *forward* ops, denoted by the proofs' flattened forms
-- `flatConv`/`maxPoolFlat`. The whole MNIST-CNN forward graph denotes the
-- proven `mnistCnnNoBnForward`. (The backward VJP — conv input-grad via the
-- reversed kernel + maxpool select_and_scatter, = `mnistCnnNoBnHasVJPAt` —
-- is the next phase.)
-- ════════════════════════════════════════════════════════════════

/-- **Conv forward faithfulness.** The (flattened) `stablehlo.convolution` op
    denotes the proven `flatConv`. -/
theorem flatConvF_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*h*w)) :
    den (.flatConvF wN bN W b e) = flatConv W b (den e) := rfl

/-- **bf16 conv forward faithfulness.** The bf16 `stablehlo.convolution` op denotes the
    proven `flatConv` on ROUNDED operands, with the accumulated sum rounded and the bias
    added afterwards in f32 — i.e. exactly what the emitted graph computes.

    ⚠ Contrast `flatConvF_faithful`, which has no rounding, and `dotInBf16`, which rounds
    the operands but NOT the result. The outer `rnd` here is not decoration: the emit gives
    the convolution a bf16-typed result, so the hardware stores the accumulator rounded. -/
theorem flatConvFBf16_faithful {ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*h*w)) :
    den (.flatConvFBf16 rnd wN bN W b e)
      = fun i => rnd (flatConv (fun o c kh kw => rnd (W o c kh kw)) 0
                               (fun j => rnd (den e j)) i)
                 + Tensor3.flatten (fun o _ _ => b o) i := rfl

/-- **The bundling is inert at the identity rounding.** At `rnd = id` the bf16 op denotes
    exactly what `flatConvF` does. The `dotInBf16_eq_dotIn_rounded` analogue: it says the op
    adds ROUNDING and nothing else — no reassociation, no dropped bias, no moved padding.
    Without it, "the emit is bf16" and "the emit is the same conv" are two separate hopes. -/
theorem flatConvFBf16_id {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*h*w)) :
    den (.flatConvFBf16 id wN bN W b e) = den (.flatConvF (h := h) (w := w) wN bN W b e) := by
  funext i
  simp only [flatConvFBf16_faithful, flatConvF_faithful, id_eq, flatConv, conv2d,
             Tensor3.flatten, Tensor3.unflatten, Pi.zero_apply, zero_add]
  ring

/-- **Max-pool forward faithfulness.** The (flattened) `reduce_window(max)` op
    denotes the proven `maxPoolFlat`. -/
theorem maxPoolF_faithful {c h w : Nat} (e : SHlo (c*(2*h)*(2*w))) :
    den (.maxPoolF e) = maxPoolFlat c h w (den e) := rfl

/-- ⭐ **3×3/s2 max-pool forward faithfulness.** The (flattened) `reduce_window(max)` op at window
    3, stride 2, symmetric padding 1 denotes the proven `maxPool3s2Flat` — He et al.'s stem pool.
    `planning/archive/rsb_a3_r50_verified.md` §4b. -/
theorem maxPool3s2F_faithful {c h w : Nat} (e : SHlo (c*(2*h)*(2*w))) :
    den (.maxPool3s2F e) = maxPool3s2Flat c h w (den e) := rfl

/-- **Conv backward faithfulness.** The reversed-kernel `stablehlo.convolution`
    (transpose+reverse+conv) denotes the proven conv input-VJP — the flattened
    `conv2dHasVJP3` backward (conv is linear, so this is a global VJP). -/
theorem convBack_faithful {ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*h*w)) (e : SHlo (oc*h*w)) :
    den (.convBack wN W b v e)
      = (HasVJP3.toHasVJP (conv2dHasVJP3 W b)).backward v (den e) := rfl

/-- **Max-pool backward faithfulness (smooth point).** The emitted
    `select_and_scatter` graph denotes the proven `maxPoolFlatHasVJPAt`
    backward — routing the cotangent to each window's argmax (the codegen's
    no-ties convention), under the MaxPool smoothness hypothesis. -/
theorem maxPoolBack_faithful {c h w : Nat} (xN : String) (x : Vec (c*(2*h)*(2*w)))
    (h_smooth : MaxPool2Smooth (Tensor3.unflatten x : Tensor3 c (2*h) (2*w)))
    (e : SHlo (c*h*w)) :
    den (.maxPoolBack xN x e)
      = (maxPoolFlatHasVJPAt (Tensor3.unflatten x) h_smooth).backward (den e) := by
  funext idx
  simp only [denStepApp, maxPoolBackFlat, maxPoolFlatHasVJPAt, HasVJPAt3.toHasVJPAt,
             maxPool2HasVJPAt3]

/-- ⭐ **3×3/s2 max-pool backward faithfulness (smooth point).** The emitted `select_and_scatter`
    graph at window 3 / stride 2 / symmetric padding 1 denotes the proven
    `maxPool3s2FlatHasVJPAt` backward, under `MaxPool3s2Smooth`.

    ⚠ The hypothesis is stated over **positions**, not window offsets, and that is not a stylistic
    difference from `maxPoolBack_faithful`: with overlapping windows two offsets can name one input
    cell (the clamped duplicate at the first window), where the values are equal by construction
    and smoothness must say nothing. `maxPool2` has no analogue because there distinct offsets
    always meant distinct positions. See `MaxPool3s2.lean`'s header. -/
theorem maxPool3s2Back_faithful {c h w : Nat} (xN : String) (x : Vec (c*(2*h)*(2*w)))
    (h_smooth : MaxPool3s2Smooth (Tensor3.unflatten x : Tensor3 c (2*h) (2*w)))
    (e : SHlo (c*h*w)) :
    den (.maxPool3s2Back xN x e)
      = (maxPool3s2FlatHasVJPAt (Tensor3.unflatten x) h_smooth).backward (den e) := by
  funext idx
  simp only [denStepApp, maxPool3s2BackFlat, maxPool3s2FlatHasVJPAt, HasVJPAt3.toHasVJPAt,
             maxPool3s2HasVJPAt3]

/-- **BN forward faithfulness.** The per-example reduce/normalize/affine graph
    (γ·(x−μ)·istd + β, μ/var over the feature axis) denotes the proven
    `bnForward` (BatchNorm.lean). -/
@[simp] theorem bnF_faithful {n : Nat} (gN bN es : String) (ε γ β : ℝ) (e : SHlo n) :
    den (.bnF gN bN es ε γ β e) = bnForward n ε γ β (den e) := rfl

/-- **Residual-add faithfulness** (= `den_addV`). The binary `stablehlo.add`
    denotes pointwise vector addition — the fan-in of a residual/skip
    connection. (`rfl`, so kept out of the axiom audit.) -/
theorem addV_faithful {n : Nat} (a b : SHlo n) :
    den (.addV a b) = fun j => den a j + den b j := rfl

/-- **Global-average-pool faithfulness.** The reduce-over-spatial / ÷h·w graph
    denotes the proven `globalAvgPoolFlat` (CNN.lean). -/
@[simp] theorem gapF_faithful {c h w : Nat} (e : SHlo (c*h*w)) :
    den (.gapF e) = globalAvgPoolFlat c h w (den e) := rfl

/-- **Strided-conv forward faithfulness.** The `window_strides=[2,2]`
    `stablehlo.convolution` denotes the proven `flatConvStride2`
    (= decimate ∘ stride-1 conv, StridedConv.lean). -/
@[simp] theorem flatConvStridedF_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*(2*h)*(2*w))) :
    den (.flatConvStridedF wN bN W b e) = flatConvStride2 W b (den e) := rfl
/-- The XLA-`SAME` peer's faithfulness. ⚠ `flatConvStride2Xla`, NOT `flatConvStride2` — identical
    types, so this `rfl` is the only place the distinction is recorded. -/
@[simp] theorem flatConvStridedXlaF_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*(2*h)*(2*w))) :
    den (.flatConvStridedXlaF wN bN W b e) = flatConvStride2Xla W b (den e) := rfl

/-- **Strided-conv input-VJP faithfulness.** The zero-upsample (`lhs_dilation`)
    + reversed-kernel conv denotes the proven `flatConvStride2HasVJP` backward. -/
theorem convStridedBack_faithful {ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*(2*h)*(2*w))) (e : SHlo (oc*h*w)) :
    den (.convStridedBack wN W b v e) = (flatConvStride2HasVJP W b).backward v (den e) := rfl

/-- **Stride-4 conv forward faithfulness.** The `window_strides=[4,4]`
    `stablehlo.convolution` (the ConvNeXt 4×4/s4 patchify stem) denotes the proven
    `flatConvStride4` (= decimate ∘ decimate ∘ stride-1 conv, StridedConv.lean). -/
@[simp] theorem flatConvStride4F_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*(2*(2*h))*(2*(2*w)))) :
    den (.flatConvStride4F wN bN W b e) = flatConvStride4 W b (den e) := rfl

/-- The scalar-BN backward node denotes the grad-input helper at its cotangent. -/
@[simp] theorem den_bnBack {n : Nat} (gN xN es : String) (ε γ : ℝ) (x : Vec n) (e : SHlo n) :
    den (.bnBack gN xN es ε γ x e) = bnGradInput n ε γ x (den e) := rfl

/-- **BN backward faithfulness.** The consolidated three-term graph denotes the
    proven BN input-VJP — equal to the `pdiv`-contracted Jacobian of `bnForward`
    (`bn_input_grad_correct`), under `0 < ε`. β-independent (a constant shift
    does not enter the Jacobian). -/
theorem bnBack_faithful {n : Nat} (gN xN es : String) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (e : SHlo n) (i : Fin n) :
    den (.bnBack gN xN es ε γ x e) i
      = ∑ j : Fin n, pdiv (bnForward n ε γ β) x i j * den e j := by
  rw [den_bnBack]
  exact bn_input_grad_correct n ε γ β hε x (den e) i

/-- **Per-channel BN forward faithfulness.** The 4-D reshape + per-channel
    reduce/normalize (μ/var over the spatial axes `[2,3]`, rank-1 γ/β `dims=[1]`)
    denotes the proven `bnPerChannelTensor3` (PerChannelBN.lean). (`rfl`, so kept
    out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem bnPerChannelF_faithful {oc h w : Nat} (gN bN es : String) (ε : ℝ)
    (γ β : Vec oc) (e : SHlo (oc*h*w)) :
    den (.bnPerChannelF gN bN es ε γ β e) = bnPerChannelTensor3 oc h w ε γ β (den e) := rfl

-- ════════════════════════════════════════════════════════════════
-- § Param gradients + AdamW: faithfulness, and consistency with the SGD ops
-- ════════════════════════════════════════════════════════════════

/-- **Dense weight-gradient faithfulness** — the outer product `xᵢ·dyⱼ`. -/
@[simp] theorem weightGrad_faithful {m n : Nat} (xN : String) (x : Vec m) (e : SHlo n) :
    den (.weightGrad xN x e) = Mat.flatten (fun i j => x i * den e j) := rfl

/-- **Conv weight-gradient faithfulness** — the proven `conv2dWeightGrad` VJP. -/
@[simp] theorem convWeightGrad_faithful {ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (e : SHlo (oc*h*w)) :
    den (.convWeightGrad xN b x W e)
      = (conv2dWeightGradHasVJP b x).backward (Kernel4.flatten W) (den e) := rfl

/-- **Conv bias-gradient faithfulness** — the proven `conv2dBiasGrad` VJP. -/
@[simp] theorem convBiasGrad_faithful {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (e : SHlo (oc*h*w)) :
    den (.convBiasGrad W x b e) = (conv2dBiasGradHasVJP W x).backward b (den e) := rfl

/-! **The gradient ops agree with the SGD ops they were split out of.** Each says
`den (θSgd …) = θ − lr · den (θGrad …)` coordinatewise — so un-fusing the update did not
quietly change the gradient, and anything already proven about a `*Sgd` output transfers to
`θ − lr·(*Grad)`. All `rfl`: the `*Grad` `den` is literally the subterm. -/

@[simp] theorem weightSgd_eq_grad {m n : Nat} (xN wN lrS : String) (x : Vec m) (W : Mat m n)
    (lr : ℝ) (e : SHlo n) (idx : Fin (m*n)) :
    den (.weightSgd xN wN lrS x W lr e) idx
      = Mat.flatten W idx - lr * den (.weightGrad xN x e) idx := rfl

/-! The TRANSFORMER peers of the same statement — the ViT family §2a left fused, which is why
`vit_adam_train_step` had no certified render until these existed. Same `rfl` discipline. -/

@[simp] theorem rowDenseWeightSgd_eq_grad {N a c : Nat} (xN wN lrS : String) (x : Vec (N*a))
    (W : Mat a c) (lr : ℝ) (e : SHlo (N*c)) (idx : Fin (a*c)) :
    den (.rowDenseWeightSgd xN wN lrS x W lr e) idx
      = Mat.flatten W idx - lr * den (.rowDenseWeightGrad xN x e) idx := rfl

@[simp] theorem rowDenseBiasSgd_eq_grad {N c : Nat} (bN lrS : String) (b : Vec c) (lr : ℝ)
    (e : SHlo (N*c)) (j : Fin c) :
    den (.rowDenseBiasSgd bN lrS b lr e) j = b j - lr * den (.rowDenseBiasGrad e) j := rfl

@[simp] theorem veclnGammaSgd_eq_grad {N D : Nat} (gN xN esS lrS : String) (ε : ℝ) (x : Vec (N*D))
    (γ : Vec D) (lr : ℝ) (e : SHlo (N*D)) (k : Fin D) :
    den (.veclnGammaSgd gN xN esS lrS ε x γ lr e) k
      = γ k - lr * den (.veclnGammaGrad xN esS ε x e) k := rfl

@[simp] theorem patchEmbedWeightSgd_eq_grad {ic H W P N D : Nat} (wN xN lrS : String)
    (x : Vec (ic*H*W)) (Wp : Kernel4 D ic P P) (lr : ℝ) (e : SHlo ((N+1)*D))
    (idx : Fin (D*ic*P*P)) :
    den (.patchEmbedWeightSgd wN xN lrS x Wp lr e) idx
      = Kernel4.flatten Wp idx
        - lr * den (.patchEmbedWeightGrad (N := N) xN x e) idx := rfl

@[simp] theorem patchEmbedBiasSgd_eq_grad {N c : Nat} (bN lrS : String) (b : Vec c) (lr : ℝ)
    (e : SHlo ((N+1)*c)) (i : Fin c) :
    den (.patchEmbedBiasSgd bN lrS b lr e) i
      = b i - lr * den (.patchEmbedBiasGrad (N := N) e) i := rfl

@[simp] theorem depthwiseWeightSgdB_eq_grad {N c h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec c) (x : Vec (N*(c*h*w))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
    (e : SHlo (N*(c*h*w))) (idx : Fin (c*kH*kW)) :
    den (.depthwiseWeightSgdB xN wN lrS b x W lr e) idx
      = Tensor3.flatten W idx - lr * den (.depthwiseWeightGradB xN b x W e) idx := rfl

@[simp] theorem depthwiseStridedWeightSgdB_eq_grad {N c h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec c) (x : Vec (N*(c*(2*h)*(2*w)))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
    (e : SHlo (N*(c*h*w))) (idx : Fin (c*kH*kW)) :
    den (.depthwiseStridedWeightSgdB xN wN lrS b x W lr e) idx
      = Tensor3.flatten W idx - lr * den (.depthwiseStridedWeightGradB xN b x W e) idx := rfl

/-! ### The depthwise BIAS gradients (§2f, MobileNetV2)

`MobileNetV2RenderB` is AdamW-only, like `ResNet34RenderB` — mnv2's SGD render stays at the
per-example index, so there is deliberately no fused `depthwise{,Strided}BiasSgdB` peer and hence
no `*SgdB_eq_grad` statement to make. What pins these two ops instead is that `den` IS the
shared-parameter batch sum of the proven per-example depthwise bias VJP, which is what the emitted
`reduce … [0, 2, 3]` computes. The emit side is covered separately by the byte-PREFIX case in
[`tests/TestBatchedEmitTie.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestBatchedEmitTie.lean) against the per-example fused `depthwiseBiasSgd`. -/

@[simp] theorem depthwiseBiasGradB_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (N*(c*h*w))) (b : Vec c)
    (e : SHlo (N*(c*h*w))) (o : Fin c) :
    den (.depthwiseBiasGradB W x b e) o
      = ∑ n : Fin N,
          (depthwiseBiasGradHasVJP W (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward b
            (batchSlice N (c*h*w) (den e) n) o := rfl

@[simp] theorem depthwiseStridedBiasGradB_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (N*(c*(2*h)*(2*w)))) (b : Vec c)
    (e : SHlo (N*(c*h*w))) (o : Fin c) :
    den (.depthwiseStridedBiasGradB W x b e) o
      = ∑ n : Fin N,
          (depthwiseStride2BiasGradHasVJP W (batchSlice N (c*(2*h)*(2*w)) x n)).backward b
            (batchSlice N (c*h*w) (den e) n) o := rfl

/-! ## The ConvNeXt five — same statement, the last `*Sgd`/`*Grad` pairs the kit was missing (§2f)

`den (xSgd …) = θ − lr · den (xGrad …)`, all `rfl`. Together with the emit-side byte-PREFIX checks
in [`tests/TestBatchedEmitTie.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestBatchedEmitTie.lean) this is what lets `convnext_adam_train_step` hand its gradients
to `adamWParamF` instead of to the SGD tail — the fusion was the blocker, never Adam (§2a). -/

@[simp] theorem depthwiseWeightSgd_eq_grad {c h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW) (lr : ℝ)
    (e : SHlo (c*h*w)) (idx : Fin (c*kH*kW)) :
    den (.depthwiseWeightSgd xN wN lrS b x W lr e) idx
      = Tensor3.flatten W idx - lr * den (.depthwiseWeightGrad xN b x W e) idx := rfl

@[simp] theorem depthwiseBiasSgd_eq_grad {c h w kH kW : Nat} (bN lrS : String)
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c) (lr : ℝ)
    (e : SHlo (c*h*w)) (o : Fin c) :
    den (.depthwiseBiasSgd bN lrS W x b lr e) o
      = b o - lr * den (.depthwiseBiasGrad W x b e) o := rfl

@[simp] theorem lnGammaSgd_eq_grad {n : Nat} (gN xN es lrS : String) (ε : ℝ) (x : Vec n)
    (γ : Vec 1) (lr : ℝ) (e : SHlo n) (c : Fin 1) :
    den (.lnGammaSgd gN xN es lrS ε x γ lr e) c
      = γ 0 - lr * den (.lnGammaGrad xN es ε x e) c := rfl

@[simp] theorem lnBetaSgd_eq_grad {n : Nat} (bN lrS : String) (β : Vec 1) (lr : ℝ)
    (e : SHlo n) (c : Fin 1) :
    den (.lnBetaSgd bN lrS β lr e) c = β 0 - lr * den (.lnBetaGrad e) c := rfl

@[simp] theorem layerScaleChGammaSgd_eq_grad {c h w : Nat} (gN xN lrS : String)
    (x : Vec (c*h*w)) (γ : Vec c) (lr : ℝ) (e : SHlo (c*h*w)) (cc : Fin c) :
    den (.layerScaleChGammaSgd gN xN lrS x γ lr e) cc
      = γ cc - lr * den (.layerScaleChGammaGrad xN x e) cc := rfl

@[simp] theorem posEmbedSgd_eq_grad {N D : Nat} (pN lrS : String) (pos : Mat (N+1) D) (lr : ℝ)
    (e : SHlo ((N+1)*D)) (i : Fin ((N+1)*D)) :
    den (.posEmbedSgd pN lrS pos lr e) i
      = Mat.flatten pos i - lr * den (.posEmbedGrad e) i := rfl

@[simp] theorem biasSgd_eq_grad {n : Nat} (bN lrS : String) (b : Vec n) (lr : ℝ)
    (e : SHlo n) (j : Fin n) :
    den (.biasSgd bN lrS b lr e) j = b j - lr * den (.biasGrad e) j := rfl

@[simp] theorem convWeightSgd_eq_grad {ic oc h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (lr : ℝ)
    (e : SHlo (oc*h*w)) (idx : Fin (oc*ic*kH*kW)) :
    den (.convWeightSgd xN wN lrS b x W lr e) idx
      = Kernel4.flatten W idx - lr * den (.convWeightGrad xN b x W e) idx := rfl

@[simp] theorem convBiasSgd_eq_grad {ic oc h w kH kW : Nat} (bN lrS : String)
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (lr : ℝ)
    (e : SHlo (oc*h*w)) (o : Fin oc) :
    den (.convBiasSgd bN lrS W x b lr e) o
      = b o - lr * den (.convBiasGrad W x b e) o := rfl

/-! The strided + BN peers, same shape: `den (xSgd …) = θ − lr · den (xGrad …)`, all `rfl`. -/

@[simp] theorem convStridedWeightSgd_eq_grad {ic oc h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec oc) (x : Vec (ic*(2*h)*(2*w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
    (e : SHlo (oc*h*w)) (idx : Fin (oc*ic*kH*kW)) :
    den (.convStridedWeightSgd xN wN lrS b x W lr e) idx
      = Kernel4.flatten W idx - lr * den (.convStridedWeightGrad xN b x W e) idx := rfl

/-- **Stride-4 weight-gradient faithfulness** (ConvNeXt's patchify stem). `den` IS the proven
    `flatConvStride4WeightGradHasVJP` backward. There is no fused `convStride4WeightSgd` peer —
    nothing but ConvNeXt's stem is stride-4, and its AdamW render consumes the un-fused gradient
    directly — so this, not a `*Sgd_eq_grad` statement, is what pins the op's `den`. -/
@[simp] theorem convStride4WeightGrad_faithful {ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (oc*h*w)) :
    den (.convStride4WeightGrad xN b x W e)
      = (flatConvStride4WeightGradHasVJP b x).backward (Kernel4.flatten W) (den e) := rfl

@[simp] theorem convStridedBiasSgd_eq_grad {ic oc h w kH kW : Nat} (bN lrS : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w))) (b : Vec oc) (lr : ℝ)
    (e : SHlo (oc*h*w)) (o : Fin oc) :
    den (.convStridedBiasSgd bN lrS W x b lr e) o
      = b o - lr * den (.convStridedBiasGrad W x b e) o := rfl

@[simp] theorem bnGammaSgd_eq_grad {oc h w : Nat} (gN vN es lrS : String) (ε : ℝ) (γ : Vec oc)
    (v : Vec (oc*h*w)) (lr : ℝ) (e : SHlo (oc*h*w)) (c : Fin oc) :
    den (.bnGammaSgd gN vN es lrS ε γ v lr e) c
      = γ c - lr * den (.bnGammaGrad vN es ε v e) c := rfl

@[simp] theorem bnBetaSgd_eq_grad {oc h w : Nat} (bN lrS : String) (β : Vec oc) (lr : ℝ)
    (e : SHlo (oc*h*w)) (c : Fin oc) :
    den (.bnBetaSgd bN lrS β lr e) c = β c - lr * den (.bnBetaGrad (h := h) (w := w) e) c := rfl

/-! ## `den (xSgdB …) = θ − lr · den (xGradB …)` — the BATCHED peers of the `*Sgd_eq_grad` set

All `rfl`, and all carrying the same content as §2a's per-example eight: the fused `*SgdB` op IS
`θ − lr·` applied to the un-fused gradient, so handing the gradient to AdamW instead of to the SGD
tail changes nothing about what is computed. This is what unblocks a batched `resnet34_adam_train_step`
rendered from `Proofs/` — the blocker was the fusion, never Adam. -/

@[simp] theorem convWeightSgdB_eq_grad {N ic oc h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec oc) (x : Vec (N*(ic*h*w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
    (e : SHlo (N*(oc*h*w))) (idx : Fin (oc*ic*kH*kW)) :
    den (.convWeightSgdB xN wN lrS b x W lr e) idx
      = Kernel4.flatten W idx - lr * den (.convWeightGradB xN b x W e) idx := rfl

@[simp] theorem convStridedWeightSgdB_eq_grad {N ic oc h w kH kW : Nat} (xN wN lrS : String)
    (b : Vec oc) (x : Vec (N*(ic*(2*h)*(2*w)))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
    (e : SHlo (N*(oc*h*w))) (idx : Fin (oc*ic*kH*kW)) :
    den (.convStridedWeightSgdB xN wN lrS b x W lr e) idx
      = Kernel4.flatten W idx - lr * den (.convStridedWeightGradB xN b x W e) idx := rfl

@[simp] theorem convBiasSgdB_eq_grad {N ic oc h w kH kW : Nat} (bN lrS : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N*(ic*h*w))) (b : Vec oc) (lr : ℝ)
    (e : SHlo (N*(oc*h*w))) (o : Fin oc) :
    den (.convBiasSgdB bN lrS W x b lr e) o
      = b o - lr * den (.convBiasGradB (h := h) (w := w) W x b e) o := rfl

@[simp] theorem convStridedBiasSgdB_eq_grad {N ic oc h w kH kW : Nat} (bN lrS : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N*(ic*(2*h)*(2*w)))) (b : Vec oc) (lr : ℝ)
    (e : SHlo (N*(oc*h*w))) (o : Fin oc) :
    den (.convStridedBiasSgdB bN lrS W x b lr e) o
      = b o - lr * den (.convStridedBiasGradB (h := h) (w := w) W x b e) o := rfl

@[simp] theorem bnGammaSgdB_eq_grad {N oc h w : Nat} (gN vN es lrS : String) (ε : ℝ) (γ : Vec oc)
    (v : Vec (N*(oc*(h*w)))) (lr : ℝ) (e : SHlo (N*(oc*(h*w)))) (c : Fin oc) :
    den (.bnGammaSgdB gN vN es lrS ε γ v lr e) c
      = γ c - lr * den (.bnGammaGradB vN es ε v e) c := rfl

@[simp] theorem bnBetaSgdB_eq_grad {N oc h w : Nat} (bN lrS : String) (β : Vec oc) (lr : ℝ)
    (e : SHlo (N*(oc*(h*w)))) (c : Fin oc) :
    den (.bnBetaSgdB bN lrS β lr e) c
      = β c - lr * den (.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) e) c := rfl

@[simp] theorem denseWeightSgdB_eq_grad {N a c : Nat} (xN wN lrS : String) (x : Vec (N*a))
    (W : Mat a c) (lr : ℝ) (e : SHlo (N*c)) (idx : Fin (a*c)) :
    den (.denseWeightSgdB xN wN lrS x W lr e) idx
      = Mat.flatten W idx - lr * den (.denseWeightGradB (c := c) xN x e) idx := rfl

@[simp] theorem denseBiasSgdB_eq_grad {N c : Nat} (bN lrS : String) (b : Vec c) (lr : ℝ)
    (e : SHlo (N*c)) (j : Fin c) :
    den (.denseBiasSgdB bN lrS b lr e) j
      = b j - lr * den (.denseBiasGradB (N := N) e) j := rfl

/-- **AdamW first-moment faithfulness** — `m' = β₁·m + (1−β₁)·g`, the proven `adamMNext`. -/
@[simp] theorem adamMNextF_faithful {n : Nat} (mN b1N ob1N : String) (ds : List Nat)
    (β₁ : ℝ) (m : Vec n) (e : SHlo n) :
    den (.adamMNextF mN b1N ob1N ds β₁ m e) = adamMNext β₁ m (den e) := rfl

/-- **AdamW second-moment faithfulness** — `v' = β₂·v + (1−β₂)·g²`, the proven `adamVNext`. -/
@[simp] theorem adamVNextF_faithful {n : Nat} (vN b2N ob2N : String) (ds : List Nat)
    (β₂ : ℝ) (v : Vec n) (e : SHlo n) :
    den (.adamVNextF vN b2N ob2N ds β₂ v e) = adamVNext β₂ v (den e) := rfl

/-- **AdamW parameter-step faithfulness.** The emitted 26-op block denotes exactly
    `Proofs.adamWParam` of the child's gradient — the theorem that moves the optimizer from a
    trusted hand-written emitter (`ViTRender.emitAdamV`, which only *claimed* to be op-for-op
    `adamWParam`) into the proven kit. Well-definedness of the `√v̂ + ε` denominator is
    `Proofs.adam_denom_pos`; there is deliberately no descent claim, because Adam is not a
    monotone descent method (AMSGrad counterexample). -/
@[simp] theorem adamWParamF_faithful {n : Nat}
    (θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN : String) (ds : List Nat)
    (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n) (e : SHlo n) :
    den (.adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
          β₁ β₂ ε lr wd bc₁ bc₂ θ m v e)
      = adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v (den e) := rfl

/-- **The rendered AdamW triple is `Proofs.adamWStep`.** Bundles the three ops into the
    `(θ', m', v')` a train step returns per parameter — the whole optimizer, denoted. -/
theorem adamW_triple_faithful {n : Nat}
    (θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN : String) (ds : List Nat)
    (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n) (e : SHlo n) :
    (den (.adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
            β₁ β₂ ε lr wd bc₁ bc₂ θ m v e),
     den (.adamMNextF mN b1N ob1N ds β₁ m e),
     den (.adamVNextF vN b2N ob2N ds β₂ v e))
      = adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v (den e) := rfl

/-- **Rendered plain SGD is `Proofs.sgdParam`** — `θ − lr·g` with `lr` a runtime arg. -/
@[simp] theorem sgdParamF_faithful {n : Nat} (θN lrN : String) (ds : List Nat)
    (lr : ℝ) (θ : Vec n) (e : SHlo n) :
    den (.sgdParamF θN lrN ds lr θ e) = sgdParam lr θ (den e) := rfl

/-- **Rendered Nesterov velocity is `Proofs.momVNext`** — `v' = μ·v + g`. -/
@[simp] theorem momVNextF_faithful {n : Nat} (vN muN : String) (ds : List Nat)
    (μ : ℝ) (v : Vec n) (e : SHlo n) :
    den (.momVNextF vN muN ds μ v e) = momVNext μ v (den e) := rfl

/-- **Rendered Nesterov update is `Proofs.momParam`** — `θ' = θ − lr·(g + μ·v')`. -/
@[simp] theorem momParamF_faithful {n : Nat} (θN vN muN lrN : String) (ds : List Nat)
    (μ lr : ℝ) (θ v : Vec n) (e : SHlo n) :
    den (.momParamF θN vN muN lrN ds μ lr θ v e) = momParam μ lr θ v (den e) := rfl

/-- **The rendered Nesterov pair is `Proofs.momStep`.** The momentum analogue of
    `adamW_triple_faithful`: the `(θ', v')` a momentum train step returns per parameter, denoted.
    The `m` slot is a passthrough and so appears nowhere here — that is the packed-`[θ|m|v]`
    signature being shared verbatim with the AdamW render, not an omission. -/
theorem mom_pair_faithful {n : Nat} (θN vN muN lrN : String) (ds : List Nat)
    (μ lr : ℝ) (θ v : Vec n) (e : SHlo n) :
    (den (.momParamF θN vN muN lrN ds μ lr θ v e), den (.momVNextF vN muN ds μ v e))
      = momStep μ lr θ v (den e) := rfl

/-- **`μ = 0` makes the rendered Nesterov update the rendered SGD update.** Ties the two new op
    families to each other at the denotation level, so the `mom` and `sgd` renders provably agree
    in the limit rather than merely looking similar. -/
theorem momParamF_mu_zero {n : Nat} (θN vN muN lrN : String) (ds : List Nat)
    (lr : ℝ) (θ v : Vec n) (e : SHlo n) :
    den (.momParamF θN vN muN lrN ds 0 lr θ v e) = den (.sgdParamF θN lrN ds lr θ e) := by
  simp only [momParamF_faithful, sgdParamF_faithful]
  exact momParam_mu_zero lr θ v (den e)

/-- **Rendered RMSProp buffer is `Proofs.rmsBufNext`** — `b' = μ·b + g/√(ρ·s + (1−ρ)·g² + ε)`,
    TensorFlow's ε placement. -/
@[simp] theorem rmsBufNextF_faithful {n : Nat} (sqN bufN rhoN orhoN muN epsN : String)
    (ds : List Nat) (ρ μ ε : ℝ) (sq buf : Vec n) (e : SHlo n) :
    den (.rmsBufNextF sqN bufN rhoN orhoN muN epsN ds ρ μ ε sq buf e)
      = rmsBufNext ρ μ ε sq buf (den e) := rfl

/-- **The rendered RMSProp triple is `Proofs.rmsPropStep`.** The RMSProp analogue of
    `adamW_triple_faithful` / `mom_pair_faithful`: `(θ', b', s')` as the three ops the render
    actually emits, denoted together.

    ▶ **Read the composition off this statement** — it is the whole "one new op" claim, checked:
    the parameter slot is `sgdParamF` applied to *this op's SSA output* (`.operand`, so the buffer
    is emitted once and threaded, per §4's no-CSE rule), and the mean-square slot is the EXISTING
    `adamVNextF` at `β₂ := ρ`. Only `rmsBufNextF` is new. -/
theorem rmsProp_triple_faithful {n : Nat} (θN sqN bufN rhoN orhoN muN epsN lrN : String)
    (ds : List Nat) (ρ μ ε lr : ℝ) (θ sq buf : Vec n) (e : SHlo n) (b' : Vec n)
    (hb : b' = rmsBufNext ρ μ ε sq buf (den e)) :
    (den (.sgdParamF θN lrN ds lr θ (.operand "%buf" b')),
     den (.rmsBufNextF sqN bufN rhoN orhoN muN epsN ds ρ μ ε sq buf e),
     den (.adamVNextF sqN rhoN orhoN ds ρ sq e))
      = rmsPropStep ρ μ ε lr θ sq buf (den e) := by
  subst hb; rfl

/-- **The mean-square slot really is the Adam op.** `adamVNextF` at `β₂ := ρ` denotes RMSProp's
    `s'`, so reusing it is licensed rather than assumed — the emit-side twin of
    `Proofs.rmsSqNext_eq_adamVNext`, and the reason this optimizer cost ONE op and not three. -/
theorem adamVNextF_as_rmsSqNext {n : Nat} (sqN rhoN orhoN : String) (ds : List Nat)
    (ρ : ℝ) (sq : Vec n) (e : SHlo n) :
    den (.adamVNextF sqN rhoN orhoN ds ρ sq e) = rmsSqNext ρ sq (den e) := rfl

/-- **`μ = 0` makes the rendered RMSProp buffer the bare normalised gradient.** The `mu_zero`
    bridge `momParamF_mu_zero` provides for Nesterov, at the denotation level. -/
theorem rmsBufNextF_mu_zero {n : Nat} (sqN bufN rhoN orhoN muN epsN : String)
    (ds : List Nat) (ρ ε : ℝ) (sq buf : Vec n) (e : SHlo n) :
    den (.rmsBufNextF sqN bufN rhoN orhoN muN epsN ds ρ 0 ε sq buf e)
      = fun i => (den e) i / Real.sqrt (rmsSqNext ρ sq (den e) i + ε) := by
  simp only [rmsBufNextF_faithful]
  exact rmsBufNext_mu_zero ρ ε sq buf (den e)

-- ════════════════════════════════════════════════════════════════
-- § Global-norm gradient clipping — faithfulness (`GradClip.lean`, `planning/archive/grad_clip.md`)
-- ════════════════════════════════════════════════════════════════

/-- **The scalar fold is `Proofs.gradSumSq` accumulated** — `acc + ∑ᵢ gᵢ²` for one parameter,
    reduced to a rank-0 scalar. `SHlo 1` denoting a rank-0 `tensor<f32>` is `lnBetaGrad`'s
    established reading, not a new convention.

    ▶ **This op is what makes the global reduction an ordinary `SHlo` TREE.** The norm reads like a
    shared DAG node — one scalar consumed by 200 sites — and `SHlo` is a tree; the resolution is
    that `SHlo` is single-OUTPUT, not single-INPUT, so folding 200 subtrees into one scalar is just
    a left-nested chain of this constructor, seeded at `%zero`. Nothing is recomputed, because every
    gradient it consumes is already an `.operand` leaf. -/
@[simp] theorem gradSumSqAccF_faithful {n : Nat} (ds : List Nat) (acc : SHlo 1) (e : SHlo n) :
    den (.gradSumSqAccF ds acc e) = fun _ => scalarOf (den acc) + gradSumSq (den e) := rfl

/-- **`lambDirF` denotes `Proofs.lambDir`** — `rfl`, i.e. the rendered LAMB direction IS the ℝ
    definition, structurally. Same bar as `adamWParamF_faithful`. -/
@[simp] theorem lambDirF_faithful {n : Nat}
    (θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN : String) (ds : List Nat)
    (β₁ β₂ ε wd bc₁ bc₂ : ℝ) (θ m v : Vec n) (e : SHlo n) :
    den (.lambDirF θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN ds β₁ β₂ ε wd bc₁ bc₂ θ m v e)
      = lambDir β₁ β₂ ε wd bc₁ bc₂ θ m v (den e) := rfl

/-- **`lambScaleF` denotes `Proofs.lambScale`.** ⚠ The trust ratio is computed from THIS tensor's
    own norm, which is what makes it layer-wise; `clipScaleF`'s factor is shared across every
    parameter. The two ops look alike and differ in exactly that quantifier. -/
@[simp] theorem lambScaleF_faithful {n : Nat} (ds : List Nat) (s : SHlo 1) (e : SHlo n) :
    den (.lambScaleF ds s e) = lambScale (scalarOf (den s)) (den e) := rfl

/-- **The rescale is `Proofs.clipScale` at `Proofs.clipFactor` of the summed total** — the
    reference's `g * jnp.minimum(1.0, CLIP / (gn + 1e-6))` with `gn = sqrt(total)`.

    ⚠ The factor is derived from the op's FIRST CHILD, the already-summed global total, so this
    constructor cannot express a per-parameter clip: it never receives enough to compute one. The
    `c`/`ε` ℝ fields pair with `clipStr`/`epsStr` exactly as `bnF`'s `ε`/`epsStr` do.
    ⚠ The factor is recomputed at every site rather than emitted once and threaded, for
    `adamWParamF`'s reason — `SHlo` is single-result, so each output is its own node, and XLA's CSE
    folds the duplicates (§2b-bis measured that on R34's 108 → 36 rsqrt at no run-time cost). -/
@[simp] theorem clipScaleF_faithful {n : Nat} (clipS epsS : String) (c ε : ℝ) (ds : List Nat)
    (s : SHlo 1) (e : SHlo n) :
    den (.clipScaleF clipS epsS c ε ds s e)
      = clipScale (clipFactor c ε (scalarOf (den s))) (den e) := rfl

/-- ▶ **THE WHOLE CLIP, END TO END, FOR TWO PARAMETERS — this is the transcription check.**

    Read the reference's two lines off the right-hand side: `gn = √(Σ_leaves Σ g²)` folded from
    `%zero`, then `g * min(1, CLIP/(gn + 1e-6))`. Stated at TWO parameters because one cannot
    exhibit the property that matters — see `clipShared_faithful`. Holds by `rfl`. -/
theorem clipGrad_faithful {n m : Nat} (dsN dsM : List Nat) (clipS epsS : String) (c ε : ℝ)
    (gN : SHlo n) (gM : SHlo m) :
    den (.clipScaleF clipS epsS c ε dsN
          (.gradSumSqAccF dsM (.gradSumSqAccF dsN (.operand "%zero" (fun _ => 0)) gN) gM) gN)
      = clipGrad c ε (0 + gradSumSq (den gN) + gradSumSq (den gM)) (den gN) := rfl

/-- ▶⚠ **THE FACTOR IS SHARED ACROSS PARAMETERS — the statement the numeric gate drives.**

    Two parameters clipped off the SAME total (and the same `c`/`ε`) satisfy
    `g'₁ᵢ · g₂ⱼ = g'₂ⱼ · g₁ᵢ`, i.e. the ratio `g'/g` is one constant across every coordinate of
    every parameter.

    ⚠ **This is the ONLY property that separates the reference from a per-parameter clip.** A
    per-parameter clip scales, never amplifies, and is the identity below the threshold — it
    satisfies everything else in `GradClip.lean`. It differs here and nowhere else, which is why
    `clip-tie` measures the ratio's CONSTANCY across all 200/180 parameters instead of checking
    that any one parameter got smaller (`wdx-tie`'s *gate the partition, not the count*). -/
theorem clipShared_faithful {n m : Nat} (dsN dsM : List Nat) (clipS epsS : String) (c ε : ℝ)
    (s : SHlo 1) (gN : SHlo n) (gM : SHlo m) (i : Fin n) (j : Fin m) :
    den (.clipScaleF clipS epsS c ε dsN s gN) i * den gM j
      = den (.clipScaleF clipS epsS c ε dsM s gM) j * den gN i := by
  simp only [clipScaleF_faithful]
  exact clipFactor_shared (clipFactor c ε (scalarOf (den s))) (den gN) (den gM) i j

/-- **Below the threshold the rendered clip is the EXACT identity**, so a clip-on render at a large
    `c` must agree with the clip-off render on every byte (`x * 1.0` is exact in binary32). The
    emit-side reading of `Proofs.clipGrad_id_below`, and the licence for `clip-tie`'s gate 3.
    ⚠ It is also why gate 3 alone is not evidence: at factor 1 a per-parameter clip and a global one
    are the SAME FUNCTION, so an identity gate cannot see which was rendered. -/
theorem clipScaleF_id_below {n : Nat} (clipS epsS : String) (c ε : ℝ) (ds : List Nat)
    (s : SHlo 1) (e : SHlo n) (hε : 0 < ε)
    (h : Real.sqrt (scalarOf (den s)) + ε ≤ c) :
    den (.clipScaleF clipS epsS c ε ds s e) = den e := by
  simp only [clipScaleF_faithful, clipFactor_eq_one_below c ε (scalarOf (den s)) h hε,
             clipScale_one]

/-- **Inference per-channel BN forward faithfulness.** The 4-D reshape + affine
    `γ·(x−μ)·rsqrt(var+ε)+β` with rank-1 μ/var/γ/β (`dims=[1]`) denotes the proven
    `bnPerChannelEvalTensor3` (PerChannelBN.lean). (`rfl`, so kept out of the axiom audit.) -/
@[simp] theorem bnPerChannelEvalF_faithful {oc h w : Nat} (gN bN muN varN es : String) (ε : ℝ)
    (γ β μ var : Vec oc) (e : SHlo (oc*h*w)) :
    den (.bnPerChannelEvalF gN bN muN varN es ε γ β μ var e)
      = bnPerChannelEvalTensor3 oc h w ε γ β μ var (den e) := rfl

/-- The per-channel BN backward node denotes the per-channel grad-input helper at its cotangent. -/
@[simp] theorem den_bnPerChannelBack {oc h w : Nat} (gN xN es : String) (ε : ℝ) (γ : Vec oc)
    (x : Vec (oc*h*w)) (e : SHlo (oc*h*w)) :
    den (.bnPerChannelBack gN xN es ε γ x e) = bnPerChannelTensor3GradInput oc h w ε γ x (den e) :=
  rfl

/-- **Per-channel BN backward faithfulness.** The block-diagonal three-term graph
    (per-channel, reducing over the spatial axes) denotes the proven per-channel BN
    input-VJP — equal to the `pdiv`-contracted (block-diagonal) Jacobian of
    `bnPerChannelTensor3` (`bnPerChannelTensor3GradInput_correct`), under `0 < ε`. -/
theorem bnPerChannelBack_faithful {oc h w : Nat} (gN xN es : String) (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec oc) (x : Vec (oc*h*w)) (e : SHlo (oc*h*w)) (i : Fin (oc*h*w)) :
    den (.bnPerChannelBack gN xN es ε γ x e) i
      = ∑ j : Fin (oc*h*w), pdiv (bnPerChannelTensor3 oc h w ε γ β) x i j * den e j := by
  rw [den_bnPerChannelBack]
  exact bnPerChannelTensor3GradInput_correct oc h w ε hε γ β x (den e) i

/-- **Depthwise-conv forward faithfulness.** The `feature_group_count = c`
    `stablehlo.convolution` (with a `[c,1,kH,kW]` kernel, one filter per channel)
    denotes the proven `depthwiseFlat` (= flatten ∘ depthwiseConv2d ∘ unflatten,
    Depthwise.lean). (`rfl`, so kept out of the axiom audit — `roundtrip` covers it
    structurally.) -/
@[simp] theorem depthwiseF_faithful {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (c*h*w)) :
    den (.depthwiseF wN bN W b e) = depthwiseFlat W b (den e) := rfl

/-- **Depthwise-conv input-VJP faithfulness.** The reversed-kernel depthwise
    `stablehlo.convolution` (reverse the per-channel filters over the spatial axes
    `[2,3]`; the channel groups are 1×1 so no o↔i transpose, same
    `feature_group_count = c`) denotes the proven `depthwiseFlatHasVJP` backward
    (depthwise is linear, so this is a global VJP). -/
theorem depthwiseBack_faithful {c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*h*w)) (e : SHlo (c*h*w)) :
    den (.depthwiseBack wN W b v e) = (depthwiseFlatHasVJP W b).backward v (den e) := rfl

/-- **Strided-depthwise forward faithfulness.** The `window_strides=[2,2]`,
    `feature_group_count = c` `stablehlo.convolution` denotes the proven
    `depthwiseStride2Flat` (= decimate ∘ stride-1 depthwise, Depthwise.lean). -/
@[simp] theorem depthwiseStridedF_faithful {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (c*(2*h)*(2*w))) :
    den (.depthwiseStridedF wN bN W b e) = depthwiseStride2Flat W b (den e) := rfl
@[simp] theorem depthwiseStridedXlaF_faithful {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (c*(2*h)*(2*w))) :
    den (.depthwiseStridedXlaF wN bN W b e) = depthwiseStride2FlatXla W b (den e) := rfl

/-- **Strided-depthwise input-VJP faithfulness.** The zero-upsample (`stablehlo.pad`
    interior=1) + reversed-kernel stride-1 depthwise denotes the proven
    `depthwiseStride2FlatHasVJP` backward. -/
theorem depthwiseStridedBack_faithful {c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) (e : SHlo (c*h*w)) :
    den (.depthwiseStridedBack wN W b v e) = (depthwiseStride2FlatHasVJP W b).backward v (den e) := rfl

/-- **Swish forward faithfulness.** The `multiply(x, logistic(x))` graph denotes
    the proven `swish` (= `x · σ(x)`, LayerNorm.lean). Smooth everywhere; no kink,
    no smoothness hypothesis. (`rfl`, so kept out of the axiom audit — `roundtrip`
    covers it structurally.) -/
@[simp] theorem swishF_faithful {n : Nat} (e : SHlo n) :
    den (.swishF e) = swish n (den e) := rfl

/-- **Swish input-VJP faithfulness.** The closed-form `dy ⊙ σ(x)·(1 + x·(1−σ(x)))`
    graph (recomputing σ from the saved pre-activation `x`) denotes the proven GLOBAL
    `swishHasVJP` backward (`dy ⊙ swishScalarDeriv x`; swish is smooth everywhere, so
    this is a global VJP — no smoothness hypothesis). -/
theorem swishBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.swishBack xN x e) = (swishHasVJP n).backward x (den e) := rfl

/-- **Sigmoid forward faithfulness.** The `stablehlo.logistic(x)` graph denotes the
    proven `sigmoid` (= σ(x), SE.lean) — the SE gate's output nonlinearity.
    Smooth everywhere. (`rfl`, so kept out of the axiom audit — `roundtrip` covers it.) -/
@[simp] theorem sigmoidF_faithful {n : Nat} (e : SHlo n) :
    den (.sigmoidF e) = sigmoid n (den e) := rfl

/-- **Sigmoid input-VJP faithfulness.** The closed-form `dy ⊙ σ(x)·(1−σ(x))` graph
    (recomputing σ from the saved pre-activation `x`) denotes the proven GLOBAL
    `sigmoidHasVJP` backward (`dy ⊙ sigmoidScalarDeriv x`; sigmoid is smooth
    everywhere, so this is a global VJP — no smoothness hypothesis). -/
theorem sigmoidBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.sigmoidBack xN x e) = (sigmoidHasVJP n).backward x (den e) := rfl

/-- **GELU forward faithfulness.** The tanh-approximation graph
    `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` denotes the proven `gelu`
    (LayerNorm.lean). Smooth everywhere; no kink, no smoothness hypothesis.
    (`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem geluF_faithful {n : Nat} (e : SHlo n) :
    den (.geluF e) = gelu n (den e) := rfl

/-- **Layer-scale faithfulness.** The per-element multiply `γ ⊙ x` denotes the proven
    `layerScale` (LayerNorm.lean). (`rfl`.) -/
@[simp] theorem layerScaleF_faithful {n : Nat} (γN : String) (γ : Vec n) (e : SHlo n) :
    den (.layerScaleF γN γ e) = layerScale γ (den e) := rfl

/-- **Per-channel layer-scale faithfulness.** The `[c]`-broadcast multiply denotes
    the proven `layerScale` at the channel-expanded vector. (`rfl`.) -/
@[simp] theorem layerScaleChF_faithful {c h w : Nat} (γN : String) (γ : Vec c)
    (e : SHlo (c*h*w)) :
    den (.layerScaleChF γN γ e) = layerScale (fun k => γ (chanIdx c h w k)) (den e) := rfl

/-- **GELU input-VJP faithfulness.** The closed-form `dy ⊙ gelu'(x)` graph
    (recomputing `tanh(u(x))` from the saved pre-activation `x`) denotes the proven
    GLOBAL `geluHasVJP` backward (`dy ⊙ geluScalarDeriv x`; GELU is smooth
    everywhere, so this is a global VJP — no smoothness hypothesis). -/
theorem geluBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.geluBack xN x e) = (geluHasVJP n).backward x (den e) := rfl

/-- **Row-softmax forward faithfulness.** The per-row `exp / reduce[last] / divide`
    graph denotes `rowSoftmaxFlat` (= flattened `rowSoftmax`, Attention.lean). Plain
    exp/sum, no max-shift (matches the proven `softmax`). Smooth everywhere.
    (`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem softmaxRowF_faithful {m n : Nat} (e : SHlo (m*n)) :
    den (.softmaxRowF e) = rowSoftmaxFlat m n (den e) := rfl

/-- **Row-softmax input-VJP faithfulness.** The per-row closed-form
    `p ⊙ (dy − ⟨p,dy⟩)` graph (recomputing `p` from the saved pre-softmax scores)
    denotes `rowSoftmaxBackFlat` (= flattened `rowSoftmaxHasVJPMat.backward`).
    Softmax is smooth, so this is a global VJP — no smoothness hypothesis. -/
theorem softmaxRowBack_faithful {m n : Nat} (xN : String) (preAct : Vec (m*n)) (e : SHlo (m*n)) :
    den (.softmaxRowBack xN preAct e) = rowSoftmaxBackFlat m n preAct (den e) := rfl

/-- **Matrix-multiply faithfulness.** The reshape + batching-dim-0 `dot_general`
    (contracting `[2] x [1]`) + reshape graph denotes `matMulFlat` (= the flattened
    `Mat.mul`). Bilinear; the attention backwards reuse this token (`dA = dC·Bᵀ`,
    `dB = Aᵀ·dC`). (`rfl`, so kept out of the axiom audit — `roundtrip` covers it
    structurally.) -/
@[simp] theorem matmulF_faithful {m k n : Nat} (a : SHlo (m*k)) (b : SHlo (k*n)) :
    den (.matmulF a b) = matMulFlat m k n (den a) (den b) := rfl

/-- **Transpose faithfulness.** `stablehlo.transpose dims=[0,2,1]` (after reshape
    to rank 3) denotes `transposeFlat` (= the flattened `Mat.transpose`). (`rfl`.) -/
@[simp] theorem transposeF_faithful {m n : Nat} (e : SHlo (m*n)) :
    den (.transposeF e) = transposeFlat m n (den e) := rfl

/-- **Scalar-scale faithfulness.** The splat-constant `stablehlo.multiply` denotes
    pointwise `s · x` — SDPA's `1/√d`. (`rfl`; the `sStr ↔ s` literal agreement is
    the audited lexical boundary, like `bnF`'s `epsStr`.) -/
@[simp] theorem scaleF_faithful {n : Nat} (sN : String) (s : ℝ) (e : SHlo n) :
    den (.scaleF sN s e) = fun i => s * den e i := rfl

/-- **Row-LayerNorm forward faithfulness.** The rank-3 reduce[2]/normalize/affine
    graph (per token row, scalar γ/β) denotes `rowLNFlat` (rowwise `bnForward` =
    rowwise `layerNormForward`, definitionally). (`rfl`.) -/
@[simp] theorem lnRowF_faithful {m n : Nat} (gN bN es : String) (ε γ β : ℝ) (e : SHlo (m*n)) :
    den (.lnRowF gN bN es ε γ β e) = rowLNFlat m n ε γ β (den e) := rfl

/-- **Row-LayerNorm input-VJP faithfulness.** The per-row consolidated three-term
    graph (recomputing x̂/istd from the saved pre-LN input, reductions over the row
    axis) denotes `rowLNBackFlat` (rowwise `bnGradInput` — faithful to the
    pdiv-Jacobian per row under `0 < ε`, `bn_input_grad_correct`). -/
theorem lnRowBack_faithful {m n : Nat} (gN xN es : String) (ε γ : ℝ) (x : Vec (m*n))
    (e : SHlo (m*n)) :
    den (.lnRowBack gN xN es ε γ x e) = rowLNBackFlat m n ε γ x (den e) := rfl

/-- **Per-token dense forward faithfulness.** The `dot_general [2] x [0]` + bias
    broadcast `dims=[2]` graph denotes `rowDenseFlat` (rowwise `dense W b`). (`rfl`.) -/
@[simp] theorem denseRowF_faithful {N a c : Nat} (wN bN : String) (W : Mat a c) (b : Vec c)
    (e : SHlo (N*a)) :
    den (.denseRowF wN bN W b e) = rowDenseFlat N a c W b (den e) := rfl

/-- **Per-token dense input-VJP faithfulness.** The `dot_general [2] x [1]` graph
    (dy against W's output axis) denotes `rowDenseBackFlat` (rowwise `Mat.mulVec W`
    = the proven `denseHasVJP` backward; dense is affine — global VJP). -/
theorem denseRowBack_faithful {N a c : Nat} (wN : String) (W : Mat a c) (e : SHlo (N*c)) :
    den (.denseRowBack wN W e) = rowDenseBackFlat N a c W (den e) := rfl

/-- **Patch-embedding faithfulness.** The stride-P VALID conv + channels-last
    flatten + CLS concatenate + position-embed add graph denotes `patchEmbedFlat`
    (the local re-spelling of the proven `patchEmbedFlat`; the tie is `rfl` in
    ViTFwdGraph). (`rfl`, coarse-token like `seBlock`.) -/
@[simp] theorem patchEmbedF_faithful {ic H W P N D : Nat} (wN bN cN pN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N+1) D) (e : SHlo (ic*H*W)) :
    den (.patchEmbedF wN bN cN pN Wc bc cls pos e)
      = patchEmbedFlat ic H W P N D Wc bc cls pos (den e) := rfl

/-- **CLS-slice faithfulness.** The row-0 `stablehlo.slice` denotes `clsSliceFlat`
    (= the proven `clsTokenFlat`). (`rfl`.) -/
@[simp] theorem clsSliceF_faithful {N D : Nat} (e : SHlo ((N+1)*D)) :
    den (.clsSliceF e) = clsSliceFlat N D (den e) := rfl

/-- **CLS-pad faithfulness.** The zero-pad scatter-to-row-0 denotes `clsPadFlat`
    (= the proven `clsTokenFlatHasVJP.backward`; linear — global VJP). (`rfl`.) -/
@[simp] theorem clsPadF_faithful {N D : Nat} (e : SHlo D) :
    den (.clsPadF (N := N) e) = clsPadFlat N D (den e) := rfl

/-- **Per-head slice faithfulness.** The feature-axis `stablehlo.slice` of head `h`'s
    contiguous column block denotes `headSliceFlat` (= `mhsaLayer`'s per-head column
    gather). Linear reindex. (`rfl`.) -/
@[simp] theorem headSliceF_faithful {N heads d : Nat} (h : Fin heads)
    (e : SHlo (N*(heads*d))) :
    den (.headSliceF h e) = headSliceFlat N heads d h (den e) := rfl

/-- **Per-head pad faithfulness.** The feature-axis zero-pad into head `h`'s column
    block denotes `headPadFlat` (the slice's VJP; summed over heads it is
    `mhsaLayer`'s concat). Linear. (`rfl`.) -/
@[simp] theorem headPadF_faithful {N heads d : Nat} (h : Fin heads) (e : SHlo (N*d)) :
    den (.headPadF h e) = headPadFlat N heads d h (den e) := rfl

/-- **Row-broadcast scale faithfulness.** The reshape + broadcast-γ-over-rows +
    multiply graph denotes `rowScaleFlat` (rowwise `layerScale γ`). Diagonal-linear —
    its own input-VJP, so the backward reuses this token on the cotangent. (`rfl`.) -/
@[simp] theorem rowScaleF_faithful {m n : Nat} (gN : String) (γ : Vec n) (e : SHlo (m*n)) :
    den (.rowScaleF gN γ e) = rowScaleFlat m n γ (den e) := rfl

/-- **Row-broadcast bias faithfulness.** The broadcast-β-over-rows + add graph denotes
    `rowBiasFlat`. Translation — identity input-VJP. (`rfl`.) -/
@[simp] theorem rowBiasF_faithful {m n : Nat} (bN : String) (β : Vec n) (e : SHlo (m*n)) :
    den (.rowBiasF bN β e) = rowBiasFlat m n β (den e) := rfl

/-- Whole MNIST-CNN **forward** graph:
    `dense ∘ relu ∘ dense ∘ relu ∘ dense ∘ maxPool ∘ relu ∘ conv ∘ relu ∘ conv`. -/
def cnnFwdGraph {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (x : Vec (ic*(2*h)*(2*w))) : SHlo nClasses :=
  denseF "%W5" "%b5" W₅ b₅
    (.reluF (denseF "%W4" "%b4" W₄ b₄
      (.reluF (denseF "%W3" "%b3" W₃ b₃
        (.maxPoolF (c := c) (h := h) (w := w)
          (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W2" "%b2" W₂ b₂
            (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W1" "%b1" W₁ b₁
              (.operand "%x" x))))))))))

/-- Whole **CIFAR-CNN forward** graph (Chapter 4): two conv→relu→conv→relu→maxPool
    stages (channels `ic→c1→c1`, then `c1→c2→c2`) then `dense→relu→dense→relu→dense`.
    The Chapter-4 peer of `cnnFwdGraph`. -/
def cifarFwdGraph {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : SHlo nClasses :=
  denseF "%W7" "%b7" W₇ b₇
    (.reluF (denseF "%W6" "%b6" W₆ b₆
      (.reluF (denseF "%W5" "%b5" W₅ b₅
        (.maxPoolF (c := c2) (h := h) (w := w)
          (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄
            (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃
              (.maxPoolF (c := c1) (h := 2*h) (w := 2*w)
                (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂
                  (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁
                    (.operand "%x" x)))))))))))))))

/-- Whole **deeper (8-conv) CIFAR-CNN forward** graph: four conv→relu→conv→relu→maxPool
    stages (channels `ic→c1→c1`, `c1→c2→c2`, `c2→c3→c3`, `c3→c4→c4`) then
    `dense→relu→dense→relu→dense`. The 4-stage peer of `cifarFwdGraph`. -/
def cifar8FwdGraph {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : SHlo nClasses :=
  denseF "%Wb" "%bb" Wb bb
    (.reluF (denseF "%Wa" "%ba" Wa ba
      (.reluF (denseF "%W9" "%b9" W₉ b₉
        (.maxPoolF (c := c4) (h := h) (w := w)
          (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W8" "%b8" W₈ b₈
            (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W7" "%b7" W₇ b₇
              (.maxPoolF (c := c3) (h := 2*h) (w := 2*w)
                (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W6" "%b6" W₆ b₆
                  (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W5" "%b5" W₅ b₅
                    (.maxPoolF (c := c2) (h := 2*(2*h)) (w := 2*(2*w))
                      (.reluF (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W4" "%b4" W₄ b₄
                        (.reluF (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W3" "%b3" W₃ b₃
                          (.maxPoolF (c := c1) (h := 2*(2*(2*h))) (w := 2*(2*(2*w)))
                            (.reluF (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W2" "%b2" W₂ b₂
                              (.reluF (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W1" "%b1" W₁ b₁
                                (.operand "%x" x)))))))))))))))))))))))))

/-- Whole **deeper (8-conv) BN-CIFAR forward** graph: each of the eight convs is followed
    by a per-channel `bnPerChannelF` before its ReLU. `epsStr` is the shared ε literal; the
    eight BN layers carry per-channel γ/β inputs `%g{i}`/`%bt{i}`. The BN peer of
    `cifar8FwdGraph`. -/
def cifar8BnFwdGraph {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : SHlo nClasses :=
  denseF "%Wb" "%bb" Wb bb
    (.reluF (denseF "%Wa" "%ba" Wa ba
      (.reluF (denseF "%W9" "%b9" W₉ b₉
        (.maxPoolF (c := c4) (h := h) (w := w)
          (.reluF (.bnPerChannelF (oc := c4) (h := 2*h) (w := 2*w) "%g8" "%bt8" epsStr ε₈ γ₈ β₈
            (.flatConvF (h := 2*h) (w := 2*w) "%W8" "%b8" W₈ b₈
            (.reluF (.bnPerChannelF (oc := c4) (h := 2*h) (w := 2*w) "%g7" "%bt7" epsStr ε₇ γ₇ β₇
              (.flatConvF (h := 2*h) (w := 2*w) "%W7" "%b7" W₇ b₇
              (.maxPoolF (c := c3) (h := 2*h) (w := 2*w)
                (.reluF (.bnPerChannelF (oc := c3) (h := 2*(2*h)) (w := 2*(2*w)) "%g6" "%bt6" epsStr ε₆ γ₆ β₆
                  (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W6" "%b6" W₆ b₆
                  (.reluF (.bnPerChannelF (oc := c3) (h := 2*(2*h)) (w := 2*(2*w)) "%g5" "%bt5" epsStr ε₅ γ₅ β₅
                    (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W5" "%b5" W₅ b₅
                    (.maxPoolF (c := c2) (h := 2*(2*h)) (w := 2*(2*w))
                      (.reluF (.bnPerChannelF (oc := c2) (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%g4" "%bt4" epsStr ε₄ γ₄ β₄
                        (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W4" "%b4" W₄ b₄
                        (.reluF (.bnPerChannelF (oc := c2) (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%g3" "%bt3" epsStr ε₃ γ₃ β₃
                          (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W3" "%b3" W₃ b₃
                          (.maxPoolF (c := c1) (h := 2*(2*(2*h))) (w := 2*(2*(2*w)))
                            (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%g2" "%bt2" epsStr ε₂ γ₂ β₂
                              (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W2" "%b2" W₂ b₂
                              (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%g1" "%bt1" epsStr ε₁ γ₁ β₁
                                (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W1" "%b1" W₁ b₁
                                (.operand "%x" x)))))))))))))))))))))))))))))))))

-- ════════════════════════════════════════════════════════════════
-- § Chapter 3 — CNN: the whole-chain backward graph (the MLP-analog of `mlpBackGraph`).
--   That it denotes `mnistCnnNoBnHasVJPAt.backward`, and that the chapter 3–4 forward graphs
--   above denote their nets, is `Nets/Small/ChapterGraphTies` — the IR imports no net.
-- ════════════════════════════════════════════════════════════════

/-- Whole MNIST-CNN **backward** (input-VJP) graph, reversing `cnnFwdGraph`:
    `convBack W₁ ∘ select(a₁) ∘ convBack W₂ ∘ select(a₂) ∘ maxPoolBack ∘
     dotOut W₃ ∘ select(a₃) ∘ dotOut W₄ ∘ select(a₄) ∘ dotOut W₅`, with `aᵢ` the
    ReLU pre-activations and the conv/maxpool saved inputs threaded as in §4. -/
noncomputable def cnnBackGraph
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses)
    (x : Vec (ic * (2*h) * (2*w))) (dy : Vec nClasses) :
    SHlo (ic * (2*h) * (2*w)) :=
  let z1 := (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x
  let zmp := (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂) z1
  let zd3 := maxPoolFlat c h w zmp
  let zd4 := (relu d1 ∘ dense W₃ b₃) zd3
  .convBack "%W1" W₁ b₁ x
    (.selectPos "%a1" (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x)
      (.convBack "%W2" W₂ b₂ z1
        (.selectPos "%a2" (flatConv (h := 2*h) (w := 2*w) W₂ b₂ z1)
          (.maxPoolBack "%z2" zmp
            (.dotOut "%W3" W₃
              (.selectPos "%a3" (dense W₃ b₃ zd3)
                (.dotOut "%W4" W₄
                  (.selectPos "%a4" (dense W₄ b₄ zd4)
                    (.dotOut "%W5" W₅ (.operand "%dy" dy))))))))))

/-- **The conv-bias SSA name** — §2l step B. Every conv in ResNet-34 is immediately followed by
    BatchNorm, and BN subtracts the batch mean, so in ℝ a conv bias cannot reach the BN output and
    its gradient is identically zero. He et al.'s `.convBn` therefore carries no conv bias, and
    this repo's render did — 8,512 parameters the reference does not have (§2k).

    With `convBias := false` the bias operand becomes a zero CONSTANT rather than a function
    argument: the op is the same proven `flatConvF`/`flatConvStridedF` at `bias = 0`, so `den` and
    every faithfulness theorem are untouched, and `x + 0.0` is exact in IEEE, so the forward is
    **bit-identical** to the biased render fed zeros. What changes is the signature.

    ⚠ MEASURED, and it corrects §2l's stated reason: in f32 the gradient is NOT exactly zero — the
    BN mean is a rounded sum, leaving a residue ~1e-6 of the conv-weight gradient — and under
    AdamW's scale-free update that residue still moves θ by ~lr per step. In the 80-epoch run all
    8,512 biases drifted to |θ|max 0.041. They are safe to drop because the FORWARD does not depend
    on them (zeroing all of them moves the trained logits by rel 1e-6, against 0.79 for the same
    ablation on BN β), not because they stay zero. See [`tests/TestConvBiasZero.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestConvBiasZero.lean). -/
def biasName (convBias : Bool) (nm : String) (c : Nat) : String :=
  if convBias then nm else s!"%zb{c}"

end StableHLO
end Proofs
