import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Architectures.CifarCNN
import LeanMlir.Proofs.Foundation.StridedConv
import LeanMlir.Proofs.Foundation.PerChannelBN
import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.MobileNetV2
import LeanMlir.Proofs.Architectures.LayerNorm
import LeanMlir.Proofs.Architectures.EfficientNet
import LeanMlir.Proofs.Architectures.ConvNeXt
-- The ℝ AdamW spec (`adamMNext`/`adamVNext`/`adamWParam`), so the optimizer ops can denote it.
-- AdamStep only imports Foundation.Tensor + Mathlib, so this adds no cycle.
import LeanMlir.Proofs.Codegen.AdamStep
-- The ℝ global-norm clip spec (`gradSumSq`/`clipFactor`/`clipScale`), so the four clip ops can
-- denote it. GradClip imports only AdamStep, so this adds no cycle either.
import LeanMlir.Proofs.Codegen.GradClip
-- LAMB (`lambDir`/`lambTrust`/`lambScale`), RSB-A3's optimizer. Imports GradClip for `scalarOf`
-- and `gradSumSq` — the per-leaf squared norm is SHARED with the clip rather than re-derived, so
-- the two features cannot drift on what a norm is.
import LeanMlir.Proofs.Codegen.Lamb
import LeanMlir.Proofs.Codegen.SgdMomentumStep
-- RmsPropStep imports only the two above, so this adds no cycle either.
import LeanMlir.Proofs.Codegen.RmsPropStep
-- DropPath imports only Architectures.ConvNeXt (for `layerScale`, which this file already has in
-- scope), so it adds no cycle either. `planning/stochastic_depth.md`.
import LeanMlir.Proofs.Codegen.DropPath
-- He et al.'s 3×3/s2 stem pool (`maxPool3s2Flat` + its VJP witness), so the stem-pool ops can
-- denote it. MaxPool3s2 imports only Architectures.CNN, which this file already has in scope
-- transitively, so it adds no cycle. `planning/rsb_a3_r50_verified.md` §4b.
import LeanMlir.Proofs.Architectures.MaxPool3s2

/-! # R4 — printer faithfulness, Stage A (Chapter 1: the linear classifier)

The seed of `planning/validated_codegen_book.md`'s `Proofs/Hlo/{Syntax,Denote}`.

`IR.lean` gives the backward/forward IR a denotation in `ℝ` and proves it equals
the Mathlib-`fderiv` math. The remaining trusted link — **R4** — is that the
StableHLO **text** the printer emits means the same function. This file closes
R4 for Chapter 1, *both halves*, over a single typed AST `SHlo`:

* **Semantic half** (`den`, load-bearing): a denotation in StableHLO-spec terms
  (explicit contraction / reduce / divide), and faithfulness theorems
  `den (emit …) = <proven math>` for every piece of the linear train step —
  forward logits, dense input-VJP, softmax-CE cotangent (to the proven
  ∂CE/∂logits), the weight/bias parameter Jacobians, **and the SGD update**
  (`θ' = θ − lr·∇`, now proven rather than trusted).

* **Syntactic half** (`pretty`): the same `SHlo` carries SSA-name annotations
  (denotation-irrelevant — `den` ignores them) so it renders to real StableHLO
  text. The emitted modules — including the **whole `@linear_train_step`** —
  are `pretty (emit g)` (the doc's "Step 0 consolidation": one AST, both
  denotable and renderable).

**All together (the R4 chain for ch 1):**
`render text = pretty (emit g)` (syntactic, by construction);
`den (emit g) = Mathlib fderiv` (semantic, the theorems below).

**Scope / residue.** Per-example semantics (`Vec`/`Mat`): the batch axis is an
outer map, a printer concern (the doc's "D1 shortcut"). `pretty`'s lexical
conformance to the StableHLO spec is the audited/validated residue (the doc's
"4b": cross-checked by `iree-compile` + execution — the verified-rendered train
step trains MNIST to ~92%), not a verified `parse` round-trip ("4a"). Everything
here closes under `[propext, Classical.choice, Quot.sound]` (`tests/AuditAxioms.lean`).
-/

open Finset BigOperators

namespace Proofs
namespace StableHLO

-- Each extension of the `SHlo` constructor list raises the per-`whnf` cost of unfolding the
-- `den` match (the `brecOn`/`below` packaging scales with the constructor count), so the
-- `den (emit …) = <math>` faithfulness proofs below — even the trivial ones — sit closer to the
-- heartbeat ceiling with every added op. The MobileNetV2 depthwise-SGD ops (the 4 `depthwise*Sgd`
-- constructors) pushed several of these over the 200000 default; raise the file floor (the heavy
-- `cnnBackGraph_faithful` keeps its own larger `2000000` bump below).
--
-- ⚠ 2026-08-03, the batched-index move (§0.2 ▶2, increment 2): **1000000 → 4000000**. Adding
-- `geluBackB` + `lnRowBackB` put NINE of the proofs below over the 1M floor at once, and it is a
-- threshold effect rather than anything about these two ops — ONE new arm, shape-identical to the
-- existing `swishBackB`, trips the same nine. Measured: 4× clears every one, whole-file elaboration
-- 3m32s. ⚠ A `set_option … in` on each proof does NOT work here: the option does not attach across
-- the declaration's doc comment, and the proofs still report the 1M ceiling. The file floor is the
-- knob, which is what this comment's own history already said.
--
-- ⚠⚠ This REFINES §0.8's finding, it does not repeat it. There, two ops with NO `{n : Nat}` binder
-- made unfolding `den` so much dearer that **4× did not help at any budget tried** and the fix was
-- to remove arms. Here the arms are parametric, the per-arm cost is modest, and the budget IS the
-- fix. The rule to carry forward: *parametric arms are affordable at a price; fixed-index arms are
-- not affordable at all.* Anyone adding the ~11 remaining ViT/ConvNeXt forms should expect to move
-- this number again, and should check `den`'s elaboration time before assuming it still scales.
set_option maxHeartbeats 4000000

-- ════════════════════════════════════════════════════════════════
-- § Batched lift (EfficientNet) — per-example block-apply over N examples,
--   plus the one genuinely batch-coupled op (true batch-norm).
-- ════════════════════════════════════════════════════════════════

/-- **Per-example block-apply.** Lift a per-example map `f : Vec a → Vec b` to a
    batch of `N` examples laid out row-major `[N, a] ↦ [N, b]` (the network's
    `[N,C,H,W]`-style flattening): example `n` occupies the `finProdFinEquiv`
    block `{(n, ·)}`. Every spatial/channel op in EfficientNet is batch-separable
    and lifts this way; only true batch-norm (`bnBatchTensor4`) couples the batch. -/
noncomputable def batchMap (N : Nat) {a b : Nat} (f : Vec a → Vec b) :
    Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f (fun i : Fin a => x (finProdFinEquiv (p.1, i))) p.2

/-- The `n`-th example's slice of a batch laid out row-major `[N, a]`. A shared
    weight's batched gradient is the sum over `n` of the per-example gradient on
    `batchSlice n` — the form the batched param-SGD dens take (so the §1 fold closes
    via the per-example cert + sum-linearity). -/
def batchSlice (N a : Nat) (v : Vec (N * a)) (n : Fin N) : Vec a :=
  fun i => v (finProdFinEquiv (n, i))

/-- **Per-example block-apply with per-example AUXILIARY data.** `batchMap` lifts one *fixed*
    function across the batch; this lifts a family indexed by each example's own saved value —
    example `n` is handed `batchSlice n aux`, not the whole `aux` and not example 0's.

    Every batched backward that recomputes from a saved forward activation has this shape, and
    that is exactly why such ops cannot be `BatchableOp` descriptors: a descriptor's
    `batchMap N (denOp op)` would apply ONE example's saved value to all `N`. Cf. `swishBackB`,
    `sigmoidBackB`, `selectPosB` (pointwise, so they take the whole-batch `x` directly) and
    `seBackBatched` (which inlines this shape). -/
noncomputable def batchMapAux (N : Nat) {s a b : Nat} (f : Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) : Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f (batchSlice N s aux p.1) (batchSlice N a x p.1) p.2

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
  -- the reference on shared weights separates them (`planning/mnv4_verified.md` §3b/§3d measured
  -- 6.16e-2 on mnv4's stem and 2.9e-1 across mnv2's five sites). Pick by which reference the net
  -- has: TF-origin → this one; torchvision-origin → `convStrided`.
  --
  -- `den` is `flatConvStride2Xla` = `decimateOddFlat ∘ flatConv` — the SAME stride-1 conv, read at
  -- the odd phase. So this adds no proof obligation: the forward, input-VJP, weight-VJP and
  -- bias-VJP are all `vjp_comp`s of results already proven (`Foundation/StridedConv.lean`).
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
  -- (`jax/Jax/Codegen.lean:679`), so every strided depthwise in those references pads
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
  -- ⭐ **3×3/s2 max-pool FORWARD** — He et al.'s ResNet stem pool (`planning/rsb_a3_r50_verified.md`
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
  -- `planning/bf16_renderer.md` §9.2 measured that `dot_general` reaches the tensor cores with
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
  -- Mixed-precision matmul (planning/bf16_renderer.md): BOTH operands rounded by `rnd`,
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
  -- LinearFaithfulPoC proves both `den`s = the certified loss-descent step.
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
  -- Mixed precision (planning/bf16_renderer.md): the in-graph ROUND node. `den` is
  -- literally `rnd ∘ den e`, so it is `den`-faithful for ANY rounding — bf16
  -- round-to-nearest being the instance we emit. This is the op
  -- `Proofs/Float/Bf16FaithfulPoC.lean` names as the depth > 1 ingredient
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
  -- `convWeightSgd`: `W − lr·(conv2d_weight_grad(b,x)·dy)` via the transpose-trick conv
  -- (transpose→transpose→convolution→transpose, then const→multiply→subtract), `den`
  -- = `cnn_render_convW_certified`. `convBiasSgd`: `b − lr·(conv2d_bias_grad(W,x)·dy)`
  -- (reduce over batch+spatial [0,2,3], then SGD). `xName`/`wName`/`bName` are the saved
  -- activation/kernel/bias SSA names; `W,x,b,lr` carry the den. CnnFaithfulPoC proves
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
  -- `epsStr` the ε literal. CifarBnFaithfulPoC proves both `den`s = the certified step.
  | bnGammaSgd {oc h w : Nat} (gName vName epsStr lrStr : String) (ε : ℝ) (γ : Vec oc)
      (v : Vec (oc*h*w)) (lr : ℝ)                          : SHlo (oc*h*w) → SHlo oc
  | bnBetaSgd  {oc h w : Nat} (bName lrStr : String) (β : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- Chapter 4 (BatchNorm): per-example normalization over the whole feature
  -- vec (reduce mean/var over axis [1], scalar γ/β). `gName,bName` are the γ,β
  -- scalar SSA inputs, `epsStr` the rendered ε literal; ε,γ,β carry the den.
  | bnF        {n : Nat} (gName bName epsStr : String) (ε γ β : ℝ)   : SHlo n → SHlo n
  -- BN input-VJP — the consolidated O(N) three-term gradient (`bn_grad_input`),
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
  -- conv). `den` via the proven `flatConvStride2` / `flatConvStride2_has_vjp`.
  | flatConvStridedF {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)              : SHlo (ic*(2*h)*(2*w)) → SHlo (oc*h*w)
  -- The XLA-`SAME` peer, for the TF-origin nets' per-example chains (`planning/mnv4_verified.md`
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
  -- VJP). ResNet34FaithfulPoC proves both `den`s = the certified loss-descent step.
  | convStridedWeightSgd {ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (ic*(2*h)*(2*w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convStridedBiasSgd   {ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w))) (b : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- MobileNetV2 (inverted-residual) param-SGD tail: the depthwise kernel/bias update ops,
  -- the depthwise analogues of `convWeightSgd`/`convBiasSgd`. `depthwiseWeightSgd` (stride-1,
  -- blocks b2/b4): `W − lr·(depthwise_weight_grad(b,x)·dy)` via the per-channel transpose-trick
  -- conv (`batch_group_count = c`, output [1,c,kH,kW]→[c,1,kH,kW]); `den` =
  -- `mnv2_render_depthwiseW_certified`. `depthwiseStridedWeightSgd` (stride-2, blocks b1/b3/b5/b6):
  -- zero-upsample dy (interior=1 → 2h×2w) then the SAME per-channel weight-grad on the 2h×2w grid;
  -- `den` = `mnv2_render_depthwiseW_strided_certified`. The depthwise bias grad is stride-INDEPENDENT
  -- (`Σ_{batch,spatial} dy`), so both bias ops emit the SAME `reduce` text as `convBiasSgd` (their
  -- `skel` aliases that op's Raw); only their `den` differs. MobileNetV2FaithfulPoC proves all four
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
  -- Chapter 8 (ConvNeXt-T) param-SGD tail. `layerScaleChGammaSgd`: the PER-CHANNEL layer-scale γ
  -- update `γ_c − lr·dγ_c`, `dγ_c = Σ_{b,h,w} x⊙dy` (the saved layer input `x` ⊙ the cotangent,
  -- reduced over batch+spatial per channel — `lsGradCh`). `γ : Vec c`, broadcast over spatial via
  -- `chanIdx` by the `layerScaleChF` forward; `den` = ConvNeXtFaithfulPoC's `cnx_render_lsgammaCh`.
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
  -- activation layout) / its renderable backward `bnPerChannelTensor3_grad_input`.
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
  -- `feature_group_count`). `den` via the proven `depthwiseFlat` / `depthwiseFlat_has_vjp`.
  | depthwiseF    {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*h*w) → SHlo (c*h*w)
  | depthwiseBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*h*w)) : SHlo (c*h*w) → SHlo (c*h*w)
  -- Chapter 6 C3: STRIDE-2 depthwise conv forward (`window_strides=[2,2]`,
  -- `feature_group_count = c`, `[c,1,kH,kW]` kernel — halves spatial, the MNv2
  -- downsampling op) and its input-VJP (zero-upsample the cotangent via
  -- `stablehlo.pad` interior=1 then the reversed-kernel stride-1 depthwise — the
  -- `convStridedBack` shape, per-channel). `den` via the proven `depthwiseStride2Flat`
  -- / `depthwiseStride2Flat_has_vjp` (= decimate ∘ depthwise).
  | depthwiseStridedF    {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  | depthwiseStridedXlaF {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  | depthwiseStridedBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- Chapter 7 (EfficientNet): swish forward (`x · σ(x)`, σ = `stablehlo.logistic`)
  -- and its input-VJP (`dy · swish'(x)`, closed form `σ(x)·(1 + x·(1−σ(x)))`).
  -- Swish is SMOOTH everywhere (no kink, NO smoothness hyp — unlike relu6); the
  -- VJP is the GLOBAL `swish_has_vjp` (no `_at`). `swishBack`'s `xName`/`x` is the
  -- saved pre-activation. `den` via the proven `swish` / `swish_has_vjp` (LayerNorm.lean).
  | swishF     {n : Nat}                                        : SHlo n → SHlo n
  | swishBack  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 7 (EfficientNet): sigmoid forward (`σ(x) = stablehlo.logistic`, the SE
  -- gate's output nonlinearity) and its input-VJP (`dy · σ(x)·(1−σ(x))`). Like swish,
  -- SMOOTH everywhere (no kink, NO smoothness hyp — GLOBAL `sigmoid_has_vjp`, not `_at`).
  -- `sigmoidBack`'s `xName`/`x` is the saved pre-activation. `den` via the proven
  -- `sigmoid` / `sigmoid_has_vjp` (EfficientNet.lean).
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
  -- `add`. `planning/rsb_a3_r50_verified.md` §4b.
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
  -- ▶ STOCHASTIC DEPTH (`planning/stochastic_depth.md`): `branch * keep / keep_prob`, the
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
  -- applies immediately before the classifier dense (`jax/Jax/Codegen.lean:1971`). `mName` is a
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
  --    denotes `Proofs.sigmoid` and already carries a global hypothesis-free `sigmoid_has_vjp`, but
  --    it is indexed PER EXAMPLE and emits at `ty [B, n]`; the loss cotangent lives at `SHlo (N*n)`
  --    with the `*B` family (`subB`, `divConstB`). So this is the same function at the batched
  --    index — one constructor, and `planning/next_session_pipeline_then_r50.md` §4 estimated
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
  -- the GLOBAL `gelu_has_vjp`, not `_at`). `geluBack`'s `xName`/`x` is the saved
  -- pre-activation. `den` via the proven `gelu` / `gelu_has_vjp` (LayerNorm.lean).
  | geluF      {n : Nat}                                        : SHlo n → SHlo n
  | geluBack   {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 8 (ConvNeXt): per-element layer-scale `γ ⊙ x` (diagonal linear, `γ : Vec n`
  -- over the flattened `c·h·w` map). `den` via the proven `layerScale` (ConvNeXt.lean).
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
  -- `rowSoftmaxBackFlat` (= `Mat.flatten ∘ rowSoftmax_has_vjp_mat.backward ∘ Mat.unflatten`).
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
  -- `Mat.mulVec W` = the proven `dense_has_vjp` backward). Linear — global VJP.
  | denseRowBack {N a c : Nat} (wName : String) (W : Mat a c)   : SHlo (N*c) → SHlo (N*a)
  -- ViT patch embedding (one coarse token, like `seBlock`): stride-P VALID conv
  -- (kernel `[D,ic,P,P]`, the non-overlapping patch projection) + bias, channels-
  -- last transpose + flatten to `[N,D]` tokens, prepend the CLS token, add the
  -- position embedding. `den` via `patchEmbedFlat` (a local re-spelling of the
  -- proven `patchEmbed_flat`, Attention.lean — the tie is `rfl` in ViTFwdGraph).
  | patchEmbedF {ic H W P N D : Nat} (wName bName clsName posName : String)
      (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N+1) D) :
      SHlo (ic*H*W) → SHlo ((N+1)*D)
  -- ViT patch-embedding input-VJP: the strided-P patchify conv's input gradient
  -- (reversed-kernel `conv_transpose` on the patch-token rows of the `[N+1,D]`
  -- cotangent; the CLS row and position-add contribute nothing — input-VJP = id
  -- on a +constant). `den` via `patchEmbedBackFlat` (= the proven
  -- `patchEmbed_input_grad_formula` = `patchEmbed_flat_has_vjp.backward`, the tie
  -- is `rfl` in ViTBackB0). Linear in the cotangent — activation-independent, so
  -- it routes through the generic `batched` Raw/Tok tag (like the strided-conv
  -- backward batched ops) rather than a bespoke top-level Raw/Tok constructor.
  | patchEmbedBack {ic H W P N D : Nat} (wName : String)
      (Wc : Kernel4 D ic P P) :
      SHlo ((N+1)*D) → SHlo (ic*H*W)
  -- CLS-token gather: row 0 of the `[N+1,D]` flat (`stablehlo.slice` after
  -- reshape) — the classifier head's input. `den` via `clsSliceFlat` (= the
  -- proven `cls_slice_flat`, Attention.lean).
  | clsSliceF  {N D : Nat}                                      : SHlo ((N+1)*D) → SHlo D
  -- CLS-slice VJP: scatter `dy` to row 0, zeros elsewhere (`stablehlo.pad` with
  -- `high = [0, N, 0]`). `den` via `clsPadFlat` (= the proven
  -- `cls_slice_flat_has_vjp.backward`). Linear — global VJP.
  | clsPadF    {N D : Nat}                                      : SHlo D → SHlo ((N+1)*D)
  -- Multi-head (ch10 scaling pass): per-head column slice — head `h`'s `[N,d]`
  -- block of the `[N,heads·d]` flat (columns `[h·d,(h+1)·d)` are contiguous in the
  -- row-major layout: `stablehlo.slice` on the feature axis after reshape).
  -- `den` via `headSliceFlat` (= `mhsa_layer`'s `finProdFinEquiv (h, ·)` column
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
  -- `bnBatchTensor4_grad_input` (reduce over [0,2,3] per channel). `den` is the
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
  -- strided-conv input-grad (`flatConvStride2_has_vjp` — activation-independent,
  -- strided conv = `decimate ∘ conv` is linear). The downsample basic-block's
  -- stride-2 conv1 backward; halves spatial vs `convBackBatched`. Routes through
  -- the generic `batched` tag like the stride-1 batched ops.
  | convStridedBackBatched {N ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * (2 * h) * (2 * w)))
  -- ⭐ The **bf16** input-VJP peers. These are where the money is: the backward is ~60% of the
  -- conv step (measured on R34's own layer shapes, `planning/bf16_renderer.md`), and unlike JAX
  -- — which autodiffs the backward FROM the cast forward and so inherits bf16 for free — every
  -- hand-written VJP here needs its own bf16 twin. dgrad is itself a convolution, so it takes
  -- the same emit shape and the same `den` discipline as the forward.
  | convBackBatchedBf16 {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wName : String)
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
  -- strided-depthwise input-grad (`depthwiseStride2Flat_has_vjp` — activation-
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
  -- The XLA-`SAME` peer (`planning/mnv4_verified.md` §3e/§3g). ⚠ The backward must place the SAME
  -- asymmetry the forward did — its `den` scatters onto the ODD positions, so the emitted
  -- transposed-conv padding shifts by one. Pairing an `Xla` forward with the SYMMETRIC backward
  -- above type-checks, trains and descends, and computes a gradient for a different net.
  | depthwiseStridedXlaBackBatched {N c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w)))
  -- ⭐ Its bf16 peer. ⚠⚠ Keeps the `[p+1, p-1]` pad — the OPPOSITE shift from the weight grads,
  -- because the kernel is reversed here. `scripts/xla_pad_op_check.py` caught that once already;
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
  -- `e` = the SE-output cotangent. `den` = batched `broadcastFlat_has_vjp.backward (x⊙dy)`.
  | seReduceB {N c h w : Nat} (xName : String) (x : Vec (N * (c * h * w))) :
      SHlo (N * (c * h * w)) → SHlo (N * c)
  -- Batched GLOBAL-AVERAGE-POOL backward (VJP): `dx[n,c,h,w] = dgap[n,c]/(h·w)` — the
  -- per-example `globalAvgPoolFlat_has_vjp` backward (broadcast over spatial, ÷h·w),
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
  -- ══ Pointwise affine-by-a-LITERAL at the batched index — the pieces a label-smoothed
  --    softmax-CE cotangent is composed from. `scaleB` is `scaleF`'s batched peer; `shiftB`
  --    and `divConstB` had no per-example peer at all.
  --    `divConstB` emits a real `divide` rather than `scaleB (1/c)` ON PURPOSE: the caller
  --    divides by the batch, and `1/B` is only exact in binary32 when `B` is a power of two.
  --    At the bs192/bs256 renders §2d wants, `x * (1/192) ≠ x / 192`. ══
  | scaleB    {N n : Nat} (sStr : String) (s : ℝ)         : SHlo (N*n) → SHlo (N*n)
  | shiftB    {N n : Nat} (sStr : String) (s : ℝ)         : SHlo (N*n) → SHlo (N*n)
  | divConstB {N n : Nat} (sStr : String) (s : ℝ)         : SHlo (N*n) → SHlo (N*n)
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
  -- `den` = flatten W − lr·(Σ_n per-example `conv2d_weight_grad` on `batchSlice n`).
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
  -- `flatConvStride4_has_vjp` (input) were already proven; this op's `den` is the matching
  -- `flatConvStride4_weight_grad_has_vjp`, which is two `vjp_comp` steps over the stride-1
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
  -- ── ▶ GLOBAL-NORM GRADIENT CLIPPING (`GradClip.lean`, `planning/grad_clip.md`), the ViT /
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
  --    tree would be duplicated at all 200 sites. Emit once, thread the name — `resnetFwdGraph`'s
  --    "tree-safe via operand leaves" trick used for its other purpose.
  --
  --    ⚠⚠ TWO ops, and the earlier four-op split (a separate `addScalarF : SHlo 1 → SHlo 1 →
  --    SHlo 1` and `gradClipFacF : SHlo 1 → SHlo 1`) was RETRACTED for a reason worth knowing
  --    before adding any op here: **a constructor with NO `{n : Nat}` binder is a shape this AST
  --    does not otherwise have**, and adding two of them made NINE unrelated `simp only [… den …]`
  --    proofs elsewhere in this file die with a `whnf` timeout — `den` is a ~200-case dependent
  --    match, and fully-index-fixed arms make unfolding it markedly more expensive. **4× the
  --    heartbeat budget did not fix it.** Both ops below are parametric in `n`, like every other
  --    constructor here. See `planning/grad_clip.md` §3.
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
-- matching `maxPool2_has_vjp_at3.backward` lifted through the flatten bridge.
-- Total in the saved input `xv` (the no-ties proof lives only in `.correct`).
open Classical in
noncomputable def maxPoolBackFlat (c h w : Nat)
    (xv : Vec (c*(2*h)*(2*w))) (dyv : Vec (c*h*w)) : Vec (c*(2*h)*(2*w)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    if MaxPool2IsArgmax (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w)) q.1 q.2 p.2
    then (Tensor3.unflatten dyv : Tensor3 c h w) q.1 (winRow q.2) (winCol p.2) else 0

/-- **3×3/s2 max-pool backward (flattened)** — the peer of `maxPoolBackFlat` at He et al.'s stem
    pool, matching `maxPool3s2_has_vjp_at3.backward` lifted through `hasVJPAt3_to_hasVJPAt`. Total
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
    `Mat.flatten ∘ rowSoftmax ∘ Mat.unflatten` (Attention.lean's `rowSoftmax`);
    spelled with MLP's `softmax` so `StableHLO` needn't import `Attention`
    (the tie to `rowSoftmax` is an `rfl` faithfulness lemma in `TestSoftmaxRow`). -/
noncomputable def rowSoftmaxFlat (m n : Nat) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => softmax n ((Mat.unflatten v) i))

/-- **Row-softmax backward (flattened)** — per row, the proven closed form
    `pᵢ⊙(dyᵢ − ⟨pᵢ,dyᵢ⟩)` with `pᵢ = softmax(preActᵢ)`. Definitionally equal to
    `Mat.flatten ∘ rowSoftmax_has_vjp_mat.backward (Mat.unflatten preAct) ∘ Mat.unflatten`
    (since `softmax_has_vjp.backward z dy i = let p := softmax z; p i·(dy i − ⟨p,dy⟩)`);
    spelled with MLP's `softmax` to keep `Attention` out of `StableHLO`'s imports. -/
noncomputable def rowSoftmaxBackFlat (m n : Nat) (preAct dy : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i =>
    let p := softmax n ((Mat.unflatten preAct) i)
    let dyi := (Mat.unflatten dy) i
    let s := ∑ j, p j * dyi j
    fun c => p c * (dyi c - s))

-- ── Chapter 9 (ViT) den helpers — flattened matrix/row-wise forms, spelled
--    with `Mat`/`bnForward`/`dense` so `StableHLO` needn't import `Attention`
--    (the rfl ties to `rowSoftmax`-style Attention forms live in ViTFwdGraph). ──

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
    three-term `bn_grad_input`, recomputing x̂/istd from the saved pre-LN input. -/
noncomputable def rowLNBackFlat (m n : Nat) (ε γ : ℝ) (x dy : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => bn_grad_input n ε γ ((Mat.unflatten x) i) ((Mat.unflatten dy) i))

/-- **Per-token dense (flattened)** — every row of the `[N,a]` flat through the
    same `dense W b`. -/
noncomputable def rowDenseFlat (N a c : Nat) (W : Mat a c) (b : Vec c) (v : Vec (N*a)) :
    Vec (N*c) :=
  Mat.flatten (fun i => dense W b ((Mat.unflatten v) i))

/-- **Per-token dense input-VJP (flattened)** — per row `dX = W·dy` (=
    `(dense_has_vjp W b).backward`'s `Mat.mulVec W`, MLP.lean). -/
noncomputable def rowDenseBackFlat (N a c : Nat) (W : Mat a c) (dy : Vec (N*c)) :
    Vec (N*a) :=
  Mat.flatten (fun i => Mat.mulVec W ((Mat.unflatten dy) i))

/-- **ViT patch embedding (flattened)** — a LOCAL re-spelling of the proven
    `patchEmbed_flat` (Attention.lean), kept here so `StableHLO` needn't import
    `Attention` (the tie is an `rfl` lemma in ViTFwdGraph). Output row `n`:
    CLS token at `n = 0`, else conv-projection of patch `n−1` + bias; plus the
    position embedding everywhere. -/
noncomputable def patchEmbedFlat
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
          ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv d c kh kw *
              (let W' := W / patchSize
               let p := n.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : hh < H ∧ ww < W then
                 img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
               else 0))

/-- **ViT patch-embedding input-VJP (flattened)** — a LOCAL re-spelling of the
    proven `patchEmbed_input_grad_formula` (Attention.lean), kept here so
    `StableHLO` needn't import `Attention` (the tie is an `rfl` lemma in
    ViTBackB0). The closed-form image cotangent: a sum over patches `p : Fin N`
    with reconstructed kernel offsets `(kh, kw)` matching the decoded input
    position `(c, hh, ww)`. The CLS row (`n = 0`) and the position-add (a
    +constant, input-VJP = id) contribute nothing — `idx_in` only flows through
    the conv-projection branch (`n = p+1`), so this is purely the strided 16×16
    patchify conv's input-VJP on the patch-token part of the cotangent. -/
noncomputable def patchEmbedBackFlat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize)
    (dy : Vec ((N + 1) * D)) : Vec (ic * H * W) :=
  fun idx_in =>
    let c  := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1
    let hh := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let ww := (finProdFinEquiv.symm idx_in).2
    ∑ p : Fin N, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
      let W' := W / patchSize
      let h' := p.val / W'
      let w' := p.val % W'
      if _h_match : h' * patchSize + kh.val = hh.val ∧
                    w' * patchSize + kw.val = ww.val then
        ∑ d : Fin D, W_conv d c kh kw *
          dy (finProdFinEquiv (p.succ, d))
      else 0

/-- **ViT patch-embedding weight-grad (flattened)** — a LOCAL re-spelling of the proven
    `patchEmbed_weight_grad` (Attention.lean), kept here so `StableHLO` needn't import
    `Attention` (the tie is the §1-fold `vit_render_patchW_certified`). The non-overlapping
    16×16/s16 patchify conv's weight-VJP: `dW_(d,c,kh,kw) = Σ_patches (patch pixel read)·
    dy_(patch.succ, d)` — token 0 is the CLS row (excluded); the pixel read mirrors
    `patchEmbedFlat`'s, and `dy (finProdFinEquiv (p.succ, d))` mirrors `patchEmbedBackFlat`. -/
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
      and the trap `planning/bf16_renderer.md` §9.2 exists to name.
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
    `cls_slice_flat`, Attention.lean; tie is `rfl` in ViTFwdGraph). -/
noncomputable def clsSliceFlat (N D : Nat) (v : Vec ((N+1)*D)) : Vec D :=
  fun k => v (finProdFinEquiv ((0 : Fin (N + 1)), k))

/-- **CLS pad (flattened)** — scatter `dy` to row 0, zeros elsewhere (= the proven
    `cls_slice_flat_has_vjp.backward`; tie is `rfl` in ViTFwdGraph). -/
noncomputable def clsPadFlat (N D : Nat) (dy : Vec D) : Vec ((N+1)*D) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    if p.1 = (0 : Fin (N + 1)) then dy p.2 else 0

/-- **Per-head column slice (flattened)** — head `h`'s `[N,d]` block of the
    `[N,heads·d]` flat: the `finProdFinEquiv (h, ·)` column gather `mhsa_layer`
    uses to feed each head's SDPA. -/
noncomputable def headSliceFlat (N heads d : Nat) (h : Fin heads)
    (v : Vec (N*(heads*d))) : Vec (N*d) :=
  Mat.flatten (fun (r : Fin N) (j : Fin d) =>
    (Mat.unflatten v) r (finProdFinEquiv (h, j)))

/-- **Per-head column pad (flattened)** — scatter an `[N,d]` head block into head
    `h`'s columns of a zero `[N,heads·d]`. `mhsa_layer`'s concat is the sum of
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
  | _, _, .convStridedBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (flatConvStride2 (fun o c a d => rnd (W o c a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun o _ _ => bias o) i
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
  | _, _, .depthwiseStridedXla _ _ W bias => depthwiseStride2FlatXla W bias
  | _, _, .depthwiseStridedXlaBf16 (h := h) (w := w) rnd _ _ W bias =>
      fun x i => rnd (depthwiseStride2FlatXla (fun cc a d => rnd (W cc a d)) 0 (fun j => rnd (x j)) i)
                 + Tensor3.flatten (fun cc _ _ => bias cc) i
  | _, _, .dense _ _ W bias => dense W bias
  | _, _, .gap (c := c) (h := h) (w := w) => globalAvgPoolFlat c h w
  | _, _, .seBlock (h := h) (w := w) _ _ _ _ W₁ b₁ W₂ b₂ => seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂
  | _, _, .bnEval (oc := oc) (h := h) (w := w) _ _ _ _ _ ε γ β μ var =>
      bnPerChannelEvalTensor3 oc h w ε γ β μ var
  | _, _, .swish (n := n) => swish n
  | _, _, .relu (n := n) => relu n
  | _, _, .relu6 (n := n) => relu6 n
  | _, _, .maxPool (c := c) (h := h) (w := w) => maxPoolFlat c h w
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

/-- **True batch-norm at the network's left-assoc `[N,C,H,W]` flat index.** The
    proven `bnBatchTensor4` (typed at `N·(oc·(h·w))`) conjugated by the `mul_assoc`
    reindex so it slots into the `N·(oc·h·w)` batched composition (where conv/etc.
    produce `oc·h·w = (oc·h)·w`). Reindex only — the function IS `bnBatchTensor4`. -/
noncomputable def bnBatchLA (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (oc * h * w)) → Vec (N * (oc * h * w)) :=
  fun v =>
    (fun y => y ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))
      (bnBatchTensor4 N oc h w ε γ β
        (v ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm))

/-- Which BatchNorm a forward chain emits — the batched-index peer of `ResNet34Render.R34Bn`,
    shared by the EfficientNet and MobileNetV2 renders so one traversal can produce both the
    training forward and its frozen-stats eval partner.

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
        - lr * (conv2d_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (conv2d_bias_grad_has_vjp W x).backward b (den e) o
  -- Param gradients, un-fused (the `*Sgd` bodies above with `θ − lr·` stripped off).
  | _, .weightGrad _ x e     => Mat.flatten (fun i j => x i * den e j)
  | _, .biasGrad e           => den e
  | _, .convWeightGrad _ b x W e =>
      (conv2d_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e)
  | _, .convBiasGrad W x b e => (conv2d_bias_grad_has_vjp W x).backward b (den e)
  | _, .convStridedWeightGrad _ b x W e =>
      (flatConvStride2_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e)
  | _, .convStride4WeightGrad _ b x W e =>
      (flatConvStride4_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e)
  | _, .convStridedBiasGrad W x b e => (flatConvStride2_bias_grad_has_vjp W x).backward b (den e)
  | _, .bnGammaGrad (oc := oc) (h := h) (w := w) _ _ ε v e =>
      bnPerChannel_grad_gamma oc (h*w) ε (reassocFwd oc h w v) (reassocFwd oc h w (den e))
  | _, .bnBetaGrad (oc := oc) (h := h) (w := w) e =>
      bnPerChannel_grad_beta oc (h*w) (reassocFwd oc h w (den e))
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
        bnPerChannel_grad_gamma oc (h*w) ε (reassocFwd oc h w v) (reassocFwd oc h w (den e)) c
  | _, .bnBetaSgd (oc := oc) (h := h) (w := w) _ _ β lr e =>
      fun c => β c - lr * bnPerChannel_grad_beta oc (h*w) (reassocFwd oc h w (den e)) c
  | _, .layerScaleChGammaSgd (c := c) (h := h) (w := w) _ _ _ x γ lr e =>
      fun cc => γ cc - lr * ∑ k : Fin (c*h*w), (if chanIdx c h w k = cc then x k * den e k else 0)
  | _, .lnGammaSgd (n := n) _ _ _ _ ε x γ lr e =>
      fun _ => γ 0 - lr * bn_grad_gamma n ε x (den e)
  | _, .lnBetaSgd (n := n) _ _ β lr e =>
      fun _ => β 0 - lr * bn_grad_beta n (den e)
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
        bnPerChannel_grad_gamma oc (N*(h*w)) ε (bnchwFwd N oc h w v) (bnchwFwd N oc h w (den e)) c
  | _, .bnBetaSgdB (N := N) (oc := oc) (h := h) (w := w) _ _ β lr e =>
      fun c => β c - lr * bnPerChannel_grad_beta oc (N*(h*w)) (bnchwFwd N oc h w (den e)) c
  | _, .denseWeightSgdB (N := N) (a := a) (c := c) _ _ _ x W lr e =>
      Mat.flatten (fun i j => W i j - lr * ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .denseBiasSgdB (N := N) (c := c) _ _ b lr e =>
      fun j => b j - lr * ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .convWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (conv2d_weight_grad_has_vjp b (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (flatConvStride2_weight_grad_has_vjp b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (conv2d_weight_grad_has_vjp b
          (Tensor3.unflatten (fun j => rnd (batchSlice N (ic*h*w) x n j)))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convStridedWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (flatConvStride2_weight_grad_has_vjp b
          (fun j => rnd (batchSlice N (ic*(2*h)*(2*w)) x n j))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convBiasGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (conv2d_bias_grad_has_vjp W (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .convStridedBiasGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (flatConvStride2_bias_grad_has_vjp W (batchSlice N (ic*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  -- The XLA-`SAME` peers. ⚠ Only the CERT changes (`…Xla…`); the shape of the batch sum is
  -- identical, which is precisely why this is easy to get wrong by copy-paste.
  | _, .convStridedXlaWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (flatConvStride2Xla_weight_grad_has_vjp b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedXlaWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (flatConvStride2Xla_weight_grad_has_vjp b
          (fun j => rnd (batchSlice N (ic*(2*h)*(2*w)) x n j))).backward
          (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc*h*w) (den e) n j)) idx)
  | _, .convStridedXlaBiasGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (flatConvStride2Xla_bias_grad_has_vjp W (batchSlice N (ic*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .bnGammaGradB (N := N) (oc := oc) (h := h) (w := w) _ _ ε v e =>
      fun c =>
        bnPerChannel_grad_gamma oc (N*(h*w)) ε (bnchwFwd N oc h w v) (bnchwFwd N oc h w (den e)) c
  | _, .bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) e =>
      fun c => bnPerChannel_grad_beta oc (N*(h*w)) (bnchwFwd N oc h w (den e)) c
  | _, .denseWeightGradB (N := N) (a := a) (c := c) _ x e =>
      Mat.flatten (fun i j => ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .denseBiasGradB (N := N) (c := c) e =>
      fun j => ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .bnBatchMeanB (N := N) (oc := oc) (h := h) (w := w) e =>
      fun c => bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c)
  | _, .bnBatchVarB (N := N) (oc := oc) (h := h) (w := w) e =>
      fun c => bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c)
  | _, .scaleB _ s e    => fun i => den e i * s
  | _, .shiftB _ s e    => fun i => den e i + s
  | _, .divConstB _ s e => fun i => den e i / s
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
        (conv2d_weight_grad_has_vjp b (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (flatConvStride2_weight_grad_has_vjp b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedXlaWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (flatConvStride2Xla_weight_grad_has_vjp b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .depthwiseWeightSgdB (N := N) (c := c) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Tensor3.flatten W idx - lr * ∑ n : Fin N,
        Tensor3.flatten ((depthwise_weight_grad_has_vjp3 b (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward
          W (Tensor3.unflatten (batchSlice N (c*h*w) (den e) n))) idx
  | _, .depthwiseStridedWeightSgdB (N := N) (c := c) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Tensor3.flatten W idx - lr * ∑ n : Fin N,
        (depthwiseStride2_weight_grad_has_vjp b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
  | _, .depthwiseWeightGradB (N := N) (c := c) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        Tensor3.flatten ((depthwise_weight_grad_has_vjp3 b (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward
          W (Tensor3.unflatten (batchSlice N (c*h*w) (den e) n))) idx
  -- ⚠ `rnd` OUTSIDE the `Σ_n`: the emit makes the batch the convolution's contraction dim, so the
  -- whole sum is ONE convolution and therefore ONE bf16 store. Inside would model N stores.
  | _, .depthwiseWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        Tensor3.flatten ((depthwise_weight_grad_has_vjp3 b
          (Tensor3.unflatten (fun j => rnd (batchSlice N (c*h*w) x n j)))).backward
          W (Tensor3.unflatten (fun j => rnd (batchSlice N (c*h*w) (den e) n j)))) idx)
  -- The ConvNeXt five. Each is exactly the gradient half of its `*Sgd` peer's `den`, so the
  -- `*Sgd_eq_grad` theorems below are `rfl`.
  | _, .depthwiseWeightGrad _ b x W e =>
      fun idx => Tensor3.flatten
        ((depthwise_weight_grad_has_vjp3 b x).backward W (Tensor3.unflatten (den e))) idx
  | _, .depthwiseBiasGrad W x b e => fun o => (depthwise_bias_grad_has_vjp W x).backward b (den e) o
  | _, .lnGammaGrad (n := n) _ _ ε x e => fun _ => bn_grad_gamma n ε x (den e)
  | _, .lnBetaGrad (n := n) e => fun _ => bn_grad_beta n (den e)
  | _, .layerScaleChGammaGrad (c := c) (h := h) (w := w) _ x e =>
      fun cc => ∑ k : Fin (c*h*w), (if chanIdx c h w k = cc then x k * den e k else 0)
  | _, .depthwiseStridedWeightGradB (N := N) (c := c) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (depthwiseStride2_weight_grad_has_vjp b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
  | _, .depthwiseStridedWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (depthwiseStride2_weight_grad_has_vjp b
          (fun j => rnd (batchSlice N (c*(2*h)*(2*w)) x n j))).backward
          (Tensor3.flatten W) (fun j => rnd (batchSlice N (c*h*w) (den e) n j)) idx)
  -- The depthwise bias grads: the shared-parameter batch sum every `*GradB` takes, `Σ_n dβ_n`.
  -- Same shape as `convBiasGradB`/`convStridedBiasGradB` one row up, with the depthwise VJP certs.
  | _, .depthwiseBiasGradB (N := N) (c := c) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (depthwise_bias_grad_has_vjp W (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward b
          (batchSlice N (c*h*w) (den e) n) o
  | _, .depthwiseStridedBiasGradB (N := N) (c := c) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (depthwiseStride2_bias_grad_has_vjp W (batchSlice N (c*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (c*h*w) (den e) n) o
  | _, .depthwiseStridedXlaWeightGradB (N := N) (c := c) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (depthwiseStride2Xla_weight_grad_has_vjp b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
  | _, .depthwiseStridedXlaWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (depthwiseStride2Xla_weight_grad_has_vjp b
          (fun j => rnd (batchSlice N (c*(2*h)*(2*w)) x n j))).backward
          (Tensor3.flatten W) (fun j => rnd (batchSlice N (c*h*w) (den e) n j)) idx)
  | _, .depthwiseStridedXlaBiasGradB (N := N) (c := c) (h := h) (w := w) W x b e =>
      fun o => ∑ n : Fin N,
        (depthwiseStride2Xla_bias_grad_has_vjp W (batchSlice N (c*(2*h)*(2*w)) x n)).backward b
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
  | _, .convBack _ W b v e => (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward v (den e)
  | _, .maxPoolBack (c := c) (h := h) (w := w) _ x e => maxPoolBackFlat c h w x (den e)
  | _, .maxPool3s2Back (c := c) (h := h) (w := w) _ x e => maxPool3s2BackFlat c h w x (den e)
  | _, .bnF (n := n) _ _ _ ε γ β e => bnForward n ε γ β (den e)
  | _, .bnBack (n := n) _ _ _ ε γ x e => bn_grad_input n ε γ x (den e)
  | _, .addV a b       => fun j => den a j + den b j
  | _, .addVB a b      => fun j => den a j + den b j
  | _, .subB a b       => fun j => den a j - den b j
  | _, .gapF (c := c) (h := h) (w := w) e => globalAvgPoolFlat c h w (den e)
  | _, .gapBack (c := c) (h := h) (w := w) e =>
      (globalAvgPoolFlat_has_vjp c h w).backward (fun _ => 0) (den e)
  | _, .broadcastBack (c := c) (h := h) (w := w) e =>
      fun k => ∑ idx : Fin (c * h * w), if flatChannel c h w idx = k then den e idx else 0
  | _, .flatConvStridedF _ _ W b e => flatConvStride2 W b (den e)
  | _, .flatConvStridedXlaF _ _ W b e => flatConvStride2Xla W b (den e)
  | _, .flatConvStride4F _ _ W b e => flatConvStride4 W b (den e)
  | _, .convStridedBack _ W b v e => (flatConvStride2_has_vjp W b).backward v (den e)
  | _, .convStridedWeightSgd _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx
        - lr * (flatConvStride2_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convStridedBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (flatConvStride2_bias_grad_has_vjp W x).backward b (den e) o
  | _, .depthwiseWeightSgd _ _ _ b x W lr e => depthwiseWeightSgdDen b x W lr (den e)
  | _, .depthwiseBiasSgd _ _ W x b lr e => depthwiseBiasSgdDen W x b lr (den e)
  | _, .depthwiseStridedWeightSgd _ _ _ b x W lr e => depthwiseStridedWeightSgdDen b x W lr (den e)
  | _, .depthwiseStridedBiasSgd _ _ W x b lr e => depthwiseStridedBiasSgdDen W x b lr (den e)
  | _, .bnPerChannelF (oc := oc) (h := h) (w := w) _ _ _ ε γ β e =>
      bnPerChannelTensor3 oc h w ε γ β (den e)
  | _, .bnPerChannelBack (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      bnPerChannelTensor3_grad_input oc h w ε γ x (den e)
  | _, .bnPerChannelEvalF (oc := oc) (h := h) (w := w) _ _ _ _ _ ε γ β μ var e =>
      bnPerChannelEvalTensor3 oc h w ε γ β μ var (den e)
  | _, .depthwiseF _ _ W b e => depthwiseFlat W b (den e)
  | _, .depthwiseBack _ W b v e => (depthwiseFlat_has_vjp W b).backward v (den e)
  | _, .depthwiseStridedF _ _ W b e => depthwiseStride2Flat W b (den e)
  | _, .depthwiseStridedXlaF _ _ W b e => depthwiseStride2FlatXla W b (den e)
  | _, .depthwiseStridedBack _ W b v e => (depthwiseStride2Flat_has_vjp W b).backward v (den e)
  | _, .swishF (n := n) e => swish n (den e)
  | _, .swishBack (n := n) _ x e => (swish_has_vjp n).backward x (den e)
  | _, .sigmoidF (n := n) e => sigmoid n (den e)
  | _, .sigmoidBack (n := n) _ x e => (sigmoid_has_vjp n).backward x (den e)
  | _, .maxPoolBackB (N := N) (c := c) (h := h) (w := w) _ x e =>
      batchMapAux N (maxPoolBackFlat c h w) x (den e)
  | _, .maxPool3s2BackB (N := N) (c := c) (h := h) (w := w) _ x e =>
      batchMapAux N (maxPool3s2BackFlat c h w) x (den e)
  | _, .convBiasSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ W x b lr e =>
      fun o => b o - lr * ∑ n : Fin N,
        (conv2d_bias_grad_has_vjp W (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .convStridedBiasSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ W x b lr e =>
      fun o => b o - lr * ∑ n : Fin N,
        (flatConvStride2_bias_grad_has_vjp W (batchSlice N (ic*(2*h)*(2*w)) x n)).backward b
          (batchSlice N (oc*h*w) (den e) n) o
  | _, .selectPosB _ x e => fun i => if x i > 0 then den e i else 0
  | _, .selectMidB _ x e => fun i => if 0 < x i ∧ x i < 6 then den e i else 0
  | _, .dropPathB (N := N) (n := n) _ s e => Proofs.dropPath N n s (den e)
  | _, .dropoutB (N := N) (n := n) _ mask e => Proofs.dropout N n mask (den e)
  | _, .swishBackB (N := N) (n := n) _ x e => (swish_has_vjp (N*n)).backward x (den e)
  -- `gelu` is POINTWISE, so its VJP at the batched width `N*n` is already the batch-lift of the
  -- per-example one — the same argument `swishBackB` rests on, and why neither needs `batchMapAux`.
  | _, .geluBackB (N := N) (n := n) _ x e => (gelu_has_vjp (N*n)).backward x (den e)
  -- The `Σ_n` shape, verbatim from `convWeightGradB`: a shared parameter's batched gradient is the
  -- sum over examples of the per-example gradient on `batchSlice n`.
  | _, .convStride4WeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ b x W e =>
      fun idx => ∑ n : Fin N,
        (flatConvStride4_weight_grad_has_vjp b
            (batchSlice N (ic*(2*(2*h))*(2*(2*w))) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  -- The bf16 peer. ⚠ `rnd` OUTSIDE the `Σ_n`: the emit contracts the batch inside one convolution
  -- and stores its bf16 result once, so a rounding per summand would claim a coarser computation
  -- than the hardware performs — the same reason `convWeightGradBBf16` is written this way.
  | _, .convStride4WeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) rnd _ b x W e =>
      fun idx => rnd (∑ n : Fin N,
        (flatConvStride4_weight_grad_has_vjp b
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
  | _, .sigmoidBackB (N := N) (n := n) _ x e => (sigmoid_has_vjp (N*n)).backward x (den e)
  | _, .geluF (n := n) e => gelu n (den e)
  | _, .geluBack (n := n) _ x e => (gelu_has_vjp n).backward x (den e)
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
      bnBatchTensor4_grad_input N oc h w ε γ x (den e)
  | _, .convBackBatched (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward (fun _ => 0) dy) (den e)
  | _, .convStridedBackBatched (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (flatConvStride2_has_vjp W b).backward (fun _ => 0) dy) (den e)
  | _, .convBackBatchedBf16 (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((hasVJP3_to_hasVJP
        (conv2d_has_vjp3 (fun o c a d => rnd (W o c a d)) b)).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .convStridedBackBatchedBf16 (N := N) (ic := ic) (oc := _oc) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((flatConvStride2_has_vjp
        (fun o c a d => rnd (W o c a d)) b).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .depthwiseBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (hasVJP3_to_hasVJP (depthwise_has_vjp3 W b)).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((hasVJP3_to_hasVJP
        (depthwise_has_vjp3 (fun cc a d => rnd (W cc a d)) b)).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .depthwiseStridedBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (depthwiseStride2Flat_has_vjp W b).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseStridedBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((depthwiseStride2Flat_has_vjp
        (fun cc a d => rnd (W cc a d)) b).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .depthwiseStridedXlaBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (depthwiseStride2FlatXla_has_vjp W b).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseStridedXlaBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) rnd _ W b e =>
      batchMap N (fun dy i => rnd ((depthwiseStride2FlatXla_has_vjp
        (fun cc a d => rnd (W cc a d)) b).backward
          (fun _ => 0) (fun j => rnd (dy j)) i)) (den e)
  | _, .bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      fun i => ∑ k, if i = (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) k then
        bnBatchTensor4_grad_input N oc h w ε γ
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
          (fun i' => ∑ k', if i' = (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w))) k'
                           then den e k' else 0) k
        else 0
  | _, .seBackBatched (h := h) (w := w) _ _ _ _ _ W₁ b₁ W₂ b₂ v e =>
      fun idx =>
        (seBlockFull_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward
          (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (Mat.unflatten (den e) (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2
  | _, .seReduceB (N := N) (c := c) (h := h) (w := w) _ x e =>
      -- the SE gate cotangent: per example, the broadcast-adjoint of `x ⊙ dy`
      -- (`broadcastFlat_has_vjp.backward` = sum each channel's spatial Hadamard).
      fun idx =>
        ∑ q : Fin (c * h * w),
          if flatChannel c h w q = (finProdFinEquiv.symm idx).2 then
            batchSlice N (c * h * w) x (finProdFinEquiv.symm idx).1 q
              * batchSlice N (c * h * w) (den e) (finProdFinEquiv.symm idx).1 q
          else 0
  | _, .gapBackBatched (N := N) (c := c) (h := h) (w := w) e =>
      batchMap N (fun dgap => (globalAvgPoolFlat_has_vjp c h w).backward (fun _ => 0) dgap) (den e)

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
@[simp] theorem den_sub {n : Nat} (a b : SHlo n) :
    den (.sub a b) = fun j => den a j - den b j := rfl
@[simp] theorem den_addV {n : Nat} (a b : SHlo n) :
    den (.addV a b) = fun j => den a j + den b j := rfl
@[simp] theorem den_reluF {n : Nat} (e : SHlo n) :
    den (.reluF e) = fun i => max (den e i) 0 := rfl
@[simp] theorem den_selectPos {n : Nat} (s : String) (x : Vec n) (e : SHlo n) :
    den (.selectPos s x e) = fun i => if x i > 0 then den e i else 0 := rfl
/-- **The round node is `den`-faithful for any rounding.** This is the equation
    `Proofs/Float/Bf16FaithfulPoC.lean` asks for by name to lift its depth-1 tie to
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

-- Batched-lift faithfulness: `den` of each batched token = `batchMap N` of the
-- proven per-example op (rfl, since `denOp` returns that op directly), and the
-- true-batch-norm token denotes `bnBatchLA` (= the proven `bnBatchTensor4`).
@[simp] theorem den_batchOp_conv {N ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (bias : Vec oc) (e : SHlo (N * (ic*h*w))) :
    den (.batchOp (N := N) (.conv (h := h) (w := w) wN bN W bias) e)
      = batchMap N (flatConv W bias) (den e) := rfl
@[simp] theorem den_batchOp_convStrided {N ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (bias : Vec oc) (e : SHlo (N * (ic*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.convStrided (h := h) (w := w) wN bN W bias) e)
      = batchMap N (flatConvStride2 W bias) (den e) := rfl
/-- The XLA-`SAME` peer. ⚠ Note it denotes `flatConvStride2Xla`, NOT `flatConvStride2` — the two
    tokens have identical types and identical emitted shapes, so this `rfl` is the only place the
    distinction is recorded. Getting it wrong would make the render provably compute one net while
    emitting the other. -/
@[simp] theorem den_batchOp_convStridedXla {N ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (bias : Vec oc) (e : SHlo (N * (ic*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.convStridedXla (h := h) (w := w) wN bN W bias) e)
      = batchMap N (flatConvStride2Xla W bias) (den e) := rfl
@[simp] theorem den_batchOp_depthwise {N c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.depthwise (h := h) (w := w) wN bN W bias) e)
      = batchMap N (depthwiseFlat W bias) (den e) := rfl
@[simp] theorem den_batchOp_depthwiseStrided {N c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) (e : SHlo (N * (c*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.depthwiseStrided (h := h) (w := w) wN bN W bias) e)
      = batchMap N (depthwiseStride2Flat W bias) (den e) := rfl
/-- The XLA-`SAME` depthwise peer. ⚠ Denotes `depthwiseStride2FlatXla`, NOT `depthwiseStride2Flat`
    — same caveat as `den_batchOp_convStridedXla`: this `rfl` is the only place the two are
    distinguished. -/
@[simp] theorem den_batchOp_depthwiseStridedXla {N c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) (e : SHlo (N * (c*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.depthwiseStridedXla (h := h) (w := w) wN bN W bias) e)
      = batchMap N (depthwiseStride2FlatXla W bias) (den e) := rfl
@[simp] theorem den_batchOp_dense {N a c : Nat} (wN bN : String)
    (W : Mat a c) (bias : Vec c) (e : SHlo (N * a)) :
    den (.batchOp (N := N) (.dense wN bN W bias) e)
      = batchMap N (dense W bias) (den e) := rfl
/-- **Batched inference-BN faithfulness.** The `bnEval` descriptor at the batched index denotes the
    proven `bnPerChannelEvalTensor3` applied to **each example independently, with the same frozen
    statistics**. That is the formal statement of "eval is class-batch-independent" at `N := B`: no
    `N` appears on the right except as the number of independent applications, so an example's
    logits cannot depend on which others share its batch — unlike `bnBatchF`, whose `den`
    (`bnBatchLA`) genuinely couples the batch. Being affine in `x`, it needs no `0 < ε`. -/
@[simp] theorem den_batchOp_bnEval {N oc h w : Nat} (gN bN muN varN es : String) (ε : ℝ)
    (γ β μ var : Vec oc) (e : SHlo (N * (oc*h*w))) :
    den (.batchOp (N := N) (.bnEval (h := h) (w := w) gN bN muN varN es ε γ β μ var) e)
      = batchMap N (bnPerChannelEvalTensor3 oc h w ε γ β μ var) (den e) := rfl
@[simp] theorem den_batchOp_gap {N c h w : Nat} (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.gap (c := c) (h := h) (w := w)) e)
      = batchMap N (globalAvgPoolFlat c h w) (den e) := rfl
@[simp] theorem den_batchOp_seBlock {N c h w r : Nat} (w1 b1 w2 b2 : String)
    (W₁ : Mat c r) (β₁ : Vec r) (W₂ : Mat r c) (β₂ : Vec c) (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.seBlock (h := h) (w := w) w1 b1 w2 b2 W₁ β₁ W₂ β₂) e)
      = batchMap N (seBlockFull (h := h) (w := w) W₁ β₁ W₂ β₂) (den e) := rfl
@[simp] theorem den_batchOp_swish {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.swish (n := n)) e) = batchMap N (swish n) (den e) := rfl

/-- **Pointwise maps are `batchMap`-free.** Lifting an elementwise map across `N` examples IS the
    elementwise map at the batched index `N·n`. This is why moving the pointwise nodes onto
    descriptors was denotation-preserving, and it is the half of that claim the artifact cannot
    witness: the render is value-independent, so a descriptor with the wrong `den` emits the same
    bytes. Cf. `swishBackB`/`sigmoidBackB`, which are NOT descriptors precisely because their
    backward is not of this shape — it reads a per-example saved activation. -/
theorem batchMap_pointwise {N n : Nat} (g : ℝ → ℝ) (v : Vec (N * n)) :
    batchMap N (fun (x : Vec n) i => g (x i)) v = fun idx => g (v idx) := by
  funext idx
  simp only [batchMap]
  exact congrArg g (congrArg v (Equiv.apply_symm_apply finProdFinEquiv idx))

/-- The descriptor form of swish denotes exactly what the descriptor-less `swishF` denoted at the
    same index — the batched graph computes the same function, only the emit width now travels
    separately from the batch. -/
theorem den_batchOp_swish_eq_swishF {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.swish (n := n)) e) = den (.swishF e) :=
  batchMap_pointwise swishScalar (den e)
@[simp] theorem den_batchOp_softmaxRow {N m n : Nat} (e : SHlo (N * (m*n))) :
    den (.batchOp (N := N) (.softmaxRow (m := m) (n := n)) e)
      = batchMap N (rowSoftmaxFlat m n) (den e) := rfl
@[simp] theorem den_batchOp_denseRowBack {N rows a c : Nat} (wN : String) (W : Mat a c)
    (e : SHlo (N * (rows*c))) :
    den (.batchOp (N := N) (.denseRowBack (rows := rows) wN W) e)
      = batchMap N (rowDenseBackFlat rows a c W) (den e) := rfl
@[simp] theorem den_batchOp_relu {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.relu (n := n)) e) = batchMap N (relu n) (den e) := rfl
-- ── ViT / ConvNeXt's five row/pointwise descriptors (§0.2 ▶2). The DENOTATION half of the
--    batched-index move, and it is the half `tests/TestBatchedEmitTie.lean` structurally cannot
--    see: `skel` erases values, so a descriptor with the wrong `denOp` emits identical bytes.
--    Each is `batchMap N` of the SAME proven per-example map its descriptor-less peer denotes,
--    by `rfl` — which is what makes the batched node honest at `N := B` rather than a one-example
--    function wearing a `tensor<Bxn>` type.
@[simp] theorem den_batchOp_gelu {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.gelu (n := n)) e) = batchMap N (gelu n) (den e) := rfl
@[simp] theorem den_batchOp_dotOut {N m n : Nat} (wN : String) (W : Mat m n) (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.dotOut wN W) e)
      = batchMap N (fun v i => ∑ j, W i j * v j) (den e) := rfl
@[simp] theorem den_batchOp_expe {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.expe (n := n)) e)
      = batchMap N (fun v j => Real.exp (v j)) (den e) := rfl
@[simp] theorem den_batchOp_softmaxDiv {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.softmaxDiv (n := n)) e)
      = batchMap N (fun v j => v j / ∑ k, v k) (den e) := rfl
@[simp] theorem den_batchOp_layerScaleCh {N c h w : Nat} (gN : String) (γ : Vec c)
    (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.layerScaleCh (c := c) (h := h) (w := w) gN γ) e)
      = batchMap N (fun v => layerScale (fun k => γ (chanIdx c h w k)) v) (den e) := rfl
@[simp] theorem den_batchOp_convStride4 {N ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (bias : Vec oc)
    (e : SHlo (N * (ic*(2*(2*h))*(2*(2*w))))) :
    den (.batchOp (N := N) (.convStride4 (h := h) (w := w) wN bN W bias) e)
      = batchMap N (flatConvStride4 W bias) (den e) := rfl

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
  simp [den_batchOp_softmaxDiv, batchMap, batchSlice]
@[simp] theorem den_batchOp_transpose {N m n : Nat} (e : SHlo (N * (m*n))) :
    den (.batchOp (N := N) (.transpose (m := m) (n := n)) e)
      = batchMap N (transposeFlat m n) (den e) := rfl
@[simp] theorem den_batchOp_lnRow {N m n : Nat} (gN bN es : String) (ε γ β : ℝ)
    (e : SHlo (N * (m*n))) :
    den (.batchOp (N := N) (.lnRow (m := m) (n := n) gN bN es ε γ β) e)
      = batchMap N (rowLNFlat m n ε γ β) (den e) := rfl
@[simp] theorem den_batchOp_rowScale {N m n : Nat} (gN : String) (γ : Vec n)
    (e : SHlo (N * (m*n))) :
    den (.batchOp (N := N) (.rowScale (m := m) (n := n) gN γ) e)
      = batchMap N (rowScaleFlat m n γ) (den e) := rfl
@[simp] theorem den_batchOp_rowBias {N m n : Nat} (bN : String) (β : Vec n)
    (e : SHlo (N * (m*n))) :
    den (.batchOp (N := N) (.rowBias (m := m) (n := n) bN β) e)
      = batchMap N (rowBiasFlat m n β) (den e) := rfl

/-! ### ViT increment 1 — the six batch-invariant forms

⚠ Read the binders: `N` is the BATCH and `tk` is ViT's token count. The per-example renderer calls
the token axis `N`, so these two statements are the place where that name is re-pointed, and
getting them the wrong way round type-checks (both are `Nat` and both appear multiplied). -/

@[simp] theorem den_batchOp_denseRow {N tk a c : Nat} (wN bN : String) (W : Mat a c) (b : Vec c)
    (e : SHlo (N * (tk*a))) :
    den (.batchOp (N := N) (.denseRow (N := tk) wN bN W b) e)
      = batchMap N (rowDenseFlat tk a c W b) (den e) := rfl

@[simp] theorem den_batchOp_patchEmbed {N ic H W P tk D : Nat} (wN bN clsN posN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (tk+1) D) (e : SHlo (N * (ic*H*W))) :
    den (.batchOp (N := N) (.patchEmbed (N := tk) wN bN clsN posN Wc bc cls pos) e)
      = batchMap N (patchEmbedFlat ic H W P tk D Wc bc cls pos) (den e) := rfl

@[simp] theorem den_batchOp_clsSlice {N tk D : Nat} (e : SHlo (N * ((tk+1)*D))) :
    den (.batchOp (N := N) (.clsSlice (N := tk) (D := D)) e)
      = batchMap N (clsSliceFlat tk D) (den e) := rfl

@[simp] theorem den_batchOp_clsPad {N tk D : Nat} (e : SHlo (N * D)) :
    den (.batchOp (N := N) (.clsPad (N := tk) (D := D)) e)
      = batchMap N (clsPadFlat tk D) (den e) := rfl

@[simp] theorem den_batchOp_headSlice {N tk heads d : Nat} (h : Fin heads)
    (e : SHlo (N * (tk*(heads*d)))) :
    den (.batchOp (N := N) (.headSlice (N := tk) (d := d) h) e)
      = batchMap N (headSliceFlat tk heads d h) (den e) := rfl

@[simp] theorem den_batchOp_headPad {N tk heads d : Nat} (h : Fin heads)
    (e : SHlo (N * (tk*d))) :
    den (.batchOp (N := N) (.headPad (N := tk) (heads := heads) h) e)
      = batchMap N (headPadFlat tk heads d h) (den e) := rfl

/-- ⭐ **THE CLS SLICE IS THE ONE PLACE THE BATCH AND THE TOKEN AXIS COULD SWAP SILENTLY.**
    `clsSlice` takes `(tk+1)*D` to `D` — it CONTRACTS — and `batchMap N` of it takes `N*((tk+1)*D)`
    to `N*D`. A render that read the batch as the token axis would take `(N+1)*D` to `D`, i.e. drop
    every example but one and still type-check at `N = tk`. Stated so the two indices are pinned
    apart by a theorem rather than by a naming convention. -/
theorem den_batchOp_clsSlice_per_example {N tk D : Nat} (e : SHlo (N * ((tk+1)*D)))
    (k : Fin N) (i : Fin D) :
    den (.batchOp (N := N) (.clsSlice (N := tk) (D := D)) e) (finProdFinEquiv (k, i))
      = clsSliceFlat tk D (batchSlice N ((tk+1)*D) (den e) k) i := by
  simp only [den_batchOp_clsSlice, batchMap, Equiv.symm_apply_apply]
  rfl

/-- **The two halves agree, per form.** The batched descriptor denotes the batch-lift of exactly
    what its per-example peer denotes — stated against `den (.lnRowF …)` rather than against
    `rowLNFlat` so the claim is *"same function as the op the renderer is replacing"*, which is
    what a reader of the swapped render needs. `rfl` on both sides; kept as five separate
    statements because a `simp` set of five `batchMap` rewrites is what the whole-net faithfulness
    proof will consume. -/
theorem den_batchOp_lnRow_eq_lnRowF {N m n : Nat} (gN bN es : String) (ε γ β : ℝ)
    (e : SHlo (N * (m*n))) :
    den (.batchOp (N := N) (.lnRow (m := m) (n := n) gN bN es ε γ β) e)
      = batchMap N (fun v => den (.lnRowF gN bN es ε γ β (.operand "" v))) (den e) := rfl

/-- ViT increment 1's peer of the above, on the form that carries the most data. ⚠ Its per-example
    peer takes the TOKEN count as `N`; this one takes the BATCH as `N` and the token count as `tk`,
    and both `N`s are `Nat`. Writing the equation out is what makes the two visible at once. -/
theorem den_batchOp_denseRow_eq_denseRowF {N tk a c : Nat} (wN bN : String)
    (W : Mat a c) (b : Vec c) (e : SHlo (N * (tk*a))) :
    den (.batchOp (N := N) (.denseRow (N := tk) wN bN W b) e)
      = batchMap N (fun v => den (.denseRowF (N := tk) wN bN W b (.operand "" v))) (den e) := rfl

theorem den_batchOp_gelu_eq_geluF {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.gelu (n := n)) e)
      = batchMap N (fun v => den (.geluF (n := n) (.operand "" v))) (den e) := rfl
@[simp] theorem den_scaleB {N n : Nat} (sS : String) (s : ℝ) (e : SHlo (N*n)) :
    den (.scaleB sS s e) = fun i => den e i * s := rfl
@[simp] theorem den_shiftB {N n : Nat} (sS : String) (s : ℝ) (e : SHlo (N*n)) :
    den (.shiftB sS s e) = fun i => den e i + s := rfl
@[simp] theorem den_divConstB {N n : Nat} (sS : String) (s : ℝ) (e : SHlo (N*n)) :
    den (.divConstB sS s e) = fun i => den e i / s := rfl
@[simp] theorem den_bnBatchMeanB {N oc h w : Nat} (e : SHlo (N * (oc * (h * w)))) :
    den (.bnBatchMeanB e)
      = fun c => bnMean (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) := rfl
@[simp] theorem den_bnBatchVarB {N oc h w : Nat} (e : SHlo (N * (oc * (h * w)))) :
    den (.bnBatchVarB e)
      = fun c => bnVar (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w (den e)) c) := rfl
@[simp] theorem den_batchOp_maxPool {N c h w : Nat} (e : SHlo (N * (c*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.maxPool (c := c) (h := h) (w := w)) e)
      = batchMap N (maxPoolFlat c h w) (den e) := rfl
@[simp] theorem den_maxPoolBackB {N c h w : Nat} (xN : String) (x : Vec (N*(c*(2*h)*(2*w))))
    (e : SHlo (N*(c*h*w))) :
    den (.maxPoolBackB xN x e) = batchMapAux N (maxPoolBackFlat c h w) x (den e) := rfl
/-- ⭐ The batched 3×3/s2 pool forward denotes He et al.'s pool lifted across the batch. ⚠ Read it
    beside `den_batchOp_maxPool` directly above: **same type, different function.** The two
    descriptors are indistinguishable to every structural check the repo has — arity, op counts,
    the prefix audit and the shape of the emitted text — which is exactly how the deviation
    survived undocumented on every ResNet here. `maxPool3s2_ne_maxPool_descr` pins them apart. -/
@[simp] theorem den_batchOp_maxPool3s2 {N c h w : Nat} (e : SHlo (N * (c*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.maxPool3s2 (c := c) (h := h) (w := w)) e)
      = batchMap N (maxPool3s2Flat c h w) (den e) := rfl
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
    den (.swishBackB xN x e) = (swish_has_vjp (N*n)).backward x (den e) := rfl
@[simp] theorem den_geluBackB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.geluBackB xN x e) = (gelu_has_vjp (N*n)).backward x (den e) := rfl
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
  simp [den_rowDenseBiasGradB, Finset.sum_fin_eq_sum_range]
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
  simp [den_lnRowBackB, batchMapAux]
@[simp] theorem den_sigmoidBackB {N n : Nat} (xN : String) (x : Vec (N*n)) (e : SHlo (N*n)) :
    den (.sigmoidBackB xN x e) = (sigmoid_has_vjp (N*n)).backward x (den e) := rfl

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
  simp [den_matmulFB, batchMapAux]

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
  simp [den_softmaxRowBackB, batchMapAux]

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
  simp [den_posEmbedGradB]
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
  funext j; simp only [fwdGraph, den, mnistLinear, dense]

/-- **Dense input-VJP faithfulness.** The backward graph denotes the proven
    dense VJP backward `(dense_has_vjp W b).backward x = Mat.mulVec W`. -/
theorem backGraph_faithful (dy : Vec n) :
    den (backGraph W dy) = (dense_has_vjp W b).backward x dy := by
  funext i; simp only [backGraph, den, dense_has_vjp, Mat.mulVec]

/-- The softmax sub-graph denotes the proven `softmax`. -/
theorem softmaxDiv_expe_faithful (z : Vec n) :
    den (.softmaxDiv (.expe (.operand "%logits" z))) = softmax n z := by
  funext j; simp only [den, softmax]

/-- **Loss-cotangent faithfulness (spec level).** -/
theorem lossCotGraph_faithful (label : Fin n) :
    den (lossCotGraph W b x (oneHot n label)) = IR.emitLossCot n (mnistLinear W b x) label := by
  funext j
  simp only [lossCotGraph, IR.emitLossCot, den, oneHot, softmax, fwdGraph_faithful,
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
theorem sgdW_descends_certified_grad (lr : ℝ) (label : Fin n) (i : Fin m) (j : Fin n) :
    sgdW W b x lr label i j
      = W i j - lr * ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k
            * den (lossCotGraph W b x (oneHot n label)) k := by
  unfold sgdW
  rw [wGrad_isWeightJacobian W b x (den (lossCotGraph W b x (oneHot n label))) i j]

/-- **SGD bias-step faithfulness.** Likewise for `b`. -/
theorem sgdB_descends_certified_grad (lr : ℝ) (label : Fin n) (j : Fin n) :
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
-- whole-MLP VJP is *conditional* (`mlp_has_vjp_at`, off the kink) — exactly the
-- regime the codegen's subgradient (`relu'(0)=0`) targets. The parameter grads
-- and SGD update reuse the layer-agnostic `wGrad`/`bGrad`/`sgd*` theorems above.
-- ════════════════════════════════════════════════════════════════

/-- `maximum(a,0)` equals ReLU's pointwise `if a>0 then a else 0`. -/
private theorem max_zero_eq (a : ℝ) : max a 0 = if a > 0 then a else 0 := by
  by_cases h : (0 : ℝ) < a
  · rw [if_pos h, max_eq_left h.le]
  · rw [if_neg h, max_eq_right (not_lt.1 h)]

/-- **ReLU forward faithfulness.** `maximum(·,0)` denotes the proven `relu`. -/
theorem reluF_faithful {k : Nat} (e : SHlo k) : den (.reluF e) = relu k (den e) := by
  funext i; simp only [den, relu]; exact max_zero_eq _

/-- **ReLU backward faithfulness (smooth point).** `select(x>0,·,0)` denotes the
    proven `relu_has_vjp_at` backward — the codegen's `relu'(0)=0` convention. -/
theorem selectPos_faithful {k : Nat} (s : String) (x : Vec k) (hx : ∀ i, x i ≠ 0)
    (e : SHlo k) :
    den (.selectPos s x e) = (relu_has_vjp_at k x hx).backward (den e) := rfl

/-- The `relu` descriptor denotes exactly what the descriptor-less `reluF` denoted at the same
    index: the batched graph computes the same function, only the emit width now travels
    separately from the batch. The ResNet-34 peer of `den_batchOp_swish_eq_swishF`. -/
theorem den_batchOp_relu_eq_reluF {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.relu (n := n)) e) = den (.reluF e) := by
  rw [reluF_faithful]
  exact batchMap_pointwise (fun y => if y > 0 then y else 0) (den e)

/-- **Batched ReLU backward faithfulness.** `selectPosB` denotes the same proven
    `relu_has_vjp_at` backward as `selectPos`, now over the whole batch — which is what the
    emitted `xName` holds. This is the statement that would be FALSE had `selectPos` been made
    a `BatchableOp` descriptor (that `den` would apply one example's mask to all `N`). -/
theorem selectPosB_faithful {N n : Nat} (s : String) (x : Vec (N*n)) (hx : ∀ i, x i ≠ 0)
    (e : SHlo (N*n)) :
    den (.selectPosB s x e) = (relu_has_vjp_at (N*n) x hx).backward (den e) := rfl

/-- **ReLU6 forward faithfulness.** `min(max(·,0),6)` denotes the proven `relu6`
    (MobileNetV2.lean). (`rfl` — `relu6` is defined as exactly this clamp.) -/
@[simp] theorem relu6F_faithful {k : Nat} (e : SHlo k) :
    den (.relu6F e) = relu6 k (den e) := rfl

/-- **ReLU6 backward faithfulness (smooth point).** `select(0<x<6,·,0)` denotes the
    proven `relu6_has_vjp_at` backward — the two-sided kink's mask, smooth iff
    `x≠0 ∧ x≠6` (both bounds, unlike ReLU's one-sided `x≠0`). -/
theorem selectMid_faithful {k : Nat} (s : String) (x : Vec k)
    (h_smooth : ∀ i, x i ≠ 0 ∧ x i ≠ 6) (e : SHlo k) :
    den (.selectMid s x e) = (relu6_has_vjp_at k x h_smooth).backward (den e) := rfl

/-- **Batched ReLU6 forward faithfulness (§2f).** The `relu6` descriptor at the batched index
    denotes exactly `relu6F`'s per-example clamp applied across the batch — the MobileNetV2 peer
    of `den_batchOp_relu_eq_reluF`. This is the statement that keeps the emit width off the SHlo
    index: at `N := B` the descriptor emits `tensor<B×n>`, not `tensor<B×(N·n)>`. -/
theorem den_batchOp_relu6_eq_relu6F {N n : Nat} (e : SHlo (N * n)) :
    den (.batchOp (N := N) (.relu6 (n := n)) e) = den (.relu6F e) := by
  rw [relu6F_faithful]
  exact batchMap_pointwise (fun y => min (max y 0) 6) (den e)

/-- **Batched ReLU6 backward faithfulness.** `selectMidB` denotes the same proven
    `relu6_has_vjp_at` backward as `selectMid`, now over the whole batch — which is what the
    emitted `xName` holds. FALSE had `selectMid` been made a `BatchableOp` descriptor beside
    `relu6` (that `den` would apply one example's two-sided mask to all `N`). Note the smoothness
    hypothesis is TWO-sided (`x ≠ 0 ∧ x ≠ 6`), unlike `selectPosB_faithful`'s `x ≠ 0`. -/
theorem selectMidB_faithful {N n : Nat} (s : String) (x : Vec (N*n))
    (h_smooth : ∀ i, x i ≠ 0 ∧ x i ≠ 6) (e : SHlo (N*n)) :
    den (.selectMidB s x e) = (relu6_has_vjp_at (N*n) x h_smooth).backward (den e) := rfl

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
    den (.dropPathB mN s e) = (Proofs.dropPath_has_vjp N n s).backward x (den e) := rfl

@[simp] theorem den_dropPathB_ones {N n : Nat} (mN : String) (e : SHlo (N*n)) :
    den (.dropPathB mN (fun _ => 1) e) = den e := by
  simp [den_dropPathB]

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
    den (.dropoutB mN mask e) = (Proofs.dropout_has_vjp N n mask).backward x (den e) := rfl

/-- ⭐ **The ones-mask identity on the AST**, which is what licenses emitting the dropout site in
    the FORWARD artifact: `@efficientnet_do_fwd` and `@efficientnet_adamdo_train_step` are then one
    graph differing only in the mask the driver supplies, and the prefix audit survives. -/
@[simp] theorem den_dropoutB_ones {N n : Nat} (mN : String) (e : SHlo (N*n)) :
    den (.dropoutB mN (fun _ => 1) e) = den e := by
  simp [den_dropoutB]

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
  funext j; simp only [denseF, den, dense]

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
    the proven `mlp_has_vjp_at.backward` — the per-op `dot_general`/`select`
    ops assembled into the proven whole-network VJP (cf. `IR.mlp_whole_bridge`). -/
theorem mlpBackGraph_faithful (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu e₁ (dense W₀ b₀ x)) k ≠ 0) (dy : Vec e₃) :
    den (mlpBackGraph W₀ W₁ W₂ (dense W₀ b₀ x)
          (dense W₁ b₁ (relu e₁ (dense W₀ b₀ x))) dy)
      = (mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1).backward dy := by
  simp only [mlpBackGraph, den, mlp_has_vjp_at, vjp_comp_at, dense_has_vjp, relu_has_vjp_at,
             HasVJP.toHasVJPAt, Mat.mulVec, id_eq, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Chapter 3 — CNN: conv + maxpool (forward, semantic)
--
-- The conv/maxpool *forward* ops, denoted by the proofs' flattened forms
-- `flatConv`/`maxPoolFlat`. The whole MNIST-CNN forward graph denotes the
-- proven `mnistCnnNoBnForward`. (The backward VJP — conv input-grad via the
-- reversed kernel + maxpool select_and_scatter, = `mnistCnnNoBn_has_vjp_at` —
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
    `planning/rsb_a3_r50_verified.md` §4b. -/
theorem maxPool3s2F_faithful {c h w : Nat} (e : SHlo (c*(2*h)*(2*w))) :
    den (.maxPool3s2F e) = maxPool3s2Flat c h w (den e) := rfl

/-- **Conv backward faithfulness.** The reversed-kernel `stablehlo.convolution`
    (transpose+reverse+conv) denotes the proven conv input-VJP — the flattened
    `conv2d_has_vjp3` backward (conv is linear, so this is a global VJP). -/
theorem convBack_faithful {ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*h*w)) (e : SHlo (oc*h*w)) :
    den (.convBack wN W b v e)
      = (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward v (den e) := rfl

/-- **Max-pool backward faithfulness (smooth point).** The emitted
    `select_and_scatter` graph denotes the proven `maxPoolFlat_has_vjp_at`
    backward — routing the cotangent to each window's argmax (the codegen's
    no-ties convention), under the MaxPool smoothness hypothesis. -/
theorem maxPoolBack_faithful {c h w : Nat} (xN : String) (x : Vec (c*(2*h)*(2*w)))
    (h_smooth : MaxPool2Smooth (Tensor3.unflatten x : Tensor3 c (2*h) (2*w)))
    (e : SHlo (c*h*w)) :
    den (.maxPoolBack xN x e)
      = (maxPoolFlat_has_vjp_at (Tensor3.unflatten x) h_smooth).backward (den e) := by
  funext idx
  simp only [den, maxPoolBackFlat, maxPoolFlat_has_vjp_at, hasVJPAt3_to_hasVJPAt,
             maxPool2_has_vjp_at3]

/-- ⭐ **3×3/s2 max-pool backward faithfulness (smooth point).** The emitted `select_and_scatter`
    graph at window 3 / stride 2 / symmetric padding 1 denotes the proven
    `maxPool3s2Flat_has_vjp_at` backward, under `MaxPool3s2Smooth`.

    ⚠ The hypothesis is stated over **positions**, not window offsets, and that is not a stylistic
    difference from `maxPoolBack_faithful`: with overlapping windows two offsets can name one input
    cell (the clamped duplicate at the first window), where the values are equal by construction
    and smoothness must say nothing. `maxPool2` has no analogue because there distinct offsets
    always meant distinct positions. See `MaxPool3s2.lean`'s header. -/
theorem maxPool3s2Back_faithful {c h w : Nat} (xN : String) (x : Vec (c*(2*h)*(2*w)))
    (h_smooth : MaxPool3s2Smooth (Tensor3.unflatten x : Tensor3 c (2*h) (2*w)))
    (e : SHlo (c*h*w)) :
    den (.maxPool3s2Back xN x e)
      = (maxPool3s2Flat_has_vjp_at (Tensor3.unflatten x) h_smooth).backward (den e) := by
  funext idx
  simp only [den, maxPool3s2BackFlat, maxPool3s2Flat_has_vjp_at, hasVJPAt3_to_hasVJPAt,
             maxPool3s2_has_vjp_at3]

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
    + reversed-kernel conv denotes the proven `flatConvStride2_has_vjp` backward. -/
theorem convStridedBack_faithful {ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*(2*h)*(2*w))) (e : SHlo (oc*h*w)) :
    den (.convStridedBack wN W b v e) = (flatConvStride2_has_vjp W b).backward v (den e) := rfl

/-- **Stride-4 conv forward faithfulness.** The `window_strides=[4,4]`
    `stablehlo.convolution` (the ConvNeXt 4×4/s4 patchify stem) denotes the proven
    `flatConvStride4` (= decimate ∘ decimate ∘ stride-1 conv, StridedConv.lean). -/
@[simp] theorem flatConvStride4F_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*(2*(2*h))*(2*(2*w)))) :
    den (.flatConvStride4F wN bN W b e) = flatConvStride4 W b (den e) := rfl

/-- **BN backward faithfulness.** The consolidated three-term graph denotes the
    proven BN input-VJP — equal to the `pdiv`-contracted Jacobian of `bnForward`
    (`bn_input_grad_correct`), under `0 < ε`. β-independent (a constant shift
    does not enter the Jacobian). -/
theorem bnBack_faithful {n : Nat} (gN xN es : String) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (e : SHlo n) (i : Fin n) :
    den (.bnBack gN xN es ε γ x e) i
      = ∑ j : Fin n, pdiv (bnForward n ε γ β) x i j * den e j := by
  show bn_grad_input n ε γ x (den e) i = _
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

/-- **Conv weight-gradient faithfulness** — the proven `conv2d_weight_grad` VJP. -/
@[simp] theorem convWeightGrad_faithful {ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (e : SHlo (oc*h*w)) :
    den (.convWeightGrad xN b x W e)
      = (conv2d_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e) := rfl

/-- **Conv bias-gradient faithfulness** — the proven `conv2d_bias_grad` VJP. -/
@[simp] theorem convBiasGrad_faithful {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (e : SHlo (oc*h*w)) :
    den (.convBiasGrad W x b e) = (conv2d_bias_grad_has_vjp W x).backward b (den e) := rfl

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
`tests/TestBatchedEmitTie.lean` against the per-example fused `depthwiseBiasSgd`. -/

@[simp] theorem depthwiseBiasGradB_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (N*(c*h*w))) (b : Vec c)
    (e : SHlo (N*(c*h*w))) (o : Fin c) :
    den (.depthwiseBiasGradB W x b e) o
      = ∑ n : Fin N,
          (depthwise_bias_grad_has_vjp W (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward b
            (batchSlice N (c*h*w) (den e) n) o := rfl

@[simp] theorem depthwiseStridedBiasGradB_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (N*(c*(2*h)*(2*w)))) (b : Vec c)
    (e : SHlo (N*(c*h*w))) (o : Fin c) :
    den (.depthwiseStridedBiasGradB W x b e) o
      = ∑ n : Fin N,
          (depthwiseStride2_bias_grad_has_vjp W (batchSlice N (c*(2*h)*(2*w)) x n)).backward b
            (batchSlice N (c*h*w) (den e) n) o := rfl

/-! ## The ConvNeXt five — same statement, the last `*Sgd`/`*Grad` pairs the kit was missing (§2f)

`den (xSgd …) = θ − lr · den (xGrad …)`, all `rfl`. Together with the emit-side byte-PREFIX checks
in `tests/TestBatchedEmitTie.lean` this is what lets `convnext_adam_train_step` hand its gradients
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
    `flatConvStride4_weight_grad_has_vjp` backward. There is no fused `convStride4WeightSgd` peer —
    nothing but ConvNeXt's stem is stride-4, and its AdamW render consumes the un-fused gradient
    directly — so this, not a `*Sgd_eq_grad` statement, is what pins the op's `den`. -/
@[simp] theorem convStride4WeightGrad_faithful {ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (oc*h*w)) :
    den (.convStride4WeightGrad xN b x W e)
      = (flatConvStride4_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e) := rfl

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
-- § Global-norm gradient clipping — faithfulness (`GradClip.lean`, `planning/grad_clip.md`)
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

/-- **Per-channel BN backward faithfulness.** The block-diagonal three-term graph
    (per-channel, reducing over the spatial axes) denotes the proven per-channel BN
    input-VJP — equal to the `pdiv`-contracted (block-diagonal) Jacobian of
    `bnPerChannelTensor3` (`bnPerChannelTensor3_grad_input_correct`), under `0 < ε`. -/
theorem bnPerChannelBack_faithful {oc h w : Nat} (gN xN es : String) (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec oc) (x : Vec (oc*h*w)) (e : SHlo (oc*h*w)) (i : Fin (oc*h*w)) :
    den (.bnPerChannelBack gN xN es ε γ x e) i
      = ∑ j : Fin (oc*h*w), pdiv (bnPerChannelTensor3 oc h w ε γ β) x i j * den e j := by
  show bnPerChannelTensor3_grad_input oc h w ε γ x (den e) i = _
  exact bnPerChannelTensor3_grad_input_correct oc h w ε hε γ β x (den e) i

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
    `feature_group_count = c`) denotes the proven `depthwiseFlat_has_vjp` backward
    (depthwise is linear, so this is a global VJP). -/
theorem depthwiseBack_faithful {c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*h*w)) (e : SHlo (c*h*w)) :
    den (.depthwiseBack wN W b v e) = (depthwiseFlat_has_vjp W b).backward v (den e) := rfl

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
    `depthwiseStride2Flat_has_vjp` backward. -/
theorem depthwiseStridedBack_faithful {c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) (e : SHlo (c*h*w)) :
    den (.depthwiseStridedBack wN W b v e) = (depthwiseStride2Flat_has_vjp W b).backward v (den e) := rfl

/-- **Swish forward faithfulness.** The `multiply(x, logistic(x))` graph denotes
    the proven `swish` (= `x · σ(x)`, LayerNorm.lean). Smooth everywhere; no kink,
    no smoothness hypothesis. (`rfl`, so kept out of the axiom audit — `roundtrip`
    covers it structurally.) -/
@[simp] theorem swishF_faithful {n : Nat} (e : SHlo n) :
    den (.swishF e) = swish n (den e) := rfl

/-- **Swish input-VJP faithfulness.** The closed-form `dy ⊙ σ(x)·(1 + x·(1−σ(x)))`
    graph (recomputing σ from the saved pre-activation `x`) denotes the proven GLOBAL
    `swish_has_vjp` backward (`dy ⊙ swishScalarDeriv x`; swish is smooth everywhere, so
    this is a global VJP — no smoothness hypothesis). -/
theorem swishBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.swishBack xN x e) = (swish_has_vjp n).backward x (den e) := rfl

/-- **Sigmoid forward faithfulness.** The `stablehlo.logistic(x)` graph denotes the
    proven `sigmoid` (= σ(x), EfficientNet.lean) — the SE gate's output nonlinearity.
    Smooth everywhere. (`rfl`, so kept out of the axiom audit — `roundtrip` covers it.) -/
@[simp] theorem sigmoidF_faithful {n : Nat} (e : SHlo n) :
    den (.sigmoidF e) = sigmoid n (den e) := rfl

/-- **Sigmoid input-VJP faithfulness.** The closed-form `dy ⊙ σ(x)·(1−σ(x))` graph
    (recomputing σ from the saved pre-activation `x`) denotes the proven GLOBAL
    `sigmoid_has_vjp` backward (`dy ⊙ sigmoidScalarDeriv x`; sigmoid is smooth
    everywhere, so this is a global VJP — no smoothness hypothesis). -/
theorem sigmoidBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.sigmoidBack xN x e) = (sigmoid_has_vjp n).backward x (den e) := rfl

/-- **GELU forward faithfulness.** The tanh-approximation graph
    `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` denotes the proven `gelu`
    (LayerNorm.lean). Smooth everywhere; no kink, no smoothness hypothesis.
    (`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem geluF_faithful {n : Nat} (e : SHlo n) :
    den (.geluF e) = gelu n (den e) := rfl

/-- **Layer-scale faithfulness.** The per-element multiply `γ ⊙ x` denotes the proven
    `layerScale` (ConvNeXt.lean). (`rfl`.) -/
@[simp] theorem layerScaleF_faithful {n : Nat} (γN : String) (γ : Vec n) (e : SHlo n) :
    den (.layerScaleF γN γ e) = layerScale γ (den e) := rfl

/-- **Per-channel layer-scale faithfulness.** The `[c]`-broadcast multiply denotes
    the proven `layerScale` at the channel-expanded vector. (`rfl`.) -/
@[simp] theorem layerScaleChF_faithful {c h w : Nat} (γN : String) (γ : Vec c)
    (e : SHlo (c*h*w)) :
    den (.layerScaleChF γN γ e) = layerScale (fun k => γ (chanIdx c h w k)) (den e) := rfl

/-- **GELU input-VJP faithfulness.** The closed-form `dy ⊙ gelu'(x)` graph
    (recomputing `tanh(u(x))` from the saved pre-activation `x`) denotes the proven
    GLOBAL `gelu_has_vjp` backward (`dy ⊙ geluScalarDeriv x`; GELU is smooth
    everywhere, so this is a global VJP — no smoothness hypothesis). -/
theorem geluBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.geluBack xN x e) = (gelu_has_vjp n).backward x (den e) := rfl

/-- **Row-softmax forward faithfulness.** The per-row `exp / reduce[last] / divide`
    graph denotes `rowSoftmaxFlat` (= flattened `rowSoftmax`, Attention.lean). Plain
    exp/sum, no max-shift (matches the proven `softmax`). Smooth everywhere.
    (`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem softmaxRowF_faithful {m n : Nat} (e : SHlo (m*n)) :
    den (.softmaxRowF e) = rowSoftmaxFlat m n (den e) := rfl

/-- **Row-softmax input-VJP faithfulness.** The per-row closed-form
    `p ⊙ (dy − ⟨p,dy⟩)` graph (recomputing `p` from the saved pre-softmax scores)
    denotes `rowSoftmaxBackFlat` (= flattened `rowSoftmax_has_vjp_mat.backward`).
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
    axis) denotes `rowLNBackFlat` (rowwise `bn_grad_input` — faithful to the
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
    = the proven `dense_has_vjp` backward; dense is affine — global VJP). -/
theorem denseRowBack_faithful {N a c : Nat} (wN : String) (W : Mat a c) (e : SHlo (N*c)) :
    den (.denseRowBack wN W e) = rowDenseBackFlat N a c W (den e) := rfl

/-- **Patch-embedding faithfulness.** The stride-P VALID conv + channels-last
    flatten + CLS concatenate + position-embed add graph denotes `patchEmbedFlat`
    (the local re-spelling of the proven `patchEmbed_flat`; the tie is `rfl` in
    ViTFwdGraph). (`rfl`, coarse-token like `seBlock`.) -/
@[simp] theorem patchEmbedF_faithful {ic H W P N D : Nat} (wN bN cN pN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N+1) D) (e : SHlo (ic*H*W)) :
    den (.patchEmbedF wN bN cN pN Wc bc cls pos e)
      = patchEmbedFlat ic H W P N D Wc bc cls pos (den e) := rfl

/-- **Patch-embedding input-VJP faithfulness.** The reversed-kernel strided
    `conv_transpose` (on the patch-token rows of the `[N+1,D]` cotangent) denotes
    `patchEmbedBackFlat` (= the proven `patchEmbed_input_grad_formula`; the tie to
    `patchEmbed_flat_has_vjp.backward` is `rfl` in ViTBackB0). (`rfl`.) -/
@[simp] theorem patchEmbedBack_faithful {ic H W P N D : Nat} (wN : String)
    (Wc : Kernel4 D ic P P) (e : SHlo ((N+1)*D)) :
    den (.patchEmbedBack wN Wc e) = patchEmbedBackFlat ic H W P N D Wc (den e) := rfl

/-- **CLS-slice faithfulness.** The row-0 `stablehlo.slice` denotes `clsSliceFlat`
    (= the proven `cls_slice_flat`). (`rfl`.) -/
@[simp] theorem clsSliceF_faithful {N D : Nat} (e : SHlo ((N+1)*D)) :
    den (.clsSliceF e) = clsSliceFlat N D (den e) := rfl

/-- **CLS-pad faithfulness.** The zero-pad scatter-to-row-0 denotes `clsPadFlat`
    (= the proven `cls_slice_flat_has_vjp.backward`; linear — global VJP). (`rfl`.) -/
@[simp] theorem clsPadF_faithful {N D : Nat} (e : SHlo D) :
    den (.clsPadF (N := N) e) = clsPadFlat N D (den e) := rfl

/-- **Per-head slice faithfulness.** The feature-axis `stablehlo.slice` of head `h`'s
    contiguous column block denotes `headSliceFlat` (= `mhsa_layer`'s per-head column
    gather). Linear reindex. (`rfl`.) -/
@[simp] theorem headSliceF_faithful {N heads d : Nat} (h : Fin heads)
    (e : SHlo (N*(heads*d))) :
    den (.headSliceF h e) = headSliceFlat N heads d h (den e) := rfl

/-- **Per-head pad faithfulness.** The feature-axis zero-pad into head `h`'s column
    block denotes `headPadFlat` (the slice's VJP; summed over heads it is
    `mhsa_layer`'s concat). Linear. (`rfl`.) -/
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

/-- **CNN forward faithfulness.** The forward graph denotes the proven
    `mnistCnnNoBnForward`. -/
theorem cnnFwdGraph_faithful {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) :
    den (cnnFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)
      = mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x := by
  simp only [cnnFwdGraph, mnistCnnNoBnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

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

/-- **CIFAR-CNN forward faithfulness.** The forward graph denotes the proven
    `cifarCnnForward`. -/
theorem cifarFwdGraph_faithful {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) :
    den (cifarFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)
      = cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x := by
  simp only [cifarFwdGraph, cifarCnnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- Whole **BN-CIFAR forward** graph (Chapter 4, BatchNorm variant): each conv is
    followed by a per-example `bnF` before its ReLU. `epsStr` is the shared ε
    literal; the four BN layers carry scalar γ/β inputs `%g{i}`/`%bt{i}`. -/
def cifarBnFwdGraph {ic c1 c2 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : SHlo nClasses :=
  denseF "%W7" "%b7" W₇ b₇
    (.reluF (denseF "%W6" "%b6" W₆ b₆
      (.reluF (denseF "%W5" "%b5" W₅ b₅
        (.maxPoolF (c := c2) (h := h) (w := w)
          (.reluF (.bnPerChannelF (oc := c2) (h := 2*h) (w := 2*w) "%g4" "%bt4" epsStr ε₄ γ₄ β₄
            (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄
            (.reluF (.bnPerChannelF (oc := c2) (h := 2*h) (w := 2*w) "%g3" "%bt3" epsStr ε₃ γ₃ β₃
              (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃
              (.maxPoolF (c := c1) (h := 2*h) (w := 2*w)
                (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g2" "%bt2" epsStr ε₂ γ₂ β₂
                  (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂
                  (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g1" "%bt1" epsStr ε₁ γ₁ β₁
                    (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁
                    (.operand "%x" x)))))))))))))))))))

/-- **BN-CIFAR forward faithfulness.** The forward graph denotes the proven
    `cifarCnnBnForward`. -/
theorem cifarBnFwdGraph_faithful {ic c1 c2 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) :
    den (cifarBnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇ x)
      = cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇ x := by
  simp only [cifarBnFwdGraph, cifarCnnBnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful,
             bnPerChannelF_faithful, den_operand]

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

/-- **Deeper (8-conv) CIFAR-CNN forward faithfulness.** The forward graph denotes the
    proven `cifarCnn8Forward`. -/
theorem cifar8FwdGraph_faithful {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) :
    den (cifar8FwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
          W₉ b₉ Wa ba Wb bb x)
      = cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
          W₉ b₉ Wa ba Wb bb x := by
  simp only [cifar8FwdGraph, cifarCnn8Forward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- Whole **deeper (8-conv) BN-CIFAR forward** graph: each of the eight convs is followed
    by a per-channel `bnPerChannelF` before its ReLU. `epsStr` is the shared ε literal; the
    eight BN layers carry per-channel γ/β inputs `%g{i}`/`%bt{i}`. The 4-stage peer of
    `cifarBnFwdGraph`. -/
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

/-- **Deeper (8-conv) BN-CIFAR forward faithfulness.** The forward graph denotes the
    proven `cifarCnnBn8Forward`. -/
theorem cifar8BnFwdGraph_faithful {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (epsStr : String)
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
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) :
    den (cifar8BnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          W₉ b₉ Wa ba Wb bb x)
      = cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          W₉ b₉ Wa ba Wb bb x := by
  simp only [cifar8BnFwdGraph, cifarCnnBn8Forward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful,
             bnPerChannelF_faithful, den_operand]

/-- Whole **ResNet-style forward** graph (Chapter 5): the structure the proven
    whole-net VJP `cnn_has_vjp_at` already covers —
    `dense ∘ GAP ∘ rblkP ∘ rblk ∘ maxPool ∘ cbr(stem)`. The stem is `convBnRelu`
    (SAME conv on the `2h×2w` input), one maxpool to `h×w`, an identity basic
    block (`rblk`: `relu(F(y)+y)`), a projection basic block (`rblkP`:
    `relu(proj(y)+F(y))`, `c→oc`), global-average-pool, then dense. Each block's
    skip reuses the block-input **subtree** in BOTH `addV` operands, so the graph
    stays a tree (the §7 "tree-safe via operand leaves" trick, generalized to a
    computed input). `epsStr` is the shared ε literal; each BN carries scalar γ/β
    SSA inputs (`%g*`/`%bt*`). The Chapter-5 peer of `cifarBnFwdGraph`. -/
def resnetFwdGraph
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip : ℝ)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (x : Vec (ic*(2*h)*(2*w))) : SHlo nClasses :=
  -- stem (convBnRelu on the 2h×2w input) → maxpool to h×w
  let pooled : SHlo (c*h*w) :=
    .maxPoolF (c := c) (h := h) (w := w)
      (.reluF (.bnF "%gs" "%bts" epsStr εs γs βs
        (.flatConvF (h := 2*h) (w := 2*w) "%Ws" "%bs" Ws bs (.operand "%x" x))))
  -- identity basic block: relu(F(pooled) + pooled),  F = bn∘conv ∘ relu∘bn∘conv
  let rblkOut : SHlo (c*h*w) :=
    .reluF (.addV
      (.bnF "%g2" "%bt2" epsStr f₂ h₂ i₂
        (.flatConvF (h := h) (w := w) "%W2" "%b2" W₂ b₂
          (.reluF (.bnF "%g1" "%bt1" epsStr f₁ h₁ i₁
            (.flatConvF (h := h) (w := w) "%W1" "%b1" W₁ b₁ pooled)))))
      pooled)
  -- projection basic block: relu(proj(rblkOut) + F'(rblkOut)),  c→oc
  let rblkPOut : SHlo (oc*h*w) :=
    .reluF (.addV
      (.bnF "%gp" "%btp" epsStr fp hp ip
        (.flatConvF (h := h) (w := w) "%Wp" "%bp" Wp bp rblkOut))
      (.bnF "%g2p" "%bt2p" epsStr e₂ g₂ bb₂
        (.flatConvF (h := h) (w := w) "%W2p" "%b2p" W₂' b₂'
          (.reluF (.bnF "%g1p" "%bt1p" epsStr e₁ g₁ bb₁
            (.flatConvF (h := h) (w := w) "%W1p" "%b1p" W₁' b₁' rblkOut))))))
  denseF "%Wd" "%bd" Wd bd (.gapF (c := oc) (h := h) (w := w) rblkPOut)

/-- **ResNet-style forward faithfulness.** The forward graph denotes the proven
    `cnnForward` — the net whose whole-network VJP is `cnn_has_vjp_at` (discharged
    unconditionally by `CnnConcrete.cnnConcrete_has_vjp_correct`). The residual
    `addV`s denote the `+` of `residual`/`residualProj` (`biPath`); each skip's
    duplicated subtree denotes the same block-input value, so `den` reads it
    twice and the fan-in is exact. -/
theorem resnetFwdGraph_faithful
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip : ℝ)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (x : Vec (ic*(2*h)*(2*w))) :
    den (resnetFwdGraph epsStr Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
          W₁' b₁' W₂' b₂' Wp bp f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip Wd bd x)
      = cnnForward Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
          W₁' b₁' W₂' b₂' Wp bp f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip Wd bd x := by
  -- LHS: collapse the graph denotation to its explicit nested form.
  simp only [resnetFwdGraph, denseF_faithful, gapF_faithful, reluF_faithful,
             bnF_faithful, flatConvF_faithful, maxPoolF_faithful, den_addV, den_operand]
  -- RHS: unfold the abbreviations (incl. `biPath`, which `simp` can't unfold below
  -- its arity), then peel the `∘`s. Both sides land on the same `+`-nested form.
  unfold cnnForward cbr rblk rblkP residual residualProj biPath
  simp only [Function.comp_apply]

/-- Whole **MobileNetV2 forward** graph (representative, ch7 peer of `resnetFwdGraph`):
    stem (conv→bn→relu6) → skip inverted-residual `addV(invresBody, stem)` → no-skip
    inverted-residual → global-average-pool → dense. Each inverted-residual body is
    `bn∘conv(project) ∘ relu6∘bn∘depthwise ∘ relu6∘bn∘conv(expand)`; the skip's `addV`
    reuses the block-input subtree (linear bottleneck — no relu6 after the add). Uses the
    MobileNetV2 ops `relu6F`/`depthwiseF` (SAME-spatial representative; the stride-2
    `depthwiseStridedF`/`flatConvStridedF` of the full render are exercised at the op level,
    not assembled here — full strided graph deferred, see planning doc). `epsStr` = shared ε
    literal; each scalar BN carries γ/β SSA inputs `%g*`/`%bt*`. -/
def mobilenetv2FwdGraph
    {ic c mid₁ oc mid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic*h*w)) : SHlo nClasses :=
  -- stem: relu6(bn(conv x))
  let stemOut : SHlo (c*h*w) :=
    .relu6F (.bnF "%gs" "%bts" epsStr εs γs βs
      (.flatConvF (h := h) (w := w) "%Ws" "%bs" Ws bs (.operand "%x" x)))
  -- block1 body (inverted residual, c→mid₁→c): project ∘ depthwise ∘ expand
  let b1Body : SHlo (c*h*w) :=
    .bnF "%gp1" "%btp1" epsStr p₁ gp₁ bp1
      (.flatConvF (h := h) (w := w) "%Wp1" "%bp1" Wp₁ bp₁
        (.relu6F (.bnF "%gd1" "%btd1" epsStr d₁ gd₁ bd1
          (.depthwiseF (h := h) (w := w) "%Wd1" "%bd1" Wd₁ bd₁
            (.relu6F (.bnF "%ge1" "%bte1" epsStr e₁ ge₁ be1
              (.flatConvF (h := h) (w := w) "%We1" "%be1" We₁ be₁ stemOut)))))))
  -- block1 (skip): linear-bottleneck residual, no relu6 after the add
  let b1Out : SHlo (c*h*w) := .addV b1Body stemOut
  -- block2 body (inverted residual, c→mid₂→oc, no skip)
  let b2Out : SHlo (oc*h*w) :=
    .bnF "%gp2" "%btp2" epsStr p₂ gp₂ bp2
      (.flatConvF (h := h) (w := w) "%Wp2" "%bp2" Wp₂ bp₂
        (.relu6F (.bnF "%gd2" "%btd2" epsStr d₂ gd₂ bd2
          (.depthwiseF (h := h) (w := w) "%Wd2" "%bd2" Wd₂ bd₂
            (.relu6F (.bnF "%ge2" "%bte2" epsStr e₂ ge₂ be2
              (.flatConvF (h := h) (w := w) "%We2" "%be2" We₂ be₂ b1Out)))))))
  denseF "%Wh" "%bh" Wh bh (.gapF (c := oc) (h := h) (w := w) b2Out)

/-- **MobileNetV2 forward faithfulness.** The representative forward graph denotes the
    proven `mobilenetv2Forward` (whose end-to-end VJP at a smooth point is
    `mobilenetv2_has_vjp_at`). The skip `addV` denotes the `+` of `residual`/`biPath`;
    the inverted-residual body's `bn/conv/depthwise/relu6` ops denote
    `invresBody = ivProject ∘ ivDepthwise ∘ ivExpand`. ch7 peer of `resnetFwdGraph_faithful`. -/
theorem mobilenetv2FwdGraph_faithful
    {ic c mid₁ oc mid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic*h*w)) :
    den (mobilenetv2FwdGraph epsStr Ws bs εs γs βs
          We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
          We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh x)
      = mobilenetv2Forward Ws bs εs γs βs
          We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
          We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh x := by
  simp only [mobilenetv2FwdGraph, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnF_faithful, flatConvF_faithful, depthwiseF_faithful, den_addV, den_operand]
  unfold mobilenetv2Forward invresBody ivExpand ivDepthwise ivProject residual biPath
  simp only [Function.comp_apply]

/-- Whole **MobileNetV2 forward** graph at the FULL ch7 render dims (3×224² → 7×7×64):
    strided stem (`flatConvStridedF`, 224→112) → 6 inverted-residual blocks (`b1/b3/b5/b6`
    stride-2 downsample via `depthwiseStridedF`, `b2/b4` stride-1 SAME with an `addV` skip)
    → 1×1 conv-bn-relu6 head → global-avg-pool → dense. Concrete (not symbolic) peer of
    `mobilenetv2FwdGraph`, tied to the *full* forward `mobilenetv2Forward_full`. Scalar BN. -/
def mobilenetv2FwdGraphFull
    (epsStr : String)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  let stemOut : SHlo (16 * 112 * 112) :=
    .relu6F (.bnF "%gs" "%bts" epsStr εs γs βs
      (.flatConvStridedF (h := 112) (w := 112) "%Ws" "%bs" Ws bs (.operand "%x" x)))
  let b1Out : SHlo (24 * 56 * 56) :=
    .bnF "%gp1" "%btp1" epsStr εp1 γp1 βp1
      (.flatConvF (h := 56) (w := 56) "%Wp1" "%bp1" Wp1 bp1
        (.relu6F (.bnF "%gd1" "%btd1" epsStr εd1 γd1 βd1
          (.depthwiseStridedF (h := 56) (w := 56) "%Wd1" "%bd1" Wd1 bd1
            (.relu6F (.bnF "%ge1" "%bte1" epsStr εe1 γe1 βe1
              (.flatConvF (h := 112) (w := 112) "%We1" "%be1" We1 be1 stemOut)))))))
  let b2Out : SHlo (24 * 56 * 56) :=
    .addV (.bnF "%gp2" "%btp2" epsStr εp2 γp2 βp2
      (.flatConvF (h := 56) (w := 56) "%Wp2" "%bp2" Wp2 bp2
        (.relu6F (.bnF "%gd2" "%btd2" epsStr εd2 γd2 βd2
          (.depthwiseF (h := 56) (w := 56) "%Wd2" "%bd2" Wd2 bd2
            (.relu6F (.bnF "%ge2" "%bte2" epsStr εe2 γe2 βe2
              (.flatConvF (h := 56) (w := 56) "%We2" "%be2" We2 be2 b1Out)))))))) b1Out
  let b3Out : SHlo (32 * 28 * 28) :=
    .bnF "%gp3" "%btp3" epsStr εp3 γp3 βp3
      (.flatConvF (h := 28) (w := 28) "%Wp3" "%bp3" Wp3 bp3
        (.relu6F (.bnF "%gd3" "%btd3" epsStr εd3 γd3 βd3
          (.depthwiseStridedF (h := 28) (w := 28) "%Wd3" "%bd3" Wd3 bd3
            (.relu6F (.bnF "%ge3" "%bte3" epsStr εe3 γe3 βe3
              (.flatConvF (h := 56) (w := 56) "%We3" "%be3" We3 be3 b2Out)))))))
  let b4Out : SHlo (32 * 28 * 28) :=
    .addV (.bnF "%gp4" "%btp4" epsStr εp4 γp4 βp4
      (.flatConvF (h := 28) (w := 28) "%Wp4" "%bp4" Wp4 bp4
        (.relu6F (.bnF "%gd4" "%btd4" epsStr εd4 γd4 βd4
          (.depthwiseF (h := 28) (w := 28) "%Wd4" "%bd4" Wd4 bd4
            (.relu6F (.bnF "%ge4" "%bte4" epsStr εe4 γe4 βe4
              (.flatConvF (h := 28) (w := 28) "%We4" "%be4" We4 be4 b3Out)))))))) b3Out
  let b5Out : SHlo (64 * 14 * 14) :=
    .bnF "%gp5" "%btp5" epsStr εp5 γp5 βp5
      (.flatConvF (h := 14) (w := 14) "%Wp5" "%bp5" Wp5 bp5
        (.relu6F (.bnF "%gd5" "%btd5" epsStr εd5 γd5 βd5
          (.depthwiseStridedF (h := 14) (w := 14) "%Wd5" "%bd5" Wd5 bd5
            (.relu6F (.bnF "%ge5" "%bte5" epsStr εe5 γe5 βe5
              (.flatConvF (h := 28) (w := 28) "%We5" "%be5" We5 be5 b4Out)))))))
  let b6Out : SHlo (64 * 7 * 7) :=
    .bnF "%gp6" "%btp6" epsStr εp6 γp6 βp6
      (.flatConvF (h := 7) (w := 7) "%Wp6" "%bp6" Wp6 bp6
        (.relu6F (.bnF "%gd6" "%btd6" epsStr εd6 γd6 βd6
          (.depthwiseStridedF (h := 7) (w := 7) "%Wd6" "%bd6" Wd6 bd6
            (.relu6F (.bnF "%ge6" "%bte6" epsStr εe6 γe6 βe6
              (.flatConvF (h := 14) (w := 14) "%We6" "%be6" We6 be6 b5Out)))))))
  let headOut : SHlo (128 * 7 * 7) :=
    .relu6F (.bnF "%gh" "%bth" epsStr εh γh βh
      (.flatConvF (h := 7) (w := 7) "%Wh" "%bh" Wh bh b6Out))
  denseF "%Wfc" "%bfc" Wfc bfc (.gapF (c := 128) (h := 7) (w := 7) headOut)

/-- **Full MobileNetV2 forward faithfulness.** The full strided render graph denotes the
    proven `mobilenetv2Forward_full` (the representative 6-block net, tied by
    `mobilenetv2Rep_denote_eq` in `SpecVJP.lean` — the committed 17-block spec's tie is
    `mobilenetv2Verified_denote_eq` there, against `mobilenetv2ForwardPaper`). `simp`-based
    — so unlike the VJP fold it does not hit the concrete-dim `isDefEq` wall. -/
theorem mobilenetv2FwdGraphFull_faithful
    (epsStr : String)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphFull epsStr Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x)
      = mobilenetv2Forward_full Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x := by
  simp only [mobilenetv2FwdGraphFull, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnF_faithful, flatConvF_faithful, flatConvStridedF_faithful, depthwiseF_faithful,
             depthwiseStridedF_faithful, den_addV, den_operand]
  unfold mobilenetv2Forward_full invresBodyStrided invresBody ivExpand ivDepthwiseStrided
         ivDepthwise ivProject residual biPath
  simp only [Function.comp_apply]


/-- Whole **ConvNeXt forward** graph (representative, ch9 peer of `resnetFwdGraph`): 1×1
    patchify conv → stem-LN → 2 residual ConvNeXt blocks (depthwise → LN → 1×1 expand →
    GELU → 1×1 project → layerScale, then `addV` skip) → GAP → head-LN → dense. Scalar LN
    (`= bnForward`, via `bnF`); uses `geluF` + the new `layerScaleF`. Denotes the proven
    `convNextForward`. -/
def convNextFwdGraph {ic c cExp h w kH kW nClasses : Nat}
    (epsStr : String)
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) : SHlo nClasses :=
  let patchOut : SHlo (c * h * w) :=
    .flatConvF (h := h) (w := w) "%Wst" "%bst" Wst bst (.operand "%x" x)
  let stemLn : SHlo (c * h * w) :=
    .bnF "%gst" "%btst" epsStr εst γst βst patchOut
  let b1Body : SHlo (c * h * w) :=
    .layerScaleF "%gls1" γls₁
      (.flatConvF (h := h) (w := w) "%Wpr1" "%bpr1" Wpr₁ bpr₁
        (.geluF (.flatConvF (h := h) (w := w) "%Wex1" "%bex1" Wex₁ bex₁
          (.bnF "%gn1" "%btn1" epsStr εn₁ γn₁ βn₁
            (.depthwiseF (h := h) (w := w) "%Wdw1" "%bdw1" Wdw₁ bdw₁ stemLn)))))
  let b1Out : SHlo (c * h * w) := .addV b1Body stemLn
  let b2Body : SHlo (c * h * w) :=
    .layerScaleF "%gls2" γls₂
      (.flatConvF (h := h) (w := w) "%Wpr2" "%bpr2" Wpr₂ bpr₂
        (.geluF (.flatConvF (h := h) (w := w) "%Wex2" "%bex2" Wex₂ bex₂
          (.bnF "%gn2" "%btn2" epsStr εn₂ γn₂ βn₂
            (.depthwiseF (h := h) (w := w) "%Wdw2" "%bdw2" Wdw₂ bdw₂ b1Out)))))
  let b2Out : SHlo (c * h * w) := .addV b2Body b1Out
  let headLn : SHlo c :=
    .bnF "%ghd" "%bthd" epsStr εhd γhd βhd (.gapF (c := c) (h := h) (w := w) b2Out)
  denseF "%Wd" "%bd" Wd bd headLn

/-- **ConvNeXt forward faithfulness.** The representative forward graph denotes the proven
    `convNextForward`. Scalar LN (`layerNormForward = bnForward`); `simp`-based. -/
theorem convNextFwdGraph_faithful {ic c cExp h w kH kW nClasses : Nat}
    (epsStr : String)
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) :
    den (convNextFwdGraph epsStr Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x) = convNextForward Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x := by
  simp only [convNextFwdGraph, denseF_faithful, gapF_faithful, geluF_faithful, bnF_faithful,
             flatConvF_faithful, depthwiseF_faithful, layerScaleF_faithful, den_addV, den_operand]
  unfold convNextForward convNextBlock convNextBlockBody residual biPath layerNormForward
  simp only [Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § Chapter 3 — CNN: whole-chain backward (A2c, the MLP-analog of
--   `mlpBackGraph_faithful`). The full backward graph denotes the proven
--   conditional whole-network VJP `mnistCnnNoBn_has_vjp_at.backward`.
-- ════════════════════════════════════════════════════════════════

/-- Pointwise-VJP backwards are unique: `.correct` pins `backward` to the
    `pdiv`-contracted Jacobian, so any two `HasVJPAt f x` agree on `backward`.
    Lets us swap the maxpool's `flatten∘unflatten` transport (built into
    `mnistCnnNoBn_has_vjp_at`) for the cast-free witness below. -/
theorem hasVJPAt_backward_det {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (v v' : HasVJPAt f x) (dy : Vec n) : v.backward dy = v'.backward dy := by
  funext i; rw [v.correct, v'.correct]

/-- Max-pool VJP at a *raw* flattened point (no `flatten ∘ unflatten` index), so
    it composes without a transport cast; `backward` is `maxPoolBackFlat`. The
    `correct` field reuses `maxPoolFlat_has_vjp_at.correct`, aligning the point
    via `Tensor3.flatten_unflatten`. -/
noncomputable def maxPoolFlat_has_vjp_at' {c h w : Nat} (v : Vec (c*(2*h)*(2*w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    HasVJPAt (maxPoolFlat c h w) v where
  backward := maxPoolBackFlat c h w v
  correct := fun dy i => by
    have hbk : maxPoolBackFlat c h w v dy i
                = (maxPoolFlat_has_vjp_at (Tensor3.unflatten v) hs).backward dy i := by
      simp only [maxPoolFlat_has_vjp_at, hasVJPAt3_to_hasVJPAt, maxPool2_has_vjp_at3, maxPoolBackFlat]
    rw [hbk, (maxPoolFlat_has_vjp_at (Tensor3.unflatten v) hs).correct dy i,
        Tensor3.flatten_unflatten]

@[simp] theorem maxPoolFlat_has_vjp_at'_backward {c h w : Nat} (v : Vec (c*(2*h)*(2*w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    (maxPoolFlat_has_vjp_at' v hs).backward = maxPoolBackFlat c h w v := rfl

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

-- **CNN backward faithfulness (smooth point) — A2c.** The whole-chain backward
-- graph denotes the proven conditional whole-network VJP
-- `mnistCnnNoBn_has_vjp_at.backward` (the Chapter-3 peer of
-- `mlpBackGraph_faithful`). The per-op `convBack`/`selectPos`/`dotOut` ops
-- assemble through `vjp_comp_at`; the one `maxPoolBack` matches via VJP
-- uniqueness (`hasVJPAt_backward_det`) — sidestepping the `flatten∘unflatten`
-- transport in `mnistCnnNoBn_has_vjp_at`'s maxpool step.
set_option maxHeartbeats 2000000 in
theorem cnnBackGraph_faithful
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h1 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₁ b₁ x k ≠ 0)
    (h2 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₂ b₂
            ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)
            : Tensor3 c (2*h) (2*w)))
    (h3 : ∀ k, dense W₃ b₃ (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)) k ≠ 0)
    (h4 : ∀ k, dense W₄ b₄ ((relu d1 ∘ dense W₃ b₃) (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x))) k ≠ 0)
    (dy : Vec nClasses) :
    den (cnnBackGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ x dy)
      = (mnistCnnNoBn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
          hc hh hw x h1 h2 h_mp h3 h4).backward dy := by
  simp only [cnnBackGraph, den, mnistCnnNoBn_has_vjp_at, convRelu_has_vjp_at,
    denseRelu_has_vjp_at, vjp_comp_at, dense_has_vjp, relu_has_vjp_at,
    hasVJP3_to_hasVJP, HasVJP.toHasVJPAt, Mat.mulVec, id_eq, Function.comp_apply]
  rw [hasVJPAt_backward_det _ (maxPoolFlat_has_vjp_at'
        ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
          ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x)) h_mp)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Syntactic half: `pretty` renders the AST to real StableHLO text
-- ════════════════════════════════════════════════════════════════

/-- Tensor-type string `tensor<d₀x…xf32>`. -/
def ty (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["f32"]) ++ ">"

/-- Boolean (i1) tensor-type string, for `compare`/`select` masks. -/
def tyI1 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["i1"]) ++ ">"

/-- bf16 tensor-type string, for the `convertF` round node (planning/bf16_renderer.md).
    Only the round trip uses it today; when a bf16-operand `dot_general` lands (rung 2+)
    this is the type its operands carry. -/
def tyBf16 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["bf16"]) ++ ">"

/-- Fresh SSA name `%v{k}`. -/
def fresh : StateM Nat String := do
  let k ← get; set (k + 1); pure s!"%v{k}"

/-- **The 3×3/s2 pool's emitted forward text**, given already-freshened names.

    ⚠⚠ It is a shared helper rather than two copies for the reason `sWGradGeom` is (§2f-bis): the
    per-example `.maxPool3s2F` and the batched `BatchableOp.maxPool3s2` are two `emitTok` arms
    emitting **one** program, and a window or padding that drifted between them would be a pair of
    renders that agree on every structural check and compute different functions — which is the
    exact failure this whole op exists to fix. With one writer they cannot drift, and
    `TestBatchedEmitTie` then measures rather than assumes it.

    `window_dimensions = 3, window_strides = 2, padding = [[1,1],[1,1]]` on the spatial axes: He
    et al./torchvision `MaxPool2d(3, stride=2, padding=1)`, window `i` = input `[2i−1, 2i+1]`.
    ⚠ NOT XLA `'SAME'`, which pads `(0,1)` and slides the grid one input position — the two are
    different functions everywhere. -/
def maxPool3s2FwdText (B c h w : Nat) (r xn ninf p o : String) : String :=
  s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
  s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  s!"    {p} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
  "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
  "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
  "        stablehlo.return %pm : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, " ++
  "padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>}" ++
  s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
  s!"    {o} = stablehlo.reshape {p} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n"

/-- **The 3×3/s2 pool's emitted backward text**, given already-freshened names. Shared by the
    per-example and batched arms, for `maxPool3s2FwdText`'s reason.

    ⭐ Only the window attributes differ from `maxPoolBack`'s emit — **nothing else** — because
    `select_and_scatter`'s scatter region already reduces with `add`, which is exactly the
    accumulation overlapping windows need. The emitter was general enough before the op existed.

    ⚠ `%sa`/`%sb`/`%sc`/`%sd` are hardcoded region block arguments and are therefore RESERVED SSA
    names (§4): a top-level value of the same name is a redefinition error that surfaces only at
    XLA compile time. -/
def maxPool3s2BackText (B c h w : Nat) (xN r xr dr z scn o : String) : String :=
  s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
  s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
  s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
  "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
  "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %sge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
  "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
  "        stablehlo.return %ss : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, " ++
  "padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>}" ++
  s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
  s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n"

/-- **The stochastic-depth mask input name for ramp index `i`** — the `mName` a `dropPathB` carries,
    and the `tensor<Bxf32>` the signature declares for it.

    ⚠ It lives HERE, beside the emitter, rather than in one net's renderer, because the spelling is
    load-bearing in three places that must agree and only one of them is Lean: `dropPathP`'s emit
    reads it as an operand, every SD render's signature declares it, and
    **`scripts/misplace_drop_sites.py` matches `%dp\d+` textually** to build the placement control.
    A second definition would be the double-writer disease with a committed shell script as the
    third writer. (It started in `EfficientNetRender.lean` and moved when ConvNeXt needed it too;
    both renderers are in this namespace, so no call site changed and no artifact byte moved.) -/
def dpName (i : Nat) : String := s!"%dp{i}"

/-- **The classifier-dropout mask input name** — the `mName` a `dropoutB` carries, and the
    `tensor<B×n×f32>` the signature declares for it.

    ⚠⚠ **IT IS DELIBERATELY NOT `%dp{i}`-SHAPED, and that is not cosmetic.**
    `scripts/misplace_drop_sites.py` builds the stochastic-depth placement control by matching
    `%dp\d+` textually; a dropout input spelled `%dp9` would be swept into that rewrite, silently
    changing a control's meaning on a render it was never written for. Handoff §0.11 records the
    other half of this hazard on ViT — a control that quietly does nothing reads exactly like a
    control that ran — and the cheap defence is a name the SD tooling cannot match.
    `grep -c '%do' verified_mlir/*.mlir` is 0 across every committed artifact. -/
def doName : String := "%do"

-- ── Renderable skeleton + postorder tokenization (one form, shared with the
--    parser in StableHLOParse.lean) ──

/-- The renderable skeleton of an `SHlo` graph: opcodes + shapes + leaf SSA
    names, with `ℝ` operand values and the shape index erased — exactly what
    reaches the emitted text. -/
inductive Raw where
  | operand    (name : String) (n : Nat)  : Raw
  | dotIn      (w : String) (m n : Nat)    : Raw → Raw
  | dotInBf16  (w : String) (m n : Nat)    : Raw → Raw
  | dotOut     (w : String) (m n : Nat)    : Raw → Raw
  | addBcast   (b : String) (n : Nat)      : Raw → Raw
  | expe       (n : Nat)                   : Raw → Raw
  | softmaxDiv (n : Nat)                   : Raw → Raw
  | sub        (n : Nat)                   : Raw → Raw → Raw
  | weightSgd  (xName wName lrStr : String) (m n : Nat) : Raw → Raw
  | biasSgd    (bName lrStr : String) (n : Nat)         : Raw → Raw
  | convWeightSgd (xName wName lrStr : String) (ic oc h w kH kW : Nat) : Raw → Raw
  | convBiasSgd   (bName lrStr : String) (oc h w : Nat)               : Raw → Raw
  | bnGammaSgd    (gName vName epsStr lrStr : String) (oc h w : Nat)  : Raw → Raw
  | bnBetaSgd     (bName lrStr : String) (oc h w : Nat)               : Raw → Raw
  | layerScaleChGammaSgd (gName xName lrStr : String) (c h w : Nat)   : Raw → Raw
  | lnGammaSgd    (gName xName epsStr lrStr : String) (n : Nat)       : Raw → Raw
  | lnBetaSgd     (bName lrStr : String) (n : Nat)                    : Raw → Raw
  | veclnGammaSgd (gName xName epsStr lrStr : String) (N D : Nat)     : Raw → Raw
  | patchEmbedWeightSgd (wName xName lrStr : String) (ic H W P N D : Nat) : Raw → Raw
  | reluF      (n : Nat)                   : Raw → Raw
  | selectPos  (x : String) (n : Nat)      : Raw → Raw
  | relu6F     (n : Nat)                   : Raw → Raw
  | selectMid  (x : String) (n : Nat)      : Raw → Raw
  | convertF   (n : Nat)                   : Raw → Raw
  | flatConvF  (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | flatConvFBf16 (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | maxPoolF   (c h w : Nat)               : Raw → Raw
  | convBack   (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | maxPoolBack (x : String) (c h w : Nat) : Raw → Raw
  | bnF        (g b eps : String) (n : Nat) : Raw → Raw
  | bnBack     (g x eps : String) (n : Nat) : Raw → Raw
  | addV       (n : Nat)                   : Raw → Raw → Raw
  | gapF       (c h w : Nat)               : Raw → Raw
  | gapBack    (c h w : Nat)               : Raw → Raw
  | broadcastBack (c h w : Nat)            : Raw → Raw
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | flatConvStridedXlaF (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | depthwiseWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | flatConvStride4F (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | bnPerChannelF    (g b eps : String) (oc h w : Nat) : Raw → Raw
  | bnPerChannelBack (g x eps : String) (oc h w : Nat) : Raw → Raw
  | bnPerChannelEvalF (g b mu var eps : String) (oc h w : Nat) : Raw → Raw
  | weightGrad (x : String) (m n : Nat) : Raw → Raw
  | biasGrad (n : Nat) : Raw → Raw
  | convWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convBiasGrad (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | bnGammaGrad (v eps : String) (oc h w' : Nat) : Raw → Raw
  | bnBetaGrad (oc h w' : Nat) : Raw → Raw
  | adamMNextF (m b1 ob1 : String) (ds : List Nat) : Raw → Raw
  | adamVNextF (v b2 ob2 : String) (ds : List Nat) : Raw → Raw
  | adamWParamF (θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd : String) (ds : List Nat) : Raw → Raw
  | sgdParamF (θ lr : String) (ds : List Nat) : Raw → Raw
  | momVNextF (v mu : String) (ds : List Nat) : Raw → Raw
  | momParamF (θ v mu lr : String) (ds : List Nat) : Raw → Raw
  | rmsBufNextF (sq buf rho orho mu eps : String) (ds : List Nat) : Raw → Raw
  -- Global-norm grad clip. `gradClipFacF` keeps only its two literal strings (the ℝs are
  -- denotation-only, as everywhere in `Raw`); `clipScaleF`/`addScalarF` are BINARY.
  | gradSumSqAccF (ds : List Nat)                            : Raw → Raw → Raw
  | clipScaleF    (clipStr epsStr : String) (ds : List Nat)  : Raw → Raw → Raw
  | lambDirF (θ m v b1 ob1 b2 ob2 bc1 bc2 eps wd : String) (ds : List Nat) : Raw → Raw
  | lambScaleF    (ds : List Nat)                            : Raw → Raw → Raw
  | depthwiseF    (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedF    (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedXlaF (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | swishF     (n : Nat)                   : Raw → Raw
  | swishBack  (x : String) (n : Nat)      : Raw → Raw
  | sigmoidF   (n : Nat)                   : Raw → Raw
  | sigmoidBack (x : String) (n : Nat)     : Raw → Raw
  | geluF      (n : Nat)                   : Raw → Raw
  | geluBack   (x : String) (n : Nat)      : Raw → Raw
  | layerScaleF (γ : String) (n : Nat)     : Raw → Raw
  | layerScaleChF (γ : String) (c h w : Nat) : Raw → Raw
  | softmaxRowF    (m n : Nat)             : Raw → Raw
  | softmaxRowBack (x : String) (m n : Nat) : Raw → Raw
  | matmulF    (m k n : Nat)               : Raw → Raw → Raw
  | transposeF (m n : Nat)                 : Raw → Raw
  | scaleF     (s : String) (n : Nat)      : Raw → Raw
  | lnRowF     (g b eps : String) (m n : Nat) : Raw → Raw
  | lnRowBack  (g x eps : String) (m n : Nat) : Raw → Raw
  | denseRowF  (w b : String) (N a c : Nat) : Raw → Raw
  | denseRowBack (w : String) (N a c : Nat) : Raw → Raw
  | patchEmbedF (w b cls pos : String) (ic H W P N D : Nat) : Raw → Raw
  | clsSliceF  (N D : Nat)                 : Raw → Raw
  | clsPadF    (N D : Nat)                 : Raw → Raw
  | headSliceF (N heads d hIdx : Nat)      : Raw → Raw
  | headPadF   (N heads d hIdx : Nat)      : Raw → Raw
  | rowScaleF  (g : String) (m n : Nat)    : Raw → Raw
  | rowBiasF   (b : String) (m n : Nat)    : Raw → Raw
  -- EfficientNet batched ops (`batchOp`/`bnBatchF`/the batched backward ops): the
  -- renderable skeleton keeps a tag discriminating the op, the SSA names the emit
  -- references (weight/bias/BN-input/γ/ε/SE-input names), and shape info. The tag
  -- is the BatchableOp variant ("conv"/"depthwise"/"seBlock"/…) for forward ops or
  -- the backward op name; this is what lets `emitTok` reconstruct real StableHLO.
  | batched    (tag : String) (names : List String) (info : List Nat) : Raw → Raw
  -- The BINARY peer of `batched`, for the pointwise two-operand ops (`addV`/`sub`)
  -- at the batched index. Same reason as the unary descriptors: their
  -- descriptor-less tokens read the emit width off the SHlo index, so they cannot
  -- sit at `N·n`. `info` is `[N, n]`; the emit uses `n`, like every batched tag.
  | batched2   (tag : String) (names : List String) (info : List Nat) : Raw → Raw → Raw
deriving DecidableEq, Repr, Inhabited

/-- The `(tag, names, info)` skeleton descriptor of a batched per-example op — the
    discriminator + the SSA names the emit references + the shape dims. Keeps the
    `batchOp` skel one line and isolates the 7-variant match into a pure function. -/
def batchOpDescr {a b : Nat} (N : Nat) : BatchableOp a b → (String × List String × List Nat)
  | .conv (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("conv", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStrided (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStrided", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- ⚠ DISTINCT tags, for the `convStridedXla` reason one case down and one more: the emitted
  -- TEXT differs (converts + a bf16 result type), so sharing a Raw with the f32 tag would make
  -- two different graphs indistinguishable after `skel`.
  | .convBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStridedBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convStridedBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- ⚠ A DISTINCT tag, deliberately. Aliasing this onto "convStrided" (the way the bias-grads
  -- legitimately alias, because their emitted text is stride-independent) would be wrong here:
  -- the emitted `pad` differs, so the two tags must not share a Raw.
  | .convStridedXla (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStridedXla", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStridedXlaBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convStridedXlaBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .depthwise (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwise", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseBf16 (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("depthwiseBf16", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStrided (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwiseStrided", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStridedBf16 (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("depthwiseStridedBf16", [wN, bN], [N, c, h, w, kH, kW])
  -- Distinct tag, for the same reason `convStridedXla` is: the emitted `pad` differs.
  | .depthwiseStridedXla (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwiseStridedXla", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStridedXlaBf16 (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("depthwiseStridedXlaBf16", [wN, bN], [N, c, h, w, kH, kW])
  | .dense (c := c) wN bN _ _ => ("dense", [wN, bN], [N, a, c])
  | .gap (c := c) (h := h) (w := w) => ("gap", [], [N, c, h, w])
  | .seBlock (c := c) (h := h) (w := w) (r := r) w1 b1 w2 b2 _ _ _ _ =>
      ("seBlock", [w1, b1, w2, b2], [N, c, h, w, r])
  | .bnEval (oc := oc) (h := h) (w := w) gN bN muN varN es _ _ _ _ _ =>
      ("bnEval", [gN, bN, muN, varN, es], [N, oc, h, w])
  | .swish (n := n) => ("swish", [], [N, n])
  | .relu (n := n) => ("relu", [], [N, n])
  | .relu6 (n := n) => ("relu6", [], [N, n])
  | .maxPool (c := c) (h := h) (w := w) => ("maxPool", [], [N, c, h, w])
  -- ⚠ A DIFFERENT tag from `.maxPool`, deliberately. The two denote different functions at the
  -- same type, so sharing a tag would make the emitted text the only thing separating them — and
  -- the emitted text is what a reader checks last. `maxPool3s2_ne_maxPool_descr` is the den-side
  -- half of the same pin.
  | .maxPool3s2 (c := c) (h := h) (w := w) => ("maxPool3s2", [], [N, c, h, w])
  | .softmaxRow (m := m) (n := n) => ("softmaxRow", [], [N, m, n])
  | .denseRowBack (rows := rows) (a := a) (c := c) wN _ => ("denseRowBackP", [wN], [N, rows, a, c])
  -- ⚠ A DISTINCT tag, for the reason every bf16 tag in this function is distinct: the emitted TEXT
  -- differs (two operand converts and bf16 operand types), so sharing a Raw with the f32 tag would
  -- make two different graphs indistinguishable after `skel`.
  | .denseRowBackBf16 (rows := rows) (a := a) (c := c) _ wN _ =>
      ("denseRowBackPBf16", [wN], [N, rows, a, c])
  -- ⚠ `epsStr` rides in `names` though it is a LITERAL, not an SSA name — `bnEval` set that
  -- precedent and `emitTok` splices both the same way. The alternative is a second string list.
  | .gelu (n := n) => ("gelu", [], [N, n])
  | .transpose (m := m) (n := n) => ("transposeP", [], [N, m, n])
  | .convStride4 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStride4P", [wN, bN], [N, ic, oc, h, w, kH, kW])
  -- ⚠ A DISTINCT tag, for the reason every other bf16 conv tag is distinct: the emitted TEXT
  -- differs (two operand converts, a bf16 result type, a convert back), so sharing a Raw with the
  -- f32 tag would make two different graphs indistinguishable after `skel`.
  | .convStride4Bf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ =>
      ("convStride4PBf16", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .layerScaleCh (c := c) (h := h) (w := w) gN _ => ("layerScaleChP", [gN], [N, c, h, w])
  | .dotOut (m := m) (n := n) wN _ => ("dotOutP", [wN], [N, m, n])
  | .expe (n := n) => ("expeP", [], [N, n])
  | .softmaxDiv (n := n) => ("softmaxDivP", [], [N, n])
  | .lnRow (m := m) (n := n) gN bN es _ _ _ => ("lnRowP", [gN, bN, es], [N, m, n])
  | .rowScale (m := m) (n := n) gN _ => ("rowScaleP", [gN], [N, m, n])
  | .rowBias (m := m) (n := n) bN _ => ("rowBiasP", [bN], [N, m, n])
  -- ViT increment 1. ⚠ The FIRST entry of `info` is the batch and every following one is the tag's
  -- own dims — so `[N, tk, a, c]` reads "batch N, token count tk". The emitter below uses only the
  -- tail (it takes the batch from `pretty`'s `B`), which is why the batched text is its
  -- per-example peer's byte for byte; `tests/TestBatchedEmitTie.lean` is what pins that.
  | .denseRow (N := tk) (a := a) (c := c) wN bN _ _ => ("denseRowP", [wN, bN], [N, tk, a, c])
  | .denseRowBf16 (N := tk) (a := a) (c := c) _ wN bN _ _ =>
      ("denseRowPBf16", [wN, bN], [N, tk, a, c])
  | .patchEmbed (ic := ic) (H := H) (W := W) (P := P) (N := tk) (D := D) wN bN clsN posN _ _ _ _ =>
      ("patchEmbedP", [wN, bN, clsN, posN], [N, ic, H, W, P, tk, D])
  | .patchEmbedBf16 (ic := ic) (H := H) (W := W) (P := P) (N := tk) (D := D)
        _ wN bN clsN posN _ _ _ _ =>
      ("patchEmbedPBf16", [wN, bN, clsN, posN], [N, ic, H, W, P, tk, D])
  | .clsSlice (N := tk) (D := D) => ("clsSliceP", [], [N, tk, D])
  | .clsPad (N := tk) (D := D) => ("clsPadP", [], [N, tk, D])
  | .headSlice (N := tk) (heads := heads) (d := d) h => ("headSliceP", [], [N, tk, heads, d, h.val])
  | .headPad (N := tk) (heads := heads) (d := d) h => ("headPadP", [], [N, tk, heads, d, h.val])

/-- Erase an `SHlo` graph to its renderable skeleton (drops `ℝ` values + shape
    index; keeps op structure, shapes, leaf names). -/
def skel : {k : Nat} → SHlo k → Raw
  | k, .operand name _        => .operand name k
  | k, .dotIn (m := m) w _ e  => .dotIn w m k (skel e)
  | k, .dotInBf16 (m := m) _ w _ e => .dotInBf16 w m k (skel e)
  | k, .dotOut (n := n) w _ e => .dotOut w k n (skel e)
  | k, .addBcast b _ e        => .addBcast b k (skel e)
  | k, .expe e                => .expe k (skel e)
  | k, .softmaxDiv e          => .softmaxDiv k (skel e)
  | k, .sub a b               => .sub k (skel a) (skel b)
  | _, .weightSgd (m := m) (n := n) xN wN lrS _ _ _ e => .weightSgd xN wN lrS m n (skel e)
  | k, .biasSgd bN lrS _ _ e  => .biasSgd bN lrS k (skel e)
  | _, .convWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .bnGammaSgd (oc := oc) (h := h) (w := w) gN vN es lrS _ _ _ _ e =>
      .bnGammaSgd gN vN es lrS oc h w (skel e)
  | _, .bnBetaSgd (oc := oc) (h := h) (w := w) bN lrS _ _ e =>
      .bnBetaSgd bN lrS oc h w (skel e)
  | _, .layerScaleChGammaSgd (c := c) (h := h) (w := w) gN xN lrS _ _ _ e =>
      .layerScaleChGammaSgd gN xN lrS c h w (skel e)
  | _, .lnGammaSgd (n := n) gN xN es lrS _ _ _ _ e =>
      .lnGammaSgd gN xN es lrS n (skel e)
  | _, .lnBetaSgd (n := n) bN lrS _ _ e =>
      .lnBetaSgd bN lrS n (skel e)
  | _, .veclnGammaSgd (N := N) (D := D) gN xN es lrS _ _ _ _ e =>
      .veclnGammaSgd gN xN es lrS N D (skel e)
  | _, .patchEmbedWeightSgd (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) wN xN lrS _ _ _ e =>
      .patchEmbedWeightSgd wN xN lrS ic H W P N D (skel e)
  | k, .reluF e               => .reluF k (skel e)
  | k, .selectPos x _ e       => .selectPos x k (skel e)
  | k, .relu6F e              => .relu6F k (skel e)
  | k, .selectMid x _ e       => .selectMid x k (skel e)
  | k, .convertF _ e          => .convertF k (skel e)
  | _, .flatConvF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvF wN bN ic oc h w kH kW (skel e)
  | _, .flatConvFBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN bN _ _ e =>
      .flatConvFBf16 wN bN ic oc h w kH kW (skel e)
  | _, .maxPoolF (c := c) (h := h) (w := w) e => .maxPoolF c h w (skel e)
  | _, .convBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convBack wN ic oc h w kH kW (skel e)
  | _, .maxPoolBack (c := c) (h := h) (w := w) xN _ e => .maxPoolBack xN c h w (skel e)
  | k, .bnF gN bN es _ _ _ e => .bnF gN bN es k (skel e)
  | k, .bnBack gN xN es _ _ _ e => .bnBack gN xN es k (skel e)
  | k, .addV a b              => .addV k (skel a) (skel b)
  | _, .maxPoolBackB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "maxPoolBackP" [xN] [N, c, h, w] (skel e)
  -- ⭐ The three 3×3/s2 pool forms all ride the generic `.batched` tag, so they cost NO
  -- `Raw`/`Tok`/`toToks`/`parseStack`/`parse_toToks` work (§0.2 increment 2's five-site route).
  -- ⚠ The per-example and batched BACKWARDS share one tag and are distinguished by the nat list's
  -- ARITY (3 vs 4) — `depthwiseWeightGrad`'s convention, and legitimate here for its reason: the
  -- emitter ignores `N` (it reads the batch off `pretty`'s `B`), so the two emit identical text by
  -- construction and the arity carries only the `den`-side difference.
  | _, .maxPool3s2F (c := c) (h := h) (w := w) e =>
      .batched "maxPool3s2" [] [c, h, w] (skel e)
  | _, .maxPool3s2Back (c := c) (h := h) (w := w) xN _ e =>
      .batched "maxPool3s2BackP" [xN] [c, h, w] (skel e)
  | _, .maxPool3s2BackB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "maxPool3s2BackP" [xN] [N, c, h, w] (skel e)
  | _, .convBiasSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .batched "convBiasSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .convStridedBiasSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .batched "convBiasSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .selectPosB (N := N) (n := n) xN _ e => .batched "selectPosP" [xN] [N, n] (skel e)
  | _, .selectMidB (N := N) (n := n) xN _ e => .batched "selectMidP" [xN] [N, n] (skel e)
  -- Two name slots: the mask INPUT and the baked `1/keep` literal. Same two-string shape
  -- `convStridedWeightSgd` uses for `xN`/`lrS`, so the generic `.batched` tag needs no widening.
  | _, .dropPathB (N := N) (n := n) mN _ e => .batched "dropPathP" [mN] [N, n] (skel e)
  -- ⚠ A DIFFERENT TAG, not a flag on `dropPathP`. The two emit different text and denote different
  -- functions, so they must be distinguishable in the skeleton — a shared tag would make the
  -- round-trip parser unable to tell a per-sample render from a per-element one.
  | _, .dropoutB (N := N) (n := n) mN _ e => .batched "dropoutP" [mN] [N, n] (skel e)
  | _, .swishBackB (N := N) (n := n) xN _ e => .batched "swishBackP" [xN] [N, n] (skel e)
  -- ⚠ Routed through the GENERIC `.batched` tag, like every batched op above — which is why these
  -- cost five sites (ctor, den, the `rfl` theorem, this line, `emitTok`) and not §4's ten: `Raw`,
  -- `Tok`, `toToks`, `parseStack` and the `parse_toToks` induction all already handle `.batched`.
  | _, .geluBackB (N := N) (n := n) xN _ e => .batched "geluBackP" [xN] [N, n] (skel e)
  -- ⚠⚠ THESE FOUR ALIAS THEIR PER-EXAMPLE PEER'S `Raw` — deliberately, and it is the pattern the
  -- depthwise bias grads already use. Their emitted MLIR ALREADY contracts the batch axis
  -- (`layerScaleChGammaGrad` reduces `dimensions = [0, 2, 3]`, `rowDenseBiasGrad` `[0, 1]`), because
  -- values flow as `tensor<B, …>` and `B` is `pretty`'s, never the SHlo index. So the batched form
  -- emits the same text BY CONSTRUCTION rather than by a copied body — no new `emitTok` case, and
  -- no way for the two to drift. It is the `den` that was per-example and is now honest.
  --
  -- ⚠ Note what is dropped: the batch `N` does NOT ride in `info`. The emitter never used it (it
  -- reads `B`), so passing it would add a number that means nothing at the emit and everything at
  -- the denotation — the exact conflation this whole thread exists to remove.
  | _, .convStride4WeightGradB (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStride4WeightGrad" [xN] [ic, oc, h, w, kH, kW] (skel e)
  | _, .convStride4WeightGradBBf16 (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convStride4WeightGradBf16" [xN] [ic, oc, h, w, kH, kW] (skel e)
  | _, .layerScaleChGammaGradB (c := c) (h := h) (w := w) xN _ e =>
      .batched "layerScaleChGammaGrad" [xN] [c, h, w] (skel e)
  | _, .veclnGammaGradB (R := R) (D := D) xN es _ _ e =>
      .batched "veclnGammaGrad" [xN, es] [R, D] (skel e)
  | _, .rowDenseBiasGradB (R := R) (c := c) e =>
      .batched "rowDenseBiasGrad" [] [R, c] (skel e)
  -- ── ViT increment 2: SIX ALIASES AND NOT ONE NEW EMIT CASE. Every one of these emits its
  --    per-example peer's `Raw` verbatim — the batch never appears in the tag, because the emitter
  --    reads `B` from `pretty` and its dims from the tag, and (for the four gradients) already
  --    contracts the batch axis. So these forms cannot drift from their peers by construction
  --    rather than by a copied body kept honest by a test. Increment 3's finding, generalised.
  --    ⚠ `matmulFB` rides the BINARY `.matmulF` tag: `.batched2` exists for `addVB`/`subB`, but a
  --    direct alias is better still, because it shares the emit rather than restating it.
  | _, .matmulFB (m := m) (k := k) (n := n) a b => .matmulF m k n (skel a) (skel b)
  -- ⚠ Its bf16 peer can NOT alias `.matmulF` — the text differs — so it rides the generic BINARY
  -- skeleton `.batched2` that `addVB`/`subB` already use, and needs no new `Raw`/`Tok` of its own.
  | _, .matmulFBBf16 (m := m) (k := k) (n := n) _ a b =>
      .batched2 "matmulFBf16" [] [m, k, n] (skel a) (skel b)
  | _, .softmaxRowBackB (m := m) (n := n) x _ e => .softmaxRowBack x m n (skel e)
  | _, .rowDenseWeightGradB (tk := tk) (a := a) (c := c) xN _ e =>
      .batched "rowDenseWeightGrad" [xN] [tk, a, c] (skel e)
  | _, .rowDenseWeightGradBBf16 (tk := tk) (a := a) (c := c) _ xN _ e =>
      .batched "rowDenseWeightGradBf16" [xN] [tk, a, c] (skel e)
  | _, .posEmbedGradB (tk := tk) (D := D) e =>
      .batched "posEmbedGrad" [] [tk, D] (skel e)
  | _, .patchEmbedWeightGradB (ic := ic) (H := H) (W := W) (P := P) (tk := tk) (D := D) xN _ e =>
      .batched "patchEmbedWeightGrad" [xN] [ic, H, W, P, tk, D] (skel e)
  | _, .patchEmbedWeightGradBBf16 (ic := ic) (H := H) (W := W) (P := P) (tk := tk) (D := D)
        _ xN _ e =>
      .batched "patchEmbedWeightGradBf16" [xN] [ic, H, W, P, tk, D] (skel e)
  | _, .patchEmbedBiasGradB (tk := tk) (c := c) e =>
      .batched "patchEmbedBiasGrad" [] [tk, c] (skel e)
  | _, .weightGradB (m := m) (n := n) xN _ e => .weightGrad xN m n (skel e)
  | _, .biasGradB (n := n) e => .biasGrad n (skel e)
  | _, .lnRowBackB (N := N) (m := m) (n := n) gN xN es _ _ _ e =>
      .batched "lnRowBackP" [gN, xN, es] [N, m, n] (skel e)
  | _, .sigmoidB (N := N) (n := n) e => .batched "sigmoidP" [] [N, n] (skel e)
  | _, .sigmoidBackB (N := N) (n := n) xN _ e => .batched "sigmoidBackP" [xN] [N, n] (skel e)
  | _, .addVB (N := N) (n := n) a b => .batched2 "addV" [] [N, n] (skel a) (skel b)
  | _, .subB (N := N) (n := n) a b  => .batched2 "sub" [] [N, n] (skel a) (skel b)
  | _, .gapF (c := c) (h := h) (w := w) e => .gapF c h w (skel e)
  | _, .gapBack (c := c) (h := h) (w := w) e => .gapBack c h w (skel e)
  | _, .broadcastBack (c := c) (h := h) (w := w) e => .broadcastBack c h w (skel e)
  | _, .flatConvStridedF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStridedF wN bN ic oc h w kH kW (skel e)
  | _, .flatConvStridedXlaF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStridedXlaF wN bN ic oc h w kH kW (skel e)
  | _, .convStridedBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convStridedBack wN ic oc h w kH kW (skel e)
  | _, .convStridedWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convStridedWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convStridedBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .depthwiseWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .depthwiseStridedWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseStridedWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseStridedBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .flatConvStride4F (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStride4F wN bN ic oc h w kH kW (skel e)
  | _, .bnPerChannelF (oc := oc) (h := h) (w := w) gN bN es _ _ _ e =>
      .bnPerChannelF gN bN es oc h w (skel e)
  | _, .bnPerChannelBack (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .bnPerChannelBack gN xN es oc h w (skel e)
  | _, .bnPerChannelEvalF (oc := oc) (h := h) (w := w) gN bN muN varN es _ _ _ _ _ e =>
      .bnPerChannelEvalF gN bN muN varN es oc h w (skel e)
  | _, .weightGrad (m := m) (n := n) xN _ e => .weightGrad xN m n (skel e)
  | _, .biasGrad (n := n) e => .biasGrad n (skel e)
  | _, .convWeightGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .convWeightGrad xN ic oc h w kH kW (skel e)
  | _, .convBiasGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .convBiasGrad ic oc h w kH kW (skel e)
  | _, .convStridedWeightGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .convStridedWeightGrad xN ic oc h w kH kW (skel e)
  -- rides the generic `.batched` tag (the four-site route, §4) — no new Raw/Tok/parse case.
  | _, .convStride4WeightGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStride4WeightGrad" [xN] [ic, oc, h, w, kH, kW] (skel e)
  -- aliases convBiasGrad's Raw: the bias grad is stride-independent, so the text is identical
  | _, .convStridedBiasGrad (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .convBiasGrad ic oc h w kH kW (skel e)
  | _, .bnGammaGrad (oc := oc) (h := h) (w := w) vN es _ _ e => .bnGammaGrad vN es oc h w (skel e)
  | _, .bnBetaGrad (oc := oc) (h := h) (w := w) e => .bnBetaGrad oc h w (skel e)
  | _, .adamMNextF mN b1N ob1N ds _ _ e => .adamMNextF mN b1N ob1N ds (skel e)
  | _, .adamVNextF vN b2N ob2N ds _ _ e => .adamVNextF vN b2N ob2N ds (skel e)
  | _, .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
        _ _ _ _ _ _ _ _ _ _ e =>
      .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds (skel e)
  | _, .sgdParamF θN lrN ds _ _ e => .sgdParamF θN lrN ds (skel e)
  | _, .momVNextF vN muN ds _ _ e => .momVNextF vN muN ds (skel e)
  | _, .momParamF θN vN muN lrN ds _ _ _ _ e => .momParamF θN vN muN lrN ds (skel e)
  | _, .rmsBufNextF sqN bufN rhoN orhoN muN epsN ds _ _ _ _ _ e =>
      .rmsBufNextF sqN bufN rhoN orhoN muN epsN ds (skel e)
  | _, .gradSumSqAccF ds acc e         => .gradSumSqAccF ds (skel acc) (skel e)
  | _, .clipScaleF cS eS _ _ ds s e    => .clipScaleF cS eS ds (skel s) (skel e)
  | _, .lambDirF a b c d e' f g h i j k ds _ _ _ _ _ _ _ _ _ x =>
      .lambDirF a b c d e' f g h i j k ds (skel x)
  | _, .lambScaleF ds s e              => .lambScaleF ds (skel s) (skel e)
  | _, .depthwiseF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseF wN bN c h w kH kW (skel e)
  | _, .depthwiseBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseBack wN c h w kH kW (skel e)
  | _, .depthwiseStridedF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseStridedF wN bN c h w kH kW (skel e)
  | _, .depthwiseStridedXlaF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseStridedXlaF wN bN c h w kH kW (skel e)
  | _, .depthwiseStridedBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseStridedBack wN c h w kH kW (skel e)
  | k, .swishF e             => .swishF k (skel e)
  | k, .swishBack x _ e      => .swishBack x k (skel e)
  | k, .sigmoidF e           => .sigmoidF k (skel e)
  | k, .sigmoidBack x _ e    => .sigmoidBack x k (skel e)
  | k, .geluF e              => .geluF k (skel e)
  | k, .geluBack x _ e       => .geluBack x k (skel e)
  | k, .layerScaleF γN _ e   => .layerScaleF γN k (skel e)
  | _, .layerScaleChF (c := c) (h := h) (w := w) γN _ e => .layerScaleChF γN c h w (skel e)
  | _, .softmaxRowF (m := m) (n := n) e => .softmaxRowF m n (skel e)
  | _, .softmaxRowBack (m := m) (n := n) x _ e => .softmaxRowBack x m n (skel e)
  | _, .matmulF (m := m) (k := k) (n := n) a b => .matmulF m k n (skel a) (skel b)
  | _, .transposeF (m := m) (n := n) e => .transposeF m n (skel e)
  | k, .scaleF sStr _ e => .scaleF sStr k (skel e)
  | _, .lnRowF (m := m) (n := n) gN bN es _ _ _ e => .lnRowF gN bN es m n (skel e)
  | _, .lnRowBack (m := m) (n := n) gN xN es _ _ _ e => .lnRowBack gN xN es m n (skel e)
  | _, .denseRowF (N := N) (a := a) (c := c) wN bN _ _ e => .denseRowF wN bN N a c (skel e)
  | _, .denseRowBack (N := N) (a := a) (c := c) wN _ e => .denseRowBack wN N a c (skel e)
  | _, .patchEmbedF (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) wN bN cN pN _ _ _ _ e =>
      .patchEmbedF wN bN cN pN ic H W P N D (skel e)
  | _, .patchEmbedBack (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ e =>
      .batched "patchEmbedBack" [] [ic, H, W, P, N, D] (skel e)
  | _, .clsSliceF (N := N) (D := D) e => .clsSliceF N D (skel e)
  | _, .clsPadF (N := N) (D := D) e => .clsPadF N D (skel e)
  | _, .headSliceF (N := N) (heads := heads) (d := d) h e => .headSliceF N heads d h.val (skel e)
  | _, .headPadF (N := N) (heads := heads) (d := d) h e => .headPadF N heads d h.val (skel e)
  | _, .rowScaleF (m := m) (n := n) gN _ e => .rowScaleF gN m n (skel e)
  | _, .rowBiasF (m := m) (n := n) bN _ e => .rowBiasF bN m n (skel e)
  | _, .batchOp (N := N) op e =>
      let (tag, nms, inf) := batchOpDescr N op; .batched tag nms inf (skel e)
  | _, .bnBatchF (N := N) (oc := oc) (h := h) (w := w) gN bN es _ _ _ e =>
      .batched "bnBatch" [gN, bN, es] [N, oc, h, w] (skel e)
  | _, .bnBatchBack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .batched "bnBatchBack" [gN, xN, es] [N, oc, h, w] (skel e)
  | _, .convBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "convBackBatched" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "convStridedBackBatched" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convBackBatchedBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "convBackBatchedBf16" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedBackBatchedBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "convStridedBackBatchedBf16" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .depthwiseBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "depthwiseBackBatchedBf16" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseStridedBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "depthwiseStridedBackBatchedBf16" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseStridedXlaBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaBackBatchedBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ wN _ _ e =>
      .batched "depthwiseStridedXlaBackBatchedBf16" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .batched "bnBatchLABack" [gN, xN, es] [N, oc, h, w] (skel e)
  | _, .seBackBatched (N := N) (c := c) (h := h) (w := w) (r := r) w1 b1 w2 b2 vN _ _ _ _ _ e =>
      .batched "seBackBatched" [w1, b1, w2, b2, vN] [N, c, h, w, r] (skel e)
  | _, .seReduceB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "seReduceB" [xN] [N, c, h, w] (skel e)
  | _, .gapBackBatched (N := N) (c := c) (h := h) (w := w) e =>
      .batched "gapBackBatched" [] [N, c, h, w] (skel e)
  | _, .bnGammaSgdB (N := N) (oc := oc) (h := h) (w := w) gN vN es lrS _ _ _ _ e =>
      .batched "bnGammaSgd" [gN, vN, es, lrS] [N, oc, h, w] (skel e)
  | _, .bnBetaSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ e =>
      .batched "bnBetaSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .denseWeightSgdB (N := N) (a := a) (c := c) xN wN lrS _ _ _ e =>
      .batched "denseWeightSgd" [xN, wN, lrS] [N, a, c] (skel e)
  | _, .denseBiasSgdB (N := N) (c := c) bN lrS _ _ e =>
      .batched "denseBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .convWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convWeightGrad" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStridedWeightGrad" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convWeightGradBf16" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convStridedWeightGradBf16" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedXlaWeightGradB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "convStridedXlaWeightGrad" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedXlaWeightGradBBf16 (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "convStridedXlaWeightGradBf16" [xN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convBiasGradB (N := N) (oc := oc) (h := h) (w := w) _ _ _ e =>
      .batched "convBiasGrad" [] [N, oc, h, w] (skel e)
  | _, .convStridedBiasGradB (N := N) (oc := oc) (h := h) (w := w) _ _ _ e =>
      .batched "convBiasGrad" [] [N, oc, h, w] (skel e)
  -- ⭐ Aliases "convBiasGrad" DELIBERATELY: `Σ_{batch,spatial} dy` is padding-independent, so the
  -- emitted text is character-identical and only `den` distinguishes them. Same aliasing
  -- `convStridedBiasGradB` already does for stride.
  | _, .convStridedXlaBiasGradB (N := N) (oc := oc) (h := h) (w := w) _ _ _ e =>
      .batched "convBiasGrad" [] [N, oc, h, w] (skel e)
  | _, .bnGammaGradB (N := N) (oc := oc) (h := h) (w := w) vN es _ _ e =>
      .batched "bnGammaGrad" [vN, es] [N, oc, h, w] (skel e)
  | _, .bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) e =>
      .batched "bnBetaGrad" [] [N, oc, h, w] (skel e)
  | _, .denseWeightGradB (N := N) (a := a) (c := c) xN _ e =>
      .batched "denseWeightGrad" [xN] [N, a, c] (skel e)
  | _, .denseBiasGradB (N := N) (c := c) e =>
      .batched "denseBiasGrad" [] [N, c] (skel e)
  | _, .bnBatchMeanB (N := N) (oc := oc) (h := h) (w := w) e =>
      .batched "bnBatchMean" [] [N, oc, h, w] (skel e)
  | _, .bnBatchVarB (N := N) (oc := oc) (h := h) (w := w) e =>
      .batched "bnBatchVar" [] [N, oc, h, w] (skel e)
  | _, .scaleB (N := N) (n := n) sS _ e    => .batched "scale" [sS] [N, n] (skel e)
  | _, .shiftB (N := N) (n := n) sS _ e    => .batched "shift" [sS] [N, n] (skel e)
  | _, .divConstB (N := N) (n := n) sS _ e => .batched "divConst" [sS] [N, n] (skel e)
  | _, .rowDenseWeightSgd (N := N) (a := a) (c := c) xN wN lrS _ _ _ e =>
      .batched "rowDenseWeightSgd" [xN, wN, lrS] [N, a, c] (skel e)
  | _, .rowDenseBiasSgd (N := N) (c := c) bN lrS _ _ e =>
      .batched "rowDenseBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .patchEmbedBiasSgd (N := N) (c := c) bN lrS _ _ e =>
      .batched "patchEmbedBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .posEmbedSgd (N := N) (D := D) pN lrS _ _ e =>
      .batched "posEmbedSgd" [pN, lrS] [N, D] (skel e)
  -- The un-fused transformer peers. They ride the same generic `.batched` tag (so `Raw`/`Tok`/
  -- `toToks`/`parseStack`/`parse_toToks` need no new cases), carrying only the operands the
  -- gradient actually reads — no param name, no lr.
  | _, .rowDenseWeightGrad (N := N) (a := a) (c := c) xN _ e =>
      .batched "rowDenseWeightGrad" [xN] [N, a, c] (skel e)
  | _, .rowDenseBiasGrad (N := N) (c := c) e =>
      .batched "rowDenseBiasGrad" [] [N, c] (skel e)
  | _, .patchEmbedBiasGrad (N := N) (c := c) e =>
      .batched "patchEmbedBiasGrad" [] [N, c] (skel e)
  | _, .posEmbedGrad (N := N) (D := D) e =>
      .batched "posEmbedGrad" [] [N, D] (skel e)
  | _, .veclnGammaGrad (N := N) (D := D) xN es _ _ e =>
      .batched "veclnGammaGrad" [xN, es] [N, D] (skel e)
  | _, .patchEmbedWeightGrad (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) xN _ e =>
      .batched "patchEmbedWeightGrad" [xN] [ic, H, W, P, N, D] (skel e)
  | _, .convWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convStridedWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedXlaWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convStridedXlaWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightSgdB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "depthwiseWeightSgd" [xN, wN, lrS] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightGradB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseWeightGrad" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "depthwiseWeightGradBf16" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaWeightGradB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseStridedXlaWeightGrad" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedXlaWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "depthwiseStridedXlaWeightGradBf16" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedWeightGradB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseStridedWeightGrad" [xN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedWeightGradBBf16 (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ xN _ _ _ e =>
      .batched "depthwiseStridedWeightGradBf16" [xN] [N, c, h, w, kH, kW] (skel e)
  -- Both depthwise BIAS grads alias ConvNeXt's per-example `depthwiseBiasGrad` Raw — the bias
  -- gradient is `Σ_{batch,spatial} dy`, stride-independent AND kernel-independent, so one emitter
  -- serves all three. `N` is dropped for the same reason every batched tag drops it: the runtime
  -- batch is `B`. This is the aliasing route, so there is nothing to add in `emitTok`.
  | _, .depthwiseBiasGradB (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  -- ⚠ info is `[c, h, w, kH, kW]` — NO leading `N`, matching the symmetric peer below rather than
  -- the `*WeightGrad` convention. It aliases that op's Raw, so the shape must agree exactly.
  | _, .depthwiseStridedXlaBiasGradB (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBiasGradB (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  -- The ConvNeXt five ride the generic `.batched` Raw/Tok tag, so they need no new
  -- `Raw`/`Tok`/`toToks`/`parseStack`/`parse_toToks` cases — the four-site route (§4).
  --
  -- `depthwiseWeightGrad` deliberately **aliases the batched op's tag**: its emitted text is
  -- byte-identical (that emitter ignores its `N` and reads the width off the render batch `B`), so
  -- only `den` differs — per-example here, a sum over `Fin N` there. Exactly the aliasing
  -- `convStridedBiasGrad` already does against `convBiasGrad`. It is distinguished by ARITY: six
  -- nats for the batched form, five for this one, so the two `emitTok` cases cannot collide.
  | _, .depthwiseWeightGrad (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN _ _ _ e =>
      .batched "depthwiseWeightGrad" [xN] [c, h, w, kH, kW] (skel e)
  | _, .depthwiseBiasGrad (c := c) (h := h) (w := w) (kH := kH) (kW := kW) _ _ _ e =>
      .batched "depthwiseBiasGrad" [] [c, h, w, kH, kW] (skel e)
  | _, .lnGammaGrad (n := n) xN es _ _ e => .batched "lnGammaGrad" [xN, es] [n] (skel e)
  | _, .lnBetaGrad (n := n) e => .batched "lnBetaGrad" [] [n] (skel e)
  | _, .layerScaleChGammaGrad (c := c) (h := h) (w := w) xN _ e =>
      .batched "layerScaleChGammaGrad" [xN] [c, h, w] (skel e)
  | _, .depthwiseStridedWeightSgdB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "depthwiseStridedWeightSgd" [xN, wN, lrS] [N, c, h, w, kH, kW] (skel e)

/-- One serialized token: an opcode with shapes/names; operands are positional. -/
inductive Tok where
  | operand    (name : String) (n : Nat)  : Tok
  | dotIn      (w : String) (m n : Nat)    : Tok
  | dotInBf16  (w : String) (m n : Nat)    : Tok
  | dotOut     (w : String) (m n : Nat)    : Tok
  | addBcast   (b : String) (n : Nat)      : Tok
  | expe       (n : Nat)                   : Tok
  | softmaxDiv (n : Nat)                   : Tok
  | sub        (n : Nat)                   : Tok
  | weightSgd  (xName wName lrStr : String) (m n : Nat) : Tok
  | biasSgd    (bName lrStr : String) (n : Nat)         : Tok
  | convWeightSgd (xName wName lrStr : String) (ic oc h w kH kW : Nat) : Tok
  | convBiasSgd   (bName lrStr : String) (oc h w : Nat)               : Tok
  | bnGammaSgd    (gName vName epsStr lrStr : String) (oc h w : Nat)  : Tok
  | bnBetaSgd     (bName lrStr : String) (oc h w : Nat)               : Tok
  | layerScaleChGammaSgd (gName xName lrStr : String) (c h w : Nat)   : Tok
  | lnGammaSgd    (gName xName epsStr lrStr : String) (n : Nat)       : Tok
  | lnBetaSgd     (bName lrStr : String) (n : Nat)                    : Tok
  | veclnGammaSgd (gName xName epsStr lrStr : String) (N D : Nat)     : Tok
  | patchEmbedWeightSgd (wName xName lrStr : String) (ic H W P N D : Nat) : Tok
  | reluF      (n : Nat)                   : Tok
  | selectPos  (x : String) (n : Nat)      : Tok
  | relu6F     (n : Nat)                   : Tok
  | selectMid  (x : String) (n : Nat)      : Tok
  | convertF   (n : Nat)                   : Tok
  | flatConvF  (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | flatConvFBf16 (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | maxPoolF   (c h w : Nat)               : Tok
  | convBack   (w : String) (ic oc h w' kH kW : Nat) : Tok
  | maxPoolBack (x : String) (c h w : Nat) : Tok
  | bnF        (g b eps : String) (n : Nat) : Tok
  | bnBack     (g x eps : String) (n : Nat) : Tok
  | addV       (n : Nat)                   : Tok
  | gapF       (c h w : Nat)               : Tok
  | gapBack    (c h w : Nat)               : Tok
  | broadcastBack (c h w : Nat)            : Tok
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | flatConvStridedXlaF (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Tok
  | depthwiseWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | flatConvStride4F (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | bnPerChannelF    (g b eps : String) (oc h w : Nat) : Tok
  | bnPerChannelBack (g x eps : String) (oc h w : Nat) : Tok
  | bnPerChannelEvalF (g b mu var eps : String) (oc h w : Nat) : Tok
  | weightGrad (x : String) (m n : Nat) : Tok
  | biasGrad (n : Nat) : Tok
  | convWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Tok
  | convBiasGrad (ic oc h w' kH kW : Nat) : Tok
  | convStridedWeightGrad (x : String) (ic oc h w' kH kW : Nat) : Tok
  | bnGammaGrad (v eps : String) (oc h w' : Nat) : Tok
  | bnBetaGrad (oc h w' : Nat) : Tok
  | adamMNextF (m b1 ob1 : String) (ds : List Nat) : Tok
  | adamVNextF (v b2 ob2 : String) (ds : List Nat) : Tok
  | adamWParamF (θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd : String) (ds : List Nat) : Tok
  | sgdParamF (θ lr : String) (ds : List Nat) : Tok
  | momVNextF (v mu : String) (ds : List Nat) : Tok
  | momParamF (θ v mu lr : String) (ds : List Nat) : Tok
  | rmsBufNextF (sq buf rho orho mu eps : String) (ds : List Nat) : Tok
  | gradSumSqAccF (ds : List Nat)                           : Tok
  | clipScaleF    (clipStr epsStr : String) (ds : List Nat) : Tok
  | lambDirF (θ m v b1 ob1 b2 ob2 bc1 bc2 eps wd : String) (ds : List Nat) : Tok
  | lambScaleF    (ds : List Nat)                           : Tok
  | depthwiseF    (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseBack (w : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedF    (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedXlaF (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedBack (w : String) (c h w' kH kW : Nat) : Tok
  | swishF     (n : Nat)                   : Tok
  | swishBack  (x : String) (n : Nat)      : Tok
  | sigmoidF   (n : Nat)                   : Tok
  | sigmoidBack (x : String) (n : Nat)     : Tok
  | geluF      (n : Nat)                   : Tok
  | geluBack   (x : String) (n : Nat)      : Tok
  | layerScaleF (γ : String) (n : Nat)     : Tok
  | layerScaleChF (γ : String) (c h w : Nat) : Tok
  | softmaxRowF    (m n : Nat)             : Tok
  | softmaxRowBack (x : String) (m n : Nat) : Tok
  | matmulF    (m k n : Nat)               : Tok
  | transposeF (m n : Nat)                 : Tok
  | scaleF     (s : String) (n : Nat)      : Tok
  | lnRowF     (g b eps : String) (m n : Nat) : Tok
  | lnRowBack  (g x eps : String) (m n : Nat) : Tok
  | denseRowF  (w b : String) (N a c : Nat) : Tok
  | denseRowBack (w : String) (N a c : Nat) : Tok
  | patchEmbedF (w b cls pos : String) (ic H W P N D : Nat) : Tok
  | clsSliceF  (N D : Nat)                 : Tok
  | clsPadF    (N D : Nat)                 : Tok
  | headSliceF (N heads d hIdx : Nat)      : Tok
  | headPadF   (N heads d hIdx : Nat)      : Tok
  | rowScaleF  (g : String) (m n : Nat)    : Tok
  | rowBiasF   (b : String) (m n : Nat)    : Tok
  | batched    (tag : String) (names : List String) (info : List Nat) : Tok
  | batched2   (tag : String) (names : List String) (info : List Nat) : Tok
deriving DecidableEq, Repr

/-- Postorder serialization: children, then the node's opcode token. -/
def toToks : Raw → List Tok
  | .operand nm n    => [.operand nm n]
  | .dotIn w m n e   => toToks e ++ [.dotIn w m n]
  | .dotInBf16 w m n e => toToks e ++ [.dotInBf16 w m n]
  | .dotOut w m n e  => toToks e ++ [.dotOut w m n]
  | .addBcast b n e  => toToks e ++ [.addBcast b n]
  | .expe n e        => toToks e ++ [.expe n]
  | .softmaxDiv n e  => toToks e ++ [.softmaxDiv n]
  | .sub n a b       => toToks a ++ toToks b ++ [.sub n]
  | .weightSgd xN wN lrS m n e => toToks e ++ [.weightSgd xN wN lrS m n]
  | .biasSgd bN lrS n e        => toToks e ++ [.biasSgd bN lrS n]
  | .convWeightSgd xN wN lrS ic oc h w kH kW e => toToks e ++ [.convWeightSgd xN wN lrS ic oc h w kH kW]
  | .convBiasSgd bN lrS oc h w e               => toToks e ++ [.convBiasSgd bN lrS oc h w]
  | .bnGammaSgd gN vN es lrS oc h w e          => toToks e ++ [.bnGammaSgd gN vN es lrS oc h w]
  | .bnBetaSgd bN lrS oc h w e                 => toToks e ++ [.bnBetaSgd bN lrS oc h w]
  | .layerScaleChGammaSgd gN xN lrS c h w e    => toToks e ++ [.layerScaleChGammaSgd gN xN lrS c h w]
  | .lnGammaSgd gN xN es lrS n e               => toToks e ++ [.lnGammaSgd gN xN es lrS n]
  | .lnBetaSgd bN lrS n e                      => toToks e ++ [.lnBetaSgd bN lrS n]
  | .veclnGammaSgd gN xN es lrS N D e          => toToks e ++ [.veclnGammaSgd gN xN es lrS N D]
  | .patchEmbedWeightSgd wN xN lrS ic H W P N D e => toToks e ++ [.patchEmbedWeightSgd wN xN lrS ic H W P N D]
  | .reluF n e       => toToks e ++ [.reluF n]
  | .selectPos x n e => toToks e ++ [.selectPos x n]
  | .relu6F n e      => toToks e ++ [.relu6F n]
  | .selectMid x n e => toToks e ++ [.selectMid x n]
  | .convertF n e    => toToks e ++ [.convertF n]
  | .flatConvF w b ic oc h w' kH kW e => toToks e ++ [.flatConvF w b ic oc h w' kH kW]
  | .flatConvFBf16 w b ic oc h w' kH kW e => toToks e ++ [.flatConvFBf16 w b ic oc h w' kH kW]
  | .maxPoolF c h w e => toToks e ++ [.maxPoolF c h w]
  | .convBack w ic oc h w' kH kW e => toToks e ++ [.convBack w ic oc h w' kH kW]
  | .maxPoolBack x c h w e => toToks e ++ [.maxPoolBack x c h w]
  | .bnF g b eps n e => toToks e ++ [.bnF g b eps n]
  | .bnBack g x eps n e => toToks e ++ [.bnBack g x eps n]
  | .addV n a b      => toToks a ++ toToks b ++ [.addV n]
  | .gapF c h w e    => toToks e ++ [.gapF c h w]
  | .gapBack c h w e => toToks e ++ [.gapBack c h w]
  | .broadcastBack c h w e => toToks e ++ [.broadcastBack c h w]
  | .flatConvStridedF w b ic oc h w' kH kW e => toToks e ++ [.flatConvStridedF w b ic oc h w' kH kW]
  | .flatConvStridedXlaF w b ic oc h w' kH kW e => toToks e ++ [.flatConvStridedXlaF w b ic oc h w' kH kW]
  | .convStridedBack w ic oc h w' kH kW e => toToks e ++ [.convStridedBack w ic oc h w' kH kW]
  | .convStridedWeightSgd xN wN lrS ic oc h w' kH kW e => toToks e ++ [.convStridedWeightSgd xN wN lrS ic oc h w' kH kW]
  | .depthwiseWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseWeightSgd xN wN lrS c h w' kH kW]
  | .depthwiseStridedWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseStridedWeightSgd xN wN lrS c h w' kH kW]
  | .flatConvStride4F w b ic oc h w' kH kW e => toToks e ++ [.flatConvStride4F w b ic oc h w' kH kW]
  | .bnPerChannelF g b eps oc h w e => toToks e ++ [.bnPerChannelF g b eps oc h w]
  | .bnPerChannelBack g x eps oc h w e => toToks e ++ [.bnPerChannelBack g x eps oc h w]
  | .bnPerChannelEvalF g b mu var eps oc h w e => toToks e ++ [.bnPerChannelEvalF g b mu var eps oc h w]
  | .weightGrad x m n e => toToks e ++ [.weightGrad x m n]
  | .biasGrad n e => toToks e ++ [.biasGrad n]
  | .convWeightGrad x ic oc h w' kH kW e => toToks e ++ [.convWeightGrad x ic oc h w' kH kW]
  | .convBiasGrad ic oc h w' kH kW e => toToks e ++ [.convBiasGrad ic oc h w' kH kW]
  | .convStridedWeightGrad x ic oc h w' kH kW e => toToks e ++ [.convStridedWeightGrad x ic oc h w' kH kW]
  | .bnGammaGrad v eps oc h w' e => toToks e ++ [.bnGammaGrad v eps oc h w']
  | .bnBetaGrad oc h w' e => toToks e ++ [.bnBetaGrad oc h w']
  | .adamMNextF m b1 ob1 ds e => toToks e ++ [.adamMNextF m b1 ob1 ds]
  | .adamVNextF v b2 ob2 ds e => toToks e ++ [.adamVNextF v b2 ob2 ds]
  | .adamWParamF θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd ds e =>
      toToks e ++ [.adamWParamF θ m v b1 ob1 b2 ob2 bc1 bc2 lr eps wd ds]
  | .sgdParamF θ lr ds e => toToks e ++ [.sgdParamF θ lr ds]
  | .momVNextF v mu ds e => toToks e ++ [.momVNextF v mu ds]
  | .momParamF θ v mu lr ds e => toToks e ++ [.momParamF θ v mu lr ds]
  | .rmsBufNextF sq buf rho orho mu eps ds e =>
      toToks e ++ [.rmsBufNextF sq buf rho orho mu eps ds]
  -- ⚠ Both push LEFT then RIGHT, so `parseStack` pops right-then-left (`.addV`'s shape). The
  -- LEFT child is the scalar in both cases — the accumulator, and the summed global total.
  | .gradSumSqAccF ds acc e  => toToks acc ++ toToks e ++ [.gradSumSqAccF ds]
  | .clipScaleF cS eS ds s e => toToks s ++ toToks e ++ [.clipScaleF cS eS ds]
  | .lambDirF a b c d e' f g h i j k ds x =>
      toToks x ++ [.lambDirF a b c d e' f g h i j k ds]
  | .lambScaleF ds s e       => toToks s ++ toToks e ++ [.lambScaleF ds]
  | .depthwiseF w b c h w' kH kW e => toToks e ++ [.depthwiseF w b c h w' kH kW]
  | .depthwiseBack w c h w' kH kW e => toToks e ++ [.depthwiseBack w c h w' kH kW]
  | .depthwiseStridedF w b c h w' kH kW e => toToks e ++ [.depthwiseStridedF w b c h w' kH kW]
  | .depthwiseStridedXlaF w b c h w' kH kW e => toToks e ++ [.depthwiseStridedXlaF w b c h w' kH kW]
  | .depthwiseStridedBack w c h w' kH kW e => toToks e ++ [.depthwiseStridedBack w c h w' kH kW]
  | .swishF n e      => toToks e ++ [.swishF n]
  | .swishBack x n e => toToks e ++ [.swishBack x n]
  | .sigmoidF n e    => toToks e ++ [.sigmoidF n]
  | .sigmoidBack x n e => toToks e ++ [.sigmoidBack x n]
  | .geluF n e       => toToks e ++ [.geluF n]
  | .geluBack x n e  => toToks e ++ [.geluBack x n]
  | .layerScaleF γN n e => toToks e ++ [.layerScaleF γN n]
  | .layerScaleChF γN c h w e => toToks e ++ [.layerScaleChF γN c h w]
  | .softmaxRowF m n e    => toToks e ++ [.softmaxRowF m n]
  | .softmaxRowBack x m n e => toToks e ++ [.softmaxRowBack x m n]
  | .matmulF m k n a b    => toToks a ++ toToks b ++ [.matmulF m k n]
  | .transposeF m n e     => toToks e ++ [.transposeF m n]
  | .scaleF s n e         => toToks e ++ [.scaleF s n]
  | .lnRowF g b eps m n e => toToks e ++ [.lnRowF g b eps m n]
  | .lnRowBack g x eps m n e => toToks e ++ [.lnRowBack g x eps m n]
  | .denseRowF w b N a c e => toToks e ++ [.denseRowF w b N a c]
  | .denseRowBack w N a c e => toToks e ++ [.denseRowBack w N a c]
  | .patchEmbedF w b cls pos ic H W P N D e => toToks e ++ [.patchEmbedF w b cls pos ic H W P N D]
  | .clsSliceF N D e      => toToks e ++ [.clsSliceF N D]
  | .clsPadF N D e        => toToks e ++ [.clsPadF N D]
  | .headSliceF N heads d hIdx e => toToks e ++ [.headSliceF N heads d hIdx]
  | .headPadF N heads d hIdx e   => toToks e ++ [.headPadF N heads d hIdx]
  | .rowScaleF g m n e    => toToks e ++ [.rowScaleF g m n]
  | .rowBiasF b m n e     => toToks e ++ [.rowBiasF b m n]
  | .batched tag names info e   => toToks e ++ [.batched tag names info]
  | .batched2 tag names info a b => toToks a ++ toToks b ++ [.batched2 tag names info]

/-- **Stride-2 weight-gradient window geometry — odd AND even kernels.** Returns
    `(up, ext, lo, hi)` for one spatial axis: `up` is the trailing zero row of the
    decimate-backward upsample, `ext` the resulting cotangent extent, and `[lo, hi]` the
    correlation's padding.

    The strided weight grad zero-upsamples the cotangent onto the stride-1 grid (the `decimate`
    backward) and then correlates the saved input against it VALID-style, so the result is `kH×kW`.
    For an **odd** kernel the upsample carries a trailing zero row (`up = 1`, extent `2s`) and the
    correlation pads symmetrically by `p = (k−1)/2`. That is the committed spelling for the 1×1, 3×3
    and 7×7 kernels every other net uses, and it is reproduced here **byte-identically** — the odd
    branch is exactly the old inline formula.

    For an **even** kernel `p = (k−1)/2` floors to `k/2 − 1`, so a symmetric pad emits a result one
    short of `k`. Measured at `k = 2`, input 8×8 → output 4×4: it declared `2x3x2x2` against a
    convolution yielding `1x1` — **type-invalid MLIR**, which is why ConvNeXt's 2×2/s2 downsample
    weight grad was hand-written (`ConvNeXtRender.downWGrad`) rather than using this op. The fix is
    the same asymmetry `convStridedBack` already applies on the input-VJP: drop the trailing zero
    row (extent `2s−1`) and pad `[p, k−2−p]`.

    Output width is `ext_padded − ext + 1 = (2s + lo + hi) − ext + 1`:
    `k=1 → (1, 2s, 0, 0) ⇒ 1`; `k=3 → (1, 2s, 1, 1) ⇒ 3`; `k=7 → (1, 2s, 3, 3) ⇒ 7`;
    `k=2 → (0, 2s−1, 0, 0) ⇒ 2`; `k=4 → (0, 2s−1, 1, 1) ⇒ 4`. -/
private def sWGradGeom (k s : Nat) : Nat × Nat × Nat × Nat :=
  let p := (k - 1) / 2
  if k % 2 == 1 then (1, 2 * s, p, p) else (0, 2 * s - 1, p, k - 2 - p)

/-- Render one token: pop its operands' result-names off the stack, emit its
    StableHLO line(s), push its fresh result name. The per-op StableHLO *syntax*
    here is the audited lexical boundary (validated by `iree-compile` + GPU run);
    the *structure* it consumes is the proven-faithful token stream. -/
def emitTok (B : Nat) : Tok → List String → StateM Nat (String × List String)
  | .operand nm _, st => pure ("", nm :: st)
  | .dotIn w m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [0], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [m,n]}) -> {ty [B,n]}\n", o :: st)
  -- The ONLY emit shape that reaches tensor cores: both operands bf16, result f32.
  -- The f32 result type IS the "fp32 accumulate" — it is not a convert of a bf16 product.
  | .dotInBf16 w m n, r :: st => do
      let a ← fresh; let bw ← fresh; let o ← fresh
      pure (s!"    {a} = stablehlo.convert {r} : ({ty [B,m]}) -> {tyBf16 [B,m]}\n" ++
            s!"    {bw} = stablehlo.convert {w} : ({ty [m,n]}) -> {tyBf16 [m,n]}\n" ++
            s!"    {o} = stablehlo.dot_general {a}, {bw}, contracting_dims = [1] x [0], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({tyBf16 [B,m]}, {tyBf16 [m,n]}) -> {ty [B,n]}\n",
            o :: st)
  | .dotOut w m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [1], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,n]}, {ty [m,n]}) -> {ty [B,m]}\n", o :: st)
  | .addBcast b n, r :: st => do
      let bb ← fresh; let o ← fresh
      pure (s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [n]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.add {r}, {bb} : {ty [B,n]}\n", o :: st)
  | .expe n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.exponential {r} : {ty [B,n]}\n", o :: st)
  | .softmaxDiv n, r :: st => do
      let z ← fresh; let s ← fresh; let sb ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {s} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.divide {r}, {sb} : {ty [B,n]}\n", o :: st)
  | .sub n, b :: a :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.subtract {a}, {b} : {ty [B,n]}\n", o :: st)
  | .weightSgd xN wN lrS m n, r :: st => do
      let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (s!"    {dW} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [B,n]}) -> {ty [m,n]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [m,n]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [m,n]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [m,n]}\n", o :: st)
  | .biasSgd bN lrS n, r :: st => do
      let z ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dB} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,n]}, tensor<f32>) -> {ty [n]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [n]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [n]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [n]}\n", o :: st)
  | .convWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- conv weight grad (transpose trick) then SGD: reshape flat acts/cotangent to
      -- 4-D, transpose batch↔feature, convolve (batch as contraction), transpose back
      -- to [oc,ic,kH,kW], then θ' = θ − lr·dW. Same op text as `CnnRender.convWGrad`+`sgd`.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .convBiasSgd bN lrS oc h w, r :: st => do
      -- conv bias grad (reduce over batch+spatial [0,2,3]) then SGD. Same op text as
      -- `CnnRender.convBiasGrad`+`sgd`.
      let dr ← fresh; let z ← fresh; let g ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {g} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sB} = stablehlo.multiply {g}, {lB} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
  | .bnGammaSgd gN vN epsStr lrS oc h w, r :: st => do
      -- BN per-channel γ grad: recompute x̂ from the saved conv output {vN} (reduce μ/var
      -- over spatial [2,3]), dγ_c = Σ_{b,h,w} dy·x̂, then SGD. Same op text as the dγ half
      -- of `CnnRender.bnParamGradPC`+`sgd`.
      let z ← fresh; let xr ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let dyr ← fresh; let p ← fresh; let dg ← fresh
      let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {dyr}, {xhat} : {ty [B,oc,h,w]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [oc]}\n", o :: st)
  | .bnBetaSgd bN lrS oc h w, r :: st => do
      -- BN per-channel β grad: dβ_c = Σ_{b,h,w} dy, then SGD (β grad needs no x̂).
      let z ← fresh; let dyr ← fresh; let db ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {db} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sB} = stablehlo.multiply {db}, {lB} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
  | .layerScaleChGammaSgd gN xN lrS c h w, r :: st => do
      -- per-channel layer-scale γ grad: dγ_c = reduce[0,2,3](x⊙dy), then SGD (`lsGradCh` + wrap).
      let z ← fresh; let xr ← fresh; let dr ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {xr}, {dr} : {ty [B,c,h,w]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [c]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [c]}\n", o :: st)
  | .lnGammaSgd gN xN epsStr lrS n, r :: st => do
      -- scalar-LN γ grad: recompute x̂ from the saved LN input {xN} (μ/var over [1]),
      -- dγ = Σ_{b,k} dy·x̂ → tensor<f32>, reshape to the Vec-1 param, SGD (`lnParamGrad` dγ half + wrap).
      let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
      let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
      let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {p} = stablehlo.multiply {r}, {xh} : {ty [B,n]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : tensor<f32>\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : tensor<f32>\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : tensor<f32>\n", o :: st)
  | .lnBetaSgd bN lrS n, r :: st => do
      -- scalar-LN β grad: dβ = Σ_{b,k} dy → tensor<f32> (rank-0, matches the scalar-LN bnF param), SGD.
      let z ← fresh; let db ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {db} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : tensor<f32>\n" ++
        s!"    {sB} = stablehlo.multiply {db}, {lB} : tensor<f32>\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : tensor<f32>\n", o :: st)
  | .veclnGammaSgd gN xN epsStr lrS N D, r :: st => do
      -- vector-[D] LN γ grad: recompute x̂ from the saved LN input {xN} (μ/var over [2], per token),
      -- dγ = Σ_{b,n} dy·x̂ → tensor<Dxf32> (reduce over [0,1], KEEP D), SGD (`lnParamGrad` dγ half + wrap).
      -- {xN}/{r} arrive flat [B,N*D] (the SHlo thread convention) → reshape both to [B,N,D] first.
      let x3 ← fresh; let d3 ← fresh
      let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
      let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
      let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {x3} = stablehlo.reshape {xN} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
        s!"    {d3} = stablehlo.reshape {r} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{D}.0> : {ty [B,N,D]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,N,D]}\n" ++
        s!"    {smr} = stablehlo.reduce({x3} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,N,D]}\n" ++
        s!"    {xc} = stablehlo.subtract {x3}, {mu} : {ty [B,N,D]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,N,D]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,N,D]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,N,D]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,N,D]}\n" ++
        s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,N,D]}\n" ++
        s!"    {p} = stablehlo.multiply {d3}, {xh} : {ty [B,N,D]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,D]}, tensor<f32>) -> {ty [D]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [D]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [D]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [D]}\n", o :: st)
  | .patchEmbedWeightSgd wN xN lrS ic H W P N D, r :: st => do
      -- patch-embed (16×16/s16) conv WEIGHT grad: slice patch tokens [1..N] from the embed cotangent
      -- {r} [B,(N+1)*D] (drop CLS row 0), reshape→[B,ph,pw,D]→transpose→[B,D,ph,pw], dilate interior P-1,
      -- valid conv with the saved image {xN} [B,ic,H,W] → dW [D,ic,P,P], SGD: W − lr·dW.
      let ph := H / P; let pw := W / P
      let dilH := H - (P - 1); let dilW := W - (P - 1)
      let zc ← fresh; let dtr ← fresh; let dsl ← fresh; let drs ← fresh; let dy3 ← fresh
      let u ← fresh; let xt ← fresh; let dt ← fresh; let raw ← fresh; let dw ← fresh
      let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {zc} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dtr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
        s!"    {dsl} = stablehlo.slice {dtr} [0:{B}, 1:{N+1}, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,N,D]}\n" ++
        s!"    {drs} = stablehlo.reshape {dsl} : ({ty [B,N,D]}) -> {ty [B,ph,pw,D]}\n" ++
        s!"    {dy3} = stablehlo.transpose {drs}, dims = [0, 3, 1, 2] : ({ty [B,ph,pw,D]}) -> {ty [B,D,ph,pw]}\n" ++
        s!"    {u} = stablehlo.pad {dy3}, {zc}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {P-1}, {P-1}] : ({ty [B,D,ph,pw]}, tensor<f32>) -> {ty [B,D,dilH,dilW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xN}, dims = [1, 0, 2, 3] : ({ty [B,ic,H,W]}) -> {ty [ic,B,H,W]}\n" ++
        s!"    {dt} = stablehlo.transpose {u}, dims = [1, 0, 2, 3] : ({ty [B,D,dilH,dilW]}) -> {ty [D,B,dilH,dilW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,H,W]}, {ty [D,B,dilH,dilW]}) -> {ty [ic,D,P,P]}\n" ++
        s!"    {dw} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,D,P,P]}) -> {ty [D,ic,P,P]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [D,ic,P,P]}\n" ++
        s!"    {sW} = stablehlo.multiply {dw}, {lW} : {ty [D,ic,P,P]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [D,ic,P,P]}\n", o :: st)
  | .reluF n, r :: st => do
      let z ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.maximum {r}, {z} : {ty [B,n]}\n", o :: st)
  -- The round node: down to bf16 and straight back. Two converts, not one, because
  -- `den` is `ℝ → ℝ` — the VALUE stays f32 and only its precision is degraded, which
  -- is what "round to bf16" means as a function on reals.
  --
  -- ⚠⚠ **MEASURED 2026-08-01 ON ares: XLA DELETES THIS PAIR, SO THIS EMIT IS A NO-OP
  -- ON HARDWARE.** jax 0.10.2 / CUDA 12.9, `.astype(bf16).astype(f32)` under `jit`:
  -- eager rounds 1.7640524 → 1.765625, but the jitted result is 1.7640524 unchanged and
  -- the optimized HLO contains no `convert` at all — the algebraic simplifier treats the
  -- round trip as removable. So a graph carrying this node computes in FULL f32: no
  -- speedup and, worse, not even the bf16 numerics.
  --
  -- The `den` equation is still correct and the ties built on it still hold — what is
  -- refuted is this EMIT STRATEGY, not the op. To make bf16 real the value has to stay
  -- bf16 ACROSS an operation, i.e. a `dot_general` whose operands are bf16-typed with
  -- `preferred_element_type = f32`. That changes the value's type and so cannot be a
  -- `SHlo n → SHlo n` node; it is the rung-2 emitter change in
  -- planning/bf16_renderer.md. Keep this node — it is the proof-side round and the
  -- depth > 1 ingredient — but do NOT read a graph containing it as running bf16.
  | .convertF n, r :: st => do
      let b ← fresh; let o ← fresh
      pure (s!"    {b} = stablehlo.convert {r} : ({ty [B,n]}) -> {tyBf16 [B,n]}\n" ++
            s!"    {o} = stablehlo.convert {b} : ({tyBf16 [B,n]}) -> {ty [B,n]}\n", o :: st)
  | .selectPos x n, r :: st => do
      let z ← fresh; let msk ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
        s!"    {msk} = stablehlo.compare GT, {x}, {z} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
        s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,n]}, {ty [B,n]}\n", o :: st)
  | .relu6F n, r :: st => do
      -- ReLU6 forward: clamp to [0,6] as `min(max(x,0),6)` (matches `relu6`'s def).
      let z ← fresh; let six ← fresh; let mx ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
            s!"    {six} = stablehlo.constant dense<6.0> : {ty [B,n]}\n" ++
            s!"    {mx} = stablehlo.maximum {r}, {z} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.minimum {mx}, {six} : {ty [B,n]}\n", o :: st)
  | .selectMid x n, r :: st => do
      -- ReLU6 backward mask: route dy where `0 < x < 6`, else 0 (the two-sided kink).
      let z ← fresh; let six ← fresh; let g0 ← fresh; let l6 ← fresh; let msk ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
        s!"    {six} = stablehlo.constant dense<6.0> : {ty [B,n]}\n" ++
        s!"    {g0} = stablehlo.compare GT, {x}, {z} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
        s!"    {l6} = stablehlo.compare LT, {x}, {six} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
        s!"    {msk} = stablehlo.and {g0}, {l6} : {tyI1 [B,n]}\n" ++
        s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,n]}, {ty [B,n]}\n", o :: st)
  | .flatConvF w b ic oc h w' kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*h*w']}) -> {ty [B,ic,h,w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,h,w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  -- ⚠ The convolution's RESULT is bf16-typed. An f32-typed result here reads as the same
  -- computation and compiles to pure f32 — see the constructor's note. Do not "simplify".
  | .flatConvFBf16 w b ic oc h w' kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let xb ← fresh; let wb ← fresh; let cv ← fresh
      let cf ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*h*w']}) -> {ty [B,ic,h,w']}\n" ++
        s!"    {xb} = stablehlo.convert {xn} : ({ty [B,ic,h,w']}) -> {tyBf16 [B,ic,h,w']}\n" ++
        s!"    {wb} = stablehlo.convert {w} : ({ty [oc,ic,kH,kW]}) -> {tyBf16 [oc,ic,kH,kW]}\n" ++
        s!"    {cv} = stablehlo.convolution({xb}, {wb})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({tyBf16 [B,ic,h,w']}, {tyBf16 [oc,ic,kH,kW]}) -> {tyBf16 [B,oc,h,w']}\n" ++
        s!"    {cf} = stablehlo.convert {cv} : ({tyBf16 [B,oc,h,w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cf}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .maxPoolF c h w, r :: st => do
      let xn ← fresh; let ninf ← fresh; let p ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
        s!"    {p} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
        "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
        "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
        "        stablehlo.return %pm : tensor<f32>\n" ++
        "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
        s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {p} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
  | .convBack w ic oc h w' kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let wt ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {wt} = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {wr} = stablehlo.reverse {wt}, dims = [2, 3] : {ty [ic,oc,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({dn}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,h,w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,h,w']}) -> {ty [B, ic*h*w']}\n", o :: st)
  | .maxPoolBack xN c h w, r :: st => do
      let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
        "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
        "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        "        stablehlo.return %sge : tensor<i1>\n" ++
        "    }, " ++ "{\n" ++
        "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
        "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
        "        stablehlo.return %ss : tensor<f32>\n" ++
        "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
        s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
  | .bnF gN bN epsStr n, r :: st => do
      -- per-example BatchNorm forward `γ·(x−μ)·istd + β` (reduce μ/var over [1])
      let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {r}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.add {gx}, {bb} : {ty [B,n]}\n", o :: st)
  | .bnBack gN xN epsStr n, r :: st => do
      -- BN input-VJP: recompute x̂/istd from saved input {xN}, then the
      -- consolidated three-term `(istd/N)·(N·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`, dx̂ = γ·dy.
      let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {r} : {ty [B,n]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,n]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,n]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,n]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,n]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,n]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.multiply {sN}, {i2} : {ty [B,n]}\n", o :: st)
  | .addV n, b :: a :: st => do
      -- residual fan-in: dy of the two operands summed (`F(x) + skip`)
      let o ← fresh
      pure (s!"    {o} = stablehlo.add {a}, {b} : {ty [B,n]}\n", o :: st)
  | .gapF c h w, r :: st => do
      -- global average pool: reshape to [B,c,h,w], reduce-add over the spatial
      -- axes [2,3], divide by h·w. Denotes `globalAvgPoolFlat` (mean over H×W).
      let xn ← fresh; let z ← fresh; let sm ← fresh; let nf ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {sm} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
        s!"    {o} = stablehlo.divide {sm}, {nf} : {ty [B,c]}\n", o :: st)
  | .gapBack c h w, r :: st => do
      -- GAP backward (VJP): divide the per-channel cotangent by h·w, broadcast
      -- it back over the H×W spatial grid, reshape to flat. Reverse of `.gapF`.
      -- Denotes `globalAvgPoolFlat`'s VJP backward `dy[chan idx] / (h·w)`.
      -- (Text emission best-effort/unverified-vs-IREE; the `den` is proven.)
      let nf ← fresh; let dv ← fresh; let bb ← fresh; let o ← fresh
      pure (
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
        s!"    {dv} = stablehlo.divide {r}, {nf} : {ty [B,c]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {dv}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {bb} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
  | .broadcastBack c h w, r :: st => do
      -- broadcast backward (VJP) = sum over H×W per channel (adjoint of broadcast):
      -- reshape to [B,c,h,w], reduce-add over spatial axes [2,3] → [B,c]. No divide.
      -- (Text emission best-effort/unverified-vs-IREE; the `den` is proven.)
      let xn ← fresh; let z ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {o} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n", o :: st)
  | .flatConvStride4F w b ic oc h w' kH kW, r :: st => do
      -- stride-4 patchify conv (the ConvNeXt 4×4/s4 stem): reshape, convolution
      -- with window_strides=[4,4], +bias. The denotation reads the SAME conv
      -- (pad (k-1)/2) at the offset-1 positions 4i+1 (decimate ∘ decimateOdd),
      -- so the emitted pad is one less: (k-1)/2 − 1 — for the 4×4 stem pad 0,
      -- the left-aligned window x[4i..4i+3] of the paper's pad-0 Conv2d(4, s=4).
      let pH := (kH - 1) / 2 - 1; let pW := (kW - 1) / 2 - 1
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*(2*h))*(2*(2*w'))]}) -> {ty [B,ic,2*(2*h),2*(2*w')]}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [4, 4], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*(2*h),2*(2*w')]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .flatConvStridedF w b ic oc h w' kH kW, r :: st => do
      -- stride-2 SAME conv: reshape, convolution with window_strides=[2,2], +bias
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w')]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*h,2*w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .flatConvStridedXlaF w b ic oc h w' kH kW, r :: st => do
      -- stride-2 XLA-`SAME` conv: identical to `flatConvStridedF` above but `pad = [p-1, p]`,
      -- the asymmetric split XLA uses at an even input (k=3 -> (0,1)).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w')]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH-1}, {pH}], [{pW-1}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*h,2*w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .convStridedBack w ic oc h w' kH kW, r :: st => do
      -- stride-2 conv input-VJP: zero-upsample dy (pad with interior=1, high=1) to
      -- the 2h×2w grid, then the reversed-kernel stride-1 conv (= decimate.back ▸ conv.back).
      -- Transpose-conv pad: low = k−1−p, high = p (p = the forward pad (k−1)/2) —
      -- symmetric (k−1)/2 for odd k (3×3 MNV2/r34, unchanged), [[1,0]] for the
      -- even 2×2 ConvNeXt downsample (the left-aligned forward window).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wt ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w']}, tensor<f32>) -> {ty [B,oc,2*h,2*w']}\n" ++
        s!"    {wt} = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {wr} = stablehlo.reverse {wt}, dims = [2, 3] : {ty [ic,oc,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{kH - 1 - pH}, {pH}], [{kW - 1 - pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,2*h,2*w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w']}) -> {ty [B, ic*(2*h)*(2*w')]}\n", o :: st)
  | .convStridedWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- strided (stride-2) conv weight grad then SGD: reshape x to the 2h×2w grid and dy
      -- to h×w, zero-upsample dy (interior+high=1 → 2h×2w, the decimate-backward), then the
      -- SAME transpose-trick stride-1 weight-grad conv as `convWeightSgd` on the 2h×2w grid →
      -- [oc,ic,kH,kW], then θ' = θ − lr·dW. Same op text as `TestResnet34Train.convWGradStrided`.
      -- `sWGradGeom` is the odd/even split; odd reproduces the old inline formula byte-for-byte.
      let (upH, extH, loH, hiH) := sWGradGeom kH h
      let (upW, extW, loW, hiW) := sWGradGeom kW w
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .depthwiseWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- depthwise (grouped) weight grad: per-channel transpose-trick conv with
      -- `batch_group_count = c` (each output kernel reads only its own channel) → [1,c,kH,kW],
      -- reshape to the depthwise kernel layout [c,1,kH,kW], then θ' = θ − lr·dW. Same op text as
      -- `TestMobilenetV2Train.dwconvWGrad` + sgd.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
  | .depthwiseStridedWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- strided depthwise weight grad: reshape x to the 2h×2w grid and dy to h×w, zero-upsample dy
      -- (interior+high=1 → 2h×2w, the decimate-backward), then the SAME per-channel transpose-trick
      -- weight-grad conv (`batch_group_count = c`) on the 2h×2w grid → [1,c,kH,kW], reshape to the
      -- depthwise layout [c,1,kH,kW], then θ' = θ − lr·dW. Same op text as
      -- `TestMobilenetV2Train.dwconvWGradStrided` + sgd.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
  | .bnPerChannelF gN bN epsStr oc h w, r :: st => do
      -- PER-CHANNEL BatchNorm forward: reshape to [B,oc,h,w], reduce μ/var over the
      -- spatial axes [2,3] (per channel), normalize, then γ·x̂+β with rank-1 γ/β
      -- (broadcast dims=[1]). Mirrors `bnF` but 4-D + per-channel.
      let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  -- ══ PARAM GRADIENTS: the `*Sgd` emitters with the `constant lr / multiply / subtract`
  --    tail cut off. Same gradient text, so `θ − lr·grad` reproduces the SGD op exactly. ══
  | .weightGrad xN m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [B,n]}) -> {ty [m,n]}\n", o :: st)
  | .biasGrad n, r :: st => do
      let z ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,n]}, tensor<f32>) -> {ty [n]}\n", o :: st)
  | .convWeightGrad xN ic oc h w kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh; let raw ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n",
        o :: st)
  | .convBiasGrad _ic oc h w _kH _kW, r :: st => do
      let dr ← fresh; let z ← fresh; let o ← fresh
      pure (
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {o} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n",
        o :: st)
  | .convStridedWeightGrad xN ic oc h w kH kW, r :: st => do
      -- `sWGradGeom` is the odd/even split; at odd kernels it reproduces the old inline
      -- `pH = (kH-1)/2` formula byte-for-byte.
      let (upH, extH, loH, hiH) := sWGradGeom kH h
      let (upW, extW, loW, hiW) := sWGradGeom kW w
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n",
        o :: st)
  | .bnGammaGrad vN epsStr oc h w, r :: st => do
      -- dγ_c = Σ_{b,h,w} dy·x̂, with x̂ recomputed from the saved BN input {vN} (μ/var over
      -- the spatial axes [2,3] — per-channel, per-example, as `bnPerChannelF` normalises).
      let z ← fresh; let xr ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let dyr ← fresh; let p ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {dyr}, {xhat} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n",
        o :: st)
  | .bnBetaGrad oc h w, r :: st => do
      -- dβ_c = Σ_{b,h,w} dy — needs no x̂.
      let z ← fresh; let dyr ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n",
        o :: st)
  -- ══ ADAMW: op-for-op `Proofs.adamMNext` / `adamVNext` / `adamWParam`, matching the
  --    hand-written `ViTRender.emitAdamV` block it replaces. Scalar hyperparameters are
  --    `tensor<f32>` function args, broadcast to the param shape `ds`. ══
  | .adamMNextF mN b1N ob1N ds, r :: st => do
      let T := ty ds
      let bb ← fresh; let ob ← fresh; let ms ← fresh; let mg ← fresh; let o ← fresh
      pure (
        s!"    {bb} = stablehlo.broadcast_in_dim {b1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob} = stablehlo.broadcast_in_dim {ob1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ms} = stablehlo.multiply {bb}, {mN} : {T}\n" ++
        s!"    {mg} = stablehlo.multiply {ob}, {r} : {T}\n" ++
        s!"    {o} = stablehlo.add {ms}, {mg} : {T}\n", o :: st)
  | .adamVNextF vN b2N ob2N ds, r :: st => do
      let T := ty ds
      let bb ← fresh; let ob ← fresh; let vs ← fresh; let g2 ← fresh; let vg ← fresh; let o ← fresh
      pure (
        s!"    {bb} = stablehlo.broadcast_in_dim {b2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob} = stablehlo.broadcast_in_dim {ob2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vs} = stablehlo.multiply {bb}, {vN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {vg} = stablehlo.multiply {ob}, {g2} : {T}\n" ++
        s!"    {o} = stablehlo.add {vs}, {vg} : {T}\n", o :: st)
  | .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds, r :: st => do
      let T := ty ds
      -- m' and v' are recomputed here rather than shared with the two moment ops: SHlo is
      -- single-result, so each output is its own node. XLA's CSE folds the duplicates.
      let b1b ← fresh; let ob1b ← fresh; let ms ← fresh; let mg ← fresh; let mn ← fresh
      let b2b ← fresh; let ob2b ← fresh; let vs ← fresh; let g2 ← fresh; let vg ← fresh; let vn ← fresh
      let bc1b ← fresh; let bc2b ← fresh; let mh ← fresh; let vh ← fresh
      let lrb ← fresh; let epsb ← fresh; let sq ← fresh; let dn ← fresh; let rat ← fresh
      let stp ← fresh; let sub ← fresh; let wdb ← fresh; let wdlr ← fresh; let wdp ← fresh; let o ← fresh
      pure (
        s!"    {b1b} = stablehlo.broadcast_in_dim {b1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob1b} = stablehlo.broadcast_in_dim {ob1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ms} = stablehlo.multiply {b1b}, {mN} : {T}\n" ++
        s!"    {mg} = stablehlo.multiply {ob1b}, {r} : {T}\n" ++
        s!"    {mn} = stablehlo.add {ms}, {mg} : {T}\n" ++
        s!"    {b2b} = stablehlo.broadcast_in_dim {b2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob2b} = stablehlo.broadcast_in_dim {ob2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vs} = stablehlo.multiply {b2b}, {vN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {vg} = stablehlo.multiply {ob2b}, {g2} : {T}\n" ++
        s!"    {vn} = stablehlo.add {vs}, {vg} : {T}\n" ++
        s!"    {bc1b} = stablehlo.broadcast_in_dim {bc1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {bc2b} = stablehlo.broadcast_in_dim {bc2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {mh} = stablehlo.divide {mn}, {bc1b} : {T}\n" ++
        s!"    {vh} = stablehlo.divide {vn}, {bc2b} : {T}\n" ++
        s!"    {lrb} = stablehlo.broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {epsb} = stablehlo.broadcast_in_dim {epsN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {sq} = stablehlo.sqrt {vh} : {T}\n" ++
        s!"    {dn} = stablehlo.add {sq}, {epsb} : {T}\n" ++
        s!"    {rat} = stablehlo.divide {mh}, {dn} : {T}\n" ++
        s!"    {stp} = stablehlo.multiply {lrb}, {rat} : {T}\n" ++
        s!"    {sub} = stablehlo.subtract {θN}, {stp} : {T}\n" ++
        s!"    {wdb} = stablehlo.broadcast_in_dim {wdN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {wdlr} = stablehlo.multiply {wdb}, {lrb} : {T}\n" ++
        s!"    {wdp} = stablehlo.multiply {wdlr}, {θN} : {T}\n" ++
        s!"    {o} = stablehlo.subtract {sub}, {wdp} : {T}\n", o :: st)
  -- ══ SGD / NESTEROV (§2i): op-for-op `Proofs.sgdParam` / `momVNext` / `momParam`, matching the
  --    retired `tests/TestCifar8AdamTrain.emit{Sgd,Momentum}` blocks byte-for-byte modulo SSA
  --    freshness. `%lr` / `%mu` are runtime `tensor<f32>` args, broadcast to the param shape. ══
  | .sgdParamF θN lrN ds, r :: st => do
      let T := ty ds
      let lrb ← fresh; let stp ← fresh; let o ← fresh
      pure (
        s!"    {lrb} = stablehlo.broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {stp} = stablehlo.multiply {lrb}, {r} : {T}\n" ++
        s!"    {o} = stablehlo.subtract {θN}, {stp} : {T}\n", o :: st)
  | .momVNextF vN muN ds, r :: st => do
      let T := ty ds
      let mub ← fresh; let vg ← fresh; let o ← fresh
      pure (
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vg} = stablehlo.multiply {mub}, {vN} : {T}\n" ++
        s!"    {o} = stablehlo.add {vg}, {r} : {T}\n", o :: st)
  | .momParamF θN vN muN lrN ds, r :: st => do
      let T := ty ds
      -- v' is recomputed here rather than shared with `momVNextF`: SHlo is single-result, so each
      -- output is its own node. XLA's CSE folds the duplicate (§2b-bis measured that on R34's
      -- 108 → 36 rsqrt, at no run-time cost).
      let mub ← fresh; let vg ← fresh; let vel ← fresh
      let nv ← fresh; let lk ← fresh; let lrb ← fresh; let stp ← fresh; let o ← fresh
      pure (
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vg} = stablehlo.multiply {mub}, {vN} : {T}\n" ++
        s!"    {vel} = stablehlo.add {vg}, {r} : {T}\n" ++
        s!"    {nv} = stablehlo.multiply {mub}, {vel} : {T}\n" ++
        s!"    {lk} = stablehlo.add {nv}, {r} : {T}\n" ++
        s!"    {lrb} = stablehlo.broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {stp} = stablehlo.multiply {lrb}, {lk} : {T}\n" ++
        s!"    {o} = stablehlo.subtract {θN}, {stp} : {T}\n", o :: st)
  -- ══ RMSPROP (TensorFlow flavour): op-for-op `Proofs.rmsBufNext`, matching the JAX reference's
  --    `MOMENTUM * b + g / jnp.sqrt(s + EPS)` where `s = RHO*s + (1-RHO)*g*g`. `%rho`/`%orho`/
  --    `%mu`/`%eps` are runtime `tensor<f32>` args, broadcast to the param shape.
  --    ⚠ `{ep}` is added to the mean-square BEFORE the `sqrt`, not to the root after it. That one
  --    line is the entire difference from textbook RMSProp and it is a different optimizer —
  --    `Proofs.rmsBufNext_eps_placement_at_zero` states the gap (1/√ε against 1/ε).
  --    s' is recomputed here rather than shared with `adamVNextF`: SHlo is single-result, so each
  --    output is its own node. XLA's CSE folds the duplicate (§2b-bis measured that on R34's
  --    108 → 36 rsqrt, at no run-time cost). ══
  | .rmsBufNextF sqN bufN rhoN orhoN muN epsN ds, r :: st => do
      let T := ty ds
      let rhob ← fresh; let orhob ← fresh; let ss ← fresh; let g2 ← fresh; let sg ← fresh
      let sn ← fresh; let ep ← fresh; let da ← fresh; let dn ← fresh; let nrm ← fresh
      let mub ← fresh; let bs ← fresh; let o ← fresh
      pure (
        s!"    {rhob} = stablehlo.broadcast_in_dim {rhoN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {orhob} = stablehlo.broadcast_in_dim {orhoN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ss} = stablehlo.multiply {rhob}, {sqN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {sg} = stablehlo.multiply {orhob}, {g2} : {T}\n" ++
        s!"    {sn} = stablehlo.add {ss}, {sg} : {T}\n" ++
        s!"    {ep} = stablehlo.broadcast_in_dim {epsN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {da} = stablehlo.add {sn}, {ep} : {T}\n" ++
        s!"    {dn} = stablehlo.sqrt {da} : {T}\n" ++
        s!"    {nrm} = stablehlo.divide {r}, {dn} : {T}\n" ++
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {bs} = stablehlo.multiply {mub}, {bufN} : {T}\n" ++
        s!"    {o} = stablehlo.add {bs}, {nrm} : {T}\n", o :: st)
  -- ══ GLOBAL-NORM GRADIENT CLIPPING: op-for-op the reference's two lines
  --      gn    = sqrt(sum(jnp.sum(g*g) for g in tree.leaves(grads)))
  --      grads = tree.map(lambda g: g * minimum(1.0, CLIP/(gn + 1e-6)), grads)
  --    `jax/Jax/Codegen.lean:2262`. All four emit at the PARAMETER shape `ty ds`, not at
  --    `ty [B,n]` — the clip runs on parameter gradients, after the batch has been contracted. ══
  | .gradSumSqAccF ds, g :: acc :: st => do
      -- One leaf's `jnp.sum(g*g)`, reduced over EVERY axis to rank 0, added to the running total.
      -- The dims list is `List.range ds.length` rather than a literal, so it is right at rank 1 and
      -- rank 4 alike; ViT and ConvNeXt have no rank-0 parameters, so it is never empty
      -- (`vitParamSig`, ConvNeXt's `allParams` — both checked, minimum rank 1).
      -- ⚠ Emits at the PARAMETER shape `ty ds`, not `ty [B,n]`: the clip runs on parameter
      -- gradients, after the batch has been contracted.
      let T := ty ds
      let dims := String.intercalate ", " ((List.range ds.length).map toString)
      let z ← fresh; let sq ← fresh; let red ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {sq} = stablehlo.multiply {g}, {g} : {T}\n" ++
        s!"    {red} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [{dims}] : ({T}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {o} = stablehlo.add {acc}, {red} : tensor<f32>\n", o :: st)
  | .clipScaleF clipS epsS ds, g :: sN :: st => do
      -- `g * min(1, CLIP/(sqrt(total) + 1e-6))` — the reference's second line.
      -- ⚠ ε is added to the ROOT, not under it: the opposite of `rmsBufNextF`'s TF placement, and
      -- this one follows the reference literally (`CLIP / (gn + 1e-6)`).
      -- ⚠ The `minimum` against 1.0 is not decoration — without it a SMALL gradient is AMPLIFIED
      -- by `c/gn`, which compiles, trains and descends (`Proofs.clipFactor_le_one`).
      -- The broadcast-then-multiply is `adamWParamF`'s `%lr` shape verbatim; it is the
      -- scale-by-a-RUNTIME-scalar the kit lacked (`scaleF` bakes a `constant dense<…>` instead).
      let T := ty ds
      let gn ← fresh; let ep ← fresh; let dn ← fresh; let cc ← fresh
      let rat ← fresh; let one ← fresh; let fac ← fresh; let fb ← fresh; let o ← fresh
      pure (
        s!"    {gn} = stablehlo.sqrt {sN} : tensor<f32>\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsS}> : tensor<f32>\n" ++
        s!"    {dn} = stablehlo.add {gn}, {ep} : tensor<f32>\n" ++
        s!"    {cc} = stablehlo.constant dense<{clipS}> : tensor<f32>\n" ++
        s!"    {rat} = stablehlo.divide {cc}, {dn} : tensor<f32>\n" ++
        s!"    {one} = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
        s!"    {fac} = stablehlo.minimum {one}, {rat} : tensor<f32>\n" ++
        s!"    {fb} = stablehlo.broadcast_in_dim {fac}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {o} = stablehlo.multiply {fb}, {g} : {T}\n", o :: st)
  -- ══ LAMB (`Proofs.Lamb`), RSB-A3's optimizer. `lambDirF` is `adamWParamF`'s block truncated at
  --    the ratio with `wd·θ` ADDED rather than subtracted at the end — the decay is decoupled and
  --    lands INSIDE the direction, hence inside the norm the trust ratio takes. ══
  | .lambDirF θN mN vN b1N ob1N b2N ob2N bc1N bc2N epsN wdN ds, r :: st => do
      let T := ty ds
      let b1b ← fresh; let ob1b ← fresh; let ms ← fresh; let mg ← fresh; let mn ← fresh
      let b2b ← fresh; let ob2b ← fresh; let vs ← fresh; let g2 ← fresh; let vg ← fresh; let vn ← fresh
      let bc1b ← fresh; let bc2b ← fresh; let mh ← fresh; let vh ← fresh
      let epsb ← fresh; let sq ← fresh; let dn ← fresh; let rat ← fresh
      let wdb ← fresh; let wdp ← fresh; let o ← fresh
      pure (
        s!"    {b1b} = stablehlo.broadcast_in_dim {b1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob1b} = stablehlo.broadcast_in_dim {ob1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ms} = stablehlo.multiply {b1b}, {mN} : {T}\n" ++
        s!"    {mg} = stablehlo.multiply {ob1b}, {r} : {T}\n" ++
        s!"    {mn} = stablehlo.add {ms}, {mg} : {T}\n" ++
        s!"    {b2b} = stablehlo.broadcast_in_dim {b2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {ob2b} = stablehlo.broadcast_in_dim {ob2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {vs} = stablehlo.multiply {b2b}, {vN} : {T}\n" ++
        s!"    {g2} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {vg} = stablehlo.multiply {ob2b}, {g2} : {T}\n" ++
        s!"    {vn} = stablehlo.add {vs}, {vg} : {T}\n" ++
        s!"    {bc1b} = stablehlo.broadcast_in_dim {bc1N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {bc2b} = stablehlo.broadcast_in_dim {bc2N}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {mh} = stablehlo.divide {mn}, {bc1b} : {T}\n" ++
        s!"    {vh} = stablehlo.divide {vn}, {bc2b} : {T}\n" ++
        s!"    {epsb} = stablehlo.broadcast_in_dim {epsN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        -- ⚠ ε OUTSIDE the root. `sqrt(vh + eps)` is RMSProp-TF's placement and a different
        -- optimizer; the reference is literal — `mc / (jnp.sqrt(vc) + EPS)`.
        s!"    {sq} = stablehlo.sqrt {vh} : {T}\n" ++
        s!"    {dn} = stablehlo.add {sq}, {epsb} : {T}\n" ++
        s!"    {rat} = stablehlo.divide {mh}, {dn} : {T}\n" ++
        s!"    {wdb} = stablehlo.broadcast_in_dim {wdN}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {wdp} = stablehlo.multiply {wdb}, {θN} : {T}\n" ++
        s!"    {o} = stablehlo.add {rat}, {wdp} : {T}\n", o :: st)
  | .lambScaleF ds, r :: wn2 :: st => do
      -- `trust · r` with `trust = ‖θ‖/‖r‖`, guarded to 1 when either norm vanishes.
      -- ⚠⚠ `‖r‖²` is reduced HERE, from this op's own tensor child, so the trust ratio is
      -- PER PARAMETER TENSOR. That is the whole difference from `clipScaleF` above, whose factor
      -- is one scalar shared across every parameter — the two blocks look alike and differ in the
      -- quantifier (`Proofs.lambScale_not_shared` states it).
      -- ⚠ The guard is not a corner case: the driver inits every BN β and dense bias to 0, so
      -- `wn2 = 0` on those tensors at step 1 and `select` takes the `1.0` branch on a real run.
      let T := ty ds
      let dims := String.intercalate ", " ((List.range ds.length).map toString)
      let z ← fresh; let rsq ← fresh; let rn2 ← fresh
      let wn ← fresh; let rn ← fresh; let rat ← fresh; let one ← fresh
      let okW ← fresh; let okR ← fresh; let ok ← fresh; let tr ← fresh; let tb ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {rsq} = stablehlo.multiply {r}, {r} : {T}\n" ++
        s!"    {rn2} = stablehlo.reduce({rsq} init: {z}) applies stablehlo.add across dimensions = [{dims}] : ({T}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {wn} = stablehlo.sqrt {wn2} : tensor<f32>\n" ++
        s!"    {rn} = stablehlo.sqrt {rn2} : tensor<f32>\n" ++
        s!"    {rat} = stablehlo.divide {wn}, {rn} : tensor<f32>\n" ++
        s!"    {one} = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
        s!"    {okW} = stablehlo.compare GT, {wn2}, {z} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        s!"    {okR} = stablehlo.compare GT, {rn2}, {z} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        s!"    {ok} = stablehlo.and {okW}, {okR} : tensor<i1>\n" ++
        s!"    {tr} = stablehlo.select {ok}, {rat}, {one} : tensor<i1>, tensor<f32>\n" ++
        s!"    {tb} = stablehlo.broadcast_in_dim {tr}, dims = [] : (tensor<f32>) -> {T}\n" ++
        s!"    {o} = stablehlo.multiply {tb}, {r} : {T}\n", o :: st)
  | .bnPerChannelEvalF gN bN muN varN epsStr oc h w, r :: st => do
      -- INFERENCE per-channel BatchNorm: reshape to [B,oc,h,w], then the affine map
      -- γ·(x − μ)·rsqrt(var + ε) + β with μ/var/γ/β all rank-1 `[oc]` graph inputs
      -- (broadcast dims=[1]). No reduce and no normalizer constant — that is the whole
      -- difference from `bnPerChannelF`, and why eval is class-batch-independent.
      let xn ← fresh; let mub ← fresh; let xc ← fresh; let vb ← fresh; let ep ← fresh
      let ve ← fresh; let istd ← fresh; let xhat ← fresh; let gb ← fresh; let bb ← fresh
      let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mub} : {ty [B,oc,h,w]}\n" ++
        s!"    {vb} = stablehlo.broadcast_in_dim {varN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  | .bnPerChannelBack gN xN epsStr oc h w, r :: st => do
      -- PER-CHANNEL BN input-VJP: recompute x̂/istd per channel from saved input {xN},
      -- then the block-diagonal three-term `(istd/m)·(m·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`,
      -- dx̂ = γ·dy, with all Σ reductions over the spatial axes [2,3] (m = h·w).
      let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,oc,h,w]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,oc,h,w]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,oc,h,w]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,oc,h,w]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {o0} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  | .depthwiseF w b c h w' kH kW, r :: st => do
      -- depthwise conv forward: reshape to [B,c,h,w'], grouped `stablehlo.convolution`
      -- (feature_group_count = c, [c,1,kH,kW] kernel — one filter per channel, no
      -- cross-channel mixing), SAME pad, + per-channel bias, reshape back.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,h,w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseBack w c h w' kH kW, r :: st => do
      -- depthwise conv input-VJP: reshape dy, reverse the per-channel filters over the
      -- spatial axes [2,3] (the channel groups are 1×1, so no o↔i transpose), then the
      -- reversed-kernel SAME-pad depthwise conv (feature_group_count = c).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({dn}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,h,w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedF w b c h w' kH kW, r :: st => do
      -- stride-2 depthwise conv: reshape, grouped convolution with window_strides=[2,2]
      -- (feature_group_count = c, [c,1,kH,kW] kernel), SAME pad, + bias. Halves spatial.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w')]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedXlaF w b c h w' kH kW, r :: st => do
      -- stride-2 XLA-`SAME` depthwise: as above with `pad = [p-1, p]`.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w')]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH-1}, {pH}], [{pW-1}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedBack w c h w' kH kW, r :: st => do
      -- stride-2 depthwise input-VJP: zero-upsample dy (pad interior/high=1) back to
      -- 2h×2w', reverse the per-channel filters over [2,3] (no transpose, 1×1 groups),
      -- then the reversed-kernel stride-1 depthwise conv (feature_group_count = c).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w']}, tensor<f32>) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w']}) -> {ty [B, c*(2*h)*(2*w')]}\n", o :: st)
  | .swishF n, r :: st => do
      -- swish forward: y = x · σ(x), σ = logistic (smooth everywhere, no kink/mask).
      let s ← fresh; let o ← fresh
      pure (s!"    {s} = stablehlo.logistic {r} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {s} : {ty [B,n]}\n", o :: st)
  | .swishBack x n, r :: st => do
      -- swish input-VJP: dy ⊙ σ(x)·(1 + x·(1−σ(x))), recomputing σ from the saved
      -- pre-activation {x} (matches `swishScalarDeriv`'s closed form, IRPrint `swishB`).
      let s ← fresh; let one ← fresh; let om ← fresh; let xom ← fresh
      let inr ← fresh; let sp ← fresh; let o ← fresh
      pure (s!"    {s} = stablehlo.logistic {x} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {om} = stablehlo.subtract {one}, {s} : {ty [B,n]}\n" ++
            s!"    {xom} = stablehlo.multiply {x}, {om} : {ty [B,n]}\n" ++
            s!"    {inr} = stablehlo.add {one}, {xom} : {ty [B,n]}\n" ++
            s!"    {sp} = stablehlo.multiply {s}, {inr} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {sp} : {ty [B,n]}\n", o :: st)
  | .sigmoidF n, r :: st => do
      -- sigmoid forward: σ(x) = logistic(x) (smooth, the SE gate's output nonlinearity).
      let o ← fresh
      pure (s!"    {o} = stablehlo.logistic {r} : {ty [B,n]}\n", o :: st)
  | .sigmoidBack x n, r :: st => do
      -- sigmoid input-VJP: dy ⊙ σ(x)·(1−σ(x)), recomputing σ from the saved
      -- pre-activation {x} (matches `sigmoidScalarDeriv`'s closed form, IRPrint `sigmoidBackM`).
      let s ← fresh; let one ← fresh; let om ← fresh; let sp ← fresh; let o ← fresh
      pure (s!"    {s} = stablehlo.logistic {x} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {om} = stablehlo.subtract {one}, {s} : {ty [B,n]}\n" ++
            s!"    {sp} = stablehlo.multiply {s}, {om} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {sp} : {ty [B,n]}\n", o :: st)
  | .layerScaleF gN n, r :: st => do
      -- per-element layer-scale `γ ⊙ x`: broadcast γ:[n] over the batch, then multiply.
      let gb ← fresh; let o ← fresh
      pure (s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [n]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {gb} : {ty [B,n]}\n", o :: st)
  | .layerScaleChF gN c h w', r :: st => do
      -- per-channel layer-scale: reshape flat→NCHW, broadcast γ:[c] over
      -- batch+spatial (dims=[1]), multiply, reshape back.
      let xn ← fresh; let gb ← fresh; let m ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
            s!"    {m} = stablehlo.multiply {xn}, {gb} : {ty [B,c,h,w']}\n" ++
            s!"    {o} = stablehlo.reshape {m} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .geluF n, r :: st => do
      -- gelu forward (tanh approximation): y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
      -- Smooth everywhere (no kink/mask); `stablehlo.tanh` is the only non-arith op.
      let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
      let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
      let chalf ← fresh; let hx ← fresh; let o ← fresh
      pure (s!"    {x2} = stablehlo.multiply {r}, {r} : {ty [B,n]}\n" ++
            s!"    {x3} = stablehlo.multiply {x2}, {r} : {ty [B,n]}\n" ++
            s!"    {ck} = stablehlo.constant dense<0.044715> : {ty [B,n]}\n" ++
            s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty [B,n]}\n" ++
            s!"    {inn} = stablehlo.add {r}, {kx3} : {ty [B,n]}\n" ++
            s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty [B,n]}\n" ++
            s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty [B,n]}\n" ++
            s!"    {t} = stablehlo.tanh {u} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {opt} = stablehlo.add {one}, {t} : {ty [B,n]}\n" ++
            s!"    {chalf} = stablehlo.constant dense<0.5> : {ty [B,n]}\n" ++
            s!"    {hx} = stablehlo.multiply {chalf}, {r} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {hx}, {opt} : {ty [B,n]}\n", o :: st)
  | .geluBack x n, r :: st => do
      -- gelu input-VJP: dy ⊙ gelu'(x), recomputing tanh(u(x)) from the saved
      -- pre-activation {x}. gelu'(x) = 0.5·(1+t) + 0.5·x·(1−t²)·√(2/π)·(1+3·0.044715·x²),
      -- t = tanh(√(2/π)·(x+0.044715·x³)). (Matches IRPrint `renderGeluB`.)
      let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
      let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
      let chalf ← fresh; let term1 ← fresh; let t2 ← fresh; let omt2 ← fresh
      let hx ← fresh; let hxo ← fresh; let c3b ← fresh; let a3x2 ← fresh
      let in2 ← fresh; let up ← fresh; let term2 ← fresh; let gp ← fresh; let o ← fresh
      pure (s!"    {x2} = stablehlo.multiply {x}, {x} : {ty [B,n]}\n" ++
            s!"    {x3} = stablehlo.multiply {x2}, {x} : {ty [B,n]}\n" ++
            s!"    {ck} = stablehlo.constant dense<0.044715> : {ty [B,n]}\n" ++
            s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty [B,n]}\n" ++
            s!"    {inn} = stablehlo.add {x}, {kx3} : {ty [B,n]}\n" ++
            s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty [B,n]}\n" ++
            s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty [B,n]}\n" ++
            s!"    {t} = stablehlo.tanh {u} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {opt} = stablehlo.add {one}, {t} : {ty [B,n]}\n" ++
            s!"    {chalf} = stablehlo.constant dense<0.5> : {ty [B,n]}\n" ++
            s!"    {term1} = stablehlo.multiply {chalf}, {opt} : {ty [B,n]}\n" ++
            s!"    {t2} = stablehlo.multiply {t}, {t} : {ty [B,n]}\n" ++
            s!"    {omt2} = stablehlo.subtract {one}, {t2} : {ty [B,n]}\n" ++
            s!"    {hx} = stablehlo.multiply {chalf}, {x} : {ty [B,n]}\n" ++
            s!"    {hxo} = stablehlo.multiply {hx}, {omt2} : {ty [B,n]}\n" ++
            s!"    {c3b} = stablehlo.constant dense<0.134145> : {ty [B,n]}\n" ++
            s!"    {a3x2} = stablehlo.multiply {c3b}, {x2} : {ty [B,n]}\n" ++
            s!"    {in2} = stablehlo.add {one}, {a3x2} : {ty [B,n]}\n" ++
            s!"    {up} = stablehlo.multiply {csqrt}, {in2} : {ty [B,n]}\n" ++
            s!"    {term2} = stablehlo.multiply {hxo}, {up} : {ty [B,n]}\n" ++
            s!"    {gp} = stablehlo.add {term1}, {term2} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {gp} : {ty [B,n]}\n", o :: st)
  | .softmaxRowF m n, r :: st => do
      -- ROW-softmax: reshape flat `[B,m*n]` → `[B,m,n]`, exp, reduce add over the
      -- LAST axis [2] (per row), broadcast back over dims [0,1], divide, reshape to
      -- flat. Plain exp/sum (no max-shift), matching the proven `softmax` (3-D
      -- analogue of `.softmaxDiv`).
      let xn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh; let sb ← fresh
      let dv ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
        s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {dv} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {dv} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .softmaxRowBack x m n, r :: st => do
      -- ROW-softmax input-VJP `p ⊙ (dy − ⟨p,dy⟩)` per row: reshape flat→`[B,m,n]`,
      -- recompute `p` from the saved pre-softmax scores {x} (exp/reduce[2]/broadcast/
      -- divide), then the rank-1 correction (`pdy`, reduce[2], subtract, multiply),
      -- reshape to flat. {r} is dy.
      let xn ← fresh; let dn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh
      let sb ← fresh; let p ← fresh; let pdy ← fresh; let sr ← fresh; let srb ← fresh
      let d ← fresh; let dz ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {x} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
        s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {p} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
        s!"    {pdy} = stablehlo.multiply {p}, {dn} : {ty [B,m,n]}\n" ++
        s!"    {sr} = stablehlo.reduce({pdy} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {srb} = stablehlo.broadcast_in_dim {sr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {d} = stablehlo.subtract {dn}, {srb} : {ty [B,m,n]}\n" ++
        s!"    {dz} = stablehlo.multiply {p}, {d} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {dz} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .matmulF m k n, b :: a :: st => do
      -- flattened matrix multiply C = A·B: reshape both operands to rank 3,
      -- dot_general with batching dim 0 (contract A's last axis with B's middle),
      -- reshape back to flat. (Postorder pushes a then b, so b is on top.)
      let an ← fresh; let bn ← fresh; let mm ← fresh; let o ← fresh
      pure (s!"    {an} = stablehlo.reshape {a} : ({ty [B, m*k]}) -> {ty [B,m,k]}\n" ++
        s!"    {bn} = stablehlo.reshape {b} : ({ty [B, k*n]}) -> {ty [B,k,n]}\n" ++
        s!"    {mm} = stablehlo.dot_general {an}, {bn}, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,m,k]}, {ty [B,k,n]}) -> {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {mm} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .transposeF m n, r :: st => do
      -- flattened matrix transpose: reshape to rank 3, swap the matrix axes
      -- (dims = [0, 2, 1], batch axis fixed), reshape back.
      let xn ← fresh; let t ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {t} = stablehlo.transpose {xn}, dims = [0, 2, 1] : ({ty [B,m,n]}) -> {ty [B,n,m]}\n" ++
        s!"    {o} = stablehlo.reshape {t} : ({ty [B,n,m]}) -> {ty [B, n*m]}\n", o :: st)
  | .scaleF sStr n, r :: st => do
      -- scalar multiply s·x against a splat constant (SDPA's 1/√d).
      let c ← fresh; let o ← fresh
      pure (s!"    {c} = stablehlo.constant dense<{sStr}> : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {c} : {ty [B,n]}\n", o :: st)
  | .lnRowF gN bN epsStr m n, r :: st => do
      -- ROW-wise LayerNorm forward: reshape flat [B,m*n] → [B,m,n], then `bnF`'s
      -- normalize/affine graph at rank 3 — μ/var reduced over the LAST axis [2]
      -- (per token row), broadcast back over dims [0,1], scalar γ/β (dims = []),
      -- reshape to flat. LayerNorm IS per-example BN per row.
      let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,m,n]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .lnRowBack gN xN epsStr m n, r :: st => do
      -- ROW-wise LN input-VJP: recompute x̂/istd per row from the saved flat
      -- pre-LN input {xN}, then `bnBack`'s consolidated three-term
      -- `(istd/n)·(n·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))` (dx̂ = γ·dy) at rank 3, all Σ
      -- reductions over the row axis [2], reshape to flat. {r} is dy.
      let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,m,n]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,m,n]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,m,n]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,m,n]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,m,n]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {o0} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .denseRowF wN bN N a c, r :: st => do
      -- per-token dense: reshape [B,N*a] → [B,N,a], dot_general contracting the
      -- feature axis with W:[a,c] ([2] x [0] — every token row through the same W),
      -- bias broadcast dims = [2], reshape back. (ViTRender `mlpRowFwd` form.)
      let xn ← fresh; let dg ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
        s!"    {dg} = stablehlo.dot_general {xn}, {wN}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,N,a]}, {ty [a,c]}) -> {ty [B,N,c]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [c]}) -> {ty [B,N,c]}\n" ++
        s!"    {ob} = stablehlo.add {dg}, {bb} : {ty [B,N,c]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,N,c]}) -> {ty [B, N*c]}\n", o :: st)
  | .denseRowBack wN N a c, r :: st => do
      -- per-token dense input-VJP dX = dY·Wᵀ: contract dy's feature axis with W's
      -- OUTPUT axis ([2] x [1] — the GPU-validated ViTRender backward form).
      let dn ← fresh; let dg ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
        s!"    {dg} = stablehlo.dot_general {dn}, {wN}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,N,c]}, {ty [a,c]}) -> {ty [B,N,a]}\n" ++
        s!"    {o} = stablehlo.reshape {dg} : ({ty [B,N,a]}) -> {ty [B, N*a]}\n", o :: st)
  | .patchEmbedF wN bN clsN posN ic H W P N D, r :: st => do
      -- ViT patch embedding: reshape image to [B,ic,H,W], stride-P VALID conv
      -- (kernel [D,ic,P,P] — the non-overlapping patch projection) + bias, move
      -- channels last (transpose [0,2,3,1]) and flatten the patch grid to [B,N,D],
      -- prepend the broadcast CLS token (concatenate at dim 1), add the position
      -- embedding (broadcast dims = [1,2]), reshape to flat [B,(N+1)*D].
      let hp := H / P; let wp := W / P
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let cb ← fresh
      let tr ← fresh; let tk ← fresh; let clsb ← fresh; let cat ← fresh
      let pb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {wN})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [{P}, {P}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,H,W]}, {ty [D,ic,P,P]}) -> {ty [B,D,hp,wp]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [D]}) -> {ty [B,D,hp,wp]}\n" ++
        s!"    {cb} = stablehlo.add {cv}, {bb} : {ty [B,D,hp,wp]}\n" ++
        s!"    {tr} = stablehlo.transpose {cb}, dims = [0, 2, 3, 1] : ({ty [B,D,hp,wp]}) -> {ty [B,hp,wp,D]}\n" ++
        s!"    {tk} = stablehlo.reshape {tr} : ({ty [B,hp,wp,D]}) -> {ty [B,N,D]}\n" ++
        s!"    {clsb} = stablehlo.broadcast_in_dim {clsN}, dims = [2] : ({ty [D]}) -> {ty [B,1,D]}\n" ++
        s!"    {cat} = stablehlo.concatenate {clsb}, {tk}, dim = 1 : ({ty [B,1,D]}, {ty [B,N,D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {pb} = stablehlo.broadcast_in_dim {posN}, dims = [1, 2] : ({ty [N+1,D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {ob} = stablehlo.add {cat}, {pb} : {ty [B,N+1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,N+1,D]}) -> {ty [B, (N+1)*D]}\n", o :: st)
  | .clsSliceF N D, r :: st => do
      -- CLS-token gather (row 0): reshape [B,(N+1)*D] → [B,N+1,D], slice the
      -- first token row, reshape to [B,D]. (ViTRender `headFwd` slice form.)
      let xn ← fresh; let sl ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:1, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {sl} : ({ty [B,1,D]}) -> {ty [B,D]}\n", o :: st)
  | .clsPadF N D, r :: st => do
      -- CLS-slice VJP (scatter dy to row 0): reshape [B,D] → [B,1,D], zero-pad
      -- N token rows below (high = [0, N, 0]), reshape to flat [B,(N+1)*D].
      -- (ViTRender `headBack` pad form.)
      let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B,D]}) -> {ty [B,1,D]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, 0], high = [0, {N}, 0], interior = [0, 0, 0] : ({ty [B,1,D]}, tensor<f32>) -> {ty [B,N+1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {pd} : ({ty [B,N+1,D]}) -> {ty [B, (N+1)*D]}\n", o :: st)
  | .headSliceF N heads d hIdx, r :: st => do
      -- per-head column slice: reshape [B,N*(H*d)] → [B,N,H*d], slice head h's
      -- contiguous feature block [h*d:(h+1)*d] (row-major layout), reshape to flat.
      let xn ← fresh; let sl ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, N*(heads*d)]}) -> {ty [B,N,heads*d]}\n" ++
        s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:{N}, {hIdx*d}:{(hIdx+1)*d}] : ({ty [B,N,heads*d]}) -> {ty [B,N,d]}\n" ++
        s!"    {o} = stablehlo.reshape {sl} : ({ty [B,N,d]}) -> {ty [B, N*d]}\n", o :: st)
  | .headPadF N heads d hIdx, r :: st => do
      -- per-head column scatter: reshape [B,N*d] → [B,N,d], zero-pad the feature
      -- axis into head h's block (low = h*d, high = (heads-1-h)*d), reshape to flat.
      let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*d]}) -> {ty [B,N,d]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, {hIdx*d}], high = [0, 0, {(heads-1-hIdx)*d}], interior = [0, 0, 0] : ({ty [B,N,d]}, tensor<f32>) -> {ty [B,N,heads*d]}\n" ++
        s!"    {o} = stablehlo.reshape {pd} : ({ty [B,N,heads*d]}) -> {ty [B, N*(heads*d)]}\n", o :: st)
  | .rowScaleF gN m n, r :: st => do
      -- per-token broadcast scale: reshape [B,m*n] -> [B,m,n], broadcast the shared
      -- gamma:[n] over batch+rows (dims = [2]), multiply, reshape back.
      let xn <- fresh; let gb <- fresh; let mu <- fresh; let o <- fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.multiply {xn}, {gb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {mu} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .rowBiasF bN m n, r :: st => do
      -- per-token broadcast bias: same bracket, broadcast beta:[n] dims = [2], add.
      let xn <- fresh; let bb <- fresh; let ad <- fresh; let o <- fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
        s!"    {ad} = stablehlo.add {xn}, {bb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {ad} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .batched tag names info, r :: st =>
      -- EfficientNet batched op: emit the concrete `[N,C,H,W]` StableHLO from the
      -- tag (which op) + names (weight/bias/BN-input/SE-input/γ/ε SSA names) + info
      -- (shape dims). Batched values flow as 2-D `[B, c·h·w]` (B = batch); each op
      -- reshapes its operand to 4-D, computes, reshapes back — uniform with the
      -- per-example ops (`convWeightSgd`/`denseRowF` do the same). Backward ops are
      -- self-contained: they recompute forward intermediates from the carried
      -- input/weight names (the mnv2 pattern). `den` never calls `emit`; this text
      -- is iree-validated, not theorem-tied (the per-op lexing trust the whole
      -- suite carries). Backward tags are filled in the next pass.
      match tag, names, info with
      | "conv", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,h,w]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convBf16", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let xb ← fresh; let wb ← fresh; let cc ← fresh; let cf ← fresh
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {xb} = stablehlo.convert {xr} : ({ty [B,ic,h,w]}) -> {tyBf16 [B,ic,h,w]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [oc,ic,kH,kW]}) -> {tyBf16 [oc,ic,kH,kW]}\n" ++
            s!"    {cc} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,ic,h,w]}, {tyBf16 [oc,ic,kH,kW]}) -> {tyBf16 [B,oc,h,w]}\n" ++
            s!"    {cf} = stablehlo.convert {cc} : ({tyBf16 [B,oc,h,w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cf}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "convStrided", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,2*h,2*w]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      -- ⭐ The asymmetric-pad peer. `pad_low = (k-2)/2 = p-1`, `pad_high = k/2 = p` for odd `k`
      -- (k=3 → [[0,1]], k=5 → [[1,2]], k=7 → [[2,3]]) — exactly what XLA computes for `'SAME'` at
      -- an even input, which is the only input shape this token's type admits (`2*h`, `2*w`).
      -- Everything else is byte-identical to "convStrided", which is the point: the ONLY
      -- difference between the two nets is these four numbers.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convStridedBf16", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let xb ← fresh; let wb ← fresh; let cc ← fresh; let cf ← fresh
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {xb} = stablehlo.convert {xr} : ({ty [B,ic,2*h,2*w]}) -> {tyBf16 [B,ic,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [oc,ic,kH,kW]}) -> {tyBf16 [oc,ic,kH,kW]}\n" ++
            s!"    {cc} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,ic,2*h,2*w]}, {tyBf16 [oc,ic,kH,kW]}) -> {tyBf16 [B,oc,h,w]}\n" ++
            s!"    {cf} = stablehlo.convert {cc} : ({tyBf16 [B,oc,h,w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cf}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "convStridedXla", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let lo := p - 1
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{lo}, {p}], [{lo}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,2*h,2*w]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "convStridedXlaBf16", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let lo := p - 1
          let xr ← fresh; let xb ← fresh; let wb ← fresh; let cc ← fresh; let cf ← fresh
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {xb} = stablehlo.convert {xr} : ({ty [B,ic,2*h,2*w]}) -> {tyBf16 [B,ic,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [oc,ic,kH,kW]}) -> {tyBf16 [oc,ic,kH,kW]}\n" ++
            s!"    {cc} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{lo}, {p}], [{lo}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,ic,2*h,2*w]}, {tyBf16 [oc,ic,kH,kW]}) -> {tyBf16 [B,oc,h,w]}\n" ++
            s!"    {cf} = stablehlo.convert {cc} : ({tyBf16 [B,oc,h,w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cf}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "depthwise", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,h,w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⭐ The asymmetric-pad depthwise. `pad_low = p-1`, `pad_high = p` (k=3 → [[0,1]], k=5 →
      -- [[1,2]]) — XLA `'SAME'` at an even input, which is the only shape this token's type admits.
      -- Byte-identical to "depthwiseStrided" apart from those four numbers.
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseBf16", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let xb ← fresh; let wb ← fresh; let cc ← fresh; let cf ← fresh
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xb} = stablehlo.convert {xr} : ({ty [B,c,h,w]}) -> {tyBf16 [B,c,h,w]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [c,1,kH,kW]}) -> {tyBf16 [c,1,kH,kW]}\n" ++
            s!"    {cc} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({tyBf16 [B,c,h,w]}, {tyBf16 [c,1,kH,kW]}) -> {tyBf16 [B,c,h,w]}\n" ++
            s!"    {cf} = stablehlo.convert {cc} : ({tyBf16 [B,c,h,w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cf}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "depthwiseStridedXla", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let lo := p - 1
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{lo}, {p}], [{lo}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseStridedXlaBf16", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let lo := p - 1
          let xr ← fresh; let xb ← fresh; let wb ← fresh; let cc ← fresh; let cf ← fresh
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xb} = stablehlo.convert {xr} : ({ty [B,c,2*h,2*w]}) -> {tyBf16 [B,c,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [c,1,kH,kW]}) -> {tyBf16 [c,1,kH,kW]}\n" ++
            s!"    {cc} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{lo}, {p}], [{lo}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({tyBf16 [B,c,2*h,2*w]}, {tyBf16 [c,1,kH,kW]}) -> {tyBf16 [B,c,h,w]}\n" ++
            s!"    {cf} = stablehlo.convert {cc} : ({tyBf16 [B,c,h,w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cf}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "depthwiseStrided", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. The f32-result
      -- shape folds to pure f32, for grouped convolutions exactly as for ordinary ones
      -- (measured). ⚠ SYMMETRIC pad — this is the torchvision-origin variant, NOT `Xla`.
      | "depthwiseStridedBf16", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let xb ← fresh; let wb ← fresh; let cc ← fresh; let cf ← fresh
          let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xb} = stablehlo.convert {xr} : ({ty [B,c,2*h,2*w]}) -> {tyBf16 [B,c,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [c,1,kH,kW]}) -> {tyBf16 [c,1,kH,kW]}\n" ++
            s!"    {cc} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({tyBf16 [B,c,2*h,2*w]}, {tyBf16 [c,1,kH,kW]}) -> {tyBf16 [B,c,h,w]}\n" ++
            s!"    {cf} = stablehlo.convert {cc} : ({tyBf16 [B,c,h,w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cf}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "dense", [wN, bN], [_N, a, c] => do
          let dg ← fresh; let bb ← fresh; let o ← fresh
          pure (
            s!"    {dg} = stablehlo.dot_general {r}, {wN}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [a,c]}) -> {ty [B,c]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {o} = stablehlo.add {dg}, {bb} : {ty [B,c]}\n", o :: st)
      | "gap", [], [_N, c, h, w] => do
          let xr ← fresh; let z ← fresh; let sr ← fresh; let nf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {o} = stablehlo.divide {sr}, {nf} : {ty [B,c]}\n", o :: st)
      | "seBlock", [w1, b1, w2, b2], [_N, c, h, w, rr] => do
          let xr ← fresh; let z ← fresh; let sqs ← fresh; let sqnf ← fresh; let sq ← fresh
          let exd ← fresh; let exbb ← fresh; let ex ← fresh; let a1s ← fresh; let a1 ← fresh
          let h2d ← fresh; let h2bb ← fresh; let h2 ← fresh; let gate ← fresh; let gb ← fresh
          let se ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sqs} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {sqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {sq} = stablehlo.divide {sqs}, {sqnf} : {ty [B,c]}\n" ++
            s!"    {exd} = stablehlo.dot_general {sq}, {w1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [c,rr]}) -> {ty [B,rr]}\n" ++
            s!"    {exbb} = stablehlo.broadcast_in_dim {b1}, dims = [1] : ({ty [rr]}) -> {ty [B,rr]}\n" ++
            s!"    {ex} = stablehlo.add {exd}, {exbb} : {ty [B,rr]}\n" ++
            s!"    {a1s} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {a1} = stablehlo.multiply {ex}, {a1s} : {ty [B,rr]}\n" ++
            s!"    {h2d} = stablehlo.dot_general {a1}, {w2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [rr,c]}) -> {ty [B,c]}\n" ++
            s!"    {h2bb} = stablehlo.broadcast_in_dim {b2}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {h2} = stablehlo.add {h2d}, {h2bb} : {ty [B,c]}\n" ++
            s!"    {gate} = stablehlo.logistic {h2} : {ty [B,c]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gate}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {se} = stablehlo.multiply {xr}, {gb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {se} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "swish", [], [_N, n] => do
          -- Pointwise swish at the BATCHED index: byte-for-byte the `.swishF` emit,
          -- except the width comes from the descriptor's per-example `n` rather than
          -- from the SHlo index (which here is `N·n`). `_N` is discarded for the same
          -- reason every batched tag discards it — the runtime batch is `B`.
          let s ← fresh; let o ← fresh
          pure (s!"    {s} = stablehlo.logistic {r} : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {s} : {ty [B,n]}\n", o :: st)
      | "relu", [], [_N, n] => do
          -- byte-for-byte `.reluF`'s emit, width from the descriptor's `n`.
          let z ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.maximum {r}, {z} : {ty [B,n]}\n", o :: st)
      | "bnEval", [gN, bN, muN, varN, es], [_N, oc, h, w] => do
          -- byte-for-byte `.bnPerChannelEvalF`'s emit, dims from the descriptor rather than off
          -- the SHlo index (§2b). INFERENCE BN: reshape to [B,oc,h,w], then the affine map
          -- γ·(x − μ)·rsqrt(var + ε) + β with μ/var/γ/β all rank-1 `[oc]` graph inputs. No reduce
          -- and no normalizer constant — that is the whole difference from `bnBatch`, and why the
          -- descriptor form is denotationally honest at any `N`.
          let xn ← fresh; let mub ← fresh; let xc ← fresh; let vb ← fresh; let ep ← fresh
          let ve ← fresh; let istd ← fresh; let xhat ← fresh; let gb ← fresh; let bb ← fresh
          let gx ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mub} = stablehlo.broadcast_in_dim {muN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xn}, {mub} : {ty [B,oc,h,w]}\n" ++
            s!"    {vb} = stablehlo.broadcast_in_dim {varN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vb}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
            s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "relu6", [], [_N, n] => do
          -- byte-for-byte `.relu6F`'s emit, width from the descriptor's `n`.
          let z ← fresh; let six ← fresh; let mx ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
                s!"    {six} = stablehlo.constant dense<6.0> : {ty [B,n]}\n" ++
                s!"    {mx} = stablehlo.maximum {r}, {z} : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.minimum {mx}, {six} : {ty [B,n]}\n", o :: st)
      | "maxPool", [], [_N, c, h, w] => do
          -- byte-for-byte `.maxPoolF`'s emit; dims from the descriptor, batch from `B`.
          let xn ← fresh; let ninf ← fresh; let pp ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
            s!"    {pp} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
            "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
            "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
            "        stablehlo.return %pm : tensor<f32>\n" ++
            "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
            s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {pp} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⭐ The 3×3/s2 stem pool, forward — both index conventions, ONE text writer
      -- (`maxPool3s2FwdText`), so the per-example and batched forms cannot drift.
      | "maxPool3s2", [], [c, h, w] => do
          let xn ← fresh; let ninf ← fresh; let pp ← fresh; let o ← fresh
          pure (maxPool3s2FwdText B c h w r xn ninf pp o, o :: st)
      | "maxPool3s2", [], [_N, c, h, w] => do
          let xn ← fresh; let ninf ← fresh; let pp ← fresh; let o ← fresh
          pure (maxPool3s2FwdText B c h w r xn ninf pp o, o :: st)
      | "maxPool3s2BackP", [xN], [c, h, w] => do
          let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
          pure (maxPool3s2BackText B c h w xN r xr dr z scn o, o :: st)
      | "maxPool3s2BackP", [xN], [_N, c, h, w] => do
          let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
          pure (maxPool3s2BackText B c h w xN r xr dr z scn o, o :: st)
      | "maxPoolBackP", [xN], [_N, c, h, w] => do
          -- byte-for-byte `.maxPoolBack`'s emit. NOTE the region block arguments %sa/%sb/%sc/%sd
          -- are HARDCODED here, so they are reserved SSA names: a top-level value of the same
          -- name is a redefinition error, and it only surfaces at XLA compile time.
          let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
            "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
            "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
            "        stablehlo.return %sge : tensor<i1>\n" ++
            "    }, " ++ "{\n" ++
            "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
            "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
            "        stablehlo.return %ss : tensor<f32>\n" ++
            "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      | "convBiasSgd", [bN, lrS], [_N, oc, h, w] => do
          -- byte-for-byte `.convBiasSgd`'s emit. Stride-independent, so the strided peer
          -- shares this case (both `skel` to the same Raw).
          let dr ← fresh; let z ← fresh; let g ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {g} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sB} = stablehlo.multiply {g}, {lB} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
      | "selectPosP", [x], [_N, n] => do
          -- byte-for-byte `.selectPos`'s emit, width from the descriptor's `n`.
          let z ← fresh; let msk ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
            s!"    {msk} = stablehlo.compare GT, {x}, {z} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
            s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,n]}, {ty [B,n]}\n", o :: st)
      | "selectMidP", [x], [_N, n] => do
          -- byte-for-byte `.selectMid`'s emit, width from the ctor's `n`. Two-sided kink, so
          -- two compares AND-ed — unlike `selectPosP`'s single GT.
          let z ← fresh; let six ← fresh; let g0 ← fresh; let l6 ← fresh
          let msk ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
            s!"    {six} = stablehlo.constant dense<6.0> : {ty [B,n]}\n" ++
            s!"    {g0} = stablehlo.compare GT, {x}, {z} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
            s!"    {l6} = stablehlo.compare LT, {x}, {six} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
            s!"    {msk} = stablehlo.and {g0}, {l6} : {tyI1 [B,n]}\n" ++
            s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,n]}, {ty [B,n]}\n", o :: st)
      | "dropPathP", [mN], [_N, n] => do
          -- ▶ STOCHASTIC DEPTH: the per-SAMPLE residual-branch scale
          -- (`planning/stochastic_depth.md`). `mN` is a graph INPUT of type `tensor<Bxf32>` — one
          -- value per EXAMPLE, computed on the host — and `dims = [0]` is what makes it the
          -- reference's `(B, 1, …, 1)` mask: every position within an example is scaled
          -- identically, every example independently. Emitting a `tensor<B×n>` scale instead
          -- typechecks, compiles and trains, and is per-ELEMENT dropout — a different regulariser.
          -- ⚠ NO BAKED `1/keep`. The driver folds the inversion into the supplied value
          -- (`bernoulli(keep_i)/keep_i` at train, `1.0` at eval), which is what makes the ones-scale
          -- forward the EXACT identity and lets this op be emitted in the forward too — keeping the
          -- `forward ⊂ train-step` prefix audit alive. See `Proofs.dropPath`'s note on why a baked
          -- constant and that audit cannot both hold.
          let mb ← fresh; let o ← fresh
          pure (s!"    {mb} = stablehlo.broadcast_in_dim {mN}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {mb}, {r} : {ty [B,n]}\n", o :: st)
      | "sigmoidP", [], [_N, n] => do
          -- σ(z) at the batched shape, for BCE-with-logits' cotangent `(σ(z) − t)/(B·K)`.
          -- ⚠ ONE op, and `stablehlo.logistic` is the same primitive `sigmoidF` emits — the
          -- difference is only the shape it is emitted at.
          let o ← fresh
          pure (s!"    {o} = stablehlo.logistic {r} : {ty [B,n]}\n", o :: st)
      | "dropoutP", [mN], [_N, n] => do
          -- ▶ CLASSIFIER DROPOUT (`recipe_gaps.md` gap C): the per-ELEMENT inverted mask, applied
          -- immediately before the classifier dense. `mN` is a graph INPUT of type
          -- `tensor<B×n×f32>` — one value per (example, feature), computed on the host.
          --
          -- ⚠⚠ **NO `broadcast_in_dim`, AND THAT ABSENCE IS THE WHOLE CLAIM.** The mask already
          -- has the value's shape, because the reference draws `bernoulli(key, keep, x.shape)`
          -- (`jax/Jax/Codegen.lean:1971`) rather than the `(B, 1, …, 1)` shape stochastic depth
          -- uses. A `dims = [0]` broadcast off a `tensor<B>` input here typechecks, compiles, runs,
          -- descends — and is stochastic depth on the classifier, a different regulariser. That is
          -- `dropPathP`'s warning read backwards, and `tests/TestBatchedEmitTie.lean` pins both
          -- directions: that one asserts the broadcast is PRESENT, this one that it is ABSENT.
          -- ⚠ NO BAKED `1/keep`, for `dropPathP`'s reason exactly: the driver folds the inversion
          -- into the supplied mask, so the ones-mask forward is the exact identity and this op can
          -- be emitted in the forward artifact without rescaling eval.
          let o ← fresh
          pure (s!"    {o} = stablehlo.multiply {mN}, {r} : {ty [B,n]}\n", o :: st)
      | "swishBackP", [x], [_N, n] => do
          -- byte-for-byte `.swishBack`'s emit, width from the descriptor's `n`.
          let s ← fresh; let one ← fresh; let om ← fresh; let xom ← fresh
          let inr ← fresh; let sp ← fresh; let o ← fresh
          pure (s!"    {s} = stablehlo.logistic {x} : {ty [B,n]}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
                s!"    {om} = stablehlo.subtract {one}, {s} : {ty [B,n]}\n" ++
                s!"    {xom} = stablehlo.multiply {x}, {om} : {ty [B,n]}\n" ++
                s!"    {inr} = stablehlo.add {one}, {xom} : {ty [B,n]}\n" ++
                s!"    {sp} = stablehlo.multiply {s}, {inr} : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {sp} : {ty [B,n]}\n", o :: st)
      | "sigmoidBackP", [x], [_N, n] => do
          -- byte-for-byte `.sigmoidBack`'s emit, width from the descriptor's `n`.
          let s ← fresh; let one ← fresh; let om ← fresh; let sp ← fresh; let o ← fresh
          pure (s!"    {s} = stablehlo.logistic {x} : {ty [B,n]}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
                s!"    {om} = stablehlo.subtract {one}, {s} : {ty [B,n]}\n" ++
                s!"    {sp} = stablehlo.multiply {s}, {om} : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {sp} : {ty [B,n]}\n", o :: st)
      | "softmaxRow", [], [_N, m, n] => do
          -- byte-for-byte `.softmaxRowF`'s emit. `m` is rows PER EXAMPLE (it always
          -- was); the batch is `_N` on the proof side and `B` in the emit.
          let xn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh; let sb ← fresh
          let dv ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
            s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {dv} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {dv} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      -- ── the five ViT/ConvNeXt row/pointwise descriptors (§0.2 ▶2). Each body is a BYTE-FOR-BYTE
      --    copy of its descriptor-less peer's, with the width read from the descriptor's `m`/`n`
      --    instead of the SHlo index — which is the entire content of the batched-index move on
      --    the emit side. `tests/TestBatchedEmitTie.lean` ties each pair, so "byte-for-byte" is
      --    checked rather than intended.
      | "geluBackP", [x], [_N, n] => do
          -- byte-for-byte `.geluBack`'s emit, width from the descriptor's `n`.
          let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
          let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
          let chalf ← fresh; let term1 ← fresh; let t2 ← fresh; let omt2 ← fresh
          let hx ← fresh; let hxo ← fresh; let c3b ← fresh; let a3x2 ← fresh
          let in2 ← fresh; let up ← fresh; let term2 ← fresh; let gp ← fresh; let o ← fresh
          pure (s!"    {x2} = stablehlo.multiply {x}, {x} : {ty [B,n]}\n" ++
                s!"    {x3} = stablehlo.multiply {x2}, {x} : {ty [B,n]}\n" ++
                s!"    {ck} = stablehlo.constant dense<0.044715> : {ty [B,n]}\n" ++
                s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty [B,n]}\n" ++
                s!"    {inn} = stablehlo.add {x}, {kx3} : {ty [B,n]}\n" ++
                s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty [B,n]}\n" ++
                s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty [B,n]}\n" ++
                s!"    {t} = stablehlo.tanh {u} : {ty [B,n]}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
                s!"    {opt} = stablehlo.add {one}, {t} : {ty [B,n]}\n" ++
                s!"    {chalf} = stablehlo.constant dense<0.5> : {ty [B,n]}\n" ++
                s!"    {term1} = stablehlo.multiply {chalf}, {opt} : {ty [B,n]}\n" ++
                s!"    {t2} = stablehlo.multiply {t}, {t} : {ty [B,n]}\n" ++
                s!"    {omt2} = stablehlo.subtract {one}, {t2} : {ty [B,n]}\n" ++
                s!"    {hx} = stablehlo.multiply {chalf}, {x} : {ty [B,n]}\n" ++
                s!"    {hxo} = stablehlo.multiply {hx}, {omt2} : {ty [B,n]}\n" ++
                s!"    {c3b} = stablehlo.constant dense<0.134145> : {ty [B,n]}\n" ++
                s!"    {a3x2} = stablehlo.multiply {c3b}, {x2} : {ty [B,n]}\n" ++
                s!"    {in2} = stablehlo.add {one}, {a3x2} : {ty [B,n]}\n" ++
                s!"    {up} = stablehlo.multiply {csqrt}, {in2} : {ty [B,n]}\n" ++
                s!"    {term2} = stablehlo.multiply {hxo}, {up} : {ty [B,n]}\n" ++
                s!"    {gp} = stablehlo.add {term1}, {term2} : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {gp} : {ty [B,n]}\n", o :: st)
      | "lnRowBackP", [gN, xN, epsStr], [_N, m, n] => do
          -- byte-for-byte `.lnRowBack`'s emit; `m` is rows PER EXAMPLE.
          let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
          let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
          let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
          let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
          let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
          let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
          pure (
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
            s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
            s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
            s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,m,n]}\n" ++
            s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,m,n]}\n" ++
            s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,m,n]}\n" ++
            s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,m,n]}\n" ++
            s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,m,n]}\n" ++
            s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {o0} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "dotOutP", [w], [_N, m, n] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [1], " ++
                s!"precision = [DEFAULT, DEFAULT] : ({ty [B,n]}, {ty [m,n]}) -> {ty [B,m]}\n", o :: st)
      | "expeP", [], [_N, n] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.exponential {r} : {ty [B,n]}\n", o :: st)
      | "softmaxDivP", [], [_N, n] => do
          -- byte-for-byte `.softmaxDiv`'s emit — which already reduced over `dimensions = [1]`,
          -- i.e. per example. It is the DEN that this descriptor fixes.
          let z ← fresh; let s ← fresh; let sb ← fresh; let o ← fresh
          pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {s} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
            s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.divide {r}, {sb} : {ty [B,n]}\n", o :: st)
      | "layerScaleChP", [gN], [_N, c, h, w'] => do
          let xn ← fresh; let gb ← fresh; let m ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
                s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
                s!"    {m} = stablehlo.multiply {xn}, {gb} : {ty [B,c,h,w']}\n" ++
                s!"    {o} = stablehlo.reshape {m} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
      | "convStride4P", [w, b], [_N, ic, oc, h, w', kH, kW] => do
          -- byte-for-byte `.flatConvStride4F`'s emit, including its ⚠ pad-one-less rule: the
          -- denotation reads the SAME conv at the offset-1 positions 4i+1, so the emitted pad is
          -- (k-1)/2 − 1 — for the 4×4 stem that is 0, the paper's left-aligned window.
          let pH := (kH - 1) / 2 - 1; let pW := (kW - 1) / 2 - 1
          let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*(2*h))*(2*(2*w'))]}) -> {ty [B,ic,2*(2*h),2*(2*w')]}\n" ++
            s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [4, 4], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,2*(2*h),2*(2*w')]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
            s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed result reads
      -- identically and compiles to pure f32 — measured on THIS shape (4×4/s4) before the op was
      -- written, so stride 4 buys no exemption from §9.2 any more than grouping did.
      -- ⚠⚠ The pad is `convStride4P`'s `(k-1)/2 − 1`, NOT `convBf16`'s `(k-1)/2`. At the 4×4 stem
      -- that is `[[0,0]]`. The two spellings produce the same output SIZE, so nothing structural
      -- separates them — do not "tidy" this to match the other bf16 convs.
      | "convStride4PBf16", [w, b], [_N, ic, oc, h, w', kH, kW] => do
          let pH := (kH - 1) / 2 - 1; let pW := (kW - 1) / 2 - 1
          let xn ← fresh; let xb ← fresh; let wb ← fresh; let cv ← fresh; let cf ← fresh
          let bb ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*(2*h))*(2*(2*w'))]}) -> {ty [B,ic,2*(2*h),2*(2*w')]}\n" ++
            s!"    {xb} = stablehlo.convert {xn} : ({ty [B,ic,2*(2*h),2*(2*w')]}) -> {tyBf16 [B,ic,2*(2*h),2*(2*w')]}\n" ++
            s!"    {wb} = stablehlo.convert {w} : ({ty [oc,ic,kH,kW]}) -> {tyBf16 [oc,ic,kH,kW]}\n" ++
            s!"    {cv} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [4, 4], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,ic,2*(2*h),2*(2*w')]}, {tyBf16 [oc,ic,kH,kW]}) -> {tyBf16 [B,oc,h,w']}\n" ++
            s!"    {cf} = stablehlo.convert {cv} : ({tyBf16 [B,oc,h,w']}) -> {ty [B,oc,h,w']}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
            s!"    {ob} = stablehlo.add {cf}, {bb} : {ty [B,oc,h,w']}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
      | "gelu", [], [_N, n] => do
          let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
          let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
          let chalf ← fresh; let hx ← fresh; let o ← fresh
          pure (s!"    {x2} = stablehlo.multiply {r}, {r} : {ty [B,n]}\n" ++
                s!"    {x3} = stablehlo.multiply {x2}, {r} : {ty [B,n]}\n" ++
                s!"    {ck} = stablehlo.constant dense<0.044715> : {ty [B,n]}\n" ++
                s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty [B,n]}\n" ++
                s!"    {inn} = stablehlo.add {r}, {kx3} : {ty [B,n]}\n" ++
                s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty [B,n]}\n" ++
                s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty [B,n]}\n" ++
                s!"    {t} = stablehlo.tanh {u} : {ty [B,n]}\n" ++
                s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
                s!"    {opt} = stablehlo.add {one}, {t} : {ty [B,n]}\n" ++
                s!"    {chalf} = stablehlo.constant dense<0.5> : {ty [B,n]}\n" ++
                s!"    {hx} = stablehlo.multiply {chalf}, {r} : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.multiply {hx}, {opt} : {ty [B,n]}\n", o :: st)
      | "transposeP", [], [_N, m, n] => do
          let xn ← fresh; let t ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {t} = stablehlo.transpose {xn}, dims = [0, 2, 1] : ({ty [B,m,n]}) -> {ty [B,n,m]}\n" ++
            s!"    {o} = stablehlo.reshape {t} : ({ty [B,n,m]}) -> {ty [B, n*m]}\n", o :: st)
      | "lnRowP", [gN, bN, epsStr], [_N, m, n] => do
          let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
          let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
          let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
          let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh
          let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
            s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
            s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
            s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,m,n]}\n" ++
            s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "rowScaleP", [gN], [_N, m, n] => do
          let xn ← fresh; let gb ← fresh; let mu ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
            s!"    {mu} = stablehlo.multiply {xn}, {gb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {mu} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "rowBiasP", [bN], [_N, m, n] => do
          let xn ← fresh; let bb ← fresh; let ad ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
            s!"    {ad} = stablehlo.add {xn}, {bb} : {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {ad} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | "denseRowBackP", [wN], [_N, rows, a, c] => do
          -- byte-for-byte `.denseRowBack`'s emit; `rows` is per-example rows.
          let dn ← fresh; let dg ← fresh; let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, rows*c]}) -> {ty [B,rows,c]}\n" ++
            s!"    {dg} = stablehlo.dot_general {dn}, {wN}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,rows,c]}, {ty [a,c]}) -> {ty [B,rows,a]}\n" ++
            s!"    {o} = stablehlo.reshape {dg} : ({ty [B,rows,a]}) -> {ty [B, rows*a]}\n", o :: st)
      -- ⚠⚠ bf16 operands, **bf16-TYPED result**, convert back — the CONV shape, applied to a dot.
      -- §9.2 established that `dot_general` reaches the tensor cores with EITHER result type and
      -- read that as "the result type is inert for dot". It is inert for CORRECTNESS and it is not
      -- inert for SPEED: an f32 result makes the gemm write twice the bytes and take a worse
      -- epilogue. Measured on ViT's own MLP chain (§20.1): f32-result 1.18×, bf16-result **1.60×**.
      -- ▶ So the convert-back is not "a node that buys nothing" — it is most of the win.
      | "denseRowBackPBf16", [wN], [_N, rows, a, c] => do
          let dn ← fresh; let db ← fresh; let wb ← fresh; let dg ← fresh; let df ← fresh; let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, rows*c]}) -> {ty [B,rows,c]}\n" ++
            s!"    {db} = stablehlo.convert {dn} : ({ty [B,rows,c]}) -> {tyBf16 [B,rows,c]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [a,c]}) -> {tyBf16 [a,c]}\n" ++
            s!"    {dg} = stablehlo.dot_general {db}, {wb}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({tyBf16 [B,rows,c]}, {tyBf16 [a,c]}) -> {tyBf16 [B,rows,a]}\n" ++
            s!"    {df} = stablehlo.convert {dg} : ({tyBf16 [B,rows,a]}) -> {ty [B,rows,a]}\n" ++
            s!"    {o} = stablehlo.reshape {df} : ({ty [B,rows,a]}) -> {ty [B, rows*a]}\n", o :: st)
      -- ══ ViT increment 1: the six batch-invariant forms. Every one is byte-for-byte its
      --    per-example peer's emit with the TOKEN count read off the tag (`tk`) instead of off the
      --    SHlo index — which is the whole content of the move, since `B` was always `pretty`'s.
      --    ⚠ `_N` (the batch) is deliberately unused in all six: an emit that read it would be
      --    reintroducing the conflation. `tests/TestBatchedEmitTie.lean` pins each against its peer.
      | "denseRowP", [wN, bN], [_N, tk, a, c] => do
          let xn ← fresh; let dg ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, tk*a]}) -> {ty [B,tk,a]}\n" ++
            s!"    {dg} = stablehlo.dot_general {xn}, {wN}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,tk,a]}, {ty [a,c]}) -> {ty [B,tk,c]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [c]}) -> {ty [B,tk,c]}\n" ++
            s!"    {ob} = stablehlo.add {dg}, {bb} : {ty [B,tk,c]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,tk,c]}) -> {ty [B, tk*c]}\n", o :: st)
      -- ⚠⚠ bf16-TYPED result then convert back, per `denseRowBackPBf16`'s note — and this is the op
      -- that carries ViT, six sites per block × 12 blocks. The BIAS is added after the convert, in
      -- f32, which is what `den`'s outer `rnd` sits inside of.
      | "denseRowPBf16", [wN, bN], [_N, tk, a, c] => do
          let xn ← fresh; let xb ← fresh; let wb ← fresh; let dg ← fresh; let df ← fresh
          let bb ← fresh; let ob ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, tk*a]}) -> {ty [B,tk,a]}\n" ++
            s!"    {xb} = stablehlo.convert {xn} : ({ty [B,tk,a]}) -> {tyBf16 [B,tk,a]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [a,c]}) -> {tyBf16 [a,c]}\n" ++
            s!"    {dg} = stablehlo.dot_general {xb}, {wb}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({tyBf16 [B,tk,a]}, {tyBf16 [a,c]}) -> {tyBf16 [B,tk,c]}\n" ++
            s!"    {df} = stablehlo.convert {dg} : ({tyBf16 [B,tk,c]}) -> {ty [B,tk,c]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [c]}) -> {ty [B,tk,c]}\n" ++
            s!"    {ob} = stablehlo.add {df}, {bb} : {ty [B,tk,c]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,tk,c]}) -> {ty [B, tk*c]}\n", o :: st)
      | "patchEmbedP", [wN, bN, clsN, posN], [_N, ic, H, W, P, tk, D] => do
          let hp := H / P; let wp := W / P
          let xn ← fresh; let cv ← fresh; let bb ← fresh; let cb ← fresh
          let tr ← fresh; let tkn ← fresh; let clsb ← fresh; let cat ← fresh
          let pb ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
            s!"    {cv} = stablehlo.convolution({xn}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [{P}, {P}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,H,W]}, {ty [D,ic,P,P]}) -> {ty [B,D,hp,wp]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [D]}) -> {ty [B,D,hp,wp]}\n" ++
            s!"    {cb} = stablehlo.add {cv}, {bb} : {ty [B,D,hp,wp]}\n" ++
            s!"    {tr} = stablehlo.transpose {cb}, dims = [0, 2, 3, 1] : ({ty [B,D,hp,wp]}) -> {ty [B,hp,wp,D]}\n" ++
            s!"    {tkn} = stablehlo.reshape {tr} : ({ty [B,hp,wp,D]}) -> {ty [B,tk,D]}\n" ++
            s!"    {clsb} = stablehlo.broadcast_in_dim {clsN}, dims = [2] : ({ty [D]}) -> {ty [B,1,D]}\n" ++
            s!"    {cat} = stablehlo.concatenate {clsb}, {tkn}, dim = 1 : ({ty [B,1,D]}, {ty [B,tk,D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {pb} = stablehlo.broadcast_in_dim {posN}, dims = [1, 2] : ({ty [tk+1,D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {ob} = stablehlo.add {cat}, {pb} : {ty [B,tk+1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,tk+1,D]}) -> {ty [B, (tk+1)*D]}\n", o :: st)
      -- ⚠⚠ **THE CONV SHAPE, NOT THE DOT SHAPE** — bf16 operands, **bf16-TYPED convolution
      -- result**, convert back. ViT's patchify stem is the one op in this net that is a
      -- `convolution`, and giving it an f32-typed result folds the whole thing to pure f32
      -- (measured standalone at this exact shape before the constructor existed, §17.2). Stride 16
      -- buys no exemption from §9.2 any more than grouping (§12.2) or stride 4 (§16.1) did.
      -- ▶ Everything after the convert-back — bias, transpose, CLS concat, position add — is
      -- byte-for-byte "patchEmbedP" and stays f32, which is what `patchEmbedFlatBf16`'s `den` says.
      | "patchEmbedPBf16", [wN, bN, clsN, posN], [_N, ic, H, W, P, tk, D] => do
          let hp := H / P; let wp := W / P
          let xn ← fresh; let xb ← fresh; let wb ← fresh; let cv ← fresh; let cf ← fresh
          let bb ← fresh; let cb ← fresh
          let tr ← fresh; let tkn ← fresh; let clsb ← fresh; let cat ← fresh
          let pb ← fresh; let ob ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
            s!"    {xb} = stablehlo.convert {xn} : ({ty [B,ic,H,W]}) -> {tyBf16 [B,ic,H,W]}\n" ++
            s!"    {wb} = stablehlo.convert {wN} : ({ty [D,ic,P,P]}) -> {tyBf16 [D,ic,P,P]}\n" ++
            s!"    {cv} = stablehlo.convolution({xb}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [{P}, {P}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,ic,H,W]}, {tyBf16 [D,ic,P,P]}) -> {tyBf16 [B,D,hp,wp]}\n" ++
            s!"    {cf} = stablehlo.convert {cv} : ({tyBf16 [B,D,hp,wp]}) -> {ty [B,D,hp,wp]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [D]}) -> {ty [B,D,hp,wp]}\n" ++
            s!"    {cb} = stablehlo.add {cf}, {bb} : {ty [B,D,hp,wp]}\n" ++
            s!"    {tr} = stablehlo.transpose {cb}, dims = [0, 2, 3, 1] : ({ty [B,D,hp,wp]}) -> {ty [B,hp,wp,D]}\n" ++
            s!"    {tkn} = stablehlo.reshape {tr} : ({ty [B,hp,wp,D]}) -> {ty [B,tk,D]}\n" ++
            s!"    {clsb} = stablehlo.broadcast_in_dim {clsN}, dims = [2] : ({ty [D]}) -> {ty [B,1,D]}\n" ++
            s!"    {cat} = stablehlo.concatenate {clsb}, {tkn}, dim = 1 : ({ty [B,1,D]}, {ty [B,tk,D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {pb} = stablehlo.broadcast_in_dim {posN}, dims = [1, 2] : ({ty [tk+1,D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {ob} = stablehlo.add {cat}, {pb} : {ty [B,tk+1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {ob} : ({ty [B,tk+1,D]}) -> {ty [B, (tk+1)*D]}\n", o :: st)
      | "clsSliceP", [], [_N, tk, D] => do
          let xn ← fresh; let sl ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, (tk+1)*D]}) -> {ty [B,tk+1,D]}\n" ++
            s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:1, 0:{D}] : ({ty [B,tk+1,D]}) -> {ty [B,1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {sl} : ({ty [B,1,D]}) -> {ty [B,D]}\n", o :: st)
      | "clsPadP", [], [_N, tk, D] => do
          let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B,D]}) -> {ty [B,1,D]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, 0], high = [0, {tk}, 0], interior = [0, 0, 0] : ({ty [B,1,D]}, tensor<f32>) -> {ty [B,tk+1,D]}\n" ++
            s!"    {o} = stablehlo.reshape {pd} : ({ty [B,tk+1,D]}) -> {ty [B, (tk+1)*D]}\n", o :: st)
      | "headSliceP", [], [_N, tk, heads, d, hIdx] => do
          let xn ← fresh; let sl ← fresh; let o ← fresh
          pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, tk*(heads*d)]}) -> {ty [B,tk,heads*d]}\n" ++
            s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:{tk}, {hIdx*d}:{(hIdx+1)*d}] : ({ty [B,tk,heads*d]}) -> {ty [B,tk,d]}\n" ++
            s!"    {o} = stablehlo.reshape {sl} : ({ty [B,tk,d]}) -> {ty [B, tk*d]}\n", o :: st)
      | "headPadP", [], [_N, tk, heads, d, hIdx] => do
          let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
          pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, tk*d]}) -> {ty [B,tk,d]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, {hIdx*d}], high = [0, 0, {(heads-1-hIdx)*d}], interior = [0, 0, 0] : ({ty [B,tk,d]}, tensor<f32>) -> {ty [B,tk,heads*d]}\n" ++
            s!"    {o} = stablehlo.reshape {pd} : ({ty [B,tk,heads*d]}) -> {ty [B, tk*(heads*d)]}\n", o :: st)
      -- ══ The un-fused BATCHED gradients: each is its `*SgdB` peer's emit with the SGD tail
      --    (const lr / multiply / subtract) removed, so the text is a byte-PREFIX of it. ══
      | "convWeightGrad", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh; let raw ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convWeightGradBf16", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [ic,B,h,w]}) -> {tyBf16 [ic,B,h,w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [oc,B,h,w]}) -> {tyBf16 [oc,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [ic,B,h,w]}, {tyBf16 [oc,B,h,w]}) -> {tyBf16 [ic,oc,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [ic,oc,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {rf}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convStridedWeightGrad", [xN], [_N, ic, oc, h, w, kH, kW] => do
          -- odd/even split via `sWGradGeom`; odd is byte-for-byte the old inline formula.
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convStridedWeightGradBf16", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [ic,B,2*h,2*w]}) -> {tyBf16 [ic,B,2*h,2*w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [oc,B,extH,extW]}) -> {tyBf16 [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [ic,B,2*h,2*w]}, {tyBf16 [oc,B,extH,extW]}) -> {tyBf16 [ic,oc,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [ic,oc,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {rf}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⭐ The XLA-`SAME` conv weight grad. Same `sWGradGeom` extents; only the correlation pad
      -- shifts by one (`loH-1`, `hiH+1`), so the saved input is read at `2·ho + 1 + kh - p`.
      | "convStridedXlaWeightGrad", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH-1}, {hiH+1}], [{loW-1}, {hiW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "convStridedXlaWeightGradBf16", [xN], [_N, ic, oc, h, w, kH, kW] => do
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [ic,B,2*h,2*w]}) -> {tyBf16 [ic,B,2*h,2*w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [oc,B,extH,extW]}) -> {tyBf16 [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH-1}, {hiH+1}], [{loW-1}, {hiW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [ic,B,2*h,2*w]}, {tyBf16 [oc,B,extH,extW]}) -> {tyBf16 [ic,oc,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [ic,oc,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {rf}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convStride4WeightGrad", [xN], [ic, oc, h, w, kH, kW] => do
          -- ConvNeXt's 4×4/s4 patchify weight grad. `flatConvStride4` decimates TWICE, so the
          -- cotangent is zero-upsampled with `interior = 3` (extent `4h−3`, no trailing row) and
          -- correlated VALID-style against the saved input at `4h`, giving `kH×kW`.
          -- The window offset: the stride-1 SAME conv is read at position `4i+1`, so a tap `kh`
          -- lands at `x[4i + 1 + kh − p]` with `p = (kH−1)/2` — hence `lo = p − 1`, and
          -- `hi = kH − 3 − p` makes the result exactly `kH` wide. At the 4×4 stem `p = 1`, so
          -- `[[0,0]]` and extent `4h−3` — byte-for-byte `ConvNeXtRender.patchWGrad`'s geometry.
          -- Nothing else in the kit is stride-4; this is exercised only at 4×4.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let extH := 4 * h - 3; let extW := 4 * w - 3
          let loH := pH - 1; let hiH := kH - 3 - pH
          let loW := pW - 1; let hiW := kW - 3 - pW
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh
          let dt ← fresh; let raw ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(4*h)*(4*w)]}) -> {ty [B,ic,4*h,4*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,4*h,4*w]}) -> {ty [ic,B,4*h,4*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,4*h,4*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back — measured on this exact
      -- shape (`[3,B,224,224]` × `[96,B,221,221]` → `[3,96,4,4]`) before the op was written.
      -- ⚠ Every geometry number is `convStride4WeightGrad`'s verbatim: the `interior = 3`
      -- upsample, the `4h−3` extent with NO trailing row, and the `lo = p−1` / `hi = kH−3−p`
      -- window. Only the four dtypes and the two converts move. ⚠ The `stablehlo.pad`'s zero stays
      -- f32 — it pads the cotangent BEFORE the cast, so it is an f32 tensor at that point.
      | "convStride4WeightGradBf16", [xN], [ic, oc, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let extH := 4 * h - 3; let extW := 4 * w - 3
          let loH := pH - 1; let hiH := kH - 3 - pH
          let loW := pW - 1; let hiW := kW - 3 - pW
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh
          let dt ← fresh; let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh
          let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(4*h)*(4*w)]}) -> {ty [B,ic,4*h,4*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,4*h,4*w]}) -> {ty [ic,B,4*h,4*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [ic,B,4*h,4*w]}) -> {tyBf16 [ic,B,4*h,4*w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [oc,B,extH,extW]}) -> {tyBf16 [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [ic,B,4*h,4*w]}, {tyBf16 [oc,B,extH,extW]}) -> {tyBf16 [ic,oc,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [ic,oc,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {o} = stablehlo.transpose {rf}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convBiasGrad", [], [_N, oc, h, w] => do
          let dr ← fresh; let z ← fresh; let o ← fresh
          pure (
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | "bnGammaGrad", [vN, es], [_N, oc, h, w] => do
          -- x̂ recomputed with μ/var over [0,2,3] — BATCH BN, not `bnGammaGrad`'s per-example [2,3].
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ep ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let dyr ← fresh; let dgp ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dgp} = stablehlo.multiply {dyr}, {xh} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reduce({dgp} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | "bnBetaGrad", [], [_N, oc, h, w] => do
          let dyr ← fresh; let z ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n", o :: st)
      | "denseWeightGrad", [xN], [_N, a, c] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [B,c]}) -> {ty [a,c]}\n", o :: st)
      | "denseBiasGrad", [], [_N, c] => do
          let z ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {o} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,c]}, tensor<f32>) -> {ty [c]}\n", o :: st)
      | "scale", [sS], [_N, n] => do
          let c ← fresh; let o ← fresh
          pure (s!"    {c} = stablehlo.constant dense<{sS}> : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.multiply {r}, {c} : {ty [B,n]}\n", o :: st)
      | "shift", [sS], [_N, n] => do
          let c ← fresh; let o ← fresh
          pure (s!"    {c} = stablehlo.constant dense<{sS}> : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.add {r}, {c} : {ty [B,n]}\n", o :: st)
      | "divConst", [sS], [_N, n] => do
          let c ← fresh; let o ← fresh
          pure (s!"    {c} = stablehlo.constant dense<{sS}> : {ty [B,n]}\n" ++
                s!"    {o} = stablehlo.divide {r}, {c} : {ty [B,n]}\n", o :: st)
      | "bnBatchMean", [], [_N, oc, h, w] => do
          -- μ_c = reduce[0,2,3](x) / (B·h·w). Numerically the `%{p}bnmu` the hand-written
          -- emitter divides out of its own BN fragment's `smr`.
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [oc]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {o} = stablehlo.divide {smr}, {nf} : {ty [oc]}\n", o :: st)
      | "bnBatchVar", [], [_N, oc, h, w] => do
          -- var_c = reduce[0,2,3]((x−μ)²) / (B·h·w), μ recomputed inline — the biased (÷n)
          -- variance `bnVar` uses, matching `%{p}bnvar`.
          let xr ← fresh; let z ← fresh; let nfb ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let nf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nfb} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nfb} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.divide {vsr}, {nf} : {ty [oc]}\n", o :: st)
      | "bnBatch", [gN, bN, es], [_N, oc, h, w] => do
          let xr ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh
          let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh
          let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let gb ← fresh; let btb ← fresh; let gx ← fresh; let o4 ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {btb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {gx} = stablehlo.multiply {xh}, {gb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o4} = stablehlo.add {gx}, {btb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {o4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | t, [gN, xN, es], [_N, oc, h, w] =>
          -- bnBatchBack / bnBatchLABack: the 3-term true-BN input-VJP. Self-contained
          -- recompute of x̂/istd from the saved BN input `xN` + γ `gN` + ε `es`
          -- (mnv2 pattern), then dx = (istd/nf)·(nf·(γ⊙dy) − Σ(γ⊙dy) − x̂·Σ(x̂·γ⊙dy)).
          -- `r` is the upstream cotangent dy. (dγ/dβ are param grads, not here.)
          if t == "bnBatchBack" || t == "bnBatchLABack" then do
            let xr ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh
            let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh
            let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
            let gb ← fresh; let dyr ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
            let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
            let xs ← fresh; let i2 ← fresh; let sN ← fresh; let dx4 ← fresh; let o ← fresh
            pure (
              s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
              s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
              s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
              s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
              s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
              s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
              s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
              s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
              s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {dxh} = stablehlo.multiply {gb}, {dyr} : {ty [B,oc,h,w]}\n" ++
              s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {xd} = stablehlo.multiply {xh}, {dxh} : {ty [B,oc,h,w]}\n" ++
              s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,oc,h,w]}\n" ++
              s!"    {xs} = stablehlo.multiply {xh}, {sxd} : {ty [B,oc,h,w]}\n" ++
              s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
              s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {dx4} = stablehlo.multiply {sN}, {i2} : {ty [B,oc,h,w]}\n" ++
              s!"    {o} = stablehlo.reshape {dx4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
          else
            pure (s!"    // [EfficientNet Item B] batched {tag} {names} {info} — render TODO\n", r :: st)
      | "convBackBatched", [wN], [_N, ic, oc, h, w, kH, kW] => do
          -- conv input-VJP: dx = conv(dy, reverse(W,[2,3])ᵀ), reversed+transposed
          -- kernel, stride 1, same-pad p. (1×1 in enet ⇒ p=0, reverse a no-op.)
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let wt ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({dyr}, {wt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,oc,h,w]}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,h,w]}) -> {ty [B, ic*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      | "convBackBatchedBf16", [wN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let wt ← fresh; let db ← fresh; let wb ← fresh
          let dx ← fresh; let xf ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {db} = stablehlo.convert {dyr} : ({ty [B,oc,h,w]}) -> {tyBf16 [B,oc,h,w]}\n" ++
            s!"    {wb} = stablehlo.convert {wt} : ({ty [ic,oc,kH,kW]}) -> {tyBf16 [ic,oc,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({db}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,oc,h,w]}, {tyBf16 [ic,oc,kH,kW]}) -> {tyBf16 [B,ic,h,w]}\n" ++
            s!"    {xf} = stablehlo.convert {dx} : ({tyBf16 [B,ic,h,w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {xf} : ({ty [B,ic,h,w]}) -> {ty [B, ic*h*w]}\n", o :: st)
      | "convStridedBackBatched", [wN], [_N, ic, oc, h, w, kH, kW] => do
          -- stride-2 conv input-VJP: upsample dy (zero-interleave to 2h×2w) then the
          -- stride-1 conv input-VJP. Produces dx at the 2h×2w input resolution.
          -- ⚠⚠ ASYMMETRIC pad, matching `.convStridedBack`. The symmetric `[[p,p],[p,p]]` this
          -- emitted AGREES at every odd kernel (kH=3 ⇒ pH=1 ⇒ kH−1−pH=1) and is WRONG at even
          -- ones (kH=2 ⇒ [[0,0]] where the VJP needs [[1,0]]). §2f-bis fixed exactly this in the
          -- per-example emitter and it was never carried here, because no batched net had an
          -- even strided kernel until ConvNeXt's 2×2/s2 downsample. Found by the whole-net
          -- backward tie — 3 lines, at the 3 downsamples. Inert on every committed batched
          -- artifact, all of which are odd (R34 3×3, mnv2/enet 3×3 and 5×5).
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let wt ← fresh
          let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({up}, {wt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{kH - 1 - pH}, {pH}], [{kW - 1 - pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,oc,2*h,2*w]}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w]}) -> {ty [B, ic*(2*h)*(2*w)]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 (measured) — see `flatConvFBf16`.
      -- ⚠ ASYMMETRIC pad, exactly as the f32 peer above — the bf16 twin must not "tidy" it.
      | "convStridedBackBatchedBf16", [wN], [_N, ic, oc, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let wt ← fresh
          let ub ← fresh; let wb ← fresh; let dx ← fresh; let xf ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {ub} = stablehlo.convert {up} : ({ty [B,oc,2*h,2*w]}) -> {tyBf16 [B,oc,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {wt} : ({ty [ic,oc,kH,kW]}) -> {tyBf16 [ic,oc,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({ub}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{kH - 1 - pH}, {pH}], [{kW - 1 - pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [B,oc,2*h,2*w]}, {tyBf16 [ic,oc,kH,kW]}) -> {tyBf16 [B,ic,2*h,2*w]}\n" ++
            s!"    {xf} = stablehlo.convert {dx} : ({tyBf16 [B,ic,2*h,2*w]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {xf} : ({ty [B,ic,2*h,2*w]}) -> {ty [B, ic*(2*h)*(2*w)]}\n", o :: st)
      | "depthwiseBackBatched", [wN], [_N, c, h, w, kH, kW] => do
          -- depthwise input-VJP: dx = depthwise_conv(dy, reverse(W,[2,3])), fgc=c,
          -- same-pad p (no transpose — one input channel per group).
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({dyr}, {rev})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,h,w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseBackBatchedBf16", [wN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let db ← fresh; let wb ← fresh
          let dx ← fresh; let xf ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {db} = stablehlo.convert {dyr} : ({ty [B,c,h,w]}) -> {tyBf16 [B,c,h,w]}\n" ++
            s!"    {wb} = stablehlo.convert {rev} : ({ty [c,1,kH,kW]}) -> {tyBf16 [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({db}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({tyBf16 [B,c,h,w]}, {tyBf16 [c,1,kH,kW]}) -> {tyBf16 [B,c,h,w]}\n" ++
            s!"    {xf} = stablehlo.convert {dx} : ({tyBf16 [B,c,h,w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {xf} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "depthwiseStridedBackBatched", [wN], [_N, c, h, w, kH, kW] => do
          -- stride-2 depthwise input-VJP: upsample dy then the stride-1 depthwise
          -- input-VJP. dx at the 2h×2w input resolution.
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({up}, {rev})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. The f32-result
      -- shape folds to pure f32, for grouped convolutions exactly as for ordinary ones
      -- (measured). ⚠ SYMMETRIC pad — this is the torchvision-origin variant, NOT `Xla`.
      | "depthwiseStridedBackBatchedBf16", [wN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh
          let db ← fresh; let wb ← fresh; let dx ← fresh; let xf ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {db} = stablehlo.convert {up} : ({ty [B,c,2*h,2*w]}) -> {tyBf16 [B,c,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {rev} : ({ty [c,1,kH,kW]}) -> {tyBf16 [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({db}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({tyBf16 [B,c,2*h,2*w]}, {tyBf16 [c,1,kH,kW]}) -> {tyBf16 [B,c,2*h,2*w]}\n" ++
            s!"    {xf} = stablehlo.convert {dx} : ({tyBf16 [B,c,2*h,2*w]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {xf} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      -- ⭐ The XLA-`SAME` depthwise input-VJP: conv pad shifts to `[p+1, p-1]`.
      -- ⚠⚠ **NOTE THE DIRECTION — it is the OPPOSITE of the two weight grads**, which shift to
      -- `[p-1, p+1]`. The kernel is REVERSED here (`stablehlo.reverse`, dims [2,3]), and that
      -- reversal flips the sign of the index shift. Deriving it "by symmetry" with the weight
      -- grads gives `[p-1, p+1]`, which type-checks, has the right shape, descends, and is WRONG
      -- — `scripts/xla_pad_op_check.py` caught exactly that (2.6e0 against both references) and
      -- a numeric sweep over (upsample phase, pad_low) pinned the true answer at both k=3 and
      -- k=5. Do not "fix" this to match its siblings. Total pad is `2p`, so the extent stays `2h`.
      | "depthwiseStridedXlaBackBatched", [wN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({up}, {rev})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p+1}, {p-1}], [{p+1}, {p-1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      -- ⚠⚠ Keeps the `[p+1, p-1]` pad of its f32 peer — the OPPOSITE shift from the weight
      -- grads, because the kernel is reversed. Do not "fix" it to match its siblings.
      | "depthwiseStridedXlaBackBatchedBf16", [wN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh
          let db ← fresh; let wb ← fresh; let dx ← fresh; let xf ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {db} = stablehlo.convert {up} : ({ty [B,c,2*h,2*w]}) -> {tyBf16 [B,c,2*h,2*w]}\n" ++
            s!"    {wb} = stablehlo.convert {rev} : ({ty [c,1,kH,kW]}) -> {tyBf16 [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({db}, {wb})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p+1}, {p-1}], [{p+1}, {p-1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({tyBf16 [B,c,2*h,2*w]}, {tyBf16 [c,1,kH,kW]}) -> {tyBf16 [B,c,2*h,2*w]}\n" ++
            s!"    {xf} = stablehlo.convert {dx} : ({tyBf16 [B,c,2*h,2*w]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {xf} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      | "seBackBatched", [w1, b1, w2, b2, vN], [_N, c, h, w, rr] => do
          -- SE backward: recompute the SE forward (GAP → dense W₁ b₁ → swish → dense
          -- W₂ b₂ → sigmoid gate) from the SE input `vN`, then the SE-input cotangent
          -- dx = gate⊙dse + GAP-adjoint(W₁ᵀ·swish'·W₂ᵀ·(gate·(1−gate))·Σ(x⊙dse)).
          -- `r` is the SE-output cotangent dse.
          let xr ← fresh; let z ← fresh; let sqs ← fresh; let sqnf ← fresh; let sq ← fresh
          let exd ← fresh; let exbb ← fresh; let ex ← fresh; let a1s ← fresh; let a1 ← fresh
          let h2d ← fresh; let h2bb ← fresh; let h2 ← fresh; let gate ← fresh
          let dser ← fresh; let gb2 ← fresh; let dleft ← fresh; let xdse ← fresh; let dgate ← fresh
          let one ← fresh; let omg ← fresh; let sg ← fresh; let dh2 ← fresh; let da1 ← fresh
          let dexs ← fresh; let dexone ← fresh; let dexom ← fresh; let dexxom ← fresh; let dexin ← fresh
          let dexsp ← fresh; let dex ← fresh; let dsq ← fresh; let dsqnf ← fresh; let dsqd ← fresh
          let dgsp ← fresh; let dds ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sqs} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {sqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {sq} = stablehlo.divide {sqs}, {sqnf} : {ty [B,c]}\n" ++
            s!"    {exd} = stablehlo.dot_general {sq}, {w1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [c,rr]}) -> {ty [B,rr]}\n" ++
            s!"    {exbb} = stablehlo.broadcast_in_dim {b1}, dims = [1] : ({ty [rr]}) -> {ty [B,rr]}\n" ++
            s!"    {ex} = stablehlo.add {exd}, {exbb} : {ty [B,rr]}\n" ++
            s!"    {a1s} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {a1} = stablehlo.multiply {ex}, {a1s} : {ty [B,rr]}\n" ++
            s!"    {h2d} = stablehlo.dot_general {a1}, {w2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [rr,c]}) -> {ty [B,c]}\n" ++
            s!"    {h2bb} = stablehlo.broadcast_in_dim {b2}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {h2} = stablehlo.add {h2d}, {h2bb} : {ty [B,c]}\n" ++
            s!"    {gate} = stablehlo.logistic {h2} : {ty [B,c]}\n" ++
            s!"    {dser} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {gb2} = stablehlo.broadcast_in_dim {gate}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dleft} = stablehlo.multiply {gb2}, {dser} : {ty [B,c,h,w]}\n" ++
            s!"    {xdse} = stablehlo.multiply {xr}, {dser} : {ty [B,c,h,w]}\n" ++
            s!"    {dgate} = stablehlo.reduce({xdse} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,c]}\n" ++
            s!"    {omg} = stablehlo.subtract {one}, {gate} : {ty [B,c]}\n" ++
            s!"    {sg} = stablehlo.multiply {gate}, {omg} : {ty [B,c]}\n" ++
            s!"    {dh2} = stablehlo.multiply {dgate}, {sg} : {ty [B,c]}\n" ++
            s!"    {da1} = stablehlo.dot_general {dh2}, {w2}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [rr,c]}) -> {ty [B,rr]}\n" ++
            s!"    {dexs} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {dexone} = stablehlo.constant dense<1.0> : {ty [B,rr]}\n" ++
            s!"    {dexom} = stablehlo.subtract {dexone}, {dexs} : {ty [B,rr]}\n" ++
            s!"    {dexxom} = stablehlo.multiply {ex}, {dexom} : {ty [B,rr]}\n" ++
            s!"    {dexin} = stablehlo.add {dexone}, {dexxom} : {ty [B,rr]}\n" ++
            s!"    {dexsp} = stablehlo.multiply {dexs}, {dexin} : {ty [B,rr]}\n" ++
            s!"    {dex} = stablehlo.multiply {da1}, {dexsp} : {ty [B,rr]}\n" ++
            s!"    {dsq} = stablehlo.dot_general {dex}, {w1}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [c,rr]}) -> {ty [B,c]}\n" ++
            s!"    {dsqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {dsqd} = stablehlo.divide {dsq}, {dsqnf} : {ty [B,c]}\n" ++
            s!"    {dgsp} = stablehlo.broadcast_in_dim {dsqd}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dds} = stablehlo.add {dleft}, {dgsp} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dds} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "bnGammaSgd", [gN, vN, es, lrS], [_N, oc, h, w] => do
          -- BN γ update: recompute x̂ from the BN input `vN`, dγ = reduce[0,2,3](dy⊙x̂),
          -- γ' = γ − lr·dγ. Output is the channel-shaped updated γ.
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ep ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let dyr ← fresh; let dgp ← fresh; let dg ← fresh; let lc ← fresh; let sc ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dgp} = stablehlo.multiply {dyr}, {xh} : {ty [B,oc,h,w]}\n" ++
            s!"    {dg} = stablehlo.reduce({dgp} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lc} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sc} = stablehlo.multiply {dg}, {lc} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {gN}, {sc} : {ty [oc]}\n", o :: st)
      | "bnBetaSgd", [bN, lrS], [_N, oc, h, w] => do
          -- BN β update: dβ = reduce[0,2,3](dy), β' = β − lr·dβ.
          let dyr ← fresh; let z ← fresh; let db ← fresh; let lc ← fresh; let sc ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lc} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sc} = stablehlo.multiply {db}, {lc} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sc} : {ty [oc]}\n", o :: st)
      | "denseWeightSgd", [xN, wN, lrS], [_N, a, c] => do
          -- dense weight update: dW = aᵀ·dy (dot_general contracts the batch), W' = W − lr·dW.
          let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {dW} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [B,c]}) -> {ty [a,c]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [a,c]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [a,c]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [a,c]}\n", o :: st)
      | "denseBiasSgd", [bN, lrS], [_N, c] => do
          -- dense bias update: dβ = reduce[0](dy) (sum over batch), β' = β − lr·dβ.
          let z ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dB} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "rowDenseWeightSgd", [xN, wN, lrS], [N, a, c] => do
          -- per-token (rowwise) dense weight grad: reshape activation/cotangent to the [B,N,·] token
          -- matrix, dW = Σ_{B,N} xᵀ·dy (contract batch×tokens [0,1]x[0,1]), W' = W − lr·dW.
          let xn ← fresh; let dn ← fresh; let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dW} = stablehlo.dot_general {xn}, {dn}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [B,N,a]}, {ty [B,N,c]}) -> {ty [a,c]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [a,c]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [a,c]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [a,c]}\n", o :: st)
      | "rowDenseBiasSgd", [bN, lrS], [N, c] => do
          -- per-token dense bias grad: db = reduce[0,1](dy) over batch×tokens ([B,N,c] → [c]),
          -- b' = b − lr·db. (Also the vector-LN β reduce.)
          let z ← fresh; let dn ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dn} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "patchEmbedBiasSgd", [bN, lrS], [N, c] => do
          -- patch-embed bias grad: slice the N patch tokens [1..N] from the embed cotangent (drop the
          -- CLS row 0), reduce[0,1] over batch×patches → [c], b' = b − lr·db.
          let z ← fresh; let dr ← fresh; let dsl ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*c]}) -> {ty [B, N+1, c]}\n" ++
            s!"    {dsl} = stablehlo.slice {dr} [0:{B}, 1:{N+1}, 0:{c}] : ({ty [B,N+1,c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dsl} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "posEmbedSgd", [pN, lrS], [N, D] => do
          -- pos-embed grad: reshape the embed cotangent [B,(N+1)*D] → [B,N+1,D], reduce ONLY the batch
          -- axis [0] (KEEP all N+1 tokens) → [N+1,D], pos' = pos − lr·dpos. (identity pos-Jacobian.)
          let z ← fresh; let dr ← fresh; let dP ← fresh; let lP ← fresh; let sP ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dP} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,N+1,D]}, tensor<f32>) -> {ty [N+1,D]}\n" ++
            s!"    {lP} = stablehlo.constant dense<{lrS}> : {ty [N+1,D]}\n" ++
            s!"    {sP} = stablehlo.multiply {dP}, {lP} : {ty [N+1,D]}\n" ++
            s!"    {o} = stablehlo.subtract {pN}, {sP} : {ty [N+1,D]}\n", o :: st)
      -- ── the un-fused transformer gradients ─────────────────────────────────────────────────
      -- Each is its `*Sgd` peer above with the trailing `constant lr / multiply / subtract` cut
      -- off, so the emitted text is a byte PREFIX of the fused one. `tests/TestBatchedEmitTie.lean`
      -- checks exactly that, which is the emit-side twin of the `*Sgd_eq_grad` theorems.
      | "rowDenseWeightGrad", [xN], [N, a, c] => do
          let xn ← fresh; let dn ← fresh; let dW ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dW} = stablehlo.dot_general {xn}, {dn}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({ty [B,N,a]}, {ty [B,N,c]}) -> {ty [a,c]}\n", dW :: st)
      -- ⚠ Dot shape. The contraction is over BOTH the batch and the token axis in one op, so the
      -- f32-typed result IS the accumulator for the whole reduction — which is what keeps this
      -- gradient out of §9.3's vacuity argument (a bf16 accumulate at this fan-in would be).
      | "rowDenseWeightGradBf16", [xN], [N, a, c] => do
          let xn ← fresh; let dn ← fresh; let xb ← fresh; let db ← fresh; let dW ← fresh
          pure (
            s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {xb} = stablehlo.convert {xn} : ({ty [B,N,a]}) -> {tyBf16 [B,N,a]}\n" ++
            s!"    {db} = stablehlo.convert {dn} : ({ty [B,N,c]}) -> {tyBf16 [B,N,c]}\n" ++
            s!"    {dW} = stablehlo.dot_general {xb}, {db}, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : ({tyBf16 [B,N,a]}, {tyBf16 [B,N,c]}) -> {ty [a,c]}\n", dW :: st)
      | "rowDenseBiasGrad", [], [N, c] => do
          let z ← fresh; let dn ← fresh; let dB ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dn} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n", dB :: st)
      | "patchEmbedBiasGrad", [], [N, c] => do
          let z ← fresh; let dr ← fresh; let dsl ← fresh; let dB ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*c]}) -> {ty [B, N+1, c]}\n" ++
            s!"    {dsl} = stablehlo.slice {dr} [0:{B}, 1:{N+1}, 0:{c}] : ({ty [B,N+1,c]}) -> {ty [B,N,c]}\n" ++
            s!"    {dB} = stablehlo.reduce({dsl} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,c]}, tensor<f32>) -> {ty [c]}\n", dB :: st)
      | "posEmbedGrad", [], [N, D] => do
          let z ← fresh; let dr ← fresh; let dP ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dP} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,N+1,D]}, tensor<f32>) -> {ty [N+1,D]}\n", dP :: st)
      | "veclnGammaGrad", [xN, epsStr], [N, D] => do
          let x3 ← fresh; let d3 ← fresh
          let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
          let dg ← fresh
          pure (
            s!"    {x3} = stablehlo.reshape {xN} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
            s!"    {d3} = stablehlo.reshape {r} : ({ty [B, N*D]}) -> {ty [B,N,D]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{D}.0> : {ty [B,N,D]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,N,D]}\n" ++
            s!"    {smr} = stablehlo.reduce({x3} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,N,D]}\n" ++
            s!"    {xc} = stablehlo.subtract {x3}, {mu} : {ty [B,N,D]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,N,D]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,N,D]}, tensor<f32>) -> {ty [B,N]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,N]}) -> {ty [B,N,D]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,N,D]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,N,D]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,N,D]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,N,D]}\n" ++
            s!"    {p} = stablehlo.multiply {d3}, {xh} : {ty [B,N,D]}\n" ++
            s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,N,D]}, tensor<f32>) -> {ty [D]}\n", dg :: st)
      | "patchEmbedWeightGrad", [xN], [ic, H, W, P, N, D] => do
          let ph := H / P; let pw := W / P
          let dilH := H - (P - 1); let dilW := W - (P - 1)
          let zc ← fresh; let dtr ← fresh; let dsl ← fresh; let drs ← fresh; let dy3 ← fresh
          let u ← fresh; let xt ← fresh; let dt ← fresh; let raw ← fresh; let dw ← fresh
          pure (
            s!"    {zc} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dtr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dsl} = stablehlo.slice {dtr} [0:{B}, 1:{N+1}, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,N,D]}\n" ++
            s!"    {drs} = stablehlo.reshape {dsl} : ({ty [B,N,D]}) -> {ty [B,ph,pw,D]}\n" ++
            s!"    {dy3} = stablehlo.transpose {drs}, dims = [0, 3, 1, 2] : ({ty [B,ph,pw,D]}) -> {ty [B,D,ph,pw]}\n" ++
            s!"    {u} = stablehlo.pad {dy3}, {zc}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {P-1}, {P-1}] : ({ty [B,D,ph,pw]}, tensor<f32>) -> {ty [B,D,dilH,dilW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xN}, dims = [1, 0, 2, 3] : ({ty [B,ic,H,W]}) -> {ty [ic,B,H,W]}\n" ++
            s!"    {dt} = stablehlo.transpose {u}, dims = [1, 0, 2, 3] : ({ty [B,D,dilH,dilW]}) -> {ty [D,B,dilH,dilW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,H,W]}, {ty [D,B,dilH,dilW]}) -> {ty [ic,D,P,P]}\n" ++
            s!"    {dw} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,D,P,P]}) -> {ty [D,ic,P,P]}\n", dw :: st)
      -- ⚠⚠ **CONV shape** — the second of ViT's two convolutions, and the second place the result
      -- type is load-bearing. The pad/transpose preamble is byte-for-byte "patchEmbedWeightGrad";
      -- the two converts go on the CONVOLUTION's operands only, after the dilating pad, because
      -- that pad is exact data movement and casting before it would round the same values twice.
      -- ⚠ The convolution contracts the BATCH axis (`[ic,B,H,W] × [D,B,dilH,dilW]`), so the single
      -- bf16 store lands on the already-summed gradient — which is exactly where
      -- `patchEmbedWeightGradBBf16`'s `den` puts its outer `rnd`.
      | "patchEmbedWeightGradBf16", [xN], [ic, H, W, P, N, D] => do
          let ph := H / P; let pw := W / P
          let dilH := H - (P - 1); let dilW := W - (P - 1)
          let zc ← fresh; let dtr ← fresh; let dsl ← fresh; let drs ← fresh; let dy3 ← fresh
          let u ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let dw ← fresh
          pure (
            s!"    {zc} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dtr} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B, N+1, D]}\n" ++
            s!"    {dsl} = stablehlo.slice {dtr} [0:{B}, 1:{N+1}, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,N,D]}\n" ++
            s!"    {drs} = stablehlo.reshape {dsl} : ({ty [B,N,D]}) -> {ty [B,ph,pw,D]}\n" ++
            s!"    {dy3} = stablehlo.transpose {drs}, dims = [0, 3, 1, 2] : ({ty [B,ph,pw,D]}) -> {ty [B,D,ph,pw]}\n" ++
            s!"    {u} = stablehlo.pad {dy3}, {zc}, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {P-1}, {P-1}] : ({ty [B,D,ph,pw]}, tensor<f32>) -> {ty [B,D,dilH,dilW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xN}, dims = [1, 0, 2, 3] : ({ty [B,ic,H,W]}) -> {ty [ic,B,H,W]}\n" ++
            s!"    {dt} = stablehlo.transpose {u}, dims = [1, 0, 2, 3] : ({ty [B,D,dilH,dilW]}) -> {ty [D,B,dilH,dilW]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [ic,B,H,W]}) -> {tyBf16 [ic,B,H,W]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [D,B,dilH,dilW]}) -> {tyBf16 [D,B,dilH,dilW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [ic,B,H,W]}, {tyBf16 [D,B,dilH,dilW]}) -> {tyBf16 [ic,D,P,P]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [ic,D,P,P]}) -> {ty [ic,D,P,P]}\n" ++
            s!"    {dw} = stablehlo.transpose {rf}, dims = [1, 0, 2, 3] : ({ty [ic,D,P,P]}) -> {ty [D,ic,P,P]}\n", dw :: st)
      | "convWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          -- conv weight update via the transpose-trick wgrad (batch as the conv
          -- contraction), then W' = W − lr·dW. Same text as the per-example convWeightSgd.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convStridedWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          -- stem 3×3 s2 weight: zero-upsample dy to 2h×2w then the transpose-trick wgrad.
          -- odd/even split via `sWGradGeom`; odd is byte-for-byte the old inline formula.
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH}, {hiH}], [{loW}, {hiW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      -- The XLA-`SAME` peer: identical, with the weight-grad correlation pad shifted one
      -- (`loH-1`, `hiH+1`) so the saved input is read at `2*ho + 1 + kh - p`.
      | "convStridedXlaWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          let (upH, extH, loH, hiH) := sWGradGeom kH h
          let (upW, extW, loW, hiW) := sWGradGeom kW w
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, {upH}, {upW}], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,extH,extW]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,extH,extW]}) -> {ty [oc,B,extH,extW]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{loH-1}, {hiH+1}], [{loW-1}, {hiW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,extH,extW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      | "depthwiseWeightSgd", [xN, wN, lrS], [_N, c, h, w, kH, kW] => do
          -- depthwise weight: per-channel transpose-trick wgrad (batch_group_count=c).
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
      | "depthwiseStridedWeightSgd", [xN, wN, lrS], [_N, c, h, w, kH, kW] => do
          -- strided depthwise weight: upsample dy to 2h×2w then the per-channel wgrad.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
      -- ── the ConvNeXt five. Each is its `*Sgd` peer's emit with the const-lr / multiply /
      --    subtract tail cut off, so each render is a byte-PREFIX of the fused one. ──
      | "depthwiseBiasGrad", [], [c, h, w, _kH, _kW] => do
          -- depthwise bias grad: Σ_{batch,spatial} dy, per channel. Note the RESHAPE precedes the
          -- zero constant — that is `depthwiseBiasSgd`'s order, and the emit-prefix test in
          -- `tests/TestBatchedEmitTie.lean` fails if the two are emitted the other way round even
          -- though the MLIR would be equivalent. (It caught exactly that here.)
          let dr ← fresh; let z ← fresh; let db ← fresh
          pure (
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n", db :: st)
      | "lnGammaGrad", [xN, epsStr], [n] => do
          -- scalar-LN γ grad: recompute x̂ from the saved LN input, dγ = Σ_{b,k} dy·x̂ → tensor<f32>.
          let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
          let dg ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
            s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
            s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
            s!"    {p} = stablehlo.multiply {r}, {xh} : {ty [B,n]}\n" ++
            s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n", dg :: st)
      | "lnBetaGrad", [], [n] => do
          -- scalar-LN β grad: dβ = Σ_{b,k} dy → tensor<f32> (rank-0, the scalar-LN param shape).
          let z ← fresh; let db ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n", db :: st)
      | "layerScaleChGammaGrad", [xN], [c, h, w] => do
          -- per-channel layer-scale γ grad: dγ_c = reduce[0,2,3](x ⊙ dy).
          let z ← fresh; let xr ← fresh; let dr ← fresh; let p ← fresh; let dg ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {p} = stablehlo.multiply {xr}, {dr} : {ty [B,c,h,w]}\n" ++
            s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n", dg :: st)
      -- The PER-EXAMPLE depthwise weight grad — five nats, where the batched form below has six.
      -- Same emitted text (that emitter ignores its `N`), so this shares the body by construction.
      | "depthwiseWeightGrad", [xN], [c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      | "depthwiseWeightGrad", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      | "depthwiseWeightGradBf16", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [c,B,h,w]}) -> {tyBf16 [c,B,h,w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [c,B,h,w]}) -> {tyBf16 [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [c,B,h,w]}, {tyBf16 [c,B,h,w]}) -> {tyBf16 [1,c,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [1,c,kH,kW]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {rf} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. An f32-typed
      -- result reads identically and compiles to pure f32 — measured on a real grouped
      -- (depthwise) conv too, so `feature_group_count` buys no exemption. See §9.2.
      -- ⚠ Keeps the `[p-1, p+1]` weight-grad pad — the opposite direction from the dgrad.
      | "depthwiseStridedXlaWeightGradBf16", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [c,B,2*h,2*w]}) -> {tyBf16 [c,B,2*h,2*w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [c,B,2*h,2*w]}) -> {tyBf16 [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH-1}, {pH+1}], [{pW-1}, {pW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [c,B,2*h,2*w]}, {tyBf16 [c,B,2*h,2*w]}) -> {tyBf16 [1,c,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [1,c,kH,kW]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {rf} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      | "depthwiseStridedWeightGrad", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⚠ bf16 operands, **bf16-typed convolution result**, convert back. The f32-result
      -- shape folds to pure f32, for grouped convolutions exactly as for ordinary ones
      -- (measured). ⚠ SYMMETRIC pad — this is the torchvision-origin variant, NOT `Xla`.
      | "depthwiseStridedWeightGradBf16", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let xb ← fresh; let db ← fresh; let raw ← fresh; let rf ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {xb} = stablehlo.convert {xt} : ({ty [c,B,2*h,2*w]}) -> {tyBf16 [c,B,2*h,2*w]}\n" ++
            s!"    {db} = stablehlo.convert {dt} : ({ty [c,B,2*h,2*w]}) -> {tyBf16 [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xb}, {db})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({tyBf16 [c,B,2*h,2*w]}, {tyBf16 [c,B,2*h,2*w]}) -> {tyBf16 [1,c,kH,kW]}\n" ++
            s!"    {rf} = stablehlo.convert {raw} : ({tyBf16 [1,c,kH,kW]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {rf} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      -- ⭐ The XLA-`SAME` depthwise weight grad — the same one-position shift.
      | "depthwiseStridedXlaWeightGrad", [xN], [_N, c, h, w, kH, kW] => do
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH-1}, {pH+1}], [{pW-1}, {pW+1}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n", g :: st)
      | "seReduceB", [xN], [_N, c, h, w] => do
          -- SE gate cotangent: dgate = reduce[2,3](x ⊙ dy). `xN` = SE input, `r` = the
          -- SE-output cotangent dy. Output is the per-example per-channel gate cotangent
          -- [B,c] (= the broadcast-adjoint of the Hadamard x⊙dy). Feeds the SE param grads.
          let xr ← fresh; let dyr ← fresh; let z ← fresh; let xd ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {xd} = stablehlo.multiply {xr}, {dyr} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n", o :: st)
      | "gapBackBatched", [], [_N, c, h, w] => do
          -- GAP backward: broadcast the per-channel cotangent `r` ([B,c]) over the h×w
          -- grid and scale by 1/(h·w) — the `globalAvgPoolFlat` adjoint, batched.
          let bb ← fresh; let nf ← fresh; let dv ← fresh; let o ← fresh
          pure (
            s!"    {bb} = stablehlo.broadcast_in_dim {r}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c,h,w]}\n" ++
            s!"    {dv} = stablehlo.divide {bb}, {nf} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dv} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | _, _, _ =>
          pure (s!"    // [EfficientNet Item B] batched {tag} {names} {info} — backward render TODO\n", r :: st)
  | .batched2 tag _names info, b :: a :: st =>
      -- Pointwise binary ops at the batched index. Byte-for-byte the `.addV`/`.sub`
      -- emits; the width is `info`'s per-example `n`, not the SHlo index `N·n`.
      match tag, info with
      | "addV", [_N, n] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.add {a}, {b} : {ty [B,n]}\n", o :: st)
      | "sub", [_N, n] => do
          let o ← fresh
          pure (s!"    {o} = stablehlo.subtract {a}, {b} : {ty [B,n]}\n", o :: st)
      -- ⭐⭐ **The only NON-pointwise `batched2`, and the only ACTIVATION × ACTIVATION bf16 op in
      -- the kit** — SDPA's `QKᵀ` and `P·V`, plus the four backward matmuls. It rides this binary
      -- skeleton rather than aliasing `.matmulF` because its text differs; `info` is `[m, k, n]`
      -- (no batch), since `B` is `pretty`'s exactly as it is for every other case here.
      -- ⚠ Both operand converts are on VALUES, not on a weight — nothing in the emit cares, and
      -- neither does `dot_close_mixed`, which rounds both sides.
      | "matmulFBf16", [m, k, n] => do
          let an ← fresh; let bn ← fresh; let ab ← fresh; let bb ← fresh
          let mm ← fresh; let mf ← fresh; let o ← fresh
          pure (s!"    {an} = stablehlo.reshape {a} : ({ty [B, m*k]}) -> {ty [B,m,k]}\n" ++
            s!"    {bn} = stablehlo.reshape {b} : ({ty [B, k*n]}) -> {ty [B,k,n]}\n" ++
            s!"    {ab} = stablehlo.convert {an} : ({ty [B,m,k]}) -> {tyBf16 [B,m,k]}\n" ++
            s!"    {bb} = stablehlo.convert {bn} : ({ty [B,k,n]}) -> {tyBf16 [B,k,n]}\n" ++
            s!"    {mm} = stablehlo.dot_general {ab}, {bb}, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({tyBf16 [B,m,k]}, {tyBf16 [B,k,n]}) -> {tyBf16 [B,m,n]}\n" ++
            s!"    {mf} = stablehlo.convert {mm} : ({tyBf16 [B,m,n]}) -> {ty [B,m,n]}\n" ++
            s!"    {o} = stablehlo.reshape {mf} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
      | _, _ => pure (s!"    // MALFORMED batched2 {tag} {info}\n", a :: st)
  | _, st => pure ("    // MALFORMED token stream\n", st)

/-- Fold a token stream to accumulated `(code, result-name-stack)`. -/
def serializeToks (B : Nat) : List Tok → (String × List String) → StateM Nat (String × List String)
  | [], acc           => pure acc
  | t :: ts, (code, st) => do
      let (c, st') ← emitTok B t st
      serializeToks B ts (code ++ c, st')

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
    ablation on BN β), not because they stay zero. See `tests/TestConvBiasZero.lean`. -/
def biasName (convBias : Bool) (nm : String) (c : Nat) : String :=
  if convBias then nm else s!"%zb{c}"

/-- The bias's slot in a **return-name list**, gated the way `biasName` gates the operand: with
    `convBias := false` no bias SGD op is emitted, so the slot must LEAVE the list rather than carry
    the empty string the `if convBias then … else pure ("", "")` idiom hands back.

    ⚠ This exists because leaving it in is **silent twice over**. An empty name renders
    `return %a, , %b` — malformed text, but only the lowerer ever sees it; and the name list keeps
    its FULL length, so an arity `#guard` on the signature still passes. Measured on the first swap
    attempt: `mobilenetv2_train_step` at `convBias := false` returned 210 names (52 of them empty)
    against 160 types. Use this at every site where a `names := [...]` list is built from gated ops. -/
def biasSlot (convBias : Bool) (nm : String) : List String :=
  if convBias then [nm] else []

/-- The zero-bias constants the `convBias := false` render consumes, one per channel width used as
    a conv bias. Emitted once at the top of the body; XLA folds the resulting `add`. -/
def zeroBiasPrelude (convBias : Bool) (widths : List Nat) : String :=
  if convBias then "" else
    "    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s\n" ++
    "    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a\n" ++
    "    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.\n" ++
    String.join (widths.map (fun c =>
      s!"    %zb{c} = stablehlo.constant dense<0.0> : {ty [c]}\n"))

/-- Fixed-6-decimal float literal, so a computed smoothing constant emits in the SAME textual form
    the hand-written literals used and `nClasses = 10` re-renders byte-identical. -/
def fmt6 (x : Float) : String :=
  let neg := x < 0.0
  let n := ((if neg then -x else x) * 1000000.0 + 0.5).toUInt64.toNat
  let ip := n / 1000000
  let fp := n % 1000000
  let fs := (toString fp).leftpad 6 '0'
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

/-- Fixed-12-decimal float literal, for constants `fmt6` would destroy.

    ⚠ It exists because `fmt6` is not a formatting preference, it is a PRECISION CEILING, and small
    derived constants fall straight through it. Gradient accumulation's second-moment coefficient is
    `(1−β₂)/K²`; at K = 4 that is `6.25e-5`, which `fmt6` emits as `0.000063` — **0.8% wrong**, in a
    baked literal, in the optimizer, where nothing downstream would question it. Same class as §2k's
    hardcoded `0.010000` label-smoothing mass. `fmt6` stays the default so every committed artifact
    re-renders byte-identically; this is for constants that need the room. -/
def fmt12 (x : Float) : String :=
  let neg := x < 0.0
  let n := ((if neg then -x else x) * 1000000000000.0 + 0.5).toUInt64.toNat
  let ip := n / 1000000000000
  let fp := n % 1000000000000
  let fs := (toString fp).leftpad 12 '0'
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

/-- **The label-smoothing mass per class, α/K.** α = 0.1 throughout; K is `nClasses`.

    ⚠ **This was hardcoded `0.010000` — correct at K = 10 and WRONG at every other K**, and it sat
    in the COTANGENT, not just in the report-only `%loss`. At `nClasses = 1000` it made the smoothing
    term 100× too large: it removes 10.0 of probability mass instead of 0.1, i.e. a different
    objective, silently. Caught 2026-07-30 by the first ImageNet smoke run reporting loss ≈ 87 where
    1000-class CE at init must be ≈ ln(1000) = 6.9 — the number was implausible, and that is the only
    reason it surfaced. Nothing in the repo's proofs covers it: `α` is a *literal in emitted text*,
    which is exactly the carve-out class §5 says needs its own numeric check, and §2b's `%loss` bug
    is the standing precedent for it going wrong unnoticed. -/
def alphaOverK (nClasses : Nat) (alpha : Float := 0.1) : String :=
  fmt6 (alpha / nClasses.toFloat)

/-- `1 − α`, the ON-class weight of label-smoothed CE. Emitted beside `alphaOverK`, because the two
    always move together and splitting them is how one of them gets updated alone. -/
def oneMinusAlpha (alpha : Float := 0.1) : String := fmt6 (1.0 - alpha)

/-- **`1 − ρ`, the RMSProp mean-square mixing weight.** Derived from ρ, never written as a second
    literal beside it — the `oneMinusAlpha` precedent, and the K-constant lesson (§2k): *any
    emitted constant that depends on a hyperparameter must be DERIVED*, because the copy is what
    gets left behind when the original moves. Five copies of one label-smoothing constant were
    found across four nets in a single session for exactly this reason. -/
def oneMinusRho (rho : Float) : String := fmt6 (1.0 - rho)

/-- Which optimizer tail a whole-net render emits. `.adamw` is every net's committed default and
    reproduces the existing artifacts byte-identically; `.rmsprop` is what the MobileNetV2 and
    EfficientNet ImageNet references actually use (`planning/recipe_gaps.md` v1.2).

    Lives here rather than in either renderer because **both** need it: a per-net copy of a
    two-constructor choice is the double-writer disease one level down, in code — the same argument
    `vitBackAll`/`enetBackAll` exist for (§2a-quater). Each renderer threads it through ONE
    traversal, so gate 1 applies for free: at `.adamw` every committed artifact must re-render
    byte-identical. -/
inductive OptKind where
  | adamw
  | rmsprop
deriving DecidableEq, Repr

/-- The RMSProp hyperparameters, as the JAX reference configs state them. `ρ`/`μ` are 0.9 on both
    nets that use this optimizer; **ε and wd are what differ**, and ε differs in the way that
    matters most (see `Proofs.rmsBufNext_eps_placement_at_zero`). -/
structure RmsHyper where
  /-- `rmspropDecay` — the running mean-square decay. -/
  rho : Float := 0.9
  /-- `momentum` — μ for the buffer on the normalised gradient. -/
  mu  : Float := 0.9
  /-- `rmspropEps` — ⚠ emitted INSIDE the square root (TensorFlow), not added to the root. -/
  eps : Float
  /-- COUPLED L2 (folded into the gradient), not AdamW's decoupled decay. -/
  wd  : Float

/-- ρ / (1−ρ) / μ / ε / wd as graph constants — the RMSProp peer of each renderer's `adamConsts`
    block. `%lr` stays a runtime `tensor<f32>` arg so one graph serves a whole LR schedule. -/
def rmsConstsBlock (h : RmsHyper) : String :=
  s!"    %rho = stablehlo.constant dense<{fmt6 h.rho}> : tensor<f32>\n" ++
  s!"    %orho = stablehlo.constant dense<{oneMinusRho h.rho}> : tensor<f32>\n" ++
  s!"    %mu = stablehlo.constant dense<{fmt6 h.mu}> : tensor<f32>\n" ++
  s!"    %eps = stablehlo.constant dense<{fmt6 h.eps}> : tensor<f32>\n" ++
  s!"    %wd = stablehlo.constant dense<{fmt6 h.wd}> : tensor<f32>\n"

/-- **MobileNetV2's RMSProp knobs** (`jax/MainMobilenetV2Imagenet.lean`): ε = **1.0**. -/
def mnv2RmsHyper : RmsHyper := { eps := 1.0, wd := 4.0e-5 }

/-- **EfficientNet-B0's RMSProp knobs** (`jax/MainEfficientNetImagenet.lean`): ε = **1e-3**. -/
def enetRmsHyper : RmsHyper := { eps := 1.0e-3, wd := 1.0e-5 }

-- ▶ The DRIVER-side half of the same two recipes — peak LR, exponential decay, warmup — is
-- `RmsSchedule` in `LeanMlir/VerifiedNets.lean`, deliberately NOT here. Two reasons, and the second
-- is the load-bearing one: the four trainer entry points that read it would otherwise have to
-- import this whole proof module, and nothing that lives in this file can reach `rmsConstsBlock`
-- by accident. `%lr` is a runtime `tensor<f32>` argument precisely so one graph serves a whole
-- schedule; a learning rate must never become a graph constant.

-- ⚠ `fmt6` is a SIX-DECIMAL fixed-point formatter, so any hyperparameter below 5e-7 silently
-- renders as `0.000000` — a graph constant of zero, which for `wd` is "no weight decay" and for
-- `eps` is a divide-by-zero at a dead coordinate. Neither is a compile error and neither is
-- visible in a green build. These pin every constant these two nets actually emit; add a line
-- here before adding a third net's knobs.
#guard fmt6 mnv2RmsHyper.eps == "1.000000"
#guard fmt6 mnv2RmsHyper.wd  == "0.000040"
#guard fmt6 enetRmsHyper.eps == "0.001000"
#guard fmt6 enetRmsHyper.wd  == "0.000010"
#guard oneMinusRho 0.9 == "0.100000"
#guard fmt6 (0.9 : Float) == "0.900000"

/-- **`pretty`** — render an `SHlo` graph to StableHLO, now defined as
    `serialize ∘ toToks ∘ skel`: tokenize the graph (postorder), then print the
    tokens. The emitter shares ONE structured form with the parser, so the
    round-trip `parse (toToks (skel a)) = skel a` (StableHLOParse.lean) is about
    the very tokens this prints — the printer can't structurally drift. -/
def pretty (B : Nat) {k : Nat} (g : SHlo k) : StateM Nat (String × String) := do
  let (code, st) ← serializeToks B (toToks (skel g)) ("", [])
  match st with
  | [r] => pure (code, r)
  | _   => pure (code, "%MALFORMED")

/-- Wrap a rendered single-result graph as a `func.func` module. -/
def renderModule (name argSig : String) (B retLen : Nat) (g : SHlo retLen) : String :=
  let (body, res) := (pretty B g).run' 0
  "module @m {\n" ++ s!"  func.func @{name}({argSig}) -> {ty [B, retLen]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [B, retLen]}\n" ++ "  }\n}\n"

/-- `@linear_fwd` rendered **from the verified AST**. -/
def linearFwdModuleV (B d₀ d₁ : Nat) (W : Mat d₀ d₁) (b : Vec d₁) (x : Vec d₀) : String :=
  renderModule "linear_fwd" s!"%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}" B d₁ (fwdGraph W b x)

/-- `@linear_back` rendered **from the verified AST**. -/
def linearBackModuleV (B d₀ d₁ : Nat) (W : Mat d₀ d₁) (dy : Vec d₁) : String :=
  renderModule "linear_back" s!"%dy: {ty [B,d₁]}, %W0: {ty [d₀,d₁]}" B d₀ (backGraph W dy)

/-- The full **`@linear_train_step`** rendered from the verified AST: forward +
    softmax-CE cotangent come from `pretty (lossCotGraph …)` (the `%onehot`
    operand value is `pretty`-irrelevant, so any placeholder renders the same
    text — at runtime `%onehot` is a graph input); the weight grad
    (`dot_general` over the batch axis), bias grad (`reduce`), and the SGD
    `multiply`/`subtract` updates are appended. Returns the two updated params.
    The verified-AST peer of `IRPrint.linearTrainStepModule`. -/
def linearTrainStepModuleV (B d₀ d₁ : Nat) (lr : String)
    (W : Mat d₀ d₁) (b : Vec d₁) (x : Vec d₀) : String :=
  let (body, dy) := (pretty B (lossCotGraph W b x (fun _ => 0))).run' 0
  "module @m {\n" ++
  s!"  func.func @linear_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, " ++
  s!"%onehot: {ty [B,d₁]}) -> ({ty [d₀,d₁]}, {ty [d₁]}) " ++ "{\n" ++
  "    // ── forward + softmax-CE cotangent — rendered from the verified AST (lossCotGraph) ──\n" ++
  body ++
  s!"    // dy = {dy} = ⟦lossCotGraph⟧ = ∂CE/∂logits (lossCotGraph_isCEgrad)\n" ++
  "    // ── param grads: dW0 = x⊗dy, db0 = Σ_batch dy (wGrad/bGrad_is*Jacobian) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %dW0 = stablehlo.dot_general %x, {dy}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,d₀]}, {ty [B,d₁]}) -> {ty [d₀,d₁]}\n" ++
  s!"    %db0 = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,d₁]}, tensor<f32>) -> {ty [d₁]}\n" ++
  "    // ── SGD update θ' = θ − lr·∇ (sgdW/sgdB_descends_certified_grad) ──\n" ++
  s!"    %lW0 = stablehlo.constant dense<{lr}> : {ty [d₀,d₁]}\n" ++
  s!"    %sW0 = stablehlo.multiply %dW0, %lW0 : {ty [d₀,d₁]}\n" ++
  s!"    %W0n = stablehlo.subtract %W0, %sW0 : {ty [d₀,d₁]}\n" ++
  s!"    %lb0 = stablehlo.constant dense<{lr}> : {ty [d₁]}\n" ++
  s!"    %sb0 = stablehlo.multiply %db0, %lb0 : {ty [d₁]}\n" ++
  s!"    %b0n = stablehlo.subtract %b0, %sb0 : {ty [d₁]}\n" ++
  s!"    return %W0n, %b0n : {ty [d₀,d₁]}, {ty [d₁]}\n" ++
  "  }\n}\n"

/-- **The linear train step rendered ENTIRELY from the verified AST.** Unlike
    `linearTrainStepModuleV` (forward via `pretty`, tail hand-written), here the
    *whole* module is `pretty` of denoted nodes: the cotangent (`lossCotGraph`,
    rendered once → shared `%dy`), then the two fused SGD ops `weightSgd`/`biasSgd`
    that consume `%dy`. So every emitted line is `pretty(provenNode)` and
    `LinearFaithfulPoC` proves the two outputs' `den` = the certified loss-descent
    SGD step. The `lr` ℝ / operand values are `skel`-erased (render is
    value-independent), so placeholders here render identically to the live graph
    the `den` theorems use. -/
def linTrainStepFaithfulV (B m n : Nat) (lrStr : String)
    (W : Mat m n) (b : Vec n) (x : Vec m) : String :=
  -- FULLY TIED: each SGD op consumes the proven `lossCotGraph` node DIRECTLY (not a
  -- name-pinned `.operand %dy <placeholder>`), so `den(output) = certified` is one composed
  -- theorem with the forward = the proven `fwdGraph` (nested inside `lossCotGraph`) — no
  -- SSA-name pin. The shared cotangent is rendered once per output (2× here); iree CSEs it.
  let act : StateM Nat (String × String × String) := do
    let (wBody, wRes) ← pretty B (SHlo.weightSgd "%x" "%W0" lrStr x W 0 (lossCotGraph W b x (fun _ => 0)))
    let (bBody, bRes) ← pretty B (SHlo.biasSgd "%b0" lrStr b 0 (lossCotGraph W b x (fun _ => 0)))
    pure (wBody ++ bBody, wRes, bRes)
  let (body, wRes, bRes) := act.run' 0
  "module @m {\n" ++
  s!"  func.func @linear_train_step(%x: {ty [B,m]}, %W0: {ty [m,n]}, %b0: {ty [n]}, " ++
  s!"%onehot: {ty [B,n]}) -> ({ty [m,n]}, {ty [n]}) " ++ "{\n" ++
  "    // ── linear train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  s!"    return {wRes}, {bRes} : {ty [m,n]}, {ty [n]}\n" ++
  "  }\n}\n"

/-- `@mlp_fwd` rendered from the verified forward AST `mlpFwdGraph`. -/
def mlpFwdModuleV (B d₀ d₁ d₂ d₃ : Nat)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : String :=
  renderModule "mlp_fwd"
    s!"%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}"
    B d₃ (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x)

/-- `@cnn_fwd` rendered from the verified CNN forward AST `cnnFwdGraph`. -/
def cnnFwdModuleV (B ic c h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) : String :=
  renderModule "cnn_fwd"
    s!"%x: {ty [B,ic*(2*h)*(2*w)]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [c*h*w,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}"
    B nClasses (cnnFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)

/-- `@cifar_fwd` rendered from the verified CIFAR forward AST `cifarFwdGraph`. -/
def cifarFwdModuleV (B ic c1 c2 h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  renderModule "cifar_fwd"
    s!"%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c2*h*w,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}"
    B nClasses (cifarFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)

/-- Full **MLP** SGD train step. The forward layers emit exactly `mlpFwdGraph`'s
    ops (`dot_general`+`add`, `maximum`), saving the pre-activations `%h0,%h1`;
    the backward emits `mlpBackGraph`'s ops (`dot_general`, `compare GT`+`select`
    masks reading `%h0,%h1`); param grads + SGD as in the linear step. Each piece
    is proven faithful above (`mlpFwdGraph_faithful`, `mlpBackGraph_faithful`,
    `reluF_faithful`, `selectPos_faithful`, `wGrad/bGrad_is*Jacobian`,
    `lossCotGraph_isCEgrad`, `sgd*_descends_certified_grad`); the assembly/naming
    is the renderer (validated by `iree-compile` + the GPU run). -/
def mlpTrainStepText (B d₀ d₁ d₂ d₃ : Nat) (lr : String) : String :=
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let reduce (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @mlp_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}, %onehot: {ty [B,d₃]}) -> ({ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}) " ++ "{\n" ++
  "    // ── forward (mlpFwdGraph): %h0,%h1 pre-acts, %a0,%a1 activations, %logits ──\n" ++
  dense "%h0" "%x" "%W0" "%b0" d₀ d₁ ++ relu "%a0" "%h0" d₁ ++
  dense "%h1" "%a0" "%W1" "%b1" d₁ d₂ ++ relu "%a1" "%h1" d₂ ++
  dense "%logits" "%a1" "%W2" "%b2" d₂ d₃ ++
  "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,d₃]}\n" ++
  "    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B,d₃]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,d₃]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,d₃]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,d₃]}\n" ++
  "    // ── backward (mlpBackGraph): dotOut + select masks reading %h1,%h0 ──\n" ++
  dg "%dx2" "%dy" "%W2" "1" "1" (ty [B,d₃]) (ty [d₂,d₃]) (ty [B,d₂]) ++
  s!"    %bz1 = stablehlo.constant dense<0.0> : {ty [B,d₂]}\n" ++
  s!"    %bm1 = stablehlo.compare GT, %h1, %bz1 : ({ty [B,d₂]}, {ty [B,d₂]}) -> {tyI1 [B,d₂]}\n" ++
  s!"    %dy1 = stablehlo.select %bm1, %dx2, %bz1 : {tyI1 [B,d₂]}, {ty [B,d₂]}\n" ++
  dg "%dx1" "%dy1" "%W1" "1" "1" (ty [B,d₂]) (ty [d₁,d₂]) (ty [B,d₁]) ++
  s!"    %bz0 = stablehlo.constant dense<0.0> : {ty [B,d₁]}\n" ++
  s!"    %bm0 = stablehlo.compare GT, %h0, %bz0 : ({ty [B,d₁]}, {ty [B,d₁]}) -> {tyI1 [B,d₁]}\n" ++
  s!"    %dy0 = stablehlo.select %bm0, %dx1, %bz0 : {tyI1 [B,d₁]}, {ty [B,d₁]}\n" ++
  "    // ── param grads (wGrad/bGrad) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  dg "%dW2" "%a1" "%dy" "0" "0" (ty [B,d₂]) (ty [B,d₃]) (ty [d₂,d₃]) ++ reduce "%db2" "%dy" d₃ ++
  dg "%dW1" "%a0" "%dy1" "0" "0" (ty [B,d₁]) (ty [B,d₂]) (ty [d₁,d₂]) ++ reduce "%db1" "%dy1" d₂ ++
  dg "%dW0" "%x" "%dy0" "0" "0" (ty [B,d₀]) (ty [B,d₁]) (ty [d₀,d₁]) ++ reduce "%db0" "%dy0" d₁ ++
  "    // ── SGD θ' = θ − lr·∇ ──\n" ++
  sgd "%W0" "%dW0" (ty [d₀,d₁]) ++ sgd "%b0" "%db0" (ty [d₁]) ++
  sgd "%W1" "%dW1" (ty [d₁,d₂]) ++ sgd "%b1" "%db1" (ty [d₂]) ++
  sgd "%W2" "%dW2" (ty [d₂,d₃]) ++ sgd "%b2" "%db2" (ty [d₃]) ++
  s!"    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n : {ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}\n" ++
  "  }\n}\n"

/-- Full **CNN** SGD train step (`@cnn_train_step`), the ch4 peer of
    `mlpTrainStepText`. Architecture (= `mnistCnnNoBnForward`):
    `conv W₁ → relu → conv W₂ → relu → maxpool → flatten → dense W₃ → relu →
     dense W₄ → relu → dense W₅`. Each mathematical op is a rendering of a
    proof-backed piece:
    * forward conv/maxpool/dense/relu — `flatConvF_faithful`, `maxPoolF_faithful`,
      `denseF_faithful`, `reluF_faithful` (and `cnnFwdGraph_faithful` for the whole);
    * loss cotangent `%dy = softmax(logits) − onehot` — `lossCotGraph_isCEgrad`;
    * backward dense (`dot_general`, contract output axis) + relu masks
      (`compare GT`+`select`) — `mlpBackGraph_faithful`/`selectPos_faithful`;
    * maxpool backward (`select_and_scatter`, GE/add, route dy to the window
      argmax) — `maxPoolBack_faithful`; conv input-VJP (transpose+reverse+conv)
      — `convBack_faithful`;
    * dense W/b grads (`dot_general` over batch / `reduce`) — `wGrad/bGrad`;
    * conv weight grad — the **transpose trick** (`conv2d_weight_grad_has_vjp`):
      the SAME `stablehlo.convolution` with the batch axis as the contraction
      feature; rendered here, validated by the GPU run (a `convWGrad_faithful`
      theorem is optional polish, see §B2 of the handoff);
    * SGD `θ' = θ − lr·∇` — `sgd*_descends_certified_grad`.
    The op text mirrors the GPU-validated emitter (`emitTok`) byte-for-byte for
    conv/maxpool/convBack/select_and_scatter; assembly + SSA naming is the
    renderer. `lr = 0.1/B` (grads sum over the batch). -/
def cnnTrainStepText (B ic c H W kH kW d1 nClasses : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2; let flat := c * H2 * W2
  -- dense dot_general (explicit contraction dims), as in mlpTrainStepText
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,c,H,W]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,c,H,W]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  -- relu-backward masks (`select(pre>0, dy, 0)`), 2-D and 4-D forms
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,c,H,W]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,c,H,W]}, {ty [B,c,H,W]}) -> {tyI1 [B,c,H,W]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,c,H,W]}, {ty [B,c,H,W]}\n"
  -- conv forward (SAME pad, stride 1) + bias bcast over channel dim 1
  let convFwd (o lhs w bnm : String) (oc icc : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,H,W]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,H,W]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,H,W]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,H,W]}\n"
  -- conv input-VJP: transpose[1,0,2,3] + reverse[2,3] + convolution (= emitTok convBack)
  let convBack (o dh w : String) (icc oc : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,H,W]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,H,W]}\n"
  -- conv weight grad (transpose trick): dW[o,i,·] = Σ_{b,y,x} x[b,i,·]·dh[b,o,·];
  -- realized as a convolution with the batch axis as the contraction feature.
  let convWGrad (o inp grad : String) (icc oc : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,H,W]}) -> {ty [icc,B,H,W]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,H,W]}) -> {ty [oc,B,H,W]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,H,W]}, {ty [oc,B,H,W]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,H,W]}, tensor<f32>) -> {ty [oc]}\n"
  -- maxpool forward (`reduce_window` max) and backward (`select_and_scatter`)
  let maxpoolFwd (o a : String) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,c,H,W]}, tensor<f32>) -> {ty [B,c,H2,W2]}\n"
  let scatter (o src dgrad : String) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,c,H,W]}, {ty [B,c,H2,W2]}, tensor<f32>) -> {ty [B,c,H,W]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cnn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [flat,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: conv→relu→conv→relu→maxpool→flatten→dense→relu→dense→relu→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c ic ++ relu4 "%ac1" "%hc1" ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c c ++ relu4 "%ac2" "%hc2" ++
  maxpoolFwd "%pool" "%ac2" ++
  s!"    %flat = stablehlo.reshape %pool : ({ty [B,c,H2,W2]}) -> {ty [B,flat]}\n" ++
  dense "%h3" "%flat" "%W3" "%b3" flat d1 ++ relu "%a3" "%h3" d1 ++
  dense "%h4" "%a3" "%W4" "%b4" d1 d1 ++ relu "%a4" "%h4" d1 ++
  dense "%logits" "%a4" "%W5" "%b5" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut) + relu masks → reshape → select_and_scatter → convBack ──\n" ++
  dg "%dx5" "%dy" "%W5" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dy4" "%h4" "%dx5" d1 ++
  dg "%dx4" "%dy4" "%W4" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy3" "%h3" "%dx4" d1 ++
  dg "%dx3" "%dy3" "%W3" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  s!"    %dpool = stablehlo.reshape %dx3 : ({ty [B,flat]}) -> {ty [B,c,H2,W2]}\n" ++
  scatter "%dac2" "%ac2" "%dpool" ++
  selMask4 "%dhc2" "%hc2" "%dac2" ++
  convBack "%dac1" "%dhc2" "%W2" c c ++
  selMask4 "%dhc1" "%hc1" "%dac1" ++
  "    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dW5" "%a4" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db5" "%dy" nClasses ++
  dg "%dW4" "%a3" "%dy4" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db4" "%dy4" d1 ++
  dg "%dW3" "%flat" "%dy3" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db3" "%dy3" d1 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c c ++ convBiasGrad "%db2" "%dhc2" c ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c ++ convBiasGrad "%db1" "%dhc1" c ++
  "    // ── SGD θ' = θ − lr·∇ (all 10 params) ──\n" ++
  sgd "%W1" "%dW1" (ty [c,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c]) ++
  sgd "%W2" "%dW2" (ty [c,c,kH,kW]) ++ sgd "%b2" "%db2" (ty [c]) ++
  sgd "%W3" "%dW3" (ty [flat,d1]) ++ sgd "%b3" "%db3" (ty [d1]) ++
  sgd "%W4" "%dW4" (ty [d1,d1]) ++ sgd "%b4" "%db4" (ty [d1]) ++
  sgd "%W5" "%dW5" (ty [d1,nClasses]) ++ sgd "%b5" "%db5" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n : {ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- Full **CIFAR CNN** SGD train step (`@cifar_train_step`), the Chapter-4 peer of
    `cnnTrainStepText`. Architecture (= `cifarCnnForward`):
    `conv 3→32 → relu → conv 32→32 → relu → maxpool → conv 32→64 → relu →
     conv 64→64 → relu → maxpool → flatten → dense 4096→512 → relu →
     dense 512→512 → relu → dense 512→10` + softmax-CE. Two conv→conv→pool
    stages at two spatial sizes (`H×W` then `H/2×W/2`), with channel changes.

    Every mathematical op is the SAME proof-backed render as `cnnTrainStepText`,
    just instantiated at more layers / two spatial scales — forward
    conv/maxpool/dense/relu (`cifarFwdGraph_faithful`); loss cotangent
    (`lossCotGraph_isCEgrad`); backward dense (`dot_general`) + relu masks
    (`selectPos_faithful`); maxpool backward (`select_and_scatter`,
    `maxPoolBack_faithful`); conv input-VJP (transpose+reverse+conv,
    `convBack_faithful`); dense W/b grads; conv weight grad (transpose trick);
    SGD `θ' = θ − lr·∇`. The per-op text mirrors the GPU-validated `emitTok`
    byte-for-byte; assembly + SSA naming is the renderer (validated by
    `iree-compile` + the GPU run). `lr = 0.1/B`. -/
def cifarTrainStepText (B ic c1 c2 H W kH kW d1 nClasses : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2          -- stage-2 spatial (16)
  let Hp := H2 / 2; let Wp := W2 / 2        -- final pooled (8)
  let flat := c2 * Hp * Wp
  -- dense dot_general (explicit contraction dims), as in cnnTrainStepText
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,C,Hh,Ww]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}) -> {tyI1 [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}\n"
  -- conv forward (SAME pad, stride 1) + bias bcast over channel dim 1
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  -- conv input-VJP: transpose[1,0,2,3] + reverse[2,3] + convolution (= emitTok convBack)
  let convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  -- conv weight grad (transpose trick): dW[o,i,·] = Σ_{b,y,x} x[b,i,·]·dh[b,o,·]
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  -- maxpool forward (`reduce_window` max) and backward (`select_and_scatter`)
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→relu)×2→pool →(conv→relu)×2→pool →flatten→(dense→relu)×2→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ relu4 "%ac1" "%hc1" c1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ relu4 "%ac2" "%hc2" c1 H W ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ relu4 "%ac3" "%hc3" c2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ relu4 "%ac4" "%hc4" c2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  s!"    %flat = stablehlo.reshape %pool2 : ({ty [B,c2,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h5" "%flat" "%W5" "%b5" flat d1 ++ relu2 "%a5" "%h5" d1 ++
  dense "%h6" "%a5" "%W6" "%b6" d1 d1 ++ relu2 "%a6" "%h6" d1 ++
  dense "%logits" "%a6" "%W7" "%b7" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut)+relu masks → scatter → convBack, twice through ──\n" ++
  dg "%dx7" "%dy" "%W7" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dy6" "%h6" "%dx7" d1 ++
  dg "%dx6" "%dy6" "%W6" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy5" "%h5" "%dx6" d1 ++
  dg "%dx5" "%dy5" "%W5" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  s!"    %dpool2 = stablehlo.reshape %dx5 : ({ty [B,flat]}) -> {ty [B,c2,Hp,Wp]}\n" ++
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++
  selMask4 "%dhc4" "%hc4" "%dac4" c2 H2 W2 ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++
  selMask4 "%dhc3" "%hc3" "%dac3" c2 H2 W2 ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++
  selMask4 "%dhc2" "%hc2" "%dac2" c1 H W ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++
  selMask4 "%dhc1" "%hc1" "%dac1" c1 H W ++
  "    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dW7" "%a6" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db7" "%dy" nClasses ++
  dg "%dW6" "%a5" "%dy6" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db6" "%dy6" d1 ++
  dg "%dW5" "%flat" "%dy5" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db5" "%dy5" d1 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 14 params) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
  sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
  sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- `@cifar_bn_fwd` rendered from the verified BN-CIFAR forward AST. γ/β are
    scalar `tensor<f32>` inputs (`%g{i}`/`%bt{i}`); `epsStr` the ε literal. -/
def cifarBnFwdModuleV (B ic c1 c2 h w d1 nClasses kH kW : Nat) (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  renderModule "cifar_bn_fwd"
    s!"%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c2*h*w,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}"
    B nClasses (cifarBnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ W₆ b₆ W₇ b₇ x)

/-- `@cifar8_fwd` rendered from the verified 8-conv CIFAR forward AST `cifar8FwdGraph`
    (`cifar8FwdGraph_faithful` proves it denotes `cifarCnn8Forward`). The 4-stage peer of
    `cifarFwdModuleV` — closes the cifar8 `_fwd` bytes (committed `verified_mlir/cifar8_fwd.mlir`
    is now `renderModule(provenGraph)`, replacing the hand-written `cifar8FwdText`). -/
def cifar8FwdModuleV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  renderModule "cifar8_fwd"
    s!"%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [c4*h*w,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
    B nClasses (cifar8FwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x)

/-- `@cifar8_bn_fwd` rendered from the verified 8-conv per-channel-BN CIFAR forward AST
    `cifar8BnFwdGraph` (`cifar8BnFwdGraph_faithful` proves it denotes `cifarCnnBn8Forward`).
    The BN peer of `cifar8FwdModuleV` — closes the cifar8-bn `_fwd` bytes, replacing the
    hand-written `cifar8BnFwdTextPC`. -/
def cifar8BnFwdModuleV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr : String)
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
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  renderModule "cifar8_bn_fwd"
    s!"%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [c4*h*w,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
    B nClasses (cifar8BnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x)

/-- Full **BN-CIFAR** SGD train step (`@cifar_bn_train_step`). The Chapter-4
    BatchNorm peer of `cifarTrainStepText`: each conv→relu block becomes
    conv→BN→relu. The per-example BN forward (`bnFwd` = `renderLN`: reduce μ/var
    over the feature axis, normalize, scalar-affine — denotes `bnForward`), its
    consolidated three-term input-VJP (`bnBack` = `renderLNBack` — the proven
    `bn_grad_input`, `bnBack_faithful`), and the scalar param grads
    `dγ = Σ dy·x̂`, `dβ = Σ dy` are inserted. BN runs on the flattened
    `[B, oc·H·W]` per-example feature vec (reshape around the 4-D conv). 22
    params (4×{W,b,γ,β} + 3×{W,b}). The whole-net backward is
    `cifarCnnBn_has_vjp_at`. `lr = 0.1/B`. -/
def cifarBnTrainStepText (B ic c1 c2 H W kH kW d1 nClasses : Nat) (epsStr lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2
  let Hp := H2 / 2; let Wp := W2 / 2
  let flat := c2 * Hp * Wp
  let M1 := c1 * H * W            -- stage-1 flattened feature size (= c1·S1)
  let M2 := c2 * H2 * W2          -- stage-2 flattened feature size (= c2·S2)
  let S1 := H * W                 -- stage-1 per-channel spatial size
  let S2 := H2 * W2               -- stage-2 per-channel spatial size
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  -- Per-channel BN forward: reshape [B,C·S]→[B,C,S], reduce μ/var over the spatial
  -- axis [2] per channel, normalize, per-channel affine (γ,β : [C]), reshape back to
  -- [B,C·S]. Saves {o}_xhat,_istd,_nf in [B,C,S] for the backward. (= bnPerChannelFlat.)
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  -- Per-channel BN input-VJP (consolidated three-term form, reductions over spatial
  -- axis [2]); reuses {bn}_xhat/_istd/_nf ([B,C,S]). Output reshaped back to [B,C·S].
  let bnBack (o bn g dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_dxh = stablehlo.multiply {o}_gb, {o}_dyr : {ty [B,C,S]}\n" ++
    s!"    {o}_sdxr = stablehlo.reduce({o}_dxh init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sdx = stablehlo.broadcast_in_dim {o}_sdxr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_xd = stablehlo.multiply {bn}_xhat, {o}_dxh : {ty [B,C,S]}\n" ++
    s!"    {o}_sxdr = stablehlo.reduce({o}_xd init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sxd = stablehlo.broadcast_in_dim {o}_sxdr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_t1 = stablehlo.multiply {o}_dxh, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_i1 = stablehlo.subtract {o}_t1, {o}_sdx : {ty [B,C,S]}\n" ++
    s!"    {o}_xs = stablehlo.multiply {bn}_xhat, {o}_sxd : {ty [B,C,S]}\n" ++
    s!"    {o}_i2 = stablehlo.subtract {o}_i1, {o}_xs : {ty [B,C,S]}\n" ++
    s!"    {o}_s = stablehlo.divide {bn}_istd, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_dx3 = stablehlo.multiply {o}_s, {o}_i2 : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_dx3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  -- Per-channel BN param grads dγ_c = Σ_{b,s} dy·x̂, dβ_c = Σ_{b,s} dy (reduce [0,2] → [C]).
  let bnParamGrad (dgr dbe bn dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {dgr}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {dgr}_p = stablehlo.multiply {dgr}_dyr, {bn}_xhat : {ty [B,C,S]}\n" ++
    s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n" ++
    s!"    {dbe} = stablehlo.reduce({dgr}_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n"
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→BN→relu)×2→pool →(conv→BN→relu)×2→pool →flatten→(dense→relu)×2→dense ──\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  rs "%flat" "%pool2" [B,c2,Hp,Wp] [B,flat] ++
  dense "%h5" "%flat" "%W5" "%b5" flat d1 ++ relu2 "%a5" "%h5" d1 ++
  dense "%h6" "%a5" "%W6" "%b6" d1 d1 ++ relu2 "%a6" "%h6" d1 ++
  dense "%logits" "%a6" "%W7" "%b7" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut)+relu → scatter → (relu→BN-back→convBack)×stage, twice ──\n" ++
  dg "%dx7" "%dy" "%W7" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dy6" "%h6" "%dx7" d1 ++
  dg "%dx6" "%dy6" "%W6" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy5" "%h5" "%dx6" d1 ++
  dg "%dx5" "%dy5" "%W5" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  rs "%dpool2" "%dx5" [B,flat] [B,c2,Hp,Wp] ++
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++ rs "%dac4f" "%dac4" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn4" "%bn4" "%dac4f" M2 ++
  bnBack "%dhc4f" "%bn4" "%g4" "%dbn4" c2 S2 ++ bnParamGrad "%dg4" "%dbt4" "%bn4" "%dbn4" c2 S2 ++
  rs "%dhc4" "%dhc4f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++ rs "%dac3f" "%dac3" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn3" "%bn3" "%dac3f" M2 ++
  bnBack "%dhc3f" "%bn3" "%g3" "%dbn3" c2 S2 ++ bnParamGrad "%dg3" "%dbt3" "%bn3" "%dbn3" c2 S2 ++
  rs "%dhc3" "%dhc3f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++ rs "%dac2f" "%dac2" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn2" "%bn2" "%dac2f" M1 ++
  bnBack "%dhc2f" "%bn2" "%g2" "%dbn2" c1 S1 ++ bnParamGrad "%dg2" "%dbt2" "%bn2" "%dbn2" c1 S1 ++
  rs "%dhc2" "%dhc2f" [B,M1] [B,c1,H,W] ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++ rs "%dac1f" "%dac1" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn1" "%bn1" "%dac1f" M1 ++
  bnBack "%dhc1f" "%bn1" "%g1" "%dbn1" c1 S1 ++ bnParamGrad "%dg1" "%dbt1" "%bn1" "%dbn1" c1 S1 ++
  rs "%dhc1" "%dhc1f" [B,M1] [B,c1,H,W] ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dW7" "%a6" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db7" "%dy" nClasses ++
  dg "%dW6" "%a5" "%dy6" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db6" "%dy6" d1 ++
  dg "%dW5" "%flat" "%dy5" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db5" "%dy5" d1 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 22 params, incl. scalar γ/β) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++ sgd "%g1" "%dg1" (ty [c1]) ++ sgd "%bt1" "%dbt1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++ sgd "%g2" "%dg2" (ty [c1]) ++ sgd "%bt2" "%dbt2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++ sgd "%g3" "%dg3" (ty [c2]) ++ sgd "%bt3" "%dbt3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++ sgd "%g4" "%dg4" (ty [c2]) ++ sgd "%bt4" "%dbt4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
  sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
  sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- Per-channel **BN-CIFAR** eval forward (`@cifar_bn_fwd`): the forward half of
    `cifarBnTrainStepText` (conv→per-channel-BN→relu ×4, 2 pools, 3 dense), returning
    logits `[B,nClasses]`. Per-channel BN (`m=H·W`) is per-example ⇒ train=eval (no
    running stats). String-rendered (peer of the train-step) until the typed
    `cifarBnFwdGraph` is reconciled to per-channel in the proof pass. -/
def cifarBnFwdTextPC (B ic c1 c2 H W kH kW d1 nClasses : Nat) (epsStr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2
  let Hp := H2 / 2; let Wp := W2 / 2
  let flat := c2 * Hp * Wp
  let M1 := c1 * H * W; let M2 := c2 * H2 * W2
  let S1 := H * W; let S2 := H2 * W2
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    s!"    {oh}d = stablehlo.dot_general {a}, {w}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,mm]}, {ty [mm,nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  -- per-channel BN forward (reshape [B,C·S]→[B,C,S], reduce spatial [2], affine γ,β:[C]).
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_bn_fwd(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}) -> {ty [B,nClasses]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  rs "%flat" "%pool2" [B,c2,Hp,Wp] [B,flat] ++
  dense "%h5" "%flat" "%W5" "%b5" flat d1 ++ relu2 "%a5" "%h5" d1 ++
  dense "%h6" "%a5" "%W6" "%b6" d1 d1 ++ relu2 "%a6" "%h6" d1 ++
  dense "%logits" "%a6" "%W7" "%b7" d1 nClasses ++
  s!"    return %logits : {ty [B,nClasses]}\n" ++
  "  }\n}\n"

/-! ### Deeper 8-conv CIFAR CNN (FOUR conv→conv→pool stages) train-step + fwd text

The 4-stage peers of `cifarTrainStepText` / `cifarBnTrainStepText` and their forwards.
Channels `c1 c2 c3 c4`; spatial `H → H/2 → H/4 → H/8 → H/16` (CIFAR 32→16→8→4→2). The
forward is `(conv→[BN→]relu)×2 → pool` four times → flatten `c4·Hp·Wp` → 3-dense head; the
backward is the exact transpose/reverse mirror (the same op templates as the 2-stage text).
The whole-net VJPs are `Proofs.cifarCnn8_has_vjp_at` / `cifarCnnBn8_has_vjp_at`. `lr = 0.1/B`. -/

/-- 8-conv CIFAR train step (`@cifar8_train_step`, no BN). 4 conv→conv→pool stages
    (channels `ic→c1→c1`, `c1→c2→c2`, `c2→c3→c3`, `c3→c4→c4`) + 3-dense head. -/
def cifar8TrainStepText (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2          -- after pool1 (16)
  let H3 := H2 / 2; let W3 := W2 / 2         -- after pool2 (8)
  let H4 := H3 / 2; let W4 := W3 / 2         -- after pool3 (4)
  let Hp := H4 / 2; let Wp := W4 / 2         -- after pool4 (2)
  let flat := c4 * Hp * Wp
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,C,Hh,Ww]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}) -> {tyI1 [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ relu4 "%ac1" "%hc1" c1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ relu4 "%ac2" "%hc2" c1 H W ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ relu4 "%ac3" "%hc3" c2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ relu4 "%ac4" "%hc4" c2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ relu4 "%ac5" "%hc5" c3 H3 W3 ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ relu4 "%ac6" "%hc6" c3 H3 W3 ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ relu4 "%ac7" "%hc7" c4 H4 W4 ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ relu4 "%ac8" "%hc8" c4 H4 W4 ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  s!"    %flat = stablehlo.reshape %pool4 : ({ty [B,c4,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut)+relu masks → scatter → convBack, four stages ──\n" ++
  dg "%dxb" "%dy" "%Wb" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dya" "%ha" "%dxb" d1 ++
  dg "%dxa" "%dya" "%Wa" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy9" "%h9" "%dxa" d1 ++
  dg "%dx9" "%dy9" "%W9" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  s!"    %dpool4 = stablehlo.reshape %dx9 : ({ty [B,flat]}) -> {ty [B,c4,Hp,Wp]}\n" ++
  -- stage 4
  scatter "%dac8" "%ac8" "%dpool4" c4 H4 W4 ++
  selMask4 "%dhc8" "%hc8" "%dac8" c4 H4 W4 ++
  convBack "%dac7" "%dhc8" "%W8" c4 c4 H4 W4 ++
  selMask4 "%dhc7" "%hc7" "%dac7" c4 H4 W4 ++
  convBack "%dpool3" "%dhc7" "%W7" c3 c4 H4 W4 ++
  -- stage 3
  scatter "%dac6" "%ac6" "%dpool3" c3 H3 W3 ++
  selMask4 "%dhc6" "%hc6" "%dac6" c3 H3 W3 ++
  convBack "%dac5" "%dhc6" "%W6" c3 c3 H3 W3 ++
  selMask4 "%dhc5" "%hc5" "%dac5" c3 H3 W3 ++
  convBack "%dpool2" "%dhc5" "%W5" c2 c3 H3 W3 ++
  -- stage 2
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++
  selMask4 "%dhc4" "%hc4" "%dac4" c2 H2 W2 ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++
  selMask4 "%dhc3" "%hc3" "%dac3" c2 H2 W2 ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  -- stage 1
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++
  selMask4 "%dhc2" "%hc2" "%dac2" c1 H W ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++
  selMask4 "%dhc1" "%hc1" "%dac1" c1 H W ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dWb" "%aa" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%dbb" "%dy" nClasses ++
  dg "%dWa" "%a9" "%dya" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%dba" "%dya" d1 ++
  dg "%dW9" "%flat" "%dy9" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db9" "%dy9" d1 ++
  convWGrad "%dW8" "%ac7" "%dhc8" c4 c4 H4 W4 ++ convBiasGrad "%db8" "%dhc8" c4 H4 W4 ++
  convWGrad "%dW7" "%pool3" "%dhc7" c3 c4 H4 W4 ++ convBiasGrad "%db7" "%dhc7" c4 H4 W4 ++
  convWGrad "%dW6" "%ac5" "%dhc6" c3 c3 H3 W3 ++ convBiasGrad "%db6" "%dhc6" c3 H3 W3 ++
  convWGrad "%dW5" "%pool2" "%dhc5" c2 c3 H3 W3 ++ convBiasGrad "%db5" "%dhc5" c3 H3 W3 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 22 params) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [c3,c2,kH,kW]) ++ sgd "%b5" "%db5" (ty [c3]) ++
  sgd "%W6" "%dW6" (ty [c3,c3,kH,kW]) ++ sgd "%b6" "%db6" (ty [c3]) ++
  sgd "%W7" "%dW7" (ty [c4,c3,kH,kW]) ++ sgd "%b7" "%db7" (ty [c4]) ++
  sgd "%W8" "%dW8" (ty [c4,c4,kH,kW]) ++ sgd "%b8" "%db8" (ty [c4]) ++
  sgd "%W9" "%dW9" (ty [flat,d1]) ++ sgd "%b9" "%db9" (ty [d1]) ++
  sgd "%Wa" "%dWa" (ty [d1,d1]) ++ sgd "%ba" "%dba" (ty [d1]) ++
  sgd "%Wb" "%dWb" (ty [d1,nClasses]) ++ sgd "%bb" "%dbb" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n, %W8n, %b8n, %W9n, %b9n, %Wan, %ban, %Wbn, %bbn : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- 8-conv CIFAR eval forward (`@cifar8_fwd`, no BN), returning logits `[B,nClasses]`. -/
def cifar8FwdText (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := c4 * Hp * Wp
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    s!"    {oh}d = stablehlo.dot_general {a}, {w}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,mm]}, {ty [mm,nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,C,Hh,Ww]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_fwd(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}) -> {ty [B,nClasses]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ relu4 "%ac1" "%hc1" c1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ relu4 "%ac2" "%hc2" c1 H W ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ relu4 "%ac3" "%hc3" c2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ relu4 "%ac4" "%hc4" c2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ relu4 "%ac5" "%hc5" c3 H3 W3 ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ relu4 "%ac6" "%hc6" c3 H3 W3 ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ relu4 "%ac7" "%hc7" c4 H4 W4 ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ relu4 "%ac8" "%hc8" c4 H4 W4 ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  s!"    %flat = stablehlo.reshape %pool4 : ({ty [B,c4,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  s!"    return %logits : {ty [B,nClasses]}\n" ++
  "  }\n}\n"

/-- 8-conv CIFAR **per-channel BN** train step (`@cifar8_bn_train_step`). Each of the 8
    convs is followed by `bnFwd` (per-channel BN, reduce spatial axis [2]); the backward
    inserts the relu-mask → BN input-VJP (`bnBack`) → conv-back per block + BN param grads
    (`dγ=Σ dy·x̂, dβ=Σ dy`). 38 params (8×{W,b,γ,β} + 3×{W,b}). Whole-net VJP:
    `Proofs.cifarCnnBn8_has_vjp_at`. `lr = 0.1/B`. -/
def cifar8BnTrainStepText (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) (epsStr lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := c4 * Hp * Wp
  let M1 := c1 * H * W;   let S1 := H * W
  let M2 := c2 * H2 * W2; let S2 := H2 * W2
  let M3 := c3 * H3 * W3; let S3 := H3 * W3
  let M4 := c4 * H4 * W4; let S4 := H4 * W4
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  let bnBack (o bn g dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_dxh = stablehlo.multiply {o}_gb, {o}_dyr : {ty [B,C,S]}\n" ++
    s!"    {o}_sdxr = stablehlo.reduce({o}_dxh init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sdx = stablehlo.broadcast_in_dim {o}_sdxr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_xd = stablehlo.multiply {bn}_xhat, {o}_dxh : {ty [B,C,S]}\n" ++
    s!"    {o}_sxdr = stablehlo.reduce({o}_xd init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sxd = stablehlo.broadcast_in_dim {o}_sxdr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_t1 = stablehlo.multiply {o}_dxh, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_i1 = stablehlo.subtract {o}_t1, {o}_sdx : {ty [B,C,S]}\n" ++
    s!"    {o}_xs = stablehlo.multiply {bn}_xhat, {o}_sxd : {ty [B,C,S]}\n" ++
    s!"    {o}_i2 = stablehlo.subtract {o}_i1, {o}_xs : {ty [B,C,S]}\n" ++
    s!"    {o}_s = stablehlo.divide {bn}_istd, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_dx3 = stablehlo.multiply {o}_s, {o}_i2 : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_dx3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  let bnParamGrad (dgr dbe bn dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {dgr}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {dgr}_p = stablehlo.multiply {dgr}_dyr, {bn}_xhat : {ty [B,C,S]}\n" ++
    s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n" ++
    s!"    {dbe} = stablehlo.reduce({dgr}_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n"
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→BN→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ rs "%hc5f" "%hc5" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn5" "%hc5f" "%g5" "%bt5" c3 S3 ++ relu2 "%ac5f" "%bn5" M3 ++ rs "%ac5" "%ac5f" [B,M3] [B,c3,H3,W3] ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ rs "%hc6f" "%hc6" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn6" "%hc6f" "%g6" "%bt6" c3 S3 ++ relu2 "%ac6f" "%bn6" M3 ++ rs "%ac6" "%ac6f" [B,M3] [B,c3,H3,W3] ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ rs "%hc7f" "%hc7" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn7" "%hc7f" "%g7" "%bt7" c4 S4 ++ relu2 "%ac7f" "%bn7" M4 ++ rs "%ac7" "%ac7f" [B,M4] [B,c4,H4,W4] ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ rs "%hc8f" "%hc8" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn8" "%hc8f" "%g8" "%bt8" c4 S4 ++ relu2 "%ac8f" "%bn8" M4 ++ rs "%ac8" "%ac8f" [B,M4] [B,c4,H4,W4] ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  rs "%flat" "%pool4" [B,c4,Hp,Wp] [B,flat] ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense+relu → scatter → (relu→BN-back→convBack)×stage, four stages ──\n" ++
  dg "%dxb" "%dy" "%Wb" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dya" "%ha" "%dxb" d1 ++
  dg "%dxa" "%dya" "%Wa" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy9" "%h9" "%dxa" d1 ++
  dg "%dx9" "%dy9" "%W9" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  rs "%dpool4" "%dx9" [B,flat] [B,c4,Hp,Wp] ++
  -- stage 4
  scatter "%dac8" "%ac8" "%dpool4" c4 H4 W4 ++ rs "%dac8f" "%dac8" [B,c4,H4,W4] [B,M4] ++
  selMask2 "%dbn8" "%bn8" "%dac8f" M4 ++
  bnBack "%dhc8f" "%bn8" "%g8" "%dbn8" c4 S4 ++ bnParamGrad "%dg8" "%dbt8" "%bn8" "%dbn8" c4 S4 ++
  rs "%dhc8" "%dhc8f" [B,M4] [B,c4,H4,W4] ++
  convBack "%dac7" "%dhc8" "%W8" c4 c4 H4 W4 ++ rs "%dac7f" "%dac7" [B,c4,H4,W4] [B,M4] ++
  selMask2 "%dbn7" "%bn7" "%dac7f" M4 ++
  bnBack "%dhc7f" "%bn7" "%g7" "%dbn7" c4 S4 ++ bnParamGrad "%dg7" "%dbt7" "%bn7" "%dbn7" c4 S4 ++
  rs "%dhc7" "%dhc7f" [B,M4] [B,c4,H4,W4] ++
  convBack "%dpool3" "%dhc7" "%W7" c3 c4 H4 W4 ++
  -- stage 3
  scatter "%dac6" "%ac6" "%dpool3" c3 H3 W3 ++ rs "%dac6f" "%dac6" [B,c3,H3,W3] [B,M3] ++
  selMask2 "%dbn6" "%bn6" "%dac6f" M3 ++
  bnBack "%dhc6f" "%bn6" "%g6" "%dbn6" c3 S3 ++ bnParamGrad "%dg6" "%dbt6" "%bn6" "%dbn6" c3 S3 ++
  rs "%dhc6" "%dhc6f" [B,M3] [B,c3,H3,W3] ++
  convBack "%dac5" "%dhc6" "%W6" c3 c3 H3 W3 ++ rs "%dac5f" "%dac5" [B,c3,H3,W3] [B,M3] ++
  selMask2 "%dbn5" "%bn5" "%dac5f" M3 ++
  bnBack "%dhc5f" "%bn5" "%g5" "%dbn5" c3 S3 ++ bnParamGrad "%dg5" "%dbt5" "%bn5" "%dbn5" c3 S3 ++
  rs "%dhc5" "%dhc5f" [B,M3] [B,c3,H3,W3] ++
  convBack "%dpool2" "%dhc5" "%W5" c2 c3 H3 W3 ++
  -- stage 2
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++ rs "%dac4f" "%dac4" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn4" "%bn4" "%dac4f" M2 ++
  bnBack "%dhc4f" "%bn4" "%g4" "%dbn4" c2 S2 ++ bnParamGrad "%dg4" "%dbt4" "%bn4" "%dbn4" c2 S2 ++
  rs "%dhc4" "%dhc4f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++ rs "%dac3f" "%dac3" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn3" "%bn3" "%dac3f" M2 ++
  bnBack "%dhc3f" "%bn3" "%g3" "%dbn3" c2 S2 ++ bnParamGrad "%dg3" "%dbt3" "%bn3" "%dbn3" c2 S2 ++
  rs "%dhc3" "%dhc3f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  -- stage 1
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++ rs "%dac2f" "%dac2" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn2" "%bn2" "%dac2f" M1 ++
  bnBack "%dhc2f" "%bn2" "%g2" "%dbn2" c1 S1 ++ bnParamGrad "%dg2" "%dbt2" "%bn2" "%dbn2" c1 S1 ++
  rs "%dhc2" "%dhc2f" [B,M1] [B,c1,H,W] ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++ rs "%dac1f" "%dac1" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn1" "%bn1" "%dac1f" M1 ++
  bnBack "%dhc1f" "%bn1" "%g1" "%dbn1" c1 S1 ++ bnParamGrad "%dg1" "%dbt1" "%bn1" "%dbn1" c1 S1 ++
  rs "%dhc1" "%dhc1f" [B,M1] [B,c1,H,W] ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dWb" "%aa" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%dbb" "%dy" nClasses ++
  dg "%dWa" "%a9" "%dya" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%dba" "%dya" d1 ++
  dg "%dW9" "%flat" "%dy9" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db9" "%dy9" d1 ++
  convWGrad "%dW8" "%ac7" "%dhc8" c4 c4 H4 W4 ++ convBiasGrad "%db8" "%dhc8" c4 H4 W4 ++
  convWGrad "%dW7" "%pool3" "%dhc7" c3 c4 H4 W4 ++ convBiasGrad "%db7" "%dhc7" c4 H4 W4 ++
  convWGrad "%dW6" "%ac5" "%dhc6" c3 c3 H3 W3 ++ convBiasGrad "%db6" "%dhc6" c3 H3 W3 ++
  convWGrad "%dW5" "%pool2" "%dhc5" c2 c3 H3 W3 ++ convBiasGrad "%db5" "%dhc5" c3 H3 W3 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 38 params, incl. per-channel γ/β) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++ sgd "%g1" "%dg1" (ty [c1]) ++ sgd "%bt1" "%dbt1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++ sgd "%g2" "%dg2" (ty [c1]) ++ sgd "%bt2" "%dbt2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++ sgd "%g3" "%dg3" (ty [c2]) ++ sgd "%bt3" "%dbt3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++ sgd "%g4" "%dg4" (ty [c2]) ++ sgd "%bt4" "%dbt4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [c3,c2,kH,kW]) ++ sgd "%b5" "%db5" (ty [c3]) ++ sgd "%g5" "%dg5" (ty [c3]) ++ sgd "%bt5" "%dbt5" (ty [c3]) ++
  sgd "%W6" "%dW6" (ty [c3,c3,kH,kW]) ++ sgd "%b6" "%db6" (ty [c3]) ++ sgd "%g6" "%dg6" (ty [c3]) ++ sgd "%bt6" "%dbt6" (ty [c3]) ++
  sgd "%W7" "%dW7" (ty [c4,c3,kH,kW]) ++ sgd "%b7" "%db7" (ty [c4]) ++ sgd "%g7" "%dg7" (ty [c4]) ++ sgd "%bt7" "%dbt7" (ty [c4]) ++
  sgd "%W8" "%dW8" (ty [c4,c4,kH,kW]) ++ sgd "%b8" "%db8" (ty [c4]) ++ sgd "%g8" "%dg8" (ty [c4]) ++ sgd "%bt8" "%dbt8" (ty [c4]) ++
  sgd "%W9" "%dW9" (ty [flat,d1]) ++ sgd "%b9" "%db9" (ty [d1]) ++
  sgd "%Wa" "%dWa" (ty [d1,d1]) ++ sgd "%ba" "%dba" (ty [d1]) ++
  sgd "%Wb" "%dWb" (ty [d1,nClasses]) ++ sgd "%bb" "%dbb" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %g5n, %bt5n, %W6n, %b6n, %g6n, %bt6n, %W7n, %b7n, %g7n, %bt7n, %W8n, %b8n, %g8n, %bt8n, %W9n, %b9n, %Wan, %ban, %Wbn, %bbn : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- 8-conv CIFAR **per-channel BN** eval forward (`@cifar8_bn_fwd`), returning logits. -/
def cifar8BnFwdTextPC (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) (epsStr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := c4 * Hp * Wp
  let M1 := c1 * H * W;   let S1 := H * W
  let M2 := c2 * H2 * W2; let S2 := H2 * W2
  let M3 := c3 * H3 * W3; let S3 := H3 * W3
  let M4 := c4 * H4 * W4; let S4 := H4 * W4
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    s!"    {oh}d = stablehlo.dot_general {a}, {w}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,mm]}, {ty [mm,nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_bn_fwd(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}) -> {ty [B,nClasses]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ rs "%hc5f" "%hc5" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn5" "%hc5f" "%g5" "%bt5" c3 S3 ++ relu2 "%ac5f" "%bn5" M3 ++ rs "%ac5" "%ac5f" [B,M3] [B,c3,H3,W3] ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ rs "%hc6f" "%hc6" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn6" "%hc6f" "%g6" "%bt6" c3 S3 ++ relu2 "%ac6f" "%bn6" M3 ++ rs "%ac6" "%ac6f" [B,M3] [B,c3,H3,W3] ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ rs "%hc7f" "%hc7" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn7" "%hc7f" "%g7" "%bt7" c4 S4 ++ relu2 "%ac7f" "%bn7" M4 ++ rs "%ac7" "%ac7f" [B,M4] [B,c4,H4,W4] ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ rs "%hc8f" "%hc8" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn8" "%hc8f" "%g8" "%bt8" c4 S4 ++ relu2 "%ac8f" "%bn8" M4 ++ rs "%ac8" "%ac8f" [B,M4] [B,c4,H4,W4] ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  rs "%flat" "%pool4" [B,c4,Hp,Wp] [B,flat] ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  s!"    return %logits : {ty [B,nClasses]}\n" ++
  "  }\n}\n"

end StableHLO
end Proofs

-- Emit the verified-renderer modules at the real ch-1 shapes (784→10, B=128).
-- `pretty` ignores operand/param *values* (only names/shapes reach the text),
-- so the constant placeholders below render exactly the text `den` is faithful
-- to. The train-step lr literal is 0.1/128 (mean-loss equiv of the book's 0.1).
#eval IO.FS.writeFile "/tmp/linear_fwd_v.mlir"
  (Proofs.StableHLO.linearFwdModuleV 128 784 10 (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
#eval IO.FS.writeFile "/tmp/linear_back_v.mlir"
  (Proofs.StableHLO.linearBackModuleV 128 784 10 (fun _ _ => 0) (fun _ => 0))
#eval IO.FS.writeFile "/tmp/linear_train_step_v.mlir"
  (Proofs.StableHLO.linearTrainStepModuleV 128 784 10 "0.00078125" (fun _ _ => 0) (fun _ => 0) (fun _ => 0))

-- Committed verified-rendered artifacts (the exact `pretty (emit g)` text the
-- `mnist-linear-verified` trainer compiles + runs through the real Lean/IREE
-- FFI on GPU). Regenerate with `lake env lean LeanMlir/Proofs/Codegen/StableHLO.lean`.
#eval (do
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/linear_fwd.mlir"
    (Proofs.StableHLO.linearFwdModuleV 128 784 10 (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
  -- Whole train step rendered from the verified AST (cotangent + weightSgd/biasSgd
  -- nodes), the den-certified renderer LinearFaithfulPoC proves; see that file +
  -- planning/verified_faithful_sweep.md. (linearTrainStepModuleV — forward-AST +
  -- hand-written tail — is its structural predecessor, kept for reference.)
  IO.FS.writeFile "verified_mlir/linear_train_step.mlir"
    (Proofs.StableHLO.linTrainStepFaithfulV 128 784 10 "0.00078125"
       (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
  -- Chapter 2 MLP (784→512→512→10): forward + full SGD train step.
  IO.FS.writeFile "verified_mlir/mlp_fwd.mlir"
    (Proofs.StableHLO.mlpFwdModuleV 128 784 512 512 10
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- mlp_train_step.mlir is now generated by the faithful renderer in MlpRender.lean
  -- (mlpTrainStepFaithfulV, den-certified in MlpFaithfulPoC.lean), not mlpTrainStepText.
  -- Chapter 3 CNN forward (1→32→32 conv, 28×28→14×14 maxpool, 6272→512→512→10).
  IO.FS.writeFile "verified_mlir/cnn_fwd.mlir"
    (Proofs.StableHLO.cnnFwdModuleV 128 1 32 14 14 512 10 3 3
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- cnn_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cnnTrainStepFaithfulV, den-certified in CnnFaithfulPoC.lean), not cnnTrainStepText.
  -- Chapter 4 CIFAR forward (3→32→32 conv, 32×32→16×16 pool, 32→64→64 conv,
  -- 16×16→8×8 pool, flatten 4096→512→512→10). h=w=8 ⇒ input 3·32·32 = 3072.
  IO.FS.writeFile "verified_mlir/cifar_fwd.mlir"
    (Proofs.StableHLO.cifarFwdModuleV 128 3 32 64 8 8 512 10 3 3
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- cifar_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifarTrainStepFaithfulV, den-certified in CifarFaithfulPoC.lean), not cifarTrainStepText.
  -- Chapter 4 CIFAR **per-channel BatchNorm** forward (per-example per-channel BN
  -- after each conv; ε=1e-5; H=W=32 full input spatial). String renderer (peer of
  -- the train-step) until the typed cifarBnFwdGraph is reconciled to per-channel.
  IO.FS.writeFile "verified_mlir/cifar_bn_fwd.mlir"
    (Proofs.StableHLO.cifarBnFwdTextPC 128 3 32 64 32 32 3 3 512 10 "1.0e-05")
  -- cifar_bn_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifarBnTrainStepFaithfulV, den-certified in CifarBnFaithfulPoC.lean), not cifarBnTrainStepText.
  -- Deeper 8-conv CIFAR (no BN): 4 conv→conv→pool stages, channels [16,16,32,32],
  -- 32→16→8→4→2 spatial, flat 32·2·2 = 128 → 64 → 64 → 10. lr = 0.1/128.
  -- cifar8_fwd.mlir is now rendered from the verified cifar8FwdGraph (cifar8FwdGraph_faithful),
  -- not the hand-written cifar8FwdText. Dims: h=w=2 final pooled (image 32×32, 4 pools).
  IO.FS.writeFile "verified_mlir/cifar8_fwd.mlir"
    (Proofs.StableHLO.cifar8FwdModuleV 128 3 16 16 32 32 2 2 64 10 3 3
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0))
  -- cifar8_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifar8TrainStepFaithfulV, den-certified in Cifar8FaithfulPoC.lean), not cifar8TrainStepText.
  -- Deeper 8-conv CIFAR **per-channel BatchNorm** (ε=1e-5; lr = 0.1/128).
  -- cifar8_bn_fwd.mlir is now rendered from the verified cifar8BnFwdGraph (cifar8BnFwdGraph_faithful),
  -- not the hand-written cifar8BnFwdTextPC. Per conv layer: W b ε γ β (ε render-erased; epsStr carries it).
  IO.FS.writeFile "verified_mlir/cifar8_bn_fwd.mlir"
    (Proofs.StableHLO.cifar8BnFwdModuleV 128 3 16 16 32 32 2 2 64 10 3 3 "1.0e-05"
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0))
  -- cifar8_bn_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifar8BnTrainStepFaithfulV, den-certified by the existing generics), not cifar8BnTrainStepText.
  -- ── §2i: the cifar8-WIDE eval forwards, d1 := 512 ────────────────────────────────────────
  -- `cifar8w` is `cifar8` with the MNIST-style 2×512 head — measured 2026-07-30, the two
  -- `VerifiedNetSpec`s agree layer-for-layer up to the head width, and the committed
  -- `cifar8w_bn_adam_train_step.mlir` is BYTE-IDENTICAL (modulo the entry name) to the width-sweep's
  -- `cifar8_bn_512_adam_train_step.mlir`. So the wide family needs NO new renderer: it is these same
  -- certified modules at `d1 := 512`, which is also what the `cifar8_bn_{d}` sweep already does.
  -- It is NOT prunable — `runs/ablation_cifar8w/README.md` is the Chapter-5 "bridge" table (6 cells,
  -- both forwards), and the width sweep is adam-only so it covers just 1 of those 6.
  -- ✅ SWAPPED 2026-07-30: both ties came back logits BIT-EXACT 1280/1280 against the retired
  -- hand-written text emitters, and each `#eval` is now its artifact's ONLY writer. Entry renamed because these renderers emit
  -- `@cifar8{,_bn}_fwd` — the same `.replace` the retired hand-written writer used, so
  -- `trainAdamSched`'s `m.cifar8w{,_bn}_fwd` eval call still resolves.
  IO.FS.writeFile "verified_mlir/cifar8w_fwd.mlir"
    ((Proofs.StableHLO.cifar8FwdModuleV 128 3 16 16 32 32 2 2 512 10 3 3
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0)).replace "@cifar8_fwd" "@cifar8w_fwd")
  IO.FS.writeFile "verified_mlir/cifar8w_bn_fwd.mlir"
    ((Proofs.StableHLO.cifar8BnFwdModuleV 128 3 16 16 32 32 2 2 512 10 3 3 "1.0e-05"
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0)).replace "@cifar8_bn_fwd" "@cifar8w_bn_fwd")
  pure () : IO Unit)

