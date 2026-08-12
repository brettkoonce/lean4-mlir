import LeanMlir.VerifiedTrain

/-! # NetSpec-style layer DSL for the verified trainers (Tier-2)

A verified trainer should read like the reference `MainResnetTrain.lean` — a layer
list + config + `train` — with the *only* difference being the formalization underneath.
This file provides that surface:

  * `VLayer`           — the verified-vocabulary layer constructors (the ops that have
                         proven `HasVJP` witnesses), mirroring the reference `Layer`;
  * `VerifiedNetSpec`  — a `{ layers := [...] }` architecture, the single source of truth;
  * `toSpecs`          — folds `layers` into the `(dims, initKind)` param layout, so the
                         layout is *derived* from the architecture rather than hand-listed
                         a second time. (Kernel-check it against the audited `XLayout.specs`
                         with `#guard spec.toSpecs == XLayout.specs` — see `MainResnet34Verified`.)

The architecture's *faithfulness* is the audited `<net>_has_vjp` theorem, which is itself a
hand-unrolled `foldl` of the generic `vjp_comp` chain-rule combinator (`Proofs/Tensor.lean`)
over these same layers — so the spec and the proof describe the same fold. Generating the
verified StableHLO from `layers` (folding the proven op-emitters) and folding the proof via
a `netVjp` term are the remaining Tier-2 / Tier-3 steps; for now the slug names the committed,
audited render of this architecture.
-/

/-- A verified-vocabulary layer. Restricted to ops with proven `HasVJP` witnesses; each
    carries enough to derive its slice of the param layout. -/
inductive VLayer where
  /-- conv (`oc←ic`, `k×k`, `stride`) → per-channel BN → relu. Params `{W,b,γ,β}`. -/
  | convBn (ic oc k stride : Nat)
  /-- conv → per-channel BN → relu with **no conv bias** — `{W, γ, β}`. BN removes a conv bias
      (`(x+b) − mean(x+b) = x − mean(x)`), so a BN-followed conv carries none in He et al.'s
      `.convBn`; ResNet-34 uses this and the nets that genuinely ship a bias use `convBn`
      (§2l step B, measured in `tests/TestConvBiasZero.lean`). -/
  | convBnNB (ic oc k stride : Nat)
  /-- max pool `k×k` / `stride`. No params. -/
  | maxPool (k stride : Nat)
  /-- A basic-block residual stage: `nBlocks` blocks at `oc` channels. The first block
      downsamples (and projects the skip) iff `stride ≠ 1 ∨ ic ≠ oc`; the rest are identity. -/
  | residualStage (ic oc nBlocks stride : Nat)
  /-- A **bottleneck** residual stage (ResNet-50/101/152): `nBlocks` blocks, each
      `1×1 (oc/4) → BN → relu`, `3×3 (oc/4) → BN → relu`, `1×1 (oc) → BN`, `+skip`, `relu`.
      The first block projects the skip (`1×1 → BN`) iff `stride ≠ 1 ∨ ic ≠ oc` — the same
      dispatch `residualStage` uses, and for R50 that fires on **all four** stages, because
      stage 1 changes 64→256 at stride 1 where R34's stage 1 is `ic = oc`.

      ⚠ **This is ResNet v1.5, not He et al.'s v1**: the stride sits on the **3×3** (and on the
      projection), with the leading 1×1 at stride 1. Measured off the reference
      (`jax/Jax/Codegen.lean`'s `bottleneck_block_down`), not assumed — putting it on the first
      1×1 compiles, trains, descends and is a different net (§2k's heavy-ball trap one layer up).

      **No conv biases** — every conv here is BN-followed, so a bias cannot reach the output
      (`convBnNB`'s argument, four convs at a time). torchvision's R50 carries none either, which
      is why the derived count lands on the reference's 25,557,032 with no adjustment (§2m). -/
  | bottleneckStage (ic oc nBlocks stride : Nat)
  /-- global average pool. No params. -/
  | globalAvgPool
  /-- dense `ic→oc`. Params `{W,b}`. -/
  | dense (ic oc : Nat)
  /-- ReLU activation (pointwise). No params. -/
  | relu
  /-- plain conv (`oc←ic`, `k×k`, `stride`) + bias, NO batch-norm. Params `{W,b}`. -/
  | conv (ic oc k stride : Nat)
  /-- flatten `[C,H,W]` → vector. No params (a reshape). -/
  | flatten
  /-- scalar-global BatchNorm (the proven `bnForward`): normalize over the whole
      `c·h·w` feature map per example, **scalar** γ/β. Params `{γ, β}` (rank-0). -/
  | bn
  /-- per-channel (per-example) BatchNorm (the proven `bnPerChannelFlat`, `m=h·w`):
      normalize each of `oc` channels over its own `h·w` spatial map per example,
      **per-channel** γ/β `[oc]`. Train=eval (no running stats). Params `{γ:[oc], β:[oc]}`. -/
  | bnPerChannel (oc : Nat)
  /-- MobileNetV2 inverted-residual block (`ic→mid→oc`, depthwise `stride`): expand 1×1
      conv→per-channel BN→relu6, depthwise 3×3→BN→relu6, project 1×1→BN (linear bottleneck),
      + residual when `stride=1 ∧ ic=oc`. Params `{W,b,γ,β}` ×3 (expand/depthwise/project). -/
  | invertedResidual (ic mid oc stride : Nat)
  /-- MobileNetV2 inverted-residual block with **no conv biases** — `{W,γ,β}` ×3. Every conv in the
      block is BN-followed, so a bias cannot reach the output (the `convBnNB` argument, three convs
      at a time); the torchvision/JAX reference carries none, and ours carried 52 across the net
      (§2m: the +17,056-param gap to the reference, closed exactly). Kept beside
      `invertedResidual` rather than replacing it for `convBnNB`'s reason — a net whose blocks
      genuinely ship biases should still be able to say so. -/
  | invertedResidualNB (ic mid oc stride : Nat)
  /-- EfficientNet MBConv block (`ic→mid=t·ic→oc`, depthwise `k×k`, SE ratio `r`): expand 1×1
      (skipped when `mid=ic`, i.e. t=1) → BN → swish, depthwise k×k → BN → swish, squeeze-excite
      (`Ws₁[mid,r]`,`bs₁[r]`,`Ws₂[r,mid]`,`bs₂[mid]`, sigmoid gate), project 1×1 → BN. Params:
      (expand{W,b,γ,β} if t≠1) ++ depthwise{W,b,γ,β} ++ SE{Ws₁,bs₁,Ws₂,bs₂} ++ project{W,b,γ,β}. -/
  | mbConvSE (ic mid oc r k : Nat)
  /-- EfficientNet MBConv with **no conv biases on the BN-followed convs** — expand/depthwise/
      project become `{W,γ,β}`. ⚠ **The squeeze-excite biases STAY.** SE's two 1×1 convs are
      followed by an ACTIVATION (sigmoid gate), not BN, so nothing absorbs them and the reference
      carries them; only a BN-followed conv can drop its bias. That distinction is what made the
      +21,008 gap close exactly (§2m) — the audit's rule was "a rank-1 kind-2 param immediately
      after a **rank-4** kernel", and SE's params are rank-2. -/
  | mbConvSENB (ic mid oc r k : Nat)
  /-- **MobileNetV4 Universal Inverted Bottleneck** (`planning/mnv4_verified.md`):
      `optional pre-DW (preDWk) → 1×1 expand ic→mid → optional post-DW (postDWk) → 1×1 project
      mid→oc`, every conv BN-followed and therefore **bias-free** (`convBnNB`'s argument, and what
      `Spec.lean`'s baseline count already assumes). `mid = ic * expand`.

      ⭐ `k = 0` means "omit that depthwise", which is how ONE constructor expresses all four of
      MNv4's block families — ExtraDW (both DWs), IB/MBConv (post only), ConvNeXt-like (pre only),
      FFN (neither). That is the architecture's whole "stop adding new block types" claim, and it
      is why this is one `VLayer` case rather than four.

      ⚠⚠ **A wrong pre/post dispatch is INVISIBLE to this function.** A pre-DW and a post-DW at the
      same `k` and the same channel count contribute *identical* parameter shapes, so emitting one
      where the table says the other yields a net that type-checks, trains, descends, and is not
      MobileNetV4. Same class as R50's stride-on-the-3×3 and the 2×2 stem pool. The gate is a
      forward tie against the reference on shared weights — **not** a param count, and not this. -/
  | uib (ic oc expand stride preDWk postDWk : Nat)
  /-- **Fused inverted bottleneck**, single block, no squeeze-excite — EfficientNetV2's early-stage
      block, and MobileNetV4's stage 0. `k×k regular conv ic→mid (stride) → BN → swish →
      1×1 project mid→oc → BN`, no activation after the project; skip iff `stride = 1 ∧ ic = oc`.
      "Fused" = the MBConv expand-1×1 and depthwise collapse into ONE regular `k×k` conv, which is
      why nothing here is depthwise. `mid = ic * expand`. Bias-free — both convs are BN-followed.

      ⚠⚠ **THE ACTIVATION IS SWISH, NOT RELU, AND THAT IS A PAPER DEVIATION.** MobileNetV4-Conv is
      a ReLU network, but both emitters that produced the 84.58% use swish here
      (`jax/Jax/Codegen.lean:1031` — the reference — and `MlirCodegen.lean:6148`'s
      `emitConvBnTrainSwish`), inherited from the block being shared with EfficientNetV2. Matching
      the REFERENCE is what lets the number be reproduced and tied; matching the PAPER would be a
      different net from the one with the result. Recorded rather than quietly fixed.

      ⚠ Deliberately narrower than the baseline `Layer.fusedMbConv`, which also carries `nBlocks`
      and `useSE`. MNv4 uses `n = 1, useSE = false`, and a layout whose render does not exist is a
      trap — so the constructor cannot express what this file cannot emit. -/
  | fusedMbConvNB (ic oc expand k stride : Nat)
  /-- ConvNeXt block @ `c` channels (expand ratio 4): depthwise 7×7 → scalar-LN → 1×1 expand
      c→4c → GELU → 1×1 project 4c→c → layerScale (per-channel γ). Params: depthwise{W,b};
      LN{γ,β scalar}; expand{W,b}; project{W,b}; layerScale{γ:[c]}. -/
  | convNextBlock (c : Nat)
  /-- ConvNeXt block with the **real channel LayerNorm** — `γ,β : [c]` instead of two rank-0
      scalars (§2m). The normalisation axis changes with it (over `c` per spatial position, not
      over the whole `c·h·w` map), but that is invisible to the LAYOUT; what the layout sees is the
      affine going from 2 floats to 2c. Kept beside `convNextBlock` for `convBnNB`'s reason. -/
  | convNextBlockCh (c : Nat)
  /-- Per-channel LayerNorm over `d` features (normalize ∘ `[d]` affine). Params `{γ:[d], β:[d]}`
      — the non-scalar form (cf. `bn`, which is scalar-global). -/
  | layerNorm (d : Nat)
  /-- Pre-norm transformer block (dim `d`, MLP hidden `m`): LN1 → MHSA (Wq/Wk/Wv/Wo `[d,d]`) →
      +x → LN2 → MLP (`d→m→d`) → +x. Params: LN1{γ,β}; {Wq,bq,Wk,bk,Wv,bv,Wo,bo};
      LN2{γ,β}; {Wfc1[d,m],bfc1, Wfc2[m,d],bfc2} (per-channel `[d]` LN). -/
  | transformerBlock (d m : Nat)
  /-- A bare learned parameter tensor `(dims, initKind)` — e.g. ViT's CLS token / positional
      embedding (not produced by any standard layer). -/
  | param (dims : Array Nat) (kind : Nat)
deriving Repr

namespace VLayer

/-- conv → per-channel BN → relu: `{W=[oc,ic,k,k], b=[oc], γ=[oc], β=[oc]}`. -/
private def convBnSpec (ic oc k : Nat) : Array (Array Nat × Nat) :=
  #[(#[oc,ic,k,k],0),(#[oc],2),(#[oc],1),(#[oc],2)]
/-- conv → per-channel BN → relu, **no conv bias**: `{W=[oc,ic,k,k], γ=[oc], β=[oc]}`. -/
private def convBnNBSpec (ic oc k : Nat) : Array (Array Nat × Nat) :=
  #[(#[oc,ic,k,k],0),(#[oc],1),(#[oc],2)]
/-- identity basic block @ `c`: two conv→BN→relu units, no projection. -/
private def idBlk (c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,c,3,3],0),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],1),(#[c],2)]
/-- downsampling basic block `cin→c`: two conv→BN→relu + the **1×1** option-B projection
    shortcut (He et al. §3.3). It was 3×3 here until 2026-07-30 — §2k/§2l. -/
private def downBlk (cin c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,cin,3,3],0),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],1),(#[c],2),
    (#[c,cin,1,1],0),(#[c],1),(#[c],2)]   -- §2l step A: option-B 1×1 projection
private def stageSpec (ic oc count stride : Nat) : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) :=
    if stride != 1 || ic != oc then downBlk ic oc else idBlk oc
  for _ in [0:count-1] do a := a ++ idBlk oc
  return a

/-- Identity bottleneck block `cin→mid→mid→oc`, **no conv biases**: `1×1 → BN → relu`,
    `3×3 → BN → relu`, `1×1 → BN`, skip = identity. Nine tensors. -/
private def bneckIdBlk (cin mid oc : Nat) : Array (Array Nat × Nat) :=
  #[(#[mid,cin,1,1],0),(#[mid],1),(#[mid],2),
    (#[mid,mid,3,3],0),(#[mid],1),(#[mid],2),
    (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]
/-- Projecting bottleneck block: the identity body plus the `1×1` shortcut conv → BN.
    The projection comes **last**, matching the reference's `conv_bn` call order
    (`bottleneck_block_down` does idx, idx+1, idx+2, then idx+3 for the shortcut) — the same
    convention `downBlk` uses. Twelve tensors. -/
private def bneckDownBlk (cin mid oc : Nat) : Array (Array Nat × Nat) :=
  bneckIdBlk cin mid oc ++ #[(#[oc,cin,1,1],0),(#[oc],1),(#[oc],2)]
/-- A bottleneck stage. Width is `mid = oc/4` (the standard 4× expansion); the later blocks
    take `oc` as their input, so only the first one can change channels. -/
private def bottleneckStageSpec (ic oc count stride : Nat) : Array (Array Nat × Nat) := Id.run do
  let mid := oc / 4
  let mut a : Array (Array Nat × Nat) :=
    if stride != 1 || ic != oc then bneckDownBlk ic mid oc else bneckIdBlk ic mid oc
  for _ in [0:count-1] do a := a ++ bneckIdBlk oc mid oc
  return a

/-- The `(dims, initKind)` params this layer contributes, in func-arg order
    (`initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros (β / bias)). -/
def toSpecs : VLayer → Array (Array Nat × Nat)
  | convBn ic oc k _        => convBnSpec ic oc k
  | convBnNB ic oc k _      => convBnNBSpec ic oc k
  | maxPool _ _             => #[]
  | residualStage ic oc n s => stageSpec ic oc n s
  | bottleneckStage ic oc n s => bottleneckStageSpec ic oc n s
  | globalAvgPool           => #[]
  | dense ic oc             => #[(#[ic,oc],0),(#[oc],2)]
  | relu                    => #[]
  | conv ic oc k _          => #[(#[oc,ic,k,k],0),(#[oc],2)]
  | flatten                 => #[]
  | bn                      => #[(#[],1),(#[],2)]   -- scalar γ (ones), β (zeros)
  | bnPerChannel oc         => #[(#[oc],1),(#[oc],2)] -- per-channel γ:[oc] (ones), β:[oc] (zeros)
  | invertedResidual ic mid oc _ =>                 -- (expand 1×1 if t≠1, i.e. mid≠ic) | depthwise 3×3 | project 1×1, each +BN
    (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],2),(#[mid],1),(#[mid],2)] else #[]) ++
    #[(#[mid,1,3,3],0),(#[mid],2),(#[mid],1),(#[mid],2),
      (#[oc,mid,1,1],0),(#[oc],2),(#[oc],1),(#[oc],2)]
  | invertedResidualNB ic mid oc _ =>               -- as above, minus the three conv biases (§2m)
    (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] else #[]) ++
    #[(#[mid,1,3,3],0),(#[mid],1),(#[mid],2),
      (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]
  | mbConvSE ic mid oc r k =>                        -- (expand if t≠1) | depthwise k×k | SE | project, +BN
    (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],2),(#[mid],1),(#[mid],2)] else #[]) ++
    #[(#[mid,1,k,k],0),(#[mid],2),(#[mid],1),(#[mid],2),
      (#[mid,r],0),(#[r],2),(#[r,mid],0),(#[mid],2),
      (#[oc,mid,1,1],0),(#[oc],2),(#[oc],1),(#[oc],2)]
  | mbConvSENB ic mid oc r k =>                      -- as above, minus the BN-followed convs' biases;
    (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] else #[]) ++  -- SE's two KEEP theirs
    #[(#[mid,1,k,k],0),(#[mid],1),(#[mid],2),
      (#[mid,r],0),(#[r],2),(#[r,mid],0),(#[mid],2),
      (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]
  | uib ic oc expand _stride preDWk postDWk =>       -- (pre-DW if k≠0) | expand 1×1 | (post-DW if k≠0) | project 1×1, each +BN, all bias-free
    let mid := ic * expand
    (if preDWk > 0 then #[(#[ic,1,preDWk,preDWk],0),(#[ic],1),(#[ic],2)] else #[]) ++
    #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] ++
    (if postDWk > 0 then #[(#[mid,1,postDWk,postDWk],0),(#[mid],1),(#[mid],2)] else #[]) ++
    #[(#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]
  | fusedMbConvNB ic oc expand k _stride =>          -- fused k×k conv (ic→mid) +BN+swish | project 1×1 (mid→oc) +BN
    let mid := if expand == 1 then oc else ic * expand
    #[(#[mid,ic,k,k],0),(#[mid],1),(#[mid],2)] ++
    (if expand == 1 then #[] else #[(#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)])
  | convNextBlock c =>                               -- depthwise 7×7 | LN(scalar) | expand | project | layerScale
    #[(#[c,1,7,7],0),(#[c],2),(#[],1),(#[],2),
      (#[4*c,c,1,1],0),(#[4*c],2),(#[c,4*c,1,1],0),(#[c],2),(#[c],1)]
  | convNextBlockCh c =>                             -- as above with a per-channel LN affine
    #[(#[c,1,7,7],0),(#[c],2),(#[c],1),(#[c],2),
      (#[4*c,c,1,1],0),(#[4*c],2),(#[c,4*c,1,1],0),(#[c],2),(#[c],1)]
  | layerNorm d             => #[(#[d],1),(#[d],2)]   -- per-channel γ,β
  | transformerBlock d m =>                          -- LN1 | Wq/Wk/Wv/Wo | LN2 | MLP(d→m→d)
    #[(#[d],1),(#[d],2),
      (#[d,d],0),(#[d],2),(#[d,d],0),(#[d],2),(#[d,d],0),(#[d],2),(#[d,d],0),(#[d],2),
      (#[d],1),(#[d],2),
      (#[d,m],0),(#[m],2),(#[m,d],0),(#[d],2)]
  | param dims kind         => #[(dims, kind)]

end VLayer

/-- A verified net as a NetSpec-style architecture. `layers` is the single source of truth;
    the param layout (`toSpecs`) and input width (`d0`) are derived from it. -/
structure VerifiedNetSpec where
  name     : String
  /-- Names the committed, audited render `verified_mlir/<slug>_{train_step,fwd}.mlir`. -/
  slug     : String
  inC      : Nat
  imageH   : Nat
  imageW   : Nat
  nClasses : Nat := 10
  data     : VerifiedData
  layers   : List VLayer
  blurb    : String
  /-- Per-BN-layer channel counts in forward order (empty = LayerNorm / no-BN). Drives running-stats
      BN threading in `trainAdamSched` — see `VerifiedNet.bnChannels`. -/
  bnChannels : Array Nat := #[]
  /-- **Stochastic-depth keep probabilities**, one per drop site, in the render's signature order
      (`planning/stochastic_depth.md`). Empty on every net without a `*sd` render.

      ⚠ A SECOND hand-list against the renderer's `enetDropIdxs`/`enetDropTotal` — the same
      two-lists shape as `toSpecs == XLayout.specs`, and for the same structural reason: this file
      sits DOWNSTREAM of `VerifiedTrain`, so the renderer cannot share the definition by import
      without inverting the dependency. `tests/TestDropPathRamp.lean` is the `#guard` that pins
      them, and it is what stops the ramp drifting the way §2k's `α/K` did. -/
  dropKeeps : Array Float := #[]
  /-- Which directory this net's artifacts live in — see `VerifiedNet.mlirDir`. Default
      `verified_mlir/` (the certified, pinned corpus); the width/batch SWEEP specs set
      `.lake/build` because they render from argv at run time and their output is a build product. -/
  mlirDir : String := "verified_mlir"
  /-- ▶ CLASSIFIER DROPOUT (`recipe_gaps.md` gap C) — `(keep_prob, per-example width)`, `none` when
      the net has none. See `VerifiedNet.dropoutKeep` for why the WIDTH is carried: this mask is
      per-ELEMENT where `dropKeeps` above is per-example, and every downstream difference (blob
      shape, draw count, DP shard split) falls out of that one number. -/
  dropoutKeep : Option (Float × Nat) := none
  /-- The generated ImageNet batch shim this net streams — see `VerifiedNet.shimScript` for why
      there is no default and why an empty one refuses instead of falling back. Required on every
      `.imagenet` spec; meaningless on the others. -/
  shimScript : String := ""
  /-- Does `<slug>_train_step.mlir` return the trailing report-only `%loss` scalar? Per-RENDER,
      and only the `mlp` / `cnn` chapter-2/3 renders carry it. See `VerifiedNet.lossSlot`. -/
  lossSlot : Bool := false

namespace VerifiedNetSpec

/-- The full `(dims, initKind)` param list, folded from `layers` (func-arg order). -/
def toSpecs (s : VerifiedNetSpec) : Array (Array Nat × Nat) :=
  s.layers.foldl (fun acc L => acc ++ L.toSpecs) #[]

/-- Per-example flattened input width. -/
def d0 (s : VerifiedNetSpec) : Nat := s.inC * s.imageH * s.imageW

/-- Lower to the runtime `VerifiedNet` the driver consumes. -/
def toNet (s : VerifiedNetSpec) : VerifiedNet :=
  { name := s.name, slug := s.slug, specs := s.toSpecs, d0 := s.d0,
    nClasses := s.nClasses, data := s.data, blurb := s.blurb, bnChannels := s.bnChannels,
    dropKeeps := s.dropKeeps, dropoutKeep := s.dropoutKeep, shimScript := s.shimScript,
    mlirDir := s.mlirDir, lossSlot := s.lossSlot }

/-- Train end-to-end (delegates to the shared `VerifiedNet.train` driver). -/
def train (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.train cfg dataDir

/-- Train the 2-parameter linear path (Chapter 1); see `VerifiedNet.trainLinear`. -/
def trainLinear (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.trainLinear cfg dataDir

/-- Phase-3 PGD adversarial attack (Chapter 1 linear); see `VerifiedNet.attackPgd`. -/
def attackPgd (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgd cfg dataDir

/-- Phase-3 PGD attack on the MLP (Chapter 2); see `VerifiedNet.attackPgdMlp`. -/
def attackPgdMlp (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdMlp cfg dataDir

/-- Phase-3 PGD attack on the CNN (Chapter 3, the conv rung); see `VerifiedNet.attackPgdCnn`. -/
def attackPgdCnn (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdCnn cfg dataDir

/-- Spectral-norm-constrained MLP training study; see `VerifiedNet.attackPgdSpectralMlp`. -/
def attackPgdSpectralMlp (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  s.toNet.attackPgdSpectralMlp cfg dataDir caps

/-- Spectral-norm-constrained CNN training study; see `VerifiedNet.attackPgdSpectralCnn`. -/
def attackPgdSpectralCnn (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  s.toNet.attackPgdSpectralCnn cfg dataDir caps

/-- PGD attack on the CIFAR-10 CNN (the deeper conv rung); see `VerifiedNet.attackPgdCifar`. -/
def attackPgdCifar (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdCifar cfg dataDir

/-- PGD attack on the CIFAR-10 CNN + per-channel BatchNorm; see `VerifiedNet.attackPgdCifarBn`. -/
def attackPgdCifarBn (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdCifarBn cfg dataDir

/-- Spectral-norm-constrained CIFAR training study; see `VerifiedNet.attackPgdSpectralCifar`. -/
def attackPgdSpectralCifar (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  s.toNet.attackPgdSpectralCifar cfg dataDir caps

/-- Randomized-smoothing certificate (Cohen 2019, depth-independent); see
    `VerifiedNet.smoothCertify`. Forward-only — works on any spec via its rendered fwd. -/
def smoothCertify (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (sigmas : List Float) : IO Unit :=
  s.toNet.smoothCertify cfg dataDir sigmas

end VerifiedNetSpec
