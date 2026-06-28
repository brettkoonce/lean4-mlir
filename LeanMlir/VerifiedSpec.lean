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
  /-- max pool `k×k` / `stride`. No params. -/
  | maxPool (k stride : Nat)
  /-- A basic-block residual stage: `nBlocks` blocks at `oc` channels. The first block
      downsamples (and projects the skip) iff `stride ≠ 1 ∨ ic ≠ oc`; the rest are identity. -/
  | residualStage (ic oc nBlocks stride : Nat)
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
  /-- EfficientNet MBConv block (`ic→mid=t·ic→oc`, depthwise `k×k`, SE ratio `r`): expand 1×1
      (skipped when `mid=ic`, i.e. t=1) → BN → swish, depthwise k×k → BN → swish, squeeze-excite
      (`Ws₁[mid,r]`,`bs₁[r]`,`Ws₂[r,mid]`,`bs₂[mid]`, sigmoid gate), project 1×1 → BN. Params:
      (expand{W,b,γ,β} if t≠1) ++ depthwise{W,b,γ,β} ++ SE{Ws₁,bs₁,Ws₂,bs₂} ++ project{W,b,γ,β}. -/
  | mbConvSE (ic mid oc r k : Nat)
  /-- ConvNeXt block @ `c` channels (expand ratio 4): depthwise 7×7 → scalar-LN → 1×1 expand
      c→4c → GELU → 1×1 project 4c→c → layerScale (per-channel γ). Params: depthwise{W,b};
      LN{γ,β scalar}; expand{W,b}; project{W,b}; layerScale{γ:[c]}. -/
  | convNextBlock (c : Nat)
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
/-- identity basic block @ `c`: two conv→BN→relu units, no projection. -/
private def idBlk (c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2)]
/-- downsampling basic block `cin→c`: two conv→BN→relu + a 1-conv projection shortcut. -/
private def downBlk (cin c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,cin,3,3],0),(#[c],2),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2),
    (#[c,cin,3,3],0),(#[c],2),(#[c],1),(#[c],2)]
private def stageSpec (ic oc count stride : Nat) : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) :=
    if stride != 1 || ic != oc then downBlk ic oc else idBlk oc
  for _ in [0:count-1] do a := a ++ idBlk oc
  return a

/-- The `(dims, initKind)` params this layer contributes, in func-arg order
    (`initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros (β / bias)). -/
def toSpecs : VLayer → Array (Array Nat × Nat)
  | convBn ic oc k _        => convBnSpec ic oc k
  | maxPool _ _             => #[]
  | residualStage ic oc n s => stageSpec ic oc n s
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
  | mbConvSE ic mid oc r k =>                        -- (expand if t≠1) | depthwise k×k | SE | project, +BN
    (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],2),(#[mid],1),(#[mid],2)] else #[]) ++
    #[(#[mid,1,k,k],0),(#[mid],2),(#[mid],1),(#[mid],2),
      (#[mid,r],0),(#[r],2),(#[r,mid],0),(#[mid],2),
      (#[oc,mid,1,1],0),(#[oc],2),(#[oc],1),(#[oc],2)]
  | convNextBlock c =>                               -- depthwise 7×7 | LN(scalar) | expand | project | layerScale
    #[(#[c,1,7,7],0),(#[c],2),(#[],1),(#[],2),
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

namespace VerifiedNetSpec

/-- The full `(dims, initKind)` param list, folded from `layers` (func-arg order). -/
def toSpecs (s : VerifiedNetSpec) : Array (Array Nat × Nat) :=
  s.layers.foldl (fun acc L => acc ++ L.toSpecs) #[]

/-- Per-example flattened input width. -/
def d0 (s : VerifiedNetSpec) : Nat := s.inC * s.imageH * s.imageW

/-- Lower to the runtime `VerifiedNet` the driver consumes. -/
def toNet (s : VerifiedNetSpec) : VerifiedNet :=
  { name := s.name, slug := s.slug, specs := s.toSpecs, d0 := s.d0,
    nClasses := s.nClasses, data := s.data, blurb := s.blurb, bnChannels := s.bnChannels }

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

end VerifiedNetSpec
