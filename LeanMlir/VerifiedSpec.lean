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

namespace VerifiedNetSpec

/-- The full `(dims, initKind)` param list, folded from `layers` (func-arg order). -/
def toSpecs (s : VerifiedNetSpec) : Array (Array Nat × Nat) :=
  s.layers.foldl (fun acc L => acc ++ L.toSpecs) #[]

/-- Per-example flattened input width. -/
def d0 (s : VerifiedNetSpec) : Nat := s.inC * s.imageH * s.imageW

/-- Lower to the runtime `VerifiedNet` the driver consumes. -/
def toNet (s : VerifiedNetSpec) : VerifiedNet :=
  { name := s.name, slug := s.slug, specs := s.toSpecs, d0 := s.d0,
    nClasses := s.nClasses, data := s.data, blurb := s.blurb }

/-- Train end-to-end (delegates to the shared `VerifiedNet.train` driver). -/
def train (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.train cfg dataDir

/-- Train the 2-parameter linear path (Chapter 2); see `VerifiedNet.trainLinear`. -/
def trainLinear (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.trainLinear cfg dataDir

end VerifiedNetSpec
