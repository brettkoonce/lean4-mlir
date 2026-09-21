import LeanMlir.Proofs.Codegen.StableHLO

/-! # The three BatchNorm emit sites every batch-BN renderer shares — batch BN or SYNCHRONISED BN

Lifted out of `ResNet34RenderB.lean` (2026-09-21, `planning/global_bn_verified.md` §3.3) so the
MobileNetV2 and EfficientNet-B0 renderers emit the same sync-BN composition ResNet-34 does. Each
site is one BatchNorm node at `replicas ≤ 1` — byte-for-byte what the renderers emitted before —
and at `replicas > 1` the sync-BN subgraph whose `den`
[`Foundation/DataParallelSync.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/DataParallelSync.lean) states
(the forward's two collectives, the backward's one, the γ gradient reading the forward's
statistics).

⚠ **Tags.** A site's `tag` is its γ parameter's name without `%`; the collectives it emits are
named from it (`{tag}mu`, `{tag}var`, and the backward's `{tag}dst`), so tags must be unique per
site and must not collide with any parameter's name.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- **One BatchNorm FORWARD site.** At `replicas ≤ 1` the batch-BN node `bnBatchF`. At
    `replicas > 1` the sync-BN composition of `planning/global_bn_verified.md` §2b, in TWO rounds
    (Chan's parallel variance): this replica's μ_r (`bnBatchMeanB`) all-reduced to the global μ;
    then `σ²_r + (μ_r − μ)²` (`bnBatchVarAtB`) all-reduced to the global σ² — each by
    `prettyAllReduceMean`, the SAME collective node the parameter gradients ride; packed
    (`bnPackB`); then `bnSyncF`, normalising with the global statistics. Returns `(code, y, st)`,
    `st` the packed `[μ ‖ σ²]` SSA name (`""` when there is no collective): the backward, the γ
    gradient and the handed-back running stats all read it, which is what makes the four agree
    on ONE `x̂`.

    ⚠ `tag` names the collectives' SSA values (`%arsum{tag}mu` / `%armean{tag}var` …), so it must
    be unique per site and disjoint from every parameter's (`{p}g1` is a parameter; `{p}g1mu`
    is this). -/
def bnFwdSite (B oc hh ww : Nat) (sync : Bool) (replicas : Nat)
    (epsStr gN btN tag xIn : String) :
    StateM Proofs.StableHLO.EmitS (String × String × String) := do
  let zoc : Vec oc := fun _ => 0
  let zin : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zst : Vec (oc+oc) := fun _ => 0
  if !sync then
    let (c, n) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) gN btN epsStr 0 zoc zoc (.operand xIn zin))
    pure (c, n, "")
  else
    let (cM, nM)   ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := ww) (.operand xIn zbn))
    let (cAM, nAM) ← prettyAllReduceMean nM [oc] s!"{tag}mu" replicas
    let (cV, nV)   ← pretty B (.bnBatchVarAtB (N := B) (oc := oc) (h := hh) (w := ww) (.operand xIn zbn) (.operand nAM zoc))
    let (cAV, nAV) ← prettyAllReduceMean nV [oc] s!"{tag}var" replicas
    let (cP, nP)   ← pretty B (.bnPackB (oc := oc) (.operand nAM zoc) (.operand nAV zoc))
    let (cY, nY)   ← pretty B (.bnSyncF (N := B) (oc := oc) (h := hh) (w := ww) gN btN epsStr 0 zoc zoc (.operand xIn zbn) (.operand nP zst))
    pure (cM ++ cAM ++ cV ++ cAV ++ cP ++ cY, nY, nP)

/-- **One BatchNorm BACKWARD site** — the input cotangent. At `replicas ≤ 1` `bnBatchBack`; at
    `replicas > 1` this replica's `[μ ‖ σ² ‖ mean(γ·dy) ‖ mean(x̂·γ·dy)]` (`bnSyncDyStatsB`, reading
    the forward's packed `st`, so its `x̂` is the forward's), all-reduced, then `bnSyncBack`. One
    collective per BN layer in the backward (three per layer per step with the forward's two). -/
def bnBackSite (B oc hh ww : Nat) (sync : Bool) (replicas : Nat)
    (epsStr gN xN tag dyIn st : String) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
  let zoc : Vec oc := fun _ => 0
  let zbn : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zst : Vec (oc+oc) := fun _ => 0
  let zds : Vec (oc+oc+(oc+oc)) := fun _ => 0
  if !sync then
    pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) gN xN epsStr 0 zoc zbn (.operand dyIn zbn))
  else
    let (cD, nD) ← pretty B (.bnSyncDyStatsB (N := B) (oc := oc) (h := hh) (w := ww) gN xN epsStr 0 zoc zbn (.operand dyIn zbn) (.operand st zst))
    let (cA, nA) ← prettyAllReduceMean nD [oc+oc+(oc+oc)] tag replicas
    let (cX, nX) ← pretty B (.bnSyncBack (N := B) (oc := oc) (h := hh) (w := ww) gN xN epsStr 0 zoc zbn (.operand dyIn zbn) (.operand nA zds))
    pure (cD ++ cA ++ cX, nX)

/-- **One BatchNorm γ-GRADIENT site.** `bnGammaGradB` rebuilds `x̂` from ITS OWN batch, which under
    sync-BN is not the `x̂` the forward used — so at `replicas > 1` it is `bnSyncGammaGradB`,
    reading the forward's packed global statistics (§2b's fifth op). β's gradient is `Σ dy`, reads
    no statistic, and stays `bnBetaGradB` at every replica count. -/
def bnGammaSite (B oc hh ww : Nat) (sync : Bool) (epsStr xN dyIn st : String) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
  let zbn : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zst : Vec (oc+oc) := fun _ => 0
  if !sync then
    pretty B (.bnGammaGradB (N := B) (oc := oc) (h := hh) (w := ww) xN epsStr 0 zbn (.operand dyIn zbn))
  else
    pretty B (.bnSyncGammaGradB (N := B) (oc := oc) (h := hh) (w := ww) xN epsStr 0 zbn (.operand dyIn zbn) (.operand st zst))

end Proofs.StableHLO
