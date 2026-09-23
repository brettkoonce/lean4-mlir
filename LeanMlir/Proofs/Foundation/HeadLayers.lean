import LeanMlir.Proofs.Foundation.CertifiedChain
import LeanMlir.Proofs.Foundation.BatchedStages
import LeanMlir.Proofs.Foundation.BackwardMaps

/-! # Shared stem-pool and head `CertLayer`s — batched GAP, the dense classifier, the 3×3/s2 pool

Every conv net in the suite ends in global average pooling and a dense classifier, and both are
globally certified (GAP is linear, dense is affine), so `ok := True`. Written once here and shared
by MobileNetV4, ResNet-34 and ResNet-50. The ResNets' 3×3/s2 stem pool (`r34PoolLayer`) is here
too; it is certified where no example's window ties (`R34PoolSmoothAt`).

⭐ **Both backward graphs tie by `rfl`.** `den` of `.gapBackBatched` is definitionally the row-wise
GAP VJP, and `den` of `.denseRowBack` is `rowDenseBackFlat`, which is what `batchMap_has_vjp`
reduces to. ⚠ GAP's VJP does not depend on its input, and `den .gapBackBatched` uses that by
evaluating the backward at `fun _ => 0`. That is sound because GAP is linear, and it is why the tie
holds at every `x`.
-/

namespace Proofs.StableHLO

/-- Batched **global average pool** as a `CertLayer`. Globally certified — GAP is linear. -/
noncomputable def gapLayer (N : Nat) {c h w : Nat} :
    CertLayer (N * (c * h * w)) (N * c) where
  fwd := batchMap N (globalAvgPoolFlat c h w)
  ok := fun _ => True
  diff := fun x _ => (batchMap_differentiable (globalAvgPoolFlat c h w)
    (globalAvgPoolFlat_differentiable c h w)) x
  vjp := fun x _ => (batchMap_has_vjp (N := N) (globalAvgPoolFlat c h w)
    (globalAvgPoolFlat_has_vjp c h w) (globalAvgPoolFlat_differentiable c h w)).toHasVJPAt x
  graph := fun _ e => .gapBackBatched (N := N) (c := c) (h := h) (w := w) e
  faithful := fun _ _ _ => rfl

/-- The GAP layer's forward is the batched global average pool. -/
theorem gapLayer_fwd_apply (N : Nat) {c h w : Nat} (v : Vec (N * (c * h * w))) :
    (gapLayer N (c := c) (h := h) (w := w)).fwd v
      = StableHLO.batchMap N (globalAvgPoolFlat c h w) v := rfl

/-- Batched **dense classifier** as a `CertLayer`. Globally certified — dense is affine. -/
noncomputable def denseLayer (N : Nat) {a nC : Nat} (W : Mat a nC) (b : Vec nC) :
    CertLayer (N * a) (N * nC) where
  fwd := batchMap N (dense W b)
  ok := fun _ => True
  diff := fun x _ => (batchMap_differentiable (dense W b) (dense_differentiable W b)) x
  vjp := fun x _ => (batchMap_has_vjp (N := N) (dense W b) (dense_has_vjp W b)
    (dense_differentiable W b)).toHasVJPAt x
  graph := fun _ e => .denseRowBack (N := N) (a := a) (c := nC) "%Wd" W e
  faithful := fun _ _ _ => rfl

/-- The classifier layer's forward is the batched dense. -/
theorem denseLayer_fwd_apply (N : Nat) {a nC : Nat} (W : Mat a nC) (b : Vec nC)
    (v : Vec (N * a)) :
    (denseLayer N W b).fwd v = StableHLO.batchMap N (Proofs.dense W b) v := rfl

end Proofs.StableHLO

namespace Proofs

/-- The stem pool has no argmax tie, **per example**: a tie is a property of one image's 3×3
    window, so the condition is stated on each row of the batched activation. This is the shape
    `batchMap_has_vjp_at` consumes. -/
def R34PoolSmoothAt (N h w : Nat) {oc : Nat} (v : Vec (N * (oc * (2 * h) * (2 * w)))) : Prop :=
  ∀ r : Fin N,
    MaxPool3s2Smooth (Tensor3.unflatten (Mat.unflatten v r) : Tensor3 oc (2 * h) (2 * w))

/-- The batched 3×3/s2 stem pool as a `CertLayer`, certified where no example's window ties. Its
    backward graph is the render's `maxPool3s2BackB`, and it denotes `batchMap_has_vjp_at`'s
    backward definitionally once the two spellings of the scatter are identified. -/
noncomputable def r34PoolLayer (N : Nat) {c h w : Nat} (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    StableHLO.CertLayer (N * (c * (2 * h) * (2 * w))) (N * (c * h * w)) where
  fwd := StableHLO.batchMap N (maxPool3s2Flat c h w)
  ok := R34PoolSmoothAt N h w
  diff := fun v hv => batchMap_differentiableAt _ _
    (fun r => maxPool3s2Flat_differentiableAt_vec _ (hv r) hc hh hw)
  vjp := fun v hv => batchMap_has_vjp_at _ _ (fun r => maxPool3s2Flat_has_vjp_at_vec _ (hv r))
    (fun r => maxPool3s2Flat_differentiableAt_vec _ (hv r) hc hh hw)
  graph := fun v e => .maxPool3s2BackB "%stemR" v e
  faithful := fun v _ e => by rw [den_maxPool3s2BackB_eq_flatBackB]; rfl

end Proofs
