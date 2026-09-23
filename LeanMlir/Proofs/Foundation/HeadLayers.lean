import LeanMlir.Proofs.Foundation.CertifiedChain
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose

/-! # The classifier head as `CertLayer`s — batched GAP and the dense classifier

Every conv net in the suite ends in global average pooling and a dense classifier, and both are
globally certified (GAP is linear, dense is affine), so `ok := True`. Written once here and shared
by MobileNetV4, ResNet-34 and ResNet-50.

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
