import LeanMlir.Proofs.Foundation.Bf16Erasure
import LeanMlir.Proofs.Foundation.DataParallel.SyncKit

/-! # The tie predicates at either precision — `ConvWTiedBAt bf16`, `ConvWSyncAt bf16`, …

`GradNodesB` states each parameter gradient node against the certified `Σ_n` gradient
(`ConvWTiedB`, …), `ParamGradNodes` makes it a loss derivative (`convW_hasGradAt`, …) and
`SyncKit` ties the all-reduced collective to the batch-`R·N` node (`ConvWSync`, …) — all at the
f32 constructor. The ImageNet renders the book reports train from are bf16, and a bf16 render
emits the `*GradBBf16` kind at every conv weight gradient. This file states the same three
things with the node chosen by the renderers' own switch (`StableHLO.PrecisionSwitch`:
`convWeightGradBAt bf16 id …` is the f32 kind at `false`, the bf16 kind at `true`), and proves
each for either value from the f32 lemma and `Bf16Erasure` (`den_convWeightGradBAt_id`): at the
identity rounding the bf16 node denotes what the f32 node does, so a tie stated on the switch
reads the bf16 artifact's text over ℝ exactly as the f32 tie reads the f32 artifact's.

Only the kinds ResNet-34 emits for now — `conv` and `convStrided`; the other nets' kinds
(XLA-`SAME`, depthwise, stride-4, row-dense, patch-embed) follow the same pattern as each net's
tie moves onto the switch (planning/bf16_tie.md §3.3). At `false` each predicate is its f32
original by `rfl` (`convWTiedBAt_false`, …).

Nothing here is about the size of the rounding: `rnd` is `id` throughout, as in the renderers. -/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.GradNodeB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The nodes against the certified gradient
-- ════════════════════════════════════════════════════════════════

/-- `ConvWTiedB` with the node chosen by the switch: the conv weight gradient node, at either
    precision, denotes the certified batched `Σ_n` gradient. -/
def ConvWTiedBAt (bf16 : Bool) (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) :
    Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (SHlo.convWeightGradBAt bf16 (h := h) (w := w) id xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j

theorem convWTiedBAt_false (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) :
    ConvWTiedBAt false N h w xN cotN b x W cot = ConvWTiedB N h w xN cotN b x W cot := rfl

theorem convWTiedBAt_holds (bf16 : Bool) {N h w ic oc kH kW : Nat} {xN cotN : String} {b : Vec oc}
    {x : Vec (N * (ic * h * w))} {W : Kernel4 oc ic kH kW} {cot : Vec (N * (oc * h * w))} :
    ConvWTiedBAt bf16 N h w xN cotN b x W cot := fun idx => by
  rw [Bf16Fold.den_convWeightGradBAt_id]
  exact convWTiedB_holds idx

/-- `ConvStridedWTiedB` on the switch (symmetric padding, ResNet's stride-2 sites). -/
def ConvStridedWTiedBAt (bf16 : Bool) (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (SHlo.convStridedWeightGradBAt bf16 (h := h) (w := w) id xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2 (Kernel4.unflatten v') b
                    (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j

theorem convStridedWTiedBAt_false (N h w : Nat) {ic oc kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) :
    ConvStridedWTiedBAt false N h w xN cotN b x W cot = ConvStridedWTiedB N h w xN cotN b x W cot :=
  rfl

theorem convStridedWTiedBAt_holds (bf16 : Bool) {N h w ic oc kH kW : Nat} {xN cotN : String}
    {b : Vec oc} {x : Vec (N * (ic * (2 * h) * (2 * w)))} {W : Kernel4 oc ic kH kW}
    {cot : Vec (N * (oc * h * w))} : ConvStridedWTiedBAt bf16 N h w xN cotN b x W cot :=
  fun idx => by
    rw [Bf16Fold.den_convStridedWeightGradBAt_id]
    exact convStridedWTiedB_holds idx

-- ════════════════════════════════════════════════════════════════
-- § The nodes as loss derivatives
-- ════════════════════════════════════════════════════════════════

/-- `convW_hasGradAt` on the switch: the conv weight gradient node, at either precision, is the
    gradient of `G` after the conv in the weight. -/
theorem convWAt_hasGradAt (bf16 : Bool) {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConv W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConv (Kernel4.unflatten θ) b) x)) (Kernel4.flatten W)
      (den (SHlo.convWeightGradBAt bf16 (h := h) (w := w) id xN b x W (.operand cotN cot))) := by
  rw [Bf16Fold.den_convWeightGradBAt_id]
  exact convW_hasGradAt xN cotN b x W hG

/-- `convStridedW_hasGradAt` on the switch. -/
theorem convStridedWAt_hasGradAt (bf16 : Bool) {N ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    {G : Vec (N * (oc * h * w)) → Vec 1} {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (batchMap N (flatConvStride2 W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConvStride2 (Kernel4.unflatten θ) b) x))
      (Kernel4.flatten W)
      (den (SHlo.convStridedWeightGradBAt bf16 (h := h) (w := w) id xN b x W
        (.operand cotN cot))) := by
  rw [Bf16Fold.den_convStridedWeightGradBAt_id]
  exact convStridedW_hasGradAt xN cotN b x W hG

end Proofs.GradNodeB

-- ════════════════════════════════════════════════════════════════
-- § The collectives
-- ════════════════════════════════════════════════════════════════

namespace Proofs.SyncKit

open scoped BigOperators

/-- `ConvWSync` on the switch: the all-reduced mean of the replicas' conv weight gradient nodes,
    at either precision, is the batch-`R·N` node of the same kind. -/
def ConvWSyncAt (bf16 : Bool) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cots : Fin R → Vec (N * (oc * h * w))) (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (.allReduceMeanF R hR t [oc, ic, kH, kW] (fun r =>
          SHlo.convWeightGradBAt bf16 (h := h) (w := w) id xN b (batchShard R N (ic * h * w) X r) W
            (.operand cotN (cots r)))) idx
      = den (SHlo.convWeightGradBAt bf16 (h := h) (w := w) id xN b X W (.operand cotN COT)) idx

theorem convWSyncAt_false (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cots : Fin R → Vec (N * (oc * h * w))) (COT : Vec ((R * N) * (oc * h * w))) :
    ConvWSyncAt false R hR N h w t xN cotN b X W cots COT
      = ConvWSync R hR N h w t xN cotN b X W cots COT := rfl

theorem convWSyncAt_of_scaled (bf16 : Bool) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc kH kW : Nat} (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * h * w)))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    ConvWSyncAt bf16 R hR N h w t xN cotN b X W cots COT := by
  intro idx
  have h := convWSync_of_scaled R hR N h w t xN cotN b X W cots COT hc idx
  simp only [den_allReduceMeanF, Bf16Fold.den_convWeightGradBAt_id] at h ⊢
  exact h

/-- `ConvStridedWSync` on the switch. -/
def ConvStridedWSyncAt (bf16 : Bool) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (.allReduceMeanF R hR t [oc, ic, kH, kW] (fun r =>
          SHlo.convStridedWeightGradBAt bf16 (h := h) (w := w) id xN b
            (batchShard R N (ic * (2 * h) * (2 * w)) X r) W (.operand cotN (cots r)))) idx
      = den (SHlo.convStridedWeightGradBAt bf16 (h := h) (w := w) id xN b X W
          (.operand cotN COT)) idx

theorem convStridedWSyncAt_false (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w))) :
    ConvStridedWSyncAt false R hR N h w t xN cotN b X W cots COT
      = ConvStridedWSync R hR N h w t xN cotN b X W cots COT := rfl

theorem convStridedWSyncAt_of_scaled (bf16 : Bool) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc kH kW : Nat} (t xN cotN : String) (b : Vec oc)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cots : Fin R → Vec (N * (oc * h * w))) (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    ConvStridedWSyncAt bf16 R hR N h w t xN cotN b X W cots COT := by
  intro idx
  have h := convStridedWSync_of_scaled R hR N h w t xN cotN b X W cots COT hc idx
  simp only [den_allReduceMeanF, Bf16Fold.den_convStridedWeightGradBAt_id] at h ⊢
  exact h

end Proofs.SyncKit
