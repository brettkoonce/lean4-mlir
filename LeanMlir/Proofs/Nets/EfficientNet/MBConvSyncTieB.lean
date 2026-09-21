import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB

/-! # The inverted-residual pieces every MBConv net's sync-BN step twin shares

MobileNetV2 (`MobileNetV2SyncStepTieB.lean`) and EfficientNet-B0 (`EfficientNetSyncStepTieG.lean`)
both tie their sync-BN data-parallel step to the single-device step at `R·N` on
`ResNet34SyncStepTieB.lean`'s four steps. Their blocks share the ops ResNet-34 does not have — the
depthwise conv and its symmetric-strided peer, the XLA-`SAME` strided stem conv, global average
pooling and the row-wise dense — and this file states each once, net-agnostically:

* **§1 homogeneity** — each input-VJP and weight-gradient node is linear in its cotangent;
* **§2 sharding** — each input-VJP is a per-example map, so it commutes with the batch cut;
* **§3 the collectives** — the replica mean of a weight-gradient node is `1/R` of the global node;
* **§4 the per-parameter DP ties** — a replica family at `R ×` the shards of a global cotangent
  gives the single-device node at `R·N` (`*_of_scaled`).

MobileNetV4's render uses the same three weight-gradient kinds. The XLA-`SAME` strided depthwise
is MobileNetV2's alone and stays in its twin.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.MBConvSyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (dInB dStridedInB gapInB)
open Proofs.ResNet34SyncTieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — linear in the cotangent
-- ════════════════════════════════════════════════════════════════

theorem dInB_smul (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (dy : Vec (N * (c * h * w))) (s : ℝ) :
    dInB N (h := h) (w := w) W b (fun i => s * dy i)
      = fun i => s * dInB N (h := h) (w := w) W b dy i :=
  batchMap_smul _ (fun s v => HasVJP.backward_smul _ _ s v) s dy

theorem dStridedInB_smul (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (dy : Vec (N * (c * h * w))) (s : ℝ) :
    dStridedInB N (h := h) (w := w) W b (fun i => s * dy i)
      = fun i => s * dStridedInB N (h := h) (w := w) W b dy i :=
  batchMap_smul _ (fun s v => HasVJP.backward_smul _ _ s v) s dy

theorem gapInB_smul (N c h w : Nat) (dy : Vec (N * c)) (s : ℝ) :
    gapInB N c h w (fun i => s * dy i) = fun i => s * gapInB N c h w dy i :=
  batchMap_smul _ (fun s v => HasVJP.backward_smul _ _ s v) s dy

/-- The row-wise input-VJP `dX = W·dy` (the classifier's, and the SE excite dense's) is linear in
    `dy`. -/
theorem rowDenseBackFlat_smul (N a c : Nat) (W : Mat a c) (dy : Vec (N * c)) (s : ℝ) :
    rowDenseBackFlat N a c W (fun i => s * dy i) = fun i => s * rowDenseBackFlat N a c W dy i := by
  funext idx
  simp only [rowDenseBackFlat, Mat.flatten, Mat.unflatten, Mat.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

/-- A `HasVJP3` backward is linear in its cotangent — `HasVJP.backward_smul`'s three-axis peer,
    read off `HasVJP3.correct`. The stride-1 depthwise weight gradient is stated through one. -/
theorem hasVJP3_backward_smul {c₁ h₁ w₁ c₂ h₂ w₂ : Nat} {f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂}
    (hf : HasVJP3 f) (x : Tensor3 c₁ h₁ w₁) (a : ℝ) (dy : Tensor3 c₂ h₂ w₂) :
    hf.backward x (fun i₁ i₂ i₃ => a * dy i₁ i₂ i₃)
      = fun j₁ j₂ j₃ => a * hf.backward x dy j₁ j₂ j₃ := by
  funext j₁ j₂ j₃
  rw [hf.correct, hf.correct, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => ?_)
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => ?_)
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

theorem depthwiseWeightGradB_smul {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (s : ℝ)
    (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [den, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [show Tensor3.unflatten (fun i => s * batchSlice N (c * h * w) cot n i)
        = fun i₁ i₂ i₃ => s * Tensor3.unflatten (batchSlice N (c * h * w) cot n) i₁ i₂ i₃ from rfl,
      hasVJP3_backward_smul]
  rfl

theorem depthwiseStridedWeightGradB_smul {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (s : ℝ) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [den, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

theorem convStridedXlaWeightGradB_smul {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (s : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [den, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

-- ════════════════════════════════════════════════════════════════
-- § 2. Sharding — every per-example input-VJP commutes with the batch cut
-- ════════════════════════════════════════════════════════════════

theorem dInB_shard {R N : Nat} {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    dInB N (h := h) (w := w) W b (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * h * w) (dInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem dStridedInB_shard {R N : Nat} {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    dStridedInB N (h := h) (w := w) W b (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * (2 * h) * (2 * w)) (dStridedInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem gapInB_shard {R N : Nat} (c h w : Nat) (DY : Vec ((R * N) * c)) (r : Fin R) :
    gapInB N c h w (batchShard R N c DY r) = batchShard R N (c * h * w) (gapInB (R * N) c h w DY) r :=
  (batchShard_batchMap _ DY r).symm

/-- The row-wise input-VJP is `batchMap` of `W·`, so it shards like every per-example lift. -/
theorem rowDenseBackFlat_shard {R N : Nat} (a c : Nat) (W : Mat a c) (DY : Vec ((R * N) * c))
    (r : Fin R) :
    rowDenseBackFlat N a c W (batchShard R N c DY r)
      = batchShard R N a (rowDenseBackFlat (R * N) a c W DY) r :=
  (batchShard_batchMap (Mat.mulVec W) DY r).symm

-- ════════════════════════════════════════════════════════════════
-- § 3. The collectives — `DataParallelSync`'s P4 at the three MBConv weight kinds
-- ════════════════════════════════════════════════════════════════

/-- **P4 at the depthwise weight** — each replica's `Σ_n` over its own examples, averaged, is `1/R`
    of the global batch's `Σ_n`. -/
theorem den_allReduceMeanF_depthwiseWeightGradB_shard {N c h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec c) (W : DepthwiseKernel c kH kW)
    (X DY : Vec ((R * N) * (c * h * w))) (dy : Fin R → SHlo (N * (c * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (c * h * w) DY r) (idx : Fin (c * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .depthwiseWeightGradB xN b (batchShard R N (c * h * w) X r) W (dy r))) idx
      = (1 / (R : ℝ)) * den (.depthwiseWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **P4 at the strided depthwise weight.** -/
theorem den_allReduceMeanF_depthwiseStridedWeightGradB_shard {N c h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec c) (W : DepthwiseKernel c kH kW)
    (X : Vec ((R * N) * (c * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (c * h * w)))
    (dy : Fin R → SHlo (N * (c * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (c * h * w) DY r) (idx : Fin (c * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .depthwiseStridedWeightGradB xN b (batchShard R N (c * (2 * h) * (2 * w)) X r) W
            (dy r))) idx
      = (1 / (R : ℝ)) * den (.depthwiseStridedWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **P4 at the XLA-`SAME` strided conv weight** (the stem) — `den_allReduceMeanF_convWeightGradB_shard`'s
    peer. Only the certificate differs from the symmetric strided one; the batch split is the same. -/
theorem den_allReduceMeanF_convStridedXlaWeightGradB_shard {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec oc) (W : Kernel4 oc ic kH kW)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (oc * h * w)))
    (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convStridedXlaWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) X r) W
            (dy r))) idx
      = (1 / (R : ℝ)) * den (.convStridedXlaWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

-- ════════════════════════════════════════════════════════════════
-- § 4. Per-parameter DP ties
-- ════════════════════════════════════════════════════════════════

/-- **A depthwise weight, DP-tied** — the collective over the `[c, 1, kH, kW]` kernel the render
    all-reduces, against the single-device node at the global batch. -/
def DepthwiseWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat} (t xN cotN : String)
    (b : Vec c) (X : Vec ((R * N) * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (cots : Fin R → Vec (N * (c * h * w))) (COT : Vec ((R * N) * (c * h * w))) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (.allReduceMeanF R hR t [c, 1, kH, kW] (fun r =>
          .depthwiseWeightGradB xN b (batchShard R N (c * h * w) X r) W (.operand cotN (cots r)))) idx
      = den (.depthwiseWeightGradB xN b X W (.operand cotN COT)) idx

/-- The strided depthwise weight, DP-tied. -/
def DepthwiseStridedWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat} (t xN cotN : String)
    (b : Vec c) (X : Vec ((R * N) * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cots : Fin R → Vec (N * (c * h * w))) (COT : Vec ((R * N) * (c * h * w))) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (.allReduceMeanF R hR t [c, 1, kH, kW] (fun r =>
          .depthwiseStridedWeightGradB xN b (batchShard R N (c * (2 * h) * (2 * w)) X r) W
            (.operand cotN (cots r)))) idx
      = den (.depthwiseStridedWeightGradB xN b X W (.operand cotN COT)) idx

/-- The stem's XLA-`SAME` strided conv weight, DP-tied. Tags are the render's: the collective is
    named for the parameter, over its `[oc, ic, kH, kW]` shape. -/
def ConvStridedXlaWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (.allReduceMeanF R hR t [oc, ic, kH, kW] (fun r =>
          .convStridedXlaWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) X r) W
            (.operand cotN (cots r)))) idx
      = den (.convStridedXlaWeightGradB xN b X W (.operand cotN COT)) idx

theorem depthwiseWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat}
    (t xN cotN : String) (b : Vec c) (X : Vec ((R * N) * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (cots : Fin R → Vec (N * (c * h * w))) (COT : Vec ((R * N) * (c * h * w)))
    (hc : ∀ r, cots r = batchShard R N (c * h * w) (fun i => (R : ℝ) * COT i) r) :
    DepthwiseWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_depthwiseWeightGradB_shard R hR t xN cotN _ b W X (fun i => (R : ℝ) * COT i)
      (fun r => .operand cotN (cots r)) (fun r => hc r) idx, depthwiseWeightGradB_smul, inv_mul_R R hR]

theorem depthwiseStridedWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat}
    (t xN cotN : String) (b : Vec c) (X : Vec ((R * N) * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) (cots : Fin R → Vec (N * (c * h * w)))
    (COT : Vec ((R * N) * (c * h * w)))
    (hc : ∀ r, cots r = batchShard R N (c * h * w) (fun i => (R : ℝ) * COT i) r) :
    DepthwiseStridedWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_depthwiseStridedWeightGradB_shard R hR t xN cotN _ b W X
      (fun i => (R : ℝ) * COT i) (fun r => .operand cotN (cots r)) (fun r => hc r) idx,
    depthwiseStridedWeightGradB_smul, inv_mul_R R hR]

theorem convStridedXlaWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    ConvStridedXlaWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_convStridedXlaWeightGradB_shard R hR t xN cotN _ b W X
      (fun i => (R : ℝ) * COT i) (fun r => .operand cotN (cots r)) (fun r => hc r) idx,
    convStridedXlaWeightGradB_smul, inv_mul_R R hR]

end Proofs.MBConvSyncTieB
