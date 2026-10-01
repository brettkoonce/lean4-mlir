import LeanMlir.Proofs.Foundation.ParamGrad
import LeanMlir.Proofs.Foundation.GradNodesB
import LeanMlir.Proofs.Foundation.SmoothedBatchLoss
import LeanMlir.Proofs.Foundation.CertifiedChain

/-! # ParamGradNodes — each batched parameter gradient node is a loss derivative

`GradNodesB` states each emitted parameter gradient node as its layer's parameter Jacobian
contracted with an arbitrary output cotangent. Here the cotangent is the gradient of a scalar `G`
at the layer's output (`HasGradAt`), and the node becomes `∂G/∂θ` with the layer's parameter
varied: one lemma per node kind, shared by every net, each concluding `HasGradAt` in the
parameter (the loss is differentiable there and the node is its gradient).

The BatchNorm γ/β nodes are stated in the transposed `[C, N·H·W]` layout at the `reassocB`
index; their lemmas re-sum the Jacobian over that permutation (`bnLAPerm`). A conv or depthwise
bias that the render reads with the β op (EfficientNet-B0's) is `biasBeta_hasGradAt`: the bias enters
as a channel broadcast (`*_bias_split`), so the channel sum the β node computes is its derivative.
-/

namespace Proofs.GradNodeB

open Proofs Proofs.StableHLO Proofs.BackLinks
open scoped BigOperators

/-- `cInB` — the emitted conv input-cotangent — is the batched conv VJP's backward, at any saved
    input. -/
theorem cInB_eq_batchMapBackward {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) :
    cInB N W b dy
      = (batchMapHasVJP (flatConv W b) (flatConvHasVJP W b) (flatConv_differentiable W b)).backward
          x dy :=
  convBackBatched_faithful "" W b x (.operand "" dy)

/-- `cStridedInB` — the emitted strided-conv input-cotangent — is the batched strided conv VJP's
    backward, at any saved input. -/
theorem cStridedInB_eq_batchMapBackward {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) :
    cStridedInB N W b dy
      = (batchMapHasVJP (flatConvStride2 W b) (flatConvStride2HasVJP W b)
          (flatConvStride2_differentiable W b)).backward x dy :=
  convStridedBackBatched_faithful "" W b x (.operand "" dy)

/-- `dInB` — the emitted depthwise input-cotangent — is the batched depthwise VJP's backward. -/
theorem dInB_eq_batchMapBackward {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Vec (N * (c * h * w))) (dy : Vec (N * (c * h * w))) :
    dInB N W b dy
      = (batchMapHasVJP (depthwiseFlat W b) (depthwiseFlatHasVJP W b)
          (depthwiseFlat_differentiable W b)).backward x dy :=
  depthwiseBackBatched_faithful "" W b x (.operand "" dy)

/-- `dStridedInB` — the emitted (symmetric) strided depthwise input-cotangent — is the batched
    strided depthwise VJP's backward, at any saved input. -/
theorem dStridedInB_eq_batchMapBackward {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (dy : Vec (N * (c * h * w))) :
    dStridedInB N W b dy
      = (batchMapHasVJP (depthwiseStride2Flat W b) (depthwiseStride2FlatHasVJP W b)
          (depthwiseStride2Flat_differentiable W b)).backward x dy :=
  depthwiseStridedBackBatched_faithful "" W b x (.operand "" dy)

-- ════════════════════════════════════════════════════════════════
-- § One stage back: the loss gradient pulled through each emitted backward op
--   Each lemma is `HasGradAt.comp` through a certified VJP, restated at the cotangent the render
--   emits (`bnInB`, `reluMaskB`, `cInB`, …), so a net's chain is one line per stage.
-- ════════════════════════════════════════════════════════════════

/-- Back through batch BN: the gradient at its input is `bnInB` of the gradient at its output. -/
theorem hasGradAt_bnBatchLA {N c h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (z : Vec (N * (c * h * w))) {G : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hG : HasGradAt G (bnBatchLA N c h w ε γ β z) dy) :
    HasGradAt (fun z' => G (bnBatchLA N c h w ε γ β z')) z (bnInB N c h w ε γ z dy) :=
  (hG.comp ((bnBatchLA_differentiable N c h w ε hε γ β) _)
    ((bnBatchLAHasVJP N c h w ε hε γ β).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N c h w ε hε γ β _ _).symm

/-- Back through relu, off its kink: `reluMaskB`. -/
theorem hasGradAt_relu {n : Nat} (x : Vec n) (hs : ∀ k, x k ≠ 0) {G : Vec n → Vec 1} {dy : Vec n}
    (hG : HasGradAt G (relu n x) dy) :
    HasGradAt (fun u => G (relu n u)) x (reluMaskB n x dy) :=
  HasGradAt.comp (f := relu n) (x := x) hG (relu_differentiableAt_of_smooth _ _ hs)
    (reluHasVJPAt _ _ hs)

/-- Back through relu6, off both kinks: `relu6MaskB`. -/
theorem hasGradAt_relu6 {n : Nat} (x : Vec n) (hs : ∀ k, x k ≠ 0 ∧ x k ≠ 6) {G : Vec n → Vec 1}
    {dy : Vec n} (hG : HasGradAt G (relu6 n x) dy) :
    HasGradAt (fun u => G (relu6 n u)) x (relu6MaskB n x dy) :=
  HasGradAt.comp (f := relu6 n) (x := x) hG (relu6_differentiableAt_of_smooth _ _ hs)
    (relu6HasVJPAt _ _ hs)

/-- Back through `u ↦ u + v` with the skip `v` held fixed: the gradient passes unchanged. This is
    how a residual block's body sees the loss once one of its parameters varies. -/
theorem hasGradAt_addConst {n : Nat} (u v : Vec n) {G : Vec n → Vec 1} {dy : Vec n}
    (hG : HasGradAt G (fun i => u i + v i) dy) :
    HasGradAt (fun u' => G (fun i => u' i + v i)) u dy :=
  HasGradAt.comp (f := fun u' i => u' i + v i) (x := u) hG (differentiableAt_id.add_const v)
    (addConstHasVJPAt (fun u => u) v _ differentiableAt_id (identityHasVJPAt _ _))

/-- …and through `u ↦ v + u`, the fixed branch on the left (a projected skip seen from the body). -/
theorem hasGradAt_constAdd {n : Nat} (v u : Vec n) {G : Vec n → Vec 1} {dy : Vec n}
    (hG : HasGradAt G (fun i => v i + u i) dy) :
    HasGradAt (fun u' => G (fun i => v i + u' i)) u dy :=
  HasGradAt.comp (f := fun u' i => v i + u' i) (x := u) hG (differentiableAt_id.const_add v)
    (constAddHasVJPAt v (fun u => u) _ differentiableAt_id (identityHasVJPAt _ _))

/-- At a residual layer the loss read at the BODY output is `u ↦ G (u + v)`: the skip is a constant
    once a body parameter varies, so its gradient there is still the block-output cotangent. -/
theorem _root_.Proofs.HasGradAt.residual_body {n : Nat} (L : CertLayer n n) (v : Vec n)
    {G : Vec n → Vec 1} {dy : Vec n} (hG : HasGradAt G ((CertLayer.residual L).fwd v) dy) :
    HasGradAt (fun u => G (fun i => u i + v i)) (L.fwd v) dy :=
  hasGradAt_addConst (L.fwd v) v hG

/-- Back through a relabelling `Fin n ≃ Fin m` along `n = m` (MobileNetV4's head reads `[N, c]` as
    `[N, c, 1, 1]`): the cotangent is read back along the same cast. -/
theorem hasGradAt_cast {n m : Nat} (e : n = m) (x : Vec n) {G : Vec m → Vec 1} {dy : Vec m}
    (hG : HasGradAt G (fun j => x (Fin.cast e.symm j)) dy) :
    HasGradAt (fun u => G (fun j => u (Fin.cast e.symm j))) x (fun i => dy (Fin.cast e i)) := by
  refine (HasGradAt.comp (f := reindexCLM (Fin.cast e.symm)) (x := x) hG
    (reindexCLM _).differentiableAt ((reindexHasVJP (Fin.cast e.symm)).toHasVJPAt x)).of_eq ?_
  funext i
  -- `reindexHasVJP`'s backward is this indicator sum by definition
  show ∑ k : Fin m, (if i = Fin.cast e.symm k then dy k else 0) = dy (Fin.cast e i)
  have hk : ∀ k : Fin m, i = Fin.cast e.symm k ↔ Fin.cast e i = k := fun k => by
    constructor
    · rintro rfl; exact Fin.ext rfl
    · rintro rfl; exact Fin.ext rfl
  simp only [hk, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

/-- Back through a certified layer, at a point it certifies: the layer's own backward. -/
theorem _root_.Proofs.StableHLO.CertLayer.hasGradAt_comp {m n : Nat} (L : CertLayer m n)
    (x : Vec m) (hx : L.ok x) {G : Vec n → Vec 1} {dy : Vec n} (hG : HasGradAt G (L.fwd x) dy) :
    HasGradAt (fun y => G (L.fwd y)) x ((L.vjp x hx).backward dy) :=
  HasGradAt.comp (f := L.fwd) (x := x) hG (L.diff x hx) (L.vjp x hx)

/-- Back through batch BN, at the certified backward's own spelling `bnBackB` (EfficientNet-B0's
    tie threads this form; the ResNets and MobileNets emit `bnInB`). -/
theorem hasGradAt_bnBackB {N c h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (z : Vec (N * (c * h * w))) {G : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hG : HasGradAt G (bnBatchLA N c h w ε γ β z) dy) :
    HasGradAt (fun z' => G (bnBatchLA N c h w ε γ β z')) z (bnBackB N c h w ε hε γ β z dy) :=
  hG.comp ((bnBatchLA_differentiable N c h w ε hε γ β) _)
    ((bnBatchLAHasVJP N c h w ε hε γ β).toHasVJPAt _)

/-- Back through swish: `swBackB`. Swish has no kink, so there is no smoothness hypothesis. -/
theorem hasGradAt_swish {n : Nat} (x : Vec n) {G : Vec n → Vec 1} {dy : Vec n}
    (hG : HasGradAt G (swish n x) dy) :
    HasGradAt (fun u => G (swish n u)) x (swBackB n x dy) :=
  HasGradAt.comp (f := swish n) (x := x) hG ((swish_differentiable n) x)
    ((swishHasVJP n).toHasVJPAt x)

/-- Back through a batched conv: `cInB`. -/
theorem hasGradAt_conv {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (batchMap N (flatConv W b) x) dy) :
    HasGradAt (fun y => G (batchMap N (flatConv W b) y)) x (cInB N W b dy) :=
  (hG.comp ((batchMap_differentiable _ (flatConv_differentiable W b)) _)
    ((batchMapHasVJP _ (flatConvHasVJP W b) (flatConv_differentiable W b)).toHasVJPAt _)).of_eq
    (cInB_eq_batchMapBackward (h := h) (w := w) W b _ _).symm

/-- Back through a batched symmetric strided conv: `cStridedInB`. -/
theorem hasGradAt_convStrided {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConvStride2 W b) x) dy) :
    HasGradAt (fun y => G (batchMap N (flatConvStride2 W b) y)) x (cStridedInB N W b dy) :=
  (hG.comp ((batchMap_differentiable _ (flatConvStride2_differentiable W b)) _)
    ((batchMapHasVJP _ (flatConvStride2HasVJP W b)
      (flatConvStride2_differentiable W b)).toHasVJPAt _)).of_eq
    (cStridedInB_eq_batchMapBackward (h := h) (w := w) W b _ _).symm

/-- Back through a batched depthwise: `dInB`. -/
theorem hasGradAt_depthwise {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Vec (N * (c * h * w))) {G : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hG : HasGradAt G (batchMap N (depthwiseFlat W b) x) dy) :
    HasGradAt (fun y => G (batchMap N (depthwiseFlat W b) y)) x (dInB N W b dy) :=
  (hG.comp ((batchMap_differentiable _ (depthwiseFlat_differentiable W b)) _)
    ((batchMapHasVJP _ (depthwiseFlatHasVJP W b)
      (depthwiseFlat_differentiable W b)).toHasVJPAt _)).of_eq
    (dInB_eq_batchMapBackward (h := h) (w := w) W b _ _).symm

/-- Back through a batched symmetric strided depthwise: `dStridedInB`. -/
theorem hasGradAt_depthwiseStrided {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) {G : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hG : HasGradAt G (batchMap N (depthwiseStride2Flat W b) x) dy) :
    HasGradAt (fun y => G (batchMap N (depthwiseStride2Flat W b) y)) x (dStridedInB N W b dy) :=
  (hG.comp ((batchMap_differentiable _ (depthwiseStride2Flat_differentiable W b)) _)
    ((batchMapHasVJP _ (depthwiseStride2FlatHasVJP W b)
      (depthwiseStride2Flat_differentiable W b)).toHasVJPAt _)).of_eq
    (dStridedInB_eq_batchMapBackward (h := h) (w := w) W b _ _).symm

-- ════════════════════════════════════════════════════════════════
-- § Convolutions and dense layers
-- ════════════════════════════════════════════════════════════════

/-- **Conv weight node = `∇_W G`.** -/
theorem convW_hasGradAt {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConv W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConv (Kernel4.unflatten θ) b) x)) (Kernel4.flatten W)
      (den (SHlo.convWeightGradB xN b x W (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => flatConv (Kernel4.unflatten θ) b y) (Kernel4.flatten W)) x) cot := by
    simpa only [Kernel4.unflatten_flatten] using hG
  have hP := hG'.param_batchMap (fun θ y => flatConv (Kernel4.unflatten θ) b y) x
    (fun y => (conv2d_weight_differentiable b (Tensor3.unflatten y)) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, convWGradB_den]
  rfl

/-- **Conv bias node = `∇_b G`.** -/
theorem convB_hasGradAt {N ic oc h w kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (N * (ic * h * w))) (b : Vec oc) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConv W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConv W θ) x)) b
      (den (SHlo.convBiasGradB (h := h) (w := w) W x b (.operand cotN cot))) := by
  have hP := hG.param_batchMap (fun θ y => flatConv W θ y) x
    (fun y => (conv2d_bias_differentiable W (Tensor3.unflatten y)) _)
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o, convBGradB_den]
  rfl

theorem flatConvStride2_weight_differentiable {ic oc h w kH kW : Nat} (b : Vec oc)
    (y : Vec (ic * (2 * h) * (2 * w))) :
    Differentiable ℝ (fun θ : Vec (oc * ic * kH * kW) =>
      (flatConvStride2 (Kernel4.unflatten θ) b y : Vec (oc * h * w))) := by
  unfold flatConvStride2 decimateFlat
  exact (reindexCLM _).differentiable.comp
    (conv2d_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten y))

theorem flatConvStride2_bias_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (y : Vec (ic * (2 * h) * (2 * w))) :
    Differentiable ℝ (fun θ : Vec oc => (flatConvStride2 W θ y : Vec (oc * h * w))) := by
  unfold flatConvStride2 decimateFlat
  exact (reindexCLM _).differentiable.comp
    (conv2d_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten y))

/-- **Stride-2 (symmetric) conv weight node = `∇_W G`.** -/
theorem convStridedW_hasGradAt {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    {G : Vec (N * (oc * h * w)) → Vec 1} {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (batchMap N (flatConvStride2 W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConvStride2 (Kernel4.unflatten θ) b) x))
      (Kernel4.flatten W) (den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => (flatConvStride2 (Kernel4.unflatten θ) b y : Vec (oc * h * w)))
        (Kernel4.flatten W)) x) cot := by
    simpa only [Kernel4.unflatten_flatten] using hG
  have hP := hG'.param_batchMap
    (fun θ y => (flatConvStride2 (Kernel4.unflatten θ) b y : Vec (oc * h * w))) x
    (fun y => (flatConvStride2_weight_differentiable b y) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, convStridedWGradB_den]

/-- **Stride-2 (symmetric) conv bias node = `∇_b G`.** -/
theorem convStridedB_hasGradAt {N ic oc h w kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConvStride2 W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConvStride2 W θ) x)) b
      (den (SHlo.convStridedBiasGradB (h := h) (w := w) W x b (.operand cotN cot))) := by
  have hP := hG.param_batchMap (fun θ y => (flatConvStride2 W θ y : Vec (oc * h * w))) x
    (fun y => (flatConvStride2_bias_differentiable W y) _)
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o, convStridedBGradB_den]

/-- **XLA-`SAME` strided conv weight node = `∇_W G`** (MobileNetV2's and B0's stem). -/
theorem convStridedXlaW_hasGradAt {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    {G : Vec (N * (oc * h * w)) → Vec 1} {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (batchMap N (flatConvStride2Xla W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConvStride2Xla (Kernel4.unflatten θ) b) x))
      (Kernel4.flatten W) (den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => (flatConvStride2Xla (Kernel4.unflatten θ) b y : Vec (oc * h * w)))
        (Kernel4.flatten W)) x) cot := by
    simpa only [Kernel4.unflatten_flatten] using hG
  have hP := hG'.param_batchMap
    (fun θ y => (flatConvStride2Xla (Kernel4.unflatten θ) b y : Vec (oc * h * w))) x
    (fun y => ((decimateOddFlat_differentiable oc h w).comp
      (conv2d_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten y))) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, convStridedXlaWGradB_den]

/-- **XLA-`SAME` strided conv bias node = `∇_b G`.** -/
theorem convStridedXlaB_hasGradAt {N ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc)
    {G : Vec (N * (oc * h * w)) → Vec 1} {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (batchMap N (flatConvStride2Xla W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (flatConvStride2Xla W θ) x)) b
      (den (SHlo.convStridedXlaBiasGradB (h := h) (w := w) W x b (.operand cotN cot))) := by
  have hP := hG.param_batchMap (fun θ y => (flatConvStride2Xla W θ y : Vec (oc * h * w))) x
    (fun y => ((decimateOddFlat_differentiable oc h w).comp
      (conv2d_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten y))) _)
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o, convStridedXlaBGradB_den]

/-- **Depthwise weight node = `∇_W G`.** -/
theorem depthwiseW_hasGradAt {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) {G : Vec (N * (c * h * w)) → Vec 1}
    {cot : Vec (N * (c * h * w))} (hG : HasGradAt G (batchMap N (depthwiseFlat W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (depthwiseFlat (Tensor3.unflatten θ) b) x))
      (Tensor3.flatten W) (den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => depthwiseFlat (h := h) (w := w)
        (Tensor3.unflatten θ : DepthwiseKernel c kH kW) b y) (Tensor3.flatten W)) x) cot := by
    simpa only [Tensor3.unflatten_flatten] using hG
  have hP := hG'.param_batchMap (fun θ y => depthwiseFlat (h := h) (w := w)
      (Tensor3.unflatten θ : DepthwiseKernel c kH kW) b y) x
    (fun y => (depthwise_weight_differentiable b (Tensor3.unflatten y)) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, depthwiseWGradB_den]
  rfl

/-- **Depthwise bias node = `∇_b G`.** -/
theorem depthwiseB_hasGradAt {N c h w kH kW : Nat} (cotN : String) (W : DepthwiseKernel c kH kW)
    (x : Vec (N * (c * h * w))) (b : Vec c) {G : Vec (N * (c * h * w)) → Vec 1}
    {cot : Vec (N * (c * h * w))} (hG : HasGradAt G (batchMap N (depthwiseFlat W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (depthwiseFlat W θ) x)) b
      (den (SHlo.depthwiseBiasGradB W x b (.operand cotN cot))) := by
  have hP := hG.param_batchMap (fun θ y => depthwiseFlat (h := h) (w := w) W θ y) x
    (fun y => (depthwise_bias_differentiable W (Tensor3.unflatten y)) _)
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o, depthwiseBGradB_den]
  rfl

/-- **Symmetric strided depthwise weight node = `∇_W G`** (MobileNetV4's `dw_mid` at its three
    downsampling rows). -/
theorem depthwiseStridedW_hasGradAt {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    {G : Vec (N * (c * h * w)) → Vec 1} {cot : Vec (N * (c * h * w))}
    (hG : HasGradAt G (batchMap N (depthwiseStride2Flat W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (depthwiseStride2Flat (Tensor3.unflatten θ) b) x))
      (Tensor3.flatten W) (den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => (depthwiseStride2Flat
        (Tensor3.unflatten θ : DepthwiseKernel c kH kW) b y : Vec (c * h * w)))
        (Tensor3.flatten W)) x) cot := by
    simpa only [Tensor3.unflatten_flatten] using hG
  have hP := hG'.param_batchMap (fun θ y => (depthwiseStride2Flat
      (Tensor3.unflatten θ : DepthwiseKernel c kH kW) b y : Vec (c * h * w))) x
    (fun y => by
      unfold depthwiseStride2Flat decimateFlat
      exact ((reindexCLM _).differentiable.comp
        (depthwise_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten y))) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, depthwiseStridedWGradB_den]

/-- **XLA-`SAME` strided depthwise weight node = `∇_W G`.** -/
theorem depthwiseStridedXlaW_hasGradAt {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    {G : Vec (N * (c * h * w)) → Vec 1} {cot : Vec (N * (c * h * w))}
    (hG : HasGradAt G (batchMap N (depthwiseStride2FlatXla W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (depthwiseStride2FlatXla (Tensor3.unflatten θ) b) x))
      (Tensor3.flatten W)
      (den (SHlo.depthwiseStridedXlaWeightGradB xN b x W (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => (depthwiseStride2FlatXla
        (Tensor3.unflatten θ : DepthwiseKernel c kH kW) b y : Vec (c * h * w)))
        (Tensor3.flatten W)) x) cot := by
    simpa only [Tensor3.unflatten_flatten] using hG
  have hP := hG'.param_batchMap (fun θ y => (depthwiseStride2FlatXla
      (Tensor3.unflatten θ : DepthwiseKernel c kH kW) b y : Vec (c * h * w))) x
    (fun y => ((decimateOddFlat_differentiable c h w).comp
      (depthwise_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten y))) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, depthwiseStridedXlaWGradB_den]

/-- **XLA-`SAME` strided depthwise bias node = `∇_b G`.** -/
theorem depthwiseStridedXlaB_hasGradAt {N c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * (2 * h) * (2 * w)))) (b : Vec c)
    {G : Vec (N * (c * h * w)) → Vec 1} {cot : Vec (N * (c * h * w))}
    (hG : HasGradAt G (batchMap N (depthwiseStride2FlatXla W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (depthwiseStride2FlatXla W θ) x)) b
      (den (SHlo.depthwiseStridedXlaBiasGradB (h := h) (w := w) W x b (.operand cotN cot))) := by
  have hP := hG.param_batchMap (fun θ y => (depthwiseStride2FlatXla W θ y : Vec (c * h * w))) x
    (fun y => ((decimateOddFlat_differentiable c h w).comp
      (depthwise_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten y))) _)
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o, depthwiseStridedXlaBGradB_den]

/-- **Dense weight node = `∇_W G`**, at the flat `Mat.flatten` index `finProdFinEquiv (i, j)`. -/
theorem denseW_hasGradAt {N a c : Nat} (xN cotN : String) (x : Vec (N * a)) (W : Mat a c)
    (b : Vec c) {G : Vec (N * c) → Vec 1} {cot : Vec (N * c)}
    (hG : HasGradAt G (batchMap N (dense W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (dense (Mat.unflatten θ) b) x)) (Mat.flatten W)
      (den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot))) := by
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => dense (Mat.unflatten θ) b y) (Mat.flatten W)) x) cot := by
    simpa only [Mat.unflatten_flatten] using hG
  have hP := hG'.param_batchMap (fun θ y => dense (Mat.unflatten θ) b y) x
    (fun y => (denseWeightMap_differentiable b y) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
  rw [hP.pdiv_eq, denseWGradB_den xN cotN x W b cot i j]

theorem dense_bias_differentiable {a c : Nat} (W : Mat a c) (x : Vec a) :
    Differentiable ℝ (fun b' : Vec c => dense W b' x) := by
  unfold dense; fun_prop

/-- **Dense bias node = `∇_b G`.** The node's statement carries one activation `x₀` for every
    example; the bias Jacobian is the identity whatever the activation, so any `x₀` serves. -/
theorem denseB_hasGradAt {N a c : Nat} (cotN : String) (W : Mat a c) (x₀ : Vec a)
    (x : Vec (N * a)) (b : Vec c) {G : Vec (N * c) → Vec 1} {cot : Vec (N * c)}
    (hG : HasGradAt G (batchMap N (dense W b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (dense W θ) x)) b
      (den (SHlo.denseBiasGradB (N := N) (.operand cotN cot))) := by
  have hP := hG.param_batchMap (fun θ y => dense W θ y) x
    (fun y => (dense_bias_differentiable W y) _)
  refine ⟨hP.differentiableAt, fun j => ?_⟩
  rw [hP.pdiv_eq j, denseBGradB_den cotN W x₀ b cot j]
  simp only [pdiv_dense_b]

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm γ / β, through the `[N,C,H,W] ↔ [C, N·H·W]` permutation
-- ════════════════════════════════════════════════════════════════

/-- The permutation `bnBatchLA` reads its per-channel core through: network index `J` ↦ the
    `[C, N·H·W]` cell `bnchwBackIdx (J at the mul_assoc cast)`. `bnBatchLA … v J` is
    `bnPerChannelFlat … (bnchwFwd … (reassocB … v)) (bnLAPerm … J)` by `rfl`. -/
noncomputable def bnLAPerm (N oc h w : Nat) : Fin (N * (oc * h * w)) ≃ Fin (oc * (N * (h * w))) where
  toFun J := bnchwBackIdx N oc h w (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) J)
  invFun j := Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm (bnchwFwdIdx N oc h w j)
  left_inv J := by simp [bnchwFwdIdx_bnchwBackIdx]
  right_inv j := by simp [bnchwBackIdx_bnchwFwdIdx]

theorem bnPerChannelFlat_gamma_differentiable (oc m : Nat) (ε : ℝ) (β : Vec oc)
    (v : Vec (oc * m)) : Differentiable ℝ (fun γ' : Vec oc => bnPerChannelFlat oc m ε γ' β v) := by
  unfold bnPerChannelFlat bnPerChannelMat Mat.flatten bnForward
  fun_prop

theorem bnPerChannelFlat_beta_differentiable (oc m : Nat) (ε : ℝ) (γ : Vec oc)
    (v : Vec (oc * m)) : Differentiable ℝ (fun β' : Vec oc => bnPerChannelFlat oc m ε γ β' v) := by
  unfold bnPerChannelFlat bnPerChannelMat Mat.flatten bnForward
  fun_prop

/-- A parameter entering `bnBatchLA` through its per-channel core: the loss gradient re-sums
    over the permutation into the core's layout. -/
private theorem bnLA_param_hasGradAt {P N oc h w : Nat} (F : Vec P → Vec (oc * (N * (h * w))))
    (hF : Differentiable ℝ F) {G : Vec (N * (oc * h * w)) → Vec 1} {θ : Vec P}
    {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (fun J => F θ (bnLAPerm N oc h w J)) cot) :
    HasGradAt (fun θ' => G (fun J => F θ' (bnLAPerm N oc h w J))) θ
      (fun i => ∑ j, pdiv F θ i j * bnchwFwd N oc h w (reassocB N oc h w cot) j) := by
  have hl : Differentiable ℝ (fun θ' => fun J => F θ' (bnLAPerm N oc h w J)) :=
    (reindexCLM (bnLAPerm N oc h w)).differentiable.comp hF
  have hP := hG.param (layer := fun θ' => fun J => F θ' (bnLAPerm N oc h w J)) (hl θ)
  refine ⟨hP.differentiableAt, fun i => ?_⟩
  rw [hP.pdiv_eq i]
  rw [← (bnLAPerm N oc h w).symm.sum_comp]
  refine Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [pdiv_eq_fderiv_coord (hl θ), pdiv_eq_fderiv_coord (hF θ)]
  simp only [Equiv.apply_symm_apply]

/-- **BatchNorm γ node = `∇_γ G`**, at the `reassocB` index the render's node reads. -/
theorem bnGamma_hasGradAt {N oc h w : Nat} (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (bnBatchLA N oc h w ε γ β v) cot) :
    HasGradAt (fun θ => G (bnBatchLA N oc h w ε θ β v)) γ
      (den (SHlo.bnGammaGradB vN epsStr ε (reassocB N oc h w v)
        (.operand cotN (reassocB N oc h w cot)))) := by
  have hP := bnLA_param_hasGradAt
    (fun θ => bnPerChannelFlat oc (N * (h * w)) ε θ β (bnchwFwd N oc h w (reassocB N oc h w v)))
    (bnPerChannelFlat_gamma_differentiable _ _ _ _ _) hG
  refine ⟨hP.differentiableAt, fun c => ?_⟩
  refine (hP.pdiv_eq c).trans ?_
  rw [bnGammaGradB_den vN epsStr cotN ε γ β]

/-- **BatchNorm β node = `∇_β G`**, at the `reassocB` index. -/
theorem bnBeta_hasGradAt {N oc h w : Nat} (cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (bnBatchLA N oc h w ε γ β v) cot) :
    HasGradAt (fun θ => G (bnBatchLA N oc h w ε γ θ v)) β
      (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
        (.operand cotN (reassocB N oc h w cot)))) := by
  have hP := bnLA_param_hasGradAt
    (fun θ => bnPerChannelFlat oc (N * (h * w)) ε γ θ (bnchwFwd N oc h w (reassocB N oc h w v)))
    (bnPerChannelFlat_beta_differentiable _ _ _ _ _) hG
  refine ⟨hP.differentiableAt, fun c => ?_⟩
  refine (hP.pdiv_eq c).trans ?_
  rw [bnBetaGradB_den cotN ε γ β (bnchwFwd N oc h w (reassocB N oc h w v))]

-- ════════════════════════════════════════════════════════════════
-- § A conv or depthwise bias, emitted as a BatchNorm β node
--   EfficientNet-B0's render computes every conv and depthwise bias gradient with the
--   `bnBetaGradB` op, the channel sum of the op's output cotangent. The bias enters each of these
--   ops as a channel broadcast (`*_bias_split`), so that channel sum is `∂G/∂b`.
-- ════════════════════════════════════════════════════════════════

/-- The emitted β node's cell `(o, (n, s))` is the network cell `(n, (o, s))`. -/
private theorem reassocB_bnchwFwd {N oc h w : Nat} (cot : Vec (N * (oc * h * w))) (o : Fin oc)
    (n : Fin N) (hi : Fin h) (wi : Fin w) :
    bnchwFwd N oc h w (reassocB N oc h w cot)
        (finProdFinEquiv (o, finProdFinEquiv (n, finProdFinEquiv (hi, wi))))
      = batchSlice N (oc * h * w) cot n (finProdFinEquiv (finProdFinEquiv (o, hi), wi)) := by
  simp only [bnchwFwd, reassocB, bnchwFwdIdx, batchSlice, Equiv.symm_apply_apply]
  congr 1
  apply Fin.ext
  simp only [Fin.val_cast, finProdFinEquiv_apply_val]
  ring

/-- A channel-broadcast bias's Jacobian is the channel indicator. -/
theorem pdiv_bias_of_split {a oc h w : Nat} (per : Vec oc → Vec a → Vec (oc * h * w))
    (hsplit : ∀ θ y, per θ y = fun k => per 0 y k + broadcastFlat oc h w θ k) (y : Vec a)
    (b : Vec oc) (o : Fin oc) (j : Fin (oc * h * w)) :
    pdiv (fun θ => per θ y) b o j = if o = flatChannel oc h w j then 1 else 0 := by
  rw [show (fun θ => per θ y) = fun θ => fun k => per 0 y k + broadcastFlat oc h w θ k from
      funext fun θ => hsplit θ y,
    pdiv_add (fun _ => per 0 y) (broadcastFlat oc h w) b (differentiableAt_const _)
      ((broadcastFlat_differentiable oc h w) b), pdiv_const, zero_add]
  exact pdiv_reindex (flatChannel oc h w) b o j

/-- **A channel-broadcast bias, read by the β node, = `∇_b G`.** For any per-example op whose
    bias enters as `per 0 y + broadcast b`, the emitted `bnBetaGradB` on the op's output cotangent
    is the loss gradient in the bias. -/
theorem biasBeta_hasGradAt {N a oc h w : Nat} (cotN : String)
    (per : Vec oc → Vec a → Vec (oc * h * w))
    (hsplit : ∀ θ y, per θ y = fun k => per 0 y k + broadcastFlat oc h w θ k)
    (x : Vec (N * a)) (b : Vec oc) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (per b) x) cot) :
    HasGradAt (fun θ => G (batchMap N (per θ) x)) b
      (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
        (.operand cotN (reassocB N oc h w cot)))) := by
  have hfun : ∀ y, (fun θ => per θ y) = fun θ => fun k => per 0 y k + broadcastFlat oc h w θ k :=
    fun y => funext fun θ => hsplit θ y
  have hd : ∀ y, DifferentiableAt ℝ (fun θ => per θ y) b := fun y => by
    rw [hfun y]
    exact (differentiableAt_const (per 0 y)).add ((broadcastFlat_differentiable oc h w) b)
  have hP := hG.param_batchMap per x hd
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o]
  symm
  simp_rw [pdiv_bias_of_split per hsplit]
  -- the β node's denotation is the per-channel β gradient of the transposed cotangent, by `rfl`
  show bnPerChannelGradBeta oc (N * (h * w)) (bnchwFwd N oc h w (reassocB N oc h w cot)) o = _
  unfold bnPerChannelGradBeta
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun n _ => ?_
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  simp only [flatChannel, Equiv.symm_apply_apply, ite_mul, one_mul, zero_mul, reassocB_bnchwFwd]
  rw [Fintype.sum_eq_single o fun c hc => by simp [Ne.symm hc]]
  simp

/-- A conv's bias is a channel broadcast. -/
theorem flatConv_bias_split {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (θ : Vec oc)
    (y : Vec (ic * h * w)) :
    (flatConv W θ y : Vec (oc * h * w)) = fun k => flatConv W 0 y k + broadcastFlat oc h w θ k := by
  funext k
  simp only [flatConv, Tensor3.flatten, conv2d, broadcastFlat, flatChannel, Pi.zero_apply, zero_add]
  ring

/-- A depthwise conv's bias is a channel broadcast. -/
theorem depthwiseFlat_bias_split {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (θ : Vec c)
    (y : Vec (c * h * w)) :
    (depthwiseFlat W θ y : Vec (c * h * w))
      = fun k => depthwiseFlat W 0 y k + broadcastFlat c h w θ k := by
  funext k
  simp only [depthwiseFlat, Tensor3.flatten, depthwiseConv2d, broadcastFlat, flatChannel,
    Pi.zero_apply, zero_add]
  ring

private theorem flatChannel_decimateIdx (c h w : Nat) (k : Fin (c * h * w)) :
    flatChannel c (2 * h) (2 * w) (decimateIdx c h w k) = flatChannel c h w k := by
  simp [flatChannel, decimateIdx]

private theorem flatChannel_decimateOddIdx (c h w : Nat) (k : Fin (c * h * w)) :
    flatChannel c (2 * h) (2 * w) (decimateOddIdx c h w k) = flatChannel c h w k := by
  simp [flatChannel, decimateOddIdx]

/-- A symmetric strided depthwise conv's bias is a channel broadcast: decimation keeps channels. -/
theorem depthwiseStride2Flat_bias_split {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (θ : Vec c) (y : Vec (c * (2 * h) * (2 * w))) :
    (depthwiseStride2Flat W θ y : Vec (c * h * w))
      = fun k => depthwiseStride2Flat W 0 y k + broadcastFlat c h w θ k := by
  funext k
  simp only [depthwiseStride2Flat, Function.comp_apply, decimateFlat]
  rw [depthwiseFlat_bias_split W θ y]
  simp only [broadcastFlat, flatChannel_decimateIdx]

/-- An XLA-`SAME` strided conv's bias is a channel broadcast. -/
theorem flatConvStride2Xla_bias_split {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (θ : Vec oc) (y : Vec (ic * (2 * h) * (2 * w))) :
    (flatConvStride2Xla W θ y : Vec (oc * h * w))
      = fun k => flatConvStride2Xla W 0 y k + broadcastFlat oc h w θ k := by
  funext k
  simp only [flatConvStride2Xla, Function.comp_apply, decimateOddFlat]
  rw [flatConv_bias_split W θ y]
  simp only [broadcastFlat, flatChannel_decimateOddIdx]

/-- The stride-4 patchify conv's bias is a channel broadcast: both decimations keep channels. -/
theorem flatConvStride4_bias_split {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (θ : Vec oc) (y : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) :
    (flatConvStride4 W θ y : Vec (oc * h * w))
      = fun k => flatConvStride4 W 0 y k + broadcastFlat oc h w θ k := by
  funext k
  simp only [flatConvStride4, Function.comp_apply, decimateFlat, decimateOddFlat]
  rw [flatConv_bias_split W θ y]
  simp only [broadcastFlat, flatChannel_decimateOddIdx, flatChannel_decimateIdx]

/-- **The stride-4 stem's bias Jacobian is a stride-1 conv's, at any input.** Both are the channel
    indicator (`pdiv_bias_of_split`), so a node the render emits as a stride-1 bias reduce at the
    output resolution is the patchify conv's bias gradient at the real image `x`. -/
theorem pdiv_flatConvStride4_bias_eq_conv2d {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) (y : Tensor3 ic h w) (b : Vec oc) (o : Fin oc)
    (j : Fin (oc * h * w)) :
    pdiv (fun b' : Vec oc => (flatConvStride4 W b' x : Vec (oc * h * w))) b o j
      = pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' y)) b o j := by
  have h1 := pdiv_bias_of_split (fun θ v => flatConv (h := h) (w := w) W θ v)
    (flatConv_bias_split W) (Tensor3.flatten y) b o j
  simp only [flatConv, Tensor3.unflatten_flatten] at h1
  rw [h1]
  exact pdiv_bias_of_split (fun θ v => flatConvStride4 (h := h) (w := w) W θ v)
    (flatConvStride4_bias_split W) x b o j

end Proofs.GradNodeB
