import LeanMlir.Proofs.Foundation.ParamGrad
import LeanMlir.Proofs.Foundation.GradNodesB
import LeanMlir.Proofs.Foundation.SmoothedBatchLoss

/-! # ParamGradNodes — each batched parameter gradient node is a loss derivative

`GradNodesB` states each emitted parameter gradient node as its layer's parameter Jacobian
contracted with an arbitrary output cotangent. Here the cotangent is the gradient of a scalar `G`
at the layer's output (`HasGradAt`), and the node becomes `∂G/∂θ` with the layer's parameter
varied: one lemma per node kind, shared by every net.

The BatchNorm γ/β nodes are stated in the transposed `[C, N·H·W]` layout at the `reassocB`
index; their lemmas re-sum the Jacobian over that permutation (`bnLA_perm`).
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

-- ════════════════════════════════════════════════════════════════
-- § Convolutions and dense layers
-- ════════════════════════════════════════════════════════════════

/-- **Conv weight node = `∂G/∂W`.** -/
theorem convW_eq_pdiv {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConv W b) x) cot)
    (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
      = pdiv (fun θ => G (batchMap N (flatConv (Kernel4.unflatten θ) b) x))
          (Kernel4.flatten W) idx 0 := by
  rw [convWGradB_den]
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => flatConv (Kernel4.unflatten θ) b y) (Kernel4.flatten W)) x) cot := by
    simpa only [Kernel4.unflatten_flatten] using hG
  rw [hG'.pdiv_param_batchMap (fun θ y => flatConv (Kernel4.unflatten θ) b y) x
    (fun y => (conv2d_weight_differentiable b (Tensor3.unflatten y)) _) idx]
  rfl

/-- **Conv bias node = `∂G/∂b`.** -/
theorem convB_eq_pdiv {N ic oc h w kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (N * (ic * h * w))) (b : Vec oc) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConv W b) x) cot)
    (o : Fin oc) :
    den (SHlo.convBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = pdiv (fun θ => G (batchMap N (flatConv W θ) x)) b o 0 := by
  rw [convBGradB_den, hG.pdiv_param_batchMap (fun θ y => flatConv W θ y) x
    (fun y => (conv2d_bias_differentiable W (Tensor3.unflatten y)) _) o]
  rfl

private theorem flatConvStride2_weight_differentiable {ic oc h w kH kW : Nat} (b : Vec oc)
    (y : Vec (ic * (2 * h) * (2 * w))) :
    Differentiable ℝ (fun θ : Vec (oc * ic * kH * kW) =>
      (flatConvStride2 (Kernel4.unflatten θ) b y : Vec (oc * h * w))) := by
  unfold flatConvStride2 decimateFlat
  exact (reindexCLM _).differentiable.comp
    (conv2d_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten y))

private theorem flatConvStride2_bias_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (y : Vec (ic * (2 * h) * (2 * w))) :
    Differentiable ℝ (fun θ : Vec oc => (flatConvStride2 W θ y : Vec (oc * h * w))) := by
  unfold flatConvStride2 decimateFlat
  exact (reindexCLM _).differentiable.comp
    (conv2d_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten y))

/-- **Stride-2 (symmetric) conv weight node = `∂G/∂W`.** -/
theorem convStridedW_eq_pdiv {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    {G : Vec (N * (oc * h * w)) → Vec 1} {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (batchMap N (flatConvStride2 W b) x) cot) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
      = pdiv (fun θ => G (batchMap N (flatConvStride2 (Kernel4.unflatten θ) b) x))
          (Kernel4.flatten W) idx 0 := by
  rw [convStridedWGradB_den]
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => (flatConvStride2 (Kernel4.unflatten θ) b y : Vec (oc * h * w)))
        (Kernel4.flatten W)) x) cot := by
    simpa only [Kernel4.unflatten_flatten] using hG
  rw [hG'.pdiv_param_batchMap (fun θ y => (flatConvStride2 (Kernel4.unflatten θ) b y : Vec (oc * h * w))) x
    (fun y => (flatConvStride2_weight_differentiable b y) _) idx]

/-- **Stride-2 (symmetric) conv bias node = `∂G/∂b`.** -/
theorem convStridedB_eq_pdiv {N ic oc h w kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (batchMap N (flatConvStride2 W b) x) cot)
    (o : Fin oc) :
    den (SHlo.convStridedBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = pdiv (fun θ => G (batchMap N (flatConvStride2 W θ) x)) b o 0 := by
  rw [convStridedBGradB_den, hG.pdiv_param_batchMap
    (fun θ y => (flatConvStride2 W θ y : Vec (oc * h * w))) x
    (fun y => (flatConvStride2_bias_differentiable W y) _) o]

/-- **Dense weight node = `∂G/∂W`.** -/
theorem denseW_eq_pdiv {N a c : Nat} (xN cotN : String) (x : Vec (N * a)) (W : Mat a c)
    (b : Vec c) {G : Vec (N * c) → Vec 1} {cot : Vec (N * c)}
    (hG : HasGradAt G (batchMap N (dense W b) x) cot) (i : Fin a) (j : Fin c) :
    den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) (finProdFinEquiv (i, j))
      = pdiv (fun θ => G (batchMap N (dense (Mat.unflatten θ) b) x)) (Mat.flatten W)
          (finProdFinEquiv (i, j)) 0 := by
  rw [denseWGradB_den xN cotN x W b cot i j]
  have hG' : HasGradAt G
      (batchMap N ((fun θ y => dense (Mat.unflatten θ) b y) (Mat.flatten W)) x) cot := by
    simpa only [Mat.unflatten_flatten] using hG
  rw [hG'.pdiv_param_batchMap (fun θ y => dense (Mat.unflatten θ) b y) x
    (fun y => (denseWeightMap_differentiable b y) _) (finProdFinEquiv (i, j))]

private theorem dense_bias_differentiable {a c : Nat} (W : Mat a c) (x : Vec a) :
    Differentiable ℝ (fun b' : Vec c => dense W b' x) := by
  unfold dense; fun_prop

/-- **Dense bias node = `∂G/∂b`.** The node's statement carries one activation `x₀` for every
    example; the bias Jacobian is the identity whatever the activation, so any `x₀` serves. -/
theorem denseB_eq_pdiv {N a c : Nat} (cotN : String) (W : Mat a c) (x₀ : Vec a)
    (x : Vec (N * a)) (b : Vec c) {G : Vec (N * c) → Vec 1} {cot : Vec (N * c)}
    (hG : HasGradAt G (batchMap N (dense W b) x) cot) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
      = pdiv (fun θ => G (batchMap N (dense W θ) x)) b j 0 := by
  rw [denseBGradB_den cotN W x₀ b cot j, hG.pdiv_param_batchMap (fun θ y => dense W θ y) x
    (fun y => (dense_bias_differentiable W y) _) j]
  simp only [pdiv_dense_b]

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm γ / β, through the `[N,C,H,W] ↔ [C, N·H·W]` permutation
-- ════════════════════════════════════════════════════════════════

/-- The permutation `bnBatchLA` reads its per-channel core through: network index `J` ↦ the
    `[C, N·H·W]` cell `bnchwBackIdx (J at the mul_assoc cast)`. -/
noncomputable def bnLAPerm (N oc h w : Nat) : Fin (N * (oc * h * w)) ≃ Fin (oc * (N * (h * w))) where
  toFun J := bnchwBackIdx N oc h w (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) J)
  invFun j := Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm (bnchwFwdIdx N oc h w j)
  left_inv J := by simp [bnchwFwdIdx_bnchwBackIdx]
  right_inv j := by simp [bnchwBackIdx_bnchwFwdIdx]

/-- `bnBatchLA` at a network index IS the per-channel core at the permuted cell. -/
theorem bnBatchLA_apply_perm (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) (v : Vec (N * (oc * h * w)))
    (J : Fin (N * (oc * h * w))) :
    bnBatchLA N oc h w ε γ β v J
      = bnPerChannelFlat oc (N * (h * w)) ε γ β (bnchwFwd N oc h w (reassocB N oc h w v))
          (bnLAPerm N oc h w J) := rfl

private theorem bnPerChannelFlat_gamma_differentiable (oc m : Nat) (ε : ℝ) (β : Vec oc)
    (v : Vec (oc * m)) : Differentiable ℝ (fun γ' : Vec oc => bnPerChannelFlat oc m ε γ' β v) := by
  unfold bnPerChannelFlat bnPerChannelMat Mat.flatten bnForward
  fun_prop

private theorem bnPerChannelFlat_beta_differentiable (oc m : Nat) (ε : ℝ) (γ : Vec oc)
    (v : Vec (oc * m)) : Differentiable ℝ (fun β' : Vec oc => bnPerChannelFlat oc m ε γ β' v) := by
  unfold bnPerChannelFlat bnPerChannelMat Mat.flatten bnForward
  fun_prop

/-- A parameter entering `bnBatchLA` through its per-channel core: the loss derivative re-sums
    over the permutation into the core's layout. -/
private theorem bnLA_param_pdiv {P N oc h w : Nat} (F : Vec P → Vec (oc * (N * (h * w))))
    (hF : Differentiable ℝ F) {G : Vec (N * (oc * h * w)) → Vec 1} {θ : Vec P}
    {cot : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (fun J => F θ (bnLAPerm N oc h w J)) cot) (i : Fin P) :
    pdiv (fun θ' => G (fun J => F θ' (bnLAPerm N oc h w J))) θ i 0
      = ∑ j, pdiv F θ i j * bnchwFwd N oc h w (reassocB N oc h w cot) j := by
  have hl : Differentiable ℝ (fun θ' => fun J => F θ' (bnLAPerm N oc h w J)) :=
    (reindexCLM (bnLAPerm N oc h w)).differentiable.comp hF
  rw [hG.pdiv_param (layer := fun θ' => fun J => F θ' (bnLAPerm N oc h w J)) (hl θ) i]
  rw [← (bnLAPerm N oc h w).symm.sum_comp]
  refine Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [pdiv_eq_fderiv_coord (hl θ), pdiv_eq_fderiv_coord (hF θ)]
  simp only [Equiv.apply_symm_apply]

/-- **BatchNorm γ node = `∂G/∂γ`**, at the `reassocB` index the render's node reads. -/
theorem bnGamma_eq_pdiv {N oc h w : Nat} (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (bnBatchLA N oc h w ε γ β v) cot)
    (c : Fin oc) :
    den (SHlo.bnGammaGradB vN epsStr ε (reassocB N oc h w v)
        (.operand cotN (reassocB N oc h w cot))) c
      = pdiv (fun θ => G (bnBatchLA N oc h w ε θ β v)) γ c 0 := by
  rw [bnGammaGradB_den vN epsStr cotN ε γ β]
  exact (bnLA_param_pdiv
    (fun θ => bnPerChannelFlat oc (N * (h * w)) ε θ β (bnchwFwd N oc h w (reassocB N oc h w v)))
    (bnPerChannelFlat_gamma_differentiable _ _ _ _ _) hG c).symm

/-- **BatchNorm β node = `∂G/∂β`**, at the `reassocB` index. -/
theorem bnBeta_eq_pdiv {N oc h w : Nat} (cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {cot : Vec (N * (oc * h * w))} (hG : HasGradAt G (bnBatchLA N oc h w ε γ β v) cot)
    (c : Fin oc) :
    den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
        (.operand cotN (reassocB N oc h w cot))) c
      = pdiv (fun θ => G (bnBatchLA N oc h w ε γ θ v)) β c 0 := by
  rw [bnBetaGradB_den cotN ε γ β (bnchwFwd N oc h w (reassocB N oc h w v))]
  exact (bnLA_param_pdiv
    (fun θ => bnPerChannelFlat oc (N * (h * w)) ε γ θ (bnchwFwd N oc h w (reassocB N oc h w v)))
    (bnPerChannelFlat_beta_differentiable _ _ _ _ _) hG c).symm

end Proofs.GradNodeB
