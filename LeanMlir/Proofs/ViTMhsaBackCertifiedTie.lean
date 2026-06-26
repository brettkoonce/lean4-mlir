import LeanMlir.Proofs.MhsaBackFloatBridge
import LeanMlir.Proofs.ViTBackB0

/-! # §B: the ViT MHSA backward float bridge targets the CERTIFIED VJP (the sdpa adjoint)

The substantive vit-specific §B leaf: the float-bridge multi-head self-attention backward `mhsaBackFlat`
(`MhsaBackFloatBridge.lean`) IS the certified MHSA input-gradient VJP `mhsa_has_vjp_mat` (`Attention.lean`),
flattened — the attention analogue of the depthwise/conv adjoint gates.

Unlike the CNN `convFlatBack` (a free reversed-kernel conv that needed a gate), the ViT sdpa cores are
ALREADY certified-`sdpa_back`-grounded by construction (`coreQFlat = flatten ∘ mhsaSdpaBackQ ∘ unflatten`,
`mhsaSdpaBackQ = sdpa_back_Q` per `mhSlab` head). What this file closes is the **assembly reconciliation**:
the float `mhsaBackFlat` is a flat per-head fan-in with SEPARATE `dense Wᵀq/Wᵀk/Wᵀv` projection-backwards
(`perRowFlat`), while the certified `mhsa_has_vjp_mat.backward` is a Mat-space VJP over the qkv-MERGED
projection. ViTBackB0's `mhsa_backward_collapseMH` already collapses the certified Mat backward to the clean
per-head merged sum `mhsaBackCollapsedMH = ∑ₕ (Σⱼ Wq c (h,j)·dQ + Σⱼ Wk·dK + Σⱼ Wv·dV)`; this file shows
`mhsaBackFlat` (Q/K/V pinned to the actual projections `dense W· bq (X·)`) equals that, coordinatewise:
`dense Wᵀ 0 = Mat.mulVec W`, the `Σ k` over `h·dh` reindexes to `Σₕ Σⱼ`, and the float's separate projBack
sums regroup into the certified `∑ₕ(Q+K+V)` by `Finset.sum_add_distrib`. So `mhsaBackFlat` provably bounds the
certified attention gradient — the genuinely-new (sdpa) half of the ViT block §B tie.

The remaining ViT block tie = wrapping this in the per-token LN/dense/gelu sublayer reconciliations + the
residual fan-in (the standard `perRowFlat`/`dense_transpose`/`diagBack` flat↔Mat reindex, no new analysis).
3-axiom-clean.
-/

namespace Proofs

open scoped Real

variable {h N dh : Nat}

/-- **The projection-back leaf coordinate.** A per-token `dense (Wᵀ) 0` projection-backward
    (`perRowFlat`) applied to a flattened saved cotangent `Mat.flatten S` reads, at output coordinate
    `(r, c)`, as `Σ k, W c k · S r k` — i.e. `Mat.mulVec W` of row `r`. The `dense_transpose = mulVec`
    fact lifted through the `perRowFlat` / `flatten`-`unflatten` reindex. The shared leaf for all four
    projections (Wq/Wk/Wv contracted against the cores, Wo against the block cotangent). -/
theorem projBack_core_coord (W : Mat (h * dh) (h * dh)) (S : Mat N (h * dh))
    (idx : Fin (N * (h * dh))) :
    perRowFlat N (h * dh) (Proofs.dense (Mat.transpose W) (0 : Vec (h * dh))) (Mat.flatten S) idx
      = ∑ k : Fin (h * dh), W (finProdFinEquiv.symm idx).2 k * S (finProdFinEquiv.symm idx).1 k := by
  rw [perRowFlat_apply, Mat.unflatten_flatten]
  simp only [Proofs.dense, Mat.transpose, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl (fun k _ => mul_comm _ _)

/-- **The Wo-back, unflattened.** `unflatten (perRowFlat (dense Wᵀo 0) dconcat) = fun i c => mulVec Wo
    (unflatten dconcat i) c` — the block cotangent run through the output-projection backward, in Mat
    form. This is the per-head slab the cores read (`mhSlab h (unflatten woflat) = dAttg h`). -/
theorem woback_unflatten (Wo : Mat (h * dh) (h * dh)) (dconcat : Vec (N * (h * dh))) :
    Mat.unflatten (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) (0 : Vec (h * dh))) dconcat)
      = (fun (i : Fin N) (c : Fin (h * dh)) => Mat.mulVec Wo (Mat.unflatten dconcat i) c) := by
  funext i c
  show perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat (finProdFinEquiv (i, c)) = _
  rw [perRowFlat_apply]
  simp only [Equiv.symm_apply_apply, Proofs.dense, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl (fun k _ => mul_comm _ _)

/-- **THE ViT MHSA BACKWARD §B TIE.** The float-bridge MHSA backward `mhsaBackFlat`, with its saved Q/K/V
    projections pinned to the actual `dense W· b· (X·)` projections at the saved block input `X`, IS the
    certified MHSA input-gradient VJP `(mhsa_has_vjp_mat …).backward X`, flattened. So the deployed-float
    attention backward is within an explicit budget (via `floatBridges_mhsaBack`) of THE certified
    attention gradient, not a look-alike. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem mhsaBackFlat_eq_mhsa_vjp
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh)) (X : Mat N (h * dh)) :
    mhsaBackFlat Wq Wk Wv Wo
        (fun r => Proofs.dense Wq bq (X r)) (fun r => Proofs.dense Wk bk (X r))
        (fun r => Proofs.dense Wv bv (X r))
      = (fun dconcat => Mat.flatten
          ((mhsa_has_vjp_mat N h dh Wq Wk Wv Wo bq bk bv bo).backward X (Mat.unflatten dconcat))) := by
  funext dconcat
  rw [StableHLO.mhsa_backward_collapseMH N h dh Wq Wk Wv Wo bq bk bv bo X (Mat.unflatten dconcat)]
  funext idx
  -- abbreviations matching `mhsaBackCollapsedMH`
  set r := (finProdFinEquiv.symm idx).1 with hr
  set c := (finProdFinEquiv.symm idx).2 with hc
  -- the Wo-back applied to the block cotangent, in Mat form
  have hwo := woback_unflatten Wo dconcat
  -- LHS: expand `mhsaBackFlat` and the three projection-back cores via `projBack_core_coord`
  show (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wq) 0)
          (coreQFlat (fun r => Proofs.dense Wq bq (X r)) (fun r => Proofs.dense Wk bk (X r))
            (fun r => Proofs.dense Wv bv (X r))
            (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat)) idx)
        + ((perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wk) 0)
              (coreKFlat (fun r => Proofs.dense Wq bq (X r)) (fun r => Proofs.dense Wk bk (X r))
                (fun r => Proofs.dense Wv bv (X r))
                (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat)) idx)
          + (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wv) 0)
              (coreVFlat (fun r => Proofs.dense Wq bq (X r)) (fun r => Proofs.dense Wk bk (X r))
                (fun r => Proofs.dense Wv bv (X r))
                (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat)) idx))
      = _
  unfold coreQFlat coreKFlat coreVFlat
  rw [projBack_core_coord Wq _ idx, projBack_core_coord Wk _ idx, projBack_core_coord Wv _ idx]
  -- reindex each `Σ k : Fin (h*dh)` to `Σ h' Σ j` and recognize the per-head `sdpa_back_*`
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin h × Fin dh ≃ Fin (h * dh))
        (fun k => Wq c k * mhsaSdpaBackQ (fun r => Proofs.dense Wq bq (X r))
          (fun r => Proofs.dense Wk bk (X r)) (fun r => Proofs.dense Wv bv (X r))
          (Mat.unflatten (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat)) r k),
      ← Equiv.sum_comp (finProdFinEquiv : Fin h × Fin dh ≃ Fin (h * dh))
        (fun k => Wk c k * mhsaSdpaBackK (fun r => Proofs.dense Wq bq (X r))
          (fun r => Proofs.dense Wk bk (X r)) (fun r => Proofs.dense Wv bv (X r))
          (Mat.unflatten (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat)) r k),
      ← Equiv.sum_comp (finProdFinEquiv : Fin h × Fin dh ≃ Fin (h * dh))
        (fun k => Wv c k * mhsaSdpaBackV (fun r => Proofs.dense Wq bq (X r))
          (fun r => Proofs.dense Wk bk (X r)) (fun r => Proofs.dense Wv bv (X r))
          (Mat.unflatten (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat)) r k)]
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type, Fintype.sum_prod_type]
  -- RHS: unfold `mhsaBackCollapsedMH` and combine the three `Σ h'` via `sum_add_distrib`
  simp only [Mat.flatten, StableHLO.mhsaBackCollapsedMH, mhsaSdpaBackQ, mhsaSdpaBackK, mhsaSdpaBackV,
    hwo, Equiv.symm_apply_apply, Finset.sum_add_distrib]
  unfold mhSlab
  ring

-- ════════════════════════════════════════════════════════════════
-- § The attention-sublayer reconciliation — grounding the MHSA leaf in the block
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 10000000 in
/-- **The certified attention-sublayer VJP decomposes** (the `biPathMat` unfold, `rfl`): the residual
    skip passes the cotangent through (`identityMat` backward = `dY`), and the non-trivial arm is the
    chain `LN₁-back ∘ mhsa-back` at the saved LayerNorm output `LN₁ A` (`vjpMat_comp`). The Mat-space
    analogue of `transformerBlock_backward_unfold`. -/
theorem transformerAttnSublayer_backward_decomp (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh)) (A dY : Mat N (h * dh)) :
    (transformerAttnSublayer_has_vjp_mat N h dh ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward A dY
      = (fun i j => dY i j +
          (layerNorm_per_token_has_vjp_mat N (h * dh) ε γ1 β1 hε).backward A
            ((mhsa_has_vjp_mat N h dh Wq Wk Wv Wo bq bk bv bo).backward
              (fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) dY) i j) := rfl

/-- **The attention-sublayer backward, flat, with the MHSA leaf plugged in.** The certified
    attention-sublayer VJP (flattened) IS the residual skip `v` plus the certified **per-token**
    LayerNorm backward of (the unflatten of) `mhsaBackFlat` — the proven sdpa-adjoint leaf
    (`mhsaBackFlat_eq_mhsa_vjp`). So the MHSA half of the ViT encoder block backward is now grounded in
    the certified gradient end to end through the sublayer.

    Note the LayerNorm-back is `layerNorm_per_token_has_vjp_mat.backward A` — `rowwise` of the
    single-token LN VJP, which threads each token's saved input `A r` (its Jacobian differs per token).
    This is precisely why the float bridge's single-`lnB₁` `perRowFlat` lift (one cotangent→grad map for
    every token) is the remaining piece for a full `vitBlockBack` tie: the forward LN rides one pure
    function per token, but the backward needs the per-token saved input. The sdpa half ties; the LN-back
    half wants a per-token-input-aware lift. 3-axiom-clean. -/
theorem transformerAttnSublayerBack_flat_decomp (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (A : Mat N (h * dh)) (v : Vec (N * (h * dh))) :
    Mat.flatten ((transformerAttnSublayer_has_vjp_mat N h dh ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward
        A (Mat.unflatten v))
      = (fun idx => v idx +
          Mat.flatten ((layerNorm_per_token_has_vjp_mat N (h * dh) ε γ1 β1 hε).backward A
            (Mat.unflatten (mhsaBackFlat Wq Wk Wv Wo
              (fun r => Proofs.dense Wq bq ((fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) r))
              (fun r => Proofs.dense Wk bk ((fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) r))
              (fun r => Proofs.dense Wv bv ((fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) r))
              v))) idx) := by
  have hmhsa : (mhsa_has_vjp_mat N h dh Wq Wk Wv Wo bq bk bv bo).backward
        (fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) (Mat.unflatten v)
      = Mat.unflatten (mhsaBackFlat Wq Wk Wv Wo
          (fun r => Proofs.dense Wq bq ((fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) r))
          (fun r => Proofs.dense Wk bk ((fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) r))
          (fun r => Proofs.dense Wv bv ((fun n => layerNormForward (h * dh) ε γ1 β1 (A n)) r))
          v) := by
    rw [congrFun (mhsaBackFlat_eq_mhsa_vjp Wq Wk Wv Wo bq bk bv bo
          (fun n => layerNormForward (h * dh) ε γ1 β1 (A n))) v, Mat.unflatten_flatten]
  rw [transformerAttnSublayer_backward_decomp ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo A (Mat.unflatten v),
      hmhsa]
  funext idx
  simp only [Mat.flatten]
  congr 1
  exact congrFun (Mat.flatten_unflatten v) idx

end Proofs
