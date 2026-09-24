import LeanMlir.Proofs.Nets.ViT.ViTBackChains
import LeanMlir.Proofs.Nets.ViT.ViTBackB0

/-! # §B: the ViT MHSA backward chain IS the certified VJP (the sdpa adjoint)

The substantive vit-specific §B leaf: the hand-composed multi-head self-attention backward `mhsaBackFlat`
(`ViTBackChains.lean`) IS the certified MHSA input-gradient VJP `mhsa_has_vjp_mat` (`Attention.lean`),
flattened — the attention analogue of the depthwise/conv adjoint gates.

Unlike the CNN `convFlatBack` (a free reversed-kernel conv that needed a gate), the ViT sdpa cores are
ALREADY certified-`sdpa_back`-grounded by construction (`coreQFlat = flatten ∘ mhsaSdpaBackQ ∘ unflatten`,
`mhsaSdpaBackQ = sdpa_back_Q` per `headSliceMat` head). What this file closes is the **assembly reconciliation**:
`mhsaBackFlat` is a flat per-head fan-in with SEPARATE `dense Wᵀq/Wᵀk/Wᵀv` projection-backwards
(`perRowFlat`), while the certified `mhsa_has_vjp_mat.backward` is a Mat-space VJP over the qkv-MERGED
projection. ViTBackB0's `mhsa_backward_collapseMH` already collapses the certified Mat backward to the clean
per-head merged sum `mhsaBackCollapsedMH = ∑ₕ (Σⱼ Wq c (h,j)·dQ + Σⱼ Wk·dK + Σⱼ Wv·dV)`; this file shows
`mhsaBackFlat` (Q/K/V pinned to the actual projections `dense W· bq (X·)`) equals that, coordinatewise:
`dense Wᵀ 0 = Mat.mulVec W`, the `Σ k` over `h·dh` reindexes to `Σₕ Σⱼ`, and the chain's separate projBack
sums regroup into the certified `∑ₕ(Q+K+V)` by `Finset.sum_add_distrib`. So `mhsaBackFlat` IS the
certified attention gradient — the genuinely-new (sdpa) half of the ViT block §B tie.

The block tie that wraps this in the per-token LN/dense/gelu sublayer reconciliations and the residual
fan-in is `vitBlockBackV_eq_transformerBlockV_vjp` (`ViTVecLNBackCertifiedTie.lean`). 3-axiom-clean.
-/

namespace Proofs

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
    form. This is the per-head slab the cores read (`headSliceMat _ _ _ h (unflatten woflat) = dAttg h`). -/
theorem woback_unflatten (Wo : Mat (h * dh) (h * dh)) (dconcat : Vec (N * (h * dh))) :
    Mat.unflatten (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) (0 : Vec (h * dh))) dconcat)
      = (fun (i : Fin N) (c : Fin (h * dh)) => Mat.mulVec Wo (Mat.unflatten dconcat i) c) := by
  funext i c
  show perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) 0) dconcat (finProdFinEquiv (i, c)) = _
  rw [perRowFlat_apply]
  simp only [Equiv.symm_apply_apply, Proofs.dense, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl (fun k _ => mul_comm _ _)

/-- **THE ViT MHSA BACKWARD §B TIE.** The MHSA backward chain `mhsaBackFlat`, with its saved Q/K/V
    projections pinned to the actual `dense W· b· (X·)` projections at the saved block input `X`, IS the
    certified MHSA input-gradient VJP `(mhsa_has_vjp_mat …).backward X`, flattened. So the attention
    backward the ViT chain is spelled in IS the certified attention gradient, not a look-alike.
    Closes under `[propext, Classical.choice, Quot.sound]`. -/
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
  unfold headSliceMat
  ring

-- ════════════════════════════════════════════════════════════════
-- § The MLP-sublayer reconciliation — the per-token-aware leaves
-- ════════════════════════════════════════════════════════════════

/-- The chain's dense input-VJP `dense (Wᵀ) 0` IS the certified contraction `Mat.mulVec W`
    (the certified `dense_has_vjp.backward`, which ignores its affine activation); `mul_comm` per
    term. The function-level form (no `x` arg) the `simp` matches against. -/
theorem dense_transpose_eq_mulVec {m n : Nat} (W : Mat m n) :
    Proofs.dense (Mat.transpose W) (0 : Vec m) = Mat.mulVec W := by
  funext dy i
  simp only [Proofs.dense, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- The chain's GELU backward `diagBack (act'(s))` IS the certified `gelu_has_vjp.backward`
    at the saved pre-activation `s` (the elementwise derivative scaling — `gelu_has_vjp.backward s
    dy i = dy i · geluScalarDeriv (s i)`, `diagBack` is the same scaling, `mul_comm`). -/
theorem diagBack_eq_gelu_vjp {n : Nat} (s : Vec n) :
    diagBack (fun c => geluScalarDeriv (s c)) = (gelu_has_vjp n).backward s := by
  funext dy i
  simp only [diagBack, gelu_has_vjp, mul_comm]

/-- **The `transformerMlp` backward in explicit per-token form.** The nested `vjpMat_comp`
    (`dense₂ ∘ gelu ∘ dense₁`, per token) reduces to: each token's `dz r` runs `mulVec Wfc2`,
    the GELU backward at the saved pre-activation `dense₁(Y r)`, then `mulVec Wfc1`. Pure
    `rfl` (the per-token VJPs are `rowwise`/`vjpMat_comp` structure projections). -/
theorem transformerMlp_backward_pertoken (N D dff : Nat)
    (Wfc1 : Mat D dff) (bfc1 : Vec dff) (Wfc2 : Mat dff D) (bfc2 : Vec D)
    (Y : Mat N D) (dz : Mat N D) :
    (transformerMlp_has_vjp_mat N D dff Wfc1 bfc1 Wfc2 bfc2).backward Y dz
      = fun r => Mat.mulVec Wfc1
          ((gelu_has_vjp dff).backward (Proofs.dense Wfc1 bfc1 (Y r)) (Mat.mulVec Wfc2 (dz r))) := by
  rfl

/-- **L2 — the `transformerMlp` backward, flattened, IS `perRowFlatPR` of the flat chain.**
    The certified per-token MLP-body backward (`mulVec Wfc1 ∘ gelu-back ∘ mulVec Wfc2`) equals the
    chain's `dense Wᵀ₁ 0 ∘ diagBack(act'(dense₁ Y)) ∘ dense Wᵀ₂ 0`, row by row. -/
theorem transformerMlp_back_flat_eq_perRowFlatPR (N D dff : Nat)
    (Wfc1 : Mat D dff) (bfc1 : Vec dff) (Wfc2 : Mat dff D) (bfc2 : Vec D)
    (Y : Mat N D) (v : Vec (N * D)) :
    Mat.flatten ((transformerMlp_has_vjp_mat N D dff Wfc1 bfc1 Wfc2 bfc2).backward Y (Mat.unflatten v))
      = perRowFlatPR N D
          (fun r => Proofs.dense (Mat.transpose Wfc1) (0 : Vec D)
            ∘ diagBack (fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1 (Y r) c))
            ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec dff)) v := by
  rw [transformerMlp_backward_pertoken]
  funext idx
  rw [perRowFlatPR_apply]
  simp only [Function.comp_apply, dense_transpose_eq_mulVec, diagBack_eq_gelu_vjp, Mat.flatten]

end Proofs
