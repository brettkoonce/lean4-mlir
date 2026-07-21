import LeanMlir.Proofs.Float.MhsaBackFloatBridge
import LeanMlir.Proofs.Architectures.ViTBackB0

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

-- ════════════════════════════════════════════════════════════════
-- § The MLP-sublayer reconciliation — the per-token-aware leaves
-- ════════════════════════════════════════════════════════════════

/-- The float-bridge dense input-VJP `dense (Wᵀ) 0` IS the certified contraction `Mat.mulVec W`
    (the certified `dense_has_vjp.backward`, which ignores its affine activation); `mul_comm` per
    term. The function-level form (no `x` arg) the `simp` matches against. -/
theorem dense_transpose_eq_mulVec {m n : Nat} (W : Mat m n) :
    Proofs.dense (Mat.transpose W) (0 : Vec m) = Mat.mulVec W := by
  funext dy i
  simp only [Proofs.dense, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- The float-bridge GELU backward `diagBack (act'(s))` IS the certified `gelu_has_vjp.backward`
    at the saved pre-activation `s` (the elementwise derivative scaling — `gelu_has_vjp.backward s
    dy i = dy i · geluScalarDeriv (s i)`, `diagBack` is the same scaling, `mul_comm`). -/
theorem diagBack_eq_gelu_vjp {n : Nat} (s : Vec n) :
    diagBack (fun c => geluScalarDeriv (s c)) = (gelu_has_vjp n).backward s := by
  funext dy i
  simp only [diagBack, gelu_has_vjp, mul_comm]

/-- **The per-token LayerNorm backward IS `perRowFlatPR` of the single-token LN VJP.** The
    certified `layerNorm_per_token_has_vjp_mat.backward A` (`rowwise` of the single-token
    `layerNorm_has_vjp.backward`, threading each token's saved input `A r`), flattened, equals
    the per-token-input-aware lift `perRowFlatPR` of `fun r => layerNorm_has_vjp.backward (A r)`.
    The flat reflection of `rowwise`'s per-row backward — the seam `vitBlockBackPR` needs. -/
theorem perRowFlatPR_LN_back (N D : Nat) (ε γ β : ℝ) (hε : 0 < ε)
    (A : Mat N D) (X : Vec (N * D)) :
    perRowFlatPR N D (fun r => (layerNorm_has_vjp D ε γ β hε).backward (A r)) X
      = Mat.flatten ((layerNorm_per_token_has_vjp_mat N D ε γ β hε).backward A (Mat.unflatten X)) := by
  rfl

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

/-- `perRowFlatPR` of a per-row residual = the per-row body lift plus the cotangent skip
    (`residual g x = g x + x`, lifted row by row to `+ v`). -/
theorem perRowFlatPR_residual {n d : Nat} (g : Fin n → (Vec d → Vec d)) (v : Vec (n * d)) :
    perRowFlatPR n d (fun r => Proofs.residual (g r)) v
      = fun idx => perRowFlatPR n d g v idx + v idx := by
  funext idx
  rw [perRowFlatPR_apply, perRowFlatPR_apply]
  show Proofs.residual (g (finProdFinEquiv.symm idx).1) (Mat.unflatten v (finProdFinEquiv.symm idx).1)
        (finProdFinEquiv.symm idx).2 = _
  simp only [Proofs.residual, biPath]
  congr 1
  show Mat.unflatten v (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = v idx
  show v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2)) = v idx
  rw [Prod.mk.eta, Equiv.apply_symm_apply]

/-- **L2 — the `transformerMlp` backward, flattened, IS `perRowFlatPR` of the float chain.**
    The certified per-token MLP-body backward (`mulVec Wfc1 ∘ gelu-back ∘ mulVec Wfc2`) equals the
    float bridge's `dense Wᵀ₁ 0 ∘ diagBack(act'(dense₁ Y)) ∘ dense Wᵀ₂ 0`, row by row. -/
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

set_option maxHeartbeats 4000000 in
/-- **The MLP-sublayer VJP backward decomposes** (the `biPathMat` unfold, `rfl`): residual skip
    passes the cotangent through, the non-trivial arm is `LN₂-back ∘ transformerMlp-back` at the
    saved LN₂ output. The MLP analogue of `transformerAttnSublayer_backward_decomp`. -/
theorem transformerMlpSublayer_backward_decomp (dff : Nat) (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (hM dz : Mat N (h * dh)) :
    (transformerMlpSublayer_has_vjp_mat N h dh dff ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).backward hM dz
      = fun i j => dz i j +
          (layerNorm_per_token_has_vjp_mat N (h * dh) ε γ2 β2 hε).backward hM
            ((transformerMlp_has_vjp_mat N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2).backward
              (fun n => layerNormForward (h * dh) ε γ2 β2 (hM n)) dz) i j := rfl

set_option maxHeartbeats 4000000 in
/-- **The transformer-block VJP backward unfolds** (general heads): `block.backward A dz =
    attn.backward A (mlp.backward (attn A) dz)`. The general-heads peer of ViTBackB0's heads=1
    `transformerBlock_backward_unfold`; `rfl` (outer `vjpMat_comp`). -/
theorem transformerBlock_backward_unfold_gen (dff : Nat)
    (ε γ1 β1 γ2 β2 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (A dz : Mat N (h * dh)) :
    (transformerBlock_has_vjp_mat N h dh dff ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A dz
      = (transformerAttnSublayer_has_vjp_mat N h dh ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward A
          ((transformerMlpSublayer_has_vjp_mat N h dh dff ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).backward
            (transformerAttnSublayer N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A) dz) := rfl

/-- **The attention-sublayer backward float-half IS the certified attn-sublayer VJP, flat.**
    The float bridge `residual (perRowFlatPR lnB₁ ∘ mhsaBackFlat)` (with `lnB₁ r =` the single-token
    LN₁ backward at `A r` and Q/K/V pinned at `LN₁(A)`) equals `flatten ∘ attnSublayer.backward A ∘
    unflatten`. The standalone packaging of `transformerAttnSublayerBack_flat_decomp`. -/
theorem attnSubFlatTie (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (A : Mat N (h * dh)) (w : Vec (N * (h * dh))) :
    Proofs.residual (perRowFlatPR N (h * dh)
        (fun r => (layerNorm_has_vjp (h * dh) ε γ1 β1 hε).backward (A r))
        ∘ mhsaBackFlat Wq Wk Wv Wo
            (fun r => Proofs.dense Wq bq (layerNormForward (h * dh) ε γ1 β1 (A r)))
            (fun r => Proofs.dense Wk bk (layerNormForward (h * dh) ε γ1 β1 (A r)))
            (fun r => Proofs.dense Wv bv (layerNormForward (h * dh) ε γ1 β1 (A r)))) w
      = Mat.flatten ((transformerAttnSublayer_has_vjp_mat N h dh ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo).backward A (Mat.unflatten w)) := by
  rw [transformerAttnSublayerBack_flat_decomp ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo A w]
  funext idx
  simp only [Proofs.residual, biPath, Function.comp_apply]
  rw [perRowFlatPR_LN_back]
  show perRowFlatPR N (h * dh)
        (fun r => (layerNorm_has_vjp (h * dh) ε γ1 β1 hε).backward (A r))
        (mhsaBackFlat Wq Wk Wv Wo _ _ _ w) idx + w idx
      = w idx + Mat.flatten _ idx
  rw [perRowFlatPR_LN_back]
  exact add_comm _ _

/-- **The MLP-sublayer backward float-half IS the certified MLP-sublayer VJP, flat.** The float
    bridge `perRowFlatPR (fun r => residual (lnB₂ r ∘ dense Wᵀ₁ 0 ∘ diagBack(sgelu r) ∘ dense Wᵀ₂ 0))`
    (LN₂-back at `hM r`, `sgelu r =` the GELU derivative at `dense₁(LN₂ hM r)`) equals `flatten ∘
    mlpSublayer.backward hM ∘ unflatten`. The MLP peer of `attnSubFlatTie`. -/
theorem mlpSubFlatTie (dff : Nat) (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (hM : Mat N (h * dh)) (v : Vec (N * (h * dh))) :
    perRowFlatPR N (h * dh) (fun r => Proofs.residual
        ((layerNorm_has_vjp (h * dh) ε γ2 β2 hε).backward (hM r)
          ∘ Proofs.dense (Mat.transpose Wfc1) (0 : Vec (h * dh))
          ∘ diagBack (fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
              (layerNormForward (h * dh) ε γ2 β2 (hM r)) c))
          ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec dff))) v
      = Mat.flatten ((transformerMlpSublayer_has_vjp_mat N h dh dff ε γ2 β2 hε
          Wfc1 bfc1 Wfc2 bfc2).backward hM (Mat.unflatten v)) := by
  rw [transformerMlpSublayer_backward_decomp dff ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2 hM (Mat.unflatten v)]
  set lnB₂ : Fin N → (Vec (h * dh) → Vec (h * dh)) :=
    fun r => (layerNorm_has_vjp (h * dh) ε γ2 β2 hε).backward (hM r) with hlnB₂
  set chain : Fin N → (Vec (h * dh) → Vec (h * dh)) :=
    fun r => Proofs.dense (Mat.transpose Wfc1) (0 : Vec (h * dh))
      ∘ diagBack (fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
          (layerNormForward (h * dh) ε γ2 β2 (hM r)) c))
      ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec dff) with hchain
  rw [perRowFlatPR_residual (fun r => lnB₂ r ∘ chain r) v]
  have hbody : perRowFlatPR N (h * dh) (fun r => lnB₂ r ∘ chain r) v
      = Mat.flatten ((layerNorm_per_token_has_vjp_mat N (h * dh) ε γ2 β2 hε).backward hM
          ((transformerMlp_has_vjp_mat N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2).backward
            (fun n => layerNormForward (h * dh) ε γ2 β2 (hM n)) (Mat.unflatten v))) := by
    calc perRowFlatPR N (h * dh) (fun r => lnB₂ r ∘ chain r) v
        = perRowFlatPR N (h * dh) lnB₂ (perRowFlatPR N (h * dh) chain v) := by
            rw [← perRowFlatPR_comp lnB₂ chain, Function.comp_apply]
      _ = perRowFlatPR N (h * dh) lnB₂
            (Mat.flatten ((transformerMlp_has_vjp_mat N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2).backward
              (fun n => layerNormForward (h * dh) ε γ2 β2 (hM n)) (Mat.unflatten v))) := by
            rw [hchain,
              ← transformerMlp_back_flat_eq_perRowFlatPR N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2
                (fun n => layerNormForward (h * dh) ε γ2 β2 (hM n)) v]
      _ = Mat.flatten ((layerNorm_per_token_has_vjp_mat N (h * dh) ε γ2 β2 hε).backward hM
            ((transformerMlp_has_vjp_mat N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2).backward
              (fun n => layerNormForward (h * dh) ε γ2 β2 (hM n)) (Mat.unflatten v))) := by
            rw [hlnB₂, perRowFlatPR_LN_back, Mat.unflatten_flatten]
  rw [hbody]
  funext idx
  simp only [Mat.flatten]
  have hv : Mat.unflatten v (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = v idx := by
    show v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2)) = v idx
    rw [Prod.mk.eta, Equiv.apply_symm_apply]
  rw [hv]
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § THE CAPSTONE — the full per-token-aware ViT block backward tie
-- ════════════════════════════════════════════════════════════════

/-- **THE FULL `vitBlockBackPR` §B TIE.** The per-token-input-aware ViT encoder-block backward
    float bridge `vitBlockBackPR`, with every saved activation pinned to the real forward
    (Q/K/V projections at `LN₁ A`; the LN₁/LN₂ backwards at each token's own saved input `A r` /
    `(attn A) r`; the GELU derivative at `dense₁(LN₂(attn A))`), IS the certified transformer-block
    input-gradient VJP `transformerBlock_has_vjp_mat`, flattened. So the deployed float ViT-block
    backward is within an explicit budget (via `floatBridges_vitBlockBackPR`) of THE certified block
    gradient — the per-token-LN enrichment closes the structural gap the attn-sublayer tie left open.
    Assembled from the block unfold (general heads) + the attn/MLP sublayer flat ties.
    Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem vitBlockBackPR_eq_transformerBlock_vjp (dff : Nat)
    (ε γ1 β1 γ2 β2 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (A : Mat N (h * dh)) :
    vitBlockBackPR Wq Wk Wv Wo
        (fun r => Proofs.dense Wq bq (layerNormForward (h * dh) ε γ1 β1 (A r)))
        (fun r => Proofs.dense Wk bk (layerNormForward (h * dh) ε γ1 β1 (A r)))
        (fun r => Proofs.dense Wv bv (layerNormForward (h * dh) ε γ1 β1 (A r)))
        (fun r => (layerNorm_has_vjp (h * dh) ε γ1 β1 hε).backward (A r))
        Wfc1 Wfc2
        (fun r => fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
          (layerNormForward (h * dh) ε γ2 β2
            (transformerAttnSublayer N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A r)) c))
        (fun r => (layerNorm_has_vjp (h * dh) ε γ2 β2 hε).backward
          (transformerAttnSublayer N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A r))
      = fun dY => Mat.flatten ((transformerBlock_has_vjp_mat N h dh dff ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A (Mat.unflatten dY)) := by
  funext dY
  set hM : Mat N (h * dh) := transformerAttnSublayer N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A with hhM
  show Proofs.residual (perRowFlatPR N (h * dh)
      (fun r => (layerNorm_has_vjp (h * dh) ε γ1 β1 hε).backward (A r))
      ∘ mhsaBackFlat Wq Wk Wv Wo
          (fun r => Proofs.dense Wq bq (layerNormForward (h * dh) ε γ1 β1 (A r)))
          (fun r => Proofs.dense Wk bk (layerNormForward (h * dh) ε γ1 β1 (A r)))
          (fun r => Proofs.dense Wv bv (layerNormForward (h * dh) ε γ1 β1 (A r))))
      (perRowFlatPR N (h * dh) (fun r => Proofs.residual
        ((layerNorm_has_vjp (h * dh) ε γ2 β2 hε).backward (hM r)
          ∘ Proofs.dense (Mat.transpose Wfc1) (0 : Vec (h * dh))
          ∘ diagBack (fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
              (layerNormForward (h * dh) ε γ2 β2 (hM r)) c))
          ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec dff))) dY) = _
  rw [mlpSubFlatTie dff ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2 hM dY,
      attnSubFlatTie ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo A _,
      Mat.unflatten_flatten,
      ← transformerBlock_backward_unfold_gen dff ε γ1 β1 γ2 β2 hε Wq Wk Wv Wo bq bk bv bo
        Wfc1 bfc1 Wfc2 bfc2 A (Mat.unflatten dY)]

end Proofs
