import LeanMlir.Proofs.Nets.ViT.ViTStepTie
import LeanMlir.Proofs.Foundation.Batched.Indexed

/-! # One ViT block with its two drop sites, per example — forward, VJP, input cotangent

The `*drop*` ViT renders scale the attention branch (after the out-dense, before the first skip
add) and the MLP branch (after fc2, before the second) by the example's own mask entry
(`ViTRenderB.vBlockFwdB`), and the backward puts the same op on each branch's cotangent while the
skip fan-ins read the raw one (`ViTRenderB`'s block backward; `dropPath_vjp_is_self`). Per
example, a site is a scalar or absent (`dropScalarOpt`, `Foundation.Batched.Indexed`), so this
file states the block at two `Option ℝ` sites `sA sM`; the batched ties lift it with
`batchMapIdx`, example `n` at `exampleSite` of the masks.

* `BlockParamsV.fwdOD` — the block forward spelled as `vitBlockSpelledMHV` with the two sites;
  at `none none` it IS `fwdO` (`fwdOD_none`, `rfl`), at `some a, some m` it is
  `vitBlockSpelledMHVDrop` (`ViTFwdDrop`).
* `BlockParamsV.cotInD` — the input cotangent the render's chain threads: `vitBlockCotInAtMHV`
  with the MLP branch fed `sM ⊙ dy` (`vitCotHVD`) and the attention branch `sA ⊙ cotH`; `cotIn` at
  `none none` (`cotInD_none`, `rfl`).
* `BlockParamsV.fwdODHasVJP` / `cotInD_eq_vjp` — the forward is two site residuals
  (`fwdOD_eq_sites`), each `siteResHasVJP` over its branch's certified VJP, and the chain IS that
  witness's backward. The branch backwards are read off the sublayer ties
  (`attnSubFlat_tie_v`, `mlpSubFlat_tie_v`) minus their skip.
-/

open Proofs Proofs.StableHLO

namespace Proofs.ViTTieGB

open scoped BigOperators
open Proofs.ViTTie (vitBlockFwdOMHV vitBlockCotInAtMHV)

/-- Cot at the attention-sublayer output `h` with the MLP branch's site: `dyOut` raw on the skip,
    `sM ⊙ dyOut` into fc2's backward (`vitCotHV` at `none`). -/
noncomputable def vitCotHVD (gf : GeluForm) {Np1 D mlpDim : Nat} (ε : ℝ) (γ2 : Vec D)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (sM : Option ℝ) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) : Vec (Np1 * D) :=
  fun i => dyOut i + StableHLO.rowLNBackFlat Np1 D ε 1 h
    (StableHLO.rowScaleFlat Np1 D γ2 (vitCotLn2 gf Wfc1 Wfc2 m1 (dropScalarOpt sM dyOut))) i

/-- Cot at the SDPA output with both sites: `Woᵀ` of `sA ⊙ cotH` (`vitCotAttV` at `none none`). -/
noncomputable def vitCotAttVD (gf : GeluForm) {Np1 D mlpDim : Nat} (ε : ℝ) (γ2 : Vec D)
    (Wo : Mat D D) (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (sA sM : Option ℝ)
    (h : Vec (Np1 * D)) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) : Vec (Np1 * D) :=
  StableHLO.rowDenseBackFlat Np1 D D Wo (dropScalarOpt sA (vitCotHVD gf ε γ2 Wfc1 Wfc2 sM h m1 dyOut))

section DropBlock
variable {Np1 heads d mlpDim : Nat}

/-- **The multi-head block with its two drop sites, spelled as the render emits it** —
    `vitBlockSpelledMHV` with `siteScale sA` on the out-projection's output and `siteScale sM` on
    fc2's, each before its skip add. -/
noncomputable def vitBlockSpelledMHVD (gf : GeluForm) (Np1 heads d mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : Vec (heads * d))
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (γ2 β2 : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d)) (sA sM : Option ℝ)
    (X : Mat Np1 (heads * d)) : Mat Np1 (heads * d) :=
  let xh1 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε 1 0 (X r)
  let sc1 : Mat Np1 (heads * d) := fun r => layerScale γ1 (xh1 r)
  let ln1 : Mat Np1 (heads * d) := fun r k => sc1 r k + β1 k
  let Q : Mat Np1 (heads * d) := fun r => dense Wq bq (ln1 r)
  let K : Mat Np1 (heads * d) := fun r => dense Wk bk (ln1 r)
  let V : Mat Np1 (heads * d) := fun r => dense Wv bv (ln1 r)
  let att : Mat Np1 (heads * d) := ∑ h : Fin heads, headPadMat Np1 heads d h
    (Mat.mul
      (rowSoftmax (fun i j => sdpaScale d *
        Mat.mul (headSliceMat Np1 heads d h Q)
          (Mat.transpose (headSliceMat Np1 heads d h K)) i j))
      (headSliceMat Np1 heads d h V))
  let O : Mat Np1 (heads * d) := fun r => dense Wo bo (att r)
  let hres : Mat Np1 (heads * d) := fun r s => X r s + siteScale sA (O r s)
  let xh2 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε 1 0 (hres r)
  let sc2 : Mat Np1 (heads * d) := fun r => layerScale γ2 (xh2 r)
  let ln2 : Mat Np1 (heads * d) := fun r k => sc2 r k + β2 k
  let m1 : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g : Mat Np1 mlpDim := fun r => gf.map mlpDim (m1 r)
  let m2 : Mat Np1 (heads * d) := fun r => dense Wfc2 bfc2 (g r)
  fun r s => hres r s + siteScale sM (m2 r s)

/-- The block's forward at its two sites (`vitBlockFwdOMHV` at `none none`). -/
noncomputable abbrev _root_.Proofs.BlockParamsV.fwdOD (gf : GeluForm) {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (ε : ℝ) (sA sM : Option ℝ) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun xin => Mat.flatten (vitBlockSpelledMHVD gf Np1 heads d mlpDim ε p.γ1 p.β1 p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 sA sM (Mat.unflatten xin))

/-- With no site rendered the block is the drop-free one. -/
theorem fwdOD_none {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    p.fwdOD gf (Np1 := Np1) ε none none = p.fwdO gf ε := rfl

/-- **The block's input cotangent at its two sites** — `vitBlockCotInAtMHV`'s `let` chain with the
    saves recomputed at the attention site (`h` reads `siteScale sA` of the out-projection), the
    MLP branch fed `sM ⊙ dyOut` (`vitCotHVD`) and the attention branch `sA ⊙ cotH`; both skip
    fan-ins raw. -/
noncomputable def vitBlockCotInAtMHVD (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let X    : Mat Np1 (heads * d) := Mat.unflatten xin
  let ln1  : Mat Np1 (heads * d) := fun r kk => layerScale γ1 (fun s => layerNormForward (heads * d) ε 1 0 (X r) s) kk + β1 kk
  let Q    : Mat Np1 (heads * d) := fun r => dense Wq bq (ln1 r)
  let K    : Mat Np1 (heads * d) := fun r => dense Wk bk (ln1 r)
  let V    : Mat Np1 (heads * d) := fun r => dense Wv bv (ln1 r)
  let att  : Mat Np1 (heads * d) := ∑ hh : Fin heads, headPadMat Np1 heads d hh
    (Mat.mul (rowSoftmax (fun i j => sdpaScale d *
        Mat.mul (headSliceMat Np1 heads d hh Q) (Mat.transpose (headSliceMat Np1 heads d hh K)) i j))
      (headSliceMat Np1 heads d hh V))
  let h    : Mat Np1 (heads * d) := fun r s => X r s + siteScale sA (dense Wo bo (att r) s)
  let ln2  : Mat Np1 (heads * d) := fun r kk => layerScale γ2 (fun s => layerNormForward (heads * d) ε 1 0 (h r) s) kk + β2 kk
  let m1   : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let cotH := vitCotHVD gf ε γ2 Wfc1 Wfc2 sM (Mat.flatten h) (Mat.flatten m1) dyOut
  let dAtt := StableHLO.rowDenseBackFlat Np1 (heads * d) (heads * d) Wo (dropScalarOpt sA cotH)
  let dQ   := vitCotDQmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let dK   := vitCotDKmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let dV   := vitCotDVmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  vitCotXinV ε γ1 Wq Wk Wv xin dQ dK dV cotH

/-- The block's input cotangent at its two sites (`vitBlockCotInAtMHV` at `none none`). -/
noncomputable abbrev _root_.Proofs.BlockParamsV.cotInD (gf : GeluForm) {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (ε : ℝ) (sA sM : Option ℝ) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  vitBlockCotInAtMHVD gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1
    p.bfc1 p.Wfc2 sA sM

/-- With no site rendered the chain is the drop-free one. -/
theorem cotInD_none {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    p.cotInD gf (Np1 := Np1) ε none none = p.cotIn gf ε := rfl

-- ════════════════════════════════════════════════════════════════
-- § The two branches, flat, with their certified VJPs
-- ════════════════════════════════════════════════════════════════

/-- The attention branch `mhsa ∘ LN₁`, flat. -/
noncomputable def vitAttnBrF (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v => Mat.flatten (((mhsaLayer Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo) ∘
    (fun X : Mat Np1 (heads * d) => fun n => layerNormVec (heads * d) ε p.γ1 p.β1 (X n)))
    (Mat.unflatten v))

/-- The MLP branch `mlp ∘ LN₂`, flat. -/
noncomputable def vitMlpBrF (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v => Mat.flatten (((transformerMlp gf Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) ∘
    (fun X : Mat Np1 (heads * d) => fun n => layerNormVec (heads * d) ε p.γ2 p.β2 (X n)))
    (Mat.unflatten v))

theorem vitAttnBrF_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) :
    Differentiable ℝ (vitAttnBrF (Np1 := Np1) ε p) :=
  flat_differentiable_comp (layerNormVec_per_token_flat_differentiable Np1 (heads * d) ε p.γ1 p.β1 hε)
    (mhsaLayer_flat_differentiable Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo)

theorem vitMlpBrF_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) :
    Differentiable ℝ (vitMlpBrF gf (Np1 := Np1) ε p) :=
  flat_differentiable_comp (layerNormVec_per_token_flat_differentiable Np1 (heads * d) ε p.γ2 p.β2 hε)
    (transformerMlp_flat_differentiable Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)

/-- The attention branch's VJP — the branch half of `transformerAttnSublayerVHasVJPMat`. -/
noncomputable def vitAttnBrHasVJP (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) :
    HasVJP (vitAttnBrF (Np1 := Np1) ε p) :=
  (vjpMatComp _ _ (layerNormVec_per_token_flat_differentiable Np1 (heads * d) ε p.γ1 p.β1 hε)
    (mhsaLayer_flat_differentiable Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo)
    (layerNormVecPerTokenHasVJPMat Np1 (heads * d) ε p.γ1 p.β1 hε)
    (mhsaHasVJPMat Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo)).toHasVJP

/-- The MLP branch's VJP — the branch half of `transformerMlpSublayerVHasVJPMat`. -/
noncomputable def vitMlpBrHasVJP (gf : GeluForm) (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) :
    HasVJP (vitMlpBrF gf (Np1 := Np1) ε p) :=
  (vjpMatComp _ _ (layerNormVec_per_token_flat_differentiable Np1 (heads * d) ε p.γ2 p.β2 hε)
    (transformerMlp_flat_differentiable Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)
    (layerNormVecPerTokenHasVJPMat Np1 (heads * d) ε p.γ2 p.β2 hε)
    (transformerMlpHasVJPMat gf Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)).toHasVJP

/-- **The attention branch's backward is the render's chain** (`LN₁`-back after the multi-head
    backward, at the saves of input `v`): `attnSubFlat_tie_v` with its skip taken off. -/
theorem vitAttnBr_back (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (v w : Vec (Np1 * (heads * d))) :
    (vitAttnBrHasVJP ε hε p).backward v w
      = (rowLNVecFlatBack Np1 (heads * d) ε p.γ1 v
          ∘ mhsaBackFlat p.Wq p.Wk p.Wv p.Wo
              (fun r => Proofs.dense p.Wq p.bq (layerNormVec (heads * d) ε p.γ1 p.β1 (Mat.unflatten v r)))
              (fun r => Proofs.dense p.Wk p.bk (layerNormVec (heads * d) ε p.γ1 p.β1 (Mat.unflatten v r)))
              (fun r => Proofs.dense p.Wv p.bv (layerNormVec (heads * d) ε p.γ1 p.β1 (Mat.unflatten v r)))) w := by
  funext idx
  have h := congrFun (attnSubFlat_tie_v ε hε p.γ1 p.β1 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo
    (Mat.unflatten v) w) idx
  rw [Mat.flatten_unflatten] at h
  simp only [Proofs.residual, biPath, Mat.flatten_apply, transformerAttnSublayerVHasVJPMat,
    preLNResHasVJPMat, biPathMatHasVJP, identityMatHasVJP] at h
  unfold vitAttnBrHasVJP
  rw [HasVJPMat.toHasVJP_backward]
  have hw : Mat.unflatten w (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = w idx := by
    simp only [Mat.unflatten, Prod.mk.eta, Equiv.apply_symm_apply]
  rw [hw] at h
  linarith

/-- **The MLP branch's backward is the render's chain** (`LN₂`-back after the per-row MLP back,
    at the saves of input `v`): `mlpSubFlat_tie_v` with its skip taken off. -/
theorem vitMlpBr_back {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (v w : Vec (Np1 * (heads * d))) :
    (vitMlpBrHasVJP gf ε hε p).backward v w
      = (rowLNVecFlatBack Np1 (heads * d) ε p.γ2 v
          ∘ perRowFlatPR Np1 (heads * d) (fun r =>
              Proofs.dense (Mat.transpose p.Wfc1) (0 : Vec (heads * d))
                ∘ diagBack (fun c => gf.scalarDeriv (Proofs.dense p.Wfc1 p.bfc1
                    (layerNormVec (heads * d) ε p.γ2 p.β2 (Mat.unflatten v r)) c))
                ∘ Proofs.dense (Mat.transpose p.Wfc2) (0 : Vec mlpDim))) w := by
  funext idx
  have h := congrFun (mlpSubFlat_tie_v (gf := gf) mlpDim ε hε p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
    (Mat.unflatten v) w) idx
  rw [Mat.flatten_unflatten] at h
  simp only [Proofs.residual, biPath, Mat.flatten_apply, transformerMlpSublayerVHasVJPMat,
    preLNResHasVJPMat, biPathMatHasVJP, identityMatHasVJP] at h
  unfold vitMlpBrHasVJP
  rw [HasVJPMat.toHasVJP_backward]
  have hw : Mat.unflatten w (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = w idx := by
    simp only [Mat.unflatten, Prod.mk.eta, Equiv.apply_symm_apply]
  rw [hw] at h
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The block as two site residuals — its VJP, and the chain as that VJP's backward
-- ════════════════════════════════════════════════════════════════

/-- The attention sublayer with its site, flat: `v ↦ v + sA ⊙ (mhsa ∘ LN₁) v`. -/
noncomputable def vitAttnSiteF (sA : Option ℝ) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v i => v i + dropScalarOpt sA (vitAttnBrF ε p v) i

/-- The MLP sublayer with its site, flat: `v ↦ v + sM ⊙ (mlp ∘ LN₂) v`. -/
noncomputable def vitMlpSiteF (gf : GeluForm) (sM : Option ℝ) (ε : ℝ)
    (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v i => v i + dropScalarOpt sM (vitMlpBrF gf ε p v) i

/-- The attention sublayer's output at its site, as a matrix. -/
theorem vitAttnSiteF_eq (sA : Option ℝ) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (xin : Vec (Np1 * (heads * d))) :
    vitAttnSiteF sA ε p xin
      = Mat.flatten (fun r s => Mat.unflatten xin r s + siteScale sA
          (mhsaLayer Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo
            (fun n => layerNormVec (heads * d) ε p.γ1 p.β1 (Mat.unflatten xin n)) r s)) := by
  funext idx
  simp only [vitAttnSiteF, vitAttnBrF, dropScalarOpt, Function.comp_apply, Mat.flatten_apply,
    Mat.unflatten, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **The spelled block with its sites is the two site residuals, composed.** -/
theorem fwdOD_eq_sites {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) :
    p.fwdOD gf (Np1 := Np1) ε sA sM = vitMlpSiteF gf sM ε p ∘ vitAttnSiteF sA ε p := by
  funext v
  rw [Function.comp_apply, vitAttnSiteF_eq]
  unfold vitMlpSiteF vitMlpBrF
  rw [Mat.unflatten_flatten, mhsaLayer_spelled]
  funext idx
  simp only [Mat.flatten_apply, dropScalarOpt, Function.comp_apply]
  rfl

theorem vitAttnSiteF_differentiable (sA : Option ℝ) (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) : Differentiable ℝ (vitAttnSiteF (Np1 := Np1) sA ε p) :=
  siteRes_differentiable sA _ (vitAttnBrF_differentiable ε hε p)

theorem vitMlpSiteF_differentiable {gf : GeluForm} (sM : Option ℝ) (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) : Differentiable ℝ (vitMlpSiteF gf (Np1 := Np1) sM ε p) :=
  siteRes_differentiable sM _ (vitMlpBrF_differentiable ε hε p)

/-- The attention site residual's VJP (`siteResHasVJP` over the branch). -/
noncomputable def vitAttnSiteHasVJP (sA : Option ℝ) (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) : HasVJP (vitAttnSiteF (Np1 := Np1) sA ε p) :=
  siteResHasVJP sA _ (vitAttnBrF_differentiable ε hε p) (vitAttnBrHasVJP ε hε p)

/-- The MLP site residual's VJP (`siteResHasVJP` over the branch). -/
noncomputable def vitMlpSiteHasVJP (gf : GeluForm) (sM : Option ℝ) (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) : HasVJP (vitMlpSiteF gf (Np1 := Np1) sM ε p) :=
  siteResHasVJP sM _ (vitMlpBrF_differentiable ε hε p) (vitMlpBrHasVJP gf ε hε p)

theorem fwdOD_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) : Differentiable ℝ (p.fwdOD gf (Np1 := Np1) ε sA sM) := by
  rw [fwdOD_eq_sites]
  exact (vitMlpSiteF_differentiable sM ε hε p).comp (vitAttnSiteF_differentiable sA ε hε p)

/-- **The block's VJP at its two sites**: the attention site residual, then the MLP one, each
    `siteResHasVJP` over its branch's certified VJP. -/
noncomputable def _root_.Proofs.BlockParamsV.fwdODHasVJP (gf : GeluForm) {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (ε : ℝ) (hε : 0 < ε) (sA sM : Option ℝ) :
    HasVJP (p.fwdOD gf (Np1 := Np1) ε sA sM) :=
  (vjpComp (vitAttnSiteF sA ε p) (vitMlpSiteF gf sM ε p)
    (vitAttnSiteF_differentiable sA ε hε p) (vitMlpSiteF_differentiable sM ε hε p)
    (vitAttnSiteHasVJP sA ε hε p) (vitMlpSiteHasVJP gf sM ε hε p)).congr
    (fwdOD_eq_sites ε p sA sM).symm

/-- The block VJP's backward, unfolded: the MLP branch reads `sM ⊙ dy` at the attention output, the
    attention branch `sA ⊙` the resulting skip cotangent; both skips raw. -/
theorem fwdODHasVJP_backward {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (x dy : Vec (Np1 * (heads * d))) :
    (p.fwdODHasVJP gf ε hε sA sM).backward x dy
      = let c : Vec (Np1 * (heads * d)) := fun i => dy i
          + (vitMlpBrHasVJP gf ε hε p).backward (vitAttnSiteF sA ε p x) (dropScalarOpt sM dy) i
        fun i => c i + (vitAttnBrHasVJP ε hε p).backward x (dropScalarOpt sA c) i := rfl

/-- The attention half of the chain's algebra: the block-input fan-in of the three dense
    cotangents the core hands back from `Woᵀ w` is `c` plus the attention branch's backward of `w`
    (`vitCotXin_eq_blockBack`'s attention step, at a general skip cotangent `c`). -/
theorem vitCotXinV_attn (ε : ℝ) (γ1 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d))
    (Q K V : Mat Np1 (heads * d)) (xin w c : Vec (Np1 * (heads * d))) :
    vitCotXinV ε γ1 Wq Wk Wv xin
        (vitCotDQmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V)
          (StableHLO.rowDenseBackFlat Np1 (heads * d) (heads * d) Wo w))
        (vitCotDKmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V)
          (StableHLO.rowDenseBackFlat Np1 (heads * d) (heads * d) Wo w))
        (vitCotDVmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V)
          (StableHLO.rowDenseBackFlat Np1 (heads * d) (heads * d) Wo w)) c
      = fun i => c i + (rowLNVecFlatBack Np1 (heads * d) ε γ1 xin
          ∘ mhsaBackFlat Wq Wk Wv Wo Q K V) w i := by
  have hL1 : ∀ a b c : Vec (Np1 * (heads * d)), vitCotLn1 Wq Wk Wv a b c
      = fun j => perRowFlat Np1 (heads * d) (Proofs.dense (Mat.transpose Wq) 0) a j
          + (perRowFlat Np1 (heads * d) (Proofs.dense (Mat.transpose Wk) 0) b j
            + perRowFlat Np1 (heads * d) (Proofs.dense (Mat.transpose Wv) 0) c j) := by
    intro a b c; funext j
    simp only [vitCotLn1, rowDenseBackFlat_eq_perRowFlat]
    ring
  rw [vitCotDQmh_eq_core, vitCotDKmh_eq_core, vitCotDVmh_eq_core]
  funext i
  simp only [vitCotXinV, rowLNBack_affine_eq, hL1, rowDenseBackFlat_eq_perRowFlat,
    Function.comp_apply, mhsaBackFlat]

/-- The MLP half: `vitCotHVD` is the raw skip plus the MLP branch's backward of `sM ⊙ dy`. -/
theorem vitCotHVD_eq {gf : GeluForm} {D : Nat} (ε : ℝ) (γ2 : Vec D) (Wfc1 : Mat D mlpDim)
    (Wfc2 : Mat mlpDim D) (sM : Option ℝ) (H : Mat Np1 D) (m1 : Mat Np1 mlpDim) (dy : Vec (Np1 * D)) :
    vitCotHVD gf ε γ2 Wfc1 Wfc2 sM (Mat.flatten H) (Mat.flatten m1) dy
      = fun i => dy i + (rowLNVecFlatBack Np1 D ε γ2 (Mat.flatten H)
          ∘ perRowFlatPR Np1 D (fun r => Proofs.dense (Mat.transpose Wfc1) (0 : Vec D)
            ∘ diagBack (fun c => gf.scalarDeriv (m1 r c))
            ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec mlpDim))) (dropScalarOpt sM dy) i := by
  funext i
  simp only [vitCotHVD, rowLNBack_affine_eq, vitCotLn2_eq_perRowFlatPR, Function.comp_apply]

/-- **The chain's block-input cotangent at the two sites is the block VJP's backward.** -/
theorem cotInD_eq_vjp {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) :
    p.cotInD gf ε sA sM xin dyOut = (p.fwdODHasVJP gf ε hε sA sM).backward xin dyOut := by
  rw [fwdODHasVJP_backward]
  simp only [vitAttnBr_back, vitMlpBr_back]
  rw [vitAttnSiteF_eq, Mat.unflatten_flatten, mhsaLayer_spelled]
  simp only [BlockParamsV.cotInD, vitBlockCotInAtMHVD]
  rw [vitCotXinV_attn, vitCotHVD_eq]
  rfl

end DropBlock

end Proofs.ViTTieGB
