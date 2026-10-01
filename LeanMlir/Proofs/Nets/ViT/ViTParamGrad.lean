import LeanMlir.Proofs.Nets.ViT.ViTStepTieGB
import LeanMlir.Proofs.Foundation.ParamGradNodes

/-! # ViT-Tiny — every parameter gradient node IS the loss's derivative in that parameter

`vit_net_tiedGB` says each of the 200 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it, the chain's top
being the smoothed-loss cotangent. `vit_net_lossGrad` composes that with the chain: for any loss
`L` of the logits whose gradient at the net's output is `g`, the loss of the WHOLE net `vitNetB`
with that one parameter varied is differentiable in it and every node is its gradient
(`HasGradAt`). `vitNetB` is the canonical forward, batched: `batchMap N` of `vitForwardKV` at the
twelve blocks (`vitNetB_eq_vitForwardKV`). `vit_net_lossGrad_smoothedCE` discharges `hL` for
the label-smoothed loss the artifacts ship (`smoothedBatchLossDiv`, whose gradient is the
`softmaxDiv` cotangent the render emits).

**How.** ConvNeXt's shape (`ConvNeXtParamGrad.lean`): no ViT op couples examples, so the work is
per example and lifted once.

* **Per example** (at variable widths): the loss read after each of a block's eight parameterised
  ops has the tie's own per-example cotangent as its gradient (`vitPostL1_hasGradAt`,
  `vitPostQ_hasGradAt`, …, `vitPostF1_hasGradAt`). Each is a short chain through certified VJPs:
  the MLP sublayer (`mlpSubFlat_tie_v` is its backward in the chain's spelling), the out-projection,
  the full attention layer for LN₁ (`mhsaBackFlat_eq_mhsa_vjp`).
* **The attention core in one of `Q`, `K`, `V`** is the new piece. The render's per-head core is a
  column-slab map with a different function on each head's slab (head `h` reads `K`'s and `V`'s
  slab `h`), so it is Attention's `colSlabApplyH`, whose Jacobian stays block-diagonal
  (`pdivMat_colIndepH`). Each head's VJP is the certified single-head
  `sdpaBack{Q,K,V}`, and the lifted backward IS the tie's `coreQFlat` / `coreKFlat` /
  `coreVFlat` (`attnCoreQ_backward`, …, all `rfl`).
* **Lifted** (`HasGradAt.param_batchMap_through`, ParamGrad): read against the linear loss
  `⟨·, dyₙ⟩` per example, those gradients turn each tied node's `Σ_n Σ_j ∂per/∂θ · cotₙ` into
  `∂G/∂θ` of the batched block, `G` the loss at the block's output. Each node needs the block
  forward read as "the rest of the block ∘ the node's op ∘ the prefix" (`vit_fwd_Wq`, …), an
  equation up to `Mat.unflatten_flatten`.
* **Per net**: the loss read after each stage (`vitSuf*`), pulled back through the certified
  batched block and head VJPs (`vitBlockCotInB_eq_vjp`, `vitCotTowerOutB_eq_vjp`), and each `Φ`
  identified with the whole net at updated weights by a standalone `vit_factor_*` theorem.

**Two nodes carry the batch sum inside `den`,** as their text does: the classifier bias
(`biasGradB`) and the CLS token.

**Hypotheses.** `0 < ε` (the LayerNorms' VJPs); no smoothness hypothesis (GELU has no kink). For
the smoothed loss, every example's target sums to one and `0 < nC`. Drop-path and the bf16 nodes
are outside this statement, as they are outside the tie. Stated at ViT-Tiny's literal dims, as the
tie is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTTieGB

open scoped BigOperators
open Proofs.ViTTie (ViTTieWeights)
open Proofs.ViTFoldGB (rowDenseBTiedB_holds rowDenseWTiedB_holds)
open Proofs.GradNodeB (vecLNBetaTiedB_holds vecLNGammaTiedB_holds hasGradAt_constAdd)

-- ════════════════════════════════════════════════════════════════
-- § The attention core in one of `Q`, `K`, `V`, the other two fixed
-- ════════════════════════════════════════════════════════════════

section Core
variable {Np1 heads d : Nat}

/-- The multi-head attention core as the render spells it: per head, slice → scaled `Q·Kᵀ` →
    row softmax → `·V` → pad, summed over heads (`blkSaves`' `att`). -/
noncomputable def attnCore (Q K V : Mat Np1 (heads * d)) : Mat Np1 (heads * d) :=
  ∑ hh : Fin heads, headPadMat Np1 heads d hh
    (Mat.mul (rowSoftmax (fun i j => sdpaScale d *
        Mat.mul (headSliceMat Np1 heads d hh Q) (Mat.transpose (headSliceMat Np1 heads d hh K)) i j))
      (headSliceMat Np1 heads d hh V))

/-- The core in `Q` is a per-slab map: each head's attention on its own `Q` slab
    (`sum_headPadMat_apply`). -/
theorem attnCore_eq_colSlabQ (K V : Mat Np1 (heads * d)) :
    (fun Q => attnCore Q K V) = colSlabApplyH (fun hh Q' =>
      sdpa Np1 d Q' (headSliceMat Np1 heads d hh K) (headSliceMat Np1 heads d hh V)) := by
  funext Q r hj; rw [attnCore, sum_headPadMat_apply]; rfl

/-- …in `K`. -/
theorem attnCore_eq_colSlabK (Q V : Mat Np1 (heads * d)) :
    (fun K => attnCore Q K V) = colSlabApplyH (fun hh K' =>
      sdpa Np1 d (headSliceMat Np1 heads d hh Q) K' (headSliceMat Np1 heads d hh V)) := by
  funext K r hj; rw [attnCore, sum_headPadMat_apply]; rfl

/-- …in `V`. -/
theorem attnCore_eq_colSlabV (Q K : Mat Np1 (heads * d)) :
    (fun V => attnCore Q K V) = colSlabApplyH (fun hh V' =>
      sdpa Np1 d (headSliceMat Np1 heads d hh Q) (headSliceMat Np1 heads d hh K) V') := by
  funext V r hj; rw [attnCore, sum_headPadMat_apply]; rfl

/-- The attention core's VJP in `Q`, `K`, `V` fixed: per head, the certified `sdpaBackQ`. -/
noncomputable def attnCoreQHasVJPMat (K V : Mat Np1 (heads * d)) :
    HasVJPMat (fun Q => attnCore Q K V) where
  backward := (colSlabwiseHasVJPMatH (g := fun hh Q' =>
      sdpa Np1 d Q' (headSliceMat Np1 heads d hh K) (headSliceMat Np1 heads d hh V))
    (fun hh => ⟨fun Q' dY => sdpaBackQ Np1 d Q' (headSliceMat Np1 heads d hh K)
        (headSliceMat Np1 heads d hh V) dY, fun Q' dY i j => sdpaBackQ_correct Np1 d Q' _ _ dY i j⟩)
    (fun hh => sdpaQ_flat_differentiable Np1 d _ _)).backward
  correct A dY i j := by
    rw [attnCore_eq_colSlabQ]
    exact (colSlabwiseHasVJPMatH _ _).correct A dY i j

/-- …in `K`: per head, `sdpaBackK`. -/
noncomputable def attnCoreKHasVJPMat (Q V : Mat Np1 (heads * d)) :
    HasVJPMat (fun K => attnCore Q K V) where
  backward := (colSlabwiseHasVJPMatH (g := fun hh K' =>
      sdpa Np1 d (headSliceMat Np1 heads d hh Q) K' (headSliceMat Np1 heads d hh V))
    (fun hh => ⟨fun K' dY => sdpaBackK Np1 d (headSliceMat Np1 heads d hh Q) K'
        (headSliceMat Np1 heads d hh V) dY, fun K' dY i j => sdpaBackK_correct Np1 d _ K' _ dY i j⟩)
    (fun hh => sdpaK_flat_differentiable Np1 d _ _)).backward
  correct A dY i j := by
    rw [attnCore_eq_colSlabK]
    exact (colSlabwiseHasVJPMatH _ _).correct A dY i j

/-- …in `V`: per head, `sdpaBackV`. -/
noncomputable def attnCoreVHasVJPMat (Q K : Mat Np1 (heads * d)) :
    HasVJPMat (fun V => attnCore Q K V) where
  backward := (colSlabwiseHasVJPMatH (g := fun hh V' =>
      sdpa Np1 d (headSliceMat Np1 heads d hh Q) (headSliceMat Np1 heads d hh K) V')
    (fun hh => ⟨fun V' dY => sdpaBackV Np1 d (headSliceMat Np1 heads d hh Q)
        (headSliceMat Np1 heads d hh K) V' dY, fun V' dY i j => sdpaBackV_correct Np1 d _ _ V' dY i j⟩)
    (fun hh => sdpaV_flat_differentiable Np1 d _ _)).backward
  correct A dY i j := by
    rw [attnCore_eq_colSlabV]
    exact (colSlabwiseHasVJPMatH _ _).correct A dY i j

/-- The core is differentiable in `Q`, flattened. -/
theorem attnCoreQ_flat_differentiable (K V : Mat Np1 (heads * d)) :
    Differentiable ℝ (fun v : Vec (Np1 * (heads * d)) =>
      Mat.flatten (attnCore (Mat.unflatten v) K V)) := by
  have h := colSlabApplyH_flat_differentiable (fun hh Q' =>
    sdpa Np1 d Q' (headSliceMat Np1 heads d hh K) (headSliceMat Np1 heads d hh V))
    (fun hh => sdpaQ_flat_differentiable Np1 d _ _)
  rw [← attnCore_eq_colSlabQ] at h
  exact h

/-- …in `K`. -/
theorem attnCoreK_flat_differentiable (Q V : Mat Np1 (heads * d)) :
    Differentiable ℝ (fun v : Vec (Np1 * (heads * d)) =>
      Mat.flatten (attnCore Q (Mat.unflatten v) V)) := by
  have h := colSlabApplyH_flat_differentiable (fun hh K' =>
    sdpa Np1 d (headSliceMat Np1 heads d hh Q) K' (headSliceMat Np1 heads d hh V))
    (fun hh => sdpaK_flat_differentiable Np1 d _ _)
  rw [← attnCore_eq_colSlabK] at h
  exact h

/-- …in `V`. -/
theorem attnCoreV_flat_differentiable (Q K : Mat Np1 (heads * d)) :
    Differentiable ℝ (fun v : Vec (Np1 * (heads * d)) =>
      Mat.flatten (attnCore Q K (Mat.unflatten v))) := by
  have h := colSlabApplyH_flat_differentiable (fun hh V' =>
    sdpa Np1 d (headSliceMat Np1 heads d hh Q) (headSliceMat Np1 heads d hh K) V')
    (fun hh => sdpaV_flat_differentiable Np1 d _ _)
  rw [← attnCore_eq_colSlabV] at h
  exact h

/-- **The core's `Q` backward IS the tie's `coreQFlat`** — the per-head `sdpaBackQ` on each slab,
    concatenated; `rfl` once the saved `Q` is unflattened. -/
theorem attnCoreQ_backward (Q K V : Mat Np1 (heads * d)) (dA : Vec (Np1 * (heads * d))) :
    (attnCoreQHasVJPMat K V).toHasVJP.backward (Mat.flatten Q) dA = coreQFlat Q K V dA := by
  funext idx
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

/-- …`K`: `coreKFlat`. -/
theorem attnCoreK_backward (Q K V : Mat Np1 (heads * d)) (dA : Vec (Np1 * (heads * d))) :
    (attnCoreKHasVJPMat Q V).toHasVJP.backward (Mat.flatten K) dA = coreKFlat Q K V dA := by
  funext idx
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

/-- …`V`: `coreVFlat`. -/
theorem attnCoreV_backward (Q K V : Mat Np1 (heads * d)) (dA : Vec (Np1 * (heads * d))) :
    (attnCoreVHasVJPMat Q K).toHasVJP.backward (Mat.flatten V) dA = coreVFlat Q K V dA := by
  funext idx
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

end Core

-- ════════════════════════════════════════════════════════════════
-- § A ViT block, per example — the loss read after each parameterised op
--   xin → LN₁ → Q/K/V → core → out-proj → + xin = h → LN₂ → fc1 → GELU → fc2 → + h
--   `vitPost*` is the rest of the block after each op, as a function of its output.
-- ════════════════════════════════════════════════════════════════

section Chain
variable {Np1 heads d mlpDim : Nat}

/-- The flat MLP sublayer `h ↦ h + MLP(LN₂ h)`. -/
noncomputable def vitMlpSubF (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v => Mat.flatten (transformerMlpSublayerV Np1 heads d mlpDim ε p.γ2 p.β2 p.Wfc1 p.bfc1
    p.Wfc2 p.bfc2 (Mat.unflatten v))

/-- The flat MLP sublayer is differentiable (`0 < ε`, the LayerNorm). -/
theorem vitMlpSubF_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) :
    Differentiable ℝ (vitMlpSubF (Np1 := Np1) ε p) := by
  have hin : Differentiable ℝ (fun v : Vec (Np1 * (heads * d)) =>
      Mat.flatten (((transformerMlp Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) ∘
        (fun X : Mat Np1 (heads * d) => fun n => layerNormVec (heads * d) ε p.γ2 p.β2 (X n)))
        (Mat.unflatten v))) := by
    simpa [Function.comp_def, Mat.unflatten_flatten] using
      (transformerMlp_flat_differentiable Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2).comp
        (layerNormVec_per_token_flat_differentiable Np1 (heads * d) ε p.γ2 p.β2 hε)
  exact (identity_mat_flat_differentiable Np1 (heads * d)).add hin

/-- `vitCotHV` is the MLP sublayer's backward in the chain's spelling (`vitCotXin_eq_blockBack`'s
    first step). -/
theorem vitCotHV_eq_residual (ε : ℝ) (γ2 : Vec (heads * d)) (Wfc1 : Mat (heads * d) mlpDim)
    (Wfc2 : Mat mlpDim (heads * d)) (H : Mat Np1 (heads * d)) (m1 : Mat Np1 mlpDim)
    (dy : Vec (Np1 * (heads * d))) :
    vitCotHV ε γ2 Wfc1 Wfc2 (Mat.flatten H) (Mat.flatten m1) dy
      = Proofs.residual (rowLNVecFlatBack Np1 (heads * d) ε γ2 (Mat.flatten H)
          ∘ perRowFlatPR Np1 (heads * d) (fun r => Proofs.dense (Mat.transpose Wfc1) 0
            ∘ diagBack (fun c => geluScalarDeriv (m1 r c))
            ∘ Proofs.dense (Mat.transpose Wfc2) 0)) dy := by
  funext i
  simp only [vitCotHV, Proofs.residual, biPath, Function.comp_apply, rowLNBack_affine_eq,
    vitCotLn2_eq_perRowFlatPR]
  ring

/-- **The MLP sublayer's input gradient is `vitCotHV`**, at any saved sublayer input `H`. -/
theorem vitMlpSub_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (H : Mat Np1 (heads * d)) (dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun v => linLoss dy (vitMlpSubF ε p v)) (Mat.flatten H)
      (vitCotHV ε p.γ2 p.Wfc1 p.Wfc2 (Mat.flatten H)
        (Mat.flatten (fun r => Proofs.dense p.Wfc1 p.bfc1 (layerNormVec (heads * d) ε p.γ2 p.β2 (H r))))
        dy) := by
  refine (HasGradAt.comp_global (f := vitMlpSubF ε p) (x := Mat.flatten H) (hasGradAt_linLoss dy _)
    (vitMlpSubF_differentiable ε hε p)
    (transformerMlpSublayerVHasVJPMat Np1 heads d mlpDim ε p.γ2 p.β2 hε p.Wfc1 p.bfc1 p.Wfc2
      p.bfc2).toHasVJP).of_eq ?_
  rw [vitCotHV_eq_residual, mlpSubFlat_tie_v mlpDim ε hε p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 H dy]
  funext idx
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl


/-! The block's forward, per example, as named `Mat`s — `blkSaves`' `let` chain verbatim. -/

/-- LN₁'s output. -/
noncomputable def vitLn1M (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) :=
  fun r kk => layerScale p.γ1 (fun s => layerNormForward (heads * d) ε 1 0 (Mat.unflatten y r) s) kk
    + p.β1 kk

/-- The Q projection. -/
noncomputable def vitQM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) := fun r => Proofs.dense p.Wq p.bq (vitLn1M ε p y r)

/-- The K projection. -/
noncomputable def vitKM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) := fun r => Proofs.dense p.Wk p.bk (vitLn1M ε p y r)

/-- The V projection. -/
noncomputable def vitVM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) := fun r => Proofs.dense p.Wv p.bv (vitLn1M ε p y r)

/-- The attention core's output (the out-projection's input). -/
noncomputable def vitAttM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) := attnCore (vitQM ε p y) (vitKM ε p y) (vitVM ε p y)

/-- The out-projection's output. -/
noncomputable def vitOM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) := fun r => Proofs.dense p.Wo p.bo (vitAttM ε p y r)

/-- The attention sublayer's output `h`. -/
noncomputable def vitHM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) := fun r s => Mat.unflatten y r s + vitOM ε p y r s

/-- LN₂'s output. -/
noncomputable def vitLn2M (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 (heads * d) :=
  fun r kk => layerScale p.γ2 (fun s => layerNormForward (heads * d) ε 1 0 (vitHM ε p y r) s) kk
    + p.β2 kk

/-- fc1's output (pre-GELU). -/
noncomputable def vitM1M (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Mat Np1 mlpDim := fun r => Proofs.dense p.Wfc1 p.bfc1 (vitLn2M ε p y r)

/-- The out-projection, flat. -/
noncomputable def vitWoF (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v => Mat.flatten (fun r => Proofs.dense p.Wo p.bo (Mat.unflatten v r))

/-- The block after the out-projection: the attention residual, then the MLP sublayer. -/
noncomputable def vitPostO (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitMlpSubF ε p (fun i => y i + u i)

/-- The attention residual's flat sum is `h`, flattened. -/
theorem vitHM_flat (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    (fun i => y i + Mat.flatten (vitOM ε p y) i) = Mat.flatten (vitHM ε p y) := by
  funext i
  simp only [Mat.flatten, vitHM, Mat.unflatten_apply, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **The out-projection's output cotangent is `cH`** — the loss after the out-projection. -/
theorem vitPostO_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostO ε p y u)) (Mat.flatten (vitOM ε p y))
      (cH ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) :=
  HasGradAt.comp (f := fun u i => y i + u i) (x := Mat.flatten (vitOM ε p y))
    ((vitMlpSub_hasGradAt ε hε p (vitHM ε p y) dy).congr_point (vitHM_flat ε p y).symm)
    (differentiableAt_id.const_add y)
    (constAddHasVJPAt y (fun u => u) _ differentiableAt_id (identityHasVJPAt _ _))


/-- The block after the attention core: out-projection, then `vitPostO`. -/
theorem vitPostAtt_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun a => linLoss dy (vitPostO ε p y (vitWoF p a))) (Mat.flatten (vitAttM ε p y))
      (cAtt ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) :=
  HasGradAt.comp_global (f := vitWoF p) (x := Mat.flatten (vitAttM ε p y))
    ((vitPostO_hasGradAt ε hε p y dy).congr_point (by rw [vitWoF, Mat.unflatten_flatten]; rfl))
    (dense_per_token_flat_differentiable p.Wo p.bo)
    (densePerTokenHasVJPMat Np1 (heads * d) (heads * d) p.Wo p.bo).toHasVJP

/-- The block after the Q projection. -/
noncomputable def vitPostQ (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO ε p y (vitWoF p (Mat.flatten (attnCore (Mat.unflatten u) (vitKM ε p y) (vitVM ε p y))))

/-- The block after the K projection. -/
noncomputable def vitPostK (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO ε p y (vitWoF p (Mat.flatten (attnCore (vitQM ε p y) (Mat.unflatten u) (vitVM ε p y))))

/-- The block after the V projection. -/
noncomputable def vitPostV (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO ε p y (vitWoF p (Mat.flatten (attnCore (vitQM ε p y) (vitKM ε p y) (Mat.unflatten u))))

/-- **The Q projection's output cotangent is `cQ`**: `cAtt` pulled back through the core in `Q`. -/
theorem vitPostQ_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostQ ε p y u)) (Mat.flatten (vitQM ε p y))
      (cQ ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
  refine (HasGradAt.comp_global (f := fun u => Mat.flatten (attnCore (Mat.unflatten u) (vitKM ε p y) (vitVM ε p y)))
    (x := Mat.flatten (vitQM ε p y))
    ((vitPostAtt_hasGradAt ε hε p y dy).congr_point (by rw [Mat.unflatten_flatten]; rfl))
    (attnCoreQ_flat_differentiable _ _) (attnCoreQHasVJPMat _ _).toHasVJP).of_eq ?_
  rw [attnCoreQ_backward]
  exact (vitCotDQmh_eq_core _ _ _ _).symm

/-- **The K projection's output cotangent is `cK`.** -/
theorem vitPostK_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostK ε p y u)) (Mat.flatten (vitKM ε p y))
      (cK ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
  refine (HasGradAt.comp_global (f := fun u => Mat.flatten (attnCore (vitQM ε p y) (Mat.unflatten u) (vitVM ε p y)))
    (x := Mat.flatten (vitKM ε p y))
    ((vitPostAtt_hasGradAt ε hε p y dy).congr_point (by rw [Mat.unflatten_flatten]; rfl))
    (attnCoreK_flat_differentiable _ _) (attnCoreKHasVJPMat _ _).toHasVJP).of_eq ?_
  rw [attnCoreK_backward]
  exact (vitCotDKmh_eq_core _ _ _ _).symm

/-- **The V projection's output cotangent is `cV`.** -/
theorem vitPostV_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostV ε p y u)) (Mat.flatten (vitVM ε p y))
      (cV ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
  refine (HasGradAt.comp_global (f := fun u => Mat.flatten (attnCore (vitQM ε p y) (vitKM ε p y) (Mat.unflatten u)))
    (x := Mat.flatten (vitVM ε p y))
    ((vitPostAtt_hasGradAt ε hε p y dy).congr_point (by rw [Mat.unflatten_flatten]; rfl))
    (attnCoreV_flat_differentiable _ _) (attnCoreVHasVJPMat _ _).toHasVJP).of_eq ?_
  rw [attnCoreV_backward]
  exact (vitCotDVmh_eq_core _ _ _ _).symm


/-- The block after LN₁: the multi-head attention layer, then `vitPostO`. -/
noncomputable def vitPostL1 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO ε p y (Mat.flatten
    (mhsaLayer Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo (Mat.unflatten u)))

/-- **LN₁'s output cotangent is `cLn1`**: `cH` pulled back through the certified attention layer
    (`mhsaBackFlat_eq_mhsa_vjp`), whose three paths are the Q/K/V fan-in. -/
theorem vitPostL1_hasGradAt (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostL1 ε p y u)) (Mat.flatten (vitLn1M ε p y))
      (cLn1 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
  refine (HasGradAt.comp_global
    (f := fun u => Mat.flatten (mhsaLayer Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo (Mat.unflatten u)))
    (x := Mat.flatten (vitLn1M ε p y))
    ((vitPostO_hasGradAt ε hε p y dy).congr_point
      (by rw [Mat.unflatten_flatten, mhsaLayer_spelled]; rfl))
    (mhsaLayer_flat_differentiable Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo)
    (mhsaHasVJPMat Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo).toHasVJP).of_eq ?_
  have hb : (mhsaHasVJPMat Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo).toHasVJP.backward
        (Mat.flatten (vitLn1M ε p y))
      = mhsaBackFlat p.Wq p.Wk p.Wv p.Wo (vitQM ε p y) (vitKM ε p y) (vitVM ε p y) := by
    refine Eq.trans ?_ (mhsaBackFlat_eq_mhsa_vjp p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo
      (vitLn1M ε p y)).symm
    funext dc idx
    rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
    rfl
  have hc : cLn1 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy
      = vitCotLn1 p.Wq p.Wk p.Wv
          (vitCotDQmh Np1 heads d (Mat.flatten (vitQM ε p y)) (Mat.flatten (vitKM ε p y))
            (Mat.flatten (vitVM ε p y))
            (cAtt ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy))
          (vitCotDKmh Np1 heads d (Mat.flatten (vitQM ε p y)) (Mat.flatten (vitKM ε p y))
            (Mat.flatten (vitVM ε p y))
            (cAtt ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy))
          (vitCotDVmh Np1 heads d (Mat.flatten (vitQM ε p y)) (Mat.flatten (vitKM ε p y))
            (Mat.flatten (vitVM ε p y))
            (cAtt ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy)) :=
    rfl
  have ha : cAtt ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy
      = perRowFlat Np1 (heads * d) (Proofs.dense (Mat.transpose p.Wo) 0)
          (cH ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
    rw [← rowDenseBackFlat_eq_perRowFlat]; rfl
  rw [hb, hc, vitCotDQmh_eq_core, vitCotDKmh_eq_core, vitCotDVmh_eq_core, ha]
  funext i
  simp only [mhsaBackFlat, vitCotLn1, rowDenseBackFlat_eq_perRowFlat, Function.comp_apply]
  ring

/-- The block after LN₂: the MLP body, then the residual. -/
noncomputable def vitPostL2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u i => Mat.flatten (vitHM ε p y) i
    + Mat.flatten (transformerMlp Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 (Mat.unflatten u)) i

/-- **LN₂'s output cotangent is `cLn2`**: the MLP body's certified backward
    (`transformerMlp_back_flat_eq_perRowFlatPR`). -/
theorem vitPostL2_hasGradAt (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostL2 ε p y u)) (Mat.flatten (vitLn2M ε p y))
      (cLn2 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
  refine (HasGradAt.comp_global
    (f := fun u => Mat.flatten (transformerMlp Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 (Mat.unflatten u)))
    (x := Mat.flatten (vitLn2M ε p y)) (hasGradAt_constAdd (G := linLoss dy) _ _ (hasGradAt_linLoss dy _))
    (transformerMlp_flat_differentiable Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)
    (transformerMlpHasVJPMat Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2).toHasVJP).of_eq ?_
  have h := transformerMlp_back_flat_eq_perRowFlatPR Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
    (vitLn2M ε p y) dy
  refine (funext fun idx => ?_ : _ = _).trans (h.trans (vitCotLn2_eq_perRowFlatPR p.Wfc1 p.Wfc2
    (vitM1M ε p y) dy).symm)
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

/-- The block after fc1: GELU, fc2, then the residual. -/
noncomputable def vitPostF1 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * mlpDim) → Vec (Np1 * (heads * d)) :=
  fun u i => Mat.flatten (vitHM ε p y) i
    + Mat.flatten (fun r => Proofs.dense p.Wfc2 p.bfc2 (gelu mlpDim (Mat.unflatten u r))) i

/-- **fc1's output cotangent is `cM1`**: through fc2 and the GELU. -/
theorem vitPostF1_hasGradAt (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostF1 ε p y u)) (Mat.flatten (vitM1M ε p y))
      (cM1 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 y dy) := by
  refine (HasGradAt.comp_global
    (f := fun u => Mat.flatten (((fun Y : Mat Np1 mlpDim => fun n => Proofs.dense p.Wfc2 p.bfc2 (Y n)) ∘
      (fun Y : Mat Np1 mlpDim => fun n => gelu mlpDim (Y n))) (Mat.unflatten u)))
    (x := Mat.flatten (vitM1M ε p y)) (hasGradAt_constAdd (G := linLoss dy) _ _ (hasGradAt_linLoss dy _))
    (flat_differentiable_comp (gelu_per_token_flat_differentiable Np1 mlpDim)
      (dense_per_token_flat_differentiable p.Wfc2 p.bfc2))
    (vjpMatComp _ _ (gelu_per_token_flat_differentiable Np1 mlpDim)
      (dense_per_token_flat_differentiable p.Wfc2 p.bfc2) (geluPerTokenHasVJPMat Np1 mlpDim)
      (densePerTokenHasVJPMat Np1 mlpDim (heads * d) p.Wfc2 p.bfc2)).toHasVJP).of_eq ?_
  funext idx
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

/-- The block after fc2: the residual. -/
noncomputable def vitPostF2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u i => Mat.flatten (vitHM ε p y) i + u i


/-! ### The block forward, read after each node -/

/-- The block forward is `vitPostO` at the out-projection's output. -/
theorem vit_fwd_eq_postO (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    p.fwdO ε y = vitPostO ε p y (Mat.flatten (vitOM ε p y)) := by
  rw [vitPostO, vitMlpSubF, vitHM_flat, Mat.unflatten_flatten]; rfl

/-- The block forward is `vitPostF2` at fc2's output (`rfl`). -/
theorem vit_fwd_eq_postF2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    p.fwdO ε y = vitPostF2 ε p y
      (Mat.flatten (fun r => Proofs.dense p.Wfc2 p.bfc2 (gelu mlpDim (vitM1M ε p y r)))) := rfl

/-- The block with `γ1` varied is `vitPostL1` after LN₁ at that `γ1`. The fifteen lemmas below
    say the same for each other parameter: the node's op at the varied parameter, between the
    block's prefix and the rest of the block (`vitPost*`). -/
theorem vit_fwd_γ1 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with γ1 := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostL1 ε p y (Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β1 (Mat.unflatten y r))) := by
  rw [vit_fwd_eq_postO, vitPostL1, Mat.unflatten_flatten, mhsaLayer_spelled]; rfl

theorem vit_fwd_β1 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with β1 := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostL1 ε p y (Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ1 θ (Mat.unflatten y r))) := by
  rw [vit_fwd_eq_postO, vitPostL1, Mat.unflatten_flatten, mhsaLayer_spelled]; rfl

theorem vit_fwd_Wq (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (W : Mat (heads * d) (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with Wq := W } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostQ ε p y (Mat.flatten (fun r => Proofs.dense W p.bq
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostQ, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bq (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with bq := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostQ ε p y (Mat.flatten (fun r => Proofs.dense p.Wq θ
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostQ, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wk (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (W : Mat (heads * d) (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with Wk := W } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostK ε p y (Mat.flatten (fun r => Proofs.dense W p.bk
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostK, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bk (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with bk := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostK ε p y (Mat.flatten (fun r => Proofs.dense p.Wk θ
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostK, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wv (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (W : Mat (heads * d) (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with Wv := W } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostV ε p y (Mat.flatten (fun r => Proofs.dense W p.bv
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostV, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bv (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with bv := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostV ε p y (Mat.flatten (fun r => Proofs.dense p.Wv θ
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostV, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wo (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (W : Mat (heads * d) (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with Wo := W } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostO ε p y (Mat.flatten (fun r => Proofs.dense W p.bo
          (Mat.unflatten (Mat.flatten (vitAttM ε p y)) r))) := by
  rw [vit_fwd_eq_postO, Mat.unflatten_flatten]; rfl

theorem vit_fwd_bo (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with bo := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostO ε p y (Mat.flatten (fun r => Proofs.dense p.Wo θ
          (Mat.unflatten (Mat.flatten (vitAttM ε p y)) r))) := by
  rw [vit_fwd_eq_postO, Mat.unflatten_flatten]; rfl

theorem vit_fwd_γ2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with γ2 := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostL2 ε p y (Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β2
          (Mat.unflatten (Mat.flatten (vitHM ε p y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostL2 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_β2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with β2 := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostL2 ε p y (Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ2 θ
          (Mat.unflatten (Mat.flatten (vitHM ε p y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostL2 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wfc1 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (W : Mat (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) :
    ({ p with Wfc1 := W } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostF1 ε p y (Mat.flatten (fun r => Proofs.dense W p.bfc1
          (Mat.unflatten (Mat.flatten (vitLn2M ε p y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostF1 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bfc1 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec mlpDim)
    (y : Vec (Np1 * (heads * d))) :
    ({ p with bfc1 := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostF1 ε p y (Mat.flatten (fun r => Proofs.dense p.Wfc1 θ
          (Mat.unflatten (Mat.flatten (vitLn2M ε p y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostF1 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wfc2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (W : Mat mlpDim (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with Wfc2 := W } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostF2 ε p y (Mat.flatten (fun r => Proofs.dense W p.bfc2
          (Mat.unflatten (Mat.flatten (fun r' => gelu mlpDim (vitM1M ε p y r'))) r))) := by
  rw [vit_fwd_eq_postF2, Mat.unflatten_flatten]; rfl

theorem vit_fwd_bfc2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (θ : Vec (heads * d))
    (y : Vec (Np1 * (heads * d))) :
    ({ p with bfc2 := θ } : BlockParamsV (heads * d) mlpDim).fwdO ε y
      = vitPostF2 ε p y (Mat.flatten (fun r => Proofs.dense p.Wfc2 θ
          (Mat.unflatten (Mat.flatten (fun r' => gelu mlpDim (vitM1M ε p y r'))) r))) := by
  rw [vit_fwd_eq_postF2, Mat.unflatten_flatten]; rfl


/-! ### Differentiability -/

/-- Each `vitPost*` is differentiable (`0 < ε` where the MLP sublayer's LN₂ is inside). -/
theorem vitPostO_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostO ε p y) := by
  unfold vitPostO; exact (vitMlpSubF_differentiable ε hε p).comp (by fun_prop)

theorem vitWoF_differentiable (p : BlockParamsV (heads * d) mlpDim) :
    Differentiable ℝ (vitWoF (Np1 := Np1) p) :=
  dense_per_token_flat_differentiable p.Wo p.bo

theorem vitPostL1_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostL1 ε p y) :=
  (vitPostO_differentiable ε hε p y).comp
    (mhsaLayer_flat_differentiable Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo)

theorem vitPostQ_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostQ ε p y) :=
  (vitPostO_differentiable ε hε p y).comp ((vitWoF_differentiable p).comp
    (attnCoreQ_flat_differentiable _ _))

theorem vitPostK_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostK ε p y) :=
  (vitPostO_differentiable ε hε p y).comp ((vitWoF_differentiable p).comp
    (attnCoreK_flat_differentiable _ _))

theorem vitPostV_differentiable (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostV ε p y) :=
  (vitPostO_differentiable ε hε p y).comp ((vitWoF_differentiable p).comp
    (attnCoreV_flat_differentiable _ _))

theorem vitPostL2_differentiable (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostL2 ε p y) := by
  have h := transformerMlp_flat_differentiable Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
  unfold vitPostL2; fun_prop

theorem vitPostF1_differentiable (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostF1 ε p y) := by
  have h := flat_differentiable_comp (G := fun Y : Mat Np1 mlpDim => fun n => Proofs.dense p.Wfc2 p.bfc2 (Y n))
    (F := fun Y : Mat Np1 mlpDim => fun n => gelu mlpDim (Y n))
    (gelu_per_token_flat_differentiable Np1 mlpDim) (dense_per_token_flat_differentiable p.Wfc2 p.bfc2)
  unfold vitPostF1; exact (differentiable_const _).add h

theorem vitPostF2_differentiable (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostF2 ε p y) := by
  unfold vitPostF2; fun_prop

end Chain

-- ════════════════════════════════════════════════════════════════
-- § A ViT block, batched — the sixteen nodes against the loss at the block's output
-- ════════════════════════════════════════════════════════════════

/-- **ViT block, every parameter node a loss derivative** — the sixteen nodes `vitBlockTiedGB`
    ties, at the tie's batched activations and cotangents, `Φ` the loss at the block's output as a
    function of the block's record. -/
def vitBlockLossTiedGB (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : BlockParamsV (heads * d) mlpDim) (xin : Vec (N * (Np1 * (heads * d))))
    (Φ : BlockParamsV (heads * d) mlpDim → Vec 1) (dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  -- forward saves, as the tie reads them
  let ln1B : Vec (N * (Np1 * (heads * d))) := batchMap N (fun x => (blkSaves ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 x).ln1) xin
  let attB : Vec (N * (Np1 * (heads * d))) := batchMap N (fun x => (blkSaves ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 x).att) xin
  let hB   : Vec (N * (Np1 * (heads * d))) := batchMap N (fun x => (blkSaves ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 x).h) xin
  let ln2B : Vec (N * (Np1 * (heads * d))) := batchMap N (fun x => (blkSaves ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 x).ln2) xin
  let gB   : Vec (N * (Np1 * mlpDim))      := batchMap N (fun x => (blkSaves ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 x).g) xin
  -- backward chain cotangents
  let cotLn1B : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cLn1 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  let dQB     : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cQ ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  let dKB     : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cK ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  let dVB     : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cV ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  let cotHB   : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cH ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  let cotLn2B : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cLn2 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  let cotM1B  : Vec (N * (Np1 * mlpDim))      := batchMapAux N (cM1 ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2) xin dyOut
  -- LN₁ γ/β
  (HasGradAt (fun θ => Φ { p with γ1 := θ }) p.γ1
        (den (SHlo.veclnGammaGradB (N := N) (R := Np1) (D := heads * d) xN epsStr ε xin
          (.operand cotN cotLn1B))))
  ∧ (HasGradAt (fun θ => Φ { p with β1 := θ }) p.β1
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotLn1B))))
  -- Q / K / V / out-projection W, b
  ∧ (HasGradAt (fun θ => Φ { p with Wq := Mat.unflatten θ }) (Mat.flatten p.Wq)
        (den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := heads * d) (c := heads * d) xN ln1B
          (.operand cotN dQB))))
  ∧ (HasGradAt (fun θ => Φ { p with bq := θ }) p.bq
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dQB))))
  ∧ (HasGradAt (fun θ => Φ { p with Wk := Mat.unflatten θ }) (Mat.flatten p.Wk)
        (den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := heads * d) (c := heads * d) xN ln1B
          (.operand cotN dKB))))
  ∧ (HasGradAt (fun θ => Φ { p with bk := θ }) p.bk
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dKB))))
  ∧ (HasGradAt (fun θ => Φ { p with Wv := Mat.unflatten θ }) (Mat.flatten p.Wv)
        (den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := heads * d) (c := heads * d) xN ln1B
          (.operand cotN dVB))))
  ∧ (HasGradAt (fun θ => Φ { p with bv := θ }) p.bv
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dVB))))
  ∧ (HasGradAt (fun θ => Φ { p with Wo := Mat.unflatten θ }) (Mat.flatten p.Wo)
        (den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := heads * d) (c := heads * d) xN attB
          (.operand cotN cotHB))))
  ∧ (HasGradAt (fun θ => Φ { p with bo := θ }) p.bo
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotHB))))
  -- LN₂ γ/β
  ∧ (HasGradAt (fun θ => Φ { p with γ2 := θ }) p.γ2
        (den (SHlo.veclnGammaGradB (N := N) (R := Np1) (D := heads * d) xN epsStr ε hB
          (.operand cotN cotLn2B))))
  ∧ (HasGradAt (fun θ => Φ { p with β2 := θ }) p.β2
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotLn2B))))
  -- fc1 W/b
  ∧ (HasGradAt (fun θ => Φ { p with Wfc1 := Mat.unflatten θ }) (Mat.flatten p.Wfc1)
        (den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := heads * d) (c := mlpDim) xN ln2B
          (.operand cotN cotM1B))))
  ∧ (HasGradAt (fun θ => Φ { p with bfc1 := θ }) p.bfc1
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := mlpDim) (.operand cotN cotM1B))))
  -- fc2 W/b
  ∧ (HasGradAt (fun θ => Φ { p with Wfc2 := Mat.unflatten θ }) (Mat.flatten p.Wfc2)
        (den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := mlpDim) (c := heads * d) xN gB
          (.operand cotN dyOut))))
  ∧ (HasGradAt (fun θ => Φ { p with bfc2 := θ }) p.bfc2
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dyOut))))

/-- One node of the block bundle: `HasGradAt.param_batchMap_through` at the factoring `hF` of the
    block with one slot varied, restated against `Φ'` (that slot of the loss) and at the node's own
    denotation `hnode`. -/
private theorem vit_node_lossTied {P N D b m : Nat} {Lb : Vec (N * D) → Vec 1}
    {X dY : Vec (N * D)} {Φ' : Vec P → Vec 1} {F : Vec P → Vec D → Vec D}
    (pre : Vec D → Vec b) (per : Vec P → Vec b → Vec m) (post : Vec D → Vec m → Vec D)
    (cot : Vec D → Vec D → Vec m) {θ node : Vec P}
    (hF : ∀ θ' y, F θ' y = post y (per θ' (pre y)))
    (hΦ : ∀ θ', Φ' θ' = Lb (batchMap N (F θ') X))
    (hG : HasGradAt Lb (batchMap N (fun y => post y (per θ (pre y))) X) dY)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ)
    (hpost : ∀ y, Differentiable ℝ (post y))
    (hcot : ∀ y dy, HasGradAt (fun u => linLoss dy (post y u)) (per θ (pre y)) (cot y dy))
    (A : Vec (N * b)) (COT : Vec (N * m))
    (hA : ∀ n, batchSlice N b A n = pre (batchSlice N D X n))
    (hC : ∀ n, batchSlice N m COT n = cot (batchSlice N D X n) (batchSlice N D dY n))
    (hnode : (fun i => ∑ n : Fin N, ∑ j : Fin m,
      pdiv (fun θ' => per θ' (batchSlice N b A n)) θ i j * batchSlice N m COT n j) = node) :
    HasGradAt Φ' θ node :=
  ((HasGradAt.param_batchMap_through pre per post cot X hG hper hpost hcot A COT hA hC).congr_left
    (funext fun θ' => (congrArg (fun f => Lb (batchMap N f X)) (funext fun y => (hF θ' y).symm)).trans (hΦ θ').symm)).of_eq
    hnode

theorem vit_block_lossTiedGB (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) (xin : Vec (N * (Np1 * (heads * d))))
    {Lb : Vec (N * (Np1 * (heads * d))) → Vec 1} {dyOut : Vec (N * (Np1 * (heads * d)))}
    (hLb : HasGradAt Lb (batchMap N (p.fwdO ε) xin) dyOut)
    {Φ : BlockParamsV (heads * d) mlpDim → Vec 1}
    (hΦ : ∀ p', Φ p' = Lb (batchMap N (p'.fwdO ε) xin)) :
    vitBlockLossTiedGB N xN epsStr cotN ε p xin Φ dyOut := by
  have hG : ∀ {f : Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d))},
      (∀ y, p.fwdO ε y = f y) → HasGradAt Lb (batchMap N f xin) dyOut :=
    fun h => hLb.congr_point (congrArg (batchMap N · xin) (funext h))
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- LN₁ γ
    exact vit_node_lossTied (fun y => y) (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β1 (Mat.unflatten x r)))
      (vitPostL1 ε p) _ (vit_fwd_γ1 ε p) (fun _ => hΦ _) (hG (vit_fwd_γ1 ε p p.γ1))
      (fun y => (rowLNVecFlat_gamma_differentiable _ _ ε p.β1 y) _) (vitPostL1_differentiable ε hε p)
      (vitPostL1_hasGradAt ε hε p) xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun k => (vecLNGammaTiedB_holds k).symm)
  · -- LN₁ β
    exact vit_node_lossTied (fun y => y) (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ1 θ (Mat.unflatten x r)))
      (vitPostL1 ε p) _ (vit_fwd_β1 ε p) (fun _ => hΦ _) (hG (vit_fwd_β1 ε p p.β1))
      (fun y => (rowLNVecFlat_beta_differentiable _ _ ε p.γ1 y) _) (vitPostL1_differentiable ε hε p)
      (vitPostL1_hasGradAt ε hε p) xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (vecLNBetaTiedB_holds i).symm)
  · -- Q W
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bq (Mat.unflatten x r)))
      (vitPostQ ε p) _ (fun θ => vit_fwd_Wq ε p (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun y => (vit_fwd_Wq ε p p.Wq y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bq y) _) (vitPostQ_differentiable ε hε p)
      (fun y dy => (vitPostQ_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedB_holds i j).symm)
  · -- Q b
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wq θ (Mat.unflatten x r)))
      (vitPostQ ε p) _ (vit_fwd_bq ε p) (fun _ => hΦ _) (hG (vit_fwd_bq ε p p.bq))
      (fun y => (rowDense_bias_differentiable p.Wq y) _) (vitPostQ_differentiable ε hε p)
      (fun y dy => (vitPostQ_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- K W
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bk (Mat.unflatten x r)))
      (vitPostK ε p) _ (fun θ => vit_fwd_Wk ε p (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun y => (vit_fwd_Wk ε p p.Wk y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bk y) _) (vitPostK_differentiable ε hε p)
      (fun y dy => (vitPostK_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedB_holds i j).symm)
  · -- K b
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wk θ (Mat.unflatten x r)))
      (vitPostK ε p) _ (vit_fwd_bk ε p) (fun _ => hΦ _) (hG (vit_fwd_bk ε p p.bk))
      (fun y => (rowDense_bias_differentiable p.Wk y) _) (vitPostK_differentiable ε hε p)
      (fun y dy => (vitPostK_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- V W
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bv (Mat.unflatten x r)))
      (vitPostV ε p) _ (fun θ => vit_fwd_Wv ε p (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun y => (vit_fwd_Wv ε p p.Wv y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bv y) _) (vitPostV_differentiable ε hε p)
      (fun y dy => (vitPostV_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedB_holds i j).symm)
  · -- V b
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wv θ (Mat.unflatten x r)))
      (vitPostV ε p) _ (vit_fwd_bv ε p) (fun _ => hΦ _) (hG (vit_fwd_bv ε p p.bv))
      (fun y => (rowDense_bias_differentiable p.Wv y) _) (vitPostV_differentiable ε hε p)
      (fun y dy => (vitPostV_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- out-projection W
    exact vit_node_lossTied (fun y => Mat.flatten (vitAttM ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bo (Mat.unflatten x r)))
      (vitPostO ε p) _ (fun θ => vit_fwd_Wo ε p (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun y => (vit_fwd_Wo ε p p.Wo y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bo y) _) (vitPostO_differentiable ε hε p)
      (fun y dy => (vitPostO_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitAttM ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedB_holds i j).symm)
  · -- out-projection b
    exact vit_node_lossTied (fun y => Mat.flatten (vitAttM ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wo θ (Mat.unflatten x r)))
      (vitPostO ε p) _ (vit_fwd_bo ε p) (fun _ => hΦ _) (hG (vit_fwd_bo ε p p.bo))
      (fun y => (rowDense_bias_differentiable p.Wo y) _) (vitPostO_differentiable ε hε p)
      (fun y dy => (vitPostO_hasGradAt ε hε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitAttM ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- LN₂ γ
    exact vit_node_lossTied (fun y => Mat.flatten (vitHM ε p y))
      (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β2 (Mat.unflatten x r))) (vitPostL2 ε p) _ (vit_fwd_γ2 ε p)
      (fun _ => hΦ _) (hG (vit_fwd_γ2 ε p p.γ2))
      (fun y => (rowLNVecFlat_gamma_differentiable _ _ ε p.β2 y) _) (vitPostL2_differentiable ε p)
      (fun y dy => (vitPostL2_hasGradAt ε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitHM ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun k => (vecLNGammaTiedB_holds k).symm)
  · -- LN₂ β
    exact vit_node_lossTied (fun y => Mat.flatten (vitHM ε p y))
      (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ2 θ (Mat.unflatten x r))) (vitPostL2 ε p) _ (vit_fwd_β2 ε p)
      (fun _ => hΦ _) (hG (vit_fwd_β2 ε p p.β2))
      (fun y => (rowLNVecFlat_beta_differentiable _ _ ε p.γ2 y) _) (vitPostL2_differentiable ε p)
      (fun y dy => (vitPostL2_hasGradAt ε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitHM ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (vecLNBetaTiedB_holds i).symm)
  · -- fc1 W
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn2M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) mlpDim) p.bfc1 (Mat.unflatten x r)))
      (vitPostF1 ε p) _ (fun θ => vit_fwd_Wfc1 ε p (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun y => (vit_fwd_Wfc1 ε p p.Wfc1 y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bfc1 y) _) (vitPostF1_differentiable ε p)
      (fun y dy => (vitPostF1_hasGradAt ε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn2M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedB_holds i j).symm)
  · -- fc1 b
    exact vit_node_lossTied (fun y => Mat.flatten (vitLn2M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wfc1 θ (Mat.unflatten x r)))
      (vitPostF1 ε p) _ (vit_fwd_bfc1 ε p) (fun _ => hΦ _) (hG (vit_fwd_bfc1 ε p p.bfc1))
      (fun y => (rowDense_bias_differentiable p.Wfc1 y) _) (vitPostF1_differentiable ε p)
      (fun y dy => (vitPostF1_hasGradAt ε p y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMap N (fun y => Mat.flatten (vitLn2M ε p y)) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- fc2 W
    exact vit_node_lossTied (fun y => Mat.flatten (fun r => gelu mlpDim (vitM1M ε p y r)))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat mlpDim (heads * d)) p.bfc2 (Mat.unflatten x r)))
      (vitPostF2 ε p) _ (fun θ => vit_fwd_Wfc2 ε p (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun y => (vit_fwd_Wfc2 ε p p.Wfc2 y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bfc2 y) _) (vitPostF2_differentiable ε p)
      (fun y dy => hasGradAt_constAdd (G := linLoss dy) _ _ (hasGradAt_linLoss dy _))
      (batchMap N (fun y => Mat.flatten (fun r => gelu mlpDim (vitM1M ε p y r))) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedB_holds i j).symm)
  · -- fc2 b
    exact vit_node_lossTied (fun y => Mat.flatten (fun r => gelu mlpDim (vitM1M ε p y r)))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wfc2 θ (Mat.unflatten x r)))
      (vitPostF2 ε p) _ (vit_fwd_bfc2 ε p) (fun _ => hΦ _) (hG (vit_fwd_bfc2 ε p p.bfc2))
      (fun y => (rowDense_bias_differentiable p.Wfc2 y) _) (vitPostF2_differentiable ε p)
      (fun y dy => hasGradAt_constAdd (G := linLoss dy) _ _ (hasGradAt_linLoss dy _))
      (batchMap N (fun y => Mat.flatten (fun r => gelu mlpDim (vitM1M ε p y r))) xin) _
      (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)
      (funext fun i => (rowDenseBTiedB_holds i).symm)


-- ════════════════════════════════════════════════════════════════
-- § The head — final LN, CLS slice, classifier
-- ════════════════════════════════════════════════════════════════

section Head

/-- The head per example: the final vector LN on every token, then the CLS-slice classifier —
    `vitHeadHasVJP`'s map. -/
noncomputable def vitHeadO {nC : Nat} (ε : ℝ) (γF βF : Vec 192) (Wcls : Mat 192 nC) (bcls : Vec nC) :
    Vec (197 * 192) → Vec nC :=
  classifierFlat 196 192 nC Wcls bcls ∘
    fun v : Vec ((196 + 1) * 192) => Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten v r))

/-- **Head, every parameter node a loss derivative** — the final LN's two nodes
    (`vitFinalLNTiedGB`) and the classifier's two (`vitHeadTiedGB`). -/
def vitHeadLossTiedGB (N : Nat) {nC : Nat} (xN aN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (bcls : Vec nC) (b12out : Vec (N * (197 * 192)))
    (Φ : Vec 192 → Vec 192 → Mat 192 nC → Vec nC → Vec 1) (g : Vec (N * nC)) : Prop :=
  let cotFlB : Vec (N * (197 * 192)) := batchMap N (vitCotFl 196 192 nC Wcls) g
  let hnB : Vec (N * 192) := batchMap N (clsSliceFlat 196 192)
    (batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten b r))) b12out)
  (HasGradAt (fun θ => Φ θ βF Wcls bcls) γF
        (den (SHlo.veclnGammaGradB (N := N) (R := 197) (D := 192) xN epsStr ε b12out
          (.operand cotN cotFlB))))
  ∧ (HasGradAt (fun θ => Φ γF θ Wcls bcls) βF
        (den (SHlo.rowDenseBiasGradB (N := N) (R := 197) (c := 192) (.operand cotN cotFlB))))
  ∧ HasGradAt (fun θ => Φ γF βF (Mat.unflatten θ) bcls) (Mat.flatten Wcls)
      (den (SHlo.weightGradB (N := N) (m := 192) (n := nC) aN hnB (.operand cotN g)))
  ∧ HasGradAt (fun θ => Φ γF βF Wcls θ) bcls
      (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g)))

theorem vit_head_lossTiedGB (N : Nat) {nC : Nat} (xN aN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (bcls : Vec nC) (b12out : Vec (N * (197 * 192)))
    {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)}
    (hL : HasGradAt L (batchMap N (vitHeadO ε γF βF Wcls bcls) b12out) g)
    {Φ : Vec 192 → Vec 192 → Mat 192 nC → Vec nC → Vec 1}
    (hΦ : ∀ a b W bb, Φ a b W bb = L (batchMap N (vitHeadO ε a b W bb) b12out)) :
    vitHeadLossTiedGB N xN aN epsStr cotN ε γF βF Wcls bcls b12out Φ g := by
  rw [show Φ = fun a b W bb => L (batchMap N (vitHeadO ε a b W bb) b12out) from
    funext fun a => funext fun b => funext fun W => funext fun bb => hΦ a b W bb]
  have hc : ∀ (x : Vec (197 * 192)) (dy : Vec nC),
      HasGradAt (fun u => linLoss dy (classifierFlat 196 192 nC Wcls bcls u)) x
        (vitCotFl 196 192 nC Wcls dy) := fun x dy =>
    HasGradAt.comp_global (f := classifierFlat 196 192 nC Wcls bcls) (x := x) (hasGradAt_linLoss dy _)
      (classifierFlat_differentiable 196 192 nC Wcls bcls) (classifierFlatHasVJP 196 192 nC Wcls bcls)
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ x => Mat.flatten (fun r => layerNormVec 192 ε θ βF (Mat.unflatten x r)))
        (fun _ u => classifierFlat 196 192 nC Wcls bcls u) (fun _ dy => vitCotFl 196 192 nC Wcls dy)
        b12out (θ := γF) hL (fun y => (rowLNVecFlat_gamma_differentiable _ _ ε βF y) _)
        (fun _ => classifierFlat_differentiable 196 192 nC Wcls bcls) (fun _ dy => hc _ dy)
        b12out _ (fun _ => rfl) (fun n => batchSlice_batchMap _ _ n)).of_eq
      (funext fun k => ((vecLNGammaTiedB_holds : GradNodeB.VecLNGammaTiedB N 197 xN epsStr cotN ε βF b12out γF
      (batchMap N (vitCotFl 196 192 nC Wcls) g)) k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ x => Mat.flatten (fun r => layerNormVec 192 ε γF θ (Mat.unflatten x r)))
        (fun _ u => classifierFlat 196 192 nC Wcls bcls u) (fun _ dy => vitCotFl 196 192 nC Wcls dy)
        b12out (θ := βF) hL (fun y => (rowLNVecFlat_beta_differentiable _ _ ε γF y) _)
        (fun _ => classifierFlat_differentiable 196 192 nC Wcls bcls) (fun _ dy => hc _ dy)
        b12out _ (fun _ => rfl) (fun n => batchSlice_batchMap _ _ n)).of_eq
      (funext fun i => ((vecLNBetaTiedB_holds : GradNodeB.VecLNBetaTiedB N 197 cotN ε γF b12out βF
      (batchMap N (vitCotFl 196 192 nC Wcls) g)) i).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => clsSliceFlat 196 192 (Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten y r))))
        (fun θ z => Proofs.dense (Mat.unflatten θ) bcls z) (fun _ l => l) (fun _ dy => dy)
        b12out (θ := Mat.flatten Wcls) (by rw [Mat.unflatten_flatten]; exact hL)
        (fun y => (denseWeightMap_differentiable bcls y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        (batchMap N (clsSliceFlat 196 192)
          (batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten b r)))
            b12out)) g
        (fun n => by rw [batchSlice_batchMap, batchSlice_batchMap]) (fun _ => rfl)).of_eq
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (GradNodeB.headWGradB_den aN cotN _ Wcls bcls g i j).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => clsSliceFlat 196 192 (Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten y r))))
        (fun θ z => Proofs.dense Wcls θ z) (fun _ l => l) (fun _ dy => dy) b12out (θ := bcls) hL
        (fun y => (GradNodeB.dense_bias_differentiable Wcls y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => by rw [batchSlice_batchMap, batchSlice_batchMap]) (fun _ => rfl)).of_eq
      (funext fun i => (GradNodeB.headBGradB_den cotN Wcls
        (batchSlice N 192 (batchMap N (clsSliceFlat 196 192)
          (batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten b r)))
            b12out))) bcls g i).symm)

end Head

-- ════════════════════════════════════════════════════════════════
-- § The patch embedding
-- ════════════════════════════════════════════════════════════════

section Embed

/-- The patch embedding is differentiable in its conv weight. -/
theorem patchEmbedFlat_weight_differentiable (bc cls : Vec 192) (pos : Mat 197 192)
    (x : Vec (3 * 224 * 224)) :
    Differentiable ℝ (fun θ : Vec (192 * 3 * 16 * 16) =>
      patchEmbedFlat 3 224 224 16 196 192 (Kernel4.unflatten θ) bc cls pos x) := by
  rw [differentiable_pi]; intro idx
  unfold patchEmbedFlat Kernel4.unflatten
  by_cases hn : (finProdFinEquiv.symm idx).1.val = 0 <;> simp only [hn, ite_true, ite_false] <;> fun_prop

/-- …in its conv bias. -/
theorem patchEmbedFlat_bias_differentiable (Wc : Kernel4 192 3 16 16) (cls : Vec 192)
    (pos : Mat 197 192) (x : Vec (3 * 224 * 224)) :
    Differentiable ℝ (fun θ : Vec 192 => patchEmbedFlat 3 224 224 16 196 192 Wc θ cls pos x) := by
  rw [differentiable_pi]; intro idx
  unfold patchEmbedFlat
  by_cases hn : (finProdFinEquiv.symm idx).1.val = 0 <;> simp only [hn, ite_true, ite_false] <;> fun_prop

/-- …in the CLS token. -/
theorem patchEmbedFlat_cls_differentiable (Wc : Kernel4 192 3 16 16) (bc : Vec 192)
    (pos : Mat 197 192) (x : Vec (3 * 224 * 224)) :
    Differentiable ℝ (fun θ : Vec 192 => patchEmbedFlat 3 224 224 16 196 192 Wc bc θ pos x) := by
  rw [differentiable_pi]; intro idx
  unfold patchEmbedFlat
  by_cases hn : (finProdFinEquiv.symm idx).1.val = 0 <;> simp only [hn, ite_true, ite_false] <;> fun_prop

/-- …in the position embedding. -/
theorem patchEmbedFlat_pos_differentiable (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192)
    (x : Vec (3 * 224 * 224)) :
    Differentiable ℝ (fun θ : Vec (197 * 192) =>
      patchEmbedFlat 3 224 224 16 196 192 Wc bc cls (Mat.unflatten θ) x) := by
  rw [differentiable_pi]; intro idx
  unfold patchEmbedFlat Mat.unflatten
  by_cases hn : (finProdFinEquiv.symm idx).1.val = 0 <;> simp only [hn, ite_true, ite_false] <;> fun_prop

/-- **Patch embedding, every parameter node a loss derivative** — the four nodes `vitEmbedTiedGB`
    ties (the CLS token's with the batch sum inside `den`). -/
def vitEmbedLossTiedGB (N : Nat) (xN cotN : String) (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192)
    (pos : Mat 197 192) (img : Vec (N * (3 * 224 * 224)))
    (Φ : Kernel4 192 3 16 16 → Vec 192 → Vec 192 → Mat 197 192 → Vec 1)
    (dyEmbed : Vec (N * (197 * 192))) : Prop :=
  HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bc cls pos) (Kernel4.flatten Wc)
      (den (SHlo.patchEmbedWeightGradB (N := N) (ic := 3) (H := 224) (W := 224) (P := 16)
        (tk := 196) (D := 192) xN img (.operand cotN dyEmbed)))
  ∧ HasGradAt (fun θ => Φ Wc θ cls pos) bc
      (den (SHlo.patchEmbedBiasGradB (N := N) (tk := 196) (c := 192) (.operand cotN dyEmbed)))
  ∧ HasGradAt (fun θ => Φ Wc bc θ pos) cls
      (den (SHlo.denseBiasGradB (N := N) (c := 192)
        (.operand cotN (batchMap N (clsSliceFlat 196 192) dyEmbed))))
  ∧ HasGradAt (fun θ => Φ Wc bc cls (Mat.unflatten θ)) (Mat.flatten pos)
      (den (SHlo.posEmbedGradB (N := N) (tk := 196) (D := 192) (.operand cotN dyEmbed)))

theorem vit_embed_lossTiedGB (N : Nat) (xN cotN : String) (Wc : Kernel4 192 3 16 16)
    (bc cls : Vec 192) (pos : Mat 197 192) (img : Vec (N * (3 * 224 * 224)))
    {Lb : Vec (N * (197 * 192)) → Vec 1} {dyEmbed : Vec (N * (197 * 192))}
    (hLb : HasGradAt Lb (batchMap N (patchEmbedFlat 3 224 224 16 196 192 Wc bc cls pos) img) dyEmbed)
    {Φ : Kernel4 192 3 16 16 → Vec 192 → Vec 192 → Mat 197 192 → Vec 1}
    (hΦ : ∀ W b c q, Φ W b c q = Lb (batchMap N (patchEmbedFlat 3 224 224 16 196 192 W b c q) img)) :
    vitEmbedLossTiedGB N xN cotN Wc bc cls pos img Φ dyEmbed := by
  rw [show Φ = fun W b c q => Lb (batchMap N (patchEmbedFlat 3 224 224 16 196 192 W b c q) img) from
    funext fun W => funext fun b => funext fun c => funext fun q => hΦ W b c q]
  refine ⟨?_, ?_, ?_, ?_⟩
  · refine (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => patchEmbedFlat 3 224 224 16 196 192 (Kernel4.unflatten θ) bc cls pos y)
        (fun _ z => z) (fun _ dy => dy) img (θ := Kernel4.flatten Wc)
        (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (patchEmbedFlat_weight_differentiable bc cls pos y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        img dyEmbed (fun _ => rfl) (fun _ => rfl)).of_eq (funext fun idx => ?_)
    obtain ⟨⟨a, kw⟩, rfl⟩ := finProdFinEquiv.surjective idx
    obtain ⟨⟨b, kh⟩, rfl⟩ := finProdFinEquiv.surjective a
    obtain ⟨⟨dd, c⟩, rfl⟩ := finProdFinEquiv.surjective b
    exact (ViTFoldGB.patchEmbedWeightGradB_den xN cotN bc cls pos img Wc dyEmbed dd c kh kw).symm
  · refine (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => patchEmbedFlat 3 224 224 16 196 192 Wc θ cls pos y)
        (fun _ z => z) (fun _ dy => dy) img (θ := bc) hLb
        (fun y => (patchEmbedFlat_bias_differentiable Wc cls pos y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        img dyEmbed (fun _ => rfl) (fun _ => rfl)).of_eq
      (funext fun i => (ViTFoldGB.patchEmbedBiasGradB_den cotN Wc bc cls pos img dyEmbed i).symm)
  · refine (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => patchEmbedFlat 3 224 224 16 196 192 Wc bc θ pos y)
        (fun _ z => z) (fun _ dy => dy) img (θ := cls) hLb
        (fun y => (patchEmbedFlat_cls_differentiable Wc bc pos y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        img dyEmbed (fun _ => rfl) (fun _ => rfl)).of_eq
      (funext fun i => (ViTFoldGB.clsGrad_denB cotN Wc bc cls pos img dyEmbed i).symm)
  · refine (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => patchEmbedFlat 3 224 224 16 196 192 Wc bc cls (Mat.unflatten θ) y)
        (fun _ z => z) (fun _ dy => dy) img (θ := Mat.flatten pos)
        (by rw [Mat.unflatten_flatten]; exact hLb)
        (fun y => (patchEmbedFlat_pos_differentiable Wc bc cls y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        img dyEmbed (fun _ => rfl) (fun _ => rfl)).of_eq
      (funext fun i => (ViTFoldGB.posEmbedGradB_den cotN Wc bc cls pos img dyEmbed i).symm)

end Embed


-- ════════════════════════════════════════════════════════════════
-- § The whole net: the prefix before each stage, the loss after it, the net with one stage varied
-- ════════════════════════════════════════════════════════════════

section Net

/-- Pull the loss gradient back through a batched block's certified VJP: the cotangent is the tie's
    `batchMapAux N (p.cotIn ε)` (`vitBlockCotInB_eq_vjp`). -/
theorem vitBlkB_hasGradAt_comp (N : Nat) {Np1 heads d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) (X : Vec (N * (Np1 * (heads * d))))
    {G : Vec (N * (Np1 * (heads * d))) → Vec 1} {dY : Vec (N * (Np1 * (heads * d)))}
    (hG : HasGradAt G (batchMap N (p.fwdO ε) X) dY) :
    HasGradAt (fun y => G (batchMap N (p.fwdO ε) y)) X (batchMapAux N (p.cotIn ε) X dY) := by
  have hf : p.fwdO (Np1 := Np1) ε = fun v => Mat.flatten (transformerBlockV Np1 heads d mlpDim ε
      p.γ1 p.β1 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
      (Mat.unflatten v)) :=
    funext fun v => congrArg Mat.flatten (vitBlockSpelledMHV_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)
  rw [hf] at hG ⊢
  exact (HasGradAt.comp (x := X) hG
    ((batchMap_differentiable _ (transformerBlockV_flat_differentiable Np1 heads d mlpDim ε p.γ1 p.β1 hε
      p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)) X)
    (batchMapHasVJPAt _ X
      (fun _ => (HasVJPMat.toHasVJP (transformerBlockVHasVJPMat Np1 heads d mlpDim ε
        p.γ1 p.β1 hε p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2
        p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)).toHasVJPAt _)
      (fun _ => (transformerBlockV_flat_differentiable Np1 heads d mlpDim ε p.γ1 p.β1 hε
        p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) _))).of_eq
    (congrFun (vitBlockCotInB_eq_vjp N ε hε p X) dY).symm

/-- …and through the batched head (`vitCotTowerOutB_eq_vjp`). -/
theorem vitHeadB_hasGradAt_comp (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε) (γF βF : Vec 192)
    (Wcls : Mat 192 nC) (bcls : Vec nC) (X : Vec (N * (197 * 192))) {L : Vec (N * nC) → Vec 1}
    {g : Vec (N * nC)} (hL : HasGradAt L (batchMap N (vitHeadO ε γF βF Wcls bcls) X) g) :
    HasGradAt (fun y => L (batchMap N (vitHeadO ε γF βF Wcls bcls) y)) X
      (batchMapAux N (vitCotTowerOutV 196 192 nC ε γF Wcls) X g) :=
  (HasGradAt.comp (x := X) hL
    ((batchMap_differentiable _ ((classifierFlat_differentiable 196 192 nC Wcls bcls).comp
      (layerNormVec_per_token_flat_differentiable (196 + 1) 192 ε γF βF hε))) X)
    (batchMapHasVJPAt _ X (fun _ => (vitHeadHasVJP 196 192 nC ε hε γF βF Wcls bcls).toHasVJPAt _)
      (fun _ => ((classifierFlat_differentiable 196 192 nC Wcls bcls).comp
        (layerNormVec_per_token_flat_differentiable (196 + 1) 192 ε γF βF hε)) _))).of_eq
    (congrFun (vitCotTowerOutB_eq_vjp N ε hε γF βF Wcls bcls X) g).symm

/-- **ViT-Tiny, batched**: the tie's forward, stage by stage — `batchMap N` of the patch
    embedding, each block, then of the head. It is `batchMap N` of `vitForwardKV` at the twelve
    blocks (`vitNetB_eq_vitForwardKV`). -/
noncomputable def vitNetB (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    Vec (N * nC) :=
  batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls)
    (batchMap N (w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε)
    (batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img)))))))))))))

/-- The patch embedding's output — block `b1`'s input (the tie's `ib1`). -/
noncomputable def vitPreE (N : Nat) {nC : Nat} (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos)

/-- Block `b1`'s output. -/
noncomputable def vitPreB1 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreE N w

/-- Block `b2`'s output. -/
noncomputable def vitPreB2 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB1 N ε w

/-- Block `b3`'s output. -/
noncomputable def vitPreB3 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB2 N ε w

/-- Block `b4`'s output. -/
noncomputable def vitPreB4 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB3 N ε w

/-- Block `b5`'s output. -/
noncomputable def vitPreB5 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB4 N ε w

/-- Block `b6`'s output. -/
noncomputable def vitPreB6 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB5 N ε w

/-- Block `b7`'s output. -/
noncomputable def vitPreB7 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB6 N ε w

/-- Block `b8`'s output. -/
noncomputable def vitPreB8 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB7 N ε w

/-- Block `b9`'s output. -/
noncomputable def vitPreB9 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB8 N ε w

/-- Block `b10`'s output. -/
noncomputable def vitPreB10 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB9 N ε w

/-- Block `b11`'s output. -/
noncomputable def vitPreB11 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB10 N ε w

/-- Block `b12`'s output. -/
noncomputable def vitPreB12 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ∘ vitPreB11 N ε w

theorem vitPreE_apply (N : Nat) {nC : Nat} (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreE N w img = batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img := by
  rw [vitPreE]

theorem vitPreB1_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB1 N ε w img = batchMap N (w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreE N w img) := by
  rw [vitPreB1, Function.comp_apply]

theorem vitPreB2_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB2 N ε w img = batchMap N (w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB1 N ε w img) := by
  rw [vitPreB2, Function.comp_apply]

theorem vitPreB3_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB3 N ε w img = batchMap N (w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB2 N ε w img) := by
  rw [vitPreB3, Function.comp_apply]

theorem vitPreB4_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB4 N ε w img = batchMap N (w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB3 N ε w img) := by
  rw [vitPreB4, Function.comp_apply]

theorem vitPreB5_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB5 N ε w img = batchMap N (w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB4 N ε w img) := by
  rw [vitPreB5, Function.comp_apply]

theorem vitPreB6_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB6 N ε w img = batchMap N (w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB5 N ε w img) := by
  rw [vitPreB6, Function.comp_apply]

theorem vitPreB7_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB7 N ε w img = batchMap N (w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB6 N ε w img) := by
  rw [vitPreB7, Function.comp_apply]

theorem vitPreB8_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB8 N ε w img = batchMap N (w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB7 N ε w img) := by
  rw [vitPreB8, Function.comp_apply]

theorem vitPreB9_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB9 N ε w img = batchMap N (w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB8 N ε w img) := by
  rw [vitPreB9, Function.comp_apply]

theorem vitPreB10_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB10 N ε w img = batchMap N (w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB9 N ε w img) := by
  rw [vitPreB10, Function.comp_apply]

theorem vitPreB11_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB11 N ε w img = batchMap N (w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB10 N ε w img) := by
  rw [vitPreB11, Function.comp_apply]

theorem vitPreB12_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB12 N ε w img = batchMap N (w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB11 N ε w img) := by
  rw [vitPreB12, Function.comp_apply]

/-- The net after block `b12` — the head. -/
noncomputable def vitSufB12 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls)

/-- The net after block `b11`: block `b12`, then the rest. -/
noncomputable def vitSufB11 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB12 N ε w (batchMap N (w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b10`: block `b11`, then the rest. -/
noncomputable def vitSufB10 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB11 N ε w (batchMap N (w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b9`: block `b10`, then the rest. -/
noncomputable def vitSufB9 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB10 N ε w (batchMap N (w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b8`: block `b9`, then the rest. -/
noncomputable def vitSufB8 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB9 N ε w (batchMap N (w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b7`: block `b8`, then the rest. -/
noncomputable def vitSufB7 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB8 N ε w (batchMap N (w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b6`: block `b7`, then the rest. -/
noncomputable def vitSufB6 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB7 N ε w (batchMap N (w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b5`: block `b6`, then the rest. -/
noncomputable def vitSufB5 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB6 N ε w (batchMap N (w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b4`: block `b5`, then the rest. -/
noncomputable def vitSufB4 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB5 N ε w (batchMap N (w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b3`: block `b4`, then the rest. -/
noncomputable def vitSufB3 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB4 N ε w (batchMap N (w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b2`: block `b3`, then the rest. -/
noncomputable def vitSufB2 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB3 N ε w (batchMap N (w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after block `b1`: block `b2`, then the rest. -/
noncomputable def vitSufB1 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB2 N ε w (batchMap N (w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- The net after the patch embedding: block `b1`, then the rest. -/
noncomputable def vitSufE (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB1 N ε w (batchMap N (w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) y)

/-- **The net with the patch embedding varied** is the suffix after it at the varied embedding. -/
theorem vit_factor_embed (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 192 3 16 16) (b c : Vec 192) (q : Mat 197 192) :
    vitNetB N ε { w with Wc := W, bc := b, cls := c, pos := q } img
      = vitSufE N ε w (batchMap N (patchEmbedFlat 3 224 224 16 196 192 W b c q) img) := rfl

/-- **The net with block `b1`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b1 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b1 := p } img
      = vitSufB1 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreE N w img)) := by
  rw [vitPreE_apply]; rfl

/-- **The net with block `b2`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b2 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b2 := p } img
      = vitSufB2 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB1 N ε w img)) := by
  rw [vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b3`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b3 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b3 := p } img
      = vitSufB3 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB2 N ε w img)) := by
  rw [vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b4`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b4 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b4 := p } img
      = vitSufB4 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB3 N ε w img)) := by
  rw [vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b5`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b5 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b5 := p } img
      = vitSufB5 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB4 N ε w img)) := by
  rw [vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b6`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b6 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b6 := p } img
      = vitSufB6 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB5 N ε w img)) := by
  rw [vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b7`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b7 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b7 := p } img
      = vitSufB7 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB6 N ε w img)) := by
  rw [vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b8`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b8 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b8 := p } img
      = vitSufB8 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB7 N ε w img)) := by
  rw [vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b9`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b9 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b9 := p } img
      = vitSufB9 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB8 N ε w img)) := by
  rw [vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b10`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b10 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b10 := p } img
      = vitSufB10 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB9 N ε w img)) := by
  rw [vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b11`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b11 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b11 := p } img
      = vitSufB11 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB10 N ε w img)) := by
  rw [vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b12`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b12 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB N ε { w with b12 := p } img
      = vitSufB12 N ε w (batchMap N (p.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB11 N ε w img)) := by
  rw [vitPreB11_apply, vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with the head varied** is the head at the varied parameters. -/
theorem vit_factor_head (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224)))
    (a b : Vec 192) (W : Mat 192 nC) (bb : Vec nC) :
    vitNetB N ε { w with γF := a, βF := b, Wcls := W, bcls := bb } img
      = batchMap N (vitHeadO ε a b W bb) (vitPreB12 N ε w img) := by
  rw [vitPreB12_apply, vitPreB11_apply, vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- The net's output is the head at block `b12`'s output. -/
theorem vit_forward_eq_head (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitNetB N ε w img = batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls) (vitPreB12 N ε w img) := by
  rw [vitPreB12_apply, vitPreB11_apply, vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The logits the tie's loss cotangent reads are `vitNetB`'s.** The tie spells the head as
    three batched ops (`batchMap_comp`). -/
theorem vit_logitsB_eq (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    batchMap N (Proofs.dense w.Wcls w.bcls) (batchMap N (clsSliceFlat 196 192)
      (batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b r)))
        (vitPreB12 N ε w img))) = vitNetB N ε w img := by
  rw [vit_forward_eq_head, vitHeadO, classifierFlat, batchMap_comp, batchMap_comp]; rfl

/-- A block's forward is the depth-`k` fold's flat block: the spelled multi-head block
    (`vitBlockFwdOMHV`) is `blockV` (`vitBlockSpelledMHV_eq`). -/
theorem fwdO_eq_blockVFlat {Np1 heads d mlpDim : Nat} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    p.fwdO (Np1 := Np1) ε = blockVFlat Np1 heads d mlpDim ε p :=
  funext fun _ => congrArg Mat.flatten (vitBlockSpelledMHV_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)

/-- **`vitNetB` is the canonical ViT-Tiny forward, batched**: `vitForwardKV` at the twelve blocks
    `![w.b1, …, w.b12]`, the forward whose graph `vitFwdGraphKMHV_faithful` ties to the render and
    whose VJP is `vitForwardKVHasVJP`, applied per example. Per example it unfolds the depth-12 fold
    block by block (`fwdO_eq_blockVFlat`); `batchMap_comp` splits the batched composite into the
    capstone's stage-by-stage chain. -/
theorem vitNetB_eq_vitForwardKV (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC)
    (img : Vec (N * (3 * 224 * 224))) :
    vitNetB N ε w img = batchMap N (vitForwardKV 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls
      w.pos ε ![w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7, w.b8, w.b9, w.b10, w.b11, w.b12]
      w.γF w.βF w.Wcls w.bcls) img := by
  have hper : ∀ y, vitForwardKV 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls w.pos ε
      ![w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7, w.b8, w.b9, w.b10, w.b11, w.b12]
      w.γF w.βF w.Wcls w.bcls y
      = (vitHeadO ε w.γF w.βF w.Wcls w.bcls ∘ w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) y := by
    intro y
    simp only [vitForwardKV, vitBodyKVFlat, fwdO_eq_blockVFlat, Function.comp_apply, vitHeadO,
      Matrix.cons_val_zero, Matrix.cons_val_succ]
  rw [show vitForwardKV 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls w.pos ε
      ![w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7, w.b8, w.b9, w.b10, w.b11, w.b12]
      w.γF w.βF w.Wcls w.bcls = _ from funext hper, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp]
  unfold vitNetB
  simp only [Function.comp_apply]

/-- **Every ViT-Tiny parameter gradient node is the derivative of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 200 nodes
    `vit_net_tiedGB` ties, each at the cotangent the tie threads to it from `g`, stated against
    `L` of `vitNetB` with that one parameter varied. -/
def ViTNetLossTiedGB (xN aN epsStr cotN : String) (N : Nat) {nC : Nat} (ε : ℝ)
    (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) (L : Vec (N * nC) → Vec 1) (g : Vec (N * nC)) : Prop :=
  let dy12 := batchMapAux N (vitCotTowerOutV 196 192 nC ε w.γF w.Wcls) (vitPreB12 N ε w img) g
  let dy11 := batchMapAux N (w.b12.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB11 N ε w img) dy12
  let dy10 := batchMapAux N (w.b11.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB10 N ε w img) dy11
  let dy9 := batchMapAux N (w.b10.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB9 N ε w img) dy10
  let dy8 := batchMapAux N (w.b9.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB8 N ε w img) dy9
  let dy7 := batchMapAux N (w.b8.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB7 N ε w img) dy8
  let dy6 := batchMapAux N (w.b7.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB6 N ε w img) dy7
  let dy5 := batchMapAux N (w.b6.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB5 N ε w img) dy6
  let dy4 := batchMapAux N (w.b5.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB4 N ε w img) dy5
  let dy3 := batchMapAux N (w.b4.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB3 N ε w img) dy4
  let dy2 := batchMapAux N (w.b3.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB2 N ε w img) dy3
  let dy1 := batchMapAux N (w.b2.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreB1 N ε w img) dy2
  let dyEmbed := batchMapAux N (w.b1.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) (vitPreE N w img) dy1
  vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b1 (vitPreE N w img)
      (fun p => L (vitNetB N ε { w with b1 := p } img)) dy1
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b2 (vitPreB1 N ε w img)
      (fun p => L (vitNetB N ε { w with b2 := p } img)) dy2
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b3 (vitPreB2 N ε w img)
      (fun p => L (vitNetB N ε { w with b3 := p } img)) dy3
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b4 (vitPreB3 N ε w img)
      (fun p => L (vitNetB N ε { w with b4 := p } img)) dy4
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b5 (vitPreB4 N ε w img)
      (fun p => L (vitNetB N ε { w with b5 := p } img)) dy5
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b6 (vitPreB5 N ε w img)
      (fun p => L (vitNetB N ε { w with b6 := p } img)) dy6
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b7 (vitPreB6 N ε w img)
      (fun p => L (vitNetB N ε { w with b7 := p } img)) dy7
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b8 (vitPreB7 N ε w img)
      (fun p => L (vitNetB N ε { w with b8 := p } img)) dy8
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b9 (vitPreB8 N ε w img)
      (fun p => L (vitNetB N ε { w with b9 := p } img)) dy9
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b10 (vitPreB9 N ε w img)
      (fun p => L (vitNetB N ε { w with b10 := p } img)) dy10
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b11 (vitPreB10 N ε w img)
      (fun p => L (vitNetB N ε { w with b11 := p } img)) dy11
  ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b12 (vitPreB11 N ε w img)
      (fun p => L (vitNetB N ε { w with b12 := p } img)) dy12
  ∧ vitHeadLossTiedGB N xN aN epsStr cotN ε w.γF w.βF w.Wcls w.bcls (vitPreB12 N ε w img)
      (fun a b W bb => L (vitNetB N ε { w with γF := a, βF := b, Wcls := W, bcls := bb } img)) g
  ∧ vitEmbedLossTiedGB N xN cotN w.Wc w.bc w.cls w.pos img
      (fun W b c q => L (vitNetB N ε { w with Wc := W, bc := b, cls := c, pos := q } img)) dyEmbed

/-- **Every ViT-Tiny parameter gradient node is the derivative of the loss in that parameter.**
    For any loss `L` of the logits with gradient `g` at the net's output, each of the 200 nodes
    `vit_net_tiedGB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net, `vitNetB` with
    that one parameter varied (a patch-embedding field, a block's record `w.bk := p` with one slot
    changed, or a head field).

    Hypothesis: `0 < ε`, the LayerNorms' (the tie itself needs none). The loss enters only through
    `hL`; `vit_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship. -/
theorem vit_net_lossGrad (xN aN epsStr cotN : String) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)}
    (hL : HasGradAt L (vitNetB N ε w img) g) :
    ViTNetLossTiedGB xN aN epsStr cotN N ε w img L g := by
  unfold ViTNetLossTiedGB
  intro dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  have hL' : HasGradAt L (batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls) (vitPreB12 N ε w img)) g :=
    hL.congr_point (vit_forward_eq_head N ε w img)
  have hB12 : HasGradAt (fun y => L (vitSufB12 N ε w y)) (vitPreB12 N ε w img) dy12 :=
    vitHeadB_hasGradAt_comp N ε hε w.γF w.βF w.Wcls w.bcls _ hL'
  have hB11 : HasGradAt (fun y => L (vitSufB11 N ε w y)) (vitPreB11 N ε w img) dy11 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b12 _ (hB12.congr_point (vitPreB12_apply N ε w img))
  have hB10 : HasGradAt (fun y => L (vitSufB10 N ε w y)) (vitPreB10 N ε w img) dy10 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b11 _ (hB11.congr_point (vitPreB11_apply N ε w img))
  have hB9 : HasGradAt (fun y => L (vitSufB9 N ε w y)) (vitPreB9 N ε w img) dy9 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b10 _ (hB10.congr_point (vitPreB10_apply N ε w img))
  have hB8 : HasGradAt (fun y => L (vitSufB8 N ε w y)) (vitPreB8 N ε w img) dy8 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b9 _ (hB9.congr_point (vitPreB9_apply N ε w img))
  have hB7 : HasGradAt (fun y => L (vitSufB7 N ε w y)) (vitPreB7 N ε w img) dy7 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b8 _ (hB8.congr_point (vitPreB8_apply N ε w img))
  have hB6 : HasGradAt (fun y => L (vitSufB6 N ε w y)) (vitPreB6 N ε w img) dy6 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b7 _ (hB7.congr_point (vitPreB7_apply N ε w img))
  have hB5 : HasGradAt (fun y => L (vitSufB5 N ε w y)) (vitPreB5 N ε w img) dy5 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b6 _ (hB6.congr_point (vitPreB6_apply N ε w img))
  have hB4 : HasGradAt (fun y => L (vitSufB4 N ε w y)) (vitPreB4 N ε w img) dy4 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b5 _ (hB5.congr_point (vitPreB5_apply N ε w img))
  have hB3 : HasGradAt (fun y => L (vitSufB3 N ε w y)) (vitPreB3 N ε w img) dy3 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b4 _ (hB4.congr_point (vitPreB4_apply N ε w img))
  have hB2 : HasGradAt (fun y => L (vitSufB2 N ε w y)) (vitPreB2 N ε w img) dy2 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b3 _ (hB3.congr_point (vitPreB3_apply N ε w img))
  have hB1 : HasGradAt (fun y => L (vitSufB1 N ε w y)) (vitPreB1 N ε w img) dy1 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b2 _ (hB2.congr_point (vitPreB2_apply N ε w img))
  have hE : HasGradAt (fun y => L (vitSufE N ε w y)) (vitPreE N w img) dyEmbed :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b1 _ (hB1.congr_point (vitPreB1_apply N ε w img))
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b1 _
      (hB1.congr_point (vitPreB1_apply N ε w img)) (fun p => by rw [vit_factor_b1]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b2 _
      (hB2.congr_point (vitPreB2_apply N ε w img)) (fun p => by rw [vit_factor_b2]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b3 _
      (hB3.congr_point (vitPreB3_apply N ε w img)) (fun p => by rw [vit_factor_b3]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b4 _
      (hB4.congr_point (vitPreB4_apply N ε w img)) (fun p => by rw [vit_factor_b4]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b5 _
      (hB5.congr_point (vitPreB5_apply N ε w img)) (fun p => by rw [vit_factor_b5]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b6 _
      (hB6.congr_point (vitPreB6_apply N ε w img)) (fun p => by rw [vit_factor_b6]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b7 _
      (hB7.congr_point (vitPreB7_apply N ε w img)) (fun p => by rw [vit_factor_b7]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b8 _
      (hB8.congr_point (vitPreB8_apply N ε w img)) (fun p => by rw [vit_factor_b8]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b9 _
      (hB9.congr_point (vitPreB9_apply N ε w img)) (fun p => by rw [vit_factor_b9]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b10 _
      (hB10.congr_point (vitPreB10_apply N ε w img)) (fun p => by rw [vit_factor_b10]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b11 _
      (hB11.congr_point (vitPreB11_apply N ε w img)) (fun p => by rw [vit_factor_b11]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b12 _
      (hB12.congr_point (vitPreB12_apply N ε w img)) (fun p => by rw [vit_factor_b12]), ?_⟩
  refine ⟨vit_head_lossTiedGB N xN aN epsStr cotN ε w.γF w.βF w.Wcls w.bcls _ hL'
    (fun a b W bb => by rw [vit_factor_head]), ?_⟩
  exact vit_embed_lossTiedGB N xN cotN w.Wc w.bc w.cls w.pos img
    (hE.congr_point (vitPreE_apply N w img)) (fun W b c q => by rw [vit_factor_embed])

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLossDiv`, `g` the `softmaxDiv` cotangent the render emits — the
    tie's own `g`, whose logits are `vitNetB N ε w img` (`vit_logitsB_eq`). -/
theorem vit_net_lossGrad_smoothedCE (xN aN epsStr cotN aStr negAK bStr logN ohN : String)
    (N : Nat) {nC : Nat} (hK : 0 < nC) (ε α B : ℝ) (hε : 0 < ε) (w : ViTTieWeights nC)
    (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC)) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    ViTNetLossTiedGB xN aN epsStr cotN N ε w img (smoothedBatchLossDiv N nC α B t)
      (den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN (vitNetB N ε w img) t)) :=
  vit_net_lossGrad xN aN epsStr cotN N ε hε w img
    ⟨(smoothedBatchLossDiv_differentiable N nC α B t) _,
      fun J => smoothedBatchLossDiv_grad N nC hK α B aStr negAK bStr logN ohN t _ ht J⟩

/-- **The emitted ViT-Tiny step's gradient nodes ARE the loss's gradient, at one chain.** For each
    of the 200 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`vit_net_tiedGB`), and the batched smoothed loss of `vitNetB` with that one slot varied is
    differentiable there with the node as its gradient (`vit_net_lossGrad_smoothedCE`). The tie's
    final-LN and classifier conjuncts pair with the one head conjunct of the loss side. The tie
    spells each block input as its own let; the proof rewrites the loss side's `vitPre*` into those
    lets (`vitPreE_apply`, …) and the loss side's logits into the tie's (`vit_logitsB_eq`). -/
theorem vit_net_tied_lossGrad (N : Nat) {nC : Nat}
    (xN aN epsStr cotN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    (w : ViTTieWeights nC)
    (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC))
    (hK : 0 < nC) (hε : 0 < ε) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    let ib1    : Vec (N * (197 * 192)) := batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img
    let ib2    : Vec (N * (197 * 192)) := batchMap N (w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib1
    let ib3    : Vec (N * (197 * 192)) := batchMap N (w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib2
    let ib4    : Vec (N * (197 * 192)) := batchMap N (w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib3
    let ib5    : Vec (N * (197 * 192)) := batchMap N (w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib4
    let ib6    : Vec (N * (197 * 192)) := batchMap N (w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib5
    let ib7    : Vec (N * (197 * 192)) := batchMap N (w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib6
    let ib8    : Vec (N * (197 * 192)) := batchMap N (w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib7
    let ib9    : Vec (N * (197 * 192)) := batchMap N (w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib8
    let ib10   : Vec (N * (197 * 192)) := batchMap N (w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib9
    let ib11   : Vec (N * (197 * 192)) := batchMap N (w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib10
    let ib12   : Vec (N * (197 * 192)) := batchMap N (w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib11
    let b12out : Vec (N * (197 * 192)) := batchMap N (w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib12
    -- final LN → CLS row → dense head, then the SMOOTHED loss cotangent at a general target `t`
    let flB     : Vec (N * (197 * 192)) :=
      batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b r))) b12out
    let hnB     : Vec (N * 192) := batchMap N (clsSliceFlat 196 192) flB
    let logitsB : Vec (N * nC)  := batchMap N (dense w.Wcls w.bcls) hnB
    let g       : Vec (N * nC)  :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    let dy12    : Vec (N * (197 * 192)) := batchMapAux N (vitCotTowerOutV 196 192 nC ε w.γF w.Wcls) b12out g
    let dy11   : Vec (N * (197 * 192)) := batchMapAux N (w.b12.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib12 dy12
    let dy10   : Vec (N * (197 * 192)) := batchMapAux N (w.b11.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib11 dy11
    let dy9    : Vec (N * (197 * 192)) := batchMapAux N (w.b10.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib10 dy10
    let dy8    : Vec (N * (197 * 192)) := batchMapAux N (w.b9.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib9 dy9
    let dy7    : Vec (N * (197 * 192)) := batchMapAux N (w.b8.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib8 dy8
    let dy6    : Vec (N * (197 * 192)) := batchMapAux N (w.b7.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib7 dy7
    let dy5    : Vec (N * (197 * 192)) := batchMapAux N (w.b6.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib6 dy6
    let dy4    : Vec (N * (197 * 192)) := batchMapAux N (w.b5.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib5 dy5
    let dy3    : Vec (N * (197 * 192)) := batchMapAux N (w.b4.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib4 dy4
    let dy2    : Vec (N * (197 * 192)) := batchMapAux N (w.b3.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib3 dy3
    let dy1    : Vec (N * (197 * 192)) := batchMapAux N (w.b2.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib2 dy2
    let dyEmbed: Vec (N * (197 * 192)) := batchMapAux N (w.b1.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib1 dy1
    let L := smoothedBatchLossDiv N nC α B t
    (w.b1.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib1 dy1
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b1 ib1
        (fun p => L (vitNetB N ε { w with b1 := p } img)) dy1)
  ∧ (w.b2.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib2 dy2
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b2 ib2
        (fun p => L (vitNetB N ε { w with b2 := p } img)) dy2)
  ∧ (w.b3.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib3 dy3
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b3 ib3
        (fun p => L (vitNetB N ε { w with b3 := p } img)) dy3)
  ∧ (w.b4.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib4 dy4
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b4 ib4
        (fun p => L (vitNetB N ε { w with b4 := p } img)) dy4)
  ∧ (w.b5.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib5 dy5
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b5 ib5
        (fun p => L (vitNetB N ε { w with b5 := p } img)) dy5)
  ∧ (w.b6.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib6 dy6
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b6 ib6
        (fun p => L (vitNetB N ε { w with b6 := p } img)) dy6)
  ∧ (w.b7.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib7 dy7
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b7 ib7
        (fun p => L (vitNetB N ε { w with b7 := p } img)) dy7)
  ∧ (w.b8.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib8 dy8
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b8 ib8
        (fun p => L (vitNetB N ε { w with b8 := p } img)) dy8)
  ∧ (w.b9.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib9 dy9
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b9 ib9
        (fun p => L (vitNetB N ε { w with b9 := p } img)) dy9)
  ∧ (w.b10.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib10 dy10
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b10 ib10
        (fun p => L (vitNetB N ε { w with b10 := p } img)) dy10)
  ∧ (w.b11.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib11 dy11
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b11 ib11
        (fun p => L (vitNetB N ε { w with b11 := p } img)) dy11)
  ∧ (w.b12.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib12 dy12
      ∧ vitBlockLossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b12 ib12
        (fun p => L (vitNetB N ε { w with b12 := p } img)) dy12)
  ∧ (vitFinalLNTiedGB N xN epsStr cotN ε w.γF w.βF w.Wcls b12out g
      ∧ vitHeadTiedGB N aN cotN hnB w.Wcls w.bcls g
      ∧ vitHeadLossTiedGB N xN aN epsStr cotN ε w.γF w.βF w.Wcls w.bcls b12out
        (fun a b W bb => L (vitNetB N ε { w with γF := a, βF := b, Wcls := W, bcls := bb } img)) g)
  ∧ (vitEmbedTiedGB N xN cotN w.Wc w.bc w.cls w.pos img dyEmbed
      ∧ vitEmbedLossTiedGB N xN cotN w.Wc w.bc w.cls w.pos img
        (fun W b c q => L (vitNetB N ε { w with Wc := W, bc := b, cls := c, pos := q } img)) dyEmbed) := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out flB hnB logitsB g dy12 dy11 dy10
    dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed L
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14⟩ :=
    vit_net_tiedGB N xN aN epsStr cotN aStr negAK bStr logN ohN ε α B w img t
  have hl :=
    vit_net_lossGrad_smoothedCE xN aN epsStr cotN aStr negAK bStr logN ohN N hK ε α B hε w img t ht
  -- the loss side's activations and logits, in the tie's spelling
  have e0 : vitPreE N w img = ib1 := by rw [vitPreE_apply N w img]
  have e1 : vitPreB1 N ε w img = ib2 := by rw [vitPreB1_apply N ε w img, e0]
  have e2 : vitPreB2 N ε w img = ib3 := by rw [vitPreB2_apply N ε w img, e1]
  have e3 : vitPreB3 N ε w img = ib4 := by rw [vitPreB3_apply N ε w img, e2]
  have e4 : vitPreB4 N ε w img = ib5 := by rw [vitPreB4_apply N ε w img, e3]
  have e5 : vitPreB5 N ε w img = ib6 := by rw [vitPreB5_apply N ε w img, e4]
  have e6 : vitPreB6 N ε w img = ib7 := by rw [vitPreB6_apply N ε w img, e5]
  have e7 : vitPreB7 N ε w img = ib8 := by rw [vitPreB7_apply N ε w img, e6]
  have e8 : vitPreB8 N ε w img = ib9 := by rw [vitPreB8_apply N ε w img, e7]
  have e9 : vitPreB9 N ε w img = ib10 := by rw [vitPreB9_apply N ε w img, e8]
  have e10 : vitPreB10 N ε w img = ib11 := by rw [vitPreB10_apply N ε w img, e9]
  have e11 : vitPreB11 N ε w img = ib12 := by rw [vitPreB11_apply N ε w img, e10]
  have e12 : vitPreB12 N ε w img = b12out := by rw [vitPreB12_apply N ε w img, e11]
  have eg : den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN
      (vitNetB N ε w img) t) = g := by rw [← vit_logitsB_eq, e12]
  unfold ViTNetLossTiedGB at hl
  rw [eg, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0] at hl
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, t13, l12⟩, ⟨t14, l13⟩⟩

end Net

end Proofs.ViTTieGB

