import LeanMlir.Proofs.Nets.ViT.ViTStepTieGB

/-! # ViT-Tiny — every parameter gradient node IS the loss's derivative in that parameter

`vit_net_tiedGB` says each of the 200 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it, the chain's top
being the smoothed-loss cotangent. `vit_net_lossGrad` composes that with the chain: for any loss
`L` of the logits whose gradient at the net's output is `g`, the loss of the WHOLE net `vitNetB`
with that one parameter varied is differentiable in it and every node is its gradient
(`HasGradAt`). `vitNetB` is the tie's forward, batched, at its stochastic-depth sites `sd`; at
`none` it is `batchMap N` of the canonical `vitForwardKV` at the twelve blocks
(`vitNetB_eq_vitForwardKV`). `vit_net_lossGrad_smoothedCE` discharges `hL` for
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
* **Lifted** (`HasGradAt.param_batchMapIdx_through`, `Foundation.Batched.Indexed`): read against
  the linear loss `⟨·, dyₙ⟩` per example, those gradients turn each tied node's
  `Σ_n Σ_j ∂per/∂θ · cotₙ` into `∂G/∂θ` of the batched block, `G` the loss at the block's output.
  Example `n`'s prefix and suffix are at its own drop sites (`exampleSite`), so the lift is the
  indexed one; the parameterised op itself is shared. Each node needs the block
  forward read as "the rest of the block ∘ the node's op ∘ the prefix" (`vit_fwd_Wq`, …), an
  equation up to `Mat.unflatten_flatten`.
* **Per net**: the loss read after each stage (`vitSuf*`), pulled back through the certified
  batched block and head VJPs (`vitBlockCotInB_eq_vjp`, `vitCotTowerOutB_eq_vjp`), and each `Φ`
  identified with the whole net at updated weights by a standalone `vit_factor_*` theorem.

**Two nodes carry the batch sum inside `den`,** as their text does: the classifier bias
(`biasGradB`) and the CLS token.

**Hypotheses.** `0 < ε` (the LayerNorms' VJPs); no smoothness hypothesis (GELU has no kink). For
the smoothed loss, every example's target sums to one and `0 < nC`. Stated at ViT-Tiny's literal
dims, as the tie is.

**Stochastic depth.** As in the tie: `sd` is the render's `drop` flag, `none` the drop-free
chain, `some` the `*drop*` artifacts'. Per example a site is `siteScale` on the out-projection's
or fc2's output (`vitHM`, `vitPostO`, `vitPostL2`/`F1`/`F2`), so the loss read after each op runs
through it, and the cotangents the bundle states are the tie's at the same sites.

**Precision.** As in the tie: every per-token dense weight node is
`rowDenseWeightGradBAt bf16 id …` and the patch-embed weight node `patchEmbedWeightGradBAt (bf16 &&
bf16ConvW) id …` (`StableHLO.PrecisionSwitch`), so `bf16 := true` is the kind the `vitin_*bf16`
artifacts emit, read over ℝ at the identity rounding (`Bf16Erasure`), and `bf16 := false` the f32
artifacts'. The bias, LayerNorm, CLS, position and classifier nodes carry no flag because no render
switches them.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTTieGB

open scoped BigOperators
open Proofs.ViTTie (ViTTieWeights)
open Proofs.ViTFoldGB (rowDenseBTiedB_holds rowDenseWTiedBAt_holds)
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

/-- **The MLP sublayer's input gradient at its site is `vitCotHVD`**, at any saved sublayer input
    `H`: the raw skip plus the MLP branch's certified backward of `sM ⊙ dy` (`vitMlpBr_back`). -/
theorem vitMlpSite_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sM : Option ℝ) (H : Mat Np1 (heads * d)) (dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun v => linLoss dy (vitMlpSiteF gf sM ε p v)) (Mat.flatten H)
      (vitCotHVD gf ε p.γ2 p.Wfc1 p.Wfc2 sM (Mat.flatten H)
        (Mat.flatten (fun r => Proofs.dense p.Wfc1 p.bfc1 (layerNormVec (heads * d) ε p.γ2 p.β2 (H r))))
        dy) := by
  refine (HasGradAt.comp_global (f := vitMlpSiteF gf sM ε p) (x := Mat.flatten H) (hasGradAt_linLoss dy _)
    (vitMlpSiteF_differentiable sM ε hε p) (vitMlpSiteHasVJP gf sM ε hε p)).of_eq ?_
  rw [vitCotHVD_eq]
  funext i
  show dy i + (vitMlpBrHasVJP gf ε hε p).backward (Mat.flatten H) (dropScalarOpt sM dy) i = _
  rw [vitMlpBr_back, Mat.unflatten_flatten]


/-! The block's forward, per example, as named `Mat`s — `blkSaves`' `let` chain verbatim, the
    attention site in `h`. -/

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

/-- The attention sublayer's output `h`, the out-projection through the attention site. -/
noncomputable def vitHM (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA : Option ℝ)
    (y : Vec (Np1 * (heads * d))) : Mat Np1 (heads * d) :=
  fun r s => Mat.unflatten y r s + siteScale sA (vitOM ε p y r s)

/-- LN₂'s output. -/
noncomputable def vitLn2M (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA : Option ℝ)
    (y : Vec (Np1 * (heads * d))) : Mat Np1 (heads * d) :=
  fun r kk => layerScale p.γ2 (fun s => layerNormForward (heads * d) ε 1 0 (vitHM ε p sA y r) s) kk
    + p.β2 kk

/-- fc1's output (pre-GELU). -/
noncomputable def vitM1M (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA : Option ℝ)
    (y : Vec (Np1 * (heads * d))) : Mat Np1 mlpDim :=
  fun r => Proofs.dense p.Wfc1 p.bfc1 (vitLn2M ε p sA y r)

/-- The out-projection, flat. -/
noncomputable def vitWoF (p : BlockParamsV (heads * d) mlpDim) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun v => Mat.flatten (fun r => Proofs.dense p.Wo p.bo (Mat.unflatten v r))

/-- The block after the out-projection: the attention site and residual, then the MLP sublayer at
    its site. -/
noncomputable def vitPostO (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitMlpSiteF gf sM ε p (fun i => y i + dropScalarOpt sA u i)

/-- The attention residual's flat sum is `h`, flattened. -/
theorem vitHM_flat (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA : Option ℝ)
    (y : Vec (Np1 * (heads * d))) :
    (fun i => y i + dropScalarOpt sA (Mat.flatten (vitOM ε p y)) i) = Mat.flatten (vitHM ε p sA y) := by
  funext i
  simp only [Mat.flatten, vitHM, dropScalarOpt, Mat.unflatten_apply, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **The out-projection's output cotangent is `sA ⊙ cH`** — the loss after the out-projection;
    the site's backward is the site (`dropScalarOptHasVJP`). -/
theorem vitPostO_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostO gf ε p sA sM y u)) (Mat.flatten (vitOM ε p y))
      (dropScalarOpt sA
        (cH gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy)) :=
  HasGradAt.comp (f := fun u i => y i + dropScalarOpt sA u i) (x := Mat.flatten (vitOM ε p y))
    ((vitMlpSite_hasGradAt ε hε p sM (vitHM ε p sA y) dy).congr_point (vitHM_flat ε p sA y).symm)
    (((dropScalarOpt_differentiable sA) _).const_add y)
    (constAddHasVJPAt y (dropScalarOpt sA) _ ((dropScalarOpt_differentiable sA) _)
      ((dropScalarOptHasVJP sA).toHasVJPAt _))


/-- The block after the attention core: out-projection, then `vitPostO`. -/
theorem vitPostAtt_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun a => linLoss dy (vitPostO gf ε p sA sM y (vitWoF p a))) (Mat.flatten (vitAttM ε p y))
      (cAtt gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) :=
  HasGradAt.comp_global (f := vitWoF p) (x := Mat.flatten (vitAttM ε p y))
    ((vitPostO_hasGradAt ε hε p sA sM y dy).congr_point (by rw [vitWoF, Mat.unflatten_flatten]; rfl))
    (dense_per_token_flat_differentiable p.Wo p.bo)
    (densePerTokenHasVJPMat Np1 (heads * d) (heads * d) p.Wo p.bo).toHasVJP

/-- The block after the Q projection. -/
noncomputable def vitPostQ (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO gf ε p sA sM y (vitWoF p (Mat.flatten (attnCore (Mat.unflatten u) (vitKM ε p y) (vitVM ε p y))))

/-- The block after the K projection. -/
noncomputable def vitPostK (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO gf ε p sA sM y (vitWoF p (Mat.flatten (attnCore (vitQM ε p y) (Mat.unflatten u) (vitVM ε p y))))

/-- The block after the V projection. -/
noncomputable def vitPostV (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO gf ε p sA sM y (vitWoF p (Mat.flatten (attnCore (vitQM ε p y) (vitKM ε p y) (Mat.unflatten u))))

/-- **The Q projection's output cotangent is `cQ`**: `cAtt` pulled back through the core in `Q`. -/
theorem vitPostQ_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostQ gf ε p sA sM y u)) (Mat.flatten (vitQM ε p y))
      (cQ gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) := by
  refine (HasGradAt.comp_global (f := fun u => Mat.flatten (attnCore (Mat.unflatten u) (vitKM ε p y) (vitVM ε p y)))
    (x := Mat.flatten (vitQM ε p y))
    ((vitPostAtt_hasGradAt ε hε p sA sM y dy).congr_point (by rw [Mat.unflatten_flatten]; rfl))
    (attnCoreQ_flat_differentiable _ _) (attnCoreQHasVJPMat _ _).toHasVJP).of_eq ?_
  rw [attnCoreQ_backward]
  exact (vitCotDQmh_eq_core _ _ _ _).symm

/-- **The K projection's output cotangent is `cK`.** -/
theorem vitPostK_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostK gf ε p sA sM y u)) (Mat.flatten (vitKM ε p y))
      (cK gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) := by
  refine (HasGradAt.comp_global (f := fun u => Mat.flatten (attnCore (vitQM ε p y) (Mat.unflatten u) (vitVM ε p y)))
    (x := Mat.flatten (vitKM ε p y))
    ((vitPostAtt_hasGradAt ε hε p sA sM y dy).congr_point (by rw [Mat.unflatten_flatten]; rfl))
    (attnCoreK_flat_differentiable _ _) (attnCoreKHasVJPMat _ _).toHasVJP).of_eq ?_
  rw [attnCoreK_backward]
  exact (vitCotDKmh_eq_core _ _ _ _).symm

/-- **The V projection's output cotangent is `cV`.** -/
theorem vitPostV_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostV gf ε p sA sM y u)) (Mat.flatten (vitVM ε p y))
      (cV gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) := by
  refine (HasGradAt.comp_global (f := fun u => Mat.flatten (attnCore (vitQM ε p y) (vitKM ε p y) (Mat.unflatten u)))
    (x := Mat.flatten (vitVM ε p y))
    ((vitPostAtt_hasGradAt ε hε p sA sM y dy).congr_point (by rw [Mat.unflatten_flatten]; rfl))
    (attnCoreV_flat_differentiable _ _) (attnCoreVHasVJPMat _ _).toHasVJP).of_eq ?_
  rw [attnCoreV_backward]
  exact (vitCotDVmh_eq_core _ _ _ _).symm


/-- The block after LN₁: the multi-head attention layer, then `vitPostO`. -/
noncomputable def vitPostL1 (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u => vitPostO gf ε p sA sM y (Mat.flatten
    (mhsaLayer Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo (Mat.unflatten u)))

/-- **LN₁'s output cotangent is `cLn1`**: `sA ⊙ cH` pulled back through the certified attention
    layer (`mhsaBackFlat_eq_mhsa_vjp`), whose three paths are the Q/K/V fan-in. -/
theorem vitPostL1_hasGradAt {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostL1 gf ε p sA sM y u)) (Mat.flatten (vitLn1M ε p y))
      (cLn1 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) := by
  refine (HasGradAt.comp_global
    (f := fun u => Mat.flatten (mhsaLayer Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo (Mat.unflatten u)))
    (x := Mat.flatten (vitLn1M ε p y))
    ((vitPostO_hasGradAt ε hε p sA sM y dy).congr_point
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
  have hc : cLn1 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy
      = vitCotLn1 p.Wq p.Wk p.Wv
          (vitCotDQmh Np1 heads d (Mat.flatten (vitQM ε p y)) (Mat.flatten (vitKM ε p y))
            (Mat.flatten (vitVM ε p y))
            (cAtt gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy))
          (vitCotDKmh Np1 heads d (Mat.flatten (vitQM ε p y)) (Mat.flatten (vitKM ε p y))
            (Mat.flatten (vitVM ε p y))
            (cAtt gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy))
          (vitCotDVmh Np1 heads d (Mat.flatten (vitQM ε p y)) (Mat.flatten (vitKM ε p y))
            (Mat.flatten (vitVM ε p y))
            (cAtt gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy)) :=
    rfl
  have ha : cAtt gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy
      = perRowFlat Np1 (heads * d) (Proofs.dense (Mat.transpose p.Wo) 0)
          (dropScalarOpt sA
            (cH gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy)) := by
    rw [← rowDenseBackFlat_eq_perRowFlat]; rfl
  rw [hb, hc, vitCotDQmh_eq_core, vitCotDKmh_eq_core, vitCotDVmh_eq_core, ha]
  funext i
  simp only [mhsaBackFlat, vitCotLn1, rowDenseBackFlat_eq_perRowFlat, Function.comp_apply]
  ring

/-- The block after LN₂: the MLP body through the MLP site, then the residual. -/
noncomputable def vitPostL2 (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u i => Mat.flatten (vitHM ε p sA y) i
    + dropScalarOpt sM (Mat.flatten (transformerMlp gf Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
        (Mat.unflatten u))) i

/-- The loss read after a branch's last op, through its site: `u ↦ ⟨c + sM ⊙ u, dy⟩` has gradient
    `sM ⊙ dy` (the site's backward is the site). -/
theorem hasGradAt_constAdd_site {n : Nat} (c : Vec n) (sM : Option ℝ) (u dy : Vec n) :
    HasGradAt (fun u' => linLoss dy (fun i => c i + dropScalarOpt sM u' i)) u (dropScalarOpt sM dy) :=
  HasGradAt.comp_global (f := dropScalarOpt sM) (x := u)
    (hasGradAt_constAdd (G := linLoss dy) c _ (hasGradAt_linLoss dy _))
    (dropScalarOpt_differentiable sM) (dropScalarOptHasVJP sM)

/-- **LN₂'s output cotangent is `cLn2`**: the MLP body's certified backward
    (`transformerMlp_back_flat_eq_perRowFlatPR`) at `sM ⊙ dy`. -/
theorem vitPostL2_hasGradAt {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostL2 gf ε p sA sM y u)) (Mat.flatten (vitLn2M ε p sA y))
      (cLn2 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) := by
  refine (HasGradAt.comp_global
    (f := fun u => Mat.flatten (transformerMlp gf Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 (Mat.unflatten u)))
    (x := Mat.flatten (vitLn2M ε p sA y)) (hasGradAt_constAdd_site _ sM _ dy)
    (transformerMlp_flat_differentiable Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)
    (transformerMlpHasVJPMat gf Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2).toHasVJP).of_eq ?_
  have h := transformerMlp_back_flat_eq_perRowFlatPR (gf := gf) Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
    (vitLn2M ε p sA y) (dropScalarOpt sM dy)
  refine (funext fun idx => ?_ : _ = _).trans (h.trans (vitCotLn2_eq_perRowFlatPR p.Wfc1 p.Wfc2
    (vitM1M ε p sA y) (dropScalarOpt sM dy)).symm)
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

/-- The block after fc1: GELU, fc2, the MLP site, then the residual. -/
noncomputable def vitPostF1 (gf : GeluForm) (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * mlpDim) → Vec (Np1 * (heads * d)) :=
  fun u i => Mat.flatten (vitHM ε p sA y) i
    + dropScalarOpt sM (Mat.flatten (fun r => Proofs.dense p.Wfc2 p.bfc2 (gf.map mlpDim (Mat.unflatten u r)))) i

/-- **fc1's output cotangent is `cM1`**: through fc2 and the GELU, at `sM ⊙ dy`. -/
theorem vitPostF1_hasGradAt {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y dy : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u => linLoss dy (vitPostF1 gf ε p sA sM y u)) (Mat.flatten (vitM1M ε p sA y))
      (cM1 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 sA sM y dy) := by
  refine (HasGradAt.comp_global
    (f := fun u => Mat.flatten (((fun Y : Mat Np1 mlpDim => fun n => Proofs.dense p.Wfc2 p.bfc2 (Y n)) ∘
      (fun Y : Mat Np1 mlpDim => fun n => gf.map mlpDim (Y n))) (Mat.unflatten u)))
    (x := Mat.flatten (vitM1M ε p sA y)) (hasGradAt_constAdd_site _ sM _ dy)
    (flat_differentiable_comp (gelu_per_token_flat_differentiable Np1 mlpDim)
      (dense_per_token_flat_differentiable p.Wfc2 p.bfc2))
    (vjpMatComp _ _ (gelu_per_token_flat_differentiable Np1 mlpDim)
      (dense_per_token_flat_differentiable p.Wfc2 p.bfc2) (geluPerTokenHasVJPMat gf Np1 mlpDim)
      (densePerTokenHasVJPMat Np1 mlpDim (heads * d) p.Wfc2 p.bfc2)).toHasVJP).of_eq ?_
  funext idx
  rw [HasVJPMat.toHasVJP_backward, Mat.unflatten_flatten]
  rfl

/-- The block after fc2: the MLP site, then the residual. -/
noncomputable def vitPostF2 (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (y : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  fun u i => Mat.flatten (vitHM ε p sA y) i + dropScalarOpt sM u i

/-- **fc2's output cotangent is `sM ⊙ dy`.** -/
theorem vitPostF2_hasGradAt (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (y dy u : Vec (Np1 * (heads * d))) :
    HasGradAt (fun u' => linLoss dy (vitPostF2 ε p sA sM y u')) u (dropScalarOpt sM dy) :=
  hasGradAt_constAdd_site _ sM u dy


/-! ### The block forward, read after each node -/

/-- The block forward is `vitPostO` at the out-projection's output. -/
theorem vit_fwd_eq_postO {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    p.fwdOD gf ε sA sM y = vitPostO gf ε p sA sM y (Mat.flatten (vitOM ε p y)) := by
  rw [fwdOD_eq_sites, Function.comp_apply, vitPostO]
  congr 1
  funext i
  simp only [vitAttnSiteF, vitAttnBrF, Function.comp_apply]
  rw [mhsaLayer_spelled]
  rfl

/-- The block forward is `vitPostF2` at fc2's output (`rfl`). -/
theorem vit_fwd_eq_postF2 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) :
    p.fwdOD gf ε sA sM y = vitPostF2 ε p sA sM y
      (Mat.flatten (fun r => Proofs.dense p.Wfc2 p.bfc2 (gf.map mlpDim (vitM1M ε p sA y r)))) := rfl

/-- The block with `γ1` varied is `vitPostL1` after LN₁ at that `γ1`. The fifteen lemmas below
    say the same for each other parameter: the node's op at the varied parameter, between the
    block's prefix and the rest of the block (`vitPost*`). -/
theorem vit_fwd_γ1 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with γ1 := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostL1 gf ε p sA sM y (Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β1 (Mat.unflatten y r))) := by
  rw [vit_fwd_eq_postO, vitPostL1, Mat.unflatten_flatten, mhsaLayer_spelled]; rfl

theorem vit_fwd_β1 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with β1 := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostL1 gf ε p sA sM y (Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ1 θ (Mat.unflatten y r))) := by
  rw [vit_fwd_eq_postO, vitPostL1, Mat.unflatten_flatten, mhsaLayer_spelled]; rfl

theorem vit_fwd_Wq {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (W : Mat (heads * d) (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with Wq := W } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostQ gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense W p.bq
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostQ, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bq {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with bq := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostQ gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense p.Wq θ
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostQ, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wk {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (W : Mat (heads * d) (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with Wk := W } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostK gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense W p.bk
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostK, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bk {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with bk := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostK gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense p.Wk θ
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostK, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wv {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (W : Mat (heads * d) (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with Wv := W } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostV gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense W p.bv
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostV, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bv {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with bv := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostV gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense p.Wv θ
          (Mat.unflatten (Mat.flatten (vitLn1M ε p y)) r))) := by
  rw [vit_fwd_eq_postO, vitPostV, vitWoF]; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wo {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (W : Mat (heads * d) (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with Wo := W } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostO gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense W p.bo
          (Mat.unflatten (Mat.flatten (vitAttM ε p y)) r))) := by
  rw [vit_fwd_eq_postO, Mat.unflatten_flatten]; rfl

theorem vit_fwd_bo {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with bo := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostO gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense p.Wo θ
          (Mat.unflatten (Mat.flatten (vitAttM ε p y)) r))) := by
  rw [vit_fwd_eq_postO, Mat.unflatten_flatten]; rfl

theorem vit_fwd_γ2 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with γ2 := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostL2 gf ε p sA sM y (Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β2
          (Mat.unflatten (Mat.flatten (vitHM ε p sA y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostL2 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_β2 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with β2 := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostL2 gf ε p sA sM y (Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ2 θ
          (Mat.unflatten (Mat.flatten (vitHM ε p sA y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostL2 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wfc1 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (W : Mat (heads * d) mlpDim) (y : Vec (Np1 * (heads * d))) :
    ({ p with Wfc1 := W } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostF1 gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense W p.bfc1
          (Mat.unflatten (Mat.flatten (vitLn2M ε p sA y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostF1 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_bfc1 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec mlpDim) (y : Vec (Np1 * (heads * d))) :
    ({ p with bfc1 := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostF1 gf ε p sA sM y (Mat.flatten (fun r => Proofs.dense p.Wfc1 θ
          (Mat.unflatten (Mat.flatten (vitLn2M ε p sA y)) r))) := by
  rw [vit_fwd_eq_postF2]; unfold vitPostF1 vitPostF2; simp only [Mat.unflatten_flatten]; rfl

theorem vit_fwd_Wfc2 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (W : Mat mlpDim (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with Wfc2 := W } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostF2 ε p sA sM y (Mat.flatten (fun r => Proofs.dense W p.bfc2
          (Mat.unflatten (Mat.flatten (fun r' => gf.map mlpDim (vitM1M ε p sA y r'))) r))) := by
  rw [vit_fwd_eq_postF2, Mat.unflatten_flatten]; rfl

theorem vit_fwd_bfc2 {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (θ : Vec (heads * d)) (y : Vec (Np1 * (heads * d))) :
    ({ p with bfc2 := θ } : BlockParamsV (heads * d) mlpDim).fwdOD gf ε sA sM y
      = vitPostF2 ε p sA sM y (Mat.flatten (fun r => Proofs.dense p.Wfc2 θ
          (Mat.unflatten (Mat.flatten (fun r' => gf.map mlpDim (vitM1M ε p sA y r'))) r))) := by
  rw [vit_fwd_eq_postF2, Mat.unflatten_flatten]; rfl


/-! ### Differentiability -/

/-- Each `vitPost*` is differentiable (`0 < ε` where the MLP sublayer's LN₂ is inside). -/
theorem vitPostO_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostO gf ε p sA sM y) := by
  unfold vitPostO
  exact (vitMlpSiteF_differentiable sM ε hε p).comp
    ((differentiable_const y).add (dropScalarOpt_differentiable sA))

theorem vitWoF_differentiable (p : BlockParamsV (heads * d) mlpDim) :
    Differentiable ℝ (vitWoF (Np1 := Np1) p) :=
  dense_per_token_flat_differentiable p.Wo p.bo

theorem vitPostL1_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostL1 gf ε p sA sM y) :=
  (vitPostO_differentiable ε hε p sA sM y).comp
    (mhsaLayer_flat_differentiable Np1 heads d p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo)

theorem vitPostQ_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostQ gf ε p sA sM y) :=
  (vitPostO_differentiable ε hε p sA sM y).comp ((vitWoF_differentiable p).comp
    (attnCoreQ_flat_differentiable _ _))

theorem vitPostK_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostK gf ε p sA sM y) :=
  (vitPostO_differentiable ε hε p sA sM y).comp ((vitWoF_differentiable p).comp
    (attnCoreK_flat_differentiable _ _))

theorem vitPostV_differentiable {gf : GeluForm} (ε : ℝ) (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostV gf ε p sA sM y) :=
  (vitPostO_differentiable ε hε p sA sM y).comp ((vitWoF_differentiable p).comp
    (attnCoreV_flat_differentiable _ _))

theorem vitPostL2_differentiable {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostL2 gf ε p sA sM y) := by
  have h := transformerMlp_flat_differentiable (gf := gf) Np1 (heads * d) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2
  unfold vitPostL2
  exact (differentiable_const _).add ((dropScalarOpt_differentiable sM).comp h)

theorem vitPostF1_differentiable {gf : GeluForm} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim)
    (sA sM : Option ℝ) (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostF1 gf ε p sA sM y) := by
  have h := flat_differentiable_comp (G := fun Y : Mat Np1 mlpDim => fun n => Proofs.dense p.Wfc2 p.bfc2 (Y n))
    (F := fun Y : Mat Np1 mlpDim => fun n => gf.map mlpDim (Y n))
    (gelu_per_token_flat_differentiable Np1 mlpDim) (dense_per_token_flat_differentiable p.Wfc2 p.bfc2)
  unfold vitPostF1
  exact (differentiable_const _).add ((dropScalarOpt_differentiable sM).comp h)

theorem vitPostF2_differentiable (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option ℝ)
    (y : Vec (Np1 * (heads * d))) : Differentiable ℝ (vitPostF2 ε p sA sM y) := by
  unfold vitPostF2
  exact (differentiable_const _).add (dropScalarOpt_differentiable sM)

end Chain

-- ════════════════════════════════════════════════════════════════
-- § A ViT block, batched — the sixteen nodes against the loss at the block's output
-- ════════════════════════════════════════════════════════════════

/-- **ViT block, every parameter node a loss derivative** — the sixteen nodes `vitBlockTiedGB`
    ties, at the tie's batched activations and cotangents, `Φ` the loss at the block's output as a
    function of the block's record. -/
def vitBlockLossTiedGB (gf : GeluForm) (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : BlockParamsV (heads * d) mlpDim) (bf16 : Bool) (sA sM : Option (Vec N))
    (xin : Vec (N * (Np1 * (heads * d))))
    (Φ : BlockParamsV (heads * d) mlpDim → Vec 1) (dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  -- forward saves, as the tie reads them
  let ln1B : Vec (N * (Np1 * (heads * d))) := batchMapIdx N (fun n x => (blkSaves gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 (exampleSite sA n) x).ln1) xin
  let attB : Vec (N * (Np1 * (heads * d))) := batchMapIdx N (fun n x => (blkSaves gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 (exampleSite sA n) x).att) xin
  let hB   : Vec (N * (Np1 * (heads * d))) := batchMapIdx N (fun n x => (blkSaves gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 (exampleSite sA n) x).h) xin
  let ln2B : Vec (N * (Np1 * (heads * d))) := batchMapIdx N (fun n x => (blkSaves gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 (exampleSite sA n) x).ln2) xin
  let gB   : Vec (N * (Np1 * mlpDim))      := batchMapIdx N (fun n x => (blkSaves gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 (exampleSite sA n) x).g) xin
  -- backward chain cotangents
  let cotLn1B : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cLn1 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let dQB     : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cQ gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let dKB     : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cK gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let dVB     : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cV gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let cotHB   : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cH gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let cotLn2B : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cLn2 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let cotM1B  : Vec (N * (Np1 * mlpDim))      := batchMapAuxIdx N (fun n => cM1 gf ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  -- the two sites on the branch cotangents (the skips read the raw ones)
  let cotOB   : Vec (N * (Np1 * (heads * d))) := dropPathOpt N (Np1 * (heads * d)) sA cotHB
  let dyOutD  : Vec (N * (Np1 * (heads * d))) := dropPathOpt N (Np1 * (heads * d)) sM dyOut
  -- LN₁ γ/β
  (HasGradAt (fun θ => Φ { p with γ1 := θ }) p.γ1
        (den (SHlo.veclnGammaGradB (N := N) (R := Np1) (D := heads * d) xN epsStr ε xin
          (.operand cotN cotLn1B))))
  ∧ (HasGradAt (fun θ => Φ { p with β1 := θ }) p.β1
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotLn1B))))
  -- Q / K / V / out-projection W, b
  ∧ (HasGradAt (fun θ => Φ { p with Wq := Mat.unflatten θ }) (Mat.flatten p.Wq)
        (den (SHlo.rowDenseWeightGradBAt bf16 (N := N) (tk := Np1) (a := heads * d) (c := heads * d) id xN ln1B
          (.operand cotN dQB))))
  ∧ (HasGradAt (fun θ => Φ { p with bq := θ }) p.bq
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dQB))))
  ∧ (HasGradAt (fun θ => Φ { p with Wk := Mat.unflatten θ }) (Mat.flatten p.Wk)
        (den (SHlo.rowDenseWeightGradBAt bf16 (N := N) (tk := Np1) (a := heads * d) (c := heads * d) id xN ln1B
          (.operand cotN dKB))))
  ∧ (HasGradAt (fun θ => Φ { p with bk := θ }) p.bk
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dKB))))
  ∧ (HasGradAt (fun θ => Φ { p with Wv := Mat.unflatten θ }) (Mat.flatten p.Wv)
        (den (SHlo.rowDenseWeightGradBAt bf16 (N := N) (tk := Np1) (a := heads * d) (c := heads * d) id xN ln1B
          (.operand cotN dVB))))
  ∧ (HasGradAt (fun θ => Φ { p with bv := θ }) p.bv
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dVB))))
  ∧ (HasGradAt (fun θ => Φ { p with Wo := Mat.unflatten θ }) (Mat.flatten p.Wo)
        (den (SHlo.rowDenseWeightGradBAt bf16 (N := N) (tk := Np1) (a := heads * d) (c := heads * d) id xN attB
          (.operand cotN cotOB))))
  ∧ (HasGradAt (fun θ => Φ { p with bo := θ }) p.bo
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotOB))))
  -- LN₂ γ/β
  ∧ (HasGradAt (fun θ => Φ { p with γ2 := θ }) p.γ2
        (den (SHlo.veclnGammaGradB (N := N) (R := Np1) (D := heads * d) xN epsStr ε hB
          (.operand cotN cotLn2B))))
  ∧ (HasGradAt (fun θ => Φ { p with β2 := θ }) p.β2
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotLn2B))))
  -- fc1 W/b
  ∧ (HasGradAt (fun θ => Φ { p with Wfc1 := Mat.unflatten θ }) (Mat.flatten p.Wfc1)
        (den (SHlo.rowDenseWeightGradBAt bf16 (N := N) (tk := Np1) (a := heads * d) (c := mlpDim) id xN ln2B
          (.operand cotN cotM1B))))
  ∧ (HasGradAt (fun θ => Φ { p with bfc1 := θ }) p.bfc1
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := mlpDim) (.operand cotN cotM1B))))
  -- fc2 W/b
  ∧ (HasGradAt (fun θ => Φ { p with Wfc2 := Mat.unflatten θ }) (Mat.flatten p.Wfc2)
        (den (SHlo.rowDenseWeightGradBAt bf16 (N := N) (tk := Np1) (a := mlpDim) (c := heads * d) id xN gB
          (.operand cotN dyOutD))))
  ∧ (HasGradAt (fun θ => Φ { p with bfc2 := θ }) p.bfc2
        (den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN dyOutD))))

/-- One node of the block bundle: `HasGradAt.param_batchMapIdx_through` at the factoring `hF` of
    the block with one slot varied (example `n` at its own sites), restated against `Φ'` (that slot
    of the loss) and at the node's own denotation `hnode`. -/
private theorem vit_node_lossTied {P N D b m : Nat} {Lb : Vec (N * D) → Vec 1}
    {X dY : Vec (N * D)} {Φ' : Vec P → Vec 1} {F : Fin N → Vec P → Vec D → Vec D}
    (pre : Fin N → Vec D → Vec b) (per : Vec P → Vec b → Vec m)
    (post : Fin N → Vec D → Vec m → Vec D) (cot : Fin N → Vec D → Vec D → Vec m) {θ node : Vec P}
    (hF : ∀ n θ' y, F n θ' y = post n y (per θ' (pre n y)))
    (hΦ : ∀ θ', Φ' θ' = Lb (batchMapIdx N (fun n => F n θ') X))
    (hG : HasGradAt Lb (batchMapIdx N (fun n y => post n y (per θ (pre n y))) X) dY)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ)
    (hpost : ∀ n y, Differentiable ℝ (post n y))
    (hcot : ∀ n y dy, HasGradAt (fun u => linLoss dy (post n y u)) (per θ (pre n y)) (cot n y dy))
    (A : Vec (N * b)) (COT : Vec (N * m))
    (hA : ∀ n, batchSlice N b A n = pre n (batchSlice N D X n))
    (hC : ∀ n, batchSlice N m COT n = cot n (batchSlice N D X n) (batchSlice N D dY n))
    (hnode : (fun i => ∑ n : Fin N, ∑ j : Fin m,
      pdiv (fun θ' => per θ' (batchSlice N b A n)) θ i j * batchSlice N m COT n j) = node) :
    HasGradAt Φ' θ node :=
  ((HasGradAt.param_batchMapIdx_through pre per post cot X hG hper hpost hcot A COT hA hC).congr_left
    (funext fun θ' => (congrArg (fun f => Lb (batchMapIdx N f X))
      (funext fun n => funext fun y => (hF n θ' y).symm)).trans (hΦ θ').symm)).of_eq
    hnode

theorem vit_block_lossTiedGB {gf : GeluForm} (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (hε : 0 < ε) (p : BlockParamsV (heads * d) mlpDim) (bf16 : Bool) (sA sM : Option (Vec N))
    (xin : Vec (N * (Np1 * (heads * d))))
    {Lb : Vec (N * (Np1 * (heads * d))) → Vec 1} {dyOut : Vec (N * (Np1 * (heads * d)))}
    (hLb : HasGradAt Lb
      (batchMapIdx N (fun n => p.fwdOD gf ε (exampleSite sA n) (exampleSite sM n)) xin) dyOut)
    {Φ : BlockParamsV (heads * d) mlpDim → Vec 1}
    (hΦ : ∀ p', Φ p' = Lb
      (batchMapIdx N (fun n => p'.fwdOD gf ε (exampleSite sA n) (exampleSite sM n)) xin)) :
    vitBlockLossTiedGB gf N xN epsStr cotN ε p bf16 sA sM xin Φ dyOut := by
  have hG : ∀ {f : Fin N → Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d))},
      (∀ n y, p.fwdOD gf ε (exampleSite sA n) (exampleSite sM n) y = f n y) →
        HasGradAt Lb (batchMapIdx N f xin) dyOut :=
    fun h => hLb.congr_point (congrArg (batchMapIdx N · xin) (funext fun n => funext (h n)))
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- LN₁ γ
    exact vit_node_lossTied (fun _ y => y) (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β1 (Mat.unflatten x r)))
      (fun n => vitPostL1 gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_γ1 ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_γ1 ε p (exampleSite sA n) (exampleSite sM n) p.γ1)
      (fun y => (rowLNVecFlat_gamma_differentiable _ _ ε p.β1 y) _) (fun n => vitPostL1_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n => vitPostL1_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n)) xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun k => (vecLNGammaTiedB_holds k).symm)
  · -- LN₁ β
    exact vit_node_lossTied (fun _ y => y) (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ1 θ (Mat.unflatten x r)))
      (fun n => vitPostL1 gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_β1 ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_β1 ε p (exampleSite sA n) (exampleSite sM n) p.β1)
      (fun y => (rowLNVecFlat_beta_differentiable _ _ ε p.γ1 y) _) (fun n => vitPostL1_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n => vitPostL1_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n)) xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun i => (vecLNBetaTiedB_holds i).symm)
  · -- Q W
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bq (Mat.unflatten x r)))
      (fun n => vitPostQ gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n θ => vit_fwd_Wq ε p (exampleSite sA n) (exampleSite sM n) (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun n y => (vit_fwd_Wq ε p (exampleSite sA n) (exampleSite sM n) p.Wq y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bq y) _) (fun n => vitPostQ_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostQ_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedBAt_holds bf16 i j).symm)
  · -- Q b
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wq θ (Mat.unflatten x r)))
      (fun n => vitPostQ gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_bq ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_bq ε p (exampleSite sA n) (exampleSite sM n) p.bq)
      (fun y => (rowDense_bias_differentiable p.Wq y) _) (fun n => vitPostQ_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostQ_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- K W
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bk (Mat.unflatten x r)))
      (fun n => vitPostK gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n θ => vit_fwd_Wk ε p (exampleSite sA n) (exampleSite sM n) (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun n y => (vit_fwd_Wk ε p (exampleSite sA n) (exampleSite sM n) p.Wk y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bk y) _) (fun n => vitPostK_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostK_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedBAt_holds bf16 i j).symm)
  · -- K b
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wk θ (Mat.unflatten x r)))
      (fun n => vitPostK gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_bk ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_bk ε p (exampleSite sA n) (exampleSite sM n) p.bk)
      (fun y => (rowDense_bias_differentiable p.Wk y) _) (fun n => vitPostK_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostK_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- V W
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bv (Mat.unflatten x r)))
      (fun n => vitPostV gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n θ => vit_fwd_Wv ε p (exampleSite sA n) (exampleSite sM n) (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun n y => (vit_fwd_Wv ε p (exampleSite sA n) (exampleSite sM n) p.Wv y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bv y) _) (fun n => vitPostV_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostV_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedBAt_holds bf16 i j).symm)
  · -- V b
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitLn1M ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wv θ (Mat.unflatten x r)))
      (fun n => vitPostV gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_bv ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_bv ε p (exampleSite sA n) (exampleSite sM n) p.bv)
      (fun y => (rowDense_bias_differentiable p.Wv y) _) (fun n => vitPostV_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostV_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitLn1M ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- out-projection W
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitAttM ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) (heads * d)) p.bo (Mat.unflatten x r)))
      (fun n => vitPostO gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n θ => vit_fwd_Wo ε p (exampleSite sA n) (exampleSite sM n) (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun n y => (vit_fwd_Wo ε p (exampleSite sA n) (exampleSite sM n) p.Wo y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bo y) _) (fun n => vitPostO_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostO_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitAttM ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => (batchSlice_dropPathOpt _ _ n).trans
        (congrArg _ (batchSlice_batchMapAuxIdx _ _ _ n)))
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedBAt_holds bf16 i j).symm)
  · -- out-projection b
    exact vit_node_lossTied (fun _ y => Mat.flatten (vitAttM ε p y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wo θ (Mat.unflatten x r)))
      (fun n => vitPostO gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_bo ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_bo ε p (exampleSite sA n) (exampleSite sM n) p.bo)
      (fun y => (rowDense_bias_differentiable p.Wo y) _) (fun n => vitPostO_differentiable ε hε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostO_hasGradAt ε hε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun _ y => Mat.flatten (vitAttM ε p y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => (batchSlice_dropPathOpt _ _ n).trans
        (congrArg _ (batchSlice_batchMapAuxIdx _ _ _ n)))
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- LN₂ γ
    exact vit_node_lossTied (fun n y => Mat.flatten (vitHM ε p (exampleSite sA n) y))
      (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε θ p.β2 (Mat.unflatten x r))) (fun n => vitPostL2 gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_γ2 ε p (exampleSite sA n) (exampleSite sM n))
      (fun _ => hΦ _) (hG fun n => vit_fwd_γ2 ε p (exampleSite sA n) (exampleSite sM n) p.γ2)
      (fun y => (rowLNVecFlat_gamma_differentiable _ _ ε p.β2 y) _) (fun n => vitPostL2_differentiable ε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostL2_hasGradAt ε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun n y => Mat.flatten (vitHM ε p (exampleSite sA n) y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun k => (vecLNGammaTiedB_holds k).symm)
  · -- LN₂ β
    exact vit_node_lossTied (fun n y => Mat.flatten (vitHM ε p (exampleSite sA n) y))
      (fun θ x => Mat.flatten (fun r => layerNormVec (heads * d) ε p.γ2 θ (Mat.unflatten x r))) (fun n => vitPostL2 gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_β2 ε p (exampleSite sA n) (exampleSite sM n))
      (fun _ => hΦ _) (hG fun n => vit_fwd_β2 ε p (exampleSite sA n) (exampleSite sM n) p.β2)
      (fun y => (rowLNVecFlat_beta_differentiable _ _ ε p.γ2 y) _) (fun n => vitPostL2_differentiable ε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostL2_hasGradAt ε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun n y => Mat.flatten (vitHM ε p (exampleSite sA n) y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun i => (vecLNBetaTiedB_holds i).symm)
  · -- fc1 W
    exact vit_node_lossTied (fun n y => Mat.flatten (vitLn2M ε p (exampleSite sA n) y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat (heads * d) mlpDim) p.bfc1 (Mat.unflatten x r)))
      (fun n => vitPostF1 gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n θ => vit_fwd_Wfc1 ε p (exampleSite sA n) (exampleSite sM n) (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun n y => (vit_fwd_Wfc1 ε p (exampleSite sA n) (exampleSite sM n) p.Wfc1 y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bfc1 y) _) (fun n => vitPostF1_differentiable ε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostF1_hasGradAt ε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun n y => Mat.flatten (vitLn2M ε p (exampleSite sA n) y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedBAt_holds bf16 i j).symm)
  · -- fc1 b
    exact vit_node_lossTied (fun n y => Mat.flatten (vitLn2M ε p (exampleSite sA n) y))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wfc1 θ (Mat.unflatten x r)))
      (fun n => vitPostF1 gf ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_bfc1 ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_bfc1 ε p (exampleSite sA n) (exampleSite sM n) p.bfc1)
      (fun y => (rowDense_bias_differentiable p.Wfc1 y) _) (fun n => vitPostF1_differentiable ε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => (vitPostF1_hasGradAt ε p (exampleSite sA n) (exampleSite sM n) y dy).congr_point
        (by simp only [Mat.unflatten_flatten]; rfl))
      (batchMapIdx N (fun n y => Mat.flatten (vitLn2M ε p (exampleSite sA n) y)) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_batchMapAuxIdx _ _ _ n)
      (funext fun i => (rowDenseBTiedB_holds i).symm)
  · -- fc2 W
    exact vit_node_lossTied (fun n y => Mat.flatten (fun r => gf.map mlpDim (vitM1M ε p (exampleSite sA n) y r)))
      (fun θ x => Mat.flatten (fun r => Proofs.dense (Mat.unflatten θ : Mat mlpDim (heads * d)) p.bfc2 (Mat.unflatten x r)))
      (fun n => vitPostF2 ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n θ => vit_fwd_Wfc2 ε p (exampleSite sA n) (exampleSite sM n) (Mat.unflatten θ)) (fun _ => hΦ _)
      (hG fun n y => (vit_fwd_Wfc2 ε p (exampleSite sA n) (exampleSite sM n) p.Wfc2 y).trans (by simp only [Mat.unflatten_flatten]))
      (fun y => (rowDense_weight_differentiable p.bfc2 y) _) (fun n => vitPostF2_differentiable ε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => vitPostF2_hasGradAt ε p (exampleSite sA n) (exampleSite sM n) y dy _)
      (batchMapIdx N (fun n y => Mat.flatten (fun r => gf.map mlpDim (vitM1M ε p (exampleSite sA n) y r))) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_dropPathOpt _ _ n)
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (rowDenseWTiedBAt_holds bf16 i j).symm)
  · -- fc2 b
    exact vit_node_lossTied (fun n y => Mat.flatten (fun r => gf.map mlpDim (vitM1M ε p (exampleSite sA n) y r)))
      (fun θ x => Mat.flatten (fun r => Proofs.dense p.Wfc2 θ (Mat.unflatten x r)))
      (fun n => vitPostF2 ε p (exampleSite sA n) (exampleSite sM n)) _ (fun n => vit_fwd_bfc2 ε p (exampleSite sA n) (exampleSite sM n)) (fun _ => hΦ _)
      (hG fun n => vit_fwd_bfc2 ε p (exampleSite sA n) (exampleSite sM n) p.bfc2)
      (fun y => (rowDense_bias_differentiable p.Wfc2 y) _) (fun n => vitPostF2_differentiable ε p (exampleSite sA n) (exampleSite sM n))
      (fun n y dy => vitPostF2_hasGradAt ε p (exampleSite sA n) (exampleSite sM n) y dy _)
      (batchMapIdx N (fun n y => Mat.flatten (fun r => gf.map mlpDim (vitM1M ε p (exampleSite sA n) y r))) xin) _
      (fun n => batchSlice_batchMapIdx _ _ n) (fun n => batchSlice_dropPathOpt _ _ n)
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
    (pos : Mat 197 192) (bf16 : Bool) (img : Vec (N * (3 * 224 * 224)))
    (Φ : Kernel4 192 3 16 16 → Vec 192 → Vec 192 → Mat 197 192 → Vec 1)
    (dyEmbed : Vec (N * (197 * 192))) : Prop :=
  HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bc cls pos) (Kernel4.flatten Wc)
      (den (SHlo.patchEmbedWeightGradBAt bf16 (N := N) (ic := 3) (H := 224) (W := 224) (P := 16)
        (tk := 196) (D := 192) id xN img (.operand cotN dyEmbed)))
  ∧ HasGradAt (fun θ => Φ Wc θ cls pos) bc
      (den (SHlo.patchEmbedBiasGradB (N := N) (tk := 196) (c := 192) (.operand cotN dyEmbed)))
  ∧ HasGradAt (fun θ => Φ Wc bc θ pos) cls
      (den (SHlo.denseBiasGradB (N := N) (c := 192)
        (.operand cotN (batchMap N (clsSliceFlat 196 192) dyEmbed))))
  ∧ HasGradAt (fun θ => Φ Wc bc cls (Mat.unflatten θ)) (Mat.flatten pos)
      (den (SHlo.posEmbedGradB (N := N) (tk := 196) (D := 192) (.operand cotN dyEmbed)))

theorem vit_embed_lossTiedGB (N : Nat) (xN cotN : String) (Wc : Kernel4 192 3 16 16)
    (bc cls : Vec 192) (pos : Mat 197 192) (bf16 : Bool) (img : Vec (N * (3 * 224 * 224)))
    {Lb : Vec (N * (197 * 192)) → Vec 1} {dyEmbed : Vec (N * (197 * 192))}
    (hLb : HasGradAt Lb (batchMap N (patchEmbedFlat 3 224 224 16 196 192 Wc bc cls pos) img) dyEmbed)
    {Φ : Kernel4 192 3 16 16 → Vec 192 → Vec 192 → Mat 197 192 → Vec 1}
    (hΦ : ∀ W b c q, Φ W b c q = Lb (batchMap N (patchEmbedFlat 3 224 224 16 196 192 W b c q) img)) :
    vitEmbedLossTiedGB N xN cotN Wc bc cls pos bf16 img Φ dyEmbed := by
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
    rw [Bf16Fold.den_patchEmbedWeightGradBAt_id]
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

/-- Pull the loss gradient back through a batched block's certified VJP at its sites: the
    cotangent is the tie's `batchMapAuxIdx N (p.cotInD ε …)` (`vitBlockCotInB_eq_vjp`). -/
theorem vitBlkB_hasGradAt_comp {gf : GeluForm} (N : Nat) {Np1 heads d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option (Vec N)) (X : Vec (N * (Np1 * (heads * d))))
    {G : Vec (N * (Np1 * (heads * d))) → Vec 1} {dY : Vec (N * (Np1 * (heads * d)))}
    (hG : HasGradAt G
      (batchMapIdx N (fun n => p.fwdOD gf ε (exampleSite sA n) (exampleSite sM n)) X) dY) :
    HasGradAt (fun y => G (batchMapIdx N (fun n => p.fwdOD gf ε (exampleSite sA n) (exampleSite sM n)) y)) X
      (batchMapAuxIdx N (fun n => p.cotInD gf ε (exampleSite sA n) (exampleSite sM n)) X dY) :=
  (HasGradAt.comp (x := X) hG
    (batchMapIdx_differentiableAt _ X (fun _ => fwdOD_differentiable ε hε p _ _ _))
    (batchMapIdxHasVJPAt _ X
      (fun n => (p.fwdODHasVJP gf ε hε (exampleSite sA n) (exampleSite sM n)).toHasVJPAt _)
      (fun _ => fwdOD_differentiable ε hε p _ _ _))).of_eq
    (congrFun (vitBlockCotInB_eq_vjp N ε hε p sA sM X) dY).symm

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
    embedding, `batchMapIdx N` of each block at its sites, then `batchMap N` of the head. At
    `sd = none` it is `batchMap N` of `vitForwardKV` at the twelve blocks
    (`vitNetB_eq_vitForwardKV`). -/
noncomputable def vitNetB (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    Vec (N * nC) :=
  batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls)
    (batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n))
    (batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n))
    (batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n))
    (batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n))
    (batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n))
    (batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n))
    (batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n))
    (batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n))
    (batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n))
    (batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n))
    (batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n))
    (batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n))
    (batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img)))))))))))))

/-- The patch embedding's output — block `b1`'s input (the tie's `ib1`). -/
noncomputable def vitPreE (N : Nat) {nC : Nat} (w : ViTTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos)

/-- Block `b1`'s output. -/
noncomputable def vitPreB1 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) ∘ vitPreE N w

/-- Block `b2`'s output. -/
noncomputable def vitPreB2 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) ∘ vitPreB1 gf N ε w sd

/-- Block `b3`'s output. -/
noncomputable def vitPreB3 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) ∘ vitPreB2 gf N ε w sd

/-- Block `b4`'s output. -/
noncomputable def vitPreB4 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) ∘ vitPreB3 gf N ε w sd

/-- Block `b5`'s output. -/
noncomputable def vitPreB5 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) ∘ vitPreB4 gf N ε w sd

/-- Block `b6`'s output. -/
noncomputable def vitPreB6 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) ∘ vitPreB5 gf N ε w sd

/-- Block `b7`'s output. -/
noncomputable def vitPreB7 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) ∘ vitPreB6 gf N ε w sd

/-- Block `b8`'s output. -/
noncomputable def vitPreB8 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) ∘ vitPreB7 gf N ε w sd

/-- Block `b9`'s output. -/
noncomputable def vitPreB9 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) ∘ vitPreB8 gf N ε w sd

/-- Block `b10`'s output. -/
noncomputable def vitPreB10 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) ∘ vitPreB9 gf N ε w sd

/-- Block `b11`'s output. -/
noncomputable def vitPreB11 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) ∘ vitPreB10 gf N ε w sd

/-- Block `b12`'s output. -/
noncomputable def vitPreB12 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (197 * 192)) :=
  batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) ∘ vitPreB11 gf N ε w sd

theorem vitPreE_apply (N : Nat) {nC : Nat} (w : ViTTieWeights nC) (img : Vec (N * (3 * 224 * 224))) :
    vitPreE N w img = batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img := by
  rw [vitPreE]

theorem vitPreB1_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB1 gf N ε w sd img = batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) (vitPreE N w img) := by
  rw [vitPreB1, Function.comp_apply]

theorem vitPreB2_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB2 gf N ε w sd img = batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) (vitPreB1 gf N ε w sd img) := by
  rw [vitPreB2, Function.comp_apply]

theorem vitPreB3_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB3 gf N ε w sd img = batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) (vitPreB2 gf N ε w sd img) := by
  rw [vitPreB3, Function.comp_apply]

theorem vitPreB4_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB4 gf N ε w sd img = batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) (vitPreB3 gf N ε w sd img) := by
  rw [vitPreB4, Function.comp_apply]

theorem vitPreB5_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB5 gf N ε w sd img = batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) (vitPreB4 gf N ε w sd img) := by
  rw [vitPreB5, Function.comp_apply]

theorem vitPreB6_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB6 gf N ε w sd img = batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) (vitPreB5 gf N ε w sd img) := by
  rw [vitPreB6, Function.comp_apply]

theorem vitPreB7_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB7 gf N ε w sd img = batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) (vitPreB6 gf N ε w sd img) := by
  rw [vitPreB7, Function.comp_apply]

theorem vitPreB8_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB8 gf N ε w sd img = batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) (vitPreB7 gf N ε w sd img) := by
  rw [vitPreB8, Function.comp_apply]

theorem vitPreB9_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB9 gf N ε w sd img = batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) (vitPreB8 gf N ε w sd img) := by
  rw [vitPreB9, Function.comp_apply]

theorem vitPreB10_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB10 gf N ε w sd img = batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) (vitPreB9 gf N ε w sd img) := by
  rw [vitPreB10, Function.comp_apply]

theorem vitPreB11_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB11 gf N ε w sd img = batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) (vitPreB10 gf N ε w sd img) := by
  rw [vitPreB11, Function.comp_apply]

theorem vitPreB12_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitPreB12 gf N ε w sd img = batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) (vitPreB11 gf N ε w sd img) := by
  rw [vitPreB12, Function.comp_apply]

/-- The net after block `b12` — the head. -/
noncomputable def vitSufB12 (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls)

/-- The net after block `b11`: block `b12`, then the rest. -/
noncomputable def vitSufB11 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB12 N ε w (batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) y)

/-- The net after block `b10`: block `b11`, then the rest. -/
noncomputable def vitSufB10 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB11 gf N ε w sd (batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) y)

/-- The net after block `b9`: block `b10`, then the rest. -/
noncomputable def vitSufB9 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB10 gf N ε w sd (batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) y)

/-- The net after block `b8`: block `b9`, then the rest. -/
noncomputable def vitSufB8 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB9 gf N ε w sd (batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) y)

/-- The net after block `b7`: block `b8`, then the rest. -/
noncomputable def vitSufB7 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB8 gf N ε w sd (batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) y)

/-- The net after block `b6`: block `b7`, then the rest. -/
noncomputable def vitSufB6 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB7 gf N ε w sd (batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) y)

/-- The net after block `b5`: block `b6`, then the rest. -/
noncomputable def vitSufB5 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB6 gf N ε w sd (batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) y)

/-- The net after block `b4`: block `b5`, then the rest. -/
noncomputable def vitSufB4 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB5 gf N ε w sd (batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) y)

/-- The net after block `b3`: block `b4`, then the rest. -/
noncomputable def vitSufB3 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB4 gf N ε w sd (batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) y)

/-- The net after block `b2`: block `b3`, then the rest. -/
noncomputable def vitSufB2 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB3 gf N ε w sd (batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) y)

/-- The net after block `b1`: block `b2`, then the rest. -/
noncomputable def vitSufB1 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB2 gf N ε w sd (batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) y)

/-- The net after the patch embedding: block `b1`, then the rest. -/
noncomputable def vitSufE (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) :
    Vec (N * (197 * 192)) → Vec (N * nC) :=
  fun y => vitSufB1 gf N ε w sd (batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) y)

/-- **The net with the patch embedding varied** is the suffix after it at the varied embedding. -/
theorem vit_factor_embed {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 192 3 16 16) (b c : Vec 192) (q : Mat 197 192) :
    vitNetB gf N ε { w with Wc := W, bc := b, cls := c, pos := q } sd img
      = vitSufE gf N ε w sd (batchMap N (patchEmbedFlat 3 224 224 16 196 192 W b c q) img) := rfl

/-- **The net with block `b1`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b1 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b1 := p } sd img
      = vitSufB1 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) (vitPreE N w img)) := by
  rw [vitPreE_apply]; rfl

/-- **The net with block `b2`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b2 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b2 := p } sd img
      = vitSufB2 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) (vitPreB1 gf N ε w sd img)) := by
  rw [vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b3`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b3 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b3 := p } sd img
      = vitSufB3 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) (vitPreB2 gf N ε w sd img)) := by
  rw [vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b4`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b4 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b4 := p } sd img
      = vitSufB4 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) (vitPreB3 gf N ε w sd img)) := by
  rw [vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b5`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b5 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b5 := p } sd img
      = vitSufB5 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) (vitPreB4 gf N ε w sd img)) := by
  rw [vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b6`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b6 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b6 := p } sd img
      = vitSufB6 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) (vitPreB5 gf N ε w sd img)) := by
  rw [vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b7`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b7 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b7 := p } sd img
      = vitSufB7 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) (vitPreB6 gf N ε w sd img)) := by
  rw [vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b8`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b8 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b8 := p } sd img
      = vitSufB8 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) (vitPreB7 gf N ε w sd img)) := by
  rw [vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b9`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b9 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b9 := p } sd img
      = vitSufB9 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) (vitPreB8 gf N ε w sd img)) := by
  rw [vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b10`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b10 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b10 := p } sd img
      = vitSufB10 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) (vitPreB9 gf N ε w sd img)) := by
  rw [vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b11`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b11 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b11 := p } sd img
      = vitSufB11 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) (vitPreB10 gf N ε w sd img)) := by
  rw [vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with block `b12`'s weights varied** is the suffix after it at the varied block. -/
theorem vit_factor_b12 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (p : BlockParamsV 192 768) :
    vitNetB gf N ε { w with b12 := p } sd img
      = vitSufB12 N ε w (batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) (vitPreB11 gf N ε w sd img)) := by
  rw [vitPreB11_apply, vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The net with the head varied** is the head at the varied parameters. -/
theorem vit_factor_head {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224)))
    (a b : Vec 192) (W : Mat 192 nC) (bb : Vec nC) :
    vitNetB gf N ε { w with γF := a, βF := b, Wcls := W, bcls := bb } sd img
      = batchMap N (vitHeadO ε a b W bb) (vitPreB12 gf N ε w sd img) := by
  rw [vitPreB12_apply, vitPreB11_apply, vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- The net's output is the head at block `b12`'s output. -/
theorem vit_forward_eq_head {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    vitNetB gf N ε w sd img = batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls) (vitPreB12 gf N ε w sd img) := by
  rw [vitPreB12_apply, vitPreB11_apply, vitPreB10_apply, vitPreB9_apply, vitPreB8_apply, vitPreB7_apply, vitPreB6_apply, vitPreB5_apply, vitPreB4_apply, vitPreB3_apply, vitPreB2_apply, vitPreB1_apply, vitPreE_apply]; rfl

/-- **The logits the tie's loss cotangent reads are `vitNetB`'s.** The tie spells the head as
    three batched ops (`batchMap_comp`). -/
theorem vit_logitsB_eq {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) :
    batchMap N (Proofs.dense w.Wcls w.bcls) (batchMap N (clsSliceFlat 196 192)
      (batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b r)))
        (vitPreB12 gf N ε w sd img))) = vitNetB gf N ε w sd img := by
  rw [vit_forward_eq_head, vitHeadO, classifierFlat, batchMap_comp, batchMap_comp]; rfl

/-- A block's forward is the depth-`k` fold's flat block: the spelled multi-head block
    (`vitBlockFwdOMHV`) is `blockV` (`vitBlockSpelledMHV_eq`). -/
theorem fwdO_eq_blockVFlat {gf : GeluForm} {Np1 heads d mlpDim : Nat} (ε : ℝ) (p : BlockParamsV (heads * d) mlpDim) :
    p.fwdO gf (Np1 := Np1) ε = blockVFlat gf Np1 heads d mlpDim ε p :=
  funext fun _ => congrArg Mat.flatten (vitBlockSpelledMHV_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)

/-- **Without stochastic depth `vitNetB` is the canonical ViT-Tiny forward, batched**: `vitForwardKV` at the twelve blocks
    `![w.b1, …, w.b12]`, the forward whose graph `vitFwdGraphKMHV_faithful` ties to the render and
    whose VJP is `vitForwardKVHasVJP`, applied per example. Per example it unfolds the depth-12 fold
    block by block (`fwdO_eq_blockVFlat`); `batchMap_comp` splits the batched composite into the
    capstone's stage-by-stage chain. -/
theorem vitNetB_eq_vitForwardKV {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC)
    (img : Vec (N * (3 * 224 * 224))) :
    vitNetB gf N ε w none img = batchMap N (vitForwardKV gf 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls
      w.pos ε ![w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7, w.b8, w.b9, w.b10, w.b11, w.b12]
      w.γF w.βF w.Wcls w.bcls) img := by
  have hper : ∀ y, vitForwardKV gf 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls w.pos ε
      ![w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7, w.b8, w.b9, w.b10, w.b11, w.b12]
      w.γF w.βF w.Wcls w.bcls y
      = (vitHeadO ε w.γF w.βF w.Wcls w.bcls ∘ w.b12.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b11.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b10.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b9.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b8.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b7.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b6.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b5.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b4.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b3.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ w.b2.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε ∘ w.b1.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε
        ∘ patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) y := by
    intro y
    simp only [vitForwardKV, vitBodyKVFlat, fwdO_eq_blockVFlat, Function.comp_apply, vitHeadO,
      Matrix.cons_val_zero, Matrix.cons_val_succ]
  rw [show vitForwardKV gf 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls w.pos ε
      ![w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7, w.b8, w.b9, w.b10, w.b11, w.b12]
      w.γF w.βF w.Wcls w.bcls = _ from funext hper, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp]
  have hb : ∀ (p : BlockParamsV 192 768) (y : Vec (N * (197 * 192))),
      batchMapIdx N (fun n => p.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (none : Option (Vec N)) n) (exampleSite (none : Option (Vec N)) n)) y
        = batchMap N (p.fwdO gf (Np1 := 197) (heads := 3) (d := 64) ε) y := fun _ _ => rfl
  unfold vitNetB
  simp only [vitSdA_none, vitSdM_none, hb, Function.comp_apply]

/-- **Every ViT-Tiny parameter gradient node is the derivative of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 200 nodes
    `vit_net_tiedGB` ties, each at the cotangent the tie threads to it from `g`, stated against
    `L` of `vitNetB` with that one parameter varied. -/
def ViTNetLossTiedGB (gf : GeluForm) (xN aN epsStr cotN : String) (N : Nat) {nC : Nat} (ε : ℝ)
    (w : ViTTieWeights nC) (bf16 bf16ConvW : Bool) (sd : Option (Fin 12 → Vec N × Vec N))
    (img : Vec (N * (3 * 224 * 224))) (L : Vec (N * nC) → Vec 1) (g : Vec (N * nC)) : Prop :=
  let dy12 := batchMapAux N (vitCotTowerOutV 196 192 nC ε w.γF w.Wcls) (vitPreB12 gf N ε w sd img) g
  let dy11 := batchMapAuxIdx N (fun n => w.b12.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) (vitPreB11 gf N ε w sd img) dy12
  let dy10 := batchMapAuxIdx N (fun n => w.b11.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) (vitPreB10 gf N ε w sd img) dy11
  let dy9 := batchMapAuxIdx N (fun n => w.b10.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) (vitPreB9 gf N ε w sd img) dy10
  let dy8 := batchMapAuxIdx N (fun n => w.b9.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) (vitPreB8 gf N ε w sd img) dy9
  let dy7 := batchMapAuxIdx N (fun n => w.b8.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) (vitPreB7 gf N ε w sd img) dy8
  let dy6 := batchMapAuxIdx N (fun n => w.b7.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) (vitPreB6 gf N ε w sd img) dy7
  let dy5 := batchMapAuxIdx N (fun n => w.b6.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) (vitPreB5 gf N ε w sd img) dy6
  let dy4 := batchMapAuxIdx N (fun n => w.b5.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) (vitPreB4 gf N ε w sd img) dy5
  let dy3 := batchMapAuxIdx N (fun n => w.b4.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) (vitPreB3 gf N ε w sd img) dy4
  let dy2 := batchMapAuxIdx N (fun n => w.b3.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) (vitPreB2 gf N ε w sd img) dy3
  let dy1 := batchMapAuxIdx N (fun n => w.b2.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) (vitPreB1 gf N ε w sd img) dy2
  let dyEmbed := batchMapAuxIdx N (fun n => w.b1.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) (vitPreE N w img) dy1
  vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b1 bf16 (vitSdA sd 0) (vitSdM sd 0) (vitPreE N w img)
      (fun p => L (vitNetB gf N ε { w with b1 := p } sd img)) dy1
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b2 bf16 (vitSdA sd 1) (vitSdM sd 1) (vitPreB1 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b2 := p } sd img)) dy2
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b3 bf16 (vitSdA sd 2) (vitSdM sd 2) (vitPreB2 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b3 := p } sd img)) dy3
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b4 bf16 (vitSdA sd 3) (vitSdM sd 3) (vitPreB3 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b4 := p } sd img)) dy4
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b5 bf16 (vitSdA sd 4) (vitSdM sd 4) (vitPreB4 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b5 := p } sd img)) dy5
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b6 bf16 (vitSdA sd 5) (vitSdM sd 5) (vitPreB5 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b6 := p } sd img)) dy6
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b7 bf16 (vitSdA sd 6) (vitSdM sd 6) (vitPreB6 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b7 := p } sd img)) dy7
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b8 bf16 (vitSdA sd 7) (vitSdM sd 7) (vitPreB7 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b8 := p } sd img)) dy8
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b9 bf16 (vitSdA sd 8) (vitSdM sd 8) (vitPreB8 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b9 := p } sd img)) dy9
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b10 bf16 (vitSdA sd 9) (vitSdM sd 9) (vitPreB9 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b10 := p } sd img)) dy10
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b11 bf16 (vitSdA sd 10) (vitSdM sd 10) (vitPreB10 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b11 := p } sd img)) dy11
  ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b12 bf16 (vitSdA sd 11) (vitSdM sd 11) (vitPreB11 gf N ε w sd img)
      (fun p => L (vitNetB gf N ε { w with b12 := p } sd img)) dy12
  ∧ vitHeadLossTiedGB N xN aN epsStr cotN ε w.γF w.βF w.Wcls w.bcls (vitPreB12 gf N ε w sd img)
      (fun a b W bb => L (vitNetB gf N ε { w with γF := a, βF := b, Wcls := W, bcls := bb } sd img)) g
  ∧ vitEmbedLossTiedGB N xN cotN w.Wc w.bc w.cls w.pos (bf16 && bf16ConvW) img
      (fun W b c q => L (vitNetB gf N ε { w with Wc := W, bc := b, cls := c, pos := q } sd img)) dyEmbed

/-- **Every ViT-Tiny parameter gradient node is the derivative of the loss in that parameter.**
    For any loss `L` of the logits with gradient `g` at the net's output, each of the 200 nodes
    `vit_net_tiedGB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net, `vitNetB` with
    that one parameter varied (a patch-embedding field, a block's record `w.bk := p` with one slot
    changed, or a head field).

    Hypothesis: `0 < ε`, the LayerNorms' (the tie itself needs none). The loss enters only through
    `hL`; `vit_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship. -/
theorem vit_net_lossGrad {gf : GeluForm} (xN aN epsStr cotN : String) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (bf16 bf16ConvW : Bool) (sd : Option (Fin 12 → Vec N × Vec N))
    (img : Vec (N * (3 * 224 * 224))) {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)}
    (hL : HasGradAt L (vitNetB gf N ε w sd img) g) :
    ViTNetLossTiedGB gf xN aN epsStr cotN N ε w bf16 bf16ConvW sd img L g := by
  unfold ViTNetLossTiedGB
  intro dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  have hL' : HasGradAt L (batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls) (vitPreB12 gf N ε w sd img)) g :=
    hL.congr_point (vit_forward_eq_head N ε w sd img)
  have hB12 : HasGradAt (fun y => L (vitSufB12 N ε w y)) (vitPreB12 gf N ε w sd img) dy12 :=
    vitHeadB_hasGradAt_comp N ε hε w.γF w.βF w.Wcls w.bcls _ hL'
  have hB11 : HasGradAt (fun y => L (vitSufB11 gf N ε w sd y)) (vitPreB11 gf N ε w sd img) dy11 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b12 (vitSdA sd 11) (vitSdM sd 11) _ (hB12.congr_point (vitPreB12_apply N ε w sd img))
  have hB10 : HasGradAt (fun y => L (vitSufB10 gf N ε w sd y)) (vitPreB10 gf N ε w sd img) dy10 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b11 (vitSdA sd 10) (vitSdM sd 10) _ (hB11.congr_point (vitPreB11_apply N ε w sd img))
  have hB9 : HasGradAt (fun y => L (vitSufB9 gf N ε w sd y)) (vitPreB9 gf N ε w sd img) dy9 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b10 (vitSdA sd 9) (vitSdM sd 9) _ (hB10.congr_point (vitPreB10_apply N ε w sd img))
  have hB8 : HasGradAt (fun y => L (vitSufB8 gf N ε w sd y)) (vitPreB8 gf N ε w sd img) dy8 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b9 (vitSdA sd 8) (vitSdM sd 8) _ (hB9.congr_point (vitPreB9_apply N ε w sd img))
  have hB7 : HasGradAt (fun y => L (vitSufB7 gf N ε w sd y)) (vitPreB7 gf N ε w sd img) dy7 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b8 (vitSdA sd 7) (vitSdM sd 7) _ (hB8.congr_point (vitPreB8_apply N ε w sd img))
  have hB6 : HasGradAt (fun y => L (vitSufB6 gf N ε w sd y)) (vitPreB6 gf N ε w sd img) dy6 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b7 (vitSdA sd 6) (vitSdM sd 6) _ (hB7.congr_point (vitPreB7_apply N ε w sd img))
  have hB5 : HasGradAt (fun y => L (vitSufB5 gf N ε w sd y)) (vitPreB5 gf N ε w sd img) dy5 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b6 (vitSdA sd 5) (vitSdM sd 5) _ (hB6.congr_point (vitPreB6_apply N ε w sd img))
  have hB4 : HasGradAt (fun y => L (vitSufB4 gf N ε w sd y)) (vitPreB4 gf N ε w sd img) dy4 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b5 (vitSdA sd 4) (vitSdM sd 4) _ (hB5.congr_point (vitPreB5_apply N ε w sd img))
  have hB3 : HasGradAt (fun y => L (vitSufB3 gf N ε w sd y)) (vitPreB3 gf N ε w sd img) dy3 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b4 (vitSdA sd 3) (vitSdM sd 3) _ (hB4.congr_point (vitPreB4_apply N ε w sd img))
  have hB2 : HasGradAt (fun y => L (vitSufB2 gf N ε w sd y)) (vitPreB2 gf N ε w sd img) dy2 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b3 (vitSdA sd 2) (vitSdM sd 2) _ (hB3.congr_point (vitPreB3_apply N ε w sd img))
  have hB1 : HasGradAt (fun y => L (vitSufB1 gf N ε w sd y)) (vitPreB1 gf N ε w sd img) dy1 :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b2 (vitSdA sd 1) (vitSdM sd 1) _ (hB2.congr_point (vitPreB2_apply N ε w sd img))
  have hE : HasGradAt (fun y => L (vitSufE gf N ε w sd y)) (vitPreE N w img) dyEmbed :=
    vitBlkB_hasGradAt_comp N (Np1 := 197) (heads := 3) (d := 64) ε hε w.b1 (vitSdA sd 0) (vitSdM sd 0) _ (hB1.congr_point (vitPreB1_apply N ε w sd img))
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b1 bf16 (vitSdA sd 0) (vitSdM sd 0) _
      (hB1.congr_point (vitPreB1_apply N ε w sd img)) (fun p => by rw [vit_factor_b1]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b2 bf16 (vitSdA sd 1) (vitSdM sd 1) _
      (hB2.congr_point (vitPreB2_apply N ε w sd img)) (fun p => by rw [vit_factor_b2]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b3 bf16 (vitSdA sd 2) (vitSdM sd 2) _
      (hB3.congr_point (vitPreB3_apply N ε w sd img)) (fun p => by rw [vit_factor_b3]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b4 bf16 (vitSdA sd 3) (vitSdM sd 3) _
      (hB4.congr_point (vitPreB4_apply N ε w sd img)) (fun p => by rw [vit_factor_b4]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b5 bf16 (vitSdA sd 4) (vitSdM sd 4) _
      (hB5.congr_point (vitPreB5_apply N ε w sd img)) (fun p => by rw [vit_factor_b5]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b6 bf16 (vitSdA sd 5) (vitSdM sd 5) _
      (hB6.congr_point (vitPreB6_apply N ε w sd img)) (fun p => by rw [vit_factor_b6]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b7 bf16 (vitSdA sd 6) (vitSdM sd 6) _
      (hB7.congr_point (vitPreB7_apply N ε w sd img)) (fun p => by rw [vit_factor_b7]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b8 bf16 (vitSdA sd 7) (vitSdM sd 7) _
      (hB8.congr_point (vitPreB8_apply N ε w sd img)) (fun p => by rw [vit_factor_b8]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b9 bf16 (vitSdA sd 8) (vitSdM sd 8) _
      (hB9.congr_point (vitPreB9_apply N ε w sd img)) (fun p => by rw [vit_factor_b9]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b10 bf16 (vitSdA sd 9) (vitSdM sd 9) _
      (hB10.congr_point (vitPreB10_apply N ε w sd img)) (fun p => by rw [vit_factor_b10]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b11 bf16 (vitSdA sd 10) (vitSdM sd 10) _
      (hB11.congr_point (vitPreB11_apply N ε w sd img)) (fun p => by rw [vit_factor_b11]), ?_⟩
  refine ⟨vit_block_lossTiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε hε w.b12 bf16 (vitSdA sd 11) (vitSdM sd 11) _
      (hB12.congr_point (vitPreB12_apply N ε w sd img)) (fun p => by rw [vit_factor_b12]), ?_⟩
  refine ⟨vit_head_lossTiedGB N xN aN epsStr cotN ε w.γF w.βF w.Wcls w.bcls _ hL'
    (fun a b W bb => by rw [vit_factor_head]), ?_⟩
  exact vit_embed_lossTiedGB N xN cotN w.Wc w.bc w.cls w.pos (bf16 && bf16ConvW) img
    (hE.congr_point (vitPreE_apply N w img)) (fun W b c q => by rw [vit_factor_embed])

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLossDiv`, `g` the `softmaxDiv` cotangent the render emits — the
    tie's own `g`, whose logits are `vitNetB N ε w img` (`vit_logitsB_eq`). -/
theorem vit_net_lossGrad_smoothedCE {gf : GeluForm} (xN aN epsStr cotN aStr negAK bStr logN ohN : String)
    (N : Nat) {nC : Nat} (hK : 0 < nC) (ε α B : ℝ) (hε : 0 < ε) (w : ViTTieWeights nC)
    (bf16 bf16ConvW : Bool) (sd : Option (Fin 12 → Vec N × Vec N)) (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC)) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    ViTNetLossTiedGB gf xN aN epsStr cotN N ε w bf16 bf16ConvW sd img (smoothedBatchLossDiv N nC α B t)
      (den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN (vitNetB gf N ε w sd img) t)) :=
  vit_net_lossGrad xN aN epsStr cotN N ε hε w bf16 bf16ConvW sd img
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
theorem vit_net_tied_lossGrad {gf : GeluForm} (N : Nat) {nC : Nat}
    (xN aN epsStr cotN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    (w : ViTTieWeights nC) (bf16 bf16ConvW : Bool) (sd : Option (Fin 12 → Vec N × Vec N))
    (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC))
    (hK : 0 < nC) (hε : 0 < ε) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    let ib1    : Vec (N * (197 * 192)) := batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img
    let ib2    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) ib1
    let ib3    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) ib2
    let ib4    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) ib3
    let ib5    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) ib4
    let ib6    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) ib5
    let ib7    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) ib6
    let ib8    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) ib7
    let ib9    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) ib8
    let ib10   : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) ib9
    let ib11   : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) ib10
    let ib12   : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) ib11
    let b12out : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) ib12
    -- final LN → CLS row → dense head, then the SMOOTHED loss cotangent at a general target `t`
    let flB     : Vec (N * (197 * 192)) :=
      batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b r))) b12out
    let hnB     : Vec (N * 192) := batchMap N (clsSliceFlat 196 192) flB
    let logitsB : Vec (N * nC)  := batchMap N (dense w.Wcls w.bcls) hnB
    let g       : Vec (N * nC)  :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    let dy12    : Vec (N * (197 * 192)) := batchMapAux N (vitCotTowerOutV 196 192 nC ε w.γF w.Wcls) b12out g
    let dy11   : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b12.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) ib12 dy12
    let dy10   : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b11.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) ib11 dy11
    let dy9    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b10.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) ib10 dy10
    let dy8    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b9.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) ib9 dy9
    let dy7    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b8.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) ib8 dy8
    let dy6    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b7.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) ib7 dy7
    let dy5    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b6.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) ib6 dy6
    let dy4    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b5.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) ib5 dy5
    let dy3    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b4.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) ib4 dy4
    let dy2    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b3.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) ib3 dy3
    let dy1    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b2.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) ib2 dy2
    let dyEmbed: Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b1.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) ib1 dy1
    let L := smoothedBatchLossDiv N nC α B t
    (w.b1.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 0) (vitSdM sd 0) ib1 dy1
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b1 bf16 (vitSdA sd 0) (vitSdM sd 0) ib1
        (fun p => L (vitNetB gf N ε { w with b1 := p } sd img)) dy1)
  ∧ (w.b2.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 1) (vitSdM sd 1) ib2 dy2
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b2 bf16 (vitSdA sd 1) (vitSdM sd 1) ib2
        (fun p => L (vitNetB gf N ε { w with b2 := p } sd img)) dy2)
  ∧ (w.b3.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 2) (vitSdM sd 2) ib3 dy3
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b3 bf16 (vitSdA sd 2) (vitSdM sd 2) ib3
        (fun p => L (vitNetB gf N ε { w with b3 := p } sd img)) dy3)
  ∧ (w.b4.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 3) (vitSdM sd 3) ib4 dy4
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b4 bf16 (vitSdA sd 3) (vitSdM sd 3) ib4
        (fun p => L (vitNetB gf N ε { w with b4 := p } sd img)) dy4)
  ∧ (w.b5.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 4) (vitSdM sd 4) ib5 dy5
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b5 bf16 (vitSdA sd 4) (vitSdM sd 4) ib5
        (fun p => L (vitNetB gf N ε { w with b5 := p } sd img)) dy5)
  ∧ (w.b6.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 5) (vitSdM sd 5) ib6 dy6
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b6 bf16 (vitSdA sd 5) (vitSdM sd 5) ib6
        (fun p => L (vitNetB gf N ε { w with b6 := p } sd img)) dy6)
  ∧ (w.b7.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 6) (vitSdM sd 6) ib7 dy7
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b7 bf16 (vitSdA sd 6) (vitSdM sd 6) ib7
        (fun p => L (vitNetB gf N ε { w with b7 := p } sd img)) dy7)
  ∧ (w.b8.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 7) (vitSdM sd 7) ib8 dy8
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b8 bf16 (vitSdA sd 7) (vitSdM sd 7) ib8
        (fun p => L (vitNetB gf N ε { w with b8 := p } sd img)) dy8)
  ∧ (w.b9.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 8) (vitSdM sd 8) ib9 dy9
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b9 bf16 (vitSdA sd 8) (vitSdM sd 8) ib9
        (fun p => L (vitNetB gf N ε { w with b9 := p } sd img)) dy9)
  ∧ (w.b10.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 9) (vitSdM sd 9) ib10 dy10
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b10 bf16 (vitSdA sd 9) (vitSdM sd 9) ib10
        (fun p => L (vitNetB gf N ε { w with b10 := p } sd img)) dy10)
  ∧ (w.b11.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 10) (vitSdM sd 10) ib11 dy11
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b11 bf16 (vitSdA sd 10) (vitSdM sd 10) ib11
        (fun p => L (vitNetB gf N ε { w with b11 := p } sd img)) dy11)
  ∧ (w.b12.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 11) (vitSdM sd 11) ib12 dy12
      ∧ vitBlockLossTiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε w.b12 bf16 (vitSdA sd 11) (vitSdM sd 11) ib12
        (fun p => L (vitNetB gf N ε { w with b12 := p } sd img)) dy12)
  ∧ (vitFinalLNTiedGB N xN epsStr cotN ε w.γF w.βF w.Wcls b12out g
      ∧ vitHeadTiedGB N aN cotN hnB w.Wcls w.bcls g
      ∧ vitHeadLossTiedGB N xN aN epsStr cotN ε w.γF w.βF w.Wcls w.bcls b12out
        (fun a b W bb => L (vitNetB gf N ε { w with γF := a, βF := b, Wcls := W, bcls := bb } sd img)) g)
  ∧ (vitEmbedTiedGB N xN cotN w.Wc w.bc w.cls w.pos (bf16 && bf16ConvW) img dyEmbed
      ∧ vitEmbedLossTiedGB N xN cotN w.Wc w.bc w.cls w.pos (bf16 && bf16ConvW) img
        (fun W b c q => L (vitNetB gf N ε { w with Wc := W, bc := b, cls := c, pos := q } sd img)) dyEmbed) := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out flB hnB logitsB g dy12 dy11 dy10
    dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed L
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14⟩ :=
    vit_net_tiedGB (gf := gf) N xN aN epsStr cotN aStr negAK bStr logN ohN ε α B w bf16 bf16ConvW sd img t
  have hl :=
    vit_net_lossGrad_smoothedCE (gf := gf) xN aN epsStr cotN aStr negAK bStr logN ohN N hK ε α B hε w bf16 bf16ConvW sd img t ht
  -- the loss side's activations and logits, in the tie's spelling
  have e0 : vitPreE N w img = ib1 := by rw [vitPreE_apply N w img]
  have e1 : vitPreB1 gf N ε w sd img = ib2 := by rw [vitPreB1_apply N ε w sd img, e0]
  have e2 : vitPreB2 gf N ε w sd img = ib3 := by rw [vitPreB2_apply N ε w sd img, e1]
  have e3 : vitPreB3 gf N ε w sd img = ib4 := by rw [vitPreB3_apply N ε w sd img, e2]
  have e4 : vitPreB4 gf N ε w sd img = ib5 := by rw [vitPreB4_apply N ε w sd img, e3]
  have e5 : vitPreB5 gf N ε w sd img = ib6 := by rw [vitPreB5_apply N ε w sd img, e4]
  have e6 : vitPreB6 gf N ε w sd img = ib7 := by rw [vitPreB6_apply N ε w sd img, e5]
  have e7 : vitPreB7 gf N ε w sd img = ib8 := by rw [vitPreB7_apply N ε w sd img, e6]
  have e8 : vitPreB8 gf N ε w sd img = ib9 := by rw [vitPreB8_apply N ε w sd img, e7]
  have e9 : vitPreB9 gf N ε w sd img = ib10 := by rw [vitPreB9_apply N ε w sd img, e8]
  have e10 : vitPreB10 gf N ε w sd img = ib11 := by rw [vitPreB10_apply N ε w sd img, e9]
  have e11 : vitPreB11 gf N ε w sd img = ib12 := by rw [vitPreB11_apply N ε w sd img, e10]
  have e12 : vitPreB12 gf N ε w sd img = b12out := by rw [vitPreB12_apply N ε w sd img, e11]
  have eg : den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN
      (vitNetB gf N ε w sd img) t) = g := by rw [← vit_logitsB_eq, e12]
  unfold ViTNetLossTiedGB at hl
  rw [eg, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0] at hl
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, t13, l12⟩, ⟨t14, l13⟩⟩

end Net

end Proofs.ViTTieGB

