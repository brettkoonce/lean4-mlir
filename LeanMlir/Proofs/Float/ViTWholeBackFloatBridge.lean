import LeanMlir.Proofs.Float.PatchEmbedBackFloatBridge
import LeanMlir.Proofs.Float.ChannelLNFloatBridge
import LeanMlir.Proofs.Architectures.ViTDepthK

/-! # The whole-net ViT-Tiny input-gradient chain, at the VECTOR LayerNorm the net runs

`MhsaBackFloatBridge.lean` defines `vitGradFlat`, the ViT input-gradient skeleton, with its
encoder tower supplied as a `List` of opaque per-block backwards and its final LayerNorm as a
single shared `Vec D → Vec D`. That shape is what a `FloatBridges` fold needs and it is not what
a certified tie can be stated at, for two reasons the ConvNeXt thread already paid for:

* **The LayerNorm slots are per-token.** `layerNormVec_per_token_has_vjp_mat.backward A` runs the
  single-token backward at *each token's own* saved input `A r`; one shared `finalLNBack` cannot
  carry that. `vitBlockBackPR` is the enrichment on the block side; this file is the same for the
  final LN, and it takes the form ConvNeXt already built — `rowLNVecFlatBack`, which is
  `perRowIdxFlat` of `bn_grad_input ∘ diagBack γ` at each row's saved activation and whose header
  says in as many words that it is *"literally ViT's per-token LN with 'token' read as 'spatial
  position'"*.
* ⛔ **`vitBlockBackPR`'s tie is at the SCALAR LayerNorm.** `vitBlockBackPR_eq_transformerBlock_vjp`
  is stated at `γ1 β1 γ2 β2 : ℝ` against `transformerBlock_has_vjp_mat`; the shipped depth-12 net
  is `vitForwardKV`, whose blocks are `transformerBlockV` at `γ β : Vec D`. That is exactly the
  hole package 3.1 closed for ConvNeXt-T (a tie true of a net the repo stopped running), and the
  audit in `planning/proofs_tier_to_paper_nets.md` §3.4 named the wrong one — it recorded the tie
  as being at `heads = 1`, which it is not (`ViTMhsaBackCertifiedTie` is general in `h`
  throughout). The LayerNorm form is the real gap.

So this file re-spells the chain at the shipped conventions and names every term, so that
`ViTWholeBackCertifiedTie.lean` can be about a named object and `ViTBackFloatBudget.lean` can put
a numeral on the same one. It is `EfficientNetFullWholeBackFloatBridge.lean`'s role for ViT.

**Convention table for the net this chain reverses** (`vitForwardKV` at ViT-Tiny, read from
`apps/baselines/MainVitTrain.lean` and `LeanMlir/Proofs/Codegen/ViTRender.lean`):
depth 12 with distinct per-block parameters, `D = 192 = 3 heads × 64`, MLP dim 768, 197 tokens
(196 patches + CLS), `16×16/s16` patchify (no `conv2d`, so no padding phase and none of the
even-kernel question `EvenKernelConvBack.lean` found for ConvNeXt), **vector-`[D]` LayerNorm at
all 25 sites**, GELU, `ε = 1e-5`. No BatchNorm anywhere, so the BN-world axis of §4 does not
touch ViT.
-/

namespace Proofs

variable {h N dh : Nat}

-- ════════════════════════════════════════════════════════════════
-- § 1. The vector-LN encoder-block backward
-- ════════════════════════════════════════════════════════════════

/-- **The vector-LayerNorm ViT encoder-block input-gradient backward.** `vitBlockBackPR` with its
    two per-token LN slots replaced by `rowLNVecFlatBack` at the site's flat saved input, and the
    MLP sublayer's residual lifted out of the per-token fold (`perRowFlatPR_residual`: a per-row
    `residual` is the row lift plus the cotangent, so the two spellings agree).

    Reading right to left, this is the reverse of `transformerBlockV`: the MLP sublayer's
    `dense W₂ ∘ gelu ∘ dense W₁ ∘ LN₂` backward under a residual, then the attention sublayer's
    `mhsa ∘ LN₁` backward under a residual. Every factor is a term the float tier already
    bridges — `rowLNVecFlatBack` (`floatBridgesTo_rowLNVecFlatBack`), `mhsaBackFlat`
    (`floatBridges_mhsaBack`), the free `dense Wᵀ 0` input-VJPs and the saved-derivative
    `diagBack`. -/
noncomputable def vitBlockBackV {dff : Nat} (Wq Wk Wv Wo : Mat (h * dh) (h * dh))
    (Q K V : Mat N (h * dh)) (ε : ℝ) (γ1 : Vec (h * dh)) (X1 : Vec (N * (h * dh)))
    (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh)) (sgelu : Fin N → Vec dff)
    (γ2 : Vec (h * dh)) (X2 : Vec (N * (h * dh))) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  Proofs.residual (rowLNVecFlatBack N (h * dh) ε γ1 X1 ∘ mhsaBackFlat Wq Wk Wv Wo Q K V)
    ∘ Proofs.residual (rowLNVecFlatBack N (h * dh) ε γ2 X2
        ∘ perRowFlatPR N (h * dh) (fun r =>
            Proofs.dense (Mat.transpose W₁) (0 : Vec (h * dh))
              ∘ diagBack (sgelu r)
              ∘ Proofs.dense (Mat.transpose W₂) (0 : Vec dff)))

/-- The block's attention-sublayer output at a flat saved input — the LN₂ site's saved activation,
    and the point the MLP sublayer's backward is taken at. Named because it appears three times in
    `vitBlockBackVAt` and once more in every tie about it. -/
noncomputable def vitAttnOutAt (Np1 heads d_head mlpDim : Nat) (ε : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) (v : Vec (Np1 * (heads * d_head))) :
    Mat Np1 (heads * d_head) :=
  transformerAttnSublayerV Np1 heads d_head ε p.γ1 p.β1 p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo (Mat.unflatten v)

/-- **`vitBlockBackV` with every saved slot pinned to the real forward at the block's own input.**
    The Q/K/V projections at `LN₁(A)`, the LN₁ backward at `A` and the LN₂ backward at the
    attention sublayer's output, the GELU derivative at `dense₁(LN₂(attn A))` — one function of
    the block's flat input `v`, so the tower recursion can be written down. `cnxBlockChBackAt`'s
    shape. -/
noncomputable def vitBlockBackVAt (Np1 heads d_head mlpDim : Nat) (ε : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) (v : Vec (Np1 * (heads * d_head))) :
    Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head)) :=
  vitBlockBackV p.Wq p.Wk p.Wv p.Wo
    (fun r => Proofs.dense p.Wq p.bq
      (layerNormVec (heads * d_head) ε p.γ1 p.β1 (Mat.unflatten v r)))
    (fun r => Proofs.dense p.Wk p.bk
      (layerNormVec (heads * d_head) ε p.γ1 p.β1 (Mat.unflatten v r)))
    (fun r => Proofs.dense p.Wv p.bv
      (layerNormVec (heads * d_head) ε p.γ1 p.β1 (Mat.unflatten v r)))
    ε p.γ1 v p.Wfc1 p.Wfc2
    (fun r => fun c => geluScalarDeriv (Proofs.dense p.Wfc1 p.bfc1
      (layerNormVec (heads * d_head) ε p.γ2 p.β2
        (vitAttnOutAt Np1 heads d_head mlpDim ε p v r)) c))
    p.γ2 (Mat.flatten (vitAttnOutAt Np1 heads d_head mlpDim ε p v))

-- ════════════════════════════════════════════════════════════════
-- § 2. The depth-`k` encoder-tower backward
-- ════════════════════════════════════════════════════════════════

/-- **The depth-`k` encoder-tower backward at a saved tower input `v`.**

    ⚠ **Head-first, like the forward it reverses.** `vitBodyKVFlat (k+1) ps =
    vitBodyKVFlat k (ps ∘ succ) ∘ blockVFlat (ps 0)` runs block `0` FIRST, so the backward applies
    block `0`'s reverse LAST, and the tail's saved input is block `0`'s forward OUTPUT.
    `cnxStageChKBack`'s recursion verbatim, one architecture over.

    ⛔ This is NOT `towerBack` of a list. `towerBack (f :: fs) = towerBack fs ∘ f` applies the head
    FIRST, so a list in block order would run the shallowest block's backward first; the ordering
    was never pinned because the only `towerBack` result in the repo is at `List.replicate`
    (`towerBack_replicate`, uniform blocks, order-invariant). Writing the fold as its own
    recursion is what makes the saved-activation thread visible, and the thread is the content. -/
noncomputable def vitTowerBackK (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    (k : Nat) → (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) →
    Vec (Np1 * (heads * d_head)) → (Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head)))
  | 0, _, _ => id
  | k + 1, ps, v =>
      vitBlockBackVAt Np1 heads d_head mlpDim ε (ps 0) v ∘
        vitTowerBackK Np1 heads d_head mlpDim ε k (fun i => ps i.succ)
          (blockVFlat Np1 heads d_head mlpDim ε (ps 0) v)

-- ════════════════════════════════════════════════════════════════
-- § 3. The two saved prefixes and the whole-net chain
-- ════════════════════════════════════════════════════════════════

/-- The patch-embed output — the encoder tower's saved input. Named as a FUNCTION of the image so
    that the same constant is both the activation the tower's slots are saved at and the `f`
    argument of the chain's first `vjp_comp` (`cnxSavedA0 … cnxSavedA10`'s reason). -/
noncomputable def vitSavedPE (ic H W patchSize N heads d_head : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (x : Vec (ic * H * W)) : Vec ((N + 1) * (heads * d_head)) :=
  patchEmbed_flat ic H W patchSize N (heads * d_head) W_conv b_conv cls_token pos_embed x

/-- The encoder tower's output — the final LayerNorm's saved input. -/
noncomputable def vitSavedBody (ic H W patchSize N mlpDim heads d_head k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (x : Vec (ic * H * W)) : Vec ((N + 1) * (heads * d_head)) :=
  vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps
    (vitSavedPE ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x)

/-- **THE WHOLE-NET ViT-TINY INPUT GRADIENT**, at the depth, head count and LayerNorm spelling the
    net runs. The reverse of `vitForwardKV = classifier_flat ∘ LNᵥ ∘ vitBodyKVFlat ∘ patchEmbed`:

      patchEmbedBack ∘ towerBack ∘ finalLNBack ∘ clsScatter ∘ dense Wclsᵀ

    with every slot concrete — the patch-embed backward is `patchEmbed_input_grad_formula` (which
    IS `patchEmbed_flat_has_vjp`'s backward, definitionally), the head is the free `linBack`
    followed by the CLS scatter, the final LN is `rowLNVecFlatBack` at the tower's output, and the
    tower is `vitTowerBackK`. `vitGradFlat`'s three supplied slots are all discharged.

    A3 = the input gradient at a point. Every factor is linear in the cotangent — a VJP is linear
    at a fixed point — so the whole chain is, which is why its budget is a FOLD and not a cap. -/
noncomputable def vitInputGradK (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF : Vec (heads * d_head)) (Wcls : Mat (heads * d_head) nClasses)
    (x : Vec (ic * H * W)) : Vec nClasses → Vec (ic * H * W) :=
  patchEmbed_input_grad_formula ic H W patchSize N (heads * d_head) W_conv
    ∘ vitTowerBackK (N + 1) heads d_head mlpDim ε k ps
        (vitSavedPE ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x)
    ∘ rowLNVecFlatBack (N + 1) (heads * d_head) ε γF
        (vitSavedBody ic H W patchSize N mlpDim heads d_head k
          W_conv b_conv cls_token pos_embed ε ps x)
    ∘ clsScatter N (heads * d_head)
    ∘ Proofs.dense (Mat.transpose Wcls) (0 : Vec (heads * d_head))

end Proofs
