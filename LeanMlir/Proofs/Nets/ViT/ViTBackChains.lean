import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Architectures.ChannelLNBack
import LeanMlir.Proofs.Nets.ViT.ViTDepthK
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The ViT-Tiny backward chains — the ℝ maps the ViT ties are about

The hand-composed reverse of the committed ViT-Tiny forward, as plain `def`s on the cotangent,
from the attention core outwards:

* the multi-head sdpa backward (`mhSlab`, `mhsaSdpaBackQ`/`K`/`V`): the certified single-head
  `sdpa_back_{Q,K,V}` (`Attention.lean`) on each head slab, concatenated by the
  `finProdFinEquiv` column layout, and its flattened forms `coreQFlat`/`coreKFlat`/`coreVFlat`;
* the full MHSA input-gradient backward `mhsaBackFlat` — output-projection backward, the three
  cores, the three projection backwards fanning in at `X` (`mhsaBackFlat_eq_mhsa_vjp` ties it to
  `mhsa_has_vjp_mat`);
* the encoder-block backward, in three spellings: `vitBlockBack` (one shared LN map per site),
  `vitBlockBackPR` (per-token LN and GELU slots — the form the certified per-token block VJP
  takes, `vitBlockBackPR_eq_transformerBlock_vjp`), and `vitBlockBackV` (the vector-`[D]`
  LayerNorm the shipped net runs, with `rowLNVecFlatBack` in both LN slots);
* `vitBlockBackVAt`, `vitTowerBackK` (the head-first depth-`k` tower fold), the two saved
  prefixes and the whole-net chain `vitInputGradK` — the reverse of `vitForwardKV`, tied by
  `vitInputGradK_eq_vitForwardKV_vjp` (`ViTWholeBackCertifiedTie.lean`);
* `clsScatter`, the CLS-slice adjoint the head backward scatters through.

**Convention table for the net `vitInputGradK` reverses** (`vitForwardKV` at ViT-Tiny, read from
`apps/baselines/MainVitTrain.lean` and `LeanMlir/Proofs/Codegen/ViTRender.lean`): depth 12 with
distinct per-block parameters, `D = 192 = 3 heads × 64`, MLP dim 768, 197 tokens (196 patches +
CLS), `16×16/s16` patchify (no `conv2d`, so no padding phase and none of the even-kernel question
`EvenKernelConvBack.lean` found for ConvNeXt), vector-`[D]` LayerNorm at all 25 sites, GELU,
`ε = 1e-5`. No BatchNorm anywhere. ⚠ `N` throughout is the TOKEN count, not a batch; the batch is
`B`, the binder of `vitInputGradKB` below — the per-example chain lifted stage by stage over `B`
examples, the way the batched T3 tie (`ViTStepTieGB`) lifts every activation and cotangent — tied
by `vitInputGradKB_eq_batchMap_vitForwardKV_vjp` (`ViTWholeBackCertifiedTieB.lean`).

Moved here from the three float bridges that defined them beside their float twins on 2026-09-08
(`planning/archive/float_second_pass.md`); no number is stated about any of these chains. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The multi-head sdpa backward — the certified single-head adjoint on each head slab
-- ════════════════════════════════════════════════════════════════

/-- The column slab `[hd·dh, (hd+1)·dh)` of a `Mat n (h·dh)` as a `Mat n dh` (head hd's view) — the
    `finProdFinEquiv (hd, ·)` column restriction, matching `mhsa_layer`'s per-head extraction. -/
noncomputable def mhSlab {n h dh : Nat} (hd : Fin h) (Q : Mat n (h * dh)) : Mat n dh :=
  fun i c => Q i (finProdFinEquiv (hd, c))

/-- **Multi-head sdpa backward w.r.t. V** — per head, the certified `sdpa_back_V` on the head
    slabs; concatenated by the `finProdFinEquiv` column layout. -/
noncomputable def mhsaSdpaBackV {h N dh : Nat} (Q K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => sdpa_back_V N dh (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
              (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. Q.** -/
noncomputable def mhsaSdpaBackQ {h N dh : Nat} (Q K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => sdpa_back_Q N dh (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
              (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. K.** -/
noncomputable def mhsaSdpaBackK {h N dh : Nat} (Q K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => sdpa_back_K N dh (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
              (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

variable {h N dh : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The flattened cores and the full MHSA backward (cotangent dY ↦ input gradient dX)
-- ════════════════════════════════════════════════════════════════

/-- Flattened multi-head sdpa backward w.r.t. V (saved projections `Q K V` fixed). -/
noncomputable def coreVFlat (Q K V : Mat N (h * dh)) (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (mhsaSdpaBackV Q K V (Mat.unflatten v))

/-- Flattened multi-head sdpa backward w.r.t. Q. -/
noncomputable def coreQFlat (Q K V : Mat N (h * dh)) (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (mhsaSdpaBackQ Q K V (Mat.unflatten v))

/-- Flattened multi-head sdpa backward w.r.t. K. -/
noncomputable def coreKFlat (Q K V : Mat N (h * dh)) (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (mhsaSdpaBackK Q K V (Mat.unflatten v))

/-- **The full multi-head self-attention input-gradient backward** (cotangent `dY ↦ dX`):
    output-projection backward (`dense Woᵀ 0`, per token) → the three sdpa cores → Q/K/V projection
    backwards (`dense Wᵀ 0`, per token), fanning in at `X` (the three paths add). The certified
    MHSA backward at the input (`mhsa_layer`, `Attention.lean`) is
    `dconcat = dY·Woᵀ`, `(dQ, dK, dV) = sdpa_back(dconcat)` per head,
    `dX = dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ`; `mhsaBackFlat_eq_mhsa_vjp` says this chain is that. -/
noncomputable def mhsaBackFlat (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (Q K V : Mat N (h * dh)) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  (fun dconcat j =>
      (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wq) (0 : Vec (h * dh))) ∘ coreQFlat Q K V)
        dconcat j
      + ((perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wk) (0 : Vec (h * dh))) ∘ coreKFlat Q K V)
          dconcat j
        + (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wv) (0 : Vec (h * dh))) ∘ coreVFlat Q K V)
          dconcat j))
    ∘ perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) (0 : Vec (h * dh)))

-- ════════════════════════════════════════════════════════════════
-- § The encoder-block backward, in its three spellings
-- ════════════════════════════════════════════════════════════════

/-- **The ViT encoder-block input-gradient backward** — the reverse of `LN → MHSA → +x → LN → MLP → +x`.
    The block is `mlpResidual ∘ attnSub` (forward), so the backward is `attnSubBack ∘ mlpResidualBack`:

    * **MLP-residual backward** (per token): `residual (LN₂-back ∘ dense W₁ᵀ ∘ geluBack ∘ dense W₂ᵀ)`
      — the reverse of `dense W₂ ∘ gelu ∘ dense W₁ ∘ LN₂`, lifted over the sequence (`perRowFlat`);
    * **attention-sublayer backward**: `residual (LN₁-back ∘ mhsaBackFlat)` — the residual skip's
      cotangent flows both through the MHSA backward and directly to `x`.

    The LN backwards (`lnB₁`/`lnB₂`) are supplied as one shared map per site; `geluBack` is the
    saved-derivative `diagBack`. The per-token form is `vitBlockBackPR`. -/
noncomputable def vitBlockBack {dff : Nat} (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (Q K V : Mat N (h * dh))
    (lnB₁ : Vec (h * dh) → Vec (h * dh)) (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh))
    (sgelu : Vec dff) (lnB₂ : Vec (h * dh) → Vec (h * dh)) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  Proofs.residual (perRowFlat N (h * dh) lnB₁ ∘ mhsaBackFlat Wq Wk Wv Wo Q K V)
    ∘ perRowFlat N (h * dh) (Proofs.residual
        (lnB₂ ∘ Proofs.dense (Mat.transpose W₁) (0 : Vec (h * dh)) ∘ diagBack sgelu
          ∘ Proofs.dense (Mat.transpose W₂) (0 : Vec dff)))

/-- **The per-token-input-aware ViT encoder-block backward** — the enrichment of
    `vitBlockBack` whose LayerNorm and GELU slots thread *each token's* saved activation
    (`perRowFlatPR` instead of `perRowFlat`). Structurally identical to `vitBlockBack`
    (residual MLP-sublayer back, then residual attention-sublayer back), but `lnB₁`/`lnB₂`
    are now per-token *families* `Fin N → (Vec → Vec)` and the GELU derivative `sgelu` is a
    per-token family `Fin N → Vec dff`. This is the form the certified per-token block VJP
    actually takes: `layerNorm_per_token_has_vjp_mat.backward A` runs the single-token LN
    backward at each token's own saved input `A r` (its Jacobian differs per token), which a
    single shared `lnB₁` cannot carry. `vitBlockBack` is the special case of all rows sharing
    one map; this is the general one `vitBlockBackPR_eq_transformerBlock_vjp` equals. -/
noncomputable def vitBlockBackPR {dff : Nat} (Wq Wk Wv Wo : Mat (h * dh) (h * dh))
    (Q K V : Mat N (h * dh))
    (lnB₁ : Fin N → (Vec (h * dh) → Vec (h * dh))) (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh))
    (sgelu : Fin N → Vec dff) (lnB₂ : Fin N → (Vec (h * dh) → Vec (h * dh))) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  Proofs.residual (perRowFlatPR N (h * dh) lnB₁ ∘ mhsaBackFlat Wq Wk Wv Wo Q K V)
    ∘ perRowFlatPR N (h * dh) (fun r => Proofs.residual
        (lnB₂ r ∘ Proofs.dense (Mat.transpose W₁) (0 : Vec (h * dh)) ∘ diagBack (sgelu r)
          ∘ Proofs.dense (Mat.transpose W₂) (0 : Vec dff)))

/-- **The vector-LayerNorm ViT encoder-block input-gradient backward.** `vitBlockBackPR` with its
    two per-token LN slots replaced by `rowLNVecFlatBack` at the site's flat saved input, and the
    MLP sublayer's residual lifted out of the per-token fold (`perRowFlatPR_residual`: a per-row
    `residual` is the row lift plus the cotangent, so the two spellings agree).

    Reading right to left, this is the reverse of `transformerBlockV`: the MLP sublayer's
    `dense W₂ ∘ gelu ∘ dense W₁ ∘ LN₂` backward under a residual, then the attention sublayer's
    `mhsa ∘ LN₁` backward under a residual. ⛔ `vitBlockBackPR`'s tie is at the SCALAR LayerNorm
    (`γ β : ℝ`); the shipped depth-12 net is `vitForwardKV`, whose blocks are `transformerBlockV`
    at `γ β : Vec D`, and this is the spelling the whole-net tie is stated at. -/
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
-- § The depth-`k` encoder-tower backward
-- ════════════════════════════════════════════════════════════════

/-- **The depth-`k` encoder-tower backward at a saved tower input `v`.**

    ⚠ **Head-first, like the forward it reverses.** `vitBodyKVFlat (k+1) ps =
    vitBodyKVFlat k (ps ∘ succ) ∘ blockVFlat (ps 0)` runs block `0` FIRST, so the backward applies
    block `0`'s reverse LAST, and the tail's saved input is block `0`'s forward OUTPUT.
    `cnxStageChKBack`'s recursion verbatim, one architecture over. Writing the fold as its own
    recursion (rather than a list fold) is what makes the saved-activation thread visible, and the
    thread is the content. -/
noncomputable def vitTowerBackK (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    (k : Nat) → (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) →
    Vec (Np1 * (heads * d_head)) → (Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head)))
  | 0, _, _ => id
  | k + 1, ps, v =>
      vitBlockBackVAt Np1 heads d_head mlpDim ε (ps 0) v ∘
        vitTowerBackK Np1 heads d_head mlpDim ε k (fun i => ps i.succ)
          (blockVFlat Np1 heads d_head mlpDim ε (ps 0) v)

-- ════════════════════════════════════════════════════════════════
-- § The endpoints: the CLS-slice scatter, the two saved prefixes, and the whole-net chain
-- ════════════════════════════════════════════════════════════════

/-- **The CLS-slice backward** — the adjoint of `cls_slice_flat` (gather row 0 of the `(N+1)×D`
    sequence): scatter the head cotangent `dy` back to row 0 (the CLS token), zero on the patch rows.
    The certified `cls_slice_flat_has_vjp.backward`. -/
noncomputable def clsScatter (N D : Nat) (dy : Vec D) : Vec ((N + 1) * D) :=
  fun idx =>
    if (finProdFinEquiv.symm idx).1 = (0 : Fin (N + 1)) then dy (finProdFinEquiv.symm idx).2 else 0

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
    IS `patchEmbed_flat_has_vjp`'s backward, definitionally), the head is the free `dense Wᵀ 0`
    followed by the CLS scatter, the final LN is `rowLNVecFlatBack` at the tower's output, and the
    tower is `vitTowerBackK`. `vitInputGradK_eq_vitForwardKV_vjp` is the apex that says this is the
    certified whole-net gradient. -/
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

-- ═══════════════════════════════════════════════════════════════
-- § The batched whole-net chain — a variable batch `B`; `N` stays the token count
-- ═══════════════════════════════════════════════════════════════

/-- The batched patch embedding: `StableHLO.batchMap B` of `vitSavedPE`, the tower's saved input at
    every example. -/
noncomputable def vitSavedPEB (B ic H W patchSize N heads d_head : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (x : Vec (B * (ic * H * W))) : Vec (B * ((N + 1) * (heads * d_head))) :=
  StableHLO.batchMap B
    (vitSavedPE ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed) x

/-- The batched tower output: `StableHLO.batchMap B` of the tower at the batched patch embedding —
    saved stage by stage, not `batchMap B` of the composed per-example prefix (the two agree only
    up to `batchMap_comp`, `ViTWholeBackCertifiedTieB.lean`). -/
noncomputable def vitSavedBodyB (B ic H W patchSize N mlpDim heads d_head k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (x : Vec (B * (ic * H * W))) : Vec (B * ((N + 1) * (heads * d_head))) :=
  StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)
    (vitSavedPEB B ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x)

/-- **THE BATCHED WHOLE-NET ViT INPUT GRADIENT** — `vitInputGradK` at each of `B` examples, stage
    by stage, as the batched render computes it. The head backward and the CLS scatter are
    `StableHLO.batchMap B` of their per-example leaves (both input-independent); the final-LN and
    tower backwards are `StableHLO.batchMapAux B` of their per-example maps, each at the batched
    saved activation (`vitSavedBodyB`, `vitSavedPEB`); the patch-embed backward is `batchMap B` of
    the linear formula. Every slot is a lift because no ViT op couples examples — the same
    honesty argument `ViTStepTieGB` makes for the batched T3 tie. `B` is a variable: this chain
    carries no batch numeral. `vitInputGradKB_eq_batchMap_vitForwardKV_vjp`
    (`ViTWholeBackCertifiedTieB.lean`) says it IS the certified gradient of
    `batchMap B vitForwardKV`. -/
noncomputable def vitInputGradKB (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF : Vec (heads * d_head)) (Wcls : Mat (heads * d_head) nClasses)
    (x : Vec (B * (ic * H * W))) : Vec (B * nClasses) → Vec (B * (ic * H * W)) :=
  StableHLO.batchMap B
      (patchEmbed_input_grad_formula ic H W patchSize N (heads * d_head) W_conv)
  ∘ StableHLO.batchMapAux B (vitTowerBackK (N + 1) heads d_head mlpDim ε k ps)
      (vitSavedPEB B ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x)
  ∘ StableHLO.batchMapAux B (rowLNVecFlatBack (N + 1) (heads * d_head) ε γF)
      (vitSavedBodyB B ic H W patchSize N mlpDim heads d_head k
        W_conv b_conv cls_token pos_embed ε ps x)
  ∘ StableHLO.batchMap B (clsScatter N (heads * d_head))
  ∘ StableHLO.batchMap B (Proofs.dense (Mat.transpose Wcls) (0 : Vec (heads * d_head)))

end Proofs
