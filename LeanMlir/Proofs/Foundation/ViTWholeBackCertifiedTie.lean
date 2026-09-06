import LeanMlir.Proofs.Architectures.ViTVecLNBackCertifiedTie

/-! # ⭐⭐ `vitInputGradK` IS the certified whole-net ViT-Tiny gradient

The ViT peer of `r34InputGrad_eq_resnet34_vjp`, `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp`,
`convnextInputGrad_eq_convNextForwardTCh_vjp` and
`efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp` — tier T6 of
`planning/proofs_tier_to_paper_nets.md`, at ViT-Tiny's shipped configuration: depth 12 with
distinct per-block parameters, `D = 192 = 3 × 64`, 197 tokens, vector-`[D]` LayerNorm.

Four things assemble it, and only the second is a proof rather than an enumeration:

1. **The three endpoint leaf ties.** The classifier head's backward is the free `linBack` then the
   CLS scatter (`vitHeadBack_eq_classifier_vjp`); the final LayerNorm's is `rowLNVecFlatBack` at
   the tower's output (`rowLNVecFlat_has_vjp_backward_eq`, ConvNeXt's, since `vitForwardKV`'s
   final-LN witness IS `hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat …)`); and the
   patch embed's is `patchEmbed_input_grad_formula`, which `patchEmbed_flat_has_vjp` gives as its
   backward *definitionally* — so that endpoint is `rfl` and needs no lemma at all.
2. ⭐ **The depth-`k` tower fold** (`vitTowerBackK_eq_vjp`), the one real proof. `vitBodyKVFlat`'s
   `HasVJP` is built head-first (block `0` runs first), so the backward composes the block
   backwards in the OPPOSITE order, each at its own saved activation, and the tail's saved input
   is block `0`'s forward OUTPUT. `cnxStageChKBack_eq_vjp`'s induction verbatim: one rewrite of
   the block tie and one of the inductive hypothesis.
3. **A term-mode apex.** `vitForwardKV_has_vjp` is tactic-built and opens with `unfold
   vitForwardKV`, so its `.backward` sits behind an `Eq.mpr` the kernel will not reduce through.
   `vitApexVJP` is the same four-factor `vjp_comp` chain written as a term, and
   `HasVJP.backward_unique` carries the tie from it to the committed witness — the escape
   `EfficientNetFullWholeBackCertifiedTie.lean` used, and the `▸`-transport trap it names.
4. **A shape check.** `vitForwardKV_eq_chain` says the four-factor composition the apex is stated
   at IS the committed `vitForwardKV`, by `rfl`.

⭐ The result is UNCONDITIONAL except `0 < ε`, at EVERY input and every cotangent. ViT has no kink
anywhere — softmax, GELU and LayerNorm are all smooth — so like ConvNeXt-T and unlike ResNet-34 /
MobileNetV2 / EfficientNet-B0 this is a `HasVJP` and not a smooth-point `HasVJPAt`, and it carries
no operating point, no batch size and no smoothness witness. 3-axiom-clean.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 1. The endpoint leaf ties
-- ════════════════════════════════════════════════════════════════

/-- **The classifier-head backward tie.** `clsScatter ∘ dense Wclsᵀ 0` — the head dense's free
    input-VJP, then the CLS-slice scatter — IS `(classifier_flat_has_vjp …).backward x`, at every
    saved input (the head is linear, so the backward does not depend on `x`). -/
theorem vitHeadBack_eq_classifier_vjp (n D nClasses : Nat) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) (x : Vec ((n + 1) * D)) :
    clsScatter n D ∘ Proofs.dense (Mat.transpose Wcls) (0 : Vec D)
      = (classifier_flat_has_vjp n D nClasses Wcls bcls).backward x := by
  funext dy
  simp only [Function.comp_apply, dense_transpose_eq_mulVec]
  rfl

/-- **The final-LayerNorm backward tie.** `vitForwardKV`'s final-LN witness is
    `hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat …)`, which is exactly
    `rowLNVecFlat_has_vjp` — so ConvNeXt's `rowLNVecFlat_has_vjp_backward_eq` is this tie, read
    at the ViT token layout. -/
theorem vitFinalLNBack_eq_vjp (n D : Nat) (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (X : Vec (n * D)) :
    rowLNVecFlatBack n D ε γF X
      = (hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat n D ε γF βF hε)).backward X := by
  funext dy
  exact (rowLNVecFlat_has_vjp_backward_eq ε hε γF βF X dy).symm

/-- **The patch-embed backward tie is `rfl`.** `patchEmbed_flat_has_vjp` is a structure whose
    `backward` field IS `patchEmbed_input_grad_formula` — the one endpoint in this repo that
    needed no reconciliation. -/
theorem vitPatchEmbedBack_eq_vjp (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D) (cls_token : Vec D)
    (pos_embed : Mat (N + 1) D) (x : Vec (ic * H * W)) :
    patchEmbed_input_grad_formula ic H W patchSize N D W_conv
      = (patchEmbed_flat_has_vjp ic H W patchSize N D W_conv b_conv cls_token
          pos_embed).backward x := rfl

-- ════════════════════════════════════════════════════════════════
-- § 2. ⭐ The depth-`k` tower fold — the one real proof
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **THE TOWER-FOLD TIE.** The hand-composed depth-`k` encoder-tower backward IS
    `(vitBodyKVFlat_has_vjp k ps).backward`. Induction on `k`: the base is `identity_has_vjp`'s
    `fun _ dy => dy`, and the step is one rewrite of the inductive hypothesis at the shifted saved
    activation (`blockVFlat (ps 0) v`, block `0`'s OUTPUT) and one of the block tie at `v`. -/
theorem vitTowerBackK_eq_vjp (Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
      (v : Vec (Np1 * (heads * d_head))),
      vitTowerBackK Np1 heads d_head mlpDim ε k ps v
        = (vitBodyKVFlat_has_vjp Np1 heads d_head mlpDim ε hε k ps).backward v
  | 0, _, _ => rfl
  | k + 1, ps, v => by
      show vitBlockBackVAt Np1 heads d_head mlpDim ε (ps 0) v ∘
        vitTowerBackK Np1 heads d_head mlpDim ε k (fun i => ps i.succ)
          (blockVFlat Np1 heads d_head mlpDim ε (ps 0) v) = _
      rw [vitTowerBackK_eq_vjp Np1 heads d_head mlpDim ε hε k (fun i => ps i.succ)
            (blockVFlat Np1 heads d_head mlpDim ε (ps 0) v),
          vitBlockBackVAt_eq_vjp Np1 heads d_head mlpDim ε hε (ps 0) v]
      rfl

-- ════════════════════════════════════════════════════════════════
-- § 3. A term-mode apex, and the shape check
-- ════════════════════════════════════════════════════════════════

/-- **The whole-net witness as a TERM.** The same four-factor `vjp_comp` chain
    `vitForwardKV_has_vjp` builds, written without the leading `unfold` — so `.backward` is a
    projection the kernel reduces, where the committed witness's sits behind an `Eq.mpr`. The
    two are tied by `HasVJP.backward_unique` below; nothing here is a second definition of the
    gradient, it is the same one spelled so that it computes. -/
noncomputable def vitApexVJP
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) :=
  vjp_comp _ (classifier_flat N (heads * d_head) nClasses Wcls bcls)
    ((layerNormVec_per_token_flat_diff (N + 1) (heads * d_head) ε γF βF hε).comp
      ((vitBodyKVFlat_diff (N + 1) heads d_head mlpDim ε hε k ps).comp
        (patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed)))
    (classifier_flat_diff N (heads * d_head) nClasses Wcls bcls)
    (vjp_comp _ _
      ((vitBodyKVFlat_diff (N + 1) heads d_head mlpDim ε hε k ps).comp
        (patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed))
      (layerNormVec_per_token_flat_diff (N + 1) (heads * d_head) ε γF βF hε)
      (vjp_comp _ _
        (patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed)
        (vitBodyKVFlat_diff (N + 1) heads d_head mlpDim ε hε k ps)
        (patchEmbed_flat_has_vjp ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed)
        (vitBodyKVFlat_has_vjp (N + 1) heads d_head mlpDim ε hε k ps))
      (hasVJPMat_to_hasVJP
        (layerNormVec_per_token_has_vjp_mat (N + 1) (heads * d_head) ε γF βF hε)))
    (classifier_flat_has_vjp N (heads * d_head) nClasses Wcls bcls)

/-- The four-factor composition the apex is stated at IS the committed `vitForwardKV`. The shape
    check `Resnet34BackCertifiedTie.lean` lacked and ConvNeXt wrote before anyone needed it. -/
theorem vitForwardKV_eq_chain
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls
      = classifier_flat N (heads * d_head) nClasses Wcls bcls
          ∘ (fun v : Vec ((N + 1) * (heads * d_head)) =>
              Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) n)))
          ∘ vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps
          ∘ patchEmbed_flat ic H W patchSize N (heads * d_head)
              W_conv b_conv cls_token pos_embed := rfl

/-- **`vitInputGradK` IS the term-mode apex's backward.** Four rewrites, one per factor: the
    tower fold, the final-LN tie, the head tie, and the patch-embed endpoint (`rfl`). -/
theorem vitInputGradK_eq_vitApexVJP
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    vitInputGradK ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (vitApexVJP ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x := by
  funext dy
  show patchEmbed_input_grad_formula ic H W patchSize N (heads * d_head) W_conv
      (vitTowerBackK (N + 1) heads d_head mlpDim ε k ps
        (vitSavedPE ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x)
        (rowLNVecFlatBack (N + 1) (heads * d_head) ε γF
          (vitSavedBody ic H W patchSize N mlpDim heads d_head k
            W_conv b_conv cls_token pos_embed ε ps x)
          (clsScatter N (heads * d_head)
            (Proofs.dense (Mat.transpose Wcls) (0 : Vec (heads * d_head)) dy)))) = _
  rw [vitTowerBackK_eq_vjp (N + 1) heads d_head mlpDim ε hε k ps _,
      vitFinalLNBack_eq_vjp (βF := βF) (N + 1) (heads * d_head) ε hε γF _]
  simp only [dense_transpose_eq_mulVec]
  rfl

/-- ⭐⭐ **THE APEX.** `vitInputGradK` — the whole-net ViT-Tiny input gradient, every slot pinned
    to the certified per-op backward at its own saved activation — IS
    `(vitForwardKV_has_vjp …).backward x`, the committed depth-12 witness. Carried from the
    term-mode apex by `HasVJP.backward_unique`: two witnesses for one map have one backward. -/
theorem vitInputGradK_eq_vitForwardKV_vjp
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    vitInputGradK ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (vitForwardKV_has_vjp ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x := by
  funext dy
  rw [vitInputGradK_eq_vitApexVJP (βF := βF) (bcls := bcls) ic H W patchSize N mlpDim heads
        d_head nClasses k W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x]
  exact HasVJP.backward_unique _ _ x dy

/-- **The apex, read as the Jacobian.** `vitInputGradK` is the `pdiv`-contracted Jacobian
    transpose of the committed `vitForwardKV`, at EVERY image and EVERY cotangent — the ViT peer
    of `efficientnetInputGradB_full_correct`. Only `0 < ε`. -/
theorem vitInputGradK_correct
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) (i : Fin (ic * H * W)) :
    vitInputGradK ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x dy i
      = ∑ j : Fin nClasses,
          pdiv (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) x i j * dy j := by
  rw [vitInputGradK_eq_vitForwardKV_vjp (βF := βF) (bcls := bcls) ic H W patchSize N mlpDim
        heads d_head nClasses k W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x]
  exact vitForwardKV_has_vjp_correct ic H W patchSize N mlpDim heads d_head nClasses k
    W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls x dy i

-- ════════════════════════════════════════════════════════════════
-- § 4. The production capstone — ViT-Tiny at its real dimensions
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **ViT-Tiny's whole-net backward tie — tier T6 at the paper net.**
    `vitInputGradK_eq_vitForwardKV_vjp` instantiated at the exact `MainVitTrain.lean` `vitTiny`
    spec: a `3×224×224` image, `16×16` patches (196 patch tokens + CLS), `D = 192 = 3 heads × 64`,
    MLP dim 768, **12 transformer blocks with DISTINCT per-block parameters**, vector-`[D]`
    LayerNorm at all 25 sites, and Imagenette's 10 classes.

    So the hand-written float-tier input-gradient chain — the one `ViTBackFloatBudget.lean` puts a
    numeral on — IS the certified gradient of the committed depth-12 forward, at every image.
    The backward peer of `vitTiny_has_vjp_correct`, and ViT's entry in the T6 column beside
    `convnextInputGrad_eq_convNextForwardTCh_vjp` and
    `efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp`. -/
theorem vitTinyInputGrad_eq_vitTiny_vjp
    (W_conv : Kernel4 (3 * 64) 3 16 16) (b_conv : Vec (3 * 64)) (cls_token : Vec (3 * 64))
    (pos_embed : Mat (196 + 1) (3 * 64)) (ε : ℝ) (hε : 0 < ε)
    (ps : Fin 12 → BlockParamsV (3 * 64) 768) (γF βF : Vec (3 * 64))
    (Wcls : Mat (3 * 64) 10) (bcls : Vec 10) (x : Vec (3 * 224 * 224)) :
    vitInputGradK 3 224 224 16 196 768 3 64 10 12
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (vitForwardKV_has_vjp 3 224 224 16 196 768 3 64 10 12
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x :=
  vitInputGradK_eq_vitForwardKV_vjp (βF := βF) (bcls := bcls) 3 224 224 16 196 768 3 64 10 12
    W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x

end Proofs
