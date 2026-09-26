import LeanMlir.Proofs.Nets.ViT.ViTVecLNBackCertifiedTie

/-! # `vitInputGradK` is the certified whole-net ViT gradient

The per-example whole-net backward tie for ViT, the peer of ConvNeXt-T's
`convnextInputGrad_eq_convNextForwardTCh_vjp` and EfficientNet-B0's
`efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp`; the batched ties of the other nets
are `r34InputGradB_eq_r34B_full_vjp` and `mnv2InputGradB_eq_mobilenetv2B_full_vjp`, and ViT's own
batched form is `ViTWholeBackCertifiedTieB.lean`. It is stated generically and instantiated at
ViT-Tiny's configuration: depth 12 with distinct per-block parameters, `D = 192 = 3 × 64`,
197 tokens, vector-`[D]` LayerNorm.

Four things assemble it, and only the second is a proof rather than an enumeration:

1. **The three endpoint leaf ties.** The classifier head's backward is the transposed dense
   (`dense Wclsᵀ 0`) then the CLS scatter (`vitHeadBack_eq_classifier_vjp`); the final LayerNorm's is `rowLNVecFlatBack` at
   the tower's output (`rowLNVecFlatHasVJP_backward_eq`, ConvNeXt's, since `vitForwardKV`'s
   final-LN witness IS `HasVJPMat.toHasVJP (layerNormVecPerTokenHasVJPMat …)`); and the
   patch embed's is `patchEmbedInputGradFormula`, which `patchEmbedFlatHasVJP` gives as its
   backward *definitionally* — so that endpoint is `rfl` and needs no lemma at all.
2. **The depth-`k` tower fold** (`vitTowerBackK_eq_vjp`), the one real proof. `vitBodyKVFlat`'s
   `HasVJP` is built head-first (block `0` runs first), so the backward composes the block
   backwards in the opposite order, each at its own saved activation, and the tail's saved input
   is block `0`'s forward output. `cnxStageChKBack_eq_vjp`'s induction verbatim: one rewrite of
   the block tie and one of the inductive hypothesis.
3. **The apex witness.** `vitApexVJP` names the committed `vitForwardKVHasVJP`; its
   `.backward` reduces through the four `vjpComp` factors by `rfl`.
4. **A shape check.** `vitForwardKV_eq_chain` says the four-factor composition the apex is stated
   at IS the committed `vitForwardKV`, by `rfl`.

The only hypothesis is `0 < ε`; the result holds at every input and every cotangent. ViT has no
kink anywhere — softmax, GELU and LayerNorm are all smooth — so, like ConvNeXt-T and
EfficientNet-B0 and unlike ResNet-34 / MobileNetV2, this is a global `HasVJP` and not a
smooth-point `HasVJPAt`, with no smoothness witness. The forward `vitForwardKV` has no drop-path
and is in exact arithmetic.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 1. The endpoint leaf ties
-- ════════════════════════════════════════════════════════════════

/-- **The classifier-head backward tie.** `clsScatter ∘ dense Wclsᵀ 0` — the head dense's free
    input-VJP, then the CLS-slice scatter — IS `(classifierFlatHasVJP …).backward x`, at every
    saved input (the head is linear, so the backward does not depend on `x`). -/
theorem vitHeadBack_eq_classifier_vjp (n D nClasses : Nat) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) (x : Vec ((n + 1) * D)) :
    clsScatter n D ∘ Proofs.dense (Mat.transpose Wcls) (0 : Vec D)
      = (classifierFlatHasVJP n D nClasses Wcls bcls).backward x := by
  funext dy
  simp only [Function.comp_apply, dense_transpose_eq_mulVec]
  rfl

/-- **The final-LayerNorm backward tie.** `vitForwardKV`'s final-LN witness is
    `HasVJPMat.toHasVJP (layerNormVecPerTokenHasVJPMat …)`, which is exactly
    `rowLNVecFlatHasVJP` — so ConvNeXt's `rowLNVecFlatHasVJP_backward_eq` is this tie, read
    at the ViT token layout. -/
theorem vitFinalLNBack_eq_vjp (n D : Nat) (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (X : Vec (n * D)) :
    rowLNVecFlatBack n D ε γF X
      = (HasVJPMat.toHasVJP (layerNormVecPerTokenHasVJPMat n D ε γF βF hε)).backward X := by
  funext dy
  exact (rowLNVecFlatHasVJP_backward_eq ε hε γF βF X dy).symm

-- ════════════════════════════════════════════════════════════════
-- § 2. ⭐ The depth-`k` tower fold — the one real proof
-- ════════════════════════════════════════════════════════════════

/-- **The tower-fold tie.** The hand-composed depth-`k` encoder-tower backward is
    `(vitBodyKVFlatHasVJP k ps).backward`. Induction on `k`: the base is `identityHasVJP`'s
    `fun _ dy => dy`, and the step is one rewrite of the inductive hypothesis at the shifted saved
    activation (`blockVFlat (ps 0) v`, block `0`'s OUTPUT) and one of the block tie at `v`. -/
theorem vitTowerBackK_eq_vjp (Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
      (v : Vec (Np1 * (heads * d_head))),
      vitTowerBackK Np1 heads d_head mlpDim ε k ps v
        = (vitBodyKVFlatHasVJP Np1 heads d_head mlpDim ε hε k ps).backward v
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
-- § 3. The apex witness, and the shape check
-- ════════════════════════════════════════════════════════════════

/-- **The whole-net witness the apex is stated at** — the committed `vitForwardKVHasVJP`, whose
    `.backward` is the four-factor `vjpComp` chain by `rfl`. -/
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
  vitForwardKVHasVJP ic H W patchSize N mlpDim heads d_head nClasses k
    W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls

/-- The four-factor composition the apex is stated at is the committed `vitForwardKV`, by `rfl`. -/
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
      = classifierFlat N (heads * d_head) nClasses Wcls bcls
          ∘ (fun v : Vec ((N + 1) * (heads * d_head)) =>
              Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) n)))
          ∘ vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps
          ∘ patchEmbedFlat ic H W patchSize N (heads * d_head)
              W_conv b_conv cls_token pos_embed := rfl

/-- **`vitInputGradK` IS the apex witness's backward.** Four rewrites, one per factor: the
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
  show patchEmbedInputGradFormula ic H W patchSize N (heads * d_head) W_conv
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

/-- **The apex.** `vitInputGradK` — the whole-net ViT-Tiny input gradient, every slot pinned
    to the certified per-op backward at its own saved activation — IS
    `(vitForwardKVHasVJP …).backward x`, the committed depth-12 witness — `vitApexVJP` by name. -/
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
      = (vitForwardKVHasVJP ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x :=
  vitInputGradK_eq_vitApexVJP (βF := βF) (bcls := bcls) ic H W patchSize N mlpDim heads
    d_head nClasses k W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x

/-- **The apex, read as the Jacobian.** `vitInputGradK` is the `pdiv`-contracted Jacobian
    transpose of the committed `vitForwardKV`, at EVERY image and EVERY cotangent — the ViT peer
    of `efficientnetInputGradBFull_correct`. Only `0 < ε`. -/
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
  exact vitForwardKVHasVJP_correct ic H W patchSize N mlpDim heads d_head nClasses k
    W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls x dy i

-- ════════════════════════════════════════════════════════════════
-- § 4. The production capstone — ViT-Tiny at its real dimensions
-- ════════════════════════════════════════════════════════════════

/-- **ViT-Tiny's whole-net backward tie.**
    `vitInputGradK_eq_vitForwardKV_vjp` instantiated at the exact ViT-Tiny
    spec: a `3×224×224` image, `16×16` patches (196 patch tokens + CLS), `D = 192 = 3 heads × 64`,
    MLP dim 768, **12 transformer blocks with DISTINCT per-block parameters**, vector-`[D]`
    LayerNorm at all 25 sites, and Imagenette's 10 classes.

    So the hand-written input-gradient chain IS the certified gradient of the committed depth-12
    forward, at every image.
    The backward peer of `vitTinyHasVJP_correct`, beside
    `convnextInputGrad_eq_convNextForwardTCh_vjp` and
    `efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp`. -/
theorem vitTinyInputGrad_eq_vitTiny_vjp
    (W_conv : Kernel4 (3 * 64) 3 16 16) (b_conv : Vec (3 * 64)) (cls_token : Vec (3 * 64))
    (pos_embed : Mat (196 + 1) (3 * 64)) (ε : ℝ) (hε : 0 < ε)
    (ps : Fin 12 → BlockParamsV (3 * 64) 768) (γF βF : Vec (3 * 64))
    (Wcls : Mat (3 * 64) 10) (bcls : Vec 10) (x : Vec (3 * 224 * 224)) :
    vitInputGradK 3 224 224 16 196 768 3 64 10 12
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (vitForwardKVHasVJP 3 224 224 16 196 768 3 64 10 12
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x :=
  vitInputGradK_eq_vitForwardKV_vjp (βF := βF) (bcls := bcls) 3 224 224 16 196 768 3 64 10 12
    W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x

end Proofs
