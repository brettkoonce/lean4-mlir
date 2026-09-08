import LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.BatchMapVJPAt

/-! # ⭐⭐ `vitInputGradKB` IS the certified whole-net ViT-Tiny gradient AT A BATCH

`ViTWholeBackCertifiedTie.lean` closed T6 for ONE image: `vitInputGradK`, the reverse of
`vitForwardKV` over that image's `N + 1` tokens, IS the certified gradient. Every shipped ViT
artifact runs a batch — `vit_adam_train_step` and the `vitin_*` family at 128 or 512 per
device — and its batched T3 tie (`ViTStepTieGB.lean`) states every activation as
`StableHLO.batchMap B` of the per-example prefix and every cotangent as `batchMapAux B` of the
per-example chain, because no ViT op couples examples. This file closes T6 at that index: the
five-stage batched chain `vitInputGradKB` (`ViTBackChains.lean`) IS the certified gradient of
`batchMap B vitForwardKV` at every batch `x`, for every `B`. ViT was the one net without a
batched whole-net tie; with this the `*InputGradB_eq_*_vjp` family covers all seven nets.

Nothing here is new mathematics, and it is lighter than ResNet-34's batched tie because ViT is
smooth everywhere: every stage has a GLOBAL `HasVJP`, so its batched witness is
`batchMap_has_vjp_at` (4.1c's field-by-field lift, `BatchMapVJPAt.lean`) over
`HasVJP.toHasVJPAt` at each row — no smooth-point hypothesis anywhere, only `0 < ε`.

1. `batchMap_comp` — `batchMap B (g ∘ f) = batchMap B g ∘ batchMap B f`, the lemma the shape
   check needs and the reason the chain saves its activations stage by stage: the two spellings
   agree only up to `finProdFinEquiv.symm_apply_apply`, which is not `rfl`.
2. The four batched leaf ties, one per stage. The patch-embed one is `rfl` (its per-example tie
   is); the other three are `funext` to one example, one rewrite of the per-example tie at that
   example's row (`vitTowerBackK_eq_vjp`, `vitFinalLNBack_eq_vjp`,
   `vitHeadBack_eq_classifier_vjp`), then `rfl` — `batchMapAux`'s slice and the lift's
   `.backward` row are the same term, as `maxPool3s2FlatBackB_eq_vjp_backward` found for r34.
3. `vitKVB_has_vjp_at` — the four-stage apex, three `vjp_comp_diff_at`s over the batched stage
   witnesses — and `vitInputGradKB_eq_vitKVB_vjp`, the tie: three leaf rewrites, then `rfl`.
4. `vitForwardKVB_eq_chain` — the shape check: the four batched stages compose to
   `batchMap B vitForwardKV`, by `vitForwardKV_eq_chain` and three `batchMap_comp`s — and
   `vitInputGradKB_eq_batchMap_vitForwardKV_vjp`, the tie carried to the committed GLOBAL
   witness `batchMap_has_vjp (vitForwardKV …)` through `HasVJPAt.backward_unique_of_eq`
   (`batchMap_has_vjp` is `▸`-transported, so its `.backward` does not reduce; uniqueness is the
   escape every whole-net tie in this repo takes), plus the `∑ pdiv` reading.
5. `vitTinyInputGradB_eq_vitTiny_vjp` — the capstone at ViT-Tiny's literal dims, `B` a binder.
-/

namespace Proofs

open scoped BigOperators

-- ═════════════════════════════════════════════════
-- § `batchMap` distributes over composition
-- ═════════════════════════════════════════════════

/-- **`batchMap B (g ∘ f) = batchMap B g ∘ batchMap B f`.** Both sides read example `p.1`'s slice
    of the input and run `g ∘ f` on it; peeling the inner lift off at one example is
    `batchSlice_batchMap`. -/
theorem batchMap_comp (B : Nat) {a b c : Nat} (f : Vec a → Vec b) (g : Vec b → Vec c) :
    StableHLO.batchMap B (g ∘ f) = StableHLO.batchMap B g ∘ StableHLO.batchMap B f := by
  funext x idx
  show g (f (StableHLO.batchSlice B a x (finProdFinEquiv.symm idx).1)) (finProdFinEquiv.symm idx).2
    = g (StableHLO.batchSlice B b (StableHLO.batchMap B f x) (finProdFinEquiv.symm idx).1)
        (finProdFinEquiv.symm idx).2
  rw [StableHLO.batchSlice_batchMap]

/-- Two `HasVJPAt` witnesses for EQUAL maps at one point have the same backward — the pointwise
    peer of `HasVJP.backward_unique_of_eq`, through `.correct` rather than a transport. -/
theorem HasVJPAt.backward_unique_of_eq {m n : Nat} {f g : Vec m → Vec n} {x : Vec m}
    (hfg : f = g) (h₁ : HasVJPAt f x) (h₂ : HasVJPAt g x) (dy : Vec n) :
    h₁.backward dy = h₂.backward dy := by
  subst hfg
  funext i
  rw [h₁.correct, h₂.correct]

-- ═════════════════════════════════════════════════
-- § The four batched stage witnesses, and their leaf ties
-- ═════════════════════════════════════════════════

/-- The batched patch-embed witness at `x`: `batchMap_has_vjp_at` over the global per-example
    witness at each row. -/
noncomputable def vitEmbedB_at (B ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D)
    (pos_embed : Mat (N + 1) D) (x : Vec (B * (ic * H * W))) :
    HasVJPAt (StableHLO.batchMap B
      (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed)) x :=
  batchMap_has_vjp_at _ x
    (fun _ => (patchEmbed_flat_has_vjp ic H W patchSize N D W_conv b_conv cls_token
      pos_embed).toHasVJPAt _)
    (fun _ => (patchEmbed_flat_diff ic H W patchSize N D W_conv b_conv cls_token
      pos_embed).differentiableAt)

/-- The batched tower witness at `v`. -/
noncomputable def vitTowerB_at (B Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) (k : Nat)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) (v : Vec (B * (Np1 * (heads * d_head)))) :
    HasVJPAt (StableHLO.batchMap B (vitBodyKVFlat Np1 heads d_head mlpDim ε k ps)) v :=
  batchMap_has_vjp_at _ v
    (fun _ => (vitBodyKVFlat_has_vjp Np1 heads d_head mlpDim ε hε k ps).toHasVJPAt _)
    (fun _ => (vitBodyKVFlat_diff Np1 heads d_head mlpDim ε hε k ps).differentiableAt)

/-- The batched final-LayerNorm witness at `v`. -/
noncomputable def vitLNB_at (B n D : Nat) (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (v : Vec (B * (n * D))) :
    HasVJPAt (StableHLO.batchMap B (fun v : Vec (n * D) =>
      Mat.flatten (fun r => layerNormVec D ε γF βF ((Mat.unflatten v) r)))) v :=
  batchMap_has_vjp_at _ v
    (fun _ => (hasVJPMat_to_hasVJP
      (layerNormVec_per_token_has_vjp_mat n D ε γF βF hε)).toHasVJPAt _)
    (fun _ => (layerNormVec_per_token_flat_diff n D ε γF βF hε).differentiableAt)

/-- The batched classifier-head witness at `v`. -/
noncomputable def vitHeadB_at (B N D nClasses : Nat) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) (v : Vec (B * ((N + 1) * D))) :
    HasVJPAt (StableHLO.batchMap B (classifier_flat N D nClasses Wcls bcls)) v :=
  batchMap_has_vjp_at _ v
    (fun _ => (classifier_flat_has_vjp N D nClasses Wcls bcls).toHasVJPAt _)
    (fun _ => (classifier_flat_diff N D nClasses Wcls bcls).differentiableAt)

/-- **The batched patch-embed tie is `rfl`**, as the per-example one is: `batchMap B` of the
    linear formula IS the lift's row-wise backward, term for term. -/
theorem vitEmbedBackB_eq_vjp (B ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D)
    (pos_embed : Mat (N + 1) D) (x : Vec (B * (ic * H * W))) :
    StableHLO.batchMap B (patchEmbed_input_grad_formula ic H W patchSize N D W_conv)
      = (vitEmbedB_at B ic H W patchSize N D W_conv b_conv cls_token pos_embed x).backward := rfl

/-- **The batched tower tie.** `batchMapAux B` of the depth-`k` tower backward at the batched
    saved input IS the lift's backward: one example, one rewrite of `vitTowerBackK_eq_vjp` at
    that example's row, `rfl`. -/
theorem vitTowerBackB_eq_vjp (B Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) (k : Nat)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) (v : Vec (B * (Np1 * (heads * d_head)))) :
    StableHLO.batchMapAux B (vitTowerBackK Np1 heads d_head mlpDim ε k ps) v
      = (vitTowerB_at B Np1 heads d_head mlpDim ε hε k ps v).backward := by
  funext dy idx
  show vitTowerBackK Np1 heads d_head mlpDim ε k ps (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [vitTowerBackK_eq_vjp Np1 heads d_head mlpDim ε hε k ps]
  rfl

/-- **The batched final-LayerNorm tie.** `batchMapAux B` of `rowLNVecFlatBack` at the batched
    tower output IS the lift's backward — `vitFinalLNBack_eq_vjp` at one example's row. -/
theorem vitLNBackB_eq_vjp (B n D : Nat) (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (v : Vec (B * (n * D))) :
    StableHLO.batchMapAux B (rowLNVecFlatBack n D ε γF) v
      = (vitLNB_at B n D ε hε γF βF v).backward := by
  funext dy idx
  show rowLNVecFlatBack n D ε γF (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [vitFinalLNBack_eq_vjp n D ε hε γF βF]
  rfl

/-- **The batched head tie.** `batchMap B` of the CLS scatter after `batchMap B` of the free dense
    backward IS the lift's backward at any saved `v` (the head is linear): fuse the two lifts by
    `batchMap_comp`, then `vitHeadBack_eq_classifier_vjp` at one example's row. -/
theorem vitHeadBackB_eq_vjp (B N D nClasses : Nat) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) (v : Vec (B * ((N + 1) * D))) :
    StableHLO.batchMap B (clsScatter N D)
        ∘ StableHLO.batchMap B (Proofs.dense (Mat.transpose Wcls) (0 : Vec D))
      = (vitHeadB_at B N D nClasses Wcls bcls v).backward := by
  rw [← batchMap_comp]
  funext dy idx
  show (clsScatter N D ∘ Proofs.dense (Mat.transpose Wcls) (0 : Vec D))
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [vitHeadBack_eq_classifier_vjp N D nClasses Wcls bcls
        (Mat.unflatten v (finProdFinEquiv.symm idx).1)]
  rfl

-- ═════════════════════════════════════════════════
-- § The batched apex, the tie, and the shape check
-- ═════════════════════════════════════════════════

/-- **The batched whole-net witness**, four batched stages composed by `vjp_comp_diff_at`, each
    at the batched saved activation the chain uses (`vitSavedPEB`, `vitSavedBodyB`). -/
noncomputable def vitKVB_has_vjp_at (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) :
    HasVJPAt
      (StableHLO.batchMap B (classifier_flat N (heads * d_head) nClasses Wcls bcls)
        ∘ StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
            Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))
        ∘ StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)
        ∘ StableHLO.batchMap B (patchEmbed_flat ic H W patchSize N (heads * d_head)
            W_conv b_conv cls_token pos_embed)) x :=
  (vjp_comp_diff_at _ (StableHLO.batchMap B (classifier_flat N (heads * d_head) nClasses Wcls bcls)) x
    (vjp_comp_diff_at _
      (StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
        Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))) x
      (vjp_comp_diff_at
        (StableHLO.batchMap B (patchEmbed_flat ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed))
        (StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)) x
        ⟨vitEmbedB_at B ic H W patchSize N (heads * d_head) W_conv b_conv cls_token pos_embed x,
         batchMap_differentiableAt _ x (fun _ => (patchEmbed_flat_diff ic H W patchSize N
           (heads * d_head) W_conv b_conv cls_token pos_embed).differentiableAt)⟩
        ⟨vitTowerB_at B (N + 1) heads d_head mlpDim ε hε k ps
           (vitSavedPEB B ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x),
         batchMap_differentiableAt _ _ (fun _ => (vitBodyKVFlat_diff (N + 1) heads d_head mlpDim
           ε hε k ps).differentiableAt)⟩)
      ⟨vitLNB_at B (N + 1) (heads * d_head) ε hε γF βF
         (vitSavedBodyB B ic H W patchSize N mlpDim heads d_head k
           W_conv b_conv cls_token pos_embed ε ps x),
       batchMap_differentiableAt _ _ (fun _ => (layerNormVec_per_token_flat_diff (N + 1)
         (heads * d_head) ε γF βF hε).differentiableAt)⟩)
    ⟨vitHeadB_at B N (heads * d_head) nClasses Wcls bcls _,
     batchMap_differentiableAt _ _ (fun _ => (classifier_flat_diff N (heads * d_head) nClasses
       Wcls bcls).differentiableAt)⟩).fst

/-- ⭐⭐ **THE BATCHED TIE.** `vitInputGradKB` — the five-stage batched chain, every slot a lift of
    the per-example backward at the batched saved activation — IS the batched apex's backward.
    Three leaf rewrites (tower, final LN, head), then `rfl`: the patch-embed leaf is definitional. -/
theorem vitInputGradKB_eq_vitKVB_vjp (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) :
    vitInputGradKB B ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (vitKVB_has_vjp_at B ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls x).backward := by
  unfold vitInputGradKB
  rw [vitTowerBackB_eq_vjp B (N + 1) heads d_head mlpDim ε hε k ps,
      vitLNBackB_eq_vjp B (N + 1) (heads * d_head) ε hε γF βF,
      vitHeadBackB_eq_vjp B N (heads * d_head) nClasses Wcls bcls
        (StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
            Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))
          (vitSavedBodyB B ic H W patchSize N mlpDim heads d_head k
            W_conv b_conv cls_token pos_embed ε ps x))]
  rfl

/-- **The shape check.** The four batched stages the apex is stated at compose to
    `batchMap B vitForwardKV`, the committed per-example forward lifted whole: the per-example
    shape check `vitForwardKV_eq_chain` and three `batchMap_comp`s. -/
theorem vitForwardKVB_eq_chain (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    StableHLO.batchMap B (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
      = StableHLO.batchMap B (classifier_flat N (heads * d_head) nClasses Wcls bcls)
        ∘ StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
            Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))
        ∘ StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)
        ∘ StableHLO.batchMap B (patchEmbed_flat ic H W patchSize N (heads * d_head)
            W_conv b_conv cls_token pos_embed) := by
  rw [vitForwardKV_eq_chain, batchMap_comp, batchMap_comp, batchMap_comp]

/-- `vitForwardKV` is differentiable everywhere (only `0 < ε`): the four stage lemmas composed. -/
theorem vitForwardKV_differentiable (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Differentiable ℝ (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) := by
  rw [vitForwardKV_eq_chain]
  exact (classifier_flat_diff N (heads * d_head) nClasses Wcls bcls).comp
    ((layerNormVec_per_token_flat_diff (N + 1) (heads * d_head) ε γF βF hε).comp
      ((vitBodyKVFlat_diff (N + 1) heads d_head mlpDim ε hε k ps).comp
        (patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed)))

/-- ⭐⭐ **THE APEX, at the committed batched witness.** `vitInputGradKB` IS
    `(batchMap_has_vjp (vitForwardKV …) …).backward x` — the certified gradient of the per-example
    net lifted whole over `B` examples. Carried from the chain-shaped apex by
    `HasVJPAt.backward_unique_of_eq` along the shape check. -/
theorem vitInputGradKB_eq_batchMap_vitForwardKV_vjp
    (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) :
    vitInputGradKB B ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (batchMap_has_vjp (N := B)
          (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
          (vitForwardKV_has_vjp ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)
          (vitForwardKV_differentiable ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)).backward x := by
  funext dy
  rw [vitInputGradKB_eq_vitKVB_vjp (βF := βF) (bcls := bcls) B ic H W patchSize N mlpDim heads
        d_head nClasses k W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x]
  exact HasVJPAt.backward_unique_of_eq
    (vitForwardKVB_eq_chain B ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls).symm
    (vitKVB_has_vjp_at B ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls x)
    ((batchMap_has_vjp (N := B)
        (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
        (vitForwardKV_has_vjp ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)
        (vitForwardKV_differentiable ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)).toHasVJPAt x) dy

/-- **The batched apex, read as the Jacobian.** `vitInputGradKB` is the `pdiv`-contracted Jacobian
    transpose of `batchMap B vitForwardKV`, at EVERY batch and EVERY cotangent. Only `0 < ε`. -/
theorem vitInputGradKB_correct (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) (dy : Vec (B * nClasses)) (i : Fin (B * (ic * H * W))) :
    vitInputGradKB B ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x dy i
      = ∑ j : Fin (B * nClasses),
          pdiv (StableHLO.batchMap B (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)) x i j * dy j := by
  rw [vitInputGradKB_eq_batchMap_vitForwardKV_vjp (βF := βF) (bcls := bcls) B ic H W patchSize N
        mlpDim heads d_head nClasses k W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x]
  exact (batchMap_has_vjp (N := B) _ _ _).correct x dy i

-- ═════════════════════════════════════════════════
-- § The production capstone — ViT-Tiny at its real dimensions, `B` a binder
-- ═════════════════════════════════════════════════

/-- ⭐⭐ **ViT-Tiny's BATCHED whole-net backward tie — tier T6 at the paper net and the shipped
    index.** `vitInputGradKB_eq_batchMap_vitForwardKV_vjp` at the exact `vitTiny` spec
    (`3×224×224`, `16×16` patches, 196 + CLS tokens, `D = 192 = 3 × 64`, MLP 768, 12 distinct
    blocks, vector-`[D]` LayerNorm, 10 classes), at a variable batch `B` — 128 or 512 per device
    in the shipped `vitin_*` artifacts, and neither number appears here. ViT's entry in the batched
    T6 column beside `r34InputGradB_eq_r34B_full_vjp` and `mnv2InputGradB_eq_mobilenetv2B_full_vjp`. -/
theorem vitTinyInputGradB_eq_vitTiny_vjp (B : Nat)
    (W_conv : Kernel4 (3 * 64) 3 16 16) (b_conv : Vec (3 * 64)) (cls_token : Vec (3 * 64))
    (pos_embed : Mat (196 + 1) (3 * 64)) (ε : ℝ) (hε : 0 < ε)
    (ps : Fin 12 → BlockParamsV (3 * 64) 768) (γF βF : Vec (3 * 64))
    (Wcls : Mat (3 * 64) 10) (bcls : Vec 10) (x : Vec (B * (3 * 224 * 224))) :
    vitInputGradKB B 3 224 224 16 196 768 3 64 10 12
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (batchMap_has_vjp (N := B)
          (vitForwardKV 3 224 224 16 196 768 3 64 10 12
            W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
          (vitForwardKV_has_vjp 3 224 224 16 196 768 3 64 10 12
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)
          (vitForwardKV_differentiable 3 224 224 16 196 768 3 64 10 12
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)).backward x :=
  vitInputGradKB_eq_batchMap_vitForwardKV_vjp (βF := βF) (bcls := bcls) B 3 224 224 16 196 768
    3 64 10 12 W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x

end Proofs
