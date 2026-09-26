import LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.BatchMapVJPAt

/-! # `vitInputGradKB` is the certified whole-net ViT gradient at a batch

`ViTWholeBackCertifiedTie.lean` proves for one image that `vitInputGradK`, the reverse of
`vitForwardKV` over that image's `N + 1` tokens, is the certified gradient. The batched step tie
(`ViTStepTieGB.lean`) states every activation as `StableHLO.batchMap B` of the per-example prefix
and every cotangent as `batchMapAux B` of the per-example chain, because no ViT op couples
examples. This file proves the same at that index: the five-stage batched chain `vitInputGradKB`
(`ViTBackChains.lean`) is the certified gradient of `batchMap B vitForwardKV` at every batch `x`,
for every `B`. The forward `vitForwardKV` has no drop-path and is in exact arithmetic.

ViT is smooth everywhere: every stage has a global `HasVJP`, so its batched witness is
`batchMapHasVJPAt` (the field-by-field lift, `BatchMapVJPAt.lean`) over `HasVJP.toHasVJPAt` at
each row. The only hypothesis is `0 < ε`; there is no smooth-point condition.

1. `batchMap_comp` (`BatchMapVJPAt.lean`, shared with ConvNeXt's batched tie) — `batchMap B
   (g ∘ f) = batchMap B g ∘ batchMap B f`, the lemma the shape check needs and the reason the
   chain saves its activations stage by stage: the two spellings agree only up to
   `finProdFinEquiv.symm_apply_apply`, which is not `rfl`.
2. The batched leaf ties. The patch-embed stage needs none (its backward is `rfl`); the other
   three are `batchMapAux_eq_batchMapHasVJPAt` (the head: its `batchMap` form, after
   `batchMap_comp`) over the per-example tie at each row (`vitTowerBackK_eq_vjp`,
   `vitFinalLNBack_eq_vjp`, `vitHeadBack_eq_classifier_vjp`).
3. `vitKVBHasVJPAt` — the four-stage apex, three `vjpCompDiffAt`s over the batched stage
   witnesses — and `vitInputGradKB_eq_vitKVB_vjp`, the tie: three leaf rewrites, then `rfl`.
4. `vitForwardKVB_eq_chain` — the shape check: the four batched stages compose to
   `batchMap B vitForwardKV`, by `vitForwardKV_eq_chain` and three `batchMap_comp`s — and
   `vitInputGradKB_eq_batchMap_vitForwardKV_vjp`, the tie carried to the committed GLOBAL
   witness `batchMapHasVJP (vitForwardKV …)` through `HasVJPAt.backward_unique_of_eq`
   (`batchMapHasVJP` is `▸`-transported, so its `.backward` does not reduce), plus the `∑ pdiv`
   reading.
5. `vitTinyInputGradB_eq_vitTiny_vjp` — the capstone at ViT-Tiny's literal dims, `B` a binder.
-/

namespace Proofs

open scoped BigOperators

-- ═════════════════════════════════════════════════
-- § The four batched stage witnesses, and their leaf ties
-- ═════════════════════════════════════════════════

/-- The batched patch-embed witness at `x`: `batchMapHasVJPAt` over the global per-example
    witness at each row. -/
noncomputable def vitEmbedBAt (B ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D)
    (pos_embed : Mat (N + 1) D) (x : Vec (B * (ic * H * W))) :
    HasVJPAt (StableHLO.batchMap B
      (patchEmbedFlat ic H W patchSize N D W_conv b_conv cls_token pos_embed)) x :=
  batchMapHasVJPAt _ x
    (fun _ => (patchEmbedFlatHasVJP ic H W patchSize N D W_conv b_conv cls_token
      pos_embed).toHasVJPAt _)
    (fun _ => (patchEmbedFlat_differentiable ic H W patchSize N D W_conv b_conv cls_token
      pos_embed).differentiableAt)

/-- The batched tower witness at `v`. -/
noncomputable def vitTowerBAt (B Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) (k : Nat)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) (v : Vec (B * (Np1 * (heads * d_head)))) :
    HasVJPAt (StableHLO.batchMap B (vitBodyKVFlat Np1 heads d_head mlpDim ε k ps)) v :=
  batchMapHasVJPAt _ v
    (fun _ => (vitBodyKVFlatHasVJP Np1 heads d_head mlpDim ε hε k ps).toHasVJPAt _)
    (fun _ => (vitBodyKVFlat_differentiable Np1 heads d_head mlpDim ε hε k ps).differentiableAt)

/-- The batched final-LayerNorm witness at `v`. -/
noncomputable def vitLNBAt (B n D : Nat) (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (v : Vec (B * (n * D))) :
    HasVJPAt (StableHLO.batchMap B (fun v : Vec (n * D) =>
      Mat.flatten (fun r => layerNormVec D ε γF βF ((Mat.unflatten v) r)))) v :=
  batchMapHasVJPAt _ v
    (fun _ => (HasVJPMat.toHasVJP
      (layerNormVecPerTokenHasVJPMat n D ε γF βF hε)).toHasVJPAt _)
    (fun _ => (layerNormVec_per_token_flat_differentiable n D ε γF βF hε).differentiableAt)

/-- The batched classifier-head witness at `v`. -/
noncomputable def vitHeadBAt (B N D nClasses : Nat) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) (v : Vec (B * ((N + 1) * D))) :
    HasVJPAt (StableHLO.batchMap B (classifierFlat N D nClasses Wcls bcls)) v :=
  batchMapHasVJPAt _ v
    (fun _ => (classifierFlatHasVJP N D nClasses Wcls bcls).toHasVJPAt _)
    (fun _ => (classifierFlat_differentiable N D nClasses Wcls bcls).differentiableAt)

/-- **The batched tower tie.** `batchMapAux B` of the depth-`k` tower backward at the batched
    saved input IS the lift's backward: `vitTowerBackK_eq_vjp` at each row. -/
theorem vitTowerBackB_eq_vjp (B Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) (k : Nat)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) (v : Vec (B * (Np1 * (heads * d_head)))) :
    StableHLO.batchMapAux B (vitTowerBackK Np1 heads d_head mlpDim ε k ps) v
      = (vitTowerBAt B Np1 heads d_head mlpDim ε hε k ps v).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ v _ _ fun _ =>
    vitTowerBackK_eq_vjp Np1 heads d_head mlpDim ε hε k ps _

/-- **The batched final-LayerNorm tie.** `batchMapAux B` of `rowLNVecFlatBack` at the batched
    tower output IS the lift's backward — `vitFinalLNBack_eq_vjp` at one example's row. -/
theorem vitLNBackB_eq_vjp (B n D : Nat) (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (v : Vec (B * (n * D))) :
    StableHLO.batchMapAux B (rowLNVecFlatBack n D ε γF) v
      = (vitLNBAt B n D ε hε γF βF v).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ v _ _ fun _ => vitFinalLNBack_eq_vjp n D ε hε γF βF _

/-- **The batched head tie.** `batchMap B` of the CLS scatter after `batchMap B` of the free dense
    backward IS the lift's backward at any saved `v` (the head is linear): fuse the two lifts by
    `batchMap_comp`, then `vitHeadBack_eq_classifier_vjp` at one example's row. -/
theorem vitHeadBackB_eq_vjp (B N D nClasses : Nat) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) (v : Vec (B * ((N + 1) * D))) :
    StableHLO.batchMap B (clsScatter N D)
        ∘ StableHLO.batchMap B (Proofs.dense (Mat.transpose Wcls) (0 : Vec D))
      = (vitHeadBAt B N D nClasses Wcls bcls v).backward := by
  rw [← batchMap_comp]
  exact batchMap_eq_batchMapHasVJPAt _ _ v _ _ fun _ =>
    vitHeadBack_eq_classifier_vjp N D nClasses Wcls bcls _

-- ═════════════════════════════════════════════════
-- § The batched apex, the tie, and the shape check
-- ═════════════════════════════════════════════════

/-- **The batched whole-net witness**, four batched stages composed by `vjpCompDiffAt`, each
    at the batched saved activation the chain uses (`vitSavedPEB`, `vitSavedBodyB`). -/
noncomputable def vitKVBHasVJPAt (B ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head)) (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) :
    HasVJPAt
      (StableHLO.batchMap B (classifierFlat N (heads * d_head) nClasses Wcls bcls)
        ∘ StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
            Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))
        ∘ StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)
        ∘ StableHLO.batchMap B (patchEmbedFlat ic H W patchSize N (heads * d_head)
            W_conv b_conv cls_token pos_embed)) x :=
  (vjpCompDiffAt _ (StableHLO.batchMap B (classifierFlat N (heads * d_head) nClasses Wcls bcls)) x
    (vjpCompDiffAt _
      (StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
        Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))) x
      (vjpCompDiffAt
        (StableHLO.batchMap B (patchEmbedFlat ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed))
        (StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)) x
        ⟨vitEmbedBAt B ic H W patchSize N (heads * d_head) W_conv b_conv cls_token pos_embed x,
         batchMap_differentiableAt _ x (fun _ => (patchEmbedFlat_differentiable ic H W patchSize N
           (heads * d_head) W_conv b_conv cls_token pos_embed).differentiableAt)⟩
        ⟨vitTowerBAt B (N + 1) heads d_head mlpDim ε hε k ps
           (vitSavedPEB B ic H W patchSize N heads d_head W_conv b_conv cls_token pos_embed x),
         batchMap_differentiableAt _ _ (fun _ => (vitBodyKVFlat_differentiable (N + 1) heads d_head mlpDim
           ε hε k ps).differentiableAt)⟩)
      ⟨vitLNBAt B (N + 1) (heads * d_head) ε hε γF βF
         (vitSavedBodyB B ic H W patchSize N mlpDim heads d_head k
           W_conv b_conv cls_token pos_embed ε ps x),
       batchMap_differentiableAt _ _ (fun _ => (layerNormVec_per_token_flat_differentiable (N + 1)
         (heads * d_head) ε γF βF hε).differentiableAt)⟩)
    ⟨vitHeadBAt B N (heads * d_head) nClasses Wcls bcls _,
     batchMap_differentiableAt _ _ (fun _ => (classifierFlat_differentiable N (heads * d_head) nClasses
       Wcls bcls).differentiableAt)⟩).fst

/-- **The batched tie.** `vitInputGradKB` — the five-stage batched chain, every slot a lift of
    the per-example backward at the batched saved activation — is the backward of
    `vitKVBHasVJPAt`. Three leaf rewrites (tower, final LN, head), then `rfl`: the patch-embed leaf
    is definitional. -/
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
      = (vitKVBHasVJPAt B ic H W patchSize N mlpDim heads d_head nClasses k
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
      = StableHLO.batchMap B (classifierFlat N (heads * d_head) nClasses Wcls bcls)
        ∘ StableHLO.batchMap B (fun v : Vec ((N + 1) * (heads * d_head)) =>
            Mat.flatten (fun r => layerNormVec (heads * d_head) ε γF βF ((Mat.unflatten v) r)))
        ∘ StableHLO.batchMap B (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps)
        ∘ StableHLO.batchMap B (patchEmbedFlat ic H W patchSize N (heads * d_head)
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
  exact (classifierFlat_differentiable N (heads * d_head) nClasses Wcls bcls).comp
    ((layerNormVec_per_token_flat_differentiable (N + 1) (heads * d_head) ε γF βF hε).comp
      ((vitBodyKVFlat_differentiable (N + 1) heads d_head mlpDim ε hε k ps).comp
        (patchEmbedFlat_differentiable ic H W patchSize N (heads * d_head)
          W_conv b_conv cls_token pos_embed)))

/-- **The batched apex at the global witness.** `vitInputGradKB` is
    `(batchMapHasVJP (vitForwardKV …) …).backward x` — the certified gradient of the drop-free
    per-example forward lifted over `B` examples, under `0 < ε`. Carried from the chain-shaped apex
    by `HasVJPAt.backward_unique_of_eq` along the shape check. -/
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
      = (batchMapHasVJP (N := B)
          (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
          (vitForwardKVHasVJP ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)
          (vitForwardKV_differentiable ic H W patchSize N mlpDim heads d_head nClasses k
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)).backward x := by
  funext dy
  rw [vitInputGradKB_eq_vitKVB_vjp (βF := βF) (bcls := bcls) B ic H W patchSize N mlpDim heads
        d_head nClasses k W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x]
  exact HasVJPAt.backward_unique_of_eq
    (vitForwardKVB_eq_chain B ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls).symm
    (vitKVBHasVJPAt B ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls x)
    ((batchMapHasVJP (N := B)
        (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
        (vitForwardKVHasVJP ic H W patchSize N mlpDim heads d_head nClasses k
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
  exact (batchMapHasVJP (N := B) _ _ _).correct x dy i

-- ═════════════════════════════════════════════════
-- § The production capstone — ViT-Tiny at its real dimensions, `B` a binder
-- ═════════════════════════════════════════════════

/-- **ViT-Tiny's batched whole-net backward tie.**
    `vitInputGradKB_eq_batchMap_vitForwardKV_vjp` at the exact `vitTiny` spec
    (`3×224×224`, `16×16` patches, 196 + CLS tokens, `D = 192 = 3 × 64`, MLP 768, 12 distinct
    blocks, vector-`[D]` LayerNorm, 10 classes), at a variable batch `B` (the `vitin_*` artifacts
    run 32, 128 or 256 per device; 512 is the global batch of the `128x4` runs). The forward is the
    drop-free `vitForwardKV` in exact arithmetic; the `*drop*` artifacts compute another function.
    The batched peers of the other nets include `r34InputGradB_eq_r34B_full_vjp` and
    `mnv2InputGradB_eq_mobilenetv2B_full_vjp`. -/
theorem vitTinyInputGradB_eq_vitTiny_vjp (B : Nat)
    (W_conv : Kernel4 (3 * 64) 3 16 16) (b_conv : Vec (3 * 64)) (cls_token : Vec (3 * 64))
    (pos_embed : Mat (196 + 1) (3 * 64)) (ε : ℝ) (hε : 0 < ε)
    (ps : Fin 12 → BlockParamsV (3 * 64) 768) (γF βF : Vec (3 * 64))
    (Wcls : Mat (3 * 64) 10) (bcls : Vec 10) (x : Vec (B * (3 * 224 * 224))) :
    vitInputGradKB B 3 224 224 16 196 768 3 64 10 12
        W_conv b_conv cls_token pos_embed ε ps γF Wcls x
      = (batchMapHasVJP (N := B)
          (vitForwardKV 3 224 224 16 196 768 3 64 10 12
            W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
          (vitForwardKVHasVJP 3 224 224 16 196 768 3 64 10 12
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)
          (vitForwardKV_differentiable 3 224 224 16 196 768 3 64 10 12
            W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls)).backward x :=
  vitInputGradKB_eq_batchMap_vitForwardKV_vjp (βF := βF) (bcls := bcls) B 3 224 224 16 196 768
    3 64 10 12 W_conv b_conv cls_token pos_embed ε hε ps γF Wcls x

end Proofs
