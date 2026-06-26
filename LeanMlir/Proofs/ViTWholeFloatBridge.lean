import LeanMlir.Proofs.ViTBlockFloatBridge
import LeanMlir.Proofs.MhsaBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE ViT FORWARD — the encoder-tower fold

The forward peer of `vit_grad_floatBridges` (`MhsaBackFloatBridge.lean`). The repo had the ViT forward
float story only at *block* level (`floatBridges_vitBlock`, `ViTBlockFloatBridge.lean`) while the
backward already folded the whole net; this closes that asymmetry by assembling the forward whole net
in the SAME blueprint the backward uses.

`vit_full = classifier ∘ (flatten ∘ vit_body ∘ unflatten) ∘ patchEmbed`, and the backward decomposes
the body into `finalLN ∘ tower`. So the forward whole net is

  `classifier ∘ perRowFlat finalLN ∘ tower blocks ∘ patchEmbed`

— the exact reverse of `vitGradFlat = patchEmbedBack ∘ towerBack ∘ perRowFlat finalLNBack ∘ clsScatter
∘ linBack Wcls`. The classifier head (`dense ∘ cls-slice`) is **concrete**; the per-row final LN, the
`k` transformer blocks, and the patch-embed are **supplied as `FloatBridges`** (each separately
dischargeable — the blocks by the pre-existing `floatBridges_vitBlock`, the LN by `floatBridges_bn`,
the patch-embed by the forward strided-conv/scatter machinery), exactly as `vit_grad_floatBridges`
supplies its `blockBacks`/`finalLNBack`/`patchEmbedBack` around concrete endpoints.

Two reuses make this thin: the **encoder tower** is `towerBack` (its fold composes a list head-first,
which IS the forward order — direction-agnostic), discharged by the existing `floatBridges_towerBack`;
and the **per-row LN** rides `FloatBridges.perRow`. The one new forward op-bridge is the cls-slice
gather `floatBridges_clsSlice` (the forward peer of `clsScatter`): `cls_slice_flat` reads row 0 of the
`(N+1)×D` sequence — exact in float, magnitude-stable (`B = A`, modulus id).
-/

namespace Proofs

open scoped Real

-- ════════════════════════════════════════════════════════════════
-- § The cls-slice gather as a `FloatBridges`  (the head's first stage)
-- ════════════════════════════════════════════════════════════════

/-- **The cls-slice gather is `FloatClose`** with modulus `id` — exact (real = float), magnitude-stable
    (`B = A`): `cls_slice_flat N D v k = v (row 0, col k)` reads a single input entry, so `|·| ≤ A` and
    `|vt(·) - va(·)| ≤ e` carry over verbatim. The forward peer of `floatClose_clsScatter`. -/
theorem floatClose_clsSlice (N D : Nat) {A : ℝ} :
    FloatClose A A (cls_slice_flat N D) (cls_slice_flat N D) (fun e => e) :=
  ⟨fun _v hv k => ⟨hv (finProdFinEquiv ((0 : Fin (N + 1)), k)),
                   hv (finProdFinEquiv ((0 : Fin (N + 1)), k))⟩,
   fun _vt _va _e _ _ hd k => hd (finProdFinEquiv ((0 : Fin (N + 1)), k))⟩

/-- The cls-slice gather float-bridges (magnitude-stable). -/
theorem floatBridges_clsSlice (N D : Nat) : FloatBridges (cls_slice_flat N D) :=
  fun A hA => ⟨A, _, _, hA, floatClose_clsSlice N D⟩

/-- **The ViT classifier head float-bridges** — the forward peer of `floatBridges_vitHeadBack`.
    `classifier_flat = dense Wcls bcls ∘ cls_slice_flat`: gather the CLS token (row 0), then the dense
    classifier. -/
theorem floatBridges_vitHead {N D nClasses : Nat} (M : FloatModel)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses)
    {w' β : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hD : 0 < D)
    (hWcls : ∀ i j, |Wcls i j| ≤ w') (hbcls : ∀ j, |bcls j| ≤ β) :
    FloatBridges (classifier_flat N D nClasses Wcls bcls) := by
  unfold classifier_flat
  exact (floatBridges_clsSlice N D).comp (floatBridges_dense M Wcls bcls hw' hβ hD hWcls hbcls)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net forward (the encoder-tower fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole ViT forward — the structural skeleton of `vit_full` (reverse of `vitGradFlat`):
    `classifier ∘ perRowFlat finalLN ∘ tower blocks ∘ patchEmbed`. The classifier head (`dense ∘
    cls-slice`) is concrete; the per-row final LN, the `k` transformer blocks, and the patch-embed are
    supplied (each `FloatBridges`, discharged by `floatBridges_vitBlock` / `floatBridges_bn` / the
    forward patch-embed machinery). The forward peer of `vitGradFlat`. The encoder tower is `towerBack`
    — its head-first list fold IS the forward order, so the existing `floatBridges_towerBack` discharges
    it unchanged. -/
noncomputable def vitForwardFlat {N D nClasses imgDim : Nat}
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) (finalLN : Vec D → Vec D)
    (blocks : List (Vec ((N + 1) * D) → Vec ((N + 1) * D)))
    (patchEmbed : Vec imgDim → Vec ((N + 1) * D)) : Vec imgDim → Vec nClasses :=
  classifier_flat N D nClasses Wcls bcls
    ∘ perRowFlat (N + 1) D finalLN
    ∘ towerBack blocks
    ∘ patchEmbed

/-- **THE WHOLE-NET ViT FORWARD FLOAT-BRIDGES.** One `.comp` thread over the supplied patch-embed, the
    encoder tower (`floatBridges_towerBack` over the per-block bridges), the supplied per-row final LN,
    and the concrete classifier head — the forward peer of `vit_grad_floatBridges`, the
    `r34_grad_floatBridges` blueprint for ViT. The deployed float forward of the whole transformer is
    within an explicit budget of the certified `ℝ` forward. Closes under `[propext, Classical.choice,
    Quot.sound]`. -/
theorem vit_floatBridges {N D nClasses imgDim : Nat} (M : FloatModel)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) (finalLN : Vec D → Vec D)
    (blocks : List (Vec ((N + 1) * D) → Vec ((N + 1) * D)))
    (patchEmbed : Vec imgDim → Vec ((N + 1) * D))
    {w' β : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hD : 0 < D)
    (hWcls : ∀ i j, |Wcls i j| ≤ w') (hbcls : ∀ j, |bcls j| ≤ β)
    (hFinalLN : FloatBridges finalLN) (hblocks : ∀ f ∈ blocks, FloatBridges f)
    (hPatch : FloatBridges patchEmbed) :
    FloatBridges (vitForwardFlat Wcls bcls finalLN blocks patchEmbed) := by
  unfold vitForwardFlat
  exact (((hPatch.comp (floatBridges_towerBack blocks hblocks)).comp
    (FloatBridges.perRow (N + 1) hFinalLN)).comp
    (floatBridges_vitHead M Wcls bcls hw' hβ hD hWcls hbcls))

end Proofs
