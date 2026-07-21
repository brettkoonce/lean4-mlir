import LeanMlir.Proofs.Float.ViTBlockFloatBridge
import LeanMlir.Proofs.Float.MhsaBackFloatBridge

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

-- ════════════════════════════════════════════════════════════════
-- § The forward tie — `vit_full` IS `vitForwardFlat` at concrete blocks
-- ════════════════════════════════════════════════════════════════

/-- `towerBack` of `k` copies of one block is the `k`-fold iterate: the head-first list fold
    (`towerBack (g :: gs) = towerBack gs ∘ g`) over `List.replicate k g` is `g^[k]`. -/
theorem towerBack_replicate {m : Nat} (g : Vec m → Vec m) (k : Nat) :
    towerBack (List.replicate k g) = g^[k] := by
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [List.replicate_succ]
      show towerBack (List.replicate k g) ∘ g = g^[k + 1]
      rw [ih, Function.iterate_succ]

/-- **The flattened transformer tower is the `k`-fold iterate of the flattened block.** The encoder
    tower `transformerTower k` shares one parameter tuple across blocks, so its flatten/unflatten
    conjugation is exactly `blockFlat^[k]` — the structural bridge between the real-net `Nat.rec` fold
    and the `towerBack`/`List.replicate` fold of `vitForwardFlat`. Induction on `k`: the base is the
    `Mat.flatten_unflatten` round-trip; the step pushes `Mat.unflatten_flatten` through one block and
    matches `Function.iterate_succ'` (`f^[k+1] = f ∘ f^[k]`, block applied last, as in the tower). -/
theorem transformerTower_flatten_eq_iterate
    (k N heads d_head mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    (fun v : Vec (N * (heads * d_head)) =>
       Mat.flatten (transformerTower k N heads d_head mlpDim ε γ1 β1
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v)))
      = (fun v : Vec (N * (heads * d_head)) =>
           Mat.flatten (transformerBlock N heads d_head mlpDim ε γ1 β1
             Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v)))^[k] := by
  induction k with
  | zero =>
      funext v
      simp only [Function.iterate_zero, id_eq]
      show Mat.flatten (Mat.unflatten v) = v
      exact Mat.flatten_unflatten v
  | succ k ih =>
      show (fun v : Vec (N * (heads * d_head)) =>
             Mat.flatten (((transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
               (transformerTower k N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
              (Mat.unflatten v)))
          = (fun v : Vec (N * (heads * d_head)) =>
               Mat.flatten (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v)))^[k + 1]
      have h_eq : (fun v : Vec (N * (heads * d_head)) =>
             Mat.flatten (((transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
               (transformerTower k N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
              (Mat.unflatten v)))
          = (fun u : Vec (N * (heads * d_head)) =>
               Mat.flatten (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten u)))
            ∘ (fun v : Vec (N * (heads * d_head)) =>
                 Mat.flatten (transformerTower k N heads d_head mlpDim ε γ1 β1
                   Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
        funext v; simp [Function.comp, Mat.unflatten_flatten]
      rw [h_eq, ih, ← Function.iterate_succ']

/-- **THE ViT FORWARD TIE.** The committed real-net `vit_full` IS the `vitForwardFlat` skeleton with
    the final LayerNorm, the `kBlocks` (shared-parameter) flattened transformer blocks, and the
    patch-embed plugged into its supplied slots — the ViT peer of `r34Forward`/`mnv2Forward`/
    `convnextForward`'s `_eq_skeleton` ties. Unlike those (pure `rfl`), ViT needs a genuine
    decomposition: `flatten ∘ vit_body ∘ unflatten = perRowFlat finalLN ∘ towerBack blocks`. Since
    `vit_body = (per-token finalLN) ∘ transformerTower`, the LN rides a `perRowFlat`/`Mat.unflatten_flatten`
    reindex (`hmid`'s final `simp`), and the tower's `Nat.rec` fold is reconciled with the
    `towerBack`/`List.replicate` fold through `transformerTower_flatten_eq_iterate` +
    `towerBack_replicate` (both routed via `Function.iterate`). So `vit_floatBridges` provably bounds
    the actual `vit_full`, not a look-alike skeleton. -/
theorem vit_full_eq_vitForwardFlat
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    vit_full ic H W patchSize N mlpDim heads d_head kBlocks nClasses
        W_conv b_conv cls_token pos_embed ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls
      = vitForwardFlat Wcls bcls
          (layerNormForward (heads * d_head) ε γF βF)
          (List.replicate kBlocks
            (fun v : Vec ((N + 1) * (heads * d_head)) =>
               Mat.flatten (transformerBlock (N + 1) heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))))
          (patchEmbed_flat ic H W patchSize N (heads * d_head)
            W_conv b_conv cls_token pos_embed) := by
  have hmid :
      (fun v : Vec ((N + 1) * (heads * d_head)) =>
         Mat.flatten (vit_body kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
           Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF (Mat.unflatten v)))
        = perRowFlat (N + 1) (heads * d_head) (layerNormForward (heads * d_head) ε γF βF)
          ∘ towerBack (List.replicate kBlocks
              (fun v : Vec ((N + 1) * (heads * d_head)) =>
                 Mat.flatten (transformerBlock (N + 1) heads d_head mlpDim ε γ1 β1
                   Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v)))) := by
    rw [towerBack_replicate,
        ← transformerTower_flatten_eq_iterate kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
            Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2]
    funext v
    show Mat.flatten (vit_body kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
           Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF (Mat.unflatten v))
       = perRowFlat (N + 1) (heads * d_head) (layerNormForward (heads * d_head) ε γF βF)
           (Mat.flatten (transformerTower kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
             Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v)))
    unfold vit_body perRowFlat
    simp only [Function.comp_apply, Mat.unflatten_flatten]
  unfold vit_full vitForwardFlat
  rw [hmid, Function.comp_assoc]

end Proofs
