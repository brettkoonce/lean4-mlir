import LeanMlir.Proofs.Float.ResNet34BlockBridge
-- He et al.'s 3×3/s2 stem pool, for `floatClose_maxPool3s2` below. It imports only
-- `Architectures.CNN`, which this file already has transitively (it uses `maxPoolFlat_abs_le`),
-- so this adds no cycle. `planning/archive/rsb_a3_r50_verified.md` §4b.
import LeanMlir.Proofs.Architectures.MaxPool3s2

/-!
# ℝ→Float32 bridge: the composition backbone (whole-net certificate)

A whole-net float certificate is a *fold* of the per-op budgets. `FloatClose`
packages exactly what's needed to fold: on inputs within magnitude `A`, the float
`fF` is within an error modulus `L e` of the real `f` (per coordinate, at input
error `e`), and both real and float outputs are within `B` (so the next layer's
magnitude precondition is met). `FloatClose.comp` proves this **composes** — the
moduli compose as `Lg ∘ Lf`, magnitudes thread `A → B → C` — so a whole net is
`FloatClose` with the composed modulus, no per-net re-proof.

Instances proved here: `relu` (exact in float, modulus `id`) and `flatConv`
(modulus = the conv-fan-in `layerBudget`). The remaining r34 ops are already
`*_close` lemmas and slot in the same way: BN→relu via `bnRelu_close` (use the
operating-point `bnIstd_close_at` for the `eistd`, else the budget is vacuous),
maxpool via `maxPoolFlat_close`, the skip via `reluAdd_close`. The whole-net
certificate is then `.comp` folded over the layer list.
-/

namespace Proofs

open FloatModel

/-- A-posteriori-magnitude, proved-error float closeness, built to compose.
    `A` bounds the inputs (both real `va` and float `vt`), `B` both outputs;
    `L` is the input-error → output-error modulus. -/
def FloatClose {m n : Nat} (A B : ℝ) (f fF : Vec m → Vec n) (L : ℝ → ℝ) : Prop :=
  (∀ v, (∀ k, |v k| ≤ A) → ∀ i, |f v i| ≤ B ∧ |fF v i| ≤ B) ∧
  (∀ vt va e, (∀ k, |va k| ≤ A) → (∀ k, |vt k| ≤ A) → (∀ k, |vt k - va k| ≤ e)
      → ∀ i, |fF vt i - f va i| ≤ L e)

/-- **Float-closeness composes** — the whole-net certificate backbone. Magnitudes
    thread `A → B → C`, error moduli compose `Lg ∘ Lf`. -/
theorem FloatClose.comp {m n p : Nat} {A B C : ℝ}
    {f fF : Vec m → Vec n} {g gF : Vec n → Vec p} {Lf Lg : ℝ → ℝ}
    (hf : FloatClose A B f fF Lf) (hg : FloatClose B C g gF Lg) :
    FloatClose A C (g ∘ f) (gF ∘ fF) (Lg ∘ Lf) := by
  obtain ⟨hfm, hfe⟩ := hf
  obtain ⟨hgm, hge⟩ := hg
  refine ⟨?_, ?_⟩
  · intro v hv i
    exact ⟨(hgm (f v) (fun k => (hfm v hv k).1) i).1,
           (hgm (fF v) (fun k => (hfm v hv k).2) i).2⟩
  · intro vt va e hva hvt hd i
    exact hge (fF vt) (f va) (Lf e)
      (fun k => (hfm va hva k).1) (fun k => (hfm vt hvt k).2)
      (fun k => hfe vt va e hva hvt hd k) i

/-- **The standard way to build a `FloatClose` instance.** A real bound `R` on the box, a
    rounding bound `E` at an exactly-represented input, and the error modulus give
    `FloatClose A (R + E)`: the float output is within `E` of the real one, so its magnitude is
    at most `R + E`. Every per-op instance below with a fresh-input rounding term is this. -/
theorem FloatClose.of_close {m n : Nat} {A R E : ℝ} {f fF : Vec m → Vec n} {L : ℝ → ℝ}
    (hreal : ∀ v, (∀ k, |v k| ≤ A) → ∀ i, |f v i| ≤ R)
    (hround : ∀ v, (∀ k, |v k| ≤ A) → ∀ i, |fF v i - f v i| ≤ E)
    (herr : ∀ vt va e, (∀ k, |va k| ≤ A) → (∀ k, |vt k| ≤ A) → (∀ k, |vt k - va k| ≤ e)
      → ∀ i, |fF vt i - f va i| ≤ L e) :
    FloatClose A (R + E) f fF L := by
  refine ⟨fun v hv i => ?_, herr⟩
  have h1 := hreal v hv i; have h2 := hround v hv i
  have h3 := abs_sub_abs_le_abs_sub (fF v i) (f v i)
  exact ⟨by linarith [abs_nonneg (fF v i - f v i)], by linarith⟩

/-- **ReLU is `FloatClose` with modulus `id`** — exact in float (real = float map),
    1-Lipschitz on the inherited error, never grows magnitudes. -/
theorem floatClose_relu {n : Nat} (A : ℝ) :
    FloatClose A A (relu n) (relu n) (fun e => e) := by
  refine ⟨fun v hv i => ⟨(relu_abs_le v i).trans (hv i), (relu_abs_le v i).trans (hv i)⟩,
          fun vt va e _ _ hd i => relu_close vt va e hd i⟩

/-- **Convolution is `FloatClose`** with modulus the conv-fan-in `layerBudget`.
    Real output ≤ `layerAct`; float output ≤ `layerAct + layerBudget(e=0)` (the
    extra rounding) — that sum is the propagated magnitude `B`. -/
theorem floatClose_flatConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (_hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0)
      (flatConv (h := h) (w := w) W b) (M.flatConvF (h := h) (w := w) W b)
      (fun e => layerBudget M.u (ic * kH * kW) w' β A e) :=
  FloatClose.of_close (fun v hv i => flatConv_abs_le hA hW hb hv i)
    (fun v hv i => M.flatConvF_close W b v v hw' hA le_rfl hW hb hv (fun k => by simp) i)
    (fun vt va e hva _ hd i => M.flatConvF_close W b vt va hw' hA
      ((abs_nonneg _).trans (hd ⟨0, hn⟩)) hW hb hva hd i)

/-- **Dense layer is `FloatClose`** with modulus the fan-in `layerBudget` (the dense
    analogue of `floatClose_flatConv`). Real output ≤ `layerAct`; float output ≤ that
    + the fresh-input rounding `layerBudget(e=0)`. The SE excite/reduce denses and the
    classifier head are this instance; the ViT MLP denses reuse it too. -/
theorem floatClose_dense {m n : Nat} (M : FloatModel) (W : Mat m n) (b : Vec n)
    {w' β A : ℝ} (hw' : 0 ≤ w') (_hβ : 0 ≤ β) (hA : 0 ≤ A) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β) :
    FloatClose A
      (layerAct m w' β A + layerBudget M.u m w' β A 0)
      (Proofs.dense W b) (M.dense W b)
      (fun e => layerBudget M.u m w' β A e) :=
  FloatClose.of_close (fun v hv i => dense_abs_le hA hW hb hv i)
    (fun v hv i => (M.dense_close_fresh W b v i).trans
      (M.denseErr_le_uniform hw' le_rfl hW hb hv i))
    (fun vt va e hva _ hd i => by
      have he : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, hm⟩)
      exact (M.dense_close W b vt va e he hd i).trans (M.denseErr_le_uniform hw' he hW hb hva i))

/-- **Demo: a conv→relu unit is `FloatClose`** — `(conv).comp (relu)` folds the
    conv `layerBudget` modulus and ReLU's `id`. A 2-conv chain
    `relu∘conv∘relu∘conv` is two more `.comp`s; the whole r34 net is this fold
    over its layer list (with the BN/maxpool/skip instances slotted in). -/
theorem floatClose_reluConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0)
      (relu (oc * h * w) ∘ flatConv (h := h) (w := w) W b)
      (relu (oc * h * w) ∘ M.flatConvF (h := h) (w := w) W b)
      ((fun e => e) ∘ (fun e => layerBudget M.u (ic * kH * kW) w' β A e)) :=
  (floatClose_flatConv M W b hw' hβ hA hn hW hb).comp
    (floatClose_relu (layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0))

/-- **MaxPool is `FloatClose` with modulus `id`** — exact in float, 1-Lipschitz,
    never grows magnitudes (`maxPoolFlat_close` / `maxPoolFlat_abs_le`). -/
theorem floatClose_maxPool {c h w : Nat} (A : ℝ) :
    FloatClose A A (maxPoolFlat c h w) (maxPoolFlat c h w) (fun e => e) :=
  ⟨fun _v hv i => ⟨maxPoolFlat_abs_le hv i, maxPoolFlat_abs_le hv i⟩,
   fun vt va _e _ _ hd i => maxPoolFlat_close vt va hd i⟩

/-- ⭐ **He et al.'s 3×3/s2 stem pool is `FloatClose`** — the peer of `floatClose_maxPool`, and
    identical in shape: a max is EXACT (it selects an existing cell, so the modulus is `id` and the
    magnitude is unchanged) whatever the window size. The overlap that makes the *backward*
    accumulate is invisible here, because the forward at one output still reads one cell.
    `planning/archive/rsb_a3_r50_verified.md` §4b. -/
theorem floatClose_maxPool3s2 {c h w : Nat} (A : ℝ) :
    FloatClose A A (maxPool3s2Flat c h w) (maxPool3s2Flat c h w) (fun e => e) :=
  ⟨fun _v hv i => ⟨maxPool3s2Flat_abs_le hv i, maxPool3s2Flat_abs_le hv i⟩,
   fun vt va _e _ _ hd i => maxPool3s2Flat_close vt va hd i⟩

/-- **Global-average-pool is `FloatClose`** — `Vec (c·h·w) → Vec c`, the SE squeeze.
    GAP is a per-channel `bnMean` (`globalAvgPoolFlat_eq_bnMean`), so the real output
    never exceeds the input magnitude `A` (`bnMean_abs_le`) and is 1-Lipschitz in the
    input (`bnMean_input_close`, the spatial mean averages the per-coordinate error
    back to `e`); the float roundoff is `gapFlat_close`'s budget `gb`. Output magnitude
    `A + gb`, modulus `e ↦ gb + e`. -/
theorem floatClose_gap {c h w : Nat} (M : FloatModel) {A : ℝ}
    (_hA0 : 0 ≤ A) (hhw : 0 < h * w) :
    FloatClose A
      (A + (M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A))
      (globalAvgPoolFlat c h w) M.gapFlatF
      (fun e => (M.u * ((1 + M.u) ^ (h * w + 1) * A)
                 + ((1 + M.u) ^ (h * w + 1) - 1) * A) + e) := by
  have hhwR : (0:ℝ) < ((h * w : ℕ) : ℝ) := by exact_mod_cast hhw
  set gb := M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A
    with hgbdef
  refine FloatClose.of_close (fun v hv ci => ?_) (fun v hv ci => ?_) (fun vt va e hva hvt hd ci => ?_)
  · rw [globalAvgPoolFlat_eq_bnMean v ci]
    exact bnMean_abs_le _ hhw (fun s => hv _)
  · rw [hgbdef]; exact M.gapFlat_close v hhw (fun _ _ _ => hv _) ci
  · -- error: vt within e of va per coordinate
    have hround : |M.gapFlatF vt ci - globalAvgPoolFlat c h w vt ci| ≤ gb := by
      rw [hgbdef]; exact M.gapFlat_close vt hhw (fun _ _ _ => hvt _) ci
    have hshift : |globalAvgPoolFlat c h w vt ci - globalAvgPoolFlat c h w va ci| ≤ e := by
      rw [globalAvgPoolFlat_eq_bnMean vt ci, globalAvgPoolFlat_eq_bnMean va ci]
      refine (bnMean_input_close _ _ hhw).trans ?_
      rw [div_le_iff₀ hhwR]
      calc ∑ s, |Tensor3.unflatten vt ci (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2
                - Tensor3.unflatten va ci (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2|
          ≤ ∑ _s : Fin (h * w), e := Finset.sum_le_sum (fun s _ => hd _)
        _ = e * ((h * w : ℕ) : ℝ) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
    calc |M.gapFlatF vt ci - globalAvgPoolFlat c h w va ci|
        ≤ |M.gapFlatF vt ci - globalAvgPoolFlat c h w vt ci|
          + |globalAvgPoolFlat c h w vt ci - globalAvgPoolFlat c h w va ci| := abs_sub_le _ _ _
      _ ≤ gb + e := add_le_add hround hshift

/-- **THE FOLD: a whole CIFAR stage is `FloatClose`.** `conv→relu→conv→relu→maxpool`
    folded through `.comp` into a single certificate — there exist a propagated
    magnitude `B` and an error modulus `L` (the composition of the two conv
    `layerBudget`s through the three `id` moduli) with the whole float stage within
    `L e` of the real stage at input error `e`. No bespoke proof: the five per-op
    `FloatClose` facts chained. The whole r34 net is this same fold at scale (with
    the BN/skip instances slotted in). ⚠ The `∃ B L` closes the modulus, so on its own
    this statement says only that both maps are bounded on the box. -/
theorem floatClose_cifarStage {ic c h w : Nat} (M : FloatModel)
    (W₁ : Kernel4 c ic 3 3) (b₁ : Vec c) (W₂ : Kernel4 c c 3 3) (b₂ : Vec c)
    {w' β A : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A)
    (hn1 : 0 < ic * (2*h) * (2*w)) (hn2 : 0 < c * (2*h) * (2*w))
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w') (hb₁ : ∀ o, |b₁ o| ≤ β)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w') (hb₂ : ∀ o, |b₂ o| ≤ β) :
    ∃ B L, FloatClose A B
      (maxPoolFlat c h w ∘ relu (c*(2*h)*(2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂
        ∘ relu (c*(2*h)*(2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)
      (maxPoolFlat c h w ∘ relu (c*(2*h)*(2*w)) ∘ M.flatConvF (h := 2*h) (w := 2*w) W₂ b₂
        ∘ relu (c*(2*h)*(2*w)) ∘ M.flatConvF (h := 2*h) (w := 2*w) W₁ b₁)
      L := by
  set B1 := layerAct (ic*3*3) w' β A + layerBudget M.u (ic*3*3) w' β A 0 with hB1def
  have hB1 : 0 ≤ B1 :=
    add_nonneg (layerAct_nonneg hw' hβ hA) (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)
  have hc1 := floatClose_flatConv (h := 2*h) (w := 2*w) M W₁ b₁ hw' hβ hA hn1 hW₁ hb₁
  have hr1 := floatClose_relu (n := c*(2*h)*(2*w)) B1
  have hc2 := floatClose_flatConv (h := 2*h) (w := 2*w) M W₂ b₂ hw' hβ hB1 hn2 hW₂ hb₂
  set B2 := layerAct (c*3*3) w' β B1 + layerBudget M.u (c*3*3) w' β B1 0 with hB2def
  have hr2 := floatClose_relu (n := c*(2*h)*(2*w)) B2
  have hmp := floatClose_maxPool (c := c) (h := h) (w := w) B2
  exact ⟨_, _, (((hc1.comp hr1).comp hc2).comp hr2).comp hmp⟩

-- ════════════════════════════════════════════════════════════════
-- § The residual skip (a branching combinator, not a plain .comp)
-- ════════════════════════════════════════════════════════════════

/-- **Additive residual `F(x) + x` (no trailing activation) is `FloatClose`** — the
    MBConv / transformer skip, the skip of `floatClose_residualBlock` without its ReLU. The
    rounded skip-add `fl(FF(x) ⊕ x)` is within `add_close`'s budget of the real
    `F(x) + x`; output magnitude `(1+u)(B+A)`. -/
theorem floatClose_addResidual {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (fun v => fun j => F v j + v j)
      (fun v => fun j => M.add (FF v j) (v j))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) := by
  have hu := M.u_nonneg
  obtain ⟨hFm, hFe⟩ := hF
  refine ⟨fun v hv i => ⟨?_, ?_⟩, fun vt va e hva hvt hd i => ?_⟩
  · have hb := (hFm v hv i).1
    nlinarith [abs_add_le (F v i) (v i), hv i, abs_nonneg (F v i), abs_nonneg (v i)]
  · have hsum : |FF v i + v i| ≤ B + A := (abs_add_le _ _).trans (add_le_add (hFm v hv i).2 (hv i))
    refine (M.abs_rnd_le _).trans ?_
    nlinarith [abs_nonneg (FF v i + v i)]
  · refine (M.add_close (hFe vt va e hva hvt hd i) (hd i)).trans ?_
    have h1 : M.u * (|F va i| + LF e + |va i| + e) ≤ M.u * (B + LF e + A + e) :=
      mul_le_mul_of_nonneg_left (by linarith [(hFm va hva i).1, hva i]) hu
    linarith

/-- **Residual block `relu(F(x) + x)` is `FloatClose`** — the branching combinator
    (the skip reuses the input, so it's not a plain `.comp`). Given the body `F`
    `FloatClose A B`, the block's float (rounded skip-add) is within
    `reluAdd_close`'s budget of the real `relu(F(x)+x)`; output magnitude
    `(1+u)(B+A)`. The defining ResNet op: `floatClose_addResidual` then `floatClose_relu`. -/
theorem floatClose_residualBlock {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (fun v => relu m (fun j => F v j + v j))
      (fun v => relu m (fun j => M.add (FF v j) (v j)))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) :=
  (floatClose_addResidual M hF).comp (floatClose_relu _)

/-- **THE RESIDUAL FOLD: a (no-BN) ResNet basic block is `FloatClose`.** Body
    `conv₂ → relu → conv₁` folded via `.comp`, then wrapped by the residual
    combinator into `relu(F(x) + x)` — one certificate for the whole block,
    skip included. The r34 identity block is this with BN inserted (the BN→relu
    `FloatClose` instance is the remaining wrap). ⚠ Same caveat as `floatClose_cifarStage`. -/
theorem floatClose_resBlock {c h w : Nat} (M : FloatModel)
    (W₁ W₂ : Kernel4 c c 3 3) (b₁ b₂ : Vec c) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < c * h * w)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w') (hb₁ : ∀ o, |b₁ o| ≤ β)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w') (hb₂ : ∀ o, |b₂ o| ≤ β) :
    ∃ B L, FloatClose A B
      (fun v => relu (c*h*w)
        (fun j => (flatConv W₂ b₂ ∘ relu (c*h*w) ∘ flatConv W₁ b₁) v j + v j))
      (fun v => relu (c*h*w)
        (fun j => M.add ((M.flatConvF W₂ b₂ ∘ relu (c*h*w) ∘ M.flatConvF W₁ b₁) v j) (v j)))
      L := by
  have hB1 : 0 ≤ layerAct (c*3*3) w' β A + layerBudget M.u (c*3*3) w' β A 0 :=
    add_nonneg (layerAct_nonneg hw' hβ hA) (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)
  have hbody :=
    ((floatClose_flatConv (h := h) (w := w) M W₁ b₁ hw' hβ hA hn hW₁ hb₁).comp
      (floatClose_relu (n := c*h*w)
        (layerAct (c*3*3) w' β A + layerBudget M.u (c*3*3) w' β A 0))).comp
      (floatClose_flatConv (h := h) (w := w) M W₂ b₂ hw' hβ hB1 hn hW₂ hb₂)
  exact ⟨_, _, floatClose_residualBlock M hbody⟩

-- ════════════════════════════════════════════════════════════════
-- § BN → relu as a FloatClose instance (the other r34 wrap)
-- ════════════════════════════════════════════════════════════════

/-- **BN (no activation) is `FloatClose`** (per-example, training-mode). The float BN computes
    its stats from the input via the supplied `fμ`/`fistdv` (within `emean`/`eistd` of the true
    stats on the magnitude domain — discharged by `bnMean_close` / `bnVar_close` +
    `bnIstd_close_at` when instantiated). Error from `bnStep_close` (rounding
    `bnForward_close_of` + input-shift `bnForward_input_close`); magnitude the real
    `|γ|·|x̂| + |β|` plus that rounding. The BN-before-swish steps in EfficientNet's MBConv (and
    BN-before-GELU positions generally) are this instance. -/
theorem floatClose_bn {m : Nat} (M : FloatModel)
    {ε γ β emean eistd D S G Bbnd A : ℝ} (fμ fistdv : Vec m → ℝ)
    (hn : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ v, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean)
    (histd : ∀ v, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd)
    (hD : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |v j - bnMean m v| ≤ D)
    (hSabs : ∀ v, (∀ k, |v k| ≤ A) → |bnIstd m v ε| ≤ S) :
    FloatClose A (G * (D * S) + Bbnd + bnNormBudget M.u D S G Bbnd emean eistd)
      (fun v => bnForward m ε γ β v)
      (fun v => M.bnForwardF γ β (fμ v) (fistdv v) v)
      (fun e => bnReluBudget M.u D S G Bbnd emean eistd A e ε) := by
  refine FloatClose.of_close (fun v hv i => ?_)
    (fun v hv i => M.bnForward_close_of (ε := ε) v i (hmean v hv) (histd v hv)
      (hD v hv i) (hSabs v hv) hγ hβ)
    (fun vt va e hva hvt hd i => M.bnStep_close vt va i hn hε hd hvt hva (hmean vt hvt)
      (histd vt hvt) (hD vt hvt) (hSabs vt hvt) hγ hβ)
  have hxhat : |bnXhat m ε v i| ≤ D * S := by
    unfold bnXhat; rw [abs_mul]
    exact mul_le_mul (hD v hv i) (hSabs v hv) (abs_nonneg _) ((abs_nonneg _).trans (hD v hv i))
  show |bnForward m ε γ β v i| ≤ _
  unfold bnForward
  refine (abs_add_le _ _).trans (add_le_add ?_ hβ)
  rw [abs_mul]; exact mul_le_mul hγ hxhat (abs_nonneg _) ((abs_nonneg _).trans hγ)

/-- **BN→relu is `FloatClose`** — `floatClose_bn` followed by `floatClose_relu` (ReLU is exact
    and 1-Lipschitz, so the BN modulus and magnitude pass through unchanged). With this +
    `floatClose_flatConv` + the residual combinator, the r34 identity block folds entirely
    through `.comp`. -/
theorem floatClose_bnRelu {m : Nat} (M : FloatModel)
    {ε γ β emean eistd D S G Bbnd A : ℝ} (fμ fistdv : Vec m → ℝ)
    (hn : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ v, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean)
    (histd : ∀ v, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd)
    (hD : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |v j - bnMean m v| ≤ D)
    (hSabs : ∀ v, (∀ k, |v k| ≤ A) → |bnIstd m v ε| ≤ S) :
    FloatClose A (G * (D * S) + Bbnd + bnNormBudget M.u D S G Bbnd emean eistd)
      (fun v => relu m (bnForward m ε γ β v))
      (fun v => relu m (M.bnForwardF γ β (fμ v) (fistdv v) v))
      (fun e => bnReluBudget M.u D S G Bbnd emean eistd A e ε) :=
  (floatClose_bn M fμ fistdv hn hε hγ hβ hmean histd hD hSabs).comp (floatClose_relu _)

-- ════════════════════════════════════════════════════════════════
-- § The final fold: a block iterated to depth (r34's [3,4,6,3] stages)
-- ════════════════════════════════════════════════════════════════

/-- The identity map is `FloatClose` (modulus `id`). -/
theorem floatClose_id {m : Nat} (A : ℝ) :
    FloatClose A A (id : Vec m → Vec m) (id : Vec m → Vec m) (id : ℝ → ℝ) :=
  ⟨fun _v hv i => ⟨hv i, hv i⟩, fun _vt _va _e _ _ hd i => hd i⟩

/-- **THE FINAL FOLD: a magnitude-stable block iterated `n` times is `FloatClose`.**
    A dim-preserving block that is `FloatClose A A f fF L` (its activations stay
    within the a-posteriori bound `A` — BN keeps them O(1), as the probe confirms)
    composes with itself to any depth: `f^[n]` is `FloatClose A A` with modulus
    `L^[n]`. This is r34's within-stage depth (`n = 3,4,6,3`); the whole net is
    these iterates `.comp`-joined with the stem / downsamples / GAP / dense. The
    depth-generic whole-net certificate — no per-depth re-proof. -/
theorem floatClose_iterate {m : Nat} {A : ℝ} {f fF : Vec m → Vec m} {L : ℝ → ℝ}
    (hf : FloatClose A A f fF L) (n : ℕ) :
    FloatClose A A (f^[n]) (fF^[n]) (L^[n]) := by
  induction n with
  | zero => simpa using floatClose_id A
  | succ k ih =>
      rw [Function.iterate_succ', Function.iterate_succ', Function.iterate_succ']
      exact ih.comp hf

/-- **r34's four stages, folded.** Given an identity block that is magnitude-stable
    `FloatClose A A` (the a-posteriori-bounded regime), the `[3,4,6,3]` block stack
    of each stage is `FloatClose A A` — the four `floatClose_iterate` instances at
    r34's depths. The full `r34_float_close` is these `.comp` the stem / strided
    downsamples / GAP / dense (each its own `FloatClose` instance). -/
theorem floatClose_r34_stages {m : Nat} {A : ℝ} {blk blkF : Vec m → Vec m} {L : ℝ → ℝ}
    (hblk : FloatClose A A blk blkF L) :
    FloatClose A A (blk^[3]) (blkF^[3]) (L^[3]) ∧
    FloatClose A A (blk^[4]) (blkF^[4]) (L^[4]) ∧
    FloatClose A A (blk^[6]) (blkF^[6]) (L^[6]) ∧
    FloatClose A A (blk^[3]) (blkF^[3]) (L^[3]) :=
  ⟨floatClose_iterate hblk 3, floatClose_iterate hblk 4,
   floatClose_iterate hblk 6, floatClose_iterate hblk 3⟩

-- ═════════════════════════════════════════════════
-- § The additive skip
-- ═════════════════════════════════════════════════

/-- **Additive residual `residual f = f(x) + x` is `FloatClose`** — the MBConv /
    transformer skip in the `Residual.lean` API (`residual = biPath f id`, defeq to
    `floatClose_addResidual`'s `fun v j => F v j + v j`). -/
theorem floatClose_residual {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (residual F) (fun v j => M.add (FF v j) (v j))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) :=
  floatClose_addResidual M hF

end Proofs
