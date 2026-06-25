import LeanMlir.Proofs.Resnet34BlockBridge

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
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0)
      (flatConv (h := h) (w := w) W b) (M.flatConvF (h := h) (w := w) W b)
      (fun e => layerBudget M.u (ic * kH * kW) w' β A e) := by
  have hLB0 : 0 ≤ layerBudget M.u (ic * kH * kW) w' β A 0 :=
    layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hreal := flatConv_abs_le hA hW hb hv i
    have hround : |M.flatConvF W b v i - flatConv W b v i|
        ≤ layerBudget M.u (ic * kH * kW) w' β A 0 :=
      M.flatConvF_close W b v v hw' hA le_rfl hW hb hv (fun k => by simp) i
    have htri : |M.flatConvF W b v i|
        ≤ |M.flatConvF W b v i - flatConv W b v i| + |flatConv W b v i| := by
      simpa using abs_sub_le (M.flatConvF W b v i) (flatConv W b v i) 0
    exact ⟨hreal.trans (le_add_of_nonneg_right hLB0), by
      calc |M.flatConvF W b v i|
          ≤ |M.flatConvF W b v i - flatConv W b v i| + |flatConv W b v i| := htri
        _ ≤ layerBudget M.u (ic * kH * kW) w' β A 0 + layerAct (ic * kH * kW) w' β A :=
            add_le_add hround hreal
        _ = layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0 := by ring⟩
  · have he : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, hn⟩)
    exact M.flatConvF_close W b vt va hw' hA he hW hb hva hd i

/-- **Dense layer is `FloatClose`** with modulus the fan-in `layerBudget` (the dense
    analogue of `floatClose_flatConv`). Real output ≤ `layerAct`; float output ≤ that
    + the fresh-input rounding `layerBudget(e=0)`. The SE excite/reduce denses and the
    classifier head are this instance; the ViT MLP denses reuse it too. -/
theorem floatClose_dense {m n : Nat} (M : FloatModel) (W : Mat m n) (b : Vec n)
    {w' β A : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β) :
    FloatClose A
      (layerAct m w' β A + layerBudget M.u m w' β A 0)
      (Proofs.dense W b) (M.dense W b)
      (fun e => layerBudget M.u m w' β A e) := by
  have hLB0 : 0 ≤ layerBudget M.u m w' β A 0 :=
    layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hreal : |Proofs.dense W b v i| ≤ layerAct m w' β A :=
      dense_abs_le hA hW hb hv i
    have hround : |M.dense W b v i - Proofs.dense W b v i| ≤ layerBudget M.u m w' β A 0 :=
      (M.dense_close_fresh W b v i).trans (M.denseErr_le_uniform hw' le_rfl hW hb hv i)
    refine ⟨hreal.trans (le_add_of_nonneg_right hLB0), ?_⟩
    calc |M.dense W b v i|
        ≤ |M.dense W b v i - Proofs.dense W b v i| + |Proofs.dense W b v i| := by
          simpa using abs_sub_le (M.dense W b v i) (Proofs.dense W b v i) 0
      _ ≤ layerBudget M.u m w' β A 0 + layerAct m w' β A := add_le_add hround hreal
      _ = layerAct m w' β A + layerBudget M.u m w' β A 0 := by ring
  · have he : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, hm⟩)
    exact (M.dense_close W b vt va e he hd i).trans
      (M.denseErr_le_uniform hw' he hW hb hva i)

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

/-- **Global-average-pool is `FloatClose`** — `Vec (c·h·w) → Vec c`, the SE squeeze.
    GAP is a per-channel `bnMean` (`globalAvgPoolFlat_eq_bnMean`), so the real output
    never exceeds the input magnitude `A` (`bnMean_abs_le`) and is 1-Lipschitz in the
    input (`bnMean_input_close`, the spatial mean averages the per-coordinate error
    back to `e`); the float roundoff is `gapFlat_close`'s budget `gb`. Output magnitude
    `A + gb`, modulus `e ↦ gb + e`. -/
theorem floatClose_gap {c h w : Nat} (M : FloatModel) {A : ℝ}
    (hA0 : 0 ≤ A) (hhw : 0 < h * w) :
    FloatClose A
      (A + (M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A))
      (globalAvgPoolFlat c h w) M.gapFlatF
      (fun e => (M.u * ((1 + M.u) ^ (h * w + 1) * A)
                 + ((1 + M.u) ^ (h * w + 1) - 1) * A) + e) := by
  have hu := M.u_nonneg
  have hhwR : (0:ℝ) < ((h * w : ℕ) : ℝ) := by exact_mod_cast hhw
  set gb := M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A
    with hgbdef
  have hgb0 : 0 ≤ gb := by
    rw [hgbdef]
    have hpow : (1:ℝ) ≤ (1 + M.u) ^ (h * w + 1) := one_le_pow₀ (by linarith)
    have h2 : 0 ≤ ((1 + M.u) ^ (h * w + 1) - 1) * A := mul_nonneg (by linarith) hA0
    have h1 : 0 ≤ M.u * ((1 + M.u) ^ (h * w + 1) * A) := by positivity
    linarith
  refine ⟨fun v hv ci => ?_, fun vt va e hva hvt hd ci => ?_⟩
  · -- magnitude at v
    have hAt : ∀ ci' hi wi, |Tensor3.unflatten v ci' hi wi| ≤ A := fun _ _ _ => hv _
    have hreal : |globalAvgPoolFlat c h w v ci| ≤ A := by
      rw [globalAvgPoolFlat_eq_bnMean v ci]
      exact bnMean_abs_le _ hhw (fun s => hAt _ _ _)
    have hround : |M.gapFlatF v ci - globalAvgPoolFlat c h w v ci| ≤ gb := by
      rw [hgbdef]; exact M.gapFlat_close v hhw hAt ci
    refine ⟨hreal.trans (le_add_of_nonneg_right hgb0), ?_⟩
    calc |M.gapFlatF v ci|
        ≤ |M.gapFlatF v ci - globalAvgPoolFlat c h w v ci| + |globalAvgPoolFlat c h w v ci| := by
          simpa using abs_sub_le (M.gapFlatF v ci) (globalAvgPoolFlat c h w v ci) 0
      _ ≤ gb + A := add_le_add hround hreal
      _ = A + gb := by ring
  · -- error: vt within e of va per coordinate
    have hAtt : ∀ ci' hi wi, |Tensor3.unflatten vt ci' hi wi| ≤ A := fun _ _ _ => hvt _
    have hround : |M.gapFlatF vt ci - globalAvgPoolFlat c h w vt ci| ≤ gb := by
      rw [hgbdef]; exact M.gapFlat_close vt hhw hAtt ci
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
    the BN/skip instances slotted in). -/
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

/-- **Residual block `relu(F(x) + x)` is `FloatClose`** — the branching combinator
    (the skip reuses the input, so it's not a plain `.comp`). Given the body `F`
    `FloatClose A B`, the block's float (rounded skip-add) is within
    `reluAdd_close`'s budget of the real `relu(F(x)+x)`; output magnitude
    `(1+u)(B+A)`. The defining ResNet op. -/
theorem floatClose_residualBlock {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (fun v => relu m (fun j => F v j + v j))
      (fun v => relu m (fun j => M.add (FF v j) (v j)))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) := by
  have hu := M.u_nonneg
  obtain ⟨hFm, hFe⟩ := hF
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hb : |F v i| ≤ B := (hFm v hv i).1
    have hfb : |FF v i| ≤ B := (hFm v hv i).2
    have hvi : |v i| ≤ A := hv i
    have hBA : 0 ≤ B + A := by
      have := (abs_nonneg (F v i)).trans hb; have := (abs_nonneg (v i)).trans hvi; linarith
    refine ⟨?_, ?_⟩
    · calc |relu m (fun j => F v j + v j) i| ≤ |F v i + v i| := relu_abs_le _ i
        _ ≤ |F v i| + |v i| := abs_add_le _ _
        _ ≤ B + A := add_le_add hb hvi
        _ ≤ B + A + M.u * (B + A) := le_add_of_nonneg_right (mul_nonneg hu hBA)
    · have hsum : |FF v i + v i| ≤ B + A := (abs_add_le _ _).trans (add_le_add hfb hvi)
      calc |relu m (fun j => M.add (FF v j) (v j)) i| ≤ |M.add (FF v i) (v i)| := relu_abs_le _ i
        _ ≤ |M.add (FF v i) (v i) - (FF v i + v i)| + |FF v i + v i| := by
            simpa using abs_sub_le (M.add (FF v i) (v i)) (FF v i + v i) 0
        _ ≤ M.u * |FF v i + v i| + |FF v i + v i| := add_le_add (M.err _) le_rfl
        _ ≤ M.u * (B + A) + (B + A) := add_le_add (mul_le_mul_of_nonneg_left hsum hu) hsum
        _ = B + A + M.u * (B + A) := by ring
  · exact M.reluAdd_close (fun k => hFe vt va e hva hvt hd k) hd
      (fun k => (hFm va hva k).1) hva i

/-- **Additive residual `F(x) + x` (no trailing activation) is `FloatClose`** — the
    MBConv / transformer skip, the no-ReLU cousin of `floatClose_residualBlock`. The
    rounded skip-add `fl(FF(x) ⊕ x)` is within `add_close`'s budget of the real
    `F(x) + x`; output magnitude `(1+u)(B+A)`. Reused by EfficientNet's MBConv skip
    (`floatClose_smoothResBlock`) and by the ViT block's two additive skips. -/
theorem floatClose_addResidual {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (fun v => fun j => F v j + v j)
      (fun v => fun j => M.add (FF v j) (v j))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) := by
  have hu := M.u_nonneg
  obtain ⟨hFm, hFe⟩ := hF
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hb : |F v i| ≤ B := (hFm v hv i).1
    have hfb : |FF v i| ≤ B := (hFm v hv i).2
    have hvi : |v i| ≤ A := hv i
    have hBA : 0 ≤ B + A := by
      have := (abs_nonneg (F v i)).trans hb; have := (abs_nonneg (v i)).trans hvi; linarith
    refine ⟨?_, ?_⟩
    · calc |F v i + v i| ≤ |F v i| + |v i| := abs_add_le _ _
        _ ≤ B + A := add_le_add hb hvi
        _ ≤ B + A + M.u * (B + A) := le_add_of_nonneg_right (mul_nonneg hu hBA)
    · have hsum : |FF v i + v i| ≤ B + A := (abs_add_le _ _).trans (add_le_add hfb hvi)
      calc |M.add (FF v i) (v i)|
          ≤ |M.add (FF v i) (v i) - (FF v i + v i)| + |FF v i + v i| := by
            simpa using abs_sub_le (M.add (FF v i) (v i)) (FF v i + v i) 0
        _ ≤ M.u * |FF v i + v i| + |FF v i + v i| := add_le_add (M.err _) le_rfl
        _ ≤ M.u * (B + A) + (B + A) := add_le_add (mul_le_mul_of_nonneg_left hsum hu) hsum
        _ = B + A + M.u * (B + A) := by ring
  · refine (M.add_close (hFe vt va e hva hvt hd i) (hd i)).trans ?_
    have hb : |F va i| ≤ B := (hFm va hva i).1
    have ha : |va i| ≤ A := hva i
    have h1 : M.u * (|F va i| + LF e + |va i| + e) ≤ M.u * (B + LF e + A + e) :=
      mul_le_mul_of_nonneg_left (by linarith) hu
    linarith

/-- **THE RESIDUAL FOLD: a (no-BN) ResNet basic block is `FloatClose`.** Body
    `conv₂ → relu → conv₁` folded via `.comp`, then wrapped by the residual
    combinator into `relu(F(x) + x)` — one certificate for the whole block,
    skip included. The r34 identity block is this with BN inserted (the BN→relu
    `FloatClose` instance is the remaining wrap). -/
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

/-- **BN→relu is `FloatClose`** (per-example, training-mode). The float BN computes
    its stats from the input via the supplied `fμ`/`fistdv` (within `emean`/`eistd`
    of the true stats on the magnitude domain — discharged by `bnMean_close` /
    `bnVar_close` + `bnIstd_close_at` when instantiated). Error from `bnRelu_close`
    (rounding + input-shift); float-output magnitude from `bnForward_close_of`.
    With this + `floatClose_flatConv` + the residual combinator, the r34 identity
    block folds entirely through `.comp`. -/
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
      (fun e => bnReluBudget M.u D S G Bbnd emean eistd A e ε) := by
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i =>
    M.bnRelu_close vt va i hn hε hd hvt hva (hmean vt hvt) (histd vt hvt)
      (hD vt hvt) (hSabs vt hvt) hγ hβ⟩
  -- magnitude: real |bnForward| ≤ G·D·S + Bbnd; float ≤ that + bnNormBudget rounding
  have hu := M.u_nonneg
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hBbnd0 : 0 ≤ Bbnd := (abs_nonneg _).trans hβ
  have hS0 : 0 ≤ S := (abs_nonneg _).trans (hSabs v hv)
  have hD0 : 0 ≤ D := (abs_nonneg _).trans (hD v hv i)
  have hem0 : 0 ≤ emean := (abs_nonneg _).trans (hmean v hv)
  have hei0 : 0 ≤ eistd := (abs_nonneg _).trans (histd v hv)
  have hnb0 : 0 ≤ bnNormBudget M.u D S G Bbnd emean eistd := by
    unfold bnNormBudget FloatModel.mulErr; positivity
  have hxhat : |bnXhat m ε v i| ≤ D * S := by
    unfold bnXhat; rw [abs_mul]
    exact mul_le_mul (hD v hv i) (hSabs v hv) (abs_nonneg _) ((abs_nonneg _).trans (hD v hv i))
  have hreal : |bnForward m ε γ β v i| ≤ G * (D * S) + Bbnd := by
    unfold bnForward
    refine (abs_add_le _ _).trans (add_le_add ?_ hβ)
    rw [abs_mul]; exact mul_le_mul hγ hxhat (abs_nonneg _) ((abs_nonneg _).trans hγ)
  have hround := M.bnForward_close_of (ε := ε) v i (hmean v hv) (histd v hv)
    (hD v hv i) (hSabs v hv) hγ hβ
  refine ⟨(relu_abs_le _ i).trans (hreal.trans (le_add_of_nonneg_right hnb0)), ?_⟩
  · refine (relu_abs_le _ i).trans ?_
    have htri : |M.bnForwardF γ β (fμ v) (fistdv v) v i|
        ≤ |M.bnForwardF γ β (fμ v) (fistdv v) v i - bnForward m ε γ β v i|
          + |bnForward m ε γ β v i| := by
      simpa using abs_sub_le (M.bnForwardF γ β (fμ v) (fistdv v) v i) (bnForward m ε γ β v i) 0
    calc |M.bnForwardF γ β (fμ v) (fistdv v) v i|
        ≤ |M.bnForwardF γ β (fμ v) (fistdv v) v i - bnForward m ε γ β v i|
          + |bnForward m ε γ β v i| := htri
      _ ≤ bnNormBudget M.u D S G Bbnd emean eistd + (G * (D * S) + Bbnd) := add_le_add hround hreal
      _ = G * (D * S) + Bbnd + bnNormBudget M.u D S G Bbnd emean eistd := by ring

/-- **BN alone (no activation) is `FloatClose`** — `floatClose_bnRelu` with the
    trailing ReLU dropped, error from `bnStep_close` (rounding `bnForward_close_of`
    + input-shift `bnForward_input_close`), same `bnReluBudget` modulus (ReLU only
    shrinks, so removing it leaves the budget unchanged). The BN-before-swish steps
    in EfficientNet's MBConv (and BN-before-GELU positions generally) are this
    instance; pair with `floatClose_swish` via `.comp`. -/
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
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i =>
    M.bnStep_close vt va i hn hε hd hvt hva (hmean vt hvt) (histd vt hvt)
      (hD vt hvt) (hSabs vt hvt) hγ hβ⟩
  have hu := M.u_nonneg
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hBbnd0 : 0 ≤ Bbnd := (abs_nonneg _).trans hβ
  have hS0 : 0 ≤ S := (abs_nonneg _).trans (hSabs v hv)
  have hD0 : 0 ≤ D := (abs_nonneg _).trans (hD v hv i)
  have hem0 : 0 ≤ emean := (abs_nonneg _).trans (hmean v hv)
  have hei0 : 0 ≤ eistd := (abs_nonneg _).trans (histd v hv)
  have hnb0 : 0 ≤ bnNormBudget M.u D S G Bbnd emean eistd := by
    unfold bnNormBudget FloatModel.mulErr; positivity
  have hxhat : |bnXhat m ε v i| ≤ D * S := by
    unfold bnXhat; rw [abs_mul]
    exact mul_le_mul (hD v hv i) (hSabs v hv) (abs_nonneg _) ((abs_nonneg _).trans (hD v hv i))
  have hreal : |bnForward m ε γ β v i| ≤ G * (D * S) + Bbnd := by
    unfold bnForward
    refine (abs_add_le _ _).trans (add_le_add ?_ hβ)
    rw [abs_mul]; exact mul_le_mul hγ hxhat (abs_nonneg _) ((abs_nonneg _).trans hγ)
  have hround := M.bnForward_close_of (ε := ε) v i (hmean v hv) (histd v hv)
    (hD v hv i) (hSabs v hv) hγ hβ
  refine ⟨hreal.trans (le_add_of_nonneg_right hnb0), ?_⟩
  have htri : |M.bnForwardF γ β (fμ v) (fistdv v) v i|
      ≤ |M.bnForwardF γ β (fμ v) (fistdv v) v i - bnForward m ε γ β v i|
        + |bnForward m ε γ β v i| := by
    simpa using abs_sub_le (M.bnForwardF γ β (fμ v) (fistdv v) v i) (bnForward m ε γ β v i) 0
  calc |M.bnForwardF γ β (fμ v) (fistdv v) v i|
      ≤ |M.bnForwardF γ β (fμ v) (fistdv v) v i - bnForward m ε γ β v i|
        + |bnForward m ε γ β v i| := htri
    _ ≤ bnNormBudget M.u D S G Bbnd emean eistd + (G * (D * S) + Bbnd) := add_le_add hround hreal
    _ = G * (D * S) + Bbnd + bnNormBudget M.u D S G Bbnd emean eistd := by ring

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

-- ════════════════════════════════════════════════════════════════
-- § FloatBridges: magnitude-threading-free whole-net assembly
-- ════════════════════════════════════════════════════════════════

/-- **`f` float-bridges** — for *any* nonnegative input magnitude there is a
    nonnegative output magnitude and a `FloatClose` certificate. This is the
    existential closure of `FloatClose` over the magnitude domain: it drops the
    bookkeeping of the exact `B`/`L` so that whole-net assembly composes in one line
    (no manual `set B0 … B7` threading). Every op that is `FloatClose A (φ A) …` with
    `0 ≤ φ A` float-bridges; `FloatBridges.comp` chains them. The form `floatClose_bn`
    delivers (with its operating-point data) and §1d's whole-net fold consumes. -/
def FloatBridges {m n : Nat} (f : Vec m → Vec n) : Prop :=
  ∀ A, 0 ≤ A → ∃ B L fF, 0 ≤ B ∧ FloatClose A B f fF L

/-- **Float-bridging composes** — the whole-net assembly backbone, magnitudes threaded
    automatically (each stage's output magnitude feeds the next). -/
theorem FloatBridges.comp {m n p : Nat} {f : Vec m → Vec n} {g : Vec n → Vec p}
    (hf : FloatBridges f) (hg : FloatBridges g) : FloatBridges (g ∘ f) := by
  intro A hA
  obtain ⟨B, L, fF, hB, hfc⟩ := hf A hA
  obtain ⟨C, Lg, gF, hC, hgc⟩ := hg B hB
  exact ⟨C, Lg ∘ L, gF ∘ fF, hC, hfc.comp hgc⟩

/-- The propagated magnitude of a `FloatClose` is nonnegative (a real output bound at
    the zero input — valid since `0` is within any nonneg magnitude domain). -/
theorem FloatClose.cod_nonneg {m n : Nat} {A B : ℝ} {f fF : Vec m → Vec n} {L : ℝ → ℝ}
    (hfc : FloatClose A B f fF L) (hA : 0 ≤ A) (hn : 0 < n) : 0 ≤ B := by
  obtain ⟨hm, _⟩ := hfc
  have hz : ∀ k : Fin m, |(0 : Vec m) k| ≤ A := fun k => by simpa using hA
  exact (abs_nonneg _).trans (hm 0 hz ⟨0, hn⟩).1

/-- A `FloatClose` error modulus is nonnegative at input error `0` (it bounds an
    absolute value at the zero input). The piece SE-branch magnitude nonnegativity needs. -/
theorem FloatClose.modulus_zero_nonneg {m n : Nat} {A B : ℝ} {f fF : Vec m → Vec n}
    {L : ℝ → ℝ} (hfc : FloatClose A B f fF L) (hA : 0 ≤ A) (hn : 0 < n) : 0 ≤ L 0 := by
  obtain ⟨_, he⟩ := hfc
  have hz : ∀ k : Fin m, |(0 : Vec m) k| ≤ A := fun k => by simpa using hA
  exact (abs_nonneg _).trans (he 0 0 0 hz hz (fun k => by simp) ⟨0, hn⟩)

/-- ReLU float-bridges (magnitude-stable). -/
theorem floatBridges_relu {n : Nat} : FloatBridges (relu n) :=
  fun A hA => ⟨A, _, _, hA, floatClose_relu A⟩

/-- MaxPool float-bridges (magnitude-stable). -/
theorem floatBridges_maxPool {c h w : Nat} : FloatBridges (maxPoolFlat c h w) :=
  fun A hA => ⟨A, _, _, hA, floatClose_maxPool A⟩

/-- Convolution float-bridges (output magnitude `layerAct + layerBudget`). -/
theorem floatBridges_flatConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatBridges (flatConv (h := h) (w := w) W b) :=
  fun _A hA => ⟨_, _, _,
    add_nonneg (layerAct_nonneg hw' hβ hA) (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl),
    floatClose_flatConv M W b hw' hβ hA hn hW hb⟩

/-- Dense float-bridges (output magnitude `layerAct + layerBudget`). -/
theorem floatBridges_dense {m n : Nat} (M : FloatModel) (W : Mat m n) (b : Vec n)
    {w' β : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β) :
    FloatBridges (Proofs.dense W b) :=
  fun _A hA => ⟨_, _, _,
    add_nonneg (layerAct_nonneg hw' hβ hA) (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl),
    floatClose_dense M W b hw' hβ hA hm hW hb⟩

/-- **Additive residual `residual f = f(x) + x` is `FloatClose`** — the MBConv /
    transformer skip in the `Residual.lean` API (`residual = biPath f id`, defeq to
    `floatClose_addResidual`'s `fun v j => F v j + v j`). -/
theorem floatClose_residual {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (residual F) (fun v j => M.add (FF v j) (v j))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) :=
  floatClose_addResidual M hF

/-- **Float-bridging survives the residual skip** — if a dim-preserving block
    float-bridges, so does `residual block`. The MBConv / ResNet / transformer skip
    in bridge form. -/
theorem FloatBridges.residual {m : Nat} (M : FloatModel) {f : Vec m → Vec m}
    (hf : FloatBridges f) : FloatBridges (Proofs.residual f) := by
  intro A hA
  obtain ⟨B, L, fF, hB, hfc⟩ := hf A hA
  refine ⟨B + A + M.u * (B + A), _, _, ?_, floatClose_residual M hfc⟩
  have hBA : 0 ≤ B + A := add_nonneg hB hA
  have := M.u_nonneg; nlinarith [mul_nonneg this hBA]

end Proofs
