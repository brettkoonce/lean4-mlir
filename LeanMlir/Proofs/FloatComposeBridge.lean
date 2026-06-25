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

end Proofs
