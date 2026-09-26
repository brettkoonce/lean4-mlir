import LeanMlir.Proofs.Float.FloatClose
import LeanMlir.Proofs.Float.ResNet34BlockBridge
-- He et al.'s 3×3/s2 stem pool, for `floatClose_maxPool3s2` below. It imports only
-- `Architectures.CNN`, which this file already has transitively (it uses `maxPoolFlat_abs_le`),
-- so this adds no cycle. `planning/archive/rsb_a3_r50_verified.md` §4b.
import LeanMlir.Proofs.Architectures.MaxPool3s2

/-!
# ℝ→Float32 bridge: per-op `FloatClose` instances for the conv-net op set

`FloatClose` (`FloatClose.lean`): on inputs within magnitude `A`, the float
`fF` is within an error modulus `L e` of the real `f` (per coordinate, at input
error `e`), and both real and float outputs are within `B` (so the next layer's
magnitude precondition is met). `FloatClose.comp` composes two — the moduli
compose as `Lg ∘ Lf`, magnitudes thread `A → B → C`.

Instances proved here: `floatClose_flatConv` (modulus = the conv-fan-in
`layerBudget`), `floatClose_dense`, the pools (`floatClose_maxPool`,
`floatClose_maxPool3s2`, `floatClose_gap`), the skips (`floatClose_addResidual`,
`floatClose_residualBlock`, `floatClose_residual`), `floatClose_bn` (use the
operating-point `bnIstd_close_at` for the `eistd`, else the budget is vacuous),
and `floatClose_r34_stages`. A whole-net bound would be `.comp` of these; none is
assembled in the repo.
-/

namespace Proofs

open FloatModel

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
    + the fresh-input rounding `layerBudget(e=0)`. Stated for any `dense W b`; nothing in
    the repo instantiates it. -/
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

/-- **MaxPool is `FloatClose` with modulus `id`** — exact in float, 1-Lipschitz,
    never grows magnitudes (`maxPoolFlat_close` / `maxPoolFlat_abs_le`). -/
theorem floatClose_maxPool {c h w : Nat} (A : ℝ) :
    FloatClose A A (maxPoolFlat c h w) (maxPoolFlat c h w) (fun e => e) :=
  ⟨fun _v hv i => ⟨maxPoolFlat_abs_le hv i, maxPoolFlat_abs_le hv i⟩,
   fun vt va _e _ _ hd i => maxPoolFlat_close vt va hd i⟩

/-- **He et al.'s 3×3/s2 stem pool is `FloatClose`** — the peer of `floatClose_maxPool`, and
    identical in shape: a max is exact (it selects an existing cell, so the modulus is `id` and the
    magnitude is unchanged) whatever the window size. The overlap that makes the backward
    accumulate is invisible here, because the forward at one output still reads one cell. -/
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
      exact (Finset.sum_le_card_nsmul _ _ _ fun s _ => hd _).trans_eq (by simp [mul_comm])
    calc |M.gapFlatF vt ci - globalAvgPoolFlat c h w va ci|
        ≤ |M.gapFlatF vt ci - globalAvgPoolFlat c h w vt ci|
          + |globalAvgPoolFlat c h w vt ci - globalAvgPoolFlat c h w va ci| := abs_sub_le _ _ _
      _ ≤ gb + e := add_le_add hround hshift

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
    `B + A + u·(B + A)` of the real `relu(F(x)+x)`; output magnitude
    `(1+u)(B+A)`. The defining ResNet op: `floatClose_addResidual` then `floatClose_relu`. -/
theorem floatClose_residualBlock {m : Nat} (M : FloatModel) {A B : ℝ}
    {F FF : Vec m → Vec m} {LF : ℝ → ℝ} (hF : FloatClose A B F FF LF) :
    FloatClose A (B + A + M.u * (B + A))
      (fun v => relu m (fun j => F v j + v j))
      (fun v => relu m (fun j => M.add (FF v j) (v j)))
      (fun e => M.u * (B + LF e + A + e) + (LF e + e)) :=
  (floatClose_addResidual M hF).comp (floatClose_relu _)

-- ════════════════════════════════════════════════════════════════
-- § BN → relu as a FloatClose instance (the other r34 wrap)
-- ════════════════════════════════════════════════════════════════

/-- **BN (no activation) is `FloatClose`** (per-example, training-mode). The float BN computes
    its stats from the input via the supplied `fμ`/`fistdv` (within `emean`/`eistd` of the true
    stats on the magnitude domain — discharged by `bnMean_close` / `bnVar_close` +
    `bnIstd_close_at` when instantiated). Error from `bnStep_close` (rounding
    `bnForward_close_of` + input-shift `bnForward_input_close`); magnitude the real
    `|γ|·|x̂| + |β|` plus that rounding. Stated for a BN with no trailing activation (the form a
    BN-before-swish or BN-before-GELU position would use); nothing in the repo instantiates it. -/
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

-- ════════════════════════════════════════════════════════════════
-- § The final fold: a block iterated to depth (r34's [3,4,6,3] stages)
-- ════════════════════════════════════════════════════════════════

/-- **The `[3,4,6,3]` iterates.** Given a dim-preserving block that is `FloatClose A A`
    (magnitude bound `A` taken as a hypothesis), its 3-, 4-, 6- and 3-fold iterates are
    `FloatClose A A` — `floatClose_iterate` at ResNet-34's per-stage block counts, for one
    block `blk` at one width `m`. No whole-net ResNet-34 float bound exists in the repo. -/
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
