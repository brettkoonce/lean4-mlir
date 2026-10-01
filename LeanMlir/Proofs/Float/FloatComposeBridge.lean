import LeanMlir.Proofs.Float.FloatClose
import LeanMlir.Proofs.Float.BnInputBridge
-- He et al.'s 3×3/s2 stem pool, for `floatClose_maxPool3s2` below. It imports only
-- `Architectures.CNN`, which this file already has transitively (it uses `maxPoolFlat_abs_le`),
-- so this adds no cycle.
import LeanMlir.Proofs.Architectures.MaxPool3s2
import LeanMlir.Proofs.Float.ResNet34FloatBridge

/-!
# ℝ→Float32 bridge: per-op `FloatClose` instances for the conv-net op set

`FloatClose` (`FloatClose.lean`): on inputs within magnitude `A`, the float
`fF` is within an error modulus `L e` of the real `f` (per coordinate, at input
error `e`), and both real and float outputs are within `B` (so the next layer's
magnitude precondition is met). `FloatClose.comp` composes two — the moduli
compose as `Lg ∘ Lf`, magnitudes thread `A → B → C`.

Instances proved here: `floatClose_flatConv` (modulus = the conv-fan-in
`layerBudget`), the pools (`floatClose_maxPool`, `floatClose_maxPool3s2`, `floatClose_gap`)
and the skips (`floatClose_addResidual`, `floatClose_residualBlock`).
A whole-net bound would be `.comp` of these; none is assembled in the repo.
-/

namespace Proofs

open FloatModel

/-- **Convolution is `FloatClose`** with modulus the conv-fan-in `layerBudget`.
    Real output ≤ `layerAct`; float output ≤ `layerAct + layerBudget(e=0)` (the
    extra rounding) — that sum is the propagated magnitude `B`. -/
theorem floatClose_flatConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0)
      (flatConv (h := h) (w := w) W b) (M.flatConvF (h := h) (w := w) W b)
      (fun e => layerBudget M.u (ic * kH * kW) w' β A e) :=
  FloatClose.of_close (fun v hv i => flatConv_abs_le hA hW hb hv i)
    (fun v hv i => M.flatConvF_close W b v v hw' hA le_rfl hW hb hv (fun k => by simp) i)
    (fun vt va e hva _ hd i => M.flatConvF_close W b vt va hw' hA
      ((abs_nonneg _).trans (hd ⟨0, hn⟩)) hW hb hva hd i)

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
    (hhw : 0 < h * w) :
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
    `F(x) + x`; output magnitude `(1+u)(B+A)`. `fun v j => F v j + v j` is `residual F`
    (`Residual.lean`) by definition, so this lemma serves that spelling too. -/
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

end Proofs
