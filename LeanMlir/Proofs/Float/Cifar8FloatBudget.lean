import LeanMlir.Proofs.Float.Cifar8FloatBridge
import LeanMlir.Proofs.Architectures.Cifar8ChainCert

/-! # A NUMBER for a whole-net float budget: CIFAR-8 at the committed shape, He profile

`FloatBridgesTo` carries a whole net's error modulus as data (`FloatComposeBridge.lean`
§`FloatBridgesTo`), so a whole-net budget can be STATED as a number and CHECKED by the
kernel — which the earlier `∃ B L` form could not (`formalization.yaml` fidelity §4d).
This file does that for the committed 8-conv CIFAR-10 net (`cifar8Verified`:
conv 3→16,16,16,16,32,32,32,32 SAME 3×3 with a max-pool after every second conv,
flatten 128, dense 64/64/10), at the He-profile bounds the adjoint-chain tier uses
(`Cifar8ChainCert.lean`: `|W| ≤ 2/5` covering the measured 0.39, `|b| ≤ 1/100`), on the
unit input window, for any rounding model at binary32 accuracy (`u ≤ 2⁻²⁴`):

    output window  ≤ 6.121·10¹⁸        (`cifar8Bridge_mag_le`)
    fresh budget   ≤ 6.37·10¹⁴         (`cifar8Bridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 6.37·10¹⁴` on `|x| ≤ 1` (`cifar8_float_logits_le`).

The method is a numeric ENVELOPE (`FloatBridgesTo.Env`): an upper bound on the output
window and on the fresh budget, pushed through each layer by two rational inequalities.
Each conv/dense step uses `layerBudget_le_num` (the monotone form of the per-op Higham
budget) with the γ-term bounded through `FloatModel.gamma_num`, so `norm_num` discharges
every step with no big-power evaluation; `relu` and the pools pass the envelope through
unchanged. Eleven steps, twenty-two numerals, one `exact`.

⚠ **What the number means.** 6.37·10¹⁴ against logits of ≈ 10 is vacuous as a
certificate, and that is the honest content: this is the interval fold — `FloatClose.comp`
threading worst-case windows (1 → 10.8 → 623 → … → 6.1·10¹⁸) — evaluated at He
magnitudes, the same regime in which `Cifar8ChainCert.lean` §"Magnitude" shows the
adjoint chain's budget is ≥ 1.8·10¹³ and `scripts/adjoint_chain_probe.py` reports
3.79·10¹⁴ for worst-case tail gains. The three numbers agree in order, which is the
consistency check. What is new is not the size of the number but that the kernel now
checks it: a wrong `layerBudget`, a dropped stage, or a misread fan-in would fail here.
The relative scale, 6.4·10¹⁴ / 6.1·10¹⁸ ≈ 10⁻⁴, is the same as the MNIST-MLP capstone's
(`mnist_mlp_float_budget`), as it should be — it is ≈ Σ (mᵢ+2)·u over the chain. -/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Numeric envelopes for bridges
-- ════════════════════════════════════════════════════════════════

/-- A numeric envelope for a bridge at input window `A`: its output window is at most `Ā`
    and its fresh budget at most `Ē`. The `Env.comp_*` lemmas below push an envelope through
    one layer with two rational inequalities, so a whole-net `.fresh` is bounded by
    `norm_num` a stage at a time. -/
structure FloatBridgesTo.Env {m n : Nat} {f fF : Vec m → Vec n} (b : FloatBridgesTo f fF)
    (A Ā Ē : ℝ) : Prop where
  /-- The output window at input window `A` is at most `Ā`. -/
  mag_le : b.mag A ≤ Ā
  /-- The fresh budget at input window `A` is at most `Ē`. -/
  fresh_le : b.fresh A ≤ Ē

/-- The fresh budget is nonnegative (it bounds an absolute value at the zero input). -/
theorem FloatBridgesTo.fresh_nonneg {m n : Nat} {f fF : Vec m → Vec n} (b : FloatBridgesTo f fF)
    {A : ℝ} (hA : 0 ≤ A) (hn : 0 < n) : 0 ≤ b.fresh A :=
  (b.floatClose hA).modulus_zero_nonneg hA hn

/-- `layerAct` under an upper bound on the activation magnitude. -/
theorem layerAct_le_num {m : ℕ} {w β A Ā : ℝ} (hw : 0 ≤ w) (hAĀ : A ≤ Ā) :
    layerAct m w β A ≤ (m : ℝ) * w * Ā + β := by
  unfold layerAct
  have hmw : (0:ℝ) ≤ (m : ℝ) * w := mul_nonneg (Nat.cast_nonneg m) hw
  have := mul_le_mul_of_nonneg_left hAĀ hmw
  linarith

/-- `layerBudget` under upper bounds on the magnitude, the inherited error and the γ-term —
    the public monotone form the numeric envelopes chain through (the per-net private
    `layerBudget_le_of` in `FloatBridge.lean`, plus monotonicity in the magnitude). -/
theorem layerBudget_le_num {u : ℝ} {m : ℕ} {w β A Ā E Ē g : ℝ}
    (hu : 0 ≤ u) (hw : 0 ≤ w) (hβ : 0 ≤ β) (hA : 0 ≤ A) (hAĀ : A ≤ Ā)
    (hE : 0 ≤ E) (hEĒ : E ≤ Ē) (hg : (1 + u) ^ (m + 2) - 1 ≤ g) :
    layerBudget u m w β A E ≤ g * ((m : ℝ) * w * (Ā + Ē) + β) + (m : ℝ) * w * Ē := by
  unfold layerBudget
  have hG0 : (0 : ℝ) ≤ (1 + u) ^ (m + 2) - 1 := sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hmw : (0 : ℝ) ≤ (m : ℝ) * w := mul_nonneg (Nat.cast_nonneg m) hw
  have hX0 : (0 : ℝ) ≤ (m : ℝ) * w * (A + E) + β :=
    add_nonneg (mul_nonneg hmw (add_nonneg hA hE)) hβ
  have hX : (m : ℝ) * w * (A + E) + β ≤ (m : ℝ) * w * (Ā + Ē) + β := by
    have := mul_le_mul_of_nonneg_left (add_le_add hAĀ hEĒ) hmw
    linarith
  have h1 : ((1 + u) ^ (m + 2) - 1) * ((m : ℝ) * w * (A + E) + β)
      ≤ g * ((m : ℝ) * w * (Ā + Ē) + β) := mul_le_mul hg hX hX0 (hG0.trans hg)
  have h2 : (m : ℝ) * w * E ≤ (m : ℝ) * w * Ē := mul_le_mul_of_nonneg_left hEĒ hmw
  linarith

namespace FloatBridgesTo

/-- The γ-term is nonnegative, so any upper bound on it is. -/
private theorem gamma_ub_nonneg (M : FloatModel) {k : ℕ} {g : ℝ}
    (hg : (1 + M.u) ^ k - 1 ≤ g) : 0 ≤ g :=
  (sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))).trans hg

/-- Envelope of a conv leaf at input window `A` (the first layer of a chain). -/
theorem Env.flatConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    {A : ℝ} (hA : 0 ≤ A) {g : ℝ} (hg : (1 + M.u) ^ (ic * kH * kW + 2) - 1 ≤ g)
    {Ā' Ē' : ℝ}
    (hĀ' : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * A + β) ≤ Ā')
    (hĒ' : g * (((ic * kH * kW : ℕ) : ℝ) * w' * A + β) ≤ Ē') :
    (floatBridgesTo_flatConv (h := h) (w := w) M W b hw' hβ hn hW hb).Env A Ā' Ē' := by
  have h1 := layerAct_le_num (m := ic * kH * kW) (β := β) hw' (le_refl A)
  have h2 := layerBudget_le_num (m := ic * kH * kW) M.u_nonneg hw' hβ hA (le_refl A)
    (le_refl 0) (le_refl 0) hg
  simp only [add_zero, mul_zero] at h2
  have hX : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * A + β)
      = (((ic * kH * kW : ℕ) : ℝ) * w' * A + β) + g * (((ic * kH * kW : ℕ) : ℝ) * w' * A + β) := by
    ring
  rw [hX] at hĀ'
  exact ⟨by show layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0 ≤ Ā'
            linarith,
         by show layerBudget M.u (ic * kH * kW) w' β A 0 ≤ Ē'
            linarith⟩

/-- Push an envelope through a conv layer. -/
theorem Env.comp_flatConv {p ic oc h w kH kW : Nat} {f fF : Vec p → Vec (ic * h * w)}
    {b : FloatBridgesTo f fF} {A Ā Ē : ℝ} (e : b.Env A Ā Ē) (hA : 0 ≤ A)
    (M : FloatModel) (W : Kernel4 oc ic kH kW) (b' : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b' o| ≤ β)
    {g : ℝ} (hg : (1 + M.u) ^ (ic * kH * kW + 2) - 1 ≤ g)
    {Ā' Ē' : ℝ}
    (hĀ' : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * (((ic * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + β)
            + ((ic * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (b.comp (floatBridgesTo_flatConv (h := h) (w := w) M W b' hw' hβ hn hW hb)).Env A Ā' Ē' := by
  have hmag0 : 0 ≤ b.mag A := b.mag_nonneg hA
  have hfr0 : 0 ≤ b.fresh A := b.fresh_nonneg hA hn
  have h1 := layerAct_le_num (m := ic * kH * kW) (β := β) hw' e.mag_le
  have h2 := layerBudget_le_num (m := ic * kH * kW) M.u_nonneg hw' hβ hmag0 e.mag_le
    (le_refl 0) (le_refl 0) hg
  simp only [add_zero, mul_zero] at h2
  have hX : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β)
      = (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β) + g * (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β) := by
    ring
  rw [hX] at hĀ'
  exact ⟨by show layerAct (ic * kH * kW) w' β (b.mag A)
              + layerBudget M.u (ic * kH * kW) w' β (b.mag A) 0 ≤ Ā'
            linarith,
         by show layerBudget M.u (ic * kH * kW) w' β (b.mag A) (b.fresh A) ≤ Ē'
            exact (layerBudget_le_num M.u_nonneg hw' hβ hmag0 e.mag_le hfr0 e.fresh_le hg).trans hĒ'⟩

/-- Push an envelope through a dense layer. -/
theorem Env.comp_dense {p m n : Nat} {f fF : Vec p → Vec m}
    {b : FloatBridgesTo f fF} {A Ā Ē : ℝ} (e : b.Env A Ā Ē) (hA : 0 ≤ A)
    (M : FloatModel) (W : Mat m n) (b' : Vec n) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b' j| ≤ β)
    {g : ℝ} (hg : (1 + M.u) ^ (m + 2) - 1 ≤ g)
    {Ā' Ē' : ℝ}
    (hĀ' : (1 + g) * ((m : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * ((m : ℝ) * w' * (Ā + Ē) + β) + (m : ℝ) * w' * Ē ≤ Ē') :
    (b.comp (floatBridgesTo_dense M W b' hw' hβ hm hW hb)).Env A Ā' Ē' := by
  have hmag0 : 0 ≤ b.mag A := b.mag_nonneg hA
  have hfr0 : 0 ≤ b.fresh A := b.fresh_nonneg hA hm
  have h1 := layerAct_le_num (m := m) (β := β) hw' e.mag_le
  have h2 := layerBudget_le_num (m := m) M.u_nonneg hw' hβ hmag0 e.mag_le (le_refl 0) (le_refl 0) hg
  simp only [add_zero, mul_zero] at h2
  have hX : (1 + g) * ((m : ℝ) * w' * Ā + β)
      = ((m : ℝ) * w' * Ā + β) + g * ((m : ℝ) * w' * Ā + β) := by ring
  rw [hX] at hĀ'
  exact ⟨by show layerAct m w' β (b.mag A) + layerBudget M.u m w' β (b.mag A) 0 ≤ Ā'
            linarith,
         by show layerBudget M.u m w' β (b.mag A) (b.fresh A) ≤ Ē'
            exact (layerBudget_le_num M.u_nonneg hw' hβ hmag0 e.mag_le hfr0 e.fresh_le hg).trans hĒ'⟩

/-- ReLU passes an envelope through unchanged (exact in float, never grows magnitudes). -/
theorem Env.comp_relu {p n : Nat} {f fF : Vec p → Vec n} {b : FloatBridgesTo f fF}
    {A Ā Ē : ℝ} (e : b.Env A Ā Ē) : (b.comp (floatBridgesTo_relu (n := n))).Env A Ā Ē :=
  ⟨e.mag_le, e.fresh_le⟩

/-- Max-pool passes an envelope through unchanged. -/
theorem Env.comp_maxPool {p c h w : Nat} {f fF : Vec p → Vec (c * (2 * h) * (2 * w))}
    {b : FloatBridgesTo f fF} {A Ā Ē : ℝ} (e : b.Env A Ā Ē) :
    (b.comp (floatBridgesTo_maxPool (c := c) (h := h) (w := w))).Env A Ā Ē :=
  ⟨e.mag_le, e.fresh_le⟩

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The committed CIFAR-8 bridge and its number
-- ════════════════════════════════════════════════════════════════

/-- A `Cifar8Weights` bound is nonnegative (the index types are inhabited). -/
theorem Cifar8Weights.w'_nonneg {w' β : ℝ} (W : Cifar8Weights w' β) : 0 ≤ w' :=
  (abs_nonneg _).trans (W.hcW1 0 0 0 0)

theorem Cifar8Weights.β_nonneg {w' β : ℝ} (W : Cifar8Weights w' β) : 0 ≤ β :=
  (abs_nonneg _).trans (W.hcb1 0)

/-- **The committed CIFAR-8 forward's bridge at a weight profile.** `Cifar8Weights w' β`
    (`Cifar8ChainCert.lean`) supplies the 22 tensors with their magnitude bounds; the real
    side is `cifarCnn8Forward` at the committed config — the map `cifar8ChainH_chainRH_eq`
    ties the adjoint chain to and `StableHLO.cifar8FwdGraph_faithful` ties the emitted
    forward graph to. -/
noncomputable def cifar8Bridge (M : FloatModel) {w' β : ℝ} (W : Cifar8Weights w' β) :
    FloatBridgesTo
      (cifarCnn8Forward (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
        (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3)
        W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
        W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8
        W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3)
      (cifarCnn8ForwardF (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
        (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3) M
        W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
        W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8
        W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3) :=
  floatBridgesTo_cifar8 (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
    (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3)
    M W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
    W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8 W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3
    W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg
    W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg
    W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg
    W.w'_nonneg W.β_nonneg W.w'_nonneg W.β_nonneg
    W.hcW1 W.hcb1 W.hcW2 W.hcb2 W.hcW3 W.hcb3 W.hcW4 W.hcb4
    W.hcW5 W.hcb5 W.hcW6 W.hcb6 W.hcW7 W.hcb7 W.hcW8 W.hcb8
    W.hdW1 W.hdb1 W.hdW2 W.hdb2 W.hdW3 W.hdb3
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

set_option maxRecDepth 100000 in
/-- **The envelope, kernel-checked.** At `|W| ≤ 2/5`, `|b| ≤ 1/100`, unit input window and
    `u ≤ 2⁻²⁴`: output window `≤ 6.121·10¹⁸`, fresh budget `≤ 6.37·10¹⁴`. Eleven
    `Env.comp_*` steps, each with its γ-term through `gamma_num` and two rational
    inequalities by `norm_num`; the per-stage numerals are the exact fold rounded up to four
    significant figures, and the kernel re-checks every inequality. The chain is built
    bottom-up so each step elaborates against a small, fully determined type; the closing
    `exact` is one structural comparison with `cifar8Bridge`'s definition. -/
theorem cifar8Bridge_env (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100)) :
    (cifar8Bridge M W).Env 1 6121000000000000000 637000000000000 := by
  have hw : (0:ℝ) ≤ 2/5 := by norm_num
  have hβ : (0:ℝ) ≤ 1/100 := by norm_num
  have e1 := FloatBridgesTo.Env.flatConv (h := 32) (w := 32) (A := 1) M W.cW1 W.cb1 hw hβ
    (by norm_num) W.hcW1 W.hcb1 (by norm_num)
    (M.gamma_num (q := 173/100000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 541/50) (Ē' := 1871/100000000) (by norm_num) (by norm_num)
  have e2 := FloatBridgesTo.Env.comp_relu e1
  have e3 := FloatBridgesTo.Env.comp_flatConv e2 (by norm_num) M W.cW2 W.cb2 hw hβ
    (by norm_num) W.hcW2 W.hcb2
    (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 6233/10) (Ē' := 6507/1000000) (by norm_num) (by norm_num)
  have e4 := FloatBridgesTo.Env.comp_relu e3
  have e5 := FloatBridgesTo.Env.comp_maxPool (c := 16) (h := 16) (w := 16) e4
  have e6 := FloatBridgesTo.Env.comp_flatConv e5 (by norm_num) M W.cW3 W.cb3 hw hβ
    (by norm_num) W.hcW3 W.hcb3
    (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 35910) (Ē' := 1719/2500) (by norm_num) (by norm_num)
  have e7 := FloatBridgesTo.Env.comp_relu e6
  have e8 := FloatBridgesTo.Env.comp_flatConv e7 (by norm_num) M W.cW4 W.cb4 hw hβ
    (by norm_num) W.hcW4 W.hcb4
    (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 2069000) (Ē' := 5763/100) (by norm_num) (by norm_num)
  have e9 := FloatBridgesTo.Env.comp_relu e8
  have e10 := FloatBridgesTo.Env.comp_maxPool (c := 16) (h := 8) (w := 8) e9
  have e11 := FloatBridgesTo.Env.comp_flatConv e10 (by norm_num) M W.cW5 W.cb5 hw hβ
    (by norm_num) W.hcW5 W.hcb5
    (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 119200000) (Ē' := 4358) (by norm_num) (by norm_num)
  have e12 := FloatBridgesTo.Env.comp_relu e11
  have e13 := FloatBridgesTo.Env.comp_flatConv e12 (by norm_num) M W.cW6 W.cb6 hw hβ
    (by norm_num) W.hcW6 W.hcb6
    (M.gamma_num (q := 173/10000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 13740000000) (Ē' := 739700) (by norm_num) (by norm_num)
  have e14 := FloatBridgesTo.Env.comp_relu e13
  have e15 := FloatBridgesTo.Env.comp_maxPool (c := 32) (h := 4) (w := 4) e14
  have e16 := FloatBridgesTo.Env.comp_flatConv e15 (by norm_num) M W.cW7 W.cb7 hw hβ
    (by norm_num) W.hcW7 W.hcb7
    (M.gamma_num (q := 173/10000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 1583000000000) (Ē' := 112600000) (by norm_num) (by norm_num)
  have e17 := FloatBridgesTo.Env.comp_relu e16
  have e18 := FloatBridgesTo.Env.comp_flatConv e17 (by norm_num) M W.cW8 W.cb8 hw hβ
    (by norm_num) W.hcW8 W.hcb8
    (M.gamma_num (q := 173/10000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 182400000000000) (Ē' := 16130000000) (by norm_num) (by norm_num)
  have e19 := FloatBridgesTo.Env.comp_relu e18
  have e20 := FloatBridgesTo.Env.comp_maxPool (c := 32) (h := 2) (w := 2) e19
  have e21 := FloatBridgesTo.Env.comp_dense e20 (by norm_num) M W.dW1 W.db1 hw hβ
    (by norm_num) W.hdW1 W.hdb1
    (M.gamma_num (q := 31/4000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 9339000000000000) (Ē' := 898300000000) (by norm_num) (by norm_num)
  have e22 := FloatBridgesTo.Env.comp_relu e21
  have e23 := FloatBridgesTo.Env.comp_dense e22 (by norm_num) M W.dW2 W.db2 hw hβ
    (by norm_num) W.hdW2 W.hdb2
    (M.gamma_num (q := 197/50000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 239100000000000000) (Ē' := 23940000000000) (by norm_num) (by norm_num)
  have e24 := FloatBridgesTo.Env.comp_relu e23
  have e25 := FloatBridgesTo.Env.comp_dense e24 (by norm_num) M W.dW3 W.db3 hw hβ
    (by norm_num) W.hdW3 W.hdb3
    (M.gamma_num (q := 197/50000000) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 6121000000000000000) (Ē' := 637000000000000) (by norm_num) (by norm_num)
  exact e25

/-- The committed CIFAR-8 bridge's fresh budget at the He profile: `≤ 6.37·10¹⁴`. -/
theorem cifar8Bridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100)) :
    (cifar8Bridge M W).fresh 1 ≤ 637000000000000 :=
  (cifar8Bridge_env M hMu W).fresh_le

/-- The committed CIFAR-8 bridge's certified output window at the He profile: `≤ 6.121·10¹⁸`
    — the worst-case logit magnitude the interval fold can promise. -/
theorem cifar8Bridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100)) :
    (cifar8Bridge M W).mag 1 ≤ 6121000000000000000 :=
  (cifar8Bridge_env M hMu W).mag_le

/-- ⭐ **The deployed CIFAR-8 float forward is within `6.37·10¹⁴` of the certified real
    forward, per logit**, on inputs of magnitude `≤ 1`, at `|W| ≤ 2/5`, `|b| ≤ 1/100`, for
    any rounding model at binary32 accuracy. The first whole-net float budget in the repo
    stated as a number over a `FloatBridgesTo`; see the file header for what the size of
    that number does and does not mean. -/
theorem cifar8_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100))
    (x : Vec (3 * (2*(2*(2*(2*2)))) * (2*(2*(2*(2*2)))))) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cifarCnn8ForwardF (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
        (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3) M
        W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
        W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8
        W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3 x j
      - cifarCnn8Forward (ic := 3) (c1 := 16) (c2 := 16) (c3 := 32) (c4 := 32)
        (h := 2) (w := 2) (d1 := 64) (nClasses := 10) (kH := 3) (kW := 3)
        W.cW1 W.cb1 W.cW2 W.cb2 W.cW3 W.cb3 W.cW4 W.cb4
        W.cW5 W.cb5 W.cW6 W.cb6 W.cW7 W.cb7 W.cW8 W.cb8
        W.dW1 W.db1 W.dW2 W.db2 W.dW3 W.db3 x j| ≤ 637000000000000 :=
  ((cifar8Bridge M W).fresh_le (by norm_num) x hx j).trans (cifar8Bridge_fresh_le M hMu W)

end Proofs
