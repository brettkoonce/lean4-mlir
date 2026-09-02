import LeanMlir.Proofs.Float.Resnet34WholeFloatBridge
import LeanMlir.Proofs.Float.BnEvalRuntimeFloatBridge

/-! # Numeric envelopes for `FloatBridgesTo`: the kit whole-net BUDGETS are numbers in

`Cifar8FloatBudget.lean` turned one whole-net `FloatBridgesTo` into a kernel-checked number by
pushing a pair `(output window, error budget)` through the net a layer at a time. That worked
because the CIFAR-8 chain is a straight `.comp` of leaves. The ImageNet-scale nets are not: a
ResNet basic block is `relu ∘ residual body`, and the residual's modulus is evaluated at the
block's **inherited** input error, not at `0`. So the CIFAR file's `FloatBridgesTo.Env` — which
fixes the input error at `0` and is therefore only closed under `.comp` with a per-op monotone
lemma — does not compose through a skip.

`FloatBridgesTo.Maps Ā Ē Ā' Ē'` is the fix: read it as *"on every input window `A ≤ Ā` and every
inherited error `E ≤ Ē`, this bridge's output window is `≤ Ā'` and its output error is `≤ Ē'`."*
Quantifying over the inputs rather than fixing them is what buys monotonicity, and monotonicity is
what makes the combinators GENERIC:

* `Maps.comp` composes two envelopes with no per-op reasoning at all — the CIFAR kit needed one
  `Env.comp_*` lemma per operation.
* `Maps.residual` / `Maps.biPathSum` push an envelope through a skip, which `Env` could not
  express.

Only the LEAVES still need work, and each is ten lines: `show` the unfolded `mag`/`mod`, one
monotone bound (`layerAct_le_num` / `layerBudget_le_num` / the γ-term through
`FloatModel.gamma_num`), `linarith`. The γ-term is always bounded by a rational `g` so `norm_num`
never evaluates a big power.

⚠ **`Maps` must stay a structure.** The CIFAR session lost a day to an `Env` defined as a `def`
unfolding to `∧`: the unifier then delta-unfolded the whole net's `.mag` chain and timed out at
20× the heartbeat budget. Inductive types unify argument-wise and never unfold.

The CIFAR-8 instance still runs on `Env` (`Cifar8FloatBudget.lean`); migrating it to `Maps` is a
coherence pass, not a correctness one — the per-stage inequalities are the same, and this file
holds the two monotone lemmas (`layerAct_le_num`, `layerBudget_le_num`) both kits share.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Modulus nonnegativity at an arbitrary inherited error
-- ════════════════════════════════════════════════════════════════

/-- A `FloatClose` error modulus is nonnegative at **every** nonnegative inherited error — it
    bounds an absolute value at the zero input, taking the float and real inputs equal (which is
    legal for any `e ≥ 0`). The `e = 0` case is `FloatClose.modulus_zero_nonneg`; the composition
    of envelopes needs the general one, because a block's body sees the prefix's budget. -/
theorem FloatClose.modulus_nonneg {m n : Nat} {A B : ℝ} {f fF : Vec m → Vec n}
    {L : ℝ → ℝ} (hfc : FloatClose A B f fF L) (hA : 0 ≤ A) (hn : 0 < n)
    {e : ℝ} (he : 0 ≤ e) : 0 ≤ L e := by
  obtain ⟨_, hE⟩ := hfc
  have hz : ∀ k : Fin m, |(0 : Vec m) k| ≤ A := fun k => by simpa using hA
  exact (abs_nonneg _).trans (hE 0 0 e hz hz (fun k => by simpa using he) ⟨0, hn⟩)

namespace FloatBridgesTo

variable {m n : Nat} {f fF : Vec m → Vec n}

/-- A bridge's modulus is nonnegative at every nonnegative window and inherited error. -/
theorem mod_nonneg (b : FloatBridgesTo f fF) {A e : ℝ} (hA : 0 ≤ A) (he : 0 ≤ e)
    (hn : 0 < n) : 0 ≤ b.mod A e :=
  (b.floatClose hA).modulus_nonneg hA hn he

/-- The fresh budget is nonnegative (the `e = 0` case). -/
theorem fresh_nonneg (b : FloatBridgesTo f fF) {A : ℝ} (hA : 0 ≤ A) (hn : 0 < n) :
    0 ≤ b.fresh A := b.mod_nonneg hA le_rfl hn

-- ════════════════════════════════════════════════════════════════
-- § The envelope
-- ════════════════════════════════════════════════════════════════

/-- **`b` carries the envelope `(Ā, Ē)` to `(Ā', Ē')`**: at every input window `A ≤ Ā` and every
    inherited input error `E ≤ Ē`, the bridge's output window is at most `Ā'` and its output
    error is at most `Ē'`. The quantification over `A`/`E` (rather than fixing them) is what makes
    the envelope monotone, hence composable without per-operation lemmas. -/
structure Maps (b : FloatBridgesTo f fF) (Ā Ē Ā' Ē' : ℝ) : Prop where
  /-- Output window, at any input window within `Ā`. -/
  mag_le : ∀ A, 0 ≤ A → A ≤ Ā → b.mag A ≤ Ā'
  /-- Output error, at any input window within `Ā` and any inherited error within `Ē`. -/
  mod_le : ∀ A E, 0 ≤ A → 0 ≤ E → A ≤ Ā → E ≤ Ē → b.mod A E ≤ Ē'

/-- The deployed-map claim an envelope makes at a concrete window: on inputs of magnitude `≤ Ā`
    the float map is within `Ē'` of the real map, per coordinate. The `Maps` peer of
    `FloatBridgesTo.fresh_le`, and the step every `<net>_float_logits_le` ends on. -/
theorem Maps.budget_le {b : FloatBridgesTo f fF} {Ā Ē Ā' Ē' : ℝ} (hM : b.Maps Ā Ē Ā' Ē')
    (hĀ : 0 ≤ Ā) (hĒ : 0 ≤ Ē) (v : Vec m) (hv : ∀ k, |v k| ≤ Ā) (i : Fin n) :
    |fF v i - f v i| ≤ Ē' :=
  (b.fresh_le hĀ v hv i).trans (hM.mod_le Ā 0 hĀ le_rfl le_rfl hĒ)

/-- Weakening: a wider output envelope is still an envelope, and a narrower input envelope
    still maps into it. -/
theorem Maps.mono {b : FloatBridgesTo f fF} {Ā Ē Ā' Ē' Ā₀ Ē₀ Ā'' Ē'' : ℝ}
    (hM : b.Maps Ā Ē Ā' Ē') (hA : Ā₀ ≤ Ā) (hE : Ē₀ ≤ Ē) (hA' : Ā' ≤ Ā'') (hE' : Ē' ≤ Ē'') :
    b.Maps Ā₀ Ē₀ Ā'' Ē'' where
  mag_le := fun A h0 hle => (hM.mag_le A h0 (hle.trans hA)).trans hA'
  mod_le := fun A E h0 hE0 hle hEle =>
    (hM.mod_le A E h0 hE0 (hle.trans hA) (hEle.trans hE)).trans hE'

/-- ⭐ **Envelopes compose, generically.** No per-operation reasoning: the second stage's envelope
    is applied at the first stage's output window and output error, which is exactly what
    `FloatBridgesTo.comp`'s `mag`/`mod` are. The intermediate dimension must be inhabited
    (`0 < n`) so the intermediate error is nonnegative. -/
theorem Maps.comp {p : Nat} {g gF : Vec n → Vec p}
    {b : FloatBridgesTo f fF} {c : FloatBridgesTo g gF} {Ā Ē Ā₁ Ē₁ Ā' Ē' : ℝ}
    (hn : 0 < n) (hb : b.Maps Ā Ē Ā₁ Ē₁) (hc : c.Maps Ā₁ Ē₁ Ā' Ē') :
    (b.comp c).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => hc.mag_le (b.mag A) (b.mag_nonneg h0) (hb.mag_le A h0 hle)
  mod_le := fun A E h0 hE0 hle hEle =>
    hc.mod_le (b.mag A) (b.mod A E) (b.mag_nonneg h0) (b.mod_nonneg h0 hE0 hn)
      (hb.mag_le A h0 hle) (hb.mod_le A E h0 hE0 hle hEle)

-- ════════════════════════════════════════════════════════════════
-- § The skip combinators
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through the additive skip `F(x) + x`.** The output window is the body's window
    plus the block's own input window, and one rounding on the sum; the output error adds the
    body's error to the inherited error, plus that rounding. `u` is bounded by a rational `q` so
    both closing inequalities are `norm_num`-able. -/
theorem Maps.residual {m : Nat} (M : FloatModel) {f fF : Vec m → Vec m}
    {bod : FloatBridgesTo f fF} {q Ā Ē Bd Ed Ā' Ē' : ℝ}
    (hm : 0 < m) (hbod : bod.Maps Ā Ē Bd Ed) (hq : M.u ≤ q)
    (hĀ' : Bd + Ā + q * (Bd + Ā) ≤ Ā') (hĒ' : q * (Bd + Ed + Ā + Ē) + (Ed + Ē) ≤ Ē') :
    (bod.residual M).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    have hb := hbod.mag_le A h0 hle
    have hB0 : 0 ≤ bod.mag A := bod.mag_nonneg h0
    have hu := M.u_nonneg
    show bod.mag A + A + M.u * (bod.mag A + A) ≤ Ā'
    have : M.u * (bod.mag A + A) ≤ q * (Bd + Ā) :=
      mul_le_mul hq (by linarith) (by linarith) (by linarith)
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    have hb := hbod.mag_le A h0 hle
    have he := hbod.mod_le A E h0 hE0 hle hEle
    have hB0 : 0 ≤ bod.mag A := bod.mag_nonneg h0
    have he0 : 0 ≤ bod.mod A E := bod.mod_nonneg h0 hE0 hm
    have hu := M.u_nonneg
    show M.u * (bod.mag A + bod.mod A E + A + E) + (bod.mod A E + E) ≤ Ē'
    have : M.u * (bod.mag A + bod.mod A E + A + E) ≤ q * (Bd + Ed + Ā + Ē) :=
      mul_le_mul hq (by linarith) (by linarith) (by linarith)
    linarith

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The two monotone forms the leaves are built from
-- ════════════════════════════════════════════════════════════════

/-- `layerAct` under an upper bound on the activation magnitude. -/
theorem layerAct_le_num' {m : ℕ} {w β A Ā : ℝ} (hw : 0 ≤ w) (hAĀ : A ≤ Ā) :
    layerAct m w β A ≤ (m : ℝ) * w * Ā + β := by
  unfold layerAct
  have hmw : (0:ℝ) ≤ (m : ℝ) * w := mul_nonneg (Nat.cast_nonneg m) hw
  have := mul_le_mul_of_nonneg_left hAĀ hmw
  linarith

/-- `layerBudget` under upper bounds on the magnitude, the inherited error and the γ-term —
    the public monotone form the numeric envelopes chain through. -/
theorem layerBudget_le_num' {u : ℝ} {m : ℕ} {w β A Ā E Ē g : ℝ}
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

-- ════════════════════════════════════════════════════════════════
-- § The exact leaves (magnitude-stable, modulus `id`)
-- ════════════════════════════════════════════════════════════════

/-- ReLU passes an envelope through unchanged (exact in float, never grows a magnitude). -/
theorem Maps.relu {n : Nat} {Ā Ē : ℝ} : (floatBridgesTo_relu (n := n)).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- Max-pool passes an envelope through unchanged. -/
theorem Maps.maxPool {c h w : Nat} {Ā Ē : ℝ} :
    (floatBridgesTo_maxPool (c := c) (h := h) (w := w)).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- He et al.'s 3×3/s2 stem pool passes an envelope through unchanged. -/
theorem Maps.maxPool3s2 {c h w : Nat} {Ā Ē : ℝ} :
    (floatBridgesTo_maxPool3s2 (c := c) (h := h) (w := w)).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

-- ════════════════════════════════════════════════════════════════
-- § The rounding leaves
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through a convolution.** Two rational inequalities: the output window is
    `(1+g)·(fan-in·w'·Ā + β)` and the output error is `g·(fan-in·w'·(Ā+Ē) + β) + fan-in·w'·Ē`,
    with the γ-term bounded by `g` through `FloatModel.gamma_num`. -/
theorem Maps.flatConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (bb : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |bb o| ≤ β)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (ic * kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * (((ic * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + β)
            + ((ic * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_flatConv (h := h) (w := w) M W bb hw' hβ hn hW hb).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0 ≤ Ā'
    have h1 := layerAct_le_num' (m := ic * kH * kW) (β := β) hw' hle
    have h2 := layerBudget_le_num' (m := ic * kH * kW) M.u_nonneg hw' hβ h0 hle
      (le_refl (0:ℝ)) (le_refl (0:ℝ)) hg
    simp only [add_zero, mul_zero] at h2
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show layerBudget M.u (ic * kH * kW) w' β A E ≤ Ē'
    exact (layerBudget_le_num' M.u_nonneg hw' hβ h0 hle hE0 hEle hg).trans hĒ'

/-- **An envelope through a dense layer** — `Maps.flatConv` at fan-in `m`. -/
theorem Maps.dense {m n : Nat} (M : FloatModel) (W : Mat m n) (bb : Vec n)
    {w' β : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |bb j| ≤ β)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (m + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * ((m : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * ((m : ℝ) * w' * (Ā + Ē) + β) + (m : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_dense M W bb hw' hβ hm hW hb).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show layerAct m w' β A + layerBudget M.u m w' β A 0 ≤ Ā'
    have h1 := layerAct_le_num' (m := m) (β := β) hw' hle
    have h2 := layerBudget_le_num' (m := m) M.u_nonneg hw' hβ h0 hle
      (le_refl (0:ℝ)) (le_refl (0:ℝ)) hg
    simp only [add_zero, mul_zero] at h2
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show layerBudget M.u m w' β A E ≤ Ē'
    exact (layerBudget_le_num' M.u_nonneg hw' hβ h0 hle hE0 hEle hg).trans hĒ'

/-- **An envelope through global average pooling.** GAP is a per-channel mean, so its window is
    the input window inflated by one `bnMean` reduction: `Ā·(1+g)·(1+q)` where `g` bounds the
    `h·w+1`-term γ and `q` bounds `u`. -/
theorem Maps.gap {c h w : Nat} (M : FloatModel) (hc : 0 < c) (hhw : 0 < h * w)
    {q g Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q) (hq0 : 0 ≤ q) (hg0 : 0 ≤ g)
    (hg : (1 + M.u) ^ (h * w + 1) - 1 ≤ g)
    (hĀ' : Ā * ((1 + g) * (1 + q)) ≤ Ā')
    (hĒ' : Ā * (q * (1 + g) + g) + Ē ≤ Ē') :
    (floatBridgesTo_gap (c := c) (h := h) (w := w) M hc hhw).Maps Ā Ē Ā' Ē' := by
  have hu := M.u_nonneg
  have hP1 : (1:ℝ) ≤ (1 + M.u) ^ (h * w + 1) := one_le_pow₀ (by linarith)
  constructor
  · intro A h0 hle
    show A + (M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A) ≤ Ā'
    have h1 : M.u * ((1 + M.u) ^ (h * w + 1) * A) ≤ q * ((1 + g) * Ā) :=
      mul_le_mul hq (mul_le_mul (by linarith) hle h0 (by linarith)) (by positivity) hq0
    have h2 : ((1 + M.u) ^ (h * w + 1) - 1) * A ≤ g * Ā :=
      mul_le_mul (by linarith) hle h0 hg0
    nlinarith
  · intro A E h0 hE0 hle hEle
    show (M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A) + E ≤ Ē'
    have h1 : M.u * ((1 + M.u) ^ (h * w + 1) * A) ≤ q * ((1 + g) * Ā) :=
      mul_le_mul hq (mul_le_mul (by linarith) hle h0 (by linarith)) (by positivity) hq0
    have h2 : ((1 + M.u) ^ (h * w + 1) - 1) * A ≤ g * Ā :=
      mul_le_mul (by linarith) hle h0 hg0
    nlinarith

/-- **An envelope through the two-branch fan-in `f(x) + g(x)`.** The `g ≠ id` cousin of
    `Maps.residual` — the downsample block's projection skip, where both branches are
    non-trivial. -/
theorem Maps.biPathSum {m n : Nat} (M : FloatModel) {f fF g gF : Vec m → Vec n}
    {bp : FloatBridgesTo f fF} {bo : FloatBridgesTo g gF}
    {q Ā Ē Pd Ep Bd Ed Ā' Ē' : ℝ} (hn : 0 < n)
    (hbp : bp.Maps Ā Ē Pd Ep) (hbo : bo.Maps Ā Ē Bd Ed) (hq : M.u ≤ q)
    (hĀ' : Pd + Bd + q * (Pd + Bd) ≤ Ā')
    (hĒ' : q * (Pd + Ep + Bd + Ed) + (Ep + Ed) ≤ Ē') :
    (FloatBridgesTo.biPathSum M bp bo).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    have h1 := hbp.mag_le A h0 hle
    have h2 := hbo.mag_le A h0 hle
    have hp0 : 0 ≤ bp.mag A := bp.mag_nonneg h0
    have ho0 : 0 ≤ bo.mag A := bo.mag_nonneg h0
    have hu := M.u_nonneg
    show bp.mag A + bo.mag A + M.u * (bp.mag A + bo.mag A) ≤ Ā'
    have : M.u * (bp.mag A + bo.mag A) ≤ q * (Pd + Bd) :=
      mul_le_mul hq (by linarith) (by linarith) (by linarith)
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    have h1 := hbp.mag_le A h0 hle
    have h2 := hbo.mag_le A h0 hle
    have e1 := hbp.mod_le A E h0 hE0 hle hEle
    have e2 := hbo.mod_le A E h0 hE0 hle hEle
    have hp0 : 0 ≤ bp.mag A := bp.mag_nonneg h0
    have ho0 : 0 ≤ bo.mag A := bo.mag_nonneg h0
    have hpe0 : 0 ≤ bp.mod A E := bp.mod_nonneg h0 hE0 hn
    have hoe0 : 0 ≤ bo.mod A E := bo.mod_nonneg h0 hE0 hn
    have hu := M.u_nonneg
    show M.u * (bp.mag A + bp.mod A E + bo.mag A + bo.mod A E)
          + (bp.mod A E + bo.mod A E) ≤ Ē'
    have : M.u * (bp.mag A + bp.mod A E + bo.mag A + bo.mod A E)
        ≤ q * (Pd + Ep + Bd + Ed) := mul_le_mul hq (by linarith) (by linarith) (by linarith)
    linarith

/-- **An envelope through a stride-2 convolution** — `Maps.flatConv` verbatim: decimating the
    output picks coordinates, so the fan-in `ic·kH·kW` and hence the budget are unchanged. -/
theorem Maps.flatConvStride2 {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (bb : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |bb o| ≤ β)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (ic * kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * (((ic * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + β)
            + ((ic * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_flatConvStride2 (h := h) (w := w) M W bb hw' hβ hn hW hb).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0 ≤ Ā'
    have h1 := layerAct_le_num' (m := ic * kH * kW) (β := β) hw' hle
    have h2 := layerBudget_le_num' (m := ic * kH * kW) M.u_nonneg hw' hβ h0 hle
      (le_refl (0:ℝ)) (le_refl (0:ℝ)) hg
    simp only [add_zero, mul_zero] at h2
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show layerBudget M.u (ic * kH * kW) w' β A E ≤ Ē'
    exact (layerBudget_le_num' M.u_nonneg hw' hβ h0 hle hE0 hEle hg).trans hĒ'

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The BatchNorm leaf: monotone forms of its two budgets
-- ════════════════════════════════════════════════════════════════

/-- `mulErr` is monotone in every argument on the nonnegative orthant (it is a polynomial with
    nonnegative coefficients). -/
theorem mulErr_mono {u u' A A' C C' ea ea' ec ec' : ℝ}
    (hu : 0 ≤ u) (huu : u ≤ u') (hA : 0 ≤ A) (hAA : A ≤ A') (hC : 0 ≤ C) (hCC : C ≤ C')
    (hea : 0 ≤ ea) (heaa : ea ≤ ea') (hec : 0 ≤ ec) (hecc : ec ≤ ec') :
    FloatModel.mulErr u A C ea ec ≤ FloatModel.mulErr u' A' C' ea' ec' := by
  unfold FloatModel.mulErr
  have h1 : (A + ea) * (C + ec) ≤ (A' + ea') * (C' + ec') :=
    mul_le_mul (by linarith) (by linarith) (by linarith) (by linarith)
  have h2 : u * ((A + ea) * (C + ec)) ≤ u' * ((A' + ea') * (C' + ec')) :=
    mul_le_mul huu h1 (by positivity) (by linarith)
  have h3 : A * ec ≤ A' * ec' := mul_le_mul hAA hecc hec (by linarith)
  have h4 : ea * C ≤ ea' * C' := mul_le_mul heaa hCC hC (by linarith)
  have h5 : ea * ec ≤ ea' * ec' := mul_le_mul heaa hecc hec (by linarith)
  linarith

/-- **`bnNormBudget` is monotone** in the rounding unit `u`, the centered bound `D`, and the two
    supplied statistic moduli — the fact that lets a symbolic `M.u` and a symbolic window be
    replaced by rationals before `norm_num` sees the expression. -/
theorem bnNormBudget_mono {u u' D D' S G Bb em em' ei ei' : ℝ}
    (hu : 0 ≤ u) (huu : u ≤ u') (hD : 0 ≤ D) (hDD : D ≤ D') (hS : 0 ≤ S) (hG : 0 ≤ G)
    (hBb : 0 ≤ Bb) (hem : 0 ≤ em) (hemm : em ≤ em') (hei : 0 ≤ ei) (heii : ei ≤ ei') :
    bnNormBudget u D S G Bb em ei ≤ bnNormBudget u' D' S G Bb em' ei' := by
  have hu'0 : 0 ≤ u' := hu.trans huu
  have hcen : u * (D + em) + em ≤ u' * (D' + em') + em' := by
    have : u * (D + em) ≤ u' * (D' + em') :=
      mul_le_mul huu (by linarith) (by linarith) hu'0
    linarith
  have hcen0 : 0 ≤ u * (D + em) + em := by positivity
  have hinner : FloatModel.mulErr u D S (u * (D + em) + em) ei
      ≤ FloatModel.mulErr u' D' S (u' * (D' + em') + em') ei' :=
    mulErr_mono hu huu hD hDD hS le_rfl hcen0 hcen hei heii
  have hinner0 : 0 ≤ FloatModel.mulErr u D S (u * (D + em) + em) ei :=
    mulErr_nonneg hu hD hS hcen0 hei
  have houter : FloatModel.mulErr u G (D * S) 0 (FloatModel.mulErr u D S (u * (D + em) + em) ei)
      ≤ FloatModel.mulErr u' G (D' * S) 0
          (FloatModel.mulErr u' D' S (u' * (D' + em') + em') ei') :=
    mulErr_mono hu huu hG le_rfl (by positivity) (mul_le_mul hDD le_rfl hS (by linarith))
      le_rfl le_rfl hinner0 hinner
  have houter0 : 0 ≤ FloatModel.mulErr u G (D * S) 0
      (FloatModel.mulErr u D S (u * (D + em) + em) ei) :=
    mulErr_nonneg hu hG (by positivity) le_rfl hinner0
  have hDS : G * (D * S) ≤ G * (D' * S) :=
    mul_le_mul_of_nonneg_left (mul_le_mul hDD le_rfl hS (by linarith)) hG
  have hDS0 : 0 ≤ G * (D * S) := by positivity
  unfold bnNormBudget
  have hmul : u * (G * (D * S)
        + FloatModel.mulErr u G (D * S) 0 (FloatModel.mulErr u D S (u * (D + em) + em) ei) + Bb)
      ≤ u' * (G * (D' * S)
        + FloatModel.mulErr u' G (D' * S) 0
            (FloatModel.mulErr u' D' S (u' * (D' + em') + em') ei') + Bb) :=
    mul_le_mul huu (by linarith) (by linarith) hu'0
  linarith

namespace FloatBridgesTo

/-- ⭐ **An envelope through a per-channel BatchNorm.** The BN leaf's window is
    `G·2Ā·S + β̄` plus the normalize chain's rounding, and its modulus is that rounding plus the
    real BN's input-sensitivity — both monotone, so a symbolic `M.u`, a symbolic window and the
    two supplied statistic moduli are replaced by rationals (`q`, `Ā`, `em`, `ei`) before
    `norm_num` sees either closing inequality. `Sq` and `Tq` are rational stand-ins for the two
    irrational constants the sensitivity carries, `1/√ε` and `1/(2ε√ε)`.

    ⚠ `em`/`ei` bound the SUPPLIED float mean/inv-stddev accuracy. A GPU `rsqrt` has no IEEE
    spec, so — exactly as EfficientNet's deployed sigmoid carries `esig` and ViT's `exp` carries
    `eexp` — the BN statistics are modelled, not derived, and the numbers below are conditional
    on those two moduli in the same way. -/
theorem Maps.bnPerChannelTensor3 {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (h * w) → ℝ) (emean eistd : ℝ → ℝ)
    {G Bbnd S : ℝ}
    (hoc : 0 < oc) (hhw : 0 < h * w) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (h * w) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (h * w) v ε| ≤ eistd A)
    (hSb : ∀ v : Vec (h * w), |bnIstd (h * w) v ε| ≤ S)
    {q em ei Sq Tq Ā Ē Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S)
    (hĀ0 : 0 ≤ Ā) (hĒ0 : 0 ≤ Ē)
    (hem : ∀ A, 0 ≤ A → A ≤ Ā → emean A ≤ em) (hei : ∀ A, 0 ≤ A → A ≤ Ā → eistd A ≤ ei)
    (hSq : 1 / Real.sqrt ε ≤ Sq) (hTq : 1 / (2 * ε * Real.sqrt ε) ≤ Tq)
    (hĀ' : G * (2 * Ā * S) + Bbnd + bnNormBudget q (2 * Ā) S G Bbnd em ei ≤ Ā')
    (hĒ' : bnNormBudget q (2 * Ā) S G Bbnd em ei
             + G * ((Ē + Ē) * Sq + 2 * Ā * (8 * Ā * Ē * Tq)) ≤ Ē') :
    (floatBridgesTo_bnPerChannelTensor3 M γ β fμ fistdv emean eistd hoc hhw hε hγ hβ
      hmean histd hSb).Maps Ā Ē Ā' Ē' := by
  have hu := M.u_nonneg
  have hq0 : 0 ≤ q := hu.trans hq
  have hsε : 0 < Real.sqrt ε := Real.sqrt_pos.mpr hε
  have hinv0 : (0:ℝ) ≤ 1 / Real.sqrt ε := by positivity
  have hlip0 : (0:ℝ) ≤ 1 / (2 * ε * Real.sqrt ε) := by positivity
  have hemn : ∀ A, 0 ≤ A → 0 ≤ emean A := fun A hA =>
    (abs_nonneg _).trans (hmean ⟨0, hoc⟩ A hA 0 (fun _ => by simpa using hA))
  have hein : ∀ A, 0 ≤ A → 0 ≤ eistd A := fun A hA =>
    (abs_nonneg _).trans (histd ⟨0, hoc⟩ A hA 0 (fun _ => by simpa using hA))
  constructor
  · intro A h0 hle
    show bnLeafMag M.u S G Bbnd emean eistd A ≤ Ā'
    unfold bnLeafMag
    have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := 2 * A) (D' := 2 * Ā)
      (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
      (hemn A h0) (hem A h0 hle) (hein A h0) (hei A h0 hle)
    have hmag : G * (2 * A * S) ≤ G * (2 * Ā * S) :=
      mul_le_mul_of_nonneg_left (by nlinarith) hG0
    linarith
  · intro A E h0 hE0 hle hEle
    show bnLeafMod M.u ε S G Bbnd emean eistd A E ≤ Ē'
    unfold bnLeafMod bnReluBudget
    have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := 2 * A) (D' := 2 * Ā)
      (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
      (hemn A h0) (hem A h0 hle) (hein A h0) (hei A h0 hle)
    -- the ReLU/input-sensitivity tail, term by term
    have ht1 : (E + E) * (1 / Real.sqrt ε) ≤ (Ē + Ē) * Sq :=
      mul_le_mul (by linarith) hSq hinv0 (by linarith)
    have hdiv : (8 * A * E) / (2 * ε * Real.sqrt ε) ≤ 8 * Ā * Ē * Tq := by
      rw [div_eq_mul_one_div]
      exact mul_le_mul (by nlinarith) hTq hlip0 (by nlinarith)
    have hdiv0 : (0:ℝ) ≤ (8 * A * E) / (2 * ε * Real.sqrt ε) := by positivity
    have ht2 : 2 * A * ((8 * A * E) / (2 * ε * Real.sqrt ε)) ≤ 2 * Ā * (8 * Ā * Ē * Tq) :=
      mul_le_mul (by linarith) hdiv hdiv0 (by linarith)
    have htail : G * ((E + E) * (1 / Real.sqrt ε)
          + 2 * A * ((8 * A * E) / (2 * ε * Real.sqrt ε)))
        ≤ G * ((Ē + Ē) * Sq + 2 * Ā * (8 * Ā * Ē * Tq)) :=
      mul_le_mul_of_nonneg_left (by linarith) hG0
    linarith

end FloatBridgesTo

namespace FloatBridgesTo

/-- ⭐ **An envelope through a per-channel INFERENCE BatchNorm.** Both closing inequalities are
    linear: the window is `G·(Ā+μ̄)·S + β̄` plus the normalize chain's rounding, and the modulus
    is that rounding plus the affine slope `G·S` on the inherited error. Contrast
    `Maps.bnPerChannelTensor3`, whose modulus carries the training-mode mean/variance shift and
    is quadratic in the window — the difference between a whole-net number that `norm_num` can
    check and one it cannot. -/
theorem Maps.bnEvalPC {oc h w : Nat} (M : FloatModel) {ε : ℝ} (γ β μ v sF : Vec oc)
    {G Bbnd Mb S es : ℝ}
    (hoc : 0 < oc) (hhw : 0 < h * w) (hε : 0 < ε) (hv : ∀ c, 0 ≤ v c)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd) (hμ : ∀ c, |μ c| ≤ Mb)
    (hes : ∀ c, |sF c - 1 / Real.sqrt (v c + ε)| ≤ es) (hS : 1 / Real.sqrt ε ≤ S)
    {q Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S) (hMb0 : 0 ≤ Mb) (hes0 : 0 ≤ es)
    (hĀ0 : 0 ≤ Ā)
    (hĀ' : G * ((Ā + Mb) * S) + Bbnd + bnNormBudget q (Ā + Mb) S G Bbnd 0 es ≤ Ā')
    (hĒ' : bnNormBudget q (Ā + Mb) S G Bbnd 0 es + G * S * Ē ≤ Ē') :
    (floatBridgesTo_bnPerChannelEvalTensor3 (h := h) (w := w) M γ β μ v sF
      hoc hhw hε hv hγ hβ hμ hes hS).Maps Ā Ē Ā' Ē' := by
  have hu := M.u_nonneg
  have hq0 : 0 ≤ q := hu.trans hq
  constructor
  · intro A h0 hle
    show bnEvalLeafMag M.u S G Bbnd Mb es A ≤ Ā'
    unfold bnEvalLeafMag
    have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := A + Mb) (D' := Ā + Mb)
      (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
      le_rfl le_rfl hes0 le_rfl
    have hmag : G * ((A + Mb) * S) ≤ G * ((Ā + Mb) * S) :=
      mul_le_mul_of_nonneg_left (mul_le_mul (by linarith) le_rfl hS0 (by linarith)) hG0
    linarith
  · intro A E h0 hE0 hle hEle
    show bnEvalLeafMod M.u S G Bbnd Mb es A E ≤ Ē'
    unfold bnEvalLeafMod
    have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := A + Mb) (D' := Ā + Mb)
      (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
      le_rfl le_rfl hes0 le_rfl
    have hlip : G * S * E ≤ G * S * Ē :=
      mul_le_mul_of_nonneg_left hEle (mul_nonneg hG0 hS0)
    linarith

end FloatBridgesTo

end Proofs
