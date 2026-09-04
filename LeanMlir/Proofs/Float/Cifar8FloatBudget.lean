import LeanMlir.Proofs.Float.Cifar8FloatBridge
import LeanMlir.Proofs.Float.FloatBudgetEnv
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

The method is the numeric ENVELOPE `FloatBridgesTo.Maps` (`FloatBudgetEnv.lean`): an upper
bound on the output window and on the output error, pushed through each layer by two
rational inequalities. Each conv/dense step uses `Maps.flatConv` / `Maps.dense`, whose
`layerBudget_le_num'` (the monotone form of the per-op Higham budget) takes the γ-term
through `FloatModel.gamma_num`, so `norm_num` discharges every step with no big-power
evaluation; `relu` and the pools pass the envelope through unchanged. Eleven rounding
stages, twenty-two numerals, twenty-four generic `Maps.comp`s and one `exact`.

⚠ **This file used to carry its own kit.** `FloatBridgesTo.Env` — the same pair of numbers,
but with the input error FIXED at `0` — lived here with one `Env.comp_*` lemma per
operation, because it was the first whole-net budget in the repo and nothing else needed a
kit. `Maps` quantifies over the input window and the inherited error instead, which is what
makes `Maps.comp` generic and what lets a skip be expressed at all (`FloatBudgetEnv.lean`
header). The per-stage inequalities are identical — the numerals below are the ones `Env`
carried, unchanged — and the statement is strictly stronger: `Maps 1 0 …` holds at every
input window `A ≤ 1`, where `Env 1 …` held only at `A = 1`.

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
/-- **The envelope, kernel-checked.** At `|W| ≤ 2/5`, `|b| ≤ 1/100`, input window `≤ 1`,
    no inherited input error and `u ≤ 2⁻²⁴`: output window `≤ 6.121·10¹⁸`, output error
    `≤ 6.37·10¹⁴`. Eleven rounding stages, each with its γ-term through `gamma_num` and two
    rational inequalities by `norm_num`, threaded by twenty-four generic `Maps.comp`s; the
    per-stage numerals are the exact fold rounded up to four significant figures, and the
    kernel re-checks every inequality. The chain is built bottom-up so each step elaborates
    against a small, fully determined type; the closing `exact` is one structural comparison
    with `cifar8Bridge`'s definition. All four numerals are pinned on every rounding leaf
    (`planning/float_budget_numbers.md` §3.7(c)) — a `by norm_num` inside a `have` runs
    before `Maps.comp` unifies anything, so an unpinned output window is a metavariable and
    the failure reads like arithmetic. -/
theorem cifar8Bridge_maps (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100)) :
    (cifar8Bridge M W).Maps 1 0 6121000000000000000 637000000000000 := by
  have hw : (0:ℝ) ≤ 2/5 := by norm_num
  have hβ : (0:ℝ) ≤ 1/100 := by norm_num
  have m1 := FloatBridgesTo.Maps.flatConv (h := 32) (w := 32) M W.cW1 W.cb1 hw hβ
      (by norm_num) W.hcW1 W.hcb1
      (Ā := 1) (Ē := 0) (Ā' := 541/50) (Ē' := 1871/100000000)
      (M.gamma_num (q := 173/100000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num)
  have m2 := FloatBridgesTo.Maps.comp (by norm_num) m1
    (FloatBridgesTo.Maps.relu (Ā := 541/50) (Ē := 1871/100000000))
  have m3 := FloatBridgesTo.Maps.comp (by norm_num) m2
    (FloatBridgesTo.Maps.flatConv (h := 32) (w := 32) M W.cW2 W.cb2 hw hβ
      (by norm_num) W.hcW2 W.hcb2
      (Ā := 541/50) (Ē := 1871/100000000) (Ā' := 6233/10) (Ē' := 6507/1000000)
      (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m4 := FloatBridgesTo.Maps.comp (by norm_num) m3
    (FloatBridgesTo.Maps.relu (Ā := 6233/10) (Ē := 6507/1000000))
  have m5 := FloatBridgesTo.Maps.comp (by norm_num) m4
    (FloatBridgesTo.Maps.maxPool (c := 16) (h := 16) (w := 16) (Ā := 6233/10) (Ē := 6507/1000000))
  have m6 := FloatBridgesTo.Maps.comp (by norm_num) m5
    (FloatBridgesTo.Maps.flatConv (h := 16) (w := 16) M W.cW3 W.cb3 hw hβ
      (by norm_num) W.hcW3 W.hcb3
      (Ā := 6233/10) (Ē := 6507/1000000) (Ā' := 35910) (Ē' := 1719/2500)
      (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m7 := FloatBridgesTo.Maps.comp (by norm_num) m6
    (FloatBridgesTo.Maps.relu (Ā := 35910) (Ē := 1719/2500))
  have m8 := FloatBridgesTo.Maps.comp (by norm_num) m7
    (FloatBridgesTo.Maps.flatConv (h := 16) (w := 16) M W.cW4 W.cb4 hw hβ
      (by norm_num) W.hcW4 W.hcb4
      (Ā := 35910) (Ē := 1719/2500) (Ā' := 2069000) (Ē' := 5763/100)
      (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m9 := FloatBridgesTo.Maps.comp (by norm_num) m8
    (FloatBridgesTo.Maps.relu (Ā := 2069000) (Ē := 5763/100))
  have m10 := FloatBridgesTo.Maps.comp (by norm_num) m9
    (FloatBridgesTo.Maps.maxPool (c := 16) (h := 8) (w := 8) (Ā := 2069000) (Ē := 5763/100))
  have m11 := FloatBridgesTo.Maps.comp (by norm_num) m10
    (FloatBridgesTo.Maps.flatConv (h := 8) (w := 8) M W.cW5 W.cb5 hw hβ
      (by norm_num) W.hcW5 W.hcb5
      (Ā := 2069000) (Ē := 5763/100) (Ā' := 119200000) (Ē' := 4358)
      (M.gamma_num (q := 871/100000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m12 := FloatBridgesTo.Maps.comp (by norm_num) m11
    (FloatBridgesTo.Maps.relu (Ā := 119200000) (Ē := 4358))
  have m13 := FloatBridgesTo.Maps.comp (by norm_num) m12
    (FloatBridgesTo.Maps.flatConv (h := 8) (w := 8) M W.cW6 W.cb6 hw hβ
      (by norm_num) W.hcW6 W.hcb6
      (Ā := 119200000) (Ē := 4358) (Ā' := 13740000000) (Ē' := 739700)
      (M.gamma_num (q := 173/10000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m14 := FloatBridgesTo.Maps.comp (by norm_num) m13
    (FloatBridgesTo.Maps.relu (Ā := 13740000000) (Ē := 739700))
  have m15 := FloatBridgesTo.Maps.comp (by norm_num) m14
    (FloatBridgesTo.Maps.maxPool (c := 32) (h := 4) (w := 4) (Ā := 13740000000) (Ē := 739700))
  have m16 := FloatBridgesTo.Maps.comp (by norm_num) m15
    (FloatBridgesTo.Maps.flatConv (h := 4) (w := 4) M W.cW7 W.cb7 hw hβ
      (by norm_num) W.hcW7 W.hcb7
      (Ā := 13740000000) (Ē := 739700) (Ā' := 1583000000000) (Ē' := 112600000)
      (M.gamma_num (q := 173/10000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m17 := FloatBridgesTo.Maps.comp (by norm_num) m16
    (FloatBridgesTo.Maps.relu (Ā := 1583000000000) (Ē := 112600000))
  have m18 := FloatBridgesTo.Maps.comp (by norm_num) m17
    (FloatBridgesTo.Maps.flatConv (h := 4) (w := 4) M W.cW8 W.cb8 hw hβ
      (by norm_num) W.hcW8 W.hcb8
      (Ā := 1583000000000) (Ē := 112600000) (Ā' := 182400000000000) (Ē' := 16130000000)
      (M.gamma_num (q := 173/10000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m19 := FloatBridgesTo.Maps.comp (by norm_num) m18
    (FloatBridgesTo.Maps.relu (Ā := 182400000000000) (Ē := 16130000000))
  have m20 := FloatBridgesTo.Maps.comp (by norm_num) m19
    (FloatBridgesTo.Maps.maxPool (c := 32) (h := 2) (w := 2) (Ā := 182400000000000) (Ē := 16130000000))
  have m21 := FloatBridgesTo.Maps.comp (by norm_num) m20
    (FloatBridgesTo.Maps.dense M W.dW1 W.db1 hw hβ (by norm_num) W.hdW1 W.hdb1
      (Ā := 182400000000000) (Ē := 16130000000) (Ā' := 9339000000000000) (Ē' := 898300000000)
      (M.gamma_num (q := 31/4000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m22 := FloatBridgesTo.Maps.comp (by norm_num) m21
    (FloatBridgesTo.Maps.relu (Ā := 9339000000000000) (Ē := 898300000000))
  have m23 := FloatBridgesTo.Maps.comp (by norm_num) m22
    (FloatBridgesTo.Maps.dense M W.dW2 W.db2 hw hβ (by norm_num) W.hdW2 W.hdb2
      (Ā := 9339000000000000) (Ē := 898300000000) (Ā' := 239100000000000000) (Ē' := 23940000000000)
      (M.gamma_num (q := 197/50000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  have m24 := FloatBridgesTo.Maps.comp (by norm_num) m23
    (FloatBridgesTo.Maps.relu (Ā := 239100000000000000) (Ē := 23940000000000))
  have m25 := FloatBridgesTo.Maps.comp (by norm_num) m24
    (FloatBridgesTo.Maps.dense M W.dW3 W.db3 hw hβ (by norm_num) W.hdW3 W.hdb3
      (Ā := 239100000000000000) (Ē := 23940000000000) (Ā' := 6121000000000000000) (Ē' := 637000000000000)
      (M.gamma_num (q := 197/50000000) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
  exact m25

/-- The committed CIFAR-8 bridge's fresh budget at the He profile: `≤ 6.37·10¹⁴`. -/
theorem cifar8Bridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100)) :
    (cifar8Bridge M W).fresh 1 ≤ 637000000000000 :=
  (cifar8Bridge_maps M hMu W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- The committed CIFAR-8 bridge's certified output window at the He profile: `≤ 6.121·10¹⁸`
    — the worst-case logit magnitude the interval fold can promise. -/
theorem cifar8Bridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32)
    (W : Cifar8Weights (2/5) (1/100)) :
    (cifar8Bridge M W).mag 1 ≤ 6121000000000000000 :=
  (cifar8Bridge_maps M hMu W).mag_le 1 (by norm_num) le_rfl

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
  (cifar8Bridge_maps M hMu W).budget_le (by norm_num) le_rfl x hx j

end Proofs
