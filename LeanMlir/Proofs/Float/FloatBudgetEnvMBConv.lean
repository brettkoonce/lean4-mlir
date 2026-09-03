import LeanMlir.Proofs.Float.FloatBudgetEnv
import LeanMlir.Proofs.Float.MobileNetV2WholeFloatBridge

/-! # `FloatBridgesTo.Maps` leaves for the inverted-bottleneck family

`FloatBudgetEnv.lean` holds the kit and the leaves ResNet-34 needed: `relu`, `maxPool`,
`flatConv`, `flatConvStride2`, `dense`, `gap`, the two BatchNorms, and the `comp`/`residual`/
`biPathSum` combinators. This file holds the ones the MBConv family needs on top — the depthwise
convolutions, `relu6`, and EfficientNet's smooth activations and squeeze-excite.

⚠ **Why they are here and not in `FloatBudgetEnv.lean`.** A `Maps` lemma names its bridge, so it
can only live where that bridge is in scope. `floatBridgesTo_depthwise` is in
`DepthwiseFloatBridge.lean`, `floatBridgesTo_swish` / `_sigmoid` / `_broadcast` and
`FloatBridgesTo.seScale` in `EnetFloatBridge.lean`, `FloatBridgesTo.batchMap` in
`EfficientNetBackFloatBridge.lean`, `_depthwiseStride2Flat` in
`EfficientNetWholeFloatBridge.lean`, `floatBridgesTo_relu6` in
`MobileNetV2WholeFloatBridge.lean` — and none of those is on `FloatBudgetEnv`'s import path.
Pulling them down there would make the ResNet-34 budget depend on the whole MobileNet/EfficientNet
cone for nothing.

Every leaf here is the `Maps.flatConv` mould: `show` the unfolded `mag`/`mod`, one monotone
lemma, `linarith`. The two that are NOT are the point of the file:

* ⭐ `Maps.relu6` **clamps** — its window step is `min Ā 6`, not `Ā`, because `relu6` is bounded
  by `6` whatever its input. On MobileNetV2 that is 97 orders of certified window
  (`planning/float_budget_numbers.md` §3.2).
* ⭐ `Maps.swish`'s modulus is the **`min`** of a multiplicative and an additive input
  sensitivity. The multiplicative branch alone multiplies the inherited error by the window at
  every swish site; on EfficientNet-B0 that is the difference between a budget of `10¹⁸⁶` and one
  of `10¹⁷³⁷`, and `norm_num` refuses numerals past ~`10³⁰⁰` (§3.4).
-/

namespace Proofs

open FloatModel

namespace FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The exact leaves (magnitude-stable or clamping, modulus `id`)
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **An envelope through `relu6` — and it CLAMPS.** Unlike `Maps.relu`, whose window step is
    the identity, relu6's output window is `min Ā 6`: relu6 is both magnitude-nonincreasing and
    bounded by `6` whatever its input (`relu6_le_six`), so a relu6 site RESETS the certified
    magnitude instead of inheriting it. Exact in float, so the modulus is still the identity and
    the error passes through untouched — the clamp buys window, not budget. -/
theorem Maps.relu6 {n : Nat} {Ā Ē Ā' Ē' : ℝ} (hĀ' : min Ā 6 ≤ Ā') (hĒ' : Ē ≤ Ē') :
    (floatBridgesTo_relu6 (n := n)).Maps Ā Ē Ā' Ē' where
  mag_le := fun _A _h0 hle => (min_le_min hle le_rfl).trans hĀ'
  mod_le := fun _A _E _h0 _hE0 _hle hEle => hEle.trans hĒ'

/-- The channel broadcast passes an envelope through unchanged (a structural gather). -/
theorem Maps.broadcast {c h w : Nat} {Ā Ē : ℝ} :
    (floatBridgesTo_broadcast (c := c) (h := h) (w := w)).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- **An envelope survives the batch lift, unchanged.** `FloatBridgesTo.batchMap`'s `mag`/`mod`
    ARE the per-example bridge's, so applying one function to `N` independent examples cannot
    change either the window or the budget — which is the envelope-level statement of "this op
    does not couple the batch". Every EfficientNet stage but true batch-norm is of this shape, and
    at inference so is that one (`EfficientNetRenderPCEval.lean`). -/
theorem Maps.batchMap {a b : Nat} (N : Nat) {f fF : Vec a → Vec b} {bf : FloatBridgesTo f fF}
    {Ā Ē Ā' Ē' : ℝ} (h : bf.Maps Ā Ē Ā' Ē') : (bf.batchMap N).Maps Ā Ē Ā' Ē' :=
  ⟨h.mag_le, h.mod_le⟩

-- ════════════════════════════════════════════════════════════════
-- § The depthwise convolutions
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through a depthwise convolution** — `Maps.flatConv` at fan-in `kH·kW`: a
    depthwise kernel touches one input channel, so its receptive field is the window alone and
    the `ic` factor drops out. Same two rational inequalities. -/
theorem Maps.depthwise {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (bb : Vec c) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < c * h * w)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |bb ch| ≤ β)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((kH * kW : ℕ) : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * (((kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + β)
            + ((kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_depthwise (h := h) (w := w) M W bb hw' hβ hn hW hb).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show layerAct (kH * kW) w' β A + layerBudget M.u (kH * kW) w' β A 0 ≤ Ā'
    have h1 := layerAct_le_num' (m := kH * kW) (β := β) hw' hle
    have h2 := layerBudget_le_num' (m := kH * kW) M.u_nonneg hw' hβ h0 hle
      (le_refl (0:ℝ)) (le_refl (0:ℝ)) hg
    simp only [add_zero, mul_zero] at h2
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show layerBudget M.u (kH * kW) w' β A E ≤ Ē'
    exact (layerBudget_le_num' M.u_nonneg hw' hβ h0 hle hE0 hEle hg).trans hĒ'

/-- **An envelope through a stride-2 depthwise convolution** — `Maps.depthwise` verbatim:
    decimating the output picks coordinates, so the fan-in and hence the budget are unchanged. -/
theorem Maps.depthwiseStride2Flat {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (bb : Vec c) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < c * (2 * h) * (2 * w))
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |bb ch| ≤ β)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((kH * kW : ℕ) : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * (((kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + β)
            + ((kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_depthwiseStride2Flat (h := h) (w := w) M W bb hw' hβ hn hW hb).Maps
      Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show layerAct (kH * kW) w' β A + layerBudget M.u (kH * kW) w' β A 0 ≤ Ā'
    have h1 := layerAct_le_num' (m := kH * kW) (β := β) hw' hle
    have h2 := layerBudget_le_num' (m := kH * kW) M.u_nonneg hw' hβ h0 hle
      (le_refl (0:ℝ)) (le_refl (0:ℝ)) hg
    simp only [add_zero, mul_zero] at h2
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show layerBudget M.u (kH * kW) w' β A E ≤ Ē'
    exact (layerBudget_le_num' M.u_nonneg hw' hβ h0 hle hE0 hEle hg).trans hĒ'

-- ════════════════════════════════════════════════════════════════
-- § EfficientNet's smooth activations and the squeeze-excite rescale
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through the deployed sigmoid.** Both inequalities are independent of the input
    window: `σ ∈ (0,1)` bounds the real gate and `esig` the deployed one, so the output window is
    `1 + esig` at ANY input magnitude — the SE branch cannot blow up. The modulus is `esig` plus
    σ's `¼`-Lipschitz input shift. -/
theorem Maps.sigmoid {n : Nat} (fsig : ℝ → ℝ) {esig : ℝ}
    (hesig : 0 ≤ esig) (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    {Ā Ē Ā' Ē' : ℝ} (hĀ' : 1 + esig ≤ Ā') (hĒ' : esig + (1/4) * Ē ≤ Ē') :
    (floatBridgesTo_sigmoid (n := n) fsig hesig hsig).Maps Ā Ē Ā' Ē' where
  mag_le := fun _A _h0 _hle => hĀ'
  mod_le := fun _A E _h0 _hE0 _hle hEle => by
    show esig + (1/4) * E ≤ Ē'
    have : (1/4 : ℝ) * E ≤ (1/4) * Ē := by linarith
    linarith

/-- ⭐ **An envelope through swish.** The window is `Ā` plus one rounded product against the
    deployed sigmoid — swish is magnitude-NON-increasing (`|x·σ(x)| ≤ |x|`), so like `relu` it
    passes the window through; unlike `relu6` it does not reset it.

    The modulus is the `min` of swish's two input sensitivities, and that `min` is load-bearing.
    `(1 + Ā/4)·Ē` (`swishScalar_lipschitz_abs`, from σ's `¼`-Lipschitz) multiplies the inherited
    error by the WINDOW at every swish site; `Ā + Ē` (`swishScalar_lipschitz_abs'`, bounding
    `|σa − σb|` by the gate's own range instead) is additive in it. They are incomparable — the
    first wins at small separation, the second at large window — and on EfficientNet-B0 keeping
    only the first puts the whole-net budget at `10¹⁷³⁷`, past what `norm_num` will evaluate. -/
theorem Maps.swish {n : Nat} (M : FloatModel) (fsig : ℝ → ℝ) {esig : ℝ}
    (hesig : 0 ≤ esig) (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    {q Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q) (hĀ0 : 0 ≤ Ā)
    (hĀ' : Ā + FloatModel.mulErr q Ā 1 0 esig ≤ Ā')
    (hĒ' : FloatModel.mulErr q Ā 1 0 esig + min ((1 + Ā/4) * Ē) (Ā + Ē) ≤ Ē') :
    (floatBridgesTo_swish (n := n) M fsig hesig hsig).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show A + FloatModel.mulErr M.u A 1 0 esig ≤ Ā'
    have := mulErr_mono (u := M.u) (u' := q) (A := A) (A' := Ā) (C := 1) (C' := 1)
      (ea := 0) (ea' := 0) (ec := esig) (ec' := esig)
      M.u_nonneg hq h0 hle (by norm_num) le_rfl le_rfl le_rfl hesig le_rfl
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show FloatModel.mulErr M.u A 1 0 esig + min ((1 + A/4) * E) (A + E) ≤ Ē'
    have hme := mulErr_mono (u := M.u) (u' := q) (A := A) (A' := Ā) (C := 1) (C' := 1)
      (ea := 0) (ea' := 0) (ec := esig) (ec' := esig)
      M.u_nonneg hq h0 hle (by norm_num) le_rfl le_rfl le_rfl hesig le_rfl
    have hmin : min ((1 + A/4) * E) (A + E) ≤ min ((1 + Ā/4) * Ē) (Ā + Ē) :=
      min_le_min (mul_le_mul (by linarith) hEle hE0 (by linarith)) (by linarith)
    linarith

/-- ⭐ **An envelope through the squeeze-excite rescale `x ⊙ gate(x)`.** The window is the input
    window times the gate's certified MAGNITUDE, plus one rounding — the gate's ERROR does not
    enter it, because `FloatClose`'s magnitude clause bounds the float gate as well as the real
    one. Deriving the window instead as `|float − real| + |real|` charges `Ā · Eg`, and on
    EfficientNet-B0 that is 10¹⁸ per SE site (`planning/float_budget_numbers.md` §3.4).

    ⚠ The MODULUS is a different story and is not slack: `mulErr q Ā Cg Ē Eg` carries `Ā · Eg`,
    the block window times the gate error — and the gate grows that error out of the same window
    through the squeeze's `GAP → dense`. **SE is quadratic in the window**, for the same
    structural reason training-mode BatchNorm and LayerNorm are (§0.1): an op that consumes a
    reduction of its own input and then multiplies by that input. B0 survives three such sites. -/
theorem Maps.seScale {m : Nat} (M : FloatModel) {g gF : Vec m → Vec m}
    {bg : FloatBridgesTo g gF} (hm : 0 < m)
    {q Ā Ē Cg Eg Ā' Ē' : ℝ} (hg : bg.Maps Ā Ē Cg Eg) (hq : M.u ≤ q) (hĀ0 : 0 ≤ Ā)
    (hĀ' : Ā * Cg + q * (Ā * Cg) ≤ Ā')
    (hĒ' : FloatModel.mulErr q Ā Cg Ē Eg ≤ Ē') :
    (FloatBridgesTo.seScale M bg hm).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show A * bg.mag A + M.u * (A * bg.mag A) ≤ Ā'
    have hgm := hg.mag_le A h0 hle
    have hg0 : 0 ≤ bg.mag A := bg.mag_nonneg h0
    have hu := M.u_nonneg
    have hprod : A * bg.mag A ≤ Ā * Cg := mul_le_mul hle hgm hg0 hĀ0
    have hp0 : 0 ≤ A * bg.mag A := mul_nonneg h0 hg0
    have : M.u * (A * bg.mag A) ≤ q * (Ā * Cg) := mul_le_mul hq hprod hp0 (hu.trans hq)
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show FloatModel.mulErr M.u A (bg.mag A) E (bg.mod A E) ≤ Ē'
    exact (mulErr_mono M.u_nonneg hq h0 hle (bg.mag_nonneg h0) (hg.mag_le A h0 hle)
      hE0 hEle (bg.mod_nonneg h0 hE0 hm) (hg.mod_le A E h0 hE0 hle hEle)).trans hĒ'

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § One worked site, so the leaves are not unexercised
-- ════════════════════════════════════════════════════════════════

/-! ⚠ A `Maps` leaf that nothing composes is a leaf nobody has checked composes — the
`stale lean_exe gates` failure mode in proof form. Until `EfficientNetFloatBudget.lean` lands,
this closes EfficientNet-B0's **b1 squeeze-excite site** end to end at the numerals
`scripts/float_budget_envelope.py`'s `b0_eval_chain` emits (`|param| ≤ 41/10` — the measured
350-epoch profile — `esig = 10⁻²`, `u ≤ 2⁻²⁴`), exercising five of the six leaves above in one
chain: `gap → dense → swish → dense → sigmoid → broadcast`, then the rescale.

It is also the check that the generator and the lemmas agree: every numeral below is copied from
the fold, and each `norm_num` is the kernel confirming the fold's arithmetic was this file's. -/
set_option maxHeartbeats 1000000 in
example (M : FloatModel) (hMu : M.u ≤ u32) (fsig : ℝ → ℝ)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ 1/100)
    (W₁ : Mat 32 8) (b₁ : Vec 8) (W₂ : Mat 8 32) (b₂ : Vec 32)
    (hW₁ : ∀ i j, |W₁ i j| ≤ 41/10) (hb₁ : ∀ j, |b₁ j| ≤ 41/10)
    (hW₂ : ∀ i j, |W₂ i j| ≤ 41/10) (hb₂ : ∀ j, |b₂ j| ≤ 41/10) :
    (FloatBridgesTo.seScale M
      (floatBridgesTo_seGate (c := 32) (h := 112) (w := 112) (r := 8) M fsig W₁ b₁ W₂ b₂
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        hsig hW₁ hb₁ hW₂ hb₂) (by norm_num)).Maps
      (7572 * 10 ^ 6) (1507 * 10 ^ 7) (7648 * 10 ^ 6) (5543 * 10 ^ 20) := by
  have g1 := FloatBridgesTo.Maps.gap (c := 32) (h := 112) (w := 112) M (by norm_num) (by norm_num)
    hMu (by norm_num [u32]) (by norm_num)
    (M.gamma_num (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 7572 * 10 ^ 6) (Ē := 1507 * 10 ^ 7)
    (Ā' := 7578 * 10 ^ 6) (Ē' := 1508 * 10 ^ 7) (by norm_num [u32]) (by norm_num [u32])
  have g2 := g1.comp (by norm_num) (FloatBridgesTo.Maps.dense M W₁ b₁ (by norm_num) (by norm_num)
    (by norm_num) hW₁ hb₁
    (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 9943 * 10 ^ 8) (Ē' := 1979 * 10 ^ 9) (by norm_num [u32]) (by norm_num [u32]))
  have g3 := g2.comp (by norm_num) (FloatBridgesTo.Maps.swish (n := 8) M fsig (by norm_num) hsig
    hMu (by norm_num)
    (Ā' := 1005 * 10 ^ 9) (Ē' := 2984 * 10 ^ 9)
    (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
  have g4 := g3.comp (by norm_num) (FloatBridgesTo.Maps.dense M W₂ b₂ (by norm_num) (by norm_num)
    (by norm_num) hW₂ hb₂
    (M.gamma_num (q := 5961 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 3297 * 10 ^ 10) (Ē' := 9788 * 10 ^ 10) (by norm_num [u32]) (by norm_num [u32]))
  have g5 := g4.comp (by norm_num) (FloatBridgesTo.Maps.sigmoid (n := 32) fsig (by norm_num) hsig
    (Ā' := 1010 / 10 ^ 3) (Ē' := 2448 * 10 ^ 10) (by norm_num) (by norm_num))
  have g6 := g5.comp (by norm_num) (FloatBridgesTo.Maps.broadcast (c := 32) (h := 112) (w := 112))
  exact FloatBridgesTo.Maps.seScale M (by norm_num) g6 hMu (by norm_num)
    (by norm_num [u32]) (by norm_num [FloatModel.mulErr, u32])

end Proofs
