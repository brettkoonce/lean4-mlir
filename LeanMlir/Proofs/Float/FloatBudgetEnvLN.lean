import LeanMlir.Proofs.Float.FloatBudgetEnvMBConv
import LeanMlir.Proofs.Float.ConvNeXtWholeFloatBridge

/-! # `FloatBridgesTo.Maps` leaves for the LayerNorm family (ConvNeXt / ViT)

`FloatBudgetEnv.lean` holds the kit and ResNet-34's leaves; `FloatBudgetEnvMBConv.lean` holds
the inverted-bottleneck family's. This file holds the ones a LayerNorm net needs: the capped
pure-normalise LN, the two halves of its affine (`diagBack` and `biasAdd`), the structural
gathers and the per-row lift they are conjugated by, GELU, and the stride-4 patchify conv.

⚠ Same reason they are here and not in `FloatBudgetEnv.lean`: a `Maps` lemma names its bridge,
and `floatBridgesTo_diagBack` / `_biasAdd` / `_gather` / `_gelu` / `_flatConvStride4` /
`FloatBridgesTo.perRow` are spread over `LinBackFloatBridge`, `ChannelLNFloatBridge`,
`ViTBlockFloatBridge`, `ViTFloatBridge` and `ConvNeXtWholeFloatBridge`, none of which is on
`FloatBudgetEnv`'s import path. ⚠ It imports `FloatBudgetEnvMBConv` rather than
`FloatBudgetEnv` for one leaf: `Maps.depthwise`. ConvNeXt's 7×7 depthwise is the same op the
MBConv family uses, and duplicating its envelope under a second name to avoid the import would
be worse than the coupling — but note the coupling is real, and a ConvNeXt budget therefore
rebuilds when the MobileNetV2 / EfficientNet kit changes.

⭐ It also holds the three **modelled device kernels** the LayerNorm and transformer families
run on — `DeviceLN` (the run-time mean and inverse-stddev), `DeviceGelu` and `DeviceExp` — with
the capped LN site's bridge (`DeviceLN.bridgeAt`) and envelope (`DeviceLN.mapsAt`) built on
them. They were `ConvNeXtFloatBudget.lean`'s until ViT-Tiny needed the same three: a budget
file should not own a structure a sibling budget file imports.

Two of these leaves are the point of the file:

* ⛔ `Maps.bnCapped` is the **capped** LayerNorm — `FloatBridgesTo.capped` applied to the
  pure-normalise `floatBridgesTo_bn`. It takes only the WINDOW inequality; the modulus is
  `2·Ā'` by the triangle inequality. That is a weaker claim than a fold and every number built
  through it must say so (`planning/float_budget_numbers.md` §9). It is here because LayerNorm
  reduces its statistics out of its own input and its modulus is therefore quadratic in the
  window, with no eval-mode escape the way inference BatchNorm has one (§0.1).
* ⭐ `Maps.gelu` closes through the **`3/2` branch** of `floatClose_gelu`'s `min` — the global
  saturation constant (`Architectures/GeluSaturation.lean`), not the cubic-in-the-window
  polynomial. At ConvNeXt-T's magnitudes the polynomial branch is ~250× worse, and 18 GELU
  sites of that is on its own enough to put the whole-net budget past `norm_num`'s ceiling.
-/

namespace Proofs

open FloatModel

namespace FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The structural leaves (exact in float, envelope unchanged)
-- ════════════════════════════════════════════════════════════════

/-- A reindex passes an envelope through unchanged (a permutation rounds nothing). -/
theorem Maps.gather {p q : Nat} (e : Fin p ≃ Fin q) {Ā Ē : ℝ} :
    (floatBridgesTo_gather e).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- The identity passes an envelope through unchanged. -/
theorem Maps.idVec {m : Nat} {Ā Ē : ℝ} : (floatBridgesTo_idVec (m := m)).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- **An envelope survives the per-row lift, unchanged.** `FloatBridgesTo.perRow`'s `mag`/`mod`
    ARE the per-row bridge's, so applying one map to `n` independent rows changes neither the
    window nor the budget — the envelope-level statement that the op does not mix rows. The
    `Maps.batchMap` of the LayerNorm family. -/
theorem Maps.perRow (n : Nat) {d : Nat} {f fF : Vec d → Vec d} {b : FloatBridgesTo f fF}
    {Ā Ē Ā' Ē' : ℝ} (hb : b.Maps Ā Ē Ā' Ē') : (FloatBridgesTo.perRow n b).Maps Ā Ē Ā' Ē' :=
  ⟨hb.mag_le, hb.mod_le⟩

-- ════════════════════════════════════════════════════════════════
-- § The capped LayerNorm
-- ════════════════════════════════════════════════════════════════

/-- ⛔ **An envelope through a CAPPED pure-normalise LayerNorm.** `layerNormForward m ε 1 0 =
    bnForward m ε 1 0` definitionally, so the leaf is `floatBridgesTo_bn` — and its modulus is
    `bnReluBudget`, which carries the training-mode mean-and-variance shift `G·2A·(8A·e/(2ε√ε))`
    and is therefore **quadratic in the window**. LayerNorm has no running statistics, so unlike
    BatchNorm there is no inference variant to switch to (`planning/float_budget_numbers.md`
    §0.1), and the fold squares at every LN site.

    `FloatBridgesTo.capped` is the escape: only the window inequality is asserted here, and the
    output error is `2·Ā'` by the triangle inequality. ⚠ **A whole-net number built through this
    leaf is not a fold** — it says the float and real forwards both land in the certified window
    — and must be labelled wherever it is stated (§9). The window itself is honest: it is the
    same `G·(2Ā·S) + β̄ + bnNormBudget` the uncapped leaf proves. -/
theorem Maps.bnCapped {m : Nat} (M : FloatModel) {ε γ β : ℝ}
    (fμ fistdv : Vec m → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd S : ℝ}
    (hm : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean A)
    (histd : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd A)
    (hS : ∀ v : Vec m, |bnIstd m v ε| ≤ S)
    {q em ei Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S)
    (hem : ∀ A, 0 ≤ A → A ≤ Ā → emean A ≤ em) (hei : ∀ A, 0 ≤ A → A ≤ Ā → eistd A ≤ ei)
    (hĀ' : G * (2 * Ā * S) + Bbnd + bnNormBudget q (2 * Ā) S G Bbnd em ei ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    ((floatBridgesTo_bn M fμ fistdv emean eistd hm hε hγ hβ hmean histd hS).capped).Maps
      Ā Ē Ā' Ē' := by
  have hu := M.u_nonneg
  have hemn : ∀ A, 0 ≤ A → 0 ≤ emean A := fun A hA =>
    (abs_nonneg _).trans (hmean A hA 0 (fun _ => by simpa using hA))
  have hein : ∀ A, 0 ≤ A → 0 ≤ eistd A := fun A hA =>
    (abs_nonneg _).trans (histd A hA 0 (fun _ => by simpa using hA))
  refine Maps.capped (fun A h0 hle => ?_) hĒ'
  show bnLeafMag M.u S G Bbnd emean eistd A ≤ Ā'
  unfold bnLeafMag
  have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := 2 * A) (D' := 2 * Ā)
    (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
    (hemn A h0) (hem A h0 hle) (hein A h0) (hei A h0 hle)
  have hmag : G * (2 * A * S) ≤ G * (2 * Ā * S) :=
    mul_le_mul_of_nonneg_left (by nlinarith) hG0
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The LayerNorm affine: the diagonal scale and the bias translation
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through a diagonal scale `x ↦ s ⊙ x`** — the LN γ multiply, and ConvNeXt's
    layer scale (`layerScale γ = diagBack γ` definitionally, at `es = 0` since γ is a stored
    weight with no transcendental in it). One rounded multiply per coordinate. -/
theorem Maps.diagBack {n : Nat} (M : FloatModel) (s fs : Vec n) {Sd es : ℝ}
    (hn : 0 < n) (hs : ∀ i, |s i| ≤ Sd) (hfs : ∀ i, |fs i - s i| ≤ es)
    {q Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q) (hSd0 : 0 ≤ Sd) (hes0 : 0 ≤ es)
    (hĀ' : Sd * Ā + FloatModel.mulErr q Sd Ā es 0 ≤ Ā')
    (hĒ' : FloatModel.mulErr q Sd Ā es 0 + Sd * Ē ≤ Ē') :
    (floatBridgesTo_diagBack M s fs hn hs hfs).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show Sd * A + FloatModel.mulErr M.u Sd A es 0 ≤ Ā'
    have hme := mulErr_mono (u := M.u) (u' := q) (A := Sd) (A' := Sd) (C := A) (C' := Ā)
      (ea := es) (ea' := es) (ec := 0) (ec' := 0)
      M.u_nonneg hq hSd0 le_rfl h0 hle hes0 le_rfl le_rfl le_rfl
    have : Sd * A ≤ Sd * Ā := mul_le_mul_of_nonneg_left hle hSd0
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show FloatModel.mulErr M.u Sd A es 0 + Sd * E ≤ Ē'
    have hme := mulErr_mono (u := M.u) (u' := q) (A := Sd) (A' := Sd) (C := A) (C' := Ā)
      (ea := es) (ea' := es) (ec := 0) (ec' := 0)
      M.u_nonneg hq hSd0 le_rfl h0 hle hes0 le_rfl le_rfl le_rfl
    have : Sd * E ≤ Sd * Ē := mul_le_mul_of_nonneg_left hEle hSd0
    linarith

/-- **An envelope through the bias translation `z ↦ z + β`** — the LN β shift. One rounded add
    per coordinate, so the window gains `β̄` and one rounding and the error gains that rounding. -/
theorem Maps.biasAdd {n : Nat} (M : FloatModel) (β : Vec n) {Bb : ℝ}
    (hn : 0 < n) (hβ : ∀ k, |β k| ≤ Bb)
    {q Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q) (hBb0 : 0 ≤ Bb)
    (hĀ' : Ā + Bb + q * (Ā + Bb) ≤ Ā') (hĒ' : q * (Ā + Bb) + Ē ≤ Ē') :
    (floatBridgesTo_biasAdd M β hn hβ).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show A + Bb + M.u * (A + Bb) ≤ Ā'
    have : M.u * (A + Bb) ≤ q * (Ā + Bb) :=
      mul_le_mul hq (by linarith) (by linarith) (M.u_nonneg.trans hq)
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show M.u * (A + Bb) + E ≤ Ē'
    have : M.u * (A + Bb) ≤ q * (Ā + Bb) :=
      mul_le_mul hq (by linarith) (by linarith) (M.u_nonneg.trans hq)
    linarith

-- ════════════════════════════════════════════════════════════════
-- § GELU, through the saturation branch
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **An envelope through GELU.** The window is `Ā + egelu` (`|gelu x| ≤ |x|`, so like relu and
    swish it passes the window through and does not reset it), and the modulus closes through the
    **right** branch of `floatClose_gelu`'s `min`: the global `3/2` saturation constant. The left
    branch is `(1 + √(2/π)/2·Ā·(1+3·0.044715·Ā²))·Ē`, cubic in the window and ~400 at ConvNeXt's
    magnitudes — and it is also irrational, so it could not be closed by `norm_num` even where it
    is the tighter of the two. This leaf is stated on the branch a big-window fold wants. -/
theorem Maps.gelu {n : Nat} (fgelu : ℝ → ℝ) {egelu : ℝ}
    (hegelu : 0 ≤ egelu) (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    {Ā Ē Ā' Ē' : ℝ} (hĀ' : Ā + egelu ≤ Ā') (hĒ' : egelu + 3 / 2 * Ē ≤ Ē') :
    (floatBridgesTo_gelu (n := n) fgelu hegelu hg).Maps Ā Ē Ā' Ē' where
  mag_le := fun A _h0 hle => by show A + egelu ≤ Ā'; linarith
  mod_le := fun A E _h0 _hE0 _hle hEle => by
    show egelu + min ((1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * E)
        (3 / 2 * E) ≤ Ē'
    have hmin := min_le_right
      ((1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * E) (3 / 2 * E)
    have : (3 : ℝ) / 2 * E ≤ 3 / 2 * Ē := by linarith
    linarith

-- ════════════════════════════════════════════════════════════════
-- § The stride-4 patchify convolution
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through the 4×4/s4 patchify stem** — `Maps.flatConv` verbatim, for the reason
    `Maps.flatConvStride2` is: the two decimations select output coordinates of the stride-1
    conv, so the fan-in `ic·kH·kW` and hence both budgets are stride-independent. -/
theorem Maps.flatConvStride4 {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (bb : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * (2 * (2 * h)) * (2 * (2 * w)))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |bb o| ≤ β)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (ic * kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((ic * kH * kW : ℕ) : ℝ) * w' * Ā + β) ≤ Ā')
    (hĒ' : g * (((ic * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + β)
            + ((ic * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_flatConvStride4 (h := h) (w := w) M W bb hw' hβ hn hW hb).Maps Ā Ē Ā' Ē' where
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


-- ════════════════════════════════════════════════════════════════
-- § The composites the ConvNeXt chain walks: LN site, block, downsample
-- ════════════════════════════════════════════════════════════════

/-! ⚠ These are stated ON the shipped bridges (`floatBridgesTo_chanLNTensor3`,
`floatBridgesTo_cnxBlockChW`, `floatBridgesTo_cnxDownChW`), not on private copies, so the
budget file's chain composes onto the same terms `convnextCh_floatBridgesTo` is built from.
That only works because those bridges REDUCE: reading a bound bundle with `obtain` compiles to
`And.casesOn`, which is stuck on a variable and defeats the unifier — see the comment on
`floatBridgesTo_cnxBlockChW`. -/

/-- ⭐ **An envelope through ConvNeXt's channel LayerNorm.** Five stages, three of them numeric:
    the four layout permutations are exact (`Maps.gather`) and the per-row lift carries the
    envelope through unchanged (`Maps.perRow`), so what is left is
    `capped LN(1,0) → diagBack γ → biasAdd β` — the affine having moved out of the normalise and
    into the bridge (`floatBridgesTo_layerNormVec`), which is why `Gd`/`Bb` appear here and not
    inside `mln`. `mln` is the LN leaf's own envelope, which at a ConvNeXt/ViT site is
    `Maps.bnCapped`'s: window honest, modulus the triangle inequality. -/
theorem Maps.chanLNTensor3 {c h w : Nat} (M : FloatModel) {ε : ℝ} (γ β : Vec c)
    (lnF : Vec c → Vec c) {Gd Bb : ℝ} (hc : 0 < c) (hhw : 0 < h * w)
    (hγ : ∀ i, |γ i| ≤ Gd) (hβ : ∀ i, |β i| ≤ Bb)
    (hln : FloatBridgesTo (layerNormForward c ε 1 0) lnF)
    {q Ā Ē A1 E1 A2 E2 Ā' Ē' : ℝ} (hq : M.u ≤ q) (hGd0 : 0 ≤ Gd) (hBb0 : 0 ≤ Bb)
    (mln : hln.Maps Ā Ē A1 E1)
    (gA : Gd * A1 + FloatModel.mulErr q Gd A1 0 0 ≤ A2)
    (gE : FloatModel.mulErr q Gd A1 0 0 + Gd * E1 ≤ E2)
    (bA : A2 + Bb + q * (A2 + Bb) ≤ Ā') (bE : q * (A2 + Bb) + E2 ≤ Ē') :
    (floatBridgesTo_chanLNTensor3 (h := h) (w := w) M γ β lnF hc hγ hβ hln).Maps Ā Ē Ā' Ē' :=
  have h1 : 0 < c * (h * w) := Nat.mul_pos hc hhw
  have h2 : 0 < h * w * c := Nat.mul_pos hhw hc
  ((((Maps.gather (reassocEquiv c h w)).comp h1 (Maps.gather _)).comp h2
      (Maps.perRow (h * w)
        ((mln.comp hc (Maps.diagBack M γ γ hc hγ (fun _ => by simp) hq hGd0 le_rfl gA gE)).comp
          hc (Maps.biasAdd M β hc hβ hq hBb0 bA bE)))).comp h2 (Maps.gather _)).comp h1
      (Maps.gather _)

/-- **An envelope through the head LayerNorm** — `rowLNVecFlat` at one row, so `Maps.perRow` of
    the same three numeric stages `Maps.chanLNTensor3` walks, without the layout permutations
    (after global-average-pooling the tensor is `[768]`, one row, and the channel LN and the
    vector LN are the same function — the cheaper spelling `CnxTWeightsCh.hε` names). -/
theorem Maps.rowLNVecFlat {s c : Nat} (M : FloatModel) {ε : ℝ} (γ β : Vec c)
    (lnF : Vec c → Vec c) {Gd Bb : ℝ} (hc : 0 < c)
    (hγ : ∀ i, |γ i| ≤ Gd) (hβ : ∀ i, |β i| ≤ Bb)
    (hln : FloatBridgesTo (layerNormForward c ε 1 0) lnF)
    {q Ā Ē A1 E1 A2 E2 Ā' Ē' : ℝ} (hq : M.u ≤ q) (hGd0 : 0 ≤ Gd) (hBb0 : 0 ≤ Bb)
    (mln : hln.Maps Ā Ē A1 E1)
    (gA : Gd * A1 + FloatModel.mulErr q Gd A1 0 0 ≤ A2)
    (gE : FloatModel.mulErr q Gd A1 0 0 + Gd * E1 ≤ E2)
    (bA : A2 + Bb + q * (A2 + Bb) ≤ Ā') (bE : q * (A2 + Bb) + E2 ≤ Ē') :
    (floatBridgesTo_rowLNVecFlat (s := s) M γ β lnF hc hγ hβ hln).Maps Ā Ē Ā' Ē' :=
  Maps.perRow s
    ((mln.comp hc (Maps.diagBack M γ γ hc hγ (fun _ => by simp) hq hGd0 le_rfl gA gE)).comp
      hc (Maps.biasAdd M β hc hβ hq hBb0 bA bE))

set_option maxHeartbeats 1000000 in
/-- ⭐ **An envelope through one ConvNeXt block.** Eight numeric stages and the skip:
    `depthwise 7×7 → chanLN (3 of them) → expand 1×1 → GELU → project 1×1 → layerScale`, then
    `Maps.residual`. Eighteen inequalities, the same shape `R34IdBlk.maps` has at ten.

    The four magnitude bounds are separate because the measured checkpoint separates them
    (`CnxBlockChBounded`): `w'` on the three convolutions, `bb` on their biases and on the LN β,
    `gl` on the LN γ, `sl` on the layer scale. -/
theorem Maps.cnxBlockChW {c cExp h w : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    (p : CnxBlockParamsCh c cExp h w 7 7) (lnF : Vec c → Vec c)
    {w' bb gl sl egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hcc : 0 < c) (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w) (hhw : 0 < h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hb : CnxBlockChBounded p w' bb gl sl)
    (hln : FloatBridgesTo (layerNormForward c p.εn 1 0) lnF)
    {q gdw gex gpr Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 A6 E6 A7 E7 A8 E8 Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hgl0 : 0 ≤ gl) (hbb0 : 0 ≤ bb) (hsl0 : 0 ≤ sl)
    (mln : hln.Maps A1 E1 A2 E2)
    (hgdw : (1 + M.u) ^ (7 * 7 + 2) - 1 ≤ gdw)
    (dwA : (1 + gdw) * (((7 * 7 : ℕ) : ℝ) * w' * Ā + bb) ≤ A1)
    (dwE : gdw * (((7 * 7 : ℕ) : ℝ) * w' * (Ā + Ē) + bb) + ((7 * 7 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (lgA : gl * A2 + FloatModel.mulErr q gl A2 0 0 ≤ A3)
    (lgE : FloatModel.mulErr q gl A2 0 0 + gl * E2 ≤ E3)
    (lbA : A3 + bb + q * (A3 + bb) ≤ A4) (lbE : q * (A3 + bb) + E3 ≤ E4)
    (hgex : (1 + M.u) ^ (c * 1 * 1 + 2) - 1 ≤ gex)
    (exA : (1 + gex) * (((c * 1 * 1 : ℕ) : ℝ) * w' * A4 + bb) ≤ A5)
    (exE : gex * (((c * 1 * 1 : ℕ) : ℝ) * w' * (A4 + E4) + bb)
            + ((c * 1 * 1 : ℕ) : ℝ) * w' * E4 ≤ E5)
    (geA : A5 + egelu ≤ A6) (geE : egelu + 3 / 2 * E5 ≤ E6)
    (hgpr : (1 + M.u) ^ (cExp * 1 * 1 + 2) - 1 ≤ gpr)
    (prA : (1 + gpr) * (((cExp * 1 * 1 : ℕ) : ℝ) * w' * A6 + bb) ≤ A7)
    (prE : gpr * (((cExp * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E6) + bb)
            + ((cExp * 1 * 1 : ℕ) : ℝ) * w' * E6 ≤ E7)
    (lsA : sl * A7 + FloatModel.mulErr q sl A7 0 0 ≤ A8)
    (lsE : FloatModel.mulErr q sl A7 0 0 + sl * E7 ≤ E8)
    (rA : A8 + Ā + q * (A8 + Ā) ≤ Ā') (rE : q * (A8 + E8 + Ā + Ē) + (E8 + Ē) ≤ Ē') :
    (floatBridgesTo_cnxBlockChW M fgelu p lnF hw' hbb hegelu hcc hc hcExp hg hb hln).Maps
      Ā Ē Ā' Ē' := by
  have hdw := Maps.depthwise (h := h) (w := w) M p.Wdw p.bdw hw' hbb hc hb.1 hb.2.1 hgdw dwA dwE
  have hlnS := Maps.chanLNTensor3 M p.γn p.βn lnF hcc hhw
    hb.2.2.2.2.2.2.2.1 hb.2.2.2.2.2.2.2.2 hln hq hgl0 hbb0 mln lgA lgE lbA lbE
  have s1 := hdw.comp hc hlnS
  have s2 := s1.comp hc (Maps.flatConv (h := h) (w := w) M p.Wex p.bex hw' hbb hc
    hb.2.2.1 hb.2.2.2.1 hgex exA exE)
  have s3 := s2.comp hcExp (Maps.gelu (n := cExp * h * w) fgelu hegelu hg geA geE)
  have s4 := s3.comp hcExp (Maps.flatConv (h := h) (w := w) M p.Wpr p.bpr hw' hbb hcExp
    hb.2.2.2.2.1 hb.2.2.2.2.2.1 hgpr prA prE)
  have s5 := s4.comp hc (Maps.diagBack M (cnxGlsCh p) (cnxGlsCh p) hc
    (fun i => hb.2.2.2.2.2.2.1 (StableHLO.chanIdx c h w i)) (fun _ => by simp)
    hq hsl0 le_rfl lsA lsE)
  exact Maps.residual M hc s5 hq rA rE

/-- **An envelope through a stage-boundary downsample** — `chanLN` at the PRE-downsample width
    over the `2h×2w` grid, then the 2×2/s2 widening convolution. Five inequalities. -/
theorem Maps.cnxDownChW {cin cout : Nat} (h w : Nat) (M : FloatModel)
    (p : CnxDownParamsCh cin cout) (lnF : Vec cin → Vec cin)
    {w' bb gl : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hcin : 0 < cin)
    (hn : 0 < cin * (2 * h) * (2 * w)) (hhw : 0 < 2 * h * (2 * w))
    (hbd : CnxDownChBounded p w' bb gl)
    (hln : FloatBridgesTo (layerNormForward cin p.ε 1 0) lnF)
    {q gc Ā Ē A1 E1 A2 E2 A3 E3 Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hgl0 : 0 ≤ gl) (hbb0 : 0 ≤ bb)
    (mln : hln.Maps Ā Ē A1 E1)
    (lgA : gl * A1 + FloatModel.mulErr q gl A1 0 0 ≤ A2)
    (lgE : FloatModel.mulErr q gl A1 0 0 + gl * E1 ≤ E2)
    (lbA : A2 + bb + q * (A2 + bb) ≤ A3) (lbE : q * (A2 + bb) + E2 ≤ E3)
    (hgc : (1 + M.u) ^ (cin * 2 * 2 + 2) - 1 ≤ gc)
    (cA : (1 + gc) * (((cin * 2 * 2 : ℕ) : ℝ) * w' * A3 + bb) ≤ Ā')
    (cE : gc * (((cin * 2 * 2 : ℕ) : ℝ) * w' * (A3 + E3) + bb)
            + ((cin * 2 * 2 : ℕ) : ℝ) * w' * E3 ≤ Ē') :
    (floatBridgesTo_cnxDownChW h w M p lnF hw' hbb hcin hn hbd hln).Maps Ā Ē Ā' Ē' :=
  (Maps.chanLNTensor3 (h := 2 * h) (w := 2 * w) M p.γ p.β lnF hcin hhw
      hbd.2.2.1 hbd.2.2.2 hln hq hgl0 hbb0 mln lgA lgE lbA lbE).comp hn
    (Maps.flatConvStride2 (h := h) (w := w) M p.W p.b hw' hbb hn hbd.1 hbd.2.1 hgc cA cE)

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The modelled device kernels
-- ════════════════════════════════════════════════════════════════

/-! Three kernels a LayerNorm/transformer net runs on device that have **no IEEE
specification**, so their accuracy is supplied rather than derived — the standing
`DeviceRsqrt` has in `Resnet34FloatBudget.lean` and `DeviceSigmoid` in
`EfficientNetFloatBudget.lean`. ⚠ They live here rather than in a budget file because two
nets now need them (ConvNeXt-T and ViT-Tiny) and a budget file should not own a structure a
sibling budget file imports. -/

/-- **The deployed LayerNorm statistics.** LayerNorm computes its mean and its inverse
    standard deviation at run time — there is nothing to freeze — so both are supplied with an
    accuracy: the mean's error RELATIVE to the window (a mean of values within `A` cannot be
    wrong by more than `O(A)`), the inverse-stddev's ABSOLUTE (the device `rsqrt`). -/
structure DeviceLN (emr ei : ℝ) where
  /-- The device mean, at any reduction width. -/
  fmu : (c : Nat) → Vec c → ℝ
  /-- The device inverse-stddev, at any reduction width and any `ε`. -/
  fistd : (c : Nat) → ℝ → Vec c → ℝ
  /-- The mean's accuracy, relative to the input window. -/
  specMu : ∀ (c : Nat) (A : ℝ), 0 ≤ A → ∀ v : Vec c, (∀ k, |v k| ≤ A) →
    |fmu c v - bnMean c v| ≤ emr * A
  /-- The inverse-stddev's absolute accuracy, at every positive `ε`. -/
  specIstd : ∀ (c : Nat) (e : ℝ), 0 < e → ∀ (A : ℝ), 0 ≤ A → ∀ v : Vec c, (∀ k, |v k| ≤ A) →
    |fistd c e v - bnIstd c v e| ≤ ei

/-- **The deployed GELU.** `stablehlo.tanh` has no IEEE specification either; `egelu` is its
    absolute accuracy against the certified `geluScalar`. -/
structure DeviceGelu (egelu : ℝ) where
  /-- The device kernel. -/
  g : ℝ → ℝ
  /-- Its accuracy at every input. -/
  spec : ∀ t, |g t - geluScalar t| ≤ egelu

/-- **The deployed `exp`** — `stablehlo.exponential`, the kernel a softmax row is built out of.
    ⚠ Its spec is **RELATIVE** where `DeviceGelu`'s and the device `rsqrt`'s are absolute, and
    that is forced rather than chosen: `softmaxF_close` divides one exponential sum by another,
    so only a relative error survives the quotient. An absolute `eexp` would say nothing about
    a row whose logits are large and negative. -/
structure DeviceExp (eexp : ℝ) where
  /-- The device kernel. -/
  e : ℝ → ℝ
  /-- Its accuracy at every input, relative to the true exponential. -/
  spec : ∀ t, |e t - Real.exp t| ≤ eexp * Real.exp t

/-- `1/√·` is antitone, so an `ε`-FLOOR's inverse-stddev bound serves every site at or above
    it — which is why one `S` covers all of a net's LayerNorm sites. -/
theorem invSqrt_le_of_floor {ε e S : ℝ} (hε : 0 < ε) (he : ε ≤ e) (hS : 1 / Real.sqrt ε ≤ S) :
    1 / Real.sqrt e ≤ S :=
  le_trans (one_div_le_one_div_of_le (Real.sqrt_pos.mpr hε) (Real.sqrt_le_sqrt he)) hS

/-- The deployed float pure-normalise LayerNorm at reduction width `c` and site `e`: the device
    mean and inverse-stddev at that width, then `bnForwardFV`'s rounded normalise chain at
    `γ = 1`, `β = 0` (the affine rides outside, in `floatBridgesTo_rowLNVecFlat` and
    `floatBridgesTo_chanLNTensor3`). -/
noncomputable def DeviceLN.lnF {emr ei : ℝ} (R : DeviceLN emr ei) (M : FloatModel) (c : Nat)
    (e : ℝ) : Vec c → Vec c := bnForwardFV M 1 0 (R.fmu c) (R.fistd c e)

/-- ⛔ **One LayerNorm site's bridge, CAPPED.** `layerNormForward c e 1 0 = bnForward c e 1 0`
    definitionally, so the leaf is `floatBridgesTo_bn` — whose modulus is quadratic in the
    window (§0.1). `.capped` replaces it by `min(that, 2·window)`. -/
noncomputable def DeviceLN.bridgeAt {emr ei : ℝ} (R : DeviceLN emr ei) (M : FloatModel)
    {ε S : ℝ} (hε : 0 < ε) (hSε : 1 / Real.sqrt ε ≤ S) (c : Nat) (hc : 0 < c) (e : ℝ)
    (he : ε ≤ e) : FloatBridgesTo (layerNormForward c e 1 0) (R.lnF M c e) :=
  (floatBridgesTo_bn (G := 1) (Bbnd := 0) (S := S) M (R.fmu c) (R.fistd c e)
    (fun A => emr * A) (fun _ => ei) hc (lt_of_lt_of_le hε he) (by norm_num) (by norm_num)
    (fun A hA v hv => R.specMu c A hA v hv)
    (fun A hA v hv => R.specIstd c e (lt_of_lt_of_le hε he) A hA v hv)
    (fun v => (bnIstd_abs_le v (lt_of_lt_of_le hε he)).trans
      (invSqrt_le_of_floor hε he hSε))).capped

/-- ⛔ **One LayerNorm site's envelope — one honest inequality and one cap.** `nA` is the real
    window `2Ā·S + bnNormBudget`, which is the fold; `nE` is `2·Ā'`, which is not (§9). -/
theorem DeviceLN.mapsAt {emr ei : ℝ} (R : DeviceLN emr ei) (M : FloatModel)
    {ε S q : ℝ} (hemr : 0 ≤ emr) (hε : 0 < ε) (hSε : 1 / Real.sqrt ε ≤ S) (hS0 : 0 ≤ S)
    (hq : M.u ≤ q) (c : Nat) (hc : 0 < c) (e : ℝ) (he : ε ≤ e) {Ā Ē Ā' Ē' : ℝ}
    (nA : 1 * (2 * Ā * S) + 0 + bnNormBudget q (2 * Ā) S 1 0 (emr * Ā) ei ≤ Ā')
    (nE : 2 * Ā' ≤ Ē') :
    (R.bridgeAt M hε hSε c hc e he).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnCapped (G := 1) (Bbnd := 0) (S := S) (em := emr * Ā) (ei := ei) M
    (R.fmu c) (R.fistd c e) (fun A => emr * A) (fun _ => ei)
    hc (lt_of_lt_of_le hε he) (by norm_num) (by norm_num)
    (fun A hA v hv => R.specMu c A hA v hv)
    (fun A hA v hv => R.specIstd c e (lt_of_lt_of_le hε he) A hA v hv)
    (fun v => (bnIstd_abs_le v (lt_of_lt_of_le hε he)).trans
      (invSqrt_le_of_floor hε he hSε))
    hq (by norm_num) (by norm_num) hS0
    (fun A h0 hle => mul_le_mul_of_nonneg_left hle hemr) (fun _ _ _ => le_rfl) nA nE

end Proofs
