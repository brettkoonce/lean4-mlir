import LeanMlir.Proofs.Float.FloatBudgetEnvBack
import LeanMlir.Proofs.Float.FloatBudgetEnvBackMBConv
import LeanMlir.Proofs.Float.FloatBudgetEnvLN
import LeanMlir.Proofs.Float.ConvNeXtBackFloatBridge

/-! # The `Maps` kit a LAYERNORM net's BACKWARD needs, on top of `FloatBudgetEnvBack`

`FloatBudgetEnvBack.lean` is ResNet-34's backward kit and `FloatBudgetEnvBackMBConv.lean`
MobileNetV2's; this is ConvNeXt-T's — the third backward net and the first whose normalisation
reduces over the CHANNEL axis (planning/float_budget_numbers.md §3.16).

⭐⭐ **The one genuinely new leaf is `Maps.rowLNVecFlatBack`, and its shape is NOT
`Maps.bnPerChannelBackGain` at a different `n`.** `floatBridgesTo_rowLNVecFlatBack` runs
`bn_grad_input` with `|(1:ℝ)|` in its `G` slot and folds the γ scale in FRONT of it as a separate
`diagBack`, so the γ multiply is a rounded stage of its own and its `mulErr` enters the fold:

    D    = Gd·Ā + mulErr q Gd Ā egam 0            — the diagBack's output window
    mag  = D·(Kr + Kb)
    mod  = D·Kb + (mulErr q Gd Ā egam 0 + Gd·Ē)·Kr

with `Kr`/`Kb` the per-unit gains **at γ = 1** — the §3.7 homogeneity factoring, which matters
more here than on r34: ConvNeXt-T has 23 LayerNorm sites but only FOUR distinct reduction widths
(96/192/384/768), so four `hKr`/`hKb` pairs serve the whole net.

⚠ The reduction width is the CHANNEL count, where r34's and mnv2's is `h·w`. `bnXhat_sq_le` still
covers `|x̂| ≤ √n` — `convnextCh_grad_floatBridges` states its LN-back hypotheses through
`bnIstd c` and `bnXhat c` at `chanLNRows`, literally r34's two quantities under a different
conjugation — but `√96` … `√768` are all irrational, so `bnXhat_abs_le_num` must be used at the
CEILING root (10/14/20/28) where r34's square feature maps gave an exact one.

The rest is composition: `Maps.chanLNTensor3Back` is that leaf between four exact permutations,
`Maps.flatConvStride4Back` is `Maps.convBack` after two exact scatters (the patchify backward),
and the block/downsample envelopes are straight `.comp` chains — no new combinator, exactly as
§3.7 step 3 found for r34's blocks and §3.13 for MobileNetV2's.
-/

namespace Proofs

namespace FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The patchify backward (the second scatter, then the conv)
-- ════════════════════════════════════════════════════════════════

/-- The odd-decimation backward is a zero-fill scatter — exact, envelope unchanged. The
    `Maps.decimateBack` of the stride-4 stem's second upsample. -/
theorem Maps.decimateOddBack (oc h w : Nat) {Ā Ē : ℝ} :
    (floatBridgesTo_decimateOddBack oc h w).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- **An envelope through the 4×4/s4 PATCHIFY input-gradient** — two zero-fill scatters, then the
    reversed-kernel conv at the quadrupled resolution. Three stages, two of them exact, so the
    arithmetic is `Maps.convBack`'s at fan-in `oc·kH·kW`. ConvNeXt's stem. -/
theorem Maps.flatConvStride4Back {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) {w' : ℝ} (hw' : 0 ≤ w')
    (hn : 0 < oc * (2 * (2 * h)) * (2 * (2 * w))) (hn2 : 0 < oc * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (oc * kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((oc * kH * kW : ℕ) : ℝ) * w' * Ā + 0) ≤ Ā')
    (hĒ' : g * (((oc * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + 0)
            + ((oc * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_flatConvStride4Back (h := h) (w := w) M W hw' hn hW).Maps Ā Ē Ā' Ē' :=
  ((Maps.decimateBack oc h w).comp hn2 (Maps.decimateOddBack oc (2 * h) (2 * w))).comp hn
    (Maps.convBack (h := 2 * (2 * h)) (w := 2 * (2 * w)) M W hw' hn hW hg hĀ' hĒ')

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ The LayerNorm BACKWARD leaf
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **An envelope through the rowwise vector-LayerNorm INPUT-GRADIENT**, in the per-unit-gain
    form a whole-net chain must use.

    `floatBridgesTo_rowLNVecFlatBack` is `diagBack γ` then `bn_grad_input` **at γ = 1**, per row,
    so — unlike `Maps.bnPerChannelBackGain`, which carries γ inside the gain — the γ multiply is a
    rounded stage of its own. `Kr`/`Kb` are therefore the BN backward's real Lipschitz constant and
    rounding budget at UNIT γ and UNIT cotangent, and the window that enters them is the diagBack's
    output `D = Gd·Ā + mulErr q Gd Ā egam 0`.

    ⭐ Both are homogeneous of degree 1 in that window (`bnGradInputReMag_homog` /
    `bnGradInputBudgetG_homog`), which is what keeps the closing inequalities flat: `D·(Kr+Kb)` and
    `D·Kb + (me + Gd·Ē)·Kr` instead of a forty-node tree at the chain's magnitude. ConvNeXt-T has
    23 LN sites and four distinct reduction widths, so four `hKr`/`hKb` pairs serve all of them. -/
theorem Maps.rowLNVecFlatBack {s c : Nat} (M : FloatModel) {ε : ℝ} (γ fγ : Vec c)
    (X : Vec (s * c)) (fs : Fin s → ℝ) (fxh : Fin s → Vec c)
    {Gd egam S Xh es exh : ℝ} (hs0 : 0 < s) (hc : 0 < c)
    (hγ : ∀ i, |γ i| ≤ Gd) (hfγ : ∀ i, |fγ i - γ i| ≤ egam)
    (hst : ∀ r, |fs r - bnIstd c (Mat.unflatten X r) ε| ≤ es)
    (hSabs : ∀ r, |bnIstd c (Mat.unflatten X r) ε| ≤ S)
    (hxh : ∀ r i, |bnXhat c ε (Mat.unflatten X r) i| ≤ Xh)
    (hfxh : ∀ r i, |fxh r i - bnXhat c ε (Mat.unflatten X r) i| ≤ exh)
    {q gn Kr Kb Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q) (hgn : (1 + M.u) ^ (c + 1) - 1 ≤ gn)
    (hGd0 : 0 ≤ Gd) (hegam0 : 0 ≤ egam) (hS0 : 0 ≤ S) (hXh0 : 0 ≤ Xh)
    (hes0 : 0 ≤ es) (hexh0 : 0 ≤ exh)
    (hKr : bnGradInputReMag c 1 1 S Xh ≤ Kr)
    (hKb : bnGradInputBudgetG q gn c 1 1 S Xh es exh ≤ Kb)
    (_hĀ0 : 0 ≤ Ā) (_hĒ0 : 0 ≤ Ē)
    (hĀ' : (Gd * Ā + FloatModel.mulErr q Gd Ā egam 0) * (Kr + Kb) ≤ Ā')
    (hĒ' : (Gd * Ā + FloatModel.mulErr q Gd Ā egam 0) * Kb
            + (FloatModel.mulErr q Gd Ā egam 0 + Gd * Ē) * Kr ≤ Ē') :
    (floatBridgesTo_rowLNVecFlatBack M γ fγ X fs fxh hs0 hc hγ hfγ hst hSabs hxh hfxh).Maps
      Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show bnGradInputReMag c |(1 : ℝ)| (Gd * A + FloatModel.mulErr M.u Gd A egam 0) S Xh
        + M.bnGradInputBudget c |(1 : ℝ)| (Gd * A + FloatModel.mulErr M.u Gd A egam 0) S Xh es exh
        ≤ Ā'
    rw [abs_one]
    set me := FloatModel.mulErr M.u Gd A egam 0 with hme
    set meB := FloatModel.mulErr q Gd Ā egam 0 with hmeB
    have hme0 : 0 ≤ me := mulErr_nonneg M.u_nonneg hGd0 h0 hegam0 le_rfl
    have hmeM : me ≤ meB := mulErr_mono M.u_nonneg hq hGd0 le_rfl h0 hle hegam0 le_rfl
      le_rfl le_rfl
    have hD0 : 0 ≤ Gd * A + me := by nlinarith
    have hDM : Gd * A + me ≤ Gd * Ā + meB := by nlinarith
    have hKr1 : 0 ≤ bnGradInputReMag c 1 1 S Xh :=
      bnGradInputReMag_nonneg c zero_le_one zero_le_one hS0 hXh0
    have hr : bnGradInputReMag c 1 (Gd * A + me) S Xh ≤ (Gd * Ā + meB) * Kr := by
      rw [bnGradInputReMag_homog]
      exact le_trans (mul_le_mul_of_nonneg_right hDM hKr1)
        (mul_le_mul_of_nonneg_left hKr (by linarith))
    have hb := bnGradInputBudget_le M c (Cdy := Gd * A + me) (Cd := Gd * Ā + meB)
      hq hgn hD0 hDM zero_le_one hS0 hXh0 hes0 hexh0
    rw [bnGradInputBudgetG_homog] at hb
    have hb2 : (Gd * Ā + meB) * bnGradInputBudgetG q gn c 1 1 S Xh es exh
        ≤ (Gd * Ā + meB) * Kb := mul_le_mul_of_nonneg_left hKb (by linarith)
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show M.bnGradInputBudget c |(1 : ℝ)| (Gd * A + FloatModel.mulErr M.u Gd A egam 0) S Xh es exh
        + bnGradInputReMag c |(1 : ℝ)| (FloatModel.mulErr M.u Gd A egam 0 + Gd * E) S Xh ≤ Ē'
    rw [abs_one]
    set me := FloatModel.mulErr M.u Gd A egam 0 with hme
    set meB := FloatModel.mulErr q Gd Ā egam 0 with hmeB
    have hme0 : 0 ≤ me := mulErr_nonneg M.u_nonneg hGd0 h0 hegam0 le_rfl
    have hmeM : me ≤ meB := mulErr_mono M.u_nonneg hq hGd0 le_rfl h0 hle hegam0 le_rfl
      le_rfl le_rfl
    have hD0 : 0 ≤ Gd * A + me := by nlinarith
    have hDM : Gd * A + me ≤ Gd * Ā + meB := by nlinarith
    have hKr1 : 0 ≤ bnGradInputReMag c 1 1 S Xh :=
      bnGradInputReMag_nonneg c zero_le_one zero_le_one hS0 hXh0
    have hEM : me + Gd * E ≤ meB + Gd * Ē := by nlinarith
    have hr : bnGradInputReMag c 1 (me + Gd * E) S Xh ≤ (meB + Gd * Ē) * Kr := by
      rw [bnGradInputReMag_homog]
      exact le_trans (mul_le_mul_of_nonneg_right hEM hKr1)
        (mul_le_mul_of_nonneg_left hKr (by nlinarith))
    have hb := bnGradInputBudget_le M c (Cdy := Gd * A + me) (Cd := Gd * Ā + meB)
      hq hgn hD0 hDM zero_le_one hS0 hXh0 hes0 hexh0
    rw [bnGradInputBudgetG_homog] at hb
    have hb2 : (Gd * Ā + meB) * bnGradInputBudgetG q gn c 1 1 S Xh es exh
        ≤ (Gd * Ā + meB) * Kb := mul_le_mul_of_nonneg_left hKb (by linarith)
    nlinarith

-- ════════════════════════════════════════════════════════════════
-- § The channel-LN backward: the leaf between four exact permutations
-- ════════════════════════════════════════════════════════════════

/-- **An envelope through the CHANNEL-LayerNorm input-gradient.** A permutation's adjoint is its
    inverse permutation, so `chanLNTensor3Back` is `rowLNVecFlatBack` conjugated by the same four
    layout gathers the forward's `Maps.chanLNTensor3` walks — all exact — and the arithmetic is
    the row leaf's, at reduction width `c` and `h·w` rows. -/
theorem Maps.chanLNTensor3Back {c h w : Nat} (M : FloatModel) {ε : ℝ} (γ fγ : Vec c)
    (x : Vec (c * h * w)) (fs : Fin (h * w) → ℝ) (fxh : Fin (h * w) → Vec c)
    {Gd egam S Xh es exh : ℝ} (hhw : 0 < h * w) (hc : 0 < c)
    (hγ : ∀ i, |γ i| ≤ Gd) (hfγ : ∀ i, |fγ i - γ i| ≤ egam)
    (hst : ∀ r, |fs r - bnIstd c (Mat.unflatten (chanLNRows c h w x) r) ε| ≤ es)
    (hSabs : ∀ r, |bnIstd c (Mat.unflatten (chanLNRows c h w x) r) ε| ≤ S)
    (hxh : ∀ r i, |bnXhat c ε (Mat.unflatten (chanLNRows c h w x) r) i| ≤ Xh)
    (hfxh : ∀ r i, |fxh r i - bnXhat c ε (Mat.unflatten (chanLNRows c h w x) r) i| ≤ exh)
    {q gn Kr Kb Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q) (hgn : (1 + M.u) ^ (c + 1) - 1 ≤ gn)
    (hGd0 : 0 ≤ Gd) (hegam0 : 0 ≤ egam) (hS0 : 0 ≤ S) (hXh0 : 0 ≤ Xh)
    (hes0 : 0 ≤ es) (hexh0 : 0 ≤ exh)
    (hKr : bnGradInputReMag c 1 1 S Xh ≤ Kr)
    (hKb : bnGradInputBudgetG q gn c 1 1 S Xh es exh ≤ Kb)
    (hĀ0 : 0 ≤ Ā) (hĒ0 : 0 ≤ Ē)
    (hĀ' : (Gd * Ā + FloatModel.mulErr q Gd Ā egam 0) * (Kr + Kb) ≤ Ā')
    (hĒ' : (Gd * Ā + FloatModel.mulErr q Gd Ā egam 0) * Kb
            + (FloatModel.mulErr q Gd Ā egam 0 + Gd * Ē) * Kr ≤ Ē') :
    (floatBridgesTo_chanLNTensor3Back (h := h) (w := w) M γ fγ x fs fxh hhw hc hγ hfγ
      hst hSabs hxh hfxh).Maps Ā Ē Ā' Ē' :=
  have h1 : 0 < c * (h * w) := Nat.mul_pos hc hhw
  have h2 : 0 < h * w * c := Nat.mul_pos hhw hc
  ((((Maps.gather (reassocEquiv c h w)).comp h1 (Maps.gather _)).comp h2
      (Maps.rowLNVecFlatBack M γ fγ _ fs fxh hhw hc hγ hfγ hst hSabs hxh hfxh hq hgn
        hGd0 hegam0 hS0 hXh0 hes0 hexh0 hKr hKb hĀ0 hĒ0 hĀ' hĒ')).comp h2
      (Maps.gather _)).comp h1 (Maps.gather _)

-- ════════════════════════════════════════════════════════════════
-- § The block body and downsample envelopes
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **An envelope through one ConvNeXt block BODY input-gradient.** Six numeric stages —
    `lsB → convBack Wpr → geluB → convBack Wex → lnB → depthwiseBack Wdw` — and no skip: the block
    is `residual (body)`, so a caller wraps this in `Maps.residual`, exactly as MobileNetV2's
    `Maps.invresBodyBackPC` leaves `b2`/`b4`'s skip to its caller. Twelve inequalities.

    ⭐ Note the two fan-ins the expand/project pair contributes on the BACKWARD: `c` and `cExp`,
    the cotangent's channel count at each stage (the 1×1 kernels contribute nothing), against the
    depthwise's `kH·kW = 49` — the whole 7×7 window, and still the smallest of the three. -/
theorem Maps.cnxBlockBodyBack {c cExp h w kHd kWd : Nat} (M : FloatModel)
    (Wdw : DepthwiseKernel c kHd kWd) (Wex : Kernel4 cExp c 1 1) (Wpr : Kernel4 c cExp 1 1)
    {lnB lsB lnBF lsBF : Vec (c * h * w) → Vec (c * h * w)}
    {geluB geluBF : Vec (cExp * h * w) → Vec (cExp * h * w)}
    {wdw wex wpr : ℝ} (hwdw : 0 ≤ wdw) (hwex : 0 ≤ wex) (hwpr : 0 ≤ wpr)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ wdw) (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ wex)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ wpr)
    (hnC : 0 < c * h * w) (hnE : 0 < cExp * h * w)
    (hlnB : FloatBridgesTo lnB lnBF) (hlsB : FloatBridgesTo lsB lsBF)
    (hgeluB : FloatBridgesTo geluB geluBF)
    {gpr gex gdw Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 Ā' Ē' : ℝ}
    (hgpr : (1 + M.u) ^ (c * 1 * 1 + 2) - 1 ≤ gpr)
    (hgex : (1 + M.u) ^ (cExp * 1 * 1 + 2) - 1 ≤ gex)
    (hgdw : (1 + M.u) ^ (kHd * kWd + 2) - 1 ≤ gdw)
    (mls : hlsB.Maps Ā Ē A1 E1)
    (prA : (1 + gpr) * (((c * 1 * 1 : ℕ) : ℝ) * wpr * A1 + 0) ≤ A2)
    (prE : gpr * (((c * 1 * 1 : ℕ) : ℝ) * wpr * (A1 + E1) + 0)
            + ((c * 1 * 1 : ℕ) : ℝ) * wpr * E1 ≤ E2)
    (mge : hgeluB.Maps A2 E2 A3 E3)
    (exA : (1 + gex) * (((cExp * 1 * 1 : ℕ) : ℝ) * wex * A3 + 0) ≤ A4)
    (exE : gex * (((cExp * 1 * 1 : ℕ) : ℝ) * wex * (A3 + E3) + 0)
            + ((cExp * 1 * 1 : ℕ) : ℝ) * wex * E3 ≤ E4)
    (mln : hlnB.Maps A4 E4 A5 E5)
    (dwA : (1 + gdw) * (((kHd * kWd : ℕ) : ℝ) * wdw * A5 + 0) ≤ Ā')
    (dwE : gdw * (((kHd * kWd : ℕ) : ℝ) * wdw * (A5 + E5) + 0)
            + ((kHd * kWd : ℕ) : ℝ) * wdw * E5 ≤ Ē') :
    (floatBridgesTo_cnxBlockBodyBack M Wdw Wex Wpr hwdw hwex hwpr hWdw hWex hWpr hnC hnE
      hlnB hlsB hgeluB).Maps Ā Ē Ā' Ē' :=
  ((((mls.comp hnC (Maps.convBack (h := h) (w := w) M Wpr hwpr hnC hWpr hgpr prA prE)).comp hnE
      mge).comp hnE
      (Maps.convBack (h := h) (w := w) M Wex hwex hnE hWex hgex exA exE)).comp hnC mln).comp hnC
      (Maps.depthwiseBack (h := h) (w := w) M Wdw hwdw hnC hWdw hgdw dwA dwE)

/-- ⭐ **An envelope through one ConvNeXt stage-boundary DOWNSAMPLE input-gradient** — the §A3
    strided-conv backward (zero-fill scatter then the reversed 2×2 kernel at the doubled grid),
    then the LayerNorm back at the INPUT resolution and channel count. Two numeric stages. -/
theorem Maps.cnxDownBack {cin cout h w : Nat} (M : FloatModel) (W : Kernel4 cout cin 2 2)
    {lnB lnBF : Vec (cin * (2 * h) * (2 * w)) → Vec (cin * (2 * h) * (2 * w))}
    {wd : ℝ} (hwd : 0 ≤ wd) (hW : ∀ o c kh kw, |W o c kh kw| ≤ wd)
    (hn : 0 < cout * (2 * h) * (2 * w)) (hnIn : 0 < cin * (2 * h) * (2 * w))
    (hlnB : FloatBridgesTo lnB lnBF)
    {g Ā Ē A1 E1 Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (cout * 2 * 2 + 2) - 1 ≤ g)
    (cA : (1 + g) * (((cout * 2 * 2 : ℕ) : ℝ) * wd * Ā + 0) ≤ A1)
    (cE : g * (((cout * 2 * 2 : ℕ) : ℝ) * wd * (Ā + Ē) + 0)
            + ((cout * 2 * 2 : ℕ) : ℝ) * wd * Ē ≤ E1)
    (mln : hlnB.Maps A1 E1 Ā' Ē') :
    (floatBridgesTo_cnxDownBack (h := h) (w := w) M W hwd hW hn hlnB).Maps Ā Ē Ā' Ē' :=
  (Maps.flatConvStride2Back (h := h) (w := w) M W hwd hn hW hg cA cE).comp hnIn mln

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § ⭐ The kit exercised: ConvNeXt-T's block `s4b2` at the probe's numerals
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **Every leaf above, composed into one real block at the numerals
    `scripts/float_budget_envelope.py`'s `cnx_back_chain` emits** — the FIRST block ConvNeXt-T's
    cotangent meets (stage 4, `c = 768`, `cExp = 3072`, `7 × 7`), at `|istd| ≤ 16`, the measured
    300-epoch per-kind profile (kernels ≤ 6/10, LN γ ≤ 48/10, layer scale ≤ 84/10), `ε ≥ 10⁻⁵`
    and saved-activation accuracies `10⁻²`.

    In `(7407, 1025/10²)` — the GAP backward's output — out `(1.421·10¹⁴, 1.365·10¹²)`. ⭐ One
    block costs ~10¹⁰ of cotangent window; eighteen put the net at 10²⁴⁹.

    §5's rule: a `Maps` leaf nothing composes is a leaf nobody has checked composes, and this is
    simultaneously the check that the generator's arithmetic IS these lemmas'. -/
example (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (_hε5 : 1 / 100000 ≤ ε)
    (Wdw : DepthwiseKernel 768 7 7) (Wex : Kernel4 3072 768 1 1) (Wpr : Kernel4 768 3072 1 1)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ 6/10)
    (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ 6/10)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ 6/10)
    -- the layer scale, a stored weight: its float peer IS itself
    (γls : Vec (768 * 7 * 7)) (hγls : ∀ i, |γls i| ≤ 84/10)
    -- the saved GELU derivative, within 10⁻² of the real one (|gelu′| ≤ 3/2 is PROVED)
    (sge fsge : Vec (3072 * 7 * 7)) (hsge : ∀ i, |sge i| ≤ 3/2)
    (hfsge : ∀ i, |fsge i - sge i| ≤ 1/100)
    -- the channel-LayerNorm backward's saved state
    (γln fγln : Vec 768) (xln : Vec (768 * 7 * 7))
    (fs : Fin (7 * 7) → ℝ) (fxh : Fin (7 * 7) → Vec 768)
    (hγln : ∀ i, |γln i| ≤ 48/10) (hfγln : ∀ i, |fγln i - γln i| ≤ 0)
    (hst : ∀ r, |fs r - bnIstd 768 (Mat.unflatten (chanLNRows 768 7 7 xln) r) ε| ≤ 1/100)
    (hSabs : ∀ r, |bnIstd 768 (Mat.unflatten (chanLNRows 768 7 7 xln) r) ε| ≤ 16)
    (hxh : ∀ r i, |bnXhat 768 ε (Mat.unflatten (chanLNRows 768 7 7 xln) r) i| ≤ 28)
    (hfxh : ∀ r i, |fxh r i - bnXhat 768 ε (Mat.unflatten (chanLNRows 768 7 7 xln) r) i| ≤ 1/100) :
    (FloatBridgesTo.residual M
      (floatBridgesTo_cnxBlockBodyBack (h := 7) (w := 7) M Wdw Wex Wpr
        (by norm_num) (by norm_num) (by norm_num) hWdw hWex hWpr (by norm_num) (by norm_num)
        (floatBridgesTo_chanLNTensor3Back (h := 7) (w := 7) M γln fγln xln fs fxh
          (by norm_num) (by norm_num) hγln hfγln hst hSabs hxh hfxh)
        (floatBridgesTo_diagBack M γls γls (es := 0) (by norm_num) hγls (fun _ => by simp))
        (floatBridgesTo_diagBack M sge fsge (by norm_num) hsge hfsge))).Maps
      7407 (1025 / 10 ^ 2) (1421 * 10 ^ 14) (1365 * 10 ^ 12) :=
  FloatBridgesTo.Maps.residual M (m := 768 * 7 * 7) (by norm_num)
    (Ā := 7407) (Ē := 1025 / 10 ^ 2) (Bd := 1420 * 10 ^ 14) (Ed := 1364 * 10 ^ 12)
    (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 7) (w := 7) M Wdw Wex Wpr
      (by norm_num) (by norm_num) (by norm_num) hWdw hWex hWpr (by norm_num) (by norm_num)
      _ _ _
      (gpr := 4590 / 10 ^ 8) (gex := 1833 / 10 ^ 7) (gdw := 3040 / 10 ^ 9)
      (A1 := 6222 * 10 ^ 1) (E1 := 8611 / 10 ^ 2)
      (A2 := 2868 * 10 ^ 4) (E2 := 4100 * 10 ^ 1)
      (A3 := 4331 * 10 ^ 4) (E3 := 3484 * 10 ^ 2)
      (A4 := 7985 * 10 ^ 7) (E4 := 6570 * 10 ^ 5)
      (A5 := 4829 * 10 ^ 12) (E5 := 4635 * 10 ^ 10)
      (Ā := 7407) (Ē := 1025 / 10 ^ 2) (Ā' := 1420 * 10 ^ 14) (Ē' := 1364 * 10 ^ 12)
      (M.gamma_num (k := 768 * 1 * 1 + 2) (q := 4590 / 10 ^ 8) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (M.gamma_num (k := 3072 * 1 * 1 + 2) (q := 1833 / 10 ^ 7) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (M.gamma_num (k := 7 * 7 + 2) (q := 3040 / 10 ^ 9) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (FloatBridgesTo.Maps.diagBack M γls γls (es := 0) (by norm_num) hγls (fun _ => by simp) hMu
        (by norm_num) le_rfl (by norm_num [FloatModel.mulErr, u32])
        (by norm_num [FloatModel.mulErr, u32]))
      (by norm_num) (by norm_num)
      (FloatBridgesTo.Maps.diagBack M sge fsge (by norm_num) hsge hfsge hMu
        (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32])
        (by norm_num [FloatModel.mulErr, u32]))
      (by norm_num) (by norm_num)
      (FloatBridgesTo.Maps.chanLNTensor3Back (h := 7) (w := 7) M γln fγln xln fs fxh
        (by norm_num) (by norm_num) hγln hfγln hst hSabs hxh hfxh hMu
        (M.gamma_num (k := 768 + 1) (q := 4584 / 10 ^ 8) hMu
          (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) le_rfl (by norm_num) (by norm_num)
        (by norm_num) (by norm_num)
        (Kr := 1258 * 10 ^ 1) (Kb := 1741 / 10 ^ 2)
        (by norm_num [bnGradInputReMag])
        (by norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
          bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32])
        (by norm_num) (by norm_num)
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
      (by norm_num) (by norm_num))
    hMu (by norm_num [u32]) (by norm_num [u32])

end Proofs
