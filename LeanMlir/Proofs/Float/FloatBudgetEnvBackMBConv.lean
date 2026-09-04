import LeanMlir.Proofs.Float.FloatBudgetEnvBack
import LeanMlir.Proofs.Float.FloatBudgetEnvMBConv
import LeanMlir.Proofs.Float.MobileNetV2BackFloatBridge

/-! # The `Maps` kit the INVERTED-RESIDUAL backward needs, on top of `FloatBudgetEnvBack`

`FloatBudgetEnvBack.lean` holds the backward envelope kit ResNet-34 needed — the BatchNorm
backward's per-unit gain, the structural leaves, `Maps.convBack` / `.linBack` / `.gapBack`, and
the two r34 block backwards. MobileNetV2's backward needs exactly three things beyond it, and
⭐ **all three are compositions of leaves that already exist**:

* `Maps.depthwiseBack` — `Maps.depthwise` at the spatially-reversed kernel and zero bias, because
  `depthwiseFlatBack W = depthwiseFlat (dwReverse W) 0` (`DepthwiseBackFloatBridge.lean`). The
  fan-in is `kH·kW = 9`, not a channel count: ⭐ **the cheapest stage in the whole chain**, and
  the structural reason MobileNetV2's backward folds ~26 orders under ResNet-34's.
* `Maps.depthwiseStride2Back` — that, composed with the exact `Maps.decimateBack`, exactly as
  `Maps.flatConvStride2Back` is `Maps.convBack` composed with it.
* `Maps.invresBodyBackPC` / `.invresBodyStridedBackPC` — the block envelopes, six numeric stages
  each (`bnBp → convBack Wp → bnBd → depthwiseBack → bnBe → convBack We`) and no skip: the
  inverted residual's skip is OUTSIDE the body, so `Maps.residual` is applied by the caller at
  the two blocks that have one (`b2`/`b4`).

⚠ **Why this file and not `FloatBudgetEnvBack.lean`.** A `Maps` lemma names its bridge, and
`floatBridgesTo_invresBodyBackPC`'s cone is the MobileNet one; putting these there would make
ResNet-34's backward budget depend on it. The same reason `FloatBudgetEnvMBConv.lean` exists for
the forward (`planning/float_budget_numbers.md` §0).

⭐ The file closes MobileNetV2's `b6` block backward as a compiled `example` at
`mnv2_back_chain`'s numerals — six leaves in one chain, and simultaneously the check that the
generator's arithmetic IS these lemmas' (§5's rule: an unexercised `Maps` leaf is the
`stale lean_exe gates` failure mode in proof form).
-/

namespace Proofs

open FloatModel

namespace FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The depthwise input-gradient leaves
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **An envelope through a depthwise conv INPUT-gradient.** `depthwiseFlatBack W =
    depthwiseFlat (dwReverse W) 0`, so this is the forward depthwise leaf at zero bias and the
    spatially-reversed kernel (`|dwReverse W| = |W|`, `dwReverse_abs_le`). ⭐ Note the fan-in:
    `kH·kW`, the kernel window ALONE — a depthwise conv mixes no channels, so unlike
    `Maps.convBack`'s `oc·kH·kW` the cotangent's channel count never enters. At MobileNetV2's
    3×3 depthwise that is `9`, against the 1×1 project/expand stages' 24…256. -/
theorem Maps.depthwiseBack {c h w kH kW : Nat} (M : FloatModel) (W : DepthwiseKernel c kH kW)
    {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * h * w)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w')
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((kH * kW : ℕ) : ℝ) * w' * Ā + 0) ≤ Ā')
    (hĒ' : g * (((kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + 0)
            + ((kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_depthwiseBack (h := h) (w := w) M W hw' hn hW).Maps Ā Ē Ā' Ē' :=
  Maps.depthwise (h := h) (w := w) M (dwReverse W) (fun _ => 0)
    hw' le_rfl hn (fun ch kh kw => dwReverse_abs_le hW ch kh kw) (fun _ => by simp)
    hg hĀ' hĒ'

/-- **An envelope through a STRIDED depthwise input-gradient** — the zero-fill scatter, then the
    reversed-kernel depthwise conv at the doubled resolution. The depthwise twin of
    `Maps.flatConvStride2Back`, and like it only one of the two stages rounds. -/
theorem Maps.depthwiseStride2Back {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * (2 * h) * (2 * w))
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w')
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((kH * kW : ℕ) : ℝ) * w' * Ā + 0) ≤ Ā')
    (hĒ' : g * (((kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + 0)
            + ((kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_depthwiseStride2Back (h := h) (w := w) M W hw' hn hW).Maps Ā Ē Ā' Ē' :=
  (Maps.decimateBack c h w).comp hn
    (Maps.depthwiseBack (h := 2 * h) (w := 2 * w) M W hw' hn hW hg hĀ' hĒ')

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The two inverted-residual body backwards, at real weights
-- ════════════════════════════════════════════════════════════════

/-- The float stride-1 inverted-residual body input-gradient — `invresBodyBackPC`'s deployed
    peer, every stage the float map its bridge names. The relu6 kinks are `reluMaskBack`, exact
    in float (⭐ and §3.9's finding 6: relu6's forward clamp buys the BACKWARD nothing — a 0/1
    select has no window to be clamped to). -/
noncomputable def invresBodyBackPCF {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    (bnBeF bnBdF : Vec (mid * h * w) → Vec (mid * h * w))
    (bnBpF : Vec (oc * h * w) → Vec (oc * h * w))
    (m_e m_d : Fin (mid * h * w) → Prop) [DecidablePred m_e] [DecidablePred m_d] :
    Vec (oc * h * w) → Vec (ic * h * w) :=
  (M.flatConvF (h := h) (w := w) (IR.reverseSwap We) (fun _ => 0) ∘ bnBeF ∘ reluMaskBack m_e)
  ∘ (M.depthwiseFlatF (h := h) (w := w) (dwReverse Wd) (fun _ => 0) ∘ bnBdF ∘ reluMaskBack m_d)
  ∘ (M.flatConvF (h := h) (w := w) (IR.reverseSwap Wp) (fun _ => 0) ∘ bnBpF)

/-- **The stride-1 inverted-residual body backward float-bridges TO its float peer** — the
    `FloatBridgesTo` peer of `floatBridges_invresBodyBackPC`, with the float map NAMED. The three
    per-channel BN-backs are supplied as bridges (discharge with
    `floatBridgesTo_bnPerChannelBack`); everything else is concrete. -/
noncomputable def floatBridgesTo_invresBodyBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    {bnBe bnBd bnBeF bnBdF : Vec (mid * h * w) → Vec (mid * h * w)}
    {bnBp bnBpF : Vec (oc * h * w) → Vec (oc * h * w)}
    (m_e m_d : Fin (mid * h * w) → Prop) [DecidablePred m_e] [DecidablePred m_d]
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnM : 0 < mid * h * w) (hnO : 0 < oc * h * w)
    (hbnBe : FloatBridgesTo bnBe bnBeF) (hbnBd : FloatBridgesTo bnBd bnBdF)
    (hbnBp : FloatBridgesTo bnBp bnBpF) :
    FloatBridgesTo (invresBodyBackPC We Wd Wp bnBe bnBd bnBp m_e m_d)
      (invresBodyBackPCF M We Wd Wp bnBeF bnBdF bnBpF m_e m_d) :=
  ((hbnBp.comp (floatBridgesTo_convBack (h := h) (w := w) M Wp hwp hnO hWp)).comp
    (((floatBridgesTo_reluMaskBack m_d).comp hbnBd).comp
      (floatBridgesTo_depthwiseBack (h := h) (w := w) M Wd hwd hnM hWd))).comp
    (((floatBridgesTo_reluMaskBack m_e).comp hbnBe).comp
      (floatBridgesTo_convBack (h := h) (w := w) M We hwe hnM hWe))

/-- The float stride-2 inverted-residual body input-gradient — the same shape with the depthwise
    stage threading the strided backward (zero-upsample scatter then reversed-kernel depthwise)
    and the expand backward at the `2h × 2w` grid. -/
noncomputable def invresBodyStridedBackPCF {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    (bnBeF : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w)))
    (bnBdF : Vec (mid * h * w) → Vec (mid * h * w))
    (bnBpF : Vec (oc * h * w) → Vec (oc * h * w))
    (m_e : Fin (mid * (2 * h) * (2 * w)) → Prop) [DecidablePred m_e]
    (m_d : Fin (mid * h * w) → Prop) [DecidablePred m_d] :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  (M.flatConvF (h := 2 * h) (w := 2 * w) (IR.reverseSwap We) (fun _ => 0)
      ∘ bnBeF ∘ reluMaskBack m_e)
  ∘ ((M.depthwiseFlatF (h := 2 * h) (w := 2 * w) (dwReverse Wd) (fun _ => 0)
        ∘ decimateBack mid h w) ∘ bnBdF ∘ reluMaskBack m_d)
  ∘ (M.flatConvF (h := h) (w := w) (IR.reverseSwap Wp) (fun _ => 0) ∘ bnBpF)

/-- **The stride-2 inverted-residual body backward float-bridges TO its float peer** — the
    downsample blocks `b1`/`b3`/`b5`/`b6`. Same `.comp` shape as the stride-1 body. -/
noncomputable def floatBridgesTo_invresBodyStridedBackPC
    {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    {bnBe bnBeF : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w))}
    {bnBd bnBdF : Vec (mid * h * w) → Vec (mid * h * w)}
    {bnBp bnBpF : Vec (oc * h * w) → Vec (oc * h * w)}
    (m_e : Fin (mid * (2 * h) * (2 * w)) → Prop) [DecidablePred m_e]
    (m_d : Fin (mid * h * w) → Prop) [DecidablePred m_d]
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnM2 : 0 < mid * (2 * h) * (2 * w)) (hnO : 0 < oc * h * w)
    (hbnBe : FloatBridgesTo bnBe bnBeF) (hbnBd : FloatBridgesTo bnBd bnBdF)
    (hbnBp : FloatBridgesTo bnBp bnBpF) :
    FloatBridgesTo (invresBodyStridedBackPC We Wd Wp bnBe bnBd bnBp m_e m_d)
      (invresBodyStridedBackPCF M We Wd Wp bnBeF bnBdF bnBpF m_e m_d) :=
  ((hbnBp.comp (floatBridgesTo_convBack (h := h) (w := w) M Wp hwp hnO hWp)).comp
    (((floatBridgesTo_reluMaskBack m_d).comp hbnBd).comp
      (floatBridgesTo_depthwiseStride2Back (h := h) (w := w) M Wd hwd hnM2 hWd))).comp
    (((floatBridgesTo_reluMaskBack m_e).comp hbnBe).comp
      (floatBridgesTo_convBack (h := 2 * h) (w := 2 * w) M We hwe hnM2 hWe))

namespace FloatBridgesTo

/-- ⭐ **An envelope through one stride-1 inverted-residual body input-gradient.** Six numeric
    stages — `bnBp → convBack Wp → bnBd → depthwiseBack Wd → bnBe → convBack We` — and no skip:
    the inverted residual's additive skip is OUTSIDE the body, so a caller closes `b2`/`b4` by
    wrapping this in `Maps.residual`. Twelve inequalities; the two relu6 masks contribute none. -/
theorem Maps.invresBodyBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    {bnBe bnBd bnBeF bnBdF : Vec (mid * h * w) → Vec (mid * h * w)}
    {bnBp bnBpF : Vec (oc * h * w) → Vec (oc * h * w)}
    (m_e m_d : Fin (mid * h * w) → Prop) [DecidablePred m_e] [DecidablePred m_d]
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnM : 0 < mid * h * w) (hnO : 0 < oc * h * w)
    (hbnBe : FloatBridgesTo bnBe bnBeF) (hbnBd : FloatBridgesTo bnBd bnBdF)
    (hbnBp : FloatBridgesTo bnBp bnBpF)
    {gp gd ge Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 Ā' Ē' : ℝ}
    (hgp : (1 + M.u) ^ (oc * kHp * kWp + 2) - 1 ≤ gp)
    (hgd : (1 + M.u) ^ (kHd * kWd + 2) - 1 ≤ gd)
    (hge : (1 + M.u) ^ (mid * kHe * kWe + 2) - 1 ≤ ge)
    (mbnp : hbnBp.Maps Ā Ē A1 E1)
    (cpA : (1 + gp) * (((oc * kHp * kWp : ℕ) : ℝ) * wp * A1 + 0) ≤ A2)
    (cpE : gp * (((oc * kHp * kWp : ℕ) : ℝ) * wp * (A1 + E1) + 0)
            + ((oc * kHp * kWp : ℕ) : ℝ) * wp * E1 ≤ E2)
    (mbnd : hbnBd.Maps A2 E2 A3 E3)
    (cdA : (1 + gd) * (((kHd * kWd : ℕ) : ℝ) * wd * A3 + 0) ≤ A4)
    (cdE : gd * (((kHd * kWd : ℕ) : ℝ) * wd * (A3 + E3) + 0)
            + ((kHd * kWd : ℕ) : ℝ) * wd * E3 ≤ E4)
    (mbne : hbnBe.Maps A4 E4 A5 E5)
    (ceA : (1 + ge) * (((mid * kHe * kWe : ℕ) : ℝ) * we * A5 + 0) ≤ Ā')
    (ceE : ge * (((mid * kHe * kWe : ℕ) : ℝ) * we * (A5 + E5) + 0)
            + ((mid * kHe * kWe : ℕ) : ℝ) * we * E5 ≤ Ē') :
    (floatBridgesTo_invresBodyBackPC M We Wd Wp m_e m_d hwe hwd hwp hWe hWd hWp hnM hnO
      hbnBe hbnBd hbnBp).Maps Ā Ē Ā' Ē' :=
  ((mbnp.comp hnO (Maps.convBack (h := h) (w := w) M Wp hwp hnO hWp hgp cpA cpE)).comp hnM
    (((Maps.reluMaskBack m_d).comp hnM mbnd).comp hnM
      (Maps.depthwiseBack (h := h) (w := w) M Wd hwd hnM hWd hgd cdA cdE))).comp hnM
    (((Maps.reluMaskBack m_e).comp hnM mbne).comp hnM
      (Maps.convBack (h := h) (w := w) M We hwe hnM hWe hge ceA ceE))

/-- ⭐ **An envelope through one stride-2 inverted-residual body input-gradient.** The same six
    stages, with the depthwise threading `Maps.depthwiseStride2Back` and the expand backward at
    the doubled grid. The downsample blocks `b1`/`b3`/`b5`/`b6` — and note MobileNetV2's
    downsample has NO projection branch to fan in against, unlike ResNet-34's: the stride change
    happens inside the body, so this is a straight line where `Maps.r34DownBlockBack` is a
    `biPathSum`. -/
theorem Maps.invresBodyStridedBackPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel)
    (We : Kernel4 mid ic kHe kWe) (Wd : DepthwiseKernel mid kHd kWd) (Wp : Kernel4 oc mid kHp kWp)
    {bnBe bnBeF : Vec (mid * (2 * h) * (2 * w)) → Vec (mid * (2 * h) * (2 * w))}
    {bnBd bnBdF : Vec (mid * h * w) → Vec (mid * h * w)}
    {bnBp bnBpF : Vec (oc * h * w) → Vec (oc * h * w)}
    (m_e : Fin (mid * (2 * h) * (2 * w)) → Prop) [DecidablePred m_e]
    (m_d : Fin (mid * h * w) → Prop) [DecidablePred m_d]
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnM : 0 < mid * h * w) (hnM2 : 0 < mid * (2 * h) * (2 * w)) (hnO : 0 < oc * h * w)
    (hbnBe : FloatBridgesTo bnBe bnBeF) (hbnBd : FloatBridgesTo bnBd bnBdF)
    (hbnBp : FloatBridgesTo bnBp bnBpF)
    {gp gd ge Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 Ā' Ē' : ℝ}
    (hgp : (1 + M.u) ^ (oc * kHp * kWp + 2) - 1 ≤ gp)
    (hgd : (1 + M.u) ^ (kHd * kWd + 2) - 1 ≤ gd)
    (hge : (1 + M.u) ^ (mid * kHe * kWe + 2) - 1 ≤ ge)
    (mbnp : hbnBp.Maps Ā Ē A1 E1)
    (cpA : (1 + gp) * (((oc * kHp * kWp : ℕ) : ℝ) * wp * A1 + 0) ≤ A2)
    (cpE : gp * (((oc * kHp * kWp : ℕ) : ℝ) * wp * (A1 + E1) + 0)
            + ((oc * kHp * kWp : ℕ) : ℝ) * wp * E1 ≤ E2)
    (mbnd : hbnBd.Maps A2 E2 A3 E3)
    (cdA : (1 + gd) * (((kHd * kWd : ℕ) : ℝ) * wd * A3 + 0) ≤ A4)
    (cdE : gd * (((kHd * kWd : ℕ) : ℝ) * wd * (A3 + E3) + 0)
            + ((kHd * kWd : ℕ) : ℝ) * wd * E3 ≤ E4)
    (mbne : hbnBe.Maps A4 E4 A5 E5)
    (ceA : (1 + ge) * (((mid * kHe * kWe : ℕ) : ℝ) * we * A5 + 0) ≤ Ā')
    (ceE : ge * (((mid * kHe * kWe : ℕ) : ℝ) * we * (A5 + E5) + 0)
            + ((mid * kHe * kWe : ℕ) : ℝ) * we * E5 ≤ Ē') :
    (floatBridgesTo_invresBodyStridedBackPC M We Wd Wp m_e m_d hwe hwd hwp hWe hWd hWp
      hnM2 hnO hbnBe hbnBd hbnBp).Maps Ā Ē Ā' Ē' :=
  ((mbnp.comp hnO (Maps.convBack (h := h) (w := w) M Wp hwp hnO hWp hgp cpA cpE)).comp hnM
    (((Maps.reluMaskBack m_d).comp hnM mbnd).comp hnM
      (Maps.depthwiseStride2Back (h := h) (w := w) M Wd hwd hnM2 hWd hgd cdA cdE))).comp hnM2
    (((Maps.reluMaskBack m_e).comp hnM2 mbne).comp hnM2
      (Maps.convBack (h := 2 * h) (w := 2 * w) M We hwe hnM2 hWe hge ceA ceE))

end FloatBridgesTo

end Proofs
