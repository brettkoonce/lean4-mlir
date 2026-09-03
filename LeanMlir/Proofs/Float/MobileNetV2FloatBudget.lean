import LeanMlir.Proofs.Float.Resnet34FloatBudget
import LeanMlir.Proofs.Float.FloatBudgetEnvMBConv
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPCEval

/-! # A NUMBER for MobileNetV2: the deployed inference forward, and a TIGHT window

The second ImageNet-scale whole-net float budget, after `Resnet34FloatBudget.lean`. For the
6-block inverted-residual MobileNetV2 forward at `224²` — 3×3/s2 stem, `[b1…b6]` with stride-2
downsamples at `b1/b3/b5/b6` and matched-channel skips at `b2/b4`, 1×1 head, GAP, dense — with
**inference** BatchNorm (frozen running statistics), on the unit input window, at the profile
measured on the 350-epoch checkpoint (`|parameter| ≤ 28/10`), for any rounding model at binary32
accuracy:

    output window  ≤ 2.154·10³       (`mnv2EvalBridge_mag_le`)
    fresh budget   ≤ 1.444·10⁹⁶      (`mnv2EvalBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 1.444·10⁹⁶` (`mnv2_float_logits_le`).

⭐⭐ **The window is essentially tight, and that is the new thing here.** ResNet-34's certified
window is `3.152·10²¹¹` against logits of a few; MobileNetV2's is `2154`. The whole difference is
one lemma: `relu6 x i = min (max (x i) 0) 6` is bounded by `6` *whatever its input*, so
`floatClose_relu6` states `FloatClose A (min A 6)` and every one of the net's 13 relu6 sites
RESETS the certified magnitude rather than passing the incoming one through. ResNet-34 cannot
have this — plain `relu` has no upper clamp, and `floatClose_relu`'s `FloatClose A A` is the best
it can do. Folded without the clamp this net's window is `4.309·10¹⁰⁰`
(`scripts/float_budget_envelope.py`, `mnv2_eval_chain(relu6_clamp := False)`); with it, `2154`.

⚠ **And the budget barely moves: `3.072·10⁹⁷` → `1.444·10⁹⁶`, one order.** That is the lesson,
not a disappointment. Error gain per stage does not care how small the window is — it is `G·S`
per BN site, and `S = 1/√ε ≤ 317` at the ε-floor. Sweeping `S` through the same fold gives
`10⁹⁶ / 10⁷⁷ / 10⁶⁰ / 10⁴⁸` at `S = 317 / 32 / 4 / 1`, about 19 orders per decade of `S` across
the 20 BN sites. **Window and budget are separate levers**: the clamp is worth 97 orders of
window and one of budget; only an operating-point variance floor moves the budget. The number is
still the interval fold and still vacuous as a certificate; what is new is that the kernel checks
it, and that the *window* half of it is now a statement one could believe on its own.

`scripts/adjoint_chain_probe.py` puts the adjoint chain's proven-H budget for this net at
`2.7·10⁶⁰`, so this fold sits ~36 orders above it — much closer than ResNet-34's 157, because the
clamped window removes the window looseness and leaves only the gain looseness.

⭐⭐ **Why inference BN.** Training-mode BatchNorm reduces its statistics out of its own input, so
its modulus is quadratic in the window and the fold squares at every one of this net's 20 BN
sites. Inference BN has no reduction — `μ` and `rsqrt(var+ε)` are frozen constants, the map is
affine in `x` with slope `γ·s`, the modulus is linear. As for ResNet-34, the choice of BN decides
whether a whole-net number exists at all (`planning/float_budget_numbers.md` §0.1).

⚠ **The one hypothesis this number rests on, named.** The deployed inverse-stddev is a device
`rsqrt` with no IEEE specification, so it is *modelled*: `DeviceRsqrt ε es` (shared with
`Resnet34FloatBudget.lean` — it is the same device kernel, and a second copy would be a second
modelled assumption) supplies it with an accuracy `es`. Everything else is proved.

**The tie is closed at the graph.** `mnv2EvalForward_eq_full_pc_eval` is a `rfl` onto
`mobilenetv2Forward_full_pc_eval` (`MobileNetV2RenderPCEval.lean`) and `mnv2EvalGraph_faithful`
carries `mobilenetv2FwdGraphFullPCEval_faithful` the rest of the way, so the typed `SHlo` graph
the render emits **denotes** the forward this file bounds. `mnv2_float_logits_le_committed`
states the number with that net on the real side.

Provenance for the numerals: `scripts/float_budget_envelope.py` (`mnv2_eval_chain`), which folds
the envelope in exactly these lemmas' semantics with exact rationals, rounds every stage UP to
four significant figures, and re-asserts each rounded inequality (`verify_mnv2`, 116
inequalities) before emitting.
-/

namespace Proofs

open FloatModel

/-! The three `Maps` leaves this net needs and ResNet-34 did not — `relu6` (⭐ it CLAMPS, so its
window step is `min Ā 6`), `depthwise` and `depthwiseStride2Flat` — live in
`FloatBudgetEnvMBConv.lean`, shared with EfficientNet-B0. -/

-- ════════════════════════════════════════════════════════════════
-- § The parameter records
-- ════════════════════════════════════════════════════════════════

/-- A convolution's stored parameters with their magnitude bounds. -/
structure MnvConv (oc ic kH kW : Nat) (w' β' : ℝ) where
  W : Kernel4 oc ic kH kW
  b : Vec oc
  hW : ∀ o c kh kw, |W o c kh kw| ≤ w'
  hb : ∀ o, |b o| ≤ β'

/-- A depthwise convolution's stored parameters with their magnitude bounds. -/
structure MnvDw (c kH kW : Nat) (w' β' : ℝ) where
  W : DepthwiseKernel c kH kW
  b : Vec c
  hW : ∀ ch kh kw, |W ch kh kw| ≤ w'
  hb : ∀ ch, |b ch| ≤ β'

/-- The classifier's stored parameters with their magnitude bounds. -/
structure MnvHead (m n : Nat) (w' β' : ℝ) where
  W : Mat m n
  b : Vec n
  hW : ∀ i j, |W i j| ≤ w'
  hb : ∀ j, |b j| ≤ β'

/-- One inference-BN site: scale, shift, and the two FROZEN running statistics. -/
structure MnvBn (c : Nat) (G Bb Mb : ℝ) where
  γ : Vec c
  β : Vec c
  μ : Vec c
  v : Vec c
  hγ : ∀ o, |γ o| ≤ G
  hβ : ∀ o, |β o| ≤ Bb
  hμ : ∀ o, |μ o| ≤ Mb
  /-- Running variances are nonnegative — what puts the inverse-stddev under the `ε`-floor. -/
  hv : ∀ o, 0 ≤ v o

/-- One inverted-residual block: expand 1×1, depthwise 3×3, project 1×1, each with its BN site.
    `ic → mid → oc`; whether it is the stride-1, stride-2 or skip form is a property of how the
    net USES it, not of the stored parameters, so one record serves all three. -/
structure MnvBlock (ic mid oc : Nat) (w' β' G Bb Mb : ℝ) where
  ex : MnvConv mid ic 1 1 w' β'
  bne : MnvBn mid G Bb Mb
  dw : MnvDw mid 3 3 w' β'
  bnd : MnvBn mid G Bb Mb
  pr : MnvConv oc mid 1 1 w' β'
  bnp : MnvBn oc G Bb Mb

/-- **The whole net's stored parameters at one uniform profile** — the stem, six
    inverted-residual blocks, the 1×1 head and the classifier, every entry within
    `w'`/`β'`/`G`/`Bb`/`Mb`. Twenty BN sites and twenty convolutions in eleven fields. -/
structure MnvWeights (w' β' G Bb Mb : ℝ) where
  stem : MnvConv 16 3 3 3 w' β'
  bns : MnvBn 16 G Bb Mb
  b1 : MnvBlock 16 64 24 w' β' G Bb Mb
  b2 : MnvBlock 24 96 24 w' β' G Bb Mb
  b3 : MnvBlock 24 96 32 w' β' G Bb Mb
  b4 : MnvBlock 32 128 32 w' β' G Bb Mb
  b5 : MnvBlock 32 128 64 w' β' G Bb Mb
  b6 : MnvBlock 64 256 64 w' β' G Bb Mb
  hd : MnvConv 128 64 1 1 w' β'
  bnh : MnvBn 128 G Bb Mb
  head : MnvHead 128 10 w' β'

/-- The numeric profile the fold runs at — `R34Profile`'s peer, field for field: every magnitude
    bound nonnegative, `ε` positive with its inverse-square-root under a rational `S`, and the
    rounding unit under a rational `q`. -/
structure MnvProfile (M : FloatModel) (ε w' β' G Bb Mb es S q : ℝ) : Prop where
  hw' : 0 ≤ w'
  hβ' : 0 ≤ β'
  hG : 0 ≤ G
  hBb : 0 ≤ Bb
  hMb : 0 ≤ Mb
  hes : 0 ≤ es
  hS0 : 0 ≤ S
  hε : 0 < ε
  hSε : 1 / Real.sqrt ε ≤ S
  hq : M.u ≤ q

-- ════════════════════════════════════════════════════════════════
-- § One BN site: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

variable {M : FloatModel} {ε w' β' G Bb Mb es S q : ℝ}

/-- The certified ℝ inference BN at this site. -/
noncomputable def MnvBn.fwd {c : Nat} (B : MnvBn c G Bb Mb) (ε : ℝ) (h w : Nat) :
    Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelEvalTensor3 c h w ε B.γ B.β B.μ B.v

/-- The deployed float inference BN at this site: the six rounded ops, with the device `rsqrt`
    evaluated at the frozen `var + ε`. -/
noncomputable def MnvBn.fwdF {c : Nat} (B : MnvBn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (h w : Nat) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelEvalTensor3FV M B.γ B.β B.μ (fun o => R.rsq (B.v o + ε))

/-- This BN site's bridge. -/
noncomputable def MnvBn.bridge {c h w : Nat} (B : MnvBn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) :
    FloatBridgesTo (B.fwd ε h w) (B.fwdF M R h w) :=
  floatBridgesTo_bnPerChannelEvalTensor3 (h := h) (w := w) M B.γ B.β B.μ B.v
    (fun o => R.rsq (B.v o + ε)) hc hhw P.hε B.hv B.hγ B.hβ B.hμ
    (fun o => R.spec (B.v o) (B.hv o)) P.hSε

/-- This BN site's numeric envelope — two linear inequalities. -/
theorem MnvBn.maps {c h w : Nat} (B : MnvBn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) {Ā Ē Ā' Ē' : ℝ} (hĀ0 : 0 ≤ Ā)
    (hĀ' : G * ((Ā + Mb) * S) + Bb + bnNormBudget q (Ā + Mb) S G Bb 0 es ≤ Ā')
    (hĒ' : bnNormBudget q (Ā + Mb) S G Bb 0 es + G * S * Ē ≤ Ē') :
    (B.bridge M R P hc hhw (h := h) (w := w)).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnEvalPC (h := h) (w := w) M B.γ B.β B.μ B.v
    (fun o => R.rsq (B.v o + ε)) hc hhw P.hε B.hv B.hγ B.hβ B.hμ
    (fun o => R.spec (B.v o) (B.hv o)) P.hSε P.hq P.hG P.hBb P.hS0 P.hMb P.hes hĀ0 hĀ' hĒ'

-- ════════════════════════════════════════════════════════════════
-- § One inverted-residual block: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

/-- The certified ℝ stride-1 body at inference: `project ∘ depthwise ∘ expand`. -/
noncomputable def MnvBlock.bodyFwd {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
    (ε : ℝ) : Vec (ic * h * w) → Vec (oc * h * w) :=
  invresBodyGen (h := h) (w := w) B.ex.W B.ex.b (B.bne.fwd ε h w)
    B.dw.W B.dw.b (B.bnd.fwd ε h w) B.pr.W B.pr.b (B.bnp.fwd ε h w)

/-- The deployed float stride-1 body. -/
noncomputable def MnvBlock.bodyFwdF {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) : Vec (ic * h * w) → Vec (oc * h * w) :=
  invresBodyGenF M B.ex.W B.ex.b (B.bne.fwdF M R h w) B.dw.W B.dw.b (B.bnd.fwdF M R h w)
    B.pr.W B.pr.b (B.bnp.fwdF M R h w)

/-- The certified ℝ stride-2 body at inference — the expand runs at `2h×2w`, so its BN does
    too, and the depthwise decimates. -/
noncomputable def MnvBlock.stridedFwd {ic mid oc h w : Nat}
    (B : MnvBlock ic mid oc w' β' G Bb Mb) (ε : ℝ) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  invresBodyStridedGen (h := h) (w := w) B.ex.W B.ex.b (B.bne.fwd ε (2 * h) (2 * w))
    B.dw.W B.dw.b (B.bnd.fwd ε h w) B.pr.W B.pr.b (B.bnp.fwd ε h w)

/-- The deployed float stride-2 body. -/
noncomputable def MnvBlock.stridedFwdF {ic mid oc h w : Nat}
    (B : MnvBlock ic mid oc w' β' G Bb Mb) (M : FloatModel) (R : DeviceRsqrt ε es) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  invresBodyStridedGenF M B.ex.W B.ex.b (B.bne.fwdF M R (2 * h) (2 * w))
    B.dw.W B.dw.b (B.bnd.fwdF M R h w) B.pr.W B.pr.b (B.bnp.fwdF M R h w)

/-- The certified ℝ matched-channel block: the stride-1 body under the additive skip
    (`b2`/`b4`). -/
noncomputable def MnvBlock.resFwd {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
    (ε : ℝ) : Vec (ic * h * w) → Vec (ic * h * w) :=
  Proofs.residual (B.bodyFwd (h := h) (w := w) ε)

/-- The deployed float matched-channel block — the skip-add is rounded. -/
noncomputable def MnvBlock.resFwdF {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) : Vec (ic * h * w) → Vec (ic * h * w) :=
  fun v j => M.add (B.bodyFwdF (h := h) (w := w) M R v j) (v j)

/-- This block's stride-1 body bridge. -/
noncomputable def MnvBlock.bodyBridge {ic mid oc h w : Nat}
    (B : MnvBlock ic mid oc w' β' G Bb Mb) (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w)
    (hni : 0 < ic * h * w) (hnm : 0 < mid * h * w) :
    FloatBridgesTo (B.bodyFwd (h := h) (w := w) ε) (B.bodyFwdF (h := h) (w := w) M R) :=
  floatBridgesTo_invresBodyGen (h := h) (w := w) M B.ex.W B.ex.b B.dw.W B.dw.b B.pr.W B.pr.b
    (B.bne.fwd ε h w) (B.bne.fwdF M R h w) (B.bnd.fwd ε h w) (B.bnd.fwdF M R h w)
    (B.bnp.fwd ε h w) (B.bnp.fwdF M R h w)
    P.hw' P.hβ' hni hnm B.ex.hW B.ex.hb B.dw.hW B.dw.hb B.pr.hW B.pr.hb
    (B.bne.bridge M R P hmid hhw) (B.bnd.bridge M R P hmid hhw) (B.bnp.bridge M R P hoc hhw)

/-- This block's stride-2 body bridge. -/
noncomputable def MnvBlock.stridedBridge {ic mid oc h w : Nat}
    (B : MnvBlock ic mid oc w' β' G Bb Mb) (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w) (hhw2 : 0 < (2 * h) * (2 * w))
    (hni : 0 < ic * (2 * h) * (2 * w)) (hnm2 : 0 < mid * (2 * h) * (2 * w))
    (hnm : 0 < mid * h * w) :
    FloatBridgesTo (B.stridedFwd (h := h) (w := w) ε) (B.stridedFwdF (h := h) (w := w) M R) :=
  floatBridgesTo_invresBodyStridedGen (h := h) (w := w) M B.ex.W B.ex.b B.dw.W B.dw.b
    B.pr.W B.pr.b (B.bne.fwd ε (2 * h) (2 * w)) (B.bne.fwdF M R (2 * h) (2 * w))
    (B.bnd.fwd ε h w) (B.bnd.fwdF M R h w) (B.bnp.fwd ε h w) (B.bnp.fwdF M R h w)
    P.hw' P.hβ' hni hnm2 hnm B.ex.hW B.ex.hb B.dw.hW B.dw.hb B.pr.hW B.pr.hb
    (B.bne.bridge M R P hmid hhw2) (B.bnd.bridge M R P hmid hhw) (B.bnp.bridge M R P hoc hhw)

/-- This block's matched-channel bridge — the body under `FloatBridgesTo.residual`. -/
noncomputable def MnvBlock.resBridge {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hic : 0 < ic) (hhw : 0 < h * w) (hni : 0 < ic * h * w)
    (hnm : 0 < mid * h * w) :
    FloatBridgesTo (B.resFwd (h := h) (w := w) ε) (B.resFwdF (h := h) (w := w) M R) :=
  (B.bodyBridge M R P hmid hic hhw hni hnm).residual M

/-- **This block's stride-1 numeric envelope** — eight numeric stages: expand conv, expand BN,
    relu6 (window `min A 6`, error unchanged), depthwise, depthwise BN, relu6, project conv,
    project BN. Sixteen inequalities. The project stage is the linear bottleneck and carries no
    relu6, so it is the one stage of a block whose window is NOT reset. -/
theorem MnvBlock.bodyMaps {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w)
    (hni : 0 < ic * h * w) (hnm : 0 < mid * h * w) (hno : 0 < oc * h * w)
    {ge gd gp Ā Ē A1 E1 A2 E2 A3 A4 E4 A5 E5 A6 A7 E7 Ā' Ē' : ℝ}
    (hge : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ ge)
    (hgd : (1 + M.u) ^ (3 * 3 + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (mid * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA40 : 0 ≤ A4) (hA70 : 0 ≤ A7)
    (eA : (1 + ge) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (eE : ge * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (enA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (enE : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (er : min A2 6 ≤ A3)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : bnNormBudget q (A4 + Mb) S G Bb 0 es + G * S * E4 ≤ E5)
    (dr : min A5 6 ≤ A6)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A6 + β') ≤ A7)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E5) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E5 ≤ E7)
    (pnA : G * ((A7 + Mb) * S) + Bb + bnNormBudget q (A7 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A7 + Mb) S G Bb 0 es + G * S * E7 ≤ Ē') :
    (B.bodyBridge M R P hmid hoc hhw hni hnm).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.ex.W B.ex.b P.hw' P.hβ' hni
    B.ex.hW B.ex.hb hge eA eE
  have s2 := s1.comp hnm (B.bne.maps M R P hmid hhw hA10 enA enE)
  have s3 := s2.comp hnm (FloatBridgesTo.Maps.relu6 (n := mid * h * w) er le_rfl)
  have s4 := s3.comp hnm (FloatBridgesTo.Maps.depthwise (h := h) (w := w) M B.dw.W B.dw.b
    P.hw' P.hβ' hnm B.dw.hW B.dw.hb hgd dA dE)
  have s5 := s4.comp hnm (B.bnd.maps M R P hmid hhw hA40 dnA dnE)
  have s6 := s5.comp hnm (FloatBridgesTo.Maps.relu6 (n := mid * h * w) dr le_rfl)
  -- the project stage is ONE `.comp` unit in `floatBridgesTo_invresBodyGen`, so the envelope
  -- has to associate the same way or the closing `exact` sees a different intermediate dim
  have pj := (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.pr.W B.pr.b
    P.hw' P.hβ' hnm B.pr.hW B.pr.hb hgp pA pE).comp hno
    (B.bnp.maps M R P hoc hhw hA70 pnA pnE)
  exact s6.comp hnm pj

/-- **This block's stride-2 numeric envelope** — the same eight stages with the expand at
    `2h×2w` and the depthwise decimating. The fan-ins, and hence every inequality, are
    identical: decimating an output picks coordinates. -/
theorem MnvBlock.stridedMaps {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w) (hhw2 : 0 < (2 * h) * (2 * w))
    (hni : 0 < ic * (2 * h) * (2 * w)) (hnm2 : 0 < mid * (2 * h) * (2 * w))
    (hnm : 0 < mid * h * w) (hno : 0 < oc * h * w)
    {ge gd gp Ā Ē A1 E1 A2 E2 A3 A4 E4 A5 E5 A6 A7 E7 Ā' Ē' : ℝ}
    (hge : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ ge)
    (hgd : (1 + M.u) ^ (3 * 3 + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (mid * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA40 : 0 ≤ A4) (hA70 : 0 ≤ A7)
    (eA : (1 + ge) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (eE : ge * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (enA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (enE : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (er : min A2 6 ≤ A3)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : bnNormBudget q (A4 + Mb) S G Bb 0 es + G * S * E4 ≤ E5)
    (dr : min A5 6 ≤ A6)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A6 + β') ≤ A7)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E5) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E5 ≤ E7)
    (pnA : G * ((A7 + Mb) * S) + Bb + bnNormBudget q (A7 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A7 + Mb) S G Bb 0 es + G * S * E7 ≤ Ē') :
    (B.stridedBridge M R P hmid hoc hhw hhw2 hni hnm2 hnm).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.flatConv (h := 2 * h) (w := 2 * w) M B.ex.W B.ex.b
    P.hw' P.hβ' hni B.ex.hW B.ex.hb hge eA eE
  have s2 := s1.comp hnm2 (B.bne.maps M R P hmid hhw2 hA10 enA enE)
  have s3 := s2.comp hnm2 (FloatBridgesTo.Maps.relu6 (n := mid * (2 * h) * (2 * w)) er le_rfl)
  have s4 := s3.comp hnm2 (FloatBridgesTo.Maps.depthwiseStride2Flat (h := h) (w := w) M
    B.dw.W B.dw.b P.hw' P.hβ' hnm2 B.dw.hW B.dw.hb hgd dA dE)
  have s5 := s4.comp hnm (B.bnd.maps M R P hmid hhw hA40 dnA dnE)
  have s6 := s5.comp hnm (FloatBridgesTo.Maps.relu6 (n := mid * h * w) dr le_rfl)
  have pj := (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.pr.W B.pr.b
    P.hw' P.hβ' hnm B.pr.hW B.pr.hb hgp pA pE).comp hno
    (B.bnp.maps M R P hoc hhw hA70 pnA pnE)
  exact s6.comp hnm pj

/-- **This block's matched-channel numeric envelope** — the body's eight stages, then the
    rounded skip fan-in against the block's own input window. Eighteen inequalities. -/
theorem MnvBlock.resMaps {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hic : 0 < ic) (hhw : 0 < h * w)
    (hni : 0 < ic * h * w) (hnm : 0 < mid * h * w)
    {ge gd gp Ā Ē A1 E1 A2 E2 A3 A4 E4 A5 E5 A6 A7 E7 Bd Ed Ā' Ē' : ℝ}
    (hge : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ ge)
    (hgd : (1 + M.u) ^ (3 * 3 + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (mid * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA40 : 0 ≤ A4) (hA70 : 0 ≤ A7)
    (eA : (1 + ge) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (eE : ge * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (enA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (enE : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (er : min A2 6 ≤ A3)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : bnNormBudget q (A4 + Mb) S G Bb 0 es + G * S * E4 ≤ E5)
    (dr : min A5 6 ≤ A6)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A6 + β') ≤ A7)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E5) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E5 ≤ E7)
    (pnA : G * ((A7 + Mb) * S) + Bb + bnNormBudget q (A7 + Mb) S G Bb 0 es ≤ Bd)
    (pnE : bnNormBudget q (A7 + Mb) S G Bb 0 es + G * S * E7 ≤ Ed)
    (rA : Bd + Ā + q * (Bd + Ā) ≤ Ā') (rE : q * (Bd + Ed + Ā + Ē) + (Ed + Ē) ≤ Ē') :
    (B.resBridge M R P hmid hic hhw hni hnm).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.residual M hni
    (B.bodyMaps M R P hmid hic hhw hni hnm hni hge hgd hgp hA10 hA40 hA70
      eA eE enA enE er dA dE dnA dnE dr pA pE pnA pnE) P.hq rA rE

-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The deployed MobileNetV2 inference forward** — the committed `mnv2Forward` skeleton with
    inference BatchNorm at every one of its 20 sites. -/
noncomputable def mnv2EvalForward (W : MnvWeights w' β' G Bb Mb) (ε : ℝ) :
    Vec (3 * 224 * 224) → Vec 10 :=
  mnv2Forward W.stem.W W.stem.b W.hd.W W.hd.b W.head.W W.head.b
    (W.bns.fwd ε 112 112) (W.bnh.fwd ε 7 7)
    (W.b1.stridedFwd (h := 56) (w := 56) ε) (W.b2.resFwd (h := 56) (w := 56) ε) (W.b3.stridedFwd (h := 28) (w := 28) ε)
    (W.b4.resFwd (h := 28) (w := 28) ε) (W.b5.stridedFwd (h := 14) (w := 14) ε) (W.b6.stridedFwd (h := 7) (w := 7) ε)

/-- **The deployed MobileNetV2 float inference forward** — every concrete slot replaced by the
    model's rounded peer, every BN by the six rounded ops the emitter writes, `relu6` unchanged
    (clamp-and-select rounds nothing). -/
noncomputable def mnv2EvalForwardF (M : FloatModel) (R : DeviceRsqrt ε es)
    (W : MnvWeights w' β' G Bb Mb) : Vec (3 * 224 * 224) → Vec 10 :=
  mnv2ForwardF M W.stem.W W.stem.b W.hd.W W.hd.b W.head.W W.head.b
    (W.bns.fwdF M R 112 112) (W.bnh.fwdF M R 7 7)
    (W.b1.stridedFwdF (h := 56) (w := 56) M R) (W.b2.resFwdF (h := 56) (w := 56) M R) (W.b3.stridedFwdF (h := 28) (w := 28) M R)
    (W.b4.resFwdF (h := 28) (w := 28) M R) (W.b5.stridedFwdF (h := 14) (w := 14) M R) (W.b6.stridedFwdF (h := 7) (w := 7) M R)

set_option maxRecDepth 100000 in
/-- ⭐ **The whole deployed MobileNetV2 inference forward float-bridges TO its float peer** — a
    CLOSED `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left: the stem conv/BN/relu6, the
    six inverted-residual blocks, the 1×1 head, GAP and the classifier are each discharged by a
    leaf. Its `.mod` is therefore a closed term over the per-op budgets, and `mnv2EvalBridge_maps`
    bounds it. -/
noncomputable def mnv2EvalBridge (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : MnvProfile M ε w' β' G Bb Mb es S q) (W : MnvWeights w' β' G Bb Mb) :
    FloatBridgesTo (mnv2EvalForward W ε) (mnv2EvalForwardF M R W) :=
  ((((((((((((
    (floatBridgesTo_flatConvStride2 (h := 112) (w := 112) M W.stem.W W.stem.b P.hw' P.hβ'
      (by norm_num) W.stem.hW W.stem.hb)
    |>.comp (W.bns.bridge M R P (by norm_num) (by norm_num) (h := 112) (w := 112)))
    |>.comp (floatBridgesTo_relu6 (n := 16 * 112 * 112)))
    |>.comp (W.b1.stridedBridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.b2.resBridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.b3.stridedBridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b4.resBridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b5.stridedBridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b6.stridedBridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (floatBridgesTo_flatConv (h := 7) (w := 7) M W.hd.W W.hd.b P.hw' P.hβ'
      (by norm_num) W.hd.hW W.hd.hb))
    |>.comp (W.bnh.bridge M R P (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (floatBridgesTo_relu6 (n := 128 * 7 * 7)))
    |>.comp (floatBridgesTo_gap (c := 128) (h := 7) (w := 7) M (by norm_num) (by norm_num)))
    |>.comp (floatBridgesTo_dense M W.head.W W.head.b P.hw' P.hβ' (by norm_num) W.head.hW W.head.hb)


-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile.** Every stored parameter within `28/10` — the global maximum over
    the 350-epoch checkpoint (`/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin`, 3.5M f32) is
    `2.7157`, its 99.99th percentile is `1.53` and only two entries exceed `2`, so the uniform
    bound is loose and the fold is not sensitive to it. `ε ≥ 10⁻⁵` (the trainer's value) puts the
    inference inverse-stddev under `317`, and the device `rsqrt` is taken accurate to `10⁻²`
    absolute — the same three choices `r34Profile_committed` makes. -/
theorem mnv2Profile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    MnvProfile M ε (28/10) (28/10) (28/10) (28/10) (28/10) (1/100) 317 u32 where
  hw' := by norm_num
  hβ' := by norm_num
  hG := by norm_num
  hBb := by norm_num
  hMb := by norm_num
  hes := by norm_num
  hS0 := by norm_num
  hε := by linarith
  hSε := by
    have hlo : (1:ℝ) / 317 ≤ Real.sqrt ε := by
      have hrw : ((1:ℝ) / 317) = Real.sqrt (((1:ℝ) / 317) ^ 2) :=
        (Real.sqrt_sq (by norm_num)).symm
      rw [hrw]
      exact Real.sqrt_le_sqrt (by nlinarith)
    have hpos : (0:ℝ) < Real.sqrt ε := lt_of_lt_of_le (by norm_num) hlo
    rw [div_le_iff₀ hpos]
    nlinarith
  hq := hMu

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 4000000 in
/-- ⭐ **The envelope, kernel-checked.** Every numeric stage closed by two rational
    inequalities, with each γ-term bounded through `FloatModel.gamma_num` so `norm_num`
    never evaluates a big power. Built bottom-up at block granularity (14 steps rather
    than ~60 leaf steps); the closing `exact` is one structural comparison with
    `mnv2EvalBridge`'s definition. -/
theorem mnv2EvalBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvWeights (28/10) (28/10) (28/10) (28/10) (28/10)) :
    (mnv2EvalBridge M R (mnv2Profile_committed M hMu hε5) W).Maps 1 0
      (2154) (1444 * 10 ^ 93) := by
  have hP := mnv2Profile_committed M hMu hε5
  have t1 := FloatBridgesTo.Maps.flatConvStride2 (h := 112) (w := 112) M W.stem.W W.stem.b
    hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb (M.gamma_num (q := 1729 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 7841 / 10 ^ 2) (Ē' := 1356 / 10 ^ 7) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
  have t2 := t1.comp (by norm_num) (W.bns.maps M R hP (by norm_num) (by norm_num) (h := 112) (w := 112)
    (Ā' := 7209 * 10 ^ 1) (Ē' := 2412 / 10 ^ 3) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t3 := t2.comp (by norm_num) (FloatBridgesTo.Maps.relu6 (n := 16 * 112 * 112)
    (Ā' := 6) (Ē' := 2412 / 10 ^ 3) (by norm_num) le_rfl)
  have t4 := t3.comp (by norm_num) (W.b1.stridedMaps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56)
    (ge := 1073 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 3934 / 10 ^ 9)
    (A1 := 2717 / 10 ^ 1) (E1 := 1081 / 10 ^ 1)
    (A2 := 2437 * 10 ^ 2) (E2 := 9596 * 10 ^ 1)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 2419 * 10 ^ 3)
    (A5 := 1393 * 10 ^ 2) (E5 := 2148 * 10 ^ 6)
    (A6 := 6)
    (A7 := 1079) (E7 := 3850 * 10 ^ 8)
    (Ā' := 9603 * 10 ^ 2) (Ē' := 3418 * 10 ^ 11)
    (M.gamma_num (q := 1073 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t5 := t4.comp (by norm_num) (W.b2.resMaps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56)
    (ge := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5842 / 10 ^ 9)
    (A1 := 6454 * 10 ^ 4) (E1 := 2297 * 10 ^ 13)
    (A2 := 5729 * 10 ^ 7) (E2 := 2039 * 10 ^ 16)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 5139 * 10 ^ 17)
    (A5 := 1393 * 10 ^ 2) (E5 := 4562 * 10 ^ 20)
    (A6 := 6)
    (A7 := 1616) (E7 := 1227 * 10 ^ 23)
    (Bd := 1437 * 10 ^ 3) (Ed := 1090 * 10 ^ 26)
    (Ā' := 2398 * 10 ^ 3) (Ē' := 1091 * 10 ^ 26)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t6 := t5.comp (by norm_num) (W.b3.stridedMaps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28)
    (ge := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5842 / 10 ^ 9)
    (A1 := 1612 * 10 ^ 5) (E1 := 7332 * 10 ^ 27)
    (A2 := 1431 * 10 ^ 8) (E2 := 6508 * 10 ^ 30)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 1641 * 10 ^ 32)
    (A5 := 1393 * 10 ^ 2) (E5 := 1457 * 10 ^ 35)
    (A6 := 6)
    (A7 := 1616) (E7 := 3917 * 10 ^ 37)
    (Ā' := 1437 * 10 ^ 3) (Ē' := 3477 * 10 ^ 40)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t7 := t6.comp (by norm_num) (W.b4.resMaps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28)
    (ge := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 7749 / 10 ^ 9)
    (A1 := 1288 * 10 ^ 5) (E1 := 3116 * 10 ^ 42)
    (A2 := 1144 * 10 ^ 8) (E2 := 2766 * 10 ^ 45)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 6971 * 10 ^ 46)
    (A5 := 1393 * 10 ^ 2) (E5 := 6188 * 10 ^ 49)
    (A6 := 6)
    (A7 := 2154) (E7 := 2218 * 10 ^ 52)
    (Bd := 1915 * 10 ^ 3) (Ed := 1969 * 10 ^ 55)
    (Ā' := 3353 * 10 ^ 3) (Ē' := 1970 * 10 ^ 55)
    (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t8 := t7.comp (by norm_num) (W.b5.stridedMaps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 7749 / 10 ^ 9)
    (A1 := 3005 * 10 ^ 5) (E1 := 1766 * 10 ^ 57)
    (A2 := 2668 * 10 ^ 8) (E2 := 1568 * 10 ^ 60)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 3952 * 10 ^ 61)
    (A5 := 1393 * 10 ^ 2) (E5 := 3508 * 10 ^ 64)
    (A6 := 6)
    (A7 := 2154) (E7 := 1258 * 10 ^ 67)
    (Ā' := 1915 * 10 ^ 3) (Ē' := 1117 * 10 ^ 70)
    (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t9 := t8.comp (by norm_num) (W.b6.stridedMaps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7)
    (ge := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 1538 / 10 ^ 8)
    (A1 := 3432 * 10 ^ 5) (E1 := 2002 * 10 ^ 72)
    (A2 := 3047 * 10 ^ 8) (E2 := 1777 * 10 ^ 75)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 4479 * 10 ^ 76)
    (A5 := 1393 * 10 ^ 2) (E5 := 3976 * 10 ^ 79)
    (A6 := 6)
    (A7 := 4304) (E7 := 2851 * 10 ^ 82)
    (Ā' := 3823 * 10 ^ 3) (Ē' := 2531 * 10 ^ 85)
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t10 := t9.comp (by norm_num) (FloatBridgesTo.Maps.flatConv (h := 7) (w := 7) M
    W.hd.W W.hd.b hP.hw' hP.hβ' (by norm_num) W.hd.hW W.hd.hb (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 6851 * 10 ^ 5) (Ē' := 4536 * 10 ^ 87) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t11 := t10.comp (by norm_num) (W.bnh.maps M R hP (by norm_num) (by norm_num) (h := 7) (w := 7)
    (Ā' := 6082 * 10 ^ 8) (Ē' := 4027 * 10 ^ 90) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t12 := t11.comp (by norm_num) (FloatBridgesTo.Maps.relu6 (n := 128 * 7 * 7)
    (Ā' := 6) (Ē' := 4027 * 10 ^ 90) (by norm_num) le_rfl)
  have t13 := t12.comp (by norm_num) (FloatBridgesTo.Maps.gap (c := 128) (h := 7) (w := 7) M (by norm_num) (by norm_num)
    hMu (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 6001 / 10 ^ 3) (Ē' := 4028 * 10 ^ 90) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t14 := t13.comp (by norm_num) (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num)
    W.head.hW W.head.hb (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 2154) (Ē' := 1444 * 10 ^ 93) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  exact t14

/-- The deployed MobileNetV2 inference bridge's certified output window at the committed profile:
    `≤ 2154`. ⭐ Against logits of a few this is the first ImageNet-scale certified window in the
    repo that is not vacuous by hundreds of orders — see the file header for why (the relu6
    clamp) and for why the budget below does not follow it down. -/
theorem mnv2EvalBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvWeights (28/10) (28/10) (28/10) (28/10) (28/10)) :
    (mnv2EvalBridge M R (mnv2Profile_committed M hMu hε5) W).mag 1 ≤ 2154 :=
  (mnv2EvalBridge_maps M hMu hε5 R W).mag_le 1 (by norm_num) le_rfl

/-- The deployed MobileNetV2 inference bridge's fresh budget at the committed profile:
    `≤ 1.444·10⁹⁶`. It is `20` BN sites × the per-site error gain `G·S`, and it is unmoved by the
    window collapse. -/
theorem mnv2EvalBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvWeights (28/10) (28/10) (28/10) (28/10) (28/10)) :
    (mnv2EvalBridge M R (mnv2Profile_committed M hMu hε5) W).fresh 1 ≤ 1444 * 10 ^ 93 :=
  (mnv2EvalBridge_maps M hMu hε5 R W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed MobileNetV2 inference forward is within `1.444·10⁹⁶` of the certified real
    forward, per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile
    (`|·| ≤ 28/10`), for `ε ≥ 10⁻⁵`, any device `rsqrt` accurate to `10⁻²`, and any rounding model
    at binary32 accuracy. The second ImageNet-scale whole-net float budget in the repo stated as a
    number over a `FloatBridgesTo`, and the first whose certified WINDOW (`2154`) is within a few
    orders of the quantity it bounds. -/
theorem mnv2_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvWeights (28/10) (28/10) (28/10) (28/10) (28/10))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |mnv2EvalForwardF M R W x j - mnv2EvalForward W ε x j| ≤ 1444 * 10 ^ 93 :=
  (mnv2EvalBridge_maps M hMu hε5 R W).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed inference forward, and the graph denotes it
-- ════════════════════════════════════════════════════════════════

/-- **The record-bundled forward IS the committed inference net.** `mnv2EvalForward` unfolds to
    `mobilenetv2Forward_full_pc_eval` at the record's projections — the eval twin of
    `mobilenetv2Forward_full_pc_eq_skeleton`, and a `rfl` for the same reason: the skeleton's
    block slots take exactly the maps the record builds. -/
theorem mnv2EvalForward_eq_full_pc_eval (W : MnvWeights w' β' G Bb Mb) (ε : ℝ) :
    mnv2EvalForward W ε = mobilenetv2Forward_full_pc_eval ε
    W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v
    W.b1.ex.W W.b1.ex.b W.b1.bne.γ W.b1.bne.β W.b1.bne.μ W.b1.bne.v W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v
    W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v
    W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v
    W.b4.ex.W W.b4.ex.b W.b4.bne.γ W.b4.bne.β W.b4.bne.μ W.b4.bne.v W.b4.dw.W W.b4.dw.b W.b4.bnd.γ W.b4.bnd.β W.b4.bnd.μ W.b4.bnd.v W.b4.pr.W W.b4.pr.b W.b4.bnp.γ W.b4.bnp.β W.b4.bnp.μ W.b4.bnp.v
    W.b5.ex.W W.b5.ex.b W.b5.bne.γ W.b5.bne.β W.b5.bne.μ W.b5.bne.v W.b5.dw.W W.b5.dw.b W.b5.bnd.γ W.b5.bnd.β W.b5.bnd.μ W.b5.bnd.v W.b5.pr.W W.b5.pr.b W.b5.bnp.γ W.b5.bnp.β W.b5.bnp.μ W.b5.bnp.v
    W.b6.ex.W W.b6.ex.b W.b6.bne.γ W.b6.bne.β W.b6.bne.μ W.b6.bne.v W.b6.dw.W W.b6.dw.b W.b6.bnd.γ W.b6.bnd.β W.b6.bnd.μ W.b6.bnd.v W.b6.pr.W W.b6.pr.b W.b6.bnp.γ W.b6.bnp.β W.b6.bnp.μ W.b6.bnp.v
    W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b := rfl

/-- ⭐ **The whole loop closes.** The typed `SHlo` inference graph denotes exactly the forward
    this file states its number about. `mobilenetv2FwdGraphFullPCEval_faithful` carried through
    the record tie. -/
theorem mnv2EvalGraph_faithful (epsStr : String) (ε : ℝ) (W : MnvWeights w' β' G Bb Mb)
    (x : Vec (3 * 224 * 224)) :
    StableHLO.den (StableHLO.mobilenetv2FwdGraphFullPCEval epsStr ε
    W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v
    W.b1.ex.W W.b1.ex.b W.b1.bne.γ W.b1.bne.β W.b1.bne.μ W.b1.bne.v W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v
    W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v
    W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v
    W.b4.ex.W W.b4.ex.b W.b4.bne.γ W.b4.bne.β W.b4.bne.μ W.b4.bne.v W.b4.dw.W W.b4.dw.b W.b4.bnd.γ W.b4.bnd.β W.b4.bnd.μ W.b4.bnd.v W.b4.pr.W W.b4.pr.b W.b4.bnp.γ W.b4.bnp.β W.b4.bnp.μ W.b4.bnp.v
    W.b5.ex.W W.b5.ex.b W.b5.bne.γ W.b5.bne.β W.b5.bne.μ W.b5.bne.v W.b5.dw.W W.b5.dw.b W.b5.bnd.γ W.b5.bnd.β W.b5.bnd.μ W.b5.bnd.v W.b5.pr.W W.b5.pr.b W.b5.bnp.γ W.b5.bnp.β W.b5.bnp.μ W.b5.bnp.v
    W.b6.ex.W W.b6.ex.b W.b6.bne.γ W.b6.bne.β W.b6.bne.μ W.b6.bne.v W.b6.dw.W W.b6.dw.b W.b6.bnd.γ W.b6.bnd.β W.b6.bnd.μ W.b6.bnd.v W.b6.pr.W W.b6.pr.b W.b6.bnp.γ W.b6.bnp.β W.b6.bnp.μ W.b6.bnp.v
    W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x)
      = mnv2EvalForward W ε x :=
  (StableHLO.mobilenetv2FwdGraphFullPCEval_faithful epsStr ε W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v W.b1.ex.W W.b1.ex.b W.b1.bne.γ W.b1.bne.β W.b1.bne.μ W.b1.bne.v W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v W.b4.ex.W W.b4.ex.b W.b4.bne.γ W.b4.bne.β W.b4.bne.μ W.b4.bne.v W.b4.dw.W W.b4.dw.b W.b4.bnd.γ W.b4.bnd.β W.b4.bnd.μ W.b4.bnd.v W.b4.pr.W W.b4.pr.b W.b4.bnp.γ W.b4.bnp.β W.b4.bnp.μ W.b4.bnp.v W.b5.ex.W W.b5.ex.b W.b5.bne.γ W.b5.bne.β W.b5.bne.μ W.b5.bne.v W.b5.dw.W W.b5.dw.b W.b5.bnd.γ W.b5.bnd.β W.b5.bnd.μ W.b5.bnd.v W.b5.pr.W W.b5.pr.b W.b5.bnp.γ W.b5.bnp.β W.b5.bnp.μ W.b5.bnp.v W.b6.ex.W W.b6.ex.b W.b6.bne.γ W.b6.bne.β W.b6.bne.μ W.b6.bne.v W.b6.dw.W W.b6.dw.b W.b6.bnd.γ W.b6.bnd.β W.b6.bnd.μ W.b6.bnd.v W.b6.pr.W W.b6.pr.b W.b6.bnp.γ W.b6.bnp.β W.b6.bnp.μ W.b6.bnp.v W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x).trans
    (congrFun (mnv2EvalForward_eq_full_pc_eval W ε).symm x)

/-- ⭐⭐ **The number, stated about the committed inference forward.** `mnv2_float_logits_le`
    with `mobilenetv2Forward_full_pc_eval` on the real side instead of the record-bundled
    `mnv2EvalForward` — so the budget is a claim about the net the eval render emits, tied
    through `mnv2EvalGraph_faithful` rather than by inspection. -/
theorem mnv2_float_logits_le_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvWeights (28/10) (28/10) (28/10) (28/10) (28/10))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |mnv2EvalForwardF M R W x j - mobilenetv2Forward_full_pc_eval ε
    W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v
    W.b1.ex.W W.b1.ex.b W.b1.bne.γ W.b1.bne.β W.b1.bne.μ W.b1.bne.v W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v
    W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v
    W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v
    W.b4.ex.W W.b4.ex.b W.b4.bne.γ W.b4.bne.β W.b4.bne.μ W.b4.bne.v W.b4.dw.W W.b4.dw.b W.b4.bnd.γ W.b4.bnd.β W.b4.bnd.μ W.b4.bnd.v W.b4.pr.W W.b4.pr.b W.b4.bnp.γ W.b4.bnp.β W.b4.bnp.μ W.b4.bnp.v
    W.b5.ex.W W.b5.ex.b W.b5.bne.γ W.b5.bne.β W.b5.bne.μ W.b5.bne.v W.b5.dw.W W.b5.dw.b W.b5.bnd.γ W.b5.bnd.β W.b5.bnd.μ W.b5.bnd.v W.b5.pr.W W.b5.pr.b W.b5.bnp.γ W.b5.bnp.β W.b5.bnp.μ W.b5.bnp.v
    W.b6.ex.W W.b6.ex.b W.b6.bne.γ W.b6.bne.β W.b6.bne.μ W.b6.bne.v W.b6.dw.W W.b6.dw.b W.b6.bnd.γ W.b6.bnd.β W.b6.bnd.μ W.b6.bnd.v W.b6.pr.W W.b6.pr.b W.b6.bnp.γ W.b6.bnp.β W.b6.bnp.μ W.b6.bnp.v
    W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x j| ≤ 1444 * 10 ^ 93 :=
  mnv2_float_logits_le M hMu hε5 R W x hx j

end Proofs
