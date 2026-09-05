import LeanMlir.Proofs.Float.MobileNetV2FloatBudget

/-! # A NUMBER for the PAPER MobileNetV2: seventeen bottlenecks, and the cap that makes it statable

`MobileNetV2FloatBudget.lean` states its number about the six-block ch7 representative. This
file states one about the net the `[t,c,n,s]` table describes and `@mobilenetv2_fwd_eval`
renders — **all seventeen bottlenecks**, stem 3×3/s2 3→32 at the XLA-`SAME` phase, the four
strided depthwises likewise, `b1` the t=1 no-expand bottleneck, ten identity skips, two
stage-first widenings, 1×1 head 320→1280, GAP, dense — with **inference** BatchNorm at all
**52** sites (the reduced net has 20), on the unit input window, at the profile measured on the
350-epoch checkpoint (`|parameter| ≤ 28/10`), for any rounding model at binary32 accuracy:

    output window  ≤ 2.152·10⁴       (`mnv2PaperEvalBridge_mag_le`)
    fresh budget   ≤ 8.176·10¹⁶      (`mnv2PaperEvalBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 8.176·10¹⁶` (`mnv2Paper_float_logits_le`).

⛔⛔ **Read the budget's label first: every one of the 52 BatchNorms goes through
`FloatBridgesTo.capped`, so at those sites this is the TRIANGLE INEQUALITY and not a fold.** A
capped site's modulus is `min(fold, 2·window)` and only the right branch is ever emitted here:
what it says is *"the float and the real activation both land in the certified window"*, never
*"the rounding error folds to this"*. The fold is what it replaces — uncapped at the ε-floor the
52-site chain reaches `2.104·10²⁶⁶`, past any numeral `norm_num` will evaluate, so there is no
theorem to state without the cap. What the number below still says on its own terms is that
**from the last cap (`head.bn`) the three remaining stages — relu6, GAP, the classifier — fold
that capped error to `8.176·10¹⁶`**; the tell that a cap is in play is `budget/window ≈ 2`, and
here it is `3.8·10¹²`, entirely because relu6 and GAP collapse the window while the classifier's
1280-wide fan-in carries the error forward.

⭐⭐ **Capping is worth more than the eleven extra blocks cost.** `8.176·10¹⁶` at seventeen
blocks is **79 orders SMALLER** than the six-block file's uncapped `1.444·10⁹⁶`. Depth is not
what makes these numbers large; the per-site error gain `G·S` at the ε-floor is, and a cap stops
it dead. The honest comparison is therefore between two different claims, and this file's is the
weaker one at the BN sites and the smaller one everywhere.

⭐ **The window is still the half one could believe, and it barely moved: `2154 → 2.152·10⁴`.**
Eleven more blocks and 21 more relu6 sites (35 against the reduced net's 14) cost it nothing —
`floatClose_relu6`'s `min Ā 6` RESETS the certified magnitude at every one of them, so the body
is flat in depth. The entire growth is the **head width**: the classifier's fan-in is 1280
against the reduced net's 128, and `gap`'s output window `6.001·10⁻³` times that fan-in against
`|W| ≤ 28/10` is the `2.152·10⁴`. Window and budget remain separate levers
(`planning/float_budget_numbers.md` §3, finding 3).

**The profile is measured on THIS net.** `MobileNetV2FloatBudget.lean` takes `|·| ≤ 28/10` from
`/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin`, whose 3,504,872 f32 entries are the
seventeen-block net — global max `2.7157`, 99.99th percentile `1.5347`, exactly two entries above
`2` (re-measured 2026-09-05). At the reduced net that profile was a transplant; here it is the
checkpoint of the net being bounded — ⚠ with one qualification. Those entries are the
**1000-class** net (the twin `@mobilenetv2_fwd_eval` renders as `mobilenetv2in_fwd_eval.mlir`),
and this file's classifier is the committed 10-class one, so the bound is a MEASUREMENT on all 52
convolutions and 52 BatchNorms and an ASSUMPTION on the `1280 × 10` head. `Maps.dense`'s
envelope depends on the fan-in `1280` and never on the output count, so both numerals hold
verbatim at 1000 classes; making the head generic in `nCls` is
`planning/proofs_tier_to_paper_nets.md` 3.2(e). `ε ≥ 10⁻⁵` puts the inference inverse-stddev
under `317`.

⚠ **The one hypothesis this number rests on, named.** The deployed inverse-stddev is a device
`rsqrt` with no IEEE specification, so it is *modelled*: `DeviceRsqrt ε es` (shared with
`Resnet34FloatBudget.lean` and the six-block file — one modelled kernel, one assumption) supplies
it with an accuracy `es = 10⁻²`. Everything else is proved.

⛔ **There is no backward number at seventeen blocks and no cap rescues one**, because a cap's
budget is `2·window` and the backward WINDOW is itself `1.246·10³²³`. Ablated before blaming the
depth (`planning/float_budget_numbers.md` §3, finding 7): the BN γ bound `1.69 → 1` buys 12
orders and the conv kernel bound `2.72 → 1` buys 24, both MEASURED bounds rather than bounds
discarded one lemma down, and only their simultaneous fiction gets under the ceiling. This is EfficientNet-B0's
backward situation and takes the same answer — no `MobileNetV2PaperBackFloatBudget.lean`.
`planning/float_budget_numbers.md` §1 carries the finding.

**What this is stated about, and what it is not.** Each of the seventeen blocks is tied by
`rfl` to the abbreviation the committed inference forward is built from — `invresBodyPCEval`,
`invresBodyStridedPCEval`, and for `b1` the project-after-depthwise pair (`*_eq_pcEval` below,
dimension-polymorphic, so one proof covers every width in the table). The ladder itself — which
block at which spatial size, with which skip — is `mobilenetv2ForwardPaper`'s, read off the same
`[t,c,n,s]` table, and the widths are `paperSig`'s.

⛔ What is NOT closed is the whole-net step: `mobilenetv2ForwardPaper` is at TRAINING BatchNorm,
the world its VJP and its typed graph live in, and the paper net's *eval* twin has no ℝ-def and
no typed graph in Lean at all (`MobileNetV2RenderPCEval.lean` covers the six-block net only). So
there is no whole-net graph-faithfulness theorem here, and no restatement of the number with the
rendered net on the real side — that is the one rung `MobileNetV2FloatBudget.lean` has and this
file does not. Closing it means the eval twin of `MobileNetV2FullPaper.lean`'s graph section:
mechanical, and scoped separately in `planning/proofs_tier_to_paper_nets.md` 3.2(b).

Provenance for the numerals: `scripts/float_budget_envelope.py` (`mnv2_paper_eval_chain`), which
reads the block table from two Lean sources rather than a fourth hand-written copy, folds the
envelope in exactly these lemmas' semantics with exact rationals, rounds every stage UP to four
significant figures, and re-asserts each rounded inequality (`verify_mnv2_paper`, 354
inequalities) before emitting.
-/

namespace Proofs

open FloatModel

variable {M : FloatModel} {ε w' β' G Bb Mb es S q : ℝ}

-- ════════════════════════════════════════════════════════════════
-- § The cap at one inference-BatchNorm site
-- ════════════════════════════════════════════════════════════════

/-- ⛔ **One BN site under `FloatBridgesTo.capped`** — the same map and the same window as
    `MnvBn.maps`, with the modulus replaced by `min(fold, 2·window)`. Note what it does NOT
    take: no bound on the inherited error `Ē`. That is the whole mechanism — the error entering
    a capped site is discarded, so the 52-site chain never builds the numeral the fold would.

    The window clause is `MnvBn.maps`'s own, at the fold's `Ē'` and `le_rfl`: capping changes the
    modulus, never the magnitude. -/
theorem MnvBn.cappedMaps {c h w : Nat} (B : MnvBn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) {Ā Ē Ā' Ē' : ℝ} (hĀ0 : 0 ≤ Ā)
    (hĀ' : G * ((Ā + Mb) * S) + Bb + bnNormBudget q (Ā + Mb) S G Bb 0 es ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    (B.bridge M R P hc hhw (h := h) (w := w)).capped.Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.capped
    (fun A hA0 hA => (B.maps M R P hc hhw (Ē := Ē) hĀ0 hĀ' le_rfl).mag_le A hA0 hA) hĒ'

-- ════════════════════════════════════════════════════════════════
-- § The t=1 first bottleneck — the one block shape the reduced net does not have
-- ════════════════════════════════════════════════════════════════

/-- The paper table's first bottleneck (`t = 1`, `32 → 16`): no expand convolution, and no skip
    because the channels change. Depthwise 3×3 on `ic`, its BN, relu6, project 1×1, its BN. -/
structure MnvBlockNoExp (ic oc : Nat) (w' β' G Bb Mb : ℝ) where
  dw : MnvDw ic 3 3 w' β'
  bnd : MnvBn ic G Bb Mb
  pr : MnvConv oc ic 1 1 w' β'
  bnp : MnvBn oc G Bb Mb

/-- The certified ℝ t=1 bottleneck at inference: `project ∘ depthwise`, the two stage
    abbreviations `invresBodyGen` is built from, with the expand stage absent. -/
noncomputable def MnvBlockNoExp.fwd {ic oc h w : Nat} (B : MnvBlockNoExp ic oc w' β' G Bb Mb)
    (ε : ℝ) : Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectGen (h := h) (w := w) B.pr.W B.pr.b (B.bnp.fwd ε h w) ∘
    ivDepthwiseGen (h := h) (w := w) B.dw.W B.dw.b (B.bnd.fwd ε h w)

/-- The deployed float t=1 bottleneck. -/
noncomputable def MnvBlockNoExp.fwdF {ic oc h w : Nat} (B : MnvBlockNoExp ic oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) : Vec (ic * h * w) → Vec (oc * h * w) :=
  (B.bnp.fwdF M R h w ∘ M.flatConvF (h := h) (w := w) B.pr.W B.pr.b) ∘
    (relu6 (ic * h * w) ∘
      (B.bnd.fwdF M R h w ∘ M.depthwiseFlatF (h := h) (w := w) B.dw.W B.dw.b))

/-- The t=1 bottleneck's bridge, with both BN sites capped. -/
noncomputable def MnvBlockNoExp.bridgeC {ic oc h w : Nat}
    (B : MnvBlockNoExp ic oc w' β' G Bb Mb) (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hic : 0 < ic) (hoc : 0 < oc) (hhw : 0 < h * w) (hni : 0 < ic * h * w) :
    FloatBridgesTo (B.fwd (h := h) (w := w) ε) (B.fwdF (h := h) (w := w) M R) :=
  (((floatBridgesTo_depthwise (h := h) (w := w) M B.dw.W B.dw.b P.hw' P.hβ' hni
      B.dw.hW B.dw.hb).comp ((B.bnd.bridge M R P hic hhw).capped)).comp
    floatBridgesTo_relu6).comp
    ((floatBridgesTo_flatConv (h := h) (w := w) M B.pr.W B.pr.b P.hw' P.hβ' hni
      B.pr.hW B.pr.hb).comp ((B.bnp.bridge M R P hoc hhw).capped))

/-- **The t=1 bottleneck's numeric envelope, capped** — five numeric stages: depthwise, BN,
    relu6, project conv, BN. Nine numeric hypotheses, two of them the cap's `2 * Ā' ≤ Ē'`. -/
theorem MnvBlockNoExp.mapsC {ic oc h w : Nat} (B : MnvBlockNoExp ic oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hic : 0 < ic) (hoc : 0 < oc) (hhw : 0 < h * w)
    (hni : 0 < ic * h * w) (hno : 0 < oc * h * w)
    {gd gp Ā Ē A1 E1 A2 E2 A3 A4 E4 Ā' Ē' : ℝ}
    (hgd : (1 + M.u) ^ (3 * 3 + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA40 : 0 ≤ A4)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (dnA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (dnE : 2 * A2 ≤ E2)
    (dr : min A2 6 ≤ A3)
    (pA : (1 + gp) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (pE : gp * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (pnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : 2 * Ā' ≤ Ē') :
    (B.bridgeC M R P hic hoc hhw hni).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.depthwise (h := h) (w := w) M B.dw.W B.dw.b
    P.hw' P.hβ' hni B.dw.hW B.dw.hb hgd dA dE
  have s2 := s1.comp hni (B.bnd.cappedMaps M R P hic hhw hA10 dnA dnE)
  have s3 := s2.comp hni (FloatBridgesTo.Maps.relu6 (n := ic * h * w) dr le_rfl)
  have pj := (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.pr.W B.pr.b
    P.hw' P.hβ' hni B.pr.hW B.pr.hb hgp pA pE).comp hno
    (B.bnp.cappedMaps M R P hoc hhw hA40 pnA pnE)
  exact s3.comp hni pj

-- ════════════════════════════════════════════════════════════════
-- § The three inverted-residual forms, capped
-- ════════════════════════════════════════════════════════════════

/-- `MnvBlock.bodyBridge` with all three BatchNorm sites capped — the same map, so the block's
    forward and float peer are unchanged and only the modulus differs. -/
noncomputable def MnvBlock.bodyBridgeC {ic mid oc h w : Nat}
    (B : MnvBlock ic mid oc w' β' G Bb Mb) (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w)
    (hni : 0 < ic * h * w) (hnm : 0 < mid * h * w) :
    FloatBridgesTo (B.bodyFwd (h := h) (w := w) ε) (B.bodyFwdF (h := h) (w := w) M R) :=
  floatBridgesTo_invresBodyGen (h := h) (w := w) M B.ex.W B.ex.b B.dw.W B.dw.b B.pr.W B.pr.b
    (B.bne.fwd ε h w) (B.bne.fwdF M R h w) (B.bnd.fwd ε h w) (B.bnd.fwdF M R h w)
    (B.bnp.fwd ε h w) (B.bnp.fwdF M R h w)
    P.hw' P.hβ' hni hnm B.ex.hW B.ex.hb B.dw.hW B.dw.hb B.pr.hW B.pr.hb
    ((B.bne.bridge M R P hmid hhw).capped) ((B.bnd.bridge M R P hmid hhw).capped)
    ((B.bnp.bridge M R P hoc hhw).capped)

/-- `MnvBlock.stridedBridge` with all three BatchNorm sites capped. -/
noncomputable def MnvBlock.stridedBridgeC {ic mid oc h w : Nat}
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
    ((B.bne.bridge M R P hmid hhw2).capped) ((B.bnd.bridge M R P hmid hhw).capped)
    ((B.bnp.bridge M R P hoc hhw).capped)

/-- `MnvBlock.resBridge` with all three BatchNorm sites capped — the capped body under the
    additive skip. The skip-add itself is NOT capped: it is one rounded sum, linear in both
    arguments, and it is where the block's own input window re-enters. -/
noncomputable def MnvBlock.resBridgeC {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : MnvProfile M ε w' β' G Bb Mb es S q)
    (hmid : 0 < mid) (hic : 0 < ic) (hhw : 0 < h * w) (hni : 0 < ic * h * w)
    (hnm : 0 < mid * h * w) :
    FloatBridgesTo (B.resFwd (h := h) (w := w) ε) (B.resFwdF (h := h) (w := w) M R) :=
  (B.bodyBridgeC M R P hmid hic hhw hni hnm).residual M

/-- **The capped stride-1 envelope** — `MnvBlock.bodyMaps`'s eight stages with the three
    BatchNorm moduli replaced by the cap's `2 * Ā' ≤ Ē'`. Fourteen numeric hypotheses, three of
    them that side condition; the relu6 stages carry no error hypothesis (it passes unchanged). -/
theorem MnvBlock.bodyMapsC {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
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
    (enE : 2 * A2 ≤ E2)
    (er : min A2 6 ≤ A3)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : 2 * A5 ≤ E5)
    (dr : min A5 6 ≤ A6)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A6 + β') ≤ A7)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E5) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E5 ≤ E7)
    (pnA : G * ((A7 + Mb) * S) + Bb + bnNormBudget q (A7 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : 2 * Ā' ≤ Ē') :
    (B.bodyBridgeC M R P hmid hoc hhw hni hnm).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.ex.W B.ex.b P.hw' P.hβ' hni
    B.ex.hW B.ex.hb hge eA eE
  have s2 := s1.comp hnm (B.bne.cappedMaps M R P hmid hhw hA10 enA enE)
  have s3 := s2.comp hnm (FloatBridgesTo.Maps.relu6 (n := mid * h * w) er le_rfl)
  have s4 := s3.comp hnm (FloatBridgesTo.Maps.depthwise (h := h) (w := w) M B.dw.W B.dw.b
    P.hw' P.hβ' hnm B.dw.hW B.dw.hb hgd dA dE)
  have s5 := s4.comp hnm (B.bnd.cappedMaps M R P hmid hhw hA40 dnA dnE)
  have s6 := s5.comp hnm (FloatBridgesTo.Maps.relu6 (n := mid * h * w) dr le_rfl)
  have pj := (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.pr.W B.pr.b
    P.hw' P.hβ' hnm B.pr.hW B.pr.hb hgp pA pE).comp hno
    (B.bnp.cappedMaps M R P hoc hhw hA70 pnA pnE)
  exact s6.comp hnm pj

/-- **The capped stride-2 envelope** — the same eight stages with the expand at `2h×2w` and the
    XLA-`SAME` decimating depthwise. -/
theorem MnvBlock.stridedMapsC {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
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
    (enE : 2 * A2 ≤ E2)
    (er : min A2 6 ≤ A3)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : 2 * A5 ≤ E5)
    (dr : min A5 6 ≤ A6)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A6 + β') ≤ A7)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E5) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E5 ≤ E7)
    (pnA : G * ((A7 + Mb) * S) + Bb + bnNormBudget q (A7 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : 2 * Ā' ≤ Ē') :
    (B.stridedBridgeC M R P hmid hoc hhw hhw2 hni hnm2 hnm).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.flatConv (h := 2 * h) (w := 2 * w) M B.ex.W B.ex.b
    P.hw' P.hβ' hni B.ex.hW B.ex.hb hge eA eE
  have s2 := s1.comp hnm2 (B.bne.cappedMaps M R P hmid hhw2 hA10 enA enE)
  have s3 := s2.comp hnm2 (FloatBridgesTo.Maps.relu6 (n := mid * (2 * h) * (2 * w)) er le_rfl)
  have s4 := s3.comp hnm2 (FloatBridgesTo.Maps.depthwiseStride2FlatXla (h := h) (w := w) M
    B.dw.W B.dw.b P.hw' P.hβ' hnm2 B.dw.hW B.dw.hb hgd dA dE)
  have s5 := s4.comp hnm (B.bnd.cappedMaps M R P hmid hhw hA40 dnA dnE)
  have s6 := s5.comp hnm (FloatBridgesTo.Maps.relu6 (n := mid * h * w) dr le_rfl)
  have pj := (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.pr.W B.pr.b
    P.hw' P.hβ' hnm B.pr.hW B.pr.hb hgp pA pE).comp hno
    (B.bnp.cappedMaps M R P hoc hhw hA70 pnA pnE)
  exact s6.comp hnm pj

/-- **The capped matched-channel envelope** — the capped body's eight stages, then the rounded
    skip fan-in against the block's own input window. -/
theorem MnvBlock.resMapsC {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
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
    (enE : 2 * A2 ≤ E2)
    (er : min A2 6 ≤ A3)
    (dA : (1 + gd) * (((3 * 3 : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((3 * 3 : ℕ) : ℝ) * w' * (A3 + E2) + β')
            + ((3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : 2 * A5 ≤ E5)
    (dr : min A5 6 ≤ A6)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A6 + β') ≤ A7)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A6 + E5) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E5 ≤ E7)
    (pnA : G * ((A7 + Mb) * S) + Bb + bnNormBudget q (A7 + Mb) S G Bb 0 es ≤ Bd)
    (pnE : 2 * Bd ≤ Ed)
    (rA : Bd + Ā + q * (Bd + Ā) ≤ Ā') (rE : q * (Bd + Ed + Ā + Ē) + (Ed + Ē) ≤ Ē') :
    (B.resBridgeC M R P hmid hic hhw hni hnm).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.residual M hni
    (B.bodyMapsC M R P hmid hic hhw hni hnm hni hge hgd hgp hA10 hA40 hA70
      eA eE enA enE er dA dE dnA dnE dr pA pE pnA pnE) P.hq rA rE

-- ════════════════════════════════════════════════════════════════
-- § The whole paper net's stored parameters
-- ════════════════════════════════════════════════════════════════

/-- **The paper-spec MobileNetV2's stored parameters at one uniform profile** — stem, the 17
    bottlenecks of the `[t,c,n,s]` table (`b1` the t=1 no-expand form, `b2/b4/b7/b14` the
    stride-2 downsamples, ten identity skips, `b11/b17` the stage-first widenings), the 1×1
    head and the classifier. Fifty-two BatchNorm sites and fifty-two convolutions.

    ⭐ The widths are `paperSig`'s (`MobileNetV2Render.lean`), the committed func-arg signature
    of `@mobilenetv2_fwd_eval`, and the kinds are `mobilenetv2ForwardPaper`'s; the probe asserts
    the two agree, block for block, including that "has an identity skip" is `ic = oc`. -/
structure MnvPaperWeights (w' β' G Bb Mb : ℝ) where
  stem : MnvConv 32 3 3 3 w' β'
  bns : MnvBn 32 G Bb Mb
  b1 : MnvBlockNoExp 32 16 w' β' G Bb Mb
  b2 : MnvBlock 16 96 24 w' β' G Bb Mb
  b3 : MnvBlock 24 144 24 w' β' G Bb Mb
  b4 : MnvBlock 24 144 32 w' β' G Bb Mb
  b5 : MnvBlock 32 192 32 w' β' G Bb Mb
  b6 : MnvBlock 32 192 32 w' β' G Bb Mb
  b7 : MnvBlock 32 192 64 w' β' G Bb Mb
  b8 : MnvBlock 64 384 64 w' β' G Bb Mb
  b9 : MnvBlock 64 384 64 w' β' G Bb Mb
  b10 : MnvBlock 64 384 64 w' β' G Bb Mb
  b11 : MnvBlock 64 384 96 w' β' G Bb Mb
  b12 : MnvBlock 96 576 96 w' β' G Bb Mb
  b13 : MnvBlock 96 576 96 w' β' G Bb Mb
  b14 : MnvBlock 96 576 160 w' β' G Bb Mb
  b15 : MnvBlock 160 960 160 w' β' G Bb Mb
  b16 : MnvBlock 160 960 160 w' β' G Bb Mb
  b17 : MnvBlock 160 960 320 w' β' G Bb Mb
  hd : MnvConv 1280 320 1 1 w' β'
  bnh : MnvBn 1280 G Bb Mb
  head : MnvHead 1280 10 w' β'

-- ════════════════════════════════════════════════════════════════
-- § Every block in the ladder IS the committed inference block
-- ════════════════════════════════════════════════════════════════

/-! ⭐ Four `rfl`s, dimension-polymorphic, so each holds at all of the widths the paper table
uses. `invresBodyPCEval` / `ivProjectPCEval` / `ivDepthwisePCEval` / `ivDepthwiseStridedPCEval`
(`MobileNetV2RenderPCEval.lean`) are the abbreviations the committed inference forward and its
faithful graph are built from; these say the blocks the number below folds over are those, at
the record's projections and at one shared `ε`. What is NOT closed here is the whole-net step —
the paper net's eval forward has no typed graph in Lean (the header says so). -/

/-- The stride-1 no-skip block is `invresBodyPCEval`. -/
theorem MnvBlock.bodyFwd_eq_pcEval {ic mid oc h w : Nat} (B : MnvBlock ic mid oc w' β' G Bb Mb)
    (ε : ℝ) :
    B.bodyFwd (h := h) (w := w) ε
      = invresBodyPCEval (h := h) (w := w) ε B.ex.W B.ex.b B.bne.γ B.bne.β B.bne.μ B.bne.v
          B.dw.W B.dw.b B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v
          B.pr.W B.pr.b B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v := rfl

/-- The stride-2 downsampling block is `invresBodyStridedPCEval` — expand at `2h×2w`, the
    XLA-`SAME` decimating depthwise, project. -/
theorem MnvBlock.stridedFwd_eq_pcEval {ic mid oc h w : Nat}
    (B : MnvBlock ic mid oc w' β' G Bb Mb) (ε : ℝ) :
    B.stridedFwd (h := h) (w := w) ε
      = invresBodyStridedPCEval (h := h) (w := w) ε B.ex.W B.ex.b B.bne.γ B.bne.β B.bne.μ B.bne.v
          B.dw.W B.dw.b B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v
          B.pr.W B.pr.b B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v := rfl

/-- The matched-channel block is that body under the identity skip. -/
theorem MnvBlock.resFwd_eq_pcEval {ic mid h w : Nat} (B : MnvBlock ic mid ic w' β' G Bb Mb)
    (ε : ℝ) :
    B.resFwd (h := h) (w := w) ε
      = residual (invresBodyPCEval (h := h) (w := w) ε B.ex.W B.ex.b
          B.bne.γ B.bne.β B.bne.μ B.bne.v B.dw.W B.dw.b B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v
          B.pr.W B.pr.b B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v) := rfl

/-- The t=1 first bottleneck is the project stage after the depthwise stage, with no expand —
    the `[t,c,n,s]` table's `(1, 16, 1, 1)` row, and the one block shape the reduced net has no
    instance of. -/
theorem MnvBlockNoExp.fwd_eq_pcEval {ic oc h w : Nat} (B : MnvBlockNoExp ic oc w' β' G Bb Mb)
    (ε : ℝ) :
    B.fwd (h := h) (w := w) ε
      = ivProjectPCEval (h := h) (w := w) B.pr.W B.pr.b ε B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v ∘
          ivDepthwisePCEval (h := h) (w := w) B.dw.W B.dw.b ε
            B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v := rfl

-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The deployed paper-spec MobileNetV2 inference forward** — the `[t,c,n,s]` ladder of
    `mobilenetv2ForwardPaper` with inference BatchNorm at every one of its 52 sites, written in
    the association the bridge below composes. -/
noncomputable def mnv2PaperEvalForward (W : MnvPaperWeights w' β' G Bb Mb) (ε : ℝ) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense W.head.W W.head.b ∘
  globalAvgPoolFlat 1280 7 7 ∘
  relu6 (1280 * 7 * 7) ∘
  W.bnh.fwd ε 7 7 ∘
  flatConv (h := 7) (w := 7) W.hd.W W.hd.b ∘
  W.b17.bodyFwd (h := 7) (w := 7) ε ∘
  W.b16.resFwd (h := 7) (w := 7) ε ∘
  W.b15.resFwd (h := 7) (w := 7) ε ∘
  W.b14.stridedFwd (h := 7) (w := 7) ε ∘
  W.b13.resFwd (h := 14) (w := 14) ε ∘
  W.b12.resFwd (h := 14) (w := 14) ε ∘
  W.b11.bodyFwd (h := 14) (w := 14) ε ∘
  W.b10.resFwd (h := 14) (w := 14) ε ∘
  W.b9.resFwd (h := 14) (w := 14) ε ∘
  W.b8.resFwd (h := 14) (w := 14) ε ∘
  W.b7.stridedFwd (h := 14) (w := 14) ε ∘
  W.b6.resFwd (h := 28) (w := 28) ε ∘
  W.b5.resFwd (h := 28) (w := 28) ε ∘
  W.b4.stridedFwd (h := 28) (w := 28) ε ∘
  W.b3.resFwd (h := 56) (w := 56) ε ∘
  W.b2.stridedFwd (h := 56) (w := 56) ε ∘
  W.b1.fwd (h := 112) (w := 112) ε ∘
  relu6 (32 * 112 * 112) ∘
  W.bns.fwd ε 112 112 ∘
  flatConvStride2Xla (h := 112) (w := 112) W.stem.W W.stem.b

/-- **The deployed paper-spec MobileNetV2 float inference forward** — every concrete slot
    replaced by the model's rounded peer, every BN by the six rounded ops the emitter writes,
    `relu6` unchanged (clamp-and-select rounds nothing). -/
noncomputable def mnv2PaperEvalForwardF (M : FloatModel) (R : DeviceRsqrt ε es)
    (W : MnvPaperWeights w' β' G Bb Mb) : Vec (3 * 224 * 224) → Vec 10 :=
  M.dense W.head.W W.head.b ∘
  M.gapFlatF ∘
  relu6 (1280 * 7 * 7) ∘
  W.bnh.fwdF M R 7 7 ∘
  M.flatConvF (h := 7) (w := 7) W.hd.W W.hd.b ∘
  W.b17.bodyFwdF (h := 7) (w := 7) M R ∘
  W.b16.resFwdF (h := 7) (w := 7) M R ∘
  W.b15.resFwdF (h := 7) (w := 7) M R ∘
  W.b14.stridedFwdF (h := 7) (w := 7) M R ∘
  W.b13.resFwdF (h := 14) (w := 14) M R ∘
  W.b12.resFwdF (h := 14) (w := 14) M R ∘
  W.b11.bodyFwdF (h := 14) (w := 14) M R ∘
  W.b10.resFwdF (h := 14) (w := 14) M R ∘
  W.b9.resFwdF (h := 14) (w := 14) M R ∘
  W.b8.resFwdF (h := 14) (w := 14) M R ∘
  W.b7.stridedFwdF (h := 14) (w := 14) M R ∘
  W.b6.resFwdF (h := 28) (w := 28) M R ∘
  W.b5.resFwdF (h := 28) (w := 28) M R ∘
  W.b4.stridedFwdF (h := 28) (w := 28) M R ∘
  W.b3.resFwdF (h := 56) (w := 56) M R ∘
  W.b2.stridedFwdF (h := 56) (w := 56) M R ∘
  W.b1.fwdF (h := 112) (w := 112) M R ∘
  relu6 (32 * 112 * 112) ∘
  W.bns.fwdF M R 112 112 ∘
  M.flatConvStride2XlaF (h := 112) (w := 112) W.stem.W W.stem.b

set_option maxRecDepth 1000000 in
/-- ⭐ **The whole deployed paper-spec MobileNetV2 inference forward float-bridges TO its float
    peer** — a CLOSED `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left: the stem
    conv/BN/relu6, the seventeen bottlenecks, the 1×1 head, GAP and the classifier are each
    discharged by a leaf. Every BatchNorm enters under `FloatBridgesTo.capped`, so `.mod` is a
    closed term in which each normalisation contributes `min(fold, 2·mag)`. -/
noncomputable def mnv2PaperEvalBridge (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : MnvProfile M ε w' β' G Bb Mb es S q) (W : MnvPaperWeights w' β' G Bb Mb) :
    FloatBridgesTo (mnv2PaperEvalForward W ε) (mnv2PaperEvalForwardF M R W) :=
  ((((((((((((((((((((((((
    (floatBridgesTo_flatConvStride2Xla (h := 112) (w := 112) M W.stem.W W.stem.b P.hw' P.hβ'
      (by norm_num) W.stem.hW W.stem.hb)
    |>.comp ((W.bns.bridge M R P (by norm_num) (by norm_num) (h := 112) (w := 112)).capped)
    |>.comp (floatBridgesTo_relu6 (n := 32 * 112 * 112))
    |>.comp ((W.b1.bridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 112) (w := 112)))
    |>.comp ((W.b2.stridedBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp ((W.b3.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp ((W.b4.stridedBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp ((W.b5.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp ((W.b6.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp ((W.b7.stridedBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b8.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b9.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b10.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b11.bodyBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b12.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b13.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp ((W.b14.stridedBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp ((W.b15.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp ((W.b16.resBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp ((W.b17.bodyBridgeC M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp ((floatBridgesTo_flatConv (h := 7) (w := 7) M W.hd.W W.hd.b P.hw' P.hβ'
      (by norm_num) W.hd.hW W.hd.hb))
    |>.comp ((W.bnh.bridge M R P (by norm_num) (by norm_num) (h := 7) (w := 7)).capped)
    |>.comp (floatBridgesTo_relu6 (n := 1280 * 7 * 7))
    |>.comp ((floatBridgesTo_gap (c := 1280) (h := 7) (w := 7) M (by norm_num) (by norm_num)))
    |>.comp ((floatBridgesTo_dense M W.head.W W.head.b P.hw' P.hβ' (by norm_num)
      W.head.hW W.head.hb))))))))))))))))))))))))))

-- ════════════════════════════════════════════════════════════════
-- § The number
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
set_option maxHeartbeats 8000000 in
/-- ⭐ **The envelope, kernel-checked, at seventeen blocks.** Every numeric stage closed by two
    rational inequalities — except the 52 BatchNorm sites, whose modulus clause is the cap's
    `2·Ā' ≤ Ē'` and whose fold is never written down. Each γ-term goes through
    `FloatModel.gamma_num` so `norm_num` never evaluates a big power. Built bottom-up at block
    granularity (25 steps rather than ~160 leaf steps); the closing `exact` is one structural
    comparison with `mnv2PaperEvalBridge`'s definition.

    ⭐ Why the numerals repeat down the ladder: at a capped site the output error is `2·window`
    and the incoming error is discarded, and every relu6 resets the window to `6`. So a block's
    stage numerals depend only on its `(ic, mid, oc)` and not on its depth — `b8`, `b9` and `b10`
    are numerically the same block. That is a property of the cap, not of the net. -/
theorem mnv2PaperEvalBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvPaperWeights (28/10) (28/10) (28/10) (28/10) (28/10)) :
    (mnv2PaperEvalBridge M R (mnv2Profile_committed M hMu hε5) W).Maps 1 0
      (2152 * 10 ^ 1) (8176 * 10 ^ 13) := by
  have hP := mnv2Profile_committed M hMu hε5
  have t0 := FloatBridgesTo.Maps.flatConvStride2Xla (h := 112) (w := 112) M W.stem.W W.stem.b
    hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb (M.gamma_num (q := 1729 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 7841 / 10 ^ 2) (Ē' := 1356 / 10 ^ 7) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
  have t1 := t0.comp (by norm_num) (W.bns.cappedMaps M R hP (by norm_num) (by norm_num)
    (h := 112) (w := 112) (Ā' := 7209 * 10 ^ 1) (Ē' := 1442 * 10 ^ 2) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t2 := t1.comp (by norm_num) (FloatBridgesTo.Maps.relu6 (n := 32 * 112 * 112)
    (Ā' := 6) (Ē' := 1442 * 10 ^ 2) (by norm_num) le_rfl)
  have t3 := t2.comp (by norm_num) (W.b1.mapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 112) (w := 112)
    (gd := 6557 / 10 ^ 10) (gp := 2027 / 10 ^ 9)
    (A1 := 1541 / 10 ^ 1) (E1 := 3634 * 10 ^ 3)
    (A2 := 1393 * 10 ^ 2) (E2 := 2786 * 10 ^ 2)
    (A3 := 6)
    (A4 := 5405 / 10 ^ 1) (E4 := 2497 * 10 ^ 4)
    (Ā' := 4823 * 10 ^ 2) (Ē' := 9646 * 10 ^ 2)
    (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t4 := t3.comp (by norm_num) (W.b2.stridedMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56)
    (ge := 1073 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5842 / 10 ^ 9)
    (A1 := 2161 * 10 ^ 4) (E1 := 4322 * 10 ^ 4)
    (A2 := 1919 * 10 ^ 7) (E2 := 3838 * 10 ^ 7)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 9672 * 10 ^ 8)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 1616) (E7 := 7489 * 10 ^ 4)
    (Ā' := 1437 * 10 ^ 3) (Ē' := 2874 * 10 ^ 3)
    (M.gamma_num (q := 1073 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t5 := t4.comp (by norm_num) (W.b3.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56)
    (ge := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 8703 / 10 ^ 9)
    (A1 := 9657 * 10 ^ 4) (E1 := 1932 * 10 ^ 5)
    (A2 := 8572 * 10 ^ 7) (E2 := 1715 * 10 ^ 8)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 4322 * 10 ^ 9)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 2423) (E7 := 1124 * 10 ^ 5)
    (Bd := 2154 * 10 ^ 3) (Ed := 4308 * 10 ^ 3)
    (Ā' := 3592 * 10 ^ 3) (Ē' := 7183 * 10 ^ 3)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t6 := t5.comp (by norm_num) (W.b4.stridedMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28)
    (ge := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 8703 / 10 ^ 9)
    (A1 := 2414 * 10 ^ 5) (E1 := 4827 * 10 ^ 5)
    (A2 := 2143 * 10 ^ 8) (E2 := 4286 * 10 ^ 8)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 1081 * 10 ^ 10)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 2423) (E7 := 1124 * 10 ^ 5)
    (Ā' := 2154 * 10 ^ 3) (Ē' := 4308 * 10 ^ 3)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t7 := t6.comp (by norm_num) (W.b5.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28)
    (ge := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 1157 / 10 ^ 8)
    (A1 := 1930 * 10 ^ 5) (E1 := 3860 * 10 ^ 5)
    (A2 := 1714 * 10 ^ 8) (E2 := 3428 * 10 ^ 8)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 8639 * 10 ^ 9)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 3229) (E7 := 1498 * 10 ^ 5)
    (Bd := 2869 * 10 ^ 3) (Ed := 5738 * 10 ^ 3)
    (Ā' := 5024 * 10 ^ 3) (Ē' := 1005 * 10 ^ 4)
    (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t8 := t7.comp (by norm_num) (W.b6.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28)
    (ge := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 1157 / 10 ^ 8)
    (A1 := 4502 * 10 ^ 5) (E1 := 9005 * 10 ^ 5)
    (A2 := 3997 * 10 ^ 8) (E2 := 7994 * 10 ^ 8)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 2015 * 10 ^ 10)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 3229) (E7 := 1498 * 10 ^ 5)
    (Bd := 2869 * 10 ^ 3) (Ed := 5738 * 10 ^ 3)
    (Ā' := 7894 * 10 ^ 3) (Ē' := 1579 * 10 ^ 4)
    (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t9 := t8.comp (by norm_num) (W.b7.stridedMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 1157 / 10 ^ 8)
    (A1 := 7074 * 10 ^ 5) (E1 := 1415 * 10 ^ 6)
    (A2 := 6280 * 10 ^ 8) (E2 := 1256 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 3166 * 10 ^ 10)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 3229) (E7 := 1498 * 10 ^ 5)
    (Ā' := 2869 * 10 ^ 3) (Ē' := 5738 * 10 ^ 3)
    (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t10 := t9.comp (by norm_num) (W.b8.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 2301 / 10 ^ 8)
    (A1 := 5142 * 10 ^ 5) (E1 := 1029 * 10 ^ 6)
    (A2 := 4565 * 10 ^ 8) (E2 := 9130 * 10 ^ 8)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 2301 * 10 ^ 10)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 6455) (E7 := 2996 * 10 ^ 5)
    (Bd := 5733 * 10 ^ 3) (Ed := 1147 * 10 ^ 4)
    (Ā' := 8603 * 10 ^ 3) (Ē' := 1721 * 10 ^ 4)
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t11 := t10.comp (by norm_num) (W.b9.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 2301 / 10 ^ 8)
    (A1 := 1542 * 10 ^ 6) (E1 := 3085 * 10 ^ 6)
    (A2 := 1369 * 10 ^ 9) (E2 := 2738 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 6900 * 10 ^ 10)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 6455) (E7 := 2996 * 10 ^ 5)
    (Bd := 5733 * 10 ^ 3) (Ed := 1147 * 10 ^ 4)
    (Ā' := 1434 * 10 ^ 4) (Ē' := 2869 * 10 ^ 4)
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t12 := t11.comp (by norm_num) (W.b10.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 2301 / 10 ^ 8)
    (A1 := 2570 * 10 ^ 6) (E1 := 5142 * 10 ^ 6)
    (A2 := 2282 * 10 ^ 9) (E2 := 4564 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 1151 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 6455) (E7 := 2996 * 10 ^ 5)
    (Bd := 5733 * 10 ^ 3) (Ed := 1147 * 10 ^ 4)
    (Ā' := 2008 * 10 ^ 4) (Ē' := 4017 * 10 ^ 4)
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t13 := t12.comp (by norm_num) (W.b11.bodyMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 2301 / 10 ^ 8)
    (A1 := 3599 * 10 ^ 6) (E1 := 7199 * 10 ^ 6)
    (A2 := 3195 * 10 ^ 9) (E2 := 6390 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 1611 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 6455) (E7 := 2996 * 10 ^ 5)
    (Ā' := 5733 * 10 ^ 3) (Ē' := 1147 * 10 ^ 4)
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t14 := t13.comp (by norm_num) (W.b12.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 5842 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 3446 / 10 ^ 8)
    (A1 := 1542 * 10 ^ 6) (E1 := 3084 * 10 ^ 6)
    (A2 := 1369 * 10 ^ 9) (E2 := 2738 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 6900 * 10 ^ 10)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 9680) (E7 := 4494 * 10 ^ 5)
    (Bd := 8595 * 10 ^ 3) (Ed := 1719 * 10 ^ 4)
    (Ā' := 1433 * 10 ^ 4) (Ē' := 2867 * 10 ^ 4)
    (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t15 := t14.comp (by norm_num) (W.b13.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14)
    (ge := 5842 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 3446 / 10 ^ 8)
    (A1 := 3852 * 10 ^ 6) (E1 := 7707 * 10 ^ 6)
    (A2 := 3420 * 10 ^ 9) (E2 := 6840 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 1724 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 9680) (E7 := 4494 * 10 ^ 5)
    (Bd := 8595 * 10 ^ 3) (Ed := 1719 * 10 ^ 4)
    (Ā' := 2293 * 10 ^ 4) (Ē' := 4587 * 10 ^ 4)
    (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t16 := t15.comp (by norm_num) (W.b14.stridedMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7)
    (ge := 5842 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 3446 / 10 ^ 8)
    (A1 := 6164 * 10 ^ 6) (E1 := 1233 * 10 ^ 7)
    (A2 := 5472 * 10 ^ 9) (E2 := 1095 * 10 ^ 10)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 2760 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 9680) (E7 := 4494 * 10 ^ 5)
    (Ā' := 8595 * 10 ^ 3) (Ē' := 1719 * 10 ^ 4)
    (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t17 := t16.comp (by norm_num) (W.b15.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7)
    (ge := 9657 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5735 / 10 ^ 8)
    (A1 := 3851 * 10 ^ 6) (E1 := 7702 * 10 ^ 6)
    (A2 := 3419 * 10 ^ 9) (E2 := 6838 * 10 ^ 9)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 1724 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 1614 * 10 ^ 1) (E7 := 7490 * 10 ^ 5)
    (Bd := 1433 * 10 ^ 4) (Ed := 2866 * 10 ^ 4)
    (Ā' := 2293 * 10 ^ 4) (Ē' := 4586 * 10 ^ 4)
    (M.gamma_num (q := 9657 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5735 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t18 := t17.comp (by norm_num) (W.b16.resMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7)
    (ge := 9657 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5735 / 10 ^ 8)
    (A1 := 1028 * 10 ^ 7) (E1 := 2055 * 10 ^ 7)
    (A2 := 9125 * 10 ^ 9) (E2 := 1825 * 10 ^ 10)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 4600 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 1614 * 10 ^ 1) (E7 := 7490 * 10 ^ 5)
    (Bd := 1433 * 10 ^ 4) (Ed := 2866 * 10 ^ 4)
    (Ā' := 3727 * 10 ^ 4) (Ē' := 7453 * 10 ^ 4)
    (M.gamma_num (q := 9657 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5735 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t19 := t18.comp (by norm_num) (W.b17.bodyMapsC M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7)
    (ge := 9657 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5735 / 10 ^ 8)
    (A1 := 1670 * 10 ^ 7) (E1 := 3339 * 10 ^ 7)
    (A2 := 1483 * 10 ^ 10) (E2 := 2966 * 10 ^ 10)
    (A3 := 6)
    (A4 := 1541 / 10 ^ 1) (E4 := 7475 * 10 ^ 11)
    (A5 := 1393 * 10 ^ 2) (E5 := 2786 * 10 ^ 2)
    (A6 := 6)
    (A7 := 1614 * 10 ^ 1) (E7 := 7490 * 10 ^ 5)
    (Ā' := 1433 * 10 ^ 4) (Ē' := 2866 * 10 ^ 4)
    (M.gamma_num (q := 9657 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5735 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t20 := t19.comp (by norm_num) (FloatBridgesTo.Maps.flatConv (h := 7) (w := 7) M
    W.hd.W W.hd.b hP.hw' hP.hβ' (by norm_num) W.hd.hW W.hd.hb (M.gamma_num (q := 1920 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 1284 * 10 ^ 7) (Ē' := 2569 * 10 ^ 7) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t21 := t20.comp (by norm_num) (W.bnh.cappedMaps M R hP (by norm_num) (by norm_num)
    (h := 7) (w := 7) (Ā' := 1140 * 10 ^ 10) (Ē' := 2280 * 10 ^ 10) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t22 := t21.comp (by norm_num) (FloatBridgesTo.Maps.relu6 (n := 1280 * 7 * 7)
    (Ā' := 6) (Ē' := 2280 * 10 ^ 10) (by norm_num) le_rfl)
  have t23 := t22.comp (by norm_num) (FloatBridgesTo.Maps.gap (c := 1280) (h := 7) (w := 7) M
    (by norm_num) (by norm_num) hMu (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 6001 / 10 ^ 3) (Ē' := 2281 * 10 ^ 10) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t24 := t23.comp (by norm_num) (FloatBridgesTo.Maps.dense M W.head.W W.head.b
    hP.hw' hP.hβ' (by norm_num) W.head.hW W.head.hb (M.gamma_num (q := 7642 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā' := 2152 * 10 ^ 1) (Ē' := 8176 * 10 ^ 13) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  exact t24

/-- The paper-net inference bridge's certified output window at the committed profile:
    `≤ 2.152·10⁴`. ⭐ Eleven more blocks than the six-block net cost the window nothing — the
    entire growth from `2154` is the classifier's 1280-wide fan-in, because `relu6`'s clamp
    resets the certified magnitude at all 35 activation sites. -/
theorem mnv2PaperEvalBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvPaperWeights (28/10) (28/10) (28/10) (28/10) (28/10)) :
    (mnv2PaperEvalBridge M R (mnv2Profile_committed M hMu hε5) W).mag 1 ≤ 2152 * 10 ^ 1 :=
  (mnv2PaperEvalBridge_maps M hMu hε5 R W).mag_le 1 (by norm_num) le_rfl

/-- The paper-net inference bridge's fresh budget at the committed profile: `≤ 8.176·10¹⁶`.
    ⛔ CAPPED at all 52 BatchNorm sites — see the file header for what that claim is and is
    not. It is 79 orders below the six-block net's uncapped `1.444·10⁹⁶`. -/
theorem mnv2PaperEvalBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvPaperWeights (28/10) (28/10) (28/10) (28/10) (28/10)) :
    (mnv2PaperEvalBridge M R (mnv2Profile_committed M hMu hε5) W).fresh 1 ≤ 8176 * 10 ^ 13 :=
  (mnv2PaperEvalBridge_maps M hMu hε5 R W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed seventeen-block MobileNetV2 inference forward is within `8.176·10¹⁶` of
    the certified real forward, per logit**, on inputs of magnitude `≤ 1`, at the measured
    parameter profile (`|·| ≤ 28/10`), for `ε ≥ 10⁻⁵`, any device `rsqrt` accurate to `10⁻²`,
    and any rounding model at binary32 accuracy. The paper-spec peer of `mnv2_float_logits_le`,
    and the first whole-net number in the repo stated at a net's real depth AND real widths with
    the cap doing the work at every normalisation. ⛔ CAP: see the header. -/
theorem mnv2Paper_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceRsqrt ε (1/100))
    (W : MnvPaperWeights (28/10) (28/10) (28/10) (28/10) (28/10))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |mnv2PaperEvalForwardF M R W x j - mnv2PaperEvalForward W ε x j| ≤ 8176 * 10 ^ 13 :=
  (mnv2PaperEvalBridge_maps M hMu hε5 R W).budget_le (by norm_num) le_rfl x hx j

/-! ### Inhabitation

`mnv2Paper_float_logits_le`'s record at the committed constants: zero weights, zero running
statistics, the exact `rsqrt`, `binary32`, `ε = 1/100000`. -/
noncomputable def MnvBlockNoExp.zero (ic oc : Nat) {w' β' G Bb Mb : ℝ} (hw : 0 ≤ w')
    (hb : 0 ≤ β') (hG : 0 ≤ G) (hBb : 0 ≤ Bb) (hMb : 0 ≤ Mb) :
    MnvBlockNoExp ic oc w' β' G Bb Mb where
  dw := MnvDw.zero _ _ _ hw hb
  bnd := MnvBn.zero _ hG hBb hMb
  pr := MnvConv.zero _ _ _ _ hw hb
  bnp := MnvBn.zero _ hG hBb hMb

noncomputable def MnvPaperWeights.zero :
    MnvPaperWeights (28/10) (28/10) (28/10) (28/10) (28/10) :=
  have h : (0:ℝ) ≤ 28/10 := by norm_num
  { stem := MnvConv.zero _ _ _ _ h h,
    bns := MnvBn.zero _ h h h,
    b1 := MnvBlockNoExp.zero _ _ h h h h h,
    b2 := MnvBlock.zero _ _ _ h h h h h,
    b3 := MnvBlock.zero _ _ _ h h h h h,
    b4 := MnvBlock.zero _ _ _ h h h h h,
    b5 := MnvBlock.zero _ _ _ h h h h h,
    b6 := MnvBlock.zero _ _ _ h h h h h,
    b7 := MnvBlock.zero _ _ _ h h h h h,
    b8 := MnvBlock.zero _ _ _ h h h h h,
    b9 := MnvBlock.zero _ _ _ h h h h h,
    b10 := MnvBlock.zero _ _ _ h h h h h,
    b11 := MnvBlock.zero _ _ _ h h h h h,
    b12 := MnvBlock.zero _ _ _ h h h h h,
    b13 := MnvBlock.zero _ _ _ h h h h h,
    b14 := MnvBlock.zero _ _ _ h h h h h,
    b15 := MnvBlock.zero _ _ _ h h h h h,
    b16 := MnvBlock.zero _ _ _ h h h h h,
    b17 := MnvBlock.zero _ _ _ h h h h h,
    hd := MnvConv.zero _ _ _ _ h h,
    bnh := MnvBn.zero _ h h h,
    head := MnvHead.zero _ _ h h }

example (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |mnv2PaperEvalForwardF binary32 (DeviceRsqrt.exact (1/100000) (es := 1/100) (by norm_num))
        MnvPaperWeights.zero x j
      - mnv2PaperEvalForward MnvPaperWeights.zero (1/100000) x j| ≤ 8176 * 10 ^ 13 :=
  mnv2Paper_float_logits_le binary32 binary32_u.le (by norm_num) _ _ x hx j

end Proofs
