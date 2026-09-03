import LeanMlir.Proofs.Float.Resnet34FloatBudget
import LeanMlir.Proofs.Float.FloatBudgetEnvMBConv

/-! # A NUMBER for EfficientNet-B0: the deployed inference forward, and the two bounds that
    had to be tightened first

The third ImageNet-scale whole-net float budget, after ResNet-34 and MobileNetV2. For the
representative batched B0 — 3×3/s2 stem, MBConv1 (no expand, SE), MBConv6 (stride-2 3×3, SE),
MBConv6 (5×5, SE, residual), 1×1 head, GAP, dense — with **inference** BatchNorm, on the unit
input window, at the profile measured on the 350-epoch checkpoint (`|parameter| ≤ 41/10`), for
any rounding model at binary32 accuracy:

    output window  ≤ 2.580·10⁵⁵      (`b0EvalBridge_mag_le`)
    fresh budget   ≤ 8.408·10²¹⁰     (`b0EvalBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 8.408·10²¹⁰` (`b0_float_logits_le`).

⛔⛔ **This number did not exist two days ago, and the reason was not the architecture.** The
first honest fold came out at window `10¹⁷¹⁷` and budget `10⁴¹⁷⁹` — `norm_num` refuses numerals
past ~`10³⁰⁰`, so there was nothing to state. Two LEAF bounds did it, and both were the relu6
pattern: a bound proved one lemma down and discarded by the generic combinator
(`planning/float_budget_numbers.md` §3.4).

* `floatClose_swish`'s modulus was `mulErr + (1 + A/4)·e`, multiplying the inherited error by the
  WINDOW at each of this net's nine swish sites. Bounding `|σa − σb|` by the gate's own range
  (`≤ 1`) instead of by `¼|a−b|` gives `A + |a−b|`, ADDITIVE in the window
  (`swishScalar_lipschitz_abs'`); the modulus is now the `min` of the two. Worth 10¹⁵⁵¹.
* `floatClose_seScale`'s window was derived as `|float − real| + |real|`, charging `A · Lg 0` —
  the block window times the GATE'S error — to the magnitude. `FloatClose`'s magnitude clause
  bounds the float gate too, so the window is `A·Bg·(1+u)`. Worth 10¹⁸ per SE site.

⚠ **What squeeze-excite still costs, and it is §0.1's shape.** Each SE site roughly DOUBLES the
budget's exponent (`10²⁵ → 10⁸³ → 10¹⁹⁹` across b1/b2/b3). `seScale`'s modulus carries `A · Eg`,
the block window times the gate error, and the gate grows that error out of the same window
through the squeeze's `GAP → dense`: **SE is quadratic in the window**, for the same structural
reason training-mode BatchNorm and LayerNorm are — an op that consumes a reduction of its own
input and then multiplies by that input. B0 survives three such sites; twenty would end it the
way r34's 33 training-BN sites end its training-mode fold. That is the rule to carry to a new
architecture, not "does it normalise".

⭐ **Why the batch is not a problem here, though `efficientnetForwardB` couples it.**
`bnBatchLA` is the one op in this net that is not `batchMap N` of a per-example op — it reduces
μ/var across examples. At frozen statistics there is no reduction, so every stage of
`efficientnetForwardBEval` is `batchMap`-of-a-per-example-op or pointwise, and
`FloatBridgesTo.Maps.batchMap` carries an envelope through the lift unchanged. So the whole
batched net folds through per-example leaves the repo already had.

⚠ **The two hypotheses this number rests on, named.** The deployed inverse-stddev is a device
`rsqrt` and the deployed gate a device `sigmoid`; neither has an IEEE specification, so both are
*modelled* — `DeviceRsqrt ε es` (shared with `Resnet34FloatBudget.lean`; the same kernel, not a
second assumption) and `DeviceSigmoid esig`. Everything else is proved.

**The tie is closed at the graph.** `b0EvalForward_eq_forwardBEval` is a `rfl` onto
`efficientnetForwardBEval` (`EfficientNetRenderPCEval.lean`) and `b0EvalGraph_faithful` carries
`efficientnetFwdGraphBEval_faithful` the rest of the way. `b0_float_logits_le_committed` states
the number with that net on the real side.

Provenance for the numerals: `scripts/float_budget_envelope.py`'s `b0_eval_chain`, whose
`verify_b0` re-asserts all 96 rounded inequalities before emitting.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The modelled device sigmoid, and the parameter records
-- ════════════════════════════════════════════════════════════════

/-- **The deployed gate.** EfficientNet's swish and squeeze-excite both evaluate a sigmoid, which
    the emitter lowers to a device kernel with no IEEE specification — so, exactly as
    `DeviceRsqrt` supplies the inverse-stddev, this supplies it with an accuracy rather than
    deriving one. `esig` is its absolute accuracy at every real argument. -/
structure DeviceSigmoid (esig : ℝ) where
  /-- The device kernel. -/
  sig : ℝ → ℝ
  /-- Its accuracy against the true logistic. -/
  spec : ∀ t, |sig t - sigmoidScalar t| ≤ esig

/-- A convolution's stored parameters with their magnitude bounds. -/
structure EnetConv (oc ic kH kW : Nat) (w' β' : ℝ) where
  W : Kernel4 oc ic kH kW
  b : Vec oc
  hW : ∀ o c kh kw, |W o c kh kw| ≤ w'
  hb : ∀ o, |b o| ≤ β'

/-- A depthwise convolution's stored parameters with their magnitude bounds. -/
structure EnetDw (c kH kW : Nat) (w' β' : ℝ) where
  W : DepthwiseKernel c kH kW
  b : Vec c
  hW : ∀ ch kh kw, |W ch kh kw| ≤ w'
  hb : ∀ ch, |b ch| ≤ β'

/-- A squeeze-excite gate's two dense layers, `c → r → c`. -/
structure EnetSE (c r : Nat) (w' β' : ℝ) where
  W₁ : Mat c r
  b₁ : Vec r
  W₂ : Mat r c
  b₂ : Vec c
  hW₁ : ∀ i j, |W₁ i j| ≤ w'
  hb₁ : ∀ j, |b₁ j| ≤ β'
  hW₂ : ∀ i j, |W₂ i j| ≤ w'
  hb₂ : ∀ j, |b₂ j| ≤ β'

/-- The classifier's stored parameters with their magnitude bounds. -/
structure EnetHead (m n : Nat) (w' β' : ℝ) where
  W : Mat m n
  b : Vec n
  hW : ∀ i j, |W i j| ≤ w'
  hb : ∀ j, |b j| ≤ β'

/-- One inference-BN site: scale, shift, and the two FROZEN running statistics. -/
structure EnetBn (c : Nat) (G Bb Mb : ℝ) where
  γ : Vec c
  β : Vec c
  μ : Vec c
  v : Vec c
  hγ : ∀ o, |γ o| ≤ G
  hβ : ∀ o, |β o| ≤ Bb
  hμ : ∀ o, |μ o| ≤ Mb
  /-- Running variances are nonnegative — what puts the inverse-stddev under the `ε`-floor. -/
  hv : ∀ o, 0 ≤ v o

/-- An MBConv1 block (`t = 1`, no expand): depthwise, its BN, the SE gate, project, its BN. -/
structure EnetNoExpBlk (ic oc r kHd kWd : Nat) (w' β' G Bb Mb : ℝ) where
  dw : EnetDw ic kHd kWd w' β'
  bnd : EnetBn ic G Bb Mb
  se : EnetSE ic r w' β'
  pr : EnetConv oc ic 1 1 w' β'
  bnp : EnetBn oc G Bb Mb

/-- An MBConv6 block: expand, its BN, depthwise, its BN, the SE gate, project, its BN. -/
structure EnetMBBlk (ic mid oc r kHd kWd : Nat) (w' β' G Bb Mb : ℝ) where
  ex : EnetConv mid ic 1 1 w' β'
  bne : EnetBn mid G Bb Mb
  dw : EnetDw mid kHd kWd w' β'
  bnd : EnetBn mid G Bb Mb
  se : EnetSE mid r w' β'
  pr : EnetConv oc mid 1 1 w' β'
  bnp : EnetBn oc G Bb Mb

/-- **The whole net's stored parameters at one uniform profile** — the stem, the three MBConv
    blocks, the 1×1 head and the classifier. Ten BN sites, three SE gates. -/
structure EnetWeights (w' β' G Bb Mb : ℝ) where
  stem : EnetConv 32 3 3 3 w' β'
  bns : EnetBn 32 G Bb Mb
  b1 : EnetNoExpBlk 32 16 8 3 3 w' β' G Bb Mb
  b2 : EnetMBBlk 16 96 24 4 3 3 w' β' G Bb Mb
  b3 : EnetMBBlk 24 144 24 6 5 5 w' β' G Bb Mb
  hd : EnetConv 1280 24 1 1 w' β'
  bnh : EnetBn 1280 G Bb Mb
  head : EnetHead 1280 10 w' β'

/-- The numeric profile the fold runs at — `R34Profile`'s peer plus `esig`, the deployed gate's
    accuracy (EfficientNet is the first net in the sweep whose forward evaluates a sigmoid). -/
structure EnetProfile (M : FloatModel) (ε w' β' G Bb Mb es esig S q : ℝ) : Prop where
  hw' : 0 ≤ w'
  hβ' : 0 ≤ β'
  hG : 0 ≤ G
  hBb : 0 ≤ Bb
  hMb : 0 ≤ Mb
  hes : 0 ≤ es
  hesig : 0 ≤ esig
  hS0 : 0 ≤ S
  hε : 0 < ε
  hSε : 1 / Real.sqrt ε ≤ S
  hq : M.u ≤ q

variable {M : FloatModel} {ε w' β' G Bb Mb es esig S q : ℝ}

-- ════════════════════════════════════════════════════════════════
-- § One BN site at the batched index: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

/-- The certified ℝ inference BN at this site — `batchMap N` of the per-example eval BN, which
    is what `den_batchOp_bnEval` proves the emitted `bnEval` descriptor denotes. -/
noncomputable def EnetBn.fwd {c : Nat} (B : EnetBn c G Bb Mb) (N : Nat) (ε : ℝ) (h w : Nat) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  StableHLO.batchMap N (bnPerChannelEvalTensor3 c h w ε B.γ B.β B.μ B.v)

/-- The deployed float inference BN at this site. -/
noncomputable def EnetBn.fwdF {c : Nat} (B : EnetBn c G Bb Mb) (N : Nat) (M : FloatModel)
    (Rq : DeviceRsqrt ε es) (h w : Nat) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  StableHLO.batchMap N (bnPerChannelEvalTensor3FV M B.γ B.β B.μ (fun o => Rq.rsq (B.v o + ε)))

/-- This BN site's bridge. -/
noncomputable def EnetBn.bridge {c h w : Nat} (B : EnetBn c G Bb Mb) (N : Nat) (M : FloatModel)
    (Rq : DeviceRsqrt ε es) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hc : 0 < c) (hhw : 0 < h * w) :
    FloatBridgesTo (B.fwd N ε h w) (B.fwdF N M Rq h w) :=
  FloatBridgesTo.batchMap N
    (floatBridgesTo_bnPerChannelEvalTensor3 (h := h) (w := w) M B.γ B.β B.μ B.v
      (fun o => Rq.rsq (B.v o + ε)) hc hhw P.hε B.hv B.hγ B.hβ B.hμ
      (fun o => Rq.spec (B.v o) (B.hv o)) P.hSε)

/-- This BN site's numeric envelope — two linear inequalities, unchanged by the batch lift. -/
theorem EnetBn.maps {c h w : Nat} (B : EnetBn c G Bb Mb) (N : Nat) (M : FloatModel)
    (Rq : DeviceRsqrt ε es) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hc : 0 < c) (hhw : 0 < h * w) {Ā Ē Ā' Ē' : ℝ} (hĀ0 : 0 ≤ Ā)
    (hĀ' : G * ((Ā + Mb) * S) + Bb + bnNormBudget q (Ā + Mb) S G Bb 0 es ≤ Ā')
    (hĒ' : bnNormBudget q (Ā + Mb) S G Bb 0 es + G * S * Ē ≤ Ē') :
    (B.bridge N M Rq P hc hhw (h := h) (w := w)).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.bnEvalPC (h := h) (w := w) M B.γ B.β B.μ B.v
      (fun o => Rq.rsq (B.v o + ε)) hc hhw P.hε B.hv B.hγ B.hβ B.hμ
      (fun o => Rq.spec (B.v o) (B.hv o)) P.hSε P.hq P.hG P.hBb P.hS0 P.hMb P.hes hĀ0 hĀ' hĒ')

-- ════════════════════════════════════════════════════════════════
-- § One squeeze-excite site
-- ════════════════════════════════════════════════════════════════

/-- This SE gate's bridge — `batchMap N` of the per-example `seBlockFull`. -/
noncomputable def EnetSE.bridge {c h w r : Nat} (Z : EnetSE c r w' β') (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w) :
    FloatBridgesTo (seB N (h := h) (w := w) Z.W₁ Z.b₁ Z.W₂ Z.b₂)
      (seBF N (h := h) (w := w) M D.sig Z.W₁ Z.b₁ Z.W₂ Z.b₂) :=
  floatBridgesTo_seB N M D.sig Z.W₁ Z.b₁ Z.W₂ Z.b₂ P.hw' P.hβ' P.hesig hhw hc hr hn
    D.spec Z.hW₁ Z.hb₁ Z.hW₂ Z.hb₂

/-- ⭐ **This SE gate's numeric envelope** — five numeric stages down the squeeze
    (`GAP → dense → swish → dense → sigmoid`; the broadcast carries the envelope through
    unchanged), then the rescale `x ⊙ gate(x)`.

    ⚠ The `sigmoid` step is what stops the SE branch's WINDOW blowing up — `1 + esig` at any
    input magnitude — and the rescale's window is that times the block's, plus one rounding. The
    BUDGET is the other story: the rescale's `mulErr q Ā Cg Ē Eg` carries `Ā · Eg`, the block
    window times the gate error, and `Eg` came out of that same window through the squeeze. -/
theorem EnetSE.maps {c h w r : Nat} (Z : EnetSE c r w' β') (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w)
    {gG g1 g2 Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 Cg Eg Ā' Ē' : ℝ}
    (hgG0 : 0 ≤ gG) (hgG : (1 + M.u) ^ (h * w + 1) - 1 ≤ gG)
    (hg1 : (1 + M.u) ^ (c + 2) - 1 ≤ g1) (hg2 : (1 + M.u) ^ (r + 2) - 1 ≤ g2)
    (hĀ0 : 0 ≤ Ā) (hA20 : 0 ≤ A2)
    (qA : Ā * ((1 + gG) * (1 + q)) ≤ A1) (qE : Ā * (q * (1 + gG) + gG) + Ē ≤ E1)
    (d1A : (1 + g1) * ((c : ℝ) * w' * A1 + β') ≤ A2)
    (d1E : g1 * ((c : ℝ) * w' * (A1 + E1) + β') + (c : ℝ) * w' * E1 ≤ E2)
    (swA : A2 + FloatModel.mulErr q A2 1 0 esig ≤ A3)
    (swE : FloatModel.mulErr q A2 1 0 esig + min ((1 + A2/4) * E2) (A2 + E2) ≤ E3)
    (d2A : (1 + g2) * ((r : ℝ) * w' * A3 + β') ≤ A4)
    (d2E : g2 * ((r : ℝ) * w' * (A3 + E3) + β') + (r : ℝ) * w' * E3 ≤ E4)
    (sgA : 1 + esig ≤ Cg) (sgE : esig + (1/4) * E4 ≤ Eg)
    (scA : Ā * Cg + q * (Ā * Cg) ≤ Ā') (scE : FloatModel.mulErr q Ā Cg Ē Eg ≤ Ē') :
    (Z.bridge N M D P hhw hc hr hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.gap (c := c) (h := h) (w := w) M hc hhw
    P.hq (M.u_nonneg.trans P.hq) hgG0 hgG qA qE
  have s2 := s1.comp hc (FloatBridgesTo.Maps.dense M Z.W₁ Z.b₁ P.hw' P.hβ' hc
    Z.hW₁ Z.hb₁ hg1 d1A d1E)
  have s3 := s2.comp hr (FloatBridgesTo.Maps.swish (n := r) M D.sig P.hesig D.spec
    P.hq hA20 swA swE)
  have s4 := s3.comp hr (FloatBridgesTo.Maps.dense M Z.W₂ Z.b₂ P.hw' P.hβ' hr
    Z.hW₂ Z.hb₂ hg2 d2A d2E)
  have s5 := s4.comp hc (FloatBridgesTo.Maps.sigmoid (n := c) D.sig P.hesig D.spec sgA sgE)
  have s6 := s5.comp hc (FloatBridgesTo.Maps.broadcast (c := c) (h := h) (w := w))
  exact FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.seScale M hn s6 P.hq hĀ0 scA scE)

-- ════════════════════════════════════════════════════════════════
-- § The three block shapes: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

/-- The certified ℝ MBConv1 block at inference. -/
noncomputable def EnetNoExpBlk.fwd {ic oc r kHd kWd h w : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  mbNoExpFwdBGen N (h := h) (w := w) B.dw.W B.dw.b (B.bnd.fwd N ε h w)
    B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b (B.bnp.fwd N ε h w)

/-- The deployed float MBConv1 block. -/
noncomputable def EnetNoExpBlk.fwdF {ic oc r kHd kWd h w : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  mbNoExpFwdBF N (h := h) (w := w) M D.sig B.dw.W B.dw.b B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂
    B.pr.W B.pr.b (B.bnd.fwdF N M Rq h w) (B.bnp.fwdF N M Rq h w)

/-- This block's bridge. -/
noncomputable def EnetNoExpBlk.bridge {ic oc r kHd kWd h w : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hic : 0 < ic) (hoc : 0 < oc) (hr : 0 < r) (hn : 0 < ic * h * w) :
    FloatBridgesTo (B.fwd (h := h) (w := w) N ε) (B.fwdF (h := h) (w := w) N M D Rq) :=
  floatBridgesTo_mbNoExpFwdBGen N M D.sig B.dw.W B.dw.b (B.bnd.fwd N ε h w)
    B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b (B.bnp.fwd N ε h w)
    (B.bnd.fwdF N M Rq h w) (B.bnp.fwdF N M Rq h w)
    P.hw' P.hβ' P.hesig hhw hic hr hn D.spec
    B.dw.hW B.dw.hb B.se.hW₁ B.se.hb₁ B.se.hW₂ B.se.hb₂ B.pr.hW B.pr.hb
    (B.bnd.bridge N M Rq P hic hhw) (B.bnp.bridge N M Rq P hoc hhw)

/-- **This block's numeric envelope.** Five numeric stages — depthwise, its BN, swish, the SE
    site (supplied as an envelope; `EnetSE.maps` discharges it), project, its BN. The project
    stage is grouped as one `.comp` unit, as `floatBridgesTo_mbNoExpFwdBGen` composes it. -/
theorem EnetNoExpBlk.maps {ic oc r kHd kWd h w : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hic : 0 < ic) (hoc : 0 < oc) (hr : 0 < r) (hn : 0 < ic * h * w)
    (hnb : 0 < N * (ic * h * w)) (hnbo : 0 < N * (oc * h * w))
    {gd gp Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 Ā' Ē' : ℝ}
    (hgd : (1 + M.u) ^ (kHd * kWd + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA20 : 0 ≤ A2) (hA50 : 0 ≤ A5)
    (dA : (1 + gd) * (((kHd * kWd : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (dE : gd * (((kHd * kWd : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((kHd * kWd : ℕ) : ℝ) * w' * Ē ≤ E1)
    (dnA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (dnE : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (swA : A2 + FloatModel.mulErr q A2 1 0 esig ≤ A3)
    (swE : FloatModel.mulErr q A2 1 0 esig + min ((1 + A2/4) * E2) (A2 + E2) ≤ E3)
    (hse : (B.se.bridge N M D P hhw hic hr hn (h := h) (w := w)).Maps A3 E3 A4 E4)
    (pA : (1 + gp) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * A4 + β') ≤ A5)
    (pE : gp * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (A4 + E4) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * E4 ≤ E5)
    (pnA : G * ((A5 + Mb) * S) + Bb + bnNormBudget q (A5 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A5 + Mb) S G Bb 0 es + G * S * E5 ≤ Ē') :
    (B.bridge N M D Rq P hhw hic hoc hr hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' := by
  have a1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.depthwise (h := h) (w := w) M
    B.dw.W B.dw.b P.hw' P.hβ' hn B.dw.hW B.dw.hb hgd dA dE)
  have a2 := a1.comp hnb (B.bnd.maps N M Rq P hic hhw hA10 dnA dnE)
  have a3 := a2.comp hnb (FloatBridgesTo.Maps.swish (n := N * (ic * h * w)) M D.sig P.hesig
    D.spec P.hq hA20 swA swE)
  have a4 := a3.comp hnb hse
  have pj := (FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M
    B.pr.W B.pr.b P.hw' P.hβ' hn B.pr.hW B.pr.hb hgp pA pE)).comp hnbo
    (B.bnp.maps N M Rq P hoc hhw hA50 pnA pnE)
  exact a4.comp hnb pj

/-- The certified ℝ MBConv6 stride-2 block at inference (the expand runs at `2h×2w`). -/
noncomputable def EnetMBBlk.stridedFwd {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  mbStridedFwdBGen N (h := h) (w := w) B.ex.W B.ex.b (B.bne.fwd N ε (2 * h) (2 * w))
    B.dw.W B.dw.b (B.bnd.fwd N ε h w) B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂
    B.pr.W B.pr.b (B.bnp.fwd N ε h w)

/-- The deployed float MBConv6 stride-2 block. -/
noncomputable def EnetMBBlk.stridedFwdF {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  mbStridedFwdBF N (h := h) (w := w) M D.sig B.ex.W B.ex.b B.dw.W B.dw.b
    B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b
    (B.bne.fwdF N M Rq (2 * h) (2 * w)) (B.bnd.fwdF N M Rq h w) (B.bnp.fwdF N M Rq h w)

/-- This block's bridge. -/
noncomputable def EnetMBBlk.stridedBridge {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hhw2 : 0 < (2 * h) * (2 * w)) (hmid : 0 < mid) (hoc : 0 < oc) (hr : 0 < r)
    (hnE : 0 < ic * (2 * h) * (2 * w)) (hnD : 0 < mid * (2 * h) * (2 * w))
    (hn : 0 < mid * h * w) :
    FloatBridgesTo (B.stridedFwd (h := h) (w := w) N ε)
      (B.stridedFwdF (h := h) (w := w) N M D Rq) :=
  floatBridgesTo_mbStridedFwdBGen N M D.sig B.ex.W B.ex.b (B.bne.fwd N ε (2 * h) (2 * w))
    B.dw.W B.dw.b (B.bnd.fwd N ε h w) B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂
    B.pr.W B.pr.b (B.bnp.fwd N ε h w)
    (B.bne.fwdF N M Rq (2 * h) (2 * w)) (B.bnd.fwdF N M Rq h w) (B.bnp.fwdF N M Rq h w)
    P.hw' P.hβ' P.hesig hhw hmid hr hnE hnD hn D.spec
    B.ex.hW B.ex.hb B.dw.hW B.dw.hb B.se.hW₁ B.se.hb₁ B.se.hW₂ B.se.hb₂ B.pr.hW B.pr.hb
    (B.bne.bridge N M Rq P hmid hhw2) (B.bnd.bridge N M Rq P hmid hhw)
    (B.bnp.bridge N M Rq P hoc hhw)

/-- **This block's numeric envelope** — expand, its BN, swish (all at `2h×2w`), then the strided
    depthwise, its BN, swish, the SE site, project and its BN. -/
theorem EnetMBBlk.stridedMaps {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hhw2 : 0 < (2 * h) * (2 * w)) (hmid : 0 < mid) (hoc : 0 < oc) (hr : 0 < r)
    (hnE : 0 < ic * (2 * h) * (2 * w)) (hnD : 0 < mid * (2 * h) * (2 * w))
    (hn : 0 < mid * h * w)
    (hnb2 : 0 < N * (mid * (2 * h) * (2 * w))) (hnb : 0 < N * (mid * h * w))
    (hnbo : 0 < N * (oc * h * w))
    {ge gd gp Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 A6 E6 A7 E7 A8 E8 Ā' Ē' : ℝ}
    (hge : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ ge)
    (hgd : (1 + M.u) ^ (kHd * kWd + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (mid * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA20 : 0 ≤ A2) (hA40 : 0 ≤ A4) (hA50 : 0 ≤ A5) (hA80 : 0 ≤ A8)
    (eA : (1 + ge) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (eE : ge * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (enA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (enE : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (eswA : A2 + FloatModel.mulErr q A2 1 0 esig ≤ A3)
    (eswE : FloatModel.mulErr q A2 1 0 esig + min ((1 + A2/4) * E2) (A2 + E2) ≤ E3)
    (dA : (1 + gd) * (((kHd * kWd : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((kHd * kWd : ℕ) : ℝ) * w' * (A3 + E3) + β')
            + ((kHd * kWd : ℕ) : ℝ) * w' * E3 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : bnNormBudget q (A4 + Mb) S G Bb 0 es + G * S * E4 ≤ E5)
    (dswA : A5 + FloatModel.mulErr q A5 1 0 esig ≤ A6)
    (dswE : FloatModel.mulErr q A5 1 0 esig + min ((1 + A5/4) * E5) (A5 + E5) ≤ E6)
    (hse : (B.se.bridge N M D P hhw hmid hr hn (h := h) (w := w)).Maps A6 E6 A7 E7)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A7 + β') ≤ A8)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A7 + E7) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E7 ≤ E8)
    (pnA : G * ((A8 + Mb) * S) + Bb + bnNormBudget q (A8 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A8 + Mb) S G Bb 0 es + G * S * E8 ≤ Ē') :
    (B.stridedBridge N M D Rq P hhw hhw2 hmid hoc hr hnE hnD hn (h := h) (w := w)).Maps
      Ā Ē Ā' Ē' := by
  have c1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv
    (h := 2 * h) (w := 2 * w) M B.ex.W B.ex.b P.hw' P.hβ' hnE B.ex.hW B.ex.hb hge eA eE)
  have c2 := c1.comp hnb2 (B.bne.maps N M Rq P hmid hhw2 hA10 enA enE)
  have c3 := c2.comp hnb2 (FloatBridgesTo.Maps.swish (n := N * (mid * (2 * h) * (2 * w))) M D.sig
    P.hesig D.spec P.hq hA20 eswA eswE)
  have a1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.depthwiseStride2Flat
    (h := h) (w := w) M B.dw.W B.dw.b P.hw' P.hβ' hnD B.dw.hW B.dw.hb hgd dA dE)
  have a2 := a1.comp hnb (B.bnd.maps N M Rq P hmid hhw hA40 dnA dnE)
  have a3 := a2.comp hnb (FloatBridgesTo.Maps.swish (n := N * (mid * h * w)) M D.sig P.hesig
    D.spec P.hq hA50 dswA dswE)
  have b1 := c3.comp hnb2 a3
  have b2 := b1.comp hnb hse
  have pj := (FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M
    B.pr.W B.pr.b P.hw' P.hβ' hn B.pr.hW B.pr.hb hgp pA pE)).comp hnbo
    (B.bnp.maps N M Rq P hoc hhw hA80 pnA pnE)
  exact b2.comp hnb pj

/-- The certified ℝ MBConv6 residual block at inference (matched channels, stride 1). -/
noncomputable def EnetMBBlk.residFwd {c mid r kHd kWd h w : Nat}
    (B : EnetMBBlk c mid c r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  mbResidFwdBGen N (h := h) (w := w) B.ex.W B.ex.b (B.bne.fwd N ε h w)
    B.dw.W B.dw.b (B.bnd.fwd N ε h w) B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂
    B.pr.W B.pr.b (B.bnp.fwd N ε h w)

/-- The deployed float MBConv6 residual block — the skip-add is rounded. -/
noncomputable def EnetMBBlk.residFwdF {c mid r kHd kWd h w : Nat}
    (B : EnetMBBlk c mid c r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  mbResidFwdBF N (h := h) (w := w) M D.sig B.ex.W B.ex.b B.dw.W B.dw.b
    B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b
    (B.bne.fwdF N M Rq h w) (B.bnd.fwdF N M Rq h w) (B.bnp.fwdF N M Rq h w)

/-- This block's bridge. -/
noncomputable def EnetMBBlk.residBridge {c mid r kHd kWd h w : Nat}
    (B : EnetMBBlk c mid c r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hc : 0 < c) (hmid : 0 < mid) (hr : 0 < r)
    (hnC : 0 < c * h * w) (hn : 0 < mid * h * w) :
    FloatBridgesTo (B.residFwd (h := h) (w := w) N ε)
      (B.residFwdF (h := h) (w := w) N M D Rq) :=
  floatBridgesTo_mbResidFwdBGen N M D.sig B.ex.W B.ex.b (B.bne.fwd N ε h w)
    B.dw.W B.dw.b (B.bnd.fwd N ε h w) B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂
    B.pr.W B.pr.b (B.bnp.fwd N ε h w)
    (B.bne.fwdF N M Rq h w) (B.bnd.fwdF N M Rq h w) (B.bnp.fwdF N M Rq h w)
    P.hw' P.hβ' P.hesig hhw hc hmid hr hnC hn D.spec
    B.ex.hW B.ex.hb B.dw.hW B.dw.hb B.se.hW₁ B.se.hb₁ B.se.hW₂ B.se.hb₂ B.pr.hW B.pr.hb
    (B.bne.bridge N M Rq P hmid hhw) (B.bnd.bridge N M Rq P hmid hhw)
    (B.bnp.bridge N M Rq P hc hhw)

/-- **This block's numeric envelope** — the eight body stages, then the rounded skip fan-in
    against the block's own input window. -/
theorem EnetMBBlk.residMaps {c mid r kHd kWd h w : Nat}
    (B : EnetMBBlk c mid c r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hc : 0 < c) (hmid : 0 < mid) (hr : 0 < r)
    (hnC : 0 < c * h * w) (hn : 0 < mid * h * w)
    (hnbc : 0 < N * (c * h * w)) (hnb : 0 < N * (mid * h * w))
    {ge gd gp Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 A6 E6 A7 E7 A8 E8 Bd Ed Ā' Ē' : ℝ}
    (hge : (1 + M.u) ^ (c * 1 * 1 + 2) - 1 ≤ ge)
    (hgd : (1 + M.u) ^ (kHd * kWd + 2) - 1 ≤ gd)
    (hgp : (1 + M.u) ^ (mid * 1 * 1 + 2) - 1 ≤ gp)
    (hA10 : 0 ≤ A1) (hA20 : 0 ≤ A2) (hA40 : 0 ≤ A4) (hA50 : 0 ≤ A5) (hA80 : 0 ≤ A8)
    (eA : (1 + ge) * (((c * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (eE : ge * (((c * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((c * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (enA : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (enE : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (eswA : A2 + FloatModel.mulErr q A2 1 0 esig ≤ A3)
    (eswE : FloatModel.mulErr q A2 1 0 esig + min ((1 + A2/4) * E2) (A2 + E2) ≤ E3)
    (dA : (1 + gd) * (((kHd * kWd : ℕ) : ℝ) * w' * A3 + β') ≤ A4)
    (dE : gd * (((kHd * kWd : ℕ) : ℝ) * w' * (A3 + E3) + β')
            + ((kHd * kWd : ℕ) : ℝ) * w' * E3 ≤ E4)
    (dnA : G * ((A4 + Mb) * S) + Bb + bnNormBudget q (A4 + Mb) S G Bb 0 es ≤ A5)
    (dnE : bnNormBudget q (A4 + Mb) S G Bb 0 es + G * S * E4 ≤ E5)
    (dswA : A5 + FloatModel.mulErr q A5 1 0 esig ≤ A6)
    (dswE : FloatModel.mulErr q A5 1 0 esig + min ((1 + A5/4) * E5) (A5 + E5) ≤ E6)
    (hse : (B.se.bridge N M D P hhw hmid hr hn (h := h) (w := w)).Maps A6 E6 A7 E7)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A7 + β') ≤ A8)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A7 + E7) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E7 ≤ E8)
    (pnA : G * ((A8 + Mb) * S) + Bb + bnNormBudget q (A8 + Mb) S G Bb 0 es ≤ Bd)
    (pnE : bnNormBudget q (A8 + Mb) S G Bb 0 es + G * S * E8 ≤ Ed)
    (rA : Bd + Ā + q * (Bd + Ā) ≤ Ā') (rE : q * (Bd + Ed + Ā + Ē) + (Ed + Ē) ≤ Ē') :
    (B.residBridge N M D Rq P hhw hc hmid hr hnC hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' := by
  have c1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M
    B.ex.W B.ex.b P.hw' P.hβ' hnC B.ex.hW B.ex.hb hge eA eE)
  have c2 := c1.comp hnb (B.bne.maps N M Rq P hmid hhw hA10 enA enE)
  have c3 := c2.comp hnb (FloatBridgesTo.Maps.swish (n := N * (mid * h * w)) M D.sig P.hesig
    D.spec P.hq hA20 eswA eswE)
  have a1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.depthwise (h := h) (w := w) M
    B.dw.W B.dw.b P.hw' P.hβ' hn B.dw.hW B.dw.hb hgd dA dE)
  have a2 := a1.comp hnb (B.bnd.maps N M Rq P hmid hhw hA40 dnA dnE)
  have a3 := a2.comp hnb (FloatBridgesTo.Maps.swish (n := N * (mid * h * w)) M D.sig P.hesig
    D.spec P.hq hA50 dswA dswE)
  have b1 := c3.comp hnb a3
  have b2 := b1.comp hnb hse
  have pj := (FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M
    B.pr.W B.pr.b P.hw' P.hβ' hn B.pr.hW B.pr.hb hgp pA pE)).comp hnbc
    (B.bnp.maps N M Rq P hc hhw hA80 pnA pnE)
  exact FloatBridgesTo.Maps.residual M hnbc (b2.comp hnb pj) P.hq rA rE


-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The deployed EfficientNet-B0 inference forward** — the `*Gen` skeleton with inference
    BatchNorm at every one of its ten sites. `rfl`-equal to `efficientnetForwardBEval`
    (`b0EvalForward_eq_forwardBEval`). -/
noncomputable def b0EvalForward (N : Nat) (W : EnetWeights w' β' G Bb Mb) (ε : ℝ) :
    Vec (N * (3 * 224 * 224)) → Vec (N * 10) :=
  headFwdBGen N (h := 56) (w := 56) W.hd.W W.hd.b (W.bnh.fwd N ε 56 56) W.head.W W.head.b ∘
    W.b3.residFwd (h := 56) (w := 56) N ε ∘
    W.b2.stridedFwd (h := 56) (w := 56) N ε ∘
    W.b1.fwd (h := 112) (w := 112) N ε ∘
    stemBGen N (h := 112) (w := 112) W.stem.W W.stem.b (W.bns.fwd N ε 112 112)

/-- **The deployed EfficientNet-B0 float inference forward** — every concrete slot replaced by
    the model's rounded peer, every BN by the six rounded ops the emitter writes, every sigmoid
    by the device kernel. -/
noncomputable def b0EvalForwardF (N : Nat) (M : FloatModel) (D : DeviceSigmoid esig)
    (Rq : DeviceRsqrt ε es) (W : EnetWeights w' β' G Bb Mb) :
    Vec (N * (3 * 224 * 224)) → Vec (N * 10) :=
  headFwdBF N (h := 56) (w := 56) M D.sig W.hd.W W.hd.b W.head.W W.head.b
      (W.bnh.fwdF N M Rq 56 56) ∘
    W.b3.residFwdF (h := 56) (w := 56) N M D Rq ∘
    W.b2.stridedFwdF (h := 56) (w := 56) N M D Rq ∘
    W.b1.fwdF (h := 112) (w := 112) N M D Rq ∘
    stemBF N (h := 112) (w := 112) M D.sig W.stem.W W.stem.b (W.bns.fwdF N M Rq 112 112)

set_option maxRecDepth 100000 in
/-- ⭐ **The whole deployed EfficientNet-B0 inference forward float-bridges TO its float peer** —
    a CLOSED `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left: stem, the three MBConv
    blocks (each with its squeeze-excite) and the head are each discharged by leaves. Its `.mod`
    is a closed term over the per-op budgets, and `b0EvalBridge_maps` bounds it. -/
noncomputable def b0EvalBridge (N : Nat) (M : FloatModel) (D : DeviceSigmoid esig)
    (Rq : DeviceRsqrt ε es) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (W : EnetWeights w' β' G Bb Mb) :
    FloatBridgesTo (b0EvalForward N W ε) (b0EvalForwardF N M D Rq W) :=
  ((((floatBridgesTo_stemBGen N M D.sig W.stem.W W.stem.b (W.bns.fwd N ε 112 112)
      (W.bns.fwdF N M Rq 112 112) P.hw' P.hβ' P.hesig (by norm_num) D.spec W.stem.hW W.stem.hb
      (W.bns.bridge N M Rq P (by norm_num) (by norm_num)))
    |>.comp (W.b1.bridge N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 112) (w := 112)))
    |>.comp (W.b2.stridedBridge N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (h := 56) (w := 56)))
    |>.comp (W.b3.residBridge N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (floatBridgesTo_headFwdBGen N M D.sig W.hd.W W.hd.b (W.bnh.fwd N ε 56 56)
      W.head.W W.head.b (W.bnh.fwdF N M Rq 56 56) P.hw' P.hβ' P.hesig (by norm_num) (by norm_num) (by norm_num) D.spec
      W.hd.hW W.hd.hb W.head.hW W.head.hb (W.bnh.bridge N M Rq P (by norm_num) (by norm_num)))


-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile.** Every stored parameter within `41/10` — the global maximum over
    the 350-epoch checkpoint (`/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin`,
    5,288,548 f32) is `4.0545`, its 99.99th percentile is `1.3949` and 100 entries exceed `2`, so
    the uniform bound is loose and the fold is not sensitive to it. `ε ≥ 10⁻⁵` (the trainer's
    value) puts the inference inverse-stddev under `317`; the device `rsqrt` and the device
    sigmoid are each taken accurate to `10⁻²` absolute. -/
theorem b0Profile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    EnetProfile M ε (41/10) (41/10) (41/10) (41/10) (41/10) (1/100) (1/100) 317 u32 where
  hw' := by norm_num
  hβ' := by norm_num
  hG := by norm_num
  hBb := by norm_num
  hMb := by norm_num
  hes := by norm_num
  hesig := by norm_num
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
/-- ⭐ **The envelope, kernel-checked.** Every numeric stage closed by two rational inequalities,
    with each γ-term bounded through `FloatModel.gamma_num` so `norm_num` never evaluates a big
    power. Built bottom-up at block granularity — the three MBConv envelopes each fold their own
    squeeze-excite through `EnetSE.maps` — and the closing `exact` is one structural comparison
    with `b0EvalBridge`'s definition. -/
theorem b0EvalBridge_maps (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetWeights (41/10) (41/10) (41/10) (41/10) (41/10)) :
    (b0EvalBridge N M D Rq (b0Profile_committed M hMu hε5) W).Maps 1 0
      (2580 * 10 ^ 52) (8408 * 10 ^ 207) := by
  have hP := b0Profile_committed M hMu hε5
  -- stem: 3×3/s2 conv, inference BN, swish
  have t1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConvStride2
    (h := 112) (w := 112) M W.stem.W W.stem.b hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb (M.gamma_num (q := 1729 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 1149 / 10 ^ 1) (Ē' := 1985 / 10 ^ 7) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t2 := t1.comp (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (W.bns.maps N M Rq hP (by norm_num) (by norm_num) (h := 112) (w := 112)
    (Ā' := 1547 * 10 ^ 2) (Ē' := 5174 / 10 ^ 3) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t3 := t2.comp (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (FloatBridgesTo.Maps.swish
    (n := N * (32 * 112 * 112)) M D.sig hP.hesig D.spec hP.hq (by norm_num)
    (Ā' := 1563 * 10 ^ 2) (Ē' := 1563 * 10 ^ 2) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
  -- b1: MBConv1, no expand (32 → 16), SE at r = 8
  have t4 := t3.comp (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (W.b1.maps N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (Nat.mul_pos hN (by norm_num : 0 < 16 * 112 * 112)) (h := 112) (w := 112)
    (gd := 6557 / 10 ^ 10) (gp := 2027 / 10 ^ 9)
    (A1 := 5768 * 10 ^ 3) (E1 := 5768 * 10 ^ 3)
    (A2 := 7497 * 10 ^ 6) (E2 := 7497 * 10 ^ 6)
    (A3 := 7572 * 10 ^ 6) (E3 := 1507 * 10 ^ 7)
    (A4 := 7648 * 10 ^ 6) (E4 := 5543 * 10 ^ 20)
    (A5 := 1004 * 10 ^ 9) (E5 := 7273 * 10 ^ 22)
    (Ā' := 1305 * 10 ^ 12) (Ē' := 9453 * 10 ^ 25)
    (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (W.b1.se.maps N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 112) (w := 112)
      (gG := 7483 / 10 ^ 7) (g1 := 2027 / 10 ^ 9) (g2 := 5961 / 10 ^ 10)
      (A1 := 7578 * 10 ^ 6) (E1 := 1508 * 10 ^ 7)
      (A2 := 9943 * 10 ^ 8) (E2 := 1979 * 10 ^ 9)
      (A3 := 1005 * 10 ^ 9) (E3 := 2984 * 10 ^ 9)
      (A4 := 3297 * 10 ^ 10) (E4 := 9788 * 10 ^ 10)
      (Cg := 1010 / 10 ^ 3) (Eg := 2448 * 10 ^ 10)
      (Ā' := 7648 * 10 ^ 6) (Ē' := 5543 * 10 ^ 20)
      (by norm_num) (M.gamma_num (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5961 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
    (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b2: MBConv6 stride-2 (16 → 96 → 24), 3×3 depthwise, SE at r = 4
  have t5 := t4.comp (Nat.mul_pos hN (by norm_num : 0 < 16 * 112 * 112)) (W.b2.stridedMaps N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 96 * 112 * 112)) (Nat.mul_pos hN (by norm_num : 0 < 96 * 56 * 56)) (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56))
    (h := 56) (w := 56) (ge := 1073 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5842 / 10 ^ 9)
    (A1 := 8561 * 10 ^ 13) (E1 := 6202 * 10 ^ 27)
    (A2 := 1113 * 10 ^ 17) (E2 := 8061 * 10 ^ 30)
    (A3 := 1125 * 10 ^ 17) (E3 := 8062 * 10 ^ 30)
    (A4 := 4152 * 10 ^ 18) (E4 := 2975 * 10 ^ 32)
    (A5 := 5397 * 10 ^ 21) (E5 := 3867 * 10 ^ 35)
    (A6 := 5451 * 10 ^ 21) (E6 := 3868 * 10 ^ 35)
    (A7 := 5506 * 10 ^ 21) (E7 := 2418 * 10 ^ 77)
    (A8 := 2168 * 10 ^ 24) (E8 := 9518 * 10 ^ 79)
    (Ā' := 2818 * 10 ^ 27) (Ē' := 1238 * 10 ^ 83)
    (M.gamma_num (q := 1073 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (W.b2.se.maps N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)
      (gG := 1871 / 10 ^ 7) (g1 := 5842 / 10 ^ 9) (g2 := 3577 / 10 ^ 10)
      (A1 := 5453 * 10 ^ 21) (E1 := 3869 * 10 ^ 35)
      (A2 := 2147 * 10 ^ 24) (E2 := 1523 * 10 ^ 38)
      (A3 := 2169 * 10 ^ 24) (E3 := 1524 * 10 ^ 38)
      (A4 := 3558 * 10 ^ 25) (E4 := 2500 * 10 ^ 39)
      (Cg := 1010 / 10 ^ 3) (Eg := 6251 * 10 ^ 38)
      (Ā' := 5506 * 10 ^ 21) (Ē' := 2418 * 10 ^ 77)
      (by norm_num) (M.gamma_num (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3577 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
    (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b3: MBConv6 residual (24 → 144 → 24), 5×5 depthwise, SE at r = 6
  have t6 := t5.comp (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56)) (W.b3.residMaps N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56)) (Nat.mul_pos hN (by norm_num : 0 < 144 * 56 * 56)) (h := 56) (w := 56)
    (ge := 1550 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 8703 / 10 ^ 9)
    (A1 := 2773 * 10 ^ 29) (E1 := 1219 * 10 ^ 85)
    (A2 := 3605 * 10 ^ 32) (E2 := 1585 * 10 ^ 88)
    (A3 := 3642 * 10 ^ 32) (E3 := 1586 * 10 ^ 88)
    (A4 := 3734 * 10 ^ 34) (E4 := 1626 * 10 ^ 90)
    (A5 := 4854 * 10 ^ 37) (E5 := 2114 * 10 ^ 93)
    (A6 := 4903 * 10 ^ 37) (E6 := 2115 * 10 ^ 93)
    (A7 := 4953 * 10 ^ 37) (E7 := 1628 * 10 ^ 193)
    (A8 := 2925 * 10 ^ 40) (E8 := 9612 * 10 ^ 195)
    (Bd := 3802 * 10 ^ 43) (Ed := 1250 * 10 ^ 199)
    (Ā' := 3803 * 10 ^ 43) (Ē' := 1251 * 10 ^ 199)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (W.b3.se.maps N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)
      (gG := 1871 / 10 ^ 7) (g1 := 8703 / 10 ^ 9) (g2 := 4769 / 10 ^ 10)
      (A1 := 4904 * 10 ^ 37) (E1 := 2116 * 10 ^ 93)
      (A2 := 2896 * 10 ^ 40) (E2 := 1250 * 10 ^ 96)
      (A3 := 2925 * 10 ^ 40) (E3 := 1251 * 10 ^ 96)
      (A4 := 7196 * 10 ^ 41) (E4 := 3078 * 10 ^ 97)
      (Cg := 1010 / 10 ^ 3) (Eg := 7696 * 10 ^ 96)
      (Ā' := 4953 * 10 ^ 37) (Ē' := 1628 * 10 ^ 193)
      (by norm_num) (M.gamma_num (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4769 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
    (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- head: 1×1 conv (24 → 1280), inference BN, swish, GAP, dense
  have h1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := 56) (w := 56) M
    W.hd.W W.hd.b hP.hw' hP.hβ' (by norm_num) W.hd.hW W.hd.hb (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 3803 * 10 ^ 43) (Ē := 1251 * 10 ^ 199)
    (Ā' := 3743 * 10 ^ 45) (Ē' := 1231 * 10 ^ 201) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have h2 := h1.comp (Nat.mul_pos hN (by norm_num : 0 < 1280 * 56 * 56)) (W.bnh.maps N M Rq hP (by norm_num) (by norm_num) (h := 56) (w := 56)
    (Ā' := 4865 * 10 ^ 48) (Ē' := 1600 * 10 ^ 204) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have h3 := h2.comp (Nat.mul_pos hN (by norm_num : 0 < 1280 * 56 * 56)) (FloatBridgesTo.Maps.swish
    (n := N * (1280 * 56 * 56)) M D.sig hP.hesig D.spec hP.hq (by norm_num)
    (Ā' := 4914 * 10 ^ 48) (Ē' := 1601 * 10 ^ 204) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
  have h4 := h3.comp (Nat.mul_pos hN (by norm_num : 0 < 1280 * 56 * 56)) (FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.gap (c := 1280) (h := 56) (w := 56) M (by norm_num) (by norm_num) hMu
      (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 4915 * 10 ^ 48) (Ē' := 1602 * 10 ^ 204) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])))
  have h5 := h4.comp (Nat.mul_pos hN (by norm_num : 0 < 1280)) (FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num) W.head.hW W.head.hb
      (M.gamma_num (q := 7642 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 2580 * 10 ^ 52) (Ē' := 8408 * 10 ^ 207) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])))
  exact t6.comp (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56)) h5

/-- The deployed EfficientNet-B0 inference bridge's certified output window at the committed
    profile: `≤ 2.580·10⁵⁵`. -/
theorem b0EvalBridge_mag_le (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetWeights (41/10) (41/10) (41/10) (41/10) (41/10)) :
    (b0EvalBridge N M D Rq (b0Profile_committed M hMu hε5) W).mag 1 ≤ 2580 * 10 ^ 52 :=
  (b0EvalBridge_maps N hN M hMu hε5 D Rq W).mag_le 1 (by norm_num) le_rfl

/-- The deployed EfficientNet-B0 inference bridge's fresh budget at the committed profile:
    `≤ 8.408·10²¹⁰`. ⚠ Three quarters of those orders are the three squeeze-excite sites — each
    roughly doubles the exponent, because `seScale`'s modulus is quadratic in the window. -/
theorem b0EvalBridge_fresh_le (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetWeights (41/10) (41/10) (41/10) (41/10) (41/10)) :
    (b0EvalBridge N M D Rq (b0Profile_committed M hMu hε5) W).fresh 1 ≤ 8408 * 10 ^ 207 :=
  (b0EvalBridge_maps N hN M hMu hε5 D Rq W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed EfficientNet-B0 inference forward is within `8.408·10²¹⁰` of the certified
    real forward, per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile
    (`|·| ≤ 41/10`), for `ε ≥ 10⁻⁵`, any device `rsqrt` and `sigmoid` accurate to `10⁻²`, and any
    rounding model at binary32 accuracy. The third ImageNet-scale whole-net float budget in the
    repo, and the first for a batched net with squeeze-excite. See the file header for the two
    leaf bounds that had to be tightened before this number existed at all. -/
theorem b0_float_logits_le (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetWeights (41/10) (41/10) (41/10) (41/10) (41/10))
    (x : Vec (N * (3 * 224 * 224))) (hx : ∀ k, |x k| ≤ 1) (j : Fin (N * 10)) :
    |b0EvalForwardF N M D Rq W x j - b0EvalForward N W ε x j| ≤ 8408 * 10 ^ 207 :=
  (b0EvalBridge_maps N hN M hMu hε5 D Rq W).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed inference forward, and the graph denotes it
-- ════════════════════════════════════════════════════════════════

/-- **The record-bundled forward IS the committed inference net.** `b0EvalForward` unfolds to
    `efficientnetForwardBEval` at the record's projections — the `*Gen` skeleton at
    `batchMap N (bnPerChannelEvalTensor3 …)` is exactly what `EfficientNetRenderPCEval.lean`
    defines (`cbsBEval_eq_gen` and its eight peers), so this is a `rfl`. -/
theorem b0EvalForward_eq_forwardBEval (N : Nat) (W : EnetWeights w' β' G Bb Mb) (ε : ℝ)
    (x : Vec (N * (3 * 224 * 224))) :
    b0EvalForward N W ε x = efficientnetForwardBEval N ε
    W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v
    W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.se.W₁ W.b1.se.b₁ W.b1.se.W₂ W.b1.se.b₂ W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v
    W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.se.W₁ W.b2.se.b₁ W.b2.se.W₂ W.b2.se.b₂ W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v
    W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.se.W₁ W.b3.se.b₁ W.b3.se.W₂ W.b3.se.b₂ W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v
    W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x := by
  -- ⚠ NOT one `rfl`: at these concrete dims the kernel times out comparing the two whole nets.
  -- Rewriting with the nine per-stage `*Eval_eq_gen` lemmas first leaves nothing to compare.
  simp only [b0EvalForward, EnetNoExpBlk.fwd, EnetMBBlk.stridedFwd, EnetMBBlk.residFwd,
    EnetBn.fwd, efficientnetForwardBEval, stemBEval_eq_gen, mbNoExpFwdBEval_eq_gen,
    mbStridedFwdBEval_eq_gen, mbResidFwdBEval_eq_gen, headFwdBEval_eq_gen,
    Function.comp_apply]

/-- ⭐ **The whole loop closes.** The typed `SHlo` inference graph denotes exactly the forward
    this file states its number about. `efficientnetFwdGraphBEval_faithful` carried through the
    record tie. -/
theorem b0EvalGraph_faithful (N : Nat) (epsStr : String) (ε : ℝ)
    (W : EnetWeights w' β' G Bb Mb) (x : Vec (N * (3 * 224 * 224))) :
    StableHLO.den (StableHLO.efficientnetFwdGraphBEval N epsStr ε
    W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v
    W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.se.W₁ W.b1.se.b₁ W.b1.se.W₂ W.b1.se.b₂ W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v
    W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.se.W₁ W.b2.se.b₁ W.b2.se.W₂ W.b2.se.b₂ W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v
    W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.se.W₁ W.b3.se.b₁ W.b3.se.W₂ W.b3.se.b₂ W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v
    W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x)
      = b0EvalForward N W ε x :=
  (StableHLO.efficientnetFwdGraphBEval_faithful N epsStr ε W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.se.W₁ W.b1.se.b₁ W.b1.se.W₂ W.b1.se.b₂ W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.se.W₁ W.b2.se.b₁ W.b2.se.W₂ W.b2.se.b₂ W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.se.W₁ W.b3.se.b₁ W.b3.se.W₂ W.b3.se.b₂ W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x).trans
    (b0EvalForward_eq_forwardBEval N W ε x).symm

/-- ⭐⭐ **The number, stated about the committed inference forward.** `b0_float_logits_le` with
    `efficientnetForwardBEval` on the real side instead of the record-bundled `b0EvalForward` —
    so the budget is a claim about the net the eval render emits, tied through
    `b0EvalGraph_faithful` rather than by inspection. -/
theorem b0_float_logits_le_committed (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32)
    {ε : ℝ} (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetWeights (41/10) (41/10) (41/10) (41/10) (41/10))
    (x : Vec (N * (3 * 224 * 224))) (hx : ∀ k, |x k| ≤ 1) (j : Fin (N * 10)) :
    |b0EvalForwardF N M D Rq W x j - efficientnetForwardBEval N ε
    W.stem.W W.stem.b W.bns.γ W.bns.β W.bns.μ W.bns.v
    W.b1.dw.W W.b1.dw.b W.b1.bnd.γ W.b1.bnd.β W.b1.bnd.μ W.b1.bnd.v W.b1.se.W₁ W.b1.se.b₁ W.b1.se.W₂ W.b1.se.b₂ W.b1.pr.W W.b1.pr.b W.b1.bnp.γ W.b1.bnp.β W.b1.bnp.μ W.b1.bnp.v
    W.b2.ex.W W.b2.ex.b W.b2.bne.γ W.b2.bne.β W.b2.bne.μ W.b2.bne.v W.b2.dw.W W.b2.dw.b W.b2.bnd.γ W.b2.bnd.β W.b2.bnd.μ W.b2.bnd.v W.b2.se.W₁ W.b2.se.b₁ W.b2.se.W₂ W.b2.se.b₂ W.b2.pr.W W.b2.pr.b W.b2.bnp.γ W.b2.bnp.β W.b2.bnp.μ W.b2.bnp.v
    W.b3.ex.W W.b3.ex.b W.b3.bne.γ W.b3.bne.β W.b3.bne.μ W.b3.bne.v W.b3.dw.W W.b3.dw.b W.b3.bnd.γ W.b3.bnd.β W.b3.bnd.μ W.b3.bnd.v W.b3.se.W₁ W.b3.se.b₁ W.b3.se.W₂ W.b3.se.b₂ W.b3.pr.W W.b3.pr.b W.b3.bnp.γ W.b3.bnp.β W.b3.bnp.μ W.b3.bnp.v
    W.hd.W W.hd.b W.bnh.γ W.bnh.β W.bnh.μ W.bnh.v W.head.W W.head.b x j| ≤ 8408 * 10 ^ 207 := by
  rw [← b0EvalForward_eq_forwardBEval N W ε x]
  exact b0_float_logits_le N hN M hMu hε5 D Rq W x hx j

end Proofs
