import LeanMlir.Proofs.Float.EfficientNetFloatBudget
import LeanMlir.Proofs.Architectures.EfficientNetFullB0Eval

/-! # A NUMBER for the PAPER EfficientNet-B0: sixteen MBConv blocks, and the cap on the gate

`EfficientNetFloatBudget.lean` states its number about the three-block representative. This
file states one about the net the `[t,c,n,s,k]` table describes and `efficientnetForwardB_full`
(`Architectures/EfficientNetFullB0.lean`) IS — **all sixteen MBConv blocks**: 3×3/s2 stem 3→32
at the XLA-`SAME` phase, `b1` the t=1 no-expand block, four stride-2 downsamples
(`b2/b4/b6/b12`), nine identity skips, two stage-first widenings with no skip (`b9/b16`), 1×1
head 320→1280 at 7×7, GAP, dense — with **inference** BatchNorm at all **49** sites (the
representative has 10) and a squeeze-excite gate in every block (**16** against 3), on the unit
input window, at the profile measured on the 350-epoch checkpoint (`|parameter| ≤ 41/10`), for
any batch size, for any rounding model at binary32 accuracy:

    output window  ≤ 1.886·10²⁷⁹     (`b0FullEvalBridge_mag_le`)
    fresh budget   ≤ 2.416·10²⁸⁷     (`b0FullEvalBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 2.416·10²⁸⁷` (`b0Full_float_logits_le`).

⛔⛔ **Read the budget's label first: the SIGMOID of every one of the sixteen squeeze-excite
gates goes through `FloatBridgesTo.capped`.** Nothing else is capped — the 49 BatchNorms, the
convolutions, the swishes, the SE rescale itself, GAP and the classifier are all the interval
fold. Why the gate, and why it is the right place: `seScale`'s modulus is `mulErr q A Cg E Eg`,
and it carries `A · Eg` — the block's window times the gate's ERROR — while the gate grows that
error out of the same window through the squeeze's `GAP → dense → swish → dense`. So squeeze-
excite is quadratic in the window (§0.1's third site), each SE roughly doubles the budget's
exponent, and sixteen of them fold to `10^1897907`. The sigmoid is the one stage in the gate
path whose WINDOW is a constant — `floatBridgesTo_sigmoid`'s magnitude is `1 + esig` at every
input — so capping there costs exactly one side condition, `2·(1 + esig) ≤ Eg`, and turns the
rescale's modulus into `≈ 2·A + 3·E`, linear in both. What the cap says at those sixteen sites
is the triangle inequality on the sigmoid's range: *"the float gate and the real gate both lie
in `[−(1+esig), 1+esig]`"*. The tell is not `budget/window ≈ 2` here but `≈ 1.3·10⁸`: sixteen
gate caps compounding as `E ↦ 2A + 3E` through the linear stages between them.

⭐ **The window is honest and it is the whole story of this number.** `2.580·10⁵⁵ → 1.886·10²⁷⁹`:
swish never resets a window (unlike relu6, which pins MobileNetV2's body flat), so every block
multiplies it by its three fan-ins, three `G·S` BatchNorm gains and the gate's `1 + esig` —
about `10¹⁷` per block, `10¹⁶` for the 3×3 ones and `10¹⁸` for the 5×5 and 1152-wide ones. No
cap touches a window, and no operating point is taken: `S = 317` is the ε-floor.

⭐⭐ **`norm_num`'s "ceiling" is an option, not a wall — and this is the first number to need
that fact.** Every earlier budget file speaks of a shape-dependent ceiling near `10²⁵³`
(`planning/archive/float_budget_numbers_log.md` §3.7(a)). Measured 2026-09-05 while writing
this file: it is Lean's `exponentiation.threshold` (default 256) — `10 ^ 256` evaluates and
`10 ^ 257` does not, in exactly the shape that was failing, and with the option raised the same
goals close at `10 ^ 290` in the same time. The kernel's `Nat.pow` is GMP-backed and never had
a limit. `b0FullEvalBridge_maps` below is stated under `set_option exponentiation.threshold 400`
and is otherwise the same `norm_num` proof as every other budget file's. ⛔ This does not make
the uncapped `10^1897907` fold a sensible theorem to state, and it does not change what any
number here MEANS (`planning/float_budget_numbers.md` §2); it retires "no theorem to state"
as a reason, and `planning/float_budget_numbers.md` §3 carries the correction.

**The profile is measured on THIS net.** `EfficientNetFloatBudget.lean` takes `|·| ≤ 41/10`
from `/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin`, 5,288,548 f32 — the
sixteen-block net's parameter count. At the representative that profile was a transplant; here
it is the checkpoint of the net being bounded, with the usual qualification that the checkpoint
is the 1000-class net and the classifier here is generic in `nCls` (the dense envelope depends
on the fan-in 1280 and never on the output count), so the bound is a measurement on all 49
convolutions, 49 BatchNorms and 32 SE denses and an assumption on the `1280 × nCls` head.

⚠ **The two hypotheses this number rests on, named.** The device `rsqrt` (`DeviceRsqrt ε es`)
and the device `sigmoid` (`DeviceSigmoid esig`), each modelled at `10⁻²` absolute accuracy, as
in the representative's file. Everything else is proved.

⚠ **The representative is a SHAPE COVER of the paper table, not a prefix of it.** Its `b1`/`b2`
are the paper's, but its `b3` carries a 5×5 depthwise (so the 5×5 shape is exercised) where the
paper's `b3` — stage 2, `k = 3` — is 3×3; and its head runs on 24 channels at 56×56 where the
paper's runs on 320 at 7×7. `b0_full_plan` in the probe asserts exactly this.

**What this is stated about, and what it is not.** Each of the sixteen blocks is tied by `rfl`
to the eval-stage abbreviations the committed inference forward is built from
(`mbNoExpFwdBEval`, `mbStridedFwdBEval`, `mbResidFwdBEval`, and for the two no-skip widenings
the body `projBEval ∘ seB ∘ dwbsBEval ∘ cbsBEval`, which is `mbResidFwdBEval`'s body without the
skip — `*_eq_eval` below, dimension-polymorphic). The ladder — which block at which spatial
size, with which skip — is `efficientnetForwardB_full`'s, and the widths are `B0Weights`'s; the
probe reads both from that file and asserts they agree, block for block.

**The tie is closed at the graph.** `efficientnetForwardB_full` is at TRAINING BatchNorm, the
world its VJP and its typed graph live in, so the number could not end there; its eval twin,
`efficientnetForwardB_fullEval` (`Architectures/EfficientNetFullB0Eval.lean`, 49 frozen-statistic
sites, the typed graph `efficientnetFwdGraphB_fullEval` and its faithfulness), was built for this
file. `b0FullEvalForward_eq_fullEval` rewrites the record-bundled forward onto it through the
per-stage `*Eval_eq_gen` lemmas — NOT one `rfl`, the three-block lesson — `b0FullEvalGraph_faithful`
carries the graph's faithfulness the rest of the way, and `b0Full_float_logits_le_committed` states
the number with that net on the real side. The typed graph is the `den`-level form of the shipped
`efficientnet_fwd_eval.mlir` (312 inputs, THIS net); its SSA names are the three-block eval graph's
and differ from the artifact's cosmetically (that file's header lists the four ways).

Provenance for the numerals: `scripts/float_budget_envelope.py` (`b0_full_eval_chain`), which
reads the block table from Lean, folds the envelope in exactly these lemmas' semantics with
exact rationals, rounds every stage UP to four significant figures, and re-asserts each rounded
inequality (`verify_b0_full`, 476 inequalities) before emitting.
-/

namespace Proofs

open FloatModel

variable {M : FloatModel} {ε w' β' G Bb Mb es esig S q : ℝ}

-- ════════════════════════════════════════════════════════════════
-- § The cap on the gate: the squeeze-excite bridge with its sigmoid capped
-- ════════════════════════════════════════════════════════════════

/-- `floatBridgesTo_seGate` with the sigmoid stage under `FloatBridgesTo.capped`. Same maps, same
    window; the modulus at the sigmoid is `min(esig + e/4, 2·(1 + esig))`, and only the right
    branch is ever emitted. -/
noncomputable def floatBridgesTo_seGateC {c h w r : Nat} (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' β esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β) :
    FloatBridgesTo (seGate (h := h) (w := w) W₁ b₁ W₂ b₂)
      (seGateF (h := h) (w := w) M fsig W₁ b₁ W₂ b₂) := by
  unfold seGate seGateF
  exact (((((floatBridgesTo_gap (c := c) (h := h) (w := w) M hc hhw).comp
      (floatBridgesTo_dense M W₁ b₁ hw' hβ hc hW₁ hb₁)).comp
      (floatBridgesTo_swish (n := r) M fsig hesig hsig)).comp
      (floatBridgesTo_dense M W₂ b₂ hw' hβ hr hW₂ hb₂)).comp
      (floatBridgesTo_sigmoid (n := c) fsig hesig hsig).capped).comp
      (floatBridgesTo_broadcast (c := c) (h := h) (w := w))

/-- The full SE block `x ⊙ gate(x)` with the gate's sigmoid capped. -/
noncomputable def floatBridgesTo_seBlockFullC {c h w r : Nat} (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' β esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β) :
    FloatBridgesTo (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂)
      (seBlockFullF (h := h) (w := w) M fsig W₁ b₁ W₂ b₂) :=
  FloatBridgesTo.seScale M
    (floatBridgesTo_seGateC M fsig W₁ b₁ W₂ b₂ hw' hβ hesig hhw hc hr hsig hW₁ hb₁ hW₂ hb₂) hn

/-- The batched SE block with the gate's sigmoid capped — `floatBridgesTo_seB`'s twin. -/
noncomputable def floatBridgesTo_seBC {c h w r : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ bb)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ bb) :
    FloatBridgesTo (seB N (h := h) (w := w) W₁ b₁ W₂ b₂)
      (seBF N (h := h) (w := w) M fsig W₁ b₁ W₂ b₂) := by
  unfold seB seBF
  exact FloatBridgesTo.batchMap N
    (floatBridgesTo_seBlockFullC M fsig W₁ b₁ W₂ b₂ hw' hbb hesig hhw hc hr hn hsig
      hW₁ hb₁ hW₂ hb₂)

/-- This SE gate's bridge with the sigmoid capped — `EnetSE.bridge`'s twin on the same map. -/
noncomputable def EnetSE.bridgeC {c h w r : Nat} (Z : EnetSE c r w' β') (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w) :
    FloatBridgesTo (seB N (h := h) (w := w) Z.W₁ Z.b₁ Z.W₂ Z.b₂)
      (seBF N (h := h) (w := w) M D.sig Z.W₁ Z.b₁ Z.W₂ Z.b₂) :=
  floatBridgesTo_seBC N M D.sig Z.W₁ Z.b₁ Z.W₂ Z.b₂ P.hw' P.hβ' P.hesig hhw hc hr hn
    D.spec Z.hW₁ Z.hb₁ Z.hW₂ Z.hb₂

/-- ⭐ **This SE gate's numeric envelope, capped at the sigmoid** — `EnetSE.maps` with the
    sigmoid's error clause `esig + E4/4 ≤ Eg` replaced by the cap's `2·Cg ≤ Eg`. The squeeze
    path's four stages (`GAP → dense → swish → dense`) still carry their honest windows and
    errors, all linear in the block window; what never forms is `Eg`'s dependence on `E4`, so
    the rescale's `mulErr q Ā Cg Ē Eg` is linear in `Ā` and `Ē`. -/
theorem EnetSE.mapsC {c h w r : Nat} (Z : EnetSE c r w' β') (N : Nat) (M : FloatModel)
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
    (sgA : 1 + esig ≤ Cg) (sgC : 2 * Cg ≤ Eg)
    (scA : Ā * Cg + q * (Ā * Cg) ≤ Ā') (scE : FloatModel.mulErr q Ā Cg Ē Eg ≤ Ē') :
    (Z.bridgeC N M D P hhw hc hr hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.gap (c := c) (h := h) (w := w) M hc hhw
    P.hq (M.u_nonneg.trans P.hq) hgG0 hgG qA qE
  have s2 := s1.comp hc (FloatBridgesTo.Maps.dense M Z.W₁ Z.b₁ P.hw' P.hβ' hc
    Z.hW₁ Z.hb₁ hg1 d1A d1E)
  have s3 := s2.comp hr (FloatBridgesTo.Maps.swish (n := r) M D.sig P.hesig D.spec
    P.hq hA20 swA swE)
  have s4 := s3.comp hr (FloatBridgesTo.Maps.dense M Z.W₂ Z.b₂ P.hw' P.hβ' hr
    Z.hW₂ Z.hb₂ hg2 d2A d2E)
  have s5 := s4.comp hc (FloatBridgesTo.Maps.capped
    (b := floatBridgesTo_sigmoid (n := c) D.sig P.hesig D.spec) (Ē := E4)
    (fun _ _ _ => sgA) sgC)
  have s6 := s5.comp hc (FloatBridgesTo.Maps.broadcast (c := c) (h := h) (w := w))
  exact FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.seScale M hn s6 P.hq hĀ0 scA scE)

-- ════════════════════════════════════════════════════════════════
-- § The four block shapes with the gate capped: bridges and envelopes
-- ════════════════════════════════════════════════════════════════

/-- The MBConv1 block's bridge with the gate's sigmoid capped — the same map as
    `EnetNoExpBlk.bridge`, assembled from the stage bridges with `floatBridgesTo_seBC` in the SE
    slot (`floatBridgesTo_mbNoExpFwdBGen`'s association). -/
noncomputable def EnetNoExpBlk.bridgeC {ic oc r kHd kWd h w : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hic : 0 < ic) (hoc : 0 < oc) (hr : 0 < r) (hn : 0 < ic * h * w) :
    FloatBridgesTo (B.fwd (h := h) (w := w) N ε) (B.fwdF (h := h) (w := w) N M D Rq) :=
  ((floatBridgesTo_dwbsBGen N M D.sig B.dw.W B.dw.b (B.bnd.fwd N ε h w) (B.bnd.fwdF N M Rq h w)
      P.hw' P.hβ' P.hesig hn D.spec B.dw.hW B.dw.hb (B.bnd.bridge N M Rq P hic hhw)).comp
    (B.se.bridgeC N M D P hhw hic hr hn (h := h) (w := w))).comp
    (floatBridgesTo_projBGen N M B.pr.W B.pr.b (B.bnp.fwd N ε h w) (B.bnp.fwdF N M Rq h w)
      P.hw' P.hβ' hn B.pr.hW B.pr.hb (B.bnp.bridge N M Rq P hoc hhw))

/-- **The MBConv1 envelope with the gate capped** — `EnetNoExpBlk.maps`'s five stages, the SE
    site supplied as a capped envelope (`EnetSE.mapsC` discharges it). -/
theorem EnetNoExpBlk.mapsC {ic oc r kHd kWd h w : Nat}
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
    (hse : (B.se.bridgeC N M D P hhw hic hr hn (h := h) (w := w)).Maps A3 E3 A4 E4)
    (pA : (1 + gp) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * A4 + β') ≤ A5)
    (pE : gp * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (A4 + E4) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * E4 ≤ E5)
    (pnA : G * ((A5 + Mb) * S) + Bb + bnNormBudget q (A5 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A5 + Mb) S G Bb 0 es + G * S * E5 ≤ Ē') :
    (B.bridgeC N M D Rq P hhw hic hoc hr hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' := by
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

/-- The stride-2 block's bridge with the gate's sigmoid capped
    (`floatBridgesTo_mbStridedFwdBGen`'s association). -/
noncomputable def EnetMBBlk.stridedBridgeC {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hhw2 : 0 < (2 * h) * (2 * w)) (hmid : 0 < mid) (hoc : 0 < oc) (hr : 0 < r)
    (hnE : 0 < ic * (2 * h) * (2 * w)) (hnD : 0 < mid * (2 * h) * (2 * w))
    (hn : 0 < mid * h * w) :
    FloatBridgesTo (B.stridedFwd (h := h) (w := w) N ε)
      (B.stridedFwdF (h := h) (w := w) N M D Rq) :=
  (((floatBridgesTo_cbsBGen N M D.sig B.ex.W B.ex.b (B.bne.fwd N ε (2 * h) (2 * w))
      (B.bne.fwdF N M Rq (2 * h) (2 * w)) P.hw' P.hβ' P.hesig hnE D.spec B.ex.hW B.ex.hb
      (B.bne.bridge N M Rq P hmid hhw2)).comp
    (floatBridgesTo_dwbsSBGen N M D.sig B.dw.W B.dw.b (B.bnd.fwd N ε h w) (B.bnd.fwdF N M Rq h w)
      P.hw' P.hβ' P.hesig hnD D.spec B.dw.hW B.dw.hb (B.bnd.bridge N M Rq P hmid hhw))).comp
    (B.se.bridgeC N M D P hhw hmid hr hn (h := h) (w := w))).comp
    (floatBridgesTo_projBGen N M B.pr.W B.pr.b (B.bnp.fwd N ε h w) (B.bnp.fwdF N M Rq h w)
      P.hw' P.hβ' hn B.pr.hW B.pr.hb (B.bnp.bridge N M Rq P hoc hhw))

/-- **The stride-2 envelope with the gate capped** — `EnetMBBlk.stridedMaps`'s nine stages. -/
theorem EnetMBBlk.stridedMapsC {ic mid oc r kHd kWd h w : Nat}
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
    (hse : (B.se.bridgeC N M D P hhw hmid hr hn (h := h) (w := w)).Maps A6 E6 A7 E7)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A7 + β') ≤ A8)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A7 + E7) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E7 ≤ E8)
    (pnA : G * ((A8 + Mb) * S) + Bb + bnNormBudget q (A8 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A8 + Mb) S G Bb 0 es + G * S * E8 ≤ Ē') :
    (B.stridedBridgeC N M D Rq P hhw hhw2 hmid hoc hr hnE hnD hn (h := h) (w := w)).Maps
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

/-- **The stride-1 no-skip block** (`ic ≠ oc`, stages 5 and 7 first block — `mbExpFwdB`'s
    shape): the residual block's body without the skip. The one MBConv form the representative
    has no instance of. -/
noncomputable def EnetMBBlk.expFwd {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projBGen N (h := h) (w := w) B.pr.W B.pr.b (B.bnp.fwd N ε h w) ∘
    seB N (h := h) (w := w) B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ ∘
    dwbsBGen N (h := h) (w := w) B.dw.W B.dw.b (B.bnd.fwd N ε h w) ∘
    cbsBGen N (h := h) (w := w) B.ex.W B.ex.b (B.bne.fwd N ε h w)

/-- The deployed float stride-1 no-skip block. -/
noncomputable def EnetMBBlk.expFwdF {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projBF N (h := h) (w := w) M B.pr.W B.pr.b (B.bnp.fwdF N M Rq h w) ∘
    seBF N (h := h) (w := w) M D.sig B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ ∘
    dwbsBF N (h := h) (w := w) M D.sig B.dw.W B.dw.b (B.bnd.fwdF N M Rq h w) ∘
    cbsBF N (h := h) (w := w) M D.sig B.ex.W B.ex.b (B.bne.fwdF N M Rq h w)

/-- The stride-1 no-skip block's bridge with the gate's sigmoid capped. -/
noncomputable def EnetMBBlk.expBridgeC {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hmid : 0 < mid) (hoc : 0 < oc) (hr : 0 < r)
    (hnI : 0 < ic * h * w) (hn : 0 < mid * h * w) :
    FloatBridgesTo (B.expFwd (h := h) (w := w) N ε) (B.expFwdF (h := h) (w := w) N M D Rq) :=
  (((floatBridgesTo_cbsBGen N M D.sig B.ex.W B.ex.b (B.bne.fwd N ε h w) (B.bne.fwdF N M Rq h w)
      P.hw' P.hβ' P.hesig hnI D.spec B.ex.hW B.ex.hb (B.bne.bridge N M Rq P hmid hhw)).comp
    (floatBridgesTo_dwbsBGen N M D.sig B.dw.W B.dw.b (B.bnd.fwd N ε h w) (B.bnd.fwdF N M Rq h w)
      P.hw' P.hβ' P.hesig hn D.spec B.dw.hW B.dw.hb (B.bnd.bridge N M Rq P hmid hhw))).comp
    (B.se.bridgeC N M D P hhw hmid hr hn (h := h) (w := w))).comp
    (floatBridgesTo_projBGen N M B.pr.W B.pr.b (B.bnp.fwd N ε h w) (B.bnp.fwdF N M Rq h w)
      P.hw' P.hβ' hn B.pr.hW B.pr.hb (B.bnp.bridge N M Rq P hoc hhw))

/-- **The stride-1 no-skip envelope with the gate capped** — the residual block's eight body
    stages and no skip fan-in. -/
theorem EnetMBBlk.expMapsC {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hmid : 0 < mid) (hoc : 0 < oc) (hr : 0 < r)
    (hnI : 0 < ic * h * w) (hn : 0 < mid * h * w)
    (hnb : 0 < N * (mid * h * w)) (hnbo : 0 < N * (oc * h * w))
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
    (hse : (B.se.bridgeC N M D P hhw hmid hr hn (h := h) (w := w)).Maps A6 E6 A7 E7)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A7 + β') ≤ A8)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A7 + E7) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E7 ≤ E8)
    (pnA : G * ((A8 + Mb) * S) + Bb + bnNormBudget q (A8 + Mb) S G Bb 0 es ≤ Ā')
    (pnE : bnNormBudget q (A8 + Mb) S G Bb 0 es + G * S * E8 ≤ Ē') :
    (B.expBridgeC N M D Rq P hhw hmid hoc hr hnI hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' := by
  have c1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M
    B.ex.W B.ex.b P.hw' P.hβ' hnI B.ex.hW B.ex.hb hge eA eE)
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
    B.pr.W B.pr.b P.hw' P.hβ' hn B.pr.hW B.pr.hb hgp pA pE)).comp hnbo
    (B.bnp.maps N M Rq P hoc hhw hA80 pnA pnE)
  exact b2.comp hnb pj

/-- The residual block's bridge with the gate's sigmoid capped — the capped body under the
    additive skip. The skip-add itself is not capped: it is one rounded sum, linear in both
    arguments, and where the block's own input window re-enters. -/
noncomputable def EnetMBBlk.residBridgeC {c mid r kHd kWd h w : Nat}
    (B : EnetMBBlk c mid c r kHd kWd w' β' G Bb Mb) (N : Nat) (M : FloatModel)
    (D : DeviceSigmoid esig) (Rq : DeviceRsqrt ε es)
    (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (hhw : 0 < h * w) (hc : 0 < c) (hmid : 0 < mid) (hr : 0 < r)
    (hnC : 0 < c * h * w) (hn : 0 < mid * h * w) :
    FloatBridgesTo (B.residFwd (h := h) (w := w) N ε)
      (B.residFwdF (h := h) (w := w) N M D Rq) :=
  FloatBridgesTo.residual M (B.expBridgeC N M D Rq P hhw hmid hc hr hnC hn (h := h) (w := w))

/-- **The residual envelope with the gate capped** — the capped body's eight stages, then the
    rounded skip fan-in against the block's own input window. -/
theorem EnetMBBlk.residMapsC {c mid r kHd kWd h w : Nat}
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
    (hse : (B.se.bridgeC N M D P hhw hmid hr hn (h := h) (w := w)).Maps A6 E6 A7 E7)
    (pA : (1 + gp) * (((mid * 1 * 1 : ℕ) : ℝ) * w' * A7 + β') ≤ A8)
    (pE : gp * (((mid * 1 * 1 : ℕ) : ℝ) * w' * (A7 + E7) + β')
            + ((mid * 1 * 1 : ℕ) : ℝ) * w' * E7 ≤ E8)
    (pnA : G * ((A8 + Mb) * S) + Bb + bnNormBudget q (A8 + Mb) S G Bb 0 es ≤ Bd)
    (pnE : bnNormBudget q (A8 + Mb) S G Bb 0 es + G * S * E8 ≤ Ed)
    (rA : Bd + Ā + q * (Bd + Ā) ≤ Ā') (rE : q * (Bd + Ed + Ā + Ē) + (Ed + Ē) ≤ Ē') :
    (B.residBridgeC N M D Rq P hhw hc hmid hr hnC hn (h := h) (w := w)).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.residual M hnbc
    (B.expMapsC N M D Rq P hhw hmid hc hr hnC hn hnb hnbc hge hgd hgp hA10 hA20 hA40 hA50 hA80
      eA eE enA enE eswA eswE dA dE dnA dnE dswA dswE hse pA pE pnA pnE) P.hq rA rE

-- ════════════════════════════════════════════════════════════════
-- § Every block in the ladder IS the committed inference block
-- ════════════════════════════════════════════════════════════════

/-! ⭐ Four `rfl`s, dimension-polymorphic, so each holds at every width in the table.
`mbNoExpFwdBEval` / `mbStridedFwdBEval` / `mbResidFwdBEval` and the four eval-stage
abbreviations (`EfficientNetRenderPCEval.lean`) are what the committed inference forward and its
faithful graph are built from; these say the blocks the number below folds over are those, at
the record's projections and at one shared `ε`. The no-skip widening has no block abbreviation
of its own in the eval render, so it is tied to the stage abbreviations directly — the body of
`mbResidFwdBEval` without `residual`, as MobileNetV2's t=1 block is tied to its two stages. -/

/-- The MBConv1 block is `mbNoExpFwdBEval`. -/
theorem EnetNoExpBlk.fwd_eq_eval {ic oc r kHd kWd h w : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    B.fwd (h := h) (w := w) N ε
      = mbNoExpFwdBEval N (h := h) (w := w) ε B.dw.W B.dw.b B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v
          B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v := rfl

/-- The stride-2 block is `mbStridedFwdBEval`. -/
theorem EnetMBBlk.stridedFwd_eq_eval {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    B.stridedFwd (h := h) (w := w) N ε
      = mbStridedFwdBEval N (h := h) (w := w) ε B.ex.W B.ex.b B.bne.γ B.bne.β B.bne.μ B.bne.v
          B.dw.W B.dw.b B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v
          B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v := rfl

/-- The residual block is `mbResidFwdBEval`. -/
theorem EnetMBBlk.residFwd_eq_eval {c mid r kHd kWd h w : Nat}
    (B : EnetMBBlk c mid c r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    B.residFwd (h := h) (w := w) N ε
      = mbResidFwdBEval N (h := h) (w := w) ε B.ex.W B.ex.b B.bne.γ B.bne.β B.bne.μ B.bne.v
          B.dw.W B.dw.b B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v
          B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ B.pr.W B.pr.b B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v := rfl

/-- The stride-1 no-skip widening is the project stage after the SE, the depthwise stage and
    the expand stage — `mbResidFwdBEval`'s body with no skip. -/
theorem EnetMBBlk.expFwd_eq_eval {ic mid oc r kHd kWd h w : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) (N : Nat) (ε : ℝ) :
    B.expFwd (h := h) (w := w) N ε
      = projBEval N (h := h) (w := w) B.pr.W B.pr.b ε B.bnp.γ B.bnp.β B.bnp.μ B.bnp.v ∘
          seB N (h := h) (w := w) B.se.W₁ B.se.b₁ B.se.W₂ B.se.b₂ ∘
          dwbsBEval N (h := h) (w := w) B.dw.W B.dw.b ε B.bnd.γ B.bnd.β B.bnd.μ B.bnd.v ∘
          cbsBEval N (h := h) (w := w) B.ex.W B.ex.b ε B.bne.γ B.bne.β B.bne.μ B.bne.v := rfl

-- ════════════════════════════════════════════════════════════════
-- § The whole paper net's stored parameters
-- ════════════════════════════════════════════════════════════════

/-- **The paper-spec EfficientNet-B0's stored parameters at one uniform profile** — stem, the
    sixteen MBConv blocks of the `[t,c,n,s,k]` table (`b1` the t=1 no-expand block,
    `b2/b4/b6/b12` the stride-2 downsamples, nine identity skips, `b9/b16` the stage-first
    widenings with no skip), the 1×1 head and the classifier. Forty-nine BatchNorm sites,
    sixteen squeeze-excite gates. ⭐ The widths, SE reductions and kernel sizes are
    `B0Weights`'s (`Architectures/EfficientNetFullB0.lean`) and the kinds and spatial sizes are
    `efficientnetForwardB_full`'s; the probe reads both and asserts they agree, block for block.
    The classifier is generic in the class count: `Maps.dense`'s envelope depends on the fan-in
    and never on the output count, so one theorem covers the 10-class and the 1000-class head. -/
structure EnetFullWeights (nCls : Nat) (w' β' G Bb Mb : ℝ) where
  stem : EnetConv 32 3 3 3 w' β'
  bns : EnetBn 32 G Bb Mb
  b1 : EnetNoExpBlk 32 16 8 3 3 w' β' G Bb Mb
  b2 : EnetMBBlk 16 96 24 4 3 3 w' β' G Bb Mb
  b3 : EnetMBBlk 24 144 24 6 3 3 w' β' G Bb Mb
  b4 : EnetMBBlk 24 144 40 6 5 5 w' β' G Bb Mb
  b5 : EnetMBBlk 40 240 40 10 5 5 w' β' G Bb Mb
  b6 : EnetMBBlk 40 240 80 10 3 3 w' β' G Bb Mb
  b7 : EnetMBBlk 80 480 80 20 3 3 w' β' G Bb Mb
  b8 : EnetMBBlk 80 480 80 20 3 3 w' β' G Bb Mb
  b9 : EnetMBBlk 80 480 112 20 5 5 w' β' G Bb Mb
  b10 : EnetMBBlk 112 672 112 28 5 5 w' β' G Bb Mb
  b11 : EnetMBBlk 112 672 112 28 5 5 w' β' G Bb Mb
  b12 : EnetMBBlk 112 672 192 28 5 5 w' β' G Bb Mb
  b13 : EnetMBBlk 192 1152 192 48 5 5 w' β' G Bb Mb
  b14 : EnetMBBlk 192 1152 192 48 5 5 w' β' G Bb Mb
  b15 : EnetMBBlk 192 1152 192 48 5 5 w' β' G Bb Mb
  b16 : EnetMBBlk 192 1152 320 48 3 3 w' β' G Bb Mb
  hd : EnetConv 1280 320 1 1 w' β'
  bnh : EnetBn 1280 G Bb Mb
  head : EnetHead 1280 nCls w' β'

variable {nCls : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The deployed paper-spec EfficientNet-B0 inference forward** — the `[t,c,n,s,k]` ladder of
    `efficientnetForwardB_full` with inference BatchNorm at every one of its 49 sites, written in
    the association the bridge below composes. -/
noncomputable def b0FullEvalForward (N : Nat) (W : EnetFullWeights nCls w' β' G Bb Mb) (ε : ℝ) :
    Vec (N * (3 * 224 * 224)) → Vec (N * nCls) :=
  headFwdBGen N (h := 7) (w := 7) W.hd.W W.hd.b (W.bnh.fwd N ε 7 7) W.head.W W.head.b ∘
    W.b16.expFwd (h := 7) (w := 7) N ε ∘
    W.b15.residFwd (h := 7) (w := 7) N ε ∘
    W.b14.residFwd (h := 7) (w := 7) N ε ∘
    W.b13.residFwd (h := 7) (w := 7) N ε ∘
    W.b12.stridedFwd (h := 7) (w := 7) N ε ∘
    W.b11.residFwd (h := 14) (w := 14) N ε ∘
    W.b10.residFwd (h := 14) (w := 14) N ε ∘
    W.b9.expFwd (h := 14) (w := 14) N ε ∘
    W.b8.residFwd (h := 14) (w := 14) N ε ∘
    W.b7.residFwd (h := 14) (w := 14) N ε ∘
    W.b6.stridedFwd (h := 14) (w := 14) N ε ∘
    W.b5.residFwd (h := 28) (w := 28) N ε ∘
    W.b4.stridedFwd (h := 28) (w := 28) N ε ∘
    W.b3.residFwd (h := 56) (w := 56) N ε ∘
    W.b2.stridedFwd (h := 56) (w := 56) N ε ∘
    W.b1.fwd (h := 112) (w := 112) N ε ∘
    stemBGen N (h := 112) (w := 112) W.stem.W W.stem.b (W.bns.fwd N ε 112 112)

/-- **The deployed paper-spec EfficientNet-B0 float inference forward** — every concrete slot
    replaced by the model's rounded peer, every BN by the six rounded ops the emitter writes,
    every sigmoid by the device kernel. -/
noncomputable def b0FullEvalForwardF (N : Nat) (M : FloatModel) (D : DeviceSigmoid esig)
    (Rq : DeviceRsqrt ε es) (W : EnetFullWeights nCls w' β' G Bb Mb) :
    Vec (N * (3 * 224 * 224)) → Vec (N * nCls) :=
  headFwdBF N (h := 7) (w := 7) M D.sig W.hd.W W.hd.b W.head.W W.head.b
      (W.bnh.fwdF N M Rq 7 7) ∘
    W.b16.expFwdF (h := 7) (w := 7) N M D Rq ∘
    W.b15.residFwdF (h := 7) (w := 7) N M D Rq ∘
    W.b14.residFwdF (h := 7) (w := 7) N M D Rq ∘
    W.b13.residFwdF (h := 7) (w := 7) N M D Rq ∘
    W.b12.stridedFwdF (h := 7) (w := 7) N M D Rq ∘
    W.b11.residFwdF (h := 14) (w := 14) N M D Rq ∘
    W.b10.residFwdF (h := 14) (w := 14) N M D Rq ∘
    W.b9.expFwdF (h := 14) (w := 14) N M D Rq ∘
    W.b8.residFwdF (h := 14) (w := 14) N M D Rq ∘
    W.b7.residFwdF (h := 14) (w := 14) N M D Rq ∘
    W.b6.stridedFwdF (h := 14) (w := 14) N M D Rq ∘
    W.b5.residFwdF (h := 28) (w := 28) N M D Rq ∘
    W.b4.stridedFwdF (h := 28) (w := 28) N M D Rq ∘
    W.b3.residFwdF (h := 56) (w := 56) N M D Rq ∘
    W.b2.stridedFwdF (h := 56) (w := 56) N M D Rq ∘
    W.b1.fwdF (h := 112) (w := 112) N M D Rq ∘
    stemBF N (h := 112) (w := 112) M D.sig W.stem.W W.stem.b (W.bns.fwdF N M Rq 112 112)

set_option maxRecDepth 1000000 in
/-- ⭐ **The whole deployed paper-spec EfficientNet-B0 inference forward float-bridges TO its
    float peer** — a CLOSED `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left: the stem,
    the sixteen MBConv blocks (each with its squeeze-excite, the gate's sigmoid capped) and the
    head are each discharged by leaves. -/
noncomputable def b0FullEvalBridge (N : Nat) (M : FloatModel) (D : DeviceSigmoid esig)
    (Rq : DeviceRsqrt ε es) (P : EnetProfile M ε w' β' G Bb Mb es esig S q)
    (W : EnetFullWeights nCls w' β' G Bb Mb) :
    FloatBridgesTo (b0FullEvalForward N W ε) (b0FullEvalForwardF N M D Rq W) :=
  (((((((((((((((((floatBridgesTo_stemBGen N M D.sig W.stem.W W.stem.b (W.bns.fwd N ε 112 112)
      (W.bns.fwdF N M Rq 112 112) P.hw' P.hβ' P.hesig (by norm_num) D.spec W.stem.hW W.stem.hb
      (W.bns.bridge N M Rq P (by norm_num) (by norm_num)))
    |>.comp (W.b1.bridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 112) (w := 112)))
    |>.comp (W.b2.stridedBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.b3.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.b4.stridedBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b5.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b6.stridedBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b7.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b8.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b9.expBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b10.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b11.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.b12.stridedBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (W.b13.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (W.b14.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (W.b15.residBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (W.b16.expBridgeC N M D Rq P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (floatBridgesTo_headFwdBGen N M D.sig W.hd.W W.hd.b (W.bnh.fwd N ε 7 7)
      W.head.W W.head.b (W.bnh.fwdF N M Rq 7 7) P.hw' P.hβ' P.hesig (by norm_num) (by norm_num) (by norm_num) D.spec
      W.hd.hW W.hd.hb W.head.hW W.head.hb (W.bnh.bridge N M Rq P (by norm_num) (by norm_num)))

-- ════════════════════════════════════════════════════════════════
-- § The number
-- ════════════════════════════════════════════════════════════════

set_option exponentiation.threshold 400 in
set_option maxRecDepth 4000000 in
set_option maxHeartbeats 16000000 in
/-- ⭐ **The envelope, kernel-checked, at sixteen blocks.** Every numeric stage closed by two
    rational inequalities — except the sixteen sigmoids, whose modulus clause is the cap's
    `2·Cg ≤ Eg` and whose fold is never written down. Each γ-term goes through
    `FloatModel.gamma_num` so `norm_num` never evaluates a big power. Built bottom-up at block
    granularity (19 steps, each block folding its own squeeze-excite through `EnetSE.mapsC`);
    the closing `exact` is one structural comparison with `b0FullEvalBridge`'s definition.

    ⭐ `exponentiation.threshold 400`: the numerals here run to `10²⁸⁷`, and Lean's default
    threshold of 256 is the only thing that ever stood between a `norm_num` goal of this shape
    and its proof (the file header). Nothing else about this proof differs from the
    representative's `b0EvalBridge_maps`. -/
theorem b0FullEvalBridge_maps (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetFullWeights nCls (41/10) (41/10) (41/10) (41/10) (41/10)) :
    (b0FullEvalBridge N M D Rq (b0Profile_committed M hMu hε5) W).Maps 1 0
      (1886 * 10 ^ 276) (2416 * 10 ^ 284) := by
  have hP := b0Profile_committed M hMu hε5
  -- stem: 3×3/s2 conv at the XLA-SAME phase, inference BN, swish
  have t1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConvStride2Xla
    (h := 112) (w := 112) M W.stem.W W.stem.b hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb (M.gamma_num (q := 1729 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 1149 / 10 ^ 1) (Ē' := 1985 / 10 ^ 7) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t2 := t1.comp (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (W.bns.maps N M Rq hP (by norm_num) (by norm_num) (h := 112) (w := 112)
    (Ā' := 1547 * 10 ^ 2) (Ē' := 5174 / 10 ^ 3) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t3 := t2.comp (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (FloatBridgesTo.Maps.swish
    (n := N * (32 * 112 * 112)) M D.sig hP.hesig D.spec hP.hq (by norm_num)
    (Ā' := 1563 * 10 ^ 2) (Ē' := 1563 * 10 ^ 2) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b1: MBConv1, no expand (32 → 16), 3×3 depthwise, SE at r = 8
  have t4 := t3.comp (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (W.b1.mapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (Nat.mul_pos hN (by norm_num : 0 < 32 * 112 * 112)) (Nat.mul_pos hN (by norm_num : 0 < 16 * 112 * 112)) (h := 112) (w := 112)
    (gd := 6557 / 10 ^ 10) (gp := 2027 / 10 ^ 9)
    (A1 := 5768 * 10 ^ 3) (E1 := 5768 * 10 ^ 3)
    (A2 := 7497 * 10 ^ 6) (E2 := 7497 * 10 ^ 6)
    (A3 := 7572 * 10 ^ 6) (E3 := 1507 * 10 ^ 7)
    (A4 := 7648 * 10 ^ 6) (E4 := 6096 * 10 ^ 7)
    (A5 := 1004 * 10 ^ 9) (E5 := 7998 * 10 ^ 9)
    (Ā' := 1305 * 10 ^ 12) (Ē' := 1040 * 10 ^ 13)
    (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b1.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 112) (w := 112)
      (gG := 7483 / 10 ^ 7) (g1 := 2027 / 10 ^ 9) (g2 := 5961 / 10 ^ 10)
      (A1 := 7578 * 10 ^ 6) (E1 := 1508 * 10 ^ 7)
      (A2 := 9943 * 10 ^ 8) (E2 := 1979 * 10 ^ 9)
      (A3 := 1005 * 10 ^ 9) (E3 := 2984 * 10 ^ 9)
      (A4 := 3297 * 10 ^ 10) (E4 := 9788 * 10 ^ 10)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 7648 * 10 ^ 6) (Ē' := 6096 * 10 ^ 7)
      (by norm_num) (M.gamma_num (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5961 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b2: MBConv6 stride-2 (16 → 96 → 24), 3×3 depthwise, SE at r = 4
  have t5 := t4.comp (Nat.mul_pos hN (by norm_num : 0 < 16 * 112 * 112)) (W.b2.stridedMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 96 * 112 * 112)) (Nat.mul_pos hN (by norm_num : 0 < 96 * 56 * 56)) (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56))
    (h := 56) (w := 56)
    (ge := 1073 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 5842 / 10 ^ 9)
    (A1 := 8561 * 10 ^ 13) (E1 := 6823 * 10 ^ 14)
    (A2 := 1113 * 10 ^ 17) (E2 := 8868 * 10 ^ 17)
    (A3 := 1125 * 10 ^ 17) (E3 := 9993 * 10 ^ 17)
    (A4 := 4152 * 10 ^ 18) (E4 := 3688 * 10 ^ 19)
    (A5 := 5397 * 10 ^ 21) (E5 := 4794 * 10 ^ 22)
    (A6 := 5451 * 10 ^ 21) (E6 := 5340 * 10 ^ 22)
    (A7 := 5506 * 10 ^ 21) (E7 := 1729 * 10 ^ 23)
    (A8 := 2168 * 10 ^ 24) (E8 := 6806 * 10 ^ 25)
    (Ā' := 2818 * 10 ^ 27) (Ē' := 8846 * 10 ^ 28)
    (M.gamma_num (q := 1073 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b2.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)
      (gG := 1871 / 10 ^ 7) (g1 := 5842 / 10 ^ 9) (g2 := 3577 / 10 ^ 10)
      (A1 := 5453 * 10 ^ 21) (E1 := 5341 * 10 ^ 22)
      (A2 := 2147 * 10 ^ 24) (E2 := 2103 * 10 ^ 25)
      (A3 := 2169 * 10 ^ 24) (E3 := 2320 * 10 ^ 25)
      (A4 := 3558 * 10 ^ 25) (E4 := 3805 * 10 ^ 26)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 5506 * 10 ^ 21) (Ē' := 1729 * 10 ^ 23)
      (by norm_num) (M.gamma_num (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3577 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b3: MBConv6 residual (24 → 144 → 24), 3×3 depthwise, SE at r = 6
  have t6 := t5.comp (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56)) (W.b3.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56)) (Nat.mul_pos hN (by norm_num : 0 < 144 * 56 * 56)) (h := 56) (w := 56)
    (ge := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 8703 / 10 ^ 9)
    (A1 := 2773 * 10 ^ 29) (E1 := 8705 * 10 ^ 30)
    (A2 := 3605 * 10 ^ 32) (E2 := 1132 * 10 ^ 34)
    (A3 := 3642 * 10 ^ 32) (E3 := 1169 * 10 ^ 34)
    (A4 := 1344 * 10 ^ 34) (E4 := 4314 * 10 ^ 35)
    (A5 := 1747 * 10 ^ 37) (E5 := 5607 * 10 ^ 38)
    (A6 := 1765 * 10 ^ 37) (E6 := 5784 * 10 ^ 38)
    (A7 := 1783 * 10 ^ 37) (E7 := 1789 * 10 ^ 39)
    (A8 := 1053 * 10 ^ 40) (E8 := 1057 * 10 ^ 42)
    (Bd := 1369 * 10 ^ 43) (Ed := 1374 * 10 ^ 45)
    (Ā' := 1370 * 10 ^ 43) (Ē' := 1375 * 10 ^ 45)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b3.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)
      (gG := 1871 / 10 ^ 7) (g1 := 8703 / 10 ^ 9) (g2 := 4769 / 10 ^ 10)
      (A1 := 1766 * 10 ^ 37) (E1 := 5785 * 10 ^ 38)
      (A2 := 1043 * 10 ^ 40) (E2 := 3416 * 10 ^ 41)
      (A3 := 1054 * 10 ^ 40) (E3 := 3522 * 10 ^ 41)
      (A4 := 2593 * 10 ^ 41) (E4 := 8665 * 10 ^ 42)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1783 * 10 ^ 37) (Ē' := 1789 * 10 ^ 39)
      (by norm_num) (M.gamma_num (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4769 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b4: MBConv6 stride-2 (24 → 144 → 40), 5×5 depthwise, SE at r = 6
  have t7 := t6.comp (Nat.mul_pos hN (by norm_num : 0 < 24 * 56 * 56)) (W.b4.stridedMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 144 * 56 * 56)) (Nat.mul_pos hN (by norm_num : 0 < 144 * 28 * 28)) (Nat.mul_pos hN (by norm_num : 0 < 40 * 28 * 28))
    (h := 28) (w := 28)
    (ge := 1550 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 8703 / 10 ^ 9)
    (A1 := 1349 * 10 ^ 45) (E1 := 1354 * 10 ^ 47)
    (A2 := 1754 * 10 ^ 48) (E2 := 1760 * 10 ^ 50)
    (A3 := 1772 * 10 ^ 48) (E3 := 1778 * 10 ^ 50)
    (A4 := 1817 * 10 ^ 50) (E4 := 1823 * 10 ^ 52)
    (A5 := 2362 * 10 ^ 53) (E5 := 2370 * 10 ^ 55)
    (A6 := 2386 * 10 ^ 53) (E6 := 2394 * 10 ^ 55)
    (A7 := 2410 * 10 ^ 53) (E7 := 7303 * 10 ^ 55)
    (A8 := 1423 * 10 ^ 56) (E8 := 4312 * 10 ^ 58)
    (Ā' := 1850 * 10 ^ 59) (Ē' := 5605 * 10 ^ 61)
    (M.gamma_num (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b4.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)
      (gG := 4680 / 10 ^ 8) (g1 := 8703 / 10 ^ 9) (g2 := 4769 / 10 ^ 10)
      (A1 := 2387 * 10 ^ 53) (E1 := 2395 * 10 ^ 55)
      (A2 := 1410 * 10 ^ 56) (E2 := 1415 * 10 ^ 58)
      (A3 := 1425 * 10 ^ 56) (E3 := 1430 * 10 ^ 58)
      (A4 := 3506 * 10 ^ 57) (E4 := 3518 * 10 ^ 59)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 2410 * 10 ^ 53) (Ē' := 7303 * 10 ^ 55)
      (by norm_num) (M.gamma_num (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4769 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b5: MBConv6 residual (40 → 240 → 40), 5×5 depthwise, SE at r = 10
  have t8 := t7.comp (Nat.mul_pos hN (by norm_num : 0 < 40 * 28 * 28)) (W.b5.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 40 * 28 * 28)) (Nat.mul_pos hN (by norm_num : 0 < 240 * 28 * 28)) (h := 28) (w := 28)
    (ge := 2504 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 1443 / 10 ^ 8)
    (A1 := 3035 * 10 ^ 61) (E1 := 9193 * 10 ^ 63)
    (A2 := 3945 * 10 ^ 64) (E2 := 1195 * 10 ^ 67)
    (A3 := 3985 * 10 ^ 64) (E3 := 1199 * 10 ^ 67)
    (A4 := 4085 * 10 ^ 66) (E4 := 1229 * 10 ^ 69)
    (A5 := 5310 * 10 ^ 69) (E5 := 1598 * 10 ^ 72)
    (A6 := 5364 * 10 ^ 69) (E6 := 1604 * 10 ^ 72)
    (A7 := 5418 * 10 ^ 69) (E7 := 4871 * 10 ^ 72)
    (A8 := 5332 * 10 ^ 72) (E8 := 4794 * 10 ^ 75)
    (Bd := 6931 * 10 ^ 75) (Ed := 6231 * 10 ^ 78)
    (Ā' := 6932 * 10 ^ 75) (Ē' := 6232 * 10 ^ 78)
    (M.gamma_num (q := 2504 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1443 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b5.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)
      (gG := 4680 / 10 ^ 8) (g1 := 1443 / 10 ^ 8) (g2 := 7153 / 10 ^ 10)
      (A1 := 5365 * 10 ^ 69) (E1 := 1605 * 10 ^ 72)
      (A2 := 5280 * 10 ^ 72) (E2 := 1580 * 10 ^ 75)
      (A3 := 5333 * 10 ^ 72) (E3 := 1586 * 10 ^ 75)
      (A4 := 2187 * 10 ^ 74) (E4 := 6503 * 10 ^ 76)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 5418 * 10 ^ 69) (Ē' := 4871 * 10 ^ 72)
      (by norm_num) (M.gamma_num (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1443 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 7153 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b6: MBConv6 stride-2 (40 → 240 → 80), 3×3 depthwise, SE at r = 10
  have t9 := t8.comp (Nat.mul_pos hN (by norm_num : 0 < 40 * 28 * 28)) (W.b6.stridedMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 240 * 28 * 28)) (Nat.mul_pos hN (by norm_num : 0 < 240 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 80 * 14 * 14))
    (h := 14) (w := 14)
    (ge := 2504 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 1443 / 10 ^ 8)
    (A1 := 1137 * 10 ^ 78) (E1 := 1023 * 10 ^ 81)
    (A2 := 1478 * 10 ^ 81) (E2 := 1330 * 10 ^ 84)
    (A3 := 1493 * 10 ^ 81) (E3 := 1332 * 10 ^ 84)
    (A4 := 5510 * 10 ^ 82) (E4 := 4916 * 10 ^ 85)
    (A5 := 7162 * 10 ^ 85) (E5 := 6390 * 10 ^ 88)
    (A6 := 7234 * 10 ^ 85) (E6 := 6398 * 10 ^ 88)
    (A7 := 7307 * 10 ^ 85) (E7 := 1941 * 10 ^ 89)
    (A8 := 7191 * 10 ^ 88) (E8 := 1910 * 10 ^ 92)
    (Ā' := 9347 * 10 ^ 91) (Ē' := 2483 * 10 ^ 95)
    (M.gamma_num (q := 2504 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1443 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b6.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)
      (gG := 1175 / 10 ^ 8) (g1 := 1443 / 10 ^ 8) (g2 := 7153 / 10 ^ 10)
      (A1 := 7235 * 10 ^ 85) (E1 := 6399 * 10 ^ 88)
      (A2 := 7120 * 10 ^ 88) (E2 := 6297 * 10 ^ 91)
      (A3 := 7192 * 10 ^ 88) (E3 := 6305 * 10 ^ 91)
      (A4 := 2949 * 10 ^ 90) (E4 := 2586 * 10 ^ 93)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 7307 * 10 ^ 85) (Ē' := 1941 * 10 ^ 89)
      (by norm_num) (M.gamma_num (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1443 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 7153 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b7: MBConv6 residual (80 → 480 → 80), 3×3 depthwise, SE at r = 20
  have t10 := t9.comp (Nat.mul_pos hN (by norm_num : 0 < 80 * 14 * 14)) (W.b7.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 80 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 480 * 14 * 14)) (h := 14) (w := 14)
    (ge := 4888 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 2874 / 10 ^ 8)
    (A1 := 3066 * 10 ^ 94) (E1 := 8145 * 10 ^ 97)
    (A2 := 3986 * 10 ^ 97) (E2 := 1059 * 10 ^ 101)
    (A3 := 4026 * 10 ^ 97) (E3 := 1060 * 10 ^ 101)
    (A4 := 1486 * 10 ^ 99) (E4 := 3912 * 10 ^ 102)
    (A5 := 1932 * 10 ^ 102) (E5 := 5085 * 10 ^ 105)
    (A6 := 1952 * 10 ^ 102) (E6 := 5087 * 10 ^ 105)
    (A7 := 1972 * 10 ^ 102) (E7 := 1542 * 10 ^ 106)
    (A8 := 3882 * 10 ^ 105) (E8 := 3035 * 10 ^ 109)
    (Bd := 5046 * 10 ^ 108) (Ed := 3945 * 10 ^ 112)
    (Ā' := 5047 * 10 ^ 108) (Ē' := 3946 * 10 ^ 112)
    (M.gamma_num (q := 4888 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2874 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b7.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)
      (gG := 1175 / 10 ^ 8) (g1 := 2874 / 10 ^ 8) (g2 := 1312 / 10 ^ 9)
      (A1 := 1953 * 10 ^ 102) (E1 := 5088 * 10 ^ 105)
      (A2 := 3844 * 10 ^ 105) (E2 := 1002 * 10 ^ 109)
      (A3 := 3883 * 10 ^ 105) (E3 := 1003 * 10 ^ 109)
      (A4 := 3185 * 10 ^ 107) (E4 := 8225 * 10 ^ 110)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1972 * 10 ^ 102) (Ē' := 1542 * 10 ^ 106)
      (by norm_num) (M.gamma_num (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2874 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1312 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b8: MBConv6 residual (80 → 480 → 80), 3×3 depthwise, SE at r = 20
  have t11 := t10.comp (Nat.mul_pos hN (by norm_num : 0 < 80 * 14 * 14)) (W.b8.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 80 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 480 * 14 * 14)) (h := 14) (w := 14)
    (ge := 4888 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (gp := 2874 / 10 ^ 8)
    (A1 := 1656 * 10 ^ 111) (E1 := 1295 * 10 ^ 115)
    (A2 := 2153 * 10 ^ 114) (E2 := 1684 * 10 ^ 118)
    (A3 := 2175 * 10 ^ 114) (E3 := 1685 * 10 ^ 118)
    (A4 := 8026 * 10 ^ 115) (E4 := 6218 * 10 ^ 119)
    (A5 := 1044 * 10 ^ 119) (E5 := 8082 * 10 ^ 122)
    (A6 := 1055 * 10 ^ 119) (E6 := 8084 * 10 ^ 122)
    (A7 := 1066 * 10 ^ 119) (E7 := 2450 * 10 ^ 123)
    (A8 := 2098 * 10 ^ 122) (E8 := 4822 * 10 ^ 126)
    (Bd := 2727 * 10 ^ 125) (Ed := 6268 * 10 ^ 129)
    (Ā' := 2728 * 10 ^ 125) (Ē' := 6269 * 10 ^ 129)
    (M.gamma_num (q := 4888 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2874 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b8.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)
      (gG := 1175 / 10 ^ 8) (g1 := 2874 / 10 ^ 8) (g2 := 1312 / 10 ^ 9)
      (A1 := 1056 * 10 ^ 119) (E1 := 8085 * 10 ^ 122)
      (A2 := 2079 * 10 ^ 122) (E2 := 1592 * 10 ^ 126)
      (A3 := 2100 * 10 ^ 122) (E3 := 1593 * 10 ^ 126)
      (A4 := 1723 * 10 ^ 124) (E4 := 1307 * 10 ^ 128)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1066 * 10 ^ 119) (Ē' := 2450 * 10 ^ 123)
      (by norm_num) (M.gamma_num (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2874 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1312 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b9: MBConv6 stride-1, no skip (80 → 480 → 112), 5×5 depthwise, SE at r = 20
  have t12 := t11.comp (Nat.mul_pos hN (by norm_num : 0 < 80 * 14 * 14)) (W.b9.expMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 480 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 112 * 14 * 14)) (h := 14) (w := 14)
    (ge := 4888 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 2874 / 10 ^ 8)
    (A1 := 8948 * 10 ^ 127) (E1 := 2057 * 10 ^ 132)
    (A2 := 1164 * 10 ^ 131) (E2 := 2674 * 10 ^ 135)
    (A3 := 1176 * 10 ^ 131) (E3 := 2675 * 10 ^ 135)
    (A4 := 1206 * 10 ^ 133) (E4 := 2742 * 10 ^ 137)
    (A5 := 1568 * 10 ^ 136) (E5 := 3564 * 10 ^ 140)
    (A6 := 1584 * 10 ^ 136) (E6 := 3565 * 10 ^ 140)
    (A7 := 1600 * 10 ^ 136) (E7 := 1081 * 10 ^ 141)
    (A8 := 3149 * 10 ^ 139) (E8 := 2128 * 10 ^ 144)
    (Ā' := 4093 * 10 ^ 142) (Ē' := 2766 * 10 ^ 147)
    (M.gamma_num (q := 4888 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2874 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b9.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)
      (gG := 1175 / 10 ^ 8) (g1 := 2874 / 10 ^ 8) (g2 := 1312 / 10 ^ 9)
      (A1 := 1585 * 10 ^ 136) (E1 := 3566 * 10 ^ 140)
      (A2 := 3120 * 10 ^ 139) (E2 := 7019 * 10 ^ 143)
      (A3 := 3152 * 10 ^ 139) (E3 := 7020 * 10 ^ 143)
      (A4 := 2585 * 10 ^ 141) (E4 := 5757 * 10 ^ 145)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1600 * 10 ^ 136) (Ē' := 1081 * 10 ^ 141)
      (by norm_num) (M.gamma_num (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2874 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1312 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b10: MBConv6 residual (112 → 672 → 112), 5×5 depthwise, SE at r = 28
  have t13 := t12.comp (Nat.mul_pos hN (by norm_num : 0 < 112 * 14 * 14)) (W.b10.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 112 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 672 * 14 * 14)) (h := 14) (w := 14)
    (ge := 6795 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 4018 / 10 ^ 8)
    (A1 := 1880 * 10 ^ 145) (E1 := 1271 * 10 ^ 150)
    (A2 := 2444 * 10 ^ 148) (E2 := 1652 * 10 ^ 153)
    (A3 := 2469 * 10 ^ 148) (E3 := 1653 * 10 ^ 153)
    (A4 := 2531 * 10 ^ 150) (E4 := 1695 * 10 ^ 155)
    (A5 := 3290 * 10 ^ 153) (E5 := 2203 * 10 ^ 158)
    (A6 := 3323 * 10 ^ 153) (E6 := 2204 * 10 ^ 158)
    (A7 := 3357 * 10 ^ 153) (E7 := 6679 * 10 ^ 158)
    (A8 := 9250 * 10 ^ 156) (E8 := 1841 * 10 ^ 162)
    (Bd := 1203 * 10 ^ 160) (Ed := 2393 * 10 ^ 165)
    (Ā' := 1204 * 10 ^ 160) (Ē' := 2394 * 10 ^ 165)
    (M.gamma_num (q := 6795 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4018 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b10.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)
      (gG := 1175 / 10 ^ 8) (g1 := 4018 / 10 ^ 8) (g2 := 1789 / 10 ^ 9)
      (A1 := 3324 * 10 ^ 153) (E1 := 2205 * 10 ^ 158)
      (A2 := 9159 * 10 ^ 156) (E2 := 6076 * 10 ^ 161)
      (A3 := 9251 * 10 ^ 156) (E3 := 6077 * 10 ^ 161)
      (A4 := 1063 * 10 ^ 159) (E4 := 6977 * 10 ^ 163)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 3357 * 10 ^ 153) (Ē' := 6679 * 10 ^ 158)
      (by norm_num) (M.gamma_num (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4018 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1789 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b11: MBConv6 residual (112 → 672 → 112), 5×5 depthwise, SE at r = 28
  have t14 := t13.comp (Nat.mul_pos hN (by norm_num : 0 < 112 * 14 * 14)) (W.b11.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 112 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 672 * 14 * 14)) (h := 14) (w := 14)
    (ge := 6795 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 4018 / 10 ^ 8)
    (A1 := 5529 * 10 ^ 162) (E1 := 1100 * 10 ^ 168)
    (A2 := 7187 * 10 ^ 165) (E2 := 1430 * 10 ^ 171)
    (A3 := 7259 * 10 ^ 165) (E3 := 1431 * 10 ^ 171)
    (A4 := 7441 * 10 ^ 167) (E4 := 1467 * 10 ^ 173)
    (A5 := 9672 * 10 ^ 170) (E5 := 1907 * 10 ^ 176)
    (A6 := 9769 * 10 ^ 170) (E6 := 1908 * 10 ^ 176)
    (A7 := 9867 * 10 ^ 170) (E7 := 5782 * 10 ^ 176)
    (A8 := 2719 * 10 ^ 174) (E8 := 1594 * 10 ^ 180)
    (Bd := 3534 * 10 ^ 177) (Ed := 2072 * 10 ^ 183)
    (Ā' := 3535 * 10 ^ 177) (Ē' := 2073 * 10 ^ 183)
    (M.gamma_num (q := 6795 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4018 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b11.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)
      (gG := 1175 / 10 ^ 8) (g1 := 4018 / 10 ^ 8) (g2 := 1789 / 10 ^ 9)
      (A1 := 9770 * 10 ^ 170) (E1 := 1909 * 10 ^ 176)
      (A2 := 2692 * 10 ^ 174) (E2 := 5260 * 10 ^ 179)
      (A3 := 2719 * 10 ^ 174) (E3 := 5261 * 10 ^ 179)
      (A4 := 3122 * 10 ^ 176) (E4 := 6040 * 10 ^ 181)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 9867 * 10 ^ 170) (Ē' := 5782 * 10 ^ 176)
      (by norm_num) (M.gamma_num (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4018 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1789 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b12: MBConv6 stride-2 (112 → 672 → 192), 5×5 depthwise, SE at r = 28
  have t15 := t14.comp (Nat.mul_pos hN (by norm_num : 0 < 112 * 14 * 14)) (W.b12.stridedMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 672 * 14 * 14)) (Nat.mul_pos hN (by norm_num : 0 < 672 * 7 * 7)) (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7))
    (h := 7) (w := 7)
    (ge := 6795 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (gp := 4018 / 10 ^ 8)
    (A1 := 1624 * 10 ^ 180) (E1 := 9520 * 10 ^ 185)
    (A2 := 2111 * 10 ^ 183) (E2 := 1238 * 10 ^ 189)
    (A3 := 2133 * 10 ^ 183) (E3 := 1239 * 10 ^ 189)
    (A4 := 2187 * 10 ^ 185) (E4 := 1270 * 10 ^ 191)
    (A5 := 2843 * 10 ^ 188) (E5 := 1651 * 10 ^ 194)
    (A6 := 2872 * 10 ^ 188) (E6 := 1652 * 10 ^ 194)
    (A7 := 2901 * 10 ^ 188) (E7 := 5006 * 10 ^ 194)
    (A8 := 7994 * 10 ^ 191) (E8 := 1380 * 10 ^ 198)
    (Ā' := 1040 * 10 ^ 195) (Ē' := 1794 * 10 ^ 201)
    (M.gamma_num (q := 6795 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4018 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b12.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)
      (gG := 2981 / 10 ^ 9) (g1 := 4018 / 10 ^ 8) (g2 := 1789 / 10 ^ 9)
      (A1 := 2873 * 10 ^ 188) (E1 := 1653 * 10 ^ 194)
      (A2 := 7917 * 10 ^ 191) (E2 := 4555 * 10 ^ 197)
      (A3 := 7997 * 10 ^ 191) (E3 := 4556 * 10 ^ 197)
      (A4 := 9181 * 10 ^ 193) (E4 := 5231 * 10 ^ 199)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 2901 * 10 ^ 188) (Ē' := 5006 * 10 ^ 194)
      (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 4018 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1789 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b13: MBConv6 residual (192 → 1152 → 192), 5×5 depthwise, SE at r = 48
  have t16 := t15.comp (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (W.b13.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (Nat.mul_pos hN (by norm_num : 0 < 1152 * 7 * 7)) (h := 7) (w := 7)
    (ge := 1157 / 10 ^ 8) (gd := 1610 / 10 ^ 9) (gp := 6879 / 10 ^ 8)
    (A1 := 8187 * 10 ^ 197) (E1 := 1413 * 10 ^ 204)
    (A2 := 1065 * 10 ^ 201) (E2 := 1837 * 10 ^ 207)
    (A3 := 1076 * 10 ^ 201) (E3 := 1838 * 10 ^ 207)
    (A4 := 1103 * 10 ^ 203) (E4 := 1884 * 10 ^ 209)
    (A5 := 1434 * 10 ^ 206) (E5 := 2449 * 10 ^ 212)
    (A6 := 1449 * 10 ^ 206) (E6 := 2450 * 10 ^ 212)
    (A7 := 1464 * 10 ^ 206) (E7 := 7424 * 10 ^ 212)
    (A8 := 6916 * 10 ^ 209) (E8 := 3507 * 10 ^ 216)
    (Bd := 8990 * 10 ^ 212) (Ed := 4559 * 10 ^ 219)
    (Ā' := 8991 * 10 ^ 212) (Ē' := 4560 * 10 ^ 219)
    (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b13.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)
      (gG := 2981 / 10 ^ 9) (g1 := 6879 / 10 ^ 8) (g2 := 2981 / 10 ^ 9)
      (A1 := 1450 * 10 ^ 206) (E1 := 2451 * 10 ^ 212)
      (A2 := 6850 * 10 ^ 209) (E2 := 1158 * 10 ^ 216)
      (A3 := 6919 * 10 ^ 209) (E3 := 1159 * 10 ^ 216)
      (A4 := 1362 * 10 ^ 212) (E4 := 2281 * 10 ^ 218)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1464 * 10 ^ 206) (Ē' := 7424 * 10 ^ 212)
      (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b14: MBConv6 residual (192 → 1152 → 192), 5×5 depthwise, SE at r = 48
  have t17 := t16.comp (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (W.b14.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (Nat.mul_pos hN (by norm_num : 0 < 1152 * 7 * 7)) (h := 7) (w := 7)
    (ge := 1157 / 10 ^ 8) (gd := 1610 / 10 ^ 9) (gp := 6879 / 10 ^ 8)
    (A1 := 7078 * 10 ^ 215) (E1 := 3590 * 10 ^ 222)
    (A2 := 9200 * 10 ^ 218) (E2 := 4666 * 10 ^ 225)
    (A3 := 9293 * 10 ^ 218) (E3 := 4667 * 10 ^ 225)
    (A4 := 9526 * 10 ^ 220) (E4 := 4784 * 10 ^ 227)
    (A5 := 1239 * 10 ^ 224) (E5 := 6218 * 10 ^ 230)
    (A6 := 1252 * 10 ^ 224) (E6 := 6219 * 10 ^ 230)
    (A7 := 1265 * 10 ^ 224) (E7 := 1885 * 10 ^ 231)
    (A8 := 5976 * 10 ^ 227) (E8 := 8904 * 10 ^ 234)
    (Bd := 7768 * 10 ^ 230) (Ed := 1158 * 10 ^ 238)
    (Ā' := 7769 * 10 ^ 230) (Ē' := 1159 * 10 ^ 238)
    (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b14.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)
      (gG := 2981 / 10 ^ 9) (g1 := 6879 / 10 ^ 8) (g2 := 2981 / 10 ^ 9)
      (A1 := 1253 * 10 ^ 224) (E1 := 6220 * 10 ^ 230)
      (A2 := 5919 * 10 ^ 227) (E2 := 2939 * 10 ^ 234)
      (A3 := 5979 * 10 ^ 227) (E3 := 2940 * 10 ^ 234)
      (A4 := 1177 * 10 ^ 230) (E4 := 5786 * 10 ^ 236)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1265 * 10 ^ 224) (Ē' := 1885 * 10 ^ 231)
      (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b15: MBConv6 residual (192 → 1152 → 192), 5×5 depthwise, SE at r = 48
  have t18 := t17.comp (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (W.b15.residMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (Nat.mul_pos hN (by norm_num : 0 < 1152 * 7 * 7)) (h := 7) (w := 7)
    (ge := 1157 / 10 ^ 8) (gd := 1610 / 10 ^ 9) (gp := 6879 / 10 ^ 8)
    (A1 := 6116 * 10 ^ 233) (E1 := 9124 * 10 ^ 240)
    (A2 := 7950 * 10 ^ 236) (E2 := 1186 * 10 ^ 244)
    (A3 := 8030 * 10 ^ 236) (E3 := 1187 * 10 ^ 244)
    (A4 := 8231 * 10 ^ 238) (E4 := 1217 * 10 ^ 246)
    (A5 := 1070 * 10 ^ 242) (E5 := 1582 * 10 ^ 249)
    (A6 := 1081 * 10 ^ 242) (E6 := 1583 * 10 ^ 249)
    (A7 := 1092 * 10 ^ 242) (E7 := 4797 * 10 ^ 249)
    (A8 := 5159 * 10 ^ 245) (E8 := 2266 * 10 ^ 253)
    (Bd := 6706 * 10 ^ 248) (Ed := 2946 * 10 ^ 256)
    (Ā' := 6707 * 10 ^ 248) (Ē' := 2947 * 10 ^ 256)
    (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b15.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)
      (gG := 2981 / 10 ^ 9) (g1 := 6879 / 10 ^ 8) (g2 := 2981 / 10 ^ 9)
      (A1 := 1082 * 10 ^ 242) (E1 := 1584 * 10 ^ 249)
      (A2 := 5111 * 10 ^ 245) (E2 := 7483 * 10 ^ 252)
      (A3 := 5163 * 10 ^ 245) (E3 := 7484 * 10 ^ 252)
      (A4 := 1017 * 10 ^ 248) (E4 := 1473 * 10 ^ 255)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 1092 * 10 ^ 242) (Ē' := 4797 * 10 ^ 249)
      (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- b16: MBConv6 stride-1, no skip (192 → 1152 → 320), 3×3 depthwise, SE at r = 48
  have t19 := t18.comp (Nat.mul_pos hN (by norm_num : 0 < 192 * 7 * 7)) (W.b16.expMapsC N M D Rq hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (Nat.mul_pos hN (by norm_num : 0 < 1152 * 7 * 7)) (Nat.mul_pos hN (by norm_num : 0 < 320 * 7 * 7)) (h := 7) (w := 7)
    (ge := 1157 / 10 ^ 8) (gd := 6557 / 10 ^ 10) (gp := 6879 / 10 ^ 8)
    (A1 := 5280 * 10 ^ 251) (E1 := 2320 * 10 ^ 259)
    (A2 := 6863 * 10 ^ 254) (E2 := 3016 * 10 ^ 262)
    (A3 := 6932 * 10 ^ 254) (E3 := 3017 * 10 ^ 262)
    (A4 := 2558 * 10 ^ 256) (E4 := 1114 * 10 ^ 264)
    (A5 := 3325 * 10 ^ 259) (E5 := 1448 * 10 ^ 267)
    (A6 := 3359 * 10 ^ 259) (E6 := 1449 * 10 ^ 267)
    (A7 := 3393 * 10 ^ 259) (E7 := 4391 * 10 ^ 267)
    (A8 := 1603 * 10 ^ 263) (E8 := 2075 * 10 ^ 271)
    (Ā' := 2084 * 10 ^ 266) (Ē' := 2697 * 10 ^ 274)
    (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
    (W.b16.se.mapsC N M D hP (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)
      (gG := 2981 / 10 ^ 9) (g1 := 6879 / 10 ^ 8) (g2 := 2981 / 10 ^ 9)
      (A1 := 3360 * 10 ^ 259) (E1 := 1450 * 10 ^ 267)
      (A2 := 1588 * 10 ^ 263) (E2 := 6850 * 10 ^ 270)
      (A3 := 1604 * 10 ^ 263) (E3 := 6851 * 10 ^ 270)
      (A4 := 3157 * 10 ^ 265) (E4 := 1349 * 10 ^ 273)
      (Cg := 1010 / 10 ^ 3) (Eg := 2020 / 10 ^ 3)
      (Ā' := 3393 * 10 ^ 259) (Ē' := 4391 * 10 ^ 267)
      (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  -- head: 1×1 conv (320 → 1280) at 7×7, inference BN, swish, GAP, dense
  have h1 := FloatBridgesTo.Maps.batchMap N (FloatBridgesTo.Maps.flatConv (h := 7) (w := 7) M
    W.hd.W W.hd.b hP.hw' hP.hβ' (by norm_num) W.hd.hW W.hd.hb (M.gamma_num (q := 1920 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 2084 * 10 ^ 266) (Ē := 2697 * 10 ^ 274)
    (Ā' := 2735 * 10 ^ 269) (Ē' := 3539 * 10 ^ 277) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have h2 := h1.comp (Nat.mul_pos hN (by norm_num : 0 < 1280 * 7 * 7)) (W.bnh.maps N M Rq hP (by norm_num) (by norm_num) (h := 7) (w := 7)
    (Ā' := 3555 * 10 ^ 272) (Ē' := 4600 * 10 ^ 280) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have h3 := h2.comp (Nat.mul_pos hN (by norm_num : 0 < 1280 * 7 * 7)) (FloatBridgesTo.Maps.swish
    (n := N * (1280 * 7 * 7)) M D.sig hP.hesig D.spec hP.hq (by norm_num)
    (Ā' := 3591 * 10 ^ 272) (Ē' := 4601 * 10 ^ 280) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have h4 := h3.comp (Nat.mul_pos hN (by norm_num : 0 < 1280 * 7 * 7)) (FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.gap (c := 1280) (h := 7) (w := 7) M (by norm_num) (by norm_num) hMu
      (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 3592 * 10 ^ 272) (Ē' := 4602 * 10 ^ 280) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])))
  have h5 := h4.comp (Nat.mul_pos hN (by norm_num : 0 < 1280)) (FloatBridgesTo.Maps.batchMap N
    (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num) W.head.hW W.head.hb
      (M.gamma_num (q := 7642 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 1886 * 10 ^ 276) (Ē' := 2416 * 10 ^ 284) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])))
  exact t19.comp (Nat.mul_pos hN (by norm_num : 0 < 320 * 7 * 7)) h5

/-- The paper-net inference bridge's certified output window at the committed profile:
    `≤ 1.886e279`. ⭐ Honest and uncapped: no cap touches a window, no operating point is
    taken (`S = 317` is the ε-floor), and swish never resets it — this is what sixteen MBConv
    blocks cost at the interval face. -/
theorem b0FullEvalBridge_mag_le (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetFullWeights nCls (41/10) (41/10) (41/10) (41/10) (41/10)) :
    (b0FullEvalBridge N M D Rq (b0Profile_committed M hMu hε5) W).mag 1 ≤ 1886 * 10 ^ 276 :=
  (b0FullEvalBridge_maps N hN M hMu hε5 D Rq W).mag_le 1 (by norm_num) le_rfl

/-- The paper-net inference bridge's fresh budget at the committed profile: `≤ 2.416e287`.
    ⛔ CAPPED at the sigmoid of all sixteen squeeze-excite gates — see the file header for what
    that claim is and is not. Uncapped the same chain folds to `10^1897907`. -/
theorem b0FullEvalBridge_fresh_le (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32)
    {ε : ℝ} (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetFullWeights nCls (41/10) (41/10) (41/10) (41/10) (41/10)) :
    (b0FullEvalBridge N M D Rq (b0Profile_committed M hMu hε5) W).fresh 1 ≤ 2416 * 10 ^ 284 :=
  (b0FullEvalBridge_maps N hN M hMu hε5 D Rq W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed sixteen-block EfficientNet-B0 inference forward is within `2.416e287`
    of the certified real forward, per logit**, on inputs of magnitude `≤ 1`, at the measured
    parameter profile (`|·| ≤ 41/10`), for `ε ≥ 10⁻⁵`, any device `rsqrt` and `sigmoid` accurate
    to `10⁻²`, any batch size, any class count, and any rounding model at binary32 accuracy. The
    paper-spec peer of `b0_float_logits_le`. ⛔ CAP at the sixteen gates: see the header. -/
theorem b0Full_float_logits_le (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetFullWeights nCls (41/10) (41/10) (41/10) (41/10) (41/10))
    (x : Vec (N * (3 * 224 * 224))) (hx : ∀ k, |x k| ≤ 1) (j : Fin (N * nCls)) :
    |b0FullEvalForwardF N M D Rq W x j - b0FullEvalForward N W ε x j| ≤ 2416 * 10 ^ 284 :=
  (b0FullEvalBridge_maps N hN M hMu hε5 D Rq W).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed inference forward, and the graph denotes it
-- ════════════════════════════════════════════════════════════════

/-- The float-tier MBConv1 record's eval-net view: weights, γ/β and the two frozen statistics. -/

noncomputable def EnetNoExpBlk.toEval {ic oc r kHd kWd : Nat}
    (B : EnetNoExpBlk ic oc r kHd kWd w' β' G Bb Mb) : MBWNoExpEval ic oc r kHd kWd :=
  { dW := B.dw.W, db := B.dw.b, dγ := B.bnd.γ, dβ := B.bnd.β, dμ := B.bnd.μ, dv := B.bnd.v
    z1 := B.se.W₁, zb1 := B.se.b₁, z2 := B.se.W₂, zb2 := B.se.b₂
    pW := B.pr.W, pb := B.pr.b, pγ := B.bnp.γ, pβ := B.bnp.β, pμ := B.bnp.μ, pv := B.bnp.v }

/-- The float-tier MBConv6 record's eval-net view. -/
noncomputable def EnetMBBlk.toEval {ic mid oc r kHd kWd : Nat}
    (B : EnetMBBlk ic mid oc r kHd kWd w' β' G Bb Mb) : MBWEval ic mid oc r kHd kWd :=
  { eW := B.ex.W, eb := B.ex.b, eγ := B.bne.γ, eβ := B.bne.β, eμ := B.bne.μ, ev := B.bne.v
    dW := B.dw.W, db := B.dw.b, dγ := B.bnd.γ, dβ := B.bnd.β, dμ := B.bnd.μ, dv := B.bnd.v
    z1 := B.se.W₁, zb1 := B.se.b₁, z2 := B.se.W₂, zb2 := B.se.b₂
    pW := B.pr.W, pb := B.pr.b, pγ := B.bnp.γ, pβ := B.bnp.β, pμ := B.bnp.μ, pv := B.bnp.v }

/-- **The record-bundled weights as the eval net's weights** — one shared `ε` is the forward's
    argument in both, so nothing is lost. -/
noncomputable def EnetFullWeights.toEval (W : EnetFullWeights nCls w' β' G Bb Mb) :
    B0WeightsEval nCls :=
  { sW := W.stem.W, sb := W.stem.b, sγ := W.bns.γ, sβ := W.bns.β, sμ := W.bns.μ, sv := W.bns.v
    b1 := W.b1.toEval, b2 := W.b2.toEval, b3 := W.b3.toEval, b4 := W.b4.toEval
    b5 := W.b5.toEval, b6 := W.b6.toEval, b7 := W.b7.toEval, b8 := W.b8.toEval
    b9 := W.b9.toEval, b10 := W.b10.toEval, b11 := W.b11.toEval, b12 := W.b12.toEval
    b13 := W.b13.toEval, b14 := W.b14.toEval, b15 := W.b15.toEval, b16 := W.b16.toEval
    hW := W.hd.W, hb := W.hd.b, hγ := W.bnh.γ, hβ := W.bnh.β, hμ := W.bnh.μ, hv := W.bnh.v
    fcW := W.head.W, fcb := W.head.b }

/-- **The record-bundled forward IS the committed sixteen-block inference net.** ⚠ NOT one `rfl`:
    at these dims the kernel times out comparing the two whole nets (the three-block lesson).
    Rewriting with the per-stage `*Eval_eq_gen` lemmas first leaves nothing to compare — the block
    wrappers on both sides unfold to the same `*Gen` skeleton at
    `batchMap N (bnPerChannelEvalTensor3 …)`. -/
theorem b0FullEvalForward_eq_fullEval (N : Nat) (W : EnetFullWeights nCls w' β' G Bb Mb) (ε : ℝ)
    (x : Vec (N * (3 * 224 * 224))) :
    b0FullEvalForward N W ε x = efficientnetForwardB_fullEval N ε W.toEval x := by
  simp only [b0FullEvalForward, EnetNoExpBlk.fwd, EnetMBBlk.stridedFwd, EnetMBBlk.residFwd,
    EnetMBBlk.expFwd, EnetBn.fwd, efficientnetForwardB_fullEval, EnetFullWeights.toEval,
    EnetNoExpBlk.toEval, EnetMBBlk.toEval, mbNoExpEvalW, mbStridedEvalW, mbResidEvalW, mbExpEvalW,
    mbExpFwdBEval, stemBEval_eq_gen, mbNoExpFwdBEval_eq_gen, mbStridedFwdBEval_eq_gen,
    mbResidFwdBEval_eq_gen, projBEval_eq_gen, dwbsBEval_eq_gen, cbsBEval_eq_gen,
    headFwdBEval_eq_gen, Function.comp_apply]

/-- ⭐ **The whole loop closes.** The typed `SHlo` inference graph of the sixteen-block net denotes
    exactly the forward this file states its number about. -/
theorem b0FullEvalGraph_faithful (N : Nat) (epsStr : String) (ε : ℝ)
    (W : EnetFullWeights nCls w' β' G Bb Mb) (x : Vec (N * (3 * 224 * 224))) :
    StableHLO.den (StableHLO.efficientnetFwdGraphB_fullEval N epsStr ε W.toEval x)
      = b0FullEvalForward N W ε x :=
  (StableHLO.efficientnetFwdGraphB_fullEval_faithful N epsStr ε W.toEval x).trans
    (b0FullEvalForward_eq_fullEval N W ε x).symm

/-- ⭐⭐ **The number, stated about the committed sixteen-block inference forward.**
    `b0Full_float_logits_le` with `efficientnetForwardB_fullEval` on the real side instead of the
    record-bundled `b0FullEvalForward` — the claim is about the net `efficientnet_fwd_eval` renders,
    tied through `b0FullEvalGraph_faithful` rather than by inspection. ⛔ CAP at the sixteen gates;
    see the header. -/
theorem b0Full_float_logits_le_committed (N : Nat) (hN : 0 < N) (M : FloatModel) (hMu : M.u ≤ u32)
    {ε : ℝ} (hε5 : 1 / 100000 ≤ ε) (D : DeviceSigmoid (1/100)) (Rq : DeviceRsqrt ε (1/100))
    (W : EnetFullWeights nCls (41/10) (41/10) (41/10) (41/10) (41/10))
    (x : Vec (N * (3 * 224 * 224))) (hx : ∀ k, |x k| ≤ 1) (j : Fin (N * nCls)) :
    |b0FullEvalForwardF N M D Rq W x j - efficientnetForwardB_fullEval N ε W.toEval x j|
      ≤ 2416 * 10 ^ 284 := by
  rw [← b0FullEvalForward_eq_fullEval N W ε x]
  exact b0Full_float_logits_le N hN M hMu hε5 D Rq W x hx j

/-! ### Inhabitation

`b0Full_float_logits_le`'s record at the committed constants, `N = 1` and ten classes: zero
weights, zero running statistics, the exact sigmoid and `rsqrt` as the device kernels,
`binary32`, `ε = 1/100000`. -/
noncomputable def EnetFullWeights.zero : EnetFullWeights 10 (41/10) (41/10) (41/10) (41/10) (41/10) :=
  have h : (0:ℝ) ≤ 41/10 := by norm_num
  { stem := EnetConv.zero _ _ _ _ h h, bns := EnetBn.zero _ h h h
    b1 := EnetNoExpBlk.zero _ _ _ _ _ h h h h h
    b2 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b3 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b4 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b5 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b6 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b7 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b8 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b9 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b10 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b11 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b12 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b13 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b14 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b15 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    b16 := EnetMBBlk.zero _ _ _ _ _ _ h h h h h
    hd := EnetConv.zero _ _ _ _ h h, bnh := EnetBn.zero _ h h h, head := EnetHead.zero _ _ h h }

example (x : Vec (1 * (3 * 224 * 224))) (hx : ∀ k, |x k| ≤ 1) (j : Fin (1 * 10)) :
    |b0FullEvalForwardF 1 binary32 (DeviceSigmoid.exact (esig := 1/100) (by norm_num))
        (DeviceRsqrt.exact (1/100000) (es := 1/100) (by norm_num)) EnetFullWeights.zero x j
      - b0FullEvalForward 1 EnetFullWeights.zero (1/100000) x j| ≤ 2416 * 10 ^ 284 :=
  b0Full_float_logits_le 1 Nat.one_pos binary32 binary32_u.le (by norm_num) _ _ _ x hx j

end Proofs
