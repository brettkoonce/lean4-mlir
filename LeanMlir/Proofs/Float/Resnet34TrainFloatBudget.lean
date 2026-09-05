import LeanMlir.Proofs.Float.Resnet34FloatBudget
import LeanMlir.Proofs.Float.BnXhatFloatBridge
import LeanMlir.Proofs.Float.Binary32Instance

/-! # A NUMBER for ResNet-34 at TRAINING-mode BatchNorm — and it is the CAP

The **training-mode** twin of `Resnet34FloatBudget.lean`: the same `[3,4,6,3]` net at `224²`,
the same measured profile, the same 90 numeric stages — with `bnPerChannelTensor3` at all 36
BatchNorm sites where that file has `bnPerChannelEvalTensor3`. This is the program the repo
actually **trains** with, and the one the input-gradient numbers in
`Resnet34BackFloatBudget.lean` are taken through.

    output window  ≤ 8.748·10⁸⁰       (`r34TrainBridge_mag_le`)
    error bound    ≤ 1.752·10⁸¹       (`r34TrainBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 1.752·10⁸¹` (`r34_train_float_logits_le`).

⭐⭐ **140 orders below where this file landed, in two steps and neither of them a modelling
concession.** It was `3.176·10²²¹ / 6.349·10²²¹` on 2026-09-05; §0.1's ESCAPE 2 at the
per-channel BatchNorm took it to `4.304·10¹⁴⁵` (76 orders, the paragraph below), and DERIVING the
device batch mean's accuracy instead of supplying it took it here (65 more, the paragraph after).

⭐⭐ **76 orders of that came from §0.1's ESCAPE 2 (2026-09-05), at no new hypothesis.** The
first version of this file was `3.176·10²²¹ / 6.349·10²²¹`, charging each site's certified
window at `|x − μ|·|istd| ≤ 2A·S` — the product of the two factors' own bounds, `S = 317` at the
`ε`-floor. The normalised activation has a bound that mentions neither: **`|x̂| ≤ √n`**
(`bnXhat_sq_le`), which is `112` at the stem and `7` at the deepest block, and which four
backward budget files have called since 2026-09-03. `Maps.bnPerChannelTensor3CappedX`
(`BnXhatFloatBridge.lean`) is that leaf lifted per channel; the profile, the modelled accuracies
and the `ε`-floor are all unchanged. ⭐ r34's reduction widths are perfect squares, so the root
is EXACT here where ConvNeXt's channel counts need the ceiling one.

⛔⛔ **This number is the CAP, not the fold, and that is not a detail.** `budget / window =
2.00` is the tell, and escape 2 does not change it — that escape has a modulus half as well as
a window half, and the modulus half is priced and NOT taken here (`BnXhatFloatBridge.lean`'s
header), which costs nothing precisely because every site is capped. All 36 BatchNorm sites go through `FloatBridgesTo.capped`, so what is proved
is *"the float and the real forward both land in the certified window"* — the triangle
inequality — and **not** *"the rounding error folds to this"*, which is what
`r34_float_logits_le` says about the inference net. The two numbers must never be tabled
together without that label (`planning/archive/float_budget_numbers_log.md` §9), and this one must never be
quoted as though it were the inference number at a different `ε`.

⭐⭐ **Why it exists, when `Resnet34FloatBudget.lean` says it cannot.** That file's header
says training-mode BatchNorm reaches `~10⁷⁴¹⁷`, *"past the point where `norm_num` will evaluate
the numeral at all"*, and concludes there is no theorem to state. The premise is right and the
conclusion does not follow: `10⁷⁴¹⁷` is the **fold's** numeral, and the same measurement puts
the **window** at `10²²¹`. `FloatBridgesTo.capped` bounds any modulus by `2·mag`, so a statable
window is already a statable theorem — and it stops the squaring cold, because `mag` depends on
the input window and not on the inherited error, so a capped site resets the error however big
it arrived.

⭐ **Why nobody noticed for a month.** `capped` was written for LayerNorm, whose forward has
the same quadratic term and no eval mode to escape to. BatchNorm has one — so the cap looked
unnecessary here, and *"there is no theorem to state"* is exactly the kind of sentence that
stops the next person re-checking. The escape LayerNorm got transfers verbatim.

⚠ **What is modelled, and it is more than the inference net models.** `R34Bn` freezes `μ` and
`v` and supplies only the device `rsqrt`'s accuracy. Training-mode BatchNorm reduces both
statistics out of its own input, so BOTH are device kernels here: `R34TrainBn.fμ` accurate to
`emr·A` — **RELATIVE** to the layer's window, which is the shape a rounded mean of `n` terms
actually has — and `R34TrainBn.fistd` accurate to `ei` absolutely, the `DeviceRsqrt` standing.

⭐⭐ **And `emr` is now DERIVED rather than supplied, per SITE — 65 orders.** It was `10⁻²`, by
analogy with the `rsqrt`. But a device `rsqrt` genuinely has no IEEE specification and a device
MEAN is a rounded reduction followed by a divide, which plainly does: `bnMean_close_of` bounds it
by `u·(1+γ) + γ` at the fan-in `γ` **every** summation order meets, and the five widths this net
reduces over give `3.041·10⁻⁹` at `7×7` up to `7.484·10⁻⁴` at `112×112`. `R34TrainWeights` pins
those five numerals; `r34TrainBn_emr_committed` is the proof that a rounded reduction achieves
each of them, so the constant is computed and not chosen. ⛔ It is per site because folding all
five at the widest throws most of it away. ⚠ `ei` stays supplied at `10⁻²` and is now the loose
one by four orders — the residue §3.27 finding 4 is about.

⚠ **What did NOT move, and it is now the whole story.** With `emr` derived the normalisation
sites SHRINK and the remaining growth is almost entirely the conv fan-in — `layerBudget`'s uniform
`m·w'·A` face, which `planning/archive/float_budget_numbers_log.md` §0 names as one of the two documented gaps
to the adjoint chain and which nothing in the float tier has attacked. That is the next problem,
and it is shared with every other number in this repo.

⭐ **No operating point.** The number is stated at the unconditional `ε`-floor `|istd| ≤ 317`,
like the inference number and unlike `Resnet34BackFloatBudget.lean`'s `|istd| ≤ 16`: the chain's
largest goals closed at `10²²¹` with 32 orders of headroom under the shape-dependent `norm_num`
ceiling before escape 2, and the two 2026-09-05 improvements leave 172. The hypothesis was never
needed and is not paid.

⚠ **`Xh` is per SITE and the batch size is 1.** The reduction width is `h·w` here; at training
with `N > 1` a BatchNorm site reduces over `N·h·w`, so `Xh` grows with the batch and this number
is a per-example statement (`R34TrainBn`'s `hmXh` is what says so). The forward's other six
numbers do not have this qualifier; `b0_grad_float_le` is the one that already did.

**The tie is closed at the graph**, exactly as the inference number's is:
`r34TrainForward_eq_full_pc` is a `rfl` onto `resnet34Forward_full_pc` — the committed training
net — and `r34TrainGraph_faithful` carries `resnet34FwdGraphFullPC_faithful` the rest of the
way, so the typed `SHlo` graph every line of `@resnet34_fwd` renders denotes the forward this
file bounds. `r34_train_float_logits_le_committed` restates the number on that net.

Provenance for the 180 numerals: `scripts/float_budget_envelope.py`'s
`r34_train_chain(lin = True, cap = 'force')`, re-asserted by `verify_r34_train(lin = True)` (180
inequalities, and it separately asserts that the cap is the smaller branch at all 36 sites, so
the label above is measured rather than assumed) before a line of this file was emitted.
⭐ The 242 slot numerals were re-emitted by first reproducing the SHIPPED file's own ordered list
exactly from the shipped chain — that reproduction is the check that the map from probe rows to
`Maps` arguments is right (`planning/archive/float_budget_numbers_log.md` §3.30).
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The modelled device statistics, and the parameter records
-- ════════════════════════════════════════════════════════════════

/-- **One TRAINING-mode BatchNorm site.** Where `R34Bn` carries frozen running statistics and
    reads a device `rsqrt` off them, this carries the two REDUCTIONS the device performs on the
    layer's own input. Both are modelled — `fμ` with an accuracy `emr·A` relative to the input
    window, `fistd` with an absolute `ei` — for the reason `BnFloatBridge.lean` gives: a GPU
    reduction and a GPU `rsqrt` have no IEEE specification.

    ⭐ There is no `μ`/`v` field and no `Mb` bound: at training there is nothing frozen to
    bound, which is precisely why this net's modulus is quadratic in the window and its number
    has to be capped. -/
structure R34TrainBn (c h w : Nat) (ε G Bb emr ei Xh : ℝ) where
  γ : Vec c
  β : Vec c
  hγ : ∀ o, |γ o| ≤ G
  hβ : ∀ o, |β o| ≤ Bb
  /-- The device's per-channel batch mean. -/
  fμ : Fin c → Vec (h * w) → ℝ
  /-- The device's per-channel inverse standard deviation. -/
  fistd : Fin c → Vec (h * w) → ℝ
  /-- ⚠ RELATIVE: a rounded mean of `h·w` terms is proportional to the window. -/
  hmean : ∀ o A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
      |fμ o v - bnMean (h * w) v| ≤ emr * A
  histd : ∀ o A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
      |fistd o v - bnIstd (h * w) v ε| ≤ ei
  /-- ⭐ §0.1's ESCAPE 2: a numeral root of the reduction width, so `bnXhat_sq_le`'s
      `|x̂| ≤ √(h·w)` is usable. Exact here — every r34 feature map is square. -/
  hXh0 : 0 ≤ Xh
  hmXh : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2
  /-- ⭐ Per SITE, because the derived mean accuracy is a function of the reduction width —
      `R34TrainWeights` pins five different numerals. It lived in `R34TrainProfile` while `emr`
      was one supplied constant for the whole net. -/
  hemr0 : 0 ≤ emr

/-- An identity basic block at training-mode BN: two 3×3 convs, two BN sites. 13 of the 16.
    ⚠ The spatial dims sit in the TYPE here where `R34IdBlk` takes them at each use site — the
    BN record needs `h*w` to state its statistics' accuracies. -/
structure R34TrainIdBlk (c h w : Nat) (w' β' ε G Bb emr ei Xh : ℝ) where
  cv1 : R34Conv c c 3 3 w' β'
  bn1 : R34TrainBn c h w ε G Bb emr ei Xh
  cv2 : R34Conv c c 3 3 w' β'
  bn2 : R34TrainBn c h w ε G Bb emr ei Xh

/-- A downsample basic block at training-mode BN: two body convs (the first stride-2), the 1×1
    stride-2 option-B projection, and three BN sites. -/
structure R34TrainDownBlk (ic oc h w : Nat) (w' β' ε G Bb emr ei Xh : ℝ) where
  cv1 : R34Conv oc ic 3 3 w' β'
  bn1 : R34TrainBn oc h w ε G Bb emr ei Xh
  cv2 : R34Conv oc oc 3 3 w' β'
  bn2 : R34TrainBn oc h w ε G Bb emr ei Xh
  cvp : R34Conv oc ic 1 1 w' β'
  bnp : R34TrainBn oc h w ε G Bb emr ei Xh

/-- **The whole training net's stored parameters** — 37 convolutions, the classifier, and
    **36** training-mode BN sites (1 stem + 13 identity blocks × 2 + 3 downsample blocks × 3).
    ⚠ Count them: the docstrings of `Resnet34FloatBudget.lean` and of `planning/` say 33, which
    is the number of *body* convolutions and not the number of normalisations. -/
structure R34TrainWeights (w' β' ε G Bb ei : ℝ) where
  stem : R34Conv 64 3 7 7 w' β'
  bns : R34TrainBn 64 112 112 ε G Bb (7484 / 10 ^ 7) ei 112
  a0 : R34TrainIdBlk 64 56 56 w' β' ε G Bb (1872 / 10 ^ 7) ei 56
  a1 : R34TrainIdBlk 64 56 56 w' β' ε G Bb (1872 / 10 ^ 7) ei 56
  a2 : R34TrainIdBlk 64 56 56 w' β' ε G Bb (1872 / 10 ^ 7) ei 56
  d2 : R34TrainDownBlk 64 128 28 28 w' β' ε G Bb (4686 / 10 ^ 8) ei 28
  b0 : R34TrainIdBlk 128 28 28 w' β' ε G Bb (4686 / 10 ^ 8) ei 28
  b1 : R34TrainIdBlk 128 28 28 w' β' ε G Bb (4686 / 10 ^ 8) ei 28
  b2 : R34TrainIdBlk 128 28 28 w' β' ε G Bb (4686 / 10 ^ 8) ei 28
  d3 : R34TrainDownBlk 128 256 14 14 w' β' ε G Bb (1181 / 10 ^ 8) ei 14
  c0 : R34TrainIdBlk 256 14 14 w' β' ε G Bb (1181 / 10 ^ 8) ei 14
  c1 : R34TrainIdBlk 256 14 14 w' β' ε G Bb (1181 / 10 ^ 8) ei 14
  c2 : R34TrainIdBlk 256 14 14 w' β' ε G Bb (1181 / 10 ^ 8) ei 14
  c3 : R34TrainIdBlk 256 14 14 w' β' ε G Bb (1181 / 10 ^ 8) ei 14
  c4 : R34TrainIdBlk 256 14 14 w' β' ε G Bb (1181 / 10 ^ 8) ei 14
  d4 : R34TrainDownBlk 256 512 7 7 w' β' ε G Bb (3041 / 10 ^ 9) ei 7
  e0 : R34TrainIdBlk 512 7 7 w' β' ε G Bb (3041 / 10 ^ 9) ei 7
  e1 : R34TrainIdBlk 512 7 7 w' β' ε G Bb (3041 / 10 ^ 9) ei 7
  head : R34Head 512 10 w' β'

/-- The numeric profile the fold runs at. `R34Profile` with the frozen-mean bound `Mb` dropped
    (there is nothing frozen) and the single device accuracy `es` replaced by the two this mode
    needs: `emr` on the batch mean and `ei` on the inverse stddev. -/
structure R34TrainProfile (M : FloatModel) (ε w' β' G Bb ei S q : ℝ) : Prop where
  hw' : 0 ≤ w'
  hβ' : 0 ≤ β'
  hG : 0 ≤ G
  hBb : 0 ≤ Bb
  hei : 0 ≤ ei
  hS0 : 0 ≤ S
  hε : 0 < ε
  hSε : 1 / Real.sqrt ε ≤ S
  hq : M.u ≤ q

-- ════════════════════════════════════════════════════════════════
-- § One BN site: forward, float peer, CAPPED bridge, envelope
-- ════════════════════════════════════════════════════════════════

variable {M : FloatModel} {ε w' β' G Bb emr ei Xh S q : ℝ}

/-- The certified ℝ training-mode BN at this site. -/
noncomputable def R34TrainBn.fwd {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei Xh) :
    Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3 c h w ε B.γ B.β

/-- The deployed float training-mode BN at this site: the rounded normalize chain over the
    device's own two reductions. -/
noncomputable def R34TrainBn.fwdF {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei Xh)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3FV M B.γ B.β B.fμ B.fistd

/-- ⛔ **This BN site's bridge, CAPPED.** The `.capped` is the whole difference between this
    file and `Resnet34FloatBudget.lean`, and it is what makes the number the triangle
    inequality rather than the fold (§9). Without it the site's modulus carries
    `G·2Ā·(8Ā·Ē/(2ε√ε))`, quadratic in the window, and 36 of them square to `10⁷⁴¹⁹`. -/
noncomputable def R34TrainBn.bridge {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei Xh)
    (M : FloatModel) (P : R34TrainProfile M ε w' β' G Bb ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) :
    FloatBridgesTo B.fwd (B.fwdF M) :=
  (floatBridgesTo_bnPerChannelTensor3X (h := h) (w := w) M B.γ B.β B.fμ B.fistd
    (fun A => emr * A) (fun _ => ei) hc hhw P.hε B.hγ B.hβ B.hmean B.histd
    (fun v => (bnIstd_abs_le v P.hε).trans P.hSε) B.hXh0 B.hmXh).capped

/-- **This BN site's numeric envelope** — and note there is only ONE numeric inequality with
    any content. The window clause is the training-mode leaf's own; the error clause is
    `2·Ā' ≤ Ē'`, which mentions neither the inherited error nor `ε`. That is the mechanism: at
    a capped site §0.1's quadratic is never turned into a numeral, so `norm_num` never meets
    it. -/
theorem R34TrainBn.maps {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei Xh) (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb ei S q) (hc : 0 < c) (hhw : 0 < h * w)
    {Ā Ē Ā' Ē' : ℝ}
    (hĀ' : G * Xh + Bb + bnNormBudgetX q Xh (2 * Ā) S G Bb (emr * Ā) ei ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    (B.bridge M P hc hhw).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnPerChannelTensor3CappedX (h := h) (w := w) M B.γ B.β B.fμ B.fistd
    (fun A => emr * A) (fun _ => ei) hc hhw P.hε B.hγ B.hβ B.hmean B.histd
    (fun v => (bnIstd_abs_le v P.hε).trans P.hSε) B.hXh0 B.hmXh P.hq P.hG P.hBb P.hS0
    (fun _A _h0 hle => mul_le_mul_of_nonneg_left hle B.hemr0) (fun _ _ _ => le_rfl) hĀ' hĒ'

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ The five `emr` numerals are DERIVED, not supplied — and here is the witness
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **Every `emr` numeral `R34TrainWeights` pins is what a ROUNDED REDUCTION achieves at that
    site's width** — the whole content of §3.31's item (D). `R34TrainBn.hmean` is still a
    hypothesis, so the theorem quantifies over every device mean at least this accurate (the
    shape `DeviceRsqrt` has had throughout); what this says is that the class is not empty and
    that the constant was not chosen. Take any device reduction `fsum` whose forward error meets
    the fan-in `γ` that EVERY summation order meets — sequential is the worst at
    `(1+u)^{n+1} − 1`, a tree is `(1+u)^{⌈log₂n⌉+1} − 1` and so a fortiori — divide by the exact
    width, and the mean is within `emr·A`.

    ⛔ **This is why `emr` had to become per-site.** The five widths give five constants,
    `3.041·10⁻⁹` at `7×7` to `7.484·10⁻⁴` at `112×112`, and folding them uniformly at the largest
    throws away most of what (D) buys. `Xh` moved into the type for the same reason one section
    earlier.

    ⚠ What is NOT claimed: that the device sums left to right. `M.sum` is a concrete left fold and
    no GPU kernel is one; `bnMean_close_of` takes the spec instead of the order, which is the
    difference between a theorem about a program we run and one about a program we do not
    (`BnEvalFloatBridge.lean`'s warning, one tier over). -/
theorem r34TrainBn_emr_derived (M : FloatModel) (hMu : M.u ≤ u32)
    {n : ℕ} {fsum : Vec n → ℝ} {gq eqn : ℝ} (hn : 0 < n)
    (hsum : ∀ x : Vec n, |fsum x - ∑ i, x i| ≤ ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i|)
    (hk : ((n + 1 : ℕ) : ℝ) * u32 < 1)
    (hgq : ((n + 1 : ℕ) : ℝ) * u32 / (1 - ((n + 1 : ℕ) : ℝ) * u32) ≤ gq)
    (heq : u32 * (1 + gq) + gq ≤ eqn) :
    ∀ A : ℝ, 0 ≤ A → ∀ v : Vec n, (∀ k, |v k| ≤ A) →
      |M.div (fsum v) (n : ℝ) - bnMean n v| ≤ eqn * A :=
  M.bnMean_num_le hMu hn hsum hk hgq heq

/-- ⭐ **The five committed numerals discharged, one per reduction width.** `112·112 = 12544`
    down to `7·7 = 49`; each is `u·(1+γ) + γ` at `gamma_num`'s rational `γ`, rounded up to four
    significant figures — the same chain `scripts/float_budget_envelope.py`'s `emr_derived` folds,
    which is what makes the emitted stage numerals match what the kernel checks. -/
theorem r34TrainBn_emr_committed (M : FloatModel) (hMu : M.u ≤ u32)
    {fs12544 : Vec 12544 → ℝ} {fs3136 : Vec 3136 → ℝ} {fs784 : Vec 784 → ℝ}
    {fs196 : Vec 196 → ℝ} {fs49 : Vec 49 → ℝ}
    (h12544 : ∀ x, |fs12544 x - ∑ i, x i| ≤ ((1 + M.u) ^ 12545 - 1) * ∑ i, |x i|)
    (h3136 : ∀ x, |fs3136 x - ∑ i, x i| ≤ ((1 + M.u) ^ 3137 - 1) * ∑ i, |x i|)
    (h784 : ∀ x, |fs784 x - ∑ i, x i| ≤ ((1 + M.u) ^ 785 - 1) * ∑ i, |x i|)
    (h196 : ∀ x, |fs196 x - ∑ i, x i| ≤ ((1 + M.u) ^ 197 - 1) * ∑ i, |x i|)
    (h49 : ∀ x, |fs49 x - ∑ i, x i| ≤ ((1 + M.u) ^ 50 - 1) * ∑ i, |x i|) :
    (∀ A : ℝ, 0 ≤ A → ∀ v : Vec 12544, (∀ k, |v k| ≤ A) →
        |M.div (fs12544 v) (12544 : ℝ) - bnMean 12544 v| ≤ (7484 / 10 ^ 7) * A)
    ∧ (∀ A : ℝ, 0 ≤ A → ∀ v : Vec 3136, (∀ k, |v k| ≤ A) →
        |M.div (fs3136 v) (3136 : ℝ) - bnMean 3136 v| ≤ (1872 / 10 ^ 7) * A)
    ∧ (∀ A : ℝ, 0 ≤ A → ∀ v : Vec 784, (∀ k, |v k| ≤ A) →
        |M.div (fs784 v) (784 : ℝ) - bnMean 784 v| ≤ (4686 / 10 ^ 8) * A)
    ∧ (∀ A : ℝ, 0 ≤ A → ∀ v : Vec 196, (∀ k, |v k| ≤ A) →
        |M.div (fs196 v) (196 : ℝ) - bnMean 196 v| ≤ (1181 / 10 ^ 8) * A)
    ∧ (∀ A : ℝ, 0 ≤ A → ∀ v : Vec 49, (∀ k, |v k| ≤ A) →
        |M.div (fs49 v) (49 : ℝ) - bnMean 49 v| ≤ (3041 / 10 ^ 9) * A) :=
  ⟨r34TrainBn_emr_derived M hMu (gq := 7483 / 10 ^ 7) (by norm_num) h12544
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]),
   r34TrainBn_emr_derived M hMu (gq := 1871 / 10 ^ 7) (by norm_num) h3136
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]),
   r34TrainBn_emr_derived M hMu (gq := 4680 / 10 ^ 8) (by norm_num) h784
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]),
   r34TrainBn_emr_derived M hMu (gq := 1175 / 10 ^ 8) (by norm_num) h196
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]),
   r34TrainBn_emr_derived M hMu (gq := 2981 / 10 ^ 9) (by norm_num) h49
      (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32])⟩

-- ════════════════════════════════════════════════════════════════
-- § One block: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

/-- The certified ℝ identity block at training-mode BN. -/
noncomputable def R34TrainIdBlk.fwd {c h w : Nat}
    (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei Xh) : Vec (c * h * w) → Vec (c * h * w) :=
  rblkGen (h := h) (w := w) B.cv1.W B.cv1.b B.bn1.fwd B.cv2.W B.cv2.b B.bn2.fwd

/-- The deployed float identity block. -/
noncomputable def R34TrainIdBlk.fwdF {c h w : Nat}
    (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei Xh) (M : FloatModel) :
    Vec (c * h * w) → Vec (c * h * w) :=
  rblkGenF M B.cv1.W B.cv1.b (B.bn1.fwdF M) B.cv2.W B.cv2.b (B.bn2.fwdF M)

/-- This block's bridge. ⭐ `floatBridgesTo_r34IdBlock` is generic in the normalisation, so the
    inference and training nets share one block bridge and the capped BN is just an argument. -/
noncomputable def R34TrainIdBlk.bridge {c h w : Nat}
    (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei Xh) (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w) :
    FloatBridgesTo B.fwd (B.fwdF M) :=
  floatBridgesTo_r34IdBlock (h := h) (w := w) M B.cv1.W B.cv1.b B.cv2.W B.cv2.b
    B.bn1.fwd (B.bn1.fwdF M) B.bn2.fwd (B.bn2.fwdF M)
    P.hw' P.hβ' hn B.cv1.hW B.cv1.hb B.cv2.hW B.cv2.hb
    (B.bn1.bridge M P hc hhw) (B.bn2.bridge M P hc hhw)

/-- **This block's numeric envelope** — four numeric stages then the residual fan-in. Ten
    inequalities, of which the two BN error clauses are caps. -/
theorem R34TrainIdBlk.maps {c h w : Nat} (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei Xh)
    (M : FloatModel) (P : R34TrainProfile M ε w' β' G Bb ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w)
    {g Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ}
    (hg : (1 + M.u) ^ (c * 3 * 3 + 2) - 1 ≤ g)
    (c1A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (c1E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (n1A : G * Xh + Bb + bnNormBudgetX q Xh (2 * A1) S G Bb (emr * A1) ei ≤ A2)
    (n1cap : 2 * A2 ≤ E2)
    (c2A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * A2 + β') ≤ A3)
    (c2E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (A2 + E2) + β')
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E3)
    (n2A : G * Xh + Bb + bnNormBudgetX q Xh (2 * A3) S G Bb (emr * A3) ei ≤ A4)
    (n2cap : 2 * A4 ≤ E4)
    (rA : A4 + Ā + q * (A4 + Ā) ≤ Ā') (rE : q * (A4 + E4 + Ā + Ē) + (E4 + Ē) ≤ Ē') :
    (B.bridge M P hc hhw hn).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.cv1.W B.cv1.b P.hw' P.hβ' hn
    B.cv1.hW B.cv1.hb hg c1A c1E
  have s2 := s1.comp hn (B.bn1.maps M P hc hhw n1A n1cap)
  have s3 := s2.comp hn FloatBridgesTo.Maps.relu
  have s4 := s3.comp hn (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.cv2.W B.cv2.b
    P.hw' P.hβ' hn B.cv2.hW B.cv2.hb hg c2A c2E)
  have s5 := s4.comp hn (B.bn2.maps M P hc hhw n2A n2cap)
  exact (FloatBridgesTo.Maps.residual M hn s5 P.hq rA rE).comp hn FloatBridgesTo.Maps.relu

/-- The certified ℝ downsample block at training-mode BN. -/
noncomputable def R34TrainDownBlk.fwd {ic oc h w : Nat}
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei Xh) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  rblkStridedGen (h := h) (w := w) B.cv1.W B.cv1.b B.bn1.fwd B.cv2.W B.cv2.b B.bn2.fwd
    B.cvp.W B.cvp.b B.bnp.fwd

/-- The deployed float downsample block. -/
noncomputable def R34TrainDownBlk.fwdF {ic oc h w : Nat}
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei Xh) (M : FloatModel) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  rblkStridedGenF M B.cv1.W B.cv1.b (B.bn1.fwdF M) B.cv2.W B.cv2.b (B.bn2.fwdF M)
    B.cvp.W B.cvp.b (B.bnp.fwdF M)

/-- This block's bridge. -/
noncomputable def R34TrainDownBlk.bridge {ic oc h w : Nat}
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei Xh) (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb ei S q)
    (hoc : 0 < oc) (hhw : 0 < h * w) (hn : 0 < oc * h * w)
    (hni : 0 < ic * (2 * h) * (2 * w)) :
    FloatBridgesTo B.fwd (B.fwdF M) :=
  floatBridgesTo_r34DownBlock (h := h) (w := w) M B.cv1.W B.cv1.b B.cv2.W B.cv2.b
    B.cvp.W B.cvp.b B.bn1.fwd (B.bn1.fwdF M) B.bn2.fwd (B.bn2.fwdF M)
    B.bnp.fwd (B.bnp.fwdF M)
    P.hw' P.hβ' hn hni B.cv1.hW B.cv1.hb B.cv2.hW B.cv2.hb B.cvp.hW B.cvp.hb
    (B.bn1.bridge M P hoc hhw) (B.bn2.bridge M P hoc hhw) (B.bnp.bridge M P hoc hhw)

/-- **This block's numeric envelope** — the projection branch, the body branch, then the
    two-branch rounded fan-in. Fourteen inequalities, of which the three BN error clauses are
    caps. -/
theorem R34TrainDownBlk.maps {ic oc h w : Nat}
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei Xh) (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb ei S q)
    (hoc : 0 < oc) (hhw : 0 < h * w) (hn : 0 < oc * h * w)
    (hni : 0 < ic * (2 * h) * (2 * w))
    {g1 g2 gp Ā Ē P1 Q1 P2 Q2 A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ}
    (hg1 : (1 + M.u) ^ (ic * 3 * 3 + 2) - 1 ≤ g1)
    (hg2 : (1 + M.u) ^ (oc * 3 * 3 + 2) - 1 ≤ g2)
    (hgp : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ gp)
    (pA : (1 + gp) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ P1)
    (pE : gp * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ Q1)
    (pnA : G * Xh + Bb + bnNormBudgetX q Xh (2 * P1) S G Bb (emr * P1) ei ≤ P2)
    (pncap : 2 * P2 ≤ Q2)
    (c1A : (1 + g1) * (((ic * 3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (c1E : g1 * (((ic * 3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (n1A : G * Xh + Bb + bnNormBudgetX q Xh (2 * A1) S G Bb (emr * A1) ei ≤ A2)
    (n1cap : 2 * A2 ≤ E2)
    (c2A : (1 + g2) * (((oc * 3 * 3 : ℕ) : ℝ) * w' * A2 + β') ≤ A3)
    (c2E : g2 * (((oc * 3 * 3 : ℕ) : ℝ) * w' * (A2 + E2) + β')
            + ((oc * 3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E3)
    (n2A : G * Xh + Bb + bnNormBudgetX q Xh (2 * A3) S G Bb (emr * A3) ei ≤ A4)
    (n2cap : 2 * A4 ≤ E4)
    (rA : P2 + A4 + q * (P2 + A4) ≤ Ā') (rE : q * (P2 + Q2 + A4 + E4) + (Q2 + E4) ≤ Ē') :
    (B.bridge M P hoc hhw hn hni).Maps Ā Ē Ā' Ē' := by
  have p1 := FloatBridgesTo.Maps.flatConvStride2 (h := h) (w := w) M B.cvp.W B.cvp.b
    P.hw' P.hβ' hni B.cvp.hW B.cvp.hb hgp pA pE
  have p2 := p1.comp hn (B.bnp.maps M P hoc hhw pnA pncap)
  have s1 := FloatBridgesTo.Maps.flatConvStride2 (h := h) (w := w) M B.cv1.W B.cv1.b
    P.hw' P.hβ' hni B.cv1.hW B.cv1.hb hg1 c1A c1E
  have s2 := s1.comp hn (B.bn1.maps M P hoc hhw n1A n1cap)
  have s3 := s2.comp hn FloatBridgesTo.Maps.relu
  have s4 := s3.comp hn (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.cv2.W B.cv2.b
    P.hw' P.hβ' hn B.cv2.hW B.cv2.hb hg2 c2A c2E)
  have s5 := s4.comp hn (B.bn2.maps M P hoc hhw n2A n2cap)
  exact (FloatBridgesTo.Maps.biPathSum M hn p2 s5 P.hq rA rE).comp hn
    FloatBridgesTo.Maps.relu

-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The deployed ResNet-34 TRAINING forward** — the committed `r34Forward` skeleton with
    training-mode BatchNorm at every one of its 36 sites. -/
noncomputable def r34TrainForward (W : R34TrainWeights w' β' ε G Bb ei) :
    Vec (3 * 224 * 224) → Vec 10 :=
  r34Forward W.stem.W W.stem.b W.head.W W.head.b
    W.bns.fwd
    W.a0.fwd
    W.a1.fwd
    W.a2.fwd
    W.d2.fwd
    W.b0.fwd
    W.b1.fwd
    W.b2.fwd
    W.d3.fwd
    W.c0.fwd
    W.c1.fwd
    W.c2.fwd
    W.c3.fwd
    W.c4.fwd
    W.d4.fwd
    W.e0.fwd
    W.e1.fwd

/-- **The deployed ResNet-34 float training forward** — every concrete slot replaced by the
    model's rounded peer, every BN by the rounded normalize chain over the device's own two
    reductions. -/
noncomputable def r34TrainForwardF (M : FloatModel) (W : R34TrainWeights w' β' ε G Bb ei) :
    Vec (3 * 224 * 224) → Vec 10 :=
  r34ForwardF M W.stem.W W.stem.b W.head.W W.head.b
    (W.bns.fwdF M)
    (W.a0.fwdF M)
    (W.a1.fwdF M)
    (W.a2.fwdF M)
    (W.d2.fwdF M)
    (W.b0.fwdF M)
    (W.b1.fwdF M)
    (W.b2.fwdF M)
    (W.d3.fwdF M)
    (W.c0.fwdF M)
    (W.c1.fwdF M)
    (W.c2.fwdF M)
    (W.c3.fwdF M)
    (W.c4.fwdF M)
    (W.d4.fwdF M)
    (W.e0.fwdF M)
    (W.e1.fwdF M)

set_option maxRecDepth 100000 in
/-- ⭐ **The whole deployed ResNet-34 TRAINING forward float-bridges TO its float peer** — a
    CLOSED `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left. Structurally identical to
    `r34EvalBridge`; the only difference is that each of the 36 BN slots carries a `.capped`
    bridge over the training-mode leaf instead of an uncapped one over the inference leaf. -/
noncomputable def r34TrainBridge (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb ei S q) (W : R34TrainWeights w' β' ε G Bb ei) :
    FloatBridgesTo (r34TrainForward W) (r34TrainForwardF M W) :=
  ((((((((((((((((((((
    (floatBridgesTo_flatConvStride2 (h := 112) (w := 112) M W.stem.W W.stem.b P.hw' P.hβ'
      (by norm_num) W.stem.hW W.stem.hb)
    |>.comp (W.bns.bridge M P (by norm_num) (by norm_num)))
    |>.comp (floatBridgesTo_relu (n := 64 * 112 * 112)))
    |>.comp (floatBridgesTo_maxPool3s2 (c := 64) (h := 56) (w := 56)))
    |>.comp (W.a0.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.a1.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.a2.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.d2.bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.b0.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.b1.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.b2.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.d3.bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.c0.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.c1.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.c2.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.c3.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.c4.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.d4.bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.e0.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (W.e1.bridge M P (by norm_num) (by norm_num) (by norm_num)))
    |>.comp (floatBridgesTo_gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num)))
    |>.comp (floatBridgesTo_dense M W.head.W W.head.b P.hw' P.hβ' (by norm_num)
      W.head.hW W.head.hb)

-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, `r34Profile_committed`'s two device accuracies re-aimed. Every
    stored parameter within `21/10` (global max `2.0741` on the 79-epoch ImageNet checkpoint),
    `ε ≥ 10⁻⁵` so the inverse-stddev is under `317`, the device batch mean accurate to `10⁻²`
    RELATIVE to the layer's window and the device inverse-stddev to `10⁻²` absolute. -/
theorem r34TrainProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    R34TrainProfile M ε (21/10) (21/10) (21/10) (21/10) (1/100) 317 u32 where
  hw' := by norm_num
  hβ' := by norm_num
  hG := by norm_num
  hBb := by norm_num
  hei := by norm_num
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
/-- ⭐ **The envelope, kernel-checked.** Ninety numeric stages and 180 rational inequalities,
    built bottom-up at block granularity. ⛔ Of those 180, the 36 BatchNorm ERROR clauses are
    `2·Ā' ≤ Ē'` — the cap — so this chain never evaluates §0.1's quadratic and the statement it
    closes is the triangle inequality (§9). ⭐ The 36 WINDOW clauses are escape 2's, stated at
    `|x̂| ≤ Xh` rather than at `2·Ā·S`; the `ε`-floor now clears the `norm_num` ceiling by 108
    orders rather than 32. -/
theorem r34TrainBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).Maps 1 0
      (8748 * 10 ^ 77) (1752 * 10 ^ 78) := by
  have hP := r34TrainProfile_committed M hMu hε5
  have t1 := FloatBridgesTo.Maps.flatConvStride2 (h := 112) (w := 112) M W.stem.W
    W.stem.b hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb
    (M.gamma_num (q := 8882 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 3109 / 10 ^ 1) (Ē' := 2761 / 10 ^ 6) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
  have t2 := t1.comp (by norm_num) (W.bns.maps M hP (by norm_num) (by norm_num)
    (Ā := 3109 / 10 ^ 1) (Ē := 2761 / 10 ^ 6) (Ā' := 4053 / 10 ^ 1) (Ē' := 8106 / 10 ^ 1) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num))
  have t3 := t2.comp (by norm_num) (FloatBridgesTo.Maps.relu (n := 64 * 112 * 112))
  have t4 := t3.comp (by norm_num)
    (FloatBridgesTo.Maps.maxPool3s2 (c := 64) (h := 56) (w := 56))
  have t5 := t4.comp (by norm_num) (W.a0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 4053 / 10 ^ 1) (Ē := 8106 / 10 ^ 1)
    (A1 := 4903 * 10 ^ 2) (E1 := 9806 * 10 ^ 2) (A2 := 8186 * 10 ^ 1) (E2 := 1638 * 10 ^ 2)
    (A3 := 9903 * 10 ^ 4) (E3 := 1982 * 10 ^ 5) (A4 := 1651 * 10 ^ 4) (E4 := 3302 * 10 ^ 4)
    (Ā' := 1652 * 10 ^ 4) (Ē' := 3303 * 10 ^ 4)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t6 := t5.comp (by norm_num) (W.a1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 1652 * 10 ^ 4) (Ē := 3303 * 10 ^ 4)
    (A1 := 1999 * 10 ^ 7) (E1 := 3996 * 10 ^ 7) (A2 := 3333 * 10 ^ 6) (E2 := 6666 * 10 ^ 6)
    (A3 := 4032 * 10 ^ 9) (E3 := 8064 * 10 ^ 9) (A4 := 6722 * 10 ^ 8) (E4 := 1345 * 10 ^ 9)
    (Ā' := 6723 * 10 ^ 8) (Ē' := 1346 * 10 ^ 9)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t7 := t6.comp (by norm_num) (W.a2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 6723 * 10 ^ 8) (Ē := 1346 * 10 ^ 9)
    (A1 := 8133 * 10 ^ 11) (E1 := 1629 * 10 ^ 12) (A2 := 1356 * 10 ^ 11) (E2 := 2712 * 10 ^ 11)
    (A3 := 1641 * 10 ^ 14) (E3 := 3281 * 10 ^ 14) (A4 := 2736 * 10 ^ 13) (E4 := 5472 * 10 ^ 13)
    (Ā' := 2737 * 10 ^ 13) (Ē' := 5473 * 10 ^ 13)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t8 := t7.comp (by norm_num) (W.d2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 3446 / 10 ^ 8) (g2 := 6879 / 10 ^ 8) (gp := 3934 / 10 ^ 9)
    (Ā := 2737 * 10 ^ 13) (Ē := 5473 * 10 ^ 13)
    (P1 := 3679 * 10 ^ 15) (Q1 := 7356 * 10 ^ 15) (P2 := 2696 * 10 ^ 14) (Q2 := 5392 * 10 ^ 14)
    (A1 := 3311 * 10 ^ 16) (E1 := 6621 * 10 ^ 16) (A2 := 2427 * 10 ^ 15) (E2 := 4854 * 10 ^ 15)
    (A3 := 5872 * 10 ^ 18) (E3 := 1175 * 10 ^ 19) (A4 := 4303 * 10 ^ 17) (E4 := 8606 * 10 ^ 17)
    (Ā' := 4306 * 10 ^ 17) (Ē' := 8612 * 10 ^ 17)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t9 := t8.comp (by norm_num) (W.b0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 4306 * 10 ^ 17) (Ē := 8612 * 10 ^ 17)
    (A1 := 1042 * 10 ^ 21) (E1 := 2084 * 10 ^ 21) (A2 := 7636 * 10 ^ 19) (E2 := 1528 * 10 ^ 20)
    (A3 := 1848 * 10 ^ 23) (E3 := 3697 * 10 ^ 23) (A4 := 1355 * 10 ^ 22) (E4 := 2710 * 10 ^ 22)
    (Ā' := 1356 * 10 ^ 22) (Ē' := 2711 * 10 ^ 22)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t10 := t9.comp (by norm_num) (W.b1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 1356 * 10 ^ 22) (Ē := 2711 * 10 ^ 22)
    (A1 := 3281 * 10 ^ 25) (E1 := 6560 * 10 ^ 25) (A2 := 2405 * 10 ^ 24) (E2 := 4810 * 10 ^ 24)
    (A3 := 5819 * 10 ^ 27) (E3 := 1164 * 10 ^ 28) (A4 := 4264 * 10 ^ 26) (E4 := 8528 * 10 ^ 26)
    (Ā' := 4265 * 10 ^ 26) (Ē' := 8529 * 10 ^ 26)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t11 := t10.comp (by norm_num) (W.b2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 4265 * 10 ^ 26) (Ē := 8529 * 10 ^ 26)
    (A1 := 1032 * 10 ^ 30) (E1 := 2064 * 10 ^ 30) (A2 := 7562 * 10 ^ 28) (E2 := 1513 * 10 ^ 29)
    (A3 := 1830 * 10 ^ 32) (E3 := 3661 * 10 ^ 32) (A4 := 1341 * 10 ^ 31) (E4 := 2682 * 10 ^ 31)
    (Ā' := 1342 * 10 ^ 31) (Ē' := 2683 * 10 ^ 31)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t12 := t11.comp (by norm_num) (W.d3.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 6879 / 10 ^ 8) (g2 := 1375 / 10 ^ 7) (gp := 7749 / 10 ^ 9)
    (Ā := 1342 * 10 ^ 31) (Ē := 2683 * 10 ^ 31)
    (P1 := 3608 * 10 ^ 33) (Q1 := 7212 * 10 ^ 33) (P2 := 1802 * 10 ^ 32) (Q2 := 3604 * 10 ^ 32)
    (A1 := 3247 * 10 ^ 34) (E1 := 6492 * 10 ^ 34) (A2 := 1622 * 10 ^ 33) (E2 := 3244 * 10 ^ 33)
    (A3 := 7849 * 10 ^ 36) (E3 := 1570 * 10 ^ 37) (A4 := 3920 * 10 ^ 35) (E4 := 7840 * 10 ^ 35)
    (Ā' := 3922 * 10 ^ 35) (Ē' := 7844 * 10 ^ 35)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t13 := t12.comp (by norm_num) (W.c0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 3922 * 10 ^ 35) (Ē := 7844 * 10 ^ 35)
    (A1 := 1898 * 10 ^ 39) (E1 := 3797 * 10 ^ 39) (A2 := 9479 * 10 ^ 37) (E2 := 1896 * 10 ^ 38)
    (A3 := 4587 * 10 ^ 41) (E3 := 9176 * 10 ^ 41) (A4 := 2291 * 10 ^ 40) (E4 := 4582 * 10 ^ 40)
    (Ā' := 2292 * 10 ^ 40) (Ē' := 4583 * 10 ^ 40)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t14 := t13.comp (by norm_num) (W.c1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 2292 * 10 ^ 40) (Ē := 4583 * 10 ^ 40)
    (A1 := 1110 * 10 ^ 44) (E1 := 2218 * 10 ^ 44) (A2 := 5544 * 10 ^ 42) (E2 := 1109 * 10 ^ 43)
    (A3 := 2683 * 10 ^ 46) (E3 := 5367 * 10 ^ 46) (A4 := 1340 * 10 ^ 45) (E4 := 2680 * 10 ^ 45)
    (Ā' := 1341 * 10 ^ 45) (Ē' := 2681 * 10 ^ 45)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t15 := t14.comp (by norm_num) (W.c2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 1341 * 10 ^ 45) (Ē := 2681 * 10 ^ 45)
    (A1 := 6490 * 10 ^ 48) (E1 := 1298 * 10 ^ 49) (A2 := 3242 * 10 ^ 47) (E2 := 6484 * 10 ^ 47)
    (A3 := 1569 * 10 ^ 51) (E3 := 3138 * 10 ^ 51) (A4 := 7836 * 10 ^ 49) (E4 := 1568 * 10 ^ 50)
    (Ā' := 7837 * 10 ^ 49) (Ē' := 1569 * 10 ^ 50)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t16 := t15.comp (by norm_num) (W.c3.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 7837 * 10 ^ 49) (Ē := 1569 * 10 ^ 50)
    (A1 := 3793 * 10 ^ 53) (E1 := 7594 * 10 ^ 53) (A2 := 1895 * 10 ^ 52) (E2 := 3790 * 10 ^ 52)
    (A3 := 9171 * 10 ^ 55) (E3 := 1835 * 10 ^ 56) (A4 := 4581 * 10 ^ 54) (E4 := 9162 * 10 ^ 54)
    (Ā' := 4582 * 10 ^ 54) (Ē' := 9163 * 10 ^ 54)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t17 := t16.comp (by norm_num) (W.c4.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 4582 * 10 ^ 54) (Ē := 9163 * 10 ^ 54)
    (A1 := 2218 * 10 ^ 58) (E1 := 4435 * 10 ^ 58) (A2 := 1108 * 10 ^ 57) (E2 := 2216 * 10 ^ 57)
    (A3 := 5362 * 10 ^ 60) (E3 := 1073 * 10 ^ 61) (A4 := 2678 * 10 ^ 59) (E4 := 5356 * 10 ^ 59)
    (Ā' := 2679 * 10 ^ 59) (Ē' := 5357 * 10 ^ 59)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t18 := t17.comp (by norm_num) (W.d4.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 1375 / 10 ^ 7) (g2 := 2749 / 10 ^ 7) (gp := 1538 / 10 ^ 8)
    (Ā := 2679 * 10 ^ 59) (Ē := 5357 * 10 ^ 59)
    (P1 := 1441 * 10 ^ 62) (Q1 := 2880 * 10 ^ 62) (P2 := 6356 * 10 ^ 60) (Q2 := 1272 * 10 ^ 61)
    (A1 := 1297 * 10 ^ 63) (E1 := 2593 * 10 ^ 63) (A2 := 5721 * 10 ^ 61) (E2 := 1145 * 10 ^ 62)
    (A3 := 5538 * 10 ^ 65) (E3 := 1109 * 10 ^ 66) (A4 := 2443 * 10 ^ 64) (E4 := 4886 * 10 ^ 64)
    (Ā' := 2444 * 10 ^ 64) (Ē' := 4888 * 10 ^ 64)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t19 := t18.comp (by norm_num) (W.e0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 2749 / 10 ^ 7)
    (Ā := 2444 * 10 ^ 64) (Ē := 4888 * 10 ^ 64)
    (A1 := 2366 * 10 ^ 68) (E1 := 4732 * 10 ^ 68) (A2 := 1044 * 10 ^ 67) (E2 := 2088 * 10 ^ 67)
    (A3 := 1011 * 10 ^ 71) (E3 := 2022 * 10 ^ 71) (A4 := 4459 * 10 ^ 69) (E4 := 8918 * 10 ^ 69)
    (Ā' := 4460 * 10 ^ 69) (Ē' := 8919 * 10 ^ 69)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t20 := t19.comp (by norm_num) (W.e1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 2749 / 10 ^ 7)
    (Ā := 4460 * 10 ^ 69) (Ē := 8919 * 10 ^ 69)
    (A1 := 4318 * 10 ^ 73) (E1 := 8635 * 10 ^ 73) (A2 := 1905 * 10 ^ 72) (E2 := 3810 * 10 ^ 72)
    (A3 := 1844 * 10 ^ 76) (E3 := 3689 * 10 ^ 76) (A4 := 8133 * 10 ^ 74) (E4 := 1627 * 10 ^ 75)
    (Ā' := 8134 * 10 ^ 74) (Ē' := 1628 * 10 ^ 75)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t21 := t20.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      hMu (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 8134 * 10 ^ 74) (Ē := 1628 * 10 ^ 75) (Ā' := 8135 * 10 ^ 74) (Ē' := 1629 * 10 ^ 75) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t22 := t21.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num)
      W.head.hW W.head.hb (M.gamma_num (q := 3064 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 8135 * 10 ^ 74) (Ē := 1629 * 10 ^ 75) (Ā' := 8748 * 10 ^ 77) (Ē' := 1752 * 10 ^ 78) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  exact t22

/-- The deployed ResNet-34 training bridge's certified output window at the committed profile:
    `≤ 8.748·10⁸⁰`. ⭐ This half is an honest fold — the cap touches only the modulus — and it
    is the half escape 2 moved, by 76 orders. -/
theorem r34TrainBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).mag 1 ≤ 8748 * 10 ^ 77 :=
  (r34TrainBridge_maps M hMu hε5 W).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ResNet-34 training bridge's error bound at the committed profile:
    `≤ 1.752·10⁸¹`, which is `2.00 ×` the window — the tell that this is
    `FloatBridgesTo.capped`'s triangle inequality and not the interval fold (§9). -/
theorem r34TrainBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).fresh 1 ≤ 1752 * 10 ^ 78 :=
  (r34TrainBridge_maps M hMu hε5 W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed ResNet-34 TRAINING-mode forward is within `1.752·10⁸¹` of the certified
    real training forward, per logit**, on inputs of magnitude `≤ 1`, at the measured parameter
    profile, for `ε ≥ 10⁻⁵`, any device batch mean accurate to `10⁻²` relative and any device
    inverse-stddev accurate to `10⁻²` absolute, and any rounding model at binary32 accuracy.

    ⭐⭐ **The first statement in this repo about the program it actually trains with** — every
    other committed forward number is at inference normalisation. ⛔ **It is the CAP**: read it
    as *"the float and the real forward both land in the certified window"*, never as *"the
    rounding error folds to this"*. `1.752 / 8.748 = 2.00` is the tell, and the fold it replaces
    is **`3.494·10⁴⁹⁹³`** — `10⁷⁴¹⁹` before escape 2, so the window half is worth 2426 orders on
    the fold as well and leaves it 4740 past the ceiling anyway. The quadratic shrinks with the
    window it is quadratic in and still does not fit; the cap is not a shortcut past a fold that
    exists (§9). -/
theorem r34_train_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |r34TrainForwardF M W x j - r34TrainForward W x j| ≤ 1752 * 10 ^ 78 :=
  (r34TrainBridge_maps M hMu hε5 W).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed training forward, and the graph denotes it
-- ════════════════════════════════════════════════════════════════

/-- **The record-bundled forward IS the committed training net.** `r34TrainForward` unfolds to
    `resnet34Forward_full_pc` at the record's projections — the training twin of
    `r34EvalForward_eq_full_pc_eval`, and a `rfl` for the same reason: `rblkPC_eq_gen` and
    `rblkPStridedPC_eq_gen` are both `rfl`, so the skeleton's block slots take exactly the maps
    the record builds. -/
theorem r34TrainForward_eq_full_pc (W : R34TrainWeights w' β' ε G Bb ei) :
    r34TrainForward W = resnet34Forward_full_pc ε
    W.stem.W W.stem.b W.bns.γ W.bns.β
    W.a0.cv1.W W.a0.cv1.b W.a0.bn1.γ W.a0.bn1.β W.a0.cv2.W W.a0.cv2.b W.a0.bn2.γ W.a0.bn2.β
    W.a1.cv1.W W.a1.cv1.b W.a1.bn1.γ W.a1.bn1.β W.a1.cv2.W W.a1.cv2.b W.a1.bn2.γ W.a1.bn2.β
    W.a2.cv1.W W.a2.cv1.b W.a2.bn1.γ W.a2.bn1.β W.a2.cv2.W W.a2.cv2.b W.a2.bn2.γ W.a2.bn2.β
    W.d2.cv1.W W.d2.cv1.b W.d2.bn1.γ W.d2.bn1.β W.d2.cv2.W W.d2.cv2.b W.d2.bn2.γ W.d2.bn2.β W.d2.cvp.W W.d2.cvp.b W.d2.bnp.γ W.d2.bnp.β
    W.b0.cv1.W W.b0.cv1.b W.b0.bn1.γ W.b0.bn1.β W.b0.cv2.W W.b0.cv2.b W.b0.bn2.γ W.b0.bn2.β
    W.b1.cv1.W W.b1.cv1.b W.b1.bn1.γ W.b1.bn1.β W.b1.cv2.W W.b1.cv2.b W.b1.bn2.γ W.b1.bn2.β
    W.b2.cv1.W W.b2.cv1.b W.b2.bn1.γ W.b2.bn1.β W.b2.cv2.W W.b2.cv2.b W.b2.bn2.γ W.b2.bn2.β
    W.d3.cv1.W W.d3.cv1.b W.d3.bn1.γ W.d3.bn1.β W.d3.cv2.W W.d3.cv2.b W.d3.bn2.γ W.d3.bn2.β W.d3.cvp.W W.d3.cvp.b W.d3.bnp.γ W.d3.bnp.β
    W.c0.cv1.W W.c0.cv1.b W.c0.bn1.γ W.c0.bn1.β W.c0.cv2.W W.c0.cv2.b W.c0.bn2.γ W.c0.bn2.β
    W.c1.cv1.W W.c1.cv1.b W.c1.bn1.γ W.c1.bn1.β W.c1.cv2.W W.c1.cv2.b W.c1.bn2.γ W.c1.bn2.β
    W.c2.cv1.W W.c2.cv1.b W.c2.bn1.γ W.c2.bn1.β W.c2.cv2.W W.c2.cv2.b W.c2.bn2.γ W.c2.bn2.β
    W.c3.cv1.W W.c3.cv1.b W.c3.bn1.γ W.c3.bn1.β W.c3.cv2.W W.c3.cv2.b W.c3.bn2.γ W.c3.bn2.β
    W.c4.cv1.W W.c4.cv1.b W.c4.bn1.γ W.c4.bn1.β W.c4.cv2.W W.c4.cv2.b W.c4.bn2.γ W.c4.bn2.β
    W.d4.cv1.W W.d4.cv1.b W.d4.bn1.γ W.d4.bn1.β W.d4.cv2.W W.d4.cv2.b W.d4.bn2.γ W.d4.bn2.β W.d4.cvp.W W.d4.cvp.b W.d4.bnp.γ W.d4.bnp.β
    W.e0.cv1.W W.e0.cv1.b W.e0.bn1.γ W.e0.bn1.β W.e0.cv2.W W.e0.cv2.b W.e0.bn2.γ W.e0.bn2.β
    W.e1.cv1.W W.e1.cv1.b W.e1.bn1.γ W.e1.bn1.β W.e1.cv2.W W.e1.cv2.b W.e1.bn2.γ W.e1.bn2.β
    W.head.W W.head.b := rfl

/-- ⭐ **The whole loop closes.** The typed `SHlo` training graph — every line of which
    `@resnet34_fwd` renders — denotes exactly the forward this file states its number about. -/
theorem r34TrainGraph_faithful (epsStr : String) (W : R34TrainWeights w' β' ε G Bb ei)
    (x : Vec (3 * 224 * 224)) :
    StableHLO.den (StableHLO.resnet34FwdGraphFullPC epsStr ε
    W.stem.W W.stem.b W.bns.γ W.bns.β
    W.a0.cv1.W W.a0.cv1.b W.a0.bn1.γ W.a0.bn1.β W.a0.cv2.W W.a0.cv2.b W.a0.bn2.γ W.a0.bn2.β
    W.a1.cv1.W W.a1.cv1.b W.a1.bn1.γ W.a1.bn1.β W.a1.cv2.W W.a1.cv2.b W.a1.bn2.γ W.a1.bn2.β
    W.a2.cv1.W W.a2.cv1.b W.a2.bn1.γ W.a2.bn1.β W.a2.cv2.W W.a2.cv2.b W.a2.bn2.γ W.a2.bn2.β
    W.d2.cv1.W W.d2.cv1.b W.d2.bn1.γ W.d2.bn1.β W.d2.cv2.W W.d2.cv2.b W.d2.bn2.γ W.d2.bn2.β W.d2.cvp.W W.d2.cvp.b W.d2.bnp.γ W.d2.bnp.β
    W.b0.cv1.W W.b0.cv1.b W.b0.bn1.γ W.b0.bn1.β W.b0.cv2.W W.b0.cv2.b W.b0.bn2.γ W.b0.bn2.β
    W.b1.cv1.W W.b1.cv1.b W.b1.bn1.γ W.b1.bn1.β W.b1.cv2.W W.b1.cv2.b W.b1.bn2.γ W.b1.bn2.β
    W.b2.cv1.W W.b2.cv1.b W.b2.bn1.γ W.b2.bn1.β W.b2.cv2.W W.b2.cv2.b W.b2.bn2.γ W.b2.bn2.β
    W.d3.cv1.W W.d3.cv1.b W.d3.bn1.γ W.d3.bn1.β W.d3.cv2.W W.d3.cv2.b W.d3.bn2.γ W.d3.bn2.β W.d3.cvp.W W.d3.cvp.b W.d3.bnp.γ W.d3.bnp.β
    W.c0.cv1.W W.c0.cv1.b W.c0.bn1.γ W.c0.bn1.β W.c0.cv2.W W.c0.cv2.b W.c0.bn2.γ W.c0.bn2.β
    W.c1.cv1.W W.c1.cv1.b W.c1.bn1.γ W.c1.bn1.β W.c1.cv2.W W.c1.cv2.b W.c1.bn2.γ W.c1.bn2.β
    W.c2.cv1.W W.c2.cv1.b W.c2.bn1.γ W.c2.bn1.β W.c2.cv2.W W.c2.cv2.b W.c2.bn2.γ W.c2.bn2.β
    W.c3.cv1.W W.c3.cv1.b W.c3.bn1.γ W.c3.bn1.β W.c3.cv2.W W.c3.cv2.b W.c3.bn2.γ W.c3.bn2.β
    W.c4.cv1.W W.c4.cv1.b W.c4.bn1.γ W.c4.bn1.β W.c4.cv2.W W.c4.cv2.b W.c4.bn2.γ W.c4.bn2.β
    W.d4.cv1.W W.d4.cv1.b W.d4.bn1.γ W.d4.bn1.β W.d4.cv2.W W.d4.cv2.b W.d4.bn2.γ W.d4.bn2.β W.d4.cvp.W W.d4.cvp.b W.d4.bnp.γ W.d4.bnp.β
    W.e0.cv1.W W.e0.cv1.b W.e0.bn1.γ W.e0.bn1.β W.e0.cv2.W W.e0.cv2.b W.e0.bn2.γ W.e0.bn2.β
    W.e1.cv1.W W.e1.cv1.b W.e1.bn1.γ W.e1.bn1.β W.e1.cv2.W W.e1.cv2.b W.e1.bn2.γ W.e1.bn2.β
    W.head.W W.head.b x)
      = r34TrainForward W x :=
  (StableHLO.resnet34FwdGraphFullPC_faithful epsStr ε
    W.stem.W W.stem.b W.bns.γ W.bns.β
    W.a0.cv1.W W.a0.cv1.b W.a0.bn1.γ W.a0.bn1.β W.a0.cv2.W W.a0.cv2.b W.a0.bn2.γ W.a0.bn2.β
    W.a1.cv1.W W.a1.cv1.b W.a1.bn1.γ W.a1.bn1.β W.a1.cv2.W W.a1.cv2.b W.a1.bn2.γ W.a1.bn2.β
    W.a2.cv1.W W.a2.cv1.b W.a2.bn1.γ W.a2.bn1.β W.a2.cv2.W W.a2.cv2.b W.a2.bn2.γ W.a2.bn2.β
    W.d2.cv1.W W.d2.cv1.b W.d2.bn1.γ W.d2.bn1.β W.d2.cv2.W W.d2.cv2.b W.d2.bn2.γ W.d2.bn2.β W.d2.cvp.W W.d2.cvp.b W.d2.bnp.γ W.d2.bnp.β
    W.b0.cv1.W W.b0.cv1.b W.b0.bn1.γ W.b0.bn1.β W.b0.cv2.W W.b0.cv2.b W.b0.bn2.γ W.b0.bn2.β
    W.b1.cv1.W W.b1.cv1.b W.b1.bn1.γ W.b1.bn1.β W.b1.cv2.W W.b1.cv2.b W.b1.bn2.γ W.b1.bn2.β
    W.b2.cv1.W W.b2.cv1.b W.b2.bn1.γ W.b2.bn1.β W.b2.cv2.W W.b2.cv2.b W.b2.bn2.γ W.b2.bn2.β
    W.d3.cv1.W W.d3.cv1.b W.d3.bn1.γ W.d3.bn1.β W.d3.cv2.W W.d3.cv2.b W.d3.bn2.γ W.d3.bn2.β W.d3.cvp.W W.d3.cvp.b W.d3.bnp.γ W.d3.bnp.β
    W.c0.cv1.W W.c0.cv1.b W.c0.bn1.γ W.c0.bn1.β W.c0.cv2.W W.c0.cv2.b W.c0.bn2.γ W.c0.bn2.β
    W.c1.cv1.W W.c1.cv1.b W.c1.bn1.γ W.c1.bn1.β W.c1.cv2.W W.c1.cv2.b W.c1.bn2.γ W.c1.bn2.β
    W.c2.cv1.W W.c2.cv1.b W.c2.bn1.γ W.c2.bn1.β W.c2.cv2.W W.c2.cv2.b W.c2.bn2.γ W.c2.bn2.β
    W.c3.cv1.W W.c3.cv1.b W.c3.bn1.γ W.c3.bn1.β W.c3.cv2.W W.c3.cv2.b W.c3.bn2.γ W.c3.bn2.β
    W.c4.cv1.W W.c4.cv1.b W.c4.bn1.γ W.c4.bn1.β W.c4.cv2.W W.c4.cv2.b W.c4.bn2.γ W.c4.bn2.β
    W.d4.cv1.W W.d4.cv1.b W.d4.bn1.γ W.d4.bn1.β W.d4.cv2.W W.d4.cv2.b W.d4.bn2.γ W.d4.bn2.β W.d4.cvp.W W.d4.cvp.b W.d4.bnp.γ W.d4.bnp.β
    W.e0.cv1.W W.e0.cv1.b W.e0.bn1.γ W.e0.bn1.β W.e0.cv2.W W.e0.cv2.b W.e0.bn2.γ W.e0.bn2.β
    W.e1.cv1.W W.e1.cv1.b W.e1.bn1.γ W.e1.bn1.β W.e1.cv2.W W.e1.cv2.b W.e1.bn2.γ W.e1.bn2.β
    W.head.W W.head.b x).trans
    (congrFun (r34TrainForward_eq_full_pc W).symm x)

/-- ⭐⭐ **The number, stated about the committed training forward.** `r34_train_float_logits_le`
    with `resnet34Forward_full_pc` on the real side — so the bound is a claim about the net
    `@resnet34_fwd` renders and the gradients of `Resnet34BackFloatBudget.lean` are taken
    through, tied through `r34TrainGraph_faithful` rather than by inspection. ⛔ Still the cap;
    §9's label travels with it. -/
theorem r34_train_float_logits_le_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |r34TrainForwardF M W x j - resnet34Forward_full_pc ε
    W.stem.W W.stem.b W.bns.γ W.bns.β
    W.a0.cv1.W W.a0.cv1.b W.a0.bn1.γ W.a0.bn1.β W.a0.cv2.W W.a0.cv2.b W.a0.bn2.γ W.a0.bn2.β
    W.a1.cv1.W W.a1.cv1.b W.a1.bn1.γ W.a1.bn1.β W.a1.cv2.W W.a1.cv2.b W.a1.bn2.γ W.a1.bn2.β
    W.a2.cv1.W W.a2.cv1.b W.a2.bn1.γ W.a2.bn1.β W.a2.cv2.W W.a2.cv2.b W.a2.bn2.γ W.a2.bn2.β
    W.d2.cv1.W W.d2.cv1.b W.d2.bn1.γ W.d2.bn1.β W.d2.cv2.W W.d2.cv2.b W.d2.bn2.γ W.d2.bn2.β W.d2.cvp.W W.d2.cvp.b W.d2.bnp.γ W.d2.bnp.β
    W.b0.cv1.W W.b0.cv1.b W.b0.bn1.γ W.b0.bn1.β W.b0.cv2.W W.b0.cv2.b W.b0.bn2.γ W.b0.bn2.β
    W.b1.cv1.W W.b1.cv1.b W.b1.bn1.γ W.b1.bn1.β W.b1.cv2.W W.b1.cv2.b W.b1.bn2.γ W.b1.bn2.β
    W.b2.cv1.W W.b2.cv1.b W.b2.bn1.γ W.b2.bn1.β W.b2.cv2.W W.b2.cv2.b W.b2.bn2.γ W.b2.bn2.β
    W.d3.cv1.W W.d3.cv1.b W.d3.bn1.γ W.d3.bn1.β W.d3.cv2.W W.d3.cv2.b W.d3.bn2.γ W.d3.bn2.β W.d3.cvp.W W.d3.cvp.b W.d3.bnp.γ W.d3.bnp.β
    W.c0.cv1.W W.c0.cv1.b W.c0.bn1.γ W.c0.bn1.β W.c0.cv2.W W.c0.cv2.b W.c0.bn2.γ W.c0.bn2.β
    W.c1.cv1.W W.c1.cv1.b W.c1.bn1.γ W.c1.bn1.β W.c1.cv2.W W.c1.cv2.b W.c1.bn2.γ W.c1.bn2.β
    W.c2.cv1.W W.c2.cv1.b W.c2.bn1.γ W.c2.bn1.β W.c2.cv2.W W.c2.cv2.b W.c2.bn2.γ W.c2.bn2.β
    W.c3.cv1.W W.c3.cv1.b W.c3.bn1.γ W.c3.bn1.β W.c3.cv2.W W.c3.cv2.b W.c3.bn2.γ W.c3.bn2.β
    W.c4.cv1.W W.c4.cv1.b W.c4.bn1.γ W.c4.bn1.β W.c4.cv2.W W.c4.cv2.b W.c4.bn2.γ W.c4.bn2.β
    W.d4.cv1.W W.d4.cv1.b W.d4.bn1.γ W.d4.bn1.β W.d4.cv2.W W.d4.cv2.b W.d4.bn2.γ W.d4.bn2.β W.d4.cvp.W W.d4.cvp.b W.d4.bnp.γ W.d4.bnp.β
    W.e0.cv1.W W.e0.cv1.b W.e0.bn1.γ W.e0.bn1.β W.e0.cv2.W W.e0.cv2.b W.e0.bn2.γ W.e0.bn2.β
    W.e1.cv1.W W.e1.cv1.b W.e1.bn1.γ W.e1.bn1.β W.e1.cv2.W W.e1.cv2.b W.e1.bn2.γ W.e1.bn2.β
    W.head.W W.head.b x j| ≤ 1752 * 10 ^ 78 :=
  r34_train_float_logits_le M hMu hε5 W x hx j

/-! ### Inhabitation

`r34_train_float_logits_le`'s record at the committed constants: zero weights and affines, the
exact batch mean and inverse-stddev as the device kernels (so `hmean`/`histd` are `0 ≤ emr·A` and
`0 ≤ ei`), the per-site `Xh` and `emr` numerals `R34TrainWeights` pins, `ε = 1/100000`. -/
noncomputable def R34TrainBn.exact (c h w : Nat) {ε G Bb emr ei Xh : ℝ}
    (hG : 0 ≤ G) (hBb : 0 ≤ Bb) (hemr : 0 ≤ emr) (hei : 0 ≤ ei) (hXh0 : 0 ≤ Xh)
    (hmXh : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) : R34TrainBn c h w ε G Bb emr ei Xh where
  γ := fun _ => 0
  β := fun _ => 0
  hγ := fun _ => by simpa using hG
  hβ := fun _ => by simpa using hBb
  fμ := fun _ v => bnMean (h * w) v
  fistd := fun _ v => bnIstd (h * w) v ε
  hmean := fun _ A hA _ _ => by simpa using mul_nonneg hemr hA
  histd := fun _ _ _ _ _ => by simpa using hei
  hXh0 := hXh0
  hmXh := hmXh
  hemr0 := hemr

noncomputable def R34TrainIdBlk.exact (c h w : Nat) {w' β' ε G Bb emr ei Xh : ℝ}
    (hw : 0 ≤ w') (hb : 0 ≤ β') (hG : 0 ≤ G) (hBb : 0 ≤ Bb) (hemr : 0 ≤ emr) (hei : 0 ≤ ei)
    (hXh0 : 0 ≤ Xh) (hmXh : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    R34TrainIdBlk c h w w' β' ε G Bb emr ei Xh where
  cv1 := R34Conv.zero _ _ _ _ hw hb
  bn1 := R34TrainBn.exact _ _ _ hG hBb hemr hei hXh0 hmXh
  cv2 := R34Conv.zero _ _ _ _ hw hb
  bn2 := R34TrainBn.exact _ _ _ hG hBb hemr hei hXh0 hmXh

noncomputable def R34TrainDownBlk.exact (ic oc h w : Nat) {w' β' ε G Bb emr ei Xh : ℝ}
    (hw : 0 ≤ w') (hb : 0 ≤ β') (hG : 0 ≤ G) (hBb : 0 ≤ Bb) (hemr : 0 ≤ emr) (hei : 0 ≤ ei)
    (hXh0 : 0 ≤ Xh) (hmXh : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei Xh where
  cv1 := R34Conv.zero _ _ _ _ hw hb
  bn1 := R34TrainBn.exact _ _ _ hG hBb hemr hei hXh0 hmXh
  cv2 := R34Conv.zero _ _ _ _ hw hb
  bn2 := R34TrainBn.exact _ _ _ hG hBb hemr hei hXh0 hmXh
  cvp := R34Conv.zero _ _ _ _ hw hb
  bnp := R34TrainBn.exact _ _ _ hG hBb hemr hei hXh0 hmXh

noncomputable def R34TrainWeights.exact {ε : ℝ} :
    R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) :=
  have h : (0:ℝ) ≤ 21/10 := by norm_num
  have he : (0:ℝ) ≤ 1/100 := by norm_num
  { stem := R34Conv.zero _ _ _ _ h h
    bns := R34TrainBn.exact _ _ _ h h (by norm_num) he (by norm_num) (by norm_num)
    a0 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    a1 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    a2 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    d2 := R34TrainDownBlk.exact _ _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    b0 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    b1 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    b2 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    d3 := R34TrainDownBlk.exact _ _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    c0 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    c1 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    c2 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    c3 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    c4 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    d4 := R34TrainDownBlk.exact _ _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    e0 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    e1 := R34TrainIdBlk.exact _ _ _ h h h h (by norm_num) he (by norm_num) (by norm_num)
    head := R34Head.zero _ _ h h }

example (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |r34TrainForwardF binary32 (R34TrainWeights.exact (ε := 1/100000)) x j
      - r34TrainForward (R34TrainWeights.exact (ε := 1/100000)) x j| ≤ 1752 * 10 ^ 78 :=
  r34_train_float_logits_le binary32 binary32_u.le (by norm_num) _ x hx j

end Proofs
