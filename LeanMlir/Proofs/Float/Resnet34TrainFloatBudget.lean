import LeanMlir.Proofs.Float.Resnet34FloatBudget
import LeanMlir.Proofs.Float.BnXhatFloatBridge

/-! # A NUMBER for ResNet-34 at TRAINING-mode BatchNorm — and it is the CAP

The **training-mode** twin of `Resnet34FloatBudget.lean`: the same `[3,4,6,3]` net at `224²`,
the same measured profile, the same 90 numeric stages — with `bnPerChannelTensor3` at all 36
BatchNorm sites where that file has `bnPerChannelEvalTensor3`. This is the program the repo
actually **trains** with, and the one the input-gradient numbers in
`Resnet34BackFloatBudget.lean` are taken through.

    output window  ≤ 4.304·10¹⁴⁵      (`r34TrainBridge_mag_le`)
    error bound    ≤ 8.605·10¹⁴⁵      (`r34TrainBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 8.605·10¹⁴⁵` (`r34_train_float_logits_le`).

⭐⭐ **76 orders of that came from §0.1's ESCAPE 2 (2026-09-05), at no new hypothesis.** The
first version of this file was `3.176·10²²¹ / 6.349·10²²¹`, charging each site's certified
window at `|x − μ|·|istd| ≤ 2A·S` — the product of the two factors' own bounds, `S = 317` at the
`ε`-floor. The normalised activation has a bound that mentions neither: **`|x̂| ≤ √n`**
(`bnXhat_sq_le`), which is `112` at the stem and `7` at the deepest block, and which four
backward budget files have called since 2026-09-03. `Maps.bnPerChannelTensor3CappedX`
(`BnXhatFloatBridge.lean`) is that leaf lifted per channel; the profile, the modelled accuracies
and the `ε`-floor are all unchanged. ⭐ r34's reduction widths are perfect squares, so the root
is EXACT here where ConvNeXt's channel counts need the ceiling one.

⚠ **What is modelled, and it is more than the inference net models.** `R34Bn` freezes `μ` and
`v` and supplies only the device `rsqrt`'s accuracy. Training-mode BatchNorm reduces both
statistics out of its own input, so BOTH are device kernels here and both are supplied:
`R34TrainBn.fμ` accurate to `emr·A` — **RELATIVE** to the layer's window, which is the shape a
rounded mean of `n` terms actually has — and `R34TrainBn.fistd` accurate to `ei` absolutely,
the `DeviceRsqrt` standing.

⛔⛔ **This number is the CAP, not the fold, and that is not a detail.** `budget / window =
2.00` is the tell, and escape 2 does not change it — that escape has a modulus half as well as
a window half, and the modulus half is priced and NOT taken here (`BnXhatFloatBridge.lean`'s
header), which costs nothing precisely because every site is capped. All 36 BatchNorm sites go through `FloatBridgesTo.capped`, so what is proved
is *"the float and the real forward both land in the certified window"* — the triangle
inequality — and **not** *"the rounding error folds to this"*, which is what
`r34_float_logits_le` says about the inference net. The two numbers must never be tabled
together without that label (`planning/float_budget_numbers.md` §9), and this one must never be
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

⚠ `emr = 10⁻²` is taken by analogy with the other device accuracies in this tier and is loose:
the rounded mean of `n` terms is `γₙ·A`, which at `n = 12544` and `u = 2⁻²⁴` is `≈ 7.5·10⁻⁴`.

⭐ **No operating point.** The number is stated at the unconditional `ε`-floor `|istd| ≤ 317`,
like the inference number and unlike `Resnet34BackFloatBudget.lean`'s `|istd| ≤ 16`: the chain's
largest goals closed at `10²²¹` with 32 orders of headroom under the shape-dependent `norm_num`
ceiling before escape 2, and escape 2 leaves 108. The hypothesis was never needed and is not
paid.

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
`Maps` arguments is right (`planning/float_budget_numbers.md` §3.30).
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
structure R34TrainWeights (w' β' ε G Bb emr ei : ℝ) where
  stem : R34Conv 64 3 7 7 w' β'
  bns : R34TrainBn 64 112 112 ε G Bb emr ei 112
  a0 : R34TrainIdBlk 64 56 56 w' β' ε G Bb emr ei 56
  a1 : R34TrainIdBlk 64 56 56 w' β' ε G Bb emr ei 56
  a2 : R34TrainIdBlk 64 56 56 w' β' ε G Bb emr ei 56
  d2 : R34TrainDownBlk 64 128 28 28 w' β' ε G Bb emr ei 28
  b0 : R34TrainIdBlk 128 28 28 w' β' ε G Bb emr ei 28
  b1 : R34TrainIdBlk 128 28 28 w' β' ε G Bb emr ei 28
  b2 : R34TrainIdBlk 128 28 28 w' β' ε G Bb emr ei 28
  d3 : R34TrainDownBlk 128 256 14 14 w' β' ε G Bb emr ei 14
  c0 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei 14
  c1 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei 14
  c2 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei 14
  c3 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei 14
  c4 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei 14
  d4 : R34TrainDownBlk 256 512 7 7 w' β' ε G Bb emr ei 7
  e0 : R34TrainIdBlk 512 7 7 w' β' ε G Bb emr ei 7
  e1 : R34TrainIdBlk 512 7 7 w' β' ε G Bb emr ei 7
  head : R34Head 512 10 w' β'

/-- The numeric profile the fold runs at. `R34Profile` with the frozen-mean bound `Mb` dropped
    (there is nothing frozen) and the single device accuracy `es` replaced by the two this mode
    needs: `emr` on the batch mean and `ei` on the inverse stddev. -/
structure R34TrainProfile (M : FloatModel) (ε w' β' G Bb emr ei S q : ℝ) : Prop where
  hw' : 0 ≤ w'
  hβ' : 0 ≤ β'
  hG : 0 ≤ G
  hBb : 0 ≤ Bb
  hemr : 0 ≤ emr
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
    (M : FloatModel) (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
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
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q) (hc : 0 < c) (hhw : 0 < h * w)
    {Ā Ē Ā' Ē' : ℝ}
    (hĀ' : G * Xh + Bb + bnNormBudgetX q Xh (2 * Ā) S G Bb (emr * Ā) ei ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    (B.bridge M P hc hhw).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnPerChannelTensor3CappedX (h := h) (w := w) M B.γ B.β B.fμ B.fistd
    (fun A => emr * A) (fun _ => ei) hc hhw P.hε B.hγ B.hβ B.hmean B.histd
    (fun v => (bnIstd_abs_le v P.hε).trans P.hSε) B.hXh0 B.hmXh P.hq P.hG P.hBb P.hS0
    (fun _A _h0 hle => mul_le_mul_of_nonneg_left hle P.hemr) (fun _ _ _ => le_rfl) hĀ' hĒ'

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
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w) :
    FloatBridgesTo B.fwd (B.fwdF M) :=
  floatBridgesTo_r34IdBlock (h := h) (w := w) M B.cv1.W B.cv1.b B.cv2.W B.cv2.b
    B.bn1.fwd (B.bn1.fwdF M) B.bn2.fwd (B.bn2.fwdF M)
    P.hw' P.hβ' hn B.cv1.hW B.cv1.hb B.cv2.hW B.cv2.hb
    (B.bn1.bridge M P hc hhw) (B.bn2.bridge M P hc hhw)

/-- **This block's numeric envelope** — four numeric stages then the residual fan-in. Ten
    inequalities, of which the two BN error clauses are caps. -/
theorem R34TrainIdBlk.maps {c h w : Nat} (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei Xh)
    (M : FloatModel) (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
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
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
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
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
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
noncomputable def r34TrainForward (W : R34TrainWeights w' β' ε G Bb emr ei) :
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
noncomputable def r34TrainForwardF (M : FloatModel) (W : R34TrainWeights w' β' ε G Bb emr ei) :
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
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q) (W : R34TrainWeights w' β' ε G Bb emr ei) :
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
    R34TrainProfile M ε (21/10) (21/10) (21/10) (21/10) (1/100) (1/100) 317 u32 where
  hw' := by norm_num
  hβ' := by norm_num
  hG := by norm_num
  hBb := by norm_num
  hemr := by norm_num
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
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).Maps 1 0
      (4304 * 10 ^ 142) (8605 * 10 ^ 142) := by
  have hP := r34TrainProfile_committed M hMu hε5
  have t1 := FloatBridgesTo.Maps.flatConvStride2 (h := 112) (w := 112) M W.stem.W
    W.stem.b hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb
    (M.gamma_num (q := 8882 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 3109 / 10 ^ 1) (Ē' := 2761 / 10 ^ 6) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
  have t2 := t1.comp (by norm_num) (W.bns.maps M hP (by norm_num) (by norm_num)
    (Ā := 3109 / 10 ^ 1) (Ē := 2761 / 10 ^ 6) (Ā' := 2321) (Ē' := 4642) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num))
  have t3 := t2.comp (by norm_num) (FloatBridgesTo.Maps.relu (n := 64 * 112 * 112))
  have t4 := t3.comp (by norm_num)
    (FloatBridgesTo.Maps.maxPool3s2 (c := 64) (h := 56) (w := 56))
  have t5 := t4.comp (by norm_num) (W.a0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 2321) (Ē := 4642)
    (A1 := 2808 * 10 ^ 3) (E1 := 5616 * 10 ^ 3) (A2 := 1882 * 10 ^ 4) (E2 := 3764 * 10 ^ 4)
    (A3 := 2277 * 10 ^ 7) (E3 := 4554 * 10 ^ 7) (A4 := 1526 * 10 ^ 8) (E4 := 3052 * 10 ^ 8)
    (Ā' := 1527 * 10 ^ 8) (Ē' := 3053 * 10 ^ 8)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t6 := t5.comp (by norm_num) (W.a1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 1527 * 10 ^ 8) (Ē := 3053 * 10 ^ 8)
    (A1 := 1848 * 10 ^ 11) (E1 := 3694 * 10 ^ 11) (A2 := 1239 * 10 ^ 12) (E2 := 2478 * 10 ^ 12)
    (A3 := 1499 * 10 ^ 15) (E3 := 2998 * 10 ^ 15) (A4 := 1005 * 10 ^ 16) (E4 := 2010 * 10 ^ 16)
    (Ā' := 1006 * 10 ^ 16) (Ē' := 2011 * 10 ^ 16)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t7 := t6.comp (by norm_num) (W.a2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 1006 * 10 ^ 16) (Ē := 2011 * 10 ^ 16)
    (A1 := 1217 * 10 ^ 19) (E1 := 2433 * 10 ^ 19) (A2 := 8154 * 10 ^ 19) (E2 := 1631 * 10 ^ 20)
    (A3 := 9864 * 10 ^ 22) (E3 := 1973 * 10 ^ 23) (A4 := 6609 * 10 ^ 23) (E4 := 1322 * 10 ^ 24)
    (Ā' := 6610 * 10 ^ 23) (Ē' := 1323 * 10 ^ 24)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t8 := t7.comp (by norm_num) (W.d2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 3446 / 10 ^ 8) (g2 := 6879 / 10 ^ 8) (gp := 3934 / 10 ^ 9)
    (Ā := 6610 * 10 ^ 23) (Ē := 1323 * 10 ^ 24)
    (P1 := 8884 * 10 ^ 25) (Q1 := 1779 * 10 ^ 26) (P2 := 5952 * 10 ^ 26) (Q2 := 1191 * 10 ^ 27)
    (A1 := 7996 * 10 ^ 26) (E1 := 1601 * 10 ^ 27) (A2 := 5357 * 10 ^ 27) (E2 := 1072 * 10 ^ 28)
    (A3 := 1297 * 10 ^ 31) (E3 := 2594 * 10 ^ 31) (A4 := 8689 * 10 ^ 31) (E4 := 1738 * 10 ^ 32)
    (Ā' := 8690 * 10 ^ 31) (Ē' := 1739 * 10 ^ 32)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t9 := t8.comp (by norm_num) (W.b0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 8690 * 10 ^ 31) (Ē := 1739 * 10 ^ 32)
    (A1 := 2103 * 10 ^ 35) (E1 := 4208 * 10 ^ 35) (A2 := 1409 * 10 ^ 36) (E2 := 2818 * 10 ^ 36)
    (A3 := 3409 * 10 ^ 39) (E3 := 6819 * 10 ^ 39) (A4 := 2284 * 10 ^ 40) (E4 := 4568 * 10 ^ 40)
    (Ā' := 2285 * 10 ^ 40) (Ē' := 4569 * 10 ^ 40)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t10 := t9.comp (by norm_num) (W.b1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 2285 * 10 ^ 40) (Ē := 4569 * 10 ^ 40)
    (A1 := 5529 * 10 ^ 43) (E1 := 1106 * 10 ^ 44) (A2 := 3705 * 10 ^ 44) (E2 := 7410 * 10 ^ 44)
    (A3 := 8964 * 10 ^ 47) (E3 := 1793 * 10 ^ 48) (A4 := 6006 * 10 ^ 48) (E4 := 1202 * 10 ^ 49)
    (Ā' := 6007 * 10 ^ 48) (Ē' := 1203 * 10 ^ 49)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t11 := t10.comp (by norm_num) (W.b2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 6007 * 10 ^ 48) (Ē := 1203 * 10 ^ 49)
    (A1 := 1454 * 10 ^ 52) (E1 := 2911 * 10 ^ 52) (A2 := 9741 * 10 ^ 52) (E2 := 1949 * 10 ^ 53)
    (A3 := 2357 * 10 ^ 56) (E3 := 4716 * 10 ^ 56) (A4 := 1580 * 10 ^ 57) (E4 := 3160 * 10 ^ 57)
    (Ā' := 1581 * 10 ^ 57) (Ē' := 3161 * 10 ^ 57)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t12 := t11.comp (by norm_num) (W.d3.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 6879 / 10 ^ 8) (g2 := 1375 / 10 ^ 7) (gp := 7749 / 10 ^ 9)
    (Ā := 1581 * 10 ^ 57) (Ē := 3161 * 10 ^ 57)
    (P1 := 4250 * 10 ^ 59) (Q1 := 8497 * 10 ^ 59) (P2 := 2848 * 10 ^ 60) (Q2 := 5696 * 10 ^ 60)
    (A1 := 3826 * 10 ^ 60) (E1 := 7648 * 10 ^ 60) (A2 := 2564 * 10 ^ 61) (E2 := 5128 * 10 ^ 61)
    (A3 := 1241 * 10 ^ 65) (E3 := 2482 * 10 ^ 65) (A4 := 8314 * 10 ^ 65) (E4 := 1663 * 10 ^ 66)
    (Ā' := 8315 * 10 ^ 65) (Ē' := 1664 * 10 ^ 66)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t13 := t12.comp (by norm_num) (W.c0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 8315 * 10 ^ 65) (Ē := 1664 * 10 ^ 66)
    (A1 := 4024 * 10 ^ 69) (E1 := 8053 * 10 ^ 69) (A2 := 2696 * 10 ^ 70) (E2 := 5392 * 10 ^ 70)
    (A3 := 1305 * 10 ^ 74) (E3 := 2610 * 10 ^ 74) (A4 := 8743 * 10 ^ 74) (E4 := 1749 * 10 ^ 75)
    (Ā' := 8744 * 10 ^ 74) (Ē' := 1750 * 10 ^ 75)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t14 := t13.comp (by norm_num) (W.c1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 8744 * 10 ^ 74) (Ē := 1750 * 10 ^ 75)
    (A1 := 4232 * 10 ^ 78) (E1 := 8469 * 10 ^ 78) (A2 := 2836 * 10 ^ 79) (E2 := 5672 * 10 ^ 79)
    (A3 := 1373 * 10 ^ 83) (E3 := 2745 * 10 ^ 83) (A4 := 9199 * 10 ^ 83) (E4 := 1840 * 10 ^ 84)
    (Ā' := 9200 * 10 ^ 83) (Ē' := 1841 * 10 ^ 84)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t15 := t14.comp (by norm_num) (W.c2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 9200 * 10 ^ 83) (Ē := 1841 * 10 ^ 84)
    (A1 := 4452 * 10 ^ 87) (E1 := 8910 * 10 ^ 87) (A2 := 2983 * 10 ^ 88) (E2 := 5966 * 10 ^ 88)
    (A3 := 1444 * 10 ^ 92) (E3 := 2888 * 10 ^ 92) (A4 := 9674 * 10 ^ 92) (E4 := 1935 * 10 ^ 93)
    (Ā' := 9675 * 10 ^ 92) (Ē' := 1936 * 10 ^ 93)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t16 := t15.comp (by norm_num) (W.c3.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 9675 * 10 ^ 92) (Ē := 1936 * 10 ^ 93)
    (A1 := 4682 * 10 ^ 96) (E1 := 9370 * 10 ^ 96) (A2 := 3137 * 10 ^ 97) (E2 := 6274 * 10 ^ 97)
    (A3 := 1519 * 10 ^ 101) (E3 := 3037 * 10 ^ 101) (A4 := 1018 * 10 ^ 102) (E4 := 2036 * 10 ^ 102)
    (Ā' := 1019 * 10 ^ 102) (Ē' := 2037 * 10 ^ 102)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t17 := t16.comp (by norm_num) (W.c4.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 1019 * 10 ^ 102) (Ē := 2037 * 10 ^ 102)
    (A1 := 4932 * 10 ^ 105) (E1 := 9858 * 10 ^ 105) (A2 := 3305 * 10 ^ 106) (E2 := 6610 * 10 ^ 106)
    (A3 := 1600 * 10 ^ 110) (E3 := 3199 * 10 ^ 110) (A4 := 1072 * 10 ^ 111) (E4 := 2144 * 10 ^ 111)
    (Ā' := 1073 * 10 ^ 111) (Ē' := 2145 * 10 ^ 111)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t18 := t17.comp (by norm_num) (W.d4.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 1375 / 10 ^ 7) (g2 := 2749 / 10 ^ 7) (gp := 1538 / 10 ^ 8)
    (Ā := 1073 * 10 ^ 111) (Ē := 2145 * 10 ^ 111)
    (P1 := 5769 * 10 ^ 113) (Q1 := 1154 * 10 ^ 114) (P2 := 3865 * 10 ^ 114) (Q2 := 7730 * 10 ^ 114)
    (A1 := 5193 * 10 ^ 114) (E1 := 1039 * 10 ^ 115) (A2 := 3479 * 10 ^ 115) (E2 := 6958 * 10 ^ 115)
    (A3 := 3368 * 10 ^ 119) (E3 := 6736 * 10 ^ 119) (A4 := 2257 * 10 ^ 120) (E4 := 4514 * 10 ^ 120)
    (Ā' := 2258 * 10 ^ 120) (Ē' := 4515 * 10 ^ 120)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t19 := t18.comp (by norm_num) (W.e0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 2749 / 10 ^ 7)
    (Ā := 2258 * 10 ^ 120) (Ē := 4515 * 10 ^ 120)
    (A1 := 2186 * 10 ^ 124) (E1 := 4371 * 10 ^ 124) (A2 := 1465 * 10 ^ 125) (E2 := 2930 * 10 ^ 125)
    (A3 := 1419 * 10 ^ 129) (E3 := 2837 * 10 ^ 129) (A4 := 9507 * 10 ^ 129) (E4 := 1902 * 10 ^ 130)
    (Ā' := 9508 * 10 ^ 129) (Ē' := 1903 * 10 ^ 130)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t20 := t19.comp (by norm_num) (W.e1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 2749 / 10 ^ 7)
    (Ā := 9508 * 10 ^ 129) (Ē := 1903 * 10 ^ 130)
    (A1 := 9204 * 10 ^ 133) (E1 := 1843 * 10 ^ 134) (A2 := 6167 * 10 ^ 134) (E2 := 1234 * 10 ^ 135)
    (A3 := 5970 * 10 ^ 138) (E3 := 1195 * 10 ^ 139) (A4 := 4000 * 10 ^ 139) (E4 := 8000 * 10 ^ 139)
    (Ā' := 4001 * 10 ^ 139) (Ē' := 8001 * 10 ^ 139)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t21 := t20.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      hMu (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 4001 * 10 ^ 139) (Ē := 8001 * 10 ^ 139) (Ā' := 4002 * 10 ^ 139) (Ē' := 8002 * 10 ^ 139) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have t22 := t21.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num)
      W.head.hW W.head.hb (M.gamma_num (q := 3064 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 4002 * 10 ^ 139) (Ē := 8002 * 10 ^ 139) (Ā' := 4304 * 10 ^ 142) (Ē' := 8605 * 10 ^ 142) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  exact t22

/-- The deployed ResNet-34 training bridge's certified output window at the committed profile:
    `≤ 4.304·10¹⁴⁵`. ⭐ This half is an honest fold — the cap touches only the modulus — and it
    is the half escape 2 moved, by 76 orders. -/
theorem r34TrainBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).mag 1 ≤ 4304 * 10 ^ 142 :=
  (r34TrainBridge_maps M hMu hε5 W).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ResNet-34 training bridge's error bound at the committed profile:
    `≤ 8.605·10¹⁴⁵`, which is `2.00 ×` the window — the tell that this is
    `FloatBridgesTo.capped`'s triangle inequality and not the interval fold (§9). -/
theorem r34TrainBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).fresh 1 ≤ 8605 * 10 ^ 142 :=
  (r34TrainBridge_maps M hMu hε5 W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed ResNet-34 TRAINING-mode forward is within `8.605·10¹⁴⁵` of the certified
    real training forward, per logit**, on inputs of magnitude `≤ 1`, at the measured parameter
    profile, for `ε ≥ 10⁻⁵`, any device batch mean accurate to `10⁻²` relative and any device
    inverse-stddev accurate to `10⁻²` absolute, and any rounding model at binary32 accuracy.

    ⭐⭐ **The first statement in this repo about the program it actually trains with** — every
    other committed forward number is at inference normalisation. ⛔ **It is the CAP**: read it
    as *"the float and the real forward both land in the certified window"*, never as *"the
    rounding error folds to this"*. `8.605 / 4.304 = 2.00` is the tell, and the fold it replaces
    is **`3.494·10⁴⁹⁹³`** — `10⁷⁴¹⁹` before escape 2, so the window half is worth 2426 orders on
    the fold as well and leaves it 4740 past the ceiling anyway. The quadratic shrinks with the
    window it is quadratic in and still does not fit; the cap is not a shortcut past a fold that
    exists (§9). -/
theorem r34_train_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |r34TrainForwardF M W x j - r34TrainForward W x j| ≤ 8605 * 10 ^ 142 :=
  (r34TrainBridge_maps M hMu hε5 W).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed training forward, and the graph denotes it
-- ════════════════════════════════════════════════════════════════

/-- **The record-bundled forward IS the committed training net.** `r34TrainForward` unfolds to
    `resnet34Forward_full_pc` at the record's projections — the training twin of
    `r34EvalForward_eq_full_pc_eval`, and a `rfl` for the same reason: `rblkPC_eq_gen` and
    `rblkPStridedPC_eq_gen` are both `rfl`, so the skeleton's block slots take exactly the maps
    the record builds. -/
theorem r34TrainForward_eq_full_pc (W : R34TrainWeights w' β' ε G Bb emr ei) :
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
theorem r34TrainGraph_faithful (epsStr : String) (W : R34TrainWeights w' β' ε G Bb emr ei)
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
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100))
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
    W.head.W W.head.b x j| ≤ 8605 * 10 ^ 142 :=
  r34_train_float_logits_le M hMu hε5 W x hx j

end Proofs
