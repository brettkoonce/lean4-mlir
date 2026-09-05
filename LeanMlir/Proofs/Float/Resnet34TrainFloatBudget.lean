import LeanMlir.Proofs.Float.Resnet34FloatBudget

/-! # A NUMBER for ResNet-34 at TRAINING-mode BatchNorm — and it is the CAP

The **training-mode** twin of `Resnet34FloatBudget.lean`: the same `[3,4,6,3]` net at `224²`,
the same measured profile, the same 90 numeric stages — with `bnPerChannelTensor3` at all 36
BatchNorm sites where that file has `bnPerChannelEvalTensor3`. This is the program the repo
actually **trains** with, and the one the input-gradient numbers in
`Resnet34BackFloatBudget.lean` are taken through.

    output window  ≤ 3.176·10²²¹      (`r34TrainBridge_mag_le`)
    error bound    ≤ 6.349·10²²¹      (`r34TrainBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 6.349·10²²¹` (`r34_train_float_logits_le`).

⛔⛔ **This number is the CAP, not the fold, and that is not a detail.** `budget / window =
2.00` is the tell. All 36 BatchNorm sites go through `FloatBridgesTo.capped`, so what is proved
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
statistics out of its own input, so BOTH are device kernels here and both are supplied:
`R34TrainBn.fμ` accurate to `emr·A` — **RELATIVE** to the layer's window, which is the shape a
rounded mean of `n` terms actually has — and `R34TrainBn.fistd` accurate to `ei` absolutely,
the `DeviceRsqrt` standing. Everything else is proved. ⚠ `emr = 10⁻²` is taken by analogy with
the other device accuracies in this tier and is loose: the rounded mean of `n` terms is `γₙ·A`,
which at `n = 12544` and `u = 2⁻²⁴` is `≈ 7.5·10⁻⁴`.

⭐ **No operating point.** The fold is stated at the unconditional `ε`-floor `|istd| ≤ 317`,
like the inference number and unlike `Resnet34BackFloatBudget.lean`'s `|istd| ≤ 16`: the
eighteen largest goals the chain asserts close at `10²²¹` with 32 orders of headroom under the
shape-dependent `norm_num` ceiling, so the hypothesis was not needed and was not paid.

**The tie is closed at the graph**, exactly as the inference number's is:
`r34TrainForward_eq_full_pc` is a `rfl` onto `resnet34Forward_full_pc` — the committed training
net — and `r34TrainGraph_faithful` carries `resnet34FwdGraphFullPC_faithful` the rest of the
way, so the typed `SHlo` graph every line of `@resnet34_fwd` renders denotes the forward this
file bounds. `r34_train_float_logits_le_committed` restates the number on that net.

Provenance for the 180 numerals: `scripts/float_budget_envelope.py`'s `r34_train_chain(cap =
True)`, re-asserted by `verify_r34_train` (180 inequalities, and it separately asserts that the
cap is the smaller branch at all 36 sites, so the label above is measured rather than assumed)
before a line of this file was emitted.
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
structure R34TrainBn (c h w : Nat) (ε G Bb emr ei : ℝ) where
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

/-- An identity basic block at training-mode BN: two 3×3 convs, two BN sites. 13 of the 16.
    ⚠ The spatial dims sit in the TYPE here where `R34IdBlk` takes them at each use site — the
    BN record needs `h*w` to state its statistics' accuracies. -/
structure R34TrainIdBlk (c h w : Nat) (w' β' ε G Bb emr ei : ℝ) where
  cv1 : R34Conv c c 3 3 w' β'
  bn1 : R34TrainBn c h w ε G Bb emr ei
  cv2 : R34Conv c c 3 3 w' β'
  bn2 : R34TrainBn c h w ε G Bb emr ei

/-- A downsample basic block at training-mode BN: two body convs (the first stride-2), the 1×1
    stride-2 option-B projection, and three BN sites. -/
structure R34TrainDownBlk (ic oc h w : Nat) (w' β' ε G Bb emr ei : ℝ) where
  cv1 : R34Conv oc ic 3 3 w' β'
  bn1 : R34TrainBn oc h w ε G Bb emr ei
  cv2 : R34Conv oc oc 3 3 w' β'
  bn2 : R34TrainBn oc h w ε G Bb emr ei
  cvp : R34Conv oc ic 1 1 w' β'
  bnp : R34TrainBn oc h w ε G Bb emr ei

/-- **The whole training net's stored parameters** — 37 convolutions, the classifier, and
    **36** training-mode BN sites (1 stem + 13 identity blocks × 2 + 3 downsample blocks × 3).
    ⚠ Count them: the docstrings of `Resnet34FloatBudget.lean` and of `planning/` say 33, which
    is the number of *body* convolutions and not the number of normalisations. -/
structure R34TrainWeights (w' β' ε G Bb emr ei : ℝ) where
  stem : R34Conv 64 3 7 7 w' β'
  bns : R34TrainBn 64 112 112 ε G Bb emr ei
  a0 : R34TrainIdBlk 64 56 56 w' β' ε G Bb emr ei
  a1 : R34TrainIdBlk 64 56 56 w' β' ε G Bb emr ei
  a2 : R34TrainIdBlk 64 56 56 w' β' ε G Bb emr ei
  d2 : R34TrainDownBlk 64 128 28 28 w' β' ε G Bb emr ei
  b0 : R34TrainIdBlk 128 28 28 w' β' ε G Bb emr ei
  b1 : R34TrainIdBlk 128 28 28 w' β' ε G Bb emr ei
  b2 : R34TrainIdBlk 128 28 28 w' β' ε G Bb emr ei
  d3 : R34TrainDownBlk 128 256 14 14 w' β' ε G Bb emr ei
  c0 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei
  c1 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei
  c2 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei
  c3 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei
  c4 : R34TrainIdBlk 256 14 14 w' β' ε G Bb emr ei
  d4 : R34TrainDownBlk 256 512 7 7 w' β' ε G Bb emr ei
  e0 : R34TrainIdBlk 512 7 7 w' β' ε G Bb emr ei
  e1 : R34TrainIdBlk 512 7 7 w' β' ε G Bb emr ei
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

variable {M : FloatModel} {ε w' β' G Bb emr ei S q : ℝ}

/-- The certified ℝ training-mode BN at this site. -/
noncomputable def R34TrainBn.fwd {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei) :
    Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3 c h w ε B.γ B.β

/-- The deployed float training-mode BN at this site: the rounded normalize chain over the
    device's own two reductions. -/
noncomputable def R34TrainBn.fwdF {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3FV M B.γ B.β B.fμ B.fistd

/-- ⛔ **This BN site's bridge, CAPPED.** The `.capped` is the whole difference between this
    file and `Resnet34FloatBudget.lean`, and it is what makes the number the triangle
    inequality rather than the fold (§9). Without it the site's modulus carries
    `G·2Ā·(8Ā·Ē/(2ε√ε))`, quadratic in the window, and 36 of them square to `10⁷⁴¹⁹`. -/
noncomputable def R34TrainBn.bridge {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei)
    (M : FloatModel) (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) :
    FloatBridgesTo B.fwd (B.fwdF M) :=
  (floatBridgesTo_bnPerChannelTensor3 (h := h) (w := w) M B.γ B.β B.fμ B.fistd
    (fun A => emr * A) (fun _ => ei) hc hhw P.hε B.hγ B.hβ B.hmean B.histd
    (fun v => (bnIstd_abs_le v P.hε).trans P.hSε)).capped

/-- **This BN site's numeric envelope** — and note there is only ONE numeric inequality with
    any content. The window clause is the training-mode leaf's own; the error clause is
    `2·Ā' ≤ Ē'`, which mentions neither the inherited error nor `ε`. That is the mechanism: at
    a capped site §0.1's quadratic is never turned into a numeral, so `norm_num` never meets
    it. -/
theorem R34TrainBn.maps {c h w : Nat} (B : R34TrainBn c h w ε G Bb emr ei) (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q) (hc : 0 < c) (hhw : 0 < h * w)
    {Ā Ē Ā' Ē' : ℝ}
    (hĀ' : G * (2 * Ā * S) + Bb + bnNormBudget q (2 * Ā) S G Bb (emr * Ā) ei ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    (B.bridge M P hc hhw).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnPerChannelTensor3Capped (h := h) (w := w) M B.γ B.β B.fμ B.fistd
    (fun A => emr * A) (fun _ => ei) hc hhw P.hε B.hγ B.hβ B.hmean B.histd
    (fun v => (bnIstd_abs_le v P.hε).trans P.hSε) P.hq P.hG P.hBb P.hS0
    (fun _A _h0 hle => mul_le_mul_of_nonneg_left hle P.hemr) (fun _ _ _ => le_rfl) hĀ' hĒ'

-- ════════════════════════════════════════════════════════════════
-- § One block: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

/-- The certified ℝ identity block at training-mode BN. -/
noncomputable def R34TrainIdBlk.fwd {c h w : Nat}
    (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei) : Vec (c * h * w) → Vec (c * h * w) :=
  rblkGen (h := h) (w := w) B.cv1.W B.cv1.b B.bn1.fwd B.cv2.W B.cv2.b B.bn2.fwd

/-- The deployed float identity block. -/
noncomputable def R34TrainIdBlk.fwdF {c h w : Nat}
    (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei) (M : FloatModel) :
    Vec (c * h * w) → Vec (c * h * w) :=
  rblkGenF M B.cv1.W B.cv1.b (B.bn1.fwdF M) B.cv2.W B.cv2.b (B.bn2.fwdF M)

/-- This block's bridge. ⭐ `floatBridgesTo_r34IdBlock` is generic in the normalisation, so the
    inference and training nets share one block bridge and the capped BN is just an argument. -/
noncomputable def R34TrainIdBlk.bridge {c h w : Nat}
    (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei) (M : FloatModel)
    (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w) :
    FloatBridgesTo B.fwd (B.fwdF M) :=
  floatBridgesTo_r34IdBlock (h := h) (w := w) M B.cv1.W B.cv1.b B.cv2.W B.cv2.b
    B.bn1.fwd (B.bn1.fwdF M) B.bn2.fwd (B.bn2.fwdF M)
    P.hw' P.hβ' hn B.cv1.hW B.cv1.hb B.cv2.hW B.cv2.hb
    (B.bn1.bridge M P hc hhw) (B.bn2.bridge M P hc hhw)

/-- **This block's numeric envelope** — four numeric stages then the residual fan-in. Ten
    inequalities, of which the two BN error clauses are caps. -/
theorem R34TrainIdBlk.maps {c h w : Nat} (B : R34TrainIdBlk c h w w' β' ε G Bb emr ei)
    (M : FloatModel) (P : R34TrainProfile M ε w' β' G Bb emr ei S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w)
    {g Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ}
    (hg : (1 + M.u) ^ (c * 3 * 3 + 2) - 1 ≤ g)
    (c1A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (c1E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (n1A : G * (2 * A1 * S) + Bb + bnNormBudget q (2 * A1) S G Bb (emr * A1) ei ≤ A2)
    (n1cap : 2 * A2 ≤ E2)
    (c2A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * A2 + β') ≤ A3)
    (c2E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (A2 + E2) + β')
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E3)
    (n2A : G * (2 * A3 * S) + Bb + bnNormBudget q (2 * A3) S G Bb (emr * A3) ei ≤ A4)
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
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  rblkStridedGen (h := h) (w := w) B.cv1.W B.cv1.b B.bn1.fwd B.cv2.W B.cv2.b B.bn2.fwd
    B.cvp.W B.cvp.b B.bnp.fwd

/-- The deployed float downsample block. -/
noncomputable def R34TrainDownBlk.fwdF {ic oc h w : Nat}
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei) (M : FloatModel) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  rblkStridedGenF M B.cv1.W B.cv1.b (B.bn1.fwdF M) B.cv2.W B.cv2.b (B.bn2.fwdF M)
    B.cvp.W B.cvp.b (B.bnp.fwdF M)

/-- This block's bridge. -/
noncomputable def R34TrainDownBlk.bridge {ic oc h w : Nat}
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei) (M : FloatModel)
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
    (B : R34TrainDownBlk ic oc h w w' β' ε G Bb emr ei) (M : FloatModel)
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
    (pnA : G * (2 * P1 * S) + Bb + bnNormBudget q (2 * P1) S G Bb (emr * P1) ei ≤ P2)
    (pncap : 2 * P2 ≤ Q2)
    (c1A : (1 + g1) * (((ic * 3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (c1E : g1 * (((ic * 3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (n1A : G * (2 * A1 * S) + Bb + bnNormBudget q (2 * A1) S G Bb (emr * A1) ei ≤ A2)
    (n1cap : 2 * A2 ≤ E2)
    (c2A : (1 + g2) * (((oc * 3 * 3 : ℕ) : ℝ) * w' * A2 + β') ≤ A3)
    (c2E : g2 * (((oc * 3 * 3 : ℕ) : ℝ) * w' * (A2 + E2) + β')
            + ((oc * 3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E3)
    (n2A : G * (2 * A3 * S) + Bb + bnNormBudget q (2 * A3) S G Bb (emr * A3) ei ≤ A4)
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
    closes is the triangle inequality (§9). ⭐ Every one of the eighteen largest goals also
    closes at `S = 8`, so the `ε`-floor was not overpaid. -/
theorem r34TrainBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).Maps 1 0
      (3176 * 10 ^ 218) (6349 * 10 ^ 218) := by
  have hP := r34TrainProfile_committed M hMu hε5
  have t1 := FloatBridgesTo.Maps.flatConvStride2 (h := 112) (w := 112) M W.stem.W
    W.stem.b hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb
    (M.gamma_num (q := 8882 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 3109 / 10 ^ 1) (Ē' := 2761 / 10 ^ 6) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
  have t2 := t1.comp (by norm_num) (W.bns.maps M hP (by norm_num) (by norm_num)
    (Ā := 3109 / 10 ^ 1) (Ē := 2761 / 10 ^ 6) (Ā' := 4161 * 10 ^ 2) (Ē' := 8322 * 10 ^ 2) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num))
  have t3 := t2.comp (by norm_num) (FloatBridgesTo.Maps.relu (n := 64 * 112 * 112))
  have t4 := t3.comp (by norm_num)
    (FloatBridgesTo.Maps.maxPool3s2 (c := 64) (h := 56) (w := 56))
  have t5 := t4.comp (by norm_num) (W.a0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 4161 * 10 ^ 2) (Ē := 8322 * 10 ^ 2)
    (A1 := 5034 * 10 ^ 5) (E1 := 1007 * 10 ^ 6) (A2 := 6736 * 10 ^ 8) (E2 := 1348 * 10 ^ 9)
    (A3 := 8149 * 10 ^ 11) (E3 := 1631 * 10 ^ 12) (A4 := 1091 * 10 ^ 15) (E4 := 2182 * 10 ^ 15)
    (Ā' := 1092 * 10 ^ 15) (Ē' := 2183 * 10 ^ 15)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t6 := t5.comp (by norm_num) (W.a1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 1092 * 10 ^ 15) (Ē := 2183 * 10 ^ 15)
    (A1 := 1321 * 10 ^ 18) (E1 := 2641 * 10 ^ 18) (A2 := 1768 * 10 ^ 21) (E2 := 3536 * 10 ^ 21)
    (A3 := 2139 * 10 ^ 24) (E3 := 4278 * 10 ^ 24) (A4 := 2863 * 10 ^ 27) (E4 := 5726 * 10 ^ 27)
    (Ā' := 2864 * 10 ^ 27) (Ē' := 5727 * 10 ^ 27)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t7 := t6.comp (by norm_num) (W.a2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 3446 / 10 ^ 8)
    (Ā := 2864 * 10 ^ 27) (Ē := 5727 * 10 ^ 27)
    (A1 := 3465 * 10 ^ 30) (E1 := 6928 * 10 ^ 30) (A2 := 4637 * 10 ^ 33) (E2 := 9274 * 10 ^ 33)
    (A3 := 5610 * 10 ^ 36) (E3 := 1122 * 10 ^ 37) (A4 := 7507 * 10 ^ 39) (E4 := 1502 * 10 ^ 40)
    (Ā' := 7508 * 10 ^ 39) (Ē' := 1503 * 10 ^ 40)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t8 := t7.comp (by norm_num) (W.d2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 3446 / 10 ^ 8) (g2 := 6879 / 10 ^ 8) (gp := 3934 / 10 ^ 9)
    (Ā := 7508 * 10 ^ 39) (Ē := 1503 * 10 ^ 40)
    (P1 := 1010 * 10 ^ 42) (Q1 := 2021 * 10 ^ 42) (P2 := 1352 * 10 ^ 45) (Q2 := 2704 * 10 ^ 45)
    (A1 := 9082 * 10 ^ 42) (E1 := 1819 * 10 ^ 43) (A2 := 1216 * 10 ^ 46) (E2 := 2432 * 10 ^ 46)
    (A3 := 2942 * 10 ^ 49) (E3 := 5885 * 10 ^ 49) (A4 := 3937 * 10 ^ 52) (E4 := 7874 * 10 ^ 52)
    (Ā' := 3938 * 10 ^ 52) (Ē' := 7875 * 10 ^ 52)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t9 := t8.comp (by norm_num) (W.b0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 3938 * 10 ^ 52) (Ē := 7875 * 10 ^ 52)
    (A1 := 9528 * 10 ^ 55) (E1 := 1906 * 10 ^ 56) (A2 := 1275 * 10 ^ 59) (E2 := 2550 * 10 ^ 59)
    (A3 := 3085 * 10 ^ 62) (E3 := 6170 * 10 ^ 62) (A4 := 4129 * 10 ^ 65) (E4 := 8258 * 10 ^ 65)
    (Ā' := 4130 * 10 ^ 65) (Ē' := 8259 * 10 ^ 65)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t10 := t9.comp (by norm_num) (W.b1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 4130 * 10 ^ 65) (Ē := 8259 * 10 ^ 65)
    (A1 := 9992 * 10 ^ 68) (E1 := 1999 * 10 ^ 69) (A2 := 1338 * 10 ^ 72) (E2 := 2676 * 10 ^ 72)
    (A3 := 3238 * 10 ^ 75) (E3 := 6475 * 10 ^ 75) (A4 := 4333 * 10 ^ 78) (E4 := 8666 * 10 ^ 78)
    (Ā' := 4334 * 10 ^ 78) (Ē' := 8667 * 10 ^ 78)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t11 := t10.comp (by norm_num) (W.b2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 6879 / 10 ^ 8)
    (Ā := 4334 * 10 ^ 78) (Ē := 8667 * 10 ^ 78)
    (A1 := 1049 * 10 ^ 82) (E1 := 2097 * 10 ^ 82) (A2 := 1404 * 10 ^ 85) (E2 := 2808 * 10 ^ 85)
    (A3 := 3397 * 10 ^ 88) (E3 := 6794 * 10 ^ 88) (A4 := 4546 * 10 ^ 91) (E4 := 9092 * 10 ^ 91)
    (Ā' := 4547 * 10 ^ 91) (Ē' := 9093 * 10 ^ 91)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t12 := t11.comp (by norm_num) (W.d3.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 6879 / 10 ^ 8) (g2 := 1375 / 10 ^ 7) (gp := 7749 / 10 ^ 9)
    (Ā := 4547 * 10 ^ 91) (Ē := 9093 * 10 ^ 91)
    (P1 := 1223 * 10 ^ 94) (Q1 := 2445 * 10 ^ 94) (P2 := 1637 * 10 ^ 97) (Q2 := 3274 * 10 ^ 97)
    (A1 := 1101 * 10 ^ 95) (E1 := 2201 * 10 ^ 95) (A2 := 1474 * 10 ^ 98) (E2 := 2948 * 10 ^ 98)
    (A3 := 7133 * 10 ^ 101) (E3 := 1427 * 10 ^ 102) (A4 := 9545 * 10 ^ 104) (E4 := 1909 * 10 ^ 105)
    (Ā' := 9546 * 10 ^ 104) (Ē' := 1910 * 10 ^ 105)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t13 := t12.comp (by norm_num) (W.c0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 9546 * 10 ^ 104) (Ē := 1910 * 10 ^ 105)
    (A1 := 4620 * 10 ^ 108) (E1 := 9244 * 10 ^ 108) (A2 := 6183 * 10 ^ 111) (E2 := 1237 * 10 ^ 112)
    (A3 := 2992 * 10 ^ 115) (E3 := 5987 * 10 ^ 115) (A4 := 4004 * 10 ^ 118) (E4 := 8008 * 10 ^ 118)
    (Ā' := 4005 * 10 ^ 118) (Ē' := 8009 * 10 ^ 118)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t14 := t13.comp (by norm_num) (W.c1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 4005 * 10 ^ 118) (Ē := 8009 * 10 ^ 118)
    (A1 := 1939 * 10 ^ 122) (E1 := 3876 * 10 ^ 122) (A2 := 2595 * 10 ^ 125) (E2 := 5190 * 10 ^ 125)
    (A3 := 1256 * 10 ^ 129) (E3 := 2512 * 10 ^ 129) (A4 := 1681 * 10 ^ 132) (E4 := 3362 * 10 ^ 132)
    (Ā' := 1682 * 10 ^ 132) (Ē' := 3363 * 10 ^ 132)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t15 := t14.comp (by norm_num) (W.c2.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 1682 * 10 ^ 132) (Ē := 3363 * 10 ^ 132)
    (A1 := 8140 * 10 ^ 135) (E1 := 1628 * 10 ^ 136) (A2 := 1090 * 10 ^ 139) (E2 := 2180 * 10 ^ 139)
    (A3 := 5275 * 10 ^ 142) (E3 := 1055 * 10 ^ 143) (A4 := 7059 * 10 ^ 145) (E4 := 1412 * 10 ^ 146)
    (Ā' := 7060 * 10 ^ 145) (Ē' := 1413 * 10 ^ 146)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t16 := t15.comp (by norm_num) (W.c3.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 7060 * 10 ^ 145) (Ē := 1413 * 10 ^ 146)
    (A1 := 3417 * 10 ^ 149) (E1 := 6839 * 10 ^ 149) (A2 := 4573 * 10 ^ 152) (E2 := 9146 * 10 ^ 152)
    (A3 := 2213 * 10 ^ 156) (E3 := 4427 * 10 ^ 156) (A4 := 2962 * 10 ^ 159) (E4 := 5924 * 10 ^ 159)
    (Ā' := 2963 * 10 ^ 159) (Ē' := 5925 * 10 ^ 159)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t17 := t16.comp (by norm_num) (W.c4.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 1375 / 10 ^ 7)
    (Ā := 2963 * 10 ^ 159) (Ē := 5925 * 10 ^ 159)
    (A1 := 1434 * 10 ^ 163) (E1 := 2868 * 10 ^ 163) (A2 := 1919 * 10 ^ 166) (E2 := 3838 * 10 ^ 166)
    (A3 := 9287 * 10 ^ 169) (E3 := 1858 * 10 ^ 170) (A4 := 1243 * 10 ^ 173) (E4 := 2486 * 10 ^ 173)
    (Ā' := 1244 * 10 ^ 173) (Ē' := 2487 * 10 ^ 173)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t18 := t17.comp (by norm_num) (W.d4.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
    (g1 := 1375 / 10 ^ 7) (g2 := 2749 / 10 ^ 7) (gp := 1538 / 10 ^ 8)
    (Ā := 1244 * 10 ^ 173) (Ē := 2487 * 10 ^ 173)
    (P1 := 6688 * 10 ^ 175) (Q1 := 1338 * 10 ^ 176) (P2 := 8950 * 10 ^ 178) (Q2 := 1790 * 10 ^ 179)
    (A1 := 6020 * 10 ^ 176) (E1 := 1204 * 10 ^ 177) (A2 := 8056 * 10 ^ 179) (E2 := 1612 * 10 ^ 180)
    (A3 := 7798 * 10 ^ 183) (E3 := 1561 * 10 ^ 184) (A4 := 1044 * 10 ^ 187) (E4 := 2088 * 10 ^ 187)
    (Ā' := 1045 * 10 ^ 187) (Ē' := 2089 * 10 ^ 187)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (M.gamma_num (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t19 := t18.comp (by norm_num) (W.e0.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 2749 / 10 ^ 7)
    (Ā := 1045 * 10 ^ 187) (Ē := 2089 * 10 ^ 187)
    (A1 := 1012 * 10 ^ 191) (E1 := 2023 * 10 ^ 191) (A2 := 1355 * 10 ^ 194) (E2 := 2710 * 10 ^ 194)
    (A3 := 1312 * 10 ^ 198) (E3 := 2624 * 10 ^ 198) (A4 := 1756 * 10 ^ 201) (E4 := 3512 * 10 ^ 201)
    (Ā' := 1757 * 10 ^ 201) (Ē' := 3513 * 10 ^ 201)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t20 := t19.comp (by norm_num) (W.e1.maps M hP (by norm_num)
    (by norm_num) (by norm_num) (g := 2749 / 10 ^ 7)
    (Ā := 1757 * 10 ^ 201) (Ē := 3513 * 10 ^ 201)
    (A1 := 1701 * 10 ^ 205) (E1 := 3401 * 10 ^ 205) (A2 := 2277 * 10 ^ 208) (E2 := 4554 * 10 ^ 208)
    (A3 := 2205 * 10 ^ 212) (E3 := 4409 * 10 ^ 212) (A4 := 2951 * 10 ^ 215) (E4 := 5902 * 10 ^ 215)
    (Ā' := 2952 * 10 ^ 215) (Ē' := 5903 * 10 ^ 215)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t21 := t20.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      hMu (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 2952 * 10 ^ 215) (Ē := 5903 * 10 ^ 215) (Ā' := 2953 * 10 ^ 215) (Ē' := 5904 * 10 ^ 215) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t22 := t21.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num)
      W.head.hW W.head.hb (M.gamma_num (q := 3064 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 2953 * 10 ^ 215) (Ē := 5904 * 10 ^ 215) (Ā' := 3176 * 10 ^ 218) (Ē' := 6349 * 10 ^ 218) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  exact t22

/-- The deployed ResNet-34 training bridge's certified output window at the committed profile:
    `≤ 3.176·10²²¹`. ⭐ This half is an honest fold — the cap touches only the modulus. -/
theorem r34TrainBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).mag 1 ≤ 3176 * 10 ^ 218 :=
  (r34TrainBridge_maps M hMu hε5 W).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ResNet-34 training bridge's error bound at the committed profile:
    `≤ 6.349·10²²¹`, which is `2.00 ×` the window — the tell that this is
    `FloatBridgesTo.capped`'s triangle inequality and not the interval fold (§9). -/
theorem r34TrainBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100)) :
    (r34TrainBridge M (r34TrainProfile_committed M hMu hε5) W).fresh 1 ≤ 6349 * 10 ^ 218 :=
  (r34TrainBridge_maps M hMu hε5 W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed ResNet-34 TRAINING-mode forward is within `6.349·10²²¹` of the certified
    real training forward, per logit**, on inputs of magnitude `≤ 1`, at the measured parameter
    profile, for `ε ≥ 10⁻⁵`, any device batch mean accurate to `10⁻²` relative and any device
    inverse-stddev accurate to `10⁻²` absolute, and any rounding model at binary32 accuracy.

    ⭐⭐ **The first statement in this repo about the program it actually trains with** — every
    other committed forward number is at inference normalisation. ⛔ **It is the CAP**: read it
    as *"the float and the real forward both land in the certified window"*, never as *"the
    rounding error folds to this"*. `6.349 / 3.176 = 2.00` is the tell, and the fold it replaces
    is `10⁷⁴¹⁹`. -/
theorem r34_train_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (W : R34TrainWeights (21/10) (21/10) ε (21/10) (21/10) (1/100) (1/100))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |r34TrainForwardF M W x j - r34TrainForward W x j| ≤ 6349 * 10 ^ 218 :=
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
    W.head.W W.head.b x j| ≤ 6349 * 10 ^ 218 :=
  r34_train_float_logits_le M hMu hε5 W x hx j

end Proofs
