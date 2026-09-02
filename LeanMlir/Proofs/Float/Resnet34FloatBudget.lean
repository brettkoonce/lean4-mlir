import LeanMlir.Proofs.Float.FloatBudgetEnv

/-! # A NUMBER for ResNet-34: the deployed inference forward at the measured profile

The ImageNet-scale peer of `Cifar8FloatBudget.lean`. For the `[3,4,6,3]` ResNet-34 forward at
`224²` — stem 7×7/s2, 3×3/s2 pool, 16 basic blocks, GAP, dense — with **inference** BatchNorm
(frozen running statistics; the forward `@resnet34_fwd_eval` renders and the accuracy runs
evaluate), on the unit input window, at the profile measured on the 79-epoch ImageNet
checkpoint (`|parameter| ≤ 21/10`), for any rounding model at binary32 accuracy:

    output window  ≤ 3.152·10²¹¹      (`r34EvalBridge_mag_le`)
    fresh budget   ≤ 1.548·10²⁰⁹      (`r34EvalBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 1.548·10²⁰⁹` (`r34_float_logits_le`).

⚠ **What the number means.** It is the interval fold — `FloatClose.comp` threading worst-case
windows through 90 numeric stages — at the measured magnitudes, and it is vacuous against
logits of ≈ 3.6 by 209 orders. What is new is that the kernel checks it: a wrong `layerBudget`,
a dropped stage, a misread fan-in, a mis-plugged block or a mis-stated BN modulus now fails to
compile. The budget is `4.9·10⁻³` of the certified window — the same relative scale as
CIFAR-8's `10⁻⁴` and the MNIST-MLP capstone's, as it should be, since it is ≈ Σ(mᵢ+2)·u over
the chain. `scripts/adjoint_chain_probe.py` §5 puts the *adjoint chain's* proven-H budget for
this net at `6.5·10⁵¹` using exact data-dependent `dot_close` coefficients at the measured
operating profile; the 157 orders between them are the two documented gaps — the uniform
`m·w'·A` face of `layerBudget` (§5 measures 257× per stage) and worst-case rather than measured
windows. The chain is ~10¹⁵⁷ tighter and still vacuous; that is the finding, not a defect here.

⭐⭐ **Why inference BN, and not the training-mode net.** Training-mode BatchNorm reduces its
statistics out of its own input, so `bnReluBudget` carries the mean-and-variance shift term
`G·2A·(8A·e/(2ε√ε))` — **quadratic in the window**. Folded through 33 BN sites the budget
squares at every one: the same fold on `resnet34Forward_full_pc` comes out at ~10⁷⁴¹⁷, past
the point where `norm_num` will evaluate the numeral at all. Inference BN has no reduction —
`μ` and `rsqrt(var+ε)` are frozen constants, the map is affine in `x` with slope `γ·s`, and its
modulus is linear (`BnEvalRuntimeFloatBridge.lean`). So the choice of BN is not a detail of
tightness; it decides whether a whole-net number for a 34-layer net exists.

⚠ **Two hypotheses this number rests on, both named.** (i) The deployed inverse-stddev is a
device `rsqrt` with no IEEE specification, so it is *modelled*: `DeviceRsqrt ε es` supplies it
with an accuracy `es`, exactly as EfficientNet's sigmoid carries `esig` and ViT's `exp` carries
`eexp`. (ii) The real side is `r34Forward` with `bnPerChannelEvalTensor3` at every BN site —
the op `SHlo.bnPerChannelEvalF` denotes (`bnPerChannelEvalF_faithful`), hence the ℝ semantics
of every BN line of the committed eval render — but the whole-graph faithfulness theorem for
the eval chain (the twin of `resnet34FwdGraphFullPC_faithful`) does not exist yet, so this ties
to the skeleton at the eval BN, not through a proved render tie. The training-mode net has that
tie (`resnet34Forward_full_pc_eq_skeleton`) and, per the paragraph above, no statable number.

Provenance for the 180 numerals: `scripts/float_budget_envelope.py`, which folds the envelope
in exactly these lemmas' semantics with exact rationals, rounds every stage UP to four
significant figures, re-asserts each rounded inequality, and emits this file's `have` chains.
Its CIFAR-8 regression case reproduces `Cifar8FloatBudget.lean`'s fold stage for stage.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The modelled device `rsqrt`, and the parameter records
-- ════════════════════════════════════════════════════════════════

/-- **The deployed inverse-stddev.** `@resnet34_fwd_eval` emits a `stablehlo.rsqrt`, which has
    no IEEE specification, so it is supplied with an accuracy rather than derived — the same
    status as `fexp`/`fsig` throughout the float tier. `es` is its absolute accuracy on the
    shifted variances the net actually evaluates it at. -/
structure DeviceRsqrt (ε es : ℝ) where
  /-- The device kernel. -/
  rsq : ℝ → ℝ
  /-- Its accuracy at every `var + ε` with `var ≥ 0`. -/
  spec : ∀ t, 0 ≤ t → |rsq (t + ε) - 1 / Real.sqrt (t + ε)| ≤ es

/-- A convolution's stored parameters with their magnitude bounds. -/
structure R34Conv (oc ic kH kW : Nat) (w' β' : ℝ) where
  W : Kernel4 oc ic kH kW
  b : Vec oc
  hW : ∀ o c kh kw, |W o c kh kw| ≤ w'
  hb : ∀ o, |b o| ≤ β'

/-- The classifier's stored parameters with their magnitude bounds. -/
structure R34Head (m n : Nat) (w' β' : ℝ) where
  W : Mat m n
  b : Vec n
  hW : ∀ i j, |W i j| ≤ w'
  hb : ∀ j, |b j| ≤ β'

/-- One inference-BN site: scale, shift, and the two FROZEN running statistics. -/
structure R34Bn (c : Nat) (G Bb Mb : ℝ) where
  γ : Vec c
  β : Vec c
  μ : Vec c
  v : Vec c
  hγ : ∀ o, |γ o| ≤ G
  hβ : ∀ o, |β o| ≤ Bb
  hμ : ∀ o, |μ o| ≤ Mb
  /-- Running variances are nonnegative — what puts the inverse-stddev under the `ε`-floor. -/
  hv : ∀ o, 0 ≤ v o

/-- An identity basic block: two 3×3 convs, two BN sites. 13 of the 16. -/
structure R34IdBlk (c : Nat) (w' β' G Bb Mb : ℝ) where
  cv1 : R34Conv c c 3 3 w' β'
  bn1 : R34Bn c G Bb Mb
  cv2 : R34Conv c c 3 3 w' β'
  bn2 : R34Bn c G Bb Mb

/-- A downsample basic block: two body convs (the first stride-2), the 1×1 stride-2 option-B
    projection, and three BN sites. -/
structure R34DownBlk (ic oc : Nat) (w' β' G Bb Mb : ℝ) where
  cv1 : R34Conv oc ic 3 3 w' β'
  bn1 : R34Bn oc G Bb Mb
  cv2 : R34Conv oc oc 3 3 w' β'
  bn2 : R34Bn oc G Bb Mb
  cvp : R34Conv oc ic 1 1 w' β'
  bnp : R34Bn oc G Bb Mb

/-- **The whole net's stored parameters at one uniform profile** — 37 convolutions, the
    classifier, and 33 inference-BN sites, every entry within `w'`/`β'`/`G`/`Bb`/`Mb`. -/
structure R34Weights (w' β' G Bb Mb : ℝ) where
  stem : R34Conv 64 3 7 7 w' β'
  bns : R34Bn 64 G Bb Mb
  a0 : R34IdBlk 64 w' β' G Bb Mb
  a1 : R34IdBlk 64 w' β' G Bb Mb
  a2 : R34IdBlk 64 w' β' G Bb Mb
  d2 : R34DownBlk 64 128 w' β' G Bb Mb
  b0 : R34IdBlk 128 w' β' G Bb Mb
  b1 : R34IdBlk 128 w' β' G Bb Mb
  b2 : R34IdBlk 128 w' β' G Bb Mb
  d3 : R34DownBlk 128 256 w' β' G Bb Mb
  c0 : R34IdBlk 256 w' β' G Bb Mb
  c1 : R34IdBlk 256 w' β' G Bb Mb
  c2 : R34IdBlk 256 w' β' G Bb Mb
  c3 : R34IdBlk 256 w' β' G Bb Mb
  c4 : R34IdBlk 256 w' β' G Bb Mb
  d4 : R34DownBlk 256 512 w' β' G Bb Mb
  e0 : R34IdBlk 512 w' β' G Bb Mb
  e1 : R34IdBlk 512 w' β' G Bb Mb
  head : R34Head 512 10 w' β'

/-- The numeric profile the fold runs at: every magnitude bound nonnegative, `ε` positive with
    its inverse-square-root under a rational `S`, and the rounding unit under a rational `q`.
    One record so the block lemmas take one argument instead of ten. -/
structure R34Profile (M : FloatModel) (ε w' β' G Bb Mb es S q : ℝ) : Prop where
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
noncomputable def R34Bn.fwd {c : Nat} (B : R34Bn c G Bb Mb) (ε : ℝ) (h w : Nat) :
    Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelEvalTensor3 c h w ε B.γ B.β B.μ B.v

/-- The deployed float inference BN at this site: the six rounded ops, with the device `rsqrt`
    evaluated at the frozen `var + ε`. -/
noncomputable def R34Bn.fwdF {c : Nat} (B : R34Bn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (h w : Nat) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelEvalTensor3FV M B.γ B.β B.μ (fun o => R.rsq (B.v o + ε))

/-- This BN site's bridge. -/
noncomputable def R34Bn.bridge {c h w : Nat} (B : R34Bn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : R34Profile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) :
    FloatBridgesTo (B.fwd ε h w) (B.fwdF M R h w) :=
  floatBridgesTo_bnPerChannelEvalTensor3 (h := h) (w := w) M B.γ B.β B.μ B.v
    (fun o => R.rsq (B.v o + ε)) hc hhw P.hε B.hv B.hγ B.hβ B.hμ
    (fun o => R.spec (B.v o) (B.hv o)) P.hSε

/-- This BN site's numeric envelope — two linear inequalities. -/
theorem R34Bn.maps {c h w : Nat} (B : R34Bn c G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : R34Profile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) {Ā Ē Ā' Ē' : ℝ} (hĀ0 : 0 ≤ Ā)
    (hĀ' : G * ((Ā + Mb) * S) + Bb + bnNormBudget q (Ā + Mb) S G Bb 0 es ≤ Ā')
    (hĒ' : bnNormBudget q (Ā + Mb) S G Bb 0 es + G * S * Ē ≤ Ē') :
    (B.bridge M R P hc hhw (h := h) (w := w)).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnEvalPC (h := h) (w := w) M B.γ B.β B.μ B.v
    (fun o => R.rsq (B.v o + ε)) hc hhw P.hε B.hv B.hγ B.hβ B.hμ
    (fun o => R.spec (B.v o) (B.hv o)) P.hSε P.hq P.hG P.hBb P.hS0 P.hMb P.hes hĀ0 hĀ' hĒ'

-- ════════════════════════════════════════════════════════════════
-- § One block: forward, float peer, bridge, envelope
-- ════════════════════════════════════════════════════════════════

/-- The certified ℝ identity block at inference. -/
noncomputable def R34IdBlk.fwd {c : Nat} (B : R34IdBlk c w' β' G Bb Mb) (ε : ℝ) (h w : Nat) :
    Vec (c * h * w) → Vec (c * h * w) :=
  rblkGen (h := h) (w := w) B.cv1.W B.cv1.b (B.bn1.fwd ε h w)
    B.cv2.W B.cv2.b (B.bn2.fwd ε h w)

/-- The deployed float identity block. -/
noncomputable def R34IdBlk.fwdF {c : Nat} (B : R34IdBlk c w' β' G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (h w : Nat) : Vec (c * h * w) → Vec (c * h * w) :=
  rblkGenF M B.cv1.W B.cv1.b (B.bn1.fwdF M R h w) B.cv2.W B.cv2.b (B.bn2.fwdF M R h w)

/-- This block's bridge. -/
noncomputable def R34IdBlk.bridge {c h w : Nat} (B : R34IdBlk c w' β' G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : R34Profile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w) :
    FloatBridgesTo (B.fwd ε h w) (B.fwdF M R h w) :=
  floatBridgesTo_r34IdBlock (h := h) (w := w) M B.cv1.W B.cv1.b B.cv2.W B.cv2.b
    (B.bn1.fwd ε h w) (B.bn1.fwdF M R h w) (B.bn2.fwd ε h w) (B.bn2.fwdF M R h w)
    P.hw' P.hβ' hn B.cv1.hW B.cv1.hb B.cv2.hW B.cv2.hb
    (B.bn1.bridge M R P hc hhw) (B.bn2.bridge M R P hc hhw)

/-- **This block's numeric envelope** — four numeric stages (conv, BN, conv, BN; the two ReLUs
    carry the envelope through unchanged) then the residual fan-in. Ten inequalities. -/
theorem R34IdBlk.maps {c h w : Nat} (B : R34IdBlk c w' β' G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : R34Profile M ε w' β' G Bb Mb es S q)
    (hc : 0 < c) (hhw : 0 < h * w) (hn : 0 < c * h * w)
    {g Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ}
    (hg : (1 + M.u) ^ (c * 3 * 3 + 2) - 1 ≤ g) (hA10 : 0 ≤ A1) (hA30 : 0 ≤ A3)
    (c1A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (c1E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (n1A : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (n1E : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (c2A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * A2 + β') ≤ A3)
    (c2E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (A2 + E2) + β')
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E3)
    (n2A : G * ((A3 + Mb) * S) + Bb + bnNormBudget q (A3 + Mb) S G Bb 0 es ≤ A4)
    (n2E : bnNormBudget q (A3 + Mb) S G Bb 0 es + G * S * E3 ≤ E4)
    (rA : A4 + Ā + q * (A4 + Ā) ≤ Ā') (rE : q * (A4 + E4 + Ā + Ē) + (E4 + Ē) ≤ Ē') :
    (B.bridge M R P hc hhw hn).Maps Ā Ē Ā' Ē' := by
  have s1 := FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.cv1.W B.cv1.b P.hw' P.hβ' hn
    B.cv1.hW B.cv1.hb hg c1A c1E
  have s2 := s1.comp hn (B.bn1.maps M R P hc hhw hA10 n1A n1E)
  have s3 := s2.comp hn FloatBridgesTo.Maps.relu
  have s4 := s3.comp hn (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.cv2.W B.cv2.b
    P.hw' P.hβ' hn B.cv2.hW B.cv2.hb hg c2A c2E)
  have s5 := s4.comp hn (B.bn2.maps M R P hc hhw hA30 n2A n2E)
  exact (FloatBridgesTo.Maps.residual M hn s5 P.hq rA rE).comp hn FloatBridgesTo.Maps.relu

/-- The certified ℝ downsample block at inference. -/
noncomputable def R34DownBlk.fwd {ic oc : Nat} (B : R34DownBlk ic oc w' β' G Bb Mb) (ε : ℝ)
    (h w : Nat) : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  rblkStridedGen (h := h) (w := w) B.cv1.W B.cv1.b (B.bn1.fwd ε h w)
    B.cv2.W B.cv2.b (B.bn2.fwd ε h w) B.cvp.W B.cvp.b (B.bnp.fwd ε h w)

/-- The deployed float downsample block. -/
noncomputable def R34DownBlk.fwdF {ic oc : Nat} (B : R34DownBlk ic oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (h w : Nat) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  rblkStridedGenF M B.cv1.W B.cv1.b (B.bn1.fwdF M R h w) B.cv2.W B.cv2.b (B.bn2.fwdF M R h w)
    B.cvp.W B.cvp.b (B.bnp.fwdF M R h w)

/-- This block's bridge. -/
noncomputable def R34DownBlk.bridge {ic oc h w : Nat} (B : R34DownBlk ic oc w' β' G Bb Mb)
    (M : FloatModel) (R : DeviceRsqrt ε es) (P : R34Profile M ε w' β' G Bb Mb es S q)
    (hoc : 0 < oc) (hhw : 0 < h * w) (hn : 0 < oc * h * w)
    (hni : 0 < ic * (2 * h) * (2 * w)) :
    FloatBridgesTo (B.fwd ε h w) (B.fwdF M R h w) :=
  floatBridgesTo_r34DownBlock (h := h) (w := w) M B.cv1.W B.cv1.b B.cv2.W B.cv2.b
    B.cvp.W B.cvp.b (B.bn1.fwd ε h w) (B.bn1.fwdF M R h w) (B.bn2.fwd ε h w)
    (B.bn2.fwdF M R h w) (B.bnp.fwd ε h w) (B.bnp.fwdF M R h w)
    P.hw' P.hβ' hn hni B.cv1.hW B.cv1.hb B.cv2.hW B.cv2.hb B.cvp.hW B.cvp.hb
    (B.bn1.bridge M R P hoc hhw) (B.bn2.bridge M R P hoc hhw) (B.bnp.bridge M R P hoc hhw)

/-- **This block's numeric envelope** — the projection branch (conv, BN), the body branch
    (conv, BN, conv, BN), then the two-branch rounded fan-in. Fourteen inequalities. -/
theorem R34DownBlk.maps {ic oc h w : Nat} (B : R34DownBlk ic oc w' β' G Bb Mb) (M : FloatModel)
    (R : DeviceRsqrt ε es) (P : R34Profile M ε w' β' G Bb Mb es S q)
    (hoc : 0 < oc) (hhw : 0 < h * w) (hn : 0 < oc * h * w)
    (hni : 0 < ic * (2 * h) * (2 * w))
    {g1 g2 gp Ā Ē P1 Q1 P2 Q2 A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ}
    (hg1 : (1 + M.u) ^ (ic * 3 * 3 + 2) - 1 ≤ g1)
    (hg2 : (1 + M.u) ^ (oc * 3 * 3 + 2) - 1 ≤ g2)
    (hgp : (1 + M.u) ^ (ic * 1 * 1 + 2) - 1 ≤ gp)
    (hP10 : 0 ≤ P1) (hA10 : 0 ≤ A1) (hA30 : 0 ≤ A3)
    (pA : (1 + gp) * (((ic * 1 * 1 : ℕ) : ℝ) * w' * Ā + β') ≤ P1)
    (pE : gp * (((ic * 1 * 1 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 1 * 1 : ℕ) : ℝ) * w' * Ē ≤ Q1)
    (pnA : G * ((P1 + Mb) * S) + Bb + bnNormBudget q (P1 + Mb) S G Bb 0 es ≤ P2)
    (pnE : bnNormBudget q (P1 + Mb) S G Bb 0 es + G * S * Q1 ≤ Q2)
    (c1A : (1 + g1) * (((ic * 3 * 3 : ℕ) : ℝ) * w' * Ā + β') ≤ A1)
    (c1E : g1 * (((ic * 3 * 3 : ℕ) : ℝ) * w' * (Ā + Ē) + β')
            + ((ic * 3 * 3 : ℕ) : ℝ) * w' * Ē ≤ E1)
    (n1A : G * ((A1 + Mb) * S) + Bb + bnNormBudget q (A1 + Mb) S G Bb 0 es ≤ A2)
    (n1E : bnNormBudget q (A1 + Mb) S G Bb 0 es + G * S * E1 ≤ E2)
    (c2A : (1 + g2) * (((oc * 3 * 3 : ℕ) : ℝ) * w' * A2 + β') ≤ A3)
    (c2E : g2 * (((oc * 3 * 3 : ℕ) : ℝ) * w' * (A2 + E2) + β')
            + ((oc * 3 * 3 : ℕ) : ℝ) * w' * E2 ≤ E3)
    (n2A : G * ((A3 + Mb) * S) + Bb + bnNormBudget q (A3 + Mb) S G Bb 0 es ≤ A4)
    (n2E : bnNormBudget q (A3 + Mb) S G Bb 0 es + G * S * E3 ≤ E4)
    (rA : P2 + A4 + q * (P2 + A4) ≤ Ā') (rE : q * (P2 + Q2 + A4 + E4) + (Q2 + E4) ≤ Ē') :
    (B.bridge M R P hoc hhw hn hni).Maps Ā Ē Ā' Ē' := by
  have p1 := FloatBridgesTo.Maps.flatConvStride2 (h := h) (w := w) M B.cvp.W B.cvp.b
    P.hw' P.hβ' hni B.cvp.hW B.cvp.hb hgp pA pE
  have p2 := p1.comp hn (B.bnp.maps M R P hoc hhw hP10 pnA pnE)
  have s1 := FloatBridgesTo.Maps.flatConvStride2 (h := h) (w := w) M B.cv1.W B.cv1.b
    P.hw' P.hβ' hni B.cv1.hW B.cv1.hb hg1 c1A c1E
  have s2 := s1.comp hn (B.bn1.maps M R P hoc hhw hA10 n1A n1E)
  have s3 := s2.comp hn FloatBridgesTo.Maps.relu
  have s4 := s3.comp hn (FloatBridgesTo.Maps.flatConv (h := h) (w := w) M B.cv2.W B.cv2.b
    P.hw' P.hβ' hn B.cv2.hW B.cv2.hb hg2 c2A c2E)
  have s5 := s4.comp hn (B.bn2.maps M R P hoc hhw hA30 n2A n2E)
  exact (FloatBridgesTo.Maps.biPathSum M hn p2 s5 P.hq rA rE).comp hn
    FloatBridgesTo.Maps.relu


-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The deployed ResNet-34 inference forward** — the committed `r34Forward` skeleton with
    inference BatchNorm at every one of its 33 sites. -/
noncomputable def r34EvalForward (W : R34Weights w' β' G Bb Mb) (ε : ℝ) :
    Vec (3 * 224 * 224) → Vec 10 :=
  r34Forward W.stem.W W.stem.b W.head.W W.head.b
    (W.bns.fwd ε 112 112)
    (W.a0.fwd (h := 56) (w := 56) ε)
    (W.a1.fwd (h := 56) (w := 56) ε)
    (W.a2.fwd (h := 56) (w := 56) ε)
    (W.d2.fwd (h := 28) (w := 28) ε)
    (W.b0.fwd (h := 28) (w := 28) ε)
    (W.b1.fwd (h := 28) (w := 28) ε)
    (W.b2.fwd (h := 28) (w := 28) ε)
    (W.d3.fwd (h := 14) (w := 14) ε)
    (W.c0.fwd (h := 14) (w := 14) ε)
    (W.c1.fwd (h := 14) (w := 14) ε)
    (W.c2.fwd (h := 14) (w := 14) ε)
    (W.c3.fwd (h := 14) (w := 14) ε)
    (W.c4.fwd (h := 14) (w := 14) ε)
    (W.d4.fwd (h := 7) (w := 7) ε)
    (W.e0.fwd (h := 7) (w := 7) ε)
    (W.e1.fwd (h := 7) (w := 7) ε)

/-- **The deployed ResNet-34 float inference forward** — every concrete slot replaced by the
    model's rounded peer, every BN by the six rounded ops the emitter writes. -/
noncomputable def r34EvalForwardF (M : FloatModel) (R : DeviceRsqrt ε es)
    (W : R34Weights w' β' G Bb Mb) : Vec (3 * 224 * 224) → Vec 10 :=
  r34ForwardF M W.stem.W W.stem.b W.head.W W.head.b
    (W.bns.fwdF M R 112 112)
    (W.a0.fwdF (h := 56) (w := 56) M R)
    (W.a1.fwdF (h := 56) (w := 56) M R)
    (W.a2.fwdF (h := 56) (w := 56) M R)
    (W.d2.fwdF (h := 28) (w := 28) M R)
    (W.b0.fwdF (h := 28) (w := 28) M R)
    (W.b1.fwdF (h := 28) (w := 28) M R)
    (W.b2.fwdF (h := 28) (w := 28) M R)
    (W.d3.fwdF (h := 14) (w := 14) M R)
    (W.c0.fwdF (h := 14) (w := 14) M R)
    (W.c1.fwdF (h := 14) (w := 14) M R)
    (W.c2.fwdF (h := 14) (w := 14) M R)
    (W.c3.fwdF (h := 14) (w := 14) M R)
    (W.c4.fwdF (h := 14) (w := 14) M R)
    (W.d4.fwdF (h := 7) (w := 7) M R)
    (W.e0.fwdF (h := 7) (w := 7) M R)
    (W.e1.fwdF (h := 7) (w := 7) M R)

set_option maxRecDepth 100000 in
/-- ⭐ **The whole deployed ResNet-34 inference forward float-bridges TO its float peer** — a
    CLOSED `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left: the stem conv/BN/ReLU, the
    3×3/s2 pool, all 16 blocks, GAP and the classifier are each discharged by a leaf. Its `.mod`
    is therefore a closed term over 90 per-op budgets, and `r34EvalBridge_maps` bounds it. -/
noncomputable def r34EvalBridge (M : FloatModel) (R : DeviceRsqrt ε es)
    (P : R34Profile M ε w' β' G Bb Mb es S q) (W : R34Weights w' β' G Bb Mb) :
    FloatBridgesTo (r34EvalForward W ε) (r34EvalForwardF M R W) :=
  ((((((((((((((((((((
    (floatBridgesTo_flatConvStride2 (h := 112) (w := 112) M W.stem.W W.stem.b P.hw' P.hβ'
      (by norm_num) W.stem.hW W.stem.hb)
    |>.comp (W.bns.bridge M R P (by norm_num) (by norm_num) (h := 112) (w := 112)))
    |>.comp (floatBridgesTo_relu (n := 64 * 112 * 112)))
    |>.comp (floatBridgesTo_maxPool3s2 (c := 64) (h := 56) (w := 56)))
    |>.comp (W.a0.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.a1.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.a2.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 56) (w := 56)))
    |>.comp (W.d2.bridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b0.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b1.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.b2.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 28) (w := 28)))
    |>.comp (W.d3.bridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.c0.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.c1.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.c2.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.c3.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.c4.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 14) (w := 14)))
    |>.comp (W.d4.bridge M R P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (W.e0.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (W.e1.bridge M R P (by norm_num) (by norm_num) (by norm_num) (h := 7) (w := 7)))
    |>.comp (floatBridgesTo_gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num)))
    |>.comp (floatBridgesTo_dense M W.head.W W.head.b P.hw' P.hβ' (by norm_num)
      W.head.hW W.head.hb)


-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile.** Every stored parameter within `21/10` — the global maximum over
    the 79-epoch ImageNet checkpoint is `2.0741`, and its conv weights sit two orders below that
    (99.99th percentile `0.43`), so the uniform bound is loose and the fold is not sensitive to
    it. `ε ≥ 10⁻⁵` (the trainer's value) puts the inference inverse-stddev under `317`, and the
    device `rsqrt` is taken accurate to `10⁻²` absolute. -/
theorem r34Profile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    R34Profile M ε (21/10) (21/10) (21/10) (21/10) (21/10) (1/100) 317 u32 where
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
/-- ⭐ **The envelope, kernel-checked.** Ninety numeric stages, each closed by two rational
    inequalities with its γ-term bounded through `FloatModel.gamma_num` so `norm_num` never
    evaluates a big power. The chain is built bottom-up so every `have` elaborates against a
    small, fully determined type, at block granularity (16 block steps rather than ~90 leaf
    steps); the closing `exact` is one structural comparison with `r34EvalBridge`'s definition. -/
theorem r34EvalBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceRsqrt ε (1/100)) (W : R34Weights (21/10) (21/10) (21/10) (21/10) (21/10)) :
    (r34EvalBridge M R (r34Profile_committed M hMu hε5) W).Maps 1 0
      (3152 * 10 ^ 208) (1548 * 10 ^ 206) := by
  have hP := r34Profile_committed M hMu hε5
  have t1 := FloatBridgesTo.Maps.flatConvStride2 (h := 112) (w := 112) M W.stem.W W.stem.b
    hP.hw' hP.hβ' (by norm_num) W.stem.hW W.stem.hb (M.gamma_num (q := 8882 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 3109 / 10 ^ 1) (Ē' := 2761 / 10 ^ 6) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
  have t2 := t1.comp (by norm_num) (W.bns.maps M R hP (by norm_num) (by norm_num)
    (h := 112) (w := 112) (Ā' := 2084 * 10 ^ 2) (Ē' := 8461 / 10 ^ 3) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t3 := t2.comp (by norm_num) (FloatBridgesTo.Maps.relu (n := 64 * 112 * 112))
  have t4 := t3.comp (by norm_num)
    (FloatBridgesTo.Maps.maxPool3s2 (c := 64) (h := 56) (w := 56))
  have t5 := t4.comp (by norm_num) (W.a0.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56) (g := 3446 / 10 ^ 8)
    (A1 := 2521 * 10 ^ 5) (E1 := 1893 * 10 ^ 1)
    (A2 := 1679 * 10 ^ 8) (E2 := 1794 * 10 ^ 4)
    (A3 := 2031 * 10 ^ 11) (E3 := 2870 * 10 ^ 7)
    (A4 := 1353 * 10 ^ 14) (E4 := 2341 * 10 ^ 10)
    (Ā' := 1354 * 10 ^ 14) (Ē' := 2342 * 10 ^ 10)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t6 := t5.comp (by norm_num) (W.a1.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56) (g := 3446 / 10 ^ 8)
    (A1 := 1638 * 10 ^ 17) (E1 := 3398 * 10 ^ 13)
    (A2 := 1091 * 10 ^ 20) (E2 := 2609 * 10 ^ 16)
    (A3 := 1320 * 10 ^ 23) (E3 := 3611 * 10 ^ 19)
    (A4 := 8788 * 10 ^ 25) (E4 := 2684 * 10 ^ 22)
    (Ā' := 8789 * 10 ^ 25) (Ē' := 2685 * 10 ^ 22)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t7 := t6.comp (by norm_num) (W.a2.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 56) (w := 56) (g := 3446 / 10 ^ 8)
    (A1 := 1064 * 10 ^ 29) (E1 := 3615 * 10 ^ 25)
    (A2 := 7084 * 10 ^ 31) (E2 := 2632 * 10 ^ 28)
    (A3 := 8570 * 10 ^ 34) (E3 := 3480 * 10 ^ 31)
    (A4 := 5706 * 10 ^ 37) (E4 := 2498 * 10 ^ 34)
    (Ā' := 5707 * 10 ^ 37) (Ē' := 2499 * 10 ^ 34)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t8 := t7.comp (by norm_num) (W.d2.maps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28) (g1 := 3446 / 10 ^ 8) (g2 := 6879 / 10 ^ 8) (gp := 3934 / 10 ^ 9)
    (P1 := 7671 * 10 ^ 39) (Q1 := 3389 * 10 ^ 36)
    (P2 := 5107 * 10 ^ 42) (Q2 := 2419 * 10 ^ 39)
    (A1 := 6904 * 10 ^ 40) (E1 := 3261 * 10 ^ 37)
    (A2 := 4597 * 10 ^ 43) (E2 := 2317 * 10 ^ 40)
    (A3 := 1113 * 10 ^ 47) (E3 := 6371 * 10 ^ 43)
    (A4 := 7410 * 10 ^ 49) (E4 := 4477 * 10 ^ 46)
    (Ā' := 7411 * 10 ^ 49) (Ē' := 4478 * 10 ^ 46)
    (M.gamma_num (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t9 := t8.comp (by norm_num) (W.b0.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28) (g := 6879 / 10 ^ 8)
    (A1 := 1793 * 10 ^ 53) (E1 := 1207 * 10 ^ 50)
    (A2 := 1194 * 10 ^ 56) (E2 := 8415 * 10 ^ 52)
    (A3 := 2889 * 10 ^ 59) (E3 := 2235 * 10 ^ 56)
    (A4 := 1924 * 10 ^ 62) (E4 := 1549 * 10 ^ 59)
    (Ā' := 1925 * 10 ^ 62) (Ē' := 1550 * 10 ^ 59)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t10 := t9.comp (by norm_num) (W.b1.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28) (g := 6879 / 10 ^ 8)
    (A1 := 4658 * 10 ^ 65) (E1 := 4071 * 10 ^ 62)
    (A2 := 3101 * 10 ^ 68) (E2 := 2809 * 10 ^ 65)
    (A3 := 7503 * 10 ^ 71) (E3 := 7313 * 10 ^ 68)
    (A4 := 4995 * 10 ^ 74) (E4 := 5028 * 10 ^ 71)
    (Ā' := 4996 * 10 ^ 74) (Ē' := 5029 * 10 ^ 71)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t11 := t10.comp (by norm_num) (W.b2.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 28) (w := 28) (g := 6879 / 10 ^ 8)
    (A1 := 1209 * 10 ^ 78) (E1 := 1300 * 10 ^ 75)
    (A2 := 8049 * 10 ^ 80) (E2 := 8910 * 10 ^ 77)
    (A3 := 1948 * 10 ^ 84) (E3 := 2290 * 10 ^ 81)
    (A4 := 1297 * 10 ^ 87) (E4 := 1566 * 10 ^ 84)
    (Ā' := 1298 * 10 ^ 87) (Ē' := 1567 * 10 ^ 84)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t12 := t11.comp (by norm_num) (W.d3.maps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14) (g1 := 6879 / 10 ^ 8) (g2 := 1375 / 10 ^ 7) (gp := 7749 / 10 ^ 9)
    (P1 := 3490 * 10 ^ 89) (Q1 := 4240 * 10 ^ 86)
    (P2 := 2324 * 10 ^ 92) (Q2 := 2897 * 10 ^ 89)
    (A1 := 3141 * 10 ^ 90) (E1 := 4008 * 10 ^ 87)
    (A2 := 2092 * 10 ^ 93) (E2 := 2735 * 10 ^ 90)
    (A3 := 1013 * 10 ^ 97) (E3 := 1463 * 10 ^ 94)
    (A4 := 6744 * 10 ^ 99) (E4 := 9954 * 10 ^ 96)
    (Ā' := 6745 * 10 ^ 99) (Ē' := 9955 * 10 ^ 96)
    (M.gamma_num (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t13 := t12.comp (by norm_num) (W.c0.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14) (g := 1375 / 10 ^ 7)
    (A1 := 3264 * 10 ^ 103) (E1 := 5267 * 10 ^ 100)
    (A2 := 2173 * 10 ^ 106) (E2 := 3576 * 10 ^ 103)
    (A3 := 1052 * 10 ^ 110) (E3 := 1876 * 10 ^ 107)
    (A4 := 7004 * 10 ^ 112) (E4 := 1272 * 10 ^ 110)
    (Ā' := 7005 * 10 ^ 112) (Ē' := 1273 * 10 ^ 110)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t14 := t13.comp (by norm_num) (W.c1.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14) (g := 1375 / 10 ^ 7)
    (A1 := 3390 * 10 ^ 116) (E1 := 6627 * 10 ^ 113)
    (A2 := 2257 * 10 ^ 119) (E2 := 4484 * 10 ^ 116)
    (A3 := 1093 * 10 ^ 123) (E3 := 2320 * 10 ^ 120)
    (A4 := 7277 * 10 ^ 125) (E4 := 1568 * 10 ^ 123)
    (Ā' := 7278 * 10 ^ 125) (Ē' := 1569 * 10 ^ 123)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t15 := t14.comp (by norm_num) (W.c2.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14) (g := 1375 / 10 ^ 7)
    (A1 := 3522 * 10 ^ 129) (E1 := 8077 * 10 ^ 126)
    (A2 := 2345 * 10 ^ 132) (E2 := 5452 * 10 ^ 129)
    (A3 := 1135 * 10 ^ 136) (E3 := 2795 * 10 ^ 133)
    (A4 := 7556 * 10 ^ 138) (E4 := 1885 * 10 ^ 136)
    (Ā' := 7557 * 10 ^ 138) (Ē' := 1886 * 10 ^ 136)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t16 := t15.comp (by norm_num) (W.c3.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14) (g := 1375 / 10 ^ 7)
    (A1 := 3657 * 10 ^ 142) (E1 := 9630 * 10 ^ 139)
    (A2 := 2435 * 10 ^ 145) (E2 := 6489 * 10 ^ 142)
    (A3 := 1179 * 10 ^ 149) (E3 := 3303 * 10 ^ 146)
    (A4 := 7849 * 10 ^ 151) (E4 := 2224 * 10 ^ 149)
    (Ā' := 7850 * 10 ^ 151) (Ē' := 2225 * 10 ^ 149)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t17 := t16.comp (by norm_num) (W.c4.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 14) (w := 14) (g := 1375 / 10 ^ 7)
    (A1 := 3799 * 10 ^ 155) (E1 := 1129 * 10 ^ 153)
    (A2 := 2530 * 10 ^ 158) (E2 := 7597 * 10 ^ 155)
    (A3 := 1225 * 10 ^ 162) (E3 := 3845 * 10 ^ 159)
    (A4 := 8156 * 10 ^ 164) (E4 := 2586 * 10 ^ 162)
    (Ā' := 8157 * 10 ^ 164) (Ē' := 2587 * 10 ^ 162)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t18 := t17.comp (by norm_num) (W.d4.maps M R hP (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7) (g1 := 1375 / 10 ^ 7) (g2 := 2749 / 10 ^ 7) (gp := 1538 / 10 ^ 8)
    (P1 := 4386 * 10 ^ 167) (Q1 := 1398 * 10 ^ 165)
    (P2 := 2920 * 10 ^ 170) (Q2 := 9400 * 10 ^ 167)
    (A1 := 3948 * 10 ^ 168) (E1 := 1307 * 10 ^ 166)
    (A2 := 2629 * 10 ^ 171) (E2 := 8785 * 10 ^ 168)
    (A3 := 2545 * 10 ^ 175) (E3 := 9203 * 10 ^ 172)
    (A4 := 1695 * 10 ^ 178) (E4 := 6181 * 10 ^ 175)
    (Ā' := 1696 * 10 ^ 178) (Ē' := 6182 * 10 ^ 175)
    (M.gamma_num (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (M.gamma_num (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t19 := t18.comp (by norm_num) (W.e0.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7) (g := 2749 / 10 ^ 7)
    (A1 := 1642 * 10 ^ 182) (E1 := 6436 * 10 ^ 179)
    (A2 := 1094 * 10 ^ 185) (E2 := 4320 * 10 ^ 182)
    (A3 := 1059 * 10 ^ 189) (E3 := 4473 * 10 ^ 186)
    (A4 := 7050 * 10 ^ 191) (E4 := 3001 * 10 ^ 189)
    (Ā' := 7051 * 10 ^ 191) (Ē' := 3002 * 10 ^ 189)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t20 := t19.comp (by norm_num) (W.e1.maps M R hP (by norm_num) (by norm_num) (by norm_num)
    (h := 7) (w := 7) (g := 2749 / 10 ^ 7)
    (A1 := 6825 * 10 ^ 195) (E1 := 3094 * 10 ^ 193)
    (A2 := 4544 * 10 ^ 198) (E2 := 2075 * 10 ^ 196)
    (A3 := 4399 * 10 ^ 202) (E3 := 2130 * 10 ^ 200)
    (A4 := 2929 * 10 ^ 205) (E4 := 1428 * 10 ^ 203)
    (Ā' := 2930 * 10 ^ 205) (Ē' := 1429 * 10 ^ 203)
    (M.gamma_num (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num) (by norm_num) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t21 := t20.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      hMu (by norm_num [u32]) (by norm_num) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 2931 * 10 ^ 205) (Ē' := 1430 * 10 ^ 203) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have t22 := t21.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M W.head.W W.head.b hP.hw' hP.hβ' (by norm_num)
      W.head.hW W.head.hb (M.gamma_num (q := 3064 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 3152 * 10 ^ 208) (Ē' := 1548 * 10 ^ 206) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  exact t22

/-- The deployed ResNet-34 inference bridge's certified output window at the committed profile:
    `≤ 3.152·10²¹¹` — the worst-case logit magnitude the interval fold can promise. -/
theorem r34EvalBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceRsqrt ε (1/100)) (W : R34Weights (21/10) (21/10) (21/10) (21/10) (21/10)) :
    (r34EvalBridge M R (r34Profile_committed M hMu hε5) W).mag 1 ≤ 3152 * 10 ^ 208 :=
  (r34EvalBridge_maps M hMu hε5 R W).mag_le 1 (by norm_num) le_rfl

/-- The deployed ResNet-34 inference bridge's fresh budget at the committed profile:
    `≤ 1.548·10²⁰⁹`, which is `4.9·10⁻³` of the certified window. -/
theorem r34EvalBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceRsqrt ε (1/100)) (W : R34Weights (21/10) (21/10) (21/10) (21/10) (21/10)) :
    (r34EvalBridge M R (r34Profile_committed M hMu hε5) W).fresh 1 ≤ 1548 * 10 ^ 206 :=
  (r34EvalBridge_maps M hMu hε5 R W).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed ResNet-34 inference forward is within `1.548·10²⁰⁹` of the certified real
    forward, per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile
    (`|·| ≤ 21/10`), for `ε ≥ 10⁻⁵`, any device `rsqrt` accurate to `10⁻²`, and any rounding
    model at binary32 accuracy. The first ImageNet-scale whole-net float budget in the repo
    stated as a number over a `FloatBridgesTo`; see the file header for what the size of that
    number does and does not mean, and why the training-mode net has no such statement. -/
theorem r34_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceRsqrt ε (1/100)) (W : R34Weights (21/10) (21/10) (21/10) (21/10) (21/10))
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |r34EvalForwardF M R W x j - r34EvalForward W ε x j| ≤ 1548 * 10 ^ 206 :=
  (r34EvalBridge_maps M hMu hε5 R W).budget_le (by norm_num) le_rfl x hx j

end Proofs
