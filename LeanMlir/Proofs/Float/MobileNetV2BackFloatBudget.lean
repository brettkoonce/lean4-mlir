import LeanMlir.Proofs.Float.FloatBudgetEnvBackMBConv

/-! # A NUMBER for MobileNetV2's whole-net BACKWARD — and it needs NO operating point

`Resnet34BackFloatBudget.lean` put a kernel-checked number on a whole-net input-gradient VJP for
the first time. This is the second, at MobileNetV2's topology, and it is the cleaner statement of
the two: **`mnv2_grad_float_le` holds at the unconditional `ε`-floor**, where ResNet-34's needs
`|istd| ≤ 16` — an assumption about the saved activations that §0.1 calls escape 2 — to come down
from `10²⁸⁸` to something `norm_num` will evaluate.

    certified window ≤ 4.750·10¹⁵³      (`mnv2GradBridge_mag_le`)
    fresh budget     ≤ 1.076·10¹⁵²      (`mnv2GradBridge_fresh_le`)
    budget / window  = 0.023            — ⭐ the interval FOLD, not a cap

at `|W| ≤ 28/10`, BatchNorm `|γ| ≤ 17/10`, `ε ≥ 10⁻⁵`, `u ≤ 2⁻²⁴`, on loss cotangents of
magnitude `≤ 1` (`|p − y| ≤ 1` for softmax cross-entropy), at **training-mode BatchNorm** — the
mode this net's own forward has no statable number for at all (`planning/float_budget_numbers.md`
§0.1). 48 numeric stages, 136 rational inequalities, generated and re-asserted by
`mnv2_back_chain` / `verify_mnv2_back`.

**⭐⭐ Why no operating point, and it is structural.** Two reasons, both measured
(§3.9 finding 2): MobileNetV2 has **20 BatchNorm sites against ResNet-34's 33**, and its
backward's conv fan-ins are the inverted residual's 1×1s (24…256) and the **depthwise's 9**,
where ResNet-34's are `512·9`. So the fold lands 135 orders lower and the `ε`-floor
`|istd| ≤ 1/√ε ≤ 317` is affordable. `MnvBnBack` therefore has **no `hS` field**: it is a
theorem, `MnvBnBack.hS`, discharged from `ε ≥ 10⁻⁵` by `bnIstd_abs_le_of`.

⛔ **Two hypotheses remain and they are the caveat, exactly as on ResNet-34.** `es` and `exh` —
the accuracies of the deployed float inverse-stddev and normalised activation *read off the saved
forward activations* — are SUPPLIED at `10⁻²`. They are quantities this repo's forward fold does
speak about, and at training-mode BatchNorm it says `10⁷⁴¹⁷`, not `10⁻²`. So the number is an
honest fold **given a forward-accuracy hypothesis its own forward cannot discharge**; §3.7's
closing point, that §0.1's wall is a fact about *composing* a backward with the forward that
feeds it, applies here verbatim. Say it that way.

⭐ **What was new to build: nothing but two leaves.** `Maps.depthwiseBack` and
`Maps.depthwiseStride2Back` (`FloatBudgetEnvBackMBConv.lean`) are `Maps.depthwise` at the
spatially-reversed kernel and that composed with `Maps.decimateBack`; the block envelopes are six
`Maps.comp`s of leaves that already existed. ⭐ And relu6's clamp — the forward's headline lever,
worth 97 orders of window there (§3.2) — buys this chain **nothing**: its backward is
`reluMaskBack`, a 0/1 select, exact in float and envelope-preserving. Window and budget are
separate levers on a forward; on a backward the clamp is not a lever at all (§3.9 finding 6). -/

namespace Proofs

open FloatModel
open Classical

-- ════════════════════════════════════════════════════════════════
-- § The numeric profile, and the net's stored parameters and saved state
-- ════════════════════════════════════════════════════════════════

/-- The numeric profile the backward fold runs at. ⭐ Note what is NOT here: an inverse-stddev
    bound. ResNet-34's `R34BackProfile` carries `S` and `hS0` because its fold needs `|istd| ≤ 16`
    to be statable; MobileNetV2's is `10¹⁵³` at the `ε`-floor, so `S = 317` is a consequence of
    `hε5` and not a choice (`MnvBnBack.hS`). ⛔ `es` and `exh` are the caveat, not the machinery
    — see the file header. -/
structure MnvBackProfile (M : FloatModel) (ε wk gl es exh q : ℝ) : Prop where
  /-- Conv, depthwise and dense kernels — the backward has no bias anywhere. -/
  hwk : 0 ≤ wk
  /-- BatchNorm γ. -/
  hgl : 0 ≤ gl
  /-- ⛔ The float inverse-stddev's accuracy. SUPPLIED. -/
  hes : 0 ≤ es
  /-- ⛔ The float normalised activation's accuracy. SUPPLIED. -/
  hexh : 0 ≤ exh
  /-- ⭐ The `ε`-floor, and the ONLY thing bounding the inverse-stddev. -/
  hε5 : 1 / 100000 ≤ ε
  hq : M.u ≤ q

/-- `ε` is positive — the `ε`-floor's immediate consequence. -/
theorem MnvBackProfile.hε {M : FloatModel} {ε wk gl es exh q : ℝ}
    (P : MnvBackProfile M ε wk gl es exh q) : 0 < ε :=
  lt_of_lt_of_le (by norm_num) P.hε5

/-- A conv kernel with its magnitude bound (bias-free: `convFlatBack` is stated at bias `0`). -/
structure MnvKerB (oc ic kH kW : Nat) (wk : ℝ) where
  W : Kernel4 oc ic kH kW
  hW : ∀ o c kh kw, |W o c kh kw| ≤ wk

/-- A depthwise kernel with its magnitude bound. -/
structure MnvDwKerB (c kH kW : Nat) (wk : ℝ) where
  W : DepthwiseKernel c kH kW
  hW : ∀ ch kh kw, |W ch kh kw| ≤ wk

/-- The classifier kernel with its bound. -/
structure MnvHeadB (m n : Nat) (wk : ℝ) where
  W : Mat m n
  hW : ∀ i j, |W i j| ≤ wk

/-- ⭐ **One per-channel BatchNorm BACKWARD site — with NO inverse-stddev field.** γ, the saved
    forward activation `x`, and the deployed float statistics computed from it with their two
    SUPPLIED accuracies. ⛔ Nothing bounds `x` and nothing needs to: `x̂` is under `√(h·w)` by
    standardisation and `istd` under `1/√ε` by the `ε`-floor, whatever `x` was. That second one
    is `MnvBnBack.hS`, a theorem here where `R34BnBack.hS` is a field. -/
structure MnvBnBack (c h w : Nat) (ε gl es exh : ℝ) where
  γ : Vec c
  /-- The SAVED forward activation this BatchNorm normalised. -/
  x : Vec (c * h * w)
  /-- The deployed float inverse-stddev, per channel. -/
  fs : Fin c → ℝ
  /-- The deployed float normalised activation, per channel. -/
  fxh : Fin c → Vec (h * w)
  hγ : ∀ k, |γ k| ≤ gl
  hs : ∀ k, |fs k - bnIstd (h * w) (Mat.unflatten (reassocFwd c h w x) k) ε| ≤ es
  hfxh : ∀ k i, |fxh k i - bnXhat (h * w) ε (Mat.unflatten (reassocFwd c h w x) k) i| ≤ exh

/-- ⭐⭐ **`|istd| ≤ 317` at every channel of this site, from the `ε`-floor ALONE.**
    `1/317² = 1/100489 ≥ 1/100000`, so `bnIstd_abs_le_of` closes it with room. ⛔ This is the
    whole of "MobileNetV2's backward needs no operating-point hypothesis": on ResNet-34 the same
    statement at `S = 317` gives `5.503·10²⁸⁸`, past §3.7(a)'s `norm_num` ceiling, which is why
    that file assumes `|istd| ≤ 16` instead. -/
theorem MnvBnBack.hS {c h w : Nat} {ε gl es exh : ℝ} (s : MnvBnBack c h w ε gl es exh)
    {M : FloatModel} {wk q : ℝ} (P : MnvBackProfile M ε wk gl es exh q) :
    ∀ k, |bnIstd (h * w) (Mat.unflatten (reassocFwd c h w s.x) k) ε| ≤ 317 :=
  fun _ => bnIstd_abs_le_of _ (by norm_num) (by norm_num; linarith [P.hε5])

/-- One stride-1 inverted-residual body's backward data (the `b2`/`b4` skip blocks' bodies):
    the 1×1 expand and project kernels, the 3×3 depthwise, three BatchNorm sites and the two
    relu6 kink masks the smooth-point VJP reads. -/
structure MnvBodyBack (ic mid oc h w : Nat) (ε wk gl es exh : ℝ) where
  ke : MnvKerB mid ic 1 1 wk
  kd : MnvDwKerB mid 3 3 wk
  kp : MnvKerB oc mid 1 1 wk
  bne : MnvBnBack mid h w ε gl es exh
  bnd : MnvBnBack mid h w ε gl es exh
  bnp : MnvBnBack oc h w ε gl es exh
  m_e : Fin (mid * h * w) → Prop
  m_d : Fin (mid * h * w) → Prop

/-- One stride-2 inverted-residual body's backward data (`b1`/`b3`/`b5`/`b6`). ⚠ The expand
    BatchNorm and its relu6 mask live at the DOUBLED grid — the stride change happens inside the
    body, between the depthwise and the expand. -/
structure MnvBodyStridedBack (ic mid oc h w : Nat) (ε wk gl es exh : ℝ) where
  ke : MnvKerB mid ic 1 1 wk
  kd : MnvDwKerB mid 3 3 wk
  kp : MnvKerB oc mid 1 1 wk
  bne : MnvBnBack mid (2 * h) (2 * w) ε gl es exh
  bnd : MnvBnBack mid h w ε gl es exh
  bnp : MnvBnBack oc h w ε gl es exh
  m_e : Fin (mid * (2 * h) * (2 * w)) → Prop
  m_d : Fin (mid * h * w) → Prop

/-- **The whole net's backward data** — the stem, the six inverted-residual bodies, the head and
    the classifier: 20 BatchNorm backward sites (each with its saved activation) against
    ResNet-34's 34, which is half of why this fold is 135 orders smaller. -/
structure MnvBackWeights (ε wk gl es exh : ℝ) where
  stemK : MnvKerB 16 3 3 3 wk
  stemBn : MnvBnBack 16 112 112 ε gl es exh
  mstem : Fin (16 * 112 * 112) → Prop
  b1 : MnvBodyStridedBack 16 64 24 56 56 ε wk gl es exh
  b2 : MnvBodyBack 24 96 24 56 56 ε wk gl es exh
  b3 : MnvBodyStridedBack 24 96 32 28 28 ε wk gl es exh
  b4 : MnvBodyBack 32 128 32 28 28 ε wk gl es exh
  b5 : MnvBodyStridedBack 32 128 64 14 14 ε wk gl es exh
  b6 : MnvBodyStridedBack 64 256 64 7 7 ε wk gl es exh
  headK : MnvKerB 128 64 1 1 wk
  headBn : MnvBnBack 128 7 7 ε gl es exh
  mhead : Fin (128 * 7 * 7) → Prop
  fc : MnvHeadB 128 10 wk

-- ════════════════════════════════════════════════════════════════
-- § The real net, its float peer, and the closed bridge
-- ════════════════════════════════════════════════════════════════

section Net

variable {ε wk gl es exh q : ℝ}

/-- The certified per-channel BatchNorm backward at this site. -/
noncomputable def MnvBnBack.real {c h w : Nat} (s : MnvBnBack c h w ε gl es exh) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun dy => bnPerChannelTensor3_grad_input c h w ε s.γ s.x dy

/-- Its deployed float peer, at the supplied float statistics. -/
noncomputable def MnvBnBack.float {c h w : Nat} (s : MnvBnBack c h w ε gl es exh)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3BackFV M s.γ s.fs s.fxh

/-- The site's bridge, at `S = 317` from the `ε`-floor and `Xh` from standardisation. -/
noncomputable def MnvBnBack.bridge {c h w : Nat} (s : MnvBnBack c h w ε gl es exh)
    (M : FloatModel) (P : MnvBackProfile M ε wk gl es exh q) (hc : 0 < c) (hhw : 0 < h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo s.real (s.float M) :=
  floatBridgesTo_bnPerChannelBack M s.γ s.x s.fs s.fxh hc hhw s.hγ s.hs (s.hS P)
    (fun _k i => bnXhat_abs_le_num (X := Xh) P.hε _ hXh0 hnX i) s.hfxh

/-- One stride-1 inverted-residual body's certified input-gradient VJP. -/
noncomputable def MnvBodyBack.real {ic mid oc h w : Nat}
    (b : MnvBodyBack ic mid oc h w ε wk gl es exh) : Vec (oc * h * w) → Vec (ic * h * w) :=
  invresBodyBackPC b.ke.W b.kd.W b.kp.W b.bne.real b.bnd.real b.bnp.real b.m_e b.m_d

/-- Its deployed float peer. -/
noncomputable def MnvBodyBack.float {ic mid oc h w : Nat}
    (b : MnvBodyBack ic mid oc h w ε wk gl es exh) (M : FloatModel) :
    Vec (oc * h * w) → Vec (ic * h * w) :=
  invresBodyBackPCF M b.ke.W b.kd.W b.kp.W (b.bne.float M) (b.bnd.float M) (b.bnp.float M)
    b.m_e b.m_d

/-- The stride-1 body's bridge, closed at real weights. -/
noncomputable def MnvBodyBack.bridge {ic mid oc h w : Nat}
    (b : MnvBodyBack ic mid oc h w ε wk gl es exh) (M : FloatModel)
    (P : MnvBackProfile M ε wk gl es exh q) (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w)
    (hnM : 0 < mid * h * w) (hnO : 0 < oc * h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_invresBodyBackPC M b.ke.W b.kd.W b.kp.W b.m_e b.m_d P.hwk P.hwk P.hwk
    b.ke.hW b.kd.hW b.kp.hW hnM hnO
    (b.bne.bridge M P hmid hhw hXh0 hnX) (b.bnd.bridge M P hmid hhw hXh0 hnX)
    (b.bnp.bridge M P hoc hhw hXh0 hnX)

/-- One stride-2 inverted-residual body's certified input-gradient VJP. -/
noncomputable def MnvBodyStridedBack.real {ic mid oc h w : Nat}
    (b : MnvBodyStridedBack ic mid oc h w ε wk gl es exh) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  invresBodyStridedBackPC b.ke.W b.kd.W b.kp.W b.bne.real b.bnd.real b.bnp.real b.m_e b.m_d

/-- Its deployed float peer. -/
noncomputable def MnvBodyStridedBack.float {ic mid oc h w : Nat}
    (b : MnvBodyStridedBack ic mid oc h w ε wk gl es exh) (M : FloatModel) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  invresBodyStridedBackPCF M b.ke.W b.kd.W b.kp.W (b.bne.float M) (b.bnd.float M)
    (b.bnp.float M) b.m_e b.m_d

/-- The stride-2 body's bridge, closed at real weights. ⚠ Two `Xh`s: the project and depthwise
    BatchNorms sit at `h·w`, the expand's at `(2h)·(2w)`. -/
noncomputable def MnvBodyStridedBack.bridge {ic mid oc h w : Nat}
    (b : MnvBodyStridedBack ic mid oc h w ε wk gl es exh) (M : FloatModel)
    (P : MnvBackProfile M ε wk gl es exh q) (hmid : 0 < mid) (hoc : 0 < oc) (hhw : 0 < h * w)
    (hhw2 : 0 < (2 * h) * (2 * w)) (hnM2 : 0 < mid * (2 * h) * (2 * w)) (hnO : 0 < oc * h * w)
    {Xh Xhe : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2)
    (hXhe0 : 0 ≤ Xhe) (hnXe : (((2 * h) * (2 * w) : ℕ) : ℝ) ≤ Xhe ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_invresBodyStridedBackPC M b.ke.W b.kd.W b.kp.W b.m_e b.m_d P.hwk P.hwk P.hwk
    b.ke.hW b.kd.hW b.kp.hW hnM2 hnO
    (b.bne.bridge M P hmid hhw2 hXhe0 hnXe) (b.bnd.bridge M P hmid hhw hXh0 hnX)
    (b.bnp.bridge M P hoc hhw hXh0 hnX)

/-- **The committed MobileNetV2 input-gradient VJP at the record's real weights and saved
    state** — `mnv2InputGrad`'s body with every abstract slot pinned to the certified per-op
    backward, and the two skip blocks wrapped in `Proofs.residual` (the inverted residual's
    additive skip routes the cotangent to both branches and adds).
    ⚠ Written with the stem's and head's three stages GROUPED, exactly as `mnv2InputGrad` writes
    them — §3.7's grouping lesson: `.comp` is not associative as a bridge. -/
noncomputable def mnv2GradR (w : MnvBackWeights ε wk gl es exh) : Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride2Back (h := 112) (w := 112) w.stemK.W ∘ w.stemBn.real ∘ reluMaskBack w.mstem)
  ∘ w.b1.real ∘ Proofs.residual w.b2.real ∘ w.b3.real ∘ Proofs.residual w.b4.real
    ∘ w.b5.real ∘ w.b6.real
  ∘ (convFlatBack (h := 7) (w := 7) w.headK.W ∘ w.headBn.real ∘ reluMaskBack w.mhead)
  ∘ gapBack 128 7 7
  ∘ dense (Mat.transpose w.fc.W) (0 : Vec 128)

/-- **The deployed float MobileNetV2 input-gradient** — the same shape with every stage the float
    map its bridge names; the relu6 masks and the decimation scatter are unchanged (a select
    rounds nothing). -/
noncomputable def mnv2GradF (M : FloatModel) (w : MnvBackWeights ε wk gl es exh) :
    Vec 10 → Vec (3 * 224 * 224) :=
  ((M.flatConvF (h := 2 * 112) (w := 2 * 112) (IR.reverseSwap w.stemK.W) (fun _ => 0)
      ∘ decimateBack 16 112 112) ∘ w.stemBn.float M ∘ reluMaskBack w.mstem)
  ∘ w.b1.float M ∘ (fun v j => M.add (w.b2.float M v j) (v j)) ∘ w.b3.float M
    ∘ (fun v j => M.add (w.b4.float M v j) (v j)) ∘ w.b5.float M ∘ w.b6.float M
  ∘ (M.flatConvF (h := 7) (w := 7) (IR.reverseSwap w.headK.W) (fun _ => 0)
      ∘ w.headBn.float M ∘ reluMaskBack w.mhead)
  ∘ gapBackF M 128 7 7
  ∘ M.dense (Mat.transpose w.fc.W) (0 : Vec 128)

set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
/-- ⭐ **The whole MobileNetV2 input-gradient VJP float-bridges TO its float peer, CLOSED** — the
    six inverted-residual bodies and all 20 BatchNorm backwards discharged at the record's real
    data, nothing left but `es`/`exh`. ⚠ Grouped exactly as `mnv2GradR` writes it. -/
noncomputable def mnv2GradBridge (M : FloatModel) (P : MnvBackProfile M ε wk gl es exh q)
    (w : MnvBackWeights ε wk gl es exh) : FloatBridgesTo (mnv2GradR w) (mnv2GradF M w) :=
  (((((((floatBridgesTo_linBack M w.fc.W P.hwk (by norm_num) w.fc.hW).comp
      (floatBridgesTo_gapBack M 128 7 7 (by norm_num) (by norm_num) (by norm_num))).comp
      (((floatBridgesTo_reluMaskBack w.mhead).comp
          (w.headBn.bridge M P (Xh := 7) (by norm_num) (by norm_num) (by norm_num)
            (by norm_num))).comp
        (floatBridgesTo_convBack (h := 7) (w := 7) M w.headK.W P.hwk (by norm_num)
          w.headK.hW))).comp
      (w.b6.bridge M P (Xh := 7) (Xhe := 14) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num))).comp
      (w.b5.bridge M P (Xh := 14) (Xhe := 28) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num))).comp
      (FloatBridgesTo.residual M
        (w.b4.bridge M P (Xh := 28) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num) (by norm_num) (by norm_num) (by norm_num)))).comp
      (w.b3.bridge M P (Xh := 28) (Xhe := 56) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num))).comp
      (FloatBridgesTo.residual M
        (w.b2.bridge M P (Xh := 56) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num) (by norm_num) (by norm_num) (by norm_num))) |>.comp
      (w.b1.bridge M P (Xh := 56) (Xhe := 112) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num)) |>.comp
    (((floatBridgesTo_reluMaskBack w.mstem).comp
        (w.stemBn.bridge M P (Xh := 112) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num))).comp
      (floatBridgesTo_flatConvStride2Back (h := 112) (w := 112) M w.stemK.W P.hwk
        (by norm_num) w.stemK.hW))

end Net

-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, measured per parameter KIND on the 350-epoch ImageNet run
    (`/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin`, 3,504,872 f32): conv, depthwise and
    dense kernels within `28/10` (global max `2.7157` over 3.47 M entries) and BatchNorm γ within
    `17/10` (max `1.6869`). ⚠ Here the maximum IS a kernel, so unlike ResNet-34's the split buys
    the BN gain and not the conv fan-in — 4 orders, not 8 (§3.9 finding 5: measure it, do not
    assume which kind is the outlier). ⭐ β and the dense bias appear nowhere: the backward is
    stated at bias `0` throughout. `ε ≥ 10⁻⁵` is the ONLY thing bounding the inverse-stddev, and
    it is enough (`MnvBnBack.hS`). ⛔ The float inverse-stddev and normalised activation are
    taken accurate to `10⁻²` — SUPPLIED; the file header is about exactly that. -/
theorem mnv2BackProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    MnvBackProfile M ε (28/10) (17/10) (1/100) (1/100) u32 where
  hwk := by norm_num
  hgl := by norm_num
  hes := by norm_num
  hexh := by norm_num
  hε5 := hε5
  hq := hMu

set_option maxRecDepth 4000000 in
set_option maxHeartbeats 8000000 in
/-- ⭐ **The envelope, kernel-checked.** 48 numeric stages, 136 rational inequalities, built
    bottom-up at block granularity (6 body steps rather than 42 leaf steps), with every γ-term
    bounded through `FloatModel.gamma_num` so `norm_num` never evaluates a big power, and every
    BatchNorm site stated at its per-unit gain (`Maps.bnPerChannelBackGain`) so the expensive
    evaluation happens once per feature-map SIZE — five constants, not sixty.

    ⭐ Of the 136, NONE is a cap: `budget / window = 0.023`, not `2.00`. Every one is the interval
    fold — at TRAINING-mode BatchNorm, and with no operating-point hypothesis anywhere. -/
theorem mnv2GradBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : MnvBackWeights ε (28/10) (17/10) (1/100) (1/100)) :
    (mnv2GradBridge M (mnv2BackProfile_committed M hMu hε5) w).Maps 1 0
      (4750 * 10 ^ 150) (1076 * 10 ^ 149) := by
  have P := mnv2BackProfile_committed M hMu hε5
  -- ⭐ the five per-resolution gain constants, each proved once
  have K49r : bnGradInputReMag (7 * 7) (17/10) 1 317 7
      ≤ 2749 * 10 ^ 1 := by norm_num [bnGradInputReMag]
  have K49b : bnGradInputBudgetG u32 (2981 / 10 ^ 9) (7 * 7) (17/10) 1 317 7
      (1/100) (1/100) ≤ 7646 / 10 ^ 2 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K196r : bnGradInputReMag (14 * 14) (17/10) 1 317 14
      ≤ 1068 * 10 ^ 2 := by norm_num [bnGradInputReMag]
  have K196b : bnGradInputBudgetG u32 (1175 / 10 ^ 8) (14 * 14) (17/10) 1 317 14
      (1/100) (1/100) ≤ 1557 / 10 ^ 1 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K784r : bnGradInputReMag (28 * 28) (17/10) 1 317 28
      ≤ 4236 * 10 ^ 2 := by norm_num [bnGradInputReMag]
  have K784b : bnGradInputBudgetG u32 (4680 / 10 ^ 8) (28 * 28) (17/10) 1 317 28
      (1/100) (1/100) ≤ 3352 / 10 ^ 1 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K3136r : bnGradInputReMag (56 * 56) (17/10) 1 317 56
      ≤ 1692 * 10 ^ 3 := by norm_num [bnGradInputReMag]
  have K3136b : bnGradInputBudgetG u32 (1871 / 10 ^ 7) (56 * 56) (17/10) 1 317 56
      (1/100) (1/100) ≤ 9741 / 10 ^ 1 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K12544r : bnGradInputReMag (112 * 112) (17/10) 1 317 112
      ≤ 6762 * 10 ^ 3 := by norm_num [bnGradInputReMag]
  have K12544b : bnGradInputBudgetG u32 (7483 / 10 ^ 7) (112 * 112) (17/10) 1 317 112
      (1/100) (1/100) ≤ 6483 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have m0 := (FloatBridgesTo.Maps.linBack M w.fc.W P.hwk (by norm_num) w.fc.hW
    (M.gamma_num (k := 10 + 2) (q := 7153 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 2801 / 10 ^ 2) (Ē' := 2003 / 10 ^ 8)
    (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
    (FloatBridgesTo.Maps.gapBack M 128 7 7 (by norm_num) (by norm_num) (by norm_num)
      P.hq (Ā' := 5717 / 10 ^ 4) (Ē' := 4429 / 10 ^ 10)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
  have mH := m0.comp (by norm_num)
    (((FloatBridgesTo.Maps.reluMaskBack w.mhead (Ā := 5717 / 10 ^ 4) (Ē := 4429 / 10 ^ 10)).comp
        (by norm_num)
      (FloatBridgesTo.Maps.bnPerChannelBackGain M w.headBn.γ w.headBn.x w.headBn.fs w.headBn.fxh
        (by norm_num) (by norm_num) w.headBn.hγ w.headBn.hs (w.headBn.hS P)
        (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
        w.headBn.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
        (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        P.hgl (by norm_num) (by norm_num) P.hes P.hexh K49r K49b
        (by norm_num) (by norm_num)
        (Ā := 5717 / 10 ^ 4) (Ē := 4429 / 10 ^ 10) (Ā' := 1576 * 10 ^ 1) (Ē' := 4373 / 10 ^ 2)
        (by norm_num) (by norm_num))).comp (by norm_num)
      (FloatBridgesTo.Maps.convBack (h := 7) (w := 7) M w.headK.W P.hwk (by norm_num) w.headK.hW
        (M.gamma_num (k := 128 * 1 * 1 + 2) (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (Ā' := 5649 * 10 ^ 3) (Ē' := 1572 * 10 ^ 1) (by norm_num [u32]) (by norm_num [u32])))
  have m6 := mH.comp (by norm_num)
      (FloatBridgesTo.Maps.invresBodyStridedBackPC (h := 7) (w := 7)
        M w.b6.ke.W w.b6.kd.W w.b6.kp.W w.b6.m_e w.b6.m_d
        P.hwk P.hwk P.hwk w.b6.ke.hW w.b6.kd.hW w.b6.kp.hW
        (by norm_num) (by norm_num) (by norm_num) _ _ _
        (gp := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 1538 / 10 ^ 8)
        (A1 := 1558 * 10 ^ 8) (E1 := 8641 * 10 ^ 5) (A2 := 2792 * 10 ^ 10) (E2 := 1550 * 10 ^ 8)
        (A3 := 7697 * 10 ^ 14) (E3 := 6396 * 10 ^ 12) (A4 := 1940 * 10 ^ 16) (E4 := 1612 * 10 ^ 14)
        (A5 := 2075 * 10 ^ 21) (E5 := 2024 * 10 ^ 19)
        (Ā := 5649 * 10 ^ 3) (Ē := 1572 * 10 ^ 1) (Ā' := 1488 * 10 ^ 24) (Ē' := 1454 * 10 ^ 22)
        (M.gamma_num (k := 64 * 1 * 1 + 2) (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 256 * 1 * 1 + 2) (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b6.bnp.γ w.b6.bnp.x w.b6.bnp.fs w.b6.bnp.fxh
          (by norm_num) (by norm_num) w.b6.bnp.hγ w.b6.bnp.hs (w.b6.bnp.hS P)
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.b6.bnp.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 5649 * 10 ^ 3) (Ē := 1572 * 10 ^ 1) (Ā' := 1558 * 10 ^ 8) (Ē' := 8641 * 10 ^ 5)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b6.bnd.γ w.b6.bnd.x w.b6.bnd.fs w.b6.bnd.fxh
          (by norm_num) (by norm_num) w.b6.bnd.hγ w.b6.bnd.hs (w.b6.bnd.hS P)
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.b6.bnd.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2792 * 10 ^ 10) (Ē := 1550 * 10 ^ 8) (Ā' := 7697 * 10 ^ 14) (Ē' := 6396 * 10 ^ 12)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b6.bne.γ w.b6.bne.x w.b6.bne.fs w.b6.bne.fxh
          (by norm_num) (by norm_num) w.b6.bne.hγ w.b6.bne.hs (w.b6.bne.hS P)
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.b6.bne.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 1940 * 10 ^ 16) (Ē := 1612 * 10 ^ 14) (Ā' := 2075 * 10 ^ 21) (Ē' := 2024 * 10 ^ 19)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]))
  have m5 := m6.comp (by norm_num)
      (FloatBridgesTo.Maps.invresBodyStridedBackPC (h := 14) (w := 14)
        M w.b5.ke.W w.b5.kd.W w.b5.kp.W w.b5.m_e w.b5.m_d
        P.hwk P.hwk P.hwk w.b5.ke.hW w.b5.kd.hW w.b5.kp.hW
        (by norm_num) (by norm_num) (by norm_num) _ _ _
        (gp := 3934 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 7749 / 10 ^ 9)
        (A1 := 1592 * 10 ^ 29) (E1 := 1785 * 10 ^ 27) (A2 := 2853 * 10 ^ 31) (E2 := 3200 * 10 ^ 29)
        (A3 := 3052 * 10 ^ 36) (E3 := 3862 * 10 ^ 34) (A4 := 7692 * 10 ^ 37) (E4 := 9733 * 10 ^ 35)
        (A5 := 3261 * 10 ^ 43) (E5 := 4381 * 10 ^ 41)
        (Ā := 1488 * 10 ^ 24) (Ē := 1454 * 10 ^ 22) (Ā' := 1169 * 10 ^ 46) (Ē' := 1572 * 10 ^ 44)
        (M.gamma_num (k := 64 * 1 * 1 + 2) (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 128 * 1 * 1 + 2) (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b5.bnp.γ w.b5.bnp.x w.b5.bnp.fs w.b5.bnp.fxh
          (by norm_num) (by norm_num) w.b5.bnp.hγ w.b5.bnp.hs (w.b5.bnp.hS P)
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.b5.bnp.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 1488 * 10 ^ 24) (Ē := 1454 * 10 ^ 22) (Ā' := 1592 * 10 ^ 29) (Ē' := 1785 * 10 ^ 27)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b5.bnd.γ w.b5.bnd.x w.b5.bnd.fs w.b5.bnd.fxh
          (by norm_num) (by norm_num) w.b5.bnd.hγ w.b5.bnd.hs (w.b5.bnd.hS P)
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.b5.bnd.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 2853 * 10 ^ 31) (Ē := 3200 * 10 ^ 29) (Ā' := 3052 * 10 ^ 36) (Ē' := 3862 * 10 ^ 34)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b5.bne.γ w.b5.bne.x w.b5.bne.fs w.b5.bne.fxh
          (by norm_num) (by norm_num) w.b5.bne.hγ w.b5.bne.hs (w.b5.bne.hS P)
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b5.bne.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 7692 * 10 ^ 37) (Ē := 9733 * 10 ^ 35) (Ā' := 3261 * 10 ^ 43) (Ē' := 4381 * 10 ^ 41)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]))
  have m4 := m5.comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (by norm_num)
      (FloatBridgesTo.Maps.invresBodyBackPC (h := 28) (w := 28)
        M w.b4.ke.W w.b4.kd.W w.b4.kp.W w.b4.m_e w.b4.m_d
        P.hwk P.hwk P.hwk w.b4.ke.hW w.b4.kd.hW w.b4.kp.hW
        (by norm_num) (by norm_num) _ _ _
        (gp := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 7749 / 10 ^ 9)
        (A1 := 4956 * 10 ^ 51) (E1 := 7051 * 10 ^ 49) (A2 := 4441 * 10 ^ 53) (E2 := 6319 * 10 ^ 51)
        (A3 := 1883 * 10 ^ 59) (E3 := 2826 * 10 ^ 57) (A4 := 4746 * 10 ^ 60) (E4 := 7122 * 10 ^ 58)
        (A5 := 2012 * 10 ^ 66) (E5 := 3176 * 10 ^ 64)
        (Ā := 1169 * 10 ^ 46) (Ē := 1572 * 10 ^ 44) (Ā' := 7212 * 10 ^ 68) (Ē' := 1139 * 10 ^ 67)
        (M.gamma_num (k := 32 * 1 * 1 + 2) (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 128 * 1 * 1 + 2) (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b4.bnp.γ w.b4.bnp.x w.b4.bnp.fs w.b4.bnp.fxh
          (by norm_num) (by norm_num) w.b4.bnp.hγ w.b4.bnp.hs (w.b4.bnp.hS P)
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b4.bnp.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 1169 * 10 ^ 46) (Ē := 1572 * 10 ^ 44) (Ā' := 4956 * 10 ^ 51) (Ē' := 7051 * 10 ^ 49)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b4.bnd.γ w.b4.bnd.x w.b4.bnd.fs w.b4.bnd.fxh
          (by norm_num) (by norm_num) w.b4.bnd.hγ w.b4.bnd.hs (w.b4.bnd.hS P)
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b4.bnd.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 4441 * 10 ^ 53) (Ē := 6319 * 10 ^ 51) (Ā' := 1883 * 10 ^ 59) (Ē' := 2826 * 10 ^ 57)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b4.bne.γ w.b4.bne.x w.b4.bne.fs w.b4.bne.fxh
          (by norm_num) (by norm_num) w.b4.bne.hγ w.b4.bne.hs (w.b4.bne.hS P)
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b4.bne.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 4746 * 10 ^ 60) (Ē := 7122 * 10 ^ 58) (Ā' := 2012 * 10 ^ 66) (Ē' := 3176 * 10 ^ 64)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])) P.hq
        (Ā' := 7213 * 10 ^ 68) (Ē' := 1140 * 10 ^ 67) (by norm_num [u32]) (by norm_num [u32]))
  have m3 := m4.comp (by norm_num)
      (FloatBridgesTo.Maps.invresBodyStridedBackPC (h := 28) (w := 28)
        M w.b3.ke.W w.b3.kd.W w.b3.kp.W w.b3.m_e w.b3.m_d
        P.hwk P.hwk P.hwk w.b3.ke.hW w.b3.kd.hW w.b3.kp.hW
        (by norm_num) (by norm_num) (by norm_num) _ _ _
        (gp := 2027 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 5842 / 10 ^ 9)
        (A1 := 3058 * 10 ^ 74) (E1 := 5071 * 10 ^ 72) (A2 := 2740 * 10 ^ 76) (E2 := 4545 * 10 ^ 74)
        (A3 := 1162 * 10 ^ 82) (E3 := 2018 * 10 ^ 80) (A4 := 2929 * 10 ^ 83) (E4 := 5086 * 10 ^ 81)
        (A5 := 4959 * 10 ^ 89) (E5 := 8891 * 10 ^ 87)
        (Ā := 7213 * 10 ^ 68) (Ē := 1140 * 10 ^ 67) (Ā' := 1333 * 10 ^ 92) (Ē' := 2391 * 10 ^ 90)
        (M.gamma_num (k := 32 * 1 * 1 + 2) (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 96 * 1 * 1 + 2) (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b3.bnp.γ w.b3.bnp.x w.b3.bnp.fs w.b3.bnp.fxh
          (by norm_num) (by norm_num) w.b3.bnp.hγ w.b3.bnp.hs (w.b3.bnp.hS P)
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b3.bnp.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 7213 * 10 ^ 68) (Ē := 1140 * 10 ^ 67) (Ā' := 3058 * 10 ^ 74) (Ē' := 5071 * 10 ^ 72)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b3.bnd.γ w.b3.bnd.x w.b3.bnd.fs w.b3.bnd.fxh
          (by norm_num) (by norm_num) w.b3.bnd.hγ w.b3.bnd.hs (w.b3.bnd.hS P)
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b3.bnd.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 2740 * 10 ^ 76) (Ē := 4545 * 10 ^ 74) (Ā' := 1162 * 10 ^ 82) (Ē' := 2018 * 10 ^ 80)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b3.bne.γ w.b3.bne.x w.b3.bne.fs w.b3.bne.fxh
          (by norm_num) (by norm_num) w.b3.bne.hγ w.b3.bne.hs (w.b3.bne.hS P)
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.b3.bne.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 2929 * 10 ^ 83) (Ē := 5086 * 10 ^ 81) (Ā' := 4959 * 10 ^ 89) (Ē' := 8891 * 10 ^ 87)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]))
  have m2 := m3.comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (by norm_num)
      (FloatBridgesTo.Maps.invresBodyBackPC (h := 56) (w := 56)
        M w.b2.ke.W w.b2.kd.W w.b2.kp.W w.b2.m_e w.b2.m_d
        P.hwk P.hwk P.hwk w.b2.ke.hW w.b2.kd.hW w.b2.kp.hW
        (by norm_num) (by norm_num) _ _ _
        (gp := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 5842 / 10 ^ 9)
        (A1 := 2257 * 10 ^ 98) (E1 := 4176 * 10 ^ 96) (A2 := 1517 * 10 ^ 100) (E2 := 2807 * 10 ^ 98)
        (A3 := 2569 * 10 ^ 106) (E3 := 4898 * 10 ^ 104) (A4 := 6474 * 10 ^ 107) (E4 := 1235 * 10 ^ 106)
        (A5 := 1097 * 10 ^ 114) (E5 := 2153 * 10 ^ 112)
        (Ā := 1333 * 10 ^ 92) (Ē := 2391 * 10 ^ 90) (Ā' := 2949 * 10 ^ 116) (Ē' := 5790 * 10 ^ 114)
        (M.gamma_num (k := 24 * 1 * 1 + 2) (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 96 * 1 * 1 + 2) (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b2.bnp.γ w.b2.bnp.x w.b2.bnp.fs w.b2.bnp.fxh
          (by norm_num) (by norm_num) w.b2.bnp.hγ w.b2.bnp.hs (w.b2.bnp.hS P)
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.b2.bnp.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 1333 * 10 ^ 92) (Ē := 2391 * 10 ^ 90) (Ā' := 2257 * 10 ^ 98) (Ē' := 4176 * 10 ^ 96)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b2.bnd.γ w.b2.bnd.x w.b2.bnd.fs w.b2.bnd.fxh
          (by norm_num) (by norm_num) w.b2.bnd.hγ w.b2.bnd.hs (w.b2.bnd.hS P)
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.b2.bnd.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 1517 * 10 ^ 100) (Ē := 2807 * 10 ^ 98) (Ā' := 2569 * 10 ^ 106) (Ē' := 4898 * 10 ^ 104)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b2.bne.γ w.b2.bne.x w.b2.bne.fs w.b2.bne.fxh
          (by norm_num) (by norm_num) w.b2.bne.hγ w.b2.bne.hs (w.b2.bne.hS P)
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.b2.bne.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 6474 * 10 ^ 107) (Ē := 1235 * 10 ^ 106) (Ā' := 1097 * 10 ^ 114) (Ē' := 2153 * 10 ^ 112)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])) P.hq
        (Ā' := 2950 * 10 ^ 116) (Ē' := 5791 * 10 ^ 114) (by norm_num [u32]) (by norm_num [u32]))
  have m1 := m2.comp (by norm_num)
      (FloatBridgesTo.Maps.invresBodyStridedBackPC (h := 56) (w := 56)
        M w.b1.ke.W w.b1.kd.W w.b1.kp.W w.b1.m_e w.b1.m_d
        P.hwk P.hwk P.hwk w.b1.ke.hW w.b1.kd.hW w.b1.kp.hW
        (by norm_num) (by norm_num) (by norm_num) _ _ _
        (gp := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 3934 / 10 ^ 9)
        (A1 := 4995 * 10 ^ 122) (E1 := 1009 * 10 ^ 121) (A2 := 3357 * 10 ^ 124) (E2 := 6782 * 10 ^ 122)
        (A3 := 5684 * 10 ^ 130) (E3 := 1181 * 10 ^ 129) (A4 := 1433 * 10 ^ 132) (E4 := 2977 * 10 ^ 130)
        (A5 := 9700 * 10 ^ 138) (E5 := 2106 * 10 ^ 137)
        (Ā := 2950 * 10 ^ 116) (Ē := 5791 * 10 ^ 114) (Ā' := 1739 * 10 ^ 141) (Ē' := 3775 * 10 ^ 139)
        (M.gamma_num (k := 24 * 1 * 1 + 2) (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 64 * 1 * 1 + 2) (q := 3934 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b1.bnp.γ w.b1.bnp.x w.b1.bnp.fs w.b1.bnp.fxh
          (by norm_num) (by norm_num) w.b1.bnp.hγ w.b1.bnp.hs (w.b1.bnp.hS P)
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.b1.bnp.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 2950 * 10 ^ 116) (Ē := 5791 * 10 ^ 114) (Ā' := 4995 * 10 ^ 122) (Ē' := 1009 * 10 ^ 121)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b1.bnd.γ w.b1.bnd.x w.b1.bnd.fs w.b1.bnd.fxh
          (by norm_num) (by norm_num) w.b1.bnd.hγ w.b1.bnd.hs (w.b1.bnd.hS P)
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.b1.bnd.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 3357 * 10 ^ 124) (Ē := 6782 * 10 ^ 122) (Ā' := 5684 * 10 ^ 130) (Ē' := 1181 * 10 ^ 129)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b1.bne.γ w.b1.bne.x w.b1.bne.fs w.b1.bne.fxh
          (by norm_num) (by norm_num) w.b1.bne.hγ w.b1.bne.hs (w.b1.bne.hS P)
          (fun c i => bnXhat_abs_le_num (X := 112) P.hε _ (by norm_num) (by norm_num) i)
          w.b1.bne.hfxh (q := u32) (gn := 7483 / 10 ^ 7) P.hq
          (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl (by norm_num) (by norm_num) P.hes P.hexh K12544r K12544b
          (by norm_num) (by norm_num)
          (Ā := 1433 * 10 ^ 132) (Ē := 2977 * 10 ^ 130) (Ā' := 9700 * 10 ^ 138) (Ē' := 2106 * 10 ^ 137)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]))
  have mS := m1.comp (by norm_num)
    (((FloatBridgesTo.Maps.reluMaskBack w.mstem (Ā := 1739 * 10 ^ 141) (Ē := 3775 * 10 ^ 139)).comp
        (by norm_num)
      (FloatBridgesTo.Maps.bnPerChannelBackGain M w.stemBn.γ w.stemBn.x w.stemBn.fs w.stemBn.fxh
        (by norm_num) (by norm_num) w.stemBn.hγ w.stemBn.hs (w.stemBn.hS P)
        (fun c i => bnXhat_abs_le_num (X := 112) P.hε _ (by norm_num) (by norm_num) i)
        w.stemBn.hfxh (q := u32) (gn := 7483 / 10 ^ 7) P.hq
        (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        P.hgl (by norm_num) (by norm_num) P.hes P.hexh K12544r K12544b
        (by norm_num) (by norm_num)
        (Ā := 1739 * 10 ^ 141) (Ē := 3775 * 10 ^ 139) (Ā' := 1178 * 10 ^ 148) (Ē' := 2666 * 10 ^ 146)
        (by norm_num) (by norm_num))).comp (by norm_num)
      (FloatBridgesTo.Maps.flatConvStride2Back (h := 112) (w := 112) M w.stemK.W
        P.hwk (by norm_num) w.stemK.hW
        (M.gamma_num (k := 16 * 3 * 3 + 2) (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (Ā' := 4750 * 10 ^ 150) (Ē' := 1076 * 10 ^ 149) (by norm_num [u32]) (by norm_num [u32])))
  exact mS

/-- The certified output window of MobileNetV2's input-gradient at the committed profile:
    `≤ 4.750·10¹⁵³` per input pixel, on loss cotangents of magnitude `≤ 1`. -/
theorem mnv2GradBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : MnvBackWeights ε (28/10) (17/10) (1/100) (1/100)) :
    (mnv2GradBridge M (mnv2BackProfile_committed M hMu hε5) w).mag 1 ≤ 4750 * 10 ^ 150 :=
  (mnv2GradBridge_maps M hMu hε5 w).mag_le 1 (by norm_num) le_rfl

/-- The fresh budget of MobileNetV2's input-gradient at the committed profile: `≤ 1.076·10¹⁵²`. -/
theorem mnv2GradBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (w : MnvBackWeights ε (28/10) (17/10) (1/100) (1/100)) :
    (mnv2GradBridge M (mnv2BackProfile_committed M hMu hε5) w).fresh 1 ≤ 1076 * 10 ^ 149 :=
  (mnv2GradBridge_maps M hMu hε5 w).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed MobileNetV2 float input-gradient is within `1.076·10¹⁵²` of the certified
    real one, per input pixel**, on loss cotangents of magnitude `≤ 1`, at `|W| ≤ 28/10`,
    `|γ| ≤ 17/10`, `ε ≥ 10⁻⁵`, `u ≤ 2⁻²⁴`. The certified window is `4.750·10¹⁵³`, so
    `budget / window = 0.023` — ⭐ **the interval FOLD**, where ConvNeXt-T's and ViT-Tiny's
    forward numbers are `2.00` caps, and **at TRAINING-mode BatchNorm**, the mode this net's own
    forward has no statable number for at all.

    ⭐⭐ **And unlike ResNet-34's, this one assumes no operating point**: `|istd| ≤ 317` comes
    from `ε ≥ 10⁻⁵` alone (`MnvBnBack.hS`). It is the first whole-net backward number in the repo
    with nothing supplied but the two saved-activation accuracies.
    ⛔ **Read the file header before quoting it**: `es` and `exh` ARE supplied, and they are the
    two quantities the forward's own training-mode fold cannot discharge. -/
theorem mnv2_grad_float_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : MnvBackWeights ε (28/10) (17/10) (1/100) (1/100))
    (dy : Vec 10) (hdy : ∀ k, |dy k| ≤ 1) (j : Fin (3 * 224 * 224)) :
    |mnv2GradF M w dy j - mnv2GradR w dy j| ≤ 1076 * 10 ^ 149 :=
  (mnv2GradBridge_maps M hMu hε5 w).budget_le (by norm_num) le_rfl dy hdy j

end Proofs
