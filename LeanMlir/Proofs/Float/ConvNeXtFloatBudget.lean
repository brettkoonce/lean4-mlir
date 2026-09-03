import LeanMlir.Proofs.Float.FloatBudgetEnvLN
import LeanMlir.Proofs.Foundation.WholeNetForwardTies

/-! # A NUMBER for ConvNeXt-T: the committed channel-LayerNorm forward, at the cap

The fourth ImageNet-scale whole-net float statement, and ⛔ **it is not the same kind of
statement as the other three.** For the `[3,3,9,3]` ConvNeXt-T forward at `224²` — 4×4/s4
patchify stem, 18 blocks of `depthwise 7×7 → channel-LN → 1×1 expand → GELU → 1×1 project →
layer scale → skip`, three LN+2×2/s2 downsamples, GAP, head LayerNorm, classifier — on the unit
input window, at the profile measured on the finished 300-epoch ImageNet checkpoint, for any
rounding model at binary32 accuracy:

    output window  ≤ 4.858·10²²⁷      (`cnxBridge_mag_le`)
    fresh budget   ≤ 9.706·10²²⁷      (`cnxBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 9.706·10²²⁷` (`cnx_float_logits_le`).

⛔ **`budget / window = 2.00`, and that ratio is the whole caveat.** Every one of the 23
LayerNorm sites goes through `FloatBridgesTo.capped`, whose modulus is `min(fold, 2·window)`,
and the right branch is what closes. So this number says *the float and the real forward both
land in the certified window* — the triangle inequality — where ResNet-34's `1.548·10²⁰⁹`,
MobileNetV2's `1.444·10⁹⁶` and EfficientNet-B0's `8.408·10²¹⁰` say *the rounding error folds to
this*. Do not table it beside them without saying so (`planning/float_budget_numbers.md` §9).

⭐⭐ **Why the cap is not optional here.** LayerNorm reduces its mean and variance out of its own
input, so perturbing the input moves the statistics and `bnReluBudget` carries a term quadratic
in the window. BatchNorm has an escape — freeze the statistics and the map becomes affine, which
is why the other three numbers are stated at inference BN — and **LayerNorm has none**: there
are no running statistics to freeze, so there is no second render to build and no eval twin to
bound. Uncapped, the same fold is `10¹¹⁶³¹`; `norm_num` refuses numerals past ~`10³⁰⁰`, so there
is no theorem to state. The cap is what makes a kernel-checked whole-net statement about a
LayerNorm net exist at all, and what it buys is weaker than a fold.

⭐ **The profile does not split uniformly, and that is the second thing that had to be right.**
On `/home/skoonce/convnext/convnext_t300_4gpu/convnext_tiny_imagenet.bin` (28,587,592 f32) the
conv and dense kernels max at `0.596` over 28.5 M entries, the biases and LN β at `2.95`, the LN
γ at `4.77`, and the layer scale at `8.38`. A single uniform bound is `8.4` — 14× loose on
exactly the entries the conv fan-in multiplies — and that fold lands at `10³⁰¹`, unstatable.
`CnxBlockChBounded` therefore carries four bounds, not two. ⚠ That checkpoint predates the
2026-08-30 head-LayerNorm restoration (it is short by exactly 1,536 = 2×768, which is how the
missing layer was found), so the head LN's γ/β are not in the measurement; they initialise at
γ=1, β=0 and the bounds cover them with room.

⚠ **Two hypotheses this number rests on, named.** The deployed LayerNorm's mean and
inverse-stddev are a device reduction and a device `rsqrt` with no IEEE specification, so they
are *modelled*: `DeviceLN` supplies them with a relative mean accuracy `emr` and an absolute
inverse-stddev accuracy `ei`, exactly as ResNet-34's `DeviceRsqrt` supplies `es` and
EfficientNet's sigmoid supplies `esig`. The deployed GELU carries `egelu` the same way
(`DeviceGelu`). Everything else is proved. ⚠ All three structures — and the capped LN site's
bridge and envelope — moved to `FloatBudgetEnvLN.lean` when ViT-Tiny needed the same ones.

**The tie.** `cnxForward_eq_committed` is `WholeNetForwardTies.convNextForwardTCh_eq_skeleton`
read backwards: the bridged skeleton IS the committed `convNextForwardTCh`, head LayerNorm
included. ⚠ It was not, until 2026-09-03 — `convnextCh_floatBridgesTo` still had `id` in the
head slot while the tie had carried `rowLNVecFlat 1 768` since 2026-08-30, so the whole-net
bridge described a net with no head LayerNorm and its docstring's claim to tie through that
lemma was false. Fixing it is also what made the cap sufficient on its own: with `id` there,
the last GELU's cubic modulus is never capped again and the fold needs the saturation constant
too (§3.3's ablation table).

Provenance for the 366 numerals: `scripts/float_budget_envelope.py`'s `cnx_eval_chain`, which
folds the envelope in exactly these lemmas' semantics with exact rationals, rounds every stage
UP to four significant figures, and `verify_cnx`, which re-asserts each rounded inequality
before any of them is emitted.
-/

namespace Proofs

open FloatModel
open FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The numeric profile (the device kernels live in `FloatBudgetEnvLN.lean`)
-- ════════════════════════════════════════════════════════════════

/-- The numeric profile the fold runs at. Four magnitude bounds because the measured checkpoint
    has four scales (see the file header), `ε` positive with its inverse square root under a
    rational `S`, and the rounding unit under a rational `q`. -/
structure CnxProfile (M : FloatModel) (ε w' bb gl sl egelu emr ei S q : ℝ) : Prop where
  /-- Conv and dense kernels. -/
  hw' : 0 ≤ w'
  /-- Conv and dense biases, and every LayerNorm β. -/
  hbb : 0 ≤ bb
  /-- LayerNorm γ. -/
  hgl : 0 ≤ gl
  /-- Layer scale. -/
  hsl : 0 ≤ sl
  hegelu : 0 ≤ egelu
  hemr : 0 ≤ emr
  hei : 0 ≤ ei
  hS0 : 0 ≤ S
  hε : 0 < ε
  hSε : 1 / Real.sqrt ε ≤ S
  hq : M.u ≤ q

/-- **The whole net's stored parameters within the four bounds** — the stem, 18 blocks, three
    downsamples, the head LayerNorm and the classifier. The ConvNeXt peer of `R34Weights`,
    stated as a `Prop` over the committed `CnxTWeightsCh` rather than as a second record. -/
structure CnxBounded (wts : CnxTWeightsCh) (w' bb gl sl : ℝ) : Prop where
  sW : ∀ o c kh kw, |wts.sW o c kh kw| ≤ w'
  sb : ∀ o, |wts.sb o| ≤ bb
  sγ : ∀ i, |wts.sγ i| ≤ gl
  sβ : ∀ i, |wts.sβ i| ≤ bb
  s1 : ∀ i, CnxBlockChBounded (wts.s1 i) w' bb gl sl
  d1 : CnxDownChBounded wts.d1 w' bb gl
  s2 : ∀ i, CnxBlockChBounded (wts.s2 i) w' bb gl sl
  d2 : CnxDownChBounded wts.d2 w' bb gl
  s3 : ∀ i, CnxBlockChBounded (wts.s3 i) w' bb gl sl
  d3 : CnxDownChBounded wts.d3 w' bb gl
  s4 : ∀ i, CnxBlockChBounded (wts.s4 i) w' bb gl sl
  hγ : ∀ i, |wts.hγ i| ≤ gl
  hβ : ∀ i, |wts.hβ i| ≤ bb
  Wd : ∀ i j, |wts.Wd i j| ≤ w'
  bd : ∀ j, |wts.bd j| ≤ bb

/-- **Every one of the 23 LayerNorm `ε`s is at or above the floor `ε`.** The inverse-stddev
    bound `S` is `1/√ε`, so a site running at a LARGER `ε` is only tighter — which is why one
    floor serves all 23 sites and no numeral depends on which site it came from. -/
structure CnxEps (wts : CnxTWeightsCh) (ε : ℝ) : Prop where
  hs : ε ≤ wts.sε
  h1 : ∀ i, ε ≤ (wts.s1 i).εn
  hd1 : ε ≤ wts.d1.ε
  h2 : ∀ i, ε ≤ (wts.s2 i).εn
  hd2 : ε ≤ wts.d2.ε
  h3 : ∀ i, ε ≤ (wts.s3 i).εn
  hd3 : ε ≤ wts.d3.ε
  h4 : ∀ i, ε ≤ (wts.s4 i).εn
  hh : ε ≤ wts.hε

-- ════════════════════════════════════════════════════════════════
-- § One LayerNorm site: float map, capped bridge, envelope
-- ════════════════════════════════════════════════════════════════

variable {M : FloatModel} {ε w' bb gl sl egelu emr ei S q : ℝ}

/-- **One LayerNorm site's bridge at this net's profile** — `DeviceLN.bridgeAt`
    (`FloatBudgetEnvLN.lean`) with the two numeric hypotheses read off `CnxProfile`. -/
noncomputable def DeviceLN.bridge (R : DeviceLN emr ei) (M : FloatModel)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (c : Nat) (hc : 0 < c) (e : ℝ)
    (he : ε ≤ e) : FloatBridgesTo (layerNormForward c e 1 0) (R.lnF M c e) :=
  R.bridgeAt M P.hε P.hSε c hc e he

/-- ⛔ **One LayerNorm site's envelope at this net's profile** — `DeviceLN.mapsAt`. `nA` is the
    real window `2Ā·S + bnNormBudget`, which is the fold; `nE` is `2·Ā'`, which is not. -/
theorem DeviceLN.maps (R : DeviceLN emr ei) (M : FloatModel)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (c : Nat) (hc : 0 < c) (e : ℝ)
    (he : ε ≤ e) {Ā Ē Ā' Ē' : ℝ}
    (nA : 1 * (2 * Ā * S) + 0 + bnNormBudget q (2 * Ā) S 1 0 (emr * Ā) ei ≤ Ā')
    (nE : 2 * Ā' ≤ Ē') :
    (R.bridge M P c hc e he).Maps Ā Ē Ā' Ē' :=
  R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq c hc e he nA nE

/-- **A channel-LayerNorm site's envelope**: the capped normalise, then γ, then β. -/
theorem cnxChanLNMaps {c h w : Nat} (M : FloatModel) (R : DeviceLN emr ei)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (γ β : Vec c)
    (hγ : ∀ i, |γ i| ≤ gl) (hβ : ∀ i, |β i| ≤ bb) {e : ℝ} (he : ε ≤ e)
    (hc : 0 < c) (hhw : 0 < h * w)
    {Ā Ē A1 E1 A2 E2 Ā' Ē' : ℝ}
    (nA : 1 * (2 * Ā * S) + 0 + bnNormBudget q (2 * Ā) S 1 0 (emr * Ā) ei ≤ A1)
    (nE : 2 * A1 ≤ E1)
    (gA : gl * A1 + FloatModel.mulErr q gl A1 0 0 ≤ A2)
    (gE : FloatModel.mulErr q gl A1 0 0 + gl * E1 ≤ E2)
    (bA : A2 + bb + q * (A2 + bb) ≤ Ā') (bE : q * (A2 + bb) + E2 ≤ Ē') :
    (floatBridgesTo_chanLNTensor3 (h := h) (w := w) M γ β (R.lnF M c e) hc hγ hβ
      (R.bridge M P c hc e he)).Maps Ā Ē Ā' Ē' :=
  Maps.chanLNTensor3 M γ β (R.lnF M c e) hc hhw hγ hβ (R.bridge M P c hc e he)
    P.hq P.hgl P.hbb (R.maps M P c hc e he nA nE) gA gE bA bE

/-- **The head LayerNorm site's envelope** — the same three stages at one row. -/
theorem cnxRowLNMaps {s c : Nat} (M : FloatModel) (R : DeviceLN emr ei)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (γ β : Vec c)
    (hγ : ∀ i, |γ i| ≤ gl) (hβ : ∀ i, |β i| ≤ bb) {e : ℝ} (he : ε ≤ e) (hc : 0 < c)
    {Ā Ē A1 E1 A2 E2 Ā' Ē' : ℝ}
    (nA : 1 * (2 * Ā * S) + 0 + bnNormBudget q (2 * Ā) S 1 0 (emr * Ā) ei ≤ A1)
    (nE : 2 * A1 ≤ E1)
    (gA : gl * A1 + FloatModel.mulErr q gl A1 0 0 ≤ A2)
    (gE : FloatModel.mulErr q gl A1 0 0 + gl * E1 ≤ E2)
    (bA : A2 + bb + q * (A2 + bb) ≤ Ā') (bE : q * (A2 + bb) + E2 ≤ Ē') :
    (floatBridgesTo_rowLNVecFlat (s := s) M γ β (R.lnF M c e) hc hγ hβ
      (R.bridge M P c hc e he)).Maps Ā Ē Ā' Ē' :=
  Maps.rowLNVecFlat M γ β (R.lnF M c e) hc hγ hβ (R.bridge M P c hc e he)
    P.hq P.hgl P.hbb (R.maps M P c hc e he nA nE) gA gE bA bE

-- ════════════════════════════════════════════════════════════════
-- § The whole net: forward, float peer, bridge
-- ════════════════════════════════════════════════════════════════

/-- **The committed ConvNeXt-T forward, in skeleton form** — `convnextForward` with the real
    channel LayerNorm in the stem slot, the [3,3,9,3] stages and the three downsamples, and the
    real head LayerNorm (`rowLNVecFlat 1 768`) in the head slot. `cnxForward_eq_committed` is
    the tie: this IS `convNextForwardTCh`. -/
noncomputable def cnxForward (wts : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec 10 :=
  convnextForward wts.sW wts.sb wts.Wd wts.bd
    (chanLNTensor3 96 56 56 wts.sε wts.sγ wts.sβ)
    (rowLNVecFlat 1 768 wts.hε wts.hγ wts.hβ)
    (convNextStageChK 3 wts.s1) (cnxDownChW 28 28 wts.d1)
    (convNextStageChK 3 wts.s2) (cnxDownChW 14 14 wts.d2)
    (convNextStageChK 9 wts.s3) (cnxDownChW 7 7 wts.d3)
    (convNextStageChK 3 wts.s4)

/-- **The deployed ConvNeXt-T float forward** — every concrete slot replaced by the model's
    rounded peer, every LayerNorm by the device mean / `rsqrt` normalise chain the emitter
    writes, every GELU by the device kernel. -/
noncomputable def cnxForwardF (M : FloatModel) (R : DeviceLN emr ei) (G : DeviceGelu egelu)
    (wts : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec 10 :=
  convnextForwardF M wts.sW wts.sb wts.Wd wts.bd
    (chanLNTensor3F M wts.sγ wts.sβ (R.lnF M 96 wts.sε))
    (rowLNVecFlatF (s := 1) M wts.hγ wts.hβ (R.lnF M 768 wts.hε))
    (convNextStageChKF M G.g 3 wts.s1 (fun i => R.lnF M 96 (wts.s1 i).εn))
    (cnxDownChWF 28 28 M wts.d1 (R.lnF M 96 wts.d1.ε))
    (convNextStageChKF M G.g 3 wts.s2 (fun i => R.lnF M 192 (wts.s2 i).εn))
    (cnxDownChWF 14 14 M wts.d2 (R.lnF M 192 wts.d2.ε))
    (convNextStageChKF M G.g 9 wts.s3 (fun i => R.lnF M 384 (wts.s3 i).εn))
    (cnxDownChWF 7 7 M wts.d3 (R.lnF M 384 wts.d3.ε))
    (convNextStageChKF M G.g 3 wts.s4 (fun i => R.lnF M 768 (wts.s4 i).εn))

set_option maxRecDepth 100000 in
/-- ⭐ **The whole deployed ConvNeXt-T forward float-bridges TO its float peer** — a CLOSED
    `FloatBridgesTo` with no `FloatBridgesTo` hypotheses left: all 23 LayerNorm slots are
    discharged by `DeviceLN.bridge`, the capped leaf. Its `.mod` is a closed term over the
    per-op budgets, and `cnxBridge_maps` bounds it. -/
noncomputable def cnxBridge (M : FloatModel) (R : DeviceLN emr ei) (G : DeviceGelu egelu)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (wts : CnxTWeightsCh)
    (B : CnxBounded wts w' bb gl sl) (Eps : CnxEps wts ε) :
    FloatBridgesTo (cnxForward wts) (cnxForwardF M R G wts) :=
  convnextCh_floatBridgesTo M G.g wts
    (R.lnF M 96 wts.sε) (fun i => R.lnF M 96 (wts.s1 i).εn) (R.lnF M 96 wts.d1.ε)
    (fun i => R.lnF M 192 (wts.s2 i).εn) (R.lnF M 192 wts.d2.ε)
    (fun i => R.lnF M 384 (wts.s3 i).εn) (R.lnF M 384 wts.d3.ε)
    (fun i => R.lnF M 768 (wts.s4 i).εn) (R.lnF M 768 wts.hε)
    P.hw' P.hbb P.hegelu G.spec B.sW B.sb B.sγ B.sβ B.Wd B.bd B.hγ B.hβ
    B.s1 B.s2 B.s3 B.s4 B.d1 B.d2 B.d3
    (R.bridge M P 96 (by norm_num) wts.sε Eps.hs)
    (R.bridge M P 768 (by norm_num) wts.hε Eps.hh)
    (fun i => R.bridge M P 96 (by norm_num) (wts.s1 i).εn (Eps.h1 i))
    (R.bridge M P 96 (by norm_num) wts.d1.ε Eps.hd1)
    (fun i => R.bridge M P 192 (by norm_num) (wts.s2 i).εn (Eps.h2 i))
    (R.bridge M P 192 (by norm_num) wts.d2.ε Eps.hd2)
    (fun i => R.bridge M P 384 (by norm_num) (wts.s3 i).εn (Eps.h3 i))
    (R.bridge M P 384 (by norm_num) wts.d3.ε Eps.hd3)
    (fun i => R.bridge M P 768 (by norm_num) (wts.s4 i).εn (Eps.h4 i))


-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, measured on the finished 300-epoch ImageNet run
    (`/home/skoonce/convnext/convnext_t300_4gpu/convnext_tiny_imagenet.bin`, 28,587,592 f32):
    conv and dense kernels within `6/10` (global max `0.5962` over 28.5 M entries), biases and
    LayerNorm β within `3` (max `2.9499`), LayerNorm γ within `48/10` (max `4.7700`), the layer
    scale within `84/10` (max `8.3766`). ⭐ The four scales are 14× apart and the fold is
    exquisitely sensitive to the first of them — see the file header. `ε ≥ 10⁻⁵` (the trainer's
    value) puts every LayerNorm's inverse-stddev under `317`; the device mean is taken accurate
    to `10⁻²` relative, the device `rsqrt` and the device GELU to `10⁻²` absolute. -/
theorem cnxProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    CnxProfile M ε (6/10) 3 (48/10) (84/10) (1/100) (1/100) (1/100) 317 u32 where
  hw' := by norm_num
  hbb := by norm_num
  hgl := by norm_num
  hsl := by norm_num
  hegelu := by norm_num
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

set_option maxRecDepth 4000000 in
set_option maxHeartbeats 4000000 in
/-- ⭐ **The envelope, kernel-checked.** 183 numeric stages, 366 rational inequalities, each
    closed with its γ-term bounded through `FloatModel.gamma_num` so `norm_num` never evaluates
    a big power. Built bottom-up at block granularity (18 block steps and 3 downsample steps
    rather than 183 leaf steps), the way `r34EvalBridge_maps` is.

    ⛔ Of the 366, the 23 that read `2 * Ā' ≤ Ē'` are the CAP, not the fold: at each LayerNorm
    site the modulus is discharged by the triangle inequality rather than by the propagated
    error. That is why `9706 / 4858 = 2.00`. -/
theorem cnxBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).Maps 1 0
      (4858 * 10 ^ 224) (9706 * 10 ^ 224) := by
  have P := cnxProfile_committed M hMu hε5
  have mStemC := FloatBridgesTo.Maps.flatConvStride4 (h := 56) (w := 56) M
    wts.sW wts.sb P.hw' P.hbb (by norm_num) B.sW B.sb
    (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā := 1) (Ē := 0)
    (Ā' := 3181 / 10 ^ 2) (Ē' := 9480 / 10 ^ 8) (by norm_num [u32]) (by norm_num [u32])
  have mStem := mStemC.comp (by norm_num)
    (cnxChanLNMaps M R P wts.sγ wts.sβ B.sγ B.sβ Eps.hs (by norm_num) (by norm_num)
      (Ā := 3181 / 10 ^ 2) (Ē := 9480 / 10 ^ 8)
      (A1 := 2027 * 10 ^ 1) (E1 := 4054 * 10 ^ 1) (A2 := 9730 * 10 ^ 1) (E2 := 1946 * 10 ^ 2)
      (Ā' := 9731 * 10 ^ 1) (Ē' := 1947 * 10 ^ 2)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  have mS1 := mStem.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 0) (R.lnF M 96 ((wts.s1 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s1 0) (R.bridge M P 96 (by norm_num) ((wts.s1 0).εn) (Eps.h1 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 96 (by norm_num) ((wts.s1 0).εn) (Eps.h1 0)
        (Ā := 2861 * 10 ^ 3) (Ā' := 1824 * 10 ^ 6) (Ē' := 3648 * 10 ^ 6) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 2861 * 10 ^ 3) (E1 := 5725 * 10 ^ 3)
      (A2 := 1824 * 10 ^ 6) (E2 := 3648 * 10 ^ 6)
      (A3 := 8756 * 10 ^ 6) (E3 := 1752 * 10 ^ 7)
      (A4 := 8757 * 10 ^ 6) (E4 := 1753 * 10 ^ 7)
      (A5 := 5045 * 10 ^ 8) (E5 := 1010 * 10 ^ 9)
      (A6 := 5046 * 10 ^ 8) (E6 := 1516 * 10 ^ 9)
      (A7 := 1163 * 10 ^ 11) (E7 := 3493 * 10 ^ 11)
      (A8 := 9770 * 10 ^ 11) (E8 := 2935 * 10 ^ 12)
      (Ā' := 9771 * 10 ^ 11) (Ē' := 2936 * 10 ^ 12)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 1) (R.lnF M 96 ((wts.s1 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s1 1) (R.bridge M P 96 (by norm_num) ((wts.s1 1).εn) (Eps.h1 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 96 (by norm_num) ((wts.s1 1).εn) (Eps.h1 1)
          (Ā := 2873 * 10 ^ 13) (Ā' := 1831 * 10 ^ 16) (Ē' := 3662 * 10 ^ 16) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (A1 := 2873 * 10 ^ 13) (E1 := 8632 * 10 ^ 13)
        (A2 := 1831 * 10 ^ 16) (E2 := 3662 * 10 ^ 16)
        (A3 := 8789 * 10 ^ 16) (E3 := 1758 * 10 ^ 17)
        (A4 := 8790 * 10 ^ 16) (E4 := 1759 * 10 ^ 17)
        (A5 := 5064 * 10 ^ 18) (E5 := 1014 * 10 ^ 19)
        (A6 := 5065 * 10 ^ 18) (E6 := 1522 * 10 ^ 19)
        (A7 := 1168 * 10 ^ 21) (E7 := 3507 * 10 ^ 21)
        (A8 := 9812 * 10 ^ 21) (E8 := 2946 * 10 ^ 22)
        (Ā' := 9813 * 10 ^ 21) (Ē' := 2947 * 10 ^ 22)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 2) (R.lnF M 96 ((wts.s1 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s1 2) (R.bridge M P 96 (by norm_num) ((wts.s1 2).εn) (Eps.h1 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 96 (by norm_num) ((wts.s1 2).εn) (Eps.h1 2)
            (Ā := 2886 * 10 ^ 23) (Ā' := 1839 * 10 ^ 26) (Ē' := 3678 * 10 ^ 26) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
          (A1 := 2886 * 10 ^ 23) (E1 := 8665 * 10 ^ 23)
          (A2 := 1839 * 10 ^ 26) (E2 := 3678 * 10 ^ 26)
          (A3 := 8828 * 10 ^ 26) (E3 := 1766 * 10 ^ 27)
          (A4 := 8829 * 10 ^ 26) (E4 := 1767 * 10 ^ 27)
          (A5 := 5086 * 10 ^ 28) (E5 := 1018 * 10 ^ 29)
          (A6 := 5087 * 10 ^ 28) (E6 := 1528 * 10 ^ 29)
          (A7 := 1173 * 10 ^ 31) (E7 := 3521 * 10 ^ 31)
          (A8 := 9854 * 10 ^ 31) (E8 := 2958 * 10 ^ 32)
          (Ā' := 9855 * 10 ^ 31) (Ē' := 2959 * 10 ^ 32)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mD1 := mS1.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 28 28 M wts.d1 (R.lnF M 96 wts.d1.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d1
      (R.bridge M P 96 (by norm_num) wts.d1.ε Eps.hd1) P.hq P.hgl P.hbb
      (R.maps M P 96 (by norm_num) wts.d1.ε Eps.hd1
        (Ā := 9855 * 10 ^ 31) (Ā' := 6280 * 10 ^ 34) (Ē' := 1256 * 10 ^ 35) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 6280 * 10 ^ 34) (E1 := 1256 * 10 ^ 35) (A2 := 3015 * 10 ^ 35) (E2 := 6029 * 10 ^ 35)
      (A3 := 3016 * 10 ^ 35) (E3 := 6030 * 10 ^ 35) (Ā' := 6950 * 10 ^ 37) (Ē' := 1390 * 10 ^ 38)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS2 := mD1.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 0) (R.lnF M 192 ((wts.s2 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s2 0) (R.bridge M P 192 (by norm_num) ((wts.s2 0).εn) (Eps.h2 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 192 (by norm_num) ((wts.s2 0).εn) (Eps.h2 0)
        (Ā := 2044 * 10 ^ 39) (Ā' := 1303 * 10 ^ 42) (Ē' := 2606 * 10 ^ 42) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 2044 * 10 ^ 39) (E1 := 4087 * 10 ^ 39)
      (A2 := 1303 * 10 ^ 42) (E2 := 2606 * 10 ^ 42)
      (A3 := 6255 * 10 ^ 42) (E3 := 1251 * 10 ^ 43)
      (A4 := 6256 * 10 ^ 42) (E4 := 1252 * 10 ^ 43)
      (A5 := 7207 * 10 ^ 44) (E5 := 1443 * 10 ^ 45)
      (A6 := 7208 * 10 ^ 44) (E6 := 2165 * 10 ^ 45)
      (A7 := 3322 * 10 ^ 47) (E7 := 9977 * 10 ^ 47)
      (A8 := 2791 * 10 ^ 48) (E8 := 8381 * 10 ^ 48)
      (Ā' := 2792 * 10 ^ 48) (Ē' := 8382 * 10 ^ 48)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 1) (R.lnF M 192 ((wts.s2 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s2 1) (R.bridge M P 192 (by norm_num) ((wts.s2 1).εn) (Eps.h2 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 192 (by norm_num) ((wts.s2 1).εn) (Eps.h2 1)
          (Ā := 8209 * 10 ^ 49) (Ā' := 5231 * 10 ^ 52) (Ē' := 1047 * 10 ^ 53) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (A1 := 8209 * 10 ^ 49) (E1 := 2465 * 10 ^ 50)
        (A2 := 5231 * 10 ^ 52) (E2 := 1047 * 10 ^ 53)
        (A3 := 2511 * 10 ^ 53) (E3 := 5026 * 10 ^ 53)
        (A4 := 2512 * 10 ^ 53) (E4 := 5027 * 10 ^ 53)
        (A5 := 2894 * 10 ^ 55) (E5 := 5792 * 10 ^ 55)
        (A6 := 2895 * 10 ^ 55) (E6 := 8689 * 10 ^ 55)
        (A7 := 1335 * 10 ^ 58) (E7 := 4005 * 10 ^ 58)
        (A8 := 1122 * 10 ^ 59) (E8 := 3365 * 10 ^ 59)
        (Ā' := 1123 * 10 ^ 59) (Ē' := 3366 * 10 ^ 59)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 2) (R.lnF M 192 ((wts.s2 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s2 2) (R.bridge M P 192 (by norm_num) ((wts.s2 2).εn) (Eps.h2 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 192 (by norm_num) ((wts.s2 2).εn) (Eps.h2 2)
            (Ā := 3302 * 10 ^ 60) (Ā' := 2105 * 10 ^ 63) (Ē' := 4210 * 10 ^ 63) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
          (A1 := 3302 * 10 ^ 60) (E1 := 9897 * 10 ^ 60)
          (A2 := 2105 * 10 ^ 63) (E2 := 4210 * 10 ^ 63)
          (A3 := 1011 * 10 ^ 64) (E3 := 2021 * 10 ^ 64)
          (A4 := 1012 * 10 ^ 64) (E4 := 2022 * 10 ^ 64)
          (A5 := 1166 * 10 ^ 66) (E5 := 2330 * 10 ^ 66)
          (A6 := 1167 * 10 ^ 66) (E6 := 3496 * 10 ^ 66)
          (A7 := 5378 * 10 ^ 68) (E7 := 1612 * 10 ^ 69)
          (A8 := 4518 * 10 ^ 69) (E8 := 1355 * 10 ^ 70)
          (Ā' := 4519 * 10 ^ 69) (Ē' := 1356 * 10 ^ 70)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mD2 := mS2.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 14 14 M wts.d2 (R.lnF M 192 wts.d2.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d2
      (R.bridge M P 192 (by norm_num) wts.d2.ε Eps.hd2) P.hq P.hgl P.hbb
      (R.maps M P 192 (by norm_num) wts.d2.ε Eps.hd2
        (Ā := 4519 * 10 ^ 69) (Ā' := 2880 * 10 ^ 72) (Ē' := 5760 * 10 ^ 72) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 2880 * 10 ^ 72) (E1 := 5760 * 10 ^ 72) (A2 := 1383 * 10 ^ 73) (E2 := 2765 * 10 ^ 73)
      (A3 := 1384 * 10 ^ 73) (E3 := 2766 * 10 ^ 73) (Ā' := 6378 * 10 ^ 75) (Ē' := 1275 * 10 ^ 76)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS3 := mD2.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 0) (R.lnF M 384 ((wts.s3 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s3 0) (R.bridge M P 384 (by norm_num) ((wts.s3 0).εn) (Eps.h3 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 384 (by norm_num) ((wts.s3 0).εn) (Eps.h3 0)
        (Ā := 1876 * 10 ^ 77) (Ā' := 1196 * 10 ^ 80) (Ē' := 2392 * 10 ^ 80) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 1876 * 10 ^ 77) (E1 := 3749 * 10 ^ 77)
      (A2 := 1196 * 10 ^ 80) (E2 := 2392 * 10 ^ 80)
      (A3 := 5741 * 10 ^ 80) (E3 := 1149 * 10 ^ 81)
      (A4 := 5742 * 10 ^ 80) (E4 := 1150 * 10 ^ 81)
      (A5 := 1323 * 10 ^ 83) (E5 := 2650 * 10 ^ 83)
      (A6 := 1324 * 10 ^ 83) (E6 := 3976 * 10 ^ 83)
      (A7 := 1221 * 10 ^ 86) (E7 := 3665 * 10 ^ 86)
      (A8 := 1026 * 10 ^ 87) (E8 := 3079 * 10 ^ 87)
      (Ā' := 1027 * 10 ^ 87) (Ē' := 3080 * 10 ^ 87)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 1) (R.lnF M 384 ((wts.s3 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s3 1) (R.bridge M P 384 (by norm_num) ((wts.s3 1).εn) (Eps.h3 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 384 (by norm_num) ((wts.s3 1).εn) (Eps.h3 1)
          (Ā := 3020 * 10 ^ 88) (Ā' := 1925 * 10 ^ 91) (Ē' := 3850 * 10 ^ 91) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (A1 := 3020 * 10 ^ 88) (E1 := 9056 * 10 ^ 88)
        (A2 := 1925 * 10 ^ 91) (E2 := 3850 * 10 ^ 91)
        (A3 := 9241 * 10 ^ 91) (E3 := 1849 * 10 ^ 92)
        (A4 := 9242 * 10 ^ 91) (E4 := 1850 * 10 ^ 92)
        (A5 := 2130 * 10 ^ 94) (E5 := 4263 * 10 ^ 94)
        (A6 := 2131 * 10 ^ 94) (E6 := 6395 * 10 ^ 94)
        (A7 := 1965 * 10 ^ 97) (E7 := 5895 * 10 ^ 97)
        (A8 := 1651 * 10 ^ 98) (E8 := 4952 * 10 ^ 98)
        (Ā' := 1652 * 10 ^ 98) (Ē' := 4953 * 10 ^ 98)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 2) (R.lnF M 384 ((wts.s3 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s3 2) (R.bridge M P 384 (by norm_num) ((wts.s3 2).εn) (Eps.h3 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 384 (by norm_num) ((wts.s3 2).εn) (Eps.h3 2)
            (Ā := 4857 * 10 ^ 99) (Ā' := 3095 * 10 ^ 102) (Ē' := 6190 * 10 ^ 102) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
          (A1 := 4857 * 10 ^ 99) (E1 := 1457 * 10 ^ 100)
          (A2 := 3095 * 10 ^ 102) (E2 := 6190 * 10 ^ 102)
          (A3 := 1486 * 10 ^ 103) (E3 := 2972 * 10 ^ 103)
          (A4 := 1487 * 10 ^ 103) (E4 := 2973 * 10 ^ 103)
          (A5 := 3427 * 10 ^ 105) (E5 := 6851 * 10 ^ 105)
          (A6 := 3428 * 10 ^ 105) (E6 := 1028 * 10 ^ 106)
          (A7 := 3160 * 10 ^ 108) (E7 := 9476 * 10 ^ 108)
          (A8 := 2655 * 10 ^ 109) (E8 := 7960 * 10 ^ 109)
          (Ā' := 2656 * 10 ^ 109) (Ē' := 7961 * 10 ^ 109)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
          ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 3) (R.lnF M 384 ((wts.s3 3).εn))
            P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
            (B.s3 3) (R.bridge M P 384 (by norm_num) ((wts.s3 3).εn) (Eps.h3 3))
            P.hq P.hgl P.hbb P.hsl
            (R.maps M P 384 (by norm_num) ((wts.s3 3).εn) (Eps.h3 3)
              (Ā := 7809 * 10 ^ 110) (Ā' := 4976 * 10 ^ 113) (Ē' := 9952 * 10 ^ 113) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
            (A1 := 7809 * 10 ^ 110) (E1 := 2341 * 10 ^ 111)
            (A2 := 4976 * 10 ^ 113) (E2 := 9952 * 10 ^ 113)
            (A3 := 2389 * 10 ^ 114) (E3 := 4777 * 10 ^ 114)
            (A4 := 2390 * 10 ^ 114) (E4 := 4778 * 10 ^ 114)
            (A5 := 5507 * 10 ^ 116) (E5 := 1101 * 10 ^ 117)
            (A6 := 5508 * 10 ^ 116) (E6 := 1652 * 10 ^ 117)
            (A7 := 5077 * 10 ^ 119) (E7 := 1523 * 10 ^ 120)
            (A8 := 4265 * 10 ^ 120) (E8 := 1280 * 10 ^ 121)
            (Ā' := 4266 * 10 ^ 120) (Ē' := 1281 * 10 ^ 121)
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
            (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
            ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 4) (R.lnF M 384 ((wts.s3 4).εn))
              P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
              (B.s3 4) (R.bridge M P 384 (by norm_num) ((wts.s3 4).εn) (Eps.h3 4))
              P.hq P.hgl P.hbb P.hsl
              (R.maps M P 384 (by norm_num) ((wts.s3 4).εn) (Eps.h3 4)
                (Ā := 1255 * 10 ^ 122) (Ā' := 7997 * 10 ^ 124) (Ē' := 1600 * 10 ^ 125) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
              (A1 := 1255 * 10 ^ 122) (E1 := 3767 * 10 ^ 122)
              (A2 := 7997 * 10 ^ 124) (E2 := 1600 * 10 ^ 125)
              (A3 := 3839 * 10 ^ 125) (E3 := 7681 * 10 ^ 125)
              (A4 := 3840 * 10 ^ 125) (E4 := 7682 * 10 ^ 125)
              (A5 := 8848 * 10 ^ 127) (E5 := 1770 * 10 ^ 128)
              (A6 := 8849 * 10 ^ 127) (E6 := 2656 * 10 ^ 128)
              (A7 := 8156 * 10 ^ 130) (E7 := 2449 * 10 ^ 131)
              (A8 := 6852 * 10 ^ 131) (E8 := 2058 * 10 ^ 132)
              (Ā' := 6853 * 10 ^ 131) (Ē' := 2059 * 10 ^ 132)
              (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
              (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
              (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
              ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 5) (R.lnF M 384 ((wts.s3 5).εn))
                P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                (B.s3 5) (R.bridge M P 384 (by norm_num) ((wts.s3 5).εn) (Eps.h3 5))
                P.hq P.hgl P.hbb P.hsl
                (R.maps M P 384 (by norm_num) ((wts.s3 5).εn) (Eps.h3 5)
                  (Ā := 2015 * 10 ^ 133) (Ā' := 1284 * 10 ^ 136) (Ē' := 2568 * 10 ^ 136) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
                (A1 := 2015 * 10 ^ 133) (E1 := 6054 * 10 ^ 133)
                (A2 := 1284 * 10 ^ 136) (E2 := 2568 * 10 ^ 136)
                (A3 := 6164 * 10 ^ 136) (E3 := 1233 * 10 ^ 137)
                (A4 := 6165 * 10 ^ 136) (E4 := 1234 * 10 ^ 137)
                (A5 := 1421 * 10 ^ 139) (E5 := 2844 * 10 ^ 139)
                (A6 := 1422 * 10 ^ 139) (E6 := 4267 * 10 ^ 139)
                (A7 := 1311 * 10 ^ 142) (E7 := 3933 * 10 ^ 142)
                (A8 := 1102 * 10 ^ 143) (E8 := 3304 * 10 ^ 143)
                (Ā' := 1103 * 10 ^ 143) (Ē' := 3305 * 10 ^ 143)
                (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
                ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 6) (R.lnF M 384 ((wts.s3 6).εn))
                  P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                  (B.s3 6) (R.bridge M P 384 (by norm_num) ((wts.s3 6).εn) (Eps.h3 6))
                  P.hq P.hgl P.hbb P.hsl
                  (R.maps M P 384 (by norm_num) ((wts.s3 6).εn) (Eps.h3 6)
                    (Ā := 3243 * 10 ^ 144) (Ā' := 2067 * 10 ^ 147) (Ē' := 4134 * 10 ^ 147) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
                  (A1 := 3243 * 10 ^ 144) (E1 := 9717 * 10 ^ 144)
                  (A2 := 2067 * 10 ^ 147) (E2 := 4134 * 10 ^ 147)
                  (A3 := 9922 * 10 ^ 147) (E3 := 1985 * 10 ^ 148)
                  (A4 := 9923 * 10 ^ 147) (E4 := 1986 * 10 ^ 148)
                  (A5 := 2287 * 10 ^ 150) (E5 := 4576 * 10 ^ 150)
                  (A6 := 2288 * 10 ^ 150) (E6 := 6865 * 10 ^ 150)
                  (A7 := 2109 * 10 ^ 153) (E7 := 6328 * 10 ^ 153)
                  (A8 := 1772 * 10 ^ 154) (E8 := 5316 * 10 ^ 154)
                  (Ā' := 1773 * 10 ^ 154) (Ē' := 5317 * 10 ^ 154)
                  (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                  (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                  (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
                  ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 7) (R.lnF M 384 ((wts.s3 7).εn))
                    P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                    (B.s3 7) (R.bridge M P 384 (by norm_num) ((wts.s3 7).εn) (Eps.h3 7))
                    P.hq P.hgl P.hbb P.hsl
                    (R.maps M P 384 (by norm_num) ((wts.s3 7).εn) (Eps.h3 7)
                      (Ā := 5213 * 10 ^ 155) (Ā' := 3322 * 10 ^ 158) (Ē' := 6644 * 10 ^ 158) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
                    (A1 := 5213 * 10 ^ 155) (E1 := 1564 * 10 ^ 156)
                    (A2 := 3322 * 10 ^ 158) (E2 := 6644 * 10 ^ 158)
                    (A3 := 1595 * 10 ^ 159) (E3 := 3190 * 10 ^ 159)
                    (A4 := 1596 * 10 ^ 159) (E4 := 3191 * 10 ^ 159)
                    (A5 := 3678 * 10 ^ 161) (E5 := 7353 * 10 ^ 161)
                    (A6 := 3679 * 10 ^ 161) (E6 := 1103 * 10 ^ 162)
                    (A7 := 3391 * 10 ^ 164) (E7 := 1017 * 10 ^ 165)
                    (A8 := 2849 * 10 ^ 165) (E8 := 8543 * 10 ^ 165)
                    (Ā' := 2850 * 10 ^ 165) (Ē' := 8544 * 10 ^ 165)
                    (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                    (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                    (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
                    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 8) (R.lnF M 384 ((wts.s3 8).εn))
                      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                      (B.s3 8) (R.bridge M P 384 (by norm_num) ((wts.s3 8).εn) (Eps.h3 8))
                      P.hq P.hgl P.hbb P.hsl
                      (R.maps M P 384 (by norm_num) ((wts.s3 8).εn) (Eps.h3 8)
                        (Ā := 8380 * 10 ^ 166) (Ā' := 5340 * 10 ^ 169) (Ē' := 1068 * 10 ^ 170) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
                      (A1 := 8380 * 10 ^ 166) (E1 := 2512 * 10 ^ 167)
                      (A2 := 5340 * 10 ^ 169) (E2 := 1068 * 10 ^ 170)
                      (A3 := 2564 * 10 ^ 170) (E3 := 5127 * 10 ^ 170)
                      (A4 := 2565 * 10 ^ 170) (E4 := 5128 * 10 ^ 170)
                      (A5 := 5910 * 10 ^ 172) (E5 := 1182 * 10 ^ 173)
                      (A6 := 5911 * 10 ^ 172) (E6 := 1774 * 10 ^ 173)
                      (A7 := 5449 * 10 ^ 175) (E7 := 1636 * 10 ^ 176)
                      (A8 := 4578 * 10 ^ 176) (E8 := 1375 * 10 ^ 177)
                      (Ā' := 4579 * 10 ^ 176) (Ē' := 1376 * 10 ^ 177)
                      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
                      (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
                      FloatBridgesTo.Maps.idVec)))))))))
  have mD3 := mS3.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 7 7 M wts.d3 (R.lnF M 384 wts.d3.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d3
      (R.bridge M P 384 (by norm_num) wts.d3.ε Eps.hd3) P.hq P.hgl P.hbb
      (R.maps M P 384 (by norm_num) wts.d3.ε Eps.hd3
        (Ā := 4579 * 10 ^ 176) (Ā' := 2918 * 10 ^ 179) (Ē' := 5836 * 10 ^ 179) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 2918 * 10 ^ 179) (E1 := 5836 * 10 ^ 179) (A2 := 1401 * 10 ^ 180) (E2 := 2802 * 10 ^ 180)
      (A3 := 1402 * 10 ^ 180) (E3 := 2803 * 10 ^ 180) (Ā' := 1293 * 10 ^ 183) (Ē' := 2584 * 10 ^ 183)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS4 := mD3.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 0) (R.lnF M 768 ((wts.s4 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s4 0) (R.bridge M P 768 (by norm_num) ((wts.s4 0).εn) (Eps.h4 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 768 (by norm_num) ((wts.s4 0).εn) (Eps.h4 0)
        (Ā := 3802 * 10 ^ 184) (Ā' := 2423 * 10 ^ 187) (Ē' := 4846 * 10 ^ 187) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
      (A1 := 3802 * 10 ^ 184) (E1 := 7597 * 10 ^ 184)
      (A2 := 2423 * 10 ^ 187) (E2 := 4846 * 10 ^ 187)
      (A3 := 1164 * 10 ^ 188) (E3 := 2327 * 10 ^ 188)
      (A4 := 1165 * 10 ^ 188) (E4 := 2328 * 10 ^ 188)
      (A5 := 5369 * 10 ^ 190) (E5 := 1073 * 10 ^ 191)
      (A6 := 5370 * 10 ^ 190) (E6 := 1610 * 10 ^ 191)
      (A7 := 9900 * 10 ^ 193) (E7 := 2969 * 10 ^ 194)
      (A8 := 8317 * 10 ^ 194) (E8 := 2494 * 10 ^ 195)
      (Ā' := 8318 * 10 ^ 194) (Ē' := 2495 * 10 ^ 195)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 1) (R.lnF M 768 ((wts.s4 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s4 1) (R.bridge M P 768 (by norm_num) ((wts.s4 1).εn) (Eps.h4 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 768 (by norm_num) ((wts.s4 1).εn) (Eps.h4 1)
          (Ā := 2446 * 10 ^ 196) (Ā' := 1559 * 10 ^ 199) (Ē' := 3118 * 10 ^ 199) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (A1 := 2446 * 10 ^ 196) (E1 := 7336 * 10 ^ 196)
        (A2 := 1559 * 10 ^ 199) (E2 := 3118 * 10 ^ 199)
        (A3 := 7484 * 10 ^ 199) (E3 := 1497 * 10 ^ 200)
        (A4 := 7485 * 10 ^ 199) (E4 := 1498 * 10 ^ 200)
        (A5 := 3450 * 10 ^ 202) (E5 := 6904 * 10 ^ 202)
        (A6 := 3451 * 10 ^ 202) (E6 := 1036 * 10 ^ 203)
        (A7 := 6363 * 10 ^ 205) (E7 := 1911 * 10 ^ 206)
        (A8 := 5345 * 10 ^ 206) (E8 := 1606 * 10 ^ 207)
        (Ā' := 5346 * 10 ^ 206) (Ē' := 1607 * 10 ^ 207)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
        (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 2) (R.lnF M 768 ((wts.s4 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s4 2) (R.bridge M P 768 (by norm_num) ((wts.s4 2).εn) (Eps.h4 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 768 (by norm_num) ((wts.s4 2).εn) (Eps.h4 2)
            (Ā := 1572 * 10 ^ 208) (Ā' := 1002 * 10 ^ 211) (Ē' := 2004 * 10 ^ 211) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
          (A1 := 1572 * 10 ^ 208) (E1 := 4725 * 10 ^ 208)
          (A2 := 1002 * 10 ^ 211) (E2 := 2004 * 10 ^ 211)
          (A3 := 4810 * 10 ^ 211) (E3 := 9620 * 10 ^ 211)
          (A4 := 4811 * 10 ^ 211) (E4 := 9621 * 10 ^ 211)
          (A5 := 2218 * 10 ^ 214) (E5 := 4434 * 10 ^ 214)
          (A6 := 2219 * 10 ^ 214) (E6 := 6652 * 10 ^ 214)
          (A7 := 4091 * 10 ^ 217) (E7 := 1227 * 10 ^ 218)
          (A8 := 3437 * 10 ^ 218) (E8 := 1031 * 10 ^ 219)
          (Ā' := 3438 * 10 ^ 218) (Ē' := 1032 * 10 ^ 219)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
          (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mGAP := mS4.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 768) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      P.hq (by norm_num [u32]) (by norm_num [u32]) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 3439 * 10 ^ 218) (Ē' := 1033 * 10 ^ 219) (by norm_num [u32]) (by norm_num [u32]))
  have mHead := mGAP.comp (by norm_num)
    (cnxRowLNMaps (s := 1) M R P wts.hγ wts.hβ B.hγ B.hβ Eps.hh (by norm_num)
      (Ā := 3439 * 10 ^ 218) (Ē := 1033 * 10 ^ 219)
      (A1 := 2192 * 10 ^ 221) (E1 := 4384 * 10 ^ 221) (A2 := 1053 * 10 ^ 222) (E2 := 2105 * 10 ^ 222)
      (Ā' := 1054 * 10 ^ 222) (Ē' := 2106 * 10 ^ 222)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
  exact mHead.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M wts.Wd wts.bd P.hw' P.hbb (by norm_num) B.Wd B.bd
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 4858 * 10 ^ 224) (Ē' := 9706 * 10 ^ 224) (by norm_num [u32]) (by norm_num [u32]))

/-- The deployed ConvNeXt-T bridge's certified output window at the committed profile. -/
theorem cnxBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).mag 1 ≤ 4858 * 10 ^ 224 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ConvNeXt-T bridge's fresh budget at the committed profile — `2.00 ×` the
    certified window, which is the tell that the cap is biting at every LayerNorm site and the
    statement is the triangle inequality rather than the fold. -/
theorem cnxBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).fresh 1 ≤ 9706 * 10 ^ 224 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐ **The deployed ConvNeXt-T forward is within `9.706·10²²⁷` of the certified real forward,
    per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile, for `ε ≥ 10⁻⁵`,
    any device LayerNorm statistics accurate to `10⁻²` and any device GELU accurate to `10⁻²`,
    for any rounding model at binary32 accuracy.

    ⛔ **Read the file header before quoting this.** It is a capped statement — the float and
    real forwards both land in the certified `4.858·10²²⁷` window — and NOT the interval fold
    that ResNet-34's, MobileNetV2's and EfficientNet-B0's numbers are. LayerNorm has no
    frozen-statistics mode, so there is no version of this net for which the fold exists. -/
theorem cnx_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε)
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cnxForwardF M R G wts x j - cnxForward wts x j| ≤ 9706 * 10 ^ 224 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed ConvNeXt-T forward
-- ════════════════════════════════════════════════════════════════

/-- **The bridged skeleton IS the committed net** — `convNextForwardTCh_eq_skeleton` read
    backwards. ⚠ It only became true on 2026-09-03: until then the whole-net bridge held `id`
    in the head slot while the tie had carried `rowLNVecFlat 1 768` since the head LayerNorm
    was restored on 2026-08-30. -/
theorem cnxForward_eq_committed (wts : CnxTWeightsCh) :
    cnxForward wts = convNextForwardTCh wts :=
  (convNextForwardTCh_eq_skeleton wts).symm

/-- ⭐ **The number, stated about the committed forward.** `cnx_float_logits_le` with
    `convNextForwardTCh` on the real side instead of the record-plugged skeleton — so the
    budget is a claim about the net `ConvNeXtRender.lean` renders. -/
theorem cnx_float_logits_le_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100))
    (wts : CnxTWeightsCh) (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε)
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cnxForwardF M R G wts x j - convNextForwardTCh wts x j| ≤ 9706 * 10 ^ 224 := by
  have h := cnx_float_logits_le M hMu hε5 R G wts B Eps x hx j
  rwa [cnxForward_eq_committed wts] at h

end Proofs
