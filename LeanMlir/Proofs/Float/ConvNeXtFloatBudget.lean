import LeanMlir.Proofs.Float.FloatBudgetEnvLN
import LeanMlir.Proofs.Foundation.WholeNetForwardTies
import LeanMlir.Proofs.Float.Binary32Instance

/-! # A NUMBER for ConvNeXt-T: the committed channel-LayerNorm forward, at the cap

The fourth ImageNet-scale whole-net float statement, and ⛔ **it is not the same kind of
statement as the other three.** For the `[3,3,9,3]` ConvNeXt-T forward at `224²` — 4×4/s4
patchify stem, 18 blocks of `depthwise 7×7 → channel-LN → 1×1 expand → GELU → 1×1 project →
layer scale → skip`, three LN+2×2/s2 downsamples, GAP, head LayerNorm, classifier — on the unit
input window, at the profile measured on the finished 300-epoch ImageNet checkpoint, for any
rounding model at binary32 accuracy:

    output window  ≤ 4.871·10¹³⁰      (`cnxBridge_mag_le`)
    fresh budget   ≤ 9.738·10¹³⁰      (`cnxBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 9.738·10¹³⁰` (`cnx_float_logits_le`).

⛔ **`budget / window = 2.00`, and that ratio is the whole caveat.** Every one of the 23
LayerNorm sites goes through `FloatBridgesTo.capped`, whose modulus is `min(fold, 2·window)`,
and the right branch is what closes. So this number says *the float and the real forward both
land in the certified window* — the triangle inequality — where ResNet-34's `1.548·10²⁰⁹`,
MobileNetV2's `1.444·10⁹⁶` and EfficientNet-B0's `8.408·10²¹⁰` say *the rounding error folds to
this*. Do not table it beside them without saying so (`planning/archive/float_budget_numbers_log.md` §9).

⭐⭐ **Why the cap is not optional here.** LayerNorm reduces its mean and variance out of its own
input, so perturbing the input moves the statistics and `bnReluBudget` carries a term quadratic
in the window. BatchNorm has an escape — freeze the statistics and the map becomes affine, which
is why the other three numbers are stated at inference BN — and **LayerNorm has none**: there
are no running statistics to freeze, so there is no second render to build and no eval twin to
bound. Uncapped at the leaf this file used to state, the same fold is `10⁵²³⁹`; `norm_num` refuses numerals past ~`10³⁰⁰`, so there
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

⭐⭐ **The window is charged at `|x̂| ≤ √n`, not at `|x − μ|·|istd| ≤ 2A·S` — §0.1's escape 2,
and it is worth 53 orders.** `Maps.bnCappedX` (`BnXhatFloatBridge.lean`) states the
pure-normalise leaf at `bnXhat_sq_le`'s bound, which mentions neither factor and holds at every
input, so `Xh` is the CEILING root of the reduction width (10/14/20/28 for 96/192/384/768) and
enters no profile. Two places charged the window and both are fixed: the real output's
magnitude, and the ROUNDING of the float product — which `bnNormBudget` bounds by
`(D+ea)·(S+ei)` when that product IS the normalised activation one rounding away. Before this
the number was `4.858·10²²⁷ / 9.706·10²²⁷`. ⭐ It cost no new hypothesis and no new mathematics:
`bnXhat_sq_le` had been in the repo since the realistic-seal work and is load-bearing on all
four whole-net BACKWARD numbers; the forward leaf simply threw it away.

⛔ **It does NOT make this a fold.** The modulus still carries §0.1's quadratic — escape 2's
modulus half is deliberately not taken (`BnXhatFloatBridge.lean`'s header prices it) — and even
with that quadratic gone the honest fold is **82 orders above the triangle inequality** here, so
`capped`'s `min` still selects the cap. ⭐ Uncapped at this leaf the fold is `2.823·10²⁵⁶`,
3 orders past §3.7(a)'s shape-dependent `norm_num` wall, and `4.710·10²²⁶` at an operating point
`|istd| ≤ 16` — statable, and 80 orders worse than the number above. **The cap is not a shortcut
past a fold that exists; it is the better bound.**

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
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (c : Nat) (Xh : ℝ) (hc : 0 < c)
    (hXh0 : 0 ≤ Xh) (hcXh : (c : ℝ) ≤ Xh ^ 2) (e : ℝ)
    (he : ε ≤ e) : FloatBridgesTo (layerNormForward c e 1 0) (R.lnF M c e) :=
  R.bridgeAt M P.hε P.hSε c hc hXh0 hcXh e he

/-- ⛔ **One LayerNorm site's envelope at this net's profile** — `DeviceLN.mapsAt`. `nA` is the
    real window `2Ā·S + bnNormBudget`, which is the fold; `nE` is `2·Ā'`, which is not. -/
theorem DeviceLN.maps (R : DeviceLN emr ei) (M : FloatModel)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (c : Nat) (Xh : ℝ) (hc : 0 < c)
    (hXh0 : 0 ≤ Xh) (hcXh : (c : ℝ) ≤ Xh ^ 2) (e : ℝ)
    (he : ε ≤ e) {Ā Ē Ā' Ē' : ℝ}
    (nA : 1 * Xh + 0 + bnNormBudgetX q Xh (2 * Ā) S 1 0 (emr * Ā) ei ≤ Ā')
    (nE : 2 * Ā' ≤ Ē') :
    (R.bridge M P c Xh hc hXh0 hcXh e he).Maps Ā Ē Ā' Ē' :=
  R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq c hc hXh0 hcXh e he nA nE

/-- **A channel-LayerNorm site's envelope**: the capped normalise, then γ, then β. -/
theorem cnxChanLNMaps {c h w : Nat} (M : FloatModel) (R : DeviceLN emr ei)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (Xh : ℝ) (γ β : Vec c)
    (hγ : ∀ i, |γ i| ≤ gl) (hβ : ∀ i, |β i| ≤ bb) {e : ℝ} (he : ε ≤ e)
    (hc : 0 < c) (hhw : 0 < h * w) (hXh0 : 0 ≤ Xh) (hcXh : (c : ℝ) ≤ Xh ^ 2)
    {Ā Ē A1 E1 A2 E2 Ā' Ē' : ℝ}
    (nA : 1 * Xh + 0 + bnNormBudgetX q Xh (2 * Ā) S 1 0 (emr * Ā) ei ≤ A1)
    (nE : 2 * A1 ≤ E1)
    (gA : gl * A1 + FloatModel.mulErr q gl A1 0 0 ≤ A2)
    (gE : FloatModel.mulErr q gl A1 0 0 + gl * E1 ≤ E2)
    (bA : A2 + bb + q * (A2 + bb) ≤ Ā') (bE : q * (A2 + bb) + E2 ≤ Ē') :
    (floatBridgesTo_chanLNTensor3 (h := h) (w := w) M γ β (R.lnF M c e) hc hγ hβ
      (R.bridge M P c Xh hc hXh0 hcXh e he)).Maps Ā Ē Ā' Ē' :=
  Maps.chanLNTensor3 M γ β (R.lnF M c e) hc hhw hγ hβ (R.bridge M P c Xh hc hXh0 hcXh e he)
    P.hq P.hgl P.hbb (R.maps M P c Xh hc hXh0 hcXh e he nA nE) gA gE bA bE

/-- **The head LayerNorm site's envelope** — the same three stages at one row. -/
theorem cnxRowLNMaps {s c : Nat} (M : FloatModel) (R : DeviceLN emr ei)
    (P : CnxProfile M ε w' bb gl sl egelu emr ei S q) (Xh : ℝ) (γ β : Vec c)
    (hγ : ∀ i, |γ i| ≤ gl) (hβ : ∀ i, |β i| ≤ bb) {e : ℝ} (he : ε ≤ e) (hc : 0 < c)
    (hXh0 : 0 ≤ Xh) (hcXh : (c : ℝ) ≤ Xh ^ 2)
    {Ā Ē A1 E1 A2 E2 Ā' Ē' : ℝ}
    (nA : 1 * Xh + 0 + bnNormBudgetX q Xh (2 * Ā) S 1 0 (emr * Ā) ei ≤ A1)
    (nE : 2 * A1 ≤ E1)
    (gA : gl * A1 + FloatModel.mulErr q gl A1 0 0 ≤ A2)
    (gE : FloatModel.mulErr q gl A1 0 0 + gl * E1 ≤ E2)
    (bA : A2 + bb + q * (A2 + bb) ≤ Ā') (bE : q * (A2 + bb) + E2 ≤ Ē') :
    (floatBridgesTo_rowLNVecFlat (s := s) M γ β (R.lnF M c e) hc hγ hβ
      (R.bridge M P c Xh hc hXh0 hcXh e he)).Maps Ā Ē Ā' Ē' :=
  Maps.rowLNVecFlat M γ β (R.lnF M c e) hc hγ hβ (R.bridge M P c Xh hc hXh0 hcXh e he)
    P.hq P.hgl P.hbb (R.maps M P c Xh hc hXh0 hcXh e he nA nE) gA gE bA bE

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
    (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) wts.sε Eps.hs)
    (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) wts.hε Eps.hh)
    (fun i => R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) (wts.s1 i).εn (Eps.h1 i))
    (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) wts.d1.ε Eps.hd1)
    (fun i => R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) (wts.s2 i).εn (Eps.h2 i))
    (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) wts.d2.ε Eps.hd2)
    (fun i => R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) (wts.s3 i).εn (Eps.h3 i))
    (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) wts.d3.ε Eps.hd3)
    (fun i => R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) (wts.s4 i).εn (Eps.h4 i))


-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, measured on the finished 300-epoch ImageNet run
    (`/home/skoonce/convnext/convnext_t300_4gpu/convnext_tiny_imagenet.bin`, 28,587,592 f32):
    conv and dense kernels within `6/10` (global max `0.5962` over 28.5 M entries), biases and
    LayerNorm β within `3` (max `2.9499`), LayerNorm γ within `48/10` (max `4.7700`), the layer
    scale within `84/10` (max `8.3766`). ⭐ The four scales are 14× apart and the fold is
    exquisitely sensitive to the first of them — see the file header. `ε ≥ 10⁻⁵` (the trainer's
    value) puts every LayerNorm's inverse-stddev under `317`; the device `rsqrt` and the device
    GELU are taken accurate to `10⁻²` absolute, and ⭐ the device MEAN to **`4.590·10⁻⁵`
    relative — DERIVED, not supplied** (`deviceLN_emr_committed`, `FloatBudgetEnvLN.lean`): a
    rounded reduction of `c` terms then a divide is within `u·(1+γ)+γ` of the certified mean at
    the fan-in every summation order meets, which at this net's widest LayerNorm (`c = 768`) is
    that numeral. It was `10⁻²` by analogy with the `rsqrt` until 2026-09-05, and the change is
    worth **44 orders**. ⛔ Uniform at the widest rather than per width, which costs 2 more
    (4.871·10¹³⁰ against 1.727·10¹²⁸); priced and declined, `FloatBudgetEnvLN.lean`. -/
theorem cnxProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    CnxProfile M ε (6/10) 3 (48/10) (84/10) (1/100) (4590 / 10 ^ 8) (1/100) 317 u32 where
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
    error. That is why `9.738·10¹³⁰ / 4.871·10¹³⁰ = 2.00`. -/
theorem cnxBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (4590 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).Maps 1 0
      (4871 * 10 ^ 127) (9738 * 10 ^ 127) := by
  have P := cnxProfile_committed M hMu hε5
  have mStemC := FloatBridgesTo.Maps.flatConvStride4 (h := 56) (w := 56) M
    wts.sW wts.sb P.hw' P.hbb (by norm_num) B.sW B.sb
    (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā := 1) (Ē := 0)
    (Ā' := 3181 / 10 ^ 2) (Ē' := 9480 / 10 ^ 8) (by norm_num [u32]) (by norm_num [u32])
  have mStem := mStemC.comp (by norm_num)
    (cnxChanLNMaps M R P 10 wts.sγ wts.sβ B.sγ B.sβ Eps.hs (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)
      (Ā := 3181 / 10 ^ 2) (Ē := 9480 / 10 ^ 8)
      (A1 := 1111 / 10 ^ 2) (E1 := 2222 / 10 ^ 2) (A2 := 5333 / 10 ^ 2) (E2 := 1067 / 10 ^ 1)
      (Ā' := 5634 / 10 ^ 2) (Ē' := 1068 / 10 ^ 1)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have mS1 := mStem.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 0) (R.lnF M 96 ((wts.s1 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s1 0) (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 0).εn) (Eps.h1 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 0).εn) (Eps.h1 0)
        (Ā := 1660) (Ā' := 6742 / 10 ^ 2) (Ē' := 1349 / 10 ^ 1) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 1660) (E1 := 3140)
      (A2 := 6742 / 10 ^ 2) (E2 := 1349 / 10 ^ 1)
      (A3 := 3237 / 10 ^ 1) (E3 := 6476 / 10 ^ 1)
      (A4 := 3268 / 10 ^ 1) (E4 := 6477 / 10 ^ 1)
      (A5 := 1883 * 10 ^ 1) (E5 := 3731 * 10 ^ 1)
      (A6 := 1884 * 10 ^ 1) (E6 := 5597 * 10 ^ 1)
      (A7 := 4341 * 10 ^ 3) (E7 := 1290 * 10 ^ 4)
      (A8 := 3647 * 10 ^ 4) (E8 := 1084 * 10 ^ 5)
      (Ā' := 3648 * 10 ^ 4) (Ē' := 1085 * 10 ^ 5)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 1) (R.lnF M 96 ((wts.s1 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s1 1) (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 1).εn) (Eps.h1 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 1).εn) (Eps.h1 1)
          (Ā := 1073 * 10 ^ 6) (Ā' := 3712 * 10 ^ 4) (Ē' := 7424 * 10 ^ 4) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 1073 * 10 ^ 6) (E1 := 3190 * 10 ^ 6)
        (A2 := 3712 * 10 ^ 4) (E2 := 7424 * 10 ^ 4)
        (A3 := 1782 * 10 ^ 5) (E3 := 3564 * 10 ^ 5)
        (A4 := 1783 * 10 ^ 5) (E4 := 3565 * 10 ^ 5)
        (A5 := 1028 * 10 ^ 7) (E5 := 2054 * 10 ^ 7)
        (A6 := 1029 * 10 ^ 7) (E6 := 3082 * 10 ^ 7)
        (A7 := 2371 * 10 ^ 9) (E7 := 7102 * 10 ^ 9)
        (A8 := 1992 * 10 ^ 10) (E8 := 5966 * 10 ^ 10)
        (Ā' := 1993 * 10 ^ 10) (Ē' := 5967 * 10 ^ 10)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 2) (R.lnF M 96 ((wts.s1 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s1 2) (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 2).εn) (Eps.h1 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 2).εn) (Eps.h1 2)
            (Ā := 5860 * 10 ^ 11) (Ā' := 2027 * 10 ^ 10) (Ē' := 4054 * 10 ^ 10) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 5860 * 10 ^ 11) (E1 := 1755 * 10 ^ 12)
          (A2 := 2027 * 10 ^ 10) (E2 := 4054 * 10 ^ 10)
          (A3 := 9730 * 10 ^ 10) (E3 := 1946 * 10 ^ 11)
          (A4 := 9731 * 10 ^ 10) (E4 := 1947 * 10 ^ 11)
          (A5 := 5606 * 10 ^ 12) (E5 := 1122 * 10 ^ 13)
          (A6 := 5607 * 10 ^ 12) (E6 := 1684 * 10 ^ 13)
          (A7 := 1292 * 10 ^ 15) (E7 := 3881 * 10 ^ 15)
          (A8 := 1086 * 10 ^ 16) (E8 := 3261 * 10 ^ 16)
          (Ā' := 1087 * 10 ^ 16) (Ē' := 3262 * 10 ^ 16)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mD1 := mS1.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 28 28 M wts.d1 (R.lnF M 96 wts.d1.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d1
      (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) wts.d1.ε Eps.hd1) P.hq P.hgl P.hbb
      (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) wts.d1.ε Eps.hd1
        (Ā := 1087 * 10 ^ 16) (Ā' := 3760 * 10 ^ 14) (Ē' := 7520 * 10 ^ 14) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 3760 * 10 ^ 14) (E1 := 7520 * 10 ^ 14) (A2 := 1805 * 10 ^ 15) (E2 := 3610 * 10 ^ 15)
      (A3 := 1806 * 10 ^ 15) (E3 := 3611 * 10 ^ 15) (Ā' := 4162 * 10 ^ 17) (Ē' := 8321 * 10 ^ 17)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS2 := mD1.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 0) (R.lnF M 192 ((wts.s2 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s2 0) (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 0).εn) (Eps.h2 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 0).εn) (Eps.h2 0)
        (Ā := 1224 * 10 ^ 19) (Ā' := 4234 * 10 ^ 17) (Ē' := 8468 * 10 ^ 17) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 1224 * 10 ^ 19) (E1 := 2447 * 10 ^ 19)
      (A2 := 4234 * 10 ^ 17) (E2 := 8468 * 10 ^ 17)
      (A3 := 2033 * 10 ^ 18) (E3 := 4065 * 10 ^ 18)
      (A4 := 2034 * 10 ^ 18) (E4 := 4066 * 10 ^ 18)
      (A5 := 2344 * 10 ^ 20) (E5 := 4685 * 10 ^ 20)
      (A6 := 2345 * 10 ^ 20) (E6 := 7028 * 10 ^ 20)
      (A7 := 1081 * 10 ^ 23) (E7 := 3239 * 10 ^ 23)
      (A8 := 9081 * 10 ^ 23) (E8 := 2721 * 10 ^ 24)
      (Ā' := 9082 * 10 ^ 23) (Ē' := 2722 * 10 ^ 24)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 1) (R.lnF M 192 ((wts.s2 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s2 1) (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 1).εn) (Eps.h2 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 1).εn) (Eps.h2 1)
          (Ā := 2671 * 10 ^ 25) (Ā' := 9239 * 10 ^ 23) (Ē' := 1848 * 10 ^ 24) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 2671 * 10 ^ 25) (E1 := 8003 * 10 ^ 25)
        (A2 := 9239 * 10 ^ 23) (E2 := 1848 * 10 ^ 24)
        (A3 := 4435 * 10 ^ 24) (E3 := 8871 * 10 ^ 24)
        (A4 := 4436 * 10 ^ 24) (E4 := 8872 * 10 ^ 24)
        (A5 := 5111 * 10 ^ 26) (E5 := 1023 * 10 ^ 27)
        (A6 := 5112 * 10 ^ 26) (E6 := 1535 * 10 ^ 27)
        (A7 := 2356 * 10 ^ 29) (E7 := 7074 * 10 ^ 29)
        (A8 := 1980 * 10 ^ 30) (E8 := 5943 * 10 ^ 30)
        (Ā' := 1981 * 10 ^ 30) (Ē' := 5944 * 10 ^ 30)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 2) (R.lnF M 192 ((wts.s2 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s2 2) (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 2).εn) (Eps.h2 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 2).εn) (Eps.h2 2)
            (Ā := 5825 * 10 ^ 31) (Ā' := 2015 * 10 ^ 30) (Ē' := 4030 * 10 ^ 30) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 5825 * 10 ^ 31) (E1 := 1748 * 10 ^ 32)
          (A2 := 2015 * 10 ^ 30) (E2 := 4030 * 10 ^ 30)
          (A3 := 9673 * 10 ^ 30) (E3 := 1935 * 10 ^ 31)
          (A4 := 9674 * 10 ^ 30) (E4 := 1936 * 10 ^ 31)
          (A5 := 1115 * 10 ^ 33) (E5 := 2231 * 10 ^ 33)
          (A6 := 1116 * 10 ^ 33) (E6 := 3347 * 10 ^ 33)
          (A7 := 5143 * 10 ^ 35) (E7 := 1543 * 10 ^ 36)
          (A8 := 4321 * 10 ^ 36) (E8 := 1297 * 10 ^ 37)
          (Ā' := 4322 * 10 ^ 36) (Ē' := 1298 * 10 ^ 37)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mD2 := mS2.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 14 14 M wts.d2 (R.lnF M 192 wts.d2.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d2
      (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) wts.d2.ε Eps.hd2) P.hq P.hgl P.hbb
      (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) wts.d2.ε Eps.hd2
        (Ā := 4322 * 10 ^ 36) (Ā' := 1495 * 10 ^ 35) (Ē' := 2990 * 10 ^ 35) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 1495 * 10 ^ 35) (E1 := 2990 * 10 ^ 35) (A2 := 7177 * 10 ^ 35) (E2 := 1436 * 10 ^ 36)
      (A3 := 7178 * 10 ^ 35) (E3 := 1437 * 10 ^ 36) (Ā' := 3308 * 10 ^ 38) (Ē' := 6623 * 10 ^ 38)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS3 := mD2.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 0) (R.lnF M 384 ((wts.s3 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s3 0) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 0).εn) (Eps.h3 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 0).εn) (Eps.h3 0)
        (Ā := 9726 * 10 ^ 39) (Ā' := 3365 * 10 ^ 38) (Ē' := 6730 * 10 ^ 38) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 9726 * 10 ^ 39) (E1 := 1948 * 10 ^ 40)
      (A2 := 3365 * 10 ^ 38) (E2 := 6730 * 10 ^ 38)
      (A3 := 1616 * 10 ^ 39) (E3 := 3231 * 10 ^ 39)
      (A4 := 1617 * 10 ^ 39) (E4 := 3232 * 10 ^ 39)
      (A5 := 3726 * 10 ^ 41) (E5 := 7447 * 10 ^ 41)
      (A6 := 3727 * 10 ^ 41) (E6 := 1118 * 10 ^ 42)
      (A7 := 3436 * 10 ^ 44) (E7 := 1031 * 10 ^ 45)
      (A8 := 2887 * 10 ^ 45) (E8 := 8661 * 10 ^ 45)
      (Ā' := 2888 * 10 ^ 45) (Ē' := 8662 * 10 ^ 45)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 1) (R.lnF M 384 ((wts.s3 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s3 1) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 1).εn) (Eps.h3 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 1).εn) (Eps.h3 1)
          (Ā := 8491 * 10 ^ 46) (Ā' := 2937 * 10 ^ 45) (Ē' := 5874 * 10 ^ 45) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 8491 * 10 ^ 46) (E1 := 2547 * 10 ^ 47)
        (A2 := 2937 * 10 ^ 45) (E2 := 5874 * 10 ^ 45)
        (A3 := 1410 * 10 ^ 46) (E3 := 2820 * 10 ^ 46)
        (A4 := 1411 * 10 ^ 46) (E4 := 2821 * 10 ^ 46)
        (A5 := 3252 * 10 ^ 48) (E5 := 6500 * 10 ^ 48)
        (A6 := 3253 * 10 ^ 48) (E6 := 9751 * 10 ^ 48)
        (A7 := 2999 * 10 ^ 51) (E7 := 8988 * 10 ^ 51)
        (A8 := 2520 * 10 ^ 52) (E8 := 7550 * 10 ^ 52)
        (Ā' := 2521 * 10 ^ 52) (Ē' := 7551 * 10 ^ 52)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 2) (R.lnF M 384 ((wts.s3 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s3 2) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 2).εn) (Eps.h3 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 2).εn) (Eps.h3 2)
            (Ā := 7412 * 10 ^ 53) (Ā' := 2564 * 10 ^ 52) (Ē' := 5128 * 10 ^ 52) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 7412 * 10 ^ 53) (E1 := 2221 * 10 ^ 54)
          (A2 := 2564 * 10 ^ 52) (E2 := 5128 * 10 ^ 52)
          (A3 := 1231 * 10 ^ 53) (E3 := 2462 * 10 ^ 53)
          (A4 := 1232 * 10 ^ 53) (E4 := 2463 * 10 ^ 53)
          (A5 := 2839 * 10 ^ 55) (E5 := 5675 * 10 ^ 55)
          (A6 := 2840 * 10 ^ 55) (E6 := 8513 * 10 ^ 55)
          (A7 := 2618 * 10 ^ 58) (E7 := 7847 * 10 ^ 58)
          (A8 := 2200 * 10 ^ 59) (E8 := 6592 * 10 ^ 59)
          (Ā' := 2201 * 10 ^ 59) (Ē' := 6593 * 10 ^ 59)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 3) (R.lnF M 384 ((wts.s3 3).εn))
            P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
            (B.s3 3) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 3).εn) (Eps.h3 3))
            P.hq P.hgl P.hbb P.hsl
            (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 3).εn) (Eps.h3 3)
              (Ā := 6471 * 10 ^ 60) (Ā' := 2239 * 10 ^ 59) (Ē' := 4478 * 10 ^ 59) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
            (A1 := 6471 * 10 ^ 60) (E1 := 1939 * 10 ^ 61)
            (A2 := 2239 * 10 ^ 59) (E2 := 4478 * 10 ^ 59)
            (A3 := 1075 * 10 ^ 60) (E3 := 2150 * 10 ^ 60)
            (A4 := 1076 * 10 ^ 60) (E4 := 2151 * 10 ^ 60)
            (A5 := 2480 * 10 ^ 62) (E5 := 4957 * 10 ^ 62)
            (A6 := 2481 * 10 ^ 62) (E6 := 7436 * 10 ^ 62)
            (A7 := 2287 * 10 ^ 65) (E7 := 6854 * 10 ^ 65)
            (A8 := 1922 * 10 ^ 66) (E8 := 5758 * 10 ^ 66)
            (Ā' := 1923 * 10 ^ 66) (Ē' := 5759 * 10 ^ 66)
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
            (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
            ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 4) (R.lnF M 384 ((wts.s3 4).εn))
              P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
              (B.s3 4) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 4).εn) (Eps.h3 4))
              P.hq P.hgl P.hbb P.hsl
              (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 4).εn) (Eps.h3 4)
                (Ā := 5654 * 10 ^ 67) (Ā' := 1956 * 10 ^ 66) (Ē' := 3912 * 10 ^ 66) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
              (A1 := 5654 * 10 ^ 67) (E1 := 1694 * 10 ^ 68)
              (A2 := 1956 * 10 ^ 66) (E2 := 3912 * 10 ^ 66)
              (A3 := 9389 * 10 ^ 66) (E3 := 1878 * 10 ^ 67)
              (A4 := 9390 * 10 ^ 66) (E4 := 1879 * 10 ^ 67)
              (A5 := 2164 * 10 ^ 69) (E5 := 4330 * 10 ^ 69)
              (A6 := 2165 * 10 ^ 69) (E6 := 6496 * 10 ^ 69)
              (A7 := 1996 * 10 ^ 72) (E7 := 5988 * 10 ^ 72)
              (A8 := 1677 * 10 ^ 73) (E8 := 5030 * 10 ^ 73)
              (Ā' := 1678 * 10 ^ 73) (Ē' := 5031 * 10 ^ 73)
              (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
              (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
              (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
              ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 5) (R.lnF M 384 ((wts.s3 5).εn))
                P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                (B.s3 5) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 5).εn) (Eps.h3 5))
                P.hq P.hgl P.hbb P.hsl
                (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 5).εn) (Eps.h3 5)
                  (Ā := 4934 * 10 ^ 74) (Ā' := 1707 * 10 ^ 73) (Ē' := 3414 * 10 ^ 73) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                (A1 := 4934 * 10 ^ 74) (E1 := 1480 * 10 ^ 75)
                (A2 := 1707 * 10 ^ 73) (E2 := 3414 * 10 ^ 73)
                (A3 := 8194 * 10 ^ 73) (E3 := 1639 * 10 ^ 74)
                (A4 := 8195 * 10 ^ 73) (E4 := 1640 * 10 ^ 74)
                (A5 := 1889 * 10 ^ 76) (E5 := 3779 * 10 ^ 76)
                (A6 := 1890 * 10 ^ 76) (E6 := 5669 * 10 ^ 76)
                (A7 := 1742 * 10 ^ 79) (E7 := 5226 * 10 ^ 79)
                (A8 := 1464 * 10 ^ 80) (E8 := 4390 * 10 ^ 80)
                (Ā' := 1465 * 10 ^ 80) (Ē' := 4391 * 10 ^ 80)
                (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 6) (R.lnF M 384 ((wts.s3 6).εn))
                  P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                  (B.s3 6) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 6).εn) (Eps.h3 6))
                  P.hq P.hgl P.hbb P.hsl
                  (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 6).εn) (Eps.h3 6)
                    (Ā := 4308 * 10 ^ 81) (Ā' := 1491 * 10 ^ 80) (Ē' := 2982 * 10 ^ 80) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                  (A1 := 4308 * 10 ^ 81) (E1 := 1291 * 10 ^ 82)
                  (A2 := 1491 * 10 ^ 80) (E2 := 2982 * 10 ^ 80)
                  (A3 := 7157 * 10 ^ 80) (E3 := 1432 * 10 ^ 81)
                  (A4 := 7158 * 10 ^ 80) (E4 := 1433 * 10 ^ 81)
                  (A5 := 1650 * 10 ^ 83) (E5 := 3302 * 10 ^ 83)
                  (A6 := 1651 * 10 ^ 83) (E6 := 4954 * 10 ^ 83)
                  (A7 := 1522 * 10 ^ 86) (E7 := 4567 * 10 ^ 86)
                  (A8 := 1279 * 10 ^ 87) (E8 := 3837 * 10 ^ 87)
                  (Ā' := 1280 * 10 ^ 87) (Ē' := 3838 * 10 ^ 87)
                  (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                  (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                  (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                  ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 7) (R.lnF M 384 ((wts.s3 7).εn))
                    P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                    (B.s3 7) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 7).εn) (Eps.h3 7))
                    P.hq P.hgl P.hbb P.hsl
                    (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 7).εn) (Eps.h3 7)
                      (Ā := 3764 * 10 ^ 88) (Ā' := 1302 * 10 ^ 87) (Ē' := 2604 * 10 ^ 87) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                    (A1 := 3764 * 10 ^ 88) (E1 := 1129 * 10 ^ 89)
                    (A2 := 1302 * 10 ^ 87) (E2 := 2604 * 10 ^ 87)
                    (A3 := 6250 * 10 ^ 87) (E3 := 1250 * 10 ^ 88)
                    (A4 := 6251 * 10 ^ 87) (E4 := 1251 * 10 ^ 88)
                    (A5 := 1441 * 10 ^ 90) (E5 := 2883 * 10 ^ 90)
                    (A6 := 1442 * 10 ^ 90) (E6 := 4325 * 10 ^ 90)
                    (A7 := 1330 * 10 ^ 93) (E7 := 3987 * 10 ^ 93)
                    (A8 := 1118 * 10 ^ 94) (E8 := 3350 * 10 ^ 94)
                    (Ā' := 1119 * 10 ^ 94) (Ē' := 3351 * 10 ^ 94)
                    (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                    (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                    (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 8) (R.lnF M 384 ((wts.s3 8).εn))
                      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                      (B.s3 8) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 8).εn) (Eps.h3 8))
                      P.hq P.hgl P.hbb P.hsl
                      (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 8).εn) (Eps.h3 8)
                        (Ā := 3290 * 10 ^ 95) (Ā' := 1138 * 10 ^ 94) (Ē' := 2276 * 10 ^ 94) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                      (A1 := 3290 * 10 ^ 95) (E1 := 9852 * 10 ^ 95)
                      (A2 := 1138 * 10 ^ 94) (E2 := 2276 * 10 ^ 94)
                      (A3 := 5463 * 10 ^ 94) (E3 := 1093 * 10 ^ 95)
                      (A4 := 5464 * 10 ^ 94) (E4 := 1094 * 10 ^ 95)
                      (A5 := 1259 * 10 ^ 97) (E5 := 2521 * 10 ^ 97)
                      (A6 := 1260 * 10 ^ 97) (E6 := 3782 * 10 ^ 97)
                      (A7 := 1162 * 10 ^ 100) (E7 := 3486 * 10 ^ 100)
                      (A8 := 9761 * 10 ^ 100) (E8 := 2929 * 10 ^ 101)
                      (Ā' := 9762 * 10 ^ 100) (Ē' := 2930 * 10 ^ 101)
                      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                      (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                      FloatBridgesTo.Maps.idVec)))))))))
  have mD3 := mS3.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 7 7 M wts.d3 (R.lnF M 384 wts.d3.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d3
      (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) wts.d3.ε Eps.hd3) P.hq P.hgl P.hbb
      (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) wts.d3.ε Eps.hd3
        (Ā := 9762 * 10 ^ 100) (Ā' := 3377 * 10 ^ 99) (Ē' := 6754 * 10 ^ 99) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 3377 * 10 ^ 99) (E1 := 6754 * 10 ^ 99) (A2 := 1621 * 10 ^ 100) (E2 := 3242 * 10 ^ 100)
      (A3 := 1622 * 10 ^ 100) (E3 := 3243 * 10 ^ 100) (Ā' := 1495 * 10 ^ 103) (Ē' := 2990 * 10 ^ 103)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS4 := mD3.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 0) (R.lnF M 768 ((wts.s4 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s4 0) (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 0).εn) (Eps.h4 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 0).εn) (Eps.h4 0)
        (Ā := 4396 * 10 ^ 104) (Ā' := 1521 * 10 ^ 103) (Ē' := 3042 * 10 ^ 103) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 4396 * 10 ^ 104) (E1 := 8791 * 10 ^ 104)
      (A2 := 1521 * 10 ^ 103) (E2 := 3042 * 10 ^ 103)
      (A3 := 7301 * 10 ^ 103) (E3 := 1461 * 10 ^ 104)
      (A4 := 7302 * 10 ^ 103) (E4 := 1462 * 10 ^ 104)
      (A5 := 3365 * 10 ^ 106) (E5 := 6738 * 10 ^ 106)
      (A6 := 3366 * 10 ^ 106) (E6 := 1011 * 10 ^ 107)
      (A7 := 6206 * 10 ^ 109) (E7 := 1864 * 10 ^ 110)
      (A8 := 5214 * 10 ^ 110) (E8 := 1566 * 10 ^ 111)
      (Ā' := 5215 * 10 ^ 110) (Ē' := 1567 * 10 ^ 111)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 1) (R.lnF M 768 ((wts.s4 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s4 1) (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 1).εn) (Eps.h4 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 1).εn) (Eps.h4 1)
          (Ā := 1534 * 10 ^ 112) (Ā' := 5306 * 10 ^ 110) (Ē' := 1062 * 10 ^ 111) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 1534 * 10 ^ 112) (E1 := 4607 * 10 ^ 112)
        (A2 := 5306 * 10 ^ 110) (E2 := 1062 * 10 ^ 111)
        (A3 := 2547 * 10 ^ 111) (E3 := 5098 * 10 ^ 111)
        (A4 := 2548 * 10 ^ 111) (E4 := 5099 * 10 ^ 111)
        (A5 := 1175 * 10 ^ 114) (E5 := 2350 * 10 ^ 114)
        (A6 := 1176 * 10 ^ 114) (E6 := 3526 * 10 ^ 114)
        (A7 := 2169 * 10 ^ 117) (E7 := 6501 * 10 ^ 117)
        (A8 := 1822 * 10 ^ 118) (E8 := 5461 * 10 ^ 118)
        (Ā' := 1823 * 10 ^ 118) (Ē' := 5462 * 10 ^ 118)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 2) (R.lnF M 768 ((wts.s4 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s4 2) (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 2).εn) (Eps.h4 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 2).εn) (Eps.h4 2)
            (Ā := 5360 * 10 ^ 119) (Ā' := 1854 * 10 ^ 118) (Ē' := 3708 * 10 ^ 118) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 5360 * 10 ^ 119) (E1 := 1606 * 10 ^ 120)
          (A2 := 1854 * 10 ^ 118) (E2 := 3708 * 10 ^ 118)
          (A3 := 8900 * 10 ^ 118) (E3 := 1780 * 10 ^ 119)
          (A4 := 8901 * 10 ^ 118) (E4 := 1781 * 10 ^ 119)
          (A5 := 4102 * 10 ^ 121) (E5 := 8208 * 10 ^ 121)
          (A6 := 4103 * 10 ^ 121) (E6 := 1232 * 10 ^ 122)
          (A7 := 7565 * 10 ^ 124) (E7 := 2272 * 10 ^ 125)
          (A8 := 6355 * 10 ^ 125) (E8 := 1909 * 10 ^ 126)
          (Ā' := 6356 * 10 ^ 125) (Ē' := 1910 * 10 ^ 126)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mGAP := mS4.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 768) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      P.hq (by norm_num [u32]) (by norm_num [u32]) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 6357 * 10 ^ 125) (Ē' := 1911 * 10 ^ 126) (by norm_num [u32]) (by norm_num [u32]))
  have mHead := mGAP.comp (by norm_num)
    (cnxRowLNMaps (s := 1) M R P 28 wts.hγ wts.hβ B.hγ B.hβ Eps.hh (by norm_num)
      (by norm_num) (by norm_num)
      (Ā := 6357 * 10 ^ 125) (Ē := 1911 * 10 ^ 126)
      (A1 := 2199 * 10 ^ 124) (E1 := 4398 * 10 ^ 124) (A2 := 1056 * 10 ^ 125) (E2 := 2112 * 10 ^ 125)
      (Ā' := 1057 * 10 ^ 125) (Ē' := 2113 * 10 ^ 125)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  exact mHead.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M wts.Wd wts.bd P.hw' P.hbb (by norm_num) B.Wd B.bd
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 4871 * 10 ^ 127) (Ē' := 9738 * 10 ^ 127) (by norm_num [u32]) (by norm_num [u32]))

/-- The deployed ConvNeXt-T bridge's certified output window at the committed profile. -/
theorem cnxBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (4590 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).mag 1 ≤ 4871 * 10 ^ 127 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ConvNeXt-T bridge's fresh budget at the committed profile — `2.00 ×` the
    certified window, which is the tell that the cap is biting at every LayerNorm site and the
    statement is the triangle inequality rather than the fold. -/
theorem cnxBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (4590 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).fresh 1 ≤ 9738 * 10 ^ 127 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐ **The deployed ConvNeXt-T forward is within `9.738·10¹³⁰` of the certified real forward,
    per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile, for `ε ≥ 10⁻⁵`,
    any device LayerNorm statistics accurate to `10⁻²` and any device GELU accurate to `10⁻²`,
    for any rounding model at binary32 accuracy.

    ⛔ **Read the file header before quoting this.** It is a capped statement — the float and
    real forwards both land in the certified `4.871·10¹³⁰` window — and NOT the interval fold
    that ResNet-34's, MobileNetV2's and EfficientNet-B0's numbers are. LayerNorm has no
    frozen-statistics mode, so there is no version of this net for which the fold exists. -/
theorem cnx_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (4590 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε)
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cnxForwardF M R G wts x j - cnxForward wts x j| ≤ 9738 * 10 ^ 127 :=
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
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceLN (4590 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100))
    (wts : CnxTWeightsCh) (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε)
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cnxForwardF M R G wts x j - convNextForwardTCh wts x j| ≤ 9738 * 10 ^ 127 := by
  have h := cnx_float_logits_le M hMu hε5 R G wts B Eps x hx j
  rwa [cnxForward_eq_committed wts] at h

/-! ### Inhabitation

`cnx_float_logits_le`'s hypotheses at the committed constants: the all-zero `CnxTWeightsCh` with
every LayerNorm `ε` field at `1/100000` (so `CnxEps` is `le_rfl` at every site), the exact device
mean, inverse-stddev and GELU, `binary32`. -/
noncomputable def CnxBlockParamsCh.zero (c cExp h w kH kW : Nat) (ε : ℝ) :
    CnxBlockParamsCh c cExp h w kH kW where
  Wdw := fun _ _ _ => 0
  bdw := fun _ => 0
  εn := ε
  γn := fun _ => 0
  βn := fun _ => 0
  Wex := fun _ _ _ _ => 0
  bex := fun _ => 0
  Wpr := fun _ _ _ _ => 0
  bpr := fun _ => 0
  γls := fun _ => 0

noncomputable def CnxDownParamsCh.zero (cin cout : Nat) (ε : ℝ) : CnxDownParamsCh cin cout where
  ε := ε
  γ := fun _ => 0
  β := fun _ => 0
  W := fun _ _ _ _ => 0
  b := fun _ => 0

noncomputable def CnxTWeightsCh.zero (ε : ℝ) : CnxTWeightsCh where
  sW := fun _ _ _ _ => 0
  sb := fun _ => 0
  sε := ε
  sγ := fun _ => 0
  sβ := fun _ => 0
  s1 := fun _ => CnxBlockParamsCh.zero _ _ _ _ _ _ ε
  d1 := CnxDownParamsCh.zero _ _ ε
  s2 := fun _ => CnxBlockParamsCh.zero _ _ _ _ _ _ ε
  d2 := CnxDownParamsCh.zero _ _ ε
  s3 := fun _ => CnxBlockParamsCh.zero _ _ _ _ _ _ ε
  d3 := CnxDownParamsCh.zero _ _ ε
  s4 := fun _ => CnxBlockParamsCh.zero _ _ _ _ _ _ ε
  hε := ε
  hγ := fun _ => 0
  hβ := fun _ => 0
  Wd := fun _ _ => 0
  bd := fun _ => 0

theorem CnxBlockParamsCh.zero_bounded {c cExp h w kH kW : Nat} {ε w' bb gl sl : ℝ}
    (hw : 0 ≤ w') (hbb : 0 ≤ bb) (hgl : 0 ≤ gl) (hsl : 0 ≤ sl) :
    CnxBlockChBounded (CnxBlockParamsCh.zero c cExp h w kH kW ε) w' bb gl sl :=
  ⟨fun _ _ _ => by simpa [CnxBlockParamsCh.zero] using hw,
   fun _ => by simpa [CnxBlockParamsCh.zero] using hbb,
   fun _ _ _ _ => by simpa [CnxBlockParamsCh.zero] using hw,
   fun _ => by simpa [CnxBlockParamsCh.zero] using hbb,
   fun _ _ _ _ => by simpa [CnxBlockParamsCh.zero] using hw,
   fun _ => by simpa [CnxBlockParamsCh.zero] using hbb,
   fun _ => by simpa [CnxBlockParamsCh.zero] using hsl,
   fun _ => by simpa [CnxBlockParamsCh.zero] using hgl,
   fun _ => by simpa [CnxBlockParamsCh.zero] using hbb⟩

theorem CnxDownParamsCh.zero_bounded {cin cout : Nat} {ε w' bb gl : ℝ}
    (hw : 0 ≤ w') (hbb : 0 ≤ bb) (hgl : 0 ≤ gl) :
    CnxDownChBounded (CnxDownParamsCh.zero cin cout ε) w' bb gl :=
  ⟨fun _ _ _ _ => by simpa [CnxDownParamsCh.zero] using hw,
   fun _ => by simpa [CnxDownParamsCh.zero] using hbb,
   fun _ => by simpa [CnxDownParamsCh.zero] using hgl,
   fun _ => by simpa [CnxDownParamsCh.zero] using hbb⟩

theorem CnxTWeightsCh.zero_bounded (ε : ℝ) {w' bb gl sl : ℝ}
    (hw : 0 ≤ w') (hbb : 0 ≤ bb) (hgl : 0 ≤ gl) (hsl : 0 ≤ sl) :
    CnxBounded (CnxTWeightsCh.zero ε) w' bb gl sl where
  sW := fun _ _ _ _ => by simpa [CnxTWeightsCh.zero] using hw
  sb := fun _ => by simpa [CnxTWeightsCh.zero] using hbb
  sγ := fun _ => by simpa [CnxTWeightsCh.zero] using hgl
  sβ := fun _ => by simpa [CnxTWeightsCh.zero] using hbb
  s1 := fun _ => CnxBlockParamsCh.zero_bounded hw hbb hgl hsl
  d1 := CnxDownParamsCh.zero_bounded hw hbb hgl
  s2 := fun _ => CnxBlockParamsCh.zero_bounded hw hbb hgl hsl
  d2 := CnxDownParamsCh.zero_bounded hw hbb hgl
  s3 := fun _ => CnxBlockParamsCh.zero_bounded hw hbb hgl hsl
  d3 := CnxDownParamsCh.zero_bounded hw hbb hgl
  s4 := fun _ => CnxBlockParamsCh.zero_bounded hw hbb hgl hsl
  hγ := fun _ => by simpa [CnxTWeightsCh.zero] using hgl
  hβ := fun _ => by simpa [CnxTWeightsCh.zero] using hbb
  Wd := fun _ _ => by simpa [CnxTWeightsCh.zero] using hw
  bd := fun _ => by simpa [CnxTWeightsCh.zero] using hbb

theorem CnxTWeightsCh.zero_eps (ε : ℝ) : CnxEps (CnxTWeightsCh.zero ε) ε where
  hs := le_rfl
  h1 := fun _ => le_rfl
  hd1 := le_rfl
  h2 := fun _ => le_rfl
  hd2 := le_rfl
  h3 := fun _ => le_rfl
  hd3 := le_rfl
  h4 := fun _ => le_rfl
  hh := le_rfl

example (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cnxForwardF binary32 (DeviceLN.exact (emr := 4590 / 10 ^ 8) (ei := 1/100) (by norm_num) (by norm_num))
        (DeviceGelu.exact (egelu := 1/100) (by norm_num)) (CnxTWeightsCh.zero (1/100000)) x j
      - cnxForward (CnxTWeightsCh.zero (1/100000)) x j| ≤ 9738 * 10 ^ 127 :=
  cnx_float_logits_le binary32 binary32_u.le (by norm_num) _ _ _
    (CnxTWeightsCh.zero_bounded _ (by norm_num) (by norm_num) (by norm_num) (by norm_num))
    (CnxTWeightsCh.zero_eps _) x hx j

end Proofs
