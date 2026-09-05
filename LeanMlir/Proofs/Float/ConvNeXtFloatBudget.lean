import LeanMlir.Proofs.Float.FloatBudgetEnvLN
import LeanMlir.Proofs.Foundation.WholeNetForwardTies

/-! # A NUMBER for ConvNeXt-T: the committed channel-LayerNorm forward, at the cap

The fourth ImageNet-scale whole-net float statement, and ⛔ **it is not the same kind of
statement as the other three.** For the `[3,3,9,3]` ConvNeXt-T forward at `224²` — 4×4/s4
patchify stem, 18 blocks of `depthwise 7×7 → channel-LN → 1×1 expand → GELU → 1×1 project →
layer scale → skip`, three LN+2×2/s2 downsamples, GAP, head LayerNorm, classifier — on the unit
input window, at the profile measured on the finished 300-epoch ImageNet checkpoint, for any
rounding model at binary32 accuracy:

    output window  ≤ 6.609·10¹⁷⁴      (`cnxBridge_mag_le`)
    fresh budget   ≤ 1.321·10¹⁷⁵      (`cnxBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 1.321·10¹⁷⁵` (`cnx_float_logits_le`).

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
    error. That is why `1.321·10¹⁷⁵ / 6.609·10¹⁷⁴ = 2.00`. -/
theorem cnxBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).Maps 1 0
      (6609 * 10 ^ 171) (1321 * 10 ^ 172) := by
  have P := cnxProfile_committed M hMu hε5
  have mStemC := FloatBridgesTo.Maps.flatConvStride4 (h := 56) (w := 56) M
    wts.sW wts.sb P.hw' P.hbb (by norm_num) B.sW B.sb
    (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā := 1) (Ē := 0)
    (Ā' := 3181 / 10 ^ 2) (Ē' := 9480 / 10 ^ 8) (by norm_num [u32]) (by norm_num [u32])
  have mStem := mStemC.comp (by norm_num)
    (cnxChanLNMaps M R P 10 wts.sγ wts.sβ B.sγ B.sβ Eps.hs (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)
      (Ā := 3181 / 10 ^ 2) (Ē := 9480 / 10 ^ 8)
      (A1 := 1115 / 10 ^ 1) (E1 := 223) (A2 := 5353 / 10 ^ 1) (E2 := 1071)
      (Ā' := 5384 / 10 ^ 1) (Ē' := 1072)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  have mS1 := mStem.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 0) (R.lnF M 96 ((wts.s1 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s1 0) (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 0).εn) (Eps.h1 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 0).εn) (Eps.h1 0)
        (Ā := 1584 * 10 ^ 1) (Ā' := 5055 * 10 ^ 1) (Ē' := 1011 * 10 ^ 2) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 1584 * 10 ^ 1) (E1 := 3152 * 10 ^ 1)
      (A2 := 5055 * 10 ^ 1) (E2 := 1011 * 10 ^ 2)
      (A3 := 2427 * 10 ^ 2) (E3 := 4853 * 10 ^ 2)
      (A4 := 2428 * 10 ^ 2) (E4 := 4854 * 10 ^ 2)
      (A5 := 1399 * 10 ^ 4) (E5 := 2796 * 10 ^ 4)
      (A6 := 1400 * 10 ^ 4) (E6 := 4195 * 10 ^ 4)
      (A7 := 3226 * 10 ^ 6) (E7 := 9666 * 10 ^ 6)
      (A8 := 2710 * 10 ^ 7) (E8 := 8120 * 10 ^ 7)
      (Ā' := 2711 * 10 ^ 7) (Ē' := 8121 * 10 ^ 7)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 1) (R.lnF M 96 ((wts.s1 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s1 1) (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 1).εn) (Eps.h1 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 1).εn) (Eps.h1 1)
          (Ā := 7971 * 10 ^ 8) (Ā' := 2543 * 10 ^ 9) (Ē' := 5086 * 10 ^ 9) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 7971 * 10 ^ 8) (E1 := 2388 * 10 ^ 9)
        (A2 := 2543 * 10 ^ 9) (E2 := 5086 * 10 ^ 9)
        (A3 := 1221 * 10 ^ 10) (E3 := 2442 * 10 ^ 10)
        (A4 := 1222 * 10 ^ 10) (E4 := 2443 * 10 ^ 10)
        (A5 := 7039 * 10 ^ 11) (E5 := 1408 * 10 ^ 12)
        (A6 := 7040 * 10 ^ 11) (E6 := 2113 * 10 ^ 12)
        (A7 := 1623 * 10 ^ 14) (E7 := 4869 * 10 ^ 14)
        (A8 := 1364 * 10 ^ 15) (E8 := 4090 * 10 ^ 15)
        (Ā' := 1365 * 10 ^ 15) (Ē' := 4091 * 10 ^ 15)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s1 2) (R.lnF M 96 ((wts.s1 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s1 2) (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 2).εn) (Eps.h1 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) ((wts.s1 2).εn) (Eps.h1 2)
            (Ā := 4014 * 10 ^ 16) (Ā' := 1281 * 10 ^ 17) (Ē' := 2562 * 10 ^ 17) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 4014 * 10 ^ 16) (E1 := 1203 * 10 ^ 17)
          (A2 := 1281 * 10 ^ 17) (E2 := 2562 * 10 ^ 17)
          (A3 := 6149 * 10 ^ 17) (E3 := 1230 * 10 ^ 18)
          (A4 := 6150 * 10 ^ 17) (E4 := 1231 * 10 ^ 18)
          (A5 := 3543 * 10 ^ 19) (E5 := 7091 * 10 ^ 19)
          (A6 := 3544 * 10 ^ 19) (E6 := 1064 * 10 ^ 20)
          (A7 := 8166 * 10 ^ 21) (E7 := 2452 * 10 ^ 22)
          (A8 := 6860 * 10 ^ 22) (E8 := 2060 * 10 ^ 23)
          (Ā' := 6861 * 10 ^ 22) (Ē' := 2061 * 10 ^ 23)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mD1 := mS1.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 28 28 M wts.d1 (R.lnF M 96 wts.d1.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d1
      (R.bridge M P 96 10 (by norm_num) (by norm_num) (by norm_num) wts.d1.ε Eps.hd1) P.hq P.hgl P.hbb
      (R.maps M P 96 10 (by norm_num) (by norm_num) (by norm_num) wts.d1.ε Eps.hd1
        (Ā := 6861 * 10 ^ 22) (Ā' := 2189 * 10 ^ 23) (Ē' := 4378 * 10 ^ 23) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 2189 * 10 ^ 23) (E1 := 4378 * 10 ^ 23) (A2 := 1051 * 10 ^ 24) (E2 := 2102 * 10 ^ 24)
      (A3 := 1052 * 10 ^ 24) (E3 := 2103 * 10 ^ 24) (Ā' := 2424 * 10 ^ 26) (Ē' := 4846 * 10 ^ 26)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS2 := mD1.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 0) (R.lnF M 192 ((wts.s2 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s2 0) (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 0).εn) (Eps.h2 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 0).εn) (Eps.h2 0)
        (Ā := 7127 * 10 ^ 27) (Ā' := 2274 * 10 ^ 28) (Ē' := 4548 * 10 ^ 28) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 7127 * 10 ^ 27) (E1 := 1425 * 10 ^ 28)
      (A2 := 2274 * 10 ^ 28) (E2 := 4548 * 10 ^ 28)
      (A3 := 1092 * 10 ^ 29) (E3 := 2184 * 10 ^ 29)
      (A4 := 1093 * 10 ^ 29) (E4 := 2185 * 10 ^ 29)
      (A5 := 1260 * 10 ^ 31) (E5 := 2518 * 10 ^ 31)
      (A6 := 1261 * 10 ^ 31) (E6 := 3778 * 10 ^ 31)
      (A7 := 5811 * 10 ^ 33) (E7 := 1742 * 10 ^ 34)
      (A8 := 4882 * 10 ^ 34) (E8 := 1464 * 10 ^ 35)
      (Ā' := 4883 * 10 ^ 34) (Ē' := 1465 * 10 ^ 35)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 1) (R.lnF M 192 ((wts.s2 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s2 1) (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 1).εn) (Eps.h2 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 1).εn) (Eps.h2 1)
          (Ā := 1436 * 10 ^ 36) (Ā' := 4582 * 10 ^ 36) (Ē' := 9164 * 10 ^ 36) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 1436 * 10 ^ 36) (E1 := 4308 * 10 ^ 36)
        (A2 := 4582 * 10 ^ 36) (E2 := 9164 * 10 ^ 36)
        (A3 := 2200 * 10 ^ 37) (E3 := 4399 * 10 ^ 37)
        (A4 := 2201 * 10 ^ 37) (E4 := 4400 * 10 ^ 37)
        (A5 := 2536 * 10 ^ 39) (E5 := 5069 * 10 ^ 39)
        (A6 := 2537 * 10 ^ 39) (E6 := 7604 * 10 ^ 39)
        (A7 := 1170 * 10 ^ 42) (E7 := 3505 * 10 ^ 42)
        (A8 := 9829 * 10 ^ 42) (E8 := 2945 * 10 ^ 43)
        (Ā' := 9830 * 10 ^ 42) (Ē' := 2946 * 10 ^ 43)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s2 2) (R.lnF M 192 ((wts.s2 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s2 2) (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 2).εn) (Eps.h2 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) ((wts.s2 2).εn) (Eps.h2 2)
            (Ā := 2891 * 10 ^ 44) (Ā' := 9223 * 10 ^ 44) (Ē' := 1845 * 10 ^ 45) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 2891 * 10 ^ 44) (E1 := 8662 * 10 ^ 44)
          (A2 := 9223 * 10 ^ 44) (E2 := 1845 * 10 ^ 45)
          (A3 := 4428 * 10 ^ 45) (E3 := 8857 * 10 ^ 45)
          (A4 := 4429 * 10 ^ 45) (E4 := 8858 * 10 ^ 45)
          (A5 := 5103 * 10 ^ 47) (E5 := 1021 * 10 ^ 48)
          (A6 := 5104 * 10 ^ 47) (E6 := 1532 * 10 ^ 48)
          (A7 := 2353 * 10 ^ 50) (E7 := 7060 * 10 ^ 50)
          (A8 := 1977 * 10 ^ 51) (E8 := 5931 * 10 ^ 51)
          (Ā' := 1978 * 10 ^ 51) (Ē' := 5932 * 10 ^ 51)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mD2 := mS2.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 14 14 M wts.d2 (R.lnF M 192 wts.d2.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d2
      (R.bridge M P 192 14 (by norm_num) (by norm_num) (by norm_num) wts.d2.ε Eps.hd2) P.hq P.hgl P.hbb
      (R.maps M P 192 14 (by norm_num) (by norm_num) (by norm_num) wts.d2.ε Eps.hd2
        (Ā := 1978 * 10 ^ 51) (Ā' := 6311 * 10 ^ 51) (Ē' := 1263 * 10 ^ 52) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 6311 * 10 ^ 51) (E1 := 1263 * 10 ^ 52) (A2 := 3030 * 10 ^ 52) (E2 := 6063 * 10 ^ 52)
      (A3 := 3031 * 10 ^ 52) (E3 := 6064 * 10 ^ 52) (Ā' := 1397 * 10 ^ 55) (Ē' := 2795 * 10 ^ 55)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS3 := mD2.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 0) (R.lnF M 384 ((wts.s3 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s3 0) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 0).εn) (Eps.h3 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 0).εn) (Eps.h3 0)
        (Ā := 4108 * 10 ^ 56) (Ā' := 1311 * 10 ^ 57) (Ē' := 2622 * 10 ^ 57) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 4108 * 10 ^ 56) (E1 := 8218 * 10 ^ 56)
      (A2 := 1311 * 10 ^ 57) (E2 := 2622 * 10 ^ 57)
      (A3 := 6293 * 10 ^ 57) (E3 := 1259 * 10 ^ 58)
      (A4 := 6294 * 10 ^ 57) (E4 := 1260 * 10 ^ 58)
      (A5 := 1451 * 10 ^ 60) (E5 := 2904 * 10 ^ 60)
      (A6 := 1452 * 10 ^ 60) (E6 := 4357 * 10 ^ 60)
      (A7 := 1339 * 10 ^ 63) (E7 := 4016 * 10 ^ 63)
      (A8 := 1125 * 10 ^ 64) (E8 := 3374 * 10 ^ 64)
      (Ā' := 1126 * 10 ^ 64) (Ē' := 3375 * 10 ^ 64)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 1) (R.lnF M 384 ((wts.s3 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s3 1) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 1).εn) (Eps.h3 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 1).εn) (Eps.h3 1)
          (Ā := 3311 * 10 ^ 65) (Ā' := 1057 * 10 ^ 66) (Ē' := 2114 * 10 ^ 66) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 3311 * 10 ^ 65) (E1 := 9923 * 10 ^ 65)
        (A2 := 1057 * 10 ^ 66) (E2 := 2114 * 10 ^ 66)
        (A3 := 5074 * 10 ^ 66) (E3 := 1015 * 10 ^ 67)
        (A4 := 5075 * 10 ^ 66) (E4 := 1016 * 10 ^ 67)
        (A5 := 1170 * 10 ^ 69) (E5 := 2341 * 10 ^ 69)
        (A6 := 1171 * 10 ^ 69) (E6 := 3512 * 10 ^ 69)
        (A7 := 1080 * 10 ^ 72) (E7 := 3238 * 10 ^ 72)
        (A8 := 9073 * 10 ^ 72) (E8 := 2720 * 10 ^ 73)
        (Ā' := 9074 * 10 ^ 72) (Ē' := 2721 * 10 ^ 73)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 2) (R.lnF M 384 ((wts.s3 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s3 2) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 2).εn) (Eps.h3 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 2).εn) (Eps.h3 2)
            (Ā := 2668 * 10 ^ 74) (Ā' := 8512 * 10 ^ 74) (Ē' := 1703 * 10 ^ 75) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 2668 * 10 ^ 74) (E1 := 8000 * 10 ^ 74)
          (A2 := 8512 * 10 ^ 74) (E2 := 1703 * 10 ^ 75)
          (A3 := 4086 * 10 ^ 75) (E3 := 8175 * 10 ^ 75)
          (A4 := 4087 * 10 ^ 75) (E4 := 8176 * 10 ^ 75)
          (A5 := 9417 * 10 ^ 77) (E5 := 1884 * 10 ^ 78)
          (A6 := 9418 * 10 ^ 77) (E6 := 2827 * 10 ^ 78)
          (A7 := 8681 * 10 ^ 80) (E7 := 2606 * 10 ^ 81)
          (A8 := 7293 * 10 ^ 81) (E8 := 2190 * 10 ^ 82)
          (Ā' := 7294 * 10 ^ 81) (Ē' := 2191 * 10 ^ 82)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 3) (R.lnF M 384 ((wts.s3 3).εn))
            P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
            (B.s3 3) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 3).εn) (Eps.h3 3))
            P.hq P.hgl P.hbb P.hsl
            (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 3).εn) (Eps.h3 3)
              (Ā := 2145 * 10 ^ 83) (Ā' := 6843 * 10 ^ 83) (Ē' := 1369 * 10 ^ 84) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
            (A1 := 2145 * 10 ^ 83) (E1 := 6442 * 10 ^ 83)
            (A2 := 6843 * 10 ^ 83) (E2 := 1369 * 10 ^ 84)
            (A3 := 3285 * 10 ^ 84) (E3 := 6572 * 10 ^ 84)
            (A4 := 3286 * 10 ^ 84) (E4 := 6573 * 10 ^ 84)
            (A5 := 7572 * 10 ^ 86) (E5 := 1515 * 10 ^ 87)
            (A6 := 7573 * 10 ^ 86) (E6 := 2273 * 10 ^ 87)
            (A7 := 6980 * 10 ^ 89) (E7 := 2096 * 10 ^ 90)
            (A8 := 5864 * 10 ^ 90) (E8 := 1761 * 10 ^ 91)
            (Ā' := 5865 * 10 ^ 90) (Ē' := 1762 * 10 ^ 91)
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
            (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
            ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 4) (R.lnF M 384 ((wts.s3 4).εn))
              P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
              (B.s3 4) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 4).εn) (Eps.h3 4))
              P.hq P.hgl P.hbb P.hsl
              (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 4).εn) (Eps.h3 4)
                (Ā := 1725 * 10 ^ 92) (Ā' := 5503 * 10 ^ 92) (Ē' := 1101 * 10 ^ 93) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
              (A1 := 1725 * 10 ^ 92) (E1 := 5181 * 10 ^ 92)
              (A2 := 5503 * 10 ^ 92) (E2 := 1101 * 10 ^ 93)
              (A3 := 2642 * 10 ^ 93) (E3 := 5285 * 10 ^ 93)
              (A4 := 2643 * 10 ^ 93) (E4 := 5286 * 10 ^ 93)
              (A5 := 6090 * 10 ^ 95) (E5 := 1218 * 10 ^ 96)
              (A6 := 6091 * 10 ^ 95) (E6 := 1828 * 10 ^ 96)
              (A7 := 5614 * 10 ^ 98) (E7 := 1685 * 10 ^ 99)
              (A8 := 4716 * 10 ^ 99) (E8 := 1416 * 10 ^ 100)
              (Ā' := 4717 * 10 ^ 99) (Ē' := 1417 * 10 ^ 100)
              (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
              (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
              (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
              ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 5) (R.lnF M 384 ((wts.s3 5).εn))
                P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                (B.s3 5) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 5).εn) (Eps.h3 5))
                P.hq P.hgl P.hbb P.hsl
                (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 5).εn) (Eps.h3 5)
                  (Ā := 1387 * 10 ^ 101) (Ā' := 4425 * 10 ^ 101) (Ē' := 8850 * 10 ^ 101) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                (A1 := 1387 * 10 ^ 101) (E1 := 4166 * 10 ^ 101)
                (A2 := 4425 * 10 ^ 101) (E2 := 8850 * 10 ^ 101)
                (A3 := 2125 * 10 ^ 102) (E3 := 4249 * 10 ^ 102)
                (A4 := 2126 * 10 ^ 102) (E4 := 4250 * 10 ^ 102)
                (A5 := 4899 * 10 ^ 104) (E5 := 9793 * 10 ^ 104)
                (A6 := 4900 * 10 ^ 104) (E6 := 1469 * 10 ^ 105)
                (A7 := 4517 * 10 ^ 107) (E7 := 1354 * 10 ^ 108)
                (A8 := 3795 * 10 ^ 108) (E8 := 1138 * 10 ^ 109)
                (Ā' := 3796 * 10 ^ 108) (Ē' := 1139 * 10 ^ 109)
                (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 6) (R.lnF M 384 ((wts.s3 6).εn))
                  P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                  (B.s3 6) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 6).εn) (Eps.h3 6))
                  P.hq P.hgl P.hbb P.hsl
                  (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 6).εn) (Eps.h3 6)
                    (Ā := 1117 * 10 ^ 110) (Ā' := 3564 * 10 ^ 110) (Ē' := 7128 * 10 ^ 110) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                  (A1 := 1117 * 10 ^ 110) (E1 := 3349 * 10 ^ 110)
                  (A2 := 3564 * 10 ^ 110) (E2 := 7128 * 10 ^ 110)
                  (A3 := 1711 * 10 ^ 111) (E3 := 3422 * 10 ^ 111)
                  (A4 := 1712 * 10 ^ 111) (E4 := 3423 * 10 ^ 111)
                  (A5 := 3945 * 10 ^ 113) (E5 := 7887 * 10 ^ 113)
                  (A6 := 3946 * 10 ^ 113) (E6 := 1184 * 10 ^ 114)
                  (A7 := 3637 * 10 ^ 116) (E7 := 1092 * 10 ^ 117)
                  (A8 := 3056 * 10 ^ 117) (E8 := 9173 * 10 ^ 117)
                  (Ā' := 3057 * 10 ^ 117) (Ē' := 9174 * 10 ^ 117)
                  (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                  (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                  (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                  ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 7) (R.lnF M 384 ((wts.s3 7).εn))
                    P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                    (B.s3 7) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 7).εn) (Eps.h3 7))
                    P.hq P.hgl P.hbb P.hsl
                    (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 7).εn) (Eps.h3 7)
                      (Ā := 8988 * 10 ^ 118) (Ā' := 2868 * 10 ^ 119) (Ē' := 5736 * 10 ^ 119) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                    (A1 := 8988 * 10 ^ 118) (E1 := 2698 * 10 ^ 119)
                    (A2 := 2868 * 10 ^ 119) (E2 := 5736 * 10 ^ 119)
                    (A3 := 1377 * 10 ^ 120) (E3 := 2754 * 10 ^ 120)
                    (A4 := 1378 * 10 ^ 120) (E4 := 2755 * 10 ^ 120)
                    (A5 := 3175 * 10 ^ 122) (E5 := 6348 * 10 ^ 122)
                    (A6 := 3176 * 10 ^ 122) (E6 := 9523 * 10 ^ 122)
                    (A7 := 2928 * 10 ^ 125) (E7 := 8778 * 10 ^ 125)
                    (A8 := 2460 * 10 ^ 126) (E8 := 7374 * 10 ^ 126)
                    (Ā' := 2461 * 10 ^ 126) (Ē' := 7375 * 10 ^ 126)
                    (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                    (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                    (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s3 8) (R.lnF M 384 ((wts.s3 8).εn))
                      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
                      (B.s3 8) (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 8).εn) (Eps.h3 8))
                      P.hq P.hgl P.hbb P.hsl
                      (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) ((wts.s3 8).εn) (Eps.h3 8)
                        (Ā := 7236 * 10 ^ 127) (Ā' := 2309 * 10 ^ 128) (Ē' := 4618 * 10 ^ 128) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
                      (A1 := 7236 * 10 ^ 127) (E1 := 2169 * 10 ^ 128)
                      (A2 := 2309 * 10 ^ 128) (E2 := 4618 * 10 ^ 128)
                      (A3 := 1109 * 10 ^ 129) (E3 := 2217 * 10 ^ 129)
                      (A4 := 1110 * 10 ^ 129) (E4 := 2218 * 10 ^ 129)
                      (A5 := 2558 * 10 ^ 131) (E5 := 5111 * 10 ^ 131)
                      (A6 := 2559 * 10 ^ 131) (E6 := 7667 * 10 ^ 131)
                      (A7 := 2359 * 10 ^ 134) (E7 := 7067 * 10 ^ 134)
                      (A8 := 1982 * 10 ^ 135) (E8 := 5937 * 10 ^ 135)
                      (Ā' := 1983 * 10 ^ 135) (Ē' := 5938 * 10 ^ 135)
                      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                      (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
                      (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
                      FloatBridgesTo.Maps.idVec)))))))))
  have mD3 := mS3.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownChW 7 7 M wts.d3 (R.lnF M 384 wts.d3.ε)
      P.hw' P.hbb (by norm_num) (by norm_num) (by norm_num) B.d3
      (R.bridge M P 384 20 (by norm_num) (by norm_num) (by norm_num) wts.d3.ε Eps.hd3) P.hq P.hgl P.hbb
      (R.maps M P 384 20 (by norm_num) (by norm_num) (by norm_num) wts.d3.ε Eps.hd3
        (Ā := 1983 * 10 ^ 135) (Ā' := 6327 * 10 ^ 135) (Ē' := 1266 * 10 ^ 136) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 6327 * 10 ^ 135) (E1 := 1266 * 10 ^ 136) (A2 := 3037 * 10 ^ 136) (E2 := 6077 * 10 ^ 136)
      (A3 := 3038 * 10 ^ 136) (E3 := 6078 * 10 ^ 136) (Ā' := 2801 * 10 ^ 139) (Ē' := 5603 * 10 ^ 139)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]))
  have mS4 := mD3.comp (by norm_num)
    ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 0) (R.lnF M 768 ((wts.s4 0).εn))
      P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
      (B.s4 0) (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 0).εn) (Eps.h4 0))
      P.hq P.hgl P.hbb P.hsl
      (R.maps M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 0).εn) (Eps.h4 0)
        (Ā := 8235 * 10 ^ 140) (Ā' := 2628 * 10 ^ 141) (Ē' := 5256 * 10 ^ 141) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
      (A1 := 8235 * 10 ^ 140) (E1 := 1648 * 10 ^ 141)
      (A2 := 2628 * 10 ^ 141) (E2 := 5256 * 10 ^ 141)
      (A3 := 1262 * 10 ^ 142) (E3 := 2523 * 10 ^ 142)
      (A4 := 1263 * 10 ^ 142) (E4 := 2524 * 10 ^ 142)
      (A5 := 5821 * 10 ^ 144) (E5 := 1164 * 10 ^ 145)
      (A6 := 5822 * 10 ^ 144) (E6 := 1747 * 10 ^ 145)
      (A7 := 1074 * 10 ^ 148) (E7 := 3221 * 10 ^ 148)
      (A8 := 9022 * 10 ^ 148) (E8 := 2706 * 10 ^ 149)
      (Ā' := 9023 * 10 ^ 148) (Ē' := 2707 * 10 ^ 149)
      (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
      ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 1) (R.lnF M 768 ((wts.s4 1).εn))
        P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
        (B.s4 1) (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 1).εn) (Eps.h4 1))
        P.hq P.hgl P.hbb P.hsl
        (R.maps M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 1).εn) (Eps.h4 1)
          (Ā := 2653 * 10 ^ 150) (Ā' := 8464 * 10 ^ 150) (Ē' := 1693 * 10 ^ 151) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (A1 := 2653 * 10 ^ 150) (E1 := 7959 * 10 ^ 150)
        (A2 := 8464 * 10 ^ 150) (E2 := 1693 * 10 ^ 151)
        (A3 := 4063 * 10 ^ 151) (E3 := 8127 * 10 ^ 151)
        (A4 := 4064 * 10 ^ 151) (E4 := 8128 * 10 ^ 151)
        (A5 := 1873 * 10 ^ 154) (E5 := 3746 * 10 ^ 154)
        (A6 := 1874 * 10 ^ 154) (E6 := 5620 * 10 ^ 154)
        (A7 := 3455 * 10 ^ 157) (E7 := 1037 * 10 ^ 158)
        (A8 := 2903 * 10 ^ 158) (E8 := 8711 * 10 ^ 158)
        (Ā' := 2904 * 10 ^ 158) (Ē' := 8712 * 10 ^ 158)
        (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
        (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
        ((FloatBridgesTo.Maps.cnxBlockChW M G.g (wts.s4 2) (R.lnF M 768 ((wts.s4 2).εn))
          P.hw' P.hbb P.hegelu (by norm_num) (by norm_num) (by norm_num) (by norm_num) G.spec
          (B.s4 2) (R.bridge M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 2).εn) (Eps.h4 2))
          P.hq P.hgl P.hbb P.hsl
          (R.maps M P 768 28 (by norm_num) (by norm_num) (by norm_num) ((wts.s4 2).εn) (Eps.h4 2)
            (Ā := 8538 * 10 ^ 159) (Ā' := 2724 * 10 ^ 160) (Ē' := 5448 * 10 ^ 160) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
          (A1 := 8538 * 10 ^ 159) (E1 := 2562 * 10 ^ 160)
          (A2 := 2724 * 10 ^ 160) (E2 := 5448 * 10 ^ 160)
          (A3 := 1308 * 10 ^ 161) (E3 := 2616 * 10 ^ 161)
          (A4 := 1309 * 10 ^ 161) (E4 := 2617 * 10 ^ 161)
          (A5 := 6033 * 10 ^ 163) (E5 := 1206 * 10 ^ 164)
          (A6 := 6034 * 10 ^ 163) (E6 := 1810 * 10 ^ 164)
          (A7 := 1113 * 10 ^ 167) (E7 := 3338 * 10 ^ 167)
          (A8 := 9350 * 10 ^ 167) (E8 := 2804 * 10 ^ 168)
          (Ā' := 9351 * 10 ^ 167) (Ē' := 2805 * 10 ^ 168)
          (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
          (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32])) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])).comp (by norm_num)
          FloatBridgesTo.Maps.idVec)))
  have mGAP := mS4.comp (by norm_num)
    (FloatBridgesTo.Maps.gap (c := 768) (h := 7) (w := 7) M (by norm_num) (by norm_num)
      P.hq (by norm_num [u32]) (by norm_num [u32]) (M.gamma_num (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 9352 * 10 ^ 167) (Ē' := 2806 * 10 ^ 168) (by norm_num [u32]) (by norm_num [u32]))
  have mHead := mGAP.comp (by norm_num)
    (cnxRowLNMaps (s := 1) M R P 28 wts.hγ wts.hβ B.hγ B.hβ Eps.hh (by norm_num)
      (by norm_num) (by norm_num)
      (Ā := 9352 * 10 ^ 167) (Ē := 2806 * 10 ^ 168)
      (A1 := 2984 * 10 ^ 168) (E1 := 5968 * 10 ^ 168) (A2 := 1433 * 10 ^ 169) (E2 := 2865 * 10 ^ 169)
      (Ā' := 1434 * 10 ^ 169) (Ē' := 2866 * 10 ^ 169)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
  exact mHead.comp (by norm_num)
    (FloatBridgesTo.Maps.dense M wts.Wd wts.bd P.hw' P.hbb (by norm_num) B.Wd B.bd
      (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])) (Ā' := 6609 * 10 ^ 171) (Ē' := 1321 * 10 ^ 172) (by norm_num [u32]) (by norm_num [u32]))

/-- The deployed ConvNeXt-T bridge's certified output window at the committed profile. -/
theorem cnxBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).mag 1 ≤ 6609 * 10 ^ 171 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ConvNeXt-T bridge's fresh budget at the committed profile — `2.00 ×` the
    certified window, which is the tell that the cap is biting at every LayerNorm site and the
    statement is the triangle inequality rather than the fold. -/
theorem cnxBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε) :
    (cnxBridge M R G (cnxProfile_committed M hMu hε5) wts B Eps).fresh 1 ≤ 1321 * 10 ^ 172 :=
  (cnxBridge_maps M hMu hε5 R G wts B Eps).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐ **The deployed ConvNeXt-T forward is within `1.321·10¹⁷⁵` of the certified real forward,
    per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile, for `ε ≥ 10⁻⁵`,
    any device LayerNorm statistics accurate to `10⁻²` and any device GELU accurate to `10⁻²`,
    for any rounding model at binary32 accuracy.

    ⛔ **Read the file header before quoting this.** It is a capped statement — the float and
    real forwards both land in the certified `6.609·10¹⁷⁴` window — and NOT the interval fold
    that ResNet-34's, MobileNetV2's and EfficientNet-B0's numbers are. LayerNorm has no
    frozen-statistics mode, so there is no version of this net for which the fold exists. -/
theorem cnx_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (wts : CnxTWeightsCh)
    (B : CnxBounded wts (6/10) 3 (48/10) (84/10)) (Eps : CnxEps wts ε)
    (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |cnxForwardF M R G wts x j - cnxForward wts x j| ≤ 1321 * 10 ^ 172 :=
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
    |cnxForwardF M R G wts x j - convNextForwardTCh wts x j| ≤ 1321 * 10 ^ 172 := by
  have h := cnx_float_logits_le M hMu hε5 R G wts B Eps x hx j
  rwa [cnxForward_eq_committed wts] at h

end Proofs
