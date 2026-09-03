import LeanMlir.Proofs.Float.ViTBlockVFloatBridge
import LeanMlir.Proofs.Foundation.SpecVJP

/-! # A NUMBER for ViT-Tiny: the committed depth-12 vector-LN forward, at the cap

The fifth ImageNet-scale whole-net float statement, and ⛔ **it is ConvNeXt-T's kind of
statement, not ResNet-34's.** For the depth-12 `vitForwardKV` at `224²` — `16×16/s16` patch
embed with CLS token and learned positions, twelve pre-norm blocks of
`LN → 3-head attention → skip → LN → fc1 → GELU → fc2 → skip`, a final per-token LayerNorm, the
CLS slice and the classifier — on the unit input window, at the profile measured per parameter
KIND on the trained checkpoint, for any rounding model at binary32 accuracy:

    output window  ≤ 3.612·10²¹⁸      (`vitBridge_mag_le`)
    fresh budget   ≤ 7.222·10²¹⁸      (`vitBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 7.222·10²¹⁸` (`vit_float_logits_le`).

⛔ **`budget / window = 2.00`, and that ratio is the whole caveat.** All 25 LayerNorm sites go
through `FloatBridgesTo.capped`, and so do all 12 attention sites — so there is no stage inside
a ViT block at which the interval fold survives, and the two skips per block carry that forward.
The statement is *the float and the real forward both land in the certified window* — the
triangle inequality — where ResNet-34's `1.548·10²⁰⁹`, MobileNetV2's `1.444·10⁹⁶` and
EfficientNet-B0's `8.408·10²¹⁰` say *the rounding error folds to this*. Do not table it beside
them without saying so (`planning/float_budget_numbers.md` §9).

⭐ **Its one honest stage is the patch embed** (`Maps.patchEmbed`, window `232.3` from a unit
image, rounding `5.633·10⁻⁴`): it does not reduce, its modulus is linear in the inherited error,
and at the net's input that error is `0`. Everything after it is capped.

⭐⭐ **Why the cap is not optional, and why ViT needed it twice over.** LayerNorm reduces its
statistics out of its own input, so `bnReluBudget` carries a term quadratic in the window and
there are no running statistics to freeze (§0.1) — uncapped, the same fold is `10³²³⁹`.
**Attention is worse and fails differently.** `floatBridges_mhProjAttnFull`'s *window* is derived
as `|real| + |float − real|`, so it carries `smErr`'s `Real.exp (2δ)` at `δ ≈ 3.6·10¹⁰` by block
0 — an argument with no rational bound, so 36 stage numerals **cannot be written down at all**,
at a magnitude *smaller* than the shipped one. Capping does not reach it (`capped` replaces the
modulus, never the window). `floatClose_mhProjAttnFullCap` does: `FloatClose`'s magnitude clause
bounds the FLOAT output directly, and the float output is a rounded dot of float softmax weights
(`≤ 1 + smCap`, exp-free) against float `V`. That is `floatClose_seScale`'s fix one net later —
**when a window contains an error term, ask why.**

⭐ **Attention is ONE leaf, and so is the patch embed.** `FloatBridgesTo` composes single-input
maps; attention fans out (`X ↦ Q,K,V`) before it rejoins, and `patchEmbed_flat` is a single
definition with an `if n.val = 0` branch selecting the CLS token — neither is the composition the
emitted graph spells. The graph says what the kernel does; the definition says what the theorem
is about, and only the second constrains a `Maps` chain.

⚠ **Four hypotheses this number rests on, named.** The deployed LayerNorm's mean and
inverse-stddev are a device reduction and a device `rsqrt` with no IEEE specification (`DeviceLN`,
`emr` relative and `ei` absolute); the deployed GELU is `stablehlo.tanh` (`DeviceGelu`, `egelu`);
the deployed `exp` is `stablehlo.exponential` (`DeviceExp`, `eexp` — ⚠ **relative**, because
`softmaxF_close` divides one exponential sum by another and only a relative error survives the
quotient); and the softmax's side condition **`smRho u eexp 197 < 1`**, which at `eexp = 10⁻²` is
`0.010012` and is a hypothesis, not a footnote. Everything else is proved.

**The tie.** `vitForwardTiny` IS `vitForwardKV` at the committed config, `vitVerified_denote_eq`
is `rfl`, and `vitVerified_fwd_faithful` says the emitted depth-12 multi-head vector-LN graph
denotes it — so `vit_float_logits_le_committed` is a claim about the net `ViTRender` renders.
⭐ No `*RenderPCEval` twin is needed or possible: LayerNorm has no running statistics, so there
is no second render to build — the same saving ConvNeXt had, and the same reason the number has
to be capped.

⚠ The committed spec's head is 10-way (the imagenette classifier `vitVerified` spells) while the
checkpoint the profile was measured on is 1000-way. `nClasses` enters no numeral — the head
dense's fan-in is `D = 192` — and the head kernel's bound is the measured one.

Provenance for the 324 numerals: `scripts/float_budget_envelope.py`'s `vit_chain`, which folds
162 stages in exactly these lemmas' semantics with exact rationals and rounds every stage UP to
four significant figures, and `verify_vit`, which re-asserts each rounded inequality before any
of them is emitted. ⚠ `vit_chain` also returns an `exp_tainted` tag list — the stages whose
numerals would contain a `Real.exp` — because for a net with a transcendental leaf "statable"
means small enough AND writable, and a Python fold hides the second half (`math.expm1` overflows
to a finite float and the chain sails on).
-/

namespace Proofs

open FloatModel
open FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The numeric profile and the parameter bounds
-- ════════════════════════════════════════════════════════════════

/-- The numeric profile the fold runs at. ⚠ Eight magnitude bounds, split by parameter kind as
    the measured checkpoint splits them — but unlike ConvNeXt-T (whose kinds are 14× apart and
    whose uniform bound is UNSTATABLE at `10³⁰¹`) ViT-Tiny's are only 2.5× apart, so the split
    buys ~18 vacuous orders rather than statability (`vit_chain(uniform=True)` is `10²³⁷`). The
    reason is structural: ConvNeXt's outlier is the LAYER SCALE, which multiplies inside every
    block; ViT's is the FINAL LayerNorm γ, which sits after everything and multiplies only the
    head. -/
structure ViTProfile (M : FloatModel)
    (ε wa wm wp wh bb gl bl pb egelu eexp emr ei S q : ℝ) : Prop where
  /-- The four attention kernels `Wq`/`Wk`/`Wv`/`Wo`. -/
  hwa : 0 ≤ wa
  /-- The two MLP kernels `Wfc1`/`Wfc2`. -/
  hwm : 0 ≤ wm
  /-- The patch-embed conv kernel. -/
  hwp : 0 ≤ wp
  /-- The classifier kernel. -/
  hwh : 0 ≤ wh
  /-- Every bias. -/
  hbb : 0 ≤ bb
  /-- LayerNorm γ. -/
  hgl : 0 ≤ gl
  /-- LayerNorm β. -/
  hbl : 0 ≤ bl
  /-- ⭐ The patch embed's SINGLE bound, covering `pos_embed`, `cls_token` and `b_conv`
      together — `floatClose_patchEmbed` takes one, so it is their max. -/
  hpb : 0 ≤ pb
  hegelu : 0 ≤ egelu
  heexp0 : 0 ≤ eexp
  heexp1 : eexp ≤ 1
  hemr : 0 ≤ emr
  hei : 0 ≤ ei
  hS0 : 0 ≤ S
  hε : 0 < ε
  hSε : 1 / Real.sqrt ε ≤ S
  hq : M.u ≤ q

/-- **The whole net's stored parameters within the eight bounds** — the patch embed, the twelve
    blocks (each through `BlockVBounded`), the final LayerNorm and the classifier. The ViT peer
    of `CnxBounded`, stated as a `Prop` over the committed `ViTTinyWeights` rather than as a
    second record. -/
structure ViTBounded (w : ViTTinyWeights) (wa wm wp wh bb gl bl pb : ℝ) : Prop where
  hWc : ∀ d c kh kw, |w.Wc d c kh kw| ≤ wp
  hbc : ∀ d, |w.bc d| ≤ pb
  hcls : ∀ d, |w.cls d| ≤ pb
  hpos : ∀ n d, |w.pos n d| ≤ pb
  hblk : ∀ i, BlockVBounded (w.blocks i) wa wm bb gl bl
  hγF : ∀ i, |w.γF i| ≤ gl
  hβF : ∀ i, |w.βF i| ≤ bl
  hWcls : ∀ i j, |w.Wcls i j| ≤ wh
  hbcls : ∀ j, |w.bcls j| ≤ bb

-- ════════════════════════════════════════════════════════════════
-- § The two side conditions the transformer leaves carry
-- ════════════════════════════════════════════════════════════════

/-- `1/√64 = 1/8`. The attention scale's bound — needed by the leaf, and absent from every
    numeral: the cap's window `mhpBCap` does not mention it. -/
theorem vitScale64 : |(1 : ℝ) / Real.sqrt ((64 : ℕ) : ℝ)| ≤ 1 / 8 := by
  have h : Real.sqrt ((64 : ℕ) : ℝ) = 8 := by
    rw [show (((64 : ℕ)) : ℝ) = 8 ^ 2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  rw [h]
  norm_num

/-- ⚠ **The softmax's side condition, at ViT-Tiny's 197 tokens and `eexp = 10⁻²`.**
    `smRho = γ₁₉₈(1 + eexp) + eexp = 0.010012 < 1`, with room — but it IS a hypothesis of every
    attention leaf, and the whole-net statement carries it. -/
theorem vit_smRho_lt_one (M : FloatModel) (hMu : M.u ≤ u32) :
    smRho M.u (1 / 100) 197 < 1 := by
  have hg := M.gamma_num (k := 198) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32])
  have h := smRho_le_of M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6) hg
    (by norm_num) (by norm_num)
  linarith

/-- `smCap ≤ 2.022·10⁻²` at the same shapes — the softmax row's absolute distance from the real
    one at the SAME logits, and the constant the capped attention window is built on. ⭐ It is
    `smErr` with its `Real.exp (2δ) − 1` term absent, which is exactly what keeps the
    exponential out of the numerals. -/
theorem vit_smCap_le (M : FloatModel) (hMu : M.u ≤ u32) :
    smCap M.u (1 / 100) 197 ≤ 2022 / 10 ^ 5 :=
  smCap_le M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6) (c := 2022 / 10 ^ 5)
    (by norm_num) hMu
    (smRho_le_of M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6)
      (M.gamma_num (k := 198) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
      (by norm_num) (by norm_num))
    (by norm_num) (by norm_num [u32])

-- ════════════════════════════════════════════════════════════════
-- § The whole net: float peer and bridge
-- ════════════════════════════════════════════════════════════════

variable {M : FloatModel} {ε wa wm wp wh bb gl bl pb egelu eexp emr ei S q : ℝ}

/-- **The deployed ViT-Tiny float forward** — every stage the float map its bridge names: the
    rounded patch embed, twelve `blockVFlatF`, the device LayerNorm's normalise chain at each of
    the 25 LN sites, the device `exp` inside every softmax, the device GELU in every MLP, and
    `M.dense` at the head. -/
noncomputable def vitForwardTinyF (M : FloatModel) (R : DeviceLN emr ei) (G : DeviceGelu egelu)
    (X : DeviceExp eexp) (w : ViTTinyWeights) : Vec (3 * 224 * 224) → Vec 10 :=
  vitForwardKVF (ic := 3) (H := 224) (W := 224) (patchSize := 16) (N := 196) (mlpDim := 768)
    (heads := 3) (d_head := 64) (nClasses := 10) (k := 12) M G.g X.e
    w.Wc w.bc w.cls w.pos w.blocks w.γF w.βF (R.lnF M (3 * 64) w.ε) w.Wcls w.bcls

/-- ⭐ **The whole deployed ViT-Tiny forward float-bridges TO its float peer** — a CLOSED
    `FloatBridgesTo` with no `FloatBridgesTo` hypothesis left: the single device LayerNorm slot
    is discharged by `DeviceLN.bridgeAt`, the capped leaf, and all 25 sites share it because the
    net shares one `ε`. Its `.mod` is a closed term over the per-op budgets, and
    `vitBridge_maps` bounds it. -/
noncomputable def vitBridge (M : FloatModel) (R : DeviceLN emr ei) (G : DeviceGelu egelu)
    (X : DeviceExp eexp) (P : ViTProfile M ε wa wm wp wh bb gl bl pb egelu eexp emr ei S q)
    (w : ViTTinyWeights) (B : ViTBounded w wa wm wp wh bb gl bl pb) (hεw : ε ≤ w.ε)
    (hρ : smRho M.u eexp 197 < 1) :
    FloatBridgesTo (vitForwardTiny w) (vitForwardTinyF M R G X w) :=
  floatBridgesTo_vitForwardKV (ic := 3) (H := 224) (W := 224) (patchSize := 16) (N := 196)
    (mlpDim := 768) (heads := 3) (d_head := 64) (nClasses := 10) (k := 12)
    M G.g X.e w.ε w.Wc w.bc w.cls w.pos w.blocks w.γF w.βF (R.lnF M (3 * 64) w.ε) w.Wcls w.bcls
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    P.hwa P.hwm P.hbb P.hegelu P.hwp P.hpb P.hwh P.hbb B.hγF B.hβF G.spec
    P.heexp0 P.heexp1 X.spec vitScale64 hρ
    B.hWc B.hpos B.hcls B.hbc B.hWcls B.hbcls B.hblk
    (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)

-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, measured per parameter KIND on the trained ViT-Tiny checkpoint
    (`/home/skoonce/vit/vit_tiny_imagenet_bf16.bin`, 5,717,416 f32): attention kernels within
    `7/10` (max `0.6594` over 1.77 M entries), MLP kernels within `8/10` (`0.7960`, 3.54 M),
    the patch-embed conv within `3/10` (`0.2522`), the classifier within `4/10` (`0.3408`),
    every bias within `9/10` (`0.8624`), LayerNorm γ within `17/10` (`1.6645`) and β within
    `6/10` (`0.5609`); the patch embed's single `pb` is `9/10`, the max of `pos_embed` (`0.7229`),
    `cls_token` (`0.5454`) and `b_conv` (`0.8624`). `ε ≥ 10⁻⁵` (`ViTRender.lean`'s value) puts
    every LayerNorm's inverse-stddev under `317`; the device mean is taken accurate to `10⁻²`
    relative, the device `rsqrt`, GELU and `exp` to `10⁻²`. -/
theorem vitProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    ViTProfile M ε (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10)
      (1/100) (1/100) (1/100) (1/100) 317 u32 where
  hwa := by norm_num
  hwm := by norm_num
  hwp := by norm_num
  hwh := by norm_num
  hbb := by norm_num
  hgl := by norm_num
  hbl := by norm_num
  hpb := by norm_num
  hegelu := by norm_num
  heexp0 := by norm_num
  heexp1 := by norm_num
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
set_option maxHeartbeats 8000000 in
/-- ⭐ **The envelope, kernel-checked.** 162 numeric stages, 324 rational inequalities, each
    closed with its γ-term bounded through `FloatModel.gamma_num` so `norm_num` never evaluates
    a big power. Four steps — patch embed, the depth-12 body, the final LayerNorm, the head —
    because `Maps.vitBodyKVFlat` is an ENVELOPE fold: the caller passes the window/error
    SEQUENCES and one `Maps` per block, where ConvNeXt's budget file spells all 183 stages out.

    ⛔ Of the 324, the 37 that read `2 * Ā' ≤ Ē'` — one per LayerNorm site and one per
    attention site — are the CAP, not the fold. That is why `7.222·10²¹⁸ / 3.612·10²¹⁸ = 2.00`.

    ⚠ The window/error sequences are `match`es on the block index rather than a closed form: the
    fold hands block `i`'s output to block `i+1`, and the numerals are what `vit_chain` emits. -/
theorem vitBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).Maps 1 0 (3612 * 10 ^ 215) (7222 * 10 ^ 215) := by
  have P := vitProfile_committed M hMu hε5
  have hρ := vit_smRho_lt_one M hMu
  have hsc := vit_smCap_le M hMu
  -- ── the patch embed: ⭐ the one stage of this chain that is an honest fold ──
  have mPatch := FloatBridgesTo.Maps.patchEmbed M 3 224 224 16 196 (3 * 64)
    w.Wc w.bc w.cls w.pos P.hwp P.hpb (by norm_num) (by norm_num) B.hWc B.hpos B.hcls B.hbc
    (Ā := 1) (Ē := 0) (rq := 5633 / 10 ^ 7) (Ā' := 2323 / 10 ^ 1) (Ē' := 5633 / 10 ^ 7)
    (le_trans (patchEmbedRoundErr_le M 3 16 hMu (by norm_num) (by norm_num) (by norm_num))
      (by norm_num [peRoundErrQ, peBranchErrQ, peTripleErrQ, redErr, patchEmbedConvMag,
                    FloatModel.mulErr, u32]))
    (by norm_num [patchEmbedMag, patchEmbedConvMag])
    (by norm_num [patchEmbedConvMag])
  -- ── the final per-token LayerNorm, then the CLS slice and the classifier ──
  have mLN := FloatBridgesTo.Maps.rowLNVecFlat (s := 197) M w.γF w.βF
    (R.lnF M (3 * 64) w.ε) (by norm_num) B.hγF B.hβF
    (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw) P.hq P.hgl P.hbl
    (Ā := 4339 * 10 ^ 210) (Ē := 1303 * 10 ^ 211)
    (A1 := 2765 * 10 ^ 213) (E1 := 5530 * 10 ^ 213) (A2 := 4701 * 10 ^ 213) (E2 := 9402 * 10 ^ 213)
    (Ā' := 4702 * 10 ^ 213) (Ē' := 9403 * 10 ^ 213)
    (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
      (Ā := 4339 * 10 ^ 210) (Ā' := 2765 * 10 ^ 213) (Ē' := 5530 * 10 ^ 213)
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32])
      (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
    (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (by norm_num [u32]) (by norm_num [u32])
  have mHead := FloatBridgesTo.Maps.vitHead 196 M w.Wcls w.bcls P.hwh P.hbb (by norm_num)
    B.hWcls B.hbcls
    (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 4702 * 10 ^ 213) (Ē := 9403 * 10 ^ 213) (Ā' := 3612 * 10 ^ 215) (Ē' := 7222 * 10 ^ 215)
    (by norm_num [u32]) (by norm_num [u32])
  -- ── the four-stage whole-net envelope; the depth-12 body is what is left ──
  refine FloatBridgesTo.Maps.vitForwardKV (ic := 3) (H := 224) (W := 224) (patchSize := 16)
    (N := 196) (mlpDim := 768) (heads := 3) (d_head := 64) (nClasses := 10) (k := 12)
    M G.g X.e w.ε w.Wc w.bc w.cls w.pos w.blocks w.γF w.βF (R.lnF M (3 * 64) w.ε)
    w.Wcls w.bcls (by norm_num) mPatch ?_ mLN mHead
  -- ── the depth-12 body: twelve `Maps.blockVFlatC`, threaded by the envelope fold ──
  refine FloatBridgesTo.Maps.vitBodyKVFlat (Np1 := 197) M G.g X.e w.ε
    (R.lnF M (3 * 64) w.ε) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
    (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw) 12 w.blocks B.hblk
    (fun j => match j with
      | 0 => 2323 / 10 ^ 1
      | 1 => 9365 * 10 ^ 16
      | 2 => 3778 * 10 ^ 34
      | 3 => 1526 * 10 ^ 52
      | 4 => 6158 * 10 ^ 69
      | 5 => 2484 * 10 ^ 87
      | 6 => 1004 * 10 ^ 105
      | 7 => 4048 * 10 ^ 122
      | 8 => 1635 * 10 ^ 140
      | 9 => 6594 * 10 ^ 157
      | 10 => 2659 * 10 ^ 175
      | 11 => 1076 * 10 ^ 193
      | _ => 4339 * 10 ^ 210)
    (fun j => match j with
      | 0 => 5633 / 10 ^ 7
      | 1 => 2811 * 10 ^ 17
      | 2 => 1135 * 10 ^ 35
      | 3 => 4572 * 10 ^ 52
      | 4 => 1848 * 10 ^ 70
      | 5 => 7448 * 10 ^ 87
      | 6 => 3005 * 10 ^ 105
      | 7 => 1216 * 10 ^ 123
      | 8 => 4900 * 10 ^ 140
      | 9 => 1980 * 10 ^ 158
      | 10 => 7977 * 10 ^ 175
      | 11 => 3221 * 10 ^ 193
      | _ => 1303 * 10 ^ 211)
    ?_
  intro i
  fin_cases i
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 0) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 0) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 1481 * 10 ^ 2) (E1 := 2962 * 10 ^ 2) (A2 := 2518 * 10 ^ 2) (E2 := 5036 * 10 ^ 2)
        (A3 := 2519 * 10 ^ 2) (E3 := 5037 * 10 ^ 2) (A4 := 6805 * 10 ^ 6) (E4 := 1361 * 10 ^ 7)
        (A5 := 9147 * 10 ^ 8) (E5 := 1830 * 10 ^ 9) (A6 := 9148 * 10 ^ 8) (E6 := 1831 * 10 ^ 9)
        (A7 := 5830 * 10 ^ 11) (E7 := 1166 * 10 ^ 12) (A8 := 9912 * 10 ^ 11) (E8 := 1983 * 10 ^ 12)
        (A9 := 9913 * 10 ^ 11) (E9 := 1984 * 10 ^ 12) (A10 := 1523 * 10 ^ 14) (E10 := 3048 * 10 ^ 14)
        (A11 := 1524 * 10 ^ 14) (E11 := 4573 * 10 ^ 14) (A12 := 9364 * 10 ^ 16) (E12 := 2810 * 10 ^ 17)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 2323 / 10 ^ 1) (Ā' := 1481 * 10 ^ 2) (Ē' := 2962 * 10 ^ 2)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 9148 * 10 ^ 8) (Ā' := 5830 * 10 ^ 11) (Ē' := 1166 * 10 ^ 12)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 1) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 1) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 5968 * 10 ^ 19) (E1 := 1194 * 10 ^ 20) (A2 := 1015 * 10 ^ 20) (E2 := 2030 * 10 ^ 20)
        (A3 := 1016 * 10 ^ 20) (E3 := 2031 * 10 ^ 20) (A4 := 2745 * 10 ^ 24) (E4 := 5490 * 10 ^ 24)
        (A5 := 3690 * 10 ^ 26) (E5 := 7379 * 10 ^ 26) (A6 := 3691 * 10 ^ 26) (E6 := 7380 * 10 ^ 26)
        (A7 := 2352 * 10 ^ 29) (E7 := 4704 * 10 ^ 29) (A8 := 3999 * 10 ^ 29) (E8 := 7997 * 10 ^ 29)
        (A9 := 4000 * 10 ^ 29) (E9 := 7998 * 10 ^ 29) (A10 := 6145 * 10 ^ 31) (E10 := 1229 * 10 ^ 32)
        (A11 := 6146 * 10 ^ 31) (E11 := 1844 * 10 ^ 32) (A12 := 3777 * 10 ^ 34) (E12 := 1134 * 10 ^ 35)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 9365 * 10 ^ 16) (Ā' := 5968 * 10 ^ 19) (Ē' := 1194 * 10 ^ 20)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 3691 * 10 ^ 26) (Ā' := 2352 * 10 ^ 29) (Ē' := 4704 * 10 ^ 29)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 2) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 2) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 2408 * 10 ^ 37) (E1 := 4816 * 10 ^ 37) (A2 := 4094 * 10 ^ 37) (E2 := 8188 * 10 ^ 37)
        (A3 := 4095 * 10 ^ 37) (E3 := 8189 * 10 ^ 37) (A4 := 1107 * 10 ^ 42) (E4 := 2214 * 10 ^ 42)
        (A5 := 1488 * 10 ^ 44) (E5 := 2976 * 10 ^ 44) (A6 := 1489 * 10 ^ 44) (E6 := 2977 * 10 ^ 44)
        (A7 := 9488 * 10 ^ 46) (E7 := 1898 * 10 ^ 47) (A8 := 1613 * 10 ^ 47) (E8 := 3227 * 10 ^ 47)
        (A9 := 1614 * 10 ^ 47) (E9 := 3228 * 10 ^ 47) (A10 := 2480 * 10 ^ 49) (E10 := 4959 * 10 ^ 49)
        (A11 := 2481 * 10 ^ 49) (E11 := 7439 * 10 ^ 49) (A12 := 1525 * 10 ^ 52) (E12 := 4571 * 10 ^ 52)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 3778 * 10 ^ 34) (Ā' := 2408 * 10 ^ 37) (Ē' := 4816 * 10 ^ 37)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1489 * 10 ^ 44) (Ā' := 9488 * 10 ^ 46) (Ē' := 1898 * 10 ^ 47)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 3) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 3) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 9724 * 10 ^ 54) (E1 := 1945 * 10 ^ 55) (A2 := 1654 * 10 ^ 55) (E2 := 3307 * 10 ^ 55)
        (A3 := 1655 * 10 ^ 55) (E3 := 3308 * 10 ^ 55) (A4 := 4471 * 10 ^ 59) (E4 := 8942 * 10 ^ 59)
        (A5 := 6010 * 10 ^ 61) (E5 := 1202 * 10 ^ 62) (A6 := 6011 * 10 ^ 61) (E6 := 1203 * 10 ^ 62)
        (A7 := 3831 * 10 ^ 64) (E7 := 7662 * 10 ^ 64) (A8 := 6513 * 10 ^ 64) (E8 := 1303 * 10 ^ 65)
        (A9 := 6514 * 10 ^ 64) (E9 := 1304 * 10 ^ 65) (A10 := 1001 * 10 ^ 67) (E10 := 2003 * 10 ^ 67)
        (A11 := 1002 * 10 ^ 67) (E11 := 3005 * 10 ^ 67) (A12 := 6157 * 10 ^ 69) (E12 := 1847 * 10 ^ 70)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1526 * 10 ^ 52) (Ā' := 9724 * 10 ^ 54) (Ē' := 1945 * 10 ^ 55)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 6011 * 10 ^ 61) (Ā' := 3831 * 10 ^ 64) (Ē' := 7662 * 10 ^ 64)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 4) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 4) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 3924 * 10 ^ 72) (E1 := 7848 * 10 ^ 72) (A2 := 6671 * 10 ^ 72) (E2 := 1335 * 10 ^ 73)
        (A3 := 6672 * 10 ^ 72) (E3 := 1336 * 10 ^ 73) (A4 := 1803 * 10 ^ 77) (E4 := 3606 * 10 ^ 77)
        (A5 := 2424 * 10 ^ 79) (E5 := 4847 * 10 ^ 79) (A6 := 2425 * 10 ^ 79) (E6 := 4848 * 10 ^ 79)
        (A7 := 1546 * 10 ^ 82) (E7 := 3092 * 10 ^ 82) (A8 := 2629 * 10 ^ 82) (E8 := 5257 * 10 ^ 82)
        (A9 := 2630 * 10 ^ 82) (E9 := 5258 * 10 ^ 82) (A10 := 4040 * 10 ^ 84) (E10 := 8077 * 10 ^ 84)
        (A11 := 4041 * 10 ^ 84) (E11 := 1212 * 10 ^ 85) (A12 := 2483 * 10 ^ 87) (E12 := 7447 * 10 ^ 87)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 6158 * 10 ^ 69) (Ā' := 3924 * 10 ^ 72) (Ē' := 7848 * 10 ^ 72)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 2425 * 10 ^ 79) (Ā' := 1546 * 10 ^ 82) (Ē' := 3092 * 10 ^ 82)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 5) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 5) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 1583 * 10 ^ 90) (E1 := 3166 * 10 ^ 90) (A2 := 2692 * 10 ^ 90) (E2 := 5383 * 10 ^ 90)
        (A3 := 2693 * 10 ^ 90) (E3 := 5384 * 10 ^ 90) (A4 := 7275 * 10 ^ 94) (E4 := 1455 * 10 ^ 95)
        (A5 := 9778 * 10 ^ 96) (E5 := 1956 * 10 ^ 97) (A6 := 9779 * 10 ^ 96) (E6 := 1957 * 10 ^ 97)
        (A7 := 6232 * 10 ^ 99) (E7 := 1247 * 10 ^ 100) (A8 := 1060 * 10 ^ 100) (E8 := 2120 * 10 ^ 100)
        (A9 := 1061 * 10 ^ 100) (E9 := 2121 * 10 ^ 100) (A10 := 1630 * 10 ^ 102) (E10 := 3258 * 10 ^ 102)
        (A11 := 1631 * 10 ^ 102) (E11 := 4888 * 10 ^ 102) (A12 := 1003 * 10 ^ 105) (E12 := 3004 * 10 ^ 105)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 2484 * 10 ^ 87) (Ā' := 1583 * 10 ^ 90) (Ē' := 3166 * 10 ^ 90)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 9779 * 10 ^ 96) (Ā' := 6232 * 10 ^ 99) (Ē' := 1247 * 10 ^ 100)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 6) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 6) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 6398 * 10 ^ 107) (E1 := 1280 * 10 ^ 108) (A2 := 1088 * 10 ^ 108) (E2 := 2177 * 10 ^ 108)
        (A3 := 1089 * 10 ^ 108) (E3 := 2178 * 10 ^ 108) (A4 := 2942 * 10 ^ 112) (E4 := 5884 * 10 ^ 112)
        (A5 := 3955 * 10 ^ 114) (E5 := 7909 * 10 ^ 114) (A6 := 3956 * 10 ^ 114) (E6 := 7910 * 10 ^ 114)
        (A7 := 2521 * 10 ^ 117) (E7 := 5042 * 10 ^ 117) (A8 := 4286 * 10 ^ 117) (E8 := 8572 * 10 ^ 117)
        (A9 := 4287 * 10 ^ 117) (E9 := 8573 * 10 ^ 117) (A10 := 6585 * 10 ^ 119) (E10 := 1317 * 10 ^ 120)
        (A11 := 6586 * 10 ^ 119) (E11 := 1976 * 10 ^ 120) (A12 := 4047 * 10 ^ 122) (E12 := 1215 * 10 ^ 123)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1004 * 10 ^ 105) (Ā' := 6398 * 10 ^ 107) (Ē' := 1280 * 10 ^ 108)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 3956 * 10 ^ 114) (Ā' := 2521 * 10 ^ 117) (Ē' := 5042 * 10 ^ 117)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 7) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 7) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 2580 * 10 ^ 125) (E1 := 5160 * 10 ^ 125) (A2 := 4387 * 10 ^ 125) (E2 := 8773 * 10 ^ 125)
        (A3 := 4388 * 10 ^ 125) (E3 := 8774 * 10 ^ 125) (A4 := 1186 * 10 ^ 130) (E4 := 2372 * 10 ^ 130)
        (A5 := 1595 * 10 ^ 132) (E5 := 3189 * 10 ^ 132) (A6 := 1596 * 10 ^ 132) (E6 := 3190 * 10 ^ 132)
        (A7 := 1017 * 10 ^ 135) (E7 := 2034 * 10 ^ 135) (A8 := 1729 * 10 ^ 135) (E8 := 3458 * 10 ^ 135)
        (A9 := 1730 * 10 ^ 135) (E9 := 3459 * 10 ^ 135) (A10 := 2658 * 10 ^ 137) (E10 := 5314 * 10 ^ 137)
        (A11 := 2659 * 10 ^ 137) (E11 := 7972 * 10 ^ 137) (A12 := 1634 * 10 ^ 140) (E12 := 4899 * 10 ^ 140)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 4048 * 10 ^ 122) (Ā' := 2580 * 10 ^ 125) (Ē' := 5160 * 10 ^ 125)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1596 * 10 ^ 132) (Ā' := 1017 * 10 ^ 135) (Ē' := 2034 * 10 ^ 135)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 8) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 8) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 1042 * 10 ^ 143) (E1 := 2084 * 10 ^ 143) (A2 := 1772 * 10 ^ 143) (E2 := 3543 * 10 ^ 143)
        (A3 := 1773 * 10 ^ 143) (E3 := 3544 * 10 ^ 143) (A4 := 4790 * 10 ^ 147) (E4 := 9580 * 10 ^ 147)
        (A5 := 6438 * 10 ^ 149) (E5 := 1288 * 10 ^ 150) (A6 := 6439 * 10 ^ 149) (E6 := 1289 * 10 ^ 150)
        (A7 := 4103 * 10 ^ 152) (E7 := 8206 * 10 ^ 152) (A8 := 6976 * 10 ^ 152) (E8 := 1396 * 10 ^ 153)
        (A9 := 6977 * 10 ^ 152) (E9 := 1397 * 10 ^ 153) (A10 := 1072 * 10 ^ 155) (E10 := 2146 * 10 ^ 155)
        (A11 := 1073 * 10 ^ 155) (E11 := 3220 * 10 ^ 155) (A12 := 6593 * 10 ^ 157) (E12 := 1979 * 10 ^ 158)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1635 * 10 ^ 140) (Ā' := 1042 * 10 ^ 143) (Ē' := 2084 * 10 ^ 143)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 6439 * 10 ^ 149) (Ā' := 4103 * 10 ^ 152) (Ē' := 8206 * 10 ^ 152)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 9) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 9) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 4202 * 10 ^ 160) (E1 := 8404 * 10 ^ 160) (A2 := 7144 * 10 ^ 160) (E2 := 1429 * 10 ^ 161)
        (A3 := 7145 * 10 ^ 160) (E3 := 1430 * 10 ^ 161) (A4 := 1931 * 10 ^ 165) (E4 := 3862 * 10 ^ 165)
        (A5 := 2596 * 10 ^ 167) (E5 := 5191 * 10 ^ 167) (A6 := 2597 * 10 ^ 167) (E6 := 5192 * 10 ^ 167)
        (A7 := 1655 * 10 ^ 170) (E7 := 3310 * 10 ^ 170) (A8 := 2814 * 10 ^ 170) (E8 := 5628 * 10 ^ 170)
        (A9 := 2815 * 10 ^ 170) (E9 := 5629 * 10 ^ 170) (A10 := 4324 * 10 ^ 172) (E10 := 8647 * 10 ^ 172)
        (A11 := 4325 * 10 ^ 172) (E11 := 1298 * 10 ^ 173) (A12 := 2658 * 10 ^ 175) (E12 := 7976 * 10 ^ 175)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 6594 * 10 ^ 157) (Ā' := 4202 * 10 ^ 160) (Ē' := 8404 * 10 ^ 160)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 2597 * 10 ^ 167) (Ā' := 1655 * 10 ^ 170) (Ē' := 3310 * 10 ^ 170)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 10) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 10) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 1695 * 10 ^ 178) (E1 := 3390 * 10 ^ 178) (A2 := 2882 * 10 ^ 178) (E2 := 5764 * 10 ^ 178)
        (A3 := 2883 * 10 ^ 178) (E3 := 5765 * 10 ^ 178) (A4 := 7788 * 10 ^ 182) (E4 := 1558 * 10 ^ 183)
        (A5 := 1047 * 10 ^ 185) (E5 := 2094 * 10 ^ 185) (A6 := 1048 * 10 ^ 185) (E6 := 2095 * 10 ^ 185)
        (A7 := 6678 * 10 ^ 187) (E7 := 1336 * 10 ^ 188) (A8 := 1136 * 10 ^ 188) (E8 := 2272 * 10 ^ 188)
        (A9 := 1137 * 10 ^ 188) (E9 := 2273 * 10 ^ 188) (A10 := 1747 * 10 ^ 190) (E10 := 3492 * 10 ^ 190)
        (A11 := 1748 * 10 ^ 190) (E11 := 5239 * 10 ^ 190) (A12 := 1075 * 10 ^ 193) (E12 := 3220 * 10 ^ 193)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 2659 * 10 ^ 175) (Ā' := 1695 * 10 ^ 178) (Ē' := 3390 * 10 ^ 178)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1048 * 10 ^ 185) (Ā' := 6678 * 10 ^ 187) (Ē' := 1336 * 10 ^ 188)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 11) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 11) (R.bridgeAt M P.hε P.hSε (3 * 64) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 6857 * 10 ^ 195) (E1 := 1372 * 10 ^ 196) (A2 := 1166 * 10 ^ 196) (E2 := 2333 * 10 ^ 196)
        (A3 := 1167 * 10 ^ 196) (E3 := 2334 * 10 ^ 196) (A4 := 3153 * 10 ^ 200) (E4 := 6306 * 10 ^ 200)
        (A5 := 4238 * 10 ^ 202) (E5 := 8476 * 10 ^ 202) (A6 := 4239 * 10 ^ 202) (E6 := 8477 * 10 ^ 202)
        (A7 := 2702 * 10 ^ 205) (E7 := 5404 * 10 ^ 205) (A8 := 4594 * 10 ^ 205) (E8 := 9187 * 10 ^ 205)
        (A9 := 4595 * 10 ^ 205) (E9 := 9188 * 10 ^ 205) (A10 := 7059 * 10 ^ 207) (E10 := 1412 * 10 ^ 208)
        (A11 := 7060 * 10 ^ 207) (E11 := 2119 * 10 ^ 208) (A12 := 4338 * 10 ^ 210) (E12 := 1302 * 10 ^ 211)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 1076 * 10 ^ 193) (Ā' := 6857 * 10 ^ 195) (Ē' := 1372 * 10 ^ 196)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) w.ε hεw
          (Ā := 4239 * 10 ^ 202) (Ā' := 2702 * 10 ^ 205) (Ē' := 5404 * 10 ^ 205)
          (by norm_num [bnNormBudget, FloatModel.mulErr, u32]) (by norm_num [bnNormBudget, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))

/-- The deployed ViT-Tiny bridge's certified output window at the committed profile. -/
theorem vitBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).mag 1 ≤ 3612 * 10 ^ 215 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ViT-Tiny bridge's fresh budget at the committed profile — `2.00 ×` the
    certified window, which is the tell that the cap is biting at every LayerNorm and every
    attention site and the statement is the triangle inequality rather than the fold. -/
theorem vitBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).fresh 1 ≤ 7222 * 10 ^ 215 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐ **The deployed ViT-Tiny forward is within `7.222·10²¹⁸` of the certified real forward,
    per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile, for
    `ε ≥ 10⁻⁵`, any device LayerNorm statistics accurate to `10⁻²`, any device GELU accurate to
    `10⁻²` and any device `exp` accurate to `10⁻²` RELATIVE, for any rounding model at binary32
    accuracy.

    ⛔ **Read the file header before quoting this.** It is a capped statement — the float and
    real forwards both land in the certified `3.612·10²¹⁸` window — and NOT the interval fold
    that ResNet-34's, MobileNetV2's and EfficientNet-B0's numbers are. Every LayerNorm and every
    attention site is capped; the patch embed is the one stage that is not. -/
theorem vit_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |vitForwardTinyF M R G X w x j - vitForwardTiny w x j| ≤ 7222 * 10 ^ 215 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed ViT-Tiny spec's forward
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The number, stated about the committed spec's denotation.** `vitVerified_denote_eq` is
    `rfl`, and `vitVerified_fwd_faithful` says the emitted depth-12 multi-head vector-LN graph
    denotes the same function — so the budget is a claim about the net `ViTRender.lean` renders,
    not about a record-plugged look-alike. -/
theorem vit_float_logits_le_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100))
    (X : DeviceExp (1/100)) (w : ViTTinyWeights)
    (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |vitForwardTinyF M R G X w x j - denoteVitTiny vitVerified.layers w x j| ≤ 7222 * 10 ^ 215 := by
  have h := vit_float_logits_le M hMu hε5 R G X w B hεw x hx j
  rwa [vitVerified_denote_eq w]

end Proofs
