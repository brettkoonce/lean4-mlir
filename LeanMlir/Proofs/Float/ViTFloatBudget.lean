import LeanMlir.Proofs.Float.ViTBlockVFloatBridge
import LeanMlir.Proofs.Foundation.SpecVJP

/-! # A NUMBER for ViT-Tiny: the committed depth-12 vector-LN forward, at the cap

The fifth ImageNet-scale whole-net float statement, and ⛔ **it is ConvNeXt-T's kind of
statement, not ResNet-34's.** For the depth-12 `vitForwardKV` at `224²` — `16×16/s16` patch
embed with CLS token and learned positions, twelve pre-norm blocks of
`LN → 3-head attention → skip → LN → fc1 → GELU → fc2 → skip`, a final per-token LayerNorm, the
CLS slice and the classifier — on the unit input window, at the profile measured per parameter
KIND on the trained checkpoint, for any rounding model at binary32 accuracy:

    output window  ≤ 1.130·10¹⁶¹      (`vitBridge_mag_le`)
    fresh budget   ≤ 2.259·10¹⁶¹      (`vitBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 2.259·10¹⁶¹` (`vit_float_logits_le`).

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
there are no running statistics to freeze (§0.1) — uncapped at the leaf this file used to state, the same fold is `10³²²⁶`.
**Attention is worse and fails differently.** `floatBridges_mhProjAttnFull`'s *window* is derived
as `|real| + |float − real|`, so it carries `smErr`'s `Real.exp (2δ)` at `δ ≈ 3.6·10¹⁰` by block
0 — an argument with no rational bound, so 36 stage numerals **cannot be written down at all**,
at a magnitude *smaller* than the shipped one. Capping does not reach it (`capped` replaces the
modulus, never the window). `floatClose_mhProjAttnFullCap` does: `FloatClose`'s magnitude clause
bounds the FLOAT output directly, and the float output is a rounded dot of float softmax weights
(`≤ 1 + smCap`, exp-free) against float `V`. That is `floatClose_seScale`'s fix one net later —
**when a window contains an error term, ask why.**

⭐⭐ **The LayerNorm window is charged at `|x̂| ≤ √n`, not at `|x − μ|·|istd| ≤ 2A·S` — §0.1's
escape 2, and it is worth 57 orders.** `Maps.bnCappedX` (`BnXhatFloatBridge.lean`) states the
pure-normalise leaf at `bnXhat_sq_le`'s bound, which mentions neither factor and holds at every
input, so `Xh = 14` — the ceiling root of ViT's `D = 3·64 = 192` — and enters no profile. Before
this the number was `3.612·10²¹⁸ / 7.222·10²¹⁸`. ⭐ It cost no new hypothesis and no new
mathematics: `bnXhat_sq_le` had been in the repo since the realistic-seal work and is
load-bearing on all four whole-net BACKWARD numbers. ⛔ It does not make this a fold — the
modulus still carries §0.1's quadratic, and even with that gone the honest fold is 7 orders
above the triangle inequality here, so `capped`'s `min` still selects the cap. ⚠ **And it does
nothing for the ATTENTION sites**, whose cap is there for REPRESENTABILITY rather than size; a
tighter window cannot reach a `Real.exp` with no rational bound.

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
    (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)

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
    attention site — are the CAP, not the fold. That is why `2.259·10¹⁶¹ / 1.130·10¹⁶¹ = 2.00`.

    ⚠ The window/error sequences are `match`es on the block index rather than a closed form: the
    fold hands block `i`'s output to block `i+1`, and the numerals are what `vit_chain` emits. -/
theorem vitBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).Maps 1 0 (1130 * 10 ^ 158) (2259 * 10 ^ 158) := by
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
    (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw) P.hq P.hgl P.hbl
    (Ā := 2709 * 10 ^ 155) (Ē := 8124 * 10 ^ 155)
    (A1 := 8643 * 10 ^ 155) (E1 := 1729 * 10 ^ 156) (A2 := 1470 * 10 ^ 156) (E2 := 2940 * 10 ^ 156)
    (Ā' := 1471 * 10 ^ 156) (Ē' := 2941 * 10 ^ 156)
    (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
      (Ā := 2709 * 10 ^ 155) (Ā' := 8643 * 10 ^ 155) (Ē' := 1729 * 10 ^ 156)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
    (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (by norm_num [u32]) (by norm_num [u32])
  have mHead := FloatBridgesTo.Maps.vitHead 196 M w.Wcls w.bcls P.hwh P.hbb (by norm_num)
    B.hWcls B.hbcls
    (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1471 * 10 ^ 156) (Ē := 2941 * 10 ^ 156) (Ā' := 1130 * 10 ^ 158) (Ē' := 2259 * 10 ^ 158)
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
    (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw) 12 w.blocks B.hblk
    (fun j => match j with
      | 0 => 2323 / 10 ^ 1
      | 1 => 2395 * 10 ^ 12
      | 2 => 2421 * 10 ^ 25
      | 3 => 2449 * 10 ^ 38
      | 4 => 2478 * 10 ^ 51
      | 5 => 2507 * 10 ^ 64
      | 6 => 2534 * 10 ^ 77
      | 7 => 2563 * 10 ^ 90
      | 8 => 2592 * 10 ^ 103
      | 9 => 2620 * 10 ^ 116
      | 10 => 2650 * 10 ^ 129
      | 11 => 2680 * 10 ^ 142
      | _ => 2709 * 10 ^ 155)
    (fun j => match j with
      | 0 => 5633 / 10 ^ 7
      | 1 => 7178 * 10 ^ 12
      | 2 => 7258 * 10 ^ 25
      | 3 => 7344 * 10 ^ 38
      | 4 => 7430 * 10 ^ 51
      | 5 => 7516 * 10 ^ 64
      | 6 => 7596 * 10 ^ 77
      | 7 => 7682 * 10 ^ 90
      | 8 => 7774 * 10 ^ 103
      | 9 => 7860 * 10 ^ 116
      | 10 => 7946 * 10 ^ 129
      | 11 => 8038 * 10 ^ 142
      | _ => 8124 * 10 ^ 155)
    ?_
  intro i
  fin_cases i
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 0) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 0) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 7551 / 10 ^ 1) (E1 := 1511) (A2 := 1284) (E2 := 2569)
        (A3 := 1285) (E3 := 2570) (A4 := 3472 * 10 ^ 4) (E4 := 6944 * 10 ^ 4)
        (A5 := 4667 * 10 ^ 6) (E5 := 9333 * 10 ^ 6) (A6 := 4668 * 10 ^ 6) (E6 := 9334 * 10 ^ 6)
        (A7 := 1490 * 10 ^ 7) (E7 := 2980 * 10 ^ 7) (A8 := 2534 * 10 ^ 7) (E8 := 5067 * 10 ^ 7)
        (A9 := 2535 * 10 ^ 7) (E9 := 5068 * 10 ^ 7) (A10 := 3894 * 10 ^ 9) (E10 := 7785 * 10 ^ 9)
        (A11 := 3895 * 10 ^ 9) (E11 := 1168 * 10 ^ 10) (A12 := 2394 * 10 ^ 12) (E12 := 7177 * 10 ^ 12)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2323 / 10 ^ 1) (Ā' := 7551 / 10 ^ 1) (Ē' := 1511)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4668 * 10 ^ 6) (Ā' := 1490 * 10 ^ 7) (Ē' := 2980 * 10 ^ 7)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 1) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 1) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 7641 * 10 ^ 12) (E1 := 1529 * 10 ^ 13) (A2 := 1299 * 10 ^ 13) (E2 := 2600 * 10 ^ 13)
        (A3 := 1300 * 10 ^ 13) (E3 := 2601 * 10 ^ 13) (A4 := 3512 * 10 ^ 17) (E4 := 7024 * 10 ^ 17)
        (A5 := 4721 * 10 ^ 19) (E5 := 9441 * 10 ^ 19) (A6 := 4722 * 10 ^ 19) (E6 := 9442 * 10 ^ 19)
        (A7 := 1507 * 10 ^ 20) (E7 := 3014 * 10 ^ 20) (A8 := 2562 * 10 ^ 20) (E8 := 5124 * 10 ^ 20)
        (A9 := 2563 * 10 ^ 20) (E9 := 5125 * 10 ^ 20) (A10 := 3937 * 10 ^ 22) (E10 := 7873 * 10 ^ 22)
        (A11 := 3938 * 10 ^ 22) (E11 := 1181 * 10 ^ 23) (A12 := 2420 * 10 ^ 25) (E12 := 7257 * 10 ^ 25)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2395 * 10 ^ 12) (Ā' := 7641 * 10 ^ 12) (Ē' := 1529 * 10 ^ 13)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4722 * 10 ^ 19) (Ā' := 1507 * 10 ^ 20) (Ē' := 3014 * 10 ^ 20)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 2) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 2) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 7724 * 10 ^ 25) (E1 := 1545 * 10 ^ 26) (A2 := 1314 * 10 ^ 26) (E2 := 2627 * 10 ^ 26)
        (A3 := 1315 * 10 ^ 26) (E3 := 2628 * 10 ^ 26) (A4 := 3553 * 10 ^ 30) (E4 := 7106 * 10 ^ 30)
        (A5 := 4776 * 10 ^ 32) (E5 := 9551 * 10 ^ 32) (A6 := 4777 * 10 ^ 32) (E6 := 9552 * 10 ^ 32)
        (A7 := 1524 * 10 ^ 33) (E7 := 3048 * 10 ^ 33) (A8 := 2591 * 10 ^ 33) (E8 := 5182 * 10 ^ 33)
        (A9 := 2592 * 10 ^ 33) (E9 := 5183 * 10 ^ 33) (A10 := 3982 * 10 ^ 35) (E10 := 7962 * 10 ^ 35)
        (A11 := 3983 * 10 ^ 35) (E11 := 1195 * 10 ^ 36) (A12 := 2448 * 10 ^ 38) (E12 := 7343 * 10 ^ 38)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2421 * 10 ^ 25) (Ā' := 7724 * 10 ^ 25) (Ē' := 1545 * 10 ^ 26)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4777 * 10 ^ 32) (Ā' := 1524 * 10 ^ 33) (Ē' := 3048 * 10 ^ 33)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 3) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 3) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 7813 * 10 ^ 38) (E1 := 1563 * 10 ^ 39) (A2 := 1329 * 10 ^ 39) (E2 := 2658 * 10 ^ 39)
        (A3 := 1330 * 10 ^ 39) (E3 := 2659 * 10 ^ 39) (A4 := 3593 * 10 ^ 43) (E4 := 7186 * 10 ^ 43)
        (A5 := 4830 * 10 ^ 45) (E5 := 9659 * 10 ^ 45) (A6 := 4831 * 10 ^ 45) (E6 := 9660 * 10 ^ 45)
        (A7 := 1542 * 10 ^ 46) (E7 := 3084 * 10 ^ 46) (A8 := 2622 * 10 ^ 46) (E8 := 5243 * 10 ^ 46)
        (A9 := 2623 * 10 ^ 46) (E9 := 5244 * 10 ^ 46) (A10 := 4029 * 10 ^ 48) (E10 := 8055 * 10 ^ 48)
        (A11 := 4030 * 10 ^ 48) (E11 := 1209 * 10 ^ 49) (A12 := 2477 * 10 ^ 51) (E12 := 7429 * 10 ^ 51)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2449 * 10 ^ 38) (Ā' := 7813 * 10 ^ 38) (Ē' := 1563 * 10 ^ 39)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4831 * 10 ^ 45) (Ā' := 1542 * 10 ^ 46) (Ē' := 3084 * 10 ^ 46)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 4) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 4) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 7906 * 10 ^ 51) (E1 := 1582 * 10 ^ 52) (A2 := 1345 * 10 ^ 52) (E2 := 2690 * 10 ^ 52)
        (A3 := 1346 * 10 ^ 52) (E3 := 2691 * 10 ^ 52) (A4 := 3636 * 10 ^ 56) (E4 := 7272 * 10 ^ 56)
        (A5 := 4887 * 10 ^ 58) (E5 := 9774 * 10 ^ 58) (A6 := 4888 * 10 ^ 58) (E6 := 9775 * 10 ^ 58)
        (A7 := 1560 * 10 ^ 59) (E7 := 3120 * 10 ^ 59) (A8 := 2653 * 10 ^ 59) (E8 := 5305 * 10 ^ 59)
        (A9 := 2654 * 10 ^ 59) (E9 := 5306 * 10 ^ 59) (A10 := 4077 * 10 ^ 61) (E10 := 8151 * 10 ^ 61)
        (A11 := 4078 * 10 ^ 61) (E11 := 1223 * 10 ^ 62) (A12 := 2506 * 10 ^ 64) (E12 := 7515 * 10 ^ 64)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2478 * 10 ^ 51) (Ā' := 7906 * 10 ^ 51) (Ē' := 1582 * 10 ^ 52)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4888 * 10 ^ 58) (Ā' := 1560 * 10 ^ 59) (Ē' := 3120 * 10 ^ 59)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 5) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 5) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 7998 * 10 ^ 64) (E1 := 1600 * 10 ^ 65) (A2 := 1360 * 10 ^ 65) (E2 := 2721 * 10 ^ 65)
        (A3 := 1361 * 10 ^ 65) (E3 := 2722 * 10 ^ 65) (A4 := 3677 * 10 ^ 69) (E4 := 7354 * 10 ^ 69)
        (A5 := 4942 * 10 ^ 71) (E5 := 9884 * 10 ^ 71) (A6 := 4943 * 10 ^ 71) (E6 := 9885 * 10 ^ 71)
        (A7 := 1577 * 10 ^ 72) (E7 := 3154 * 10 ^ 72) (A8 := 2681 * 10 ^ 72) (E8 := 5362 * 10 ^ 72)
        (A9 := 2682 * 10 ^ 72) (E9 := 5363 * 10 ^ 72) (A10 := 4120 * 10 ^ 74) (E10 := 8238 * 10 ^ 74)
        (A11 := 4121 * 10 ^ 74) (E11 := 1236 * 10 ^ 75) (A12 := 2533 * 10 ^ 77) (E12 := 7595 * 10 ^ 77)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2507 * 10 ^ 64) (Ā' := 7998 * 10 ^ 64) (Ē' := 1600 * 10 ^ 65)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4943 * 10 ^ 71) (Ā' := 1577 * 10 ^ 72) (Ē' := 3154 * 10 ^ 72)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 6) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 6) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 8084 * 10 ^ 77) (E1 := 1617 * 10 ^ 78) (A2 := 1375 * 10 ^ 78) (E2 := 2749 * 10 ^ 78)
        (A3 := 1376 * 10 ^ 78) (E3 := 2750 * 10 ^ 78) (A4 := 3717 * 10 ^ 82) (E4 := 7434 * 10 ^ 82)
        (A5 := 4996 * 10 ^ 84) (E5 := 9992 * 10 ^ 84) (A6 := 4997 * 10 ^ 84) (E6 := 9993 * 10 ^ 84)
        (A7 := 1595 * 10 ^ 85) (E7 := 3190 * 10 ^ 85) (A8 := 2712 * 10 ^ 85) (E8 := 5424 * 10 ^ 85)
        (A9 := 2713 * 10 ^ 85) (E9 := 5425 * 10 ^ 85) (A10 := 4168 * 10 ^ 87) (E10 := 8333 * 10 ^ 87)
        (A11 := 4169 * 10 ^ 87) (E11 := 1250 * 10 ^ 88) (A12 := 2562 * 10 ^ 90) (E12 := 7681 * 10 ^ 90)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2534 * 10 ^ 77) (Ā' := 8084 * 10 ^ 77) (Ē' := 1617 * 10 ^ 78)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4997 * 10 ^ 84) (Ā' := 1595 * 10 ^ 85) (Ē' := 3190 * 10 ^ 85)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 7) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 7) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 8177 * 10 ^ 90) (E1 := 1636 * 10 ^ 91) (A2 := 1391 * 10 ^ 91) (E2 := 2782 * 10 ^ 91)
        (A3 := 1392 * 10 ^ 91) (E3 := 2783 * 10 ^ 91) (A4 := 3761 * 10 ^ 95) (E4 := 7522 * 10 ^ 95)
        (A5 := 5055 * 10 ^ 97) (E5 := 1011 * 10 ^ 98) (A6 := 5056 * 10 ^ 97) (E6 := 1012 * 10 ^ 98)
        (A7 := 1613 * 10 ^ 98) (E7 := 3226 * 10 ^ 98) (A8 := 2743 * 10 ^ 98) (E8 := 5485 * 10 ^ 98)
        (A9 := 2744 * 10 ^ 98) (E9 := 5486 * 10 ^ 98) (A10 := 4215 * 10 ^ 100) (E10 := 8427 * 10 ^ 100)
        (A11 := 4216 * 10 ^ 100) (E11 := 1265 * 10 ^ 101) (A12 := 2591 * 10 ^ 103) (E12 := 7773 * 10 ^ 103)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2563 * 10 ^ 90) (Ā' := 8177 * 10 ^ 90) (Ē' := 1636 * 10 ^ 91)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 5056 * 10 ^ 97) (Ā' := 1613 * 10 ^ 98) (Ē' := 3226 * 10 ^ 98)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 8) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 8) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 8269 * 10 ^ 103) (E1 := 1654 * 10 ^ 104) (A2 := 1406 * 10 ^ 104) (E2 := 2812 * 10 ^ 104)
        (A3 := 1407 * 10 ^ 104) (E3 := 2813 * 10 ^ 104) (A4 := 3801 * 10 ^ 108) (E4 := 7602 * 10 ^ 108)
        (A5 := 5109 * 10 ^ 110) (E5 := 1022 * 10 ^ 111) (A6 := 5110 * 10 ^ 110) (E6 := 1023 * 10 ^ 111)
        (A7 := 1631 * 10 ^ 111) (E7 := 3262 * 10 ^ 111) (A8 := 2773 * 10 ^ 111) (E8 := 5546 * 10 ^ 111)
        (A9 := 2774 * 10 ^ 111) (E9 := 5547 * 10 ^ 111) (A10 := 4261 * 10 ^ 113) (E10 := 8521 * 10 ^ 113)
        (A11 := 4262 * 10 ^ 113) (E11 := 1279 * 10 ^ 114) (A12 := 2619 * 10 ^ 116) (E12 := 7859 * 10 ^ 116)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2592 * 10 ^ 103) (Ā' := 8269 * 10 ^ 103) (Ē' := 1654 * 10 ^ 104)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 5110 * 10 ^ 110) (Ā' := 1631 * 10 ^ 111) (Ē' := 3262 * 10 ^ 111)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 9) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 9) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 8359 * 10 ^ 116) (E1 := 1672 * 10 ^ 117) (A2 := 1422 * 10 ^ 117) (E2 := 2843 * 10 ^ 117)
        (A3 := 1423 * 10 ^ 117) (E3 := 2844 * 10 ^ 117) (A4 := 3844 * 10 ^ 121) (E4 := 7688 * 10 ^ 121)
        (A5 := 5167 * 10 ^ 123) (E5 := 1034 * 10 ^ 124) (A6 := 5168 * 10 ^ 123) (E6 := 1035 * 10 ^ 124)
        (A7 := 1649 * 10 ^ 124) (E7 := 3298 * 10 ^ 124) (A8 := 2804 * 10 ^ 124) (E8 := 5607 * 10 ^ 124)
        (A9 := 2805 * 10 ^ 124) (E9 := 5608 * 10 ^ 124) (A10 := 4309 * 10 ^ 126) (E10 := 8615 * 10 ^ 126)
        (A11 := 4310 * 10 ^ 126) (E11 := 1293 * 10 ^ 127) (A12 := 2649 * 10 ^ 129) (E12 := 7945 * 10 ^ 129)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2620 * 10 ^ 116) (Ā' := 8359 * 10 ^ 116) (Ē' := 1672 * 10 ^ 117)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 5168 * 10 ^ 123) (Ā' := 1649 * 10 ^ 124) (Ē' := 3298 * 10 ^ 124)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 10) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 10) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 8454 * 10 ^ 129) (E1 := 1691 * 10 ^ 130) (A2 := 1438 * 10 ^ 130) (E2 := 2875 * 10 ^ 130)
        (A3 := 1439 * 10 ^ 130) (E3 := 2876 * 10 ^ 130) (A4 := 3888 * 10 ^ 134) (E4 := 7776 * 10 ^ 134)
        (A5 := 5226 * 10 ^ 136) (E5 := 1046 * 10 ^ 137) (A6 := 5227 * 10 ^ 136) (E6 := 1047 * 10 ^ 137)
        (A7 := 1668 * 10 ^ 137) (E7 := 3336 * 10 ^ 137) (A8 := 2836 * 10 ^ 137) (E8 := 5672 * 10 ^ 137)
        (A9 := 2837 * 10 ^ 137) (E9 := 5673 * 10 ^ 137) (A10 := 4358 * 10 ^ 139) (E10 := 8714 * 10 ^ 139)
        (A11 := 4359 * 10 ^ 139) (E11 := 1308 * 10 ^ 140) (A12 := 2679 * 10 ^ 142) (E12 := 8037 * 10 ^ 142)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2650 * 10 ^ 129) (Ā' := 8454 * 10 ^ 129) (Ē' := 1691 * 10 ^ 130)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 5227 * 10 ^ 136) (Ā' := 1668 * 10 ^ 137) (Ē' := 3336 * 10 ^ 137)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))
  · exact
    FloatBridgesTo.Maps.blockVFlatC
      (FloatBridgesTo.Maps.blockVFlat (Np1 := 197) M G.g X.e w.ε (w.blocks 11) (R.lnF M (3 * 64) w.ε)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        P.hwa P.hwm P.hbb P.hegelu G.spec P.heexp0 P.heexp1 X.spec vitScale64 hρ
        (B.hblk 11) (R.bridgeAt M (Xh := 14) P.hε P.hSε (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw)
        (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
        (A1 := 8550 * 10 ^ 142) (E1 := 1710 * 10 ^ 143) (A2 := 1454 * 10 ^ 143) (E2 := 2908 * 10 ^ 143)
        (A3 := 1455 * 10 ^ 143) (E3 := 2909 * 10 ^ 143) (A4 := 3931 * 10 ^ 147) (E4 := 7862 * 10 ^ 147)
        (A5 := 5284 * 10 ^ 149) (E5 := 1057 * 10 ^ 150) (A6 := 5285 * 10 ^ 149) (E6 := 1058 * 10 ^ 150)
        (A7 := 1686 * 10 ^ 150) (E7 := 3372 * 10 ^ 150) (A8 := 2867 * 10 ^ 150) (E8 := 5733 * 10 ^ 150)
        (A9 := 2868 * 10 ^ 150) (E9 := 5734 * 10 ^ 150) (A10 := 4406 * 10 ^ 152) (E10 := 8808 * 10 ^ 152)
        (A11 := 4407 * 10 ^ 152) (E11 := 1322 * 10 ^ 153) (A12 := 2708 * 10 ^ 155) (E12 := 8123 * 10 ^ 155)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2680 * 10 ^ 142) (Ā' := 8550 * 10 ^ 142) (Ē' := 1710 * 10 ^ 143)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 5285 * 10 ^ 149) (Ā' := 1686 * 10 ^ 150) (Ē' := 3372 * 10 ^ 150)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
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
      (vit_smRho_lt_one M hMu)).mag 1 ≤ 1130 * 10 ^ 158 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ViT-Tiny bridge's fresh budget at the committed profile — `2.00 ×` the
    certified window, which is the tell that the cap is biting at every LayerNorm and every
    attention site and the statement is the triangle inequality rather than the fold. -/
theorem vitBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).fresh 1 ≤ 2259 * 10 ^ 158 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐ **The deployed ViT-Tiny forward is within `2.259·10¹⁶¹` of the certified real forward,
    per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile, for
    `ε ≥ 10⁻⁵`, any device LayerNorm statistics accurate to `10⁻²`, any device GELU accurate to
    `10⁻²` and any device `exp` accurate to `10⁻²` RELATIVE, for any rounding model at binary32
    accuracy.

    ⛔ **Read the file header before quoting this.** It is a capped statement — the float and
    real forwards both land in the certified `1.130·10¹⁶¹` window — and NOT the interval fold
    that ResNet-34's, MobileNetV2's and EfficientNet-B0's numbers are. Every LayerNorm and every
    attention site is capped; the patch embed is the one stage that is not. -/
theorem vit_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1/100) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |vitForwardTinyF M R G X w x j - vitForwardTiny w x j| ≤ 2259 * 10 ^ 158 :=
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
    |vitForwardTinyF M R G X w x j - denoteVitTiny vitVerified.layers w x j| ≤ 2259 * 10 ^ 158 := by
  have h := vit_float_logits_le M hMu hε5 R G X w B hεw x hx j
  rwa [vitVerified_denote_eq w]

end Proofs
