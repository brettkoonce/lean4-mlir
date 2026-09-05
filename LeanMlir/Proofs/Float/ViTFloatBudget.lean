import LeanMlir.Proofs.Float.ViTBlockVFloatBridge
import LeanMlir.Proofs.Foundation.SpecVJP

/-! # A NUMBER for ViT-Tiny: the committed depth-12 vector-LN forward, at the cap

The fifth ImageNet-scale whole-net float statement, and ⛔ **it is ConvNeXt-T's kind of
statement, not ResNet-34's.** For the depth-12 `vitForwardKV` at `224²` — `16×16/s16` patch
embed with CLS token and learned positions, twelve pre-norm blocks of
`LN → 3-head attention → skip → LN → fc1 → GELU → fc2 → skip`, a final per-token LayerNorm, the
CLS slice and the classifier — on the unit input window, at the profile measured per parameter
KIND on the trained checkpoint, for any rounding model at binary32 accuracy:

    output window  ≤ 2.397·10¹⁰⁸      (`vitBridge_mag_le`)
    fresh budget   ≤ 4.794·10¹⁰⁸      (`vitBridge_fresh_le`)

and hence, per logit, `|float − real| ≤ 4.794·10¹⁰⁸` (`vit_float_logits_le`).

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
    every LayerNorm's inverse-stddev under `317`; the device `rsqrt`, GELU and `exp` are taken
    accurate to `10⁻²`, and ⭐ the device MEAN to **`1.157·10⁻⁵` relative — DERIVED, not
    supplied** (`deviceLN_emr_committed`, `FloatBudgetEnvLN.lean`): a rounded reduction of `D`
    terms then a divide is within `u·(1+γ)+γ` of the certified mean at the fan-in every summation
    order meets. It was `10⁻²` by analogy with the `rsqrt` until 2026-09-05, and the change is
    worth **53 orders**. ⭐ Uniform costs this net NOTHING, unlike ConvNeXt-T: all 25 LayerNorm
    sites reduce over the same `D = 192`. -/
theorem vitProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    ViTProfile M ε (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10)
      (1/100) (1/100) (1157 / 10 ^ 8) (1/100) 317 u32 where
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
    attention site — are the CAP, not the fold. That is why `4.794·10¹⁰⁸ / 2.397·10¹⁰⁸ = 2.00`.

    ⚠ The window/error sequences are `match`es on the block index rather than a closed form: the
    fold hands block `i`'s output to block `i+1`, and the numerals are what `vit_chain` emits. -/
theorem vitBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1157 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).Maps 1 0 (2397 * 10 ^ 105) (4794 * 10 ^ 105) := by
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
    (Ā := 7739 * 10 ^ 104) (Ē := 2322 * 10 ^ 105)
    (A1 := 1835 * 10 ^ 103) (E1 := 3670 * 10 ^ 103) (A2 := 3120 * 10 ^ 103) (E2 := 6240 * 10 ^ 103)
    (Ā' := 3121 * 10 ^ 103) (Ē' := 6241 * 10 ^ 103)
    (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
      (Ā := 7739 * 10 ^ 104) (Ā' := 1835 * 10 ^ 103) (Ē' := 3670 * 10 ^ 103)
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32])
      (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
    (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (by norm_num [u32]) (by norm_num [u32])
  have mHead := FloatBridgesTo.Maps.vitHead 196 M w.Wcls w.bcls P.hwh P.hbb (by norm_num)
    B.hWcls B.hbcls
    (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 3121 * 10 ^ 103) (Ē := 6241 * 10 ^ 103) (Ā' := 2397 * 10 ^ 105) (Ē' := 4794 * 10 ^ 105)
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
      | 1 => 4677 * 10 ^ 8
      | 2 => 2611 * 10 ^ 17
      | 3 => 1461 * 10 ^ 26
      | 4 => 8157 * 10 ^ 34
      | 5 => 4555 * 10 ^ 43
      | 6 => 2544 * 10 ^ 52
      | 7 => 1423 * 10 ^ 61
      | 8 => 7948 * 10 ^ 69
      | 9 => 4441 * 10 ^ 78
      | 10 => 2480 * 10 ^ 87
      | 11 => 1385 * 10 ^ 96
      | _ => 7739 * 10 ^ 104)
    (fun j => match j with
      | 0 => 5633 / 10 ^ 7
      | 1 => 1404 * 10 ^ 9
      | 2 => 7830 * 10 ^ 17
      | 3 => 4375 * 10 ^ 26
      | 4 => 2450 * 10 ^ 35
      | 5 => 1368 * 10 ^ 44
      | 6 => 7628 * 10 ^ 52
      | 7 => 4262 * 10 ^ 61
      | 8 => 2384 * 10 ^ 70
      | 9 => 1334 * 10 ^ 79
      | 10 => 7437 * 10 ^ 87
      | 11 => 4150 * 10 ^ 96
      | _ => 2322 * 10 ^ 105)
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
        (A1 := 1951 / 10 ^ 2) (E1 := 3902 / 10 ^ 2) (A2 := 3317 / 10 ^ 2) (E2 := 6634 / 10 ^ 2)
        (A3 := 3378 / 10 ^ 2) (E3 := 6635 / 10 ^ 2) (A4 := 9127 * 10 ^ 2) (E4 := 1826 * 10 ^ 3)
        (A5 := 1227 * 10 ^ 5) (E5 := 2455 * 10 ^ 5) (A6 := 1228 * 10 ^ 5) (E6 := 2456 * 10 ^ 5)
        (A7 := 2912 * 10 ^ 3) (E7 := 5824 * 10 ^ 3) (A8 := 4951 * 10 ^ 3) (E8 := 9901 * 10 ^ 3)
        (A9 := 4952 * 10 ^ 3) (E9 := 9902 * 10 ^ 3) (A10 := 7607 * 10 ^ 5) (E10 := 1521 * 10 ^ 6)
        (A11 := 7608 * 10 ^ 5) (E11 := 2282 * 10 ^ 6) (A12 := 4675 * 10 ^ 8) (E12 := 1403 * 10 ^ 9)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2323 / 10 ^ 1) (Ā' := 1951 / 10 ^ 2) (Ē' := 3902 / 10 ^ 2)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 1228 * 10 ^ 5) (Ā' := 2912 * 10 ^ 3) (Ē' := 5824 * 10 ^ 3)
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
        (A1 := 1109 * 10 ^ 7) (E1 := 2218 * 10 ^ 7) (A2 := 1886 * 10 ^ 7) (E2 := 3771 * 10 ^ 7)
        (A3 := 1887 * 10 ^ 7) (E3 := 3772 * 10 ^ 7) (A4 := 5098 * 10 ^ 11) (E4 := 1020 * 10 ^ 12)
        (A5 := 6852 * 10 ^ 13) (E5 := 1371 * 10 ^ 14) (A6 := 6853 * 10 ^ 13) (E6 := 1372 * 10 ^ 14)
        (A7 := 1625 * 10 ^ 12) (E7 := 3250 * 10 ^ 12) (A8 := 2763 * 10 ^ 12) (E8 := 5526 * 10 ^ 12)
        (A9 := 2764 * 10 ^ 12) (E9 := 5527 * 10 ^ 12) (A10 := 4246 * 10 ^ 14) (E10 := 8490 * 10 ^ 14)
        (A11 := 4247 * 10 ^ 14) (E11 := 1274 * 10 ^ 15) (A12 := 2610 * 10 ^ 17) (E12 := 7828 * 10 ^ 17)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4677 * 10 ^ 8) (Ā' := 1109 * 10 ^ 7) (Ē' := 2218 * 10 ^ 7)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 6853 * 10 ^ 13) (Ā' := 1625 * 10 ^ 12) (Ē' := 3250 * 10 ^ 12)
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
        (A1 := 6190 * 10 ^ 15) (E1 := 1238 * 10 ^ 16) (A2 := 1053 * 10 ^ 16) (E2 := 2105 * 10 ^ 16)
        (A3 := 1054 * 10 ^ 16) (E3 := 2106 * 10 ^ 16) (A4 := 2848 * 10 ^ 20) (E4 := 5696 * 10 ^ 20)
        (A5 := 3828 * 10 ^ 22) (E5 := 7656 * 10 ^ 22) (A6 := 3829 * 10 ^ 22) (E6 := 7657 * 10 ^ 22)
        (A7 := 9077 * 10 ^ 20) (E7 := 1816 * 10 ^ 21) (A8 := 1544 * 10 ^ 21) (E8 := 3088 * 10 ^ 21)
        (A9 := 1545 * 10 ^ 21) (E9 := 3089 * 10 ^ 21) (A10 := 2374 * 10 ^ 23) (E10 := 4745 * 10 ^ 23)
        (A11 := 2375 * 10 ^ 23) (E11 := 7118 * 10 ^ 23) (A12 := 1460 * 10 ^ 26) (E12 := 4374 * 10 ^ 26)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2611 * 10 ^ 17) (Ā' := 6190 * 10 ^ 15) (Ē' := 1238 * 10 ^ 16)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 3829 * 10 ^ 22) (Ā' := 9077 * 10 ^ 20) (Ē' := 1816 * 10 ^ 21)
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
        (A1 := 3464 * 10 ^ 24) (E1 := 6928 * 10 ^ 24) (A2 := 5889 * 10 ^ 24) (E2 := 1178 * 10 ^ 25)
        (A3 := 5890 * 10 ^ 24) (E3 := 1179 * 10 ^ 25) (A4 := 1592 * 10 ^ 29) (E4 := 3184 * 10 ^ 29)
        (A5 := 2140 * 10 ^ 31) (E5 := 4280 * 10 ^ 31) (A6 := 2141 * 10 ^ 31) (E6 := 4281 * 10 ^ 31)
        (A7 := 5076 * 10 ^ 29) (E7 := 1016 * 10 ^ 30) (A8 := 8630 * 10 ^ 29) (E8 := 1728 * 10 ^ 30)
        (A9 := 8631 * 10 ^ 29) (E9 := 1729 * 10 ^ 30) (A10 := 1326 * 10 ^ 32) (E10 := 2656 * 10 ^ 32)
        (A11 := 1327 * 10 ^ 32) (E11 := 3985 * 10 ^ 32) (A12 := 8154 * 10 ^ 34) (E12 := 2449 * 10 ^ 35)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 1461 * 10 ^ 26) (Ā' := 3464 * 10 ^ 24) (Ē' := 6928 * 10 ^ 24)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2141 * 10 ^ 31) (Ā' := 5076 * 10 ^ 29) (Ē' := 1016 * 10 ^ 30)
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
        (A1 := 1934 * 10 ^ 33) (E1 := 3868 * 10 ^ 33) (A2 := 3288 * 10 ^ 33) (E2 := 6576 * 10 ^ 33)
        (A3 := 3289 * 10 ^ 33) (E3 := 6577 * 10 ^ 33) (A4 := 8885 * 10 ^ 37) (E4 := 1777 * 10 ^ 38)
        (A5 := 1195 * 10 ^ 40) (E5 := 2389 * 10 ^ 40) (A6 := 1196 * 10 ^ 40) (E6 := 2390 * 10 ^ 40)
        (A7 := 2836 * 10 ^ 38) (E7 := 5672 * 10 ^ 38) (A8 := 4822 * 10 ^ 38) (E8 := 9643 * 10 ^ 38)
        (A9 := 4823 * 10 ^ 38) (E9 := 9644 * 10 ^ 38) (A10 := 7409 * 10 ^ 40) (E10 := 1482 * 10 ^ 41)
        (A11 := 7410 * 10 ^ 40) (E11 := 2224 * 10 ^ 41) (A12 := 4553 * 10 ^ 43) (E12 := 1367 * 10 ^ 44)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 8157 * 10 ^ 34) (Ā' := 1934 * 10 ^ 33) (Ē' := 3868 * 10 ^ 33)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 1196 * 10 ^ 40) (Ā' := 2836 * 10 ^ 38) (Ē' := 5672 * 10 ^ 38)
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
        (A1 := 1080 * 10 ^ 42) (E1 := 2160 * 10 ^ 42) (A2 := 1837 * 10 ^ 42) (E2 := 3673 * 10 ^ 42)
        (A3 := 1838 * 10 ^ 42) (E3 := 3674 * 10 ^ 42) (A4 := 4965 * 10 ^ 46) (E4 := 9930 * 10 ^ 46)
        (A5 := 6674 * 10 ^ 48) (E5 := 1335 * 10 ^ 49) (A6 := 6675 * 10 ^ 48) (E6 := 1336 * 10 ^ 49)
        (A7 := 1583 * 10 ^ 47) (E7 := 3166 * 10 ^ 47) (A8 := 2692 * 10 ^ 47) (E8 := 5383 * 10 ^ 47)
        (A9 := 2693 * 10 ^ 47) (E9 := 5384 * 10 ^ 47) (A10 := 4137 * 10 ^ 49) (E10 := 8270 * 10 ^ 49)
        (A11 := 4138 * 10 ^ 49) (E11 := 1241 * 10 ^ 50) (A12 := 2543 * 10 ^ 52) (E12 := 7626 * 10 ^ 52)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4555 * 10 ^ 43) (Ā' := 1080 * 10 ^ 42) (Ē' := 2160 * 10 ^ 42)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 6675 * 10 ^ 48) (Ā' := 1583 * 10 ^ 47) (Ē' := 3166 * 10 ^ 47)
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
        (A1 := 6031 * 10 ^ 50) (E1 := 1207 * 10 ^ 51) (A2 := 1026 * 10 ^ 51) (E2 := 2052 * 10 ^ 51)
        (A3 := 1027 * 10 ^ 51) (E3 := 2053 * 10 ^ 51) (A4 := 2775 * 10 ^ 55) (E4 := 5550 * 10 ^ 55)
        (A5 := 3730 * 10 ^ 57) (E5 := 7460 * 10 ^ 57) (A6 := 3731 * 10 ^ 57) (E6 := 7461 * 10 ^ 57)
        (A7 := 8845 * 10 ^ 55) (E7 := 1769 * 10 ^ 56) (A8 := 1504 * 10 ^ 56) (E8 := 3008 * 10 ^ 56)
        (A9 := 1505 * 10 ^ 56) (E9 := 3009 * 10 ^ 56) (A10 := 2312 * 10 ^ 58) (E10 := 4622 * 10 ^ 58)
        (A11 := 2313 * 10 ^ 58) (E11 := 6934 * 10 ^ 58) (A12 := 1422 * 10 ^ 61) (E12 := 4261 * 10 ^ 61)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2544 * 10 ^ 52) (Ā' := 6031 * 10 ^ 50) (Ē' := 1207 * 10 ^ 51)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 3731 * 10 ^ 57) (Ā' := 8845 * 10 ^ 55) (Ē' := 1769 * 10 ^ 56)
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
        (A1 := 3374 * 10 ^ 59) (E1 := 6748 * 10 ^ 59) (A2 := 5736 * 10 ^ 59) (E2 := 1148 * 10 ^ 60)
        (A3 := 5737 * 10 ^ 59) (E3 := 1149 * 10 ^ 60) (A4 := 1550 * 10 ^ 64) (E4 := 3100 * 10 ^ 64)
        (A5 := 2084 * 10 ^ 66) (E5 := 4167 * 10 ^ 66) (A6 := 2085 * 10 ^ 66) (E6 := 4168 * 10 ^ 66)
        (A7 := 4943 * 10 ^ 64) (E7 := 9886 * 10 ^ 64) (A8 := 8404 * 10 ^ 64) (E8 := 1681 * 10 ^ 65)
        (A9 := 8405 * 10 ^ 64) (E9 := 1682 * 10 ^ 65) (A10 := 1292 * 10 ^ 67) (E10 := 2584 * 10 ^ 67)
        (A11 := 1293 * 10 ^ 67) (E11 := 3877 * 10 ^ 67) (A12 := 7945 * 10 ^ 69) (E12 := 2383 * 10 ^ 70)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 1423 * 10 ^ 61) (Ā' := 3374 * 10 ^ 59) (Ē' := 6748 * 10 ^ 59)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2085 * 10 ^ 66) (Ā' := 4943 * 10 ^ 64) (Ē' := 9886 * 10 ^ 64)
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
        (A1 := 1885 * 10 ^ 68) (E1 := 3770 * 10 ^ 68) (A2 := 3205 * 10 ^ 68) (E2 := 6410 * 10 ^ 68)
        (A3 := 3206 * 10 ^ 68) (E3 := 6411 * 10 ^ 68) (A4 := 8661 * 10 ^ 72) (E4 := 1733 * 10 ^ 73)
        (A5 := 1165 * 10 ^ 75) (E5 := 2330 * 10 ^ 75) (A6 := 1166 * 10 ^ 75) (E6 := 2331 * 10 ^ 75)
        (A7 := 2765 * 10 ^ 73) (E7 := 5530 * 10 ^ 73) (A8 := 4701 * 10 ^ 73) (E8 := 9402 * 10 ^ 73)
        (A9 := 4702 * 10 ^ 73) (E9 := 9403 * 10 ^ 73) (A10 := 7223 * 10 ^ 75) (E10 := 1445 * 10 ^ 76)
        (A11 := 7224 * 10 ^ 75) (E11 := 2168 * 10 ^ 76) (A12 := 4439 * 10 ^ 78) (E12 := 1333 * 10 ^ 79)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 7948 * 10 ^ 69) (Ā' := 1885 * 10 ^ 68) (Ē' := 3770 * 10 ^ 68)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 1166 * 10 ^ 75) (Ā' := 2765 * 10 ^ 73) (Ē' := 5530 * 10 ^ 73)
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
        (A1 := 1053 * 10 ^ 77) (E1 := 2106 * 10 ^ 77) (A2 := 1791 * 10 ^ 77) (E2 := 3581 * 10 ^ 77)
        (A3 := 1792 * 10 ^ 77) (E3 := 3582 * 10 ^ 77) (A4 := 4841 * 10 ^ 81) (E4 := 9682 * 10 ^ 81)
        (A5 := 6507 * 10 ^ 83) (E5 := 1302 * 10 ^ 84) (A6 := 6508 * 10 ^ 83) (E6 := 1303 * 10 ^ 84)
        (A7 := 1543 * 10 ^ 82) (E7 := 3086 * 10 ^ 82) (A8 := 2624 * 10 ^ 82) (E8 := 5247 * 10 ^ 82)
        (A9 := 2625 * 10 ^ 82) (E9 := 5248 * 10 ^ 82) (A10 := 4033 * 10 ^ 84) (E10 := 8062 * 10 ^ 84)
        (A11 := 4034 * 10 ^ 84) (E11 := 1210 * 10 ^ 85) (A12 := 2479 * 10 ^ 87) (E12 := 7435 * 10 ^ 87)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 4441 * 10 ^ 78) (Ā' := 1053 * 10 ^ 77) (Ē' := 2106 * 10 ^ 77)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 6508 * 10 ^ 83) (Ā' := 1543 * 10 ^ 82) (Ē' := 3086 * 10 ^ 82)
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
        (A1 := 5879 * 10 ^ 85) (E1 := 1176 * 10 ^ 86) (A2 := 9995 * 10 ^ 85) (E2 := 2000 * 10 ^ 86)
        (A3 := 9996 * 10 ^ 85) (E3 := 2001 * 10 ^ 86) (A4 := 2701 * 10 ^ 90) (E4 := 5402 * 10 ^ 90)
        (A5 := 3631 * 10 ^ 92) (E5 := 7261 * 10 ^ 92) (A6 := 3632 * 10 ^ 92) (E6 := 7262 * 10 ^ 92)
        (A7 := 8610 * 10 ^ 90) (E7 := 1722 * 10 ^ 91) (A8 := 1464 * 10 ^ 91) (E8 := 2928 * 10 ^ 91)
        (A9 := 1465 * 10 ^ 91) (E9 := 2929 * 10 ^ 91) (A10 := 2251 * 10 ^ 93) (E10 := 4500 * 10 ^ 93)
        (A11 := 2252 * 10 ^ 93) (E11 := 6751 * 10 ^ 93) (A12 := 1384 * 10 ^ 96) (E12 := 4149 * 10 ^ 96)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2480 * 10 ^ 87) (Ā' := 5879 * 10 ^ 85) (Ē' := 1176 * 10 ^ 86)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 3632 * 10 ^ 92) (Ā' := 8610 * 10 ^ 90) (Ē' := 1722 * 10 ^ 91)
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
        (A1 := 3284 * 10 ^ 94) (E1 := 6568 * 10 ^ 94) (A2 := 5583 * 10 ^ 94) (E2 := 1117 * 10 ^ 95)
        (A3 := 5584 * 10 ^ 94) (E3 := 1118 * 10 ^ 95) (A4 := 1509 * 10 ^ 99) (E4 := 3018 * 10 ^ 99)
        (A5 := 2029 * 10 ^ 101) (E5 := 4057 * 10 ^ 101) (A6 := 2030 * 10 ^ 101) (E6 := 4058 * 10 ^ 101)
        (A7 := 4813 * 10 ^ 99) (E7 := 9626 * 10 ^ 99) (A8 := 8183 * 10 ^ 99) (E8 := 1637 * 10 ^ 100)
        (A9 := 8184 * 10 ^ 99) (E9 := 1638 * 10 ^ 100) (A10 := 1258 * 10 ^ 102) (E10 := 2517 * 10 ^ 102)
        (A11 := 1259 * 10 ^ 102) (E11 := 3776 * 10 ^ 102) (A12 := 7736 * 10 ^ 104) (E12 := 2321 * 10 ^ 105)
        P.hq P.hgl P.hbl (by norm_num)
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 1385 * 10 ^ 96) (Ā' := 3284 * 10 ^ 94) (Ē' := 6568 * 10 ^ 94)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32])
        (R.mapsAt M (Xh := 14) P.hemr P.hε P.hSε P.hS0 P.hq (3 * 64) (by norm_num) (by norm_num) (by norm_num) w.ε hεw
          (Ā := 2030 * 10 ^ 101) (Ā' := 4813 * 10 ^ 99) (Ē' := 9626 * 10 ^ 99)
          (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]) (by norm_num [bnNormBudgetX, bnXhatErr, bnProdErr, bnCentErr, FloatModel.mulErr, u32]))
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [u32]) (by norm_num [u32])
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num) (by norm_num [u32]) (by norm_num [u32]))

/-- The deployed ViT-Tiny bridge's certified output window at the committed profile. -/
theorem vitBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1157 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).mag 1 ≤ 2397 * 10 ^ 105 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).mag_le 1 (by norm_num) le_rfl

/-- ⛔ The deployed ViT-Tiny bridge's fresh budget at the committed profile — `2.00 ×` the
    certified window, which is the tell that the cap is biting at every LayerNorm and every
    attention site and the statement is the triangle inequality rather than the fold. -/
theorem vitBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1157 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) :
    (vitBridge M R G X (vitProfile_committed M hMu hε5) w B hεw
      (vit_smRho_lt_one M hMu)).fresh 1 ≤ 4794 * 10 ^ 105 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐ **The deployed ViT-Tiny forward is within `4.794·10¹⁰⁸` of the certified real forward,
    per logit**, on inputs of magnitude `≤ 1`, at the measured parameter profile, for
    `ε ≥ 10⁻⁵`, any device LayerNorm statistics accurate to `10⁻²`, any device GELU accurate to
    `10⁻²` and any device `exp` accurate to `10⁻²` RELATIVE, for any rounding model at binary32
    accuracy.

    ⛔ **Read the file header before quoting this.** It is a capped statement — the float and
    real forwards both land in the certified `2.397·10¹⁰⁸` window — and NOT the interval fold
    that ResNet-34's, MobileNetV2's and EfficientNet-B0's numbers are. Every LayerNorm and every
    attention site is capped; the patch embed is the one stage that is not. -/
theorem vit_float_logits_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (R : DeviceLN (1157 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100)) (X : DeviceExp (1/100))
    (w : ViTTinyWeights) (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |vitForwardTinyF M R G X w x j - vitForwardTiny w x j| ≤ 4794 * 10 ^ 105 :=
  (vitBridge_maps M hMu hε5 R G X w B hεw).budget_le (by norm_num) le_rfl x hx j

-- ════════════════════════════════════════════════════════════════
-- § The tie: this IS the committed ViT-Tiny spec's forward
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The number, stated about the committed spec's denotation.** `vitVerified_denote_eq` is
    `rfl`, and `vitVerified_fwd_faithful` says the emitted depth-12 multi-head vector-LN graph
    denotes the same function — so the budget is a claim about the net `ViTRender.lean` renders,
    not about a record-plugged look-alike. -/
theorem vit_float_logits_le_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) (R : DeviceLN (1157 / 10 ^ 8) (1/100)) (G : DeviceGelu (1/100))
    (X : DeviceExp (1/100)) (w : ViTTinyWeights)
    (B : ViTBounded w (7/10) (8/10) (3/10) (4/10) (9/10) (17/10) (6/10) (9/10))
    (hεw : ε ≤ w.ε) (x : Vec (3 * 224 * 224)) (hx : ∀ k, |x k| ≤ 1) (j : Fin 10) :
    |vitForwardTinyF M R G X w x j - denoteVitTiny vitVerified.layers w x j| ≤ 4794 * 10 ^ 105 := by
  have h := vit_float_logits_le M hMu hε5 R G X w B hεw x hx j
  rwa [vitVerified_denote_eq w]

end Proofs
