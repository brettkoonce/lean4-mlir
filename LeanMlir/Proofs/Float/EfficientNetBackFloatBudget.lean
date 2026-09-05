import LeanMlir.Proofs.Float.FloatBudgetEnvBackSE
import LeanMlir.Proofs.Float.Binary32Instance

/-! # A NUMBER for EfficientNet-B0's whole-net BACKWARD — the fourth, and the SQUEEZE-EXCITE one

The backward peer of `EfficientNetFloatBudget.lean`, and the fourth whole-net input-gradient
number in the repo after ResNet-34's (`Resnet34BackFloatBudget.lean`), MobileNetV2's
(`MobileNetV2BackFloatBudget.lean`) and ConvNeXt-T's (`ConvNeXtBackFloatBudget.lean`).

    certified window ≤ 7.104·10¹⁸²      (`b0GradBridge_mag_le`)
    fresh budget     ≤ 1.578·10¹⁸²      (`b0GradBridge_fresh_le`)
    budget / window  = 0.222            — ⭐ the interval FOLD, not a cap

at `|W| ≤ 37/10`, BatchNorm `|γ| ≤ 41/10`, `ε ≥ 10⁻⁵`, `u ≤ 2⁻²⁴`, on loss cotangents of
magnitude `≤ 1` (`|p − y| ≤ 1` for softmax cross-entropy), at **training-mode BatchNorm** — the
mode this net's own forward has no statable number for at all
(`planning/archive/float_budget_numbers_log.md` §0.1). 59 numeric stages, 138 rational inequalities,
generated and re-asserted by `b0_back_chain` / `verify_b0_back` before a line of Lean was written.

**⭐⭐ Two things make this net's backward the interesting one.**

**(1) It is the SQUEEZE-EXCITE fold, and squeeze-excite was the reason to doubt that a backward
folds at all.** §0.1 lists SE as the third site whose FORWARD modulus is quadratic in the window:
`seScale`'s `A · Eg` multiplies the block window by the gate's error, and the gate is grown out of
that same window by the squeeze's `GAP → dense`. Its backward fans out and rejoins, so it looked
like the counterexample. It is not one:

    seInputGrad g xinp gateBack = biPathSum (diagBack g) (gateBack ∘ diagBack xinp)

— the gate `g` and the SE's input `xinp` are **saved constants**, both branches are `diagBack`
scales of the cotangent, and nothing multiplies the cotangent by itself. `budget / window = 0.222`
with **no `capped` anywhere**, at three SE sites. §0.1's closing sentence, as a theorem.

**(2) ⭐⭐ `|swish′| ≤ 2` is what makes it EXIST, and the crudeness of the constant is the
point.** The repo's only prior bound was `swishScalar_lipschitz_abs`'s `1 + A/4` at the forward's
pre-swish window — `1.216·10⁵¹` at the head alone — and with it this fold is `10⁴³¹`, past
`norm_num`'s ceiling and unstatable. `swishScalarDeriv_abs_le` (`Architectures/SwishSaturation.lean`)
is GLOBAL and window-free, and it is worth **262 orders**. ⭐ The nine `diagBack` slots below take
the REAL saved derivative — `fun i => swishScalarDeriv (xpre i)` — so `Ssw = 2` is discharged, not
assumed. ⛔ The sharp `≈ 1.0998` is 2.6 orders better and §3.12 says explicitly not to prove it.

**⭐⭐ And it needs NO OPERATING POINT — the second backward in the repo that does not, after
MobileNetV2's.** `|istd| ≤ 317` is a THEOREM from `ε ≥ 10⁻⁵` alone (`EnetBnBack.hS`, via
`bnIstd_abs_le_of`), where ResNet-34's and ConvNeXt-T's numbers both assume `|istd| ≤ 16`.
The ε-floor fold lands at `10¹⁸²`, **70 orders under §3.7(a)'s ~10²⁵³ shape-dependent ceiling**, so
§3.13's rule applies rather than being inherited: *an operating-point hypothesis is not a property
of backwards, it is what you pay when the ε-floor fold does not fit.* ⚠ `b0_back_chain`'s `S = 16`
default is ResNet-34's, and §3.12's `7.640·10¹⁶⁹ / 1.735·10¹⁶⁹` is that stronger hypothesis' row;
the number here is the weaker-hypothesis one and the theorem is strictly better for it.

**⛔⛔ THE NUMBER IS AT `N = 1`, AND THAT QUALIFIER IS NOT COSMETIC.** `b0_float_logits_le` holds
at **any** batch size, because at inference every stage of `efficientnetForwardBEval` is
`batchMap N` of a per-example op. This fold is at TRAINING-mode BatchNorm, and `bnBatchLA` is the
ONE op in this net that is not `batchMap N` of a per-example op — it reduces μ/var **across**
examples, so each BatchNorm site's per-channel width is `N·h·w`, not `h·w`. `bnGradInputReMag`'s
gain is `S·G·(2 + Xh²)` with `Xh² = n`, so all nine sites scale with `N`: 7.104·10¹⁸² at `N = 1`,
and (at the inherited `S = 16`) 9.877·10¹⁸⁴ at `N = 32`, 2.880·10¹⁹⁴ at `N = 256` — a fold and
statable throughout, but a different number per batch size. ⭐ At `N = 1` the two widths coincide,
which is why the per-example `floatBridgesTo_bnPerChannelBack` at width `h·w` is the right leaf
here and why the chain is `StableHLO.batchMap 1` of per-example maps. §3.24 has the table;
`b0_back_chain(N := ...)` carries it.

⛔ **Two hypotheses remain and they are the caveat, exactly as on ResNet-34 and MobileNetV2.**
`es` and `exh` — the accuracies of the deployed float inverse-stddev and normalised activation
*read off the saved forward activations* — are SUPPLIED at `10⁻²`, and this net's own
training-mode forward fold cannot discharge them. ⭐ But unlike ConvNeXt-T (§3.16 finding 1), B0
HAS an inference mode in which its forward statement is an honest fold (`b0_float_logits_le`), so
this is r34's and MobileNetV2's *quantitative* gap and not a gap in KIND. Say it that way.
⚠ A third supplied quantity, `esav`, covers every saved vector the backward scales by that is not
a BatchNorm statistic: the SE gate `g`, the SE's input `x`, `σ′(saved)` and `swish′(saved)`.

⚠ **The one remaining WINDOW import is the SE's saved input.** `Sx` is the forward's certified
pre-swish window at the block — `4.903·10⁴⁰` / `5.451·10²⁴` / `7.572·10⁹` for `b3`/`b2`/`b1` — and
it is the second of §3.9's two window imports (the first, `|swish′|`, is gone). An operating point
on it (`|x| ≤ 16`) is worth a further ~71 orders and is deliberately NOT taken: §3.13's rule
again.

⭐ **Stated DIRECTLY on the committed `efficientnetInputGradB`**, as ConvNeXt-T's is on
`convnextInputGrad` (§3.22) and unlike r34's and MobileNetV2's, which define their own `*GradR`
skeletons. `b0GradR` fills the committed def's eleven slots and the number is about that term, so
B0's whole-net certified tie — against `efficientnetForwardB_has_vjp`
(`Architectures/EfficientNetChainClose.lean`), ⛔ **not** the 16-block
`efficientnetForwardB_full_has_vjp` — will make it a statement about the certified gradient with
nothing in between. ⚠ That tie is still open: the only existing block tie
(`mbconvBodyBack_eq_mbconvBody_vjp`) is stated per-example at scalar `bnForward`, so all three
batched ties at `bnBatchLA` are missing (§3.24 correction 2).
-/

namespace Proofs

open FloatModel
open Classical

-- ════════════════════════════════════════════════════════════════
-- § The numeric profile, and the net's stored parameters and saved state
-- ════════════════════════════════════════════════════════════════

/-- The numeric profile the EfficientNet-B0 backward fold runs at. ⭐ Note what is NOT here, as in
    `MnvBackProfile` and unlike `R34BackProfile`/`CnxBackProfile`: an inverse-stddev bound. The
    ε-floor fold is `10¹⁸²`, 70 orders under `norm_num`'s ceiling, so `S = 317` is a consequence of
    `hε5` and not a choice (`EnetBnBack.hS`). ⛔ `es`, `exh` and `esav` are the caveat, not the
    machinery — see the file header. -/
structure EnetBackProfile (M : FloatModel) (ε wk gl es exh esav q : ℝ) : Prop where
  /-- Conv, depthwise and dense kernels — the backward has no bias anywhere. -/
  hwk : 0 ≤ wk
  /-- BatchNorm γ. -/
  hgl : 0 ≤ gl
  /-- ⛔ The float inverse-stddev's accuracy. SUPPLIED. -/
  hes : 0 ≤ es
  /-- ⛔ The float normalised activation's accuracy. SUPPLIED. -/
  hexh : 0 ≤ exh
  /-- ⛔ The accuracy of every OTHER saved vector the backward scales by — the SE gate, the SE's
      input, `σ′(saved)` and `swish′(saved)`. SUPPLIED. -/
  hesav : 0 ≤ esav
  /-- ⭐ The `ε`-floor, and the ONLY thing bounding the inverse-stddev. -/
  hε5 : 1 / 100000 ≤ ε
  hq : M.u ≤ q

/-- `ε` is positive — the `ε`-floor's immediate consequence. -/
theorem EnetBackProfile.hε {M : FloatModel} {ε wk gl es exh esav q : ℝ}
    (P : EnetBackProfile M ε wk gl es exh esav q) : 0 < ε :=
  lt_of_lt_of_le (by norm_num) P.hε5

/-- A conv kernel with its magnitude bound (bias-free: `convFlatBack` is stated at bias `0`). -/
structure EnetKerB (oc ic kH kW : Nat) (wk : ℝ) where
  W : Kernel4 oc ic kH kW
  hW : ∀ o c kh kw, |W o c kh kw| ≤ wk

/-- A depthwise kernel with its magnitude bound. -/
structure EnetDwKerB (c kH kW : Nat) (wk : ℝ) where
  W : DepthwiseKernel c kH kW
  hW : ∀ ch kh kw, |W ch kh kw| ≤ wk

/-- The classifier kernel with its bound. -/
structure EnetHeadB (m n : Nat) (wk : ℝ) where
  W : Mat m n
  hW : ∀ i j, |W i j| ≤ wk

/-- ⭐ **One per-channel BatchNorm BACKWARD site — with NO inverse-stddev field.** γ, the saved
    forward activation `x`, and the deployed float statistics computed from it with their two
    SUPPLIED accuracies. ⛔ Nothing bounds `x` and nothing needs to: `x̂` is under `√(h·w)` by
    standardisation (`bnXhat_sq_le`, load-bearing for the fourth net running) and `istd` under
    `1/√ε` by the `ε`-floor, whatever `x` was.
    ⚠ At `N = 1` this per-example site IS the batched `bnBatchLA`'s reduction width; at `N > 1` it
    is not, which is the whole of the file header's batch caveat. -/
structure EnetBnBack (c h w : Nat) (ε gl es exh : ℝ) where
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
    `1/317² = 1/100489 ≥ 1/100000`, so `bnIstd_abs_le_of` closes it with room. ⛔ This is the whole
    of "EfficientNet-B0's backward needs no operating-point hypothesis", and it is the second net
    to have it (`MnvBnBack.hS` was the first): on ResNet-34 the same statement at `S = 317` gives
    `5.503·10²⁸⁸` and on ConvNeXt-T `6.847·10²⁸⁰`, both past §3.7(a)'s `norm_num` ceiling, which is
    why those two files assume `|istd| ≤ 16` instead. -/
theorem EnetBnBack.hS {c h w : Nat} {ε gl es exh : ℝ} (s : EnetBnBack c h w ε gl es exh)
    {M : FloatModel} {wk esav q : ℝ} (P : EnetBackProfile M ε wk gl es exh esav q) :
    ∀ k, |bnIstd (h * w) (Mat.unflatten (reassocFwd c h w s.x) k) ε| ≤ 317 :=
  fun _ => bnIstd_abs_le_of _ (by norm_num) (by norm_num; linarith [P.hε5])

/-- ⭐⭐ **One swish BACKWARD site: the SAVED PRE-ACTIVATION, not a bound.** The site's real map is
    `diagBack (swish′(xpre))` and its `Ssw = 2` is discharged by `swishScalarDeriv_abs_le` — so
    this record carries no magnitude hypothesis at all, only the deployed float vector's accuracy.
    ⛔ That is the difference between a statable fold and `10⁴³¹`: the repo's other bound,
    `swishScalar_lipschitz_abs`'s `1 + A/4`, imports the FORWARD's certified window (§3.9 finding 3,
    §3.12). -/
structure EnetSwBack (n : Nat) (esav : ℝ) where
  /-- The SAVED pre-swish activation this stage differentiates at. -/
  xpre : Vec n
  /-- The deployed float saved derivative. -/
  fsw : Vec n
  hfsw : ∀ i, |fsw i - swishScalarDeriv (xpre i)| ≤ esav

/-- ⭐⭐ **One squeeze-excite BACKWARD site.** The two gate denses, the two saved activation
    derivatives the gate path scales by (`σ′` at the `c` channels, `swish′` at the `r` reduced
    ones), and the product rule's two saved multipliers — the gate `g` and the block's own input
    `x`.

    ⚠ `Sx` — the bound on the SE's saved input — is the fold's ONE remaining window import, and
    it is per block: the forward's certified pre-swish window there. Everything else in this record
    is bounded by something the architecture supplies: `|g| ≤ 1 + esig` because a sigmoid gate
    cannot exceed one, `|swish′| ≤ 2` by `swishScalarDeriv_abs_le`, and `|σ′| ≤ 1/4` because
    `σ′ = σ(1−σ)`. ⚠ That last one is a HYPOTHESIS and not a theorem: `sigmoidScalar_lipschitz`
    proves the constant `1/4` as a Lipschitz modulus, but the pointwise
    `|sigmoidScalarDeriv x| ≤ 1/4` is not in the repo. It is not load-bearing — it multiplies one
    stage of the gate path and never the window — so it enters as the saved vector's bound. -/
structure EnetSeBack (c r h w : Nat) (wk Sx esav : ℝ) where
  /-- The squeeze's expand dense, `Mat c r` — its backward's fan-in is `r`. -/
  W₁ : Mat c r
  /-- The squeeze's reduce dense, `Mat r c` — its backward's fan-in is `c`. -/
  W₂ : Mat r c
  hW₁ : ∀ i j, |W₁ i j| ≤ wk
  hW₂ : ∀ i j, |W₂ i j| ≤ wk
  /-- The saved sigmoid derivative, and the deployed float peer. -/
  ssig : Vec c
  fssig : Vec c
  hssig : ∀ i, |ssig i| ≤ 1 / 4
  hfssig : ∀ i, |fssig i - ssig i| ≤ esav
  /-- ⭐ The saved pre-swish activation inside the gate — the derivative is the REAL one. -/
  xsw : Vec r
  fssw : Vec r
  hfssw : ∀ i, |fssw i - swishScalarDeriv (xsw i)| ≤ esav
  /-- The saved gate, `σ(…)`, and the deployed float peer. -/
  gate : Vec (c * h * w)
  fgate : Vec (c * h * w)
  hgate : ∀ i, |gate i| ≤ 101 / 100
  hfgate : ∀ i, |fgate i - gate i| ≤ esav
  /-- ⚠ The saved SE INPUT, and its bound is the forward's window. -/
  xinp : Vec (c * h * w)
  fxinp : Vec (c * h * w)
  hxinp : ∀ i, |xinp i| ≤ Sx
  hfxinp : ∀ i, |fxinp i - xinp i| ≤ esav

/-- B0's `b1`: one NO-EXPAND MBConv body's backward data. The 3×3 depthwise, the 1×1 project, two
    BatchNorm sites, one swish site and the squeeze-excite. ⚠ No expand arm and no skip. -/
structure EnetNoExpBack (cin cout r h w : Nat) (ε wk gl es exh esav Sx : ℝ) where
  dw : EnetDwKerB cin 3 3 wk
  kp : EnetKerB cout cin 1 1 wk
  bnp : EnetBnBack cout h w ε gl es exh
  se : EnetSeBack cin r h w wk Sx esav
  swd : EnetSwBack (cin * h * w) esav
  bnd : EnetBnBack cin h w ε gl es exh

/-- B0's `b2`: one STRIDED MBConv body's backward data. ⚠ The expand BatchNorm and its swish live
    at the DOUBLED grid — the stride change happens inside the body, between the depthwise and the
    expand — so `bne`/`swe` carry their own `Xh`. -/
structure EnetStridedBack (cin cmid cout r h w : Nat) (ε wk gl es exh esav Sx : ℝ) where
  ke : EnetKerB cmid cin 1 1 wk
  dw : EnetDwKerB cmid 3 3 wk
  kp : EnetKerB cout cmid 1 1 wk
  bnp : EnetBnBack cout h w ε gl es exh
  se : EnetSeBack cmid r h w wk Sx esav
  swd : EnetSwBack (cmid * h * w) esav
  bnd : EnetBnBack cmid h w ε gl es exh
  swe : EnetSwBack (cmid * (2 * h) * (2 * w)) esav
  bne : EnetBnBack cmid (2 * h) (2 * w) ε gl es exh

/-- B0's `b3`: one RESIDUAL MBConv body's backward data, all at one resolution and with a 5×5
    depthwise. ⚠ The additive skip is NOT here — the caller wraps the body in `Proofs.residual`,
    exactly as MobileNetV2's `b2`/`b4` do (§3.13: the block record does not have to own its
    skip). -/
structure EnetResidBack (c cmid r h w : Nat) (ε wk gl es exh esav Sx : ℝ) where
  ke : EnetKerB cmid c 1 1 wk
  dw : EnetDwKerB cmid 5 5 wk
  kp : EnetKerB c cmid 1 1 wk
  bnp : EnetBnBack c h w ε gl es exh
  se : EnetSeBack cmid r h w wk Sx esav
  swd : EnetSwBack (cmid * h * w) esav
  bnd : EnetBnBack cmid h w ε gl es exh
  swe : EnetSwBack (cmid * h * w) esav
  bne : EnetBnBack cmid h w ε gl es exh

/-- **The whole net's backward data** — the stem, the three MBConv bodies, the head and the
    classifier: nine BatchNorm backward sites (each with its saved activation), nine swish sites
    and three squeeze-excites. ⚠ `Sx1`/`Sx2`/`Sx3` are `b1`/`b2`/`b3`'s SE saved-input bounds and
    they differ by 31 orders, so they are separate parameters rather than one net-level bound. -/
structure EnetBackWeights (ε wk gl es exh esav Sx1 Sx2 Sx3 : ℝ) where
  stemK : EnetKerB 32 3 3 3 wk
  stemSw : EnetSwBack (32 * 112 * 112) esav
  stemBn : EnetBnBack 32 112 112 ε gl es exh
  b1 : EnetNoExpBack 32 16 8 112 112 ε wk gl es exh esav Sx1
  b2 : EnetStridedBack 16 96 24 4 56 56 ε wk gl es exh esav Sx2
  b3 : EnetResidBack 24 144 6 56 56 ε wk gl es exh esav Sx3
  headK : EnetKerB 1280 24 1 1 wk
  headSw : EnetSwBack (1280 * 56 * 56) esav
  headBn : EnetBnBack 1280 56 56 ε gl es exh
  fc : EnetHeadB 1280 10 wk

-- ════════════════════════════════════════════════════════════════
-- § The per-site real maps, float peers and bridges
-- ════════════════════════════════════════════════════════════════

section Net

variable {ε wk gl es exh esav q Sx Sx1 Sx2 Sx3 : ℝ}

/-- The certified per-channel BatchNorm backward at this site. -/
noncomputable def EnetBnBack.real {c h w : Nat} (s : EnetBnBack c h w ε gl es exh) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun dy => bnPerChannelTensor3_grad_input c h w ε s.γ s.x dy

/-- Its deployed float peer, at the supplied float statistics. -/
noncomputable def EnetBnBack.float {c h w : Nat} (s : EnetBnBack c h w ε gl es exh)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3BackFV M s.γ s.fs s.fxh

/-- The site's bridge, at `S = 317` from the `ε`-floor and `Xh` from standardisation. -/
noncomputable def EnetBnBack.bridge {c h w : Nat} (s : EnetBnBack c h w ε gl es exh)
    (M : FloatModel) (P : EnetBackProfile M ε wk gl es exh esav q) (hc : 0 < c) (hhw : 0 < h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo s.real (s.float M) :=
  floatBridgesTo_bnPerChannelBack M s.γ s.x s.fs s.fxh hc hhw s.hγ s.hs (s.hS P)
    (fun _k i => bnXhat_abs_le_num (X := Xh) P.hε _ hXh0 hnX i) s.hfxh

/-- ⭐ The certified swish backward: the diagonal scale by the REAL saved derivative. -/
noncomputable def EnetSwBack.real {n : Nat} (s : EnetSwBack n esav) : Vec n → Vec n :=
  diagBack (fun i => swishScalarDeriv (s.xpre i))

/-- Its deployed float peer, at the supplied float saved derivative. -/
noncomputable def EnetSwBack.float {n : Nat} (s : EnetSwBack n esav) (M : FloatModel) :
    Vec n → Vec n :=
  M.diagBackF s.fsw

/-- ⭐⭐ The site's bridge, at `Ssw = 2` — **PROVED, not supplied** (`swishScalarDeriv_abs_le`). -/
noncomputable def EnetSwBack.bridge {n : Nat} (s : EnetSwBack n esav) (M : FloatModel)
    (hn : 0 < n) : FloatBridgesTo s.real (s.float M) :=
  floatBridgesTo_diagBack M _ s.fsw hn (fun _i => swishScalarDeriv_abs_le _) s.hfsw

/-- The certified squeeze-excite GATE backward — the exact reverse of the gate's six forward
    stages, with the real saved swish derivative in its `diagBack`. -/
noncomputable def EnetSeBack.gateReal {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav) :
    Vec (c * h * w) → Vec (c * h * w) :=
  seGateInputGrad (h := h) (w := w) s.W₁ s.W₂ s.ssig (fun i => swishScalarDeriv (s.xsw i))

/-- Its deployed float peer. -/
noncomputable def EnetSeBack.gateFloat {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  seGateInputGradF (h := h) (w := w) M s.W₁ s.W₂ s.fssig s.fssw

/-- The gate's bridge, at `Ssw = 2`. -/
noncomputable def EnetSeBack.gateBridge {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav)
    (M : FloatModel) (P : EnetBackProfile M ε wk gl es exh esav q)
    (hc : 0 < c) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w) :
    FloatBridgesTo s.gateReal (s.gateFloat M) :=
  floatBridgesTo_seGateBack (h := h) (w := w) M s.W₁ s.W₂ s.ssig s.fssig
    (fun i => swishScalarDeriv (s.xsw i)) s.fssw (Ssw := 2) P.hwk hc hr hh hww
    s.hW₁ s.hW₂ s.hssig s.hfssig (fun _i => swishScalarDeriv_abs_le _) s.hfssw

/-- ⭐⭐ The certified squeeze-excite BLOCK backward — the product rule's two-branch fan-in. -/
noncomputable def EnetSeBack.real {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav) :
    Vec (c * h * w) → Vec (c * h * w) :=
  seInputGrad s.gate s.xinp s.gateReal

/-- Its deployed float peer. -/
noncomputable def EnetSeBack.float {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  seInputGradF M s.fgate s.fxinp (s.gateFloat M)

/-- The squeeze-excite's bridge, closed at real weights. -/
noncomputable def EnetSeBack.bridge {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav)
    (M : FloatModel) (P : EnetBackProfile M ε wk gl es exh esav q)
    (hn : 0 < c * h * w) (hc : 0 < c) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w) :
    FloatBridgesTo s.real (s.float M) :=
  floatBridgesTo_seBack M s.gate s.fgate s.xinp s.fxinp hn s.hgate s.hfgate s.hxinp s.hfxinp
    (s.gateBridge M P hc hr hh hww)

/-- B0's `b1`'s certified body backward — no expand arm, no skip. -/
noncomputable def EnetNoExpBack.real {cin cout r h w : Nat}
    (b : EnetNoExpBack cin cout r h w ε wk gl es exh esav Sx) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  mbNoExpBodyBack b.dw.W b.kp.W b.bnd.real b.swd.real b.se.real b.bnp.real

/-- Its deployed float peer. -/
noncomputable def EnetNoExpBack.float {cin cout r h w : Nat}
    (b : EnetNoExpBack cin cout r h w ε wk gl es exh esav Sx) (M : FloatModel) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  mbNoExpBodyBackF M b.dw.W b.kp.W (b.bnd.float M) (b.swd.float M) (b.se.float M) (b.bnp.float M)

/-- `b1`'s bridge, closed at real weights. -/
noncomputable def EnetNoExpBack.bridge {cin cout r h w : Nat}
    (b : EnetNoExpBack cin cout r h w ε wk gl es exh esav Sx) (M : FloatModel)
    (P : EnetBackProfile M ε wk gl es exh esav q)
    (hcin : 0 < cin) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w)
    (hnI : 0 < cin * h * w) (hnO : 0 < cout * h * w) (hcout : 0 < cout) (hhw : 0 < h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_mbNoExpBodyBack M b.dw.W b.kp.W P.hwk P.hwk b.dw.hW b.kp.hW hnI hnO
    (b.bnd.bridge M P hcin hhw hXh0 hnX) (b.swd.bridge M hnI)
    (b.se.bridge M P hnI hcin hr hh hww) (b.bnp.bridge M P hcout hhw hXh0 hnX)

/-- B0's `b2`'s certified body backward — the strided one. -/
noncomputable def EnetStridedBack.real {cin cmid cout r h w : Nat}
    (b : EnetStridedBack cin cmid cout r h w ε wk gl es exh esav Sx) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  mbStridedBodyBack b.ke.W b.dw.W b.kp.W b.bne.real b.swe.real b.bnd.real b.swd.real
    b.se.real b.bnp.real

/-- Its deployed float peer. -/
noncomputable def EnetStridedBack.float {cin cmid cout r h w : Nat}
    (b : EnetStridedBack cin cmid cout r h w ε wk gl es exh esav Sx) (M : FloatModel) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  mbStridedBodyBackF M b.ke.W b.dw.W b.kp.W (b.bne.float M) (b.swe.float M) (b.bnd.float M)
    (b.swd.float M) (b.se.float M) (b.bnp.float M)

/-- `b2`'s bridge, closed at real weights. ⚠ Two `Xh`s: the project and depthwise BatchNorms sit
    at `h·w`, the expand's at `(2h)·(2w)`. -/
noncomputable def EnetStridedBack.bridge {cin cmid cout r h w : Nat}
    (b : EnetStridedBack cin cmid cout r h w ε wk gl es exh esav Sx) (M : FloatModel)
    (P : EnetBackProfile M ε wk gl es exh esav q)
    (hcmid : 0 < cmid) (hcout : 0 < cout) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w)
    (hhw : 0 < h * w) (hhw2 : 0 < (2 * h) * (2 * w))
    (hnM : 0 < cmid * h * w) (hnM2 : 0 < cmid * (2 * h) * (2 * w)) (hnO : 0 < cout * h * w)
    {Xh Xhe : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2)
    (hXhe0 : 0 ≤ Xhe) (hnXe : (((2 * h) * (2 * w) : ℕ) : ℝ) ≤ Xhe ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_mbStridedBodyBack M b.ke.W b.dw.W b.kp.W P.hwk P.hwk P.hwk
    b.ke.hW b.dw.hW b.kp.hW hnM2 hnO
    (b.bne.bridge M P hcmid hhw2 hXhe0 hnXe) (b.swe.bridge M hnM2)
    (b.bnd.bridge M P hcmid hhw hXh0 hnX) (b.swd.bridge M hnM)
    (b.se.bridge M P hnM hcmid hr hh hww) (b.bnp.bridge M P hcout hhw hXh0 hnX)

/-- B0's `b3`'s certified body backward — the full MBConv at one resolution. ⚠ The skip is the
    caller's. -/
noncomputable def EnetResidBack.real {c cmid r h w : Nat}
    (b : EnetResidBack c cmid r h w ε wk gl es exh esav Sx) :
    Vec (c * h * w) → Vec (c * h * w) :=
  mbconvBodyBack b.ke.W b.dw.W b.kp.W b.bne.real b.bnd.real b.swe.real b.swd.real
    b.se.real b.bnp.real

/-- Its deployed float peer. -/
noncomputable def EnetResidBack.float {c cmid r h w : Nat}
    (b : EnetResidBack c cmid r h w ε wk gl es exh esav Sx) (M : FloatModel) :
    Vec (c * h * w) → Vec (c * h * w) :=
  mbconvBodyBackF M b.ke.W b.dw.W b.kp.W (b.bne.float M) (b.bnd.float M) (b.swe.float M)
    (b.swd.float M) (b.se.float M) (b.bnp.float M)

/-- `b3`'s body bridge, closed at real weights. -/
noncomputable def EnetResidBack.bridge {c cmid r h w : Nat}
    (b : EnetResidBack c cmid r h w ε wk gl es exh esav Sx) (M : FloatModel)
    (P : EnetBackProfile M ε wk gl es exh esav q)
    (hc : 0 < c) (hcmid : 0 < cmid) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w) (hhw : 0 < h * w)
    (hnM : 0 < cmid * h * w) (hnO : 0 < c * h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_mbconvBodyBack M b.ke.W b.dw.W b.kp.W P.hwk P.hwk P.hwk
    b.ke.hW b.dw.hW b.kp.hW hnM hnO
    (b.bne.bridge M P hcmid hhw hXh0 hnX) (b.bnd.bridge M P hcmid hhw hXh0 hnX)
    (b.swe.bridge M hnM) (b.swd.bridge M hnM)
    (b.se.bridge M P hnM hcmid hr hh hww) (b.bnp.bridge M P hc hhw hXh0 hnX)

-- ════════════════════════════════════════════════════════════════
-- § The per-site ENVELOPES — one per record, so the whole-net chain stays at block granularity
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **An envelope through one BatchNorm backward site, at the PER-UNIT GAIN**
    (`Maps.bnPerChannelBackGain`, with the record's data read off and `S = 317` supplied by
    `EnetBnBack.hS`). ⭐ The factoring is what makes the numerals checkable: `Kr`/`Kb` depend only
    on the reduction width, so B0's nine sites need **two** pairs of constants (`n = 3136` and
    `n = 12544`) rather than eighteen — and §3.7(a)'s `norm_num` ceiling is a fact about the
    operation TREE, which this flattens. -/
theorem EnetBnBack.maps {c h w : Nat} (s : EnetBnBack c h w ε gl es exh)
    (M : FloatModel) (P : EnetBackProfile M ε wk gl es exh esav q)
    (hc : 0 < c) (hhw : 0 < h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2)
    {gn Kr Kb Ā Ē Ā' Ē' : ℝ} (hgn : (1 + M.u) ^ (h * w + 1) - 1 ≤ gn)
    (hKr : bnGradInputReMag (h * w) gl 1 317 Xh ≤ Kr)
    (hKb : bnGradInputBudgetG q gn (h * w) gl 1 317 Xh es exh ≤ Kb)
    (hĀ0 : 0 ≤ Ā) (hĒ0 : 0 ≤ Ē)
    (hĀ' : Ā * (Kr + Kb) ≤ Ā') (hĒ' : Ā * Kb + Ē * Kr ≤ Ē') :
    (s.bridge M P hc hhw hXh0 hnX).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.bnPerChannelBackGain M s.γ s.x s.fs s.fxh hc hhw s.hγ s.hs (s.hS P)
    (fun _k i => bnXhat_abs_le_num (X := Xh) P.hε _ hXh0 hnX i) s.hfxh
    P.hq hgn P.hgl (by norm_num) hXh0 P.hes P.hexh hKr hKb hĀ0 hĒ0 hĀ' hĒ'

/-- ⭐⭐ **An envelope through one swish backward site, at the GLOBAL `Ssw = 2`.** One rounded
    multiply by the saved derivative — and the `2` is `swishScalarDeriv_abs_le`, so the FORWARD's
    certified window appears nowhere. ⛔ With the repo's other bound (`1 + A/4` at that window)
    these nine sites put the fold at `10⁴³¹`, past `norm_num`'s ceiling. -/
theorem EnetSwBack.maps {n : Nat} (s : EnetSwBack n esav) (M : FloatModel)
    (P : EnetBackProfile M ε wk gl es exh esav q) (hn : 0 < n) {Ā Ē Ā' Ē' : ℝ}
    (hĀ' : 2 * Ā + FloatModel.mulErr q 2 Ā esav 0 ≤ Ā')
    (hĒ' : FloatModel.mulErr q 2 Ā esav 0 + 2 * Ē ≤ Ē') :
    (s.bridge M hn).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.diagBack M _ s.fsw hn (fun _i => swishScalarDeriv_abs_le _) s.hfsw
    P.hq (by norm_num) P.hesav hĀ' hĒ'

/-- ⭐⭐ **An envelope through one whole squeeze-excite backward site** — ten numeric stages: the
    product rule's two `diagBack`s (at the saved gate and at the saved input), the gate path's six
    (`broadcastBack → σ′ → linBack W₂ → swish′ → linBack W₁ → gapBack`), and the rounded join.

    ⭐ Read the gate path's first and last stages as a pair: `broadcastBack` multiplies the
    cotangent window by the spatial reduce's `c·h·w` fan-in and `gapBack` divides by `h·w`, so what
    the gate path costs is a factor of roughly `c` — magnitude, not nonlinearity. That is the whole
    of why a squeeze-excite BACKWARD folds where §0.1 makes its forward quadratic. -/
theorem EnetSeBack.maps {c r h w : Nat} (s : EnetSeBack c r h w wk Sx esav)
    (M : FloatModel) (P : EnetBackProfile M ε wk gl es exh esav q)
    (hn : 0 < c * h * w) (hc : 0 < c) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w) (hSx0 : 0 ≤ Sx)
    {gbc g2 g1 Ā Ē Pd Ep A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 A6 E6 Bd Ed Ā' Ē' : ℝ}
    (hgbc : (1 + M.u) ^ (c * h * w + 1) - 1 ≤ gbc)
    (hg2 : (1 + M.u) ^ (c + 2) - 1 ≤ g2) (hg1 : (1 + M.u) ^ (r + 2) - 1 ≤ g1)
    (mainA : 101 / 100 * Ā + FloatModel.mulErr q (101 / 100) Ā esav 0 ≤ Pd)
    (mainE : FloatModel.mulErr q (101 / 100) Ā esav 0 + 101 / 100 * Ē ≤ Ep)
    (preA : Sx * Ā + FloatModel.mulErr q Sx Ā esav 0 ≤ A1)
    (preE : FloatModel.mulErr q Sx Ā esav 0 + Sx * Ē ≤ E1)
    (bcA : ((c * h * w : ℕ) : ℝ) * A1 + gbc * (((c * h * w : ℕ) : ℝ) * A1) ≤ A2)
    (bcE : gbc * (((c * h * w : ℕ) : ℝ) * (A1 + E1)) + ((c * h * w : ℕ) : ℝ) * E1 ≤ E2)
    (sigA : 1 / 4 * A2 + FloatModel.mulErr q (1 / 4) A2 esav 0 ≤ A3)
    (sigE : FloatModel.mulErr q (1 / 4) A2 esav 0 + 1 / 4 * E2 ≤ E3)
    (d2A : (1 + g2) * ((c : ℝ) * wk * A3 + 0) ≤ A4)
    (d2E : g2 * ((c : ℝ) * wk * (A3 + E3) + 0) + (c : ℝ) * wk * E3 ≤ E4)
    (swA : 2 * A4 + FloatModel.mulErr q 2 A4 esav 0 ≤ A5)
    (swE : FloatModel.mulErr q 2 A4 esav 0 + 2 * E4 ≤ E5)
    (d1A : (1 + g1) * ((r : ℝ) * wk * A5 + 0) ≤ A6)
    (d1E : g1 * ((r : ℝ) * wk * (A5 + E5) + 0) + (r : ℝ) * wk * E5 ≤ E6)
    (gpA : 1 / ((h : ℝ) * (w : ℝ)) * A6
            + FloatModel.mulErr q (1 / ((h : ℝ) * (w : ℝ))) A6 0 0 ≤ Bd)
    (gpE : FloatModel.mulErr q (1 / ((h : ℝ) * (w : ℝ))) A6 0 0
            + 1 / ((h : ℝ) * (w : ℝ)) * E6 ≤ Ed)
    (hĀ' : Pd + Bd + q * (Pd + Bd) ≤ Ā')
    (hĒ' : q * (Pd + Ep + Bd + Ed) + (Ep + Ed) ≤ Ē') :
    (s.bridge M P hn hc hr hh hww).Maps Ā Ē Ā' Ē' :=
  FloatBridgesTo.Maps.seBack M s.gate s.fgate s.xinp s.fxinp hn s.hgate s.hfgate s.hxinp s.hfxinp
    (s.gateBridge M P hc hr hh hww)
    P.hq (by norm_num) P.hesav hSx0 P.hesav mainA mainE preA preE
    (FloatBridgesTo.Maps.seGateBack (h := h) (w := w) M s.W₁ s.W₂ s.ssig s.fssig
      (fun i => swishScalarDeriv (s.xsw i)) s.fssw (Ssw := 2) P.hwk hc hr hh hww
      s.hW₁ s.hW₂ s.hssig s.hfssig (fun _i => swishScalarDeriv_abs_le _) s.hfssw
      P.hq (by norm_num) P.hesav (by norm_num) P.hesav hgbc hg2 hg1
      bcA bcE sigA sigE d2A d2E swA swE d1A d1E gpA gpE)
    hĀ' hĒ'

-- ════════════════════════════════════════════════════════════════
-- § The committed net, its float peer, and the closed bridge
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The whole EfficientNet-B0 input-gradient, at the COMMITTED `efficientnetInputGradB`** —
    every slot pinned to the certified per-op backward at its own saved activation.
    ⛔ **At `N = 1`, and that is load-bearing, not tidiness**: the block and BatchNorm slots are
    `StableHLO.batchMap 1` of per-example maps, and the deployed `bnBatchLA` reduces μ/var ACROSS
    examples — the two agree exactly when `N = 1`. See the file header. -/
noncomputable def b0GradR (w : EnetBackWeights ε wk gl es exh esav Sx1 Sx2 Sx3) :
    Vec (1 * 10) → Vec (1 * (3 * 224 * 224)) :=
  efficientnetInputGradB 1 w.stemK.W w.headK.W w.fc.W
    (StableHLO.batchMap 1 w.stemBn.real) (StableHLO.batchMap 1 w.stemSw.real)
    (StableHLO.batchMap 1 w.headBn.real) (StableHLO.batchMap 1 w.headSw.real)
    (StableHLO.batchMap 1 w.b1.real) (StableHLO.batchMap 1 w.b2.real)
    (StableHLO.batchMap 1 (Proofs.residual w.b3.real))

/-- The deployed float peer, at the committed `efficientnetInputGradBF`. -/
noncomputable def b0GradF (M : FloatModel) (w : EnetBackWeights ε wk gl es exh esav Sx1 Sx2 Sx3) :
    Vec (1 * 10) → Vec (1 * (3 * 224 * 224)) :=
  efficientnetInputGradBF 1 M w.stemK.W w.headK.W w.fc.W
    (StableHLO.batchMap 1 (w.stemBn.float M)) (StableHLO.batchMap 1 (w.stemSw.float M))
    (StableHLO.batchMap 1 (w.headBn.float M)) (StableHLO.batchMap 1 (w.headSw.float M))
    (StableHLO.batchMap 1 (w.b1.float M)) (StableHLO.batchMap 1 (w.b2.float M))
    (StableHLO.batchMap 1 (fun v j => M.add (w.b3.float M v j) (v j)))

set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
/-- ⭐ **The whole EfficientNet-B0 input-gradient VJP float-bridges TO its float peer, CLOSED** —
    the three MBConv bodies, all nine BatchNorm backwards, all nine swish backwards and all three
    squeeze-excites discharged at the record's real data, nothing left but `es`/`exh`/`esav`.
    ⚠ Grouped and associated exactly as `efficientnet_grad_floatBridgesTo` threads
    `efficientnetInputGradB` — `.comp` is not associative as a bridge (§3.7's grouping lesson). -/
noncomputable def b0GradBridge (M : FloatModel)
    (P : EnetBackProfile M ε wk gl es exh esav q)
    (w : EnetBackWeights ε wk gl es exh esav Sx1 Sx2 Sx3) :
    FloatBridgesTo (b0GradR w) (b0GradF M w) :=
  ((((((FloatBridgesTo.batchMap 1
        (floatBridgesTo_linBack M w.fc.W P.hwk (by norm_num) w.fc.hW)).comp
      (FloatBridgesTo.batchMap 1
        (floatBridgesTo_gapBack M 1280 56 56 (by norm_num) (by norm_num) (by norm_num)))).comp
      (((FloatBridgesTo.batchMap 1 (w.headSw.bridge M (by norm_num))).comp
          (FloatBridgesTo.batchMap 1
            (w.headBn.bridge M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num)
              (by norm_num)))).comp
        (FloatBridgesTo.batchMap 1
          (floatBridgesTo_convBack (h := 56) (w := 56) M w.headK.W P.hwk (by norm_num)
            w.headK.hW)))).comp
      (FloatBridgesTo.batchMap 1 (FloatBridgesTo.residual M
        (w.b3.bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num) (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num))))).comp
      (FloatBridgesTo.batchMap 1
        (w.b2.bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Xh := 56) (by norm_num) (by norm_num) (Xhe := 112) (by norm_num) (by norm_num)))).comp
      (FloatBridgesTo.batchMap 1
        (w.b1.bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num) (by norm_num) (by norm_num) (Xh := 112) (by norm_num) (by norm_num)))).comp
      (((FloatBridgesTo.batchMap 1 (w.stemSw.bridge M (by norm_num))).comp
          (FloatBridgesTo.batchMap 1
            (w.stemBn.bridge M P (by norm_num) (by norm_num) (Xh := 112) (by norm_num)
              (by norm_num)))).comp
        (FloatBridgesTo.batchMap 1
          (floatBridgesTo_flatConvStride2Back (h := 112) (w := 112) M w.stemK.W P.hwk
            (by norm_num) w.stemK.hW)))

end Net

-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, measured per parameter KIND on the 350-epoch 4-GPU ImageNet run
    (`/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin`, 5,288,548 f32): conv,
    depthwise and dense kernels — the squeeze-excites' two among them — within `37/10` (global max
    `3.6857` over 5.24 M entries) and BatchNorm γ within `41/10` (max `4.0545`). ⚠ Here the split runs ResNet-34's way — the uniform
    bound is a BN γ — but at 1.1× it is worth under an order, against ConvNeXt-T's 68 and r34's 8
    (§3.9 finding 5: measure it, do not assume which kind is the outlier). ⭐ β, the SE biases and
    the dense bias appear nowhere: the backward is stated at bias `0` throughout.
    `ε ≥ 10⁻⁵` is the ONLY thing bounding the inverse-stddev, and it is enough (`EnetBnBack.hS`).
    ⛔ The float inverse-stddev, the float normalised activation and every other saved vector the
    backward scales by are taken accurate to `10⁻²` — SUPPLIED; the file header is about exactly
    that. -/
theorem b0BackProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    EnetBackProfile M ε (37/10) (41/10) (1/100) (1/100) (1/100) u32 where
  hwk := by norm_num
  hgl := by norm_num
  hes := by norm_num
  hexh := by norm_num
  hesav := by norm_num
  hε5 := hε5
  hq := hMu

set_option maxRecDepth 4000000 in
set_option maxHeartbeats 8000000 in
/-- ⭐ **The envelope, kernel-checked.** 59 numeric stages, 138 rational inequalities, built
    bottom-up at block granularity (3 body steps rather than 45 leaf steps), with every γ-term
    bounded through `FloatModel.gamma_num` so `norm_num` never evaluates a big power, and every
    BatchNorm site stated at its per-unit gain (`EnetBnBack.maps`) so the expensive evaluation
    happens once per reduction WIDTH — **two** constants for nine sites.

    ⭐ Of the 138, NONE is a cap: `budget / window = 0.222`, not `2.00`. Every one is the interval
    fold — at TRAINING-mode BatchNorm, through three squeeze-excites, and with no operating-point
    hypothesis anywhere. -/
theorem b0GradBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : EnetBackWeights ε (37/10) (41/10) (1/100) (1/100) (1/100)
          (7572 * 10 ^ 6) (5451 * 10 ^ 21) (4903 * 10 ^ 37)) :
    (b0GradBridge M (b0BackProfile_committed M hMu hε5) w).Maps 1 0
      (7104 * 10 ^ 179) (1578 * 10 ^ 179) := by
  have P := b0BackProfile_committed M hMu hε5
  -- ⭐ the TWO per-width gain constants, each proved once: at `N = 1` the batched BatchNorm's
  -- reduction width is `h·w`, so B0's nine sites use only these two.
  have K3136r : bnGradInputReMag (56 * 56) (41/10) 1 317 56
      ≤ 4079 * 10 ^ 3 := by norm_num [bnGradInputReMag]
  have K3136b : bnGradInputBudgetG u32 (1871 / 10 ^ 7) (56 * 56) (41/10) 1 317 56
      (1/100) (1/100) ≤ 2350 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K12544r : bnGradInputReMag (112 * 112) (41/10) 1 317 112
      ≤ 1631 * 10 ^ 4 := by norm_num [bnGradInputReMag]
  have K12544b : bnGradInputBudgetG u32 (7483 / 10 ^ 7) (112 * 112) (41/10) 1 317 112
      (1/100) (1/100) ≤ 1564 * 10 ^ 1 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  -- the classifier and the GAP scatter: the two stages of a backward chain that SHRINK
  have m0 := (FloatBridgesTo.Maps.batchMap 1
      (FloatBridgesTo.Maps.linBack M w.fc.W P.hwk (by norm_num) w.fc.hW
        (M.gamma_num (k := 10 + 2) (q := 7153 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (Ā := 1) (Ē := 0) (Ā' := 3701 / 10 ^ 2) (Ē' := 2647 / 10 ^ 8)
        (by norm_num) (by norm_num))).comp (by norm_num)
    (FloatBridgesTo.Maps.batchMap 1
      (FloatBridgesTo.Maps.gapBack M 1280 56 56 (by norm_num) (by norm_num) (by norm_num) P.hq
        (Ā := 3701 / 10 ^ 2) (Ē := 2647 / 10 ^ 8) (Ā' := 1181 / 10 ^ 5) (Ē' := 9145 / 10 ^ 12)
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])))
  -- the head: swish backward, BatchNorm backward, 1×1 conv backward (fan-in 1280)
  have mH := m0.comp (by norm_num)
    ((FloatBridgesTo.Maps.batchMap 1
        (w.headSw.maps M P (by norm_num)
          (Ā := 1181 / 10 ^ 5) (Ē := 9145 / 10 ^ 12) (Ā' := 2374 / 10 ^ 5) (Ē' := 1182 / 10 ^ 7)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.batchMap 1
        (w.headBn.maps M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num)
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K3136r K3136b (by norm_num) (by norm_num)
          (Ā := 2374 / 10 ^ 5) (Ē := 1182 / 10 ^ 7) (Ā' := 9690 * 10 ^ 1) (Ē' := 538)
          (by norm_num) (by norm_num)))
      |>.comp (by norm_num)
      (FloatBridgesTo.Maps.batchMap 1
        (FloatBridgesTo.Maps.convBack (h := 56) (w := 56) M w.headK.W P.hwk (by norm_num) w.headK.hW
          (M.gamma_num (k := 1280 * 1 * 1 + 2) (q := 7642 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          (Ā := 9690 * 10 ^ 1) (Ē := 538) (Ā' := 4590 * 10 ^ 5) (Ē' := 2584 * 10 ^ 3)
          (by norm_num) (by norm_num))))
  -- ⭐ `b3`, the residual MBConv: its body under `Maps.residual`, the skip supplied by the caller
  have mB3 := mH.comp (by norm_num)
    (FloatBridgesTo.Maps.batchMap 1
      (FloatBridgesTo.Maps.residual M (by norm_num)
      (FloatBridgesTo.Maps.mbconvBodyBack (h := 56) (w := 56) M w.b3.ke.W w.b3.dw.W
        w.b3.kp.W P.hwk P.hwk P.hwk w.b3.ke.hW w.b3.dw.hW w.b3.kp.hW
        (by norm_num) (by norm_num) _ _ _ _ _ _
        (gp := 1550 / 10 ^ 9) (gd := 1610 / 10 ^ 9) (ge := 8703 / 10 ^ 9)
        (Ā := 4590 * 10 ^ 5) (Ē := 2584 * 10 ^ 3) (A1 := 1874 * 10 ^ 12) (E1 := 1162 * 10 ^ 10)
        (A2 := 1665 * 10 ^ 14) (E2 := 1033 * 10 ^ 12) (A3 := 7479 * 10 ^ 60) (E3 := 5604 * 10 ^ 59)
        (A4 := 1504 * 10 ^ 61) (E4 := 1196 * 10 ^ 60) (A5 := 6139 * 10 ^ 67) (E5 := 4914 * 10 ^ 66)
        (A6 := 5679 * 10 ^ 69) (E6 := 4546 * 10 ^ 68) (A7 := 1142 * 10 ^ 70) (E7 := 9660 * 10 ^ 68)
        (A8 := 4661 * 10 ^ 76) (E8 := 3968 * 10 ^ 75) (Ā' := 2484 * 10 ^ 79) (Ē' := 2115 * 10 ^ 78)
        (M.gamma_num (k := 24 * 1 * 1 + 2) (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 5 * 5 + 2) (q := 1610 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 144 * 1 * 1 + 2) (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (w.b3.bnp.maps M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num)
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K3136r K3136b (by norm_num) (by norm_num)
          (Ā := 4590 * 10 ^ 5) (Ē := 2584 * 10 ^ 3) (Ā' := 1874 * 10 ^ 12) (Ē' := 1162 * 10 ^ 10)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)
        (w.b3.se.maps M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Ā := 1665 * 10 ^ 14) (Ē := 1033 * 10 ^ 12) (Pd := 1699 * 10 ^ 14) (Ep := 2709 * 10 ^ 12)
          (A1 := 8164 * 10 ^ 54) (E1 := 5065 * 10 ^ 52) (A2 := 3789 * 10 ^ 60) (E2 := 1256 * 10 ^ 59)
          (A3 := 9852 * 10 ^ 59) (E3 := 6930 * 10 ^ 58) (A4 := 5250 * 10 ^ 62) (E4 := 3693 * 10 ^ 61)
          (A5 := 1056 * 10 ^ 63) (E5 := 7912 * 10 ^ 61) (A6 := 2345 * 10 ^ 64) (E6 := 1757 * 10 ^ 63)
          (Bd := 7478 * 10 ^ 60) (Ed := 5603 * 10 ^ 59) (Ā' := 7479 * 10 ^ 60) (Ē' := 5604 * 10 ^ 59)
          (M.gamma_num (k := 144 * 56 * 56 + 1) (q := 2767 / 10 ^ 5) hMu (by norm_num [u32]) (by norm_num [u32]))
          (M.gamma_num (k := 144 + 2) (q := 8703 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (M.gamma_num (k := 6 + 2) (q := 4769 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num [u32]) (by norm_num [u32]))
        (w.b3.swd.maps M P (by norm_num)
          (Ā := 7479 * 10 ^ 60) (Ē := 5604 * 10 ^ 59) (Ā' := 1504 * 10 ^ 61) (Ē' := 1196 * 10 ^ 60)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
        (w.b3.bnd.maps M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num)
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K3136r K3136b (by norm_num) (by norm_num)
          (Ā := 1504 * 10 ^ 61) (Ē := 1196 * 10 ^ 60) (Ā' := 6139 * 10 ^ 67) (Ē' := 4914 * 10 ^ 66)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)
        (w.b3.swe.maps M P (by norm_num)
          (Ā := 5679 * 10 ^ 69) (Ē := 4546 * 10 ^ 68) (Ā' := 1142 * 10 ^ 70) (Ē' := 9660 * 10 ^ 68)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
        (w.b3.bne.maps M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num)
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K3136r K3136b (by norm_num) (by norm_num)
          (Ā := 1142 * 10 ^ 70) (Ē := 9660 * 10 ^ 68) (Ā' := 4661 * 10 ^ 76) (Ē' := 3968 * 10 ^ 75)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)) P.hq
        (Ā' := 2485 * 10 ^ 79) (Ē' := 2116 * 10 ^ 78) (by norm_num [u32]) (by norm_num [u32])))
  -- `b2`, the strided MBConv: the expand arm at 112²
  have mB2 := mB3.comp (by norm_num)
    (FloatBridgesTo.Maps.batchMap 1
      (FloatBridgesTo.Maps.mbStridedBodyBack (h := 56) (w := 56) M w.b2.ke.W w.b2.dw.W
        w.b2.kp.W P.hwk P.hwk P.hwk w.b2.ke.hW w.b2.dw.hW w.b2.kp.hW
        (by norm_num) (by norm_num) (by norm_num) _ _ _ _ _ _
        (gp := 1550 / 10 ^ 9) (gd := 6557 / 10 ^ 10) (ge := 5842 / 10 ^ 9)
        (Ā := 2485 * 10 ^ 79) (Ē := 2116 * 10 ^ 78) (A1 := 1015 * 10 ^ 86) (E1 := 8690 * 10 ^ 84)
        (A2 := 9014 * 10 ^ 87) (E2 := 7717 * 10 ^ 86) (A3 := 1323 * 10 ^ 118) (E3 := 1882 * 10 ^ 117)
        (A4 := 2660 * 10 ^ 118) (E4 := 3897 * 10 ^ 117) (A5 := 1086 * 10 ^ 125) (E5 := 1596 * 10 ^ 124)
        (A6 := 3617 * 10 ^ 126) (E6 := 5315 * 10 ^ 125) (A7 := 7271 * 10 ^ 126) (E7 := 1100 * 10 ^ 126)
        (A8 := 1188 * 10 ^ 134) (E8 := 1806 * 10 ^ 133) (Ā' := 4220 * 10 ^ 136) (Ē' := 6416 * 10 ^ 135)
        (M.gamma_num (k := 24 * 1 * 1 + 2) (q := 1550 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 96 * 1 * 1 + 2) (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (w.b2.bnp.maps M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num)
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K3136r K3136b (by norm_num) (by norm_num)
          (Ā := 2485 * 10 ^ 79) (Ē := 2116 * 10 ^ 78) (Ā' := 1015 * 10 ^ 86) (Ē' := 8690 * 10 ^ 84)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)
        (w.b2.se.maps M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Ā := 9014 * 10 ^ 87) (Ē := 7717 * 10 ^ 86) (Pd := 9195 * 10 ^ 87) (Ep := 8696 * 10 ^ 86)
          (A1 := 4914 * 10 ^ 112) (E1 := 4207 * 10 ^ 111) (A2 := 1507 * 10 ^ 118) (E2 := 1561 * 10 ^ 117)
          (A3 := 3919 * 10 ^ 117) (E3 := 5410 * 10 ^ 116) (A4 := 1393 * 10 ^ 120) (E4 := 1922 * 10 ^ 119)
          (A5 := 2800 * 10 ^ 120) (E5 := 3984 * 10 ^ 119) (A6 := 4145 * 10 ^ 121) (E6 := 5897 * 10 ^ 120)
          (Bd := 1322 * 10 ^ 118) (Ed := 1881 * 10 ^ 117) (Ā' := 1323 * 10 ^ 118) (Ē' := 1882 * 10 ^ 117)
          (M.gamma_num (k := 96 * 56 * 56 + 1) (q := 1828 / 10 ^ 5) hMu (by norm_num [u32]) (by norm_num [u32]))
          (M.gamma_num (k := 96 + 2) (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (M.gamma_num (k := 4 + 2) (q := 3577 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num [u32]) (by norm_num [u32]))
        (w.b2.swd.maps M P (by norm_num)
          (Ā := 1323 * 10 ^ 118) (Ē := 1882 * 10 ^ 117) (Ā' := 2660 * 10 ^ 118) (Ē' := 3897 * 10 ^ 117)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
        (w.b2.bnd.maps M P (by norm_num) (by norm_num) (Xh := 56) (by norm_num) (by norm_num)
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K3136r K3136b (by norm_num) (by norm_num)
          (Ā := 2660 * 10 ^ 118) (Ē := 3897 * 10 ^ 117) (Ā' := 1086 * 10 ^ 125) (Ē' := 1596 * 10 ^ 124)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)
        (w.b2.swe.maps M P (by norm_num)
          (Ā := 3617 * 10 ^ 126) (Ē := 5315 * 10 ^ 125) (Ā' := 7271 * 10 ^ 126) (Ē' := 1100 * 10 ^ 126)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
        (w.b2.bne.maps M P (by norm_num) (by norm_num) (Xh := 112) (by norm_num) (by norm_num)
          (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K12544r K12544b (by norm_num) (by norm_num)
          (Ā := 7271 * 10 ^ 126) (Ē := 1100 * 10 ^ 126) (Ā' := 1188 * 10 ^ 134) (Ē' := 1806 * 10 ^ 133)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)))
  -- `b1`, the no-expand MBConv
  have mB1 := mB2.comp (by norm_num)
    (FloatBridgesTo.Maps.batchMap 1
      (FloatBridgesTo.Maps.mbNoExpBodyBack (h := 112) (w := 112) M w.b1.dw.W w.b1.kp.W
        P.hwk P.hwk w.b1.dw.hW w.b1.kp.hW (by norm_num) (by norm_num) _ _ _ _
        (gp := 1073 / 10 ^ 9) (gd := 6557 / 10 ^ 10)
        (Ā := 4220 * 10 ^ 136) (Ē := 6416 * 10 ^ 135) (A1 := 6890 * 10 ^ 143) (E1 := 1054 * 10 ^ 143)
        (A2 := 4079 * 10 ^ 145) (E2 := 6240 * 10 ^ 144) (A3 := 1858 * 10 ^ 160) (E3 := 3948 * 10 ^ 159)
        (A4 := 3735 * 10 ^ 160) (E4 := 8082 * 10 ^ 159) (A5 := 6098 * 10 ^ 167) (E5 := 1325 * 10 ^ 167)
        (Ā' := 2031 * 10 ^ 169) (Ē' := 4413 * 10 ^ 168)
        (M.gamma_num (k := 16 * 1 * 1 + 2) (q := 1073 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 3 * 3 + 2) (q := 6557 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
        (w.b1.bnp.maps M P (by norm_num) (by norm_num) (Xh := 112) (by norm_num) (by norm_num)
          (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K12544r K12544b (by norm_num) (by norm_num)
          (Ā := 4220 * 10 ^ 136) (Ē := 6416 * 10 ^ 135) (Ā' := 6890 * 10 ^ 143) (Ē' := 1054 * 10 ^ 143)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)
        (w.b1.se.maps M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Ā := 4079 * 10 ^ 145) (Ē := 6240 * 10 ^ 144) (Pd := 4161 * 10 ^ 145) (Ep := 6711 * 10 ^ 144)
          (A1 := 3089 * 10 ^ 155) (E1 := 4725 * 10 ^ 154) (A2 := 1271 * 10 ^ 161) (E2 := 2248 * 10 ^ 160)
          (A3 := 3305 * 10 ^ 160) (E3 := 6892 * 10 ^ 159) (A4 := 3914 * 10 ^ 162) (E4 := 8161 * 10 ^ 161)
          (A5 := 7868 * 10 ^ 162) (E5 := 1672 * 10 ^ 162) (A6 := 2329 * 10 ^ 164) (E6 := 4950 * 10 ^ 163)
          (Bd := 1857 * 10 ^ 160) (Ed := 3947 * 10 ^ 159) (Ā' := 1858 * 10 ^ 160) (Ē' := 3948 * 10 ^ 159)
          (M.gamma_num (k := 32 * 112 * 112 + 1) (q := 2452 / 10 ^ 5) hMu (by norm_num [u32]) (by norm_num [u32]))
          (M.gamma_num (k := 32 + 2) (q := 2027 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (M.gamma_num (k := 8 + 2) (q := 5961 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num) (by norm_num) (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
          (by norm_num [u32]) (by norm_num [u32]))
        (w.b1.swd.maps M P (by norm_num)
          (Ā := 1858 * 10 ^ 160) (Ē := 3948 * 10 ^ 159) (Ā' := 3735 * 10 ^ 160) (Ē' := 8082 * 10 ^ 159)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
        (w.b1.bnd.maps M P (by norm_num) (by norm_num) (Xh := 112) (by norm_num) (by norm_num)
          (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K12544r K12544b (by norm_num) (by norm_num)
          (Ā := 3735 * 10 ^ 160) (Ē := 8082 * 10 ^ 159) (Ā' := 6098 * 10 ^ 167) (Ē' := 1325 * 10 ^ 167)
          (by norm_num) (by norm_num))
        (by norm_num) (by norm_num)))
  -- the stem: swish backward, BatchNorm backward, 3×3/s2 conv backward (fan-in 32·9)
  have mS := mB1.comp (by norm_num)
    ((FloatBridgesTo.Maps.batchMap 1
        (w.stemSw.maps M P (by norm_num)
          (Ā := 2031 * 10 ^ 169) (Ē := 4413 * 10 ^ 168) (Ā' := 4083 * 10 ^ 169) (Ē' := 9030 * 10 ^ 168)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.batchMap 1
        (w.stemBn.maps M P (by norm_num) (by norm_num) (Xh := 112) (by norm_num) (by norm_num)
          (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          K12544r K12544b (by norm_num) (by norm_num)
          (Ā := 4083 * 10 ^ 169) (Ē := 9030 * 10 ^ 168) (Ā' := 6666 * 10 ^ 176) (Ē' := 1480 * 10 ^ 176)
          (by norm_num) (by norm_num)))
      |>.comp (by norm_num)
      (FloatBridgesTo.Maps.batchMap 1
        (FloatBridgesTo.Maps.flatConvStride2Back (h := 112) (w := 112) M w.stemK.W P.hwk
          (by norm_num) w.stemK.hW
          (M.gamma_num (k := 32 * 3 * 3 + 2) (q := 1729 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          (Ā := 6666 * 10 ^ 176) (Ē := 1480 * 10 ^ 176) (Ā' := 7104 * 10 ^ 179) (Ē' := 1578 * 10 ^ 179)
          (by norm_num) (by norm_num))))
  exact mS

/-- The certified output window of EfficientNet-B0's input-gradient at the committed profile:
    `≤ 7.104·10¹⁸²` per input pixel, on loss cotangents of magnitude `≤ 1`, at `N = 1`. -/
theorem b0GradBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : EnetBackWeights ε (37/10) (41/10) (1/100) (1/100) (1/100)
          (7572 * 10 ^ 6) (5451 * 10 ^ 21) (4903 * 10 ^ 37)) :
    (b0GradBridge M (b0BackProfile_committed M hMu hε5) w).mag 1 ≤ 7104 * 10 ^ 179 :=
  (b0GradBridge_maps M hMu hε5 w).mag_le 1 (by norm_num) le_rfl

/-- The fresh budget of EfficientNet-B0's input-gradient at the committed profile:
    `≤ 1.578·10¹⁸²`. -/
theorem b0GradBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε)
    (w : EnetBackWeights ε (37/10) (41/10) (1/100) (1/100) (1/100)
          (7572 * 10 ^ 6) (5451 * 10 ^ 21) (4903 * 10 ^ 37)) :
    (b0GradBridge M (b0BackProfile_committed M hMu hε5) w).fresh 1 ≤ 1578 * 10 ^ 179 :=
  (b0GradBridge_maps M hMu hε5 w).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed EfficientNet-B0 float input-gradient is within `1.578·10¹⁸²` of the
    certified real one, per input pixel**, on loss cotangents of magnitude `≤ 1`, at `|W| ≤ 37/10`,
    `|γ| ≤ 41/10`, `ε ≥ 10⁻⁵`, `u ≤ 2⁻²⁴`, **batch size `N = 1`**. The certified window is
    `7.104·10¹⁸²`, so `budget / window = 0.222` — ⭐ **the interval FOLD**, where this net's own
    forward number is a fold only in its INFERENCE mode and ConvNeXt-T's and ViT-Tiny's forward
    numbers are `2.00` caps — and **at TRAINING-mode BatchNorm**, the mode this net's own forward
    has no statable number for at all.

    ⭐⭐ **Three squeeze-excites, and no cap anywhere.** `seInputGrad g x gateBack =
    biPathSum (diagBack g) (gateBack ∘ diagBack x)`: the gate and the SE's input are SAVED
    constants, so both branches are linear in the cotangent and §0.1's third quadratic site is a
    FORWARD fact.

    ⭐⭐ **And it assumes no operating point**: `|istd| ≤ 317` comes from `ε ≥ 10⁻⁵` alone
    (`EnetBnBack.hS`), which makes this the second whole-net backward number in the repo with
    nothing supplied but saved-activation accuracies — MobileNetV2's was the first, and ResNet-34's
    and ConvNeXt-T's both pay `|istd| ≤ 16`.

    ⛔ **Read the file header before quoting it**, for two reasons. `es`, `exh` and `esav` ARE
    supplied at `10⁻²`, and they are what this net's own training-mode forward fold cannot
    discharge (⭐ though unlike ConvNeXt-T, B0 has an inference mode where its forward IS a fold,
    so the gap is quantitative and not a gap in kind). And the number is **at `N = 1`**: `bnBatchLA`
    reduces μ/var across examples, so every BatchNorm site's width is `N·h·w` and the number moves
    with the batch size — it stays a fold and stays statable to `N = 256`, but it is not one
    number for all `N` the way the forward's is. -/
theorem b0_grad_float_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : EnetBackWeights ε (37/10) (41/10) (1/100) (1/100) (1/100)
          (7572 * 10 ^ 6) (5451 * 10 ^ 21) (4903 * 10 ^ 37))
    (dy : Vec (1 * 10)) (hdy : ∀ k, |dy k| ≤ 1) (j : Fin (1 * (3 * 224 * 224))) :
    |b0GradF M w dy j - b0GradR w dy j| ≤ 1578 * 10 ^ 179 :=
  (b0GradBridge_maps M hMu hε5 w).budget_le (by norm_num) le_rfl dy hdy j

/-! ### Inhabitation

`b0_grad_float_le`'s record at the ε-floor and `N = 1`: zero kernels, zero saved activations and
gates, the exact inverse-stddev, normalised activation and swish derivative as the deployed float
values, zero saved squeeze-excite inputs (the `Sx` bounds are positive numerals). No operating
point, so `ε = 1/100000`. -/
noncomputable def EnetKerB.zero (oc ic kH kW : Nat) {wk : ℝ} (h : 0 ≤ wk) : EnetKerB oc ic kH kW wk where
  W := fun _ _ _ _ => 0
  hW := fun _ _ _ _ => by simpa using h

noncomputable def EnetDwKerB.zero (c kH kW : Nat) {wk : ℝ} (h : 0 ≤ wk) : EnetDwKerB c kH kW wk where
  W := fun _ _ _ => 0
  hW := fun _ _ _ => by simpa using h

noncomputable def EnetHeadB.zero (m n : Nat) {wk : ℝ} (h : 0 ≤ wk) : EnetHeadB m n wk where
  W := fun _ _ => 0
  hW := fun _ _ => by simpa using h

noncomputable def EnetBnBack.exact (c h w : Nat) {ε gl es exh : ℝ}
    (hgl : 0 ≤ gl) (hes : 0 ≤ es) (hexh : 0 ≤ exh) : EnetBnBack c h w ε gl es exh where
  γ := fun _ => 0
  x := fun _ => 0
  fs := fun k => bnIstd (h * w) (Mat.unflatten (reassocFwd c h w (fun _ => 0)) k) ε
  fxh := fun k => bnXhat (h * w) ε (Mat.unflatten (reassocFwd c h w (fun _ => 0)) k)
  hγ := fun _ => by simpa using hgl
  hs := fun _ => by simpa using hes
  hfxh := fun _ _ => by simpa using hexh

noncomputable def EnetSwBack.exact (n : Nat) {esav : ℝ} (h : 0 ≤ esav) : EnetSwBack n esav where
  xpre := fun _ => 0
  fsw := fun _ => swishScalarDeriv 0
  hfsw := fun _ => by simpa using h

noncomputable def EnetSeBack.exact (c r h w : Nat) {wk Sx esav : ℝ}
    (hwk : 0 ≤ wk) (hSx : 0 ≤ Sx) (hesav : 0 ≤ esav) : EnetSeBack c r h w wk Sx esav where
  W₁ := fun _ _ => 0
  W₂ := fun _ _ => 0
  hW₁ := fun _ _ => by simpa using hwk
  hW₂ := fun _ _ => by simpa using hwk
  ssig := fun _ => 0
  fssig := fun _ => 0
  hssig := fun _ => by norm_num
  hfssig := fun _ => by simpa using hesav
  xsw := fun _ => 0
  fssw := fun _ => swishScalarDeriv 0
  hfssw := fun _ => by simpa using hesav
  gate := fun _ => 0
  fgate := fun _ => 0
  hgate := fun _ => by norm_num
  hfgate := fun _ => by simpa using hesav
  xinp := fun _ => 0
  fxinp := fun _ => 0
  hxinp := fun _ => by simpa using hSx
  hfxinp := fun _ => by simpa using hesav

noncomputable def EnetNoExpBack.exact (cin cout r h w : Nat) {ε wk gl es exh esav Sx : ℝ}
    (hwk : 0 ≤ wk) (hgl : 0 ≤ gl) (hes : 0 ≤ es) (hexh : 0 ≤ exh) (hesav : 0 ≤ esav) (hSx : 0 ≤ Sx) :
    EnetNoExpBack cin cout r h w ε wk gl es exh esav Sx where
  dw := EnetDwKerB.zero _ _ _ hwk
  kp := EnetKerB.zero _ _ _ _ hwk
  bnp := EnetBnBack.exact _ _ _ hgl hes hexh
  se := EnetSeBack.exact _ _ _ _ hwk hSx hesav
  swd := EnetSwBack.exact _ hesav
  bnd := EnetBnBack.exact _ _ _ hgl hes hexh

noncomputable def EnetStridedBack.exact (cin cmid cout r h w : Nat) {ε wk gl es exh esav Sx : ℝ}
    (hwk : 0 ≤ wk) (hgl : 0 ≤ gl) (hes : 0 ≤ es) (hexh : 0 ≤ exh) (hesav : 0 ≤ esav) (hSx : 0 ≤ Sx) :
    EnetStridedBack cin cmid cout r h w ε wk gl es exh esav Sx where
  ke := EnetKerB.zero _ _ _ _ hwk
  dw := EnetDwKerB.zero _ _ _ hwk
  kp := EnetKerB.zero _ _ _ _ hwk
  bnp := EnetBnBack.exact _ _ _ hgl hes hexh
  se := EnetSeBack.exact _ _ _ _ hwk hSx hesav
  swd := EnetSwBack.exact _ hesav
  bnd := EnetBnBack.exact _ _ _ hgl hes hexh
  swe := EnetSwBack.exact _ hesav
  bne := EnetBnBack.exact _ _ _ hgl hes hexh

noncomputable def EnetResidBack.exact (c cmid r h w : Nat) {ε wk gl es exh esav Sx : ℝ}
    (hwk : 0 ≤ wk) (hgl : 0 ≤ gl) (hes : 0 ≤ es) (hexh : 0 ≤ exh) (hesav : 0 ≤ esav) (hSx : 0 ≤ Sx) :
    EnetResidBack c cmid r h w ε wk gl es exh esav Sx where
  ke := EnetKerB.zero _ _ _ _ hwk
  dw := EnetDwKerB.zero _ _ _ hwk
  kp := EnetKerB.zero _ _ _ _ hwk
  bnp := EnetBnBack.exact _ _ _ hgl hes hexh
  se := EnetSeBack.exact _ _ _ _ hwk hSx hesav
  swd := EnetSwBack.exact _ hesav
  bnd := EnetBnBack.exact _ _ _ hgl hes hexh
  swe := EnetSwBack.exact _ hesav
  bne := EnetBnBack.exact _ _ _ hgl hes hexh

noncomputable def EnetBackWeights.exact {ε : ℝ} :
    EnetBackWeights ε (37/10) (41/10) (1/100) (1/100) (1/100)
      (7572 * 10 ^ 6) (5451 * 10 ^ 21) (4903 * 10 ^ 37) :=
  have hwk : (0:ℝ) ≤ 37/10 := by norm_num
  have hgl : (0:ℝ) ≤ 41/10 := by norm_num
  have he : (0:ℝ) ≤ 1/100 := by norm_num
  { stemK := EnetKerB.zero _ _ _ _ hwk, stemSw := EnetSwBack.exact _ he
    stemBn := EnetBnBack.exact _ _ _ hgl he he
    b1 := EnetNoExpBack.exact _ _ _ _ _ hwk hgl he he he (by norm_num)
    b2 := EnetStridedBack.exact _ _ _ _ _ _ hwk hgl he he he (by norm_num)
    b3 := EnetResidBack.exact _ _ _ _ _ hwk hgl he he he (by norm_num)
    headK := EnetKerB.zero _ _ _ _ hwk, headSw := EnetSwBack.exact _ he
    headBn := EnetBnBack.exact _ _ _ hgl he he, fc := EnetHeadB.zero _ _ hwk }

example (dy : Vec (1 * 10)) (hdy : ∀ k, |dy k| ≤ 1) (j : Fin (1 * (3 * 224 * 224))) :
    |b0GradF binary32 (EnetBackWeights.exact (ε := 1/100000)) dy j
      - b0GradR (EnetBackWeights.exact (ε := 1/100000)) dy j| ≤ 1578 * 10 ^ 179 :=
  b0_grad_float_le binary32 binary32_u.le (by norm_num) _ dy hdy j

end Proofs
