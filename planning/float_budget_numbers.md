# Whole-net float budgets as NUMBERS: six forwards, and the first BACKWARD

Written 2026-09-02; revised 2026-09-03 after ResNet-34, MobileNetV2, EfficientNet-B0 and
ConvNeXt-T, again the same day after ViT-Tiny, and again after **`Resnet34BackFloatBudget.lean`**
— phase 2's first number.

**Picking this up cold?** ⭐⭐ **Every forward in §0's table carries a kernel-checked number, and
so does ResNet-34's INPUT-GRADIENT VJP** — `r34_grad_float_le`, window 8.857·10²⁴⁵ / budget
6.894·10²⁴⁴, ratio 0.078, and ⭐⭐ **as of 2026-09-03 that chain IS the certified whole-net
gradient** (`r34InputGrad_eq_resnet34_vjp`, §3.10), not just a chain each of whose pieces is. That last one is the interesting one: **it is the interval FOLD, and it
is at TRAINING-mode BatchNorm — the mode this same net's forward has no statable number for at
all** (§0.1's 10⁷⁴¹⁷). A VJP reads its statistics off the saved activations, which the cotangent
does not perturb, so §0.1's quadratic never appears. Read §3.7 for that, and read it with its
three hypotheses (§3.7 step 4) — the wall relocates rather than vanishing.

⭐ **Two whole-net BACKWARD numbers now, and the second needs no operating point** —
MobileNetV2's, 4.750·10¹⁵³ / 1.076·10¹⁵² (§3.13, 2026-09-04). **§3.8's THREE items are ALL DONE** — the MobileNetV2 and EfficientNet-B0 backward probes
(**§3.9**) and r34's whole-net certified tie (**§3.10**), both 2026-09-03, and the CIFAR-8
`Env → Maps` coherence pass (**§3.11**, 2026-09-04).
Three sentences of §3.8: a
backward is *always* a fold (squeeze-excite included, so §0.1's list of quadratic sites is a
forward-only list); MobileNetV2's backward is statable with **no operating-point hypothesis at
all**, unlike ResNet-34's; and EfficientNet-B0's was a fold that could not be written down,
blocked by ONE unproved scalar lemma — a *global* bound on `|swish′|`, which §3.4 costed as
calculus and is three elementary steps. ✅ **That lemma landed 2026-09-04**
(`swishScalarDeriv_abs_le`, `Architectures/SwishSaturation.lean`), and B0's backward fold is now
**7.640·10¹⁶⁹ / 1.735·10¹⁶⁹** — statable, from 10⁴³¹. §3.12.

⭐⭐ **And §3.15's probe has been RUN: a LayerNorm net's BACKWARD folds.** ConvNeXt-T's is
5.766·10²⁴⁹ / 8.791·10²⁴⁸, ratio 0.15, no cap anywhere — **the repo's first honest whole-net fold
for a LayerNorm net**, at the net whose FORWARD number is a `2.00` cap (**§3.16**, 2026-09-04).
⛔ Read finding 1 with the number: because that forward is a cap, the fold is real and the
forward-then-backward composition it looks like it supports **does not exist** — §9 has the new
row. ⛔⛔ The probe also turned up the fifth instance of `imagenet_specs_drift_from_twins`: the
shipped ConvNeXt BACKWARD bridge held `id` in its head-LayerNorm slot — the same slot §3.3(b)
fixed on the FORWARD on 2026-09-03 — **and its docstring justified it with a LayerNorm count that
had gone stale** — and a THIRD copy of that count sits in the certified apex's docstring (§3.18).
✅ Fixed the same day; it needed no new leaves, and the `Maps` kit followed (**§3.17**).
⭐ **What is open is now §4**, and it is headed by the whole-net certified TIE, not by the number
(**§3.18**): §3.10's tie moved a committed number 4×, and this net has already produced one defect
of that class this week. ⭐ The apex it needs already exists at the committed net.

Read in this order: §0.1 (the one structural finding, with two failure modes not one), §9
(⛔ what a capped number is and is not — ViT's and ConvNeXt's are entirely of that kind), then
§3.5 for the transformer-specific work. §7 is the checklist you run before every commit —
including the "stage, then STOP and ask" rule.

⭐⭐ **The habit ViT added, and it cost three separate corrections: check that the granularity
you are folding at is EXPRESSIBLE before you trust the number.** `FloatBridgesTo` composes
single-input maps, so any op that fans out (attention: `X ↦ Q,K,V`) or that is defined as one
piece with a branch (the patch embed's `if n.val = 0`) must be ONE leaf — and the first probe had
decomposed both, at the granularity the emitted GRAPH spells. **The graph says what the kernel
does; the definition says what the theorem is about, and only the second constrains a `Maps`
chain.** Read the committed definition before planning a decomposition (§3.5, §3.5.2 item 4).

⭐ The single most useful habit from the five folds that landed: **when a fold overshoots, ablate
before concluding anything about the architecture.** Four times running the blocker was
something already true that the statement threw away — relu6's clamp, swish's modulus, seScale's
window, and then ConvNeXt's **profile** (a uniform parameter bound where the checkpoint has four
scales 14× apart). Each diagnosis was a Python probe, not a Lean session, and the last of them
was not a leaf lemma at all — which is the refinement to carry: **ablate the inputs too.**
⭐ ViT made it five, and the fifth is a repeat of the third: attention's window was derived as
`|real| + |float − real|` when a direct bound on the FLOAT side was available — `floatClose_seScale`'s
bug, one net later, and this time it was not merely loose but *unstatable* (§3.5). **When a
window contains an error term, ask why.**

## 0. Where we are

`FloatBridgesTo f fF` is a Type-valued bridge (`FloatComposeBridge.lean`): `mag : ℝ → ℝ`
(input window ↦ output window), `mod : ℝ → ℝ → ℝ` (input window ↦ error modulus), and
`close : ∀ A, 0 ≤ A → 0 ≤ mag A ∧ FloatClose A (mag A) f fF (mod A)`. Every combinator composes
`mag`/`mod` explicitly, every leaf writes them out. (Why it had to be data and not `∃ L`:
`formalization.yaml` fidelity §4d — the `∃`-modulus was discharged by `L := 2B`.)

| net | window | budget | budget/window | kind | file |
|---|---|---|---|---|---|
| CIFAR-8 (8 conv, no BN) | 6.121·10¹⁸ | 6.37·10¹⁴ | 1·10⁻⁴ | fold | `Cifar8FloatBudget.lean` |
| ResNet-34 @224², **inference BN** | 3.152·10²¹¹ | 1.548·10²⁰⁹ | 4.9·10⁻³ | fold | `Resnet34FloatBudget.lean` |
| MobileNetV2 @224², **inference BN** | **2.154·10³** | 1.444·10⁹⁶ | — | fold | `MobileNetV2FloatBudget.lean` |
| EfficientNet-B0 @224², **inference BN**, batched | 2.580·10⁵⁵ | 8.408·10²¹⁰ | — | fold | `EfficientNetFloatBudget.lean` |
| ConvNeXt-T @224², channel LN | 4.858·10²²⁷ | 9.706·10²²⁷ | **2.00** | ⛔ **cap** | `ConvNeXtFloatBudget.lean` |
| ViT-Tiny @224², depth 12, vector LN | 3.612·10²¹⁸ | 7.222·10²¹⁸ | **2.00** | ⛔ **cap** | `ViTFloatBudget.lean` |
| **ResNet-34 BACKWARD** @224², **training BN** | 8.857·10²⁴⁵ | 6.894·10²⁴⁴ | 7.8·10⁻² | ⭐ fold | `Resnet34BackFloatBudget.lean` |
| **MobileNetV2 BACKWARD** @224², **training BN**, ⭐ no operating point | 4.750·10¹⁵³ | 1.076·10¹⁵² | 2.3·10⁻² | ⭐ fold | `MobileNetV2BackFloatBudget.lean` |

Six nets now carry kernel-checked numbers — ⛔ **but not all six are the same claim.** The first
four are the interval fold and all four are vacuous as *budgets*; the point is that the kernel
checks them. **ConvNeXt-T's and ViT-Tiny's are the triangle inequality** — ConvNeXt's 23
LayerNorm sites and ViT's 25 LayerNorm *and* 12 attention sites all go through
`FloatBridgesTo.capped`, so they say "the float and the real forward both land in the certified
window" and not "the rounding error folds to this". `budget/window = 2.00` is the tell, and §9 is
the rule for saying so. There is no version of either net for which the fold exists (§0.1), so
this is not a weaker choice; it is the only statement available. ⭐ ViT's ONE honest stage is the
patch embed — it does not reduce.

The r34 number sits 157 orders above the adjoint chain's proven-H figure for
the same net (6.5·10⁵¹, `scripts/adjoint_chain_probe.py` §5), and the two documented reasons are
`layerBudget`'s uniform `m·w'·A` face (§5 of the probe measures 257× per stage) and worst-case
rather than measured windows.

⭐⭐ **MobileNetV2's WINDOW is not vacuous, and that is the result of the second net.** 2154,
against logits of a few — where ResNet-34's is 10²¹¹. One lemma does all of it: `relu6` is
bounded by 6 whatever its input, so `floatClose_relu6` now states `FloatClose A (min A 6)` and
every one of the net's 13 relu6 sites RESETS the certified magnitude. Without the clamp the same
fold gives 4.309·10¹⁰⁰ (`mnv2_eval_chain(relu6_clamp := False)`). ResNet-34 cannot have this;
plain `relu` has no upper clamp. ⚠ **And the budget moved one order, 3.072·10⁹⁷ → 1.444·10⁹⁶.**
Window and budget are separate levers (§3.2), and only `S = 1/√ε` moves the budget. MobileNetV2
also sits only ~36 orders above its chain figure (2.7·10⁶⁰) rather than r34's 157, because the
clamped window removes the window looseness and leaves only the gain looseness.

The machinery built for r34 and reusable for the rest:

* `FloatBudgetEnv.lean` — `FloatBridgesTo.Maps Ā Ē Ā' Ē'`, *"at every input window `A ≤ Ā` and
  every inherited error `E ≤ Ē`, the output window is `≤ Ā'` and the output error `≤ Ē'`."*
  Quantifying over the inputs rather than fixing them buys monotonicity, and monotonicity makes
  `Maps.comp` / `Maps.residual` / `Maps.biPathSum` **generic** — the CIFAR-8 `Env` needed one
  `comp_*` lemma per operation and could not express a skip at all. Leaves so far: `relu`,
  `maxPool`, `maxPool3s2`, `flatConv`, `flatConvStride2`, `dense`, `gap`,
  `bnPerChannelTensor3` (training), `bnEvalPC` (inference), `relu6` (⭐ window `min Ā 6`),
  `depthwise`, `depthwiseStride2Flat` (the last three in `MobileNetV2FloatBudget.lean`, which is
  where the files that define their leaves first meet the kit).
* `BnEvalRuntimeFloatBridge.lean` — inference BN as the render actually emits it.
* `Resnet34WholeFloatBridge.lean` — the block bridges are now generic in the normalisation
  (`rblkGen` / `rblkStridedGen`, with `rblkPC_eq_gen` / `rblkPStridedPC_eq_gen` both `rfl`), so
  one pair of block bridges serves the training net and the inference net.
* `MobileNetV2WholeFloatBridge.lean` — the same treatment for mnv2 (`invresBodyGen` /
  `invresBodyStridedGen`, `invresBodyPC_eq_gen` / `invresBodyStridedPC_eq_gen` both `rfl`), and
  `floatClose_relu6` strengthened to carry its clamp.
* `MobileNetV2RenderPCEval.lean` — the eval graph twin, mirroring `MobileNetV2RenderPC.lean` rung
  for rung. One shared ε (as the render emits), each BN site carrying its frozen mean and
  variance: 102 arguments → 123.
* `EnetFloatBridge.lean` — `floatClose_swish`'s modulus and `floatClose_seScale`'s window
  both tightened (§3.4); without either, EfficientNet-B0's fold is past `norm_num`'s ceiling.
* `EfficientNetWholeFloatBridge.lean` — the B0 stage/block bridges made generic in the
  normalisation (`cbsBGen` … `headFwdBGen`), with `*_eq_gen` `rfl` onto the training net and
  `*Eval_eq_gen` `rfl` onto the inference net. ⭐ Only the REAL side needed it — the float peers
  always took the float BN abstractly, since it was always a supplied hypothesis.
* `FloatBudgetEnvMBConv.lean` — the `Maps` leaves the inverted-bottleneck family needs on top of
  `FloatBudgetEnv`'s: `relu6`, `depthwise`, `depthwiseStride2Flat`, `swish`, `sigmoid`,
  `broadcast`, `seScale`, `batchMap`. They live here and not in `FloatBudgetEnv.lean` because a
  `Maps` lemma names its bridge and none of these bridges is on that file's import path; putting
  them there would make the ResNet-34 budget depend on the whole MobileNet/EfficientNet cone.
  ⚠ It also carries a worked B0 b1 squeeze-excite site as a compiled `example` — a `Maps` leaf
  nothing composes is a leaf nobody has checked composes.
* `FloatBudgetEnvLN.lean` — the three **modelled device kernels** (`DeviceLN`, `DeviceGelu`,
  ⭐ `DeviceExp` — the last added for ViT, and ⚠ its spec is RELATIVE where the other two are
  absolute, because `softmaxF_close` divides one exponential sum by another and only a relative
  error survives the quotient), with the capped LN site's bridge `DeviceLN.bridgeAt` and
  envelope `DeviceLN.mapsAt` built on them; ⚠ they were `ConvNeXtFloatBudget.lean`'s until ViT
  needed the same three, and that file's `DeviceLN.bridge`/`.maps` are now one-line delegations
  at its own profile. Plus the `Maps` kit a LayerNorm net needs: the capped pure-normalise LN
  (`Maps.bnCapped`), the two halves of its affine (`diagBack`/`biasAdd`), the gathers and the
  per-row lift they are conjugated by, `gelu` (through the `3/2` branch), `flatConvStride4`, and
  the three composites the whole-net chain walks (`Maps.chanLNTensor3` / `.cnxBlockChW` /
  `.cnxDownChW`). ⚠ It imports `FloatBudgetEnvMBConv` for one leaf, `Maps.depthwise`.
* `Architectures/SwishSaturation.lean` — `swishScalarDeriv_abs_le` (⭐ `|swish′| ≤ 2`, GLOBAL:
  the bound B0's backward `diagBack` slots take as their `Ssw`, where the only prior one was
  `swishScalar_lipschitz_abs`'s `1 + A/4` at the forward's window) plus `swishScalarDeriv_eq` and
  `swishScalar_lipschitz`. ⛔ The Lipschitz corollary is deliberately NOT wired into
  `floatClose_swish` — it is worth 8 orders on the forward and moves a committed number (§3.12).
* `Architectures/GeluSaturation.lean` — `geluScalarDeriv_abs_le` / `geluScalar_lipschitz`, moved
  down out of `Certificates/GeluLipschitz.lean` so `floatClose_gelu` can state the `min` of its
  magnitude polynomial and the global `3/2` (§3.3.0(b)).
* `FloatBudgetEnvAttn.lean` — the `Maps` kit a TRANSFORMER needs on top of the LayerNorm one:
  `Maps.softmaxRow` (⭐ window `1 + smCap`, CONSTANT in the input — a softmax RESETS the fold),
  `Maps.mhProjAttnFullCap` (⭐⭐ attention as ONE leaf at an exp-free window) and
  `Maps.patchEmbed` (⛔ also one leaf — `patchEmbed_flat` is a single definition with an
  `if n.val = 0` branch, not a composition). Plus the numeral plumbing every ViT stage needs:
  `smRho_le_of`, `smCap_le`, and `peRoundErrQ` / `patchEmbedRoundErr_le` for the one budget in
  the repo that is not already stated through a `gamma_num` rational.
* `ViTBlockVFloatBridge.lean` — the ViT assembly: the four structural ties, the vector-LN block
  (`floatBridgesTo_blockVFlat` / `Maps.blockVFlat`, 13 stages and 26 inequalities), the depth-`k`
  fold (`Maps.vitBodyKVFlat`, an envelope fold ConvNeXt does not have), and the whole net
  (`floatBridgesTo_vitForwardKV` / `Maps.vitForwardKV`). ⭐ Also `FloatBridgesTo.ofEq`, the
  transport of a bridge along a tie that keeps `mag`/`mod` DEFINITIONAL — `▸` does not, and a
  bridge whose `.mag` will not reduce cannot carry a `Maps`.
* `ViTFloatBudget.lean` — the number (731 lines, ⭐ **30 s** to elaborate against ConvNeXt's
  ~2 min at the same order of inequalities, because `Maps.vitBodyKVFlat` is an envelope fold:
  the whole-net proof is three `have`s, two `refine`s and twelve block applications, where
  ConvNeXt's spells all 183 stages out).
* `scripts/float_budget_envelope.py` — the exact-rational fold in the lemmas' semantics, the
  4-significant-figure round-up, the re-assertion passes (`verify_r34`, 180 inequalities;
  `verify_mnv2`, 116; `verify_b0`, 96; `verify_cnx`, 366; `verify_vit`, 324; and the three
  backwards' `verify_r34_back`, 252, `verify_mnv2_back`, 136, `verify_b0_back`, 138) and the
  numerals. Its CIFAR-8
  regression case reproduces `Cifar8FloatBudget.lean` stage for stage, and `cnx_eval_chain`'s
  three flags (`ln_cap`, `gelu_sat`, `head_ln`) reproduce §3.3's ablation table.
  ⭐ `vit_chain` (added 2026-09-03) is the ViT-Tiny sizing fold — 162 stages, five flags, and
  unlike the others it returns `(rows, exp_tainted)`: the tag list of stage numerals that would
  contain a `Real.exp` with no rational bound. "Statable" for a net with a transcendental leaf is
  `max(exponent) < 300` **and** no taint, and §3.5's table is the case where those two disagree.
  ⚠ It must fold with the **rounded** γ (`r4(gamma_q k)`), not the exact `(1+u)^k − 1` — the
  Lean chain passes the rounded one, and folding with the exact value silently emits stage
  numerals a hair too small, which the kernel then rejects. That is what the re-assertion pass
  is for; it caught exactly this.
  ⚠ **`ilog10`'s seed was `len(str(·))`, and CPython caps int→str at 4300 digits (3.10.7+)** — so
  every ablation past ~10⁴³⁰⁰ raised `ValueError` instead of a number, which silently included
  ConvNeXt's own §3.3 rows. Fixed 2026-09-03 to a `bit_length` seed (exact, unguarded; the two
  correction loops absorb its ±1). ⚠ With it running again, §3.3's ablation table reproduces its
  SHAPE but not its digits (the script now says 4.858·10²²⁷ / 10¹¹³⁴³ where that table says
  10²³³ / 10¹¹⁶³¹); §0's committed numbers and all four `verify_*` passes reproduce EXACTLY, so
  the script is in sync with the Lean files and the §3.3 ablation digits are the stale ones.

## 0.1 ⭐⭐ The finding: the modulus is QUADRATIC in the window wherever a normalisation reduces

This is the thing that changes the plan, so it goes first.

⭐ **And after ViT there are TWO ways a net fails to have a number, not one.** A *magnitude*
failure is this section: the fold squares at every reducing site and runs past `norm_num`'s
~10³⁰⁰ ceiling. A *representability* failure is ViT's: the numeral would be perfectly small, but
some stage carries a `Real.exp` at an argument with no rational bound, so **there is no numeral
at all**. §3.5's ablation table has a row where the magnitude is *identical* to the shipped one
and 36 stages are still unwritable. ⚠ A Python fold hides the second kind — `math.expm1`
overflows to a finite float and the chain sails on — which is why `vit_chain` returns an
`exp_tainted` tag list and "statable" means small enough AND untainted. Build that instrument
into any probe of a net with a transcendental leaf.

Training-mode BatchNorm reduces its statistics out of its own input, so perturbing the input
moves the mean AND the variance. `FloatModel.bnReluBudget` therefore carries

    G · 2A · (8A·e / (2ε√ε))          — quadratic in the window A

on top of the rounding. Fold that through a net and the budget does not grow geometrically, it
**squares at every normalisation site**. Measured with the generator: the same r34 fold on the
training-mode net gives window 10²²¹ and budget **10⁷⁴¹⁷** — past the point where `norm_num`
will evaluate the numeral at all (it refuses around 10³⁰⁰; verified). There is no numeral to
write down, so there is no theorem to state.

Inference BN has no reduction: `μ` and `rsqrt(var+ε)` are frozen constants, the map is affine in
`x` with slope `γ·s`, and the modulus is `rounding + G·S·e` — **linear**. The fold then behaves
exactly like CIFAR-8's, budget ≈ 10⁻³ of the window, and the number exists. That is the whole
reason r34's number is stated at inference BN.

**LayerNorm has the same quadratic term and no escape.** `floatClose_layerNorm`
(`ViTFloatBridge.lean`) IS `floatClose_bn` — same `bnReluBudget` modulus — and LN has no
running statistics, so there is no eval-mode variant to switch to. So ConvNeXt-T and ViT-Tiny
hit this wall unconditionally, and §3.3's "ConvNeXt-T is the cheapest ImageNet closure" is right
about the *closure* and wrong about the *number*. There are three escapes; ConvNeXt-T landed on
the first:

1. ✅ **A cap — LANDED 2026-09-03 as `FloatBridgesTo.capped` (`FloatBudgetEnv.lean`).** Any
   modulus can be replaced by `2·mag` (both float and real outputs lie in the certified window),
   so a bridge carries `mod' := min(mod, 2·mag)`. It stops the squaring cold, because `mag`
   depends on the input WINDOW and not on the inherited error: a capped site resets the error
   however big it arrived. ⛔ It is the triangle inequality, not the fold, and it must be
   labelled wherever it is used (§9). ConvNeXt-T is `10²²⁷ / 10²²⁷` with it and `10²³³ / 10¹¹⁶³¹`
   without.
   ⚠ **The earlier note here said "it is not sufficient on its own — the cap alone is still
   `10⁸³²`, because GELU is separately CUBIC in the window". That was measured against the net
   the whole-net bridge described at the time, which had `id` in its head-LayerNorm slot.** With
   the real head LN restored (§3.3) the last GELU is capped like every other, and the cap IS
   sufficient on its own: cap + cubic GELU is `10²³³`, the same as cap + the saturation constant.
   The saturation constant is still worth having — without the head LN it is the difference
   between `10⁸⁹⁶` and `10²³⁰` — but it is not what makes ConvNeXt statable, and the "neither
   escape alone is enough" finding was an artifact of a stale slot.
2. **A tighter input-sensitivity for the reducing normalisation.** The `A²/ε^{3/2}` factor is
   worst-case-at-the-ε-floor twice over. An operating-point variance floor `V` shrinks it by
   `(V/ε)^{3/2}` but leaves it quadratic, so it postpones the wall rather than removing it.
   A genuinely linear bound needs the *normalised* output's Lipschitz constant, not the
   pre-normalisation one — that is real work and probably the interesting result.
3. **Accept a per-site cap on `eistd`.** `|istd| ≤ 1/√ε` bounds the inverse-stddev, so
   `eistd ≤ 2/√ε` always; that removes the `eistd` growth but not the `D²` growth.

**⭐ Squeeze-excite is a third reducing site (added 2026-09-03, §3.4).** `seScale`'s modulus
`mulErr q A Cg E Eg` carries `A · Eg` — the block window times the gate's error — and the gate
grows that error out of the same window through the squeeze's `GAP → dense`. So SE is quadratic
in the window for the same structural reason BN and LN are: **a reduction feeds back
multiplicatively against the thing it reduced.** EfficientNet-B0 survives it (each SE site
doubles the budget's exponent, and there are three); a net with twenty would not. When looking at
a new architecture, the question to ask is not "does it normalise" but "does any op consume a
reduction of its own input and then multiply by that input".

**⭐⭐ Attention is a FOURTH site and it is not quadratic — it is EXPONENTIAL-of-quadratic
(added 2026-09-03, §3.5).** `attnOutInErr`'s second term is
`vA·(Real.exp (2·scaleA·attnScoreInErr d qA kA e) − 1)` with `attnScoreInErr = d·((qA+e)·e + kA·e)`:
the inherited error appears quadratically *inside an exponential*. Same structural cause as the
other three — softmax reduces over the tokens and the result is multiplied back against V — one
rung further up. So the question to ask a new architecture, refined once more: not "does it
normalise", not "does an op consume a reduction of its own input and multiply by that input", but
**how does the inherited error enter — linearly, quadratically, or under a transcendental?** The
third case is qualitatively different, because it fails by REPRESENTABILITY rather than by size
(§3.5's table: identical magnitude, 48 numerals that cannot be written).

Recording the size of the training-mode number (10⁷⁴¹⁷, script-computed, not kernel-checked) is
itself a result: it is why the adjoint chain exists.

⭐⭐ **AND EVERY ONE OF THOSE FOUR SITES IS A FORWARD FACT. The closing sentence, settled
2026-09-03 by the MobileNetV2 and EfficientNet-B0 backward probes (§3.8 item 1): a VJP is linear
at a fixed point, so EVERY backward folds — squeeze-excite included.** The obvious suspect was
SE, because §0.1 lists it as the third quadratic site and its backward genuinely fans out and
rejoins. It is not one. `seInputGrad g xinp gateBack = biPathSum (diagBack g) (gateBack ∘
diagBack xinp)` (`SEBackFloatBridge.lean`): the gate `g` and the input `x` are **saved
constants**, both branches are linear maps, and nothing multiplies the cotangent by itself. The
forward's quadratic came from the gate being grown out of the same input the rescale multiplies;
on the backward that input is not the cotangent. Measured ratios: r34 0.078, MobileNetV2 0.034,
EfficientNet-B0 0.186 — three folds, no cap anywhere.

⛔ **"Folds" is not "is statable", and B0 is the counterexample — which relocates the wall a
second time.** A backward's numerals are built from the SAVED ACTIVATIONS' magnitude bounds, and
§3.7's escape was that BatchNorm's enter *normalised* (`|x̂| ≤ √n`). B0 has two that do not:
`swish'(saved)`, whose only repo bound is `swishScalar_lipschitz_abs`'s window-dependent
`1 + A/4`, and the SE's own saved input `x`, a post-swish activation with no clamp at all. Both
import the forward's certified window LINEARLY — no new quadratic, just a huge constant — and B0's
backward lands at 10⁴³¹. So the refined question for a new architecture's BACKWARD is neither
"does it normalise" nor "how does the error enter": it is **which saved activations does it scale
by, and is each one bounded by something other than the forward's window?** (§3.8 has the
ablation, and the answer for B0 is one unproved scalar lemma.)

## 1. Goal, non-goals, success

**Goal.** For each committed ImageNet-scale forward, a closed `FloatBridgesTo` over the real
leaves and a theorem `<net>_float_logits_le` stating a number. ✅ **Met, six for six, 2026-09-03.**
Backwards are phase 2 (§3.7).

**Non-goals.** Making the numbers small. They are the interval fold and that is the honest
content. Do not chase the adjoint chain here.

**Success per net.** (i) `<net>Bridge` with no `FloatBridgesTo` hypotheses; (ii) a tie to a
committed real def — see the r34 caveat in §3.1; (iii) `<net>Bridge_maps` and
`<net>_float_logits_le`; (iv) the number cross-checked against the probe; (v) 3-axiom clean, in
`AuditAxioms`, disclosed in `formalization.yaml` §4d; (vi) ⛔ **if §3.3.0(a)'s cap bites
anywhere, say so** — the statement is then the triangle inequality and not the fold (§9).

## 2. The mechanism

Read `Resnet34FloatBudget.lean` top to bottom (620 lines). The pieces:

* `Maps` is a Prop-**structure**. It MUST stay one. The first CIFAR attempt used a `def`
  unfolding to `∧`; the unifier then delta-unfolded the whole `.mag` chain and timed out at 20×
  the heartbeat budget. Inductive types unify argument-wise and never unfold.
* Bundle the parameters. `R34Weights` is 19 fields, not 424, because each conv/BN/block gets its
  own little record (`R34Conv`, `R34Bn`, `R34IdBlk`, `R34DownBlk`) carrying its own bounds, and
  the numeric profile is one more record (`R34Profile`) instead of ten `0 ≤ _` arguments.
* Give each block a `maps` lemma so the whole-net chain is 22 steps, not ~90. Its proof is the
  leaf steps chained with the generic `Maps.comp`.
* Pass dims explicitly (`(h := 56) (w := 56)`) on every block and leaf; `0 < c*h*w` side goals
  elaborate before the dims are solved otherwise.
* The BN goals need `norm_num [bnNormBudget, FloatModel.mulErr, u32]`; plain `norm_num` cannot
  unfold the budget. The residual/`gap` goals need at least `[u32]` because `q := u32`.
* Elaboration: ~64 s for the r34 file at `maxHeartbeats 4000000`, `maxRecDepth 1000000`.

## 3. The work, net by net

| net | state | what closing needs |
|---|---|---|
| **ResNet-34 fwd** | ✅ **DONE** (inference BN) — `r34_float_logits_le`, 1.548·10²⁰⁹, tied to the graph | — |
| **MobileNetV2 fwd** | ✅ **DONE** (inference BN) — `mnv2_float_logits_le`, window **2154** / budget 1.444·10⁹⁶, tied to the graph | — |
| **EfficientNet-B0 fwd** | ✅ **DONE** (inference BN, any batch size) — `b0_float_logits_le`, window 2.580·10⁵⁵ / budget 8.408·10²¹⁰, tied to the graph | — |
| **ConvNeXt-T Ch fwd** | ⛔ **DONE (2026-09-03), and it is the CAP not the fold** — `cnx_float_logits_le`, window 4.858·10²²⁷ / budget 9.706·10²²⁷, tied to the committed net | — |
| **ViT-Tiny fwd** (`vitForwardKV`) | ⛔ **DONE (2026-09-03), and it is the CAP not the fold** — `vit_float_logits_le`, window 3.612·10²¹⁸ / budget 7.222·10²¹⁸, tied to the committed spec's denotation | — |
| **ResNet-34 BACKWARD** (`r34InputGrad`) | ⭐ **DONE (2026-09-03), the FOLD, and TIED to the certified whole-net VJP** — `r34_grad_float_le`, window 8.857·10²⁴⁵ / budget 6.894·10²⁴⁴, at TRAINING BN; ⛔ three hypotheses, §3.7; the tie is §3.10 | — |

### 3.1 ResNet-34 — what landed, and the one thing left

`r34EvalBridge` is the `r34Forward` skeleton with `bnPerChannelEvalTensor3` at all 33 BN sites,
and the tie is now closed at the graph. `ResNet34RenderPCEval.lean` mirrors
`ResNet34RenderPC.lean`'s part 2 one rung for one — `idBlockGraphPCEval` /
`downBlockGraphPCEval` + `_faithful`, then `resnet34FwdGraphFullPCEval` + `_faithful` against
`resnet34Forward_full_pc_eval` — and in the budget file `r34EvalForward_eq_full_pc_eval`
(`rfl`) + `r34EvalGraph_faithful` compose them, with `r34_float_logits_le_committed` restating
the number on the committed net. The eval forward now has exactly the standing the training
forward has, and the only remaining modelled input is `DeviceRsqrt`'s accuracy, which is
irreducible (a device `rsqrt` has no IEEE spec).

⚠ The eval graph needs 219 arguments where the training one needs 146 — each BN site carries
its frozen mean and variance as well as γ/β. Generate that signature, do not type it. The SSA
names follow `ResNet34Render.lean`'s `bnSite` convention (`%{p}n1mu`/`%{p}n1var`, `stn` for the
stem) so the typed graph and the emitted text name the same inputs; names never enter `den`.

⚠ **`BnEvalFloatBridge.lean`'s premise is wrong and is now corrected in place.** It models
deployed BN as a pre-folded affine `a·x + b` and says *"there is no batch reduction and no
runtime `rsqrt`"*. The first half is the point; the second is not what this repo emits —
`.bnPerChannelEvalF` expands to `subtract`, `add`, `rsqrt`, `multiply`, `multiply`, `add`
(`StableHLO.lean`), i.e. the rsqrt is on device. `BnEvalRuntimeFloatBridge.lean` bridges the
kernel we ship; `floatClose_bnEval`'s affine remains true and remains about a program we do not
run. Do not build the other nets' numbers on the affine.

### 3.2 MobileNetV2 — ✅ DONE (2026-09-03). What landed, and the one lesson to carry forward.

`mnv2_float_logits_le` / `mnv2_float_logits_le_committed`: window **2154**, budget
**1.444·10⁹⁶**, at `|param| ≤ 28/10` (global max 2.7157 on
`/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin`), `ε ≥ 10⁻⁵`, `es = 10⁻²`, `u ≤ 2⁻²⁴`.
Files: `MobileNetV2FloatBudget.lean` (725 lines, ~26 s to elaborate), plus
`MobileNetV2RenderPCEval.lean` and the `invresBody*Gen` layer in
`MobileNetV2WholeFloatBridge.lean`. The r34 recipe transferred step for step; nothing in §2 or §7
needed changing.

**⭐⭐ The finding, and it is the reason to have done a second net: WINDOW AND BUDGET ARE
SEPARATE LEVERS.** `floatClose_relu6` stated `FloatClose A A` — magnitude passed straight
through — even though `relu6 n x i = min (max (x i) 0) 6` is bounded by `6` *whatever its input*.
Strengthening it to `FloatClose A (min A 6)` (`relu6_le_six` for the magnitude clause; the error
clause is untouched) plus a `Maps.relu6` peer is ~15 lines, and:

| | output window | fresh budget |
|---|---|---|
| `relu6` as pass-through (the old leaf) | 4.309·10¹⁰⁰ | 3.072·10⁹⁷ |
| `relu6` clamped at 6 | **2.154·10³** | 1.444·10⁹⁶ |

97 orders of window; one order of budget. Error gain per stage does not care how small the
window is — it is `G·S` per BN site, and `S = 1/√ε ≤ 317` at the ε-floor. Sweeping `S` over the
same fold:

| `S` | 317 (ε-floor at 1e-5) | 32 | 4 | 1 (σ² ≈ 1) |
|---|---|---|---|---|
| budget | 10⁹⁶ | 10⁷⁷ | 10⁶⁰ | 10⁴⁸ |

~19 orders per decade of `S` across the 20 BN sites. So the operating-point variance floor
(§0.1 item 2) is worth ~48 orders here and **nothing else is**. ⭐ Carry this into every
remaining net: an activation with an upper clamp buys window and only window, and the budget
only ever moves when `S` does.

**What this bought.** MobileNetV2's certified window is `2154` against logits of a few — the
first ImageNet-scale window in the repo that is not vacuous by hundreds of orders — and the fold
sits only ~36 orders above the probe's adjoint-chain figure for this net (2.7·10⁶⁰), against
r34's 157, because the clamped window removes the window looseness and leaves only the gain
looseness.

**Two things that cost time, for the next net.**
1. **Associate the block envelope the way the block BRIDGE composes.**
   `floatBridgesTo_invresBodyGen` groups the project stage as one unit
   (`prefix.comp (conv_p.comp bnp)`), so `MnvBlock.bodyMaps` has to build `pj` separately and
   finish `s6.comp hnm pj`. Chaining eight flat `.comp`s instead gives an intermediate-dimension
   mismatch on the closing `exact` (`0 < oc*h*w` where `0 < mid*h*w` is wanted) — the error
   points at the `hn` argument, not at the association.
2. **The rounded skip fan-in needs `norm_num [… u32]`,** like every BN goal — `Maps.residual`'s
   `rA`/`rE` carry `q := u32`. Plain `norm_num` leaves them open.

⛔ The plan's earlier claim that "the blueprint's `2.7·10⁶⁰` is the proven-H budget for this net
— the fold should land in that order" was wrong on both counts: that figure is the probe's
EVAL-BN *adjoint-chain* budget, not an interval fold, and the fold lands ~36 orders above it.


### 3.4 EfficientNet-B0 — ⭐ UNBLOCKED (2026-09-03). Two leaf bounds were the wall.

**✅ DONE (2026-09-03).** `b0_float_logits_le` / `b0_float_logits_le_committed`: window
**2.580·10⁵⁵**, budget **8.408·10²¹⁰**, **for any batch size `N > 0`**. Files:
`EfficientNetFloatBudget.lean` (794 lines, ~20 s), plus `EfficientNetRenderPCEval.lean`, the
`Maps` leaves in `FloatBudgetEnvMBConv.lean`, the `*BGen` layer in
`EfficientNetWholeFloatBridge.lean`, and the two tightened leaves in `EnetFloatBridge.lean`.

⭐ **Any batch size** is new: at inference every stage is `batchMap N` of a per-example op, and
`Maps.batchMap` carries an envelope through the lift unchanged, so `N` never enters a numeral.
The other two nets' numbers are per-example by construction; this one is genuinely batched and
the bound does not degrade with `N`. At the profile measured on `/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin`
(5,288,548 f32, global max `|·| = 4.0545`, 99.99th pct 1.3949, 100 entries above 2 — so
`|·| ≤ 41/10`), `ε ≥ 10⁻⁵`, `es = esig = 10⁻²`:

    output window  ≤ 2.580·10⁵⁵      budget ≤ 8.408·10²¹⁰      (`b0_eval_chain`, `verify_b0`, 96 ineqs)

**⛔ It did not start there.** The first honest fold came out at window `10¹⁷¹⁷`, budget `10⁴¹⁷⁹` —
`norm_num` refuses numerals past ~10³⁰⁰, so there was no theorem to state, and unlike §0.1 this had
nothing to do with the BN mode (the fold was already at inference BN). Two leaf bounds did it, and
**both were the relu6 pattern: a bound proved one lemma down and thrown away by the generic
combinator.** Ablated at the measured profile:

| | window | budget | statable |
|---|---|---|---|
| both leaves as they stood | 10¹⁷¹⁷ | 10⁴¹⁷⁹ | no |
| swish fixed only | 10²¹¹ | 10²¹² | yes |
| seScale fixed only | 10⁵⁵ | 10⁷⁹⁹ | no |
| **both fixed** | **10⁵⁵** | **10²¹¹** | **yes** |

1. **`floatClose_swish`'s modulus was `mulErr + (1 + A/4)·e`** — it multiplies the inherited error
   by the WINDOW at every swish site, and B0 has nine of them. `swishScalar_lipschitz_abs` proves
   that constant from `|σa − σb| ≤ ¼|a−b|`; bounding the same factor by the gate's own *range*
   instead (`|σa − σb| ≤ 1`, `sigmoidScalar_sub_abs_le_one`) gives `A + |a−b|` —
   **additive in the window rather than multiplicative** (`swishScalar_lipschitz_abs'`). The two
   are incomparable, so `floatClose_swish` now states their `min`. ⛔ **The sentence that stood
   here — *"the true global Lipschitz constant of `x·σ(x)` is ≈ 1.1; getting THAT needs the decay
   of `σ'`, i.e. calculus, and the additive bound is what avoids needing it"* — was true of the
   SHARP constant and cost a month.** A CRUDE global constant needs no calculus at all: `≤ 2` is
   three elementary steps (§3.12), and the only property that matters is that the window does not
   appear. ⚠ It is also a forward improvement worth 8 orders here, priced and NOT taken —
   `floatClose_swish` still states the two window-dependent branches; §3.12.
2. **`floatClose_seScale`'s window was derived as `|float − real| + |real|`** — which charges
   `A · Lg 0`, the block window times the GATE'S ERROR, to the magnitude. But `FloatClose`'s
   magnitude clause bounds the *float* gate as well as the real one, so the float product is one
   rounding above `A·Bg` and the gate's error never needed to enter the window at all. The window
   is now `A·Bg·(1+u)`. Worth 10¹⁸ per SE site.

**⭐ What SE still costs, and it is §0.1's shape again.** Each SE site roughly DOUBLES the budget's
exponent (`10²³ → 10⁸⁰ → 10¹⁹⁶` across b1/b2/b3): `seScale`'s modulus is
`mulErr q A Cg E Eg`, whose `A · Eg` term multiplies the block window by the gate error — and the
gate error is itself grown from that same window by the squeeze's `dense`. **The SE block's
modulus is quadratic in the window**, exactly like training-mode BN and LayerNorm. B0 survives it
only because it has three SE sites; twenty would end it the way §0.1's 33 BN sites end r34's
training-mode fold. Add SE to §0.1's list of reducing sites.

**What is left.** ✅ (ii)+(iii) landed 2026-09-03: `EfficientNetRenderPCEval.lean` is the eval
forward + graph + `efficientnetFwdGraphBEval_faithful`, mirroring `EfficientNetRenderPC.lean` rung
for rung with `.bnBatchF` replaced by the batched `.bnEval` DESCRIPTOR. ⭐ `bnBatchLA` is the ONE
op in this net that is not `batchMap N` of a per-example op — it reduces μ/var across examples,
which is why it has its own constructor instead of being a `BatchableOp`; at frozen statistics
there is no reduction, so `bnEval` IS a descriptor and `den_batchOp_bnEval` already proved it
denotes `batchMap N (bnPerChannelEvalTensor3 …)`, `rfl`. Every stage of the eval forward is
therefore `batchMap`-of-a-per-example-op or pointwise, and every leaf the budget needs is a
per-example leaf the repo already has. ✅ (v) The numerals are generated and re-asserted
(`b0_eval_chain` / `verify_b0`, 96 inequalities).

✅ **(i)** landed the same day: `FloatBudgetEnvMBConv.lean` holds `swish`, `sigmoid`,
`broadcast`, `seScale` and `batchMap`, and `relu6` / `depthwise` / `depthwiseStride2Flat` moved
there out of `MobileNetV2FloatBudget.lean` (B0 is the third net, which is the condition §5 set for
moving them). B0's b1 SE site is closed there as a compiled `example` at the generated numerals,
so nothing in the kit is unexercised.

✅ **The budget file landed.** `EfficientNetFloatBudget.lean` on the r34 recipe: records
(`EnetConv`/`EnetDw`/`EnetSE`/`EnetBn`/`EnetNoExpBlk`/`EnetMBBlk`/`EnetWeights`/`EnetProfile`), a
`DeviceSigmoid` alongside `DeviceRsqrt`, per-site `EnetBn.maps` and `EnetSE.maps`, the three block
envelopes, the whole-net chain and the ties.

⚠ **The one thing that did not transfer: the tie is NOT one `rfl`.** At these concrete dims the
kernel times out comparing `b0EvalForward` with `efficientnetForwardBEval` — r34's and mnv2's
whole-net ties are single `rfl`s and this one cannot be. It rewrites with the nine per-stage
`*Eval_eq_gen` lemmas first, which leaves nothing to compare; that is why the `*BGen` layer's eval
ties are stated per stage rather than only whole-net, and it is the same reason
`efficientnetFwdGraphB_faithful` is `rw`-based. Budget for this on the next batched net.

✅ **The `*BGen` layer** landed too. `cbsB`/`stemB`/`dwbsB`/`dwbsSB`/`projB` and the four blocks
each have a `*Gen` peer abstract in the normalisation, a `*_eq_gen` `rfl` onto the TRAINING net
(`bnBatchLA`) and a `*Eval_eq_gen` `rfl` onto the INFERENCE net
(`batchMap N (bnPerChannelEvalTensor3 …)`), and all nine bridges are stated on the `Gen` defs with
the nine originals delegating — so one set of bridges serves both modes and there is no second
proof. ⭐ **Only the REAL side needed generalising**: `cbsBF` and its peers already took the float
BN abstractly, because the float BN was always a supplied hypothesis (the one batch-coupled op).
The file closes B0's b1 block at the eval BN as a compiled `example` with NO `FloatBridgesTo`
hypothesis left — the exact shape the budget file needs at all ten sites.

Nothing is open for B0.

⚠ **Not stated, and worth stating:** `efficientnetForwardBEval N = batchMap N (per-example
forward)` — the whole-net form of "at inference the batch decouples". Only the per-SITE claim is
proved (`den_batchOp_bnEval`). It needs a `batchMap N f ∘ batchMap N g = batchMap N (f ∘ g)` lemma
(the repo has `batchMap_pointwise` but not this) plus a per-example B0 def to be the witness.

⛔ **An earlier note in this file said "B0's activation is swish, which is unbounded above, so
§3.2's clamp does not apply and the window will compound the way ResNet-34's does." The
conclusion was right and the reason was wrong.** `|x·σ(x)| ≤ |x|`, so swish is
magnitude-NON-increasing — its `mag` is `A + mulErr`, like relu's. It does not amplify the window;
it simply does not RESET it, which is the relu case, not an unboundedness case. What actually
threatened B0 was swish's *modulus* and SE's *window*, neither of which that note mentions.

### 3.3.0 ✅ The two shared prerequisites — BOTH LANDED 2026-09-03

Both are the pattern the previous three commits established: **a bound that is already true,
which the leaf does not state.** They are what ConvNeXt-T's number was built on, and ViT-Tiny
inherits them.

**(a) ✅ `FloatBridgesTo.capped` — the §0.1 escape-1 combinator (`FloatBudgetEnv.lean`).** `FloatClose`'s magnitude clause
bounds the real AND the float output by `mag A`, so their difference is at most `2·mag A`,
always. As a bridge transformer:

```
FloatBridgesTo.capped (b : FloatBridgesTo f fF) : FloatBridgesTo f fF
  mag := b.mag
  mod := fun A e => min (b.mod A e) (2 * b.mag A)
```

⭐ The shipped `Maps.capped` takes **only the WINDOW bound** — `hmag : ∀ A, 0 ≤ A → A ≤ Ā →
b.mag A ≤ Ā'` and `2 * Ā' ≤ Ē'` — because the output error is `2·Ā'` whatever the underlying
modulus does. That is the mechanism, not a convenience: at a capped site the quadratic term is
never turned into a numeral, so `norm_num` never meets it. `Maps.bnCapped` (`FloatBudgetEnvLN`)
is the LN leaf in that shape.
⚠ **Label every use.** Wherever the `min` selects the `2·mag` branch the result is the triangle
inequality, not the fold — the statement becomes "both maps land in the certified window", which
is much weaker than "the rounding error is bounded by the fold". It bites at *every* LN site of
ConvNeXt-T, so that number is entirely of that kind (§9).

**(b) ✅ GELU's modulus — and ⛔ the bound did NOT have to be written; it was already in the
repo.** `floatClose_gelu`'s modulus was

    egelu + (1 + √(2/π)/2 · A · (1 + 3·0.044715·A²)) · e        — CUBIC in the window

and the plan here was to prove the additive `A + |a−b|` split, the way swish got one. That was
unnecessary: `Certificates/GeluLipschitz.lean` had already proved the *global* constant
`|gelu′| ≤ 3/2` (`geluScalarDeriv_abs_le` / `geluScalar_lipschitz`) — saturation-aware, because
past the small-`|x|` region `gelu′`'s `sech²` decays like `e^{−2√(2/π)|x|}` and beats the cubic
growth — and it sat **one import above** the float bridge that needed it, written for the
adjoint chain. `3/2·e` beats the additive `A + e` everywhere the window is not tiny.
The fix was therefore a MOVE, not a proof: the analysis is now
`Architectures/GeluSaturation.lean` (imports only `LayerNorm.lean`), `floatClose_gelu` states
the `min` of the polynomial and `3/2·e` as `floatClose_swish` states its `min`, and
`Maps.gelu` closes through the `3/2` branch (the polynomial branch carries `√(2/π)` and could
not be `norm_num`'d even where it is tighter).
⭐ **The lesson is the sharpest instance of this file's habit yet:** before writing a tighter
leaf bound, grep the repo for it. The chain tier and the float tier had proved and needed the
same constant for a month without meeting.

### 3.3 ConvNeXt-T — ⛔ DONE (2026-09-03), and it is the CAP, not the fold

`cnx_float_logits_le` / `cnx_float_logits_le_committed`: **window 4.858·10²²⁷, budget
9.706·10²²⁷**, at the measured 300-epoch profile, `ε ≥ 10⁻⁵`, device LayerNorm statistics and
device GELU accurate to `10⁻²`, `u ≤ 2⁻²⁴`. Files: `ConvNeXtFloatBudget.lean` (781 lines, ~2 min
to elaborate — 183 numeric stages, 366 inequalities), plus `FloatBudgetEnvLN.lean`,
`Architectures/GeluSaturation.lean`, `FloatBridgesTo.capped` in `FloatBudgetEnv.lean`, and the
head-LN and four-bound fixes in `ConvNeXtWholeFloatBridge.lean`.

⛔ **`budget / window = 2.00`.** All 23 LayerNorm sites are discharged by `Maps.bnCapped`, so the
statement is "the float and the real forward both land in the certified window" — the triangle
inequality — and NOT r34/mnv2/B0's "the rounding error folds to this". Say it every time (§9).
It is still worth having: it is the only kernel-checked whole-net statement about a LayerNorm net
in the repo, and its WINDOW half is an honest fold.

**⭐ No eval twin was needed and none is possible** — LN has no running statistics, so unlike the
other three nets there is no second render to build. That made ConvNeXt *cheaper* than B0, as
predicted, and it is also exactly why the number has to be capped (§0.1).

**⭐⭐ Two things had to be right besides the cap, and neither was about the architecture.**

**(a) The measured profile does not split uniformly, and a uniform bound is unstatable.** On the
finished 300-epoch run (`/home/skoonce/convnext/convnext_t300_4gpu/convnext_tiny_imagenet.bin`,
28,587,592 f32):

| kind | count | max | bound used |
|---|---|---|---|
| conv / dense kernels | 28,524,000 | **0.5962** | `w' = 6/10` |
| biases + LayerNorm β | 49,576 | 2.9499 | `bb = 3` |
| LayerNorm γ | 7,392 | 4.7700 | `gl = 48/10` |
| layer scale | 6,624 | **8.3766** | `sl = 84/10` |

A single uniform bound is `8.4` — 14× loose on exactly the entries the conv fan-in multiplies —
and that fold lands at **`10³⁰¹`**, past `norm_num`'s ceiling. Split it is `10²²⁷`.
`CnxBlockChBounded` / `CnxDownChBounded` therefore carry four bounds and three, not two.
⚠ This is the file's ⭐ habit one level up: the looseness was in the INPUT, not in a lemma, and
the ablation that found it is the same three-line Python probe.
⚠ The checkpoint predates the 2026-08-30 head-LN restoration (it is short by exactly
1,536 = 2×768, which is how the missing layer was found), so the head LN's γ/β are not in the
measurement; they initialise at γ=1, β=0 and the bounds cover them with room. The plan's earlier
"provisional profile, global max 1.1135 at e9" was the in-progress run — the finished one is 7×
larger, and it changes the answer.

**(b) ⛔ The head-LayerNorm slot was stale, so the whole-net bridge described a different net.**
`convnextCh_floatBridges` and `convnextCh_floatBridgesTo` both held `id` in the `lnHead` slot,
while `WholeNetForwardTies.convNextForwardTCh_eq_skeleton` had carried
`rowLNVecFlat 1 768 w.hε w.hγ w.hβ` since the head LN came back on 2026-08-30. Their docstrings
claimed to tie through that lemma; they could not. Both now take the real head LN, its two
bounds and its bridge (23 LN hypotheses, not 22). ⚠ The `imagenet_specs_drift_from_twins`
lesson exactly: *"same net as the tie" is an unchecked claim until something forces the two
statements to unify* — and what finally forced it was needing the tie for a number.

**⭐ Fixing (b) also changed the §0.1 finding.** The ablation, at the committed profile
(`cnx_eval_chain`'s `ln_cap` / `gelu_sat` / `head_ln` flags):

| variant | window | budget | statable |
|---|---|---|---|
| leaves as they stood (no cap, cubic GELU) | 10²³³ | 10¹¹⁶³¹ | no |
| saturation GELU only | 10²³³ | 10⁵³⁹⁰ | no |
| **LN cap only, cubic GELU** | 10²³³ | **10²³³** | **yes** |
| both (shipped) | 10²³³ | 10²³³ | yes |
| cap + cubic GELU, **head LN removed** | 10²²⁹ | 10⁸⁹⁶ | no |

So **the cap alone is sufficient**, and the earlier "neither escape alone is enough" was measured
against the net with `id` in the head slot — the bottom row, which reproduces it. The reason is
mechanical: `capped`'s `2·mag` depends on the input WINDOW, not on the inherited error, so a
capped site resets the error however big it arrived, and the head LN sits after the last GELU.
The saturation constant stays in anyway (it is free, it is tighter everywhere, and ViT's GELU
sites want it) — but it is not load-bearing here, and a plan that had relied on it would have
been relying on layer order.

**⚠ Three things that cost time, for ViT.**
1. **A bound bundle read with `obtain` is a bridge that does not reduce.** `rcases` on a `∧`
   compiles to `And.casesOn`, which is stuck on a variable, so `Maps.residual … ≟ (that
   bridge).Maps …` fails to unify and no numeric envelope can be attached.
   `floatBridgesTo_cnxBlockChW` / `_cnxDownChW` are now term-mode with `hb.1` / `hb.2.1` /…
   projections. Anything a budget file will compose onto must reduce.
2. **The recursive stage fold associates to the RIGHT.**
   `floatBridgesTo_convNextStageChK` at `k = 3` is `b0.comp (b1.comp (b2.comp idVec))`, not
   `((b0.comp b1).comp b2)`. The `Maps` chain has to match, and the error points at the first
   block, not at the association.
3. **Pin `(Ā := …)` on any `Maps` leaf whose input window the elaborator has not yet unified.**
   The head-LN site's `by norm_num`s ran before `Maps.comp` had determined `Ā` and met a
   metavariable; the block/downsample sites were fine only because their other arguments pinned
   it. Passing the input window explicitly costs nothing and removes the order dependence.

### 3.5 ViT-Tiny — ⛔ DONE 2026-09-03: 3.612·10²¹⁸ / 7.222·10²¹⁸, and it is the CAP

`vit_float_logits_le` / `vit_float_logits_le_committed`: window **3.612·10²¹⁸**, budget
**7.222·10²¹⁸**, at the per-kind measured profile, `ε ≥ 10⁻⁵`, device LayerNorm statistics,
device GELU and device `exp` accurate to `10⁻²`, `u ≤ 2⁻²⁴`. Files: `ViTFloatBudget.lean`
(731 lines, ⭐ **30 s** — 162 numeric stages, 324 inequalities), plus the device kernels' move
into `FloatBudgetEnvLN.lean` and `DeviceExp` beside them.

**The probe said yes and the Lean followed: the whole ViT forward bridges at real weights**
(`floatBridgesTo_vitForwardKV` / `Maps.vitForwardKV`), with nothing supplied but the device
LayerNorm's statistics and the device `exp`/`gelu` accuracies — the standing `DeviceRsqrt` has on
the other four nets. `vit_chain`
(`scripts/float_budget_envelope.py`) folds all 162 stages of `vitForwardKV` at ViT-Tiny's shapes
and lands at

    window ≤ 3.612·10²¹⁸      budget ≤ 7.222·10²¹⁸      budget/window = 1.999
    324 re-assertions (`verify_vit`)

— under `norm_num`'s ~10³⁰⁰ ceiling with 80 orders to spare, and ⛔ **it is the CAP, not the
fold** (the 1.999 is the same tell ConvNeXt's 2.00 is; §9 applies verbatim).

⚠ **That number is 27 orders above the first probe's 1.065·10¹⁹¹, and the reason is a real
constraint, not a regression.** The first fold decomposed attention into four stages
(`Q·Kᵀ → scale → softmax → ·V`) the way the emitted graph spells it. **`FloatBridgesTo` cannot
express that**: it composes single-input maps, and attention FANS OUT (`X ↦ Q,K,V`) before it
rejoins. The repo's `mhProjAttnFullFlat` already bundles attention monolithically for exactly
that reason. So the softmax's window reset happens INSIDE one leaf, the output matmul takes the
generic `n = 197` fan-in bound, and the whole net is the probe's old `convex_av=False` row —
which is why that row was worth costing. ⭐ The lesson for the next architecture: **check that
the probe's granularity is expressible in the composition framework before trusting its
number.** A `Maps` chain is a line, and any op that fans out has to be one leaf.

**⭐⭐ The finding, and it is a new category: there are TWO kinds of unstatable.** ConvNeXt's
ablations only ever failed one way — the numeral got too big. ViT fails both ways, and the two
escapes are load-bearing for *different* reasons:

| variant | window | budget | unwritable stages | statable |
|---|---|---|---|---|
| **shipped (`Maps.mhProjAttnFullCap`)** | **3.612·10²¹⁸** | **7.222·10²¹⁸** | 0 | **yes** |
| uniform param bound | 1.105·10²³⁷ | 2.210·10²³⁷ | 0 | yes |
| + convex attn·V (lemma NOT proved) | 1.055·10¹⁹¹ | 2.109·10¹⁹¹ | 0 | yes |
| **`mhpB` window** (`\|real\| + \|float−real\|`) | 8.257·10¹⁹⁰ | 1.652·10¹⁹¹ | **36** | **NO** |
| cubic GELU (no saturation) | 3.612·10²¹⁸ | 7.222·10²¹⁸ | 0 | yes |
| **LN UNCAPPED** | 3.612·10²¹⁸ | **3.523·10³²³⁹** | 0 | **NO** |
| depth 2 (`vitForward2V`) | 3.145·10⁴² | 6.290·10⁴² | 0 | yes |
| depth 6 | 8.364·10¹¹² | 1.673·10¹¹³ | 0 | yes |

* **LN uncapped is a MAGNITUDE failure** — 10³²³⁹, the §0.1 quadratic, exactly ConvNeXt's story.
* **`mhpB` is a REPRESENTABILITY failure at a SMALLER magnitude.** `smErr`'s
  `Real.exp (2δ)` sits at δ ≈ 10¹⁶ by block 0, and `Real.exp` at that argument has no rational
  bound the kernel will check. The fold's *number* does not move — the next capped LN resets the
  error two stages later regardless — but 36 stage numerals (3 per block × 12) **cannot be
  written down at all**. ⛔ Quantified in chunk 1: `δ = attnScaledErr = 3.609·10¹⁰` at block 0,
  where `exp_sub_one_le` (which the repo already has) needs `x < 1`. There is no bound. ⚠ A Python fold hides this: `math.expm1` overflows to a finite float and
  the chain sails on. `vit_chain` therefore returns an `exp_tainted` tag list alongside the rows,
  and "statable" is `max(exponent) < 300 AND no taint`. Carry that instrument to any net with a
  transcendental leaf; magnitude alone is the wrong test.

**⭐ Three things the plan budgeted for are NOT load-bearing, and the Lean work can skip them.**
1. **The per-kind profile split.** §3.3(a)'s ConvNeXt lesson does not transfer: ViT-Tiny's kinds
   are 2.5× apart, not 14×, so the uniform bound is *statable* (10²⁰⁹). Splitting buys 18 vacuous
   orders. ⭐ The reason is structural — ConvNeXt's outlier was **layer scale**, which multiplies
   inside every block; ViT's outlier is the **final** LayerNorm γ (1.6645), which sits after
   everything and multiplies only the head. Measure the profile, but expect to spend the split on
   tidiness rather than on statability.
2. **The convex attention·V bound.** Using `sdpa_abs_le`'s convex-combination window (the softmax
   row is a probability vector, so the output is bounded by `vA` with no factor of `n`) buys 27
   orders — and the generic `n = 197` fan-in bound is statable without it. ⛔ So do **not** prove
   the float-side peer (rounded weights nonnegative and summing to `≤ 1+κ`); it is not currently
   proved, it is not needed, and §1 says the numbers are not the point.
3. **GELU's saturation constant.** Identical number with the cubic polynomial, for the same
   reason it was not load-bearing on ConvNeXt: the next capped LN resets the error anyway. Keep
   `Maps.gelu` on the `3/2` branch because the polynomial carries `√(2/π)` and cannot be
   `norm_num`'d — but that is a *rationality* reason, not a size one.

**⭐ Where the size actually comes from.** Per block the window multiplies by ~10¹⁵·³, and the
LayerNorm sites are the whole story: each contributes `2S = 634` (`|x−μ| ≤ 2A` over
`istd ≤ 1/√ε`), and there are 25 of them. The dense fan-ins (192·0.7, 768·0.8) are second order.
So §0.1 item 2 — an operating-point variance floor — is worth ~10⁶⁸ here and nothing else is,
the same conclusion §3.2 reached on MobileNetV2.

**⛔⛔ Do NOT build the number on `floatBridges_mhProjAttnFull` — ✅ and the replacement landed
2026-09-03 as `floatClose_mhProjAttnFullCap` / `Maps.mhProjAttnFullCap`.** §3.5's earlier note asked
whether attention is "a fourth quadratic site". It is not — **it is worse, and it is the reason
the decomposition matters.** Reading the two definitions (`ViTBlockFloatBridge.lean` :851/:858):

* `mhpL`'s second term is `attnOutInErr n dh … (layerBudget u (h·dh) w' β A e)`, and
  `attnOutInErr n d qA kA vA scaleA e = n·(e + vA·(Real.exp (2·scaleA·attnScoreInErr d qA kA e) − 1))`
  with `attnScoreInErr d qA kA e = d·((qA+e)·e + kA·e)` — **quadratic in the inherited error,
  inside an exponential.** Not a quadratic site; an exponential-of-quadratic one.
* Worse for the fold, `mhpB` — the **window** — also contains `Real.exp`, through
  `attnOutErr`'s `attnWeightErr = smErr u eexp (attnScaledErr …) n`. Its δ is rounding-only, so it
  *is* boundable (`Real.exp x ≤ 1/(1−x)` for `x < 1`, off `Real.add_one_le_exp`) — but that is a
  lemma to write, and capping the monolithic leaf does **not** avoid it, because `capped` bounds
  the modulus by `2·mag` and the `Real.exp` is in `mag`.

⭐⭐ **The fix is `floatClose_seScale`'s fix again, one net later (§3.4 finding 2).** `mhpB`
derives the float window as `|real| + |float − real|`; but `FloatClose`'s magnitude clause bounds
the FLOAT output directly, and the float output is a rounded dot of float softmax weights
(`≤ 1 + smCap`, exp-free) against float `V`. That gives `(1+γ_{n+1})·n·(1+smCap)·vA_F` —
**exp-free AND tighter than `mhpB`**. The identical sentence closes both findings: *the error
never needed to enter the window at all.* Twice now the blocker has been a window derived through
an error term when a direct bound on the float side was available; ⭐ **add that to §0.1's
checklist — when a window contains an error term, ask why.**

⚠ The softmax's window reset therefore happens INSIDE the attention leaf rather than as its own
chain stage. `Maps.softmaxRow` exists and is proved (window `1 + smCap`, constant in the input),
but ViT's chain does not compose it — it is there for the loss head, for a decomposed variant,
and because the leaf is the clearest statement of the trick. ⚠ That makes it an unexercised
`Maps` leaf in §5's sense, which is why `FloatBudgetEnvAttn.lean` closes it as its own compiled
`example`.

### 3.5.1 ⛔ The two questions §3.5 said to settle — both settled, and the second reverses

**1. WHICH NET: `vitForwardKV`, not `vit_full`. The plan's dichotomy is stale — the
generalisation already landed.** §3.5 said to choose between stating the number on the
weight-shared `vit_full` or "generalising the committed def to per-block params first (the
`CnxBlockParamsCh` treatment)". That work exists: `Architectures/ViTDepthK.lean` has
`BlockParamsV` (the 16-field per-block record), `vitBodyKVFlat` (the depth-`k` fold),
`vitForwardKV` (whole net, **vector-[D] LN, multi-head, distinct per-block params**),
`vitForwardKV_has_vjp` — and `vitFwdGraphKMHV_faithful` ties the depth-`k` graph to it.
`VerifiedNets.lean`'s ViT-S docstring already calls `vitForwardKV_has_vjp` the theorem that
covers Tiny *and* Small. So the number's target is `vitForwardKV` at
`ic=3, H=W=224, patchSize=16, N=196, mlpDim=768, heads=3, d_head=64, nClasses=1000, k=12`, and
⛔ `vit_full` / `vitForwardFlat` / `vit_floatBridgesTo` are the **wrong tier for this** —
`vit_full` shares one parameter tuple across all 12 blocks and carries SCALAR LN affines, and the
checkpoint has neither. ⚠ `vit_full_eq_vitForwardFlat` remains true and remains about a net we do
not train; §3.1's `BnEvalFloatBridge` warning, one tier up.

**2. THE CONE IS OPEN WIDER THAN §3.5 SAID — it is not the EfficientNet job, it is a TIER
MIGRATION. ⚠ AND THAT COSTING WAS HALF WRONG; see the correction at the end of this item.** §3.5 read the gap as "the patch embed, the attention block and the MLP block each
need a closed `FloatBridgesTo` at real weights". The actual state is that **the ViT float cone is
`FloatBridges` (the ∃-existential tier) essentially everywhere**: `ViTBlockFloatBridge.lean`
(1116 lines), `ViTAttentionFloatBridge.lean` (439) and `ViTFloatBridge.lean` (233) contain
**zero** `FloatBridgesTo` definitions between them apart from `FloatBridgesTo.perRow` and
`floatBridgesTo_gather`. Only `ViTWholeFloatBridge.lean`'s top-level skeleton was migrated — so
`vit_floatBridgesTo` exists and **not one of its hypotheses can currently be discharged**.
⚠ That is the `floatbridgesto_existential_L_vacuous` migration, unfinished on this net, and it is
the bulk of the remaining work. Budget for it explicitly; it is not visible from the whole-net
file, which looks finished.

⭐⭐ **CORRECTION (chunk 2, 2026-09-03): that count was right about the ATTENTION half and wrong
about everything else, and the whole migration was a fraction of the estimate.** The grep above
searched `ViT*FloatBridge.lean`, and the LayerNorm half of ViT's cone does not live there — it
lives in ConvNeXt's, written for the head LayerNorm:

* `floatBridgesTo_rowLNVecFlat` (`ChannelLNFloatBridge.lean`) and `Maps.rowLNVecFlat`
  (`FloatBudgetEnvLN.lean`) already exist, and ⭐ **`rowLNVecFlat N D ε γ β` IS ViT's per-token
  vector LayerNorm — `rfl`.** All 25 of ViT's LN sites are served by ConvNeXt's leaf unchanged.
* `floatBridgesTo_gelu` (`ViTFloatBridge.lean`) and `floatBridgesTo_dense` were already migrated
  too, as were `Maps.gelu` / `Maps.dense` / `Maps.perRow` / `Maps.residual`.

So chunk 2 wrote **no new leaves at all** — only the assembly and the ties. ⭐ This is
§3.3.0(b)'s lesson for the third time (*before writing a bound, grep the repo for it*), now
sharpened: **grep the whole cone, not the files named after your net.** A leaf written for one
architecture is named after that architecture and will not be found by searching for yours.

### 3.5.2 The order of work — ✅ ALL SIX STEPS LANDED 2026-09-03

1. ✅ **`FloatBridgesTo` peers for the attention leaves.** `floatBridgesTo_softmaxRow` (window
   `1 + smCap`, `smCap = u(1+smKappa) + smKappa`) and `floatBridgesTo_mhProjAttnFullCap` (window
   `mhpBCap`), plus the supporting `softmaxF_abs_le`, `dot_abs_le_of`, and the two numeral
   helpers every ViT stage needs — `smRho_le_of` (the side condition from a `gamma_num` bound)
   and `smCap_le` (a rational bound on `smKappa`'s quotient). ⭐ `floatClose_mhProjAttnFullCap`
   reuses `floatClose_mhProjAttnFull`'s ERROR clause verbatim — `FloatClose`'s error clause does
   not mention the window — so only the magnitude was reproved.
   ⛔ **Still open, and it is the bulk:** the LN / MLP / patch-embed peers. The ViT float cone is
   still `FloatBridges` everywhere else (§3.5.1).
2. ✅ **`Maps` leaves** — `Maps.softmaxRow` and `Maps.mhProjAttnFullCap`, both capped. ⛔ `matmul2`
   and a standalone softmax stage turned out NOT to be needed (the fan-out finding, §3.5); still
   to write are `Maps.concatCls` and `Maps.flatConvStride16`. ⭐ Reused unchanged from
   `FloatBudgetEnvLN.lean`: `bnCapped`, `diagBack`, `biasAdd`, `gelu`, `gather`, `perRow`,
   `idVec`. Block-0's attention site and the softmax leaf are closed as compiled `example`s at
   `vit_chain`'s numerals (§5's rule).
   ⚠ **`verify_vit` earned itself on its first run.** It rejected the attention stage because the
   chain rounded `mag` and `2·mag` up INDEPENDENTLY, and `2·r4(x)` can exceed `r4(2·x)` — which
   breaks `Maps.capped`'s own `2·Ā' ≤ Ē'`. Round the window first, then double the rounded value
   (`cnx_eval_chain`'s `lnsite` already did; the ViT leaf did not). Any capped leaf has this
   trap.
3. ✅ **The BLOCK landed 2026-09-03 (`ViTBlockVFloatBridge.lean`)** — `floatBridgesTo_blockVFlat`
   (closed at real weights, float net named as `blockVFlatF`), `Maps.blockVFlat` (13 stages, two
   skips, 26 inequalities), and `blockVFlat_eq`, the structural tie to the committed definition.
   ⛔ **And it is a DIFFERENT block from the one the float tier already had.**
   `floatBridges_vitBlockMHFull` is stated about `transformerBlock`'s SCALAR LayerNorm affines
   (`γ β : ℝ`); `vitForwardKV` composes `transformerBlockV`, whose affines are VECTORS — and the
   checkpoint has vectors. ⚠ `imagenet_specs_drift_from_twins` for the third time in this file
   (§3.3(b) was the second): the two statements look interchangeable until something forces them
   to unify, and what forces it is needing the tie for a number.
   ⚠ Of the three structural ties only `flat_mhsaLayer_eq` needed a proof — `mhProjAttnFullFlat`
   ends in `Mat.flatten` where `perRowFlat` opens with `Mat.unflatten`, so the roundtrip needs
   rewriting; the LN and MLP ties are `rfl`. And ⚠ `biPathMat (fun X => X) G` puts the IDENTITY
   first where `Proofs.residual` puts the BODY first — equal by `add_comm`, and it matters because
   `FloatBridgesTo.residual` names its float map body-first, so the other association is a
   different term.
   ⭐ ViT needs NO eval twin and none is possible (LN has no running statistics), so unlike
   r34/mnv2/B0 there is no second render and no `*RenderPCEval.lean` — the same saving ConvNeXt
   had.
4. ✅ **The PATCH EMBED landed 2026-09-03 (`FloatBudgetEnvAttn.lean`)** —
   `floatBridgesTo_patchEmbed`, `Maps.patchEmbed`, and a worked site at ViT-Tiny's shapes
   (window `232.3`, rounding `5.633·10⁻⁴`).
   ⛔ **And `Maps.concatCls` / `Maps.flatConvStride16` do not exist, because that decomposition
   is wrong.** `patchEmbed_flat` is a SINGLE definition with an `if n.val = 0` branch selecting
   the CLS token — not `concatCls ∘ convStride16`. ⭐ **That is the granularity trap for the
   THIRD time** (attention's fan-out was the first, the probe's own 3-stage patch embed the
   second), so promote it to a rule: **read the committed definition before planning a
   decomposition, and never infer one from the emitted graph** — the graph spells what the
   kernel does, the definition says what the theorem is about, and only the second constrains a
   `Maps` chain.
   ⭐ It needed no new mathematics: `floatClose_patchEmbed` already carried both clauses against a
   NAMED float peer (`FloatModel.patchEmbedF`), so the bridge is a repackage. What it needed was
   **monotonicity, in two directions**: in the input window (because `Maps` quantifies over every
   `A ≤ Ā`) and ⭐ in the ROUNDING UNIT — the budget mentions `M.u`, which no `norm_num` can
   evaluate, so `peRoundErrQ` restates it over a plain rational and `patchEmbedRoundErr_le`
   bridges `M.u ≤ q` once. **Carry that pattern to any leaf whose budget is not already stated
   through a `gamma_num`-style rational.**
   ⭐ Uniquely in this repo, no `gamma_num` detour is needed here at all: the reductions are
   `ic = 3` and `patchSize = 16`, so `norm_num` takes the exact `(1+u)⁴` and `(1+u)¹⁷` directly.
   ⛔ And it is the ONE stage of ViT's chain that is NOT capped — the patch embed does not reduce,
   its modulus is linear in the inherited error, and at the net's input that error is `0`. It is
   the only honest fold ViT has.
5. ✅ **The DEPTH-`k` FOLD AND THE WHOLE NET landed 2026-09-03** (`ViTBlockVFloatBridge.lean`):
   `floatBridgesTo_vitBodyKVFlat`, `Maps.vitBodyKVFlat`, `vitForwardKV_eq` (`rfl`),
   `floatBridgesTo_vitForwardKV` and `Maps.vitForwardKV`. ⛔ **The ViT cone is now CLOSED at real
   weights** — no `FloatBridgesTo` hypothesis is left except the device LayerNorm, whose
   statistics have no IEEE specification, and the device `exp`/`gelu` accuracies. The numerals
   followed in step 6.
   ⚠ **The recursion is HEAD-FIRST** — `vitBodyKVFlat (k+1) ps = body k (ps ∘ succ) ∘ blockVFlat
   (ps 0)` — so block 0 is applied FIRST and `.comp` puts it on the left, the OPPOSITE
   association from `floatBridgesTo_convNextStageChK`. §3.3's lesson 2 said to check; the answer
   differs per net, and it is the DEFINITION that decides, never the analogy.
   ⭐⭐ **`FloatBridgesTo.ofEq` is the reusable piece.** The block bridge is stated on
   `blockVFlat_eq`'s right-hand side and the fold needs it on the committed `blockVFlat`;
   `blockVFlat_eq ▸ b` does NOT work, because the `Eq.mpr` blocks `.mag` from reducing and a
   bridge whose `.mag` does not reduce cannot carry a `Maps` (§2's unifier trap, in a new
   disguise). Rebuilding the structure field-by-field keeps `mag`/`mod` definitionally the
   originals, which is why `Maps.ofEq` is `⟨hM.mag_le, hM.mod_le⟩` and not a re-proof. **Any
   transport of a bridge along a tie needs this shape.**
   ⭐ `Maps.vitBodyKVFlat` is an ENVELOPE fold, which ConvNeXt does not have — its budget file
   spells every stage out. At depth 12 that would be twelve nested `.comp`s written by hand;
   here the caller passes window/error SEQUENCES and one `Maps` per block.
6. ✅ **`ViTFloatBudget.lean` LANDED — and it compiled on the first try, in 30 s.** What
   actually happened, against the plan above:
   1. ✅ **The device kernels moved** — `DeviceLN` and `DeviceGelu` out of
      `ConvNeXtFloatBudget.lean` into `FloatBudgetEnvLN.lean`, with a `DeviceExp` beside them
      (⚠ RELATIVE spec, as predicted, because that is the shape `softmaxF_close` needs). ⭐ The
      move is cleaner than a copy: the capped LN site's bridge and envelope generalised to
      `DeviceLN.bridgeAt` / `DeviceLN.mapsAt` (taking `0 ≤ emr`, `0 < ε`, `1/√ε ≤ S`, `0 ≤ S`,
      `M.u ≤ q` directly instead of a net's profile record), and ConvNeXt's `DeviceLN.bridge` /
      `.maps` are now one-line delegations that read those five off `CnxProfile`. ⚠ ConvNeXt's
      number re-checked unchanged (128 s) — do that check, it is the whole reason this is its
      own commit.
   2. ✅ **`ViTProfile` / `ViTBounded`** on the `CnxProfile` / `CnxBounded` pattern — eight
      magnitude bounds, `ε > 0` with `1/√ε ≤ S = 317`, `M.u ≤ q = u32`. ⭐ **`ViTBounded` is
      stated over the COMMITTED `ViTTinyWeights`** (`Proofs/Foundation/SpecVJP.lean`), not over
      a fresh record, which is what makes the tie free.
   3. ✅ **The chain is four steps**, exactly as promised: `mPatch` (the worked
      `FloatBudgetEnvAttn` example, transplanted), `mBody` (`Maps.vitBodyKVFlat` over twelve
      `Maps.blockVFlatC`), `mLN` (`Maps.rowLNVecFlat` at the head) and `mHead` (`Maps.vitHead`).
      ⭐⭐ **The envelope fold is why this file is a third of ConvNeXt's elaboration time at the
      same order of inequalities** — 30 s against ~2 min. The k = 2 fallback was not needed.
   4. ✅ **Every numeral came from `vit_chain`**, emitted by a generator that runs `verify_vit`
      (324 inequalities) *before* it writes a line of Lean.
   5. ✅ **The tie is stronger than ConvNeXt's.** `vit_float_logits_le_committed` is stated
      against `denoteVitTiny vitVerified.layers` — the committed SPEC's denotation — via
      `vitVerified_denote_eq` (`rfl`); `vitVerified_fwd_faithful` already says the emitted
      depth-12 multi-head vector-LN graph denotes the same function. No `*RenderPCEval` twin
      exists or is possible.
   ⚠ **Two shape notes for the next transformer.** (a) The whole-net envelope cannot be built
   with `have m := … ?_` — a `have` will not carry a synthetic hole — so the proof is
   `have mPatch/mLN/mHead`, then `refine Maps.vitForwardKV … mPatch ?_ mLN mHead`, then
   `refine Maps.vitBodyKVFlat … ?_`, then `intro i; fin_cases i`. The head and final-LN
   envelopes do not depend on the body, so they can be built first and the body left as the
   single hole. (b) ⚠ The committed spec's head is 10-way (imagenette) while the checkpoint the
   profile is measured on is 1000-way; `nClasses` enters no numeral, because the head dense's
   fan-in is `D = 192`. ConvNeXt-T has the same split and does not say so.

7. ✅ **`verify_vit`** — the re-assertion pass, landed with the probe (324 inequalities). Run it
   before any numeral is emitted (§0's ⚠: fold with the ROUNDED γ; the pass is what catches it).
   ⚠ It has already earned itself once: it rejected the attention stage because the chain rounded
   `mag` and `2·mag` up INDEPENDENTLY, and `2·r4(x)` can exceed `r4(2·x)`, breaking
   `Maps.capped`'s own `2·Ā' ≤ Ē'`. Any capped leaf has that trap.

⚠ Two facts the whole-net statement must carry and disclose, like `DeviceRsqrt`/`DeviceSigmoid`:
the device `exp` accuracy `eexp` (taken at 10⁻², as `es`/`esig`/`egelu` are), and the softmax side
condition **`smRho u eexp n < 1`** — at `n = 197` and `eexp = 10⁻²` it is `0.010012 < 1`, with
room, but it is a hypothesis and not a footnote.

⚠ And carry §3.3's three mechanical lessons: bound bundles by projection, not `obtain`
(`And.casesOn` is stuck on a variable and the bridge will not reduce); check which way the
recursion associates before writing the chain (`vitBodyKVFlat` recurses HEAD-first — block 0
applied first — where `floatBridgesTo_convNextStageChK` associates to the right); and `(Ā := …)`
pinned on every leaf whose input window the elaborator has not yet unified.

### 3.6 Common sub-steps for every net

* Parameter records with bounds (pattern: `R34Conv`/`R34Bn`/`R34IdBlk`), and one `R34Profile`.
* Block `Maps` lemma so the whole-net chain stays at block granularity.
* Extend `scripts/float_budget_envelope.py` with the net's stage list AND its `verify_<net>`
  re-assertion pass. Never hand-type a numeral.
* Cross-check the number against the probe BEFORE writing the docstring.

### 3.7 Phase 2 — the backwards. ⭐⭐ ResNet-34's INPUT-GRADIENT NUMBER LANDED 2026-09-03

✅ **`r34_grad_float_le` (`Resnet34BackFloatBudget.lean`): the ResNet-34 whole-net
input-gradient VJP has a kernel-checked number, and it is a FOLD, at TRAINING-mode BatchNorm** —
the mode this same net's forward has no number for at all (§0.1's 10⁷⁴¹⁷). `r34_back_chain` /
`verify_r34_back` (`scripts/float_budget_envelope.py`, 91 stages, 254 re-assertions):

    window ≤ 8.857·10²⁴⁵      budget ≤ 6.894·10²⁴⁴      budget/window = 0.078

⚠ at the operating point `|istd| ≤ 16`; at the unconditional `ε`-floor (`|istd| ≤ 317`) the same
fold is 5.503·10²⁸⁸ and is past `norm_num`'s real ceiling — see step 4(a)/(b) below.
⚠ **These numbers are 4× the ones first committed (2.188·10²⁴⁵ / 1.458·10²⁴⁴), and the reason is
§3.10: the chain was missing the 3×3/s2 stem pool's backward entirely.**

⛔ **The old text here said "§0.1 applies with more force: a backward's BN-back modulus inherits
the same reduction structure." It does not.** §0.1's quadratic came from the statistics MOVING
with the input; a VJP reads its statistics off the SAVED ACTIVATIONS, which the cotangent does
not perturb. So `floatClose_bnBack`'s modulus is `bnGradInputBudget(A) + bnGradInputReMag(e)`
with `bnGradInputReMag n G e S Xh = S·G·e·(2 + Xh²)` — **linear in `e`**, because a VJP at a
fixed point is a linear map and this is its Lipschitz constant. No cap, no squaring, ratio 0.048.

⛔⛔ **BUT THE WALL DOES NOT VANISH — IT RELOCATES, and that is the honest headline.** The saved
float activations enter the BN backward as two SUPPLIED accuracies: `es` on the inverse-stddev
and `exh` on the normalised activation `x̂`, both taken at `10⁻²` like every other device-kernel
accuracy in this file. ⚠ Unlike `DeviceRsqrt`'s, those are quantities the repo's forward fold
DOES speak about — and at training-mode BatchNorm it says `10⁷⁴¹⁷`, not `10⁻²`. So the backward
number is an honest fold *given a forward-accuracy hypothesis its own forward cannot discharge*.
Say it that way (§9). ⭐ The interesting consequence: §0.1's wall is not a fact about backwards,
it is a fact about *composing* a backward with the forward that feeds it.

⭐⭐ **The load-bearing lemma is `bnXhat_sq_le` — `|x̂| ≤ √n` — AND IT WAS ALREADY IN THE REPO.**
It sits in `Foundation/ResNet34.lean` (and again in `Training/MobileNetV2SealRealistic.lean`),
proved for the *realistic-seal* work, named after neither the float tier nor the backward. The
ablation is decisive:

| variant | window | budget | statable |
|---|---|---|---|
| **shipped leaves at the ε-floor `S = 317`** (per-kind profile, `x̂ ≤ √n`) | **5.503·10²⁸⁸** | **3.276·10²⁸⁷** | **yes** |
| the committed statement, at `\|istd\| ≤ 16` | 8.857·10²⁴⁵ | 6.894·10²⁴⁴ | yes |
| uniform `21/10` (the forward's profile) | 1.008·10²⁹⁷ | 5.997·10²⁹⁵ | yes |
| **`x̂` from the FORWARD window (`2·A·S`)** | 2.798·10⁷²⁷² | 5.480·10⁷²⁷⁰ | **NO** |
| variance floor `10⁻³` (`S = 32`) | 7.547·10²⁵⁵ | 5.130·10²⁵⁴ | yes |
| variance floor `10⁻¹` (`S = 4`) | 1.278·10²²⁶ | 1.702·10²²⁵ | yes |
| `σ² ≈ 1` (`S = 1`) | 2.211·10²⁰⁶ | 7.126·10²⁰⁵ | yes |

That is §3.3.0(b)'s lesson for the FOURTH time — *before writing a bound, grep the whole repo for
it, not the files named after your net.* Every place the activations enter the input-gradient
they enter **normalised** (`x̂`) or through `istd` (≤ `1/√ε`), so the backward's fold **does not
inherit the forward's window at all**. `Xh` enters as `Xh²`, and `Xh² = h·w` at that site.

⭐ **The per-kind profile split is worth 8 orders, and here it buys HEADROOM, not statability.**
Measured on `/home/skoonce/resnet/r34_imagenet_bf16_e79.bin` (21,797,672 f32): conv and dense
kernels max at **1.1007** over 21.78 M entries, BN γ at 2.0741, BN β at 0.8118, the dense bias at
0.0475. ⚠ The forward's uniform `21/10` is a BN **γ**, and it is 1.9× loose on exactly the
entries every conv fan-in multiplies. Uniform gives `10²⁹⁶` — four orders under `norm_num`'s
ceiling, which is too close to be comfortable; split gives `10²⁸⁸` and twelve. ⚠ Note the
backward has **no bias anywhere** (`convFlatBack` and `linBack` both carry bias `0`), so β and
the dense bias do not enter a single numeral.

**Where the Lean work is.** The cone is in the same state ViT's was before its four chunks: the
whole-net skeleton exists at the `FloatBridgesTo` tier (`r34_grad_floatBridgesTo`,
`Resnet34WholeBackFloatBridge.lean`) but **its 16 block backwards are hypotheses**, and the
blocks themselves are only at the `∃`-tier. In order:

1. ✅ **`Maps` leaves for the backward — LANDED 2026-09-03 (`FloatBudgetEnvBack.lean`).**
   `Maps.convBack` / `.linBack` / `.gapBack` reuse the forward's arithmetic (`Maps.flatConv` /
   `Maps.dense` at bias 0 — `linBack` IS `floatBridgesTo_dense` at the transpose, and
   `convFlatBack W` IS `flatConv (reverseSwap W) 0` with the channel roles swapped, so the
   fan-in is the COTANGENT's channel count). `Maps.reluMaskBack` / `.maxPoolBack` /
   `.decimateBack` are envelope-preserving; `Maps.flatConvStride2Back` is the last two composed.
   ⭐ **The real one was `bnGradInputBudget`'s numerals**, and the fix is `peRoundErrQ`'s pattern
   one tier up: `bnGradInputBudgetG` takes the rounding unit AND the reduction's
   `(1+u)^(n+1) − 1` as PARAMETERS, `bnGradInputBudget_eq_G` is `rfl`, and
   `bnGradInputBudget_le` replaces both by rationals once. ⚠ Its monotonicity is the one real
   proof in the file, and the shape that made it tractable is **naming every intermediate**: the
   budget is a fifteen-step `let` chain, and respelling it as thirteen small defs (`bgED`,
   `bgESD`, … `bgEP`) turns each monotonicity step into a three-line lemma instead of a
   hundred-line literal. A single `nlinarith` over the zeta-reduced chain does not close.
2. ✅ **`FloatBridgesTo` + `Maps` for the per-channel BN backward — LANDED with it.**
   `floatBridgesTo_bnBack` / `_bnPerChannelFlatBack` / `_bnPerChannelBack` mirror the forward's
   three rungs (`FloatClose.perRowIdx` then the `reassoc` gather conjugation), and
   ⭐ `bnXhat_abs_le_num` turns `bnXhat_sq_le`'s `x̂² ≤ n` into a RATIONAL `Xh` wherever
   `n ≤ X²` — exact for the square feature maps a conv net has (`h·w = 7² ⇒ X = 7`). ⚠ `√n`
   itself is irrational and `norm_num` cannot see it; that step is not optional.
   ⭐ The file closes the HEAD of r34's chain as a compiled `example` at `r34_back_chain`'s
   numerals — cotangent `(1, 0)` → classifier input-gradient → GAP backward → block `e1`'s
   second BN backward, out `(8348, 2.317·10⁻²)`. ⭐ Note the SHAPE that example makes visible:
   the head's fan-in is `10` and the GAP backward divides by `49`, so a backward chain's first
   two stages SHRINK and the growth only starts at the first normalisation.
3. ✅ **The two block backwards at real weights — LANDED 2026-09-03**, in the same file.
   `floatBridgesTo_r34IdBlockBack` / `_r34DownBlockBack` (float peers NAMED), and
   `Maps.r34IdBlockBack` / `.r34DownBlockBack`. ⭐ **Neither needs a new combinator, and the
   reason is worth stating**: the identity block's residual-skip backward is itself a *forward*
   `Proofs.residual` — the skip routes the cotangent to both branches and adds — and the
   downsample's two-branch fan-in is `biPathSum`. The forward's fan-IN is the backward's
   fan-OUT, and the same two combinators serve both directions.
   ⭐ Block `e1` (identity, `512 × 7 × 7`) is closed as a compiled `example` at
   `r34_back_chain`'s numerals with BOTH BatchNorm sites the real
   `floatBridgesTo_bnPerChannelBack`: in `(2.452·10⁻⁴, 1.898·10⁻¹⁰)`, out
   `(8.702·10¹², 5.299·10¹⁰)`. One block costs ~10¹⁶ of cotangent window; sixteen put the net at
   10²⁸⁸.
   ⚠ Two mechanical traps, both new: `Resnet34BackFloatBridge.lean` is NOT on
   `Resnet34WholeBackFloatBridge`'s import path (the whole-net file takes its blocks abstractly,
   so it never needed them), and `x.residual M` inside a `.comp` chain parses as
   `(A.comp x.residual) M` — write `FloatBridgesTo.residual M x` explicitly.
4. ✅ **`Resnet34BackFloatBudget.lean` — LANDED 2026-09-03.** `r34_grad_float_le`: the deployed
   ResNet-34 input-gradient is within **6.894·10²⁴⁴** of the certified one, per input pixel, on
   loss cotangents of magnitude `≤ 1` (`|p − y| ≤ 1` for softmax cross-entropy), certified window
   **8.857·10²⁴⁵**. ⭐ `budget / window = 0.078` — **the interval FOLD**, where ConvNeXt-T's and
   ViT-Tiny's forward numbers are `2.00` caps — **and at TRAINING-mode BatchNorm, the mode this
   same net's FORWARD has no statable number for at all** (§0.1's `10⁷⁴¹⁷`). 1026 lines, 90
   stages, 254 inequalities, ~80 s / 3.8 GB.

   ⛔ **Three hypotheses carry the caveat**, and they must be quoted with the number: `es` and
   `exh` (the saved float activations' inverse-stddev and normalised-activation accuracies, at
   `10⁻²` — quantities the forward's own training-mode fold cannot discharge), and the
   **operating point** `|istd| ≤ 16` i.e. `σ ≥ 1/16`, which is §0.1's escape 2 finally used.

   **Four things this cost, and three of them are corrections to earlier guesses in this file.**

   **(a) ⛔⛔ `norm_num`'s ceiling is ~10²⁵³ and SHAPE-DEPENDENT, not the `10³⁰⁰` §0.1 records.**

   | goal | closes |
   |---|---|
   | `21/10 * (4271 * 10²⁵⁰) ≤ 8971 * 10²⁵⁰` | ✅ |
   | `21/10 * (4271 * 10²⁶⁰) ≤ 8971 * 10²⁶⁰` | ⛔ |
   | `317/12544 * (12544 * (21/10 * (4271 * 10²⁵⁰))) ≤ 3571 * 10²⁵⁷` | ⛔ **same VALUE as row 1** |
   | `4271 * 10⁴⁰⁰ ≤ 4272 * 10⁴⁰⁰` | ✅ ⚠ **not evidence** — `a*c ≤ b*c` cancels via simp |

   Raising `maxHeartbeats`/`maxRecDepth` does not move it, and neither `ring_nf`, `nlinarith`,
   nor `simp only` + `norm_num` closes what `norm_num` alone will not. ⚠ ConvNeXt-T's `10²²⁷` and
   ViT-Tiny's `10²¹⁸` sit under it, which is why five nets landed without meeting it.

   **(b) ⭐ The operating-point `S` is what buys the room.** `R34BnBack.hS` — `|istd| ≤ S` at the
   saved activations — is a HYPOTHESIS, not a consequence of the `ε`-floor, so `S` is free to
   choose; it multiplies at all 33 BatchNorm sites, so `317 → 16` is worth ~43 orders:
   `5.503·10²⁸⁸ → 8.857·10²⁴⁵`. ⚠ `hSε` then has no business in the profile and was dropped — at
   `S = 16` it is false at `ε = 10⁻⁵`, and nothing in the backward uses it.

   **(c) ⚠ PIN ALL FOUR NUMERALS ON EVERY LEAF** — `(Ā := …) (Ē := …) (Ā' := …) (Ē' := …)`.
   §3.3's lesson 3 says to pin the input window; that is not enough. Every `by norm_num` inside a
   `have` runs before `Maps.comp` unifies anything, so an unpinned OUTPUT window is a
   metavariable too — and it fails as `⊢ <huge numeral> ≤ ?m.2125`, which reads like arithmetic
   and is not.

   **(d) ⭐⭐ THE BLOCKER WAS ONE IMPLICIT ARGUMENT, and every structural theory about it was
   wrong.** `floatBridgesTo_maxPoolBack` is declared over `x : Tensor3 c (2*h) (2*w)`, so putting
   it in a composition asks the elaborator to solve **`2 * ?h = 112`** — higher-order, and it
   cannot. What that looks like from outside is a whole-net `isDefEq` that runs 25 minutes to
   41 GB without terminating, which is indistinguishable from a dozen other causes. ⛔ In
   sequence I blamed, and each cost a 2–25 minute compile: the number of blocks; the `∘`
   association (flat vs `r34InputGrad`'s grouped stem); `r34_grad_floatBridgesTo` being a
   tactic-proof term that must be `whnf`'d through; and the downsample dimension indices
   (`ic * (2*h) * (2*w)` against `ic * h' * w'`). **All four were wrong.** The fix is
   `(c := 64) (h := 56) (w := 56)` — and with it a term-mode 22-stage chain typed at the
   committed maps elaborates in 80 s.

   ⭐ **The method that found it, and the one to use next time: probe compositions of GROWING
   DEPTH with explicit type ascriptions, from both ends.** 2 blocks including a downsample: 2.3 s.
   Depths 4/8/12/16: 2.6 s for all four. Head alone, stem group alone: instant. `lin + gap + all
   16 blocks` (18 stages): fine. **+ maxPool (19 stages): fails.** That is a five-minute bisect
   and it lands on the exact argument. ⚠ It only works because each probe carries a type
   ascription — see the next lesson.

   ⚠ **And time a whole-net `Maps` chain SEPARATELY from its closing step.** Every truncation
   timed here ended in `sorry`, which never performs the final unification, so a blowup in one
   `isDefEq` looked like a per-block cost for three iterations of the wrong fix. The chain alone
   is 84 s; the closing step was everything.

   ⭐⭐ **THE HOMOGENEITY IS THE REUSABLE PIECE: `bnGradInputBudgetG` is HOMOGENEOUS OF DEGREE 1
   in the cotangent window.** Every term carries exactly one factor of `Cdy` (`bgEP` is the only
   `Cdy`-free piece, and it enters as a `mulErr` *ea*, multiplied by a degree-1 *C*). So
   `budget(Cdy) = Cdy · budget(1)`, and a site's envelope is `A ↦ A·(Kr+Kb)`,
   `(A,E) ↦ A·Kb + E·Kr` — `Maps.bnPerChannelBackGain`. Stated directly, each site is a forty-node
   tree at the chain's magnitude; factored, the expensive evaluation happens once per feature-map
   SIZE — **five constants instead of sixty-eight.** Measured: 8 blocks 48 s → 31 s, 11 blocks
   >120 s → 55 s, the full chain 39 min/41 GB (not finishing) → 84 s/3.8 GB. ⭐ Ask the same of
   any future backward: a VJP is linear at a fixed point, so its budget is homogeneous, so its
   numerals factor. ⚠ The FORWARD budgets are NOT homogeneous — a bias breaks it — which is
   exactly why five forward nets never needed this.

### 3.8 What is open — ⭐ start with the probe, not the Lean

Three things, in the order I would do them. ⚠ Every one of them has a cheap first move; §3.7's
history is four wrong structural guesses at 2–25 minutes of compile each, and the thing that
finally worked was a five-minute bisect.

**1. ✅ PROBE THE MOBILENETV2 AND EFFICIENTNET-B0 BACKWARDS — DONE 2026-09-03. See §3.9.**
Both folds now live in `scripts/float_budget_envelope.py` (`mnv2_back_chain` / `verify_mnv2_back`,
136 inequalities; `b0_back_chain` / `verify_b0_back`, 138) with their ablations. The question is
answered — **yes, a backward is always a fold; SE does not break it** — and it turned up a
different blocker, which §3.9 records. What is open moved down to §3.9's "what to do next".

**2. ✅ Close the whole-net CERTIFIED-TIE fold for r34's backward — DONE 2026-09-03. See §3.10.**
`r34InputGrad_eq_resnet34_vjp` assembles the per-op ties into
`r34InputGrad = (resnet34_has_vjp_at …).backward` at the full `3×224²` dims. ⛔ **The blocker this
item named was a misreading** — `resnet34_has_vjp_at` is dimension-generic and parametric in its
component maps, so no new apex was needed. What the tie actually cost was a leaf nobody knew was
missing, and it moved the number; §3.10.

**3. ✅ Coherence: migrate `Cifar8FloatBudget.lean` off `FloatBridgesTo.Env` onto `Maps` — DONE
2026-09-04. See §3.11.** It was as cheap as advertised — same numerals, compiled first try — and
it found one thing worth keeping: the retired kit had been hiding a duplicate declaration.

⚠ **And one thing to flag rather than schedule.** §0.1's escape 2 — *"a genuinely linear bound
needs the NORMALISED output's Lipschitz constant, not the pre-normalisation one; that is real
work and probably the interesting result"* — is still open for the FORWARDS. The backward's
operating-point `S` is now a worked precedent for what such a bound is worth: ~43 orders across
33 BatchNorm sites, and the difference between a number that exists and one that does not
(§3.7 step 4(b)).

### 3.9 ⭐⭐ The MobileNetV2 and EfficientNet-B0 BACKWARD probes (2026-09-03)

`mnv2_back_chain` / `verify_mnv2_back` (48 stages, 136 re-assertions) and `b0_back_chain` /
`verify_b0_back` (59 stages, 138) fold `mnv2InputGrad` (`MobileNetV2BackFloatBridge.lean`) and
`efficientnetInputGradB` (`EfficientNetWholeBackFloatBridge.lean`) in the leaves' own semantics,
over the loss cotangent, at training-mode BatchNorm. No Lean was written.

| net | window | budget | budget/window | statable |
|---|---|---|---|---|
| ResNet-34 back (`\|istd\| ≤ 16`) | 8.857·10²⁴⁵ | 6.894·10²⁴⁴ | 0.078 | yes |
| **MobileNetV2 back** (`\|istd\| ≤ 16`) | **5.508·10¹²⁷** | **1.882·10¹²⁶** | 0.034 | **yes** |
| MobileNetV2 back, **ε-floor** `S = 317` | 4.750·10¹⁵³ | 1.076·10¹⁵² | 0.023 | **yes** |
| **EfficientNet-B0 back**, leaves as they stand | 2.491·10⁴³¹ | 4.624·10⁴³⁰ | 0.186 | ⛔ **NO** |
| EfficientNet-B0 back, global `\|swish′\|` | 2.021·10¹⁶⁷ | 5.225·10¹⁶⁶ | 0.259 | yes |

**⭐⭐ Finding 1 — the question §3.8 asked: YES, a backward is always a fold, SE included.**
The ratios are 0.034 / 0.186, not 2.00, and no `capped` appears anywhere. §0.1 now says why.

**⭐⭐ Finding 2 — MobileNetV2's backward needs NO operating-point hypothesis.** At the
unconditional ε-floor `S = 1/√ε ≤ 317` it is 4.750·10¹⁵³ — comfortably under §3.7(a)'s ~10²⁵³
shape-dependent ceiling — where ResNet-34's needs `|istd| ≤ 16` to come down from 10²⁸⁸ (§3.7
step 4(b)). Two reasons, both structural: 20 BatchNorm sites against r34's 33, and the
inverted-residual's backward fan-ins are 1×1 (24…256) and depthwise (9) where r34's are 512·9.
**So the first backward this repo could state with no hypothesis beyond the device-kernel
accuracies is MobileNetV2's, not ResNet-34's** — worth knowing before picking the next net.

**⛔ Finding 3 — B0's backward is a fold and is NOT statable, and ONE UNPROVED SCALAR LEMMA is
the entire difference.** The ablation, at the measured profile:

| variant | window | budget | statable |
|---|---|---|---|
| leaves as they stand | 10⁴³¹ | 10⁴³⁰ | no |
| **global `\|swish′\| ≤ 11/10` ONLY** | **10¹⁶⁷** | **10¹⁶⁶** | **yes** |
| operating-point `\|x\| ≤ 16` on the SE input ONLY | 10³⁵⁹ | 10³⁵⁸ | no |
| both | 10⁹⁵ | 10⁹⁵ | yes |
| both + `broadcastBack` tightened to its `h·w` nonzeros | 10⁸⁹ | 10⁸⁹ | yes |
| both, `x̂` from the forward WINDOW | 10⁶⁴⁸ | 10⁶⁴⁷ | no |

B0's backward scales the cotangent by four saved vectors r34's does not: the SE gate `g`
(bounded, `≤ 1`), `σ′(saved)` (bounded, `≤ 1/4` — `sigmoidScalar_lipschitz` already proves it),
the SE's saved input `x`, and `swish′(saved)`. The last two have no bound but the forward's
certified window, and `swishScalar_lipschitz_abs`'s `1 + A/4` is 1.216·10⁵¹ at the head alone —
**one stage, 51 orders**, visible in the chain as `head.swB` going from 10⁻² to 10⁴⁹.

**⭐⭐ And the lemma is elementary, against what §3.4 says. ✅ PROVED 2026-09-04, §3.12.** §3.4 records *"the true global
Lipschitz constant of `x·σ(x)` is ≈ 1.1; getting THAT needs the decay of `σ′`, i.e. calculus"* —
true of the sharp constant and irrelevant, because **the sharpness does not matter**: `Ssw =
11/10` gives 10⁹⁵, `137/100` gives 10⁹⁶, and the crudest global `2` gives 10⁹⁸. What is needed is
only that *the window does not appear*. And `≤ 2` is three steps: `σ′(x) = σ(x)(1−σ(x)) =
e^{−|x|}/(1+e^{−|x|})² ≤ e^{−|x|}`, then `|x|·e^{−|x|} ≤ 1` straight from `Real.add_one_le_exp`,
then `|swish′| = |σ + x·σ′| ≤ 1 + 1 = 2`. No MVT, no derivative analysis, no sup.
⭐ **This is the file's ⭐ habit for the sixth time — when a fold overshoots, ablate before
concluding anything about the architecture — with one new twist.** The previous five blockers
(relu6's clamp, swish's modulus, seScale's window, ConvNeXt's profile, attention's window) were
all bounds *already true and thrown away*, and §3.3.0(b)'s rule for them is "grep the repo before
writing one". This one is not in the repo, and the rule it needs is different: **a cost estimate
written down next to a bound is not evidence, and it is exactly what stops anyone re-checking.**
§3.4's parenthesis has been sitting there since the B0 forward landed.

**⚠ Finding 4 — `bnXhat_sq_le` is load-bearing on both nets**, as it was on r34: without it
MobileNetV2 is 10⁴²¹ and B0 10⁶⁴⁸. Three nets, three times decisive.

**⚠ Finding 5 — the per-kind profile split runs the OPPOSITE way on MobileNetV2.** Measured on
`/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet.bin` (3,504,872 f32): conv/dense kernels max
**2.7157** over 3.47 M entries, BN γ 1.6869, BN β 1.6406, dense bias 0.1029. On ResNet-34 the
uniform bound was a BN γ and the kernels were 1.9× tighter; here the maximum IS a kernel, so the
split buys the BN gain and not the conv fan-in — 4 orders, not 8. On B0
(`/home/skoonce/enet_b0_350_4gpu/…`, 5,288,548 f32) it runs r34's way — kernels 3.6857, BN γ
4.0545 — but at 1.1× it is worth under an order. ⭐ **Measure it; do not assume which kind is the
outlier.** `scripts/param_kind_profile.py` does it from the net's own generated
`init_params_from_file` (parsed, so it cannot drift from the loader that reads the checkpoint);
it reproduces §3.7's committed ResNet-34 table exactly, which is the check that it reads the
layout right.

**⭐ Finding 6 — relu6's clamp buys the backward NOTHING.** §3.2's headline is that `min A 6`
resets MobileNetV2's forward window at 13 sites (10¹⁰⁰ → 2154). Its backward is `reluMaskBack`, a
0/1 select: exact in float, envelope-*preserving*, and there is nothing for a cotangent window to
be clamped to. **Window and budget are separate levers on a forward; on a backward the clamp is
not a lever at all.**

**⚠ Finding 7 — a loose leaf, found and priced, not load-bearing.** `floatClose_broadcastBack`
(`SEBackFloatBridge.lean`) bounds the SE gate's spatial reduce by `(c·h·w)·A` — its `hsumabs`
step bounds *every* masked entry by `A`, including the `(c−1)·h·w` that are identically zero. The
honest count is `h·w`, so the window carries a spurious factor of `c` at each SE site. Worth 6
orders on B0 (10⁹⁵ → 10⁸⁹) and cheap to fix, but it is not what blocks anything.

**What to do next, in order.**
1. ✅ **`swishScalarDeriv_abs_le : |swishScalarDeriv x| ≤ 2` — DONE 2026-09-04
   (`Architectures/SwishSaturation.lean`, §3.12).** B0's backward is 7.640·10¹⁶⁹ / 1.735·10¹⁶⁹
   with it, from 10⁴³¹ — statable, and the fold's last window import is gone.
2. ✅ **MobileNetV2's backward budget file — DONE 2026-09-04 (§3.13).** 4.750·10¹⁵³ /
   1.076·10¹⁵² at the ε-floor, with no operating-point hypothesis, exactly as finding 2 said. The
   costing above was right to the leaf: two new `Maps`, both compositions of existing ones.
3. **B0's backward** then needs `Maps.diagBack`, `Maps.broadcastBack` and the `Maps.seBack`
   composite (`biPathSum` of two `.comp` chains — no new combinator, the same point §3.7 step 3
   made about the r34 blocks), plus four supplied saved-vector accuracies where r34 has two.
   ⚠ `batchMap` never enters a numeral, so the number holds at any `N`, like the forward's.

### 3.10 ⭐⭐ The whole-net CERTIFIED TIE for r34's backward (2026-09-03) — and the drift it found

`r34InputGrad_eq_resnet34_vjp` (`Foundation/Resnet34BackCertifiedTie.lean`, ~57 s):

    r34InputGrad <every slot pinned to the certified per-op backward>
      = (resnet34_has_vjp_at <the committed components at 3×224²> …).backward

So §1's success criterion (ii) is now met for the backward in the strongest available form: the
reading is no longer *"every piece of this chain is the certified gradient"* but **"the chain IS
the certified whole-net gradient"**. ⚠ It stays a SMOOTH-POINT statement — the pool's
`MaxPool3s2Smooth`, the stem's post-BN no-zero and the four per-stage `ChainData` bundles are
hypotheses, as every `HasVJPAt` in this cone is.

**⛔ §3.8 item 2's stated blocker was wrong, and that is worth recording.** It said
*"`resnet34_has_vjp_at` is parametric and only concretely instantiated at toy `resnet34Concrete`
dims, so the whole-net certified term does not exist at full dims yet … real work in
`Foundation/`."* That theorem is **dimension-generic and parametric in its component maps** — the
toy instantiation is one *use*, not a limit — so instantiating it at ImageNet dims needed no new
apex at all. One component witness was genuinely missing, the stem's: `cbrStridedPC_has_vjp_at`,
which is `convBnReluPC_has_vjp_at` with `flatConvStride2` in place of `flatConv`, twelve lines.
⭐ **A blocker written down once and not re-read is the same failure as a cost estimate written
down once and not re-derived** (§3.9's swish lemma, the same day). Both were load-bearing and both
were wrong.

**⛔⛔ WHAT THE TIE ACTUALLY COST: `r34InputGrad` WAS THE REVERSE OF THE WRONG POOL.** It used
`maxPoolFlatBack` — the **2×2** pool's backward — while the committed forward
`resnet34Forward_full_pc` pools with `maxPool3s2Flat`, He et al.'s 3×3/s2 stem pool restored
2026-08-03. `MaxPool3s2.lean`'s header warns in as many words that the two share a TYPE and are
different functions, and the whole-net backward's docstring claimed to be *"the exact reverse of
`resnet34Forward_full_pc`"*. ⚠ **`imagenet_specs_drift_from_twins` for the fourth time in this
file** (§3.3(b) ConvNeXt's head LN was the second, §3.5.2 item 3's scalar-vs-vector LN block the
third): *"the same net as the tie" is an unchecked claim until something forces the two statements
to unify, and what forces it is needing the tie.* ⭐ Note which direction it ran this time — the
FORWARD budget was right (`Maps.maxPool3s2` has been in the chain since r34's forward landed) and
the BACKWARD was wrong, so a per-tier consistency check would not have caught it; only a statement
naming both did.

**The missing leaf, and why it is not free.** `MaxPool3s2BackFloatBridge.lean`:

* `maxPool2`'s windows TILE, so each input is the argmax of at most one output, its backward is a
  LOOKUP, and `Maps.maxPoolBack` carries `Ā ↦ Ā` — exact in float, envelope-preserving.
* 3×3/s2 windows OVERLAP, so an input can be the argmax of up to FOUR outputs and the backward
  **accumulates**: a rounded reduction, window `4Ā(1+γ)`, modulus `γ·4Ā + 4Ē`.
* ⭐ **The `4` is proved, not assumed.** `win3Row_mem_le_two` (already in `MaxPool3s2.lean`, for
  the row axis) says an input row lies in at most two windows, so the fibre of the smooth-point
  reindex is inside a `1 × 2 × 2` box (`maxPool3s2Back_mask_sum_abs_le`). ⛔ Settling for the
  trivial `c·h·w` fibre bound — the shape `floatClose_broadcastBack` uses, §3.9 finding 7 — costs
  **six orders**, landing the fold at `10²⁵¹` against §3.7(a)'s shape-dependent `norm_num` wall at
  ~`10²⁵³`. This is the one place in the file where the loose bound would have been fatal rather
  than merely vacuous.
* ⭐ `maxPool3s2Flat_has_vjp_at_vec` restates the certified witness at a `Vec` point with its
  `backward` field **definitionally the leaf**. Transporting with `▸`/`rwa` gives the right TYPE
  and an `Eq.mpr`-blocked `backward` — §3.5.2 item 5's `FloatBridgesTo.ofEq` trap, one tier down,
  and the same fix: build the structure field-by-field.

**The number moved: 2.188·10²⁴⁵ → 8.857·10²⁴⁵, budget 1.458·10²⁴⁴ → 6.894·10²⁴⁴** (ratio 0.067 →
0.078; still the fold, no cap anywhere). One extra stage, ×4 on the window and a rounding term,
and it lands two stages before the input so almost nothing compounds on it.

**⚠ Two mechanical notes for the next whole-net tie.**
1. **Pin every implicit over a computed dimension, in the STATEMENT as well as the proof.** Both
   `cbrStridedPC_has_vjp_at` and `maxPool3s2Flat_has_vjp_at_vec` are declared over
   `Vec (ic·(2h)·(2w))`, so using them at `3×224²` / `64×112²` asks the elaborator to solve
   `2·?h = 224` — §3.7(d)'s trap, and it presents as a `whnf` timeout inside the theorem's TYPE,
   not inside its proof. `(ic := 3) (oc := 64) (h := 112) (w := 112)` fixes it.
2. **Keep the blocks OPAQUE.** They enter as the `ChainData`/`PProd` bundles `resnet34_has_vjp_at`
   already takes, and `r34InputGrad`'s block slots are pinned to those witnesses' `.backward`, so
   the whole-net `isDefEq` compares variables. Only the four concrete endpoints — stem, pool, GAP,
   dense — are rewritten, and the proof is `unfold r34InputGrad`, two `rw`s, `rfl`. ⚠ Do the
   rewrites at the FUNCTION level: a `funext dy` first makes the `show` that exposes the stem group
   cost more than the whole rest of the proof.

### 3.11 ✅ CIFAR-8 `Env → Maps` (2026-09-04) — the last item of §3.8, and what it found

`cifar8Bridge_maps` replaces `cifar8Bridge_env`: `(cifar8Bridge M W).Maps 1 0 6.121·10¹⁸
6.37·10¹⁴`, twenty-five stages built bottom-up from `Maps.flatConv` / `.dense` / `.relu` /
`.maxPool` threaded by twenty-four generic `Maps.comp`s. Deleted with `FloatBridgesTo.Env`: its
five `Env.comp_*` lemmas, its private `layerAct_le_num` / `layerBudget_le_num`, and one
`private theorem gamma_ub_nonneg` nothing called. 322 lines → 237, ~73 s.

**⚠ It was exactly as mechanical as §5 promised, and the reason is worth writing down once:
`Env.comp_flatConv`'s two hypotheses are LITERALLY `Maps.flatConv`'s** —
`(1+g)·(m·w'·Ā + β) ≤ Ā'` and `g·(m·w'·(Ā+Ē) + β) + m·w'·Ē ≤ Ē'`. So all twenty-two numerals
transfer unchanged, both headline numbers are the same to the digit, and the whole migration is
a re-spelling. Re-asserted independently before the edit (exact rationals, 22 stage inequalities
plus the 22 `gamma_num` side conditions) rather than trusted from the old file.

**⭐ The statement got STRONGER for free.** `Env A Ā Ē` fixed the input window at `A`; `Maps Ā Ē
Ā' Ē'` quantifies over every `A ≤ Ā` and every `E ≤ Ē`. `cifar8Bridge_maps` therefore holds at
every input window `≤ 1`, and `cifar8_float_logits_le` now closes through the generic
`Maps.budget_le` instead of `FloatBridgesTo.fresh_le` + a bespoke projection.

**⭐⭐ The finding: a superseded kit that is never imported alongside its successor hides
duplicate declarations.** `Cifar8FloatBudget.lean` and `FloatBudgetEnv.lean` each defined
`Proofs.FloatBridgesTo.fresh_nonneg`, with different proofs, for two days. Neither file was on
the other's import path, so nothing ever elaborated both and Lean never complained — the
`FloatClose is precision-agnostic` note's duplicate-`convWindow` failure, one import edge away
from firing. ⚠ The generalisation: **the moment a kit is superseded, the successor's file
inherits the obligation to be imported by the old one's consumers, and until that happens the
two namespaces are unchecked against each other.** Grepping for the collision is a one-liner;
nothing runs it.

**⚠ What it cost, and the split that was NOT done.** Importing `FloatBudgetEnv.lean` adds 15
modules to the CIFAR-8 budget's cone (70 → 85), including `Codegen.ResNet34RenderPC` and the ViT
float bridges. None of them is needed for CIFAR-8's four leaves — `floatBridgesTo_flatConv`,
`_dense`, `_relu` and `_maxPool` are all in `FloatComposeBridge.lean` — the weight comes from
`FloatBudgetEnv.lean` importing `Resnet34WholeFloatBridge` for `Maps.flatConvStride2` /
`Maps.biPathSum` and `BnEvalRuntimeFloatBridge` for `Maps.bnEvalPC`. So a `FloatBudgetEnvCore`
holding the structure, `comp`/`mono`/`capped`/`residual`, the two monotone lemmas and the six
`FloatComposeBridge` leaves would be a clean cut, on exactly the reasoning that created
`FloatBudgetEnvMBConv.lean` (§0: *a `Maps` lemma names its bridge*). ⛔ Not done: it edits a file
five landed budget files elaborate against, for zero mathematical gain, and a full build compiles
all 85 modules anyway. Do it if a sixth consumer wants the kit without a net attached.

### 3.12 ✅ `|swish′| ≤ 2` (2026-09-04) — §3.9's one unproved lemma, and B0's backward is statable

`swishScalarDeriv_abs_le` (`Architectures/SwishSaturation.lean`, 134 lines, **1.5 s**), in the
`GeluSaturation.lean` mould: pure real analysis about `swishScalar`, importing only
`LayerNorm.lean`, consumed by the float tier. Four declarations —
`swishScalarDeriv_eq` (the closed form as `σ + x·σ′`, the split the bound needs), two private
helpers, the theorem, and `swishScalar_lipschitz` as its mean-value corollary.

| B0 backward variant | window | budget | statable |
|---|---|---|---|
| leaves as they stood (`\|swish′\| ≤ 1 + A/4`) | 2.491·10⁴³¹ | 4.624·10⁴³⁰ | ⛔ no |
| global `11/10` (⛔ still not proved, and not needed) | 2.021·10¹⁶⁷ | 5.225·10¹⁶⁶ | yes |
| ⭐ **global `2` — what is PROVED** | **7.640·10¹⁶⁹** | **1.735·10¹⁶⁹** | **yes** |
| + operating point `\|x\| ≤ 16` on the SE input | 1.551·10⁹⁸ | 3.535·10⁹⁷ | yes |
| + `broadcastBack` tightened to its `h·w` nonzeros | 3.499·10⁹² | 7.994·10⁹¹ | yes |

**262 orders, and the crude constant costs 2.6 of them against the sharp one.** That is the whole
of §3.9's point made concrete: the shipped `2` is 10¹⁶⁹ and the unproved-sharp `11/10` is 10¹⁶⁷,
both a hundred orders under §3.7(a)'s ~10²⁵³ ceiling. ⛔ Do not prove the sharp constant.

**The proof, and why §3.4's cost estimate was wrong.** `swish′ = σ + x·σ′`, and each summand is
bounded by `1` for a *different* reason: `σ ≤ 1` because `1 + e^{−x} ≥ 1`, and `|x|·σ′ ≤ 1`
because `σ′(x) = e^{−x}/(1+e^{−x})² ≤ e^{−|x|}` and `|x|·e^{−|x|} ≤ 1` off
`Real.add_one_le_exp`. ⭐ The only place saturation enters is a two-line case split on the sign of
`x`: the denominator is `≥ 1` when `x ≥ 0` and `≥ (e^{−x})²` when `x < 0`. No MVT, no `deriv`
analysis, no sup — §3.4's *"needs the decay of `σ′`, i.e. calculus"* is true of the SHARP constant
and was never true of a usable one. ⚠ The estimate sat unchallenged from the day B0's forward
landed to the day someone needed the backward; **a cost written next to a bound reads as a
finding and is what stops it being re-derived.**

**⚠ Where it is exercised, and why that is an `example`.** Every consumer takes the bound as a
parameter — `floatBridges_seGateBack`'s `Ssw`, `EfficientNetBackFloatBridge`'s `swBe`/`swBd` —
so nothing in the repo *changed*; what changed is that a caller can now pass `2` instead of
deriving a window. `SEBackFloatBridge.lean` closes the SE gate's backward at
`(Ssw := 2)` with `fun _i => swishScalarDeriv_abs_le _` as a compiled `example`, at the REAL
saved derivative `fun i => swishScalarDeriv (xsw i)`. It is an `example` because B0's backward
budget file does not exist yet (§3.9 item 3) — and §5's rule is that a leaf nothing composes is a
leaf nobody has checked composes.

**⛔ THE FORWARD IMPLICATION IS PRICED AND NOT TAKEN.** `swishScalar_lipschitz` says swish is
globally `2`-Lipschitz, so `floatClose_swish`'s modulus could carry a third branch `2·e` — and
`2·e` beats BOTH shipped branches at B0's windows (the multiplicative `(1+A/4)·e` for `A > 4`, the
additive `A + e` whenever `e ≤ A`). Measured with a new `b0_eval_chain(swish_lip = …)` flag,
default `None` so the shipped chain and `verify_b0`'s 96 inequalities reproduce byte-for-byte:

| B0 forward | window | budget |
|---|---|---|
| shipped (`min` of multiplicative and additive) | 2.580·10⁵⁵ | 8.408·10²¹⁰ |
| + global `L = 2` | 2.580·10⁵⁵ | **3.679·10²⁰²** |
| + global `L = 11/10` | 2.580·10⁵⁵ | 1.814·10²⁰¹ |

**Eight orders, window unchanged** — a Lipschitz constant is a modulus fact, not a window one,
which is §3.2's separate-levers finding in its cleanest form. ⛔ Not wired: it moves
`b0_float_logits_le`, a committed number with a `formalization.yaml` entry and an `AuditAxioms`
line, and §7 says one commit per net. Do it when B0's forward is next opened.

### 3.13 ✅ MobileNetV2's BACKWARD NUMBER (2026-09-04) — §3.9 item 2, and the first with NO operating point

`mnv2_grad_float_le` (`MobileNetV2BackFloatBudget.lean`, 709 lines, **40 s**): the deployed
MobileNetV2 input-gradient is within **1.076·10¹⁵²** of the certified one, per input pixel, on
loss cotangents of magnitude `≤ 1`, certified window **4.750·10¹⁵³**, `budget/window = 0.023` —
the interval FOLD, no cap anywhere, at TRAINING-mode BatchNorm. 48 stages, 136 inequalities,
generated and re-asserted by `mnv2_back_chain` / `verify_mnv2_back` before a line of Lean was
written.

**⭐⭐ The result is the hypothesis that is ABSENT.** r34's number carries three: `es`, `exh`, and
the operating point `|istd| ≤ 16`. This one carries two. `MnvBnBack` has **no `hS` field** —
`MnvBnBack.hS` is a THEOREM, deriving `|istd| ≤ 317` from `ε ≥ 10⁻⁵` alone through the new
`bnIstd_abs_le_of` (`FloatBudgetEnvBack.lean`), the rational form of `bnIstd_abs_le`'s
irrational `1/√ε`: `317² = 100489 ≥ 100000`, with room. ⭐ **That is what §3.9 finding 2 predicted
and it is worth stating as a rule: an operating-point hypothesis is not a property of backwards,
it is what you pay when the ε-floor fold does not fit under `norm_num`'s ceiling.** r34 at the
floor is 5.503·10²⁸⁸ against §3.7(a)'s ~10²⁵³; MobileNetV2 is 10¹⁵³ and pays nothing.

| net | window | budget | ratio | hypotheses |
|---|---|---|---|---|
| ResNet-34 back | 8.857·10²⁴⁵ | 6.894·10²⁴⁴ | 0.078 | `es`, `exh`, **`\|istd\| ≤ 16`** |
| **MobileNetV2 back** | **4.750·10¹⁵³** | **1.076·10¹⁵²** | **0.023** | `es`, `exh` |

**⭐ What it cost: two `Maps` leaves, and both are compositions of leaves that existed.**
`Maps.depthwiseBack` is `Maps.depthwise` at the spatially-reversed kernel and zero bias
(`depthwiseFlatBack W = depthwiseFlat (dwReverse W) 0`, already proved); `Maps.depthwiseStride2Back`
is that composed with `Maps.decimateBack`, exactly as `Maps.flatConvStride2Back` is `Maps.convBack`
composed with it. ⭐ The fan-in is the whole story: **`kH·kW = 9`, the kernel window ALONE**,
because a depthwise conv mixes no channels — where `Maps.convBack` carries `oc·kH·kW` and r34's
blocks carry `512·9`. Plus the two block envelopes (`Maps.invresBodyBackPC` /
`.invresBodyStridedBackPC`, six stages each) and the `FloatBridgesTo` peers of the two body
backwards, which were only at the `∃`-tier. §3.9's costing was exactly right, for once.

**⭐ MobileNetV2's downsample is a LINE where ResNet-34's is a fan-out.** `Maps.r34DownBlockBack`
is a `biPathSum` — the projection skip means the cotangent goes two ways. The inverted residual
changes stride *inside* the body, so `Maps.invresBodyStridedBackPC` is a straight `.comp` chain
and the only `Maps.residual` in the net is at `b2`/`b4`, applied by the caller OUTSIDE the body.
⭐ Worth carrying: **the block record does not have to own its skip.**

**⚠ Three things that cost time, all mechanical.**
1. ⛔ **The generator was off by one at two BN sites** — it fed `bnBd` the *bnBp* output and
   `bnBe` the *bnBd* output, when the chain is `bnBp → cBp → bnBd → dwB → bnBe → cBe`. The stage
   NUMERALS were right (they come from the probe by tag); only the `(Ā := …)` annotations were
   wrong, so `verify_mnv2_back` could not catch it and the elaborator did — as a type mismatch
   naming both numerals, which reads straight. ⭐ **Pinning all four numerals (§3.7(c)) is what
   turned a silent wrong-input into a compile error.**
2. `floatBridgesTo_invresBodyStridedBackPC` does NOT need `0 < mid·h·w` — the depthwise stage
   scatters to `2h × 2w` before anything joins at the middle width, so only `hnM2` appears. The
   `Maps` peer DOES need both. An unused binder in a `def` is a lint, not an error, so it is worth
   deleting rather than `_`-prefixing.
3. The whole-net bridge `mnv2GradBridge` cost **2.5 s**, against §3.7(d)'s 25-minute
   non-terminating `isDefEq` on r34's. The difference is §3.7's grouping lesson applied from the
   start: `mnv2GradR`/`mnv2GradF` are written with the stem and head groups parenthesised exactly
   as `mnv2InputGrad` writes them.

✅ **The whole-net CERTIFIED tie followed the same day — §3.14.** When this section was written
the tie was open and the scoping said the APEX was genuinely missing (unlike r34, where §3.8's
blocker was a misreading because `resnet34_has_vjp_at` already existed and was dimension-generic).
That scoping was right, and the apex turned out to be nine `vjp_comp_diff_at`s.

### 3.14 ✅ The whole-net CERTIFIED TIE for MobileNetV2 (2026-09-04) — and the shape check r34 lacks

`mnv2InputGrad_eq_mobilenetv2_vjp` (`Foundation/MobileNetV2WholeBackCertifiedTie.lean`, 355 lines,
**~2 s**): `mnv2InputGrad`, with every slot pinned to the certified per-op backward, IS
`(mobilenetv2PC_has_vjp_at …).backward`. So §1's criterion (ii) is met for MobileNetV2's backward
in the same form r34's has since §3.10 — the reading of `mnv2_grad_float_le` is now *"the chain IS
the certified whole-net gradient"*, not *"every piece of this chain is"*.

**⭐⭐ The result is a NEGATIVE that is worth as much as §3.10's positive: no drift.** Closing
r34's tie found `r34InputGrad` reversing the **2×2** pool while the committed forward pools 3×3/s2,
and moved that number 4× (2.188·10²⁴⁵ → 8.857·10²⁴⁵). This tie went through against `mnv2InputGrad`
exactly as committed, so **4.750·10¹⁵³ / 1.076·10¹⁵² stand unchanged**. ⛔ A tie that finds nothing
is not a wasted tie — it is the only way to know the previous section's number was about the net it
claimed to be about.

**⭐⭐ And this file carries the piece `Resnet34BackCertifiedTie.lean` does NOT have.**
`mobilenetv2Forward_full_pc_eq_chain` states by `rfl` that the ten-stage chain the apex is
instantiated at IS the committed forward — `b1/b3/b5/b6` the strided inverted-residual bodies,
`b2/b4` those bodies under `Proofs.residual`, the stem and head spelled as the render spells them.
⛔ **That is the theorem that would have caught r34's wrong pool.** Both whole-net ties take their
blocks OPAQUE (r34's through `ChainData`/`PProd`, this one through `PProd`), so the apex's subject
is a chain of *variables*; nothing in either theorem says which net those variables are. §3.10's
drift survived a month for exactly that reason — *"the same net as the tie"* was prose in a
docstring. ⭐ **Add the shape `rfl` to any future whole-net tie, and add one to r34's.**

**What it cost: two genuinely missing pieces and three repackages.**

| piece | state before | work |
|---|---|---|
| `convStridedBnRelu6PC_has_vjp_at` (stem) | ⛔ missing — the repo had strided-conv-with-**relu** (r34's `cbrStridedPC`) and **non-strided** relu6, not the corner | ~25 lines, copy of r34's with `relu → relu6` |
| stem + head leaf ties | ⛔ missing (the body ties do their leaves inline) | one conv-leaf `rw` then `rfl` each |
| `mobilenetv2PC_has_vjp_at` (apex) | ⛔ missing | nine `vjp_comp_diff_at`s |
| `residualBack_eq_vjp_backward` | ⛔ missing | `rfl` |
| body VJPs + body ties, relu6 VJP, residual VJP, conv/depthwise/dense/GAP leaf ties | ✅ all existed | — |

**⭐ The apex is SIMPLER than ResNet-34's, and the reason is architectural.** `resnet34_has_vjp_at`
needs `ChainData` lists and separate downsample slots because r34's stages are *runs of same-shaped
identity blocks punctuated by a shape change*. MobileNetV2's skips live INSIDE the block maps
(`residual (invresBodyPC …)` at `b2`/`b4`) and its stride changes inside the strided bodies, so the
net is a straight ten-stage chain: nine `vjp_comp_diff_at`s and nothing else. ⭐ Same finding as
§3.13's *"the block record does not have to own its skip"*, one tier up — **where a net puts its
skip decides how much scaffolding its whole-net theorem needs.**

**⚠ Why it cost 2 s where r34's needed `maxRecDepth 400000` and a 25-minute debugging detour.**
Same two disciplines, applied from the start: the blocks stay opaque so the whole-net `isDefEq`
compares variables (§3.7(a)'s lesson in the direction that works), and `mnv2GradR`/`mnv2InputGrad`
are grouped identically so only the four concrete endpoints — stem, head, GAP, dense — are
rewritten. The proof is `unfold`, three `rw`s, `rfl`.

⚠ It stays a SMOOTH-POINT statement, as every `HasVJPAt` in this cone is: the stem's and head's
post-BN clamp windows (`≠ 0 ∧ ≠ 6`, relu6's kink) and the six blocks' own VJP witnesses are
hypotheses. ⚠ And `mobilenetv2_full_has_vjp_at` (`MobileNetV2FullVJP.lean`) is still NOT this
theorem and must not be confused with it: that one is over `MNV2PaperWeights`, the 17-block paper
net, where `mnv2InputGrad` reverses the ch7 6-block render.

### 3.15 ⭐⭐ THE NEXT PROBE: a BACKWARD number for a LAYERNORM net — scoped 2026-09-04,
✅ **RUN THE SAME DAY: it folds. 5.766·10²⁴⁹ / 8.791·10²⁴⁸, ratio 0.15 — §3.16.** Everything
below is the scoping as written before the probe ran; all four of its predictions held (the
reduction width, the GELU bound, the profile split, and ViT's cone), and it missed one thing —
the head-LN slot (§3.16 finding 6)

**The question nobody in this file has asked.** §0.1 says LayerNorm's forward is quadratic in the
window with no eval mode to escape to, so ConvNeXt-T's and ViT-Tiny's numbers are `2.00` **caps**
(§9) and there is no honest whole-net fold for a LayerNorm net anywhere in the repo. §3.9 finding 1
says **every backward folds** — a VJP is linear at a fixed point. Put those together and the
obvious question is whether **a LayerNorm net's BACKWARD is the first honest fold for one**. It has
never been probed: there is no `cnx_back_chain` or `vit_back_chain` in
`scripts/float_budget_envelope.py`, and no section of this file mentions the possibility.

⭐ **Start with the probe, not the Lean** (§3.8), and start with **ConvNeXt, not ViT** (below).

**What was verified today, so the probe does not re-derive it.**

1. ⭐⭐ **ConvNeXt's channel-LayerNorm BACKWARD is BatchNorm's backward at a different row
   decomposition, and `bnXhat_sq_le` already covers it.** `convnextCh_grad_floatBridgesTo`
   (`ConvNeXtBackFloatBridge.lean`) states its stem LN-back's hypotheses through **`bnIstd`** and
   **`bnXhat`** at `Mat.unflatten (chanLNRows 96 56 56 xstem) r` — literally the same two
   quantities `R34BnBack` carries, conjugated by `chanLNRows` instead of `reassocFwd`. ⛔ **So the
   gap you would predict — "there is no `lnXhat_sq_le`" — does not exist.** §3.7's ⭐⭐
   load-bearing lemma applies verbatim, with the reduction width the CHANNEL count. §3.3.0(b)'s
   rule for the sixth time: the bound was already there, under the other net's name.
2. ⭐ **GELU's global derivative bound is already proved.** `geluScalarDeriv_abs_le : |gelu′| ≤ 3/2`
   (`Architectures/GeluSaturation.lean`) — the analogue of the swish lemma that blocked B0's
   backward for a month (§3.12) is in the repo for these nets *before* anyone needs it.
3. ⭐ **The softmax backward's saved quantity is bounded by `1`, unconditionally.**
   `floatClose_softmaxBack` (`SoftmaxBackFloatBridge.lean`) is stated at `|p| ≤ P` and instantiates
   at `P = 1` for softmax, because a softmax row IS a probability vector. So §3.7's relocated
   question — *which saved activations does the backward scale by, and is each bounded by something
   other than the forward's window?* — is answered favourably for attention before it is asked,
   which is the opposite of B0's `swish′(saved)` and `x`.
4. ⚠ **The two nets are NOT in the same state, and this decides the order.** ConvNeXt's cone is
   closed down to the stem LN — `convnextCh_grad_floatBridgesTo` takes only the four stages and
   three downsamples abstractly, exactly the state r34's was in before §3.7 and mnv2's before
   §3.13. ViT's `vit_grad_floatBridgesTo` (`MhsaBackFloatBridge.lean`) takes the **final LN, the
   whole block LIST, and the patch embed** as supplied bridges — a tier up. ⛔ Expect §3.5.1's
   surprise to repeat there (*"the cone is at the `∃` tier essentially everywhere, and that is the
   bulk of the work"*); check it before costing ViT.

**What the probe must measure.**

* The fold for `convnextInputGrad` at ConvNeXt-T's shapes over the loss cotangent, at the
  granularity a `Maps` chain composes — the `r34_back_chain` / `mnv2_back_chain` mould, reusing
  `bn_back` for every LN site at `n = C` and `Xh = √C`.
  ⚠ **The reduction width is the CHANNEL count (96/192/384/768), where r34's and mnv2's is `h·w`
  (49…12544).** `bnGradInputReMag` is `S·G·Cdy·(2 + Xh²)` and `Xh² = n`, so a site's real gain is
  `S·G·(n+2)`: ConvNeXt's per-site gain is far smaller than r34's, and it has 23 LN sites against
  r34's 33 BN sites. ⛔ **That is arithmetic, not a prediction — the fold is NOT computed here, on
  purpose.** §3.9's rule: a number written down beside a bound and not re-derived is what stops
  anyone checking it.
* Whether it lands under §3.7(a)'s **~10²⁵³** shape-dependent `norm_num` ceiling, and whether it
  needs an operating point. §3.13's rule: *an operating-point hypothesis is not a property of
  backwards, it is what you pay when the ε-floor fold does not fit.*
* The ablations this file always wants: `x̂` from `bnXhat_sq_le` vs from the forward's window
  (decisive three times — §3.9 finding 4), the per-kind profile split (⚠ ConvNeXt's kinds are
  **14× apart** and the outlier is the LAYER SCALE, §3.3(a) — expect it to matter more here than
  on any other net), and GELU's `3/2` against the cubic polynomial.
* ⭐ For ViT, the **`exp_tainted` instrument is mandatory** (§3.5, §0.1): a Python fold hides a
  `Real.exp` at an unbounded argument, and "statable" is `max(exponent) < 253` **AND** no taint.

**⭐⭐ And the genuinely new question, which is not about size.** On r34 and mnv2 the caveat is that
`es`/`exh` — the saved float activations' accuracies — are quantities *the forward's own
training-mode fold cannot discharge* (§0.1's 10⁷⁴¹⁷). For ConvNeXt and ViT the forward has **no
fold at all**: its number is the triangle inequality, and it never claimed a rounding bound to
begin with. So a LayerNorm net's backward number would be an honest fold composed against a
forward that is a *cap*. ⛔ **Decide what that statement means before quoting one**, and write the
answer into §9 — it is a new row in the "what to say about the numbers" table, not a footnote.

### 3.16 ⭐⭐ A LAYERNORM NET'S BACKWARD FOLDS (2026-09-04) — §3.15's probe, run

`cnx_back_chain` / `verify_cnx_back` (`scripts/float_budget_envelope.py`, 137 stages, 322
re-assertions) fold `convnextInputGrad` (`ConvNeXtBackFloatBridge.lean`) at ConvNeXt-T's shapes
over the loss cotangent, in the leaves' own semantics. No Lean was written.

    window ≤ 5.766·10²⁴⁹      budget ≤ 8.791·10²⁴⁸      budget/window = 0.15
    at the operating point |istd| ≤ 16; 322 re-assertions

**⭐⭐ The answer is YES.** Ratio 0.15, no `capped` anywhere — **the repo's first honest whole-net
fold for a LayerNorm net**, at the net whose FORWARD number is a 2.00 cap. §0.1's quadratic is a
forward fact and §3.9's finding 1 now holds at the fourth net and the third normalisation: a VJP
reads its statistics off the saved activations, which the cotangent does not perturb.

| variant | window | budget | ratio | statable |
|---|---|---|---|---|
| **shipped shape: `\|istd\| ≤ 16`, per-kind profile, `x̂ ≤ √C`, GELU `3/2`** | **5.766·10²⁴⁹** | **8.791·10²⁴⁸** | 0.153 | **yes** |
| the unconditional ε-floor `S = 317` | 3.833·10²⁷⁹ | 5.394·10²⁷⁸ | 0.141 | ⛔ no |
| variance floor 10⁻³ (`S = 32`) | 4.795·10²⁵⁶ | 7.012·10²⁵⁵ | 0.146 | ⛔ no |
| variance floor 10⁻¹ (`S = 4`) | 8.520·10²³⁵ | 1.607·10²³⁵ | 0.189 | yes |
| `σ² ≈ 1` (`S = 1`) | 1.438·10²²² | 4.551·10²²¹ | 0.317 | yes |
| uniform `84/10` param bound, at `S = 16` | 2.406·10³¹⁷ | 3.663·10³¹⁶ | 0.152 | ⛔ no |
| `x̂` from the forward WINDOW (`2·A·S`) | 1.472·10⁵¹⁴⁷ | 1.889·10⁵¹⁴⁶ | 0.128 | ⛔ no |
| **GELU's cubic polynomial (no saturation constant)** | 3.769·10⁶³³⁰ | 1.685·10⁶³²⁹ | 0.045 | ⛔ **no** |
| ⛔ head-LN slot `id` — **what the SHIPPED bridge says** | 9.549·10²⁴⁴ | 1.444·10²⁴⁴ | 0.151 | yes |

**⭐⭐ Finding 1 — the genuinely new question §3.15 asked, and the answer it needs before anyone
quotes this number.** On r34 and mnv2 the caveat is that `es`/`exh` — the saved float activations'
accuracies, taken at `10⁻²` — are quantities the forward's own training-mode fold cannot discharge
(§0.1's 10⁷⁴¹⁷). ConvNeXt's forward supplies something **different in kind, not merely in size**:
its certified statement is `FloatBridgesTo.capped`, whose modulus is `2·mag` *by construction*, so
the activation accuracy it implies is `2 × window ≈ 10²²⁷` and no tightening of the profile or the
operating point moves it below that. On r34 the gap between hypothesis and forward is quantitative
and the forward has an inference mode where its statement IS a fold; ConvNeXt has no such mode,
because LayerNorm has no statistics to freeze (§0.1). **So say it this way: the ConvNeXt backward
number is an honest fold of the BACKWARD kernel's rounding, at a hypothesised operating point,
given saved-activation accuracies that this net's forward cannot supply in any mode.** It is not,
and cannot be assembled into, a statement about a deployed forward-then-backward composition.
⭐ That sharpens §3.7's closing line — *the wall is not a fact about backwards, it is a fact about
composing a backward with the forward that feeds it* — with the case where the composition is
blocked by the forward's KIND rather than its magnitude. §9 has the new row.

**⭐ Finding 2 — it pays an operating point, and it has the tightest margin of the four
backwards.** At the ε-floor it is 10²⁷⁹; `|istd| ≤ 16` buys 30 orders and lands 10²⁴⁹, **four
orders under §3.7(a)'s ~10²⁵³ shape-dependent ceiling** — where r34's is eight and MobileNetV2's
pays nothing at all. §3.13's rule for the third time: *an operating-point hypothesis is not a
property of backwards, it is what you pay when the ε-floor fold does not fit.* ⚠ Four orders is
close enough that a Lean file should be written against `S = 8` (11 orders) or `S = 4` (18) if
`norm_num` balks at the shape; the headroom sweep is in the probe.

**⭐⭐ Finding 3 — GELU's global derivative bound is LOAD-BEARING, and it is worth 6081 orders.**
`geluScalarDeriv_abs_le : |gelu′| ≤ 3/2` (`Architectures/GeluSaturation.lean`) is what the 18
`geluB = diagBack (gelu'(saved))` slots take; `floatClose_gelu`'s magnitude polynomial — cubic in
the forward's pre-GELU window — puts the same fold at 10⁶³³⁰. **This is B0's swish blocker exactly
(§3.12), one net over, with the opposite ending: the lemma was already in the repo, written for
ConvNeXt's FORWARD, where §3.3 measured it as *not* load-bearing.** ⭐ §3.15 item 2 predicted this
before the probe ran, and the pair is the sharpest statement of §3.3.0(b) yet: *a bound that is
free and not load-bearing on the forward can be the whole theorem on the backward.* Do not retire
a saturation constant because a forward ablation says it buys nothing.

**⭐ Finding 4 — the per-kind profile split is worth 68 orders, by far the largest of any net.**
r34's is 8, MobileNetV2's 4, B0's under 1; here the uniform `84/10` — ConvNeXt's layer scale, 14×
the conv kernels (§3.3(a)) — is 10³¹⁷ against the split's 10²⁴⁹, and it is the difference between
a theorem and none. §3.15 expected this and the reason is the same as the forward's: the outlier
kind multiplies *inside* every block, so every conv fan-in in the chain pays for it.

**⚠ Finding 5 — `bnXhat_sq_le` is decisive for the FOURTH net.** 10⁵¹⁴⁷ from the forward's window,
10²⁴⁹ from `|x̂| ≤ √n`. ⭐ And it needed nothing new: `convnextCh_grad_floatBridges` already states
its LN-back hypotheses through `bnIstd 96 (Mat.unflatten (chanLNRows 96 56 56 xstem) r)` and
`bnXhat 96 …` — r34's two quantities under a different conjugation — so the lemma applies verbatim
at reduction width `C`. ⚠ `√C` is irrational for all four of 96/192/384/768, so the Lean leaf needs
`bnXhat_abs_le_num` at the CEILING root (10/14/20/28), not the exact one r34's square feature maps
gave for free.

**⛔⛔ Finding 6 — THE SHIPPED BACKWARD BRIDGE HOLDS `id` IN ITS HEAD-LAYERNORM SLOT.**
`convnextCh_grad_floatBridges` and `convnextCh_grad_floatBridgesTo` both instantiate
`convnextInputGrad`'s `lnBhead` at `id` — and say so in the docstring, *"and the head slot `id`"*,
as though it were a choice. The committed forward `convNextForwardTCh` has carried
`rowLNVecFlat 1 768 w.hε w.hγ w.hβ` since 2026-08-30, and **§3.3(b) fixed this exact slot on the
FORWARD bridge on 2026-09-03 and did not look at the backward.** So the shipped backward bridge is
the reverse of a net the repo does not train. ⚠ `imagenet_specs_drift_from_twins` for the FIFTH
time in this file, and the second time (§3.10's 2×2 pool was the first) that it is the BACKWARD
that drifted while the forward was right. ⭐ Note what found it: not a review of the backward, but
needing its number — the same trigger every time. Worth 6 orders (10²⁴⁴ → 10²⁴⁹), which is the
least interesting thing about it.

⭐⭐ **And the docstring said WHY, which is what makes this instructive rather than just a bug:**
*"the head LN back becomes the identity — the reference's 22 LN sites are 1 stem + 18 block + 3
downsample, so there is no head LN to reverse."* **That count is a fossil.** It was true of the
pre-2026-08-30 net; the FORWARD bridge's docstring two files over says **23**. So the slot was not
an oversight — it was a *justified* choice, correctly derived from a number that had since changed,
and the justification is what stopped anyone re-checking it. ⛔ **That is §3.9's swish lesson in a
second guise: a cost estimate written next to a bound reads as a finding, and so does a COUNT
written next to a slot.** When a docstring justifies an `id`, check the count.

✅ **FIXED 2026-09-04** (`ConvNeXtBackFloatBridge.lean`). Both tiers —
`convnextCh_grad_floatBridges` and `convnextCh_grad_floatBridgesTo` — now take
`rowLNVecFlatBack 1 768 εh γh xhead` and its float peer `rowLNVecFlatBackF`, discharged by
`floatBridges_rowLNVecFlatBack` / `floatBridgesTo_rowLNVecFlatBack`. ⭐ **It cost no new
mathematics: every piece already existed**, written for the channel LN's own conjugation
(`chanLNTensor3Back` IS `rowLNVecFlatBack` between two permutations), so the fix is six hypotheses
and a slot — §3.3.0(b) for the eighth time. ⚠ The head carries its own `Xhh` (`√768` rounds to 28
where the stem's `√96` rounds to 10) but shares the profile's `Gd`/`egam`/`S`/`es`/`exh`, which is
the factoring a budget file wants: γ bound, float accuracies and operating point are net-level,
`Xh` is per-site. `lake build Proofs Certs` 3948 jobs, `AuditAxioms` exit 0, both theorems on
`[propext, Classical.choice, Quot.sound]`, `docstring-checkrefs` and `check_audit_coverage` clean.

**⚠ Finding 7 — ViT's backward cone is where §3.15 said, so ConvNeXt-first was the right order.**
`vit_grad_floatBridgesTo` (`MhsaBackFloatBridge.lean`) takes the final LN, the whole block LIST
(`FloatBridgesToList`) and the patch embed as supplied bridges, and the block-level facts under it
— `floatBridges_vitBlockBack`, `_mhsaBack`, `_coreQ/K/V`, `floatBridges_patchEmbedBack` — are all
at the `∃` tier, with no `FloatBridgesTo` peers but the structural `id` / `towerBack` /
`clsScatter`. §3.5.1's surprise repeats verbatim: **the migration is the bulk of ViT's backward
work**, and it is not visible from the whole-net file.

**What to do next, in order.**
1. ✅ **Fix the head-LN slot — DONE 2026-09-04** (finding 6). Both tiers take the real
   `rowLNVecFlatBack 1 768` and its float peer; no new leaves were needed, and the whole cone
   rebuilt in 1.7 s.
2. ✅ **The `Maps` KIT — DONE 2026-09-04 (`FloatBudgetEnvBackLN.lean`, §3.17).** ⛔ **And the
   costing above was wrong twice.** It said "nothing new in kind"; there was one genuinely new
   leaf, and the block bridges the budget file composes did not exist at the `FloatBridgesTo`
   tier at all. §3.17.
3. ⭐⭐ **THE APEX FIRST — the whole-net certified tie, §3.18.** ⛔ Not the number: r34 and mnv2
   folded first and tied after, and §3.10 is why not to repeat it — that tie found r34 reversing the
   wrong pool and moved the committed number 4×. This net has already produced one defect of that
   class this week (finding 6). ⭐ `convNextForwardTCh_has_vjp` already exists at the committed net,
   and so does §3.14's shape `rfl`; what is missing is the stage-fold backward tie, the downsample
   tie, the patchify leaf tie, and the assembly.
4. **`ConvNeXtBackFloatBudget.lean`** — the number, against whatever chain item 3 certifies.
   Assembly on the `Resnet34BackFloatBudget.lean` recipe (records,
   `cnxGradR`/`cnxGradF`/`cnxGradBridge` grouped exactly as `convnextInputGrad` groups, the
   137-stage `Maps` chain). ⭐ State it at `S = 16`: §3.17's ceiling probe says the shape closes.

### 3.17 ✅ The `Maps` kit for a LAYERNORM net's BACKWARD (2026-09-04) — and two costings that were wrong

`FloatBudgetEnvBackLN.lean` (330 lines, **7.5 s**), the third backward net's kit beside
`FloatBudgetEnvBack.lean`'s (r34) and `FloatBudgetEnvBackMBConv.lean`'s (MobileNetV2). With it the
ConvNeXt-T backward cone is closed at the `Maps` tier: everything
`ConvNeXtBackFloatBudget.lean` needs exists and is exercised.

**⛔ §3.16 costed this as "nothing new in kind" and was wrong twice.**

**⭐⭐ Wrong once: there IS a new leaf, and reusing `Maps.bnPerChannelBackGain` at `n = C` would
have been a silent numeral error.** `floatBridgesTo_rowLNVecFlatBack` does not carry γ inside the
gain the way the BN backward does — it runs `bn_grad_input` with `|(1:ℝ)|` in its `G` slot and
folds the γ scale in FRONT of it as a separate `diagBack`, so the γ multiply is a rounded stage of
its own and its `mulErr` enters the fold:

    D    = Gd·Ā + mulErr q Gd Ā egam 0            — the diagBack's output window
    mag  = D·(Kr + Kb)
    mod  = D·Kb + (mulErr q Gd Ā egam 0 + Gd·Ē)·Kr

with `Kr`/`Kb` the per-unit gains at **unit γ**. ⚠ The probe had folded it as `bn_back` at `n = c`,
`G = γ` — the same leading term, ~0.2 % apart per site — so the NUMBER barely moved (5.766·10²⁴⁹
window unchanged at four figures, budget 8.791 → 8.790·10²⁴⁸). **But the kernel checks the leaf's
arithmetic, not the leading term**, and a chain folded at the wrong shape emits stage numerals the
kernel then rejects. §0's ⚠ — *the fold must assert exactly what the proof asserts* — reaches one
level deeper than "use the rounded γ": ⭐ **read the leaf's `mag`/`mod` before folding it, even
when a peer leaf with the same name-shape exists.**
⭐ The homogeneity still pays, and pays MORE here than on r34: ConvNeXt-T has 23 LN sites but only
**four** distinct reduction widths (96/192/384/768), so four `hKr`/`hKb` pairs serve the whole net.

**⛔⛔ Wrong twice: the block bridges the budget file composes did not exist at the
`FloatBridgesTo` tier.** `ConvNeXtBackFloatBridge.lean` had `floatBridges_cnxBlockBodyBack`,
`_cnxBlockBack` and `_cnxDownBack` — all `∃`-tier — and a budget file cannot use those: `FloatBridges`
discards the float map and a `Maps` envelope has to name one. **That is §3.5.1's surprise, one net
over and on the backward** (*"the cone is at the `∃` tier essentially everywhere, and that is the
bulk of the work"*), and §3.16 finding 7 had flagged it for ViT while missing it for the net it was
about to build. ⭐ The rule to carry: **when scoping a budget file, grep the cone for
`floatBridgesTo_` — not for the defs, which exist either way.** Three defs, three float skeletons,
~80 lines, and all three compiled first try.

**What landed.**

| piece | what it is |
|---|---|
| `Maps.rowLNVecFlatBack` | ⭐⭐ the new leaf, in per-unit-gain form; both monotonicity branches |
| `Maps.chanLNTensor3Back` | that leaf between the four exact layout gathers — the forward's `Maps.chanLNTensor3` shape |
| `Maps.decimateOddBack` / `Maps.flatConvStride4Back` | the 4×4/s4 patchify backward: `Maps.convBack` after two exact scatters |
| `Maps.cnxBlockBodyBack` | the six-stage body envelope; the skip is the caller's (`Maps.residual`), as MobileNetV2's is |
| `Maps.cnxDownBack` | strided-conv backward then the LN back at the INPUT resolution |
| `cnxBlockBodyBackF` / `cnxDownBackF` + three `floatBridgesTo_*` | the tier migration |

⭐ **And the block is EXERCISED, not just written.** The file closes ConvNeXt-T's `s4b2` — the
first block the cotangent meets, `c = 768`, `cExp = 3072`, 7×7 — as a compiled `example` at
`cnx_back_chain`'s numerals: in `(7407, 1.025·10¹)`, out `(1.421·10¹⁴, 1.365·10¹²)`, with the real
`floatBridgesTo_chanLNTensor3Back` in the LN slot and both `diagBack`s (layer scale at `es = 0`,
saved `gelu′` at `10⁻²`) in theirs. §5's rule, and simultaneously the check that the generator's
arithmetic IS these lemmas' — which is what caught the leaf-shape error above.

**⭐⭐ And the four-orders worry in §3.16 finding 2 was unfounded — measured, not assumed.**
Before writing a line of the budget file, the largest goals the chain will assert were emitted at
both `S = 16` and `S = 8` and thrown at `norm_num`: the stem's `convBack` at 10²⁴⁹, the stem LN
gain, the deepest residual, the 49- and 384-fan-in convs. **All close, both values, 2.8 s total.**
§3.7(a)'s ceiling is shape-dependent, and the shape that hit it was a NESTED tree
(`317/12544 * (12544 * (21/10 * ⋯))`) — which `Maps.bnPerChannelBackGain`'s factoring already
flattens, and `Maps.rowLNVecFlatBack` inherits. ⭐ **So state ConvNeXt's number at `S = 16` and do
not overpay the hypothesis.** The general lesson: §3.7(a) says the ceiling depends on the operation
tree, so **probe the tree** — a ten-line file and three minutes, against choosing a stronger
hypothesis for the rest of the theorem's life.

**Process.** `lake build Proofs Certs` 3949 jobs; `AuditAxioms` exit 0 with all twelve new
declarations on `[propext, Classical.choice, Quot.sound]`; `docstring-checkrefs` 1454 citations;
`check_audit_coverage` 204 imports. The module is a lakefile `Proofs` root (nothing imports it
until the budget file does — the `render guard on a new artifact` failure mode, one tier over).

### 3.18 ⭐⭐ THE APEX: scoped 2026-09-04 — and it EXISTS, at the committed net

⭐ **Do this BEFORE the budget file, reversing the order r34 and MobileNetV2 were done in.** Both
of those stated the number first and tied afterwards (§3.7 → §3.10, §3.13 → §3.14), and §3.10 is
the reason not to: closing r34's tie found `r34InputGrad` reversing the **2×2** pool while the
committed forward pools 3×3/s2, and **moved the committed number 4×**. ConvNeXt has already shown
the same class of defect once this week — the head-LayerNorm slot (§3.16 finding 6) — so spending
137 stages of numerals before knowing the chain is the right net is buying the same lottery ticket
twice. ⭐ **Tie first, then fold.**

**⛔ The blocker one would write down — "the apex is missing" — is a misreading, for the second
time in this file** (§3.8 item 2 was the first, on r34). `convNextForwardTCh_has_vjp`
(`Architectures/ConvNeXtFullT.lean`) is the whole committed net: `[3,3,9,3]`, channel LayerNorm,
vector affines, layer scale, three downsamples, the 4×4/s4 patchify stem, GAP, **the head
`rowLNVecFlat 1 768`**, dense — chain-stated with the blocks opaque, exactly the shape §3.14 says a
whole-net tie wants. It is `HasVJP` (everywhere) rather than `HasVJPAt`, which is *stronger* than
r34's and mnv2's smooth-point statements. `convNextForwardTCh_has_vjp_correct` is beside it.

⭐⭐ **And ConvNeXt already HAS the piece §3.14 said to add to every future tie and r34 still
lacks**: `convNextForwardTCh_eq_chain` — the `rfl` saying the chain the apex is instantiated at IS
the committed forward. That is the theorem that would have caught r34's wrong pool, and here it
was written before anyone needed it.

**⛔⛔ But `convnext_has_vjp_at` is the WRONG TIER and must not be used** — it is a fixed-depth
TWO-block net over `layerNormForward`'s **scalar** γ/β normalising the whole flattened vector,
where the committed net is `chanLNTensor3`'s per-position channel LN with **vector** affines.
⚠ That is §3.5.1's `vit_full` trap and §3.5.2 item 3's scalar-vs-vector block, for the third time:
*the two statements look interchangeable until something forces them to unify.* Same name stem,
different function.

**⚠ And a THIRD copy of §3.16's fossil is sitting in that apex's docstring.**
`convNextForwardTCh_has_vjp` says *"22 LN positivities (stem + 18 blocks via the per-stage `∀ i` +
3 downsamples), **no head LN**"* — while its own statement composes `rowLNVecFlat 1 768 w.hε w.hγ
w.hβ` and takes `hhε : 0 < w.hε`. Count them: 1 + 18 + 3 + 1 = **23**. The statement is right and
the docstring is the pre-2026-08-30 count, in a third place. ⛔ `docstring-checkrefs` cannot see
this — it resolves cited identifiers, and a stale COUNT cites nothing. Fix it while you are there.

**What exists, and what is actually missing.**

| piece | state |
|---|---|
| `convNextForwardTCh_has_vjp` (apex, committed net, `HasVJP`) | ✅ exists |
| `convNextForwardTCh_eq_chain` (§3.14's shape `rfl`) | ✅ exists |
| block ties: `cnxBlockBack_eq_convNextBlock_vjp`, `cnxBlockChBack_eq_vjp`, `cnxBodyWithChanLNBack_eq_vjp` | ✅ exist |
| `chanLNTensor3Back_eq_chanLN_vjp`, `rowLNVecFlat_has_vjp_backward_eq`, `bn_grad_input_eq_vjp_backward` | ✅ exist |
| leaf ties: `convFlatBack_`, `depthwiseFlatBack_`, `dense_transpose_`, `gapBack_`, `flatConvStride2Back_`, `decimateBack_eq_vjp` | ✅ exist |
| `flatConvStride4Back_eq_vjp_backward` (the patchify backward) | ⛔ **missing** — the stride-2 peer exists; this is it with one more exact scatter |
| the STAGE-fold backward tie (`convNextStageChK`'s VJP backward = the `k`-fold composition of block backwards) | ⛔ **missing** — `convNextStageChK_has_vjp` exists, its backward is not yet identified; a `k`-recursion, the one real proof here |
| `cnxDownBack_eq_vjp` (`lnB ∘ flatConvStride2Back` = `(cnxDownChW_has_vjp …).backward`) | ⛔ **missing** — two existing ties composed |
| `convnextInputGrad_eq_convNextForwardTCh_vjp` (the assembly) | ⛔ **missing** — the §3.10 / §3.14 apex theorem |

⭐ So the shape of the job is MobileNetV2's, not ResNet-34's: one genuinely new recursion (the
stage fold), two composed ties, one leaf tie, and an assembly. ⚠ Keep the stages OPAQUE the way
§3.14 did — `convnextInputGrad`'s `s1B..s4B` slots pin to the certified stage backwards and the
whole-net `isDefEq` then compares variables, which is why mnv2's tie cost 2 s where r34's cost a
25-minute debugging detour. And ⚠ group `cnxGradR` exactly as `convnextInputGrad` groups its stem
(`flatConvStride4Back ∘ lnBstem`), for the same reason.

⚠ **Pin every implicit over a computed dimension in the STATEMENT** (§3.10 note 1): the patchify
backward is declared over `Vec (ic * (2*(2*h)) * (2*(2*w)))`, so using it at `3 × 224²` asks the
elaborator to solve `2*(2*?h) = 224` — §3.7(d)'s trap, presenting as a `whnf` timeout inside the
theorem's TYPE. `(ic := 3) (oc := 96) (h := 56) (w := 56)`. The three downsamples' stride-2
backwards have the same shape.

## 4. What is open (2026-09-04)

§3.8's three items are all closed, so this is its successor. Ordered by what I would do next.

**1. ✅ Probe a LAYERNORM net's BACKWARD — DONE 2026-09-04. See §3.16.** It folds:
5.766·10²⁴⁹ / 8.791·10²⁴⁸ at `|istd| ≤ 16`, ratio 0.15, no cap anywhere — the repo's first honest
whole-net fold for a LayerNorm net. What it opened is below.

**2. ✅ Fix ConvNeXt's head-LayerNorm BACKWARD slot — DONE 2026-09-04, §3.16 finding 6.** Both
tiers now take `rowLNVecFlatBack 1 768` and `rowLNVecFlatBackF`; every piece already existed, and
the `id` was justified in its own docstring by a LayerNorm count that had gone stale.

**3. ✅ The `Maps` kit for a LayerNorm net's backward — DONE 2026-09-04, §3.17.**
`FloatBudgetEnvBackLN.lean`: `Maps.rowLNVecFlatBack` (the one new leaf), the channel-LN
conjugation, the patchify backward, the block-body and downsample envelopes, plus the
`FloatBridgesTo` migration of the three ConvNeXt backward block defs — and ConvNeXt-T's block
`s4b2` closed as a compiled `example` at the probe's numerals.

**4. ⭐⭐ THE APEX — ConvNeXt's whole-net certified tie, `convnextInputGrad_eq_convNextForwardTCh_vjp`.
Scoped 2026-09-04, §3.18, and it goes BEFORE the number.** r34 and mnv2 folded first and tied
after; §3.10 is why not to repeat that — the tie found r34 reversing the wrong pool and moved the
committed number 4×, and ConvNeXt has already produced one defect of that class this week
(§3.16 finding 6). ⭐ The apex `convNextForwardTCh_has_vjp` **already exists at the committed net**,
and so does §3.14's shape `rfl`. Missing: the stage-fold backward tie (the one real proof), the
downsample tie, the patchify-backward leaf tie, and the assembly.

**5. `ConvNeXtBackFloatBudget.lean`** — §3.16's number, at `S = 16` (⭐ the ceiling probe in §3.17
says the shape closes; the four-orders worry was unfounded). Everything it needs exists and is
exercised; what is left is assembly on the `Resnet34BackFloatBudget.lean` recipe. ⚠ Build it
against whatever chain item 4 certifies, not against today's.

**6. EfficientNet-B0's backward budget file** — §3.9's "what to do next" item 3.
The blocker is gone (§3.12's `|swish′| ≤ 2`), the fold is **7.640·10¹⁶⁹ / 1.735·10¹⁶⁹** and
statable. Needs `Maps.diagBack`, `Maps.broadcastBack` and the `Maps.seBack` composite (`biPathSum`
of two `.comp` chains — no new combinator), plus four supplied saved-vector accuracies where r34
has two. ⚠ `batchMap` never enters a numeral, so the number holds at any `N`.

**7. B0's whole-net certified tie** — the §3.14 analogue, and §1's criterion (ii) for that net. Not
scoped; scope it the way §3.14 was, by reading the cone rather than guessing.

**8. ⭐ `resnet34Forward_full_pc_eq_chain`** — §3.14's own recommendation. ⭐ ConvNeXt already has
its peer (`convNextForwardTCh_eq_chain`, §3.18); ResNet-34 is now the only net whose whole-net
backward tie has no shape check, and it is the net the hole already bit.
`Resnet34BackCertifiedTie.lean` has no shape check: its apex's subject is a chain of VARIABLES and
nothing in the theorem says which net they are. That is precisely how §3.10's wrong pool survived a
month. Cheap, and it closes the hole that already bit once.

**Priced, deliberately NOT taken** (each is a decision, not an oversight):

* **`swishScalar_lipschitz` into `floatClose_swish`** — 8 orders on B0's FORWARD
  (8.408·10²¹⁰ → 3.679·10²⁰²), window unchanged. Moves a committed number; §3.12, §7's one-commit-
  per-net rule. Do it when B0's forward is next opened.
* **`floatClose_broadcastBack`'s spurious factor of `c`** — 6 orders on B0's backward, cheap, blocks
  nothing (§3.9 finding 7).
* **The `FloatBudgetEnvCore` split** — cone hygiene, zero mathematical gain (§3.11).

**Flagged rather than scheduled.** §0.1's escape 2 — *a genuinely linear bound needs the NORMALISED
output's Lipschitz constant, not the pre-normalisation one* — is still open for the FORWARDS, and is
still "real work and probably the interesting result". Every forward number in §0's table is vacuous
by hundreds of orders and this is the only item that would change that. ⭐ The backward's
operating-point `S` is the worked precedent for what such a bound is worth: ~43 orders across 33 BN
sites, and the difference between a number that exists and one that does not (§3.7 step 4(b)).

**Stated as missing, still missing.** `efficientnetForwardBEval N = batchMap N (per-example
forward)` — the whole-net form of "at inference the batch decouples". Only the per-SITE claim is
proved (`den_batchOp_bnEval`); it needs a `batchMap` composition lemma plus a per-example B0 def as
the witness (§3.4).

## 5. The `Maps` kit — ✅ complete for all six forwards, and for r34's backward

✅ **All eight the MBConv family needs now live in `FloatBudgetEnvMBConv.lean`**: `relu6`
(window `min Ā 6`, NOT a copy of `Maps.relu` — §3.2), `depthwise`, `depthwiseStride2Flat`,
`swish` (⭐ modulus = the `min` of a multiplicative and an additive sensitivity — §3.4),
`sigmoid` (window `1 + esig` at any input), `broadcast`, `seScale` (⭐ window = the gate's
MAGNITUDE, not mag + error) and `batchMap` (the identity on the envelope, which is the
envelope-level statement that the op does not couple the batch). They are in their own file
rather than `FloatBudgetEnv.lean` because a `Maps` lemma names its bridge and none of these
bridges is on that file's import path.

⭐ The file closes B0's b1 squeeze-excite site as a compiled `example` at the numerals
`b0_eval_chain` emits — five leaves in one chain, and simultaneously the check that the
generator's arithmetic IS these lemmas'. Do that for every new leaf: an unexercised `Maps` leaf
is the `stale lean_exe gates` failure mode in proof form.

✅ **The LayerNorm set landed too, in `FloatBudgetEnvLN.lean`**: `capped` / `Maps.capped` (the
bridge transformer, §3.3.0(a)), `Maps.bnCapped` (the capped pure-normalise LN), `Maps.diagBack`
(the LN γ multiply AND the layer scale — `layerScale γ = diagBack γ` definitionally),
`Maps.biasAdd`, `Maps.gather` / `Maps.idVec` / `Maps.perRow` (the structural steps), `Maps.gelu`
(⭐ through the `3/2` saturation branch, NOT the cubic polynomial — which is also irrational and
could not be `norm_num`'d), and `Maps.flatConvStride4`. Plus the three composites the chain
actually walks: `Maps.chanLNTensor3`, `Maps.cnxBlockChW`, `Maps.cnxDownChW`.

✅ **The attention set landed 2026-09-03, in `FloatBudgetEnvAttn.lean`**: `Maps.softmaxRow`
(⭐ window `1 + smCap`, CONSTANT in the input — a softmax RESETS — so `capped` gives modulus
`2(1+smCap)` and `smErr`'s `Real.exp` disappears rather than being bounded) and
`Maps.mhProjAttnFullCap` (⭐⭐ attention as ONE leaf at an exp-free window — `FloatBridgesTo`
composes single-input maps and attention fans out, so it CANNOT be decomposed into the graph's
tokens; §3.5). ⛔ **NOT `floatBridges_mhProjAttnFull`** — its `Real.exp` is in the *window*, where
`capped` cannot reach it (`capped` replaces `mod`, never `mag`), and at `δ = 3.6·10¹⁰` it has no
rational bound at all.

✅ **The patch embed landed too** — `Maps.patchEmbed`, one leaf. ⛔ NOT `Maps.concatCls` +
`Maps.flatConvStride16`: `patchEmbed_flat` is a single definition with an `if n.val = 0` branch,
so that decomposition describes a function the repo does not contain (§3.5.2 item 4).

✅ **The forward kit is closed**, and ✅ **`FloatBudgetEnvBack.lean` is the backward's**:
`Maps.bnPerChannelBack` and ⭐ `Maps.bnPerChannelBackGain` (the per-unit-gain form every whole-net
backward chain must use — §3.7 step 4), `Maps.convBack` / `.linBack` / `.gapBack` /
`.reluMaskBack` / `.maxPoolBack` / ⛔ `.maxPool3s2Back` / `.decimateBack` / `.flatConvStride2Back`,
both r34 block backwards, the fourteen homogeneity lemmas, and `bnXhat_abs_le_num`.
⛔ **`Maps.maxPool3s2Back` is the one backward leaf that is NOT envelope-preserving** — `Ā ↦ 4Ā`,
because He et al.'s 3×3/s2 windows overlap and the backward accumulates (§3.10). Its 2×2 sibling
`Maps.maxPoolBack` carries `Ā ↦ Ā` and using it for the 3×3 pool is exactly the drift §3.10
records; the two have the same shape and are different functions. ✅ **And `FloatBudgetEnvBackMBConv.lean` is the SECOND backward net's** (2026-09-04, §3.13):
`Maps.depthwiseBack` (`Maps.depthwise` at the reversed kernel, ⭐ fan-in `kH·kW` alone),
`Maps.depthwiseStride2Back` (that composed with `Maps.decimateBack`) and the two inverted-residual
body envelopes. §3.9's costing was exact — nothing else was needed, and `Maps.reluMaskBack`
already covered the relu6 kink. ⭐ `bnIstd_abs_le_of` joined `FloatBudgetEnvBack.lean` with them:
the rational `|istd| ≤ S` from the ε-floor, which is what lets a backward be stated without an
operating point. EfficientNet-B0 needs `Maps.diagBack`, `Maps.broadcastBack` and the
`Maps.seBack` composite, and ⛔ **a `Maps` leaf was never what blocked it** — a global `|swish′|`
bound was, and ✅ that landed 2026-09-04 (§3.12), so B0's backward budget file is now only
`Maps` leaves away. All the `∃`-tier bridges exist (`DepthwiseBackFloatBridge.lean`,
`SEBackFloatBridge.lean`, `LinBackFloatBridge.lean`).
Each is ten lines in the `Maps.flatConv` mould: `show` the unfolded `mag`/`mod`, one monotone
lemma, `linarith`. Write one only when a net in §3 needs it.

✅ **`Cifar8FloatBudget.lean` is on `Maps` too (2026-09-04, §3.11), and `FloatBridgesTo.Env` is
retired** — deleted with its five `Env.comp_*` lemmas, its private copies of the two monotone
lemmas, and one dead `private theorem`. Every whole-net budget in the repo now runs on one kit.

## 7. Process (every commit)

1. `lake build Proofs Certs` (bare `lake build` skips the Certs corpus — memory note).
2. `lake env lean tests/AuditAxioms.lean` — exit 0, every new declaration on
   `[propext, Classical.choice, Quot.sound]`, no `sorryAx`.
3. `lake exe docstring-checkrefs`; `python3 scripts/check_audit_coverage.py`.
4. New files: lakefile `Proofs` root + `AuditAxioms` import + `#print axioms`.
5. `formalization.yaml` §4d: append the net's number with the size caveat; a `declarations:`
   entry for `<net>_float_logits_le`.
6. Stage, then STOP and ask before committing. One commit per net.

## 8. Pitfalls

* `Maps`/`Env` as a `def` → unifier unfolds the whole chain → heartbeat timeout even at 20×.
  Structure, always.
* One big `exact` against the unfolded net → same. Bottom-up `have`s at block granularity.
* Implicit `{A}` on the base step → metavariable → later tactics never run. `(A := 1) (Ē := 0)`.
* Wrapper def with inferred dims → `Mat (c4 * ?h * ?w) =?= Mat 128` → timeout. Pass all dims.
* `have` inside a term-mode def → `letFun` → the final defeq has to zeta through it.
* The generator must fold with the ROUNDED γ (§0), and its re-assertion pass must run before it
  emits.
* Lean identifiers: `Ā` and `Ē` are single codepoints and legal; `B̄` and `P̄` are a letter plus
  a COMBINING macron and are not. Use `Bd`, `Pd`.
* Lint: an unused bound `A` in `fun A hA => …` warns; write `_A`.
* ⛔ `norm_num`'s ceiling is ~10²⁵³ for a nested arithmetic tree, NOT 10³⁰⁰, and it depends on
  the operation tree rather than the value — the same number closes in a flatter shape (§3.7(a)).
  Nothing moves it: heartbeats, recursion depth, `ring_nf`, `nlinarith`, `simp only` first.
* Pin ALL FOUR numerals on every `Maps` leaf, not just the input window: a `by norm_num` inside a
  `have` runs before `Maps.comp` unifies, so an unpinned output window is a metavariable and the
  error reads like arithmetic (§3.7(c)).
* ⭐⭐ An op declared over a COMPUTED dimension — `maxPoolFlatBack`'s `x : Tensor3 c (2*h) (2*w)`
  — makes any composition containing it a higher-order unification (`2 * ?h = 112`), which
  presents as a non-terminating whole-net `isDefEq`, not as a missing argument. Pin those
  implicits: `(c := 64) (h := 56) (w := 56)` (§3.7(d)).
* Diagnose a non-terminating whole-net `isDefEq` by probing compositions of GROWING DEPTH with
  explicit type ascriptions, from both ends — it takes minutes and lands on the exact argument.
* Time a whole-net `Maps` chain SEPARATELY from its closing step. A truncation that ends in
  `sorry` never performs the final unification, so a blowup there looks like a per-block cost.
* A `have` cannot carry a `?_`. A whole-net envelope whose middle stage is the open goal has to
  be two `refine`s (build the stages that do NOT depend on it first — ViT's head and final LN
  do not depend on its body).
* `layerBudget_le_of` in `FloatBridge.lean` is `private`; `layerBudget_le_num'` in
  `FloatBudgetEnv.lean` is the public monotone form — do not touch `FloatBridge.lean`
  (2000-module rebuild).

## 9. What to say about the numbers

⛔ **And say when the number is the CAP rather than the fold.** If §3.3.0(a)'s `min(mod, 2·mag)`
selects its right branch anywhere on the chain, the statement has changed from "the rounding
error folds to this" to "both maps land in the certified window" — the triangle inequality. On
ConvNeXt-T it selects the right branch at every one of the 23 LN sites and the budget comes out
at exactly **`2.00 ×`** the window, which is the tell. That number must not be tabled beside
r34/mnv2/B0's without the label — the §0 table carries a `kind` column for exactly this.
⭐ And say WHY, because the reason is the interesting part: LayerNorm reduces its statistics out
of its own input, BatchNorm escapes that by freezing them, and LayerNorm has nothing to freeze.
The cap is not a shortcut past a fold that exists; it is what makes a statement possible where no
fold does.
⛔ **ViT-Tiny IS the same kind of number, and more thoroughly so than ConvNeXt (landed
2026-09-03).** Every one of its 25 LayerNorm sites is capped, and so is every one of its 12
attention sites — so there is no stage in a ViT block at which the fold survives, and the two
skips per block carry that forward. ⭐ Its ONE honest stage is the patch embed, which does not
reduce. Say that: "the float and the real forward both land in the certified window", never "the
rounding error folds to this"; `budget / window = 2.00` is the tell, and it came out at 2.00.
⭐⭐ And say WHY ITS SECOND CAP IS THERE, because it is not ConvNeXt's reason: LayerNorm is
capped for SIZE (uncapped, 10³²³⁹), attention for REPRESENTABILITY — its uncapped window carries
a `Real.exp` at an argument with no rational bound, so 36 stage numerals cannot be written at
all, at a magnitude *smaller* than the shipped one. "Too big" and "not writable" are different
failures and the second is invisible to a Python fold.

⭐⭐ **AND SAY WHAT A FOLD COMPOSED AGAINST A CAPPED FORWARD IS** — the ConvNeXt backward's row,
new 2026-09-04 (§3.16 finding 1). r34's and mnv2's backward numbers hypothesise `es`/`exh` = 10⁻² on
the saved float activations, which their own training-mode forward fold cannot discharge (10⁷⁴¹⁷) —
a quantitative gap, and both nets have an inference mode where the forward IS a fold. ConvNeXt has
neither: its forward statement is `capped`, whose modulus is `2·mag` by construction, so the
activation accuracy it supplies is `2 × window ≈ 10²²⁷` and nothing moves it below that; and there
is no eval-mode LayerNorm to switch to. So the number is *an honest fold of the backward kernel's
rounding, at a hypothesised operating point, given saved-activation accuracies this net's forward
cannot supply in any mode* — never "the deployed ConvNeXt gradient is within this of the certified
one end to end". The fold is real and the composition does not exist; say both halves.

⭐ Say the WINDOW and the BUDGET separately — after MobileNetV2 they are not the same story.
A clamped-activation net can have a tight window and a vacuous budget at the same time, and
collapsing them into "the number" loses the only half that is currently believable.

Per net: the budget is the interval fold at the stated magnitudes; it is vacuous as a
certificate; it agrees in order with nothing tighter than itself and sits far above the adjoint
chain's figure for the same net, for two documented reasons; relative to the certified output
window it is ≈ Σ(mᵢ+2)·u. What is new is that a wrong `layerBudget`, a dropped stage, a misread
fan-in, a mis-plugged block — or, after ConvNeXt, a stale layer slot — now fails to compile.
⭐ Say the PROFILE too, and say it per parameter kind if the checkpoint has more than one scale:
ConvNeXt-T's conv kernels max at `0.60` and its layer scale at `8.38`, and quoting a single
"|param| ≤ 8.4" would be both true and 74 orders misleading. The blueprint's "adjoint chain" paragraph
carries the measured-magnitude caveat; do not quote the fold's numbers there as if they were the
chain's. And say which BN mode the number is at — after §0.1 that is not a footnote.
