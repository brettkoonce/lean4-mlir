# Whole-net float budgets as NUMBERS: five nets landed; ViT-Tiny is PROBED and reachable

Written 2026-09-02; revised 2026-09-03 after ResNet-34, MobileNetV2, EfficientNet-B0 and
ConvNeXt-T, and again the same day after ViT-Tiny's sizing probe.

**Picking this up cold?** Read §0.1 (the one structural finding) and §9 (⛔ what the ConvNeXt
number is and is not), then §3.5 (ViT-Tiny — probed 2026-09-03, reachable at 10¹⁹¹, no Lean
written yet) and §3.5.2 (the order of work, which starts with a tier migration nobody had
costed). §7 is the checklist you run before every commit — including the "stage, then STOP and
ask" rule.

⭐ The single most useful habit from the five nets that landed: **when a fold overshoots, ablate
before concluding anything about the architecture.** Four times running the blocker was
something already true that the statement threw away — relu6's clamp, swish's modulus, seScale's
window, and then ConvNeXt's **profile** (a uniform parameter bound where the checkpoint has four
scales 14× apart). Each diagnosis was a Python probe, not a Lean session, and the last of them
was not a leaf lemma at all — which is the refinement to carry: **ablate the inputs too.**

## 0. Where we are

`FloatBridgesTo f fF` is a Type-valued bridge (`FloatComposeBridge.lean`): `mag : ℝ → ℝ`
(input window ↦ output window), `mod : ℝ → ℝ → ℝ` (input window ↦ error modulus), and
`close : ∀ A, 0 ≤ A → 0 ≤ mag A ∧ FloatClose A (mag A) f fF (mod A)`. Every combinator composes
`mag`/`mod` explicitly, every leaf writes them out. (Why it had to be data and not `∃ L`:
`formalization.yaml` fidelity §4d — the `∃`-modulus was discharged by `L := 2B`.)

Five whole nets now carry kernel-checked numbers — ⛔ **but not all five are the same claim.**

| net | window | budget | budget/window | kind | file |
|---|---|---|---|---|---|
| CIFAR-8 (8 conv, no BN) | 6.121·10¹⁸ | 6.37·10¹⁴ | 1·10⁻⁴ | fold | `Cifar8FloatBudget.lean` |
| ResNet-34 @224², **inference BN** | 3.152·10²¹¹ | 1.548·10²⁰⁹ | 4.9·10⁻³ | fold | `Resnet34FloatBudget.lean` |
| MobileNetV2 @224², **inference BN** | **2.154·10³** | 1.444·10⁹⁶ | — | fold | `MobileNetV2FloatBudget.lean` |
| EfficientNet-B0 @224², **inference BN**, batched | 2.580·10⁵⁵ | 8.408·10²¹⁰ | — | fold | `EfficientNetFloatBudget.lean` |
| ConvNeXt-T @224², channel LN | 4.858·10²²⁷ | 9.706·10²²⁷ | **2.00** | ⛔ **cap** | `ConvNeXtFloatBudget.lean` |

The first four are the interval fold and all four are vacuous as *budgets*; the point is that the
kernel checks them. **ConvNeXt-T's is the triangle inequality** — its 23 LayerNorm sites all go
through `FloatBridgesTo.capped`, so it says "the float and the real forward both land in the
certified window" and not "the rounding error folds to this". `budget/window = 2.00` is the tell,
and §9 is the rule for saying so. There is no version of ConvNeXt for which the fold exists
(§0.1), so this is not a weaker choice; it is the only statement available.

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
* `FloatBudgetEnvLN.lean` — the `Maps` kit a LayerNorm net needs: the capped pure-normalise LN
  (`Maps.bnCapped`), the two halves of its affine (`diagBack`/`biasAdd`), the gathers and the
  per-row lift they are conjugated by, `gelu` (through the `3/2` branch), `flatConvStride4`, and
  the three composites the whole-net chain walks (`Maps.chanLNTensor3` / `.cnxBlockChW` /
  `.cnxDownChW`). ⚠ It imports `FloatBudgetEnvMBConv` for one leaf, `Maps.depthwise`.
* `Architectures/GeluSaturation.lean` — `geluScalarDeriv_abs_le` / `geluScalar_lipschitz`, moved
  down out of `Certificates/GeluLipschitz.lean` so `floatClose_gelu` can state the `min` of its
  magnitude polynomial and the global `3/2` (§3.3.0(b)).
* `scripts/float_budget_envelope.py` — the exact-rational fold in the lemmas' semantics, the
  4-significant-figure round-up, the re-assertion passes (`verify_r34`, 180 inequalities;
  `verify_mnv2`, 116; `verify_b0`, 96; `verify_cnx`, 366) and the numerals. Its CIFAR-8
  regression case reproduces `Cifar8FloatBudget.lean` stage for stage, and `cnx_eval_chain`'s
  three flags (`ln_cap`, `gelu_sat`, `head_ln`) reproduce §3.3's ablation table.
  ⭐ `vit_chain` (added 2026-09-03) is the ViT-Tiny sizing fold — 236 stages, five flags, and
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

## 1. Goal, non-goals, success

**Goal.** For each committed ImageNet-scale forward, a closed `FloatBridgesTo` over the real
leaves and a theorem `<net>_float_logits_le` stating a number. Backwards are phase 2 (§3.7).

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
| ViT-Tiny fwd (`vitForwardKV`) | ⭐ **PROBED + LEAVES LANDED 2026-09-03**, window 3.612·10²¹⁸ / budget 7.222·10²¹⁸ (§3.5), the cap again. `FloatBudgetEnvAttn.lean` closes the attention leaves; the whole net is not written. | ⛔ the `FloatBridgesTo` TIER MIGRATION for the LN / MLP / patch-embed cone — still all `FloatBridges`, so `vit_floatBridgesTo`'s hypotheses stay undischargeable (§3.5.1); then `Maps.concatCls` / `flatConvStride16` and `ViTFloatBudget.lean` (§3.5.2) |

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
   are incomparable, so `floatClose_swish` now states their `min`. ⚠ The true global Lipschitz
   constant of `x·σ(x)` is ≈ 1.1 (`σ' → 0` at both ends); getting *that* needs the decay of `σ'`,
   i.e. calculus, and the additive bound is what avoids needing it.
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

### 3.5 ViT-Tiny — ⭐ PROBED and the leaves LANDED 2026-09-03: 3.612·10²¹⁸ / 7.222·10²¹⁸

**The probe says yes, and chunk 1 of the Lean is in** (`FloatBudgetEnvAttn.lean`). `vit_chain`
(`scripts/float_budget_envelope.py`) folds all 164 stages of `vitForwardKV` at ViT-Tiny's shapes
and lands at

    window ≤ 3.612·10²¹⁸      budget ≤ 7.222·10²¹⁸      budget/window = 1.999
    328 re-assertions (`verify_vit`)

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
MIGRATION.** §3.5 read the gap as "the patch embed, the attention block and the MLP block each
need a closed `FloatBridgesTo` at real weights". The actual state is that **the ViT float cone is
`FloatBridges` (the ∃-existential tier) essentially everywhere**: `ViTBlockFloatBridge.lean`
(1116 lines), `ViTAttentionFloatBridge.lean` (439) and `ViTFloatBridge.lean` (233) contain
**zero** `FloatBridgesTo` definitions between them apart from `FloatBridgesTo.perRow` and
`floatBridgesTo_gather`. Only `ViTWholeFloatBridge.lean`'s top-level skeleton was migrated — so
`vit_floatBridgesTo` exists and **not one of its hypotheses can currently be discharged**.
⚠ That is the `floatbridgesto_existential_L_vacuous` migration, unfinished on this net, and it is
the bulk of the remaining work. Budget for it explicitly; it is not visible from the whole-net
file, which looks finished.

### 3.5.2 The order of work — ✅ steps 1–2 LANDED 2026-09-03 (`FloatBudgetEnvAttn.lean`)

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
3. **The `*Gen` layer** for `blockV`/`vitBodyKVFlat`, if the eval/training split needs it — ⭐ ViT
   needs NO eval twin and none is possible (LN has no running statistics), so unlike r34/mnv2/B0
   there is no second render and no `*RenderPCEval.lean`. That made ConvNeXt cheaper and it makes
   ViT cheaper the same way.
4. **`ViTFloatBudget.lean`** on the r34 recipe: records (`ViTBlockW`/`ViTProfile`/…), a
   `DeviceExp` alongside `DeviceRsqrt`, per-block `Maps` lemma, the 236-stage chain, the tie
   through `vitFwdGraphKMHV_faithful`, `vit_float_logits_le` + `_committed`.
5. **`verify_vit`** — the re-assertion pass, before any numeral is emitted (§0's ⚠: fold with the
   ROUNDED γ; the pass is what catches it).

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

### 3.7 Phase 2 — backwards

Same pattern on the `_grad_floatBridgesTo` peers. ⚠ §0.1 applies with more force: a backward's
BN-back modulus inherits the same reduction structure. Start only after the forwards.

## 5. The `Maps` kit still to add

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

Still to add for ViT: `Maps.concatCls` and `Maps.flatConvStride16` (the patch embed), and the
`FloatBridgesTo` peers for the LN / MLP / patch-embed cone (§3.5.1 — the tier migration, and the
bulk of what is left). Then, if a net needs them, identity steps for `clsSlice` and `iterate k`.
Each is ten lines in the `Maps.flatConv` mould: `show` the unfolded `mag`/`mod`, one monotone
lemma, `linarith`. Write one only when a net in §3 needs it.

⚠ `Cifar8FloatBudget.lean` still runs on the older `FloatBridgesTo.Env` (fixed input error,
per-op `comp_*` lemmas). Migrating it to `Maps` is a coherence pass, not a correctness one — the
per-stage inequalities are identical and the two shared monotone lemmas (`layerAct_le_num'`,
`layerBudget_le_num'`) already live in `FloatBudgetEnv.lean`. Do it when touching that file
anyway, not as its own commit.

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
