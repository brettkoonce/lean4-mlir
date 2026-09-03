# Whole-net float budgets as NUMBERS: ResNet-34 and MobileNetV2 landed, and the wall the rest hit

Written 2026-09-02; revised 2026-09-03 after MobileNetV2. Read §0 and §0.1 before touching
anything; §3 is the work list; §7 is the checklist you run before every commit.

## 0. Where we are

`FloatBridgesTo f fF` is a Type-valued bridge (`FloatComposeBridge.lean`): `mag : ℝ → ℝ`
(input window ↦ output window), `mod : ℝ → ℝ → ℝ` (input window ↦ error modulus), and
`close : ∀ A, 0 ≤ A → 0 ≤ mag A ∧ FloatClose A (mag A) f fF (mod A)`. Every combinator composes
`mag`/`mod` explicitly, every leaf writes them out. (Why it had to be data and not `∃ L`:
`formalization.yaml` fidelity §4d — the `∃`-modulus was discharged by `L := 2B`.)

Four whole nets now carry kernel-checked numbers:

| net | window | budget | budget/window | over the chain | file |
|---|---|---|---|---|---|
| CIFAR-8 (8 conv, no BN) | 6.121·10¹⁸ | 6.37·10¹⁴ | 1·10⁻⁴ | — | `Cifar8FloatBudget.lean` |
| ResNet-34 @224², **inference BN** | 3.152·10²¹¹ | 1.548·10²⁰⁹ | 4.9·10⁻³ | ~10¹⁵⁷ | `Resnet34FloatBudget.lean` |
| MobileNetV2 @224², **inference BN** | **2.154·10³** | 1.444·10⁹⁶ | — | ~10³⁶ | `MobileNetV2FloatBudget.lean` |
| EfficientNet-B0 @224², **inference BN**, batched | 2.580·10⁵⁵ | 8.408·10²¹⁰ | — | — | `EfficientNetFloatBudget.lean` |

All four are the interval fold and all four are vacuous as *budgets*; the point is that the
kernel checks them. The r34 number sits 157 orders above the adjoint chain's proven-H figure for
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
* `scripts/float_budget_envelope.py` — the exact-rational fold in the lemmas' semantics, the
  4-significant-figure round-up, the re-assertion passes (`verify_r34`, 180 inequalities;
  `verify_mnv2`, 116; `verify_b0`, 96) and the numerals. Its CIFAR-8 regression case reproduces `Cifar8FloatBudget.lean` stage for stage.
  ⚠ It must fold with the **rounded** γ (`r4(gamma_q k)`), not the exact `(1+u)^k − 1` — the
  Lean chain passes the rounded one, and folding with the exact value silently emits stage
  numerals a hair too small, which the kernel then rejects. That is what the re-assertion pass
  is for; it caught exactly this.

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
about the *closure* and wrong about the *number*. Do not plan a ConvNeXt or ViT number until
one of the following is done:

1. **A cap.** Any modulus can be replaced by `2·mag` (both float and real outputs lie in the
   certified window), so `Maps` could carry `Ē' := min(fold, 2Ā')`. Cheap, and it stops the
   squaring cold. But it is the triangle inequality, not the fold, and it should be labelled as
   such wherever it is used.
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
`AuditAxioms`, disclosed in `formalization.yaml` §4d.

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
| ConvNeXt-T Ch fwd | open: 22 pure-normalise LN bridges | ⛔ blocked on §0.1 — the LN modulus is quadratic and LN has no eval mode |
| ViT-Tiny fwd | open: `hFinalLN`, `hblocks`, `hPatch` | ⛔ blocked on §0.1, and additionally the softmax modulus carries `Real.exp` |

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

### 3.3, 3.5 ConvNeXt-T and ViT-Tiny

Both blocked on §0.1; if one of the escapes there works, ConvNeXt is the cheapest (one LN leaf
closes 22 hypotheses) and ViT is still gated on `Real.exp` in the softmax modulus.

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

Still to add:

`comp_flatConvStride4`,
`comp_gelu` (rational slack for `√(2/π)`), `comp_biasAdd`,
`comp_diagBack`, `comp_layerNormVec`, identity steps for `gather`/`transposeFlat`/`reassoc*`/
`broadcast`/`clsSlice`, and `seScale` / `perRow` / `batchMap` / `iterate k`. Each is ten lines
in the `Maps.flatConv` mould: `show` the unfolded `mag`/`mod`, one monotone lemma, `linarith`.
Write one only when a net in §3 needs it.

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

⭐ Say the WINDOW and the BUDGET separately — after MobileNetV2 they are not the same story.
A clamped-activation net can have a tight window and a vacuous budget at the same time, and
collapsing them into "the number" loses the only half that is currently believable.

Per net: the budget is the interval fold at the stated magnitudes; it is vacuous as a
certificate; it agrees in order with nothing tighter than itself and sits far above the adjoint
chain's figure for the same net, for two documented reasons; relative to the certified output
window it is ≈ Σ(mᵢ+2)·u. What is new is that a wrong `layerBudget`, a dropped stage, a misread
fan-in or a mis-plugged block now fails to compile. The blueprint's "adjoint chain" paragraph
carries the measured-magnitude caveat; do not quote the fold's numbers there as if they were the
chain's. And say which BN mode the number is at — after §0.1 that is not a footnote.
