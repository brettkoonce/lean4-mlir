# Whole-net float budgets as NUMBERS: ResNet-34 and MobileNetV2 landed, and the wall the rest hit

Written 2026-09-02; revised 2026-09-03 after MobileNetV2. Read §0 and §0.1 before touching
anything; §3 is the work list; §7 is the checklist you run before every commit.

## 0. Where we are

`FloatBridgesTo f fF` is a Type-valued bridge (`FloatComposeBridge.lean`): `mag : ℝ → ℝ`
(input window ↦ output window), `mod : ℝ → ℝ → ℝ` (input window ↦ error modulus), and
`close : ∀ A, 0 ≤ A → 0 ≤ mag A ∧ FloatClose A (mag A) f fF (mod A)`. Every combinator composes
`mag`/`mod` explicitly, every leaf writes them out. (Why it had to be data and not `∃ L`:
`formalization.yaml` fidelity §4d — the `∃`-modulus was discharged by `L := 2B`.)

Three whole nets now carry kernel-checked numbers:

| net | window | budget | budget/window | over the chain | file |
|---|---|---|---|---|---|
| CIFAR-8 (8 conv, no BN) | 6.121·10¹⁸ | 6.37·10¹⁴ | 1·10⁻⁴ | — | `Cifar8FloatBudget.lean` |
| ResNet-34 @224², **inference BN** | 3.152·10²¹¹ | 1.548·10²⁰⁹ | 4.9·10⁻³ | ~10¹⁵⁷ | `Resnet34FloatBudget.lean` |
| MobileNetV2 @224², **inference BN** | **2.154·10³** | 1.444·10⁹⁶ | — | ~10³⁶ | `MobileNetV2FloatBudget.lean` |

All three are the interval fold and all three are vacuous as *budgets*; the point is that the
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
* `scripts/float_budget_envelope.py` — the exact-rational fold in the lemmas' semantics, the
  4-significant-figure round-up, the re-assertion passes (`verify_r34`, 180 inequalities;
  `verify_mnv2`, 116) and the numerals. Its CIFAR-8 regression case reproduces `Cifar8FloatBudget.lean` stage for stage.
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
| EfficientNet-B0 fwd | open: 10 batched `bnBatchLA` bridges | an inference-BN leaf at the batched index (`batchMap` of the per-example eval BN — `FloatBridgesTo.batchMap` handles the lift) |
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


### 3.3–3.5 ConvNeXt-T, EfficientNet-B0, ViT-Tiny

**B0 is NEXT.** It needs a batched inference-BN leaf (`batchMap` of the per-example eval BN) and
is otherwise the r34 recipe. ⚠ Do not expect mnv2's tight window there: B0's activation is
swish, which is unbounded above, so §3.2's clamp does not apply and the window will compound the
way ResNet-34's does. The budget will look like the others (`G·S` per BN site). ConvNeXt-T and
ViT-Tiny are blocked on §0.1; if one of the three escapes there works, ConvNeXt is the cheapest
(one LN leaf closes 22 hypotheses) and ViT is still gated on `Real.exp` in the softmax
modulus.

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

✅ `relu6` (window `min Ā 6`, NOT a copy of `Maps.relu` — §3.2), `depthwise` and
`depthwiseStride2Flat` all landed with MobileNetV2 and live in `MobileNetV2FloatBudget.lean`
(`FloatBudgetEnv.lean` cannot see their leaves — `floatBridgesTo_depthwise` is in
`DepthwiseFloatBridge.lean` and `floatBridgesTo_relu6` in `MobileNetV2WholeFloatBridge.lean`, and
neither is on `FloatBudgetEnv`'s import path. Move them down only if a third net needs them).
Still to add:

`comp_flatConvStride4`,
`comp_gelu` (rational slack for `√(2/π)`), `comp_swish`, `comp_sigmoid`, `comp_biasAdd`,
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
