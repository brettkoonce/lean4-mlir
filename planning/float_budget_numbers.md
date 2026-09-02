# Whole-net float budgets as NUMBERS: ResNet-34 landed, and the wall the rest hit

Written 2026-09-02 (revised the same day, after ResNet-34). Read §0 and §0.1 before touching
anything; §3 is the work list; §7 is the checklist you run before every commit.

## 0. Where we are

`FloatBridgesTo f fF` is a Type-valued bridge (`FloatComposeBridge.lean`): `mag : ℝ → ℝ`
(input window ↦ output window), `mod : ℝ → ℝ → ℝ` (input window ↦ error modulus), and
`close : ∀ A, 0 ≤ A → 0 ≤ mag A ∧ FloatClose A (mag A) f fF (mod A)`. Every combinator composes
`mag`/`mod` explicitly, every leaf writes them out. (Why it had to be data and not `∃ L`:
`formalization.yaml` fidelity §4d — the `∃`-modulus was discharged by `L := 2B`.)

Two whole nets now carry kernel-checked numbers:

| net | window | budget | budget/window | file |
|---|---|---|---|---|
| CIFAR-8 (8 conv, no BN) | 6.121·10¹⁸ | 6.37·10¹⁴ | 1·10⁻⁴ | `Cifar8FloatBudget.lean` |
| ResNet-34 @224², **inference BN** | 3.152·10²¹¹ | 1.548·10²⁰⁹ | 4.9·10⁻³ | `Resnet34FloatBudget.lean` |

Both are the interval fold and both are vacuous as certificates; the point is that the kernel
checks them. The r34 number sits 157 orders above the adjoint chain's proven-H figure for the
same net (6.5·10⁵¹, `scripts/adjoint_chain_probe.py` §5), and the two documented reasons are
`layerBudget`'s uniform `m·w'·A` face (§5 of the probe measures 257× per stage) and worst-case
rather than measured windows.

The machinery built for r34 and reusable for the rest:

* `FloatBudgetEnv.lean` — `FloatBridgesTo.Maps Ā Ē Ā' Ē'`, *"at every input window `A ≤ Ā` and
  every inherited error `E ≤ Ē`, the output window is `≤ Ā'` and the output error `≤ Ē'`."*
  Quantifying over the inputs rather than fixing them buys monotonicity, and monotonicity makes
  `Maps.comp` / `Maps.residual` / `Maps.biPathSum` **generic** — the CIFAR-8 `Env` needed one
  `comp_*` lemma per operation and could not express a skip at all. Leaves so far: `relu`,
  `maxPool`, `maxPool3s2`, `flatConv`, `flatConvStride2`, `dense`, `gap`,
  `bnPerChannelTensor3` (training), `bnEvalPC` (inference).
* `BnEvalRuntimeFloatBridge.lean` — inference BN as the render actually emits it.
* `Resnet34WholeFloatBridge.lean` — the block bridges are now generic in the normalisation
  (`rblkGen` / `rblkStridedGen`, with `rblkPC_eq_gen` / `rblkPStridedPC_eq_gen` both `rfl`), so
  one pair of block bridges serves the training net and the inference net.
* `scripts/float_budget_envelope.py` — the exact-rational fold in the lemmas' semantics, the
  4-significant-figure round-up, the re-assertion pass (`verify_r34`, 180 inequalities) and the
  numerals. Its CIFAR-8 regression case reproduces `Cifar8FloatBudget.lean` stage for stage.
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
| **ResNet-34 fwd** | ✅ **DONE** (inference BN) — `r34_float_logits_le`, 1.548·10²⁰⁹ | the eval-graph faithfulness tie (§3.1) |
| MobileNetV2 fwd | open: `hbnS hbnH` + `b1..b6` | port the blocks to `To`-defs at the INFERENCE BN, then the r34 recipe verbatim |
| EfficientNet-B0 fwd | open: 10 batched `bnBatchLA` bridges | an inference-BN leaf at the batched index (`batchMap` of the per-example eval BN — `FloatBridgesTo.batchMap` handles the lift) |
| ConvNeXt-T Ch fwd | open: 22 pure-normalise LN bridges | ⛔ blocked on §0.1 — the LN modulus is quadratic and LN has no eval mode |
| ViT-Tiny fwd | open: `hFinalLN`, `hblocks`, `hPatch` | ⛔ blocked on §0.1, and additionally the softmax modulus carries `Real.exp` |

### 3.1 ResNet-34 — what landed, and the one thing left

`r34EvalBridge` is the `r34Forward` skeleton with `bnPerChannelEvalTensor3` at all 33 BN sites.
That op is what `SHlo.bnPerChannelEvalF` denotes (`bnPerChannelEvalF_faithful`, `rfl`), so it is
the ℝ semantics of every BN line of the committed `@resnet34_fwd_eval` render — but the
**whole-graph** faithfulness theorem for the eval chain does not exist. The training chain has
one (`resnet34FwdGraphFullPC_faithful`, `ResNet34RenderPC.lean`). Building the eval twin is the
remaining tie work: `idBlockGraphPCEval` / `downBlockGraphPCEval` + `_faithful`, then
`resnet34FwdGraphFullPCEval` + `_faithful`, mirroring the training pair one for one. Until then
the disclosure in `formalization.yaml` §4d states the tie honestly as skeleton-level.

⚠ **`BnEvalFloatBridge.lean`'s premise is wrong and is now corrected in place.** It models
deployed BN as a pre-folded affine `a·x + b` and says *"there is no batch reduction and no
runtime `rsqrt`"*. The first half is the point; the second is not what this repo emits —
`.bnPerChannelEvalF` expands to `subtract`, `add`, `rsqrt`, `multiply`, `multiply`, `add`
(`StableHLO.lean`), i.e. the rsqrt is on device. `BnEvalRuntimeFloatBridge.lean` bridges the
kernel we ship; `floatClose_bnEval`'s affine remains true and remains about a program we do not
run. Do not build the other nets' numbers on the affine.

### 3.2 MobileNetV2 — next, and it should be quick

Same recipe. The legacy per-block chain exists (`floatBridges_ivExpandPC` →
`_ivDepthwiseStridedPC` → `_ivProjectPC`); port each to a `To`-def generic in the normalisation
the way `rblkGen` is, plug in `floatBridgesTo_bnPerChannelEvalTensor3`, and add
`Maps.comp_depthwise` / `Maps.comp_relu6` / `Maps.residual` (the last already exists). Tie:
`mobilenetv2Forward_full_pc_eq_skeleton` is the training tie, so the same eval caveat applies.
⚠ The blueprint's `2.7·10⁶⁰` for this net is the probe's EVAL-BN adjoint-chain figure, not the
interval fold — expect the fold to land far above it, as r34's did.

### 3.3–3.5 ConvNeXt-T, EfficientNet-B0, ViT-Tiny

B0 needs a batched inference-BN leaf and is otherwise the r34 recipe. ConvNeXt-T and ViT-Tiny
are blocked on §0.1; if one of the three escapes there works, ConvNeXt is the cheapest (one LN
leaf closes 22 hypotheses) and ViT is still gated on `Real.exp` in the softmax modulus.

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

`comp_flatConvStride4`, `comp_depthwise`, `comp_depthwiseStride2Flat`, `comp_relu6`,
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

Per net: the number is the interval fold at the stated magnitudes; it is vacuous as a
certificate; it agrees in order with nothing tighter than itself and sits far above the adjoint
chain's figure for the same net, for two documented reasons; relative to the certified output
window it is ≈ Σ(mᵢ+2)·u. What is new is that a wrong `layerBudget`, a dropped stage, a misread
fan-in or a mis-plugged block now fails to compile. The blueprint's "adjoint chain" paragraph
carries the measured-magnitude caveat; do not quote the fold's numbers there as if they were the
chain's. And say which BN mode the number is at — after §0.1 that is not a footnote.
