# Whole-net float budgets

**Status, 2026-09-05: the research thread is closed.** Seven forwards and four input-gradient
backwards carry kernel-checked numbers; three backward chains are proven to be the certified
whole-net gradient. What remains is three closing items (section 5) and then a freeze. The
session-by-session record, with the ablations and the corrections, is
`planning/archive/float_budget_numbers_log.md`; its section numbers are what the Lean docstrings
cite (`§0.1`, `§3.16`, `§9`, ...) and are preserved there.

This document is the standing reference: what exists, what it certifies, what it does not, and
what to do if one of these files is opened again.

## 1. What exists

Every row is a theorem in `LeanMlir/Proofs/Float/`, closed over the real leaves (no
`FloatBridgesTo` hypothesis left), tied to the committed net definition and, for the forwards,
to the rendered graph. Every numeral was produced by `scripts/float_budget_envelope.py` in the
leaves' own exact-rational arithmetic, re-asserted by that script's `verify_*` pass, and then
checked again by the kernel. All are 3-axiom clean and listed in `tests/AuditAxioms.lean` and
`formalization.yaml` (status.main_results, fidelity 4d).

| net | mode and qualifiers | window | budget | kind | theorem | file |
|---|---|---|---|---|---|---|
| CIFAR-8 forward | 8 conv, no normalisation | 6.121e18 | 6.37e14 | fold | `cifar8_float_logits_le` | `Cifar8FloatBudget.lean` |
| ResNet-34 forward | inference BN | 3.152e211 | 1.548e209 | fold | `r34_float_logits_le` | `Resnet34FloatBudget.lean` |
| ResNet-34 forward | training BN, per example | 8.748e80 | 1.752e81 | cap | `r34_train_float_logits_le` | `Resnet34TrainFloatBudget.lean` |
| MobileNetV2 forward | inference BN | 2.154e3 | 1.444e96 | fold | `mnv2_float_logits_le` | `MobileNetV2FloatBudget.lean` |
| EfficientNet-B0 forward | inference BN, any batch size | 2.580e55 | 8.408e210 | fold | `b0_float_logits_le` | `EfficientNetFloatBudget.lean` |
| ConvNeXt-T forward | channel LayerNorm | 4.871e130 | 9.738e130 | cap | `cnx_float_logits_le` | `ConvNeXtFloatBudget.lean` |
| ViT-Tiny forward | vector LayerNorm, depth 12 | 2.397e108 | 4.794e108 | cap | `vit_float_logits_le` | `ViTFloatBudget.lean` |
| ResNet-34 backward | training BN, `\|istd\| <= 16` | 8.857e245 | 6.894e244 | fold | `r34_grad_float_le` | `Resnet34BackFloatBudget.lean` |
| MobileNetV2 backward | training BN, no operating point | 4.750e153 | 1.076e152 | fold | `mnv2_grad_float_le` | `MobileNetV2BackFloatBudget.lean` |
| ConvNeXt-T backward | channel LN, `\|istd\| <= 16` | 1.023e251 | 1.563e250 | fold | `cnx_grad_float_le` | `ConvNeXtBackFloatBudget.lean` |
| EfficientNet-B0 backward | training BN, N = 1, no operating point | 7.104e182 | 1.578e182 | fold | `b0_grad_float_le` | `EfficientNetBackFloatBudget.lean` |

Window is the certified bound on the output's magnitude; budget is the bound on the distance
between the float output and the real output, per logit (forwards) or per input pixel
(backwards, on loss cotangents of magnitude at most 1). Inputs are on the unit box.

The three whole-net certified ties, all in `LeanMlir/Proofs/Foundation/`:

| net | tie | shape check | apex kind |
|---|---|---|---|
| ResNet-34 | `r34InputGrad_eq_resnet34_vjp` | `resnet34Forward_full_pc_eq_chain` | `HasVJPAt` (smooth point) |
| MobileNetV2 | `mnv2InputGrad_eq_mobilenetv2_vjp` | `mobilenetv2Forward_full_pc_eq_chain` | `HasVJPAt` (smooth point) |
| ConvNeXt-T | `convnextInputGrad_eq_convNextForwardTCh_vjp` | `convNextForwardTCh_eq_chain` | `HasVJP` (everywhere) |
| EfficientNet-B0 | open (section 5, item 3) | `efficientnetForwardB_eq_chain` | `HasVJP` (everywhere) |

The tie says the hand-written backward chain the number is stated on IS the certified whole-net
VJP, not merely that each of its pieces is. The shape check says the chain of opaque block
variables the apex is instantiated at IS the committed forward, slot for slot.

**Hypotheses every number carries.** A per-parameter-kind magnitude profile measured on the
trained checkpoint (`scripts/param_kind_profile.py`; the checkpoints are outside the repo, paths
in the budget files' headers). `eps >= 1e-5`, `u <= 2^-24`. Device kernels with no IEEE
specification are supplied with an accuracy of 1e-2: `rsqrt` on every BatchNorm net
(`DeviceRsqrt`), sigmoid on B0, GELU on ConvNeXt and ViT, `exp` on ViT (relative, plus the
softmax side condition `smRho u eexp 197 < 1`). The device mean is derived, not supplied
(`FloatModel.bnMean_close_of`, parameterised by the reduction's error spec so any summation
order qualifies). The backwards additionally take the saved activations' accuracies (`es`,
`exh`, B0 also `esav`) at 1e-2.

## 2. What the numbers certify, and what they do not

**They certify the float program's shape.** Every stage is present, at the right fan-in, in the
right composition, at the right normalisation mode, and the whole is tied to the committed net
by `rfl` and to the rendered graph by the `*_faithful` theorems. A dropped stage, a misread
fan-in, a stale layer slot or a mis-plugged block fails to compile. That is the property the
thread actually exercised, and it found seven defects (section 4).

**They certify that three backward chains are the certified gradient.** That is a statement
about this repo's purpose, and it is the strongest thing here.

**They do not certify numerical accuracy.** The budgets are 1e80 to 1e251 against logits of
order 10. This cannot be fixed by more work of this kind. The bound is the interval fold at
worst-case windows through 30 or more layers; on ResNet-34 the conv fan-in face alone is 14 to
95 times loose per layer against the measured row-L1 norm, and even at that measured face the
training forward would land at 1e19. A bound that bites needs the on-trajectory Jacobian, which
the adjoint-chain probe measures and which is not a static hypothesis (`planning/adjoint_chain.md`,
`formalization.yaml` fidelity 4c). The one non-vacuous number is MobileNetV2's window, 2154,
and that is the window, not the budget.

**The three caps are the triangle inequality.** `FloatBridgesTo.capped` replaces a modulus by
`2 * mag`, so a capped statement says only that the float and the real output both lie in the
certified window. `budget / window = 2.00` is the tell. Never table a cap beside a fold without
the label.

**No forward-then-backward composition exists.** The backwards assume the saved activations
are accurate to 1e-2. No forward statement supplies that: the inference forwards are about a
different program, the training-mode forward is a cap and supplies only `2 * window`, and the
LayerNorm nets have no fold in any mode. Say: an honest fold of the backward kernel's rounding,
given saved-activation accuracies this net's forward cannot supply.

**Two numbers carry a batch or example qualifier.** B0's backward is at `N = 1`, because
`bnBatchLA` reduces across examples and the number grows with the batch (2.880e194 at
`N = 256`). ResNet-34's training forward is per example for the same reason. The six inference
forwards hold at any batch size.

**The sentence for the book.** The fold certifies the shape and composition of the float
program against the certified real program and is checked by the kernel; its numerical value is
vacuous, and a tight bound needs measured Jacobians that are not static.

## 3. Findings worth keeping

1. **A forward's modulus goes quadratic in the window wherever an op reduces a statistic out
   of its own input and multiplies back.** Training-mode BatchNorm, LayerNorm and
   squeeze-excite all do; attention does it inside an exponential. Inference BatchNorm freezes
   its statistics, is affine in `x`, and its modulus is linear. That is why four forwards are
   folds and three are caps, and why the cap is not optional: the uncapped fold exists at an
   operating point and is 80 orders worse than the cap (ConvNeXt-T, measured), so `min(mod,
   2 * mag)` selects the cap on merit.

2. **Every backward folds.** A VJP is linear at a fixed point and reads its statistics off the
   saved activations, which the cotangent does not perturb. Four backwards at ratios 0.02 to
   0.22, squeeze-excite included. The wall relocates rather than vanishing: it becomes the
   saved activations' accuracies, which are hypotheses (section 2).

3. **The normalised activation is bounded by `sqrt n` whatever the input** (`bnXhat_sq_le`,
   written for the realistic-seal work). This one lemma is decisive on all four backwards
   (1e5147 without it on ConvNeXt-T) and worth 53 to 76 orders on each of the three caps. The
   forward LayerNorm leaf had bounded the same quantity by `|x - mu| * |istd| <= 2 A S` and
   thrown the better bound away. Window and budget are separate levers: relu6's clamp gives
   MobileNetV2 a window of 2154 (97 orders) and moves its budget one order; only the
   inverse-stddev bound `S` moves a budget, at about 19 orders per decade across 20 sites.

4. **The modelled device accuracies were the wall on the normalisation nets, not the
   arithmetic.** A normalisation site resets the window only if `emr * S < 1`; at the supplied
   `emr = 1e-2` and the eps-floor `S = 317` it multiplies by 3.19 instead. Deriving `emr` from
   the reduction's own rounding (5.8e-6 at n = 96) was worth 44 to 65 orders per net. `ei`, the
   device inverse-stddev accuracy, is now the loose one by four orders, and the remaining
   growth is the conv fan-in face `m * w'` shared by every number in the table. That face is the
   interval-arithmetic floor.

5. **There are two ways a number fails to exist.** Magnitude: `norm_num` refuses around 1e253
   for a nested arithmetic tree, and the ceiling depends on the tree's shape, not the value (the
   same number closes flattened). Representability: ViT's attention window carried `Real.exp` at
   an argument with no rational bound, so 36 stage numerals could not be written at all, at a
   magnitude smaller than the shipped one. A Python fold hides the second (`math.expm1`
   overflows to a finite float); `vit_chain` returns an `exp_tainted` tag list for it.

6. **A whole-net budget is homogeneous of degree 1 in the cotangent window on a backward, and
   a bias breaks that on a forward.** Factoring a BatchNorm-backward site's gain as one constant
   per feature-map size (`Maps.bnPerChannelBackGain`) took ResNet-34's chain from 39 minutes
   and 41 GB, not finishing, to 84 s and 3.8 GB.

## 4. What the thread found in the repo

Bringing the float tier, the codegen tier and the certified VJPs into one statement forced
several definitions to unify that had drifted apart. Every item below was found by needing a
tie or a number, never by review.

| defect | where | found by | effect |
|---|---|---|---|
| `r34InputGrad` reversed the 2x2 pool; the committed forward pools 3x3/s2 (restored 2026-08-03) | `Resnet34WholeBackFloatBridge.lean` | the whole-net tie | number x4; new leaf `MaxPool3s2BackFloatBridge.lean` |
| ConvNeXt forward bridge held `id` in the head-LayerNorm slot (head LN restored 2026-08-30) | `ConvNeXtWholeFloatBridge.lean` | needing the tie for a number | 23 LN hypotheses, not 22 |
| ViT float cone stated on `transformerBlock`'s scalar LN affines; the net trains vectors | `ViTBlockFloatBridge.lean` | needing the tie for a number | new `ViTBlockVFloatBridge.lean` |
| ConvNeXt backward bridge held `id` in the same slot, justified by a stale LN count in its docstring | `ConvNeXtBackFloatBridge.lean` | the backward probe | 6 orders; fixed same day |
| `convFlatBack` is not the adjoint at an even kernel; the emitter had been fixed twice, the float tier's peer never | `EvenKernelConvBack.lean` (`padOdd`) | the ConvNeXt whole-net tie | ConvNeXt backward 1.25 orders |
| CIFAR-8 chain's dense head had three bare denses where the committed layers interleave relu | `Cifar8ChainCert.lean` | the `rfl` tie | fixed in the tie commit |
| Counts in docstrings: ResNet-34 has 36 BatchNorm sites, not 33 (nine places); ConvNeXt 23 LayerNorm sites, not 22 (three places) | budget files, `AuditAxioms.lean`, `formalization.yaml` | needing the number | prose only; no fold reads the count |

Nothing trained was affected: the emitted programs were right in every case. What drifted was
a hand-written spelling of a map that also has a certified spelling. The rule that follows is
in section 7.

## 5. Closing items, in order

Each has an acceptance criterion. None makes a number smaller.

1. **Inhabitation checks for every budget file.** Add one compiled `example` per budget file
   constructing its weights record at the committed profile (gamma 0, saved activations 0,
   float peers exact). Eleven files. The check that the theorem is not about an empty type
   currently lives in session scratch for two of them and nowhere for the other nine. A record
   with one unsatisfiable field would make a whole-net number vacuous in the bad sense, and
   this is the one way the table could be wrong rather than merely loose. Done when all eleven
   compile with the example in the file.

2. **The standing audit.** Walk the emitters' fix history (`git log -- LeanMlir/StableHLO.lean`
   and the render files, for backward and padding fixes) and for each fix grep for every other
   definition that claims to denote the same map: the float tier's `*Back` leaves, the
   `Proofs` hand-written backwards, the `Maps` leaves. The even-kernel pad was fixed twice on
   the codegen side and reached the float tier only through the ConvNeXt tie. Deliverable: a
   table in this document, one row per emitter fix, saying which peers were checked and whether
   any drifted. Done when the table exists.

3. **EfficientNet-B0's whole-net certified tie.** Aim at `efficientnetForwardB_has_vjp`
   (`Architectures/EfficientNetChainClose.lean`, `HasVJP` everywhere), not the 16-block
   `efficientnetForwardB_full_has_vjp`. Missing: the three batched block ties at `bnBatchLA`
   (`mbNoExpFwdB`, `mbStridedFwdB`, `mbResidFwdB`; the one existing tie is per example at scalar
   `bnForward`), the batched leaf ties (`batchMap_has_vjp` is built by transport so its
   `.backward` does not reduce; `hasVJPMat_to_hasVJP (rowwise_has_vjp_mat ...)` typechecks at
   `HasVJP (StableHLO.batchMap N f)` with a `.backward` that does, verified), and the assembly
   in the shape of `MobileNetV2WholeBackCertifiedTie.lean` with the blocks opaque. State it at
   `N = 1`, matching the number; general `N` needs a batched BatchNorm backward leaf
   (`bnBatchTensor4` conjugated by `bnchwFwd`/`bnchwBack`) that does not exist. Done when the
   tie compiles, `efficientnetForwardB_eq_chain` is used as its shape check, and the number in
   `EfficientNetBackFloatBudget.lean` is unchanged or the change is explained.

4. **Freeze.** Add the table in section 1 and the sentence at the end of section 2 to the
   blueprint; the blueprint currently mentions the float budgets in one clause and tables none
   of them. After that this document changes only when a number moves.

## 6. Priced and declined

Each of these was measured. None is an oversight, and none should be started without a reason
that is not "the number gets smaller".

* **The conv fan-in at the row-L1 face.** `layerAct` charges `m * w'` where the honest bound is
  `max_o ||W_o||_1`, a checkpoint fact of the same kind as `|w| <= 21/10`. Measured on
  ResNet-34 (`scripts/param_row_l1.py`): 61 orders on the training forward, 63 on the inference
  one. Declined because the result is still vacuous (1e19 against logits of 10), it edits
  `FloatBridge.lean` (98 dependent modules), every parameter record gains a field, eleven
  committed numbers move, and `Maps.convBack`'s face needs its own measurement. If the row-L1
  bound is wanted, it belongs in the adjoint-chain probe's proven tier, where it closes part of
  a gap that probe already costs.
* **`ei` stated relative rather than absolute.** 3.7x per normalisation site. Same files as the
  item above; same reason.
* **Escape 2's modulus half** (the input-sensitivity `2 e S (1 + Xh)`, free of the window). 5006
  orders on ConvNeXt's uncapped fold and nothing to any shipped statement, because the fold
  stays 82 orders (ConvNeXt-T) and 7 orders (ViT-Tiny) above the cap. Needs a reverse triangle
  inequality for the Euclidean norm over `Vec n`.
* **`swishScalar_lipschitz` wired into `floatClose_swish`.** 8 orders on B0's forward, window
  unchanged. Moves a committed number; take it only if B0's forward is reopened.
* **`floatClose_broadcastBack`'s spurious factor of `c`.** 6 orders on B0's backward; needs the
  cardinality of `flatChannel`'s fibre, a lemma that does not exist.
* **A pointwise `|sigmoidScalarDeriv x| <= 1/4`.** `EnetSeBack.hssig` is a hypothesis where the
  swish bound is a theorem. Not load-bearing.
* **The sharp `|swish'| ~ 1.1`.** 2.6 orders against the proved global 2. Do not.
* **Per-width `emr` on the LayerNorm nets.** 2 orders on ConvNeXt-T, nothing on ViT-Tiny; costs
  a `Nat -> R` inside 690 `norm_num` goals.
* **A `FloatBudgetEnvCore` split** of the `Maps` kit. Cone hygiene only.
* **MobileNetV2's and B0's training-mode forwards.** Would be two more caps. B0's would also be
  per example.
* **ViT-Tiny's backward.** `MhsaBackFloatBridge.lean` is 8 `floatBridgesTo_` against 46
  existential-tier `floatBridges_`, so it is a tier migration first, for a fifth number of a
  kind there are four of.
* **`efficientnetForwardBEval N = batchMap N (per-example forward)`**, the whole-net form of
  "inference decouples the batch". Only the per-site claim is proved.

## 7. Working on these files

**Before writing Lean.** Probe in `scripts/float_budget_envelope.py` first. When a fold
overshoots, ablate the leaves and the inputs (the per-kind profile) before concluding anything
about the architecture: five of six blockers were bounds already proved somewhere in the repo
and thrown away by a leaf (relu6's clamp, swish's modulus, seScale's window, ConvNeXt's uniform
profile, attention's window derived through an error term). Grep the whole cone for a bound
before proving one, not the files named after the net; the LayerNorm leaf that serves ViT lives
in ConvNeXt's file. A block needs three tiers, a `def`, a `floatBridgesTo_` and a `Maps.`, and
each has been the missing one.

**Tie first, then fold.** The three ties found drift; the four folds found none. Keep the
blocks opaque in the apex so the whole-net `isDefEq` compares variables, and add the shape
`rfl` in the same commit, because an opaque-block tie says nothing about which net the
variables are.

**Granularity comes from the committed definition, never from the emitted graph.** Attention
fans out, so it is one leaf; `patchEmbed_flat` has an `if` branch, so it is one leaf. A `Maps`
chain is a line.

**Emitter fix rule.** When a fix lands on an emitter, grep for every other definition that
claims to denote the same map. A `den` stated as the certified VJP cannot drift; a hand-written
peer can and did.

**Counts and cost estimates in docstrings are what stop re-checking.** "33 BN sites", "22 LN
sites", "needs calculus" each cost between a day and a month. When a docstring justifies a
slot or declines a bound with a number, re-derive the number.

**The verify pass is not a sync check.** `verify_*` re-asserts a chain against itself; the
probe's default flags drifted from the committed Lean for two nets while every count passed.
When a flag becomes the committed shape, flip its default in the same commit, so a no-argument
call reproduces the Lean file.

**Re-emitting a budget file.** The generators were session scratch six times. The method that
worked: extract the shipped file's ordered `(name := numeral)` list, reproduce it exactly from
the shipped chain (this is the check that the map from probe rows to `Maps` arguments is right;
ConvNeXt's hand-written map was wrong at the tail and the reproduction caught it), then re-emit
the same list at the new flags. ViT's depth-12 fold passes block boundaries as `| k => numeral`
match arms and a `| _ =>` fall-through, which a regex over `(name := ...)` misses.

**Reproduce a number.** `python3 scripts/float_budget_envelope.py` folds every chain and runs
every verify pass; each `*_chain()` with no arguments reproduces its committed Lean file.
`python3 scripts/param_kind_profile.py` gives the per-kind profile; `python3 scripts/param_row_l1.py`
the row-L1 face (ResNet-34 only).

**Per commit.** `lake build Proofs Certs` (bare `lake build` skips the Certs corpus);
`lake env lean tests/AuditAxioms.lean` exit 0 with every new declaration on
`[propext, Classical.choice, Quot.sound]`; `lake exe docstring-checkrefs`;
`python3 scripts/check_audit_coverage.py`. A new file needs a lakefile `Proofs` root, an
`AuditAxioms` import and a `#print axioms` line, and a `formalization.yaml` main_results entry
for its headline theorem. Stage, then stop and ask before committing. One commit per net.

**Pitfalls, in the order they bite.**

* `Maps` must stay a Prop-structure. As a `def` unfolding to a conjunction the unifier
  delta-unfolds the whole `.mag` chain and times out at 20x the heartbeat budget.
* Pin all four numerals on every leaf and on `Maps.residual`: `(A := ..) (E := ..) (A' := ..)
  (E' := ..)`. A `by norm_num` inside a `have` runs before `Maps.comp` unifies, so an unpinned
  output window is a metavariable and the error reads like arithmetic (`<44-digit numeral> <=
  ?m.3743`).
* A per-site constant a `Maps` goal must evaluate belongs in the record's type, not in a field:
  a projection `B.Xh` is opaque to `norm_num`.
* An op declared over a computed dimension (`x : Tensor3 c (2*h) (2*w)`) makes any composition
  containing it a higher-order unification (`2 * ?h = 112`) that presents as a non-terminating
  whole-net `isDefEq`. Pin the implicits. Pinning is not enough when the term is applied: two
  closed spellings of one numeral (`Vec (cin * (2*28) * (2*28))` against `Vec (96*56*56)`) still
  send the unifier into the net's semantics. Give the stage a `def` with the type ascribed in
  the chain's spelling. A leaf tie goes the other way: state it in the lemma's spelling. An
  unapplied comparison is free either way, which is the test for which case you are in.
* A `chainComp` node compared against a flat `Function.comp` chain is a kernel timeout at any
  heartbeat budget. Peel it once between variables (`chainComp [f, g] ∘ k = f ∘ g ∘ k`, an
  abstract `rfl`) and `rw`. `Function.comp_assoc` in a `simp only` set unfolds the reducible
  block bodies instead.
* After peeling a chain of `vjp_comp` reductions, close with
  `simp only [Function.comp_apply, <base witness>]`, not `rfl`.
* Transport a bridge along a tie with `FloatBridgesTo.ofEq` (rebuild the structure field by
  field), never with `▸`: an `Eq.mpr` blocks `.mag` from reducing and a bridge whose `.mag`
  does not reduce cannot carry a `Maps`. The same trap applies to a `HasVJP` witness's
  `.backward`.
* Read a bound bundle by projection (`hb.1`, `hb.2.1`), not `obtain`: `And.casesOn` is stuck on
  a variable and the bridge will not reduce.
* Check which way a recursive fold associates before writing its chain: `vitBodyKVFlat`
  recurses head-first, `convNextStageChK` right-associated. The definition decides.
* `norm_num`'s ceiling is about 1e253 for a nested tree, shape-dependent; heartbeats, recursion
  depth, `ring_nf`, `nlinarith` and `simp only` first do not move it. Flatten the tree
  (per-unit-gain factoring) or take an operating point.
* Round the window first, then double it, for any capped leaf: `2 * r4(x)` can exceed
  `r4(2 * x)` and break `Maps.capped`'s own `2 * A' <= E'`.
* Fold with the rounded gamma (`r4(gamma_q k)`), never the exact `(1+u)^k - 1`; the Lean chain
  passes the rounded one.
* Read the leaf's `mag`/`mod` before folding it, and read where the affine sits: a BatchNorm
  leaf carries gamma/beta inside, a LayerNorm leaf composes them outside, and the two spellings
  differ by `u * Bbnd` per site.
* `Elab.async` shares caches across declarations, so per-`rfl` timings are order-dependent.
  Measure with `set_option Elab.async false`.
* `lake env lean <file>` reads built oleans; a leaf edited in the same session must be
  `lake build`-ed before a file above it sees it.
* Two files that never meet on an import path can each define the same name for days
  (`FloatBridgesTo.fresh_nonneg`, twice). When a kit is superseded, import the successor from
  the old kit's consumers.
* `FloatBridge.lean` has 98 dependent modules and `BnFloatBridge.lean` 85. Develop against them
  in scratch and edit once.

## 8. Files

| file | holds |
|---|---|
| `Float/FloatComposeBridge.lean` | `FloatBridgesTo`: the Type-valued bridge (`mag`, `mod`, `close`) and its combinators |
| `Float/FloatBudgetEnv.lean` | the `Maps` numeric-envelope kit: `comp`, `residual`, `biPathSum`, `capped`, and the conv-net leaves |
| `Float/FloatBudgetEnvMBConv.lean`, `EnvLN.lean`, `EnvAttn.lean` | the inverted-bottleneck, LayerNorm and attention leaves; the modelled device kernels (`DeviceLN`, `DeviceGelu`, `DeviceExp`) |
| `Float/FloatBudgetEnvBack.lean`, `EnvBackMBConv.lean`, `EnvBackLN.lean`, `EnvBackSE.lean` | the four backward kits, including `Maps.bnPerChannelBackGain` and `bnIstd_abs_le_of` |
| `Float/BnXhatFloatBridge.lean` | the normalise leaf at `\|x-hat\| <= sqrt n` (`floatClose_bnX`, `Maps.bnCappedX`, `Maps.bnPerChannelTensor3CappedX`, `bnXhat_abs_le_num`) |
| `Float/BnEvalRuntimeFloatBridge.lean` | inference BatchNorm as the render emits it (six runtime ops, `rsqrt` on device) |
| `Codegen/*RenderPCEval.lean` | the inference-mode graph twins and their `_faithful` theorems (ResNet-34, MobileNetV2, EfficientNet-B0) |
| `Float/*FloatBudget.lean` | the eleven numbers (section 1) |
| `Foundation/*CertifiedTie.lean` | the three whole-net backward ties and their shape checks |
| `Foundation/EvenKernelConvBack.lean` | `padOdd`: the even-kernel conv backward as an odd-kernel one |
| `Float/MaxPool3s2BackFloatBridge.lean` | the 3x3/s2 pool's backward (accumulating, window `4A`) |
| `Architectures/SwishSaturation.lean`, `GeluSaturation.lean` | the global derivative bounds `\|swish'\| <= 2`, `\|gelu'\| <= 3/2` |
| `scripts/float_budget_envelope.py` | the exact-rational folds and the `verify_*` passes, one pair per number |
| `scripts/param_kind_profile.py`, `scripts/param_row_l1.py` | the checkpoint profiles |
