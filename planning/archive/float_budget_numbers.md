> ⛔ **DELETED 2026-09-08.** Everything below describes the whole-net float budgets, their envelopes and forward float chains, removed from the repo as vacuous
> (user decision; `formalization.yaml` fidelity 4c/4d records why). Historical record only.

# Whole-net float budgets

**Status, 2026-09-05: the research thread is closed.** Nine forwards and four input-gradient
backwards carry kernel-checked numbers; six backward chains are proven to be the certified
whole-net gradient. The eighth and ninth forwards — MobileNetV2 at its seventeen-block paper
depth and EfficientNet-B0 at its sixteen-block paper depth — came from
`planning/archive/proofs_tier_to_paper_nets.md` packages 3.2(b) and 3.3(a), not from re-opening this
thread: each states an existing kind of number about a bigger net, and shaves no orders off any
row below. The ninth did correct one standing claim (section 3, finding 5): `norm_num`'s
"ceiling" is an option, not a wall.
What remains is three closing items (section 5) and then a freeze. The
session-by-session record, with the ablations and the corrections, is
`planning/archive/float_budget_numbers_log.md`; its section numbers are what the Lean docstrings
cite (`§0.1`, `§3.16`, `§9`, ...) and are preserved there.

This document is the standing reference: what exists, what it certifies, what it does not, and
what to do if one of these files is opened again.

## 1. What exists

Every row is a theorem in `LeanMlir/Proofs/Float/`, closed over the real leaves (no
`FloatBridgesTo` hypothesis left), tied to the committed net definition and, for the forwards,
to the rendered graph — with no exceptions left. The two paper-net rows had none when their
numbers landed, because neither eval twin existed; `MobileNetV2FullPaperEval.lean` and
`EfficientNetFullB0Eval.lean` (packages 3.2(e) and 3.3(e), both 2026-09-05) are those twins, and
`mnv2Paper_float_logits_le_committed` / `b0Full_float_logits_le_committed` are the restatements.
Every numeral
was produced by `scripts/float_budget_envelope.py` in the leaves' own exact-rational arithmetic,
re-asserted by that script's `verify_*` pass, and then checked again by the kernel. All are
3-axiom clean and listed in `tests/AuditAxioms.lean` and `formalization.yaml`
(status.main_results, fidelity 4d).

| net | mode and qualifiers | window | budget | kind | theorem | file |
|---|---|---|---|---|---|---|
| CIFAR-8 forward | 8 conv, no normalisation | 6.121e18 | 6.37e14 | fold | `cifar8_float_logits_le` | `Cifar8FloatBudget.lean` |
| ResNet-34 forward | inference BN | 3.152e211 | 1.548e209 | fold | `r34_float_logits_le` | `Resnet34FloatBudget.lean` |
| ResNet-34 forward | training BN, per example | 8.748e80 | 1.752e81 | cap | `r34_train_float_logits_le` | `Resnet34TrainFloatBudget.lean` |
| MobileNetV2 forward | inference BN | 2.154e3 | 1.444e96 | fold | `mnv2_float_logits_le` | `MobileNetV2FloatBudget.lean` |
| MobileNetV2 forward, 17-block paper | inference BN, 52 sites, capped at every one; tied to the typed eval graph; any class count | 2.152e4 | 8.176e16 | cap | `mnv2Paper_float_logits_le` | `MobileNetV2PaperFloatBudget.lean` |
| EfficientNet-B0 forward | inference BN, any batch size | 2.580e55 | 8.408e210 | fold | `b0_float_logits_le` | `EfficientNetFloatBudget.lean` |
| EfficientNet-B0 forward, 16-block paper | inference BN, 49 sites; the sigmoid of each of the 16 SE gates capped, nothing else; tied to the typed eval graph; any batch size, any class count | 1.886e279 | 2.416e287 | cap | `b0Full_float_logits_le` | `EfficientNetFullFloatBudget.lean` |
| ConvNeXt-T forward | channel LayerNorm | 4.871e130 | 9.738e130 | cap | `cnx_float_logits_le` | `ConvNeXtFloatBudget.lean` |
| ViT-Tiny forward | vector LayerNorm, depth 12 | 2.397e108 | 4.794e108 | cap | `vit_float_logits_le` | `ViTFloatBudget.lean` |
| ResNet-34 backward | training BN, `\|istd\| <= 16` | 8.857e245 | 6.894e244 | fold | `r34_grad_float_le` | `Resnet34BackFloatBudget.lean` |
| MobileNetV2 backward | training BN, no operating point | 4.750e153 | 1.076e152 | fold | `mnv2_grad_float_le` | `MobileNetV2BackFloatBudget.lean` |
| ConvNeXt-T backward | channel LN, `\|istd\| <= 16` | 1.023e251 | 1.563e250 | fold | `cnx_grad_float_le` | `ConvNeXtBackFloatBudget.lean` |
| EfficientNet-B0 backward | training BN, N = 1, no operating point | 7.104e182 | 1.578e182 | fold | `b0_grad_float_le` | `EfficientNetBackFloatBudget.lean` |

Window is the certified bound on the output's magnitude; budget is the bound on the distance
between the float output and the real output, per logit (forwards) or per input pixel
(backwards, on loss cotangents of magnitude at most 1). Inputs are on the unit box.

The seven whole-net certified ties, all in `LeanMlir/Proofs/Foundation/`:

| net | tie | shape check | apex kind |
|---|---|---|---|
| ResNet-34 | `r34InputGrad_eq_resnet34_vjp` | `resnet34Forward_full_pc_eq_chain` | `HasVJPAt` (smooth point) |
| MobileNetV2 | `mnv2InputGrad_eq_mobilenetv2_vjp` | `mobilenetv2Forward_full_pc_eq_chain` | `HasVJPAt` (smooth point) |
| ConvNeXt-T | `convnextInputGrad_eq_convNextForwardTCh_vjp` | `convNextForwardTCh_eq_chain` | `HasVJP` (everywhere) |
| EfficientNet-B0 | `efficientnetInputGradB_eq_efficientnetForwardB_vjp` | `efficientnetForwardB_eq_chain` | `HasVJP` (everywhere) |
| MobileNetV2, 17-block paper | `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp` | `mobilenetv2ForwardPaper_eq_slots` | `HasVJPAt` (smooth point) |
| EfficientNet-B0, 16-block paper | `efficientnetInputGradB_full_correct` | `efficientnetForwardB_full_eq_chain` (inside `efficientnetForwardB_full_has_vjp_correct`) | `HasVJP` (everywhere); through `HasVJP.backward_unique` to the concrete `efficientnetForwardB_full_has_vjp` |
| ViT-Tiny, depth 12 | `vitInputGradK_eq_vitForwardKV_vjp` (`vitTinyInputGrad_eq_vitTiny_vjp` at the shipped dims) | `vitForwardKV_eq_chain` | `HasVJP` (everywhere); through `HasVJP.backward_unique` to the committed `vitForwardKV_has_vjp` |

The tie says the hand-written backward chain the number is stated on IS the certified whole-net
VJP, not merely that each of its pieces is. The shape check says the chain of opaque block
variables the apex is instantiated at IS the committed forward, slot for slot. The sixteen-block
B0 tie is the strongest form: `HasVJP.backward_unique` (two witnesses for one map have one
backward) carries the opaque-block tie to the tactic-built concrete witness without unfolding
it, and `_correct` then states the chain is the `pdiv`-contracted Jacobian of the committed
nested-application forward. The three-block B0 file could not take that step and stopped at a
`▸`-transported `_committed` witness; the difference is the lemma, not the depth.

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
thread actually exercised, and it found eight defects (section 4).

**They certify that six backward chains are the certified gradient.** That is a statement
about this repo's purpose, and it is the strongest thing here.

**They do not certify numerical accuracy.** The budgets are 1e80 to 1e251 against logits of
order 10. This cannot be fixed by more work of this kind. The bound is the interval fold at
worst-case windows through 30 or more layers; on ResNet-34 the conv fan-in face alone is 14 to
95 times loose per layer against the measured row-L1 norm, and even at that measured face the
training forward would land at 1e19. A bound that bites needs the on-trajectory Jacobian, which
the adjoint-chain probe measures and which is not a static hypothesis (`planning/archive/adjoint_chain.md`,
`formalization.yaml` fidelity 4c). The one non-vacuous number is MobileNetV2's window, 2154
(2.152e4 at the paper depth, and the growth is the wider classifier, not the eleven extra
blocks), and that is the window, not the budget.

**The five caps are the triangle inequality.** `FloatBridgesTo.capped` replaces a modulus by
`2 * mag`, so a capped statement says only that the float and the real output both lie in the
certified window. `budget / window = 2.00` is the tell — with two exceptions to read carefully.
The seventeen-block MobileNetV2 forward caps at every BatchNorm and then runs three more stages
(relu6, GAP, the classifier) which collapse the window and carry the error forward, so its ratio
is 3.8e12. The sixteen-block EfficientNet-B0 forward caps only the sigmoid of each squeeze-excite
gate — the one stage whose window is a constant, `1 + esig`, so the cap costs one side condition
and turns the rescale's quadratic `A · Eg` into `≈ 2A + 3E` — and folds everything else, so its
ratio is 1.3e8 (sixteen gate caps compounding) while its window is honest and uncapped. In both
the label has to be read off the file, not off the ratio. Never table a cap beside a fold without
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

**Two paper nets are now stated at any class count.** MobileNetV2's and B0's heads are generic in
`nCls`, because `Maps.dense`'s envelope depends on the fan-in and never on the output count. That
closes a real qualification: both `|·| ≤ 28/10` and `|·| ≤ 41/10` were measured on the 1000-class
checkpoints, while the committed artifacts are 10-class, so the bound had been a measurement on
the body and an assumption on the head. One theorem now covers both artifacts.

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

5. **There are two ways a number fails to exist — and the first is an option, not a wall
   (corrected 2026-09-05).** Magnitude: every earlier note here and in the archive puts
   `norm_num`'s ceiling near 1e253, "shape-dependent". Measured while writing the sixteen-block
   B0 forward: the ceiling is Lean's `exponentiation.threshold` (default 256). In the exact goal
   shape that was failing, `10 ^ 256` evaluates and `10 ^ 257` does not, and under
   `set_option exponentiation.threshold 400` the same goals close at `10 ^ 290` in the same time.
   The "shape dependence" was which stages' numerals happened to carry an exponent above 256.
   The kernel's `Nat.pow` is GMP-backed and never had a limit. Consequences: `b0Full_float_logits_le`
   is stated at the eps-floor with no operating point, window 1.886e279; "no theorem to state" is
   retired as a reason everywhere it was given (finding 7 below, the 3-block B0 backward's
   `1e431` history, the seventeen-block MobileNetV2 backward at 1e323); and none of that changes
   what any number means (section 2), so none of those has been or should be written down.
   Representability is unchanged: ViT's attention window carried `Real.exp` at an argument with
   no rational bound, so 36 stage numerals could not be written at all, at a magnitude smaller
   than the shipped one. A Python fold hides it (`math.expm1` overflows to a finite float);
   `vit_chain` returns an `exp_tainted` tag list for it.

6. **A whole-net budget is homogeneous of degree 1 in the cotangent window on a backward, and
   a bias breaks that on a forward.** Factoring a BatchNorm-backward site's gain as one constant
   per feature-map size (`Maps.bnPerChannelBackGain`) took ResNet-34's chain from 39 minutes
   and 41 GB, not finishing, to 84 s and 3.8 GB.

7. **A depth a fold cannot reach, ablated rather than assumed.** MobileNetV2's backward exists
   at six blocks (4.750e153 / 1.076e152, no operating point) and does not exist at seventeen. The
   shipped operating point `|istd| <= 16` gives a WINDOW of 1.246e323, so a cap cannot rescue it
   either — a cap's budget is `2·window`. Nothing in the family gets under: `|istd| <= 4` is
   6.769e291 and sigma^2 ~ 1, the crudest setting there is, is 4.901e260. And the reason is the
   depth and not a loose leaf: dropping the BN gamma bound from its measured 1.69 to 1 buys 12
   orders, dropping the conv kernel bound from 2.72 to 1 buys 24, and only their simultaneous
   fiction gets under the ceiling. Per block the chain costs 17 to 21 orders, dominated by the
   three BatchNorm-backward sites at x5.4e3 each. EfficientNet-B0's backward at 16 MBConvs is the
   same shape at a larger scale: 9.112e2648 at the shipped leaves (`b0_full_back_chain`), and the
   fiction that sets every measured bound to 1 still lands at 1e344. Same answer for both: report
   that there is no number, do not shave — and, since finding 5's correction, "cannot be stated"
   is no longer the reason; "would say nothing" is.

   ⭐ Its FORWARD is the opposite result, and it is now a theorem
   (`MobileNetV2PaperFloatBudget.lean`, 2026-09-05). Uncapped the 52-site fold is 2.104e266 — no
   theorem — but capped at the BatchNorm sites it is 8.176e16, which is 79 orders SMALLER than the
   shipped UNCAPPED six-block number (1.444e96). Capping the normalisation sites is worth more
   than the eleven extra blocks cost. ⛔ Label it: the Lean caps at ALL 52 sites, so at every
   normalisation the claim is the triangle inequality and not the fold. Taking `min(fold, 2·mag)`
   instead would select the fold at 12 of the 52 and change nothing to four figures — the last cap
   discards the history — so the uniform cap is both the simpler statement and the honest one.

   ⭐ A property of the cap worth keeping: at a capped site the output error is `2·window` and the
   inherited error is discarded, so a block's stage numerals depend only on its `(ic, mid, oc)` and
   not on its depth. `b8`, `b9` and `b10` are numerically one block. That is what makes a
   seventeen-block chain writable at all; it is not a fact about the net.

   ⭐ EfficientNet-B0's sixteen-block forward is the other kind of cap (`EfficientNetFullFloatBudget.lean`,
   2026-09-05). Squeeze-excite is quadratic in the window because `seScale`'s modulus carries
   `A · Eg` and the gate grows `Eg` out of the same window, so sixteen sites fold to 1e1897907. The
   cap goes on the gate's SIGMOID — the one stage whose window is the constant `1 + esig`, so
   `Maps.capped` there needs no error numeral at all, only `2·(1+esig) ≤ Eg` — and the rescale's
   error becomes `≈ 2A + 3E`. Nothing else is capped: 49 BatchNorms, the rescale, every conv.
   Window 1.886e279 (honest; swish never resets a window), budget 2.416e287, ratio 1.3e8. Where to
   cap is a per-op question: cap the stage whose window is bounded, not the stage whose error is
   large.

8. **A saved-activation bound taken from the forward's certified window is worth 1158 orders,
   and the proved replacement was already in the repo** (ViT-Tiny's backward probe, 2026-09-05 —
   finding 3 for the fifth time, in the one place it had not been looked for).
   `floatBridges_mhsaBack` takes `|Q i k| <= qA`, `|K i k| <= kA`, `|V i k| <= vA` as FREE
   hypotheses on the saved projections, and each sdpa core multiplies the cotangent window by
   `n * (1+n) * dh * scaleA * vA * kA`. Discharging those from the forward's own certified window
   — which grows `2 A S` per LayerNorm site and reaches 1e108 at depth 12 — gives 5.798e1557 and
   a per-block multiplier that GROWS with depth (10^225 at block 11, 10^32 at block 0).
   Discharging them from `bnXhat_sq_le` instead gives 5.686e399 and a per-block multiplier that
   is UNIFORM at 10^32. The bound is `|Q| <= (1 + gamma_{D+2}) * (D * w' * (G * sqrt D + Bl) + b)`
   = 3281 at ViT-Tiny, whatever arrives: ViT's per-token LayerNorm is literally `gamma * x-hat +
   beta`, so `|x-hat| <= sqrt D` bounds its OUTPUT by a constant and one dense bounds the
   projection. Depth-independent, and one line from a lemma the repo already has: `layerNormVec
   D eps g b x k = g k * bnXhat D eps x k + b k` by unfolding `layerNormForward`/`bnForward`, and
   `bnXhat_sq_le` is the bound. ⚠ Not written in Lean, because the number was declined.

   ⭐ **The same reading is available to ViT's committed FORWARD number and was not taken.**
   `vit_ln_leaf` / `Maps.bnCapped` bound the LayerNorm output by `G * (2 A S) + Bb`, growing;
   `min(2 A S, G * sqrt D + Bl)` is the honest window and is a constant. That is finding 3's
   sentence — *"the forward LayerNorm leaf had bounded the same quantity by `|x - mu| * |istd| <=
   2 A S` and thrown the better bound away"* — still true of every LayerNorm and BatchNorm
   forward in the table, three years of orders from being cashed. ⛔ NOT CHASED (user decision,
   2026-09-05): the thread is closed, the numbers are vacuous either way, and re-spelling
   `Maps.bnCapped`'s window moves six committed forward numbers. Recorded so that nobody
   re-derives it, and so that a future forward number is written with the `min` from the start.

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
| The Proofs tier spelled EfficientNet-B0's and MobileNetV2's stride-2 convolutions at symmetric padding; the shipped renders moved to XLA-SAME padding on 2026-08-08 (B0's stem; MobileNetV2's stem and four strided depthwises) | `EfficientNetRenderPC.lean`, `MobileNetV2RenderPC.lean`, both `*Eval` twins, `efficientnetForwardB`, `mobilenetv2Forward_full_pc`, both backward chains, all four B0/MNv2 numbers | the standing audit (section 5, item 2) | numbers unchanged; the claim "the deployed forward" was one padding phase off at those sites; **re-spelled 2026-09-05**, all four numbers reproduced |

Nothing trained was affected: the emitted programs were right in every case. What drifted was
a hand-written spelling of a map that also has a certified spelling. The rule that follows is
in section 7.

## 5. Closing items, in order

Each has an acceptance criterion. None makes a number smaller.

1. **Inhabitation checks for every budget file. Done 2026-09-05.** Each of the twelve budget
   files ends with an `Inhabitation` section: a `*.zero` / `*.exact` witness per record (zero
   weights, zero saved activations, the exact device kernels `DeviceRsqrt.exact`,
   `DeviceSigmoid.exact`, `DeviceLN.exact`, `DeviceGelu.exact`, `DeviceExp.exact`, the last three
   in `FloatBudgetEnvLN.lean`) and an `example` applying the headline theorem to the witness at
   `binary32`, so every hypothesis is discharged by data and none of the twelve theorems is about
   an empty type. The two backwards with an operating point (`|istd| ≤ 16`) are witnessed at
   `ε = 1/256`: at zero saved activations the inverse-stddev is `1/√ε`, so the operating point at
   the eps-floor needs saved activations with per-channel variance at least `1/256 − ε`, a
   checkpoint fact rather than a shape fact. The other ten are at `ε = 1/100000`.

2. **The standing audit. Done 2026-09-05.** Every commit touching `Codegen/StableHLO.lean`
   whose message names a fix (8 of 56) was read, and for each fix that changed what a map
   denotes, every other definition of the same map was checked. Emit-only, performance-only and
   proof-only fixes (`2936318` bf16 convert, `b71a596` shape table, `517224b` ite regression)
   change no `den` and have no peers to drift.

   | emitter fix | map | hand-written peers checked | drift |
   |---|---|---|---|
   | `63f6370` 2026-08-04, stem pool 2x2 to 3x3/s2 | `maxPool3s2F` / `maxPool3s2Back` | committed forward (3x3/s2), `Maps.maxPool3s2`, `r34InputGrad` (was `maxPoolFlatBack`) | found by the 2026-09-03 tie and fixed; residual: the `ResNet34Live*` non-degeneracy witnesses are stated over a 2x2-pool net, which is a witness net and not the committed one, and should say so |
   | `9e056ce` 2026-08-03, even-kernel pad in the batched strided-conv backward | `convStridedBack` at `kH` even | `flatConvStride2Back` / `flatConvStride4Back` (`convFlatBack ∘ scatter`, symmetric) | found by the 2026-09-04 ConvNeXt tie, repaired by `padOdd`; every even-kernel call site now goes through it (36 uses, no other file carries an even-kernel literal) |
   | `3d9b14d` + `601a900` 2026-08-08, XLA-SAME strided convs in the EfficientNet and MobileNetV2 renders | `convStridedXla`, `depthwiseStridedXlaF` (`flatConvStride2Xla` = `decimateOdd ∘ flatConv`) | `stemB` (`EfficientNetRenderPC.lean:54`), `efficientnetForwardB` and its `_faithful` graph, `b0EvalForward`, `efficientnetInputGradB`; `mobilenetv2Forward_full_pc`, `MobileNetV2RenderPC` / `PCEval`, `mnv2EvalForward`, `mnv2InputGrad`, `mobilenetv2PC_has_vjp_at` | **yes, and unrecorded at this tier.** No `Architectures` definition uses the XLA forms. Shipped artifacts: `efficientnet_fwd`, `_fwd_eval`, `_adam_train_step` each have 1 asymmetric site; `mobilenetv2_fwd_eval`, `_adam_train_step` 5 each; `mobilenetv2_fwd` (the SGD pair) 0, deliberately (`scripts/convention_audit.py`). So the Proofs describe MobileNetV2's SGD-pair net and, for B0, a net no shipped artifact has run since 2026-08-08 |
   | `019e09d` 2026-07-28, MobileNetV2 forward rendered at batch BN against a per-example train step | `bnBatchF` vs `bnPerChannelF` | `MobileNetV2RenderPC` (per example), both budgets (inference BN, where the two worlds coincide), `mnv2InputGrad` (per example, tied); B0 batch-BN on both sides (`bnBatchLA`) | none |

   **What the third row means and does not.** The four B0 and MobileNetV2 numbers do not move:
   `decimateOdd` selects the odd positions where `decimate` selects the even ones, the fan-in
   and the rounding are identical, so the fold is the same to the digit. What is off is the
   sentence "the deployed forward" in four docstrings and in `formalization.yaml`: at 1 (B0) or
   5 (MobileNetV2) stride-2 sites the certified program reads its window one pixel from where
   the shipped program does. Two honest resolutions, and the choice is the user's:
   (a) re-spell the Proofs chains at `flatConvStride2Xla` / `depthwiseStride2FlatXla` (the
   definitions and VJPs have existed since `3d9b14d`; the float tier needs the two `Xla` leaves,
   whose envelopes equal the symmetric ones) and re-tie the PC graphs to the XLA ops, or (b)
   leave the spelling and disclose. **Decided 2026-09-05: (a), and done the same day** in five
   commits (`ec977de`, `0773e20`, `0584ab8`, the B0 commit, `ee81d36`) plus the scalar-BN twin,
   scoped in `planning/archive/xla_same_respell_and_blueprint_audit.md`. The odd-phase backwards, their
   float leaves and the four `Maps` envelopes were built first; then B0's stem cone and
   MobileNetV2's five sites, the latter with a one-off name map
   (`scripts/respell_mnv2_xla.py`). All four numbers reproduced to the digit. Four certs are
   shared across net boundaries — ResNet-34 reuses MobileNetV2's stem pair, EfficientNet-B0 its
   strided-depthwise pair, MobileNetV4 borrowed B0's `stemB` — and each is right for one net and
   wrong for the other the moment the conventions diverge; they kept their symmetric statements
   and gained `_xla_` twins (`stemB` gained a separate `fusedConvB`). `formalization.yaml` 4d
   now records the item closed. Left open: the blueprint audit, and item 3 below.

3. **EfficientNet-B0's whole-net certified tie. Done 2026-09-05**
   (`EfficientNetWholeBackCertifiedTie.lean`), unblocked the same day by the re-spelling
   (item 2): at the symmetric stem the tie would have certified a net no shipped artifact runs.
   `efficientnetInputGradB`, with its stem and head BatchNorm and swish slots pinned to the
   certified per-op backwards and its three MBConv blocks opaque, IS
   `(efficientnetB_has_vjp …).backward` — **at every batch size**, not the `N = 1` scoped below:
   that restriction is the FLOAT chain's, and the certified chain's BatchNorm slot is
   `bnBatchLA_has_vjp`, which exists for all `N`. `7.104e182 / 1.578e182` unchanged; no drift
   found, as with MobileNetV2 and unlike ResNet-34.

   Two findings worth carrying. **(a) The `▸` in `batchMap_has_vjp` does not block the
   reduction** — the scoping note below said it would. §5's trap is real for a transport along an
   equation that is not pointwise `rfl`; `batchMap_eq_rowwiseFlat` holds by `funext … ; rfl`, and
   proof irrelevance is definitional, so `.backward` reduces straight through it and both stage
   ties close by `rw` + `rfl`. **(b) What bites instead is size.** The same transport at the
   WHOLE-NET type (`efficientnetForwardB_has_vjp_committed`, which is where the shape check
   earns its keep) typechecks but cannot be reduced through inside the kernel's deterministic
   budget, and neither can instantiating the tie's three block slots at the concrete MBConv
   blocks. Hence opaque blocks, the MobileNetV2 discipline, and the shape check as the thing
   that says which net they are.

   The original scoping, kept for the record: aim at `efficientnetForwardB_has_vjp`
   (`Nets/EfficientNet/EfficientNetChainClose.lean`, `HasVJP` everywhere), not the 16-block
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

   **And at the paper depth, 2026-09-05** (`EfficientNetFullWholeBackCertifiedTie.lean`, package
   3.3(c)): the sixteen-block chain `efficientnetInputGradB_full` is tied to the generic
   eighteen-stage apex with the blocks opaque, then — the step the three-block file could not
   take — instantiated at the concrete `mb*W` blocks and carried to
   `efficientnetForwardB_full_has_vjp` by `HasVJP.backward_unique`, and read through
   `efficientnetForwardB_full_has_vjp_correct` as the `pdiv`-contracted Jacobian of the committed
   `efficientnetForwardB_full`. ~3 s. No number at sixteen blocks (finding 7).

4. **Freeze.** The blueprint audit in `planning/archive/xla_same_respell_and_blueprint_audit.md` step 8
   adds the table in section 1 and the sentence at the end of section 2 to the blueprint, which
   currently mentions the float budgets in one clause and tables none of them. After that this
   document changes only when a number moves.

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
* **The sixteen-block B0 backward number and the seventeen-block MobileNetV2 backward number.**
  Statable since finding 5's correction (9.112e2648 and 1.246e323 at the shipped leaves; the
  threshold is an option). Declined: a number that says nothing at 1e182 says nothing at 1e2648,
  and the sixteen-block chain has its certified tie without one.
* **ViT-Tiny's backward. MEASURED 2026-09-05 and declined; see finding 8.** `vit_back_chain` /
  `verify_vit_back` put it at window 5.686e399, budget 1.703e399, ratio 0.30 — a FOLD at the
  eps-floor with NO operating point, 195 stages, 390 rounded inequalities re-asserted. Statable
  under `set_option exponentiation.threshold 500` (finding 5). Declined on the same ground as
  B0's and MobileNetV2's backwards: it says nothing, and tier T6 landed without it
  (`vitInputGradK_eq_vitForwardKV_vjp`). The cost is also the largest of the three — ViT is the
  one net whose backward needs `Maps` leaves nothing else uses (the three sdpa cores, the
  patch-embed backward, a per-token `perRowPR` lift), on top of the tier migration
  `MhsaBackFloatBridge.lean` still wants (8 `floatBridgesTo_` against 46 `floatBridges_`).
  ⛔ The scoping's predicted CAP is wrong and could not have been right: no backward stage has a
  window bounded by a constant, so `Maps.capped` has nothing to attach to. Caps are a
  forward-only instrument, which is finding 2 read the other way.
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
* `norm_num`'s "ceiling" is `exponentiation.threshold` (default 256): a `10 ^ e` literal with
  `e > 256` is left unevaluated and the goal stays open, whatever the tree's shape. Heartbeats,
  recursion depth, `ring_nf`, `nlinarith` and `simp only` first do not move it;
  `set_option exponentiation.threshold 400 in` does, at no cost (`b0FullEvalBridge_maps`). Prefer
  it to an operating point taken only to get under 1e253.
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
