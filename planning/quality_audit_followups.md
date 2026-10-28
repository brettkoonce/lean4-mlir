# Quality-audit follow-ups: what the 2026-10-28 pass left undone

The lean-quality-indicators runs of 2026-10-28 (Rémy Degenne's tool, LLM judges reading the
source) and the cleanup they drove. This lists only what is **not** done. Done work is in the
commits from `ce27bf2a` to `ad9af260`. The page with every finding and its triage is the
"lean4-mlir quality audit" artifact.

Line numbers are at `f752c7b4` unless a later commit is named.

**Status.**
- Seven commits since `03bfd879` are on local `main` and not pushed: the bug, statement and
  docstring fixes, and the four refactor groups.
- Every gate is green on that tree. All 399 artifacts are byte-identical, and `MlirCodegen`
  output is identical across 306 specs.

## 1. Decisions waiting on you

1. **The VJP oracle cannot see a mis-scaled backward.** `LeanMlir/VjpOracleNets.lean:26` runs one
   bias-corrected Adam step and `run.sh` diffs only the step-2 loss. Adam's first step is about
   `lr·sign(g)`, so a backward wrong by any positive scale (a missing `1/B`, a BN scale) still
   passes.
   - Fix: switch to SGD and diff the parameters, as `run_verified.sh` does.
   - This changes the test, and the JAX side has to follow.
   - Alternatively, narrow the `denseOnly` claim to "gradient signs plus forward".
2. **The CNN float rung's max-pool hypothesis excludes common inputs.** At
   `Proofs/Training/SgdDescent/CnnFloat.lean:608`, `MaxPool2MarginQ δ (relu …)` needs `2δ < 0` on
   every 2×2 window that is all zero after ReLU. The float conv rungs, and their cot/grad-close
   lemmas, therefore say nothing about such inputs.
   - Fix: weaken to `MaxPool2MarginQUpTo`, which admits dead windows, and thread it through the
     float lemmas.
   - Or, at minimum, state the exclusion in the header (lines 23–27).
3. **`r50FpnDet`'s stem is padded differently from its backbone.** At `ReferenceNets.lean:249`,
   `convPadStyle := .symmetric` is read only by the JAX emitter. On the `MlirCodegen` path the
   7×7/s2 stem stays (2,3), so the bootstrapped A3 backbone runs one pixel shifted.
   - The comment now says so (`728bb7b5`); the behaviour is unchanged.
   - Fix: have `MlirCodegen.samePad` honour `convPadStyle` at the stem. That changes the
     detector's train step.
4. **`Attack.lean` selects its checkpoint on the split it reports.** The best checkpoint is
   chosen by accuracy on `evalImg`/`evalLbl`. Clean, PGD and certified accuracy are then printed
   on that same split as "clean test acc (best epoch)".
   - Fix: hold out a selection split, or report the last epoch.
   - Check whether any of those numbers reach the book.
5. **The cifar8-bn step ties' opening lines still read as loss-driven.** The re-audit still
   flagged `Cifar8BnTie.cifar8Bn_train_step_tied` and `Cifar8BnTieG.cifar8Bn_train_step_tiedG`.
   Both carry the "What it does not say on its own" paragraph (`a878ae48`), but their first
   sentences still say the ops are tied to "the rendered backward chain driven by the
   softmax-CE cotangent `g`". This is a wording call on the headline sentence.
6. **The pooled 49-dim certificate model.** See `planning/pooled_cert_removal.md`: its four
   decisions are still open. `Trained/MlpWitness.lean`'s "Level 1" result is also vacuous
   (`HasVJPAt` is inhabited by `canonical` for any `f`). That file goes with the removal, so it
   was left.

## 2. Latent bugs (no current caller hits them)

| where | what | fix |
|---|---|---|
| `Proofs/Codegen/ConvNeXtRenderB.lean:604` | `bStr`, the ÷B in the smoothed cotangent, is a free string independent of `bB`. `"32.0"` with `bB := 64` renders a shape-correct step with 2× gradients. | Derive it from `bB` when `bStr` is empty, as `negAK` is derived. |
| `MlirCodegen.lean:3099` | `generateForwardCam` on a spec with no GAP or a non-dense last layer emits a no-op replace or `tensor<Bx0xf32>`. Callers check `preGAPShape` only after writing. | Return the `#unsupported_spec` stub. |
| `Proofs/Codegen/MobileNetV2RenderB.lean:480` | `ls{(α·100).round}` names every α < 0.005 (and negatives) `ls0`, colliding with α = 0. | A lossless spelling, or reject α off the 0.01 grid, with a `#guard`. |
| `Spec.lean:445` | Cosmetic. The `fusedMbConv` arm prints `,relu`, and `convNextStage`/`convNextDownsample` drop norm/act, so the ablation's BN/ReLU variants print the LN/GELU architecture string. | Print norm/act when not the defaults. |
| `Proofs/Codegen/CnnRender.lean:882` | `invBStr` (the cotangent's `1/B`) is a free string. It defaults independently of `B`; every call site passes it at `B = 128`. | Drop the default, or derive it from `B`. |
| `Verified/Attack.lean` (MLP drivers) | `attackPgdMlp`/`attackPgdSpectralMlp` hard-code hidden width 512 and slice θ at fixed offsets without checking `net.specs`. | Read the widths from `net.specs`, or refuse others. |

## 3. Docstring fixes that wanted code

The prose was corrected in `5d200dd9`. The stronger, code-side fix each note preferred was not
made.

- **Guards or `panic!`s:**
  - ConvNeXtRender `sd` without drop sites should fail, not render with no drop path.
  - The `d0` invariant guards on the S/B ConvNeXt/ViT specs (`NetsCore`).
  - `dropDenom ≥ 1` (`Spec.lean`).
  - PgdGen's xorshift needs `n ≥ 3`.
  - The `delay` refresh period (`Pong`/`TicTacToe`).
  - MobileNetV2RenderB's length-only guards should compare shapes.
  - The ResNet-50 render's decay-operand counts.
  - MobileNetV4FullB's literal-arithmetic `#guard`s.
- **Ties and lemmas:**
  - `hbStr`, tying `BceLossCot`'s printed divisor to `N·K`.
  - A theorem tying `BackwardMaps.mlpInputGrad` to `mlpHasVJPAt`. The "proven" wording in
    `lakefile.lean` and `apps/mnist/MainMnistMlpPgd.lean` is still unqualified.
  - Adam's `wd = 0` corollary.
  - `skel_castIdx`.
  - `momStep_lr_zero`.
  - The converse direction in `JacobianSeal`, giving an iff.
  - The stride-2 conv spec lemma (`StridedConv`).
  - The RMSProp non-affinity counterexample.
- **Renames:** `lrStr` in `CnnRender` (it carries `lr/B`).
- **Emitted text:**
  - `IRPrint`'s `// PROOF-BACKED` labels, now defined in the header but unchanged in the output.
  - CnnBbnRender's banner naming "proved as" theorems that don't exist (a call-site comment says
    so).
- **Nets:**
  - `cnnConcreteHasVJPAt`'s zero head, which makes its Jacobian trivially 0. Setting `Wd := 1`
    would give it content.
  - A pinning `example` for ViT-Tiny's depth in `ViTBackNet`.

## 4. Refactors not done

- **Skipped on inspection:**
  - `linear_loss_gradAt_sum`: half its uses are in the generated `Trained/LinearDescent.lean`,
    so it belongs to `trained_linear_descent.py`.
  - `pdiv_tileAff`: saves about 6 lines and needs row-selector and constant parameters.
  - The depthwise flatten split: each site still states its own `hsplit`.
- **Narrowed:**
  - The max-pool flat VJP lemma was applied everywhere except `Training/SgdDescent/Cnn.lean`.
  - `CertLayer.backward_eq_den_graph` applies at five MobileNetV4 sites. The MobileNetV2,
    ResNet-34 and ResNet-50 `*CotIn_eq_vjp` proofs go through per-block `*_faithful` theorems
    instead.
  - `HasVJPDiff` is used only at ConvNeXt-T's top-level chain. The EfficientNet ladder and
    `cnxBodyWithHasVJP` have no duplicated differentiability twin.
- **Declined earlier, and the re-audit flagged them again:**
  - The BackLinks `*_back_eq` proofs.
  - The Cifar8Bbn germ stages.
  - MobileNetV2's stride-1/stride-2 bundles.
- **Rated "maybe"** (80): see Appendix B. The ones worth a second look are:
  - The sync-BN shard step: one SyncKit lemma for about 60 sites.
  - The dense+ReLU head pullback, repeated in five small ParamGrad files.
  - The plain and dropout trunk duplicates (MobileNetV4, EfficientNet, ViT).
  - `BatchMapVJPAt`'s uniform versus indexed lift, together with the `batchMapIdx` move
    already made.

## 5. Coverage not run

- **Full-file pass, other rubrics.** It ran only `repeated-proof-machinery` and
  `vacuity-and-misstatement`. `statement-generality` and `readable-proof-simplification`
  ran only on the 20-file sample. A full run is about 560 Sonnet sessions.
- **Unnecessary hypotheses on every theorem.** It ran on 100 sampled theorems; the full
  population is 3,991, about 4,000 Opus sessions. Expect 150–200 claims, many of them "true but
  not worth it".
- **Scope.** Only `LeanMlir/` was audited. `apps/`, `demos/` and `tests/` were not, and neither
  were private declarations.
- **Not exercised.** The runtime-driver refactors in `ad9af260` (Attack, the trainers,
  SyncBnCheck, E4M3Quant) were checked by reading and by building. None has run on a GPU since.

## 6. CI

- **Comparator's critical path.** After the four-leg split and the shared `build` job, the
  tier lean4lean leg is the critical path, at 702 s. A different layout (lean4lean alone, the
  default kernel elsewhere) would shorten it. Not worth it yet.
- **Leftover branches.** The four `worktree-agent-*` branches still hold the refactor commits as
  the agents wrote them, before they were cherry-picked onto `main`. Delete them once the
  refactors are pushed.

## Appendix A — design calls from the vacuity/misstatement pass (62)

True observations the second reader judged to be documented choices or the maintainer's call.
Line numbers at `f752c7b4`.

| where | what the call is |
|---|---|
| `Proofs/Architectures/CNN.lean:716` | The docstring itself (lines 719-721) says no theorem states the equation and that the formula is the sum the backward is defined as, so nothing is hidden. Adding the one-line `conv2dBiasGrad … = conv2dBiasGradFormula dy` theorem is cheap and would let the IRPrint citation stand; maintainer call. |
| `Proofs/Architectures/ConvGrad.lean:486` | True that the bound uses the full spatial count c·(2h)·(2w) rather than the local c·kH·kW the section header (l.329-333) advertises, but the lemma's own docstring calls it '(loose, uniform)'. Maintainer call: tighten via convTap_out_l1 for CnnFloat.lean:1013 or leave it as the documented loose bound. |
| `Proofs/Architectures/Depthwise.lean:460` | True that no theorem ties depthwiseConv2dBiasGradFormula to depthwiseConv2dBiasGrad, but the docstring itself says 'No theorem in this file states the equation'. Maintainer call: add the cheap _eq_formula lemma or delete the unused def and its summary line (l.544). |
| `Proofs/Architectures/LayerNorm.lean:316` | The docstring already says it is 'a rewrite by layerNormVec_gamma_grad_bridge', so nothing is hidden. The `_sgd_certified` suffix is a repo convention across 15 theorems. Call: whether to extend the a878ae48 `_certified`→`_tied` rename to this family. |
| `Proofs/Architectures/StridedConv.lean:341` | The odd-k>=3 domain is stated in the adjacent block (lines 333-336), and the printer emits MALFORMED outside it. Maintainer's call: optionally lift it to a Lean predicate or hypothesis on the Xla-named defs. |
| `Proofs/Codegen/ChapterGraphs.lean:137` | True that `epsStr` is printed and `ε₁..ε₈` only feed `den`, but that is the repo's uniform skel-erasure of values/literals (same as `lrStr`, weights); `cifar8BnFwdGraph_faithful` is ∀ ε, so the call site's `0` placeholders are irrelevant and the only gap is the string↔real literal, which is trusted everywhere. Collapsing to one ε is a tidy-up, not a soundness fix. |
| `Proofs/Codegen/CnnRender.lean:882` | True that `invBStr` defaults to 1/128 independently of `B`, but every `some _` call site (CnnArtifacts.lean:254, :325) passes it explicitly at B=128, so nothing hits it today. Maintainer call: drop the default (as the sibling packed renders do) or derive it from `B`, in line with the signature comment's own 'state the hyperparameters' rule. |
| `Proofs/Codegen/ConvNeXtRenderB.lean:299` | The 'multiple of 32' precondition is documented on the parameter (line 299). The only non-default call passes 288 = 9·32, so a bad s fails loudly at lowering. Call: optionally declare %x from 32*(s/32), or reject s % 32 ≠ 0. |
| `Proofs/Codegen/EfficientNetRender/Basic.lean:136` | True and latent: with adam=false, bnG silently drops `sync`/`st` and uses the local-batch bnGammaSgdB, while enetBackAll accepts adam=false together with sync=true. The only SGD caller (line 1053) leaves sync at its default. Call: refuse or assert `adam ∨ ¬sync` in enetBackAll. |
| `Proofs/Codegen/EfficientNetRender/Basic.lean:1163` | The precondition (no accumulating recipe) is documented right on the `opt` parameter (line 1163) and no caller passes adamwAccum/lambAccum. Maintainer call: enforce it, either by stubbing/erroring on `.adamwAccum _ \| .lambAccum _` or by building the signature with acc := true. |
| `Proofs/Codegen/RenderKit.lean:339` | True that `adamwAccum 0`/`lambAccum 0` would bake saturated `fmt12` constants and a zero clip, but no call site constructs k = 0. Cheap hardening: panic/throw on k = 0 in `accumScalarConsts`/`clipNormStr`, or document k ≥ 1. |
| `Proofs/Codegen/ResNet34RenderB.lean:643` | True that `panic!` in the `optLabel` match prints an error and returns `""` rather than aborting, so rendering continues to `optOne` with the unsupported recipe; no committed `#eval` passes these recipes, so it is a latent hard-stop gap. Maintainer call: return `Except`/throw in the writers, or narrow the parameter type; also move the comment block from above `.lamb` to the refusing arms. |
| `Proofs/Codegen/ViTRender.lean:680` | True that with traversal := none a non-Tiny V gives a V-wide signature over Tiny-wide gradient ops. Today only ViTRenderB passes V, always with its traversal, and the doc does not claim V works alone. Maintainer's call: a cheap panic! when traversal.isNone && V != vitTiDims would make the misuse loud. |
| `Proofs/Foundation/DataParallel/SyncBf16.lean:193` | True that no lemma exhibits a nonzero difference at rnd = rndP 7; the file proves the exact difference and its vanishing at id, and 'rounded per replica' is the table's claim. Optional: add a small separating witness (∑ rnd S_r ≠ rnd ∑ S_r for rndP 7) or soften the title's 'but' to 'differs by exactly this defect'. |
| `Proofs/Foundation/MLP.lean:320` | True that mlpHasVJP/reluHasVJP are HasVJP.canonical and their _correct theorems are rfl, but both docstrings say so ('canonical witness … by definition', l.311-318, 362-363) and mlpHasVJPAt is the substantive version. Maintainer call: whether the comparator contract (config-arch.json chk_mlpHasVJP_correct/chk_reluHasVJP_correct) should keep these definitional entries or swap in an explicit-backward statement. |
| `Proofs/Nets/ConvNeXt/ConvNeXtFullT.lean:280` | Same `HasVJP` convention as Foundation/Tensor.lean l.274-278 (canonical exists; content is the constructed `backward`); hypotheses feed `vjpComp`'s differentiability. Not a per-file gap. |
| `Proofs/Nets/ConvNeXt/ConvNeXtStepTie.lean:631` | The docstring (l.627-630) says each chain cotangent is the certified VJP backward and cites `CnxTieGB.cnxBlockCotInChAt_eq_vjp`/`cnxDownCotInChAt_eq_vjp`/`cnxHeadDyXheadChN_eq_vjp`, i.e. the documented split; `0 < ε` lives there. |
| `Proofs/Nets/ConvNeXt/ConvNeXtStepTieGB.lean:504` | The docstring (l.484-488, 495-497) says the folds are `∀ cot` statements at constructed cotangents and cites `cnxHeadDyB_eq_vjp`/`cnxBlockCotInB_eq_vjp`/`cnxDownCotInB_eq_vjp` for the VJP identification; no `0 < ε` by design. |
| `Proofs/Nets/EfficientNet/EfficientNetFullWholeBackCertifiedTie.lean:223` | The header (l.18-21) says the sixteen blocks enter as `HasVJP` witnesses whose `.backward` the chain's block slots are pinned to; only the `_correct` docstring's 'hand-written sixteen-block chain' is loose. Optional: say 'the chain with hand-written stem/head and witness block slots'. |
| `Proofs/Nets/EfficientNet/EfficientNetStepTie.lean:71` | The predicate docstring (l.67-70) says the RHS is a BN β-Jacobian at zero γ/activation standing in for the channel sum; the conv/depthwise bias link to the forward is made at the G nodes by `GradNodeB.biasBeta_hasGradAt` in EfficientNetParamGrad (l.37, 404-415). Same deliberate shape as ResNet34StepTieB's head bias (l.393-397). |
| `Proofs/Nets/EfficientNet/EfficientNetStepTieG.lean:90` | The predicate docstring (l.86-89) says the RHS is the channel sum written as a BN β-Jacobian; EfficientNetParamGrad (l.37, 404-415) makes it the conv/depthwise bias gradient via `GradNodeB.biasBeta_hasGradAt`. Renaming is optional. |
| `Proofs/Nets/EfficientNet/EfficientNetStepTieG.lean:450` | The docstring (l.440-450) claims only Σ pdiv·cot 'at the cotangent that chain delivers'; the loss link (with B, t, and the head/dy16 consistency) is EfficientNetParamGrad's composition of `efficientnet_net_tiedG` (l.1235-1300, 1535). Optional: name that theorem here. |
| `Proofs/Nets/EfficientNet/EfficientNetSyncB.lean:23` | Header l.29-30 says each conv's shard step 'erases the switch through Bf16Erasure (`denOp_convAt_id`)', i.e. the repo-wide 'bf16 kinds read over ℝ at the identity rounding' convention. Optional: use the sibling wording 'read over ℝ at the identity rounding' verbatim. |
| `Proofs/Nets/MobileNet/MobileNetV2FullB.lean:418` | `epsStr` is an uninterpreted label, the same repo-wide convention as all name strings; the text is checked against the render separately. Per-site ε fields are real data the shipped weights fill; relating literal to real is the per-op text trust. |
| `Proofs/Nets/MobileNet/MobileNetV2FullPaperEval.lean:338` | The module doc (l.44-50, 67) says the artifacts are 'the zero-bias instance' and that at inference BN this is a restriction. Adding a `ConvBiasesZero` hypothesis or corollary is a maintainer call. |
| `Proofs/Nets/MobileNet/MobileNetV2FullPaperEval.lean:338` | `epsStr` and every name string are uninterpreted labels (`den` ignores strings, repo-wide); the text side is checked against the render by FwdGraphTextTies/CI, and `0 < ε` is not needed for a forward-denotation equation. |
| `Proofs/Nets/MobileNet/MobileNetV2StepTieB.lean:827` | The docstring (l.812-816) says the folds are `∀ cot` at constructed cotangents and that the `*_eq_vjp` lemmas carry the identification and hypotheses; `g` is a declared binder. This is the documented split. |
| `Proofs/Nets/MobileNet/MobileNetV2WholeBackCertifiedTieB.lean:366` | The header (l.17-44) documents opaque blocks, `mobilenetv2ForwardBFull_eq_slots` as the shape check, and the kernel-timeout reason for not instantiating concrete blocks. |
| `Proofs/Nets/MobileNet/MobileNetV4BackB0.lean:605` | The docstring (l.603-605) says the strided post-DW must exist and is 'guarded above': `#guard (mnv4Blocks.filter (fun s => s.stride2 && s.postDWk != 0)).length = 3` (l.462). Total over `UibSpec` and applied only at concrete rows; adding `postDWk ≠ 0` binders is a maintainer call. |
| `Proofs/Nets/MobileNet/MobileNetV4FullBEval.lean:220` | Generic in `UibSpec` and applied only at Conv-M's strided rows, which have `postDWk = 5`, guarded in MobileNetV4BackB0 l.462 (`stride2 && postDWk != 0` count = 3). Optional: cite that guard in the `mnv4StridedEval` docstring. |
| `Proofs/Nets/MobileNet/MobileNetV4FullBEval.lean:298` | The module doc (l.38-42) says 'The artifacts are the zero-bias instance… a restriction, not a reparametrisation.' A `ZeroConvBias` hypothesis or corollary would be a maintainer call. |
| `Proofs/Nets/MobileNet/MobileNetV4StepTieB.lean:954` | The docstring (l.943-947, 952-954) cites the `*_eq_vjp` lemmas as the VJP identification and says the folds are `∀ cot` at constructed cotangents with conditions in the `CertLayer.ok`s, i.e. the documented split. |
| `Proofs/Nets/MobileNet/MobileNetV4WholeBackCertifiedTieB.lean:551` | The header (l.26-61) documents opaque blocks, `mobilenetv4ForwardBFull_eq_slots` as the separate shape check, and the kernel-timeout reason there's no concrete instantiation. |
| `Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:355` | The header (l.17-35, build notes l.57-69) says the sixteen blocks are opaque `HasVJPDiffAt` witnesses, that `resnet34ForwardBFull_eq_slots` is the separate shape check, and that instantiating at concrete blocks is a kernel timeout. Only the title 'IS the certified whole-net gradient' is loose. |
| `Proofs/Nets/ResNet/ResNet34FullBVJP.lean:482` | The docstring (l.472-473) says the VJP is 'read at the layered chain', and `_correct` (l.488-499) states it for `resnet34ForwardBFull` itself; the chain-form type avoids the defeq walls noted in ResNet34BackCertifiedTieB l.57-60. Restating the headline at the committed forward is a maintainer call. |
| `Proofs/Nets/ResNet/ResNet34StepTieB.lean:418` | The docstring (l.420-424) says the folds are `∀ cot` instantiated at constructed cotangents and the VJP identification is `r34IdCotIn_eq_vjp`/`r34DownCotIn_eq_vjp`, 'the two halves kept apart'; the composed loss-gradient statement lives in ResNet34ParamGrad. |
| `Proofs/Nets/ResNet/ResNet50StepTieB.lean:501` | The docstring (l.515-519) states the folds are `∀ cot` at constructed cotangents and that `r50*CotIn_eq_vjp` are 'the two halves of the tie, kept apart'; the composition is `r50_net_lossGrad` in ResNet50ParamGrad. Optional: name `r50_net_lossGrad` in this docstring. |
| `Proofs/Nets/ResNet/ResNet50SyncStepTieB.lean:609` | The module docstring says the bf16 kinds are read at the identity rounding (judge concedes), the repo-wide `Bf16Erasure` convention. Optional: repeat the phrase in the `r50_net_syncTiedB` docstring. |
| `Proofs/Nets/ResNet/ResNet50WholeBackCertifiedTieB.lean:149` | The header (l.25-31, 44, build note l.57-61) documents opaque blocks, no `backward_unique` step because of the kernel timeout, and `resnet50ForwardBFull_eq_slots` as the separate shape check; only the title wording is loose. |
| `Proofs/Nets/Small/Cifar8BbnStepTieB.lean:83` | The module doc (l.36-43) says 'The other half — that this chain IS the certified whole-net backward — is Cifar8BbnWholeBackB.lean', and the loss link with `invB`/`0 < ε` is Cifar8BbnParamGrad. The per-node tie is deliberately hypothesis-free. |
| `Proofs/Nets/Small/Cifar8BnStepTie.lean:56` | The docstring itself (l.66-68) says each conjunct holds for any cotangent and that the loss-gradient claim is `Cifar8BnTieG.cifar8Bn_net_lossGrad`; the judge repeats the documented split. |
| `Proofs/Nets/Small/Cifar8BnStepTieG.lean:36` | True that every GradNode clause holds for any cotangent, but the module Scope (l.21-26) names `cifar8Bn_net_lossGrad` as the loss-gradient statement and the docstring only claims Jacobian·chain-cotangent. Optional: copy the sibling's 'What it does not say on its own' paragraph (Cifar8BnStepTie l.66-68) into this docstring. |
| `Proofs/Nets/Small/Cifar8StepTie.lean:44` | The docstring (l.55-57) says each conjunct holds for any cotangent and that the loss-gradient claim is `Cifar8TieG.cifar8_net_lossGrad`; the finding repeats the documented split. |
| `Proofs/Nets/Small/Cifar8StepTieG.lean:87` | Same as 0007:0: the module Scope (l.17-23) names `cifar8_net_lossGrad` as the loss-gradient statement and the docstring claims only Jacobian·chain-cotangent. Optional: add the 'What it does not say on its own' paragraph from Cifar8StepTie l.55-57. |
| `Proofs/Nets/Small/CifarCNN.lean:81` | True that `HasVJPAt` is inhabited by `canonical`, but Foundation/Tensor.lean l.274-278 documents that a witness says something only through its `backward` term; here that term is the vjpCompAt chain by definition (a transparent noncomputable def). Repo-wide convention, not a per-file gap. |
| `Proofs/Nets/Small/CifarCNN.lean:328` | The Tiny instance's job is to show the eight smoothness hypotheses are jointly dischargeable, which it does (l.328-361); `_correct` is just `.correct`, same `HasVJP` convention as Foundation/Tensor.lean l.274-278. Wording 'unconditional' refers to the side conditions and is accurate. |
| `Proofs/Nets/Small/CifarCNN.lean:985` | Same `HasVJPAt` convention: `_correct` restates `.correct` of the composed witness (l.985-1011); content lives in the def's backward term. No concrete Bn8 instance is a scope choice, not a misstatement. |
| `Proofs/Nets/Small/CifarFold.lean:195` | The docstring (l.191-193) says each conjunct holds for any cotangent and that the loss-gradient claim is `cifar_net_lossGrad`; the finding repeats the documented split. |
| `Proofs/Nets/Small/CnnChainClose.lean:97` | The module doc (l.24-28) states these are the generic bridges instantiated at the chain and that equality with the loss gradient 'is not stated'; that link is `cnn_net_lossGrad`. |
| `Proofs/Nets/Small/CnnFold.lean:247` | The docstring (l.243-245) explicitly says each conjunct holds for any cotangent and the loss-gradient claim is `cnn_net_lossGrad`; the finding repeats the documented split. |
| `Proofs/Nets/Small/LinearFold.lean:95` | True that `lrStr` and `lr` are unrelated binders, but strings are uninterpreted labels repo-wide (`den` ignores them) and the printed-op ↔ den correspondence is the listed per-op trust (Scope l.40-44), with committed bytes guarded by CI. Optional: name 'the printed lr literal denotes lr' explicitly in Scope. |
| `Proofs/Nets/Small/LinearFoldB.lean:176` | `lrStr` is an uninterpreted label like every name string (`den` ignores it); 0.00078125 = 0.1/128 is the docstring's reading at `α = 0.1, N = 128`, and the printed literal is part of the per-op text trust. A concrete-instance corollary is optional. |
| `Proofs/Nets/ViT/ViTStepTie.lean:344` | The docstring (l.338-340) says each chain cotangent is the certified VJP backward and cites `vitBlockCotInAtMHV_eq_vjp`/`vitCotTowerOutV_eq_vjp`, which carry `0 < ε`. That's the documented split. |
| `Proofs/Nets/ViT/ViTStepTieGB.lean:449` | The docstring (l.449-452) cites `vitCotTowerOutB_eq_vjp`/`vitBlockCotInB_eq_vjp` as where each chain cotangent is identified with the lifted VJP backward. That's the documented split; `0 < ε` lives in those lemmas. |
| `Proofs/Training/Optim/Accum.lean:107` | True that `*_eq_mean` take an arbitrary `Gt`/`k` and no theorem composes them with `accumCycle_sum`, but the module header (lines 24-28) scopes them as 'ring identities over ℝ, nothing more' and the composition is immediate. Optional: add the one-line composed corollary with `k = (g::gs).length`. |
| `Proofs/Training/Trained/CnnDescent.lean:16` | 'Classified correctly' and 'trained' are generator-side facts. The header says the weights and input are DATA from scripts/certs/trained_cnn_descent.py and that the generator prints the accuracy. Maintainer's call: attribute the claim ('per the generator') or add a logit lemma. |
| `Proofs/Training/Trained/CnnDescent.lean:2224` | The module header (lines 37-40) states that the decrease is not shown positive and explains why (the gradient's lower bound is transcendental). Maintainer's call: optionally say 'by at least lr·‖∇L‖²/2 (non-strict)' in the theorem docstring too. |
| `Proofs/Training/Trained/CnnDescentConv1.lean:2205` | The header (lines 33-34) already says the decrease lr*‖∇L‖²/2 'is not shown positive', and the weights and input are labelled DATA from the generator. 'Classified correctly' is the generator's claim. Maintainer's call: optionally reword 'decreases' to 'does not increase, by at least …' and attribute 'classified correctly' to the generator. |
| `Spec.lean:608` | validate checks channel chaining only and skips fan-in after flatten because spatial extent is not tracked. That is a scope limit, not a bug in what it checks. Maintainer's call: add 'fan-in after flatten not checked' to the docstring, or track H·W. |
| `SyncBnCheck.lean:309` | The header (lines 23-25) states the verdict is on the BN statistics and the DUPLICATED collective (whose dupNormRel does gate the gradient slot), with the other gradient columns printed against SENSITIVITY by design. The '— R×b IS 1×Rb' tail of the success line is a bit strong. Maintainer's call: optionally append 'on the statistics'. |
| `Verified/Attack.lean:412` | True that the guard is a kind!=1 blacklist that would admit kind 3/4/5/8 rank-1 scales, but its docstring (lines 412-416) states exactly that limitation and that today's callers (BN-free MNIST/CIFAR CNNs) have none. A whitelist (rank-1 only kind 2, rank-2/4 only kind 0) is cheap hardening; note `certProduct`'s 'Sound only for nets requireCertModelled accepts' (line 426) slightly overstates given that admitted gap. |
| `Verified/Attack.lean:497` | True: both conv drivers select the best epoch on the eval split that then reports clean/cert/PGD numbers, and the spectral driver ranks the unprojected iterate before the final `projectSpectral` (line 643; the cert/clean numbers are then computed on the projected net, so they are self-consistent). Methodology call: select on a train-tail holdout or say 'selected on the eval split' in the printed label/docstring. |

## Appendix B — refactors rated "maybe" (80)

Real repetition, but a modest gain or an awkward abstraction. Line numbers at `f752c7b4`.

| where | trade-off |
|---|---|
| `Proofs/Architectures/Attention.lean:573` | The repetition is real: both chains rebuild scale→rowSoftmax→·V with three vjpMatComp calls and restated diff terms. A `sdpaTailHasVJP F hF vF` would save ~15 lines, but sdpaBackQ/K_correct close by `unfold … rfl` through these defs, so the refactor risks defeq/unfold churn for a modest gain. |
| `Proofs/Architectures/Attention.lean:1108` | Two 8-line proofs differ only in the scores map, but `h1` (the fun_prop part) stays per-proof. The shared softmax-then-·V tail lemma saves only ~4 lines and is worth doing only if it is introduced together with 0207:0. |
| `Proofs/Architectures/BatchNorm.lean:723` | bnCentered differentiability is proved twice (fun_prop in pdiv_bnNormalize, and an h_eq detour in bnHasVJP); bnForward_differentiable does NOT redo the product argument, it just runs fun_prop. Named `bnCentered_differentiable`/`bnNormalize_differentiable` lemmas are natural and save ~8 lines. |
| `Proofs/Architectures/CNN.lean:642` | The conv-affine-in-bias split (`flatten (conv2d W b x) = flatten (conv2d W 0 x) + broadcast b`) is proved 3x in CNN.lean (plus depthwise peers), 4-5 lines each. A named `flatten_conv2d_eq_add_bias` saves ~8 lines and names the broadcast, but each site still needs a `funext` adapter. |
| `Proofs/Architectures/ConvGrad.lean:676` | Real: the card ≤ 1 pinned-position block (`sum_prod_type'`/`sum_filter`/`card_le_one`/`Prod.ext … omega`) is a 6-line copy in `convTap_out_l1` and `conv2d_input_l1_drift`. `sum_pinned_le` saves ~8 lines, and sharing the two `_expand` lemmas is not worth it (different bound variables). |
| `Proofs/Architectures/PerChannelBN.lean:320` | Real, with only two instances: the inner/inner_diff/inner_vjp `vjpComp` conjugation (~13 lines) plus its 2-rw `_backward_eq` appears for `reassoc` and `bnchw`. A generic `hasVJP_reindexConj σ τ f` over `reindexVJP` would also absorb the per-bridge Fwd/Back HasVJP and backward_eq lemmas, but it is a new abstraction for 2 uses (net saving ~15 lines) and needs reassocFwd/bnchwFwd recast as raw gathers. |
| `Proofs/Architectures/SE.lean:241` | This is one def, not repeated machinery across proofs: the nested differentiability chain is restated 4/3/2 times inside seGateHasVJP. Type-inferred local `have h1 := …` names would cut ~8 lines, but this is a local readability tidy. |
| `Proofs/Architectures/StridedConv.lean:136` | There is real plumbing repetition (let f, diff, vjp, decimate) across 8 defs, and the stride-4 ones rebuild `s1_diff` by hand. A `DiffVJP` bundle adds a new structure, and the `_correct` theorems may depend on the current def shapes. A lighter option is a `HasVJP.decimate4`/diff-returning decimate for the two stride-4 defs only. |
| `Proofs/Codegen/ConvNeXtRender.lean:996` | Real: ConvNeXt's clip block hand-copies optAllParams' all-reduce → global fold → clipScale stage (~35 lines) and has already drifted (`%zero` seed, literal "0.000001", panics on clip+acc). But switching changes bytes of the committed `*clip*` artifacts unless the kit's seed name and ε are parameterised, and the in-code comment defends the copy. |
| `Proofs/Codegen/EfficientNetRender/Basic.lean:394` | The fwd tail (dn BN, swish, seFwd, project conv+BN) is a real 5-line triple. An `eFwdTail` would have to return a partially filled EFwd that each caller patches with `{ t with … }`, which reads worse than the explicit records; the saving is ~10 lines. |
| `Proofs/Codegen/IRPrint.lean:1617` | The transpose+reverse+convOp triple really is hand-written ~10 times, so a `renderConvInputVjp` with name/dim parameters would save ~2 lines per site (~20 total) and centralize the reversed-kernel identity. Parametrizing the whole mbconv/invres skeleton over an activation-back function reads worse than the two explicit modules in this legacy, unimported printer. |
| `Proofs/Codegen/MobileNetV2RenderB.lean:252` | The 12-slot `ps` list is written out in `irBackStridedGradB` and `irBackStride1GradB` and duplicates `mnv2BlockSig`'s order. A helper taking 12 SSA-name strings reads worse than the list; the worthwhile version zips names with `mnv2BlockSig`-derived slots, at ~8 lines saved for a design change. |
| `Proofs/Codegen/MobileNetV4RenderB.lean:788` | The stride-1 depthwise+BN backward really is written three times (~11 lines each), with the same statements at (c,h) and (ic,2h). A `dwBnBack` with ~13 positional args saves ~18 lines; it is worth doing if 0262:0's helper lands, so the pair is consistent. |
| `Proofs/Codegen/ResNet50RenderB.lean:316` | Real: the 12-line drop-routing comment, the `match drop`, and bn3→conv3→relu2→bn2 are verbatim in all three emitters. But the W3/g3/bt3 gradients are emitted after the W1/W2 grads, so moving them into a helper would reorder SSA numbering and break byte identity. A byte-safe helper covers only the head (cDa, cDd, n3, c3, r2, n2 plus the comment), saving ~35 lines. |
| `Proofs/Codegen/StableHLO/Pretty.lean:1583` | The μ/var/istd/x̂ prologue really is repeated in ~12 arms, but the fresh-name and statement order differ between arms (e.g. `z` before the reshape in bnGammaSgd, after it in lnRowF). A shared `normStatsText` either changes the SSA text in some arms or needs ordering knobs. The lnRow pair is better covered by 0235:1 first. |
| `Proofs/Codegen/StableHLO/Pretty.lean:2001` | The transpose-trick weight-grad pipeline really is copied ~12 times (~20 lines each). The copies differ in more than the pad, though: upsample geometry (sWGradGeom vs fixed 1), group count, transpose-back vs reshape-back, and output shapes. A single `emitWGradConv` would need ~8 knobs; per-family helpers (conv vs depthwise) plus an `emitSgdTail` are the more readable route. |
| `Proofs/Float/ConvMixedComposeBridge.lean:94` | The claim is correct and the pattern (sum_le_card_nsmul + abs_mul + mul_le_mul) recurs ~8x repo-wide (FloatBridge, Cnn.lean), so `sum_abs_mul_le_card (ha : ∀ k, \|a k\| ≤ w') (hb : ∀ k, \|b k\| ≤ C) : ∑ \|a k * b k\| ≤ n*w'*C` is natural. Each use is a 2-3 line term and the RHS shapes differ elsewhere, so the saving is ~10 lines here and less elsewhere. |
| `Proofs/Float/ConvMixedFloatBridge.lean:120` | The overlap is real: both prove bracket nonneg, `\|p−P\| ≤ br·S` and `\|p\| ≤ (1+br)·S`. Sharing them needs `convBrR`/`convBrR_nonneg` moved into FloatBridge and a `dotMixed_close_abs` lemma, which saves ~15 lines, mostly in dense_close_mixed. |
| `Proofs/Float/DepthwiseFloatBridge.lean:148` | The 2-line `huf` haves (~15 across ConvFloat/DepthwiseFloatBridge/WindowMax/CnnFloat) are defeq to `fun _ _ _ => hv _`, so `Tensor3.unflatten_abs_le`/`unflatten_sub_abs_le` (or just inlining that term) saves ~15 lines total. Tiny per site. |
| `Proofs/Float/FloatBridge.lean:145` | The htri/h1/h2/h3 triangle block really is duplicated between step_bound and dense_step_bound and overlaps add_close. However, step_bound is stated over an abstract `r` (not M.add) and the call sites rely on its packaged budget, so rerouting through add_close plus new algebra lemmas is a nontrivial restructure; the saving is ~30-40 lines, uncertain until it is tried. |
| `Proofs/Float/MlpFloatBridge.lean:349` | `(reluMask_abs_le _ _ j).trans (dense_abs_le h0 (fun j i => hW i j) (fun _ => by simp) hC j)` occurs 13x across MlpFloatBridge/Mlp/CnnFloat. A `reluMask_denseT_abs_le` one-liner is natural, but it saves ~1 line per site because the restated `have` types stay. |
| `Proofs/Float/MlpFloatBridge.lean:445` | Real: w0/b0 share ~25 identical lines (hC₁0…hc₀mag), but `mlp_cot0_close` needs ~13 hypotheses and a ~10-line nested reluMask/dense conclusion, so it saves only ~15 lines net. |
| `Proofs/Foundation/BackwardMaps.lean:230` | Real but small: the two theorems share an ~8-line block (hidx + sum_finProdFinEquiv₃ + triple sum_congr + split). The proposed shared lemma's RHS is a long finProdFinEquiv-decoded triple sum, so the net saving is ~8 lines at the cost of a heavy statement. |
| `Proofs/Foundation/BatchMapVJPAt.lean:57` | Real: ~30 lines (pdiv_batchMap_at, field-by-field HasVJPAt, aux tie) duplicate the batchMapIdx versions, but the uniform lemma reads the row at `idx` while the indexed one reads it at `jdx` (needs a split_ifs bridge), and it requires moving batchMapIdx below BatchMapVJPAt. Worth it only together with 0045:0's restructure. |
| `Proofs/Foundation/Batched/Indexed.lean:139` | Same duplication as 0012:0 from the other side (pdiv/HasVJPAt/aux-tie, ~30 lines). The `idx` vs `jdx` row index in pdiv_batchMap_at needs a small bridge, and `.backward` reduction through the delegated def must be rechecked; do it with 0045:0's file move or not at all. |
| `Proofs/Foundation/DataParallel/Basic.lean:148` | Accurate: `meanLoss_shard` and `dpSyncGrad_eq_globalBatchGrad` share the same 2-line `rw` chain after a `show`, and `dpMean_shardSum` repeats the re-index tail. A scalar `shard_mean_eq e f` makes the docstring's 'meanLoss_shard's arithmetic, for vectors' link explicit, but it saves only ~3 lines. |
| `Proofs/Foundation/DataParallel/Sync.lean:107` | Two 2-line facts (`bnchwFwd y (bnchwBackIdx t) = y t` and `(finProdFinEquiv.symm (bnchwBackIdx t)).1 = bnchwChan t`) are re-derived as local haves in the two _apply lemmas. Naming them is clean, but the saving is ~8 lines. |
| `Proofs/Foundation/DataParallel/SyncBf16.lean:296` | The two bf16 `_smul` bodies (8 lines each) are identical, but the proposal imports SyncKit into SyncBf16 just to reuse 3-line f32 proofs, and the rewrite is unverified. A cheaper route is one batch-sum-homogeneity lemma in Sync.lean (see 0202:0) used by both files, saving ~8 lines. |
| `Proofs/Foundation/HeadLayers.lean:205` | It is real: `batchMapMaxPool3s2FlatReluHasVJPAt` and its differentiableAt and _backward lemmas restate gatherRelu*/gatherReluHasVJPAt_stem_backward at σ = the argmax gather. Each body is only 2–4 lines, so it saves ~6 lines and mostly follows from the reordering in 0210:0 (the HasVJPAt def is used only by its own _backward lemma, so check whether it is dead). |
| `Proofs/Foundation/Muon/Geometry.lean:226` | Real but small: `orthogonal_dot_mulVec (hW : Wᵀ*W = 1) : (W *ᵥ x) ⬝ᵥ (W *ᵥ x) = x ⬝ᵥ x` is a natural lemma, and steepest re-derives muon_polar_orthogonal's first half inline. It saves ~6 lines but needs muon_polar_orthogonal moved above muon_polar_steepest. |
| `Proofs/Foundation/PairTile.lean:149` | Real: the two Pair Jacobians are dense one-line `<;>` simps repeating the finSumFinEquiv case split 4x each. A general `pdiv_appendConst` (an appended constant block leaves the Jacobian on inl and puts 0 on inr) is the right abstraction, but it needs a new proof (affine-hypothesis form or fderiv-of-append), for ~15-20 lines saved. |
| `Proofs/Foundation/ParamGrad.lean:79` | Correct that `comp` = `param` + `vf.correct`: `(hG.param hf).of_eq (funext fun i => (vf.correct dy i).symm)` would work after moving `param`/`of_eq` above `comp`. It saves only ~4 lines of a 5-line proof, but it does make the relationship explicit. |
| `Proofs/Foundation/ParamGradNodes.lean:219` | Real: the seven weight nodes each restate hG' via simpa [unflatten_flatten] and write the per-op lambda twice. A `param_batchMap` variant at `flat W` with `unflat (flat W) = W` would save ~4 lines each (~25), but its statement carries flat/unflat/hfu/per, and the type-ascribed lambda still has to be written once per node. |
| `Proofs/Nets/ConvNeXt/ConvNeXt.lean:128` | These are 5-stage vjpComp/.comp ladders in the block body and the end-to-end net. Convert them only as a rider of the global `HasVJPDiff`/`vjpCompDiff` from 0120:0; on their own they are too short to justify a lemma. |
| `Proofs/Nets/ConvNeXt/ConvNeXtParamGrad.lean:263` | hCD/hCN/hCE/hCP are long statements with a 1-line proof. A `batchSlice_batchMapAux_dropPathOpt` lemma in Batched/Indexed.lean would let them be passed inline and drop about 20 lines. It is generic, but the gain is mostly the restated statements. |
| `Proofs/Nets/ConvNeXt/ConvNeXtParamGrad.lean:1033` | This is a quadratic `rw` list, not repeated reasoning. A named simp set or `attribute [local simp] cnxPre*_apply`, or bare `rfl`/`unfold` if heartbeats allow, would shrink it. Check why `rw` was chosen before changing it. |
| `Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean:336` | Real duplication: twelve stage witnesses, then twelve `batchMap_differentiableAt` pairs in the apex (plus four in ViTWholeBackCertifiedTieB). A Foundation `batchMapHasVJPDiffAt` returning the pair would remove about 25 lines, but the tie lemmas would then read `.1` off every cnx*BAt. |
| `Proofs/Nets/EfficientNet/EfficientNet.lean:149` | These are short (4–5 stage) vjpComp ladders. They are worth converting only as riders of the global `HasVJPDiff`/`vjpCompDiff` from 0120:0. |
| `Proofs/Nets/EfficientNet/EfficientNetFullB0.lean:442` | Defining `efficientnetForwardBFullHasVJP` as an instance of the generic apex would drop the 17-step ladder and the `backward_unique` bridge. But the file docs deliberately keep the tactic-built witness opaque and bridge by uniqueness, and the generic apex lives downstream in the import order. |
| `Proofs/Nets/EfficientNet/EfficientNetSyncB.lean:270` | There are only two copies of a 5-line chain, but the Do variant restates the whole head graph inside `hd` with `rfl`/`show`. A `headPoolGraphSync` def plus its shard lemma would remove that fragility, at the cost of reshaping the two head graph defs. |
| `Proofs/Nets/EfficientNet/EfficientNetSyncStepTieG.lean:362` | Right in substance: the branch of `mbResidDropWHasVJP` is `mbExpWHasVJP` by delta, so the proof can go through `xCotIn_eq_vjp` instead of repeating the vjpComp unfold. It saves about 3 lines; fold it in on the next touch. |
| `Proofs/Nets/MobileNet/MobileNetV2FullB.lean:487` | The 18-lemma rw chain and the trunk graph term are duplicated between plain and Do. A trunk def plus lemma saves about 25 lines, but it changes forward/graph defs other files unfold; same trade-off as 0115:0/0179:0. |
| `Proofs/Nets/MobileNet/MobileNetV2FullBVJP.lean:148` | Merging each shape's `*HasVJPAt` and `*_differentiableAt` into one `HasVJPDiffAt` def would simplify `mnv2ChainB`. But the `_differentiableAt` theorems are likely referenced elsewhere, and each pair is only 2×4 lines. |
| `Proofs/Nets/MobileNet/MobileNetV2FullBVJP.lean:205` | Restating the head as `HasVJPDiffAt` via the existing `vjpCompDiffAt` (as `mnv2ChainB` already does) would drop the duplicated differentiability term, about 15 lines. It pairs naturally with 0218:0's pool-stage sharing. |
| `Proofs/Nets/MobileNet/MobileNetV2StepTieB.lean:506` | The GAP∘cbrB VJP/diff plumbing duplicates `mnv2HeadBHasVJPAt`. A shared pool-stage `HasVJPDiffAt` in MobileNetV2FullBVJP would serve both heads, saving about 15 lines; do it together with 0197:0. |
| `Proofs/Nets/MobileNet/MobileNetV2SyncB.lean:159` | This is the SyncKit composite of 0034:0/0063:0, about 7 sites here (plus relu6). It fits as part of that one generic lemma, not a MobileNetV2-local one. |
| `Proofs/Nets/MobileNet/MobileNetV2SyncB.lean:424` | The 18-step `s0…s17` chain and the trunk graph term are copied verbatim between the plain and Do theorems. A body graph def plus a shard lemma removes about 40 lines, but it reshapes the graph defs, the same trade-off as 0115:0 and 0256:0. |
| `Proofs/Nets/MobileNet/MobileNetV4FullB.lean:751` | Three per-family `mnv4Skip<Family>_faithful` lemmas would turn each 2-line simp argument into 1 (18 sites, about 18 lines). That's a modest gain for three new lemmas. |
| `Proofs/Nets/MobileNet/MobileNetV4FullB.lean:856` | A trunk graph def plus a trunk lemma would remove the duplicated seven-deep trunk term and its rw chain between the plain and %do graphs. It is a real duplicate, but it reshapes the graph defs other files unfold, for about 15 lines. |
| `Proofs/Nets/MobileNet/MobileNetV4FullBSeal.lean:835` | There are 18 near-identical `eK` haves. A `sealZ_residual_fwd` lemma helps only if the rewrite can match `(sealW nCls).bK` against `sealZ s` syntactically; otherwise each site still needs its type ascription, and the gain shrinks to under 15 lines. |
| `Proofs/Nets/MobileNet/MobileNetV4ParamGrad.lean:901` | Same as 0102:1: a 13-lemma `simp only` list copied into 23 one-line proofs. A local macro or simp set is cheap and shortens the text, but there is no shared lemma here, just a tactic list. |
| `Proofs/Nets/MobileNet/MobileNetV4ParamGrad.lean:1212` | A `mnv4SkipBody_hasGradAt` that bakes `mnv4BodyCotIn_eq_vjp` into `mnv4Skip_hasGradAt_comp` names the body and `ok` once per step. It saves about 1 line per step over 19 steps; a natural statement but a small gain. |
| `Proofs/Nets/MobileNet/MobileNetV4SyncB.lean:210` | Same SyncKit composite (`den_opBnSyncSiteLA_shard`) as 0034:0, with about 20 sites in this file. It is generic and natural, but it only fuses two existing lemma calls of about 17 arguments, saving about 2 lines per site. |
| `Proofs/Nets/MobileNet/MobileNetV4SyncStepTieB.lean:1344` | A `mnv4SkipBodySyncCotIn_scaled` step lemma would name each of the 17 skip steps once instead of passing `sK` twice. It saves about one line per step (~17 lines) and mirrors the 0104:1 step. |
| `Proofs/Nets/ResNet/ResNet50FullBSeal.lean:297` | A `bn_zeroConv_ne_zero` lemma (zero conv then BN with β = 1 is never 0) would replace six 5-line hm1/hm2 proofs, plus two in ResNet34FullBSeal, saving about 25 lines. It is natural, but seal-local. |
| `Proofs/Nets/ResNet/ResNet50SyncB.lean:111` | The conv-shard → bf16-erase → `den_bnSyncSiteLA` triple recurs about 60 times across the five *SyncB files, so a SyncKit composite lemma would be architecture-generic. But it is a composition of two existing lemmas with roughly 17 arguments, saving about 2 lines per site, and each site would still spell out the op. |
| `Proofs/Nets/Small/Cifar8BbnParamGrad.lean:900` | The 40 twin lambdas `fun θ r ci q q' htw => htw {p with X := θ} ci` really are identical, and restating each local g_k over a parameter path ρ would drop about 40 lines. But that reshapes the germ stages the maintainer chose to keep stage-specific, and each lambda is one line. |
| `Proofs/Nets/Small/Cifar8BnParamGrad.lean:861` | A local `bnConvRelu_continuous` would replace a 3-line continuity chain in 16 bullets (about 25 lines). It is a natural statement, but local to one file whose surrounding germ structure the maintainer keeps as is. |
| `Proofs/Nets/Small/CifarCNN.lean:160` | The proposed `VJPAtD` bundle already exists as `HasVJPDiffAt`/`vjpCompDiffAt` in Tensor.lean. Porting these three older chains to it would halve them, but it is churn in stable Small files rather than a new lemma. |
| `Proofs/Nets/Small/CnnParamGrad.lean:211` | The shared core is `have hg := maxPool_relu_eventuallyEq_sel …; filter_upwards [hg]; exact congrArg Gp hθ`. Mathlib's `EventuallyEq.fun_comp` already does this, so no new lemma is needed, and it saves about 1–2 lines at about 15 germ sites across five Small files. |
| `Proofs/Nets/Small/MlpParamGrad.lean:69` | The two-layer dense+relu head pullback (hL→hP₁→hP₀) is the same 6 lines in five ParamGrad files, close in kind to the accepted row-sum factoring. But the cotangents are file-local `let`s (cotH4/cotH6/…) and need `of_eq` at each site, so the saving is about 3–4 lines per file. |
| `Proofs/Nets/Small/MnistCNN.lean:129` | Same as 0023:1: the bundle already exists as `HasVJPDiffAt`/`vjpCompDiffAt`, so this is a port of an older chain, not a new lemma. |
| `Proofs/Nets/ViT/ViTBackB0.lean:647` | A `den_addV`-plus-`add_comm` skip helper would shorten both sublayer capstones by a few lines. There are only two copies, and the VJP-unfolding `show`s stay sublayer-specific. |
| `Proofs/Nets/ViT/ViTDropBlock.lean:183` | These are two 12-line proofs with the same steps, specialise the sublayer tie and peel the skip. A `preLNBranch_back_of_tie` over L/F would save about 10 lines, but it must take the sublayer tie as a hypothesis. |
| `Proofs/Nets/ViT/ViTMhsaBackCertifiedTie.lean:46` | Correct: both proofs re-derive the library's `dense_transpose_eq_mulVec`, which the same file already uses further down. Switching to it saves only about 2 lines each; an easy cleanup on the next touch. |
| `Proofs/Nets/ViT/ViTParamGrad.lean:114` | A generic `HasVJPMat` transport along `F = colSlabApplyH g` would shorten the three attnCore witnesses and their flat-differentiability lemmas. It is real but small (three ~10-line defs) and Q/K/V-name-shaped. |
| `Proofs/Nets/ViT/ViTParamGrad.lean:807` | The twelve dense W/b bullets each pass about 10 identical boilerplate arguments to `vit_node_lossTied` (batchSlice lemmas, rowDense_*_differentiable, the unflatten congr_point). Specialised W/b wrappers could save about 60 lines, but each still takes the pre, post, fwd-factor and tied-holds terms. |
| `Proofs/Nets/ViT/ViTParamGrad.lean:1624` | A stage lemma pairing `vitBlkB_hasGradAt_comp` with `vit_block_lossTiedGB` at the same `congr_point` would state each block once (12 blocks). Same trade-off as 0104:1: about 1–2 lines per block. |
| `Proofs/Training/BatchSealKit.lean:331` | Real ~17-line centre-tap collapse + padding case split shared by conv2d_ctK and depthwise2d_ctDW, but the shared lemma must restate conv2d's padded-read body verbatim so both unfoldings match; saves ~12 lines at that cost. |
| `Proofs/Training/SgdDescent/Cnn.lean:1503` | The four kernel rungs really do repeat the same hpd/hJ blocks (k4Idx_surj + conv2d_weight_pdiv, then convPad_row_l1). Two lemmas over `fun v' => flatten (conv2d (unflatten v') b x)` (base-point independence of pdiv, and the row-ℓ1 ≤ (h*w)*a bound) would each be natural, but every block is only 2-3 lines, so the total saving is ~15-20 lines. |
| `Proofs/Training/SgdDescent/CnnFloat.lean:454` | Two copies of the hHeadMag/hc4Mag/hc3Mag chain (~15 lines each) in cnn_conv2_cot_close and cnn_conv2_cot_real_abs_le. A triple-bound lemma would be about as long as one copy, so it saves ~10 lines, but it would tie `cnnConv2CotMag` to its proof. |
| `Proofs/Training/SgdDescent/Linear.lean:186` | Real: Linear `hz` (186–194), MlpBias `hz` (96–99), Mlp `hzdrift` and Cnn `hzdrift` each do drift → `smul_l1_mass` → scale by t before `softmax_seg_drift`. `logit_seg_drift (hF : ∀ v e k, \|F (v+e) k - F v k\| ≤ c*∑\|e\|) : \|F (v+t•d) k - F v k\| ≤ t*(c*D)` is natural, but Mlp and Cnn still need their own composite `hF` arithmetic (Cnn's δ is not c*D), so the saving is ~10–15 lines in total. |
| `Proofs/Training/SgdDescent/Mlp.lean:1295` | The head-to-layer-1 cotangent chain (l2, hcot2, hC2, hcot1, hC1) really is duplicated (~35 lines) between the w1 and w0 grad-close proofs. The shared lemma's statement would be nearly as long as one copy, so the gain is ~25 lines against a heavy, many-hypothesis statement. |
| `Proofs/Training/SgdDescent/MlpBias.lean:201` | Real but small: each of 3 bias proofs (and the Linear/Mlp/Cnn weight rungs, about 7 sites) carries `simp only [stepRadius] at *` plus a 2–3-line expanded `hC0 := div_nonneg _ (by linarith)`. A `sgd_descends_ratio (N K) (hN) (hsmall : 2*K<1)` wrapper stated with `stepRadius` saves ~3–4 lines per site, and the weight rungs also need `hmain`/flatten plumbing, so the gain there is smaller. |
| `Proofs/Training/Trained/CnnSeal.lean:167` | Real: the relu∘dense Jacobian argument is in CnnSeal d4inner/d3inner (generated by trained_cnn_seal.py) and in hand-written MlpWitness hinner; the library lemma `pdiv_relu_dense` is filed as "do" under 0190:0. Using it here is a generator template change (~15 emitted lines per copy). |
| `Proofs/Training/Trained/CnnSeal.lean:230` | CnnSeal.lean is generated by scripts/certs/trained_cnn_seal.py, so this is a generator change: emitting one k-generic `S2_reduce k : pdiv (...∘ maxPoolFlat) _ k 7 = (maxPoolFlatHasVJPAt r2V _).backward t3V k` and having each S2_ck start with `rw [S2_reduce]` would drop ~7 lines x 72 of generated text. The argmax/non-argmax tails differ per k, so the 72 per-position theorems themselves stay. |
| `Proofs/Training/Trained/CnnSeal.lean:2118` | Generated file (trained_cnn_seal.py), so the change belongs in the generator: an `S1_reduce k` lemma (pdiv_comp + c2flat_eq/S2r + conv2wit.correct + unflatten_flatten) would remove ~8 repeated lines from each of the 72 S1_ck. That is a real cut in emitted size, but nothing hand-maintained gets shorter. |
| `Proofs/Training/Trained/LinearDescent.lean:201` | Same lemma as 0049:0 (`linear_loss_gradAt_sum`, filed "do" there); LinearDescent.lean is generated by trained_linear_descent.py, so using it here is a generator change saving ~3-4 lines per site. |
| `Train.lean:805` | Two adjacent ~15-line blocks share the pathExists/readBinFile/size-check logic but differ in tag, path rewrite, fallback and message wording. A `loadBnCompanion tag path expected fallback` saves ~12 lines, but the copies sit side by side, so the drift risk is low. |
| `Verified/PgdGen.lean:428` | The 9-line select_and_scatter maxpool-back block appears 3x (MNIST once, CIFAR twice), so a `maxPoolBackText res operand cot inTy cotTy attr` helper saves ~18 lines. The 3-line conv input-VJP groups each need 6+ shape strings, so a helper buys little there. |
