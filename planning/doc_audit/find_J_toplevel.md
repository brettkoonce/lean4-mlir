# Slice J — `LeanMlir.lean` + `LeanMlir/*.lean` (not `Proofs/`)

**Coverage.** Every docstring (746, extracted by script) was read in full for `LeanMlir.lean`, `VerifiedNetsCore`, `VerifiedSpec`, `VerifiedTrain`, `Types`, `Train`, `IreeRuntime`, `ParamLayouts`, `VerifiedAttack`, `VerifiedPgdGen`, `VerifiedSmoothing`, `VjpOracleNets` and `ViTRender`. Module docstrings, plus the docstrings that match claim or process keywords, were read for `F32Array`, `SpecHelpers`, `Spec`, `E4M3Quant`, `GradcheckHelpers`, `Cam`, `Ddpm`, `MnistData`, `SyncBnCheck`, `ReferenceNets`, `Blackjack`, `Pong`, `FloatFmt` and `LEBytes`. In `MlirCodegen.lean` (8.3k lines) I read the module docstring and all 19 public `def`s. Its private emitters were not read.
**Axiom/sorry check.** `grep -rnE '^\s*axiom|\bsorry\b|\badmit\b|native_decide|implemented_by'` over `LeanMlir/` turns up prose only, no declarations. The top-level files have 94 `@[extern]`s, all FFI runtime code, and no theorem sits on them. `LeanMlir.lean`'s "Zero project axioms" is consistent with that grep. I did not re-run `#print axioms`.

---

### LeanMlir.lean:112 — "The chapter nets" (landing page of the API docs)

**Kind:** overclaim
**Says:** "Whole-network VJPs — `resnet34ForwardBFullHasVJPAt`, `Proofs.mobilenetv2HasVJPAt`, `Proofs.efficientnetHasVJP`, `Proofs.convnextHasVJP`, `Proofs.vitFullHasVJP`"
**Actually states:** Only the first is a whole chapter net. `mobilenetv2HasVJPAt` (MobileNetV2.lean:275) is "stem, a skip inverted-residual, a no-skip inverted-residual, global avg pool, and dense head". `efficientnetHasVJP` (EfficientNet.lean:236) is stem + two MBConv blocks. `convnextHasVJP` (ConvNeXt.lean:209) is stem + two blocks. `vitFullHasVJP` (Attention.lean:2100) is `vitBody kBlocks` with one weight set shared across every block. The full-depth witnesses exist under other names.
**Fix:** "Whole-network VJPs — `resnet34ForwardBFullHasVJPAt`, `Proofs.mobilenetv2ForwardBFullHasVJPAt`, `Proofs.efficientnetForwardBFullHasVJP`, `Proofs.convNextForwardTChHasVJP`, `Proofs.vitForwardKVHasVJP` (the two-block `mobilenetv2HasVJPAt` / `efficientnetHasVJP` / `convnextHasVJP` and the weight-shared `vitFullHasVJP` are the small worked examples)."

### LeanMlir.lean:80 — "Start here"

**Kind:** overclaim
**Says:** "Two things are proved for every net … 2. **Descent** — that step provably decreases the loss: `Proofs.linear_sgd_descends`"
**Actually states:** Descent capstones exist only for the linear net, one MLP layer at a time (`mlp_hidden_sgd_descends`, `mlp_input_sgd_descends`, …), the CNN, and CIFAR-8's last conv (`Training/SgdDescent*.lean`). No ImageNet-tier net has one. `linear_sgd_descends` covers one example's CE, an η-inexact abstract step `W − lr•gh`, and carries hypotheses on the step size (`hsmall`) and two dominance conditions (`h1`, `h2`). It says nothing about the emitted step, and nothing about the batch-mean loss. The page's own "Around the ties" section says "the linear, MLP and CNN capstones", which contradicts "every net".
**Fix:** "Two things are proved, and the linear classifier shows both … 1. **Faithfulness** (every net) … 2. **Descent** (the linear, MLP and CNN nets): one inexact SGD step on one example's loss decreases it by ≥ `lr·‖∇L‖²/2` under a small-step and two dominance hypotheses — `Proofs.linear_sgd_descends`."

### LeanMlir.lean:114 — "The chapter nets", tie paragraph

**Kind:** overclaim
**Says:** "for every net the tiers train, the tie of the committed train-step render (the verified_mlir/ files the trainers load) to the certified chain, at the batch BatchNorm and the ImageNet head its artifacts run: each emitted parameter-update node denotes the certified descent step."
**Actually states:**
1. `efficientnet_net_tiedG` is pinned to 10 classes (`B0Weights`, `t : Vec (N * (1 * 10))`), while the B0 ImageNet artifacts (`efficientnetin_*`) are `x1000`.
2. `cnx_net_tiedGB` says "Stated at the drop-free chain; the `*drop*` artifacts' … cotangent chain carries the `dropPathB` sites, which this thread does not name". The shipping ConvNeXt and ViT ImageNet renders are `*drop*` and `*bf16`.
3. The capstones tie the parameter **gradient** nodes ("every parameter GRADIENT node … denotes the certified batched Σ_n gradient", ResNet34StepTieB:409). The optimizer update is a separate per-op tie (`adamW_triple_faithful`, `lamb_triple_faithful`).
4. The capstones cover one replica. Data parallelism is the separate `*SyncStepTieB`.
**Fix:** "… for every net the tiers train, a tie of the committed train-step render to the certified chain at batch BatchNorm: every emitted parameter-gradient node denotes the certified batched gradient, and the optimizer ops are tied per op (`adamW_triple_faithful`, `lamb_triple_faithful`). Class count is a binder except for EfficientNet-B0 (pinned at 10); drop-path and bf16 variants are not covered by the capstones."

### VerifiedAttack.lean:172, 278, 396, 676 — `attackPgdMlp`, `attackPgdSpectralMlp`, `attackPgdConvNet`, `attackPgd`

**Kind:** overclaim
**Says:** "attacks through IREE with the proven `mlpInputGrad` VJP kernel" (172). "attacks through IREE with `genKernel` — the full proven backward" (396). "the `genLinearPgdStep` StableHLO kernel (the proven `dx = (softmax−onehot)·Wᵀ` VJP)" (676). "The verified CE gradient stays in the proven kernel" (278).
**Actually states:** `VerifiedPgdGen`'s own module docstring says: "⚠ These are NOT rendered from a proven graph and no tie pins them: each follows the proven input-VJP's formula by hand … so they are unverified program code, like the reference `MlirCodegen`." Also, the runtime default is PJRT/XLA, not IREE.
**Fix:** "… then attacks through the runtime with `genMlpPgdStep`, a hand-typed StableHLO kernel that follows the formula of the proven `mlpInputGrad` VJP (unverified code; no tie pins it)." At 278: "the CE gradient comes from the proof-rendered train step; the PGD kernel and the projection are unverified host/hand-typed code."

### VjpOracleNets.lean:3 — module docstring (and :81 `residualNet`)

**Kind:** overclaim
**Says:** "The oracle trains each net one step through the verified path (Lean → MLIR → IREE, `tests/vjp_oracle/phase3/`)". At :81: "tests `biPathHasVJP` (additive fan-in VJP)".
**Actually states:** The phase-3 entry points call `VjpOracle.convBnOnly.train …`, which is `NetSpec.train`, i.e. `Train.compileVmfbs` and `MlirCodegen.generateTrainStep`. `MlirCodegen` describes itself as "the REFERENCE path … Unverified". The oracle compares two hand-written backwards (MlirCodegen and JAX). It does not exercise `biPathHasVJP` or `verified_mlir/`.
**Fix:** "… through the Lean reference codegen (`MlirCodegen`, NetSpec → MLIR at run time) and through the JAX reference …". At :81: "tests the reference codegen's additive fan-in backward (the math is `biPathHasVJP`)".

### ViTRender.lean:3 — module docstring

**Kind:** overclaim (+ stale)
**Says:** "# ch10 ViT — verified-faithful StableHLO render fragments (shared library) … each line what the proven-faithful emitter produces"
**Actually states:** This is the hand-written String emitter. `Proofs/Codegen/ViTRender.lean:8` calls it "a hand-written String emitter (faithful per-op, NOT `pretty(provenGraph)`)". No theorem ties its text. It is checked only numerically (`tests/TestViTAdamTie.lean`, `TestAdamOpTie`, the gradchecks). No trainer uses it any more: `MainViTVerifiedAdam` stopped emitting from it and loads the proof-side render, and its only importers are tests. ViT is chapter 9, not 10.
**Fix:** "# ViT — hand-written StableHLO fragments (legacy, test-only). The emitter the ViT trainer used before the proof-side `Proofs/Codegen/ViTRender.lean` render replaced it; kept as the numeric oracle for `vit-adam-tie`, `TestAdamOpTie` and the ViT gradchecks. Not tied to any theorem."

### MlirCodegen.lean:9 — module docstring

**Kind:** overclaim
**Says:** "Covers every `Layer` constructor (dense, conv, BN, depthwise, MBConv/SE, UIB, attention, …)."
**Actually states:** `mambaBlock`, `swinStage`, `patchMerging`, `fireModule` and `separableConv` never appear in MlirCodegen.lean. `Types.lean:101/106` says of `mambaBlock` and `swinStage`: "Not codegen-backed yet — used by the Bestiary as a shape-only primitive".
**Fix:** "Covers the trainable `Layer` constructors (dense, conv, BN, depthwise, MBConv/SE, UIB, attention, UNet, detection heads, …); the Bestiary-only shape primitives (`mambaBlock`, `swinStage`, `patchMerging`, `fireModule`, `separableConv`) have no codegen."

### VerifiedSpec.lean:16 — module docstring

**Kind:** stale / overclaim (+ process-narrative)
**Says:** "The architecture's *faithfulness* is the audited `<net>HasVJP` theorem, which is itself a hand-unrolled `foldl` of the generic `vjpComp` … over these same layers — so the spec and the proof describe the same fold. Generating the verified StableHLO from `layers` … and folding the proof via a `netVjp` term are the remaining Tier-2 / Tier-3 steps; for now the slug names the committed, audited render of this architecture."
**Actually states:**
1. The spec↔math link is `SpecVJP`'s `denote*`, which pattern-matches the literal layer list to the net function (`resnet34VerifiedB_denote_eq … := rfl`), plus the `*_fwd_faithful` rung-E theorems. The spec-level witness for the full nets is `HasVJP.canonical` (`resnet34VerifiedBHasVJP`), not a `vjpComp` fold.
2. Emitted-text faithfulness is the `*StepTie*` capstones, not `HasVJP`.
3. `netVjp` does not exist anywhere.
**Fix:** "`SpecVJP` ties each committed spec to its net function (`<net>Verified*_denote_eq`, by `rfl` on the literal layer list, so a spec edit breaks it) and to the rendered forward graph (`*_fwd_faithful`); the emitted train step's faithfulness is the per-net `*StepTie*` capstone. The slug names the committed render; the render is not derived from `layers`."

### VerifiedSmoothing.lean:3 (module) and :95 `clopperPearsonLower`

**Kind:** overclaim
**Says:** "a **sound** Clopper–Pearson lower confidence bound on `p_A` (a genuine 1−α lower bound, not an approximation — a certificate must under-estimate)"
**Actually states:** Float code with no proof. The incomplete beta comes from a Lanczos `lgamma` plus a Lentz continued fraction (eps 3e-12, ≤200 iterations). The function returns the bisection **midpoint** `0.5*(lo+hi)`, not `lo`, so it is not rounded in the conservative direction.
**Fix:** "the Clopper–Pearson lower bound (exact in its statistical definition — no normal approximation), evaluated in `Float` by 60-step bisection on a Numerical-Recipes incomplete beta; not proven, and not rounded conservatively."

### VerifiedNetsCore.lean:1002 — `convnextVerified`

**Kind:** stale
**Says:** "Its only hypotheses are the 22 LN positivities (stem + 18 blocks + 3 downsamples; there is no head LN)."
**Actually states:** `convNextForwardTChHasVJP` (ConvNeXtFullT.lean:294) takes 23 of them, including `hhε : 0 < w.hε` for the head LN. Its docstring records that "The count read `22 … no head LN` until 2026-09-04 — … the third place that stale number had been copied to". This is a fourth copy. The layer list 15 lines below has `.globalAvgPool, .layerNorm 768, .dense 768 10`, and the same docstring's first paragraph says the head LN was restored.
**Fix:** "Its only hypotheses are the 23 LN positivities (stem + 18 blocks + 3 downsamples + the head LN)." Also delete the paragraph that begins "⚠ Three things above were stale …" (process narrative).

### VerifiedNetsCore.lean:853 — `efficientnetVerified`

**Kind:** stale
**Says:** "the honest pointwise VJP witness is the representative `Proofs.efficientnetHasVJP`."
**Actually states:** `efficientnetHasVJP` is global (not pointwise) and covers stem + 2 MBConv blocks. The full-depth witness is `Proofs.efficientnetForwardBFullHasVJP` (EfficientNetFullB0.lean:371).
**Fix:** "the full-depth VJP is `Proofs.efficientnetForwardBFullHasVJP` (global; only `0 < ε` hypotheses)."

### VerifiedNetsCore.lean:475 — `### ResNet-50` section header

**Kind:** stale (+ process-narrative)
**Says:** "⚠⚠ **SKELETON, 2026-08-03. These two specs are the LAYOUT only.** There is no `Proofs/Architectures/ResNet50*.lean`, no `ResNet50RenderB.lean`, no artifact and no rung E — so nothing renders, trains or is gated off them yet."
**Actually states:** `Proofs/Nets/ResNet/ResNet50{FullB,FullBVJP,FullBSeal,StepTieB,SyncB,SyncStepTieB,WholeBackCertifiedTieB}.lean` exist, `r50_net_tiedB` is cited on the landing page, and `verified_mlir/resnet50in_*` artifacts have trained (commit 82445a97, 77.16%).
**Fix:** "### ResNet-50 — the bottleneck pair. Proof chain: `Proofs/Nets/ResNet/ResNet50*` (capstone `r50_net_tiedB`)."

### VerifiedNetsCore.lean:527 — `resnet50ImagenetVerified`

**Kind:** stale
**Says:** "the verified driver has **no gradient accumulation** … So a pair run is not comparable until that is settled"
**Actually states:** `VerifiedTrain` implements accumulation (`VerifiedVariant.accOn`/`accK`, the `G` blob region, "acc<k>x<B>" at :1516), and `lambaccdp8x64…` renders exist.
**Fix:** "The reference number is at effective batch 2048 (512 × 4 accumulation); the verified driver matches it with the `acc…` variants (`lambaccdp8x64…`)."

### VerifiedNetsCore.lean:602 — `resnet50Imagenet160Verified`

**Kind:** stale
**Says:** "⚠⚠ **DO NOT run this net with eval enabled until `evalD0` lands.** The driver feeds `net.d0` to both invokes …"
**Actually states:** `evalD0` has landed. `loadData` takes `(evalD0 : Nat := 0)` (VerifiedTrain:1108), and `fwdRenderedShape` (VerifiedTrain:1429) reads the 224² eval width off the artifact "under RSB-A3".
**Fix:** Drop the warning. Say: "Eval runs at 224² from `resnet50in160_fwd_eval.mlir`; the driver reads the eval width off that artifact (`fwdRenderedShape`)."

### VerifiedNetsCore.lean:1606 — `mnv4ImagenetVerified`

**Kind:** stale (overclaim group 4)
**Says:** "⭐ **This IS Conv-M as of 2026-08-14, so it is now comparable to the chapter's 75.51%.** … there is no data-parallel render, so it is single-device"
**Actually states:** The 75.51 / 75.48 JAX number was measured on the pre-timm-parity transcription. The book (content.tex:8020) says: "The 75.48% above is this network's target. It is not this network's result, and it was measured on the earlier transcription." The spec has matched timm since 2026-09-24. DP renders exist: `mnv4in_adamdp64*`, `mnv4in_emaaccdp8x128wxdowd005bf16`.
**Fix:** "Conv-M at timm 1.0.28 parity; `#guard`ed at 9,715,512 parameters. The chapter's 75.48% was measured on the earlier (pre-parity) transcription and is a target, not a comparison. Data-parallel renders: `mnv4in_adamdp64`, `mnv4in_emaaccdp8x128…`. No verified ImageNet run yet."

### VerifiedNetsCore.lean:426, 790, 1056, 1323 — ImageNet specs' "Claim ceiling"

**Kind:** stale (contradicts LeanMlir.lean)
**Says:** (`resnet34ImagenetVerified`) "The proof-carrying tier stops at Imagenette: this net has no §1a tie … The honest sentence is 'one architecture, two independent lowerings, agreeing', not 'proven'". The same ceiling appears at `mobilenetv2ImagenetVerified`, `convnextImagenetVerified` and `vitImagenetVerified`.
**Actually states:** `r34_net_tiedB`, `mnv2_net_tiedB`, `cnx_net_tiedGB` and `vit_net_tiedGB` all bind the class count (`{nCls}` / `{nC}`). `r34_net_tiedB`'s docstring names `resnet34in_momdp64`. The only ceiling that is still correct is B0's (10 classes). The landing page claims the reverse for every net, so the two documents disagree.
**Fix:** R34/MNv2/ConvNeXt/ViT: "The train-step capstone (`r34_net_tiedB`, …) binds the class count, so it covers this head at one replica; the spec-level `SpecVJP` ties are stated at 10 classes." Keep the Imagenette-only ceiling for `efficientnetImagenetVerified`, where it is still true.

### VerifiedNetsCore.lean:1 — module docstring

**Kind:** stale
**Says:** "Specs with no proof importing them yet (e.g. `resnet34Verified`) stay in their own `Main*Verified.lean`"
**Actually states:** `resnet34Verified` is defined in this file (:389), and `SpecVJP` imports it.
**Fix:** "Every verified spec lives here; a spec is added the moment a trainer or proof needs to name it."

### VerifiedNetsCore.lean:78, 134, 487 — stale declaration names

**Kind:** stale
**Says:** "(`mlpVerifiedHasVJP` / `_at`)" (78). "(`cnnVerifiedHasVJPAt`, folded through conv/maxpool/dense)" (134). "(`SHlo.maxPool3s2F` / the `BatchableOp.maxPool3s2` descriptor …)" (487).
**Actually states:** The declarations are `mlpVerifiedHasVJPAt` (SpecVJP:107), `cnnVerifiedHasVJP` (SpecVJP:154; no `…At` exists), and `BatchableOp.maxPool3s2` (no `maxPool3s2F` exists).
**Fix:** Replace the names with `mlpVerifiedHasVJPAt`, `cnnVerifiedHasVJP` and `BatchableOp.maxPool3s2`.

### VerifiedNetsCore.lean:389, 743, 853, 1002, 1273; ViTRender.lean:3,6; GradcheckHelpers.lean:12 — chapter numbers

**Kind:** stale
**Says:** "ch6 **ResNet-34**", "ch7 **MobileNetV2**", "ch8 **EfficientNet-B0**", "ch9 **ConvNeXt-T**", "ch10 **ViT-Tiny**", "ch? ResNet-50", "the ch10 ViT de-risk tests"
**Actually states:** The book (`blueprint/src/content.tex` `\chapter`s) and `LeanMlir.lean`/`ParamLayouts` put ResNet-34 at 5, MobileNetV2 at 6, EfficientNet at 7, ConvNeXt at 8 and ViT at 9.
**Fix:** Shift each down by one. ResNet-50 lives in chapter 5.

### VerifiedSpec.lean:217, VerifiedTrain.lean:58, ParamLayouts.lean:70/107/156/198 — `initKind` 0

**Kind:** stale
**Says:** "`initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros …" (and "stem fan-in = 3·7·7 = 147", "depthwise fan-in = 9", …)
**Actually states:** `VerifiedTrain.mkParam` (:479): rank-4 gets "He, fan-OUT" (`2/(oc·kH·kW)`), rank-2 gets Glorot (`2/(in+out)`), and fan-in applies only under the gate-only `heFanIn` flag. `mkParam`'s own docstring says "Both weight cases CHANGED 2026-08-04".
**Fix:** "`initKind`: 0 = random weight (conv: He fan-out; dense: Glorot; see `VerifiedTrain.mkParam`), 1 = ones, 2 = zeros, 3 = 1e-6."

### ParamLayouts.lean:156 `EfficientNetLayout`, :198 `ConvNeXtLayout`

**Kind:** stale
**Says:** EfficientNet: "(CIFAR 3×32×32, E6 …) stem … (3×3 stride-1 conv 3→32, CIFAR adaptation) … Spatial 32→16→8→4→2 (4 strided stages, stem stride 1) … (tests/TestEfficientNet*.lean)". ConvNeXt: "→ **LN** (global per-example scalar γ/β, rank-0 `#[]`) … Each downsample (4 params): LN scalar {γ,β}".
**Actually states:** `EfficientNetLayout.xShape` is `3 * 224 * 224` ("Imagenette 224²"), matching `efficientnetVerified` (224², stride-2 stem). `ConvNeXtLayout.blockSpec` carries `(#[c],1),(#[c],2)` "LN γ,β (PER-CHANNEL, §2m)". The renders come from `Proofs/Codegen/*Render*.lean`, not `tests/`.
**Fix:** EfficientNet: "Imagenette 3×224×224, stride-2 stem, 224→112→…→7". ConvNeXt: "channel LN, per-channel `[c]` γ/β". The "rendered from" pointer should name `Proofs/Codegen/EfficientNetRender.lean` / `ConvNeXtRender.lean`.

### VerifiedTrain.lean:9 — module docstring

**Kind:** stale
**Says:** "pre-rendered, audited StableHLO (`verified_mlir/<slug>_{train_step,fwd}.mlir`, emitted offline by `tests/Test*` from the proof stack)"
**Actually states:** No file in `tests/` writes `verified_mlir/`. The writers are the 12 `Proofs/Codegen/*.lean` `#eval`s, and `VerifiedNet.mlirDir` (:40) says `regen_verified_mlir.sh` pins the corpus to those.
**Fix:** "… emitted offline by the `Proofs/Codegen/` renderers (`scripts/regen_verified_mlir.sh`)".

### VerifiedTrain.lean:1366, 1464 — `trainAdamPacked`, `trainAdamSched`

**Kind:** stale
**Says:** "against the baked-hyperparameter packed render `@<slug>_adam_train_step` (`ViTRender.vitTrainStepModuleAdamPacked` …)". "Drives `ViTRender.vitTrainStepModuleAdamSched`." Also "(Phase 2)".
**Actually states:** Both drivers load `verified_mlir/<slug>_<variant>_train_step.mlir` for every net (R34, MNv2, B0, ConvNeXt, ViT, R50, MNv4), rendered by `Proofs/Codegen`. The hand-written ViTRender functions are used only by tests (`MainViTVerifiedAdam.lean:68`: "This driver no longer WRITES its artifact").
**Fix:** "… against the committed `@<slug>_<variant>_train_step` (rendered by `Proofs/Codegen`; optimizer ops tied by `adamW_triple_faithful` / `lamb_triple_faithful`)".

### VerifiedTrain.lean:2762 — `scoreCheckpoint`

**Kind:** stale
**Says:** Table row "drain the val split | `loadData` (all 50,000 as of `ccca380`)". Also "It is available today on ConvNeXt and ViT, which have `nBnStats = 0`".
**Actually states:** ImageNet val is streamed per pass (`spawnValStream`, :2948), and `loadData` holds only non-ImageNet splits. The next paragraph of the same docstring describes BN-net scoring from the `.bn` companion, so the equality gate is not limited to LN nets.
**Fix:** Row: "stream the val split | `spawnValStream` (ImageNet) / `loadData` (held splits)". Drop "available today on ConvNeXt and ViT".

### VerifiedTrain.lean:88 `VerifiedNet.dropKeeps`, VerifiedSpec.lean:294 `VerifiedNetSpec.dropKeeps`

**Kind:** stale
**Says:** "Empty = the net has no drop sites, which is every net today except EfficientNet's `*sd` variants." / "Empty on every net without a `*sd` render."
**Actually states:** `dropKeeps` is set on R50, B0, ConvNeXt-T/S/B and ViT-Ti/S/B (VerifiedNetsCore:570–1431). There are no `*sd` artifacts. The marker is `drop` (`VerifiedVariant.sdOn`: "the marker is `drop` and not `sd`").
**Fix:** "Empty = no drop sites. Consumed only by `*drop*` variants."

### IreeRuntime.lean:2, :8 — module docstring and `LowererSession`

**Kind:** stale
**Says:** "Also holds the verified nets' parameter-layout tables (`*Layout`), which the trainers and `VerifiedSpec` read." / "Opaque handle to an IREE runtime session (module + device)."
**Actually states:** The tables live in `ParamLayouts` (imported and re-exported here), and `VerifiedSpec` is import-free. The session is lowerer-agnostic (PJRT/XLA by default).
**Fix:** "Re-exports `ParamLayouts` (the `*Layout` tables)." / "Opaque handle to a lowerer session (PJRT/XLA or IREE): a loaded module on a device."

### Train.lean:110 — `compileVmfbs` (+ :480 `runTraining`)

**Kind:** stale
**Says:** "…compile each to a `.vmfb`. Cached on the MLIR content + IREE backend … Returns the path to the train-step vmfb." / "Adam + cosine-LR + running-BN-stats training loop"
**Actually states:** On XLA (the default), `runIreeCached` returns immediately and `graphArtifact` returns the `.mlir`. The loop also runs SGD+momentum when `useAdam = false` (:676).
**Fix:** "Writes the three `.mlir`s; on IREE also compiles them to cached `.vmfb`s. Returns the train-step artifact `graphArtifact` resolves (`.mlir` on XLA, `.vmfb` on IREE)." / "Adam or SGD+momentum, cosine LR, …".

### Types.lean:909 `runningBN`, :782 `dropPath`, :894 `bf16`

**Kind:** stale
**Says:** `runningBN`: "Currently wired for the convBn + invertedResidual path (MobileNetV2); extend the BN-threading to mbconv/basic/bottleneck blocks for the other convnets." `dropPath`: "currently wired for ConvNeXt blocks." `bf16`: "Measured ~2.7-3.6× … on gfx1100 … See reference_bf16_gfx1100_conv_vs_gemm."
**Actually states:** `jax/Jax/Codegen.lean` branches on `cfg.runningBN` for residual (:827), bottleneck (:855), MBConv (:986) and fused (:1239), and applies `dropPath` at several block kinds (:2284, :2347, :2378, :2421, :2484). `reference_bf16_gfx1100_conv_vs_gemm` is not a repo file; the box is now CUDA 4060 Ti.
**Fix:** `runningBN`: "Wired for convBn, residual, bottleneck, inverted-residual, MBConv and fused-MBConv blocks." `dropPath`: name the block kinds. `bf16`: drop the dangling reference, or cite a repo measurement.

### Types.lean:53 `Layer`, :339 `NetSpec`, :581 `TrainConfig`, :1042 `DatasetKind`, :3 `Activation`

**Kind:** missing
**Says:** (no docstring. The module docstring is one line: "ML specification types: Layer, NetSpec, TrainConfig, DatasetKind.")
**Actually states:** These are the central public types of the reference path. `Layer`'s ~40 constructors are documented only with `--` comments, which doc-gen4 does not render, so the API page shows `Layer.transformerEncoder`, `Layer.uib` and the rest bare.
**Fix:** Add a docstring to each type. Convert the per-constructor `--` comments on `Layer` into `/-- … -/` constructor docstrings, e.g. "`uib ic oc expand stride preDWk postDWk` — MobileNetV4 Universal Inverted Bottleneck; `k = 0` omits that depthwise."

### LeanMlir.lean:108 — "The foundation"

**Kind:** stale
**Says:** "`tests/comparator` re-runs Lean's kernel over 73 of the headline theorems"
**Actually states:** `config.json` / `config-arch.json` / `config-tier.json` list 13 + 39 + 35 = **87** theorem names. Commit 7dbe9c92 reports "(13 + 39 + 35 OK)".
**Fix:** "over 87 of the headline theorems". Lead for another slice: `tests/comparator/README.md` still says 73 and "remaining 21".

### Process narrative in docstrings (grouped: one pattern, many sites)

**Kind:** process-narrative
**Says (representative):**
- VerifiedNetsCore:1002 "⚠ Three things above were stale or wrong until 2026-08-12 … A docstring is not gated by anything."
- :1056 "⚠⚠ **This docstring used to end 'none of which exist on the verified path', and that was WRONG by four of six as of 2026-08-12**"
- :1441 "⚠⚠ **THE 'PER-DEVICE BATCH 32' PIN IS LIFTED** (2026-08-27) … **AND THE 32×4 PAIR IS DELETED**"
- :484 "⚠ The deviation this docstring used to record was **2×2 stride-2** …"
- :661 "Caught 2026-08-24 as a mean −4.90 top-1 gap … the run was killed at epoch 13"
- VerifiedTrain:449 `mkParam` "⚠ **Kind 3 is new on 2026-09-13** … ⚠ **Both weight cases CHANGED 2026-08-04**"
- :163 `cnxInit` "⛔ **Why it exists.** The 2026-09-17 ConvNeXt/ImageNet pair run was killed at epoch 67"
- :202 `wilson95` "⭐ Added 2026-08-30 because …"
- :120 `shimScript` "THIS FIELD EXISTS BECAUSE ITS DEFAULT USED TO BE R34's"
- :396 `mkSession` "All 58 call sites used to supply one"
- :1211 `printBlurb`, :634 `ShimProc`, :818 `readShimBatchPartial` "Why this exists (2026-08-14)"
- F32Array:152 `dropoutMask` "⛔⛔ **WHAT CHANGED, AND WHAT THE OLD NOTE GOT WRONG**" with a dated measurement table
- F32Array:338 `loadDetBin` "(This docstring said 6076 …)"
- Train:63 `graphArtifact` "⚠ This said '…' until 2026-08-25"
- Train:8 and SpecHelpers module docs ("Each Main*Train.lean used to inline ~250 lines" / "These were copy-pasted into every `Main*Train.lean`")
- VerifiedSpec module "(Tier-2)", "remaining Tier-2 / Tier-3 steps"
- `handoff §2k/§2p`, `§2m`, `rung E`, `§1a` cross-references to planning sections throughout VerifiedNetsCore/VerifiedSpec/ParamLayouts
- `LeanMlir.lean:62` import comment "audited in tests/AuditAxioms.lean since 2026-07-07: it rotted while orphaned …"

**Actually states:** Each declaration's docstring should say what the object is and what its invariant is. The dated histories, run IDs and "used to" corrections belong in commit messages, which already hold them.
**Fix:** Keep the present-tense invariant and cut the history. For example, `cnxInit`: "ConvNeXt `_init_weights`: σ = 0.02 on every conv and the head; must match the JAX reference's `cnxInit` or a pair run compares two inits." `mkParam`: "Matches the JAX reference: conv He fan-out, dense Glorot, γ = 1, β/bias = 0, layer scale 1e-6. Variance-matched, not distribution-matched (Bates-3 vs uniform)."

### VerifiedTrain.lean:3081, :3151 — E4M3 trainers

**Kind:** stale (minor)
**Says:** "Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-e4m3-verified data`"
**Actually states:** The default runtime is PJRT/XLA on the CUDA box. The comment at :3160 says these trainers moved to `mkSession` (the XLA-capable path).
**Fix:** "Run: `lake exe mnist-linear-e4m3-verified data`"

---

## Overclaims (fix before anyone reads the published results again)

1. **LeanMlir.lean:112** — "Whole-network VJPs" names three 2-block representatives (`mobilenetv2HasVJPAt`, `efficientnetHasVJP`, `convnextHasVJP`) and the weight-shared `vitFullHasVJP`. Cite the full-depth witnesses instead.
2. **LeanMlir.lean:80** — "Two things are proved for every net … Descent". Descent covers only linear/MLP/CNN, per example, under step-size and dominance hypotheses.
3. **LeanMlir.lean:114** — "every net … at … the ImageNet head its artifacts run: each emitted parameter-update node denotes the certified descent step". B0 is pinned at 10 classes; drop/bf16 variants are not covered; the ties are at gradient nodes, not update nodes; one replica only.
4. **VerifiedAttack.lean:172/278/396/676** — "the proven `mlpInputGrad` VJP kernel" / "the full proven backward". The PGD kernels are hand-typed and unverified by `VerifiedPgdGen`'s own admission.
5. **VjpOracleNets.lean:3** — "through the verified path". It is the unverified `MlirCodegen` reference path.
6. **ViTRender.lean:3** — "verified-faithful … what the proven-faithful emitter produces". It is a hand-written emitter with no tie, now test-only.
7. **MlirCodegen.lean:9** — "Covers every `Layer` constructor". Five Bestiary constructors have no codegen.
8. **VerifiedSpec.lean:16** — "faithfulness is the audited `<net>HasVJP` theorem … a hand-unrolled `foldl` of `vjpComp` over these same layers". The link is a literal-list `rfl` denote plus the StepTie capstones, and `netVjp` does not exist.
9. **VerifiedSmoothing.lean:3/95** — "sound … a genuine 1−α lower bound, not an approximation". Unproven Float bisection that returns the midpoint.
10. **VerifiedNetsCore.lean:1606** — "now comparable to the chapter's 75.51%". That number came from the pre-parity net; the book calls it a target.
