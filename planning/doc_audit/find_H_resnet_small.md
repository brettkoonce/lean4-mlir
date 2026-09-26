# Slice H: `LeanMlir/Proofs/Nets/ResNet/` + `LeanMlir/Proofs/Nets/Small/`

**Coverage.** Read in full (docstrings and statements): every file in `Small/` (MnistCNN, CifarCNN, ChapterGraphTies, CnnChainClose, CnnFold, CifarFold, Cifar8StepTie, Cifar8BnStepTie, LinearFold, LinearTrainStep, MlpTrainStep, MlpFold, MlpCanonical), plus ResNetBackChains, ResNet34Fold, ResNet34StepTieB (capstone and loss corollary), ResNet34BackCertifiedTieB (tie and `_correct`), ResNet34FullBSeal (seal statements). Every other ResNet file: I read all module and declaration docstrings and spot-checked the statements of the main theorems (the `_smul`, `_shard`, `nn*`, `pc*`, `ed*` and `sc_*` helper families were only skimmed).
**Axiom check.** I grepped the slice for `sorry`, `admit`, `axiom `, `native_decide`, `implemented_by` and `@[extern]`. Every hit is prose in a comment. `tests/AuditAxioms.lean` does `#print axioms` on the three-axiom claims made in `MnistCNN`, `CifarCNN` and `MlpCanonical` (lines 309–315, 533–543 and 1756–1764), so those claims hold.
**Old names from the naming pass.** None remain in this slice. Every backticked identifier was checked against a declaration index. The misses are listed below.

---

### LeanMlir/Proofs/Nets/Small/CnnFold.lean:198–213, CifarFold.lean:168–178, Cifar8StepTie.lean:406–411, Cifar8BnStepTie.lean:192–196: the small-CNN "whole train step" tie capstones

**Kind:** overclaim
**Says:** (CnnFold §"The CONV fold") "all four conv param ops denote `θ − lr·(certified ∂convₖ/∂θ · the conv backward-chain cotangent the real loss drives)` … Together with the dense head (`cnn_W5_tied_totalloss` + the `*_den` at the composed cotangent) the WHOLE cnn train step is now den-composed forward→loss→backward — no free activations, no symbolic cotangent." The docstring of `cnn_conv_tied_certified` says "denote the certified loss-descent step". `cifar_conv_tied_certified` has the same sentence ("the WHOLE cifar train step is den-composed forward→loss→backward"). So does `cifar8_convs_tied_certified` ("the WHOLE cifar8 train step is den-composed"). `cifar8Bn_convbn_tied_certified` adds "both are the genuine cifar8-bn backward chain", and the Cifar8BnStepTie module says "All 38 params … fold with the generics". The module docstrings of CnnFold (l.11, l.32–35) and CifarFold (l.12) say "each emitted SGD op denotes the certified loss-descent step".
**Actually states:** Each conjunct is `ConvWSgdTied … c` / `ConvBSgdTied … c`, which is `θ − lr·Σ pdiv(conv wrt θ)·c` at the local conv layer. The conjuncts are proved by `convWSgdTied_holds` for every `c`. The `c` fed in is `cnnChainCotW1/W2`, `cifarChainCotW2`, or `dyBnᵢ`/`cotCᵢ`. These are hand-built definitions: `if pre > 0 then (maxpool-back / conv-back).flatDenote … else 0`. No theorem anywhere relates them to `pdiv (crossEntropy ∘ forward)` at the conv output. `grep` confirms that `cnnChainCotW*` and `cifarChainCotW2` are used only in these four files. The capstones take no ReLU-kink or no-tie hypothesis, but at a kink or a pool tie the masked chain is not a derivative. `CnnChainClose`'s own header concedes this ("the further '= ∂loss/∂θ' fold is the separate `pdiv G = Back.denote` step"). The dense head is not covered either. Only the output weight has a total-loss theorem (`cnn_W5_tied_totalloss`, `cifar_W7_tied_totalloss`, and `cifar8_Wb_tied_totalloss`, which is conditional on `hlog`). The hidden dense layers (`W₃,b₃,W₄,b₄` in cnn; `W₅,W₆` in cifar; `W₉,Wa` in cifar8/cifar8-bn) and the output bias are never stated at their chain cotangent. "`denseW_den` at `g`" is the wrong cotangent for them.
**Fix:** In each capstone docstring: "All conv kernel/bias ops, at the real forward activations, denote `θ − lr·(certified ∂convₖ/∂θ · c)` with `c` the rendered backward-chain cotangent (`cnnChainCotW*`: relu masks and select-and-scatter pool-back). This theorem does not state that `c` equals the loss gradient at the conv output, which would need the ReLU and max-pool smoothness hypotheses, and it does not cover the hidden dense layers. Only the output weight is folded to `∂CE/∂W` (`cnn_W5_tied_totalloss`)." In the CnnFold and CifarFold module docstrings, replace "each emitted SGD op denotes the certified loss-descent step" with "each emitted SGD op denotes θ − lr·(certified per-layer Jacobian · the cotangent the rendered chain feeds it); only the output weight is tied to the whole-loss gradient". Delete "All 38 params … fold" from Cifar8BnStepTie. *Cross-slice:* `blueprint/src/content.tex:4290–4295` (thm:cifar8_step_tie) says "and the dense head's updates are tied to the gradient of the whole loss". `cifar8_convs_tied_certified` contains no dense-head conjunct at all. Blueprint l.3285–3288 (cnn) and l.4299–4309 (cifar8-bn: "twenty-four update pairs", which are really 24 conjuncts over 32 tensors) have the same problem.

### LeanMlir/Proofs/Nets/Small/MnistCNN.lean:16–17: module docstring

**Kind:** overclaim
**Says:** "`TrainedCnn.trainedCnnHasVJPAt` discharges every hypothesis at trained weights and a real MNIST test input."
**Actually states:** `Training/TrainedCnnWitness.lean` instantiates `mnistCnnNoBnHasVJPAt` on a reduced model: MNIST cropped to 24×24 and 4×4-average-pooled to 6×6, conv 1→2→2, dense 18→8→8→10, with /128-rationalised weights, trained with a pool-tie regulariser so that `h_mp` holds. It is not the Chapter-3 demo net that the paragraph above describes, and the input is a pooled 6×6 digit, not an MNIST image.
**Fix:** "`TrainedCnn.trainedCnnHasVJPAt` discharges every hypothesis on a reduced instance of this forward (6×6 pooled input, 2 channels, dense width 8, /128-rational trained weights; the max-pool no-tie needed a pool-tie regulariser during training) at one pooled MNIST test digit."

### LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:390–393 and ResNet50WholeBackCertifiedTieB.lean:139–142: `r34InputGradB_correct`, `r50InputGradB_correct`

**Kind:** overclaim
**Says:** "The batched chain IS the `pdiv`-contracted Jacobian of the eighteen-stage net — at every batch size, every input, every loss cotangent and every input pixel." The R50 docstring adds "every resolution".
**Actually states:** Both theorems assume `h_stem : R34StemSmoothAt …` (stem relu off its kink at `x`), `h_pool : R34PoolSmoothAt …` (no 3×3 window ties, per example), `hεs : 0 < εs`, and sixteen `HasVJPDiffAt bₖ (opaqueA… x)` witnesses for arbitrary block maps `b1…b16`. R50 also assumes `hq0 : 0 < q`. The equation holds only at such inputs, and it is about an eighteen-stage chain of variables. Only `resnet*ForwardBFull_eq_slots` connects those variables to the net.
**Fix:** "…at every batch size and loss cotangent, at any input where the stem relu is off its kink and no stem-pool window ties (`h_stem`, `h_pool`), for any block maps carrying VJP witnesses at their running activations. `resnet34ForwardBFull_eq_slots` identifies those maps with the committed net."

### LeanMlir/Proofs/Nets/Small/MlpFold.lean:13–15, 20–22 (module) and MlpCanonical.lean:347–349 (`train_step_tied_certified`)

**Kind:** overclaim / stale
**Says:** (MlpFold) "proves each output's `den` equals the certified `fderiv`-derived loss-descent step … Residual: … the ReLU smooth-point hypotheses are inherited from the bridges". (MlpCanonical) "every SGD op of the emitted graph denotes the certified loss-descent step of the REAL canonical forward."
**Actually states:** `mlp_train_step_tied_certified` folds only `W₂` to `∂CE/∂W₂`. The other five conjuncts are `θ − lr·Σ pdiv(layer wrt θ)·(mlpCotOut{1,0}).denote g`. Nothing proves `mlpCotOut1.denote g = ∂L/∂p₁`. The theorem's own docstring says this correctly. No theorem in `MlpFold` carries a ReLU smoothness hypothesis: `denseW_den`/`denseB_den` are unconditional. What is actually missing is the missing link itself (the smooth-point identification of the chain cotangent), not a hypothesis carried along.
**Fix:** MlpFold: "…proves the output weight's `den` is `W₂ − lr·∂CE/∂W₂`, and each of the other five is `θ − lr·(certified layer Jacobian · the rendered chain cotangent)`. That the chain cotangent is the loss gradient at a hidden pre-activation (true only off the ReLU kinks) is not stated here." MlpCanonical: "every SGD op denotes θ − lr·(certified per-layer Jacobian · rendered chain cotangent) of the canonical forward; the output weight is folded to the whole-loss gradient."

### LeanMlir/Proofs/Nets/ResNet/ResNet34StepTieB.lean:493–498, ResNet34SyncStepTieB.lean:611–614, ResNet50SyncStepTieB.lean:882–884: the "together" sentences

**Kind:** overclaim
**Says:** (`r34_lossCot_is_smoothedCE_grad`) "Together with the capstone this closes the top of the chain: every parameter node denotes the certified gradient at the cotangent of the loss the trainer actually minimises." (`r34_net_syncTiedB`) "this and it together say the DP step's update is the certified gradient of the global-batch step." (`r50_net_syncTiedB`) "…the DP step's update is the certified gradient of the mean loss over all `R·N` examples."
**Actually states:** `r34_net_tiedB`/`r50_net_tiedB` tie each node to `Σ_n (local op Jacobian)ᵀ·c` at constructed cotangents (`r34IdCotA/C2/N1/C1`, …), with no smoothness hypothesis. `*CotIn_eq_vjp` identifies only the block-input cotangent with a certified block backward, and only under `R34IdSmoothAt`/`R34IdPos`. No theorem states that a parameter node equals `pdiv (loss ∘ net)` with respect to that parameter. That statement would need the whole `R34SmoothAtB` bundle, which none of these capstones carries. "Update" is also wrong: the nodes are raw gradients, and LAMB/momentum sit downstream.
**Fix:** "…every parameter gradient node is the certified per-op gradient at the cotangent the rendered chain delivers from the loss the trainer minimises; at a smooth point (`R34SmoothAtB`, via `r34*CotIn_eq_vjp`) the block-input cotangents are the certified block backwards." For the sync theorems: "…so each all-reduced gradient equals the single-device node at the global batch."

### LeanMlir/Proofs/Nets/Small/CifarCNN.lean:435–440, 761: `cifarCnn8HasVJPAt`, BN8 section banner

**Kind:** wrong
**Says:** "Conditional on the twelve ReLU smoothness kinks and the four MaxPools … supplied opaquely (`hf1 … hf12`, `hp1 … hp4`) — they discharge for a concrete instance the same way `Tiny.cifarTinyCnnHasVJPAt` discharges the 2-stage ones." The BN8 banner says "the twelve post-BN ReLU kinks".
**Actually states:** Ten ReLU hypotheses: `hf1…hf8` (conv) plus `hf9`, `hfa` (dense). The BN8 version has `h1…h8` post-BN plus `h9`, `ha` on the dense layers, which are not post-BN. No `hf10–hf12` exist. No concrete 8-conv instance is proved anywhere (`grep cifarCnn8HasVJPAt` finds only `AuditAxioms` and `VerifiedNetsCore` prose). *Cross-slice:* `LeanMlir/VerifiedNetsCore.lean:212,245` repeats "12 ReLU kinks".
**Fix:** "Conditional on the ten ReLU kinks (eight conv, two dense: `hf1…hf8`, `hf9`, `hfa`) and the four MaxPools (`hp1…hp4`). No concrete instance is proved; `Tiny.cifarTinyCnnHasVJPAt` shows the 2-stage analogue is satisfiable." BN8 banner: "the eight post-BN ReLU kinks, the two dense ReLU kinks, and the four MaxPools."

### LeanMlir/Proofs/Nets/ResNet/ResNetBackChains.lean:40–41, ResNet34FullB.lean:49, ResNet34FullBVJP.lean:44, 487, ResNet34StepTieB.lean:33: "carries no numerals"

**Kind:** overclaim
**Says:** "`N` is a variable: this chain carries no numerals." / "T1 and T2 carry no numerals" / "this tier carries no numerals" / "T3 carries no numerals".
**Actually states:** Every ResNet-34 statement pins the resolution and the widths as literals: `x : Vec (N * (3 * (2 * (2 * 56)) * …))`, blocks at `56/28/14/7`, `Kernel4 64 3 7 7`, `Mat 512 nCls`. Only `N` (and `nCls`) are free. A reader would conclude the statement is resolution-generic, as R50's `q` statements are. It is not.
**Fix:** "`N` is a variable: the batch size is never pinned (the 224-px resolution and the [64…512] widths are literals)."

### LeanMlir/Proofs/Nets/ResNet/ResNet50StepTieB.lean:30–32: module docstring

**Kind:** wrong
**Says:** "The stem tie is three nodes, not four … no conv-bias gradient op is ever emitted and there is nothing to keep 'to cover the flag', unlike r34's and MobileNetV2's ties."
**Actually states:** `r50_net_tiedB`'s first conjunct is `r34StemTiedB …`, which contains `ConvStridedBTiedB` (the stem conv-bias node) as its second conjunct. The file's own capstone banner (l.480–481) says "`r34StemTiedB` carries a conv-BIAS conjunct that ResNet-50 never emits … the stem contributes 3 exercised slots of 4."
**Fix:** "The block ties carry no conv-bias conjunct (ResNet-50's render has no `convBias` flag). The stem tie is ResNet-34's `r34StemTiedB`, whose conv-bias conjunct is true at `bias = 0` but not exercised, so the stem contributes 3 exercised slots of 4."

### LeanMlir/Proofs/Nets/ResNet/ResNet50FullB.lean:42: conventions table

**Kind:** wrong
**Says:** "stride-2 padding | symmetric at all five sites (stem + three downsample 3×3 + three 1×1 skips)"
**Actually states:** The list is 1 + 3 + 3 = seven sites (the ResNet-34 table, l.40, says seven).
**Fix:** "symmetric at all seven sites (stem + three downsample 3×3 + three 1×1 skips)".

### LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:57–60: module docstring, hypothesis budget

**Kind:** stale
**Says:** "r34 carries the heaviest hypothesis budget of the five nets: two relu clauses per block … That is 4.1d's bundle list, reused verbatim — this file adds no hypothesis of its own."
**Actually states:** ResNet-50 (three clauses per block) now exists, and its tie file says "ResNet-50 carries the heaviest kink budget in the suite" (`ResNet50WholeBackCertifiedTieB.lean:36`). This file's tie does not take `R34SmoothAtB`. It takes `h_stem`, `h_pool` and sixteen opaque `HasVJPDiffAt bₖ …` witnesses, so the block clauses live inside those witnesses.
**Fix:** "A smooth-point statement: it assumes the stem relu clause and the stem pool's per-example no-tie, and takes each block as an opaque `HasVJPDiffAt` witness (whose relu clauses — two per basic block — are the caller's)."

### LeanMlir/Proofs/Nets/ResNet/ResNet34BackB0.lean:33–38, 46, 53–56 and ResNet34FullB.lean:19–21: where the batched stages live

**Kind:** stale
**Says:** (BackB0 "Structure") `cbReluB` "`_at` VJP + backward-graph faithfulness (`cbReluBackBatchedGraph` + `…_faithful`)" is listed as this file's content. `projBackBatchedGraph` is "reused VERBATIM from EfficientNetBackB0". `cbReluLayer`/`projLayer` are "(from `MobileNetV2BackB0`)". `convStridedBackBatched`'s "`_faithful` lives in `EfficientNetBackB0`". (FullB) "`ResNet34BackB0.lean` already carries the batched stages (`cbReluB`, `cbReluStridedB`, `projStridedB`, and `projB` from `EfficientNetRenderPC.lean`)".
**Actually states:** `cbReluB`, `cbReluStridedB`, `projStridedB`, `cbReluBackBatchedGraph(_faithful)`, `cbReluLayer`, `projLayer`, `cbReluStridedLayer` and `projStridedLayer` are in `Foundation/BatchedStageLayers.lean`. `projB` is in `Foundation/BatchedStages.lean`. `projBackBatchedGraph(_faithful)` and `convStridedBackBatched_faithful` are in `Foundation/BatchedBackLinks.lean`. `ResNet34BackB0.lean` defines only the `r34*` body/block graphs, layers and VJPs. (`ResNet50BackB0.lean`'s table gives the right locations.)
**Fix:** Replace those source attributions with the `Foundation/BatchedStageLayers` / `BatchedStages` / `BatchedBackLinks` locations, and drop `cbReluB` from BackB0's list of this file's contents.

### LeanMlir/Proofs/Nets/ResNet/ResNet34FullB.lean:6–8: module docstring

**Kind:** stale
**Says:** Per-example BatchNorm (reduce `[2,3]`) "is the world of `resnet34_fwd.mlir` and the Imagenette SGD trainer."
**Actually states:** `verified_mlir/resnet34_fwd.mlir` now reduces `dimensions = [0, 2, 3]` (72 sites; `regen_verified_mlir.sh:198` pairs it with `resnet34_adam_train_step.mlir`). It is batch-BN, and the per-example renderer was retired (see `ResNet34Fold.lean:11–14`).
**Fix:** Delete the sentence, or: "The retired per-example renderer stated the ladder at per-example BN; every committed ResNet-34 artifact, `resnet34_fwd.mlir` included, reduces `[0,2,3]`."

### LeanMlir/Proofs/Nets/ResNet/ResNet34StepTieB.lean:45: parameter census section

**Kind:** stale
**Says:** "`resnet34TrainStepFaithfulV` and `ResNet34RenderB` both default to `convBias := false`"
**Actually states:** No declaration `resnet34TrainStepFaithfulV` exists. The live renderers are in `Codegen/ResNet34RenderB.lean` (e.g. `resnet34AdamTrainStepFaithfulB`, `r34SigList … (convBias := false)`).
**Fix:** "`ResNet34RenderB`'s writers default to `convBias := false`".

### LeanMlir/Proofs/Nets/ResNet/ResNet50BackB0.lean:33–34 and ResNet50FullB.lean:51–52: v1 vs v1.5 cost

**Kind:** wrong (citation)
**Says:** "`VerifiedSpec` records it costing ~0.5 pt of top-1" / "worth ~0.5 pt of top-1 (`VerifiedSpec.Layer.bottleneckStage`'s note)".
**Actually states:** The `bottleneckStage` docstring (`LeanMlir/VerifiedSpec.lean:74–78`) says the v1 placement "compiles, trains, descends and is a different net" and gives no top-1 figure.
**Fix:** Drop "~0.5 pt of top-1", or cite where that number is measured.

### LeanMlir/Proofs/Nets/Small/CifarFold.lean:29–33, 41: module residual and empty section

**Kind:** stale
**Says:** "The conv cotangents here are free variables `c` … Pinning each `c` to the exact emitted backward subgraph (the `CnnChainClose` recipe, scaled to two stages) is the remaining polish." Line 41 opens a section "## Conv layers — generic `den = certified` (covers all four conv layers)".
**Actually states:** The file defines `cifarChainCotW2` and pins every conv cotangent in `cifar_conv_tied_certified`. The conv lemmas moved to `Foundation/SgdNodes.lean`, so the l.41 section is empty.
**Fix:** Residual: "The conv cotangents are the rendered chain (`cnnChainCotW*`, `cifarChainCotW2`); that they equal the loss gradient at each conv output is not stated." Delete the empty section header.

### LeanMlir/Proofs/Nets/Small/LinearFold.lean:34–39: honest residual

**Kind:** stale
**Says:** "`tailDenW`/`tailDenB` *model* what `dot_general`/… compute … Adding these as `SHlo` nodes with a `den` (whole module one `pretty(provenGraph)`) is the last mechanical step."
**Actually states:** `tailDenW`/`tailDenB` no longer exist. The file's own §"The tail fold (closed)" says the tail is already `SHlo.weightSgd`/`biasSgd` nodes printed by `pretty`.
**Fix:** "Per-op `den` ⇄ MLIR text for `weightSgd`/`biasSgd` is the same trusted op-level modelling the forward `den` relies on."

### LeanMlir/Proofs/Nets/Small/LinearFold.lean:9–10: module headline

**Kind:** overclaim (per-example vs batched artifact)
**Says:** "every value the emitted module produces is the certified … softmax-CE loss-descent SGD step."
**Actually states:** The theorems are about one example `x`; the emitted module batch-contracts (disclosed only at l.40–41). The bias output is the two-factor contraction, not folded to `∂L/∂b`.
**Fix:** "…for a single example, both emitted outputs denote the certified SGD step (the weight folded to `W − lr·∂CE/∂W`); the committed module batch-contracts, which this file does not state."

### LeanMlir/Proofs/Nets/Small/Cifar8StepTie.lean:6, 344–346: module docstring

**Kind:** wrong / stale
**Says:** "cifar8 is cifar (ch5)". Also "the generic `Cifar8PoC.denseW_den`/`denseB_den` (`MlpTrainStep.lean`, …)".
**Actually states:** CIFAR is Chapter 4 everywhere else (the title here, `CifarCNN`, `CifarFold`), and ResNet-34 is Chapter 5. `denseW_den`/`denseB_den` are in `Foundation/SgdNodes.lean`.
**Fix:** "cifar (ch4)"; "(`Foundation/SgdNodes.lean`, …)".

### LeanMlir/Proofs/Nets/Small/MlpTrainStep.lean:118–122 (section comment) and 145–147 (`mlp_hidden_total_loss_grad`)

**Kind:** stale / overclaim
**Says:** "The hidden layers (`W₁`,`W₀`) fold only at smooth points …; that conditional fold is the remaining new proof." The docstring says the inner factor "is exactly the cotangent the backward chain delivers at layer 1 (`relu'(p₁) ⊙ (W₂ · (softmax−onehot))`, cf. `mlpCotOut1_denote`)".
**Actually states:** The conditional folds are proved immediately below (`mlp_hidden_total_loss_grad`, `mlp_input_total_loss_grad`). Their right-hand side leaves the inner factor as `pdiv (fun z => crossEntropy …(relu z)…) p₁ k 0`. No lemma equates it to `mlpCotOut1.denote`.
**Fix:** Section comment: "The hidden layers fold only at smooth points (below)." Docstring: "…contracted with the loss gradient at the hidden pre-activation `∂L/∂p₁` (which, off the kinks, is the chain cotangent `mlpCotOut1_denote` computes; that identification is not stated here)."

### LeanMlir/Proofs/Nets/Small/Cifar8BnStepTie.lean:159–161: honest residual

**Kind:** stale
**Says:** "BN `0 < ε` smoothness" is listed as a residual.
**Actually states:** `cifar8Bn_convbn_tied_certified` has no `ε` hypothesis; `bnSgdPairTied_holds` is unconditional.
**Fix:** Drop "BN `0 < ε` smoothness".

### LeanMlir/Proofs/Nets/ResNet/ResNet34Fold.lean:5–51: module docstring

**Kind:** stale / process-narrative
**Says:** "This file makes every parameter update of the per-example, SGD-inline train step `den`-faithful … **Two new core ops, ZERO new theorems for 142 of the 146 params** … Honest residual … remaining polish."
**Actually states:** The file holds exactly two generic lemmas, `convStridedW_den` and `convStridedB_den` (consumed by `ConvNeXtStepTie.lean`). The artifact the fold was about is retired.
**Fix:** "`convStridedW_den` / `convStridedB_den`: any emitted strided-conv weight/bias SGD op (`convStridedWeightSgd`/`convStridedBiasSgd`) denotes `θ − lr·(certified ∂flatConvStride2/∂θ · c)`, generic in kernel size and cotangent. Used by the per-example fused-SGD ties (e.g. ConvNeXt)." Move the retirement history to the commit log.

### LeanMlir/Proofs/Nets/Small/CifarCNN.lean:765, 803, 1027: `cifarCnnBn8Forward`, `cifarCnnBn8HasVJPAt`, `cifarCnnBn8HasVJPAt_correct`

**Kind:** missing
**Says:** No docstring. The blueprint (`content.tex:4562–4565`) cites both the `HasVJPAt` def and the `_correct` theorem.
**Actually states:** These are the public forward and whole-net VJP of the Chapter-4 BN net.
**Fix:** Add docstrings, e.g. "The 8-conv CIFAR forward with per-channel BatchNorm (`bnPerChannelTensor3`) between each conv and its ReLU." / "Whole-network VJP at a smooth point, conditional on `0 < εᵢ` (×8), the eight post-BN and two dense ReLU kinks, and the four MaxPool no-tie conditions." / "Its backward equals the `pdiv`-contracted Jacobian of `cifarCnnBn8Forward`."

### LeanMlir/Proofs/Nets/Small/ChapterGraphTies.lean:102–109: `cnnBackGraph_faithful`

**Kind:** missing
**Says:** The theorem's description is a `--` line comment ("… — A2c."), so doc-gen shows it undocumented.
**Actually states:** This is the main backward-graph theorem for the Chapter-3 CNN.
**Fix:** Turn it into a `/-- … -/` docstring and drop "A2c": "The emitted CNN backward graph denotes `mnistCnnNoBnHasVJPAt.backward` at a smooth point (ReLU pre-activations nonzero, no max-pool ties)."

### Process narrative (module docstrings): one entry per file

**Kind:** process-narrative
These module docstrings carry tier and phase labels (T1/T2/T3/T6, "4.1d", "§4.2a", "4d piece 2", "4c leg 1", "§3.5(e)", "M1", "M2", "Crux A", "A2c"), dates ("retired 2026-09-19", "deleted 2026-09-08", "since 2026-09-21", "landed 2026-09-06"), planning-doc pointers (`planning/archive/*.md`, `planning/full_width_seals.md`, `planning/certlayer_nets.md`), "measured walls" (build timings and heartbeats), and run results ("the quoted 76.66%"):
- ResNet: `ResNetBackChains` l.13–14, 27–28; `ResNet34Fold` l.11–19; `ResNet34FullB` l.6–15, 30–31, 51; `ResNet34FullBVJP` l.11, 480; `ResNet34BackCertifiedTieB` l.8–13, 25–29, 33–35, 42–55; `ResNet34StepTieB` l.13–18, 35–41, 50; `ResNet34FullBSeal` l.8–13; `ResNet34SyncB` l.7; `ResNet34SyncStepTieB` l.10, 618–620; `ResNet50FullB` l.6–15, 62; `ResNet50FullBVJP` l.8, 18, 45–46; `ResNet50WholeBackCertifiedTieB` l.7–10, 31, 47; `ResNet50StepTieB` l.12–13, 20, 25, 28, 61, 494–495; `ResNet50FullBSeal` l.7, 33.
- Small: `LinearFold` l.6, 15, 26; `LinearTrainStep` l.4, 129–130; `MlpTrainStep` l.4, 22–25; `CnnChainClose` l.24; `CnnFold` l.144–146 ("see §1a of the planning doc", now also stale because the next section does it); `MnistCNN` l.274–275 ("Closes the gap that `mlpHasVJPAt` is never instantiated").
**Fix:** Keep what each file provides and what it feeds. Move tier labels, dates, retirement history, timings and accuracy figures to commit messages or `runs/*/RESULTS.md`. Where a run is the reason a binder exists, cite the artifact by name without its accuracy.

---

## Overclaims (fix before the published results are read again)

1. **Small-CNN tie capstones** (`CnnFold.cnn_conv_tied_certified`, `CifarFold.cifar_conv_tied_certified`, `Cifar8StepTie.cifar8_convs_tied_certified`, `Cifar8BnStepTie.cifar8Bn_convbn_tied_certified`, plus the CnnFold and CifarFold module docstrings). They say the "WHOLE train step is den-composed forward→loss→backward" and "denote the certified loss-descent step". What is proved is a conv-local Jacobian times a hand-built masked chain cotangent. That cotangent is never tied to `∂L`, no smoothness hypothesis is carried, and the hidden dense layers are never stated. The same claim appears in the blueprint at thm:cifar8_step_tie ("dense head's updates are tied to the gradient of the whole loss"), and in thm:cnn_fold and thm:cifar8bn_step_tie.
2. **`MnistCNN` module**: "trainedCnnHasVJPAt discharges every hypothesis at trained weights and a real MNIST test input". The witness is on a reduced 6×6-input, 2-channel model.
3. **`r34InputGradB_correct` / `r50InputGradB_correct`**: "every input". The statements assume the stem relu is off its kink, the stem pool has no ties, and sixteen opaque block VJP witnesses exist.
4. **`MlpFold` module and `MlpCanonical.train_step_tied_certified`**: "each output's den equals the certified loss-descent step". Only `W₂` is folded to `∂L/∂W₂`.
5. **R34/R50 step-tie and sync "together" sentences** (`r34_lossCot_is_smoothedCE_grad`, `r34_net_syncTiedB`, `r50_net_syncTiedB`): these say the nodes or update are "the certified gradient of the loss". No parameter-level `pdiv (loss ∘ net)` statement exists, and the within-block cotangents are not certified. The claim would also need `R34SmoothAtB`, which none of these capstones carries.
6. **`MlpTrainStep.mlp_hidden_total_loss_grad`**: says the inner factor "is exactly" the chain cotangent. That is not stated.
7. **"carries no numerals"** (R34 chain, T1, T2, T3). The 224-px resolution and all widths are literals; only `N` is free.
8. **`LinearFold` headline**: "every value the emitted module produces". The theorems are per-example (B = 1), and the module is batched.
9. **`CifarCNN.cifarCnn8HasVJPAt`**: "they discharge for a concrete instance". No such instance is proved.
