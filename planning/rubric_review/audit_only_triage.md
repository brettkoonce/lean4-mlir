# Audit-only triage — 2026-10-01, branch `wp8fg`

**2026-10-01: first pass applied.** Every row whose verdict is exactly cite (70), delete (32) or
unpin (8) is done, each re-checked against the tree first (no delete had a real consumer, every
cite target exists); none was rejected. The `status` column says what each row got. The rows
still marked `open` (every keep and every `?` row) are the to-revisit list. The script now reads
`\_` as `_` in `.tex` files, which clears the three book spellings below; after the pass it lists
50 declarations: the open rows not yet reached, the two cited MLP bias bridges (their citers are
reached only through the book's brace glob), and three this pass created
(`seBlockBackGraph_faithful`, whose only users were the deleted per-example B0 graphs, and the two
concrete `*HasVJPAt` witnesses the projection pins moved to).

Triage of the 164 declarations `python3 scripts/gates/audit_only_mentions.py --where` lists:
pinned in `tests/AuditAxioms*.lean` and reached by no root (book, yaml, comparator, scripts,
READMEs, `#guard`s, module headers) or used declaration's body. Each entry was read, grepped
repo-wide (including the defining file) and checked against newer statements.

**Counts:** cite 73 · delete 44 · keep 39 · unpin 8 (total 164); 17 of them marked `?`. Entries marked `?` are
uncertain; the reason says what decides them.

**What the script misses.** Several of the 39 keeps are real uses the name matcher cannot see:
LaTeX `\_` escapes in `\texttt{…}` (book: `mobilenetv2FwdGraphBFullDo\_faithful`,
`cifarCnnBn8HasVJPAt\_correct`, `layernorm\_back\_bridge`); brace, ellipsis and glob spellings
(`mlp_{w2,w1,w0,b2,b1,b0}_step_float_close`, `IR.mlp_fwd_preact0/1`, `…_shard_id`, the `*_smul`
lemmas, `certifiedC<i>_float`); and f-string generators (`scorecard_crown{'' if … else '_uncon'}`).
Teaching the script `\_` → `_` would clear three entries outright.

**"cite" at a render or test comment** means the place a sibling of the same kind is already cited
(for example `den_matmulFB_per_example` in `ViTRenderB.lean`, `dropPathB_back_faithful` in
`tests/TestBatchedEmitTie.lean`). **"unpin"** entries are generated per-image witnesses that
become reached once their scorecard apex is cited; the apex's pin covers them.

## Architectures — pooling

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `maxPool2_close` | Architectures/CNN.lean | delete | `maxPoolFlat_close` now goes through the generic window lemma; nothing reads this | applied |
| `max_close` | Architectures/CNN.lean | delete | only consumer is `maxPool2_close`; the content is Mathlib's `abs_max_sub_max_le_max` | applied |
| `max4_sub_abs_le` | Architectures/ConvIndex.lean | delete | only consumer is `max4_sub_abs_le_sum` (same cluster) | applied |
| `max4_sub_abs_le_sum` | Architectures/ConvIndex.lean | delete | only consumer is `maxPoolFlat_l1_contract` | applied |
| `maxPoolFlat_apply` | Architectures/ConvIndex.lean | delete | only consumer is `maxPoolFlat_l1_contract` | applied |
| `maxPoolFlat_l1_contract` | Architectures/ConvIndex.lean | delete | superseded by `poolGatherFlat_l1_contract`: the CNN descent now runs through the fixed gather (WP1 2B) | applied; ConvIndex header now names `poolGatherFlat_l1_contract` |
| `maxPool3s2Smooth_of_pairwise` | Architectures/MaxPool3s2.lean | delete | one-line instance of `windowSmooth_of_pairwise`; the seals use `maxPool3s2Smooth_of_injective` | applied |
| `maxPool3s2_eq_argmax_value` | Architectures/MaxPool3s2.lean | delete | one-line instance of `windowMax_eq_argmax_value` | applied |
| `maxPool3s2_flat_hasFDerivAt` | Architectures/MaxPool3s2.lean | delete | one-line instance of `windowMax_flat_hasFDerivAt`; `maxPool3s2Flat_differentiableAt` calls the generic | applied |
| `pdiv3_maxPool3s2_smooth` | Architectures/MaxPool3s2.lean | delete | one-line instance of `pdiv3_windowMax_smooth`; `maxPool3s2HasVJPAt3` spells the indicator itself | applied |
| `win3ColInv_first_dup` | Architectures/MaxPool3s2.lean | cite | module header, beside `win3RowInv_first_dup` ("the padding statement"): its column half | applied |

## Certificates — Gaussian quantile, IBP, Lipschitz

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `Proofs.stdNormalQuantile_strictMonoOn` | Certificates/GaussianQuantile.lean | cite | module header: it claims "continuous and inverts Φ both ways"; name these four | applied |
| `Proofs.stdNormalQuantile_surjOn` | Certificates/GaussianQuantile.lean | cite | same header claim | applied |
| `Proofs.stdNormalQuantile_continuousAt` | Certificates/GaussianQuantile.lean | cite | same header claim | applied |
| `Proofs.stdNormalQuantile_continuousOn` | Certificates/GaussianQuantile.lean | cite | same header claim | applied |
| `Proofs.IBP.deepNet_boxSound` | Certificates/IntervalBoundConv/Basic.lean | delete ? | depth demo: no scorecard uses `deepNet`; the conv scorecard composes its own `net_boxSound`. Keep only if the "depth is free" illustration is wanted | open |
| `Proofs.Robustness.certifiedC0_float` | Certificates/LipschitzCert/Float.lean | keep | the header names the family `certifiedC<i>_float` (the script misses the pattern) | open |
| `Proofs.Robustness.linear_radius_pos` | Certificates/LipschitzCert/Instance.lean | cite | Instance.lean header: it claims each "certified radius is provably positive"; name these | applied |
| `Proofs.Robustness.mlp_radius_pos` | Certificates/LipschitzCert/Instance.lean | cite | same header claim | applied |
| `Proofs.Robustness.trained_radius_pos` | Certificates/LipschitzCert/Instance.lean | cite | same header claim | applied |
| `Proofs.Robustness.trained_radius_gram_pos` | Certificates/LipschitzCert/Instance.lean | cite | same header claim (`trained_radius_gram2_pos` is reached via the s8 witness generator) | applied |
| `Proofs.Robustness.scorecard_crown_uncon` | Certificates/LipschitzCert/ScorecardCrownUncon.lean | cite | Certificates/README.md CROWN row (apex of the Uncon file; the generator's f-string hides the name) | applied |
| `Proofs.Robustness.crownUnconCertse1_certified` | Certificates/LipschitzCert/ScorecardCrownUncon.lean | unpin | conjunct of `scorecard_crown_uncon`; reached once the apex is cited | applied |
| `Proofs.Robustness.certCRTFe4_0` | Certificates/LipschitzCert/ScorecardCrownUncon.lean | unpin | per-image witness in the apex's lists | applied |
| `Proofs.Robustness.certCRTFe8_3` | Certificates/LipschitzCert/ScorecardCrownUncon.lean | unpin | per-image witness in the apex's lists | applied |
| `Proofs.Robustness.hWTF` | Certificates/LipschitzCert/ScorecardCrownUncon.lean | unpin | weight-row bridge every CROWN witness in the file uses | applied |
| `Proofs.Robustness.absrowTF` | Certificates/LipschitzCert/ScorecardIBPData.lean | unpin | row-sum data lemma the Uncon CROWN bounds rewrite with | applied |
| `Proofs.Robustness.scorecard_ibp_uncon` | Certificates/LipschitzCert/ScorecardIBPUncon.lean | cite | Certificates/README.md "IBP L∞, dense" row (apex; f-string hides the name) | applied |
| `Proofs.Robustness.ibpUnconCertse1_certified` | Certificates/LipschitzCert/ScorecardIBPUncon.lean | unpin | conjunct of `scorecard_ibp_uncon` | applied |
| `Proofs.Robustness.certIBPTFe1_0` | Certificates/LipschitzCert/ScorecardIBPUncon.lean | unpin | per-image witness in the apex's lists | applied |
| `Proofs.Robustness.certIBPTFe2_4` | Certificates/LipschitzCert/ScorecardIBPUncon.lean | unpin | per-image witness in the apex's lists | applied |

## Certificates — smoothing scorecards

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `Proofs.smoothCpMlp_tail_le` | Certificates/Smoothing/CPScorecard.lean | cite | Certificates/README.md "smoothing, CP" row: the per-net aggregate (census: a result in its own right) | applied |
| `Proofs.smoothCpCnn_tail_le` | Certificates/Smoothing/CPScorecard.lean | cite | same row | applied |
| `Proofs.smoothCpCifar_tail_le` | Certificates/Smoothing/CPScorecard.lean | cite | same row | applied |
| `Proofs.smoothDecMlp_radius_le` | Certificates/Smoothing/DecScorecard.lean | cite | Certificates/README.md "smoothing, decimal radii" row | applied |
| `Proofs.smoothDecCnn_radius_le` | Certificates/Smoothing/DecScorecard.lean | cite | same row | applied |
| `Proofs.smoothDecCifar_radius_le` | Certificates/Smoothing/DecScorecard.lean | cite | same row | applied |

## Codegen — StableHLO node lemmas

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `StableHLO.backGraph_faithful` | Codegen/StableHLO/Basic.lean | delete | linear input-VJP graph: its printer `linearBackModuleV` has no caller and no artifact (delete it too) | applied; its graph `backGraph` and printer `linearBackModuleV` deleted with it |
| `StableHLO.softmaxDiv_expe_faithful` | Codegen/StableHLO/Basic.lean | delete | superseded by `lossCotGraph_faithful` / `lossCotGraph_isCEgrad`, which state the whole emitted cotangent | applied |
| `StableHLO.bnPerChannelBack_faithful` | Codegen/StableHLO/Basic.lean | cite ? | Cifar8BnStepTie header: the only den→Jacobian link for the emitted cifar8-bn `.bnPerChannelBack` node (the tie states the math VJP) | open |
| `StableHLO.maxPool3s2F_faithful` | Codegen/StableHLO/Basic.lean | keep | only certificate of the per-example pool node the R34/R50 renders emit; the 09-19 census left it pinned on purpose | open |
| `StableHLO.den_bnStatsVarB_allReduce_R1` | Codegen/StableHLO/Basic.lean | cite | DataParallel/Sync.lean, beside the `den_bnSyncF/Back_allReduce_R1` mentions (R = 1 drop-in) | applied |
| `StableHLO.den_bnSyncGammaGradB_allReduce_R1` | Codegen/StableHLO/Basic.lean | cite | same; its docstring calls it the third R = 1 anchor (node emitted by SyncBnSites) | applied |
| `StableHLO.dropoutB_faithful` | Codegen/StableHLO/Basic.lean | cite | the dropout render comments (EfficientNetRender/Basic, MobileNetV2RenderB) or tests/TestDropoutTie.lean; yaml's tail list names `dropPathB_faithful` | applied |
| `StableHLO.dropoutB_back_faithful` | Codegen/StableHLO/Basic.lean | cite | same; only certificate of the dropout backward node in the `*do*` train steps | applied |
| `StableHLO.den_dropoutB_of_dropScale` | Codegen/StableHLO/Basic.lean | cite | EfficientNetRender/Basic.lean, where `dropout_of_dropScale` is cited (node-level form) | applied |
| `StableHLO.den_posEmbedGradB_at_one` | Codegen/StableHLO/Basic.lean | cite | ViTRenderB comment at the `posEmbedGradB` site, as `den_rowDenseBiasGradB_at_one` is | applied |
| `StableHLO.den_softmaxRowBackB_per_example` | Codegen/StableHLO/Basic.lean | cite | ViTRenderB comment at the `softmaxRowBackB` site, as `den_matmulFB_per_example` is | applied |

## Float tier

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `Proofs.Bf16Fold.bf16_render_faithful` | Float/Bf16Fold.lean | cite | book §finite precision, the paragraph that names `Bf16Fold.lean` (prose names the file, not the theorem) | applied |
| `Proofs.Bf16Fold.bf16_render_faithful_emit` | Float/Bf16Fold.lean | cite | same paragraph: the emittable `dotInBf16` form | applied |
| `Proofs.Bf16Fold.bf16_render_faithful_depth2` | Float/Bf16Fold.lean | cite | same paragraph | applied |
| `Proofs.Bf16Fold.bf16_emit_eq_prerounded` | Float/Bf16Fold.lean | delete | `rfl` restatement of `bf16_render_faithful_emit` with `bf16_render_faithful` | applied |
| `Proofs.binary32_e4m3_argmax_small` | Float/Binary32Instance.lean | cite ? | module header beside `binary32_e4m3_budget_small` (its argmax corollary); else delete as a width no net has | open |
| `FloatModel.bnForward_close` | Float/BnFloatBridge.lean | delete ? | its consumers (the per-example BN float compositions) went in the float chop; keep if per-op budgets stay as library | open |
| `bnForward_input_close` | Float/BnInputBridge.lean | delete ? | same: `bnRelu_close` / `bnStep_close` are gone | open |
| `Proofs.convMixedBudget_affine` | Float/ConvMixedComposeBridge.lean | delete | ring identity nothing composes: no whole-net bound is assembled (the file's own header) | applied; `convMixedGain_factor`'s docstring now states the f32 slope itself |
| `Proofs.layerBudget_affine` | Float/ConvMixedComposeBridge.lean | delete | same | applied |
| `FloatModel.depthwiseConv2dF_close` | Float/DepthwiseFloatBridge.lean | delete ? | f32 depthwise budget whose consumer `floatClose_depthwise` was cut; the book's depthwise bound is the bf16-mixed one | open |
| `FloatModel.depthwiseFlatF_close` | Float/DepthwiseFloatBridge.lean | delete ? | same | open |
| `FloatModel.dotMixed_exact_leaf` | Float/FloatBridge.lean | keep ? | `@[simp]`, so an unnamed `simp` may use it; delete if a build without it passes | open |
| `floatClose_id` | Float/FloatClose.lean | keep | FloatClose fold API; kept with `floatClose_iterate` by the 09-19 census ruling | open |
| `floatClose_iterate` | Float/FloatClose.lean | keep | depth-generic fold, same ruling | open |
| `FloatModel.mlp_w0_step_float_close` | Float/MlpFloatBridge.lean | keep | book §finite precision prints `mlp_{w2,w1,w0,b2,b1,b0}_step_float_close` | open |
| `FloatModel.mlp_w1_step_float_close` | Float/MlpFloatBridge.lean | keep | same book line | open |
| `FloatModel.mlp_b0_step_float_close` | Float/MlpFloatBridge.lean | keep | same book line | open |
| `FloatModel.mlp_b1_step_float_close` | Float/MlpFloatBridge.lean | keep | same book line | open |
| `FloatModel.mlp_b2_step_float_close` | Float/MlpFloatBridge.lean | keep | same book line | open |

## Foundation

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `BackLinks.bnBackB_eq_den_bnBatchBack` | Foundation/Batched/BackLinks.lean | cite | BackLinks header (or the StepTieB headers): the finding-3 fix tying the ties' BN cotangent to the emitted `.bnBatchBack` | applied |
| `Proofs.Bf16Fold.convStridedWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | Bf16GradNodes header table (add a lemma column; it names kinds only); only certificate of that node in the bf16 artifacts | applied |
| `Proofs.Bf16Fold.convStridedXlaWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `Proofs.Bf16Fold.convStride4WGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `Proofs.Bf16Fold.depthwiseWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `Proofs.Bf16Fold.depthwiseStridedWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `Proofs.Bf16Fold.depthwiseStridedXlaWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `Proofs.Bf16Fold.rowDenseWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `Proofs.Bf16Fold.patchEmbedWGradBBf16_den` | Foundation/Bf16GradNodes.lean | cite | same table | applied |
| `StableHLO.CertLayer.chain_faithful` | Foundation/CertifiedChain.lean | delete ? | no net builds a `CertLayer.chain` (they compose with `.comp`); it restates `(chain Ls).faithful`. The 09-19 census kept it as "the net-level statement" | open |
| `StableHLO.CertLayer.chain_fwd` | Foundation/CertifiedChain.lean | delete ? | same; delete `chain` with them | open |
| `Proofs.stemPoolRelu_param_eventuallyEq` | Foundation/HeadLayers.lean | delete | one-line instance of `stemPoolRelu_param_eventuallyEq_select`, which the `*_lossGrad_stemSelect` corollaries use | applied; `StemPoolTwinAt`'s docstring repointed to `_select` |
| `IR.conv_compose3` | Foundation/IR.lean | delete ? | demonstrator, held only as the user of `denote_subst3`, which IRPrint now names directly | open |
| `IR.gelu_back_bridge` | Foundation/IR.lean | cite | IRPrint `geluBackM` docstring (the A.1b ruling held it as the proof behind that printout) | applied |
| `IR.swish_back_bridge` | Foundation/IR.lean | cite | IRPrint `swishBackM` docstring, same ruling | applied |
| `IR.layernorm_back_bridge` | Foundation/IR.lean | keep | the book prints it (`layernorm\_back\_bridge` is literally `bn\_back\_bridge`) | open; reached now (the script reads `\_` as `_`) |
| `IR.mlp_fwd_preact1` | Foundation/IR.lean | keep | IRPrint names it as `IR.mlp_fwd_preact0/1` in two comments and the printed string | open |
| `relu_canonical_diagonal` | Foundation/MLP.lean | delete | restates `relu_codegen_matches_canonical` (its docstring: "same content") | applied |
| `Proofs.MuonGeometry.steepest_l2_bound` | Foundation/Muon/Geometry.lean | cite | formalization.yaml optimizer-ladder entry: its `lean:` field lists "SGD / sign" rungs by no name | applied |
| `Proofs.MuonGeometry.steepest_l2_attained` | Foundation/Muon/Geometry.lean | cite | same entry (SGD rung) | applied |
| `Proofs.MuonGeometry.steepest_linf_bound` | Foundation/Muon/Geometry.lean | cite | same entry (sign rung) | applied |
| `Proofs.MuonGeometry.steepest_linf_attained` | Foundation/Muon/Geometry.lean | cite | same entry (sign rung) | applied |
| `Proofs.hasGradAt_iff` | Foundation/ParamGrad.lean | keep | deliberate `HasGradAt` API (P-API: the unfolding lemma beside `.congr_left` / `.congr_of_eventuallyEq`) | open |
| `MathlibUpstream.strictMono_cdf_iff` | Foundation/UpstreamDraft.lean | keep | Mathlib PR1 draft, mirrored in planning/mathlib_upstream_drafts/; pinned so the draft can't rot | open |
| `MathlibUpstream.continuous_cdf_iff` | Foundation/UpstreamDraft.lean | keep | PR1 draft, same | open |
| `MathlibUpstream.cdf_mem_Ioo` | Foundation/UpstreamDraft.lean | keep | PR1 draft, same | open |
| `MathlibUpstream.cdf_gaussianReal_mem_Ioo` | Foundation/UpstreamDraft.lean | keep | PR2 draft, same | open |
| `MathlibUpstream.cdf_gaussianReal_sub_const` | Foundation/UpstreamDraft.lean | keep | PR2 draft, same | open |

## DataParallel

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `Proofs.dpMeanGrad_eq_globalBatchGrad_contiguous` | Foundation/DataParallel/Basic.lean | cite | formalization.yaml DP block: the no-coupling case beside the `_ne_` and sync rows, at the split `Verified.Train` cuts | applied: yaml row + comparator tier |
| `Proofs.dpSyncGrad_eq_globalBatchGrad_contiguous` | Foundation/DataParallel/Basic.lean | cite ? | Verified/Train.lean DP-split comment (the instance at its split); else delete as a `finProdFinEquiv` instance of the yaml row | open |
| `Proofs.dpMean_shardSum` | Foundation/DataParallel/Basic.lean | delete ? | `dpSyncGrad_eq_globalBatchGrad` proves the same re-indexing inline | open |
| `bnSyncTensor4_shard_eq_global` | Foundation/DataParallel/Sync.lean | cite | formalization.yaml DP block: P1, the forward half beside `den_bnSyncBack_allReduce` | applied: yaml row + comparator tier |
| `Proofs.den_allReduceMeanF_convStridedWeightGradBBf16_sub_global` | Foundation/DataParallel/SyncBf16.lean | cite | SyncBf16 header table and the yaml bf16 DP row, which say "its strided peer" without the name | applied: header table + the yaml bf16 DP row's comment |
| `Proofs.den_allReduceMeanF_convStridedWeightGradBBf16_shard` | Foundation/DataParallel/SyncBf16.lean | keep | step of the strided `_sub_global`; reached once that is cited | open; reached now (its apex is cited) |
| `Proofs.den_convStridedWeightGradBBf16_global_split` | Foundation/DataParallel/SyncBf16.lean | keep | step of the strided `_sub_global` | open; reached now (its apex is cited) |
| `Proofs.den_convStridedWeightGradBBf16_eq_rnd` | Foundation/DataParallel/SyncBf16.lean | keep | step of the strided `_sub_global` | open; reached now (its apex is cited) |
| `Proofs.den_allReduceMeanF_convWeightGradBBf16_shard_id` | Foundation/DataParallel/SyncBf16.lean | keep | the header cites it as `…_shard_id` (the f32 consistency check) | open |
| `Proofs.convWeightGradBBf16_smul` | Foundation/DataParallel/SyncBf16.lean | keep | the header's divisor-step section documents the `*_smul` lemmas as the per-node cases a bf16 DP twin needs | open |
| `Proofs.convStridedWeightGradBBf16_smul` | Foundation/DataParallel/SyncBf16.lean | keep | same section | open |
| `Proofs.convBackBatchedBf16_smul` | Foundation/DataParallel/SyncBf16.lean | keep | same section | open |
| `Proofs.convStridedBackBatchedBf16_smul` | Foundation/DataParallel/SyncBf16.lean | keep | same section | open |

## Nets — combined capstones and ResNet glue

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `Proofs.ResNet34TieB.r34_net_tied_lossGrad` | Nets/ResNet/ResNet34ParamGrad.lean | cite | book `thm:resnet34_loss_grad` `\lean{}` (beside `r34_net_lossGrad`) | applied |
| `Proofs.ResNet34TieB.r34LossSmoothAtB_of_smoothAtB` | Nets/ResNet/ResNet34ParamGrad.lean | cite | same block: the loss bundle is implied by the T1 bundle, so T3 applies wherever T1 does | applied |
| `Proofs.ResNet50TieB.r50_net_tied_lossGrad` | Nets/ResNet/ResNet50ParamGrad.lean | cite | book `thm:resnet50_loss_grad` | applied |
| `Proofs.ResNet50TieB.r50LossSmoothAtB_of_smoothAtB` | Nets/ResNet/ResNet50ParamGrad.lean | cite | same block, same reason | applied |
| `Proofs.MobileNetV2TieB.mnv2_net_tied_lossGrad` | Nets/MobileNet/MobileNetV2ParamGrad.lean | cite | book `thm:mobilenetv2_loss_grad` | applied |
| `Proofs.Mnv4TieB.mnv4_net_tied_lossGrad` | Nets/MobileNet/MobileNetV4ParamGrad.lean | cite | book `thm:mobilenetv4_loss_grad` | applied |
| `Proofs.EnetTieG.enet_net_tied_lossGrad` | Nets/EfficientNet/EfficientNetParamGrad.lean | cite | book `thm:efficientnet_loss_grad` | applied |
| `Proofs.CnxTieGB.cnx_net_tied_lossGrad` | Nets/ConvNeXt/ConvNeXtParamGrad.lean | cite | book `thm:convnext_loss_grad` | applied |
| `Proofs.ViTTieGB.vit_net_tied_lossGrad` | Nets/ViT/ViTParamGrad.lean | cite | book `thm:vit_loss_grad` | applied |

## Nets — EfficientNet

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `StableHLO.seGate_backGraph_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | per-example backward graph no renderer emits; superseded by the batched `mb*BackBatchedGraph_faithful` (census M.5) | applied; the cluster's seven per-example graph defs deleted with it; `seBlockBackGraph_faithful` is now audit-only (revisit) |
| `StableHLO.seGateBackGraphE_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | same cluster | applied |
| `StableHLO.seBlockFullBackGraphE_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | same cluster | applied |
| `StableHLO.convBnSwishBackGraph_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | same cluster | applied |
| `StableHLO.dwBnSwishBackGraph_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | same cluster | applied |
| `StableHLO.convBnBackGraph_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | same cluster | applied |
| `StableHLO.mbconvBodyBackGraph_faithful` | Nets/EfficientNet/EfficientNetBackB0.lean | delete | same cluster; superseded by `mbBodyBackBatchedGraph_faithful` | applied |
| `Proofs.efficientnetForwardBFullDrop_ones` | Nets/EfficientNet/EfficientNetFullB0Drop.lean | cite | module header: spell out the `…_ones` / "likewise the eval twins" it elides | applied |
| `Proofs.efficientnetForwardBFullEvalDrop_none` | Nets/EfficientNet/EfficientNetFullB0Drop.lean | cite | same header | applied |
| `Proofs.efficientnetForwardBFullEvalDrop_ones` | Nets/EfficientNet/EfficientNetFullB0Drop.lean | cite | same header | applied |
| `Proofs.StableHLO.efficientnetFwdGraphBFullEval_faithful` | Nets/EfficientNet/EfficientNetFullB0Eval.lean | cite | book `thm:efficientnetFullHasVJP` `\lean{}`: T2 of `efficientnet_fwd_eval.mlir` (MNv4's eval tie is in the book) | applied |
| `Proofs.StableHLO.mbResidGraphEvalW_faithful` | Nets/EfficientNet/EfficientNetFullB0Eval.lean | keep | step of `efficientnetFwdGraphBFullEval_faithful` | open; reached now (its apex is cited) |
| `Proofs.StableHLO.headGraphBEval_faithful` | Nets/EfficientNet/EfficientNetStagesPCEval.lean | keep | step of the same; FwdGraphTextTies guards its text | open; reached now (its apex is cited) |
| `Proofs.StableHLO.mbResidGraphBEval_faithful` | Nets/EfficientNet/EfficientNetStagesPCEval.lean | keep | step of `mbResidGraphEvalW_faithful` | open; reached now (its apex is cited) |

## Nets — ConvNeXt, MobileNet, ViT

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `StableHLO.cnxDownChBackGraph_faithful` | Nets/ConvNeXt/ConvNeXtBackB0.lean | delete ? | per-example downsample backward graph no renderer emits; the batched GB tie covers the downsamples | open |
| `Proofs.StableHLO.mobilenetv2FwdGraphBFullDo_faithful` | Nets/MobileNet/MobileNetV2FullB.lean | keep | the book prints it (`\texttt`, escaped underscores, MobileNetV2 run section) | open; reached now (the script reads `\_` as `_`) |
| `Proofs.mobilenetv2ForwardBFullDo_ones` | Nets/MobileNet/MobileNetV2FullB.lean | cite | MobileNetV2FullB header, as MobileNetV4FullBDrop's header names its `_ones` | applied |
| `Proofs.vitTinyInputGrad_eq_vitTiny_vjp` | Nets/ViT/ViTWholeBackCertifiedTie.lean | cite | book `thm:vitTinyHasVJP_correct`: the shipped-dims backward tie that theorem reads | applied |

## Nets — small nets

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `StableHLO.cifarFwdGraph_faithful` | Nets/Small/ChapterGraphTies.lean | cite | apps/cifar/MainCifarVerified.lean header, as MainMnistCnnVerified cites `cnnFwdGraph_faithful` | applied |
| `StableHLO.cnnBackGraph_faithful` | Nets/Small/ChapterGraphTies.lean | delete ? | no renderer prints `cnnBackGraph` (09-19 census #10); the trained step is `CnnFold` / `cnn_net_lossGrad` | open |
| `Cifar8Tie.cifar8LossCot_den` | Nets/Small/Cifar8StepTie.lean | delete | superseded by the generic `softmaxCELossCot_den` that `cifar8_net_lossGrad_CE` uses | applied |
| `Cifar8Tie.cifar8_Wb_tied_totalloss` | Nets/Small/Cifar8StepTie.lean | delete | superseded by `cifar8_net_lossGrad_CE` (every slot, `Wb` included); no trainer runs `cifar8_train_step` | applied; `cifar8_train_step_tied_certified`'s docstring no longer names it |
| `Cifar8BnTie.cifar8BnLossCot_den` | Nets/Small/Cifar8BnStepTie.lean | delete | superseded by `softmaxCELossCot_den` in `cifar8Bn_net_lossGrad_CE` | applied |
| `cifarCnnHasVJPAt_correct` | Nets/Small/CifarCNN.lean | delete | `.correct` field projection (the A-reuse-1 pattern); pin `cifarCnnHasVJPAt` instead | applied; pin moved to `cifarCnnHasVJPAt` |
| `cifarCnn8HasVJPAt_correct` | Nets/Small/CifarCNN.lean | delete | same; pin `cifarCnn8HasVJPAt` | applied; pin moved to `cifarCnn8HasVJPAt` |
| `cifarCnnBn8HasVJPAt_correct` | Nets/Small/CifarCNN.lean | keep ? | the book prints it; it is a projection, so delete only with that sentence repointed to `cifarCnnBn8HasVJPAt` | open; reached now (the script reads `\_` as `_`) |
| `CnnConcrete.cnnConcreteHasVJP_correct` | Nets/Small/MnistCNN.lean | delete | projection; pin `cnnConcreteHasVJPAt` (README names `CnnConcrete` as a satisfiability witness) | applied; pin moved to `CnnConcrete.cnnConcreteHasVJPAt`, which nothing cites yet (revisit) |
| `MlpConcrete.mlpConcreteHasVJP_correct` | Nets/Small/MnistCNN.lean | delete | projection; pin `mlpConcreteHasVJPAt` | applied; pin moved to `MlpConcrete.mlpConcreteHasVJPAt`, which nothing cites yet (revisit) |
| `Proofs.MlpCanonical.hasVJP_correct` | Nets/Small/MlpCanonical.lean | delete | canonical witness (`mlpHasVJP := HasVJP.canonical`): true by `rfl`, says nothing (09-24 item #2) | applied; its name_lint baseline row removed |
| `Proofs.MlpCanonical.hasVJPAt` | Nets/Small/MlpCanonical.lean | keep | canonical-dims audit surface: its header says AuditAxioms is the only importer by design; reduced-model banners link it | open |
| `Proofs.MlpCanonical.output_float_sgd_descends` | Nets/Small/MlpCanonical.lean | keep | same surface | open |
| `Proofs.MlpCanonical.hidden_float_sgd_descends` | Nets/Small/MlpCanonical.lean | keep | same surface | open |
| `Proofs.MlpCanonical.input_float_sgd_descends` | Nets/Small/MlpCanonical.lean | keep | same surface | open |
| `Proofs.MlpCanonical.w1_grad_close` | Nets/Small/MlpCanonical.lean | keep | same surface | open |
| `Proofs.MlpCanonical.w0_grad_close` | Nets/Small/MlpCanonical.lean | keep | same surface | open |
| `Proofs.MlpCanonical.train_step_tied_certified` | Nets/Small/MlpCanonical.lean | keep | same surface | open |
| `IR.mlp_layer0_weight_grad_bridge` | Nets/Small/MlpTrainStep.lean | keep | cited by `mlp_w0_step_float_close`'s docstring, which the book prints | open |
| `IR.mlp_layer0_bias_grad_bridge` | Nets/Small/MlpTrainStep.lean | cite | `mlp_b0_step_float_close` docstring, as `mlp_w0`'s cites the weight bridge | applied; still listed: its citer `mlp_b0_step_float_close` is reached only through the book's brace glob |
| `IR.mlp_layer1_bias_grad_bridge` | Nets/Small/MlpTrainStep.lean | cite | `mlp_b1_step_float_close` docstring, same pattern | applied; still listed, same reason (`mlp_b1_step_float_close`) |

## Spec rungs and training

| declaration | file | verdict | where to cite / what supersedes / why | status |
|---|---|---|---|---|
| `linearVerified_denote_eq` | SpecVJP.lean | cite | NetsCore `linearVerified` docstring, as the big-net specs cite their SpecVJP rungs | applied |
| `linearVerified_fwd_faithful` | SpecVJP.lean | cite | same docstring | applied |
| `linearVerified_lossCot_isCEgrad` | SpecVJP.lean | cite | same docstring | applied |
| `mlpVerified_fwd_faithful` | SpecVJP.lean | cite | NetsCore `mlpVerified` docstring | applied |
| `mlpVerified_back_faithful` | SpecVJP.lean | delete ? | spec-level graph no artifact prints (09-19 census #10); the emitted step is `mlp_net_lossGrad` | open |
| `cnnVerified_fwd_faithful` | SpecVJP.lean | cite | NetsCore `cnnVerified` docstring (it already cites `cnnVerified_denote_eq`) | applied |
| `cifarVerified_fwd_faithful` | SpecVJP.lean | cite | NetsCore `cifarVerified` docstring (it already cites `cifarVerified_denote_eq`) | applied |
| `Proofs.dropout_eq_reference` | Training/DropPath.lean | cite | DropPath header, with `dropPath_eq_reference`: the fed mask is the reference's inverted dropout | applied |
