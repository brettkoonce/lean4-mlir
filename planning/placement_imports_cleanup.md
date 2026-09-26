# placement_imports_cleanup.md — imports, declaration homes, directory layout

Started 2026-09-26 from a placement-and-imports audit of every built module under `LeanMlir/`,
`apps/`, `demos/` (365 modules, tree at ed4e50a7). Rubric: unused import, broad import, wrong home,
net-specific material in a generic file, hidden instead of moved, forwarding module, flat siblings
that should be a directory. Every import finding below was compiled; every move was checked for
cycles against the full import graph. Reverse-dependency ("rdeps") and closure counts are modules,
measured on that tree.

## 0. Rules for this thread

* Tooling beats judgment for imports. `lake shake` refuses non-`module` files and this Mathlib only
  has `#min_imports in <cmd>`, so the audit recomputed ImportGraph's whole-file `#min_imports`
  (`Environment.minimalRequiredModules`) from the .oleans. That sees constants only, not notation,
  tactics, `#eval` or `example` bodies, so a removal counts only once the file **compiles** without
  the import, and the whole graph is simulated for downstream files that reached something through it.
* No renames in this thread. A generic lemma that moves keeps its net-flavoured name (`mnv2_*`,
  `ResNet34PoCB.*`); renames belong to the naming pass (`LeanMlir/NAMING.md`).
* Root files move as one batch. `Tensor` (245 rdeps), `MLP` (227), `LayerNorm` (191), `AdamStep`
  (186), `Codegen.StableHLO` (315) each force most of `Certs` to rebuild; §4 collects every edit to
  them so the corpus rebuilds once.
* Gate after every batch with `lake build Certs` (bare `lake build` skips the corpus), plus
  `lake build LeanMlir Apps CertsHeavy Reference`, and capture the exit status.
* A module rename or deletion is not done until every consumer that names it by path is repointed:
  lakefile roots (`Proofs`, `Certs`, `CertsHeavy`, `ProofsMinimal`, `Reference`),
  `tests/AuditAxioms.lean` and `tests/comparator/`, `scripts/regen_verified_mlir.sh`'s module list,
  `.github/workflows/*.yml` path filters and diff lists, the scorecard generators' output paths,
  `scripts/gen_comparator_tier.py`, the READMEs' layout tables, and doc-gen4 backtick paths.

## 1. Done

| commit | change |
|---|---|
| 70d623be | `LeanMlir/MnistData.lean` (no build target reached it; MNIST loads through `F32.loadIdxImages` / `loadIdxLabels` in `ffi/f32_helpers.c`) and its three `historical/` importers deleted; README row fixed |
| 94c16b30 | Batch A: §2.1 and §2.2 as tabled, §2.3 on the leaf files (113 imports, 68 files). Deviations below |
| (staged) | Batch B: §3.1–3.6 and two of the three re-derived statements. Deviations below |

Batch A deviations:

* `MobileNetV2Close` keeps `ConvGrad` (§3.3 needs it back); `MobileNetV2Fold` drops `StagesPC` and
  keeps `SgdNodes`.
* The §2 claim that no downstream file needs a new import was wrong. Nineteen consumers had reached
  a declaration through an import §2.1/§2.2 removed, and now import its home directly:
  `Lamb` (`AdamStep`); `MobileNetV2FullPaperEval`, `EfficientNetRenderPC`, `BatchedBackLinks`
  (`StableHLO`); `FloatComposeBridge` (`ResNet34FloatBridge`); `VerifiedTrain` (`ParamLayouts`);
  `CnnFold`, `CifarFold` (`MlpTrainStep`); `ConvNeXtChannelLN` (`IndexCast`); `MobileNetV4FullB`
  (`HeadLayers`); `ResNet50FullBVJP` (`ResNet50BackB0`); `ConvNeXtStepTie` (`CifarFold`);
  `ConvNeXtStepTieGB` (`GradNodesB`, `ViTFoldGB`); `tests/AuditAxioms.lean` and the comparator
  tier (via `gen_comparator_tier.py`'s `MODULES`) (the four `*SyncB`); the four `tests/*SyncBnCheck`
  (`VerifiedNetsCore`). Several §2.1 rows therefore move an edge down a level rather than cut it;
  §3 is what cuts them.
* `scripts/certs/lipschitz_cert_pair_sdp_full.py` emits the Uncon file's narrowed imports.
* `tests/TestDropPathRamp.lean`, `TestViTFwd.lean`, `TestViTTrain.lean` lost ViT modules from their
  closure and are not checked: no `lean_exe` or lib builds them.
* Gate: `lake build Certs LeanMlir Apps CertsHeavy Reference` (3613 jobs), the ten affected test
  exes, `lake env lean` on `AuditAxioms` and both tier files, `gen_comparator_tier.py --check`.

Batch B deviations:

* §3.3 and §3.4 emptied three files, which are deleted with their lakefile roots and imports:
  `MobileNetV2Close` (its importers take `ConvGrad`), `MobileNetV2Fold` and `ResNet34Fold`.
  Certs is now 185 roots reaching 233 proof modules (lakefile and `Proofs/README.md` updated).
* `vecLNBetaSgdTied_holds` (§3.4) and `vecLNBetaTiedB_holds` (§3.5) move with their γ twins.
* `ConvGrad` imports `Depthwise` in place of `CNN` (which `Depthwise` reaches).
* Edges cut beyond the table: `ConvNeXtStepTie` imports no small-net, ViT, MobileNetV2 or
  ResNet-34 module (it takes `SgdNodes` and `SmoothedLossCot`); `ConvNeXtStepTieGB` drops
  `ViTFoldGB`; `CifarFold` and `CnnFold` take `SmoothedLossCot` in place of `MlpTrainStep`.
  `MobileNetV4WholeBackCertifiedTieB` needed `ConvBackCertifiedTie` and `Foundation.OpaquePrefix`
  directly once `ResNet34BackCertifiedTieB` went. `ViTWholeBackCertifiedTieB` needed `Foundation.BatchMapVJPAt`
  once ViT's LN tie stopped importing ConvNeXt (§3.1).
* Re-derived statements: `IR.mlp_output_total_loss_grad` and `ViTPoC.headW_den` / `headB_den`
  deleted, users repointed, their three `#print axioms` lines removed from `tests/AuditAxioms.lean`.
  `Mnv4FullBSeal.cbReluStridedB_eq` / `cbReluB_eq` stay: `BatchSealKit` has a general
  `cbReluStridedB_eq` (in `R34FullBSeal`) but no `cbReluB_eq`, so only one of the pair could go.
* Left for §4 (a root file): the comments at `Codegen/StableHLO.lean:498` and `:525` still name
  `ResNet34Fold` / `MobileNetV2Fold`; the lemmas are `ResNet34PoC.convStrided{W,B}_den` and
  `Mnv2PoC.depthwise{W,B}_den` in `SgdNodes`.
* Gate: `lake build Certs LeanMlir Apps CertsHeavy Reference ProofsMinimal` (3611 jobs) and
  `Proofs`; `AuditAxioms` (1607 prints, 1602 on the three standard axioms, the 5 others on a
  subset) and both tier files elaborate; `gen_comparator_tier.py --check`, `check_audit_coverage`,
  `check_render_coverage`, `regen_verified_mlir.sh check`, `gen_mlir_manifest --check`,
  `check_target_names`, `name_lint`, `docstring-checkrefs`, `blueprint-checkdecls` +
  `blueprint_uses --check` (lean_deps unchanged).

## 2. Batch A — leaf imports (cheap, no root file)

### 2.1 Unused imports

Not in the file's minimal set, and the file compiles without them.

| file | remove | closure Δ |
|---|---|---|
| `Proofs/Codegen/StableHLOLex.lean` | `Codegen.StableHLO` | −3268 (the lexer stops depending on Mathlib) |
| `Proofs/Foundation/ListDot.lean` | `Mathlib.Tactic` | −732 (keep `EuclideanDist`; either one suffices, not neither) |
| `Proofs/Nets/ConvNeXt/ConvNeXtFoldG.lean` | `Nets.MobileNet.MobileNetV2Fold`, `Nets.ViT.ViTFoldG` | −9 |
| `Proofs/Nets/ResNet/ResNet34Fold.lean` | `Nets.Small.CifarFold` | −9 |
| `Proofs/Float/ResNet34BlockBridge.lean` | `Float.ResNet34FloatBridge` | −7, with the §2.2 swaps in the two Float bridges |
| `Proofs/Nets/MobileNet/MobileNetV2StepTieB.lean` | `Nets.ResNet.ResNet34StepTieB` | −7 |
| `Proofs/Nets/EfficientNet/EfficientNetSyncStepTieG.lean` | `EfficientNetStepTieG`, `EfficientNetSyncB` | −6 |
| `Proofs/Nets/MobileNet/MobileNetV4SyncStepTieB.lean` | `Nets.ResNet.ResNet34SyncStepTieB`, `MobileNetV4SyncB` | −6 |
| `Proofs/Codegen/ViTRender.lean` | `Nets.ViT.ViTMultiHead` | −5 |
| `Proofs/Nets/ConvNeXt/ConvNeXtChannelLN.lean` | `ConvNeXtChainClose` | −5 |
| `Proofs/Nets/ConvNeXt/ConvNeXtChainClose.lean` | `Nets.MobileNet.MobileNetV2Close` | −4 |
| `Proofs/Nets/MobileNet/MobileNetV2Close.lean` | `Architectures.ConvGrad`, `Architectures.PerChannelBNGrad` | −3 (but §3.3 needs `ConvGrad` back) |
| `Proofs/Nets/Small/CifarFold.lean` | `Codegen.CnnRender` | −3 |
| `Proofs/Nets/Small/CnnFold.lean` | `Codegen.CnnRender` | −3 |
| `Proofs/Nets/Small/MlpFold.lean` | `Codegen.MlpRender`, `Nets.Small.LinearFold` | −3 |
| `Proofs/Nets/ViT/ViTFold.lean` | `Nets.ViT.ViTVecLN` | −3 |
| `Proofs/Float/ResNet34FloatBridge.lean` | `Architectures.StridedConv`, `Architectures.PerChannelBN` | −2 |
| `Proofs/Nets/ConvNeXt/ConvNeXtFoldGB.lean` | `Foundation.GradNodesB`, `Nets.ViT.ViTFoldGB` | −2 |
| `Proofs/Nets/MobileNet/MobileNetV4BackB0.lean` | `Foundation.HeadLayers` | −2 |
| `Proofs/SpecVJP.lean` | `Nets.MobileNet.MobileNetV2`, `Nets.EfficientNet.EfficientNet` | −2 |
| `LeanMlir/IreeRuntime.lean` | `LeanMlir.ParamLayouts` | −2, with the §2.2 swap in `SpecHelpers` |
| `Proofs/Architectures/Attention.lean` | `Architectures.SE` | −1 |
| `Proofs/Architectures/ChannelLN.lean` | `Foundation.IndexCast` | −1 |
| `Proofs/Certificates/LipschitzCert.lean` | `Mathlib.Tactic.Positivity.Finset` | −1 |
| `Proofs/Certificates/SmoothingNetSemantics.lean` | `Certificates.DenseEuclid` | −1 |
| `Proofs/Codegen/MlpRender.lean` | `Nets.Small.LinearTrainStep` | −1 |
| `Proofs/Nets/EfficientNet/EfficientNetWholeBackCertifiedTie.lean` | `EfficientNetBackChains` | −1 |
| `Proofs/Nets/MobileNet/MobileNetV2SyncStepTieB.lean` | `MobileNetV2SyncB` | −1 |
| `Proofs/Nets/ResNet/ResNet34FullBVJP.lean` | `ResNetBackChains` | −1 |
| `Proofs/Nets/ResNet/ResNet50FullB.lean` | `ResNet50BackB0` | −1 |
| `Proofs/Nets/ResNet/ResNet50SyncStepTieB.lean` | `ResNet50SyncB` | −1 |
| `Proofs/Nets/ViT/ViTChainClose.lean` | `Architectures.TokenParamGrad` | −1 |
| `LeanMlir/Ddpm.lean` | `LeanMlir.F32Array` | −1 |

`MobileNetV2Fold` compiles without `MobileNetV2StagesPC` or without `Foundation.SgdNodes`, not
without both: drop one. The root-file rows (`Tensor`, `MLP`, `LayerNorm`, `AdamStep`) are in §4.

Imports that look unused but are not (each fails to compile without it): `ChapterArtifacts` and
`MlpArtifacts` (their `#eval`s), `VerifiedNetsCore` (`#eval ResNet34Layout.specs`),
`FwdGraphTextTies` (its `example`s), `Blackjack` and `Pong` (the `export`, §5.2).

### 2.2 Broad imports — a module imported only as a route to its own imports

The file uses no declaration of the imported module. Every swap compiles, and no downstream file
needs a new import.

| file | replace | with | closure Δ |
|---|---|---|---|
| `Foundation/MuonNewtonSchulz.lean` | `Foundation.MuonGeometry` | `Mathlib.Algebra.Order.Star.Real`, `Mathlib.Analysis.Normed.Field.Basic`, `Mathlib.Topology.Instances.Matrix`, `Mathlib.Topology.Metrizable.Uniformity` | −845 |
| `Nets/MobileNet/MobileNetV2StagesPC.lean` | `Codegen.StableHLO` | `Architectures.Depthwise`, `Architectures.PerChannelBN` | −204 |
| `Nets/MobileNet/MobileNetV2StagesPCEval.lean` | `MobileNetV2StagesPC` | `Architectures.Depthwise`, `Architectures.PerChannelBN` | −205 |
| `Nets/MobileNet/MobileNetV2FullPaper.lean` | `MobileNetV2StagesPC` | `Architectures.Depthwise` | −206 |
| `Architectures/MaxPool3s2.lean` | `Architectures.CNN` | `Foundation.Tensor` | −136 |
| `Training/JacobianSeal.lean` | `Foundation.MLP` | `Foundation.Tensor`, `Mathlib.Analysis.Calculus.Deriv.Mul`, `…Deriv.Slope`, `Mathlib.Analysis.RCLike.Basic` | −79 |
| `Certificates/SmoothingCP.lean` | `Certificates.SmoothingMC` | `Certificates.SmoothingGaussian` | −55 |
| `Training/Optim/GradClip.lean` | `Training.Optim.AdamStep` | `Foundation.Tensor` | −16 |
| `Foundation/BatchedStages.lean` | `Codegen.StableHLO` | `Architectures.Depthwise`, `Architectures.SE` | −12 |
| `Float/ConvMixedComposeBridge.lean` | `Float.FloatComposeBridge` | `Float.ConvFloat`, `Float.FloatClose` | −8 |
| `Float/DepthwiseFloatBridge.lean` | `Float.FloatComposeBridge` | `Float.ConvFloat`, `Float.FloatClose` | −7 |
| `Nets/ViT/ViTFoldG.lean` | `Nets.ViT.ViTFold` | `Architectures.TokenParamGrad`, `Codegen.StableHLO` | −7 |
| `Foundation/DataParallelSync.lean` | `Foundation.DataParallelNode` | `Codegen.StableHLO` | −6 |
| `LeanMlir/SyncBnCheck.lean` | `LeanMlir.VerifiedNets` | `LeanMlir.VerifiedTrain` | −5 |
| `LeanMlir/SpecHelpers.lean` | `LeanMlir.IreeRuntime` | `LeanMlir.ParamLayouts` | −1 |
| `Nets/MobileNet/MobileNetV2FullBSeal.lean` | `Nets.ResNet.ResNet34FullBSeal` | `Training.BatchSealKit`, `Training.JacobianSeal` | −7 |
| `Nets/MobileNet/MobileNetV4FullBSeal.lean` | `Nets.ResNet.ResNet34FullBSeal` | `Training.BatchSealKit`, `Training.JacobianSeal` | −4 |
| `Nets/MobileNet/MobileNetV4StepTieB.lean` | `Nets.ResNet.ResNet34StepTieB` | `Foundation.SmoothedLossCot` | −4 |
| `Nets/ConvNeXt/ConvNeXtBackB0.lean` | `Nets.EfficientNet.EfficientNetBackB0` | `Foundation.BatchedBackLinks` | −4 |
| `Nets/EfficientNet/EfficientNetBackNet.lean` | `Nets.MobileNet.MobileNetV4BackB0` | `Foundation.BatchedBackLinks` | −9 |
| `Nets/MobileNet/MobileNetV4BackB0.lean` | `Nets.ResNet.ResNet50BackB0` | `Foundation.BatchedStageLayers` | −2 |
| `Nets/EfficientNet/EfficientNetStepTieG.lean` | `EfficientNetStepTie` | `Foundation.BatchedBackLinks`, `EfficientNetFullB0` | −4 |
| `Nets/Small/CnnChainClose.lean` | `Nets.Small.MlpTrainStep` | `Codegen.StableHLO` | −2 |

The last seven rows remove net-to-net edges across families (MobileNet → ResNet, ConvNeXt →
EfficientNet, EfficientNet → MobileNet). The two MobileNet seals use no ResNet-34 declaration;
the shared kit is `Training/BatchSealKit.lean`.

### 2.3 Implied imports

169 imports in 83 files are already reached through another import of the same file; removing
them leaves every closure unchanged. The largest: `Foundation/Tensor.lean` (11 of 14 Mathlib
imports), `Architectures/Attention.lean` (10), `Codegen/StableHLO.lean` (8), `LayerNorm` (7),
`SpecVJP` (7), `LeanMlir/Train.lean` (5). Noise only, no build-time effect; delete mechanically,
leaf files here and the root files in §4.

## 3. Batch B — generic lemmas out of net files

Every declaration under `Nets/` whose dependencies stay outside `Nets/` (transitively, through
other such declarations) and that another net family uses. The audit's fixpoint found no others:
every other cross-family edge runs through a `Codegen` renderer or `SpecVJP`, as intended. Each
target's closure already holds the dependencies unless noted, and no move creates a cycle.

| § | from | declarations | to | import changes |
|---|---|---|---|---|
| 3.1 | `Nets/ConvNeXt/ConvNeXtBackCertifiedTie` | `bnGradInput_eq_vjp_backward`, `layerNormVecHasVJP_backward_eq`, `rowLNVecFlatHasVJP_backward_eq` | `Architectures/ChannelLNBack` | `ViTVecLNBackCertifiedTie` drops `ConvNeXtBackCertifiedTie` |
| 3.2 | `Nets/ResNet/ResNet34BackCertifiedTieB` | `cbReluBBack_eq_vjp_backward`, `cbReluStridedBBack_eq_vjp_backward` | `Architectures/ConvBackCertifiedTie` | target gains `Foundation.BatchedStageLayers` (11 of its 18 rdeps newly reach it); `MobileNetV4WholeBackCertifiedTieB` drops `ResNet34BackCertifiedTieB` |
| 3.3 | `Nets/MobileNet/MobileNetV2Close` | `mnv2_render_stem_convW_certified`, `mnv2_render_stem_convb_certified`, `mnv2_render_depthwiseb_certified`, `mnv2_depthwise_bias_grad_bridge` | `Architectures/ConvGrad` (the README's home for per-op parameter-gradient bridges) | target gains `StridedConv`, `Depthwise` (58 of 59 rdeps already reach both); `ResNet34Fold` drops `MobileNetV2Close` |
| 3.4 | `MobileNetV2Fold`, `ResNet34Fold`, `CifarFold`, `CnnChainClose`, `ViTFold` | `Mnv2PoC.depthwise{W,B}_den`, `Mnv2PoC.mnv2_render_depthwiseW_flat_certified`, `ResNet34PoC.convStrided{W,B}_den`, `conv{W,B}SgdTied_holds`, `Conv{W,B}SgdTied`, `ViTPoC.VecLN{Gamma,Beta}SgdTied`, `vecLNGammaSgdTied_holds`, `veclnGammaSgd_den`, `rowDenseBiasSgd_den_lnbeta` | `Foundation/SgdNodes` (the per-example fused-SGD nodes) | after 3.3; `CnnChainClose` gains `SgdNodes`; `ConvNeXtStepTie` drops `MobileNetV2Fold`, `ResNet34Fold`, `ViTFold` |
| 3.5 | `Nets/ViT/ViTFoldGB` | `VecLN{Gamma,Beta}TiedB`, `vecLNGammaTiedB_holds`, `veclnGammaGradB_den`, `head{W,B}GradB_den`, `rowDenseBiasGradB_den_lnbeta` | `Foundation/GradNodesB` (the batched gradient nodes) | `ViTFoldGB` gains `GradNodesB` |
| 3.6 | `Nets/Small/LinearTrainStep` | `StableHLO.softmaxCELossCot_den`, `lossWeightGrad_eq_sum`, `denseWeightMap_differentiable` | `Foundation/SmoothedLossCot` (the loss-cotangent kit) | `LinearTrainStep` and `ConvNeXtStepTie` gain `SmoothedLossCot` (+1 module in `ProofsMinimal`); lowest value, since the ConvNeXt → small-net edge also runs through `CifarFold` until 3.4 lands |

What each buys: net edits stop rebuilding other families (ConvNeXt → ViT ties, ResNet-34 → MNv4
whole-back tie, MNv2 → ResNet-34 fold, three nets → ConvNeXt step tie).

Re-derived statements, deleted in the same batch (types are `Expr`-equal):

* `ViTPoC.headW_den` / `headB_den` (`ViTFold`) = `Cifar8PoC.denseW_den` / `denseB_den` (`SgdNodes`,
  already imported). One user, `ViTStepTie`.
* `IR.mlp_output_total_loss_grad` (`MlpTrainStep`) = `StableHLO.lossWeightGrad_eq_sum`, which it is
  proved from. Five users: `CnnFold`, `MlpFold`, `CifarFold`, `Cifar8StepTie`, `ConvNeXtStepTie`.
* `Mnv4FullBSeal.cbReluStridedB_eq` / `cbReluB_eq` restate `BatchSealKit`'s general-`hp`
  `cbReluStridedB_eq` (`BatchSealKit.lean:1009`) at `(1, kv 1, kv 160)`; use the kit's with `bpos`.

Hidden-instead-of-moved: none. The private general-looking lemmas (`sum_heads_3d`,
`mulVec_headPadMat` in `ViTBackB0`; `sum_channel_fiber` in `PerChannelBNGrad`) have one consumer
each and no copy elsewhere.

## 4. Batch C — root files (one `Certs` rebuild)

| file | change |
|---|---|
| `Foundation/MLP.lean` | drop `Mathlib.Analysis.SpecialFunctions.ExpDeriv` (−38); `JacobianSeal` then needs the §2.2 swap |
| `Architectures/LayerNorm.lean` | drop `Mathlib.Analysis.SpecialFunctions.Pow.Real` (−50) |
| `Training/Optim/AdamStep.lean` | drop `Mathlib.Analysis.SpecialFunctions.Sqrt` (−15) |
| `Foundation/Tensor.lean` | drop `Mathlib.Analysis.Calculus.FDeriv.Pi` (−1) |
| all of the above + `StableHLO`, `Attention` | their implied imports (§2.3) |
| `Codegen/StableHLO.lean` | move the chapter-net graphs `mlpFwdGraph`, `mlpBackGraph`, `cnnFwdGraph`, `cnnBackGraph`, `cifarFwdGraph`, `cifar8FwdGraph`, `cifar8BnFwdGraph` out (users: `ChapterGraphTies`, `SpecVJP`, their printers) |
| `Codegen/StableHLOPretty.lean` | with them, the printers `linear{Fwd,Back,TrainStep}ModuleV`, `mlpFwdModuleV`, `cnnFwdModuleV`, `cifarFwdModuleV`, `cifar8FwdModuleV`, `cifar8BnFwdModuleV` (users: `ChapterArtifacts`'s `#eval`s, `LinearFold`); and `mnv2RmsHyper` → `Codegen/MobileNetV2RenderB.lean` (its only user) |
| `Codegen/StableHLO.lean:498`, `:525` | the comments name the deleted `ResNet34Fold` / `MobileNetV2Fold`; point them at `SgdNodes` |
| new chapter render module under `Codegen/` | receives the graphs and printers; `ChapterGraphTies`, `ChapterArtifacts`, `SpecVJP`, `LinearFold` import it |
| `Float/FloatBridge.lean` | split the MLP/MNIST chain into `Float/MlpFloatBridge.lean` (the pattern `ResNet34FloatBridge` already follows): `mlpF`, `mlp_float_close`, `mlp_float_close_uniform`, `mlp_l1_close`, `mlp_{w,b}{0,1,2}_step_float_close`, `mnist_mlp_float_budget`, `mnist_w2_step_float_budget`, `mnist_cot_budget` with the private `mnist_E{0,1}_*`, `linear_e4m3_logit_budget`, `linear_e4m3_argmax_preserved`; none has a Lean user, so `tests/AuditAxioms.lean` and the `Certs` roots are the consumers |

The chapter-graph move is what makes a chapter net's edit stop rebuilding 315 modules. The
`ChapterArtifacts` `#eval` writers must still byte-reproduce `verified_mlir/` (`regen_verified_mlir.sh`
and the `proofs.yml` diff guard).

## 5. Batch D — forwarding modules

### 5.1 `LeanMlir/VerifiedNets.lean`

Four imports, no declarations, 95 importers (53 in apps/demos/lib, the rest in `tests/`). Minimal
cover among the 53: 40 need `VerifiedNetsCore` + `VerifiedTrain`, 7 `Core` + `VerifiedAttack`,
4 `Core` + `VerifiedSmoothing`, 1 `VerifiedTrain` alone, and `SyncBnCheck` nothing (§2.2). Delete
it and give each importer its pair; recompute the cover for `tests/` when doing it. Small build
win; the reason is that each entry point then states what it uses.

### 5.2 `export FloatFmt (fmt)` in `Blackjack.lean:10`, `Pong.lean:9`

A compatibility alias from 290187b1 ("so no caller changes"). Replace each with `open FloatFmt`
and add `open FloatFmt` to `MainBlackjackEnv`, `MainBlackjackDqn`, `MainPongEnv`, `MainPongDqn`.

## 6. Batch E — directory restructure (one change)

Pure `git mv` plus import edits, no renames. Counts are files naming the modules (Lean / other:
lakefile, scripts, CI, docs), measured before any batch above.

| # | group | layout | refs |
|---|---|---|---|
| 1 | `Certificates/LipschitzCert*` (18) | `Certificates/LipschitzCert/{Basic,Float,Instance,PairSDP,Scorecard,ScorecardSDP,ScorecardFull,…}` | 27 / 19; generators write these paths |
| 2 | `Certificates/Smoothing*` (14) | `Certificates/Smoothing/{CP,CPScorecard,DecChunk1..6,DecScorecard,Gaussian,MC,NetSemantics,NetWitness,PhiBounds}` | 18 / 8; generators |
| 3 | `Certificates/IbpConvScorecard*` (6) | `Certificates/IbpConvScorecard/{Basic,ImgsA..D,Net}` | 7 / 4; generator |
| 4 | `Foundation/DataParallel*` (5) | `Foundation/DataParallel/{Basic,Node,Sync,SyncBf16,SyncKit}` | 20 / 12 |
| 5 | `Foundation/Batched*` (4) | `Foundation/Batched/{Basic,Stages,StageLayers,BackLinks}` | 14 / 1 |
| 6 | `Training/SgdDescent*` (5) | `Training/SgdDescent/{Basic,Linear,Mlp,Cnn,Cifar}` | 10 / 4 |
| 7 | `Training/Trained*` (4) | `Training/Trained/{CnnSeal,CnnWitness,LinearDescent,MlpWitness}` | 5 / 8 |
| 8 | `Foundation/Muon*` (2) | `Foundation/Muon/{Geometry,NewtonSchulz}` | 5 / 5 |
| 9 | `Codegen/StableHLO*` (4) | `Codegen/StableHLO/{Basic,Lex,Parse,Pretty}` | 80 / 14; with §4, one rebuild |
| 10 | `Codegen/EfficientNetRender*` (3) | `Codegen/EfficientNetRender/{Basic,PC,PCEval}` | 19 / 9 |
| 11 | `LeanMlir/Verified*` (7, 6 after §5.1) | `LeanMlir/Verified/{Attack,NetsCore,PgdGen,Smoothing,Spec,Train}` | 111 / 83; after §5.1 |

Optional: `Architectures/{ChannelLN,ChannelLNBack}` → `ChannelLN/{Basic,Back}`,
`{PerChannelBN,PerChannelBNGrad}` → `PerChannelBN/{Basic,Grad}`,
`Foundation/{IntervalBoundConv,IntervalBoundConvQ}` → `IntervalBoundConv/{Basic,Q}`.

Not proposed: `Nets/<family>/` already groups the per-net files (2026-09-08 layout); splitting
`ResNet34*` / `ResNet50*` one level deeper adds depth without making anything easier.

## 7. Generators

The scorecard generators under `scripts/certs/` emit the same theorem into several files: about 56
`Expr`-equal pairs across `LipschitzCertScorecard{,SDP,Full,SDPFull,SDPFullUncon,Crown,CrownUncon,IBP,IBPUncon}`
(`hpsumSF*` = `hpreSF*_sum`, `certSF*` = `certifiedSSF*`, `pzTF_*` = `pzITF_*`), and
`LipschitzCertScorecardIBPData`'s three imports are interchangeable. Emit shared lemmas once into a
file both scorecards import; do it together with §6 rows 1–3, since those change the output paths anyway.

## 8. Handed to the naming pass

Generic lemmas in the right file under a net's name or namespace: `ResNet34PoCB.*`,
`Mnv2PaperPoCG.*`, `CnxPoCGB.*` (`GradNodesB`); `ResNet34SyncTieB.*`, `MBConvSyncTieB.*`
(`DataParallelSyncKit`); `ResNet34TieB.*` (`BatchedBackLinks`); `Cifar8PoC.*`, `CifarPoC.*`,
`CifarBnPoC.*` (`SgdNodes`); `R34FullBSeal.*` (`BatchSealKit`); `vit_*` (`TokenParamGrad`,
`LayerNorm`); `cifar_bn_*` (`PerChannelBNGrad`); `r34PoolLayer` / `R34PoolSmoothAt` (`HeadLayers`,
shared by R34 and R50). §3's moved `mnv2_*` names join this list.

## 9. Open

* Keep the audit's .olean `#min_imports` pass as a script (and possibly a CI ratchet on unused and
  implied imports), so the tables above do not regrow.
* After each batch, re-run it: the tables were measured once, and §2 rows can interact (the
  either-or pairs).
