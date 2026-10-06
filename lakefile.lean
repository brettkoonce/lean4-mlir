import Lake
open Lake DSL

package «lean4-mlir» where
  version := v!"0.7.1"
  buildType := .release

-- doc-gen4 is a conditional dependency only activated when the CI
-- (or a local user) passes `-Kenv=dev`. Without that flag, this
-- require is inert, so normal `lake build` invocations don't pull it
-- in. Standard pattern used by Mathlib, PrimeNumberTheoremAnd,
-- Carleson, FLT, etc.
--
-- CRITICAL: Mathlib must be required LAST so its version constraints
-- for shared transitive deps (plausible, etc.) take precedence over
-- doc-gen4's. Otherwise `lake exe cache get` can't find matching
-- Mathlib archives and the build fails.
meta if get_config? env = some "dev" then
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.34.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.34.0"

-- The umbrella root: codegen + runtime FFI + trainers' shared modules + the part of the
-- proof suite `LeanMlir.lean` imports (the per-chapter headline modules, a minority of
-- `LeanMlir/`). NOT the whole repo: `Certs` below is the full proof corpus, `Apps`
-- the entry points.
lean_lib «LeanMlir» where
  roots := #[`LeanMlir]

-- Scoped targets, so CI and contributors can build one slice without the
-- rest, along the seam between the proof suite and the program side. `Proofs` reaches no
-- program module. `Certs` reaches four, all pure data (`Verified.Spec`, `ParamLayouts` and its
-- byte writer `LEBytes`, `Verified.NetsCore`), through `SpecVJP`, the executable-spec ↔ proof
-- bridge — never the trainer or the FFI.

/-- **`lake build Proofs`** — the fast per-push slice: the IR/render layer
    every demo's import cone actually reaches (StableHLO/IR + the per-net
    op/VJP modules the proven renderers are built on) plus every renderer the
    CI drift guard re-elaborates. Builds in minutes. The
    certificate corpus lives in `Certs` below and is checked by its own
    workflow (.github/workflows/certs.yml: proof-path pushes + nightly cron),
    so demo/book/engine pushes stop paying the multi-hour corpus tail. -/
@[default_target]
lean_lib «Proofs» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Architectures.Attention,
             `LeanMlir.Proofs.Architectures.CNN,
             `LeanMlir.Proofs.Architectures.Depthwise,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNet,
             `LeanMlir.Proofs.Nets.Small.MnistCNN,
             `LeanMlir.Proofs.Architectures.StridedConv,
             `LeanMlir.Proofs.Architectures.PerChannelBN,
             `LeanMlir.Proofs.Foundation.IR,
             `LeanMlir.Proofs.Foundation.PairTile,
             `LeanMlir.Proofs.Codegen.StableHLO.Basic,
             `LeanMlir.Proofs.Codegen.StableHLO.Pretty,
             `LeanMlir.Proofs.Codegen.ChapterArtifacts,
             `LeanMlir.Proofs.Codegen.MlpArtifacts,
             `LeanMlir.Proofs.Codegen.CnnArtifacts,
             `LeanMlir.Proofs.Codegen.ResNet34RenderB,
             `LeanMlir.Proofs.Codegen.ResNet50RenderB,
             `LeanMlir.Proofs.Codegen.MobileNetV2RenderB,
             `LeanMlir.Proofs.Codegen.MobileNetV4RenderB,
             `LeanMlir.Proofs.Codegen.EfficientNetRender.Basic,
             `LeanMlir.Proofs.Codegen.ConvNeXtRender,
             `LeanMlir.Proofs.Codegen.ConvNeXtRenderB,
             `LeanMlir.Proofs.Codegen.ViTRender,
             `LeanMlir.Proofs.Codegen.ViTRenderB]

/-- **`lake build Certs`** — the certificate corpus (the certified ties, seals, descent, Lipschitz/LipSDP, smoothing,
    Muon, the float model, …): the VJP proof suite's apex modules; their transitive
    imports cover every proof file (they subsume the `Proofs` roots above, so
    building `Certs` builds everything the axiom audit needs). Built +
    3-axiom-audited by .github/workflows/certs.yml, NOT by the per-push proofs
    workflow. -/
lean_lib «Certs» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Architectures.Attention,
             `LeanMlir.Proofs.Architectures.CNN,
             `LeanMlir.Proofs.Architectures.Depthwise,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNet,
             `LeanMlir.Proofs.Nets.Small.MnistCNN,
             `LeanMlir.Proofs.Training.JacobianSeal,
             `LeanMlir.Proofs.Architectures.StridedConv,
             `LeanMlir.Proofs.Architectures.PerChannelBN,
             `LeanMlir.Proofs.Foundation.MatBridge,
             `LeanMlir.Proofs.Foundation.IR,
             `LeanMlir.Proofs.Foundation.PairTile,
             `LeanMlir.Proofs.Codegen.StableHLO.Basic,
             `LeanMlir.Proofs.Codegen.StableHLO.Pretty,
             `LeanMlir.Proofs.Codegen.FwdGraphTextTies,
             `LeanMlir.Proofs.Codegen.StableHLO.Parse,
             `LeanMlir.Proofs.Nets.Small.LinearTrainStep,
             `LeanMlir.Proofs.Nets.Small.MlpTrainStep,
             `LeanMlir.Proofs.Architectures.ConvGrad,
             `LeanMlir.Proofs.Codegen.ChapterArtifacts,
             `LeanMlir.Proofs.Codegen.MlpArtifacts,
             `LeanMlir.Proofs.Codegen.CnnArtifacts,
             `LeanMlir.Proofs.Architectures.PerChannelBNGrad,
             `LeanMlir.Proofs.Nets.Small.CnnChainClose,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPC,
             `LeanMlir.Proofs.Nets.ViT.ViTFwdGraph,
             `LeanMlir.Proofs.Architectures.TokenParamGrad,
             `LeanMlir.Proofs.Nets.ViT.ViTChainClose,
             `LeanMlir.Proofs.Nets.ViT.ViTVecLN,
             `LeanMlir.Proofs.Nets.ViT.ViTDepthK,
             `LeanMlir.Proofs.Nets.ViT.ViTFwdDrop,
             `LeanMlir.Proofs.Nets.ViT.ViTMultiHeadChain,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Eval,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Drop,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaper,
             `LeanMlir.Proofs.Float.FloatBridge,
             `LeanMlir.Proofs.Training.SgdDescent.Basic,
             `LeanMlir.Proofs.Training.SgdDescent.Linear,
             `LeanMlir.Proofs.Training.SgdDescent.Mlp,
             `LeanMlir.Proofs.Training.SgdDescent.MlpBias,
             `LeanMlir.Proofs.Training.SgdDescent.Cnn,
             `LeanMlir.Proofs.Training.SgdDescent.CnnFloat,
             `LeanMlir.Proofs.Training.SgdDescent.Cifar,
             `LeanMlir.Proofs.Float.BnFloatBridge,
             `LeanMlir.Proofs.Float.ResNet34FloatBridge,
             `LeanMlir.Proofs.Float.BnInputBridge,
             `LeanMlir.Proofs.Float.FloatComposeBridge,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPCEval,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaperEval,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStagesPCEval,
             `LeanMlir.Proofs.Foundation.BatchMapVJPAt,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34FullB,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34FullBSeal,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50FullBSeal,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBSeal,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBSeal,
             `LeanMlir.Proofs.Foundation.GradNodesB,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldG,
             `LeanMlir.Proofs.Nets.ViT.ViTFoldG,
             `LeanMlir.Proofs.Nets.ViT.ViTFoldGB,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldGB,
             `LeanMlir.Proofs.Foundation.Bf16GradNodes,
             `LeanMlir.Proofs.Foundation.SmoothedLossCot,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StepTieB,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTieG,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB,
             `LeanMlir.Proofs.Nets.ViT.ViTStepTieGB,
             `LeanMlir.Proofs.Foundation.DataParallel.Basic,
             `LeanMlir.Proofs.Foundation.DataParallel.Node,
             `LeanMlir.Proofs.Foundation.DataParallel.Sync,
             `LeanMlir.Proofs.Foundation.DataParallel.SyncBf16,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB,
             `LeanMlir.Proofs.Foundation.DataParallel.SyncKit,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncStepTieB,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncB,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncStepTieG,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50SyncB,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50SyncStepTieB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncStepTieB,
             `LeanMlir.Proofs.Codegen.LambTriple,
             `LeanMlir.Proofs.Foundation.BceLossCot,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50FullB,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBEval,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBDrop,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBVJP,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4StepTieB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4WholeBackCertifiedTieB,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB,
             `LeanMlir.Proofs.Foundation.ParamGrad,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34ParamGrad,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50ParamGrad,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2ParamGrad,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4ParamGrad,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetParamGrad,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtParamGrad,
             `LeanMlir.Proofs.Nets.ViT.ViTParamGrad,
             `LeanMlir.Proofs.Foundation.BackwardMaps,
             `LeanMlir.Proofs.Architectures.ChannelLNBack,
             `LeanMlir.Proofs.Nets.ResNet.ResNetBackChains,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackChains,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackChains,
             `LeanMlir.Proofs.Float.ConvMixedComposeBridge,
             `LeanMlir.Proofs.Float.DepthwiseMixedFloatBridge,
             `LeanMlir.Proofs.Architectures.ConvBackCertifiedTie,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetWholeBackCertifiedTie,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullWholeBackCertifiedTie,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2WholeBackCertifiedTieB,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50WholeBackCertifiedTieB,
             `LeanMlir.Proofs.Architectures.EvenKernelConvBack,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTie,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTieB,
             `LeanMlir.Proofs.Nets.ViT.ViTBackChains,
             `LeanMlir.Proofs.Nets.ViT.ViTVecLNBackCertifiedTie,
             `LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTie,
             `LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTieB,
             `LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackCertifiedTie,
             `LeanMlir.Proofs.Nets.ViT.ViTMhsaBackCertifiedTie,
             `LeanMlir.Proofs.Training.Optim.AdamStep,
             `LeanMlir.Proofs.Training.Optim.SgdMomentumStep,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackB0,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV2BackB0,
             `LeanMlir.Proofs.Nets.ResNet.ResNet34BackB0,
             `LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0,
             `LeanMlir.Proofs.Foundation.CertifiedChain,
             `LeanMlir.Proofs.Nets.MobileNet.MobileNetV4BackB0,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackNet,
             `LeanMlir.Proofs.Nets.ViT.ViTBackNet,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackB0,
             `LeanMlir.Proofs.Nets.ViT.ViTBackB0,
             `LeanMlir.Proofs.Nets.Small.LinearFold,
             `LeanMlir.Proofs.Float.E4M3Fold,
             `LeanMlir.Proofs.Float.Bf16Fold,
             `LeanMlir.Proofs.Nets.Small.MlpFold,
             `LeanMlir.Proofs.Nets.Small.CnnFold,
             `LeanMlir.Proofs.Nets.Small.CifarFold,
             `LeanMlir.Proofs.Foundation.SgdNodes,
             `LeanMlir.Proofs.Nets.Small.Cifar8StepTie,
             `LeanMlir.Proofs.Nets.Small.Cifar8BnStepTie,
             `LeanMlir.Proofs.Nets.Small.Cifar8StepTieG,
             `LeanMlir.Proofs.Nets.Small.Cifar8BnStepTieG,
             `LeanMlir.Proofs.Nets.Small.SmallParamGrad,
             `LeanMlir.Proofs.Nets.Small.LinearParamGrad,
             `LeanMlir.Proofs.Nets.Small.MlpParamGrad,
             `LeanMlir.Proofs.Nets.Small.CnnParamGrad,
             `LeanMlir.Proofs.Nets.Small.CifarParamGrad,
             `LeanMlir.Proofs.Nets.Small.Cifar8ParamGrad,
             `LeanMlir.Proofs.Nets.Small.Cifar8BnParamGrad,
             `LeanMlir.Proofs.Codegen.EfficientNetRender.Basic,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFold,
             `LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTie,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFold,
             `LeanMlir.Proofs.Codegen.ConvNeXtRender,
             `LeanMlir.Proofs.Codegen.ConvNeXtRenderB,
             `LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTie,
             `LeanMlir.Proofs.Codegen.ViTRender,
             `LeanMlir.Proofs.Codegen.ViTRenderB,
             `LeanMlir.Proofs.Nets.ViT.ViTFold,
             `LeanMlir.Proofs.Nets.ViT.ViTStepTie,
             `LeanMlir.Proofs.Certificates.LipschitzCert.Basic,
             `LeanMlir.Proofs.Foundation.UpstreamDraft,
             `LeanMlir.Proofs.Certificates.Smoothing.Gaussian,
             `LeanMlir.Proofs.Certificates.Smoothing.MC,
             `LeanMlir.Proofs.Certificates.Smoothing.CP,
             `LeanMlir.Proofs.Certificates.Smoothing.CPScorecard,
             `LeanMlir.Proofs.Certificates.Smoothing.PhiBounds,
             `LeanMlir.Proofs.Certificates.Smoothing.DecScorecard,
             `LeanMlir.Proofs.Certificates.Smoothing.NetSemantics,
             `LeanMlir.Proofs.Certificates.Smoothing.NetWitness,
             `LeanMlir.Proofs.Foundation.Muon.Geometry,
             `LeanMlir.Proofs.Foundation.Muon.NewtonSchulz,
             `LeanMlir.Proofs.Certificates.LipschitzCert.Instance,
             `LeanMlir.Proofs.Training.Trained.MlpWitness,
             `LeanMlir.Proofs.Certificates.LipschitzCert.Scorecard,
             `LeanMlir.Proofs.Certificates.LipschitzCert.PairSDP,
             `LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardSDP,
             `LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardSDPUncon,
             `LeanMlir.Proofs.Foundation.ListDot,
             `LeanMlir.Proofs.Certificates.IntervalBound,
             `LeanMlir.Proofs.Certificates.IntervalBoundConv.Basic,
             `LeanMlir.Proofs.Certificates.CrownBound,
             `LeanMlir.Proofs.Float.Binary32Instance,
             `LeanMlir.Proofs.Training.Trained.LinearDescent,
             `LeanMlir.Proofs.Training.Trained.CnnWitness,
             `LeanMlir.Proofs.Training.Trained.CnnSeal,
             `LeanMlir.Proofs.Training.Trained.CnnDescent,
             `LeanMlir.Proofs.Training.Trained.CnnDescentConv1,
             `LeanMlir.Proofs.Certificates.LipschitzCert.Float,
             `LeanMlir.Proofs.SpecVJP,
             `LeanMlir.Proofs.Nets.Small.MlpCanonical,
             `LeanMlir.Proofs.Codegen.ResNet34RenderB,
             `LeanMlir.Proofs.Codegen.ResNet50RenderB,
             `LeanMlir.Proofs.Codegen.MobileNetV2RenderB,
             `LeanMlir.Proofs.Codegen.MobileNetV4RenderB]

/-- **`lake build CertsHeavy`** — the GENERATED full-input certificate
    instances: the 784-dim L2 scorecard (`ScorecardFull`), dense IBP L∞
    (`ScorecardIBP{,Uncon}`), CROWN (`ScorecardCrown{,Uncon}`) and the conv IBP net
    (`IbpConvScorecard`), weight/image data plus per-image theorems. Split out of `Certs` for
    per-module peak elaboration memory, which took down the shared runners — long-running
    corpus work gets its OWN workflow (.github/workflows/certs-heavy.yml: weekly cron +
    on-demand + pushes touching these files) so it can never break the core. Every theorem is
    audited by tests/AuditAxiomsHeavy.lean; the measured per-100 counts are in each generated
    file's header, the proved ones are the aggregates' lengths. The pooled LipSDP files are in
    `Certs`; the full-input LipSDP files (`LipschitzCert/ScorecardSDPFull{,Uncon}`) are in NO
    lib: their linarith PSD witnesses OOM the runners, so their audit lines are commented out.
    Re-check them locally with `scripts/certs/check_sdpfull.sh`. -/
lean_lib «CertsHeavy» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardFull,
             `LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardIBP,
             `LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardIBPUncon,
             `LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardCrown,
             `LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardCrownUncon,
             `LeanMlir.Proofs.Certificates.IbpConvScorecard.Basic]

/-- **`lake build ProofsMinimal`** — the suite's "hello world": the smallest
    end-to-end story (the Linear classifier), both halves — faithfulness
    (`LinearFold`: emitted train-step = certified math) and descent
    (`SgdDescent.Linear`: that step decreases the loss). Their transitive closure is
    exactly the minimum working set (LinearTrainStep + the shared StableHLO/Tensor/
    FloatBridge/IR foundation), nothing per-net beyond Linear. Point a newcomer here
    before the full `Proofs` target. See `LeanMlir/Proofs/README.md` (Start here). -/
lean_lib «ProofsMinimal» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Nets.Small.LinearFold, `LeanMlir.Proofs.Training.SgdDescent.Linear]

/-- **`lake build Reference`** — the REFERENCE path's codegen + spec core, no proofs:
    `MlirCodegen` (NetSpec → MLIR at run time, unverified) and the modules around it. It does
    not build the verified path (`Verified.Spec`/`Verified.NetsCore`/`Verified.Train`, which load
    `verified_mlir/`) or anything in `LeanMlir/Proofs/Codegen/`. The blueprint workflow builds it,
    because checkdecls imports its roots. -/
lean_lib «Reference» where
  srcDir := "."
  roots := #[`LeanMlir.MlirCodegen, `LeanMlir.Train, `LeanMlir.Spec, `LeanMlir.ReferenceNets,
             `LeanMlir.SpecHelpers, `LeanMlir.Types, `LeanMlir.IreeRuntime,
             `LeanMlir.Ddpm, `LeanMlir.Cam, `LeanMlir.F32Array]

/-- **`lake build Apps`** — type-check every entry point, without linking any of them.

    Without it nothing in CI builds an `apps/` or `demos/` exe, and `lake build` builds
    `Proofs` alone, so an entry point could keep a stale call for weeks and be found only
    when someone ran it. Linking is what makes the exes expensive (each is a large static
    binary, and there are hundreds of them); elaborating their modules is not, which is the whole
    reason this is a `lean_lib` over the same sources rather than a `lake build <exe>` loop.

    Covers `apps/` and `demos/`, and NOT `tests/`, which holds
    `tests/comparator/`, a nested Lake package with its own toolchain whose modules a
    `.submodules` glob would try to elaborate here. The test exes CI already names
    (`argmax-check`, `label-check`, `opt-step-fixtures`, `bestiary-*`) keep their own
    explicit builds. The NAME-level companion is `scripts/gates/check_target_names.sh`. -/
lean_lib «Apps» where
  srcDir := "."
  -- `roots := #[]` is not redundant. Lake defaults `roots` to `#[<lib name>]`, and there is
  -- no `Apps.lean` — the lib IS its globs. Leaving the default makes `docstring-checkrefs` and
  -- `blueprint-checkdecls` die with `unknown module prefix 'Apps'`: both walk
  -- `ws.root.leanLibs` and import every lib's `roots`.
  roots := #[]
  globs := #[.submodules `apps, .submodules `demos]

/-- **`lake build TestSupport`** — modules the `lake env lean tests/…` scripts import:
    `tests/ViTRender.lean`, the hand-written ViT/AdamW StableHLO fragments nine tests use as a
    numeric reference. Build it before running those scripts. -/
lean_lib «TestSupport» where
  srcDir := "."
  roots := #[`tests.ViTRender]

-- IREE FFI shim: Lean ↔ C bridge for libiree_ffi.so (see ffi/).
target ireeLeanFfiO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "iree_lean_ffi.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "iree_lean_ffi.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString,
                    "-I", (pkg.dir / "ffi").toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

-- F32 ByteArray helpers (He init, argmax, data loading — all in C for speed).
target f32HelpersO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "f32_helpers.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "f32_helpers.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

-- Runtime lowerer dispatch: dlopens libpjrt_ffi.so / libiree_ffi.so and resolves
-- the ten-symbol shim surface at run time rather than from the link line, so one
-- binary serves both backends via $LEAN_MLIR_LOWERER. See ffi/lowerer.h.
target lowererO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "lowerer.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "lowerer.c"
  let weakArgs := #["-I", (pkg.dir / "ffi").toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

extern_lib libireeffi pkg := do
  let shimO ← fetch <| pkg.target ``ireeLeanFfiO
  let f32O  ← fetch <| pkg.target ``f32HelpersO
  let lowO  ← fetch <| pkg.target ``lowererO
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "ireeffi") #[shimO, f32O, lowO]

-- ═══════════════════════════════════════════════════════════════════
-- Phase 3 trainers (Lean → StableHLO → XLA/PJRT (default) or IREE → GPU)
-- ═══════════════════════════════════════════════════════════════════

/-- No shim on the link line at all. `ffi/lowerer.c` dlopens whichever of
    `libpjrt_ffi.so` / `libiree_ffi.so` `$LEAN_MLIR_LOWERER` selects (XLA by
    default, `=iree` for the other), so ONE executable serves both backends and
    a box that has only one of the two shims still starts. `lowerer.h` `#define`s every
    `iree_ffi_*` / `pjrt_ffi_*` entry point onto a dlopen'd pointer, so the executable
    resolves nothing against either library at link time.

    The selected shim is still REQUIRED at run time: a missing `ffi/libpjrt_ffi.so` or
    `ffi/libiree_ffi.so` fails at startup, not at link. `ensurePjrtShim` builds the PJRT one on
    demand. When checking a binary's link with `nm -D`, `lake build` it first: an old binary on
    disk can carry an older link scheme. -/
private def lowererLink : Array String := #["-ldl"]

-- ═══════════════════════════════════════════════════════════════════════
-- THE TOUR — the canonical set: the four
-- `lake run` tiers, the nine demos and the two gates. New readers are pointed here and
-- nowhere else; every number the README and the book quote comes from one of these.
-- ═══════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════
-- XLA/PJRT backend
--
-- Same Lean root, same verified_mlir/*.mlir, same ties — the ONLY change is
-- which trusted lowerer consumes the emitted StableHLO. `libpjrt_ffi.so` exports
-- the identical C surface as `libiree_ffi.so`, so nothing above the shim moves.
--
-- Build the shim (it is not built by lake — it only needs libc + dlopen):
--   gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
-- Still required, but at RUN time, not link time: nothing names the shim on a link
-- line, so a missing shim is a dlopen failure when the trainer starts rather than a
-- link error when it builds.
-- `ensurePjrtShim` (below) is what keeps that from biting on a fresh clone.
-- ═══════════════════════════════════════════════════════════════════

-- ─── Tier 1 — `lake run mnist`: MNIST linear / MLP / CNN on the verified renders (Chapters 1–3) ───

-- Trains MNIST-linear on the VERIFIED-rendered StableHLO
-- (`verified_mlir/`, = Proofs.StableHLO.linTrainStepText) through the
-- real Lean/IREE FFI. See MainMnistLinearVerified.lean.

lean_exe «mnist-linear-verified» where
  root := `apps.mnist.MainMnistLinearVerified
  moreLinkArgs := lowererLink

-- Rung 0 of the XLA ladder is a RUN-TIME choice, not a second executable:
-- `mnist-linear-verified` serves both lowerers via $LEAN_MLIR_LOWERER. G2 is the
-- SAME binary run twice, which is a stronger comparison than two binaries.

-- Chapter 3: trains the MNIST MLP on the VERIFIED-rendered StableHLO
-- (verified_mlir/mlp_train_step.mlir = Proofs.StableHLO.mlpTrainStepText).

lean_exe «mnist-mlp-verified» where
  root := `apps.mnist.MainMnistMlpVerified
  moreLinkArgs := lowererLink

-- Rung 1 of the XLA ladder: depth + multiple param tensors via the packed-params
-- path, and the first rung with He init. It is a RUN-TIME choice:
-- `mnist-mlp-verified` serves both lowerers via $LEAN_MLIR_LOWERER.

-- Chapter 4: trains the MNIST CNN on the VERIFIED-rendered StableHLO
-- (verified_mlir/cnn_train_step.mlir = Proofs.StableHLO.cnnTrainStepText).
/-- The verified CNN trainer — one executable for both the IREE and XLA lowerers,
    so their config and He-init seed cannot drift. -/

lean_exe «mnist-cnn-verified» where
  root := `apps.mnist.MainMnistCnnVerified
  moreLinkArgs := lowererLink

-- The first CONVOLUTIONAL graph on the XLA ladder — where IREE's slow
-- conv codegen actually bites, unlike the dense-only rungs 0-1.
-- The conv rung of the XLA ladder is a RUN-TIME choice:
-- `mnist-cnn-verified` serves both lowerers via $LEAN_MLIR_LOWERER.

-- ─── Tier 2 — `lake run cifar`: the wide 8-conv CIFAR net, SGD / momentum / AdamW × no-BN / BN (Chapter 4) ───

-- Wide-head (MNIST-style 2×512 dense, d1=512) cifar8 optimizer ablation: each exe runs SGD /
-- momentum / AdamW in sequence on the controlled pipeline. Render: Proofs/Codegen/CnnRender.lean at d1 := 512.
lean_exe «cifar8w-ablation» where
  root := `apps.cifar.MainCifar8WideAblation
  moreLinkArgs := lowererLink

lean_exe «cifar8w-bn-ablation» where
  root := `apps.cifar.MainCifar8WideBnAblation
  moreLinkArgs := lowererLink

-- ─── Tier 3 — `lake run imagenette`: the seven Part-1 nets at 224², book order (Chapters 5–9) ───

-- The suite is one chain per net, the batched one. The batched SGD trainer is
-- `LEAN_MLIR_VARIANT=sgd .lake/build/bin/resnet34-verified-adam`, which renders from `OptRecipe.sgd`
-- and threads running stats (running-stat threading lives only in `trainAdamSched`).

lean_exe «resnet34-verified-adam» where
  root := `apps.imagenette.MainResnet34VerifiedAdam
  moreLinkArgs := lowererLink

lean_exe «resnet50-verified-adam» where
  root := `apps.imagenette.MainResnet50VerifiedAdam
  moreLinkArgs := lowererLink

-- `mobilenetv2-verified-adam` is the MobileNetV2 trainer, on the batched chain.

lean_exe «mobilenetv2-verified-adam» where
  root := `apps.imagenette.MainMobilenetV2VerifiedAdam
  moreLinkArgs := lowererLink

/-- 80ep, bs32, AdamW; the target is `historical/RESULTS.md`'s MobileNetV4 row. XLA/PJRT only — no
    IREE peer exists yet, and the body is backend-agnostic if one is wanted. -/
lean_exe «mobilenetv4-verified-adam» where
  root := `apps.imagenette.MainMobilenetV4VerifiedAdam
  moreLinkArgs := lowererLink

lean_exe «efficientnet-verified-adam» where
  root := `apps.imagenette.MainEfficientNetVerifiedAdam
  moreLinkArgs := lowererLink

lean_exe «convnext-verified-adam» where
  root := `apps.imagenette.MainConvNeXtVerifiedAdam
  moreLinkArgs := lowererLink

lean_exe «vit-verified-adam» where
  root := `apps.imagenette.MainViTVerifiedAdam
  moreLinkArgs := lowererLink

-- ─── Tier 4 — `lake run imagenet`: the ImageNet-1k runners; a row is scripts/jobs/<job>.conf + scripts/supervise.sh ───

/-- **ResNet-34 on full 1000-class ImageNet** — the scale/reference tier. Same certified renderer at
    `nClasses := 1000, B := 256`, heavy-ball + coupled L2 (the `jax/MainResnetImagenet.lean` recipe),
    fed by the generated tfds shim so both paths see identical augmented batches.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). Not
    `lake exe resnet34-imagenet default --shim` — that is R34's, and every net fed from it
    streams R34's augmentation.
    Does NOT move the verification tier — proofs stop at Imagenette. -/
lean_exe «resnet34-imagenet-verified» where
  root := `apps.imagenette.MainResnet34Imagenet
  moreLinkArgs := lowererLink

/-- **ResNet-50 on full 1000-class ImageNet** — R50 phase 3. The bottleneck renderer
    (`ResNet50RenderB`) at `nClasses := 1000`, AdamW, 4-replica by default.
    NOT RSB-A3 — no LAMB, no bs2048, no gradient accumulation. And no incumbent render to tie
    against, so name the check that licenses any number off it. -/
lean_exe «resnet50-imagenet-verified» where
  root := `apps.imagenette.MainResnet50Imagenet
  moreLinkArgs := lowererLink

/-- **MobileNetV2 on full 1000-class ImageNet** — the fifth scale-tier trainer. `nClasses := 1000,
    B := 64`; four replicas is global 256, the reference's batch. Batch-BN, so it has a `_fwd_eval`
    peer and a running-stat region.

    Optimizer does NOT match the reference (RMSProp there, AdamW here). -/
lean_exe «mobilenetv2-imagenet-verified» where
  root := `apps.imagenette.MainMobileNetV2Imagenet
  moreLinkArgs := lowererLink

/-- **MobileNetV4-Conv-M on full 1000-class ImageNet** — the sixth scale-tier trainer.
    `nClasses := 1000, B := 64`; four replicas would be global 256. Batch-BN, so it has a
    `_fwd_eval` peer and a running-stat region.

    **Conv-M.** `mnv4ImagenetVerified` (`Verified.NetsCore`) carries the Conv-M block table and
    names itself "MobileNetV4-Conv-M (ImageNet-1k)".

    The job `mnv4-default-4gpu` runs the JAX reference's `default` recipe (AdamW 0.004 at effective
    batch 4096 by accumulation, EMA, wd 0.05 off norm/bias, classifier dropout); drop-path in the
    UIB blocks, used only by the 500-epoch paper tier, is absent on this path. -/
lean_exe «mobilenetv4-imagenet-verified» where
  root := `apps.imagenette.MainMobilenetV4Imagenet
  moreLinkArgs := lowererLink

/-- **EfficientNet-B0 on full 1000-class ImageNet**. Same certified renderer at `nClasses := 1000,
    B := 64`; four replicas is global 256, the reference's batch. The first ImageNet net here with
    BatchNorm, so it has a `_fwd_eval` peer and a running-stat region.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). Not
    `lake exe resnet34-imagenet default --shim` — that is R34's, and every net fed from it
    streams R34's augmentation.
    Optimizer does NOT match the reference (RMSProp there, AdamW here). -/
lean_exe «efficientnet-imagenet-verified» where
  root := `apps.imagenette.MainEfficientNetImagenet
  moreLinkArgs := lowererLink

/-- **ConvNeXt-T on full 1000-class ImageNet**. Same certified renderer at `nClasses := 1000`;
    batch stays 32 per device (`cBS` is still private), so four replicas is global 128 and 10,009
    steps/epoch — more optimizer steps than the reference's 5,004 at batch 256.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). Not
    `lake exe resnet34-imagenet default --shim` — that is R34's, and every net fed from it
    streams R34's augmentation.
    Does NOT move the verification tier, and is NOT the ConvNeXt paper recipe. -/
lean_exe «convnext-imagenet-verified» where
  root := `apps.imagenette.MainConvNeXtImagenet
  moreLinkArgs := lowererLink

/-- **ConvNeXt-Small on ImageNet-1k** — ConvNeXt-T DEEPENED (stage 3 goes 9 → 27 blocks, dims
    UNCHANGED at [96,192,384,768]), 342 param tensors / 50,222,152 scalars, the published 50.22M.

    Added by reshaping a renderer rather than writing a chain: S is pure depth, so one `Array Nat`
    threaded as a trailing defaulted parameter covers it and every hardcoded dimension literal stays
    correct. The proof side needed nothing — ConvNeXt's certificates are per-site and generic in
    `c`/`e`/`h`, so depth was never a hypothesis. Every ConvNeXt-T artifact re-renders
    byte-identical.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (it reuses ConvNeXt-T's, which
    is correct — the paper does not change the data pipeline between T and S).
    Stochastic depth is **0.4**, the paper's S value, not Tiny's 0.1 — the one recipe knob that
    moves with model size. Use `LEAN_MLIR_DROP_RATE_U=200000` on the 80-epoch tier.
    Both `adamwxclipdrop` (single device, the default) and `adamdpwxclipdrop` (4 replicas) are
    rendered, so unlike `vit-s-imagenet-verified` a plain invocation runs.
    Renders and ties its shapes; NOTHING has been trained on it. -/
lean_exe «convnext-s-imagenet-verified» where
  root := `apps.imagenette.MainConvNeXtSImagenet
  moreLinkArgs := lowererLink

/-- **ConvNeXt-Base on ImageNet-1k** — ConvNeXt-S's depth `[3,3,27,3]` at `[128,256,512,1024]`,
    342 param tensors / 88,589,416 scalars, the published 88.59M.

    B moves the stem (96→128), the head (768→1024) and every stage (~27 dimension literals), which
    is why the DIMS are a renderer parameter. Depths and dims are one `CnxDims` record so
    `(S depths, T dims)` cannot be spelled. B shares S's depth table, so anything keyed on block
    count cannot separate them — a banner keyed that way would call every B artifact a ConvNeXt-S.
    Stochastic depth is 0.5 (paper value for B; S is 0.4, T is 0.1).
    Renders and ties its shapes; NOTHING has been trained on it. -/
lean_exe «convnext-b-imagenet-verified» where
  root := `apps.imagenette.MainConvNeXtBImagenet
  moreLinkArgs := lowererLink

/-- **ViT-Tiny on full 1000-class ImageNet** — the ViT peer of the R34 scale tier. Same certified
    renderer at `nClasses := 1000, bs := 128`; at four replicas that is global batch 512, the
    reference's (`jax/MainVitImagenet.lean`). Fed by the generated tfds shim.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). Not
    `lake exe resnet34-imagenet default --shim` — that is R34's, and every net fed from it
    streams R34's augmentation.
    Set `SHIM_WORKERS=2` — one producer cannot feed a 4×128 ViT step.
    Does NOT move the verification tier, and is NOT the DeiT recipe. -/
lean_exe «vit-imagenet-verified» where
  root := `apps.imagenette.MainViTImagenet
  moreLinkArgs := lowererLink

/-- **ViT-Small on ImageNet-1k** — ViT-Tiny widened (D 384 = 6 heads × 64, MLP 1536, same depth
    12), 22,050,664 parameters. The first net added by widening rather than by a new chain: the
    proof side needed nothing, since `vitForwardKVHasVJP` is already `∀ heads d_head mlpDim k`
    and global (GELU/softmax/LayerNorm carry no kink).
    FOUR-REPLICA ONLY — `adamdp128x4wxclipdrop` is the sole rendered variant, so this needs
    `PJRT_REPLICAS=4` AND `LEAN_MLIR_REPLICAS=4`; there is no single-device peer. At 128 per
    device that is DeiT's global 512, so the recipe's LR is the rate this batch was set for.
    `scripts/supervise.sh vits-default-g512-4gpu` is the job. Renders, ties and STEPS; NOTHING
    has been trained on it. -/
lean_exe «vit-s-imagenet-verified» where
  root := `apps.imagenette.MainViTSImagenet
  moreLinkArgs := lowererLink

/-- **ViT-Base (DeiT-B) on ImageNet-1k** — 86,567,656 parameters. Per-device batch 32 (global
    128, NOT the DeiT 512) because ViT-B OOMs at 4×128 on 16 GB cards. Renders; unmeasured. -/
lean_exe «vit-b-imagenet-verified» where
  root := `apps.imagenette.MainViTBImagenet
  moreLinkArgs := lowererLink

-- ─── Demos — `lake exe <name>`, in chapter 10's order: detection, inspection, people, agriculture, segmentation; RL, language, diffusion, physics, signal, beyond vision (demos/README.md) ───

lean_exe «unet-brats-train» where
  root := `demos.MainUnetBratsTrain
  moreLinkArgs := lowererLink

lean_exe «unet-brats-r34» where
  root := `demos.MainUnetBratsR34
  moreLinkArgs := lowererLink

lean_exe «brats-predict» where
  root := `demos.MainBratsPredict
  moreLinkArgs := lowererLink

/-- Per-volume BraTS Dice (the literature's protocol) for a trained checkpoint, from
    `preprocess_brats.py --val-full`'s whole-volume export. -/
lean_exe «brats-eval» where
  root := `demos.MainBratsEval
  moreLinkArgs := lowererLink

lean_exe «bigram-shakespeare» where
  root := `demos.MainBigramShakespeare
  moreLinkArgs := lowererLink

lean_exe «yolov1-visdrone-fpn» where
  root := `demos.MainYolov1VisdroneFpn
  moreLinkArgs := lowererLink

-- The VisDrone detector, unchanged, on NEU-DET steel defects — the opposite
-- regime (2 large defects per crop vs 70 tiny cars).
lean_exe «yolov1-neudet-fpn» where
  root := `demos.MainYolov1NeuDetFpn
  moreLinkArgs := lowererLink

-- The single-grid 14×14 arm (0.0000 on VisDrone) on the same NEU-DET records.
lean_exe «yolov1-neudet448» where
  root := `demos.MainYolov1NeuDet448
  moreLinkArgs := lowererLink

lean_exe «tinygpt-shakespeare» where
  root := `demos.MainTinyGptShakespeare
  moreLinkArgs := lowererLink

lean_exe «mnist-ddpm-train» where
  root := `demos.MainMnistDdpmTrain
  moreLinkArgs := lowererLink

lean_exe «mnist-ddpm-sample» where
  root := `demos.MainMnistDdpmSample
  moreLinkArgs := lowererLink

-- The 2-D diffusion demo, the second half of the
-- diffusion section: the same 18k-param MLP on the rank-2 DDPM MSE block,
-- trained as a DDPM or (with `flow`) as a flow-matching Boltzmann generator
-- on the Müller-Brown density.
lean_exe «diffusion-2d» where
  root := `demos.MainDiffusion2d
  moreLinkArgs := lowererLink

-- The two RL environments of the DQN sidequests: pure Lean, no FFI, no GPU. Each
-- runs its own gates — the exact DP instrument and the four arms for blackjack, the
-- scripted baselines and a rendered frame strip for Pong.
lean_exe «blackjack-env» where
  root := `demos.MainBlackjackEnv

lean_exe «pong-env» where
  root := `demos.MainPongEnv

-- Blackjack: the DQN on the XLA path, zero new codegen (the
-- rank-2 DDPM MSE block is the loss). Scores its greedy policy exactly.
lean_exe «blackjack-dqn» where
  root := `demos.MainBlackjackDqn
  moreLinkArgs := lowererLink

-- Pong: the same loop on the Lean Pong, the six-number state
-- (the ceiling row) and 84 × 84 frames through Chapter 3's CNN. Zero new codegen.
lean_exe «pong-dqn» where
  root := `demos.MainPongDqn
  moreLinkArgs := lowererLink

-- Tic-tac-toe (n×n, k in a row): the game, the solved-game instrument — a C minimax
-- table over every reachable position, 43 MB at 4×4 — and the scripted players, with
-- the solver's own gates. No stack, no GPU.
lean_exe «ttt-env» where
  root := `demos.MainTttEnv

-- AlphaZero on the Lean tic-tac-toe: self-play + PUCT on AlphaGo's plain conv + ReLU
-- stack from the bestiary, the two-headed loss through the rank-2 DDPM MSE block (zero new codegen),
-- every arm scored against the solved game. `n=` is the board.
lean_exe «alphazero-ttt» where
  root := `demos.MainAlphaZeroTtt
  moreLinkArgs := lowererLink

-- The gravitational-wave detection demo: a chapter
-- CNN on H1+L1 spectrograms of real O3a strain with injected chirps, scored by
-- `scripts/demos/gw_metrics.py` against the matched filter's closed form. Zero new codegen.
lean_exe «gw-detect» where
  root := `demos.MainGwDetect
  moreLinkArgs := lowererLink

-- The CASP16 distogram demo: a `pairTile` stem on frozen ESM-2 features, the chapter residual
-- body at stride 1, per-pixel weighted CE over 66 distance bins; folded and scored against the
-- CASP16 field by scripts/demos/casp16_score.py. See planning/casp16_distogram_demo.md.
lean_exe «distogram-casp» where
  root := `demos.MainDistogramCasp
  moreLinkArgs := lowererLink

-- Neural quantum states: the ground
-- state of the transverse-field Ising chain as an MLP / ViT / GPT wavefunction
-- times the mean-field reference, energy gradient through the rank-2 DDPM MSE
-- block, exact at N ≤ 14 by enumeration and bracketed by Jordan-Wigner above.
lean_exe «nqs-ising» where
  root := `demos.MainNqsIsing
  moreLinkArgs := lowererLink

-- The people-watching demo: chapter 4's CNN on
-- Arabic sign-language letters under a random and a capture-order-blocked split of the
-- same burst frames, scored by `scripts/demos/arasl_score.py` with a leak audit. Zero new codegen.
lean_exe «arasl-signs» where
  root := `demos.MainAraslSigns
  moreLinkArgs := lowererLink

-- The agriculture demo: chapter 6's ResNet-34 from the
-- ImageNet prefix on PlantVillage's lab leaves, scored on PlantDoc's field leaves, with the
-- leaf-mask CAM statistic as the diagnosis and three fixes. Zero new codegen.
lean_exe «plant-leaf» where
  root := `demos.MainPlantLeaf
  moreLinkArgs := lowererLink

-- The remote-sensing demo: chapter 4's CNN on EuroSAT's thirteen
-- Sentinel-2 bands under five stems (rgb / rgbn / ms10 / all / ir), the same weights scored on
-- Sentinel-2 chips over the Amazon and the Cerrado labelled from MapBiomas. Zero new codegen.
lean_exe «rs-bands» where
  root := `demos.MainRsBands
  moreLinkArgs := lowererLink

-- ─── Gates — the two checkers CI and every doc commit run ───

/-- `lake exe blueprint-checkdecls blueprint/lean_decls` — the split-aware
    blueprint declaration check (checkdecls minus the `CertsHeavy` lib, whose
    oleans the blueprint workflow deliberately does not build). -/
lean_exe «blueprint-checkdecls» where
  root := `tests.BlueprintCheckDecls
  supportInterpreter := true

/-- `docstring-checkrefs` — the docstring peer of `blueprint-checkdecls`. That gate resolves
    every `\lean{}` the blueprint cites; this one resolves every `` `Ident` `` a DOCSTRING
    cites, against the same environment.
    Resolution is `Environment.find?`, not a regex — see the file header for why a regex
    falls short (it has a false-positive floor). It also checks
    that every `` `dir/File.lean` `` a `LeanMlir/` docstring cites is a markdown link into the
    repo with a live target, because doc-gen4 renders the bare form as a module link that
    404s for anything outside the LeanMlir doc build. -/
lean_exe «docstring-checkrefs» where
  root := `tests.DocstringCheckRefs
  supportInterpreter := true

-- ═══════════════════════════════════════════════════════════════════════
-- THE LAB — everything else: the evidence behind the tour's numbers and the book's
-- ablations. Nothing here is a headline runner. historical/RESULTS.md and the book cite
-- these names, so nothing is renamed; groups are by home directory.
-- ═══════════════════════════════════════════════════════════════════════

-- ─── apps/ablation/ — Chapter 4 and 5 ablation binaries: constant-lr optimizer arms, precision, the R34 recipe ───

lean_exe «ablation» where
  root := `apps.ablation.MainAblation
  moreLinkArgs := lowererLink

-- ── §4.3 "Lever 3: precision": the wide-head (d1=512) 3×3 sweep ──
-- One net (the one Levers 1-2 measure), three optimizers, three precisions, three binaries.
-- f32 and bf16 come from ONE renderer (`c8wbPacked`) differing only in the emit; fp8 is
-- host-side E4M3 and rides the f32 graph, which is why it needs no artifact of its own.
-- FORWARD-only bf16, and NO speedup by design (bf16 does not pay at cifar8's shapes) — the arms
-- exist to show the optimizer ORDERING is invariant under precision, which is the CIFAR
-- chapter's claim, not to go faster. Each binary runs its three optimizers in sequence, so
-- the nine cells of the lever are three invocations: `runs/2026-09-01-cifar8w-6arm-constlr/`.
-- The per-arm narrow-head renders stay committed and gated in .github/workflows/proofs.yml,
-- the §4.1/§5.2 provenance.
lean_exe «cifar8wb-ablation» where
  root := `apps.ablation.MainCifar8WideBatchedAblation
  moreLinkArgs := lowererLink

lean_exe «cifar8wb-bf16-ablation» where
  root := `apps.ablation.MainCifar8WideBf16Ablation
  moreLinkArgs := lowererLink

lean_exe «cifar8w-fp8-ablation» where
  root := `apps.ablation.MainCifar8WideFp8Ablation
  moreLinkArgs := lowererLink

/-- Chapter 5 §5.6's recipe ablation on ResNet-34 / Imagenette, ONE ARM PER INVOCATION
    (`resnet34-ablation data <arm>`) so the eight arms can run across four cards in under
    three hours instead of ten in series. Arms: full nowarm nocos noaug nowd nols noadam bare. -/
lean_exe «resnet34-ablation» where
  root := `apps.ablation.MainResnet34Ablation
  moreLinkArgs := lowererLink

/-- Chapter 4 Lever 3 on the NORMALIZED net: the BN net on the batched op family, f32 and bf16,
    three optimizers each. -/
lean_exe «cifar8wb-bn-ablation» where
  root := `apps.ablation.MainCifar8WideBnBf16Ablation
  moreLinkArgs := lowererLink

-- ─── apps/mnist/ — MNIST robustness (PGD / spectral / smoothing), grids and low precision ───

-- Phase-3 PGD adversarial attack on the verified linear net:
-- the attack's input gradient is the proven dx=(softmax-onehot)·Wᵀ VJP, run on the GPU.
lean_exe «mnist-linear-pgd» where
  root := `apps.mnist.MainMnistLinearPgd
  moreLinkArgs := lowererLink

-- Phase-3 PGD attack on the verified MLP: input gradient =
-- the proven mlpInputGrad VJP; certificate = the loose product of layer spectral norms.
lean_exe «mnist-mlp-pgd» where
  root := `apps.mnist.MainMnistMlpPgd
  moreLinkArgs := lowererLink

-- Phase-3 PGD attack on the verified CNN (the conv rung):
-- input gradient = the proven conv/maxpool input-VJP; certificate = the conv-aware product.
lean_exe «mnist-cnn-pgd» where
  root := `apps.mnist.MainMnistCnnPgd
  moreLinkArgs := lowererLink

-- Spectral-norm-constrained MLP training (the gap-shrinking lever): projected SGD onto ‖Wᵢ‖₂ ≤ c
-- shrinks the global L = ∏‖Wᵢ‖₂, turning the vacuous product certificate non-vacuous — the
-- empirical face of lipschitz_margin_certified_radius.
lean_exe «mnist-mlp-spectral» where
  root := `apps.mnist.MainMnistMlpSpectral
  moreLinkArgs := lowererLink

-- Spectral-norm-constrained CNN training: the conv sibling —
-- caps the dense ‖Wᵢ‖₂ and the conv tap-sum bound; a 5-layer product + loose conv-norm make
-- certifying the conv net harder than the MLP (tighter c, more clean cost).
lean_exe «mnist-cnn-spectral» where
  root := `apps.mnist.MainMnistCnnSpectral
  moreLinkArgs := lowererLink

-- Randomized-smoothing certificate (Cohen 2019): the
-- DEPTH-INDEPENDENT cert. Forward-only Monte-Carlo over the proof-rendered fwd (no kernel, no
-- input-VJP) — sample noisy copies, Clopper-Pearson lower-bound p_A, radius = σ·Φ⁻¹(p_A). Base
-- net trained with matched Gaussian augmentation. Non-vacuous where the spectral product is hopeless.
lean_exe «mnist-mlp-smooth» where
  root := `apps.mnist.MainMnistMlpSmooth
  moreLinkArgs := lowererLink

lean_exe «mnist-cnn-smooth» where
  root := `apps.mnist.MainMnistCnnSmooth
  moreLinkArgs := lowererLink

-- Chapter 2 (low precision): fp8 (E4M3) training on the SAME verified StableHLO —
-- fp32 master, per-column W / per-tensor x projected to the E4M3 grid, fp32 accumulate.
-- See MainMnistLinearE4M3Verified.lean + LeanMlir/E4M3Quant.lean.
lean_exe «mnist-linear-e4m3-verified» where
  root := `apps.mnist.MainMnistLinearE4M3Verified
  moreLinkArgs := lowererLink

-- Width-parametric MNIST MLP: `mnist-mlp-grid <d₁> <d₂> [epochs]` renders + trains
-- the 784→d₁→d₂→10 MLP on the faithful verified StableHLO (the size-sweep demo).
lean_exe «mnist-mlp-grid» where
  root := `apps.mnist.MainMnistMlpGrid
  moreLinkArgs := lowererLink

-- FC-width-parametric MNIST CNN: `mnist-cnn-grid <fc-width> [epochs]` holds the conv
-- stack at 32 channels and sweeps the dense head (…→d→d→10) on the faithful StableHLO.
lean_exe «mnist-cnn-grid» where
  root := `apps.mnist.MainMnistCnnGrid
  moreLinkArgs := lowererLink

-- Chapter 3 (low precision): fp8 (E4M3) MLP training on the SAME verified StableHLO.
-- fp32 master, per-column weight quant + per-tensor input, fp32 accumulate.
-- fp8 weights+input, fp32 intermediates. See MainMnistMlpE4M3Verified.lean.
lean_exe «mnist-mlp-e4m3-verified» where
  root := `apps.mnist.MainMnistMlpE4M3Verified
  moreLinkArgs := lowererLink

-- ─── apps/cifar/ — the CIFAR trainers behind Chapter 4: robustness, the head-width grid, the BN pair ───

-- Phase-3 PGD attack on the verified CIFAR-10 CNN (the deeper conv rung): input gradient = the
-- proven 4-conv/2-pool input-VJP (genCifarPgdStep); cert = the 7-layer conv-aware product. Reuses
-- the generic attackPgdConvNet driver.
lean_exe «cifar-pgd» where
  root := `apps.cifar.MainCifarPgd
  moreLinkArgs := lowererLink

-- Spectral-norm-constrained CIFAR-10 CNN training: the 7-layer
-- product compounds the loose conv bound harder still — tightest caps, smallest certified radii.
lean_exe «cifar-spectral» where
  root := `apps.cifar.MainCifarSpectral
  moreLinkArgs := lowererLink

-- The deep-net payoff: smoothing certifies a non-vacuous L2 radius on the 7-layer CIFAR CNN where
-- the conv-aware spectral product is too loose to certify anything. Same forward-only procedure, any depth.
lean_exe «cifar-smooth» where
  root := `apps.cifar.MainCifarSmooth
  moreLinkArgs := lowererLink

-- FC-head-parametric cifar8-BN (AdamW): `cifar8-bn-grid <fc-width> [epochs]` holds the
-- 8-conv [16,16,32,32] backbone and sweeps the dense head (128→d→d→10) on the verified
-- renders (tests/TestCifar8AdamTrain.lean), trained via trainAdamSched "adam".
lean_exe «cifar8-bn-grid» where
  root := `apps.cifar.MainCifar8BnGrid
  moreLinkArgs := lowererLink

-- Chapter 5: trains the CIFAR-10 CNN (no BN) on the VERIFIED-rendered StableHLO
-- (verified_mlir/cifar_train_step.mlir = Proofs.StableHLO.cifarTrainStepText).
lean_exe «cifar-verified» where
  root := `apps.cifar.MainCifarVerified
  moreLinkArgs := lowererLink

-- The 8-conv CIFAR-10 CNN WITH per-channel BN (narrow 64-wide head), on the VERIFIED-rendered
-- StableHLO — SGD via `.train` and AdamW via `trainAdamSched`. These two are the whole narrow-head
-- family (the wide ablation pair `cifar8w-{,bn-}ablation` runs all three optimizers per binary and
-- is what Chapter 4 reports). The book names `cifar8-bn-verified` as the 64-wide-head net the
-- head-width proof is parametric over, and `scripts/gates/residency_gate_all.sh` gates both. Their
-- renders are verified_mlir/cifar8_bn{,_adam}_train_step.mlir.
lean_exe «cifar8-bn-verified» where
  root := `apps.cifar.MainCifar8BnVerified
  moreLinkArgs := lowererLink

lean_exe «cifar8-bn-verified-adam» where
  root := `apps.cifar.MainCifar8BnVerifiedAdam
  moreLinkArgs := lowererLink

-- ─── apps/imagenette/ extras and apps/tools/ — non-tier Imagenette drivers and the checkpoint scorer ───

/-- **Score a finished checkpoint, standalone** — the verified peer of the JAX side's six
    `eval_*_full50k.py`.

        LEAN_MLIR_VARIANT=<v> .lake/build/bin/score-checkpoint <net> [dataDir]

    Without it a verified accuracy can only be produced in-training, for whichever weights are
    live at that moment — so a finished run cannot be re-scored, and an EMA run reports one of
    {live, shadow} and discards the other. `LEAN_MLIR_REGION` picks.

    Its gate is an EQUALITY, not a smoke test: the same checkpoint at the same region must score
    exactly what the training run printed for the epoch that wrote it.
    ConvNeXt and ViT only. The BN nets refuse, loudly, because the checkpoint is exactly
    `[θ|m|v(|ema)]` and the running mean/var are not in it. -/
lean_exe «score-checkpoint» where
  root := `apps.tools.MainScoreCheckpoint
  moreLinkArgs := lowererLink

-- ch8: EfficientNet-B0 (faithful [t,c,n,s,k] config — 16 MBConv layers,
-- inverted-residual + squeeze-excite + swish + BATCH norm, 3×3/5×5 depthwise) trained
-- on VERIFIED-rendered StableHLO (tests/TestEfficientNet{Train,Fwd}.lean); 262 params.
lean_exe «efficientnet-verified» where
  root := `apps.imagenette.MainEfficientNetVerified
  moreLinkArgs := lowererLink

-- Chapter 9: ConvNeXt-T (Liu et al. 2022 — patchify stem + [3,3,9,3] depthwise-7×7
-- blocks with LN + GELU + layerScale + 3 between-stage downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestConvNeXt{Train,Fwd}.lean); 180 params.
lean_exe «convnext-verified» where
  root := `apps.imagenette.MainConvNeXtVerified
  moreLinkArgs := lowererLink

-- Randomized-smoothing certificate on the verified ConvNeXt-T (Imagenette 224²): the deep / real-
-- resolution rung of the depth-INDEPENDENT cert (Cohen 2019). LayerNorm ⇒ per-sample fwd, so the
-- generic smoothCertify driver applies unchanged. σ via SMOOTH_SIGMA_MILLI (split across 2 GPUs).
lean_exe «convnext-smooth» where
  root := `apps.imagenette.MainConvNeXtSmooth
  moreLinkArgs := lowererLink

lean_exe «vit-verified» where
  root := `apps.imagenette.MainViTVerified
  moreLinkArgs := lowererLink

-- ─── demos/archive/ — an earlier demo generation: the single-grid VisDrone detector ───

-- VisDrone single-scale detector at 448 input / 14×14 grid (train + infer).
-- The resolution rung above the 224/7×7 baseline.
lean_exe «yolov1-visdrone448» where
  root := `demos.archive.MainYolov1VisDrone448
  moreLinkArgs := lowererLink

-- ─── demos/probes/ — the loss / neck / emit probes behind the detector ───

lean_exe «grad-fd-probe» where
  root := `demos.probes.MainGradFdProbe
  moreLinkArgs := lowererLink

lean_exe «gradcam» where
  root := `demos.probes.MainGradCAM
  moreLinkArgs := lowererLink

lean_exe «flash-probe» where
  root := `demos.probes.MainFlashProbe
  moreLinkArgs := lowererLink

lean_exe «seg-loss-probe» where
  root := `demos.probes.MainSegLossProbe
  moreLinkArgs := lowererLink

-- DIoU box-loss forward probe (detection infra); FD-checked by
-- scripts/probes/diou_probe_check.py against scripts/probes/diou_grad_check.py.
lean_exe «diou-loss-probe» where
  root := `demos.probes.MainDiouLossProbe
  moreLinkArgs := lowererLink

-- Anchor-YOLO-loss probe (A anchors); FD-checked by
-- scripts/probes/anchor_loss_probe_check.py.
lean_exe «anchor-loss-probe» where
  root := `demos.probes.MainAnchorLossProbe
  moreLinkArgs := lowererLink

-- FPN-neck (top-down multi-scale merge) probe; FD-checked by
-- scripts/probes/fpn_neck_probe_check.py against scripts/probes/fpn_neck_check.py's oracle.
lean_exe «fpn-neck-probe» where
  root := `demos.probes.MainFpnNeckProbe
  moreLinkArgs := lowererLink

-- FPN multi-scale-loss probe; FD-checked by
-- scripts/probes/fpn_loss_probe_check.py against a numpy Σ-of-per-scale-anchor-loss ref.
lean_exe «fpn-loss-probe» where
  root := `demos.probes.MainFpnLossProbe
  moreLinkArgs := lowererLink

-- Whole-FPN-detector probe: neck+heads+concat+loss+DAG backward,
-- γ=0 so every grad is FD-checkable; validated by scripts/probes/fpn_detect_probe_check.py.
lean_exe «fpn-detect-probe» where
  root := `demos.probes.MainFpnDetectProbe
  moreLinkArgs := lowererLink

-- Emit-only: dump the r34FpnDet train-step MLIR for eyeball / iree-compile
-- --compile-to=input parse check.
lean_exe «fpn-train-emit» where
  root := `demos.probes.MainFpnTrainEmit
  moreLinkArgs := lowererLink

-- Scores the unconditional MNIST DDPM with Chapter 3's VERIFIED CNN — the
-- 2-D demo's metric suite (coverage, per-class mass, energy distance) moved onto
-- images, using a classifier whose math VJP is proven. See the driver's header.
lean_exe «mnist-ddpm-score» where
  root := `demos.probes.MainMnistDdpmScore
  moreLinkArgs := lowererLink

-- ─── tests/ — ties, checks, smokes and benches: the gates behind the verified renders ───

/-- `uib` layout tie: `VLayer.toSpecs` vs the baseline
    `Layer.nParams`, over all four UIB families. Pins the LAYOUT; the ORDER needs a forward tie. -/
lean_exe «uib-layout-tie» where
  root := `tests.TestUibLayoutTie

/-- MNv4 forward-chain structural smoke — op counts against the Conv-S block table. -/
lean_exe «mnv4-fwd-smoke» where
  root := `tests.TestMnv4FwdSmoke

/-- MNv4 AdamW train-step smoke: arity, entry point, the
    eval forward's stat binding, and — the one no other net has — that the train step's forward
    region is `@mnv4_fwd`'s body VERBATIM. The two-worlds split cannot hide behind this.
    Also emits the batch-2 train step `scripts/parity/grad_tie.py --net mnv4` runs. -/
lean_exe «mnv4-train-smoke» where
  root := `tests.TestMnv4TrainSmoke

/-- Emits the **batch-2** ResNet-34 AdamW train step that `scripts/parity/grad_tie.py --net r34` runs, and
    pins the two-worlds split it lives with: `resnet34_fwd` is per-example BN while the Adam
    train step is batch BN, so unlike MNv4 there is no forward-prefix property to assert. Sole
    writer of `.lake/build/resnet34_adam_train_step_b2.mlir`. -/
lean_exe «r34-train-b2» where
  root := `tests.TestR34TrainB2

/-- Emits the **optimizer stage alone** — one step as a function of `(θ, g, m, v, G)` — for each of
    seven variants, which is what `scripts/parity/opt_step_tie.py` diffs against the reference optimizer.
    `vjp_oracle` ties the two implementations at the GRADIENT; this ties them at the UPDATE.

    The body is `optAllParams`, the same call `resnet50TrainStepText` makes — so this gates
    the shipped emission and not a copy of it. Sole writer of `.lake/build/opt_step_*.mlir`. -/
lean_exe «opt-step-fixtures» where
  root := `tests.TestOptStepFixtures

/-- Forward tie: feeds two renders of `@<slug>_fwd` (or, with
    `--eval`, `@<slug>_fwd_eval`) the same θ and x and compares logits. The two emitters differ
    textually by construction, so a numeric tie is the only meaningful check. XLA-linked — it
    compiles the module in-process, so this is seconds rather than the multi-minute 224²
    `iree-compile`.

        .lake/build/bin/fwd-tie <slug> [--eval] [<pathA> [<pathB>]]

    `fwd-tie resnet34` is the ResNet-34 check. It DELETES its `.vmfb` before every compile — without
    that, a second run with a different candidate silently reuses the first candidate's binary and
    reports a perfect match, which is exactly what running a negative control looks like. -/
lean_exe «fwd-tie» where
  root := `tests.TestFwdTie
  moreLinkArgs := lowererLink

/-- The batched-index forward: the ConvNeXt forward rendered at `N := B` must emit the
    committed `verified_mlir/convnext_fwd.mlir` BYTE FOR BYTE. No GPU — it is a string compare, so
    it belongs in every pre-commit sweep rather than behind a device. The committed bytes ARE
    the batched chain's, so the gate renders the PER-EXAMPLE chain and
    compares it — the same statement from the other side, load-bearing as long as both chains exist
    (the per-example one still writes the SGD-inline `convnext_train_step.mlir`).

    Pair it with `lake env lean tests/TestBatchedEmitTie.lean`: that file pins each of the 31
    batched forms against its per-example peer individually, so it localises a failure this
    whole-net diff can only report. -/
lean_exe «convnext-fwd-b-tie» where
  root := `tests.TestConvNeXtFwdBTie

/-- **EMA and stochastic depth in one ViT render: the artifact's arity is the driver's arity.**

    `vitin_emadp128x4wxclipdropbf16` is the first render carrying both axes at once, and it exists
    because the ImageNet ViT pair needs both (blueprint §9.6). `tests/TestVariantPredicates.lean`
    pins the axis predicates against the variant NAME; this pins the committed artifact against
    what those predicates make the driver pack — regions, scalar tail, drop-mask count, and the
    arity identity that closes only if every region is a full `nP` wide.
    `vit-ema-drop-render convnextin` runs the same checks on ConvNeXt-T's EMA peer of its shipping
    recipe, `convnextin_emadpwxclipdropbf16`, and `vitsin` / `vitbin` / `convnextsin` / `convnextbin`
    on ViT-S's, ViT-B's, ConvNeXt-S's and ConvNeXt-B's.

    The failure it exists for is silent: a wrongly-packed region **trains and reports a loss**. No
    crash, no NaN — just a run optimising a misaligned view of its own parameters. No GPU; a parse
    and three counts. -/
lean_exe «vit-ema-drop-render» where
  root := `tests.TestVitEmaDropRender

/-- **ViT's batched-index chain, byte-tied against the independent per-example chain**. The
    committed artifacts are the batched chain's, so the forward and train-step checks render the
    PER-EXAMPLE chain and compare it with them; the backward check compares the two traversals.

    The peer of `convnext-fwd-b-tie`, and the bar is STRICTER: ConvNeXt's batched chain differs from
    its per-example one on 78 conv-VJP lines (two emitters for one VJP that were never tied to each
    other), so its train-step tie carries an allowance. ViT uses one emitter per op, so this is
    exact byte-identity with no allowance — anything else is a defect, not a known divergence.

        lake build vit-fwd-b-tie && .lake/build/bin/vit-fwd-b-tie -/
lean_exe «vit-fwd-b-tie» where
  root := `tests.TestViTFwdBTie

/-- Can the existing ops spell a **channel** LayerNorm?

    `.bnF` (`bnForward` over the whole `C·H·W` map, scalar γ/β) is not the reference's
    `channel_layer_norm` (`H·W` statistics, each over `C`, per-channel affine). Channel-LN needs
    **no new op**: it is ViT's row-LN under a transpose. This checks that on device, and it
    carries its own control — the `.bnF` chain must NOT match, or gate 2 is measuring something
    both paths satisfy.

        lake build channel-ln && HIP_VISIBLE_DEVICES=0 .lake/build/bin/channel-ln -/
lean_exe «channel-ln» where
  root := `tests.TestChannelLN
  moreLinkArgs := lowererLink

/-- Does the emitter spell a **1×1 strided** conv, and does it compute the right one?

    The paper's ResNet-34 option-B shortcut is a 1×1 stride-2 projection, which `downFwdB` emits.
    This renders the four strided-conv ops at `k = 1` (with `k = 3` alongside as the control),
    `iree-compile`s both, then
    drives each op on device against the **closed form** `den` implies — `flatConvStride2` is
    `decimateFlat ∘ flatConv` and `decimateIdx` reads the even positions, so at `k = 1` all four
    ops are writable in one line each. The `dx` odd-position zeros are the load-bearing check: they
    distinguish `decimate ∘ conv` from a conv that read the wrong pixel, and no norm would show it.

        lake build strided-1x1 && HIP_VISIBLE_DEVICES=0 .lake/build/bin/strided-1x1 -/
lean_exe «strided-1x1» where
  root := `tests.TestStrided1x1
  moreLinkArgs := lowererLink

/-- Are the R34 conv biases inert? Dropping them is layout-only because
    every conv is BN-followed and BN removes the bias — this MEASURES it. One step of the committed
    AdamW render from `m = v = 0`, where `m' = (1−β₁)·g` recovers the gradient exactly, then
    reads the 36 conv-bias slots. The DENSE bias is the control: same shape, same zero init, no BN
    after it, so it must move — otherwise the reading is "the harness sees zeros".

        lake build conv-bias-zero && HIP_VISIBLE_DEVICES=0 .lake/build/bin/conv-bias-zero -/
lean_exe «conv-bias-zero» where
  root := `tests.TestConvBiasZero
  moreLinkArgs := lowererLink

/-- Is `resnet34_mom_train_step` really heavy-ball with COUPLED L2?

    A cross-render known answer. AdamW's stored `m' = 0.1·g` at `m = v = 0` recovers the gradient
    exactly, so on the same (θ, x, onehot) the momentum render must satisfy `v' = g + wd·θ` and
    `θ' = θ − lr·v'`. The controls are the point: the repo's `momParamF` is NESTEROV, which at
    `v = 0` differs by exactly 1.9×, and the harness requires that prediction to MISS — a gate that
    cannot separate the two optimizers would pass the wrong one.

        lake build r34-mom-tie && HIP_VISIBLE_DEVICES=0 .lake/build/bin/r34-mom-tie -/
lean_exe «r34-mom-tie» where
  root := `tests.TestMomTie
  moreLinkArgs := lowererLink

/-- **The RMSProp render, numerically certified.** The `r34-mom-tie`
    construction pointed at MobileNetV2's RMSProp tail: recover the gradient from the AdamW
    render's `m'` at `m = v = 0`, then require the RMSProp render to satisfy `s' = (1-rho)*gw^2`,
    `b' = gw/sqrt(s'+eps)` and `theta' = theta - lr*b'` where `gw = g + wd*theta`.

    Two controls it requires to FIRE: the TEXTBOOK epsilon placement `gw/(sqrt(s')+eps)`, and the
    decay dropped. The first is the whole point — TensorFlow's RMSProp puts epsilon INSIDE the
    root and that is a different optimizer.

        lake build rms-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/rms-tie -/
lean_exe «rms-tie» where
  root := `tests.TestRmsTie
  moreLinkArgs := lowererLink

/-- **Stochastic depth — the two gates that cover the op's INTERIOR**.
    The other gates pin an ENDPOINT: `dropPath = 0` re-renders every
    artifact byte-identically, keep = 1 is bit-identical to AdamW, and `TestDropPathRamp` pins the
    keep ramp across the driver/renderer seam. Neither says what a scale strictly between those
    endpoints does, and neither can — every existing tie compares the render against a peer built
    from the SAME constants.

    **A — the known answer**: drive `dropPathB` through the same `pretty` emitter and compare
    against a host-computed `s[j]·x[j,i]`. Bit-exact is the bar, not tolerance: an f32 product is
    exact in the f64 the host multiplies in, so any difference is a different function.
    **B — the all-zero-mask control**: `s ⊙ (branch + x)` compiles, trains and descends, and no
    structural check distinguishes it from the correct `s ⊙ branch + x`. A zeroed site separates
    them — it must leave the block an IDENTITY, not annihilate the signal.

    Run gate B under `scripts/det_shim.sh`: it compares two different HLO programs and the
    committed compile options autotune on CUDA. The harness
    measures its own A-vs-A floor first and degrades B1 to a bound if the floor is not bit-exact.

        lake build droppath-tie
        scripts/det_shim.sh /tmp/detshim
        LD_LIBRARY_PATH=/tmp/detshim CUDA_VISIBLE_DEVICES=0 .lake/build/bin/droppath-tie
        .lake/build/bin/droppath-tie --op --break     # gate A: a 1% wrong scale is caught
        .lake/build/bin/droppath-tie --net --cand <misplaced.mlir>   # gate B goes red, rc=1 -/
lean_exe «droppath-tie» where
  root := `tests.TestDropPathTie
  moreLinkArgs := lowererLink

/-- **Classifier dropout's two gates** — the ones its endpoint checks
    structurally cannot make. Gate A: the mask multiplies PER ELEMENT, against a host-computed
    answer, with the per-EXAMPLE mask (i.e. stochastic depth on the classifier) as the control.
    Gate W: the classifier WEIGHT GRADIENT reads the dropped activation, not the pooled one —
    invisible to every ones-mask gate because there the two activations are the same buffer.

        lake build dropout-tie
        CUDA_VISIBLE_DEVICES=0 .lake/build/bin/dropout-tie
        .lake/build/bin/dropout-tie --op --break     # gate A is falsifiable
        .lake/build/bin/dropout-tie --net            # gate W only — NO GPU, milliseconds
        scripts/probes/fault_dropout_wgrad.py verified_mlir/efficientnet_adamdo_train_step.mlir /tmp/f.mlir
        .lake/build/bin/dropout-tie --net /tmp/f.mlir              # goes red, rc=1 -/
lean_exe «dropout-tie» where
  root := `tests.TestDropoutTie
  moreLinkArgs := lowererLink

/-- **`wdExcludeNormBias`, the timm/DeiT `no_weight_decay` render.**
    `vit_adamwx` is `vit_adam` with decoupled decay switched off for the 126 params timm excludes
    (every 1-D param plus the positional embedding). The change moves NO arity, NO type and NO
    region — only which constant feeds `%wd` at 126 of 200 sites — so every structural check
    passes on both renders and only a numeric known answer can tell them apart:

      adam:  θ' = θ − lr·( m̂/(√v̂+ε) + wd·θ )
      wx:    θ' = θ − lr·( m̂/(√v̂+ε) + wd·msk·θ )

    THE PARTITION IS THE CONTROL. It does not check that 74 params match and 126 differ — a
    count is satisfied by any 74. It recovers per parameter which bucket that param EMPIRICALLY
    falls in and requires the partition to equal `vitWdDecays`' name for name, which is what
    catches a mask that excluded the wrong 126 (silent in arity, types and the prefix audit).
    m'/v' bit-exact on all 400 moment regions is the other half: AdamW's decay is DECOUPLED, so
    reaching a moment would mean coupled L2 — a different optimizer.

        lake build wdx-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/wdx-tie -/
lean_exe «wdx-tie» where
  root := `tests.TestWdExcludeTie
  moreLinkArgs := lowererLink

/-- **Global-norm gradient clipping**, ONE harness for ViT and ConvNeXt (`clip-tie <net>`).
    The reference's `g * min(1, CLIP/(‖g‖+1e-6))` with ‖g‖ taken across EVERY parameter.

    At `m = v = 0` the moment slot recovers the factor exactly — `m' = (1−β₁)·g` — so
    `m'_clip/m'_adam` is ONE number at all ~5.5M coordinates. **That constancy is the gate**, and
    it is the only property a per-parameter clip gets wrong: a per-parameter clip scales, never
    amplifies, and is the identity below the threshold, so it satisfies every other check here.
    `scripts/probes/perturb_clip.py perparam` builds it and it must fire.

    Needs the below-threshold render, which is GENERATED rather than committed (an artifact
    baking a threshold no config sets is a silent hyperparameter):

        lake build clip-tie
        python3 scripts/probes/perturb_clip.py verified_mlir/vit_adamclip_train_step.mlir \
          .lake/build/clip_hi_vit.mlir hi
        CUDA_VISIBLE_DEVICES=0 .lake/build/bin/clip-tie vit -/
lean_exe «clip-tie» where
  root := `tests.TestGradClipTie
  moreLinkArgs := lowererLink

/-- **The drop mask is SHARDED, not replicated.** The mask is per-EXAMPLE and rides in the
    PARAMETER blob, where the DP shim's rule ("x and the labels shard, everything between
    replicates") would copy it to every replica.

    The duplicated-batch `*-dp-check` gates are structurally blind to this — same rows on both
    replicas means sharded and replicated agree bit-exact — and `shard-check` needs the gated slot
    linear in the gradient, false for the RMSProp variant this net wants. So: duplicate the DATA,
    make only the MASK asymmetric, and **swap the halves**. A sharded mask is swap-invariant TO THE
    BIT (f32 addition is commutative, so the 2-replica mean is order-free); a replicated one is not.
    Optimizer-agnostic — it compares two runs of the same graph, never a device answer to a host one.

        lake build drop-shard-check && scripts/det_shim.sh /tmp/detshim
        CUDA_VISIBLE_DEVICES=0,1 PJRT_REPLICAS=2 LD_LIBRARY_PATH=/tmp/detshim \
          .lake/build/bin/drop-shard-check
        PJRT_DP_NO_MASK_SHARD=1 … .lake/build/bin/drop-shard-check    # the control, rc=1 -/
lean_exe «drop-shard-check» where
  root := `tests.TestDropShardCheck
  moreLinkArgs := lowererLink

/-- The cifar8 optimizer-render tie for ALL THREE variants — `cifar8-opt-tie <adam|sgd|mom>`.
    Gates the RECOVERED GRADIENT, never θ': a train step returns θ' = θ − lr·g and θ' is dominated
    by θ, the same input on both sides, so at lr 1e-3 a wholly wrong gradient still looks like a
    match. Each variant's gradient is exactly recoverable from its own outputs —
    adam from m', sgd from θ', mom from v'. Also gates the m/v PASSTHROUGH slots bit-exactly, since
    a tail that silently dropped a moment would still yield a plausible θ'. Deletes its .vmfb before
    every compile, unlike `cifar8-adam-tie`. -/
lean_exe «cifar8-opt-tie» where
  root := `tests.TestCifar8OptTie
  moreLinkArgs := lowererLink

/-- One AdamW step through two renders of `@cifar8_adam_train_step`, same packed
    `[θ|m|v|lr|bc1|bc2]`, every returned float compared. Compares θ' among other things and does
    NOT delete its `.vmfb`; prefer `cifar8-opt-tie`, which recovers the gradient and deletes. -/
lean_exe «cifar8-adam-tie» where
  root := `tests.TestCifar8AdamTie
  moreLinkArgs := lowererLink

/-- One AdamW step through two renders of `@resnet34_adam_train_step` — the
    hand-written emitter vs the batched `pretty(provenGraph)` — same packed
    `[θ|m|v|lr,bc1,bc2|bn stats]`, every returned float compared. Numeric and not textual on
    purpose: the two are the same function but not the same graph. -/
lean_exe «resnet34-adam-tie» where
  root := `tests.TestResnet34AdamTie
  moreLinkArgs := lowererLink

/-- One SGD step through two renders of `@<slug>_train_step` — the `tests/`
    emitter vs `pretty(provenGraph)` — on one shared θ, comparing the recovered GRADIENT
    `(θ − θ')/lr` rather than θ' (θ' is dominated by the shared θ, so it hides a wrong gradient).
    The lr is per side because the two emitters do not always agree on it. Run BEFORE deleting a
    `tests/` emitter: afterwards the comparison no longer exists. -/
lean_exe «sgd-render-tie» where
  root := `tests.TestSgdRenderTie
  moreLinkArgs := lowererLink

/-- ViT AdamW gate: one AdamW step through two renders of `@vit_adam_train_step` — the
    hand-written emitter the driver writes at startup vs `pretty(provenGraph)` — same packed
    `[θ|m|v|lr,bc1,bc2]`, every returned float compared. ViT has no BN, so there is no forward-only
    region: the gate is the gradient AND `%loss` (the only direct read of the forward).

    **`lowererLink`.** `vit-verified-adam` is on `lowererLink` and defaults to XLA. A
    `miopenStatusUnknownError` in the patch-embed weight-grad convolution is a **ROCm** fault, not
    an XLA one: the same graph trains this net end to end on CUDA. The link arg does not select
    the backend — `ffi/lowerer.c` **dlopens** whichever shim `$LEAN_MLIR_LOWERER` names. -/
lean_exe «vit-adam-tie» where
  root := `tests.TestViTAdamTie
  moreLinkArgs := lowererLink

/-- EfficientNet-B0 AdamW gate: one AdamW step through two renders of
    `@efficientnet_adam_train_step` — the hand-written emitter in `tests/TestEfficientNetTrain.lean`
    vs `pretty(provenGraph)` — same packed `[θ|m|v|lr,bc1,bc2|bn stats]`, every returned float
    compared.

    **Stronger than `vit-adam-tie`, because EfficientNet has BatchNorm.** The 98 returned batch
    statistics (μ/σ² of all 49 BN inputs) depend on the forward alone, so the `bnstat` region pins
    the whole forward chain BIT-EXACTLY and separates a forward disagreement from a backward one in
    one run. `%loss` is still gated, but as a cross-check rather than the only forward evidence.

    **`lowererLink`.** `efficientnet-verified-adam` is itself on `lowererLink` and defaults to
    XLA. The link arg never selects the backend, since `ffi/lowerer.c` dlopens the shim
    `$LEAN_MLIR_LOWERER` names. -/
lean_exe «efficientnet-adam-tie» where
  root := `tests.TestEfficientNetAdamTie
  moreLinkArgs := lowererLink

/-- MobileNetV2 AdamW gate: one AdamW step through two renders
    of `@mobilenetv2_adam_train_step` — the hand-written emitter in `tests/TestMobilenetV2TrainPC.lean`
    against `Proofs/Codegen/MobileNetV2RenderB.lean`'s `pretty(provenGraph)` — comparing all
    returned floats per region. 52 BN layers give a `bnstat` region that pins the forward
    bit-exactly, and the gate covers SPREAD as well as magnitude. Deletes its `.vmfb`
    before every compile (the false-PASS trap).

    **`lowererLink`**, as `mobilenetv2-verified-adam` is. A link flag selects no backend —
    `lowerer.h` macro-redirects the whole `iree_ffi_*` surface onto dlopen'd pointers. -/
lean_exe «mobilenetv2-adam-tie» where
  root := `tests.TestMobilenetV2AdamTie
  moreLinkArgs := lowererLink

/-- ConvNeXt-T AdamW gate: one AdamW step through two renders of `@convnext_adam_train_step`
    — the hand-written emitter in `tests/TestConvNeXtTrain.lean` vs `pretty(provenGraph)` — same
    packed `[θ|m|v|lr,bc1,bc2]`, every returned float compared.

    **ViT-grade, not EfficientNet-grade**: ConvNeXt has no BatchNorm, so there is no `bnstat`
    forward-only region and `%loss` is the only direct read of the forward — which is why it is
    gated rather than reported.

    The trainer is on `lowererLink` and defaults to XLA. Link args do not select the backend:
    `ffi/lowerer.c` dlopens the shim `$LEAN_MLIR_LOWERER` names — an IREE-linked build of this
    gate prints `[pjrt_ffi] XLA backend: PJRT 0.112` and ties on XLA. -/
lean_exe «convnext-adam-tie» where
  root := `tests.TestConvNeXtAdamTie
  moreLinkArgs := lowererLink

/-- Gate on the bs256 re-render: feed it 8 identical copies of one bs32 batch. Batch-BN
    statistics and the mean-CE cotangent are then exactly the bs32 render's, so all 68M returned
    floats must AGREE — an exact known-answer check, not a tolerance argument. -/
lean_exe «resnet34-batch-check» where
  root := `tests.TestResnet34BatchCheck
  moreLinkArgs := lowererLink

/-- ViT DP gate: ViT has no BN, so giving both replicas the SAME batch makes `all_reduce(add)/2`
    the identity — the data-parallel step must reproduce the single-device one exactly. Needs two
    GPUs and the XLA backend. -/
lean_exe «vit-dp-check» where
  root := `tests.TestViTDpCheck
  moreLinkArgs := lowererLink

/-- Soft-target gate: the committed renders are AFFINE in their `%onehot` input, so a mixed
    target gives the mixed gradient and mixup/cutmix need **no new cotangent**. Measures
    `grad(λ·y_a + (1−λ)·y_b) == λ·grad(y_a) + (1−λ)·grad(y_b)` on the committed bytes, gating
    `m` (never θ', which is nonlinear in the gradient under AdamW), against a control that runs
    every time and a vacuity refusal. So no `softLabelCE` render is needed. -/
lean_exe «soft-target-tie» where
  root := `tests.TestSoftTargetTie
  moreLinkArgs := lowererLink

/-- EfficientNet DP gate. Giving both replicas the SAME batch makes `all_reduce(add)/2` the
    identity, so the data-parallel step must reproduce the single-device one exactly. BatchNorm does
    not spoil this: the render all-reduces its BN statistics, and both replicas see the same 32
    examples, so the averaged statistics are the single device's. A split batch is
    `efficientnet-syncbn-check`'s job.

    Stronger than `vit-dp-check`: EfficientNet returns 98 BN batch statistics, so it has a
    forward-only region that must come back BIT-EXACT. Needs two GPUs and the XLA backend. -/
lean_exe «efficientnet-dp-check» where
  root := `tests.TestEfficientNetDpCheck
  moreLinkArgs := lowererLink

/-- The mnv2 peer of `efficientnet-dp-check`, gated by the same EXACT duplicated-batch identity:
    both replicas get the same 32 examples, so their BN groups are identical by construction,
    `all_reduce(add)/2 = (g+g)/2 = g`, and the DP step must reproduce the single-device one.

    mnv2 returns 104 BN batch statistics (52 layers), so it has a forward-only `bnstat` region that
    must come back BIT-EXACT. Needs two GPUs and the XLA backend — collectives exist only on the
    PJRT path. -/
lean_exe «mobilenetv2-dp-check» where
  root := `tests.TestMobilenetV2DpCheck
  moreLinkArgs := lowererLink

/-- The MNv4 peer, and the gate that makes `mnv4in_adamdp64` quotable. Same EXACT duplicated-batch
    identity: every replica gets the same 64 examples, so the all-reduced BN statistics are the
    single device's, `all_reduce(add)/4 = (4·g)/4 = g`, and the DP step must reproduce the
    single-device one — `bnstat` bit-exact, gradient within 1e-2 (bf16 5e-2): the DP backward is
    the sync-BN graph and the single-device one the two-pass graph.

    An untied collective artifact looks exactly as trustworthy as a tied one. This gate plus
    `imagenet-syncbn-check mnv4` (the split batch) ties MNv4's collectives.

    FOUR GPUs, not two, and that is forced — MNv4 renders `adamdp64` at 4 replicas only, with no
    2-replica peer, so `PJRT_REPLICAS=2` hits the shim's replica-count guard rather than degrading.
    It is also the 1000-class 224² net: there is no Imagenette-scale MNv4 DP render to gate more
    cheaply.
    `DP_VARIANT`/`DP_VARIANT_DP` re-run it over the bf16 pair. XLA backend only. -/
lean_exe «mnv4-dp-check» where
  root := `tests.TestMnv4DpCheck
  moreLinkArgs := lowererLink

/-- **`cnx-init-check` — the known-answer gate for ConvNeXt's verified weight init.**

    The two arms of the ConvNeXt/ImageNet pair must share an init: the JAX reference sets
    `cnxInit := true` (ConvNeXt `_init_weights`, `trunc_normal(0.02)` on every conv AND the head),
    where `mkParam`'s default is He. Two arms with different inits cannot isolate the lowerer,
    which is the one thing this BatchNorm-free net is in the book for.

    **Init is the hardest thing in this repo to read off the source and be right.** It is
    host-side, it never reaches a committed artifact so no drift guard covers it, and the two
    init paths disagree on the rank-4 rule — `mkParam` is He **fan-OUT** (`2/(oc·kh·kw)`) while
    `SpecHelpers.heInitLayer` is fan-**IN** (`2/(ic·kh·kw)`). Reading the wrong file gives ratios
    off by an order of magnitude and, at some shapes, in the wrong DIRECTION. So this gate
    MEASURES the emitted parameters over the real 183-spec layout rather than arguing from source.

    Asserts σ = 0.02 on every weight spec, LayerNorm γ exactly 1.0, biases exactly 0.0 and
    LayerScale γ exactly 1e-6 — and carries its own CONTROL: with the flag off the weights must
    land somewhere else, or the gate is reading a flag that does nothing. No GPU, no lowerer. -/
lean_exe «cnx-init-check» where
  root := `tests.TestCnxInit

/-- The ConvNeXt peer, gated by the same EXACT duplicated-batch identity — and the one net that
    needs no BatchNorm caveat to justify it: LayerNorm reduces within an example, never across the
    batch, so nothing couples the replicas at all.

    The flip side is that ConvNeXt returns no batch statistics, so there is no `bnstat` region and
    `%loss` is the whole of the forward evidence — this harness gates it as well as the gradient,
    the same split `convnext-adam-tie` uses. It is also the first execution anywhere here of a
    RANK-0 `all_reduce` (the 44 scalar LayerNorm γ/β). Needs two GPUs and the XLA backend —
    collectives exist only on the PJRT path. -/
lean_exe «convnext-dp-check» where
  root := `tests.TestConvNeXtDpCheck
  moreLinkArgs := lowererLink

/-- The SHARDING gate — what `convnext-dp-check` cannot see. That gate hands both replicas the same
    rows, so a shard-offset bug leaves the halves identical and it still passes bit-exact; it
    establishes "the collective averages correctly", not "the replicas saw different data".

    This one gives the replicas DIFFERENT data and checks `DP([xA|xB]) == mean(single(xA),
    single(xB))`, with a built-in control that `DP vs single(xA)` — what a broken shard would
    return — is a far larger number. Gates the first Adam moment with `m = 0` on input, because
    `m' = (1-β₁)·g` is exactly linear in the gradient while θ' and v' are not. Needs two GPUs. -/
lean_exe «convnext-shard-check» where
  root := `tests.TestConvNeXtShardCheck
  moreLinkArgs := lowererLink

/-- `shard-check <convnext|vit|convnextin> [<dpPath>]` — the asymmetric-batch SHARDING gate for
    the LayerNorm nets' DP renders, generalised from `convnext-shard-check`. The `*-dp-check` gates
    hand both replicas the SAME rows, so they are structurally blind to a shard-offset bug; this one
    gives them different data and checks `DP([xA|xB]) == mean(single(xA), single(xB))`. Needs two
    GPUs and the XLA backend.

    It has no BatchNorm rows (`efficientnet`, `mobilenetv2`, `…in`, `mnv4in`): those DP renders
    are synchronised BatchNorm, so the identity above is exactly what their `*-syncbn-check`
    CONTROL requires to FAIL. The sync-BN gates' TEST column covers them; such a slug exits 2 and
    names its replacement.

    A 4-replica render needs `SHARD_REPLICAS=4` and four GPUs, e.g. `convnextin`. -/
lean_exe «shard-check» where
  root := `tests.TestShardCheck
  moreLinkArgs := lowererLink

/-- `resnet34-syncbn-check` — the SYNC-BN gate, the identity a per-replica batch-BN net cannot
    state: `DP_sync([xA|xB])` at 2×32 equals `single([xA|xB])` at 1×64 on EVERY output region
    (θ', m', v', the 72 BN statistics), because the sync render's replicas compute their shards of
    one global-batch function (`Foundation/DataParallel/Sync.lean`).
    CONTROL: the per-replica identity `DP == mean(single_32(xA), single_32(xB))` must FAIL
    by a margin, or the statistics are not synchronised. A numeric check on the seven sync ops'
    emitted MLIR. Needs two GPUs and the XLA backend. -/
lean_exe «resnet34-syncbn-check» where
  root := `tests.TestR34SyncBnCheck
  moreLinkArgs := lowererLink

/-- `mobilenetv2-syncbn-check` / `efficientnet-syncbn-check` — `resnet34-syncbn-check`'s gate on the
    other two BN nets with a committed pair, one shared
    runner (`LeanMlir/SyncBnCheck.lean`): the committed 2×32 sync-BN DP step against the 1×64
    two-pass step and the one-replica sync graphs, both rendered at run time. Two GPUs, XLA. -/
lean_exe «mobilenetv2-syncbn-check» where
  root := `tests.TestMnv2SyncBnCheck
  moreLinkArgs := lowererLink

lean_exe «efficientnet-syncbn-check» where
  root := `tests.TestEnetSyncBnCheck
  moreLinkArgs := lowererLink

/-- `imagenet-syncbn-check <resnet34|mobilenetv2|efficientnet|resnet50|resnet50bce|mnv4> [f32]` —
    the same runner on the artifacts the ImageNet pairs train from: the committed 4×64 bf16 sync-BN
    DP step against a 1×256 single-device step rendered at run time. ResNet-50 runs 4×32 against
    1×128 with its DP step rendered at run time too (the 1×256 reference nearly fills the
    raised arena). Four GPUs, XLA. -/
lean_exe «imagenet-syncbn-check» where
  root := `tests.TestImagenetSyncBnCheck
  moreLinkArgs := lowererLink

/-- `argmax-check` — the class-count gate on `F32.argmaxN`, the eval scorer every trainer here
    reads its top-1 through. Needs no GPU and no backend: pure host arithmetic on a synthetic
    logit block, so it can run anywhere and costs nothing.

    A scorer that scans a LITERAL 10 entries is correct on every 10-class net
    (Imagenette/CIFAR/MNIST) and silently wrong on the 1000-class ImageNet tier — it reports
    a number far below the net's real top-1, because it can only ever be right on labels 0..9. The file
    ships its own CONTROL: a 10-wide window re-run on the same data, required to MISS. -/
lean_exe «argmax-check» where
  root := `tests.TestArgmaxN

/-- `label-check` — the WIDTH gate on eval label decoding, and `argmax-check`'s twin: both guard
    10-class assumptions in code that also runs at 1000 classes. `ByteArray.get!` yields a
    `UInt8`, so `(evalLbl.get! (4*i)).toNat` reads byte 0 only — `label % 256` — capping
    ImageNet top-1 at roughly a quarter of the truth. No GPU. -/
lean_exe «label-check» where
  root := `tests.TestLabelDecode

/-- `r34-dp-shard` — R34's DP gate. `shard-check` has no R34 row ("no `adamdp` peer at this
    batch to pair with") and there is no `resnet34-dp-check`. This asks the narrower question the
    full identity would subsume — do replicas 1..3 affect the update at all — using only the
    committed DP artifact, no single-device bs64 peer. TEST + a CONTROL that must fire. Needs 4 GPUs
    and the XLA backend. -/
lean_exe «r34-dp-shard» where
  root := `tests.TestR34DpShard
  moreLinkArgs := lowererLink

/-- `r50-gradcheck` — **the gate on R50's backward.** Without it R50 renders, trains and
    descends behind only a LAYOUT gate (`TestR50Contract`) with nothing on the gradient, and R50
    is the one net with no incumbent hand-written artifact to tie against.

    The train step returns its own loss next to `[θ'|m'|v']`, so a single invoke from `m = v = 0`
    gives both `L(θ)` and `g = 10·m'`. The check is then the adjoint identity
    `⟨g, δ⟩ = (L(θ+δ) − L(θ−δ))/2` on the COMMITTED bytes, one direction per block, so it localises
    to the three bottleneck forms individually — including the stride-1 projection, which nothing
    else covers.

    Two tiers, because they cover each other's blind spots: the closed-form homogeneity identities
    (`⟨g_W, W⟩ = 0` on 53 BN-followed convs, `⟨g_γ,γ⟩+⟨g_β,β⟩ = 0` on 33 pre-conv BN affines — the
    stem's being the only gradient path through `maxPool3s2`) pin the STRUCTURE to 6e-5 with no
    finite differences at all, and the adjoint probes pin the SCALE, which the identities cannot
    see. Controls: 21 sites where the invariance is FALSE must violate it, and a doubled gradient
    must not fit.

    `vjp_oracle` cases would gate `MlirCodegen.emitBottleneckBlock`, a DIFFERENT lowering from the
    one `resnet50-imagenet-verified` runs.

    This gates `adam64`; the driver defaults to `adamdp64`, whose `%loss` is replica-local while
    its gradient is all-reduced, so tier 2 cannot run there. `python3 tests/r50_dp_render_tie.py`
    carries the verdict across by text — the DP renders are sync-BN, so from the ONE-REPLICA
    SYNC graph `ResNet50RenderB` writes to `.lake/build/r50sync/`, which this gate
    certifies with `R50_GC_PATH=.lake/build/r50sync`. Needs one GPU and the XLA backend.

        lake build r50-gradcheck && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-gradcheck -/
lean_exe «r50-gradcheck» where
  root := `tests.TestR50GradCheck
  moreLinkArgs := lowererLink

/-- `r50-accum-tie` — **gradient accumulation, numerically certified.** The `.adamwAccum` render
    carries a FOURTH parameter region `G` and two runtime scalars deciding, per micro-batch,
    whether the invoke accumulates or applies.

    Run it k times on the SAME batch: every micro-gradient is then the same `g`, so the cycle must
    reproduce ONE step of the committed `resnet50in_adam64_train_step.mlir` — a different artifact,
    rendered without accumulation, whose gradient `r50-gradcheck` certifies. Plus a
    bit-exactness claim that the accumulate micro-batches leave `[θ|m|v]` untouched.

    The check with teeth is `v'`: it is QUADRATIC in the gradient, so `%ob2` must carry
    `(1−β₂)/k²` where `%ob1` carries `(1−β₁)/k`. A single shared scale gives the mean of the
    per-micro-batch second moments instead of the second moment of the mean — a different optimizer
    that descends and looks entirely normal.

    Duplicated batch, so it is blind to the combination of DIFFERENT micro-batches, exactly as
    every `*-dp-check` is blind to shard offset. CONTROL: applying one micro-batch early must miss.
    Needs one GPU and the XLA backend.

        lake build r50-accum-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-accum-tie -/
lean_exe «r50-accum-tie» where
  root := `tests.TestR50AccumTie
  moreLinkArgs := lowererLink

/-- `r50-accum-shard-tie` — accumulation over **DIFFERENT** micro-batches, which `r50-accum-tie`
    is structurally blind to (it runs k micro-steps on the same batch, so every micro-gradient is
    the same). The same hole every `*-dp-check` has and that `shard-check` closes one level up.

    The naive complement — "k micro-batches of b == one step at batch k·b" — is FALSE by design:
    k micro-batches give k BatchNorm groups where one big batch gives one. That is Ghost-BN. But
    R50's renderer draws a graph that computes exactly that: the DATA-PARALLEL step with PER-REPLICA
    BatchNorm, each replica normalising over its own b rows. So `acc(x₁..x_k) == dp([x₁|..|x_k])`
    EXACTLY, and the two sides reach it through a serial accumulator with a folded 1/k versus an
    `all_reduce` and a divide — neither a re-derivation of the other. The
    committed DP renders are sync-BN, so that peer is rendered at run time (`noSync := true`).

    It compares θ', m' AND v', unlike `shard-check` (which averages two separately-optimised
    steps and so can only compare the linear `m`). CONTROL: the duplicated batch — exactly what
    `r50-accum-tie` runs — must MISS. Needs FOUR GPUs and the XLA backend.

        lake build r50-accum-shard-tie
        CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 .lake/build/bin/r50-accum-shard-tie -/
lean_exe «r50-accum-shard-tie» where
  root := `tests.TestR50AccumShardTie
  moreLinkArgs := lowererLink

/-- `r50-lamb-tie` — **LAMB, numerically certified.** **Two** new `SHlo` constructors;
    `gradSumSqAccF` is the global-norm clip's and `sgdParamF` heavy-ball's.

    The known-answer construction: recover `g = 10·m'` from the committed AdamW render at
    `m = v = 0`, then require the LAMB render to satisfy the closed form, per parameter TENSOR.

    THE CONTROLS ARE THE POINT — three wrong LAMBs that every one of them trains and descends:
    `trust ≡ 1` (plain Adam, i.e. forgetting the layer-wise part that IS the algorithm),
    `√(v̂ + ε)` (RMSProp-TF's ε placement) and the decay applied AFTER the trust ratio (AdamW's).
    Each must miss by ≥10× the tie.

    The comparison is on the STEP `θ' − θ`, not `θ'` — the step is ~1e-3 of θ, so a relative error
    on θ' divides by the wrong thing. Needs one GPU and the XLA backend.

        lake build r50-lamb-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-lamb-tie -/
lean_exe «r50-lamb-tie» where
  root := `tests.TestR50LambTie
  moreLinkArgs := lowererLink

/-- `r50-bce-tie` — **BCE-with-logits, numerically certified.** RSB-A2/A3 do not
    train with softmax CE: every class is an independent sigmoid, `reduction='mean'` over B×K.

    The trick that makes it EXACT: zero the classifier weight, so `z = Wd·gap + bd` collapses to
    `z = bd` — a vector the harness chose, known to the last bit, with no forward to reproduce. The
    loss and the whole cotangent are then closed forms in `bd` and the targets, and `g_bd` comes out
    of AdamW's `m' = 0.1·g`. The degeneracy is the instrument.

    The control worth having is the REDUCTION: mean over B alone rather than B×K is `K = 1000×`
    on the effective step, changes no shape, no op and no arity, and descends perfectly well at
    1/1000 of the intended learning rate. Needs one GPU and the XLA backend.

        lake build r50-bce-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-bce-tie -/
lean_exe «r50-bce-tie» where
  root := `tests.TestR50BceTie
  moreLinkArgs := lowererLink

/-- Step-time bench: 1 GPU (bs 32) vs 2 GPUs (global 64) on the same certified net,
    compiled in ONE process and interleaved A,B,A,B so drift hits both equally, min statistic,
    SYNTHETIC inputs so the data loader is out of it (the data-bound trap). Reports ms/image and
    the Amdahl-implied non-parallelisable share of a step, which is the measured argument for
    device-resident parameters. Needs two GPUs and the XLA backend. -/
lean_exe «efficientnet-dp-bench» where
  root := `tests.TestEfficientNetDpBench
  moreLinkArgs := lowererLink

/-- The collective's SEMANTICS, checked where they can be. cifar8 has no BN, so
    2×128 + all_reduce must equal 1×256 to fp rounding. Needs two GPUs and the XLA backend. -/
lean_exe «cifar8-dp-check» where
  root := `tests.TestCifar8DpCheck
  moreLinkArgs := lowererLink

/-- Step-time bench for the same two renders the tie compares. The batched render is
    far more emitted ops because `pretty` has no CSE and the batched backward ops
    are self-contained recomputes; the open question is whether XLA's own CSE collapses that. Both
    are compiled in one process and their steps interleaved, so the comparison is drift-free. -/
lean_exe «resnet34-adam-bench» where
  root := `tests.TestResnet34AdamBench
  moreLinkArgs := lowererLink

-- Pins the image/label pairing invariant of `F32.shuffle` on a synthetic
-- dataset where label k is derivable from image k. A shuffle that swaps a
-- hardcoded 4 bytes of label per record silently mispairs every detection and
-- segmentation batch (the detector's mAP collapsed to near zero).
-- Hermetic — no data files, no GPU.
lean_exe «test-shuffle-pairing» where
  root := `tests.TestShufflePairing
  moreLinkArgs := lowererLink

-- A `Ddpm.sampleNoise` that seeds its xorshift by XOR alone and reads the first
-- uniform from the TOP of the word gives nearby seeds a shared Box-Muller radius:
-- the 2-D demo's 2048 starting points sit on a circle instead of filling a
-- Gaussian. Per-axis mean and variance are correct under that defect, so this
-- asserts the RADIUS is Rayleigh. Hermetic — no data files, no GPU.
lean_exe «test-sample-noise-seeding» where
  root := `tests.TestSampleNoiseSeeding
  moreLinkArgs := lowererLink

-- Checks every DatasetIO's declared `trainPixels` / `labelBytesPerRecord`
-- against what its C loader actually allocates. Skips absent datasets, so it
-- is a pre-flight check rather than a CI job — run it whenever a dataset or
-- its preprocessing script changes.
lean_exe «test-dataset-record-sizes» where
  root := `tests.TestDatasetRecordSizes
  moreLinkArgs := lowererLink

lean_exe «test-unet-forward» where
  root := `tests.TestUnetForward

lean_exe «test-yolov1-mutex» where
  root := `tests.TestYolov1Mutex
  moreLinkArgs := lowererLink

lean_exe «test-resnet-residual» where
  root := `tests.TestResnetResidual

-- Dischargeability sanity check: 11 examples confirming every
-- Differentiable hypothesis the proofs propagate is satisfiable for
-- the architecture functions (dense, softmax, layerNorm, the flat
-- transformer pieces, mhsaLayer_flat). If any goes vacuous on a
-- refactor, this will fail at build time.
lean_exe «test-diff-sanity» where
  root := `tests.TestDifferentiableSanity

-- ─── tests/vjp_oracle/ — one binary per axiom under test (tests/vjp_oracle/README.md) ───

lean_exe «vjp-oracle-dense» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDense
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-dense-relu» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDenseRelu
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-conv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConv
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-convbn» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConvBn
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-conv-pool» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConvPool
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-residual» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleResidual
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-depthwise» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDepthwise
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-attention» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleAttention
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-mbconv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleMbConv
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-global-avg-pool» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleGlobalAvgPool
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-bottleneck» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleBottleneck
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-mbconv-v3» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleMbConvV3
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-fused-mbconv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleFusedMb
  moreLinkArgs := lowererLink

lean_exe «vjp-oracle-uib» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleUib
  moreLinkArgs := lowererLink

-- ─── Bestiary/ — architecture-only NetSpec examples: print, no training ───

lean_exe «bestiary-alphazero» where
  root := `Bestiary.AlphaZero

lean_exe «bestiary-highway» where
  root := `Bestiary.Highway

lean_exe «bestiary-densenet» where
  root := `Bestiary.DenseNet

lean_exe «bestiary-vgg» where
  root := `Bestiary.VGG

lean_exe «bestiary-resnet» where
  root := `Bestiary.ResNet

lean_exe «bestiary-wrn» where
  root := `Bestiary.WRN

lean_exe «bestiary-mamba» where
  root := `Bestiary.Mamba

lean_exe «bestiary-swin» where
  root := `Bestiary.SwinT

lean_exe «bestiary-unet» where
  root := `Bestiary.UNet

lean_exe «bestiary-detr» where
  root := `Bestiary.DETR

lean_exe «bestiary-yolo» where
  root := `Bestiary.YOLO

lean_exe «bestiary-shufflenet» where
  root := `Bestiary.ShuffleNet

lean_exe «bestiary-evoformer» where
  root := `Bestiary.Evoformer

lean_exe «bestiary-muzero» where
  root := `Bestiary.MuZero

lean_exe «bestiary-mobilevit» where
  root := `Bestiary.MobileViT

lean_exe «bestiary-wavenet» where
  root := `Bestiary.WaveNet

lean_exe «bestiary-nerf» where
  root := `Bestiary.NeRF

lean_exe «bestiary-clip» where
  root := `Bestiary.CLIP

lean_exe «bestiary-squeezenet» where
  root := `Bestiary.SqueezeNet

lean_exe «bestiary-lenet» where
  root := `Bestiary.LeNet

lean_exe «bestiary-inception» where
  root := `Bestiary.Inception

lean_exe «bestiary-xception» where
  root := `Bestiary.Xception

lean_exe «bestiary-alexnet» where
  root := `Bestiary.AlexNet

lean_exe «bestiary-bert» where
  root := `Bestiary.BERT

lean_exe «bestiary-shufflenetv2» where
  root := `Bestiary.ShuffleNetV2

lean_exe «bestiary-gpt» where
  root := `Bestiary.GPT

lean_exe «bestiary-diffusion» where
  root := `Bestiary.Diffusion

lean_exe «bestiary-sam» where
  root := `Bestiary.SAM

lean_exe «bestiary-whisper» where
  root := `Bestiary.Whisper

lean_exe «bestiary-llava» where
  root := `Bestiary.LLaVA

lean_exe «bestiary-stable-diffusion» where
  root := `Bestiary.StableDiffusion

lean_exe «bestiary-boltzmann-generator» where
  root := `Bestiary.BoltzmannGenerator

lean_exe «bestiary-pinn» where
  root := `Bestiary.PINN

lean_exe «bestiary-fno» where
  root := `Bestiary.FNO

lean_exe «bestiary-fourcastnet» where
  root := `Bestiary.FourCastNet

lean_exe «bestiary-segformer» where
  root := `Bestiary.SegFormer

lean_exe «bestiary-vae» where
  root := `Bestiary.VAE

lean_exe «bestiary-deeplab» where
  root := `Bestiary.DeepLabV3Plus

lean_exe «bestiary-maskrcnn» where
  root := `Bestiary.MaskRCNN

lean_exe «bestiary-dcgan» where
  root := `Bestiary.DCGAN

lean_exe «bestiary-cyclegan» where
  root := `Bestiary.CycleGAN

lean_exe «bestiary-alphago» where
  root := `Bestiary.AlphaGo

lean_exe «bestiary-pix2pix» where
  root := `Bestiary.Pix2Pix

lean_exe «bestiary-nystromformer» where
  root := `Bestiary.Nystromformer

lean_exe «bestiary-qanet» where
  root := `Bestiary.QANet

require checkdecls from git "https://github.com/PatrickMassot/checkdecls.git"

-- ═══════════════════════════════════════════════════════════════════════
-- Demo groups: one command that builds + runs a curated chunk of trainers,
-- tiered by time budget. `lake run mnist` (the short tier) / `lake run cifar` (the longer one);
-- anything bigger is a deliberate single-model run (see run.sh). Backend
-- auto-detects (cuda if `nvidia-smi` is present, else rocm) but `IREE_BACKEND`
-- overrides; GPU honors `LEAN_DEMO_GPU` (default 0). Each trainer streams live
-- and tees to `runs/<date>-<name>/<name>.log` via run.sh.
-- ═══════════════════════════════════════════════════════════════════════

/-- cuda when an NVIDIA GPU is visible (`nvidia-smi -L` succeeds), else rocm. -/
private def detectBackend : IO String := do
  let ok (cmd : String) (args : Array String) : IO Bool := do
    try pure ((← IO.Process.output { cmd, args }).exitCode == 0) catch _ => pure false
  -- ROCm only when its tool answers and NVIDIA's does not; anything else is the NVIDIA default.
  if ← ok "nvidia-smi" #["-L"] then return "cuda"
  if ← ok "rocm-smi" #[] then return "rocm"
  return "cuda"

/-- `ffi/libpjrt_ffi.so` is **not** a lake target — it is the gcc one-liner documented at the head
    of the XLA/PJRT section. Build it when it is missing or older than its source, so
    `lake run <group>-xla` works from a fresh clone instead of failing at startup (the shim is
    dlopen'd, not linked, so its absence is not caught at build time), and so an edited shim
    cannot be silently run stale (the `.vmfb`-cache hazard's cousin). -/
private def ensurePjrtShim : IO Bool := do
  let src : System.FilePath := "ffi/pjrt_ffi.c"
  let so  : System.FilePath := "ffi/libpjrt_ffi.so"
  let stale ← if !(← so.pathExists) then pure true else do
    let a ← src.metadata; let b ← so.metadata
    pure (b.modified.sec < a.modified.sec)
  if !stale then
    IO.println s!"  ✓ {so} up to date"
    return true
  IO.println s!"  ▸ building {so} (gcc -fPIC -O2 -shared {src} -ldl)"
  let r ← IO.Process.output
    { cmd := "gcc", args := #["-fPIC", "-O2", "-shared", src.toString, "-ldl", "-o", so.toString] }
  if r.exitCode != 0 then
    IO.eprintln s!"    ✗ shim build failed:\n{r.stderr.take 2000}"
    return false
  IO.println s!"    ✓ built {so}"
  return true

/-- The XLA path `dlopen`s a PJRT plugin at run time (`$PJRT_PLUGIN`, else the path compiled
    into the shim). Nothing links against it, so a missing plugin surfaces as a `dlopen` error
    at the first step rather than at build time — say so up front. -/
private def notePjrtPlugin : IO Unit := do
  match ← IO.getEnv "PJRT_PLUGIN" with
  | some p =>
    if ← System.FilePath.pathExists p then IO.println s!"  ✓ PJRT_PLUGIN={p}"
    else IO.println s!"  ⚠ PJRT_PLUGIN={p} does NOT exist — the first step will fail in dlopen"
  | none =>
    IO.println "  ▸ PJRT_PLUGIN unset — falling back to the path compiled into ffi/pjrt_ffi.c \
(a jax rocm/cuda plugin .so). If the run dies in dlopen, set PJRT_PLUGIN to your plugin."

/-- Build then run each named trainer in sequence (streaming) via `run.sh`.

    `xla := true` selects the PJRT peers: it builds the shim first and reports the plugin, and
    it does NOT need the venv, because those binaries compile in-process through PJRT instead of
    shelling out to `iree-compile`. One loop serves both so the two paths cannot drift.

    The GPU is `$LEAN_DEMO_GPU`, else the `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` the
    caller already set, else 0 — `run.sh` sets the device variable itself, so an ordinary
    `CUDA_VISIBLE_DEVICES=1 lake run cifar` used to land on GPU 0.

    A trainer that fails does not stop the group — the rest of a 5½-hour Imagenette pass is worth
    having — but it is named at the end and the script exits 1. -/
private def runDemoGroup (names : List String) (xla : Bool := false) : IO UInt32 := do
  let backend ← match ← IO.getEnv "IREE_BACKEND" with
    | some b => pure b
    | none   => detectBackend
  let gpu ← do
    let pick := [← IO.getEnv "LEAN_DEMO_GPU", ← IO.getEnv "CUDA_VISIBLE_DEVICES",
                 ← IO.getEnv "HIP_VISIBLE_DEVICES"].filterMap id |>.filter (!·.isEmpty)
    pure (pick.head?.getD "0")
  if xla then
    IO.println "━━━ XLA/PJRT backend ━━━"
    if !(← ensurePjrtShim) then return 1
    notePjrtPlugin
  -- The IREE trainers shell out to `iree-compile`; put the project venv on PATH so
  -- `lake run` works without pre-activating it (the usual one-click footgun).
  let venvBin := (← IO.currentDir) / ".venv" / "bin"
  -- Name the lowerer explicitly for BOTH groups. Binaries on `lowererLink` pick
  -- their backend from this at run time and DEFAULT to XLA, so without it
  -- `lake run *-iree` would silently run XLA.
  let lowerer := if xla then "xla" else "iree"
  -- The XLA tiers run resident — parameters stay on the device between steps — which is how
  -- every chapter transcript ran and what Appendix B's tier times assume (1.3–2× the wall
  -- clock without it, numerics unchanged). Since 2026-09-29 that is ffi/pjrt_ffi.c's default,
  -- so nothing is injected here; PJRT_FFI_RESIDENT=0 in the env takes the copying path.
  let runEnv ← do
    if ← System.FilePath.pathExists (venvBin / "iree-compile") then
      pure #[("PATH", some s!"{venvBin}:{(← IO.getEnv "PATH").getD ""}"),
             ("LEAN_MLIR_LOWERER", some lowerer)]
    else pure #[("LEAN_MLIR_LOWERER", some lowerer)]
  let mut failed : Array String := #[]
  for n in names do
    IO.println s!"\n━━━ {n}: build ━━━"
    let bp ← IO.Process.spawn { cmd := "lake", args := #["build", n] }
    if (← bp.wait) != 0 then
      IO.eprintln s!"build failed: {n}"
      return 1
    IO.println s!"━━━ {n}: run (gpu {gpu}, {backend}) ━━━"
    let rp ← IO.Process.spawn { cmd := "./run.sh", args := #[n, gpu, backend], env := runEnv }
    let rc ← rp.wait
    if rc != 0 then
      IO.eprintln s!"━━━ {n}: FAILED (exit {rc}) — continuing with the rest ━━━"
      failed := failed.push n
  if failed.isEmpty then return 0
  IO.eprintln s!"\n━━━ {failed.size} of {names.length} failed: {", ".intercalate failed.toList} ━━━"
  return 1

/-- `lake run mnist-iree` — the same three binaries as `lake run mnist`, with the
    IREE lowerer selected instead of XLA, which is slower here. -/
script «mnist-iree» do
  runDemoGroup ["mnist-linear-verified", "mnist-mlp-verified", "mnist-cnn-verified"]

/-- `lake run cifar-iree` — the ch.4 six-arm optimizer ablation (SGD/momentum/adam ×
    bn/no-bn) on the IREE lowerer.

    These are the WIDE-head (`d1 = 512`) nets, because those are what Chapter 4
    actually runs and quotes: `cifar8w-bn-ablation` is the binary behind §4.1's
    listing and `runs/2026-08-12-cifar8w-6arm-xla-cuda/`. The narrow `d1 = 64`
    nets are still built and still proved — `cifarCnn8HasVJPAt` is parametric in
    the head width, so neither needs its own proof — they are just not what
    `lake run cifar` shows. -/
script «cifar-iree» do
  runDemoGroup ["cifar8w-ablation", "cifar8w-bn-ablation"]

/-- `lake run imagenette-iree` — the Part-I verified Imagenette trainers (the rest of
    the chapters: ResNet-34, MobileNetV2, EfficientNet-B0, ConvNeXt-T, ViT-Tiny),
    80-epoch AdamW at 224². Many hours end-to-end on one card (the ViT chapter's results table
    has the per-net wall clocks) — a real time investment, not a quick demo.

    **This is FIVE nets, not the seven of `imagenette`.** MobileNetV4 and ResNet-50 are
    XLA-only: `apps/imagenette/` has `MainMobilenetV4VerifiedAdamXla` and
    `MainResnet50VerifiedAdamXla` and **no IREE peers**, so there is nothing to put here. That is
    an omission of drivers, not of nets — both have 80-epoch numbers on their certified bytes,
    both off the XLA path. **`imagenette` is the official set**; this
    group is the IREE half of the cross-backend comparison. -/
script «imagenette-iree» do
  runDemoGroup ["resnet34-verified-adam", "mobilenetv2-verified-adam",
                "efficientnet-verified-adam", "convnext-verified-adam",
                "vit-verified-adam"]

-- ═══════════════════════════════════════════════════════════════════════
-- THE DEFAULT DEMO GROUPS — `lake run {mnist,cifar,imagenette}`.
--
-- XLA is the default because it is where every quoted number comes from, it is much faster than
-- IREE on EfficientNet, and it is the only path with MobileNetV4 and ResNet-50 at all.
--
-- Same nets, same certified artifacts, same schedules and seeds as the `-iree` group — the
-- ONLY difference is which trusted lowerer consumes the emitted StableHLO, which is the whole
-- point of the second backend.
--
-- Why you'd reach for these: XLA trains EfficientNet's 80 epochs several times faster than
-- IREE, and multi-GPU is reachable ONLY
-- here, since collectives exist on the PJRT path and the IREE shim refuses a DP
-- entry point outright. Re-measure per net rather than assuming that ratio; it is one
-- net's number, on a depthwise-convolution-heavy net.
--
-- Two coverage gaps, both named rather than papered over — see each docstring.
-- ═══════════════════════════════════════════════════════════════════════

/-- `lake run mnist` — the three verified MNIST demos (linear/MLP/CNN) on XLA.

    This names the SAME three binaries as `lake run mnist-iree`. The only difference between the
    two scripts is the lowerer `runDemoGroup` puts in the environment, which is
    the strongest form the G2 comparison can take — one binary, run twice. -/
script mnist do
  runDemoGroup ["mnist-linear-verified", "mnist-mlp-verified",
                "mnist-cnn-verified"] (xla := true)

/-- `lake run cifar` — the six-way Chapter-4 optimizer ablation on XLA
    (SGD / Nesterov-momentum / AdamW × BN / no-BN).

    A mirror of `lake run cifar-iree` in the literal sense: the SAME binaries,
    differing only in the lowerer `runDemoGroup` selects. Both sides run the WIDE-head pair so that
    the demo trains the net Chapter 4 reports on; each ablation binary runs its three optimizers in
    sequence on one controlled pipeline (shuffle + hflip + cosine-warmup), so the six arms are two
    binaries rather than six. Wide costs noticeably more wall-clock per epoch than the narrow pair
    and buys no accuracy — §4.3's head-width sweep is exactly that
    finding — but matching the chapter is worth the minutes. -/
script cifar do
  runDemoGroup ["cifar8w-ablation", "cifar8w-bn-ablation"] (xla := true)

/-- `lake run imagenette` — the seven Imagenette nets on the verified XLA path, 80 epochs of AdamW
    at 224², in book order (ch.5 → ch.9, each chapter's side quest right after it).

    **SEVEN NETS, and this is THE OFFICIAL SET** — `lake run mnist` (3) + `lake run cifar` (6) +
    this (7) is the whole demo surface; nothing else is a headline runner. MobileNetV4 and
    ResNet-50 are here and **only** here, because neither has an IREE driver (see
    `lake run imagenette-iree`'s note).

    The latest 80-epoch top-1 and wall clock on the certified bytes live in each net's chapter
    and run directory: ResNet-34 in `runs/2026-09-12-r34-ablation-fp32-seeds/`, ResNet-50 and
    MobileNetV4-Conv-M in `runs/2026-08-31-imagenette-n3/`, ConvNeXt-T in
    `runs/2026-09-13-convnext-imagenette-ls1e-6-resident/`, and MobileNetV2, EfficientNet-B0 and
    ViT-Tiny in chapters 6, 7 and 9.

    ConvNeXt's run is the paper's 1e-6 layer-scale init; `runs/2026-09-13-convnext-imagenette-ls1e-6/README.md`
    compares it with the old init of 1.0. A second
    `lake run imagenette` scores each finished checkpoint instead of training it again; set
    `LEAN_MLIR_CKPT_TAG` for a fresh run beside it. -/
script imagenette do
  -- Book order (ch.5 -> ch.9, each chapter's side quest right after it), so the
  -- group narrates in the order a reader met the nets.
  runDemoGroup ["resnet34-verified-adam", "resnet50-verified-adam",
                "mobilenetv2-verified-adam", "mobilenetv4-verified-adam",
                "efficientnet-verified-adam", "convnext-verified-adam",
                "vit-verified-adam"] (xla := true)

/-- The Orin tier's rows: (exe, pool fraction, the row's extra env), six of `imagenette`'s seven nets
    in book order. The fraction is `LEAN_MLIR_MEM_FRACTION`, the BFC pool as a share of the board's one
    DRAM — it must hold the net's train-step activation arena and leave the process the rest — each from
    the 2026-09-13 sweep in `deploy/ORIN.md` §4. ResNet-50 is the `acc2x16` render (micro-batch 16 × 2, the
    same recipe at effective batch 32), because the batch-32 step's arena does not fit at any fraction;
    `LEAN_MLIR_G2_STEPS=590` makes its epoch the 9,440 images a batch-32 epoch takes. ConvNeXt-T is
    absent: its renderer hard-codes batch 32 and that step does not fit the board. -/
private def orinRows : List (String × String × List (String × String)) :=
  [ ("resnet34-verified-adam",     "0.25", []),
    ("resnet50-verified-adam",     "0.36", [("LEAN_MLIR_VARIANT", "acc2x16"), ("LEAN_MLIR_BATCH", "16"),
                                            ("LEAN_MLIR_G2_STEPS", "590")]),
    ("mobilenetv2-verified-adam",  "0.25", []),
    ("mobilenetv4-verified-adam",  "0.25", []),
    ("efficientnet-verified-adam", "0.30", []),
    ("vit-verified-adam",          "0.25", []) ]

/-- `lake run imagenette-orin [plan] [net-prefix …]` — the Imagenette tier on a Jetson Orin Nano:
    `orinRows` through `deploy/orin_imagenette.sh`, the board's recipe (the train split streamed from
    disk, the pool reserved once at the row's fraction, the process under a memory cap) with a relaunch
    that resumes from the checkpoint while each attempt completes an epoch. About a day for the six
    rows at 80 epochs (`deploy/ORIN.md` §4 has the per-net wall clocks); the board needs `~/pjrt/orin_env.sh`
    sourced and `jetson_clocks` applied, both of which the plan reports.

    **Six nets, not `imagenette`'s seven**, and ResNet-50 on a different render (`acc2x16`), so this is a
    named tier and not an environment: the drop-in platforms (a desktop CUDA or ROCm card) run
    `lake run imagenette` itself. `plan` prints each row's env, command and checkpoint state and launches
    nothing; a net-prefix list (`lake run imagenette-orin vit r34`) runs a subset. A row that fails is
    named at the end and the rest run, as in `imagenette`; a second invocation resumes every row from
    its checkpoint and scores the finished ones. The recipe is not Tegra-specific — on a desktop the
    rows run under the box's plugin, which is how it is smoke-tested here. -/
script «imagenette-orin» (args) do
  let (plan, only) := match args with
    | "plan" :: rest => (true, rest)
    | rest           => (false, rest)
  let rows := orinRows.filter fun (exe, _, _) => only.isEmpty || only.any (exe.startsWith ·)
  if rows.isEmpty then
    IO.eprintln s!"no Orin row matches {only} — rows: {orinRows.map (·.1)}"
    return 1
  IO.println s!"━━━ lake run imagenette-orin: {rows.length} row(s){if plan then " — plan only" else ""} ━━━"
  if !plan then
    IO.println "━━━ XLA/PJRT backend ━━━"
    if !(← ensurePjrtShim) then return 1
    notePjrtPlugin
  let mut failed : Array String := #[]
  for (exe, fraction, extra) in rows do
    if !plan then
      IO.println s!"\n━━━ {exe}: build ━━━"
      let bp ← IO.Process.spawn { cmd := "lake", args := #["build", exe] }
      if (← bp.wait) != 0 then
        IO.eprintln s!"build failed: {exe}"
        return 1
    IO.println ""
    let env : Array (String × Option String) :=
      #[("BIN", some exe), ("FRACTION", some fraction)]
        ++ (if plan then #[("DRY_RUN", some "1")] else #[])
        ++ (extra.map fun (k, v) => (k, some v)).toArray
    let p ← IO.Process.spawn { cmd := "deploy/orin_imagenette.sh", env := env }
    let rc ← p.wait
    if rc != 0 then
      IO.eprintln s!"━━━ {exe}: {if plan then "plan REFUSED" else "FAILED"} (exit {rc}) — continuing with the rest ━━━"
      failed := failed.push exe
  if failed.isEmpty then return 0
  IO.eprintln s!"\n━━━ {failed.size} of {rows.length} {if plan then "refused" else "failed"}: {", ".intercalate failed.toList} ━━━"
  return 1


-- ═══════════════════════════════════════════════════════════════════════
-- `lake run imagenet` — the fourth tier — and one `lake run <job>` per ImageNet job.
--
-- A job is `scripts/jobs/<job>.conf`: the device list, both replica knobs, the variant, the
-- per-replica batch and the per-net feed setting all live THERE, and `scripts/supervise.sh` is
-- the engine that runs it with the restart policy a multi-day run on this box needs. These
-- scripts add nothing to that. They make the job's own name the command, so a seven-variable
-- incantation becomes `lake run r34-default-bf16-4gpu`.
--
-- Modes, the first argument to a job script:
--   (none)   supervised run — resumes from the job's checkpoint, restarts on AER / heat / stall
--   plan     DRY_RUN=1: the name check, the job's PRECHECK and the wall-clock on file; no launch
--   once     ONCE=1: one foreground run with the job's env and no restart policy (probes)
-- The job's knobs are edited in its conf, never passed here.
-- ═══════════════════════════════════════════════════════════════════════

/-- The ImageNet tier's rows in chapter order, each chapter's side quest right after it (the
    `imagenette` convention): (job config, the exe it runs, the book's row). The seven Track-4
    rows are the ones with chapter numbers; the seven side-quest rows are the verified halves of
    `planning/side_quest_runs.md`'s queue (their JAX peers, the `*-jax-4gpu` confs, stay
    `scripts/supervise.sh`-only), each conf's ETA measured on this box on 2026-10-05. MobileNetV4's
    row is the 500-epoch `full` job; its 100-epoch pair (`mnv4-default-4gpu`, landed 09-27) is an
    axis sibling. Axis siblings — `r34-default-4gpu`, `r50-2018-4gpu`, `r50-a3-4gpu`, `r50-a3-wxclip-4gpu`,
    `r50-a3-wxclip-bf16-4gpu` (RSB-A3 at 8 × 64: the same effective 2048 at a BatchNorm group of 256),
    `vit-default-4gpu`, `vits-default-g512-4gpu`, `vitb-default-g512-4gpu` (the non-EMA f32 S/B jobs),
    `cnx-default-emabf16-4gpu`, `cnxs-default-4gpu` / `cnxb-default-4gpu` (ConvNeXt-S/B f32 at 4 × 32) (ConvNeXt-T with the EMA shadow), the MNv4 100-epoch pair `mnv4-default-4gpu` /
    `mnv4-default-jax-4gpu` (run as overnight chunks, `START_AT`/`STOP_AT`), `selftest`, `chunktest` — stay `scripts/supervise.sh`-only.
    `r34-default-bf16-4gpu`, `r50-2018-bf16-4gpu` and `r50-a3-wxclip4x128-bf16-4gpu` are the 4× 3060
    box's confs, named by the
    book's Track-4 table as the jobs behind their rows; their PRECHECK passes here too (`_box.sh`
    picks the plugin), and their ETAs are the 3060 box's where they say so. -/
private def imagenetRows : List (String × String × String) :=
  [ ("r34-default-bf16-4gpu",  "resnet34-imagenet-verified",     "Ch. 5  ResNet-34, the 2018 recipe, bf16"),
    ("r50-2018-bf16-4gpu",     "resnet50-imagenet-verified",     "Ch. 5  ResNet-50, 2018"),
    ("r50-a3-wxclip4x128-bf16-4gpu", "resnet50-imagenet-verified", "Ch. 5  ResNet-50, RSB-A3 (train@160, 4 × 128), bf16"),
    ("r50-a2-bf16-4gpu",       "resnet50-imagenet-verified",     "Ch. 5  ResNet-50, RSB-A2 (300 ep, 4 × 128), bf16 (side quest)"),
    ("r50-a1-bf16-4gpu",       "resnet50-imagenet-verified",     "Ch. 5  ResNet-50, RSB-A1 (600 ep, 4 × 128), bf16 (side quest)"),
    ("mnv2-default-4gpu",      "mobilenetv2-imagenet-verified",  "Ch. 6  MobileNetV2"),
    ("mnv4-full-4gpu",         "mobilenetv4-imagenet-verified",  "Ch. 6  MobileNetV4-Conv-M, `full` (500 ep), bf16 (side quest)"),
    ("enet-default-4gpu",      "efficientnet-imagenet-verified", "Ch. 7  EfficientNet-B0"),
    ("cnx-default-4gpu",       "convnext-imagenet-verified",     "Ch. 8  ConvNeXt-T"),
    ("cnxs-default-emabf16-4gpu", "convnext-s-imagenet-verified", "Ch. 8  ConvNeXt-S, EMA + bf16 (side quest)"),
    ("cnxb-default-emabf16-4gpu", "convnext-b-imagenet-verified", "Ch. 8  ConvNeXt-B, EMA + bf16 (side quest)"),
    ("vit-default-emabf16-4gpu", "vit-imagenet-verified",        "Ch. 9  ViT-Tiny (DeiT-Ti), EMA + bf16"),
    ("vits-default-emabf16-4gpu", "vit-s-imagenet-verified",     "Ch. 9  ViT-S (DeiT-S), EMA + bf16 (side quest)"),
    ("vitb-default-emabf16-4gpu", "vit-b-imagenet-verified",     "Ch. 9  ViT-B (DeiT-B), EMA + bf16 (side quest)") ]

/-- One job in one mode. `plan` skips the build and says whether the binary exists; the other
    two build the exe first (the `runDemoGroup` convention) and then hand the job to the engine,
    which owns everything else — devices, env, checkpoint resume, restart policy. -/
private def runJob (job exe mode : String) : IO UInt32 := do
  let bin : System.FilePath := ".lake" / "build" / "bin" / exe
  if mode == "plan" then
    IO.println (if ← bin.pathExists then s!"  ✓ {bin}" else s!"  ▸ {bin} not built yet — a run builds it")
  else
    IO.println s!"━━━ {job}: build {exe} ━━━"
    let bp ← IO.Process.spawn { cmd := "lake", args := #["build", exe] }
    if (← bp.wait) != 0 then
      IO.eprintln s!"build failed: {exe}"
      return 1
  let env : Array (String × Option String) :=
    if mode == "plan" then #[("DRY_RUN", some "1")]
    else if mode == "once" then #[("ONCE", some "1")]
    else #[]
  IO.println s!"━━━ {job}: scripts/supervise.sh {job}{if mode == "run" then "" else s!"  ({mode})"} ━━━"
  let p ← IO.Process.spawn { cmd := "scripts/supervise.sh", args := #[job], env := env }
  pure (← p.wait)

/-- The body of every per-job script: parse the mode, check the shim once, run the row. -/
private def runJobScript (job : String) (args : List String) : IO UInt32 := do
  let mode := args.head?.getD "run"
  if !["run", "plan", "once"].contains mode then
    IO.eprintln s!"unknown mode `{mode}` — one of: (none) | plan | once"
    return 1
  match imagenetRows.find? (·.1 == job) with
  | none => IO.eprintln s!"{job} is not an imagenetRows entry"; return 1
  | some (_, exe, row) =>
    IO.println s!"━━━ {row} ━━━"
    if mode != "plan" then
      IO.println "━━━ XLA/PJRT backend ━━━"
      if !(← ensurePjrtShim) then return 1
    runJob job exe mode

script «r34-default-bf16-4gpu»  (args) do runJobScript "r34-default-bf16-4gpu" args
script «r50-2018-bf16-4gpu»     (args) do runJobScript "r50-2018-bf16-4gpu" args
script «r50-a3-wxclip4x128-bf16-4gpu» (args) do runJobScript "r50-a3-wxclip4x128-bf16-4gpu" args
script «r50-a2-bf16-4gpu»       (args) do runJobScript "r50-a2-bf16-4gpu" args
script «r50-a1-bf16-4gpu»       (args) do runJobScript "r50-a1-bf16-4gpu" args
script «mnv2-default-4gpu»      (args) do runJobScript "mnv2-default-4gpu" args
script «mnv4-full-4gpu»         (args) do runJobScript "mnv4-full-4gpu" args
script «enet-default-4gpu»      (args) do runJobScript "enet-default-4gpu" args
script «cnx-default-4gpu»       (args) do runJobScript "cnx-default-4gpu" args
script «cnxs-default-emabf16-4gpu» (args) do runJobScript "cnxs-default-emabf16-4gpu" args
script «cnxb-default-emabf16-4gpu» (args) do runJobScript "cnxb-default-emabf16-4gpu" args
script «vit-default-emabf16-4gpu» (args) do runJobScript "vit-default-emabf16-4gpu" args
script «vits-default-emabf16-4gpu» (args) do runJobScript "vits-default-emabf16-4gpu" args
script «vitb-default-emabf16-4gpu» (args) do runJobScript "vitb-default-emabf16-4gpu" args

/-- `lake run imagenet` — the fourth tier: the fourteen ImageNet rows, in chapter order.

    PLAN-ONLY unless the first argument is `start`. Bare, it runs every row's `plan` — the name
    check, the PRECHECK, the wall-clock on file — and launches nothing, because the tier is weeks
    of four-card time and this box has crashed under it. `lake run imagenet start` runs the rows
    through `scripts/supervise.sh` in order and stops at the first failure; a row whose checkpoint
    is complete exits 0 at once, so re-invoking resumes where it left off. `lake run imagenet
    start cnx vit` (job-name prefixes) runs a subset. Rows are sequential by construction: every
    job claims all four cards. -/
script imagenet (args) do
  let (go, only) := match args with
    | "start" :: rest => (true, rest)
    | rest            => (false, rest)
  let rows := imagenetRows.filter fun (j, _, _) => only.isEmpty || only.any (j.startsWith ·)
  if rows.isEmpty then
    IO.eprintln s!"no ImageNet row matches {only}"
    return 1
  IO.println s!"━━━ lake run imagenet: {rows.length} row(s) in chapter order — plan first ━━━"
  let mut refused : List String := []
  for (j, exe, row) in rows do
    IO.println s!"\n▸ {row}"
    if (← runJob j exe "plan") != 0 then refused := refused ++ [j]
  IO.println s!"\n━━━ plan: {rows.length - refused.length} of {rows.length} row(s) ready\
              {if refused.isEmpty then "" else s!" — refused: {refused}"} ━━━"
  if !go then
    IO.println "plan only, nothing launched. `lake run imagenet start [job-prefix …]` runs these rows in\n\
                order through scripts/supervise.sh, each resuming from its own checkpoint."
    return (if refused.isEmpty then 0 else 1)
  if !refused.isEmpty then
    IO.eprintln "⛔ not starting: fix the refused rows first, or name a subset that excludes them"
    return 1
  IO.println "━━━ XLA/PJRT backend ━━━"
  if !(← ensurePjrtShim) then return 1
  for (j, exe, row) in rows do
    IO.println s!"\n▸ {row}"
    if (← runJob j exe "run") != 0 then
      IO.eprintln s!"⛔ {j} did not complete — the tier stops here; re-run to resume"
      return 1
  return 0

-- ═══════════════════════════════════════════════════════════════════════
-- `lake run download` — fetch the core datasets the verified trainers + the
-- benchmark need. Each entry pairs a download script with a sentinel file that
-- proves the dataset is already on disk, so re-running is a fast no-op. Used both
-- by `lake run download` and by `lake run benchmark` (which auto-downloads any
-- missing dataset as its first step, instead of soft-failing on imagenette).
-- ═══════════════════════════════════════════════════════════════════════

/-- The core datasets: `(label, download script, sentinel that exists once it's
    downloaded)`. MNIST + CIFAR feed the benchmark's dense/conv probes; Imagenette
    feeds the ViT/attn probe and the `lake run imagenette-iree` tier. -/
def coreDatasets : List (String × String × String) :=
  [ ("MNIST",      "scripts/datasets/download_mnist.sh",      "data/train-images-idx3-ubyte"),
    ("CIFAR-10",   "scripts/datasets/download_cifar.sh",      "data/cifar-10/data_batch_1.bin"),
    ("Imagenette", "scripts/datasets/download_imagenette.sh", "data/imagenette/train.bin") ]

/-- Run a dataset's download script (via `bash`, so the exec bit doesn't matter)
    if its sentinel file is missing. Returns `false` on a download failure. -/
def ensureDataset (label sh sentinel : String) : IO Bool := do
  if ← System.FilePath.pathExists sentinel then
    IO.println s!"  ✓ {label} present ({sentinel})"
    return true
  IO.println s!"  ▸ {label} missing — running ./{sh} …"
  let rp ← IO.Process.spawn { cmd := "bash", args := #[sh] }
  if (← rp.wait) != 0 then
    IO.eprintln s!"    ✗ download failed: ./{sh}"
    return false
  pure (← System.FilePath.pathExists sentinel)

/-- Download any missing core dataset. Returns `false` if any download failed. -/
def ensureCoreData : IO Bool := do
  let mut ok := true
  for (label, sh, sentinel) in coreDatasets do
    if !(← ensureDataset label sh sentinel) then ok := false
  pure ok

/-- `lake run download` — fetch the core datasets (MNIST, CIFAR-10, Imagenette)
    that the verified trainers and `lake run benchmark` need, downloading only the
    ones not already on disk. Imagenette additionally needs `python3` + Pillow for
    the binary preprocessing step (see ./scripts/datasets/download_imagenette.sh). -/
script download do
  IO.println "━━━ lake run download ━━━ core datasets: MNIST, CIFAR-10, Imagenette"
  if ← ensureCoreData then
    IO.println "\n  ✓ all core datasets present."
    return 0
  else
    IO.eprintln "\n  ✗ one or more downloads failed (see above)."
    return 1

-- ═══════════════════════════════════════════════════════════════════════
-- `lake run benchmark` (IREE) and `lake run benchmark-xla` (XLA/PJRT) —
-- estimate the book's training time on YOUR gpu.
-- Probes two fast verified nets for a few epochs (the only thing that runs),
-- reads steady-state ms/epoch from the trainer's own per-epoch print, and
-- scales the reference per-chapter wall-clock by the measured hardware factor.
-- Backend auto-detects (cuda if nvidia-smi present, else rocm) — works on
-- either vendor out of the box. A dense factor (MNIST-MLP) and a conv factor
-- (CIFAR-8-BN) scale the dense- vs conv-dominated chapters independently.
--
-- **EACH LOWERER HAS ITS OWN REFERENCE COLUMN, AND THAT IS NOT OPTIONAL.**
-- The two backends are not within noise of each other on the same card: measured
-- on the reference 7900 XTX, XLA beats IREE by a different factor on each of the conv, dense and
-- attn anchors, the attn gap by far the largest. A single blended factor would be badly wrong in
-- both directions, which is why the split is per family AND per lowerer. So dividing an XLA
-- probe by an IREE anchor conflates *your GPU vs a 7900 XTX* with *XLA vs IREE*, and reports a
-- training estimate several times too fast with no warning. That is why `BenchItem` carries
-- `refSecXla` and why there are `probe*RefMsXla` constants: a `BenchRef` bundles one lowerer's
-- anchors so a probe can only ever be divided by a reference measured on the same path.
--
-- REFERENCE NUMBERS below are per-chapter *training* wall-clock on a single AMD
-- 7900 XTX (gfx1100, ROCm 7.2). The MNIST/CIFAR rows and all three IREE probe
-- anchors (dense/conv/attn) were MEASURED directly from these verified trainers
-- (steady-state ms/{epoch,step} × the trainer's epoch/step count); the
-- R34/MNv2/ENet/ConvNeXt IREE Imagenette rows are the verified-adam tier runs
-- and the IREE ViT row is measured here, warm (the shorter ViT figure elsewhere is the
-- JAX bf16 path, not this verified trainer). The
-- IREE rows EXCLUDE the one-time IREE compile (minutes per arch, CPU-bound,
-- ~hardware-independent); the XLA rows need no such carve-out, since XLA compiles
-- in-process in seconds. Re-running either benchmark on a 7900 XTX reproduces its
-- own anchors (every factor reads about one).
--
-- Each row's derivation (steady-state ms/epoch or ms/step × the trainer's count, per lowerer)
-- and the re-measurements behind the caveats below were measured once and are archived in
-- historical/comment_measurements.md.
--
-- Known staleness caveats in the IREE column, left as-is rather than silently changed:
--   * ch3 (MNIST CNN) reads pessimistic: re-measured on the same card, same
--     basis (real data + eval, steady state) it is markedly faster. ch1/ch2/ch4 reproduce.
--   * the IREE Imagenette rows were measured on hand-written renders, not the
--     certified bytes. The XLA Imagenette rows are the current certified bytes.
--   * the IREE DENSE anchor also reads high: repeated measurements in one session on the
--     reference card all came in below it. Unlike the XLA anchors, which are medians of
--     several samples, the IREE ones are single historical samples. So on IREE read a low
--     dense factor as anchor noise, not as your card being slow; the conv and attn anchors
--     reproduce.
-- Correcting any of these means re-anchoring the IREE column from medians, which is
-- a deliberate separate change — it moves published per-chapter estimates.
-- ═══════════════════════════════════════════════════════════════════════

structure BenchItem where
  chapter : String
  family  : String          -- "dense" | "conv" | "attn"
  refSec  : Nat             -- IREE reference training wall-clock (s) on the 7900 XTX
  /-- XLA/PJRT reference wall-clock (s) on the same card. `none` = this chapter has no
      measured XLA reference, in which case `benchmark-xla` prints the row as `n/a` and
      leaves it out of the totals rather than borrowing the IREE number (which would be
      the mismatched-baseline trap). **Every chapter is measured**; the mechanism is kept
      because it is what makes an unmeasured row honest rather than invented. -/
  refSecXla : Option Nat
  tier    : String          -- "" | "mnist" | "cifar" | "imagenette"
  /-- **This chapter's BOTTLENECK MIX is not the one its probe measures**, so a single
      hardware factor cannot describe it and the row prints a RANGE instead.

      The `family` axis says which *ops* a chapter runs. It does not say what the chapter is
      *limited by*, and on this path most of them are limited by the parameter round trip, not by
      arithmetic: on the reference card the param share of a step is smaller for the conv probe
      (`cifar8-bn`, 32²) than for ResNet-34, EfficientNet or the MNIST CNN. A card whose transport:compute ratio differs from the reference's — e.g.
      PCIe Gen3 x8 against a 7900 XTX — therefore gets a *different* factor for each, and scaling a
      transport-bound chapter by a compute-bound probe is the mismatched-baseline trap one axis
      over: same lowerer, same card, wrong bottleneck.

      **MEASURED on ares (RTX 4060 Ti, PCIe Gen3 x8):** the conv and dense probes read factors on
      opposite sides of one (idle card) — a spread that is not noise but two different
      bottlenecks. A single number predicted ch5 at well under half of what the real 80-epoch run
      took (`runs/r34_pool3s2_80ep_aug04.log`); the bracket's transport end came in slightly low.

      **SO THE BRACKET DOES NOT CONTAIN THE TRUTH, AND MUST NOT BE SOLD AS A BOUND.** R34's real
      like-for-like ratio (its `refSecCuda` over its `refSecXla`) is above *both* probe factors. The cause is
      structural and known: the probes run `LEAN_MLIR_BENCH_SYNTH=1`, so they exclude the data
      loader **by design**, and per-epoch host overhead is about the size of the miss. A bracket over two synthetic probes cannot reach a real run's
      loader term. Read it as *"the compute/transport estimate spans this"*, not as an interval the
      answer lies in. It takes ch5 from badly optimistic to nearly right; that is the whole claim.

      On the reference card both factors read ~1.0, the range collapses, and nothing about the
      published column changes.

      Do NOT "fix" this by re-pointing these rows at the dense probe. That is one measured
      chapter, and one sample is not a measurement. The honest fix is a 224² conv probe with its own
      anchor measured on the reference card; until someone has that card, a bracket is the most
      the evidence supports. -/
  transportSensitive : Bool := false
  /-- **The CUDA (RTX 4060 Ti) reference wall-clock**, measured, not scaled from
      anything. ch5 is a real 80-epoch run (`runs/r34_pool3s2_80ep_aug04.log`, 66.7 s
      marginal epoch × 80); ch6-9 are eval-inclusive marginal epochs × 80; ch1-4 are 3 steady-state
      epochs × their own epoch counts. `benchmark-xla` picks this column on a CUDA backend, so on a
      4060 Ti every factor reads ~1.00 and the estimate IS the measurement. -/
  refSecCuda : Option Nat := none
  /-- **Direct mode** (`BENCH_DIRECT=1`): the `-xla` trainer that IS this chapter, and the epoch
      count it trains for. With both set, the chapter can be measured on THIS box instead of
      scaled from the reference card — which is the only way a row that is not its own probe can
      be accurate, since no hardware factor transfers across a bottleneck change. -/
  probeXla : String := ""
  epochs   : Nat := 0
  /-- Direct mode: steps/epoch, set ONLY for `trainAdamSched` nets (the five Imagenette ones).
      Those trainers print `Epoch N/80: loss=…` with **no ms**, so `lastEpochMs` cannot read them
      and a 3-epoch probe returns nothing. They report ms/**step** through the
      `LEAN_MLIR_MAX_STEPS` PROBE line instead, which is why `runProbe` already carries a
      `stepProbe` parameter for the attn probe. A non-zero value here selects that path.
      It is also strictly better: the step probe warms 8 and times 9..40 (seconds, not epochs),
      and it `return ()`s BEFORE any checkpoint write, so it cannot leave a
      marker. It excludes the EVAL pass, so these rows are TRAIN-ONLY — see the footnote. -/
  stepsPerEpoch : Nat := 0

/-- The XLA MNIST/CIFAR rows were measured on the reference 7900 XTX with the
    SAME construction as the IREE ones — steady-state ms/epoch (real data + eval, last of
    3 epochs) × the trainer's own epoch count, so the two columns are directly comparable.
    **All five XLA Imagenette rows are measured 80-epoch single-GPU runs on the current
    certified bytes** — ch5-8 `runs/<net>_xla_80ep_jul29.log` and ch9
    `runs/vit_xla_80ep_jul30.log` (wall **3491 s**, epoch marker 80).

    ch9 is also the **validation of the marginal-epoch method** the other rows lean on: its
    `(T₃−T₁)/2` measurement × 80 extrapolates to the real 80-epoch wall within a fraction of a
    percent. So `scripts/sweeps/marginal_epoch.sh` × epochs is trustworthy at
    this scale, which is worth knowing because it is far cheaper than an 80-epoch run.
    ch4 mirrors the IREE row's
    approximation — the BN arm's cost × 6 — so that the two columns stay comparable, even
    though the 3 no-BN arms are cheaper.

    The XLA MNIST/CIFAR rows are each a SINGLE steady-state sample, so they inherit the
    per-run spread documented on `probeConvRefMsXla`; the conv-family ones (ch3, ch4)
    are the affected pair. Treat them as that band, not as exact. -/
def benchTable : List BenchItem :=
  [ { chapter := "1  MNIST linear", family := "dense", refSec := 6,     refSecXla := some 3,    tier := "mnist",
      refSecCuda := some 3, probeXla := "mnist-linear-verified", epochs := 12 },
    { chapter := "2  MNIST MLP",    family := "dense", refSec := 38,    refSecXla := some 8,    tier := "mnist",
      refSecCuda := some 11, probeXla := "mnist-mlp-verified", epochs := 12 },
    { chapter := "3  MNIST CNN",    family := "conv",  refSec := 238,   refSecXla := some 41,   tier := "mnist",
      transportSensitive := true, refSecCuda := some 49, probeXla := "mnist-cnn-verified", epochs := 10 },
    { chapter := "4  CIFAR x6",     family := "conv",  refSec := 2038,  refSecXla := some 888,  tier := "cifar",
      refSecCuda := some 1514, probeXla := "cifar8w-bn-ablation", epochs := 240 },   -- 40 ep × 6 ARMS, approximated as the BN arm ×6 (the 3 no-BN arms are cheaper) — the same approximation the ref column makes, kept so the two stay comparable
    -- **WIDE-head**, because that is what `lake run cifar` and ch.4 are. refSecCuda is
    -- the BN arm's steady-state s/epoch × 240 on `cifar8w-bn-ablation`, which
    -- keeps this field's stated meaning ("3 steady-state epochs × their own epoch counts") and
    -- matches what direct mode computes from the same probe.
    -- The TRUE 6-arm wall, measured end-to-end on one 4060 Ti
    -- (`runs/2026-08-26-cifar8w-6arm-timing/`), is well below that: the BN-arm×6 approximation
    -- OVERSHOOTS here — the no-BN arms are half the cost, not "cheaper" by a little. The approximation is kept anyway
    -- because direct mode can only ever read ONE ms/epoch and the IREE/XLA ref columns make the
    -- same one; a row that mixed a true wall against two extrapolated ones would be the
    -- mismatched-baseline trap. `refSec` (IREE 2038) and `refSecXla` (888) are the narrow-head 7900
    -- XTX numbers and are mismatched against this row's net. Not scaled here on purpose — inventing
    -- a factor is exactly what this table exists to avoid. Re-measure on the ROCm box, or read them
    -- as narrow.
    { chapter := "5  ResNet-34",    family := "conv",  refSec := 34200, refSecXla := some 3780, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 5333, probeXla := "resnet34-verified-adam-xla", epochs := 80, stepsPerEpoch := 295 },  -- the PAPER net
    { chapter := "6  MobileNetV2",  family := "conv",  refSec := 19440, refSecXla := some 5100, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 2986, probeXla := "mobilenetv2-verified-adam", epochs := 80, stepsPerEpoch := 295 },  -- XLA measured on the net with its 52 conv biases
    { chapter := "7  EfficientNet", family := "conv",  refSec := 22320, refSecXla := some 5640, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 3760, probeXla := "efficientnet-verified-adam", epochs := 80, stepsPerEpoch := 295 },
    { chapter := "8  ConvNeXt",     family := "conv",  refSec := 47880, refSecXla := some 6841, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 8080, probeXla := "convnext-verified-adam", epochs := 80, stepsPerEpoch := 295 },  -- the channel-LN net
    { chapter := "9  ViT",          family := "attn",  refSec := 27966, refSecXla := some 3491, tier := "imagenette",
      refSecCuda := some 2560, probeXla := "vit-verified-adam", epochs := 80, stepsPerEpoch := 295 } ]  -- IREE warm steady-state ms/step × 295 × 80 | XLA the measured 80-epoch wall

/-- **This chapter's reference wall-clock, for the column in play.** Three columns, not two:
    IREE, XLA-on-ROCm (7900 XTX) and XLA-on-CUDA (4060 Ti). The vendor split exists because the
    per-chapter cross-vendor ratio spans a wide range and no single probe factor fits it — see
    `probeDenseRefMsCuda`. -/
def BenchItem.refFor (it : BenchItem) (col : String) : Option Nat :=
  if col == "iree" then some it.refSec
  else if col == "cuda" then it.refSecCuda
  else it.refSecXla

/-- Steady-state ms/epoch on the reference 7900 XTX for the two anchors, measured by
    the synthetic-input probe (`LEAN_MLIR_BENCH_SYNTH`): one constant batch reused at
    the dataset's real step count, eval skipped — so the on-reference factor reads about one
    and no dataset download is needed. -/
def probeDenseRefMs : Nat := 3030   -- mnist-mlp-verified  (784→512→512→10)
def probeConvRefMs  : Nat := 8020   -- cifar8-bn-verified  (8-conv + BN, 512 head)
/-- ms/STEP on the reference 7900 XTX for the `attn` anchor — synthetic-input probe of
    vit-verified-adam, reported as the median of a 100-step window (robust to the
    cold-cache / GC-blip outliers that make a 40-step mean swing).
    Step-based, not per-epoch: a ViT epoch is too slow to probe, and ViT's
    matmul/attention cost scales unlike conv across GPUs — so transformers get their
    own factor. (The shorter ViT figure elsewhere is the JAX bf16 path, not this
    verified-IREE trainer.) -/
def probeAttnRefMs : Nat := 1173

/-- The XLA/PJRT anchors, measured on the same reference 7900 XTX, with the
    same synthetic-input probe and the same "last of 3 epochs" steady-state rule as the
    IREE ones above — the only difference is which `.so` the probe binary linked. Against
    the IREE anchors the dense and conv probes read very different ratios, which is the whole
    reason a shared reference column would be wrong.

    **The attn anchor: `vit-verified-adam` runs on this box, and at bs32 it needs no
    workaround.** The graph can die at *execution* in the patch-embed weight-gradient convolution
    (a fused interior-dilated pad+conv selects MIOpen's no-workspace `GemmFwdRest` solver,
    whose `MIOpenIm2d2Col.cpp` fails to build under HIPRTC — it uses the OpenCL builtin
    `get_global_id`; see `historical/upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`).

    **It is BATCH-DEPENDENT, and this anchor is the bs32 number.** Measured:
    * **bs32** — the fault fired on the session's FIRST ViT/XLA execution and then never again
      across 11 runs, *including the byte-identical invocation that had just failed*. So here
      `MIOPEN_DEBUG_CONV_GEMM=0` is **not needed**, and setting it makes the attn probe slower.
      Why it fired once is unexplained —
      the MIOpen on-disk cache shows no writes in that window, so cache population is not it.
    * **bs64** — the fault fires **reliably** and the variable is **REQUIRED** (see
      `vit_adamdp64_train_step` in `ViTRender.lean`).

    The mechanism, from these logs plus `historical/upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`:
    XLA fuses the interior-dilated `pad` into the patch-embed weight-gradient convolution as
    `rhs_dilation = 16` rather than materialising a 209×209 filter, and requests that conv with a
    **zero-byte workspace** (`provided ptr: 0 size: 0`). That confines MIOpen to no-workspace
    solvers; the one it lands on is `GemmFwdRest`, whose kernel source `MIOpenIm2d2Col.cpp` uses
    the **OpenCL** builtins `get_global_id`/`get_global_size` while MIOpen JIT-compiles it through
    **HIPRTC**, where those identifiers do not exist ⇒ code-object build failure ⇒
    `miopenStatusUnknownError`. A MIOpen packaging bug, not anything about the graph. The im2col
    workspace it wanted is **linear in batch** — 6,422,528 bytes at bs32, exactly **2×** that
    (12,845,056) at bs64 — which is why the batch moves the outcome. The variable removes that
    solver family from consideration, so the broken kernel is never built.

    The render is gated independently of any of this: the first three step losses agree with
    the IREE peer to **3e-6** from identical fresh init, and `vit-dp-check` passes bit-exact
    on all 16,579,041 floats against a control that fires at 0.996.

    All three are **medians over repeated runs, not single samples**, because the conv probe has
    real run-to-run spread on this card: repeated runs of the same binary scatter over a band
    with no pattern, so a single sample can read below one against its own anchor and look
    like a regression when nothing changed. The dense probe is much steadier. Read an
    on-reference factor near one as agreement, not as signal — and if you re-anchor, use a median of several runs, not the
    one number in front of you. -/
def probeDenseRefMsXla : Nat := 610    -- mnist-mlp-verified (XLA) (vs 3030 on IREE); median of 8
def probeConvRefMsXla  : Nat := 3650   -- cifar8-bn-verified (XLA) (vs 8020 on IREE); median of 10
/-- ms/STEP, `vit-verified-adam`, median of 8 in the DEFAULT configuration, i.e. with no
    MIOpen override. Against `probeAttnRefMs` that is the largest cross-lowerer gap of the three
    families and the reason ViT cannot share the conv factor. (`MIOPEN_DEBUG_CONV_GEMM=0` is a
    regression here, not a fix; see the note above.) -/
def probeAttnRefMsXla : Nat := 128

-- ═══ THE CUDA REFERENCE COLUMN — RTX 4060 Ti, measured on an idle ares ═══
--
-- WHY A SECOND COLUMN RATHER THAN A BETTER FACTOR. The per-chapter 4060Ti/7900XTX ratio,
-- measured directly on all five Imagenette nets (each row's `refSecCuda` over its `refSecXla`),
-- ranges from well below one (MobileNetV2, EfficientNet, ViT) to well above it (ResNet-34,
-- ConvNeXt), and the three probe factors fall inside that range. **No single factor fits them**, so
-- cross-vendor scaling cannot be made accurate at any coefficient — the nets differ in how much of
-- a step is parameter transport (less for the conv probe than for R34), and the
-- two vendors differ in exactly that ratio. Measuring each vendor is the fix; scaling is the
-- fallback for a box with no datasets, not the answer.
def probeDenseRefMsCuda : Nat := 814    -- mnist-mlp-verified (XLA), idle, 3 real epochs
def probeConvRefMsCuda  : Nat := 2049   -- cifar8-bn-verified (XLA), idle, 3 real epochs
def probeAttnRefMsCuda  : Nat := 95     -- vit-verified-adam,   idle, MAX_STEPS=100

/-- One lowerer's complete probe configuration: which binaries to probe and which anchors
    to divide by. Bundling them is the point — `yourSecOf` takes a `BenchRef`, so a probe
    measured on one path cannot be divided by the other path's anchor, which is the
    mismatched-baseline trap made unrepresentable rather than merely documented. -/
structure BenchRef where
  /-- Lowerer name for the table header. -/
  lowerer    : String
  xla        : Bool
  /-- Which reference column to read: "iree" | "rocm" | "cuda". -/
  col        : String := "rocm"
  /-- The card that column was measured on, for the table header. -/
  card       : String := "7900 XTX"
  denseProbe : String
  convProbe  : String
  /-- `""` = this lowerer has no runnable attn probe. -/
  attnProbe  : String
  denseRefMs : Nat
  convRefMs  : Nat
  attnRefMs  : Nat

def ireeRef : BenchRef :=
  { lowerer := "IREE", xla := false, col := "iree", card := "7900 XTX"
    denseProbe := "mnist-mlp-verified", convProbe := "cifar8-bn-verified"
    attnProbe := "vit-verified-adam"
    denseRefMs := probeDenseRefMs, convRefMs := probeConvRefMs, attnRefMs := probeAttnRefMs }

/-- XLA on ROCm — scaled from the 7900 XTX column. -/
def xlaRefRocm : BenchRef :=
  { lowerer := "XLA/PJRT", xla := true, col := "rocm", card := "7900 XTX"
    denseProbe := "mnist-mlp-verified", convProbe := "cifar8-bn-verified"
    attnProbe := "vit-verified-adam"
    denseRefMs := probeDenseRefMsXla, convRefMs := probeConvRefMsXla
    attnRefMs := probeAttnRefMsXla }

/-- XLA on CUDA — the 4060 Ti column, measured rather than scaled. On that card every factor reads
    ~1.00, so the estimate is the measurement; on another NVIDIA card it scales from a same-vendor
    baseline, which the wide cross-vendor spread (see `probeDenseRefMsCuda`) says is the best a
    scaled number can do. -/
def xlaRefCuda : BenchRef :=
  { lowerer := "XLA/PJRT", xla := true, col := "cuda", card := "RTX 4060 Ti"
    denseProbe := "mnist-mlp-verified", convProbe := "cifar8-bn-verified"
    attnProbe := "vit-verified-adam"
    denseRefMs := probeDenseRefMsCuda, convRefMs := probeConvRefMsCuda
    attnRefMs := probeAttnRefMsCuda }

/-- Scale a chapter's reference seconds by the measured per-family factor, against `ref`'s
    OWN anchors. `aMs` is the attn ms/step probe (0 when there is no attn probe or it
    failed → attn falls back to the conv factor). `none` when this chapter has no reference
    on `ref`'s lowerer.

    **The conv proxy for a transformer measured several times LOW**, which is why the attn family
    exists at all: on a 4060 Ti under IREE the attn factor came out far above the dense and conv
    ones, so scaling ViT as conv badly underestimated a measured ViT run.
    That is the 7900 XTX → 4060 Ti (ROCm → CUDA) divergence specifically, not a constant —
    on the reference card the proxy is harmless because every factor reads about one. An
    independent instance of the same principle, measured on one card across lowerers: XLA-vs-IREE
    is a far larger gap for attn than for conv. Attention and convolution do not track each other,
    whether you change the GPU or the lowerer.

    **The low proxy is an IREE fact, not a 4060 Ti fact.** Measured on
    ares (CUDA 12.9, jax 0.10.2 CUDA PJRT plugin), the SAME card on the XLA path BEATS the
    reference 7900 XTX on both conv and attn, and the conv proxy for attn would have been only
    mildly off — a statement about IREE's CUDA backend rather than about the GPU. Consequence for
    anyone re-anchoring: the attn family is still worth keeping (it costs one probe and it is what
    makes ViT honest on IREE), but on XLA/CUDA a `*proxy` row is a mild approximation, not the trap
    it is on IREE. -/
def yourSecOf (ref : BenchRef) (it : BenchItem) (dMs cMs aMs : Nat) : Option Nat :=
  (it.refFor ref.col).map fun refSec =>
    if it.family == "dense" then refSec * dMs / ref.denseRefMs
    else if it.family == "attn" then
      if aMs == 0 then refSec * cMs / ref.convRefMs            -- fallback: conv proxy
      else refSec * aMs / ref.attnRefMs
    else refSec * cMs / ref.convRefMs

/-- **The estimate**, for the reason on `BenchItem.transportSensitive`.

    A chapter whose bottleneck mix its probe does not share has two candidate factors — its own
    family's (compute-leaning) and the dense probe's (transport-leaning). This returns the LOWER
    of the two; everything else is just `yourSecOf`.

    **On the reference card the two factors both read ~1.0, so they coincide and the published
    column is unchanged.** They separate only on a card whose transport:compute ratio differs from
    the reference's. The low end is optimistic when they do: measured on ares, ch5's
    real run landed near the HIGH one (`runs/r34_pool3s2_80ep_aug04.log`).

    The single printed number is deliberate (see the note at the print site). -/
def yourSecLo (ref : BenchRef) (it : BenchItem) (dMs cMs aMs : Nat) : Option Nat :=
  (yourSecOf ref it dMs cMs aMs).map fun base =>
    if it.transportSensitive then
      Nat.min base ((it.refFor ref.col).getD 0 * dMs / ref.denseRefMs)
    else base


/-- Human duration from whole seconds: `45s` / `12m` / `9.5h`. -/
def fmtDur (sec : Nat) : String :=
  if sec < 90 then s!"{sec}s"
  else if sec < 5400 then s!"{(sec + 30) / 60}m"
  else let t := (sec * 10 + 1800) / 3600; s!"{t / 10}.{t % 10}h"



/-- `num/den` as a 2-decimal multiplier string, e.g. `1.24`. -/
def fmtFactor (num den : Nat) : String :=
  if den == 0 then "?" else
  let h := num * 100 / den
  s!"{h / 100}.{(h % 100) / 10}{h % 10}"

/-- Right-pad to a fixed width for column alignment. -/
def padR (s : String) (n : Nat) : String :=
  if s.length < n then s ++ String.ofList (List.replicate (n - s.length) ' ') else s

/-- Pull the steady-state (last) `(<n>ms)` epoch timing out of a trainer's stdout. -/
def lastEpochMs (out : String) : Option Nat :=
  let eps := (out.splitOn "\n").filter fun l =>
    l.contains "epoch " && l.contains "ms)"
  match eps.getLast? with
  | none => none
  | some line =>
    match (line.splitOn "(").getLast? with
    | none => none
    | some s => match s.splitOn "ms)" with
                | h :: _ => h.toNat?
                | []     => none

/-- Pull `<n> ms/step` out of a `PROBE:` line (the LEAN_MLIR_MAX_STEPS path). -/
def probeMsStep (out : String) : Option Nat :=
  match ((out.splitOn "\n").filter fun l => l.contains "PROBE:").getLast? with
  | none => none
  | some line => match (line.splitOn "PROBE: ").getLast? with
    | none => none
    | some s => match s.splitOn " ms/step" with
                | h :: _ => h.toNat?
                | []     => none

/-- First run of consecutive digits in `s` as a Nat (tolerates surrounding text). -/
def firstNat (s : String) : Option Nat :=
  let ds := (s.toList.dropWhile (fun c => !c.isDigit)).takeWhile (·.isDigit)
  if ds.isEmpty then none else (String.ofList ds).toNat?

/-- Best-effort GPU utilization % (rocm-smi / nvidia-smi); none if the tool is absent. -/
def gpuBusyPct (backend : String) : IO (Option Nat) := do
  try
    if backend == "cuda" then
      let o ← IO.Process.output { cmd := "nvidia-smi", args := #["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"] }
      pure (firstNat o.stdout)
    else
      let o ← IO.Process.output { cmd := "rocm-smi", args := #["--showuse"] }
      pure (((o.stdout.splitOn "\n").find? (fun l => l.contains "use (%)")).bind firstNat)
  catch _ => pure none

/-- **Direct mode's probe: measure the chapter's OWN trainer on THIS box, 3 real epochs.**

    `runProbe` answers *"how does my card compare to a 7900 XTX"* and then multiplies a reference.
    That is exact for a chapter which IS its own probe (ch1/2 are the dense probe, ch4 the conv one,
    ch9 the attn one) and an extrapolation for every other. This answers the other question
    directly: run the chapter's real trainer and
    multiply by its own epoch count. No reference card, no hardware factor, no bottleneck
    assumption. The method is validated on ch9, whose marginal-epoch extrapolation lands within a
    fraction of a percent of its real 80-epoch wall (see `benchTable`).

    **REAL data, not `LEAN_MLIR_BENCH_SYNTH`** — deliberately. The synthetic path exists to take
    the loader out of a *comparison*; here the loader is part of the answer. Excluding it is most
    of why the transport-bound estimate came in low against the measured ResNet-34 run.

    **AND IT MUST NOT TOUCH A CHECKPOINT**, in either direction. `trainAdamSched` checkpoints per
    epoch, so a naive 3-epoch probe would (a) RESUME an existing checkpoint — measuring a warm
    restart, or nothing at all if the marker is at the epoch budget (a silent no-op) — and
    (b) leave its own marker at 3 for the next real run to resume from. The caller stashes and
    restores; this function only runs. -/
def runDirectProbe (it : BenchItem) (backend gpu : String)
    (runEnv : Array (String × Option String)) : IO (Option Nat) := do
  IO.println s!"\n  ▸ measuring {it.chapter} directly — {it.probeXla}, 3 real epochs…"
  let bp ← IO.Process.spawn { cmd := "lake", args := #["build", it.probeXla] }
  if (← bp.wait) != 0 then
    IO.eprintln s!"    build failed: {it.probeXla}"
    return none
  let vis := if backend == "cuda" then "CUDA_VISIBLE_DEVICES" else "HIP_VISIBLE_DEVICES"
  let stepMode := it.stepsPerEpoch > 0
  -- MAX_STEPS warms 8 and times 9..n, so anything ≤ 8 fires NOTHING and caps nothing.
  let capEnv := if stepMode then #[("LEAN_MLIR_MAX_STEPS", some "40")]
                            else #[("LEAN_MLIR_MAX_EPOCHS", some "3")]
  let env := runEnv ++ capEnv ++ #[("IREE_BACKEND", some backend), (vis, some gpu)]
  let o ← IO.Process.output { cmd := s!".lake/build/bin/{it.probeXla}", args := #["data"], env := env }
  match (if stepMode then probeMsStep o.stdout else lastEpochMs o.stdout) with
  | none =>
      IO.eprintln s!"    no timing for {it.probeXla} (dataset present? exit {o.exitCode})"
      pure none
  | some ms =>
      if stepMode then
        let total := ms * it.stepsPerEpoch * it.epochs / 1000
        IO.println s!"    {ms} ms/step × {it.stepsPerEpoch} steps × {it.epochs} ep = {fmtDur total}   (measured here; TRAIN-ONLY)"
        pure (some total)
      else
        let total := ms * it.epochs / 1000
        IO.println s!"    {ms} ms/epoch × {it.epochs} ep = {fmtDur total}   (measured here, incl. eval)"
        pure (some total)

/-- Build + run one probe net and return its steady-state timing. With `stepProbe`
    set (`attn` anchor) it caps at N steps and reads ms/step; otherwise it runs 3
    epochs and reads ms/epoch. -/
def runProbe (bin family : String) (refMs : Nat) (card backend gpu : String)
    (runEnv : Array (String × Option String)) (stepProbe : Option Nat := none) : IO (Option Nat) := do
  let what := match stepProbe with | some n => s!"{n} steps" | none => "3 epochs"
  IO.println s!"\n  ▸ probing {bin} ({family}) — build + {what}…"
  let bp ← IO.Process.spawn { cmd := "lake", args := #["build", bin] }
  if (← bp.wait) != 0 then
    IO.eprintln s!"    build failed: {bin}"
    return none
  let vis := if backend == "cuda" then "CUDA_VISIBLE_DEVICES" else "HIP_VISIBLE_DEVICES"
  let capEnv := match stepProbe with
    | some n => #[("LEAN_MLIR_MAX_STEPS", some (toString n))]
    | none   => #[("LEAN_MLIR_MAX_EPOCHS", some "3")]
  let env := runEnv ++ capEnv ++ #[("IREE_BACKEND", some backend), (vis, some gpu),
                                    ("LEAN_MLIR_BENCH_SYNTH", some "1")]
  let o ← IO.Process.output { cmd := s!".lake/build/bin/{bin}", args := #["data"], env := env }
  let parsed := match stepProbe with | some _ => probeMsStep o.stdout | none => lastEpochMs o.stdout
  match parsed with
  | none =>
      IO.eprintln s!"    no timing for {bin} (data present? exit {o.exitCode})"
      pure none
  | some ms =>
      let unit := match stepProbe with | some _ => "ms/step" | none => "ms/epoch"
      IO.println s!"    {ms} {unit}   [ref {refMs}]   → {fmtFactor ms refMs}× the {card}"
      pure (some ms)

/-- The shared body of `lake run benchmark` and `lake run benchmark-xla`. One printer, two
    `BenchRef`s, so the two commands cannot drift on the probe recipe, the steady-state rule
    or the table layout — and, more to the point, so neither can end up dividing by the other
    lowerer's anchors.

    Rows with no reference on this lowerer print `n/a` and are excluded from the totals and
    from the tier subtotals; the footer says how many chapters were covered, so a short total
    can never read as a whole-Part-1 number. -/
def runBenchmark (ref : BenchRef) : IO UInt32 := do
  let backend ← match ← IO.getEnv "IREE_BACKEND" with
    | some b => pure b
    | none   => detectBackend
  let gpu := (← IO.getEnv "LEAN_DEMO_GPU").getD "0"
  -- The XLA binaries compile in-process through PJRT, so unlike the IREE ones they need
  -- neither the venv on PATH nor `iree-compile` — but they DO need the shim built and a
  -- resolvable plugin, the same two guards `runDemoGroup (xla := true)` applies.
  let runEnv ← if ref.xla then do
      IO.println "━━━ XLA/PJRT backend ━━━"
      if !(← ensurePjrtShim) then return 1
      notePjrtPlugin
      pure #[]
    else do
      let venvBin := (← IO.currentDir) / ".venv" / "bin"
      if ← System.FilePath.pathExists (venvBin / "iree-compile") then
        pure #[("PATH", some s!"{venvBin}:{(← IO.getEnv "PATH").getD ""}")]
      else pure #[]
  let cmdName := if ref.xla then "benchmark-xla" else "benchmark"
  -- DIRECT MODE: measure every chapter's own trainer here instead of scaling a foreign
  --   reference. See `runDirectProbe`. XLA only — the IREE path has no in-process compile, so a
  --   3-epoch probe would pay minutes of iree-compile per net.
  let direct := ref.xla && ((← IO.getEnv "BENCH_DIRECT").getD "" != "")
  IO.println s!"━━━ lake run {cmdName} ━━━ verified-NN training throughput on your GPU"
  IO.println s!"  lowerer: {ref.lowerer}   backend: {backend}   gpu: {gpu}   (synthetic-input probes — no dataset needed)"
  -- Pre-flight: a busy GPU inflates every probe. Warn if the card isn't idle.
  match ← gpuBusyPct backend with
  | some u => IO.println (if u > 20 then
      s!"  ⚠ GPU is {u}% busy — close other GPU jobs first or the estimate will inflate."
      else s!"  GPU idle ({u}%).")
  | none   => pure ()
  -- Synthetic input (LEAN_MLIR_BENCH_SYNTH, set in runProbe): one constant batch reused
  -- at the dataset's real step count, so no MNIST/CIFAR/Imagenette download is required.
  if direct then
    -- A 3-epoch probe on a checkpointing trainer is DESTRUCTIVE IN BOTH DIRECTIONS: it
    --   resumes whatever marker is on disk — measuring a warm restart, or NOTHING if the marker is
    --   at the epoch budget, which exits `done` with rc=0 and no timing — and it leaves its own
    --   marker at 3 for the next real run to resume from. So: refuse if a trainer is live, stash
    --   every XLA checkpoint, measure, restore. Stash rather than delete, because these are
    --   someone's training state.
    let ps ← IO.Process.output { cmd := "bash", args := #["-c", "ps -eo comm | grep -c verified || true"] }
    if (ps.stdout.trimAscii.toNat?.getD 0) > 0 then
      IO.eprintln "  ⛔ a *-verified* trainer is RUNNING. Direct mode stashes checkpoints and would \
corrupt it (and contend for the GPU). Stop it first."
      return 1
    let stash := ".lake/build/benchdirect-stash"
    IO.println s!"\n  ▶ DIRECT MODE — measuring each chapter's own trainer on THIS box (3 real epochs each)."
    IO.println s!"    checkpoints stashed to {stash}/ and restored afterwards (§4: a probe must not resume or leave one)."
    _ ← IO.Process.output { cmd := "bash", args := #["-c",
      s!"mkdir -p {stash} && mv .lake/build/*_ckpt_xla.bin* {stash}/ 2>/dev/null; true"] }
    let mut rows : Array (BenchItem × Option Nat) := #[]
    for it in benchTable do
      if it.probeXla.isEmpty || it.epochs == 0 then
        rows := rows.push (it, none)
      else
        rows := rows.push (it, ← runDirectProbe it backend gpu runEnv)
    -- the probes' own markers go; the stash comes back
    _ ← IO.Process.output { cmd := "bash", args := #["-c",
      s!"rm -f .lake/build/*_ckpt_xla.bin*; mv {stash}/* .lake/build/ 2>/dev/null; rmdir {stash} 2>/dev/null; true"] }
    IO.println s!"\n  MEASURED training time on THIS box ({ref.lowerer}, real data, current certified bytes):\n"
    let rule := "  " ++ String.ofList (List.replicate 47 '-')
    IO.println s!"  {padR "Chapter" 18}{padR "family" 8}{padR s!"ref({ref.card})" 18}measured here"
    IO.println rule
    let mut tot := 0
    for (it, m) in rows do
      let refCol := match it.refFor ref.col with | some r => fmtDur r | none => "n/a"
      match m with
      | some sec => tot := tot + sec
                    IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR refCol 18}{fmtDur sec}"
      | none     => IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR refCol 18}— (probe failed)"
    IO.println rule
    IO.println s!"  {padR "Full Part-1 training" 30}{padR "" 18}{fmtDur tot}"
    IO.println "\n  * every number is 3 steady-state epochs of that chapter's OWN trainer × its own"
    IO.println "    epoch count — no reference card, no hardware factor, no bottleneck assumption."
    IO.println "    §2j validated the method at 0.3% (ch9 extrapolated 3480s; real run 3491s)."
    IO.println "  * real data, so the ~6.3% per-epoch loader term (§2e-ter) is included — unlike the"
    IO.println "    scaled mode, whose probes are synthetic by design."
    IO.println "  * current certified bytes, so a re-render cannot silently invalidate it."
    IO.println "  * ⚠ the five Imagenette rows are TRAIN-ONLY. Their trainers (trainAdamSched) print no"
    IO.println "    per-epoch ms, so they are measured as median ms/step × 295 steps × 80 ep via the"
    IO.println "    MAX_STEPS probe — which returns before the eval pass. Eval adds ~5% on R34 and"
    IO.println "    ~8% on ConvNeXt (§2h). Add that back before comparing to a wall clock."
    IO.println "  * residency is on by default — parameters stay on the device (§2d.3: 2.03× on"
    IO.println "    ResNet-34 bs32). PJRT_FFI_RESIDENT=0 measures the copying path."
    return 0
  let denseMs ← runProbe ref.denseProbe "dense" ref.denseRefMs ref.card backend gpu runEnv
  let convMs  ← runProbe ref.convProbe  "conv"  ref.convRefMs  ref.card backend gpu runEnv
  let attnMs ← if ref.attnProbe.isEmpty then do
      IO.println s!"\n  ▸ attn probe SKIPPED — no runnable ViT probe on {ref.lowerer}. \
ch.9 has no {ref.lowerer} reference and prints n/a."
      pure none
    else runProbe ref.attnProbe "attn" ref.attnRefMs ref.card backend gpu runEnv (stepProbe := some 100)
  match denseMs, convMs with
  | some dMs, some cMs =>
    let aMs := attnMs.getD 0
    IO.println s!"\n  ESTIMATED training time on YOUR gpu  (ref = {ref.card}, {ref.lowerer}):\n"
    let rule := "  " ++ String.ofList (List.replicate 47 '-')
    IO.println s!"  {padR "Chapter" 18}{padR "family" 8}{padR s!"ref({ref.card})" 18}your gpu"
    IO.println rule
    let mut yourLoTotal := 0
    let mut refTotal := 0
    let mut covered := 0
    for it in benchTable do
      match it.refFor ref.col, yourSecLo ref it dMs cMs aMs with
      | some refSec, some lo =>
        yourLoTotal := yourLoTotal + lo
        refTotal := refTotal + refSec
        covered := covered + 1
        let flag := if it.family == "attn" && aMs == 0 then " *proxy" else ""
        -- Single number, deliberately. A bracket wide enough to hold the real cross-vendor
        --   spread (0.585-1.411, measured) would tell a first-time user nothing they could plan
        --   with — and with a per-vendor reference column the on-vendor factors read ~1.00 anyway,
        --   so the honest number is simply the measured one. `yourSecOf` is the estimate.
        IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR (fmtDur refSec) 18}{fmtDur lo}{flag}"
      | _, _ =>
        IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR "n/a" 18}n/a  (no {ref.lowerer} reference)"
    IO.println rule
    let totalLabel := if covered == benchTable.length then "Full Part-1 training"
                      else s!"Part-1 training ({covered} of {benchTable.length} ch.)"
    IO.println s!"  {padR totalLabel 30}{padR (fmtDur refTotal) 18}{fmtDur yourLoTotal}"
    IO.println "\n  `lake run` tiers on your gpu (training time):"
    for (tier, label) in [("mnist", "lake run mnist-iree"), ("cifar", "lake run cifar-iree"),
                          ("imagenette", "lake run imagenette-iree")] do
      let items := benchTable.filter (·.tier == tier)
      let refS := (items.filterMap (·.refFor ref.col)).foldl (· + ·) 0
      let yourLo := (items.filterMap (fun it => yourSecLo ref it dMs cMs aMs)).foldl (· + ·) 0
      let miss := items.length - (items.filterMap (·.refFor ref.col)).length
      let note := if miss == 0 then "" else s!"   ({miss} ch. n/a)"
      -- All three demo groups have an `-xla` peer, so name the command the user would
      -- actually run on this path rather than its IREE sibling.
      let suffix := if ref.xla then "-xla" else ""
      IO.println s!"    {padR (label ++ suffix) 26}{padR (fmtDur refS) 9}→  {fmtDur yourLo}{note}"
    IO.println s!"\n  * probes: {ref.denseProbe} / {ref.convProbe}\
{if ref.attnProbe.isEmpty then " / (no attn probe)" else s!" / {ref.attnProbe}"}"
    IO.println s!"  * ref column measured on {ref.card}; on that card the factors read ~1.00 and"
    IO.println "    the estimate is the measurement. On other cards it is scaled — accurate within a"
    IO.println "    vendor, approximate across one (the cross-vendor per-chapter spread is 2.4×)."
    IO.println "  * Imagenette rows are TRAIN-ONLY; eval adds ~5-10%. Training time only."
    if ref.xla then
      IO.println "  * BENCH_DIRECT=1 measures YOUR box instead of scaling (needs the datasets, ~5 min)."
    else
      IO.println "  * first run adds ~10-15 min/arch IREE compile."
    return 0
  | _, _ =>
    if ref.xla then
      IO.eprintln "\n  probe failed — check that ffi/libpjrt_ffi.so built and $PJRT_PLUGIN resolves"
      IO.eprintln "  (reported above). No estimate produced."
    else
      IO.eprintln "\n  probe failed — need data (`lake run download`) and the IREE venv from"
      IO.eprintln "  Track 2. No estimate produced."
    return 1

/-- `lake run benchmark` — probe this GPU on the **IREE** path, print a per-chapter
    training-time estimate. The XLA peer is `lake run benchmark-xla`; the two scale from
    separate reference columns measured on separate lowerers and are not interchangeable. -/
script benchmark do
  runBenchmark ireeRef

/-- `lake run benchmark-xla` — the XLA/PJRT peer of `lake run benchmark`.

    Same probe recipe, same steady-state rule, same printer — the only difference is which
    lowerer the probe binaries linked and which reference column they scale from. Both of those move
    together inside `xlaRef`, which is what stops the mismatched-baseline trap: an XLA probe divided
    by IREE's anchors would report Part-1 training ~2-5× too fast, silently.

    **All 3 probes and all 9 chapters.** `vit-verified-adam` executes on this box with no
    workaround, though its MIOpen failure is non-deterministic rather than fixed — see
    `probeAttnRefMsXla`. The `n/a` machinery is retained on purpose: it is what would keep an
    unmeasured row honest.

    All nine XLA references are measured; ch.9's is a real 80-epoch run, which confirms the
    marginal-epoch extrapolation. See `benchTable`.

    Needs no venv: the XLA binaries compile in-process rather than shelling out to
    `iree-compile`. It does build `ffi/libpjrt_ffi.so` if missing/stale and report whether
    `$PJRT_PLUGIN` resolves, exactly as `lake run {mnist,cifar,imagenette}-xla` do. -/
script «benchmark-xla» do
  -- ONE PATH PER VENDOR. The reference column is measured on that vendor's card, so on a
  --   4060 Ti (CUDA) or a 7900 XTX (ROCm) the factors read ~1.00 and the table is the measurement.
  let be ← match ← IO.getEnv "IREE_BACKEND" with | some b => pure b | none => detectBackend
  runBenchmark (if be == "cuda" then xlaRefCuda else xlaRefRocm)
