import Lake
open Lake DSL

package «lean4-mlir» where
  version := v!"0.6.0"
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
  "https://github.com/leanprover/doc-gen4" @ "main"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"

-- The everything-root: codegen + IREE FFI + trainers' shared modules +
-- the proof suite, all pulled in transitively via `LeanMlir.lean`.
-- `lake build LeanMlir` type-checks the whole repo.
lean_lib «LeanMlir» where
  roots := #[`LeanMlir]

-- Scoped targets, so CI and contributors can build one slice without the
-- rest. `LeanMlir` above is all-or-nothing; these split it along the same
-- seam the codebase already has (the proof suite imports only Mathlib —
-- never the codegen — so `Proofs` skips Types/Spec/MlirCodegen/Train).

/-- **`lake build Proofs`** — the VJP proof suite only. Roots are the apex
    modules (same set `LeanMlir.lean` aggregates); their transitive imports
    cover every proof file. Default target: a bare `lake build` now
    type-checks the formalization instead of doing nothing. -/
@[default_target]
lean_lib «Proofs» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Attention, `LeanMlir.Proofs.CNN,
             `LeanMlir.Proofs.Depthwise, `LeanMlir.Proofs.MobileNetV2,
             `LeanMlir.Proofs.ConvNeXt, `LeanMlir.Proofs.EfficientNet,
             -- Chapter-4 MNIST 2D CNN (no BN): conditional whole-net VJP
             -- + a concrete instance with every smoothness hyp discharged.
             `LeanMlir.Proofs.MnistCNN,
             -- Chapter-6 ResNet Milestone B: stride-2 SAME convolution (the hard
             -- new downsampling op) = decimate ∘ stride-1 conv, with its input-VJP.
             `LeanMlir.Proofs.StridedConv,
             -- Chapter-6 ResNet Milestone B: the deep-block chain (a list of
             -- same-type residual blocks composes to one VJP) — 16-block depth.
             `LeanMlir.Proofs.ResNet34,
             -- Chapter-6 ResNet Milestone B8: per-channel BatchNorm (block-diagonal
             -- VJP via a per-row generalization of `rowwise_has_vjp_mat`).
             `LeanMlir.Proofs.PerChannelBN,
             -- opt-in Mathlib.Matrix interop; not imported by the suite,
             -- listed here so CI keeps it green.
             `LeanMlir.Proofs.MatBridge,
             -- denoted StableHLO-subset IR (Phase 0a/0b spike); bridges the
             -- emitted backward graph to the proven HasVJP.backward.
             `LeanMlir.Proofs.IR,
             -- R4 printer-faithfulness Stage A (ch 2): StableHLO-subset AST +
             -- denotation `den` proven to match the linear train-step math.
             `LeanMlir.Proofs.StableHLO,
             -- R4 syntactic core: op-graph serialization round-trip
             -- (parse (toToks (skel a)) = a).
             `LeanMlir.Proofs.StableHLOParse,
             -- M1 (planning/verified_train_step.md): the linear train step bundled
             -- into one SGD-on-certified-softmax-CE-gradient theorem.
             `LeanMlir.Proofs.LinearTrainStep,
             -- M2: the MLP per-layer parameter-gradient assembly (Crux A).
             `LeanMlir.Proofs.MlpTrainStep,
             -- M3: the CNN convolution parameter-gradient bridges.
             `LeanMlir.Proofs.CnnTrainStep,
             -- MLP render half: the train-step text as a name-threaded render of the
             -- proven forward graphs (multi-intermediate generalization).
             `LeanMlir.Proofs.MlpRender,
             -- CNN render half: the CNN train-step text rendered from `cnnFwdGraph`,
             -- with flat→NCHW reshape glue bridging the conv param-grad tail.
             `LeanMlir.Proofs.CnnRender,
             -- CIFAR-BN close: the per-channel BN scale/shift (dγ, dβ) param-grad
             -- bridges — the affine BN analogue of `bias_grad_bridge`.
             `LeanMlir.Proofs.CifarBnClose,
             -- CNN conv-close upgrade: the conv param closes pinned to the actual
             -- backward-chain cotangent (Back3 maxpool/conv via flatDenote + relu masks).
             `LeanMlir.Proofs.CnnChainClose,
             -- MobileNetV2 close (Item C): the depthwise (stride-1/2) + strided-conv
             -- parameter-gradient bridges — every MobileNetV2 train-step param output
             -- certified θ − lr·(certified Jacobian · cotangent).
             `LeanMlir.Proofs.MobileNetV2Close,
             -- MobileNetV2 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph
             -- (matches the operational render's BN flavor) + faithfulness to the
             -- per-channel ℝ-forward. Prerequisite for the structured render (Item B).
             `LeanMlir.Proofs.MobileNetV2RenderPC,
             -- MobileNetV2 cotangent-chain close (Item D): the Item C conv/depthwise bridges
             -- pinned to the inverted-residual backward chain (relu6 kink + depthwise + stride-2).
             `LeanMlir.Proofs.MobileNetV2ChainClose,
             -- The cotangent pass / = ∂loss/∂θ fold: the certified per-layer conv/depthwise
             -- Jacobian contracted with ∂loss/∂(layer output) IS the total loss gradient (pdiv_comp
             -- at a smooth point). The conv analogue of mlp_hidden_total_loss_grad; program-wide.
             `LeanMlir.Proofs.ConvLossFold,
             -- EfficientNet-B0 close (Item C): a FREE close — every param family reuses an
             -- existing bridge (5×5 depthwise pinned; batch-norm γ/β = per-channel BN at m=N·h·w;
             -- SE squeeze/excite are dense → M2). No new VJP.
             `LeanMlir.Proofs.EfficientNetClose,
             -- ResNet-34 close (Item C): a FREE close — every r34 param family certified
             -- by an existing bridge (the 7×7 stem + 3×3 strided projection pinned to the
             -- generic strided conv W/b bridges; no new VJP).
             `LeanMlir.Proofs.ResNet34Close,
             -- ResNet-34 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph (full
             -- 16-block [3,4,6,3] net, 7×7 stem, maxpool) + per-block + whole-net faithfulness.
             `LeanMlir.Proofs.ResNet34RenderPC,
             -- ResNet-34 cotangent-chain close (Item D): the Item C conv bridges pinned to the
             -- cotangent the backward chain delivers (id/downsample block + maxpool-back stem).
             `LeanMlir.Proofs.ResNet34ChainClose]

/-- **`lake build Codegen`** — the Lean→MLIR codegen + spec core, no proofs.
    The half that actually emits StableHLO and runs on device. -/
lean_lib «Codegen» where
  srcDir := "."
  roots := #[`LeanMlir.MlirCodegen, `LeanMlir.Train, `LeanMlir.Spec,
             `LeanMlir.SpecHelpers, `LeanMlir.Types, `LeanMlir.IreeRuntime,
             `LeanMlir.Ddpm, `LeanMlir.Cam, `LeanMlir.F32Array]

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

extern_lib libireeffi pkg := do
  let shimO ← fetch <| pkg.target ``ireeLeanFfiO
  let f32O  ← fetch <| pkg.target ``f32HelpersO
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "ireeffi") #[shimO, f32O]

-- ═══════════════════════════════════════════════════════════════════
-- Phase 3 trainers (Lean → MLIR → IREE → GPU)
-- ═══════════════════════════════════════════════════════════════════

private def ireeLink : Array String :=
  #["-L", "./ffi", "-liree_ffi", "-Wl,-rpath,./ffi", "-Wl,--allow-shlib-undefined"]

lean_exe «resnet34-train» where
  root := `MainResnetTrain
  moreLinkArgs := ireeLink

lean_exe «resnet50-train» where
  root := `MainResnet50Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v2-train» where
  root := `MainMobilenetV2Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v3-train» where
  root := `MainMobilenetV3Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v4-train» where
  root := `MainMobilenetV4Train
  moreLinkArgs := ireeLink

lean_exe «efficientnet-train» where
  root := `MainEfficientNetTrain
  moreLinkArgs := ireeLink

lean_exe «efficientnet-v2-train» where
  root := `MainEfficientNetV2Train
  moreLinkArgs := ireeLink

lean_exe «convnext-tiny-train» where
  root := `MainConvNeXtTrain
  moreLinkArgs := ireeLink

lean_exe «vit-tiny-train» where
  root := `MainVitTrain
  moreLinkArgs := ireeLink

lean_exe «ablation» where
  root := `MainAblation
  moreLinkArgs := ireeLink

lean_exe «vgg-train» where
  root := `MainVggTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-cnn-train» where
  root := `MainMnistCnnTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-bn-train» where
  root := `MainCifarCnnBnTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-mlp-train» where
  root := `MainMnistMlpTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-linear-train» where
  root := `MainMnistLinearTrain
  moreLinkArgs := ireeLink

-- Trains MNIST-linear on the VERIFIED-rendered StableHLO
-- (`verified_mlir/`, = Proofs.StableHLO.linearTrainStepModuleV) through the
-- real Lean/IREE FFI. See MainMnistLinearVerified.lean.
lean_exe «mnist-linear-verified» where
  root := `MainMnistLinearVerified
  moreLinkArgs := ireeLink

-- Chapter 3: trains the MNIST MLP on the VERIFIED-rendered StableHLO
-- (verified_mlir/mlp_train_step.mlir = Proofs.StableHLO.mlpTrainStepText).
lean_exe «mnist-mlp-verified» where
  root := `MainMnistMlpVerified
  moreLinkArgs := ireeLink

-- Chapter 4: trains the MNIST CNN on the VERIFIED-rendered StableHLO
-- (verified_mlir/cnn_train_step.mlir = Proofs.StableHLO.cnnTrainStepText).
lean_exe «mnist-cnn-verified» where
  root := `MainMnistCnnVerified
  moreLinkArgs := ireeLink

-- Chapter 5: trains the CIFAR-10 CNN (no BN) on the VERIFIED-rendered StableHLO
-- (verified_mlir/cifar_train_step.mlir = Proofs.StableHLO.cifarTrainStepText).
lean_exe «cifar-verified» where
  root := `MainCifarVerified
  moreLinkArgs := ireeLink

-- Chapter 5 (BatchNorm): trains the CIFAR-10 CNN + per-example BN on the
-- VERIFIED-rendered StableHLO (Proofs.StableHLO.cifarBnTrainStepText).
lean_exe «cifar-bn-verified» where
  root := `MainCifarBnVerified
  moreLinkArgs := ireeLink

-- ch6 B9: real ResNet-34 ([3,4,6,3], per-channel BN, strided downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestResnet34{Train,Fwd}.lean); 146 params.
lean_exe «resnet34-verified» where
  root := `MainResnet34Verified
  moreLinkArgs := ireeLink

-- ch7 C4: small MobileNetV2 (inverted-residual blocks: depthwise conv + relu6 +
-- per-channel BN) trained on VERIFIED-rendered StableHLO
-- (tests/TestMobilenetV2{Train,Fwd}.lean); 30 params.
lean_exe «mobilenetv2-verified» where
  root := `MainMobilenetV2Verified
  moreLinkArgs := ireeLink

-- ch8 E4/E5/E6: EfficientNet-B0 (faithful [t,c,n,s,k] config — 16 MBConv layers,
-- inverted-residual + squeeze-excite + swish + BATCH norm, 3×3/5×5 depthwise) trained
-- on VERIFIED-rendered StableHLO (tests/TestEfficientNet{Train,Fwd}.lean); 262 params.
lean_exe «efficientnet-verified» where
  root := `MainEfficientNetVerified
  moreLinkArgs := ireeLink

-- Chapter 9: ConvNeXt-T (Liu et al. 2022 — patchify stem + [3,3,9,3] depthwise-7×7
-- blocks with LN + GELU + layerScale + 3 between-stage downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestConvNeXt{Train,Fwd}.lean); 180 params.
lean_exe «convnext-verified» where
  root := `MainConvNeXtVerified
  moreLinkArgs := ireeLink

lean_exe «vit-verified» where
  root := `MainViTVerified
  moreLinkArgs := ireeLink

lean_exe «cifar-cnn-train» where
  root := `MainCifarCnnTrain
  moreLinkArgs := ireeLink

lean_exe «autoencoder-pets-train» where
  root := `demos.MainAutoencoderPetsTrain
  moreLinkArgs := ireeLink

lean_exe «unet-pets-train» where
  root := `demos.MainUnetPetsTrain
  moreLinkArgs := ireeLink

lean_exe «pets-predict» where
  root := `demos.MainPetsPredict
  moreLinkArgs := ireeLink

lean_exe «gradcam» where
  root := `demos.MainGradCAM
  moreLinkArgs := ireeLink

lean_exe «bigram-shakespeare» where
  root := `demos.MainBigramShakespeare
  moreLinkArgs := ireeLink

lean_exe «tinygpt-shakespeare» where
  root := `demos.MainTinyGptShakespeare
  moreLinkArgs := ireeLink

lean_exe «mnist-ddpm-train» where
  root := `demos.MainMnistDdpmTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-ddpm-sample» where
  root := `demos.MainMnistDdpmSample
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-train» where
  root := `demos.MainCifarDdpmTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-sample» where
  root := `demos.MainCifarDdpmSample
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-attn-train» where
  root := `demos.MainCifarDdpmAttnTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-attn-sample» where
  root := `demos.MainCifarDdpmAttnSample
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-sincos-train» where
  root := `demos.MainCifarDdpmSincosTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-sincos-sample» where
  root := `demos.MainCifarDdpmSincosSample
  moreLinkArgs := ireeLink

-- YOLOv1 Phase 2: Pascal VOC 2007 smoke trainer.
-- See planning/yolo_demo_v3.md Phase 2.
lean_exe «yolov1-voc-train» where
  root := `demos.MainYolov1VocTrain
  moreLinkArgs := ireeLink

-- YOLOv1 Phase 4: bootstrap from R34-Imagenette pretrained backbone.
lean_exe «yolov1-voc-train-bootstrap» where
  root := `demos.MainYolov1VocTrainBootstrap
  moreLinkArgs := ireeLink

-- YOLOv1 Phase 5: inference dump (logits + images + IDs) for Python viz.
lean_exe «yolov1-voc-infer» where
  root := `demos.MainYolov1VocInfer
  moreLinkArgs := ireeLink

-- ═══════════════════════════════════════════════════════════════════
-- VJP oracle — one binary per axiom under test.
-- Trainers live in tests/vjp_oracle/phase3/ so the root isn't crowded
-- with test-only files. See tests/vjp_oracle/README.md.
-- ═══════════════════════════════════════════════════════════════════

lean_exe «vjp-oracle-dense» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDense
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-dense-relu» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDenseRelu
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-conv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConv
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-convbn» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConvBn
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-conv-pool» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConvPool
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-residual» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleResidual
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-depthwise» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDepthwise
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-attention» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleAttention
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-mbconv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleMbConv
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-global-avg-pool» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleGlobalAvgPool
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-bottleneck» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleBottleneck
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-mbconv-v3» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleMbConvV3
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-fused-mbconv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleFusedMb
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-uib» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleUib
  moreLinkArgs := ireeLink

-- ═══════════════════════════════════════════════════════════════════
-- Tests + benchmarks
-- ═══════════════════════════════════════════════════════════════════

lean_exe «test-forward» where
  root := `tests.TestForward
  moreLinkArgs := ireeLink

lean_exe «test-iree» where
  root := `tests.TestIreeRuntime
  moreLinkArgs := ireeLink

lean_exe «test-train» where
  root := `tests.TestTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-iree-load» where
  root := `tests.TestIreeLoad
  moreLinkArgs := ireeLink

lean_exe «test-f32» where
  root := `tests.TestF32
  moreLinkArgs := ireeLink

lean_exe «bench-resnet» where
  root := `tests.BenchResnet
  moreLinkArgs := ireeLink

lean_exe «test-resnet-fwd» where
  root := `tests.TestResnetForward

lean_exe «test-unet-forward» where
  root := `tests.TestUnetForward

lean_exe «test-ddpm-train-emit» where
  root := `tests.TestDdpmTrainEmit
  moreLinkArgs := ireeLink

lean_exe «test-convnext-fwd-emit» where
  root := `tests.TestConvNextForwardEmit
  moreLinkArgs := ireeLink

lean_exe «test-convnext-train-emit» where
  root := `tests.TestConvNextTrainEmit
  moreLinkArgs := ireeLink

lean_exe «test-focal-emit» where
  root := `tests.TestFocalEmit
  moreLinkArgs := ireeLink

-- YOLOv1 Phase 1 tests (planning/yolo_demo_v2.md decisions D10-D11).
lean_exe «test-yolov1-emit» where
  root := `tests.TestYolov1Emit
  moreLinkArgs := ireeLink

lean_exe «test-yolov1-mutex» where
  root := `tests.TestYolov1Mutex
  moreLinkArgs := ireeLink

lean_exe «test-yolov1-train-step» where
  root := `tests.TestYolov1TrainStep
  moreLinkArgs := ireeLink

lean_exe «test-randaugment» where
  root := `tests.TestRandAugment
  moreLinkArgs := ireeLink

lean_exe «test-cam» where
  root := `tests.TestCam
  moreLinkArgs := ireeLink

lean_exe «inspect-convnext-bn» where
  root := `tests.InspectConvNeXtBN
  moreLinkArgs := ireeLink

lean_exe «inspect-convnext» where
  root := `demos.MainInspectConvNeXt
  moreLinkArgs := ireeLink

lean_exe «test-convnext-train-step» where
  root := `tests.TestConvNextTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-convnext-bn-train-step» where
  root := `tests.TestConvNextBnTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-convnext-ablation-smoke» where
  root := `tests.TestConvNextAblationSmoke
  moreLinkArgs := ireeLink

lean_exe «test-convnext-tiny-emit» where
  root := `tests.TestConvNextTinyEmit

lean_exe «test-resnet-residual» where
  root := `tests.TestResnetResidual

lean_exe «test-spec-helpers» where
  root := `tests.TestSpecHelpers

lean_exe «test-smoke-trainers» where
  root := `tests.TestSmokeTrainers

lean_exe «test-codegen-ts» where
  root := `tests.TestCodegenTrainStep

-- Dischargeability sanity check: 11 examples confirming every
-- Differentiable hypothesis the proofs propagate is satisfiable for
-- the architecture functions (dense, softmax, layerNorm, the flat
-- transformer pieces, mhsa_layer_flat). If any goes vacuous on a
-- refactor, this will fail at build time.
lean_exe «test-diff-sanity» where
  root := `tests.TestDifferentiableSanity

-- ════════════════════════════════════════════════════════════════
-- Bestiary: architecture-only NetSpec examples (print, no training)
-- ════════════════════════════════════════════════════════════════

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
