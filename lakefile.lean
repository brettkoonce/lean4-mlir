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
             -- Nonzero-Jacobian seal (planning/whole_network_backward.md Item B): the
             -- generic "one nonzero Jacobian entry ⇒ non-trivial backward" bridge.
             `LeanMlir.Proofs.JacobianSeal,
             -- Item B2: the seal discharged at the live MobileNetV2 witness —
             -- `fderiv ℝ forward 0 ≠ 0` ⇒ non-trivial whole-net backward (level 3).
             `LeanMlir.Proofs.MobileNetV2JacobianSeal,
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
             -- Deeper (8-conv) CIFAR-CNN close: cifar8{,Bn}FwdGraph_faithful's backward
             -- peer — each conv W/b, BN γ/β, dense W/b output pinned to the actual 4-stage
             -- backward-chain cotangent (the CnnChainClose recipe + BN, two more pool stages).
             `LeanMlir.Proofs.Cifar8Close,
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
             `LeanMlir.Proofs.ResNet34ChainClose,
             -- ConvNeXt close (Item C): mostly reuse (7×7 depthwise pinned to the generic
             -- bridges) + the two genuinely-new families — layer-scale γ (dγ = x⊙dy) and
             -- scalar-LN γ/β (the Vec-1 embedding bridging bn_grad_gamma/beta).
             `LeanMlir.Proofs.ConvNeXtClose,
             -- ViT close (Item A): the distinct-param 2-block ViT forward (vitForward2 +
             -- whole-net VJP) and the heads=1 token forward graph + faithfulness
             -- (den vitFwdGraph = vitForward2 via mhsa_layer_one_head).
             `LeanMlir.Proofs.ViTFwdGraph,
             -- ViT close (Item C): the per-token dense W/b family (row-lifted M2
             -- outer product), row-lifted scalar-LN γ/β, pos-embed identity, CLS
             -- masked-gather — every representative-ViT param family except the
             -- patch conv certified.
             `LeanMlir.Proofs.ViTClose,
             -- ViT cotangent-chain close (Item D): the Item C bridges pinned to the
             -- attention-block backward chain (SDPA matmul chain = the proven
             -- sdpa_back_{Q,K,V} closed forms; the Q/K/V three-way fan-in at LN1).
             `LeanMlir.Proofs.ViTChainClose,
             -- ViT scaling pass (vector-[D] LN): layerNormVec block + vitForward2V
             -- whole-net VJP + the rowScaleF/rowBiasF token graph + faithfulness +
             -- the per-channel gamma/beta param bridges.
             `LeanMlir.Proofs.ViTVecLN,
             -- ViT scaling pass (multi-head + depth-k): headSliceF/headPadF tokens,
             -- mhsa at general heads, then the distinct-param depth-k tower
             -- (vitForwardKV). ViTDepthK imports ViTMultiHead, covering both.
             `LeanMlir.Proofs.ViTDepthK,
             -- EfficientNet-B0 at full depth (16 distinct MBConv blocks, true BN+SE):
             -- batched forward graph + whole-net VJP. Imports the EfficientNet
             -- RenderPC + ChainClose modules, covering all three.
             `LeanMlir.Proofs.EfficientNetFullB0,
             -- Full ConvNeXt-T [3,3,9,3]: forward graph + faithfulness + whole-net
             -- VJP. Imports ConvNeXtChainClose, covering both.
             `LeanMlir.Proofs.ConvNeXtFullT,
             -- Paper-spec full MobileNetV2 (all 17 [t,c,n,s] bottlenecks): forward
             -- graph + faithfulness.
             `LeanMlir.Proofs.MobileNetV2FullPaper,
             -- ℝ→Float32 bridge, Tier 1: standard-model rounding (hypothesis-style,
             -- no axioms) + forward error bounds for the linear/MLP nets
             -- (dot/dense budgets, ReLU exact-in-float Lipschitz pass-through).
             `LeanMlir.Proofs.FloatBridge,
             -- Inexact-gradient descent over ℝ (MVT form): an η-accurate gradient
             -- oracle + segment smoothness ⇒ the SGD step still decreases the loss,
             -- with an explicit decrease. The keystone the FloatBridge budgets
             -- plug into ("close" ⇒ "still trains").
             `LeanMlir.Proofs.SgdDescent,
             -- The smoothness hypothesis DISCHARGED for the Chapter-2 linear
             -- softmax-CE loss: explicit segment-Lipschitz constant 2a²/(1−2aD)
             -- via the softmax ratio sandwich (no Hessian), and the capstone —
             -- one inexact SGD step provably decreases the cross-entropy loss.
             `LeanMlir.Proofs.SgdDescentLinear,
             -- The smoothness hypothesis discharged through the Chapter-3
             -- MLP: under quantitative ReLU margins (the step cannot flip a
             -- mask sign) the loss-of-one-layer maps get explicit
             -- segment-Lipschitz constants, and one inexact SGD step on each
             -- weight layer provably decreases the cross-entropy loss.
             `LeanMlir.Proofs.SgdDescentMlp,
             -- The descent program reaches the Chapter-4 CNN: quantitative
             -- max-pool selection margins (the argmax freezes along the step
             -- segment), pool 1-Lipschitz/ℓ1-contraction, conv kernel drift.
             `LeanMlir.Proofs.SgdDescentCnn,
             -- The optimizer rung beyond SGD: the ℝ Adam/AdamW step mirroring
             -- the emitted update (Phase 3a of vit_train_to_vit_verified.md).
             -- Faithfulness target + denominator well-definedness; NO descent
             -- claim (Adam isn't monotone).
             `LeanMlir.Proofs.AdamStep,
             -- Phase 3b: the AdamW render-close — emitted weight/bias update =
             -- adamWScalar of the certified gradient (sgdW_descends_certified_grad
             -- analogue, optimizer swapped for AdamW).
             `LeanMlir.Proofs.AdamRender,
             -- WIP toward a live ResNet-34 witness (Mnv2Live peer): Stage-1
             -- `liveDown` mechanism only — `liveFwd` is still constant-output at
             -- 1 channel (structural: gap averages BN deviations back to β; see
             -- the file header). Build-checked so the mechanism doesn't rot; NOT
             -- a live witness and NOT in the AuditAxioms headline set.
             `LeanMlir.Proofs.ResNet34Live,
             -- Stage 2 of the live ResNet-34 (Item A2): the channel-order invariant
             -- kit (maxpool/BN/ReLU preserve strict pointwise channel domination —
             -- the non-vacuity carrier). Build-checked; not yet a live witness, so
             -- also NOT in the AuditAxioms headline set.
             `LeanMlir.Proofs.ResNet34Live2,
             -- Stage 3 (Item A1): the 2-channel layer rebuild — liveDownPC (the
             -- signal-carrying 2-channel downsample, full VJP/diff) + per-coordinate
             -- BN injectivity. Build-checked; not yet a live witness.
             `LeanMlir.Proofs.ResNet34LivePC,
             -- Backward-graph faithfulness (den-level): fan-in bricks
             -- (residual/SE), per-op backward ops (gap/broadcast/true-batch-norm/
             -- batched conv+depthwise), the whole per-example MBConv block, and
             -- the batched-stage backward primitives.
             `LeanMlir.Proofs.EfficientNetBackB0,
             -- MobileNetV2 backward-graph faithfulness (den-level): the batched
             -- relu6 conv/depthwise stages (selectMid kink), the SE-less inverted-
             -- residual body, and the whole-block capstone — the relu6 (_at)
             -- peer of EfficientNetBackB0.
             `LeanMlir.Proofs.MobileNetV2BackB0,
             -- ResNet-34 backward-graph faithfulness (den-level): the batched
             -- conv-bn-relu stage (selectPos one-sided kink), the basic-block
             -- body (conv-bn ∘ conv-bn-relu), and the identity-block capstone —
             -- relu (_at) with an OUTER post-residual relu (the extra factor
             -- vs the MBConv/inverted-residual blocks).
             `LeanMlir.Proofs.ResNet34BackB0,
             -- ConvNeXt backward-graph faithfulness (den-level): the per-example
             -- (batch-1) peer of EfficientNetBackB0. LayerNorm is per-example
             -- separable, so no batched machinery — the block-body backward graph
             -- (depthwise → LN → expand → gelu → project → layerScale) + identity-skip
             -- residual capstone, plus the LN+2×2/s2 downsample capstone. GELU is a
             -- global VJP, so everything stays in the clean global HasVJP form.
             `LeanMlir.Proofs.ConvNeXtBackB0,
             -- ViT whole-block backward-graph faithfulness (den-level, heads = 1):
             -- the per-token Mat-VJP peer of the conv nets' *BackB0 capstones. MLP +
             -- attention sublayer backward graphs (residual fan-in + LN-back), with
             -- the MHSA backward collapsed at heads = 1 to the plain three-way dense
             -- fan-in over the proven sdpa_back_{Q,K,V} (tied to mhsa_has_vjp_mat by
             -- VJP determinism), assembled into the whole transformerBlock VJP.
             `LeanMlir.Proofs.ViTBackB0]

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

-- Deeper 8-conv CIFAR-10 CNN (no BN; [16,16,32,32], 4 pools) on the VERIFIED-rendered
-- StableHLO (verified_mlir/cifar8_train_step.mlir = Proofs.StableHLO.cifar8TrainStepText).
lean_exe «cifar8-verified» where
  root := `MainCifar8Verified
  moreLinkArgs := ireeLink

-- Deeper 8-conv CIFAR-10 CNN + per-channel BN on the VERIFIED-rendered StableHLO
-- (Proofs.StableHLO.cifar8BnTrainStepText). The pedagogical BN-acceleration demo.
lean_exe «cifar8-bn-verified» where
  root := `MainCifar8BnVerified
  moreLinkArgs := ireeLink

-- ch6 B9: real ResNet-34 ([3,4,6,3], per-channel BN, strided downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestResnet34{Train,Fwd}.lean); 146 params.
lean_exe «resnet34-verified» where
  root := `MainResnet34Verified
  moreLinkArgs := ireeLink

-- r34 peer of mnv2/enet-verified-adam: the proof-rendered train step (per-channel BN + strided
-- downsamples) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime
-- lr/bc threading via trainAdamSched. Recipe matches the reference (lr 1e-3, wd 1e-4, cosine+warmup
-- 3, label-smoothing 0.1). Render: tests/TestResnet34Train.lean.
lean_exe «resnet34-verified-adam» where
  root := `MainResnet34VerifiedAdam
  moreLinkArgs := ireeLink

-- ch7 C4: small MobileNetV2 (inverted-residual blocks: depthwise conv + relu6 +
-- per-channel BN) trained on VERIFIED-rendered StableHLO
-- (tests/TestMobilenetV2{Train,Fwd}.lean); 30 params.
lean_exe «mobilenetv2-verified» where
  root := `MainMobilenetV2Verified
  moreLinkArgs := ireeLink

-- mnv2 peer of vit-verified-adam: the proof-rendered train step with the SGD update swapped for
-- AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime lr/bc threading via trainAdamSched. Recipe
-- matches mobilenet-v2-train (lr 1e-3, wd 1e-4, cosine+warmup 3, label-smoothing 0.1). Loss-curve
-- parity; batch-BN eval (running-stats BN deferred). Render: tests/TestMobilenetV2TrainPC.lean.
lean_exe «mobilenetv2-verified-adam» where
  root := `MainMobilenetV2VerifiedAdam
  moreLinkArgs := ireeLink

-- ch8 E4/E5/E6: EfficientNet-B0 (faithful [t,c,n,s,k] config — 16 MBConv layers,
-- inverted-residual + squeeze-excite + swish + BATCH norm, 3×3/5×5 depthwise) trained
-- on VERIFIED-rendered StableHLO (tests/TestEfficientNet{Train,Fwd}.lean); 262 params.
lean_exe «efficientnet-verified» where
  root := `MainEfficientNetVerified
  moreLinkArgs := ireeLink

-- enet peer of mnv2-verified-adam: the proof-rendered train step (all-swish + squeeze-excite +
-- batch-norm) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime
-- lr/bc threading via trainAdamSched. Recipe matches efficientnet-train (lr 1e-3, wd 1e-4,
-- cosine+warmup 3, label-smoothing 0.1). Render: tests/TestEfficientNetTrain.lean.
lean_exe «efficientnet-verified-adam» where
  root := `MainEfficientNetVerifiedAdam
  moreLinkArgs := ireeLink

-- Chapter 9: ConvNeXt-T (Liu et al. 2022 — patchify stem + [3,3,9,3] depthwise-7×7
-- blocks with LN + GELU + layerScale + 3 between-stage downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestConvNeXt{Train,Fwd}.lean); 180 params.
lean_exe «convnext-verified» where
  root := `MainConvNeXtVerified
  moreLinkArgs := ireeLink

-- convnext peer of r34-verified-adam: the proof-rendered train step (all-smooth — LayerNorm +
-- GELU + layerScale, no BN) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed
-- θ|m|v + runtime lr/bc threading via trainAdamSched. Recipe matches the reference (lr 1e-3, wd
-- 1e-4, cosine+warmup 3, label-smoothing 0.1). Render: tests/TestConvNeXtTrain.lean.
lean_exe «convnext-verified-adam» where
  root := `MainConvNeXtVerifiedAdam
  moreLinkArgs := ireeLink

lean_exe «vit-verified» where
  root := `MainViTVerified
  moreLinkArgs := ireeLink

-- Phase 3c: ViT-Tiny with the VERIFIED-rendered AdamW step (packed θ|m|v threading
-- through the generic FFI; ViTRender.vitTrainStepModuleAdamPacked / trainAdamPacked).
lean_exe «vit-verified-adam» where
  root := `MainViTVerifiedAdam
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
