import LeanMlir.VerifiedSpec

/-! # Concrete verified architectures — the shared specs

Readable layer-list specs that are referenced by **both** a trainer (`Main*Verified`)
and a proof (`LeanMlir/Proofs/*`). Kept in this light module (no Mathlib) so the proof
side can import the *exact* object the trainer runs — there's then a single source of
truth, and "the spec the trainer runs is the proven one" is literally true, not a twin.

Specs with no proof importing them yet (e.g. `resnet34Verified`) stay in their own
`Main*Verified.lean`; a spec moves here the moment a proof needs to name it. -/

/-- The Chapter-1 linear classifier: a single dense 784→10. Trained by
    `MainMnistLinearVerified`; its math VJP is proven in `Proofs/SpecVJP.lean`
    (`linearVerified_has_vjp`) — both over *this* object. -/
def linearVerified : VerifiedNetSpec where
  name     := "MNIST-Linear"
  slug     := "linear"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 10]
  blurb    := "MNIST-Linear via the VERIFIED renderer (pretty∘emit) → IREE FFI → GPU"

-- Shape tie: the derived param layout is W:[784,10] (He) + b:[10] (zeros).
#guard linearVerified.toSpecs == #[(#[784, 10], 0), (#[10], 2)]

/-- The Chapter-2 MLP: dense 784→512 → relu → dense 512→512 → relu → dense 512→10.
    Trained by `MainMnistMlpVerified`; its math VJP is proven in `Proofs/SpecVJP.lean`
    (`mlpVerified_has_vjp` / `_at`) — both over *this* object. -/
def mlpVerified : VerifiedNetSpec where
  name     := "MNIST-MLP"
  slug     := "mlp"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "MNIST-MLP via the VERIFIED renderer (784→512→512→10) → IREE FFI → GPU"

-- Shape tie: W₀:[784,512] b₀:[512] | W₁:[512,512] b₁:[512] | W₂:[512,10] b₂:[10].
#guard mlpVerified.toSpecs ==
  #[(#[784, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- **Width-parametric MNIST MLP** `dense 784→d₁ → relu → dense d₁→d₂ → relu → dense d₂→10`.
    The canonical `mlpVerified` is `mlpG 512 512`. Every instance shares the exact same
    architecture shape as the proven `mlpForward {d₀ d₁ d₂ d₃}` (VJP: `mlp_has_vjp`, which is
    polymorphic in all four dims), so any `(d₁, d₂)` is covered by that one theorem — the
    grid is a single proof instantiated, not a new proof per point. `mnist-mlp-grid` renders
    `verified_mlir/mlp_{d₁}x{d₂}_{train_step,fwd}.mlir` from the faithful renderer at run time
    and trains on it. Slug `mlp_{d₁}x{d₂}`. -/
def mlpG (d₁ d₂ : Nat) : VerifiedNetSpec where
  name     := s!"MNIST-MLP-{d₁}x{d₂}"
  slug     := s!"mlp_{d₁}x{d₂}"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 d₁, .relu, .dense d₁ d₂, .relu, .dense d₂ 10]
  blurb    := s!"MNIST-MLP-{d₁}x{d₂} via the VERIFIED renderer (784→{d₁}→{d₂}→10) → IREE FFI → GPU"

-- `mlpG 512 512` is exactly the canonical `mlpVerified` architecture.
#guard (mlpG 512 512).toSpecs == mlpVerified.toSpecs

/-- The Chapter-3 MNIST CNN (no BN): conv 1→32 → relu → conv 32→32 → relu → maxpool
    28→14 → flatten(6272) → dense 6272→512 → relu → dense 512→512 → relu → dense 512→10.
    Trained by `MainMnistCnnVerified`; its math VJP is proven in `Proofs/SpecVJP.lean`
    (`cnnVerified_has_vjp_at`, folded through conv/maxpool/dense). -/
def cnnVerified : VerifiedNetSpec where
  name     := "MNIST-CNN"
  slug     := "cnn"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.conv 1 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2, .flatten,
               .dense 6272 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "MNIST-CNN via the VERIFIED renderer (conv→conv→pool→512→512→10) → IREE FFI → GPU"

-- Shape tie: conv0[32,1,3,3]+b | conv1[32,32,3,3]+b | dense 6272→512→512→10 (+biases).
#guard cnnVerified.toSpecs ==
  #[(#[32, 1, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[6272, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- The Chapter-4 CIFAR-10 CNN (no BN): conv 3→32 → relu → conv 32→32 → relu → maxpool
    → conv 32→64 → relu → conv 64→64 → relu → maxpool → flatten(4096) → dense 4096→512
    → relu → dense 512→512 → relu → dense 512→10. VJP: `cifarCnn_has_vjp` (Proofs/SpecVJP). -/
def cifarVerified : VerifiedNetSpec where
  name     := "CIFAR-CNN"
  slug     := "cifar"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2,
               .conv 32 64 3 1, .relu, .conv 64 64 3 1, .relu, .maxPool 2 2, .flatten,
               .dense 4096 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "CIFAR-10 CNN via the VERIFIED renderer (3→32→32→pool→32→64→64→pool→512→512→10) → IREE FFI → GPU"

#guard cifarVerified.toSpecs ==
  #[(#[32, 3, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[64, 32, 3, 3], 0), (#[64], 2), (#[64, 64, 3, 3], 0), (#[64], 2),
    (#[4096, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- The Chapter-4 CIFAR-10 CNN **with per-channel BatchNorm** (`.bnPerChannel`, γ/β
    per channel) after each conv. Same backbone as `cifarVerified` + 4 `.bnPerChannel` layers.
    VJP: `cifarBnVerified_has_vjp` (the conditional fold is `cifarCnnBn_has_vjp_at`). -/
def cifarBnVerified : VerifiedNetSpec where
  name     := "CIFAR-CNN-BN"
  slug     := "cifar_bn"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2,
               .conv 32 64 3 1, .bnPerChannel 64, .relu, .conv 64 64 3 1, .bnPerChannel 64, .relu, .maxPool 2 2, .flatten,
               .dense 4096 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "CIFAR-10 CNN + per-channel BatchNorm via the VERIFIED renderer (conv→BN→relu ×4, 2 pools, 512→512→10) → IREE FFI → GPU"

-- conv{W,b} then per-channel BN{γ:[c],β:[c]} ×4, then 3 dense{W,b}.
#guard cifarBnVerified.toSpecs ==
  #[(#[32, 3, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[64, 32, 3, 3], 0), (#[64], 2), (#[64], 1), (#[64], 2),
    (#[64, 64, 3, 3], 0), (#[64], 2), (#[64], 1), (#[64], 2),
    (#[4096, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- The deeper **8-conv CIFAR-10 CNN (no BN)** — the pedagogical BN-demo backbone: four
    `conv→conv→pool` stages, channels `[16,16,32,32]`, 32→16→8→4→2 spatial, then the
    reused 3-dense head (`d1=64`): flatten 128 → 64 → relu → 64 → relu → 10. VJP:
    `Proofs.cifarCnn8_has_vjp_at` (12 ReLU kinks + 4 maxpools), 3-axiom clean. -/
def cifar8Verified : VerifiedNetSpec where
  name     := "CIFAR-CNN8"
  slug     := "cifar8"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 16 3 1, .relu, .conv 16 16 3 1, .relu, .maxPool 2 2,
               .conv 16 16 3 1, .relu, .conv 16 16 3 1, .relu, .maxPool 2 2,
               .conv 16 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2,
               .conv 32 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2, .flatten,
               .dense 128 64, .relu, .dense 64 64, .relu, .dense 64 10]
  blurb    := "Deeper CIFAR-10 CNN (8 convs, [16,16,32,32], 4 pools 32→2 → 128→64→64→10) via the VERIFIED renderer → IREE FFI → GPU"

#guard cifar8Verified.toSpecs ==
  #[(#[16, 3, 3, 3], 0), (#[16], 2), (#[16, 16, 3, 3], 0), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16, 16, 3, 3], 0), (#[16], 2),
    (#[32, 16, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[128, 64], 0), (#[64], 2), (#[64, 64], 0), (#[64], 2), (#[64, 10], 0), (#[10], 2)]

/-- The deeper **8-conv CIFAR-10 CNN with per-channel BatchNorm** — `cifar8Verified` + a
    `.bnPerChannel` after each of the 8 convs (γ=1/β=0 init, before relu). The pedagogical
    BN-acceleration demo. VJP: `Proofs.cifarCnnBn8_has_vjp_at` (12 ReLU kinks + 4 maxpools +
    `0<εᵢ` ×8), 3-axiom clean. Per-channel BN is per-example ⇒ train=eval. -/
def cifar8BnVerified : VerifiedNetSpec where
  name     := "CIFAR-CNN8-BN"
  slug     := "cifar8_bn"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 16 3 1, .bnPerChannel 16, .relu, .conv 16 16 3 1, .bnPerChannel 16, .relu, .maxPool 2 2,
               .conv 16 16 3 1, .bnPerChannel 16, .relu, .conv 16 16 3 1, .bnPerChannel 16, .relu, .maxPool 2 2,
               .conv 16 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2,
               .conv 32 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2, .flatten,
               .dense 128 64, .relu, .dense 64 64, .relu, .dense 64 10]
  blurb    := "Deeper CIFAR-10 CNN + per-channel BatchNorm (8× conv→BN→relu, [16,16,32,32], 4 pools → 128→64→64→10) via the VERIFIED renderer → IREE FFI → GPU"

-- conv{W,b} then per-channel BN{γ:[c],β:[c]} ×8, then 3 dense{W,b}.
#guard cifar8BnVerified.toSpecs ==
  #[(#[16, 3, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[32, 16, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[128, 64], 0), (#[64], 2), (#[64, 64], 0), (#[64], 2), (#[64, 10], 0), (#[10], 2)]

/-- `cifar8Verified` with the MNIST-style **wide 2×512 dense head** (`d1=512`): flatten 128 →
    512 → relu → 512 → relu → 10. Same 8-conv backbone; the head jumps from 13K to 334K floats
    (whole net 52,858 → 373,626). Same parametric VJP `Proofs.cifarCnn8_has_vjp_at` (the dense
    bridge is generic in width). Slug `cifar8w` (render `tests/TestCifar8WideTrain.lean`). -/
def cifar8wVerified : VerifiedNetSpec where
  name     := "CIFAR-CNN8-wide"
  slug     := "cifar8w"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 16 3 1, .relu, .conv 16 16 3 1, .relu, .maxPool 2 2,
               .conv 16 16 3 1, .relu, .conv 16 16 3 1, .relu, .maxPool 2 2,
               .conv 16 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2,
               .conv 32 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2, .flatten,
               .dense 128 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "Deeper CIFAR-10 CNN, MNIST-style wide head (8 convs, [16,16,32,32], 4 pools 32→2 → 128→512→512→10) via the VERIFIED renderer → IREE FFI → GPU"

#guard cifar8wVerified.toSpecs ==
  #[(#[16, 3, 3, 3], 0), (#[16], 2), (#[16, 16, 3, 3], 0), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16, 16, 3, 3], 0), (#[16], 2),
    (#[32, 16, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[128, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- `cifar8BnVerified` with the wide 2×512 dense head (`d1=512`). Slug `cifar8w_bn`. -/
def cifar8wBnVerified : VerifiedNetSpec where
  name     := "CIFAR-CNN8-wide-BN"
  slug     := "cifar8w_bn"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 16 3 1, .bnPerChannel 16, .relu, .conv 16 16 3 1, .bnPerChannel 16, .relu, .maxPool 2 2,
               .conv 16 16 3 1, .bnPerChannel 16, .relu, .conv 16 16 3 1, .bnPerChannel 16, .relu, .maxPool 2 2,
               .conv 16 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2,
               .conv 32 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2, .flatten,
               .dense 128 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "Deeper CIFAR-10 CNN + per-channel BatchNorm, MNIST-style wide head (8× conv→BN→relu → 128→512→512→10) via the VERIFIED renderer → IREE FFI → GPU"

#guard cifar8wBnVerified.toSpecs ==
  #[(#[16, 3, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16], 1), (#[16], 2),
    (#[32, 16, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32], 1), (#[32], 2),
    (#[128, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- ch6 **ResNet-34** on Imagenette 224²: 7×7-s2 stem → BN → relu → maxpool →
    [3,4,6,3] basic-block stages (per-channel BN, strided downsample at the first block of
    stages 2–4) → GAP → dense. 146 params. Tied at the FULL spec in `Proofs/SpecVJP.lean`
    (`resnet34Verified_denote_eq` → `resnet34Forward_full_pc`, + rung E
    `resnet34Verified_fwd_faithful`); the honest pointwise VJP is the audited parametric
    skeleton `Proofs.resnet34_has_vjp_at`. -/
def resnet34Verified : VerifiedNetSpec where
  name     := "ResNet-34"
  slug     := "resnet34"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .convBn 3 64 7 2,            -- 7×7-s2 stem → BN → relu       224→112
    .maxPool 2 2,                --                                112→56
    .residualStage  64  64 3 1,  -- stage1: 3 identity            @56
    .residualStage  64 128 4 2,  -- stage2: downsample + 3        56→28
    .residualStage 128 256 6 2,  -- stage3: downsample + 5        28→14
    .residualStage 256 512 3 2,  -- stage4: downsample + 2        14→7
    .globalAvgPool,
    .dense 512 10 ]
  blurb := "Real ResNet-34 on Imagenette 224² (7×7-s2 stem→pool→[3,4,6,3] blocks w/ batch-norm + strided downsamples, 56→28→14→7→GAP→dense) via the VERIFIED renderer → IREE FFI → GPU"
  -- 36 BN layers in forward order (stem; then per basic block 2, per downsample block 3) — the
  -- running-stats layout for trainAdamSched + @resnet34_fwd_eval. Matches TestResnet34Train.bnLayers.
  bnChannels := #[64,
    64,64, 64,64, 64,64,                              -- stage1: 3 id blocks
    128,128,128, 128,128, 128,128, 128,128,           -- d2 + stage2: 3 id blocks
    256,256,256, 256,256, 256,256, 256,256, 256,256, 256,256,  -- d3 + stage3: 5 id blocks
    512,512,512, 512,512, 512,512]                    -- d4 + stage4: 2 id blocks

-- Derived layout (146 params) == the audited hand-list ResNet34Layout.specs.
#guard resnet34Verified.toSpecs == ResNet34Layout.specs

/-- ch7 **MobileNetV2** on Imagenette 224²: 3×3-s2 stem → BN → relu6 → 17 inverted-residual
    blocks (full-paper `[t,c,n,s]` config, strided depthwise downsamples, per-channel BN,
    relu6, linear bottleneck) → 1×1 head conv (320→1280) → BN → relu6 → GAP → dense.
    (Tied at the FULL paper spec in `Proofs/SpecVJP.lean`: `mobilenetv2Verified_denote_eq`
    → `mobilenetv2ForwardPaper`, + rung E `mobilenetv2Verified_fwd_faithful`. The honest
    pointwise VJP-fold witness `Proofs.mobilenetv2_has_vjp_at` remains the representative
    stem+2-block net, see planning doc.) -/
def mobilenetv2Verified : VerifiedNetSpec where
  name     := "MobileNetV2"
  slug     := "mobilenetv2"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .convBn 3 32 3 2,               -- stem
    .invertedResidual 32  32  16 1,
    .invertedResidual 16  96  24 2, .invertedResidual 24 144  24 1,
    .invertedResidual 24 144  32 2, .invertedResidual 32 192  32 1, .invertedResidual 32 192  32 1,
    .invertedResidual 32 192  64 2, .invertedResidual 64 384  64 1, .invertedResidual 64 384  64 1, .invertedResidual 64 384  64 1,
    .invertedResidual 64 384  96 1, .invertedResidual 96 576  96 1, .invertedResidual 96 576  96 1,
    .invertedResidual 96 576 160 2, .invertedResidual 160 960 160 1, .invertedResidual 160 960 160 1,
    .invertedResidual 160 960 320 1,
    .convBn 320 1280 1 1,           -- head
    .globalAvgPool,
    .dense 1280 10 ]
  blurb := "MobileNetV2 on Imagenette 224² (stem-s2 → 17 inverted-residual blocks, full-paper [t,c,n,s] config, stride-2 depthwise downsamples 224→7 → head conv-BN-relu6 → GAP → dense) via the VERIFIED renderer → IREE FFI → GPU"
  -- 52 BN layers in forward order (stem; per inverted-residual block expand-BN/depthwise-BN/project-BN,
  -- but b1 is t=1 → NO expand, so only depthwise-BN/project-BN; head) — running-stats layout for
  -- trainAdamSched + @mobilenetv2_fwd_eval. Matches TestMobilenetV2TrainPC.bnLayers. True batch-norm
  -- (reduce [0,2,3]) → batch-BN eval degenerate on sorted val, so the adam trainer evals through running stats.
  bnChannels := #[32,
    32,16,  96,96,24, 144,144,24,  144,144,32, 192,192,32, 192,192,32,
    192,192,64, 384,384,64, 384,384,64, 384,384,64,
    384,384,96, 576,576,96, 576,576,96,
    576,576,160, 960,960,160, 960,960,160,
    960,960,320,
    1280]

-- Derived layout (210 param tensors == the canonical no-t=1-expand net, torchvision-standard:
-- b1 is t=1 so its expand 1×1 is skipped) == the audited hand-list MobileNetV2Layout.specs.
#guard mobilenetv2Verified.toSpecs == MobileNetV2Layout.specs

/-- ch8 **EfficientNet-B0** on Imagenette 224²: 3×3-s2 stem → 16 MBConv blocks (`[t,c,n,s,k]`
    B0 config; expand 1×1 [skip when t=1] → depthwise k×k → squeeze-excite → project 1×1, all
    BN + swish) → 1×1 head (320→1280) → GAP → dense. 262 params. The 16 `mbConvSE ic mid oc r k`
    args are the B0 generator unrolled (mid=t·ic, r=ic/4, ic threads stage→stage). Tied at the
    FULL spec in `Proofs/SpecVJP.lean` (`efficientnetVerified_denote_eq` →
    `efficientnetForwardB_full`, batched ∀N, + rung E `efficientnetVerified_fwd_faithful`);
    the honest pointwise VJP witness is the representative `Proofs.efficientnet_has_vjp`. -/
def efficientnetVerified : VerifiedNetSpec where
  name     := "EfficientNet-B0"
  slug     := "efficientnet"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .convBn 3 32 3 2,            -- stem 3×3-s2
    .mbConvSE   32   32  16  8 3,  -- s1 t1 (no expand)
    .mbConvSE   16   96  24  4 3,  -- s2
    .mbConvSE   24  144  24  6 3,
    .mbConvSE   24  144  40  6 5,  -- s3
    .mbConvSE   40  240  40 10 5,
    .mbConvSE   40  240  80 10 3,  -- s4
    .mbConvSE   80  480  80 20 3,
    .mbConvSE   80  480  80 20 3,
    .mbConvSE   80  480 112 20 5,  -- s5
    .mbConvSE  112  672 112 28 5,
    .mbConvSE  112  672 112 28 5,
    .mbConvSE  112  672 192 28 5,  -- s6
    .mbConvSE  192 1152 192 48 5,
    .mbConvSE  192 1152 192 48 5,
    .mbConvSE  192 1152 192 48 5,
    .mbConvSE  192 1152 320 48 3,  -- s7
    .convBn 320 1280 1 1,         -- head 1×1 (320→1280)
    .globalAvgPool,
    .dense 1280 10 ]
  blurb := "EfficientNet-B0 on Imagenette 224² (stem-s2 → 16 MBConv [t,c,n,s,k], swish + squeeze-excite + batch-norm, 5 downsamples 224→7 → head 320→1280 → GAP → dense) via the VERIFIED renderer → IREE FFI → GPU"
  -- 49 BN layers in forward order (stem; per MBConv: expand-BN [t≠1 only], depthwise-BN, project-BN;
  -- head) — running-stats layout for trainAdamSched + @efficientnet_fwd_eval. Printed by
  -- TestEfficientNetTrain.bnChannelsList; true batch-norm makes batch-BN eval degenerate on sorted val.
  bnChannels := #[32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 40, 240, 240, 40, 240, 240, 80,
    480, 480, 80, 480, 480, 80, 480, 480, 112, 672, 672, 112, 672, 672, 112, 672, 672, 192,
    1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 320, 1280]

-- Derived layout (262 params) == the audited hand-list EfficientNetLayout.specs.
#guard efficientnetVerified.toSpecs == EfficientNetLayout.specs

/-- ch9 **ConvNeXt-T** on Imagenette 224²: 4×4-s4 patchify → [3,3,9,3] ConvNeXt blocks @
    [96,192,384,768] (depthwise 7×7 → scalar-LN → 1×1 expand → GELU → 1×1 project → layerScale)
    with 3 between-stage (LN + 2×2-s2) downsamples (56→28→14→7) → GAP → LN → dense. 180 params.
    Tied at the FULL spec in `Proofs/SpecVJP.lean` (`convnextVerified_denote_eq` →
    `convNextForwardTC`, the committed 180-param config, + rung E
    `convnextVerified_fwd_faithful`); the full-depth REAL VJP is
    `Proofs.convNextForwardTC_has_vjp_correct` (ConvNeXtFullT.lean). -/
def convnextVerified : VerifiedNetSpec where
  name     := "ConvNeXt-T"
  slug     := "convnext"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .conv 3 96 4 4,                                              -- patchify 4×4/s4   224→56
    .convNextBlock 96, .convNextBlock 96, .convNextBlock 96,     -- stage 1 (3) @56
    .bn, .conv 96 192 2 2,                                       -- downsample 96→192  56→28
    .convNextBlock 192, .convNextBlock 192, .convNextBlock 192,  -- stage 2 (3) @28
    .bn, .conv 192 384 2 2,                                      -- downsample 192→384 28→14
    .convNextBlock 384, .convNextBlock 384, .convNextBlock 384,  -- stage 3 (9) @14
    .convNextBlock 384, .convNextBlock 384, .convNextBlock 384,
    .convNextBlock 384, .convNextBlock 384, .convNextBlock 384,
    .bn, .conv 384 768 2 2,                                      -- downsample 384→768 14→7
    .convNextBlock 768, .convNextBlock 768, .convNextBlock 768,  -- stage 4 (3) @7
    .globalAvgPool, .bn, .dense 768 10 ]                         -- head: GAP → LN → dense
  blurb := "ConvNeXt-T on Imagenette 224² (patchify /4 → [3,3,9,3] blocks @ [96,192,384,768] depthwise-7×7 + LN + GELU + layerScale + 3 downsamples 56→7 → GAP → LN → dense) via the VERIFIED renderer → IREE FFI → GPU"

-- Derived layout (180 params) == the audited hand-list ConvNeXtLayout.specs.
#guard convnextVerified.toSpecs == ConvNeXtLayout.specs

/-- ch10 **ViT-Tiny** on Imagenette 224² (patch-16): 16×16-s16 conv patch embed (3→192,
    →196 patches), learned CLS token + positional embed (→197 tokens), 12 pre-norm transformer
    blocks (dim 192, 3 heads, MLP 768), final per-channel LayerNorm, CLS-slice dense head 192→10.
    200 params. Tied at the FULL spec in `Proofs/SpecVJP.lean` (`vitVerified_denote_eq` →
    `vitForwardKV` depth-12 distinct-param vector-LN, retiring the old weight-shared
    scalar-LN caveats), with the REAL whole-net VJP `Proofs.vitVerified_has_vjp`
    (all-smooth, `0 < ε` only) and rung E `vitVerified_fwd_faithful` (the depth-12
    multi-head vector-LN graph `vitFwdGraphKMHV`). -/
def vitVerified : VerifiedNetSpec where
  name     := "ViT-Tiny"
  slug     := "vit"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .conv 3 192 16 16,            -- patch embed 16×16/s16 (3→192)   224→14×14=196
    .param #[192] 2,              -- CLS token  [192] (1D — matches the proof-tied render's `cls : Vec 192`)
    .param #[197, 192] 2,         -- positional embedding  [197,192]
    .transformerBlock 192 768,    -- 12 pre-norm blocks @ dim 192, MLP 768
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .transformerBlock 192 768,
    .layerNorm 192,               -- final LayerNorm (per-channel [192])
    .dense 192 10 ]               -- CLS-head 192→10
  blurb := "ViT-Tiny on Imagenette 224² (patch-16 → CLS+pos → 12 transformer blocks @ dim192/3heads/MLP768 → final LN → CLS-head 10) via the VERIFIED renderer → IREE FFI → GPU"

-- Derived layout (200 params) == the audited hand-list ViTLayout.specs.
#guard vitVerified.toSpecs == ViTLayout.specs
