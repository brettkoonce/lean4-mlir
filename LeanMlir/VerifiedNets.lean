import LeanMlir.VerifiedSpec

/-! # Concrete verified architectures — the shared specs

Readable layer-list specs that are referenced by **both** a trainer (`Main*Verified`)
and a proof (`LeanMlir/Proofs/*`). Kept in this light module (no Mathlib) so the proof
side can import the *exact* object the trainer runs — there's then a single source of
truth, and "the spec the trainer runs is the proven one" is literally true, not a twin.

Specs with no proof importing them yet (e.g. `resnet34Verified`) stay in their own
`Main*Verified.lean`; a spec moves here the moment a proof needs to name it. -/

/-- The Chapter-2 linear classifier: a single dense 784→10. Trained by
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

/-- The Chapter-3 MLP: dense 784→512 → relu → dense 512→512 → relu → dense 512→10.
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

/-- The Chapter-4 MNIST CNN (no BN): conv 1→32 → relu → conv 32→32 → relu → maxpool
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

/-- The Chapter-5 CIFAR-10 CNN (no BN): conv 3→32 → relu → conv 32→32 → relu → maxpool
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

/-- The Chapter-5 CIFAR-10 CNN **with scalar BatchNorm** (`bnForward`, one γ/β over the
    whole c·h·w map) after each conv. Same backbone as `cifarVerified` + 4 `.bn` layers.
    VJP: `cifarBnVerified_has_vjp` (the conditional fold is `cifarCnnBn_has_vjp_at`). -/
def cifarBnVerified : VerifiedNetSpec where
  name     := "CIFAR-CNN-BN"
  slug     := "cifar_bn"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 32 3 1, .bn, .relu, .conv 32 32 3 1, .bn, .relu, .maxPool 2 2,
               .conv 32 64 3 1, .bn, .relu, .conv 64 64 3 1, .bn, .relu, .maxPool 2 2, .flatten,
               .dense 4096 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "CIFAR-10 CNN + scalar BatchNorm via the VERIFIED renderer (conv→BN→relu ×4, 2 pools, 512→512→10) → IREE FFI → GPU"

-- conv{W,b} then scalar BN{γ:[],β:[]} ×4, then 3 dense{W,b}.
#guard cifarBnVerified.toSpecs ==
  #[(#[32, 3, 3, 3], 0), (#[32], 2), (#[], 1), (#[], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[], 1), (#[], 2),
    (#[64, 32, 3, 3], 0), (#[64], 2), (#[], 1), (#[], 2),
    (#[64, 64, 3, 3], 0), (#[64], 2), (#[], 1), (#[], 2),
    (#[4096, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- ch6 **ResNet-34** on Imagenette 224²: 7×7-s2 stem → BN → relu → maxpool →
    [3,4,6,3] basic-block stages (per-channel BN, strided downsample at the first block of
    stages 2–4) → GAP → dense. 146 params. VJP: the audited parametric skeleton
    `Proofs.resnet34_has_vjp_at`. -/
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
  blurb := "Real ResNet-34 on Imagenette 224² (7×7-s2 stem→pool→[3,4,6,3] blocks w/ per-channel BN + strided downsamples, 56→28→14→7→GAP→dense) via the VERIFIED renderer → IREE FFI → GPU"

-- Derived layout (146 params) == the audited hand-list ResNet34Layout.specs.
#guard resnet34Verified.toSpecs == ResNet34Layout.specs
