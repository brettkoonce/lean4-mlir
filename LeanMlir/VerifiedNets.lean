import LeanMlir.VerifiedSpec

/-! # Concrete verified architectures — the shared specs

Readable layer-list specs that are referenced by **both** a trainer (`Main*Verified`)
and a proof (`LeanMlir/Proofs/*`). Kept in this light module (no Mathlib) so the proof
side can import the *exact* object the trainer runs — there's then a single source of
truth, and "the spec the trainer runs is the proven one" is literally true, not a twin.

Specs with no proof importing them yet (e.g. `resnet34Verified`) stay in their own
`Main*Verified.lean`; a spec moves here the moment a proof needs to name it. -/

/-- **The driver-side half of the MobileNetV2 / EfficientNet RMSProp recipe** — peak LR, the
    exponential decay `VerifiedNet.trainAdamSched` runs, and the warmup length.

    The *emitted* half (ρ, μ, ε, coupled wd) is `Proofs.StableHLO.RmsHyper`, which the renderers
    bake into each graph via `rmsConstsBlock`. These three do NOT belong there: `%lr` is a runtime
    `tensor<f32>` argument exactly so one render serves a whole schedule, and a learning rate that
    became a graph constant would be a silent, uncheckable hyperparameter — the
    `RenderCifar8Sgd02` / EfficientNet-16× failure this repo has already paid for twice
    (handoff §2a-quater, §2a-quinquies). Keeping the two halves in two modules makes that
    impossible rather than merely discouraged.

    It lives here, in the light shared-spec module, for this file's own stated reason: four entry
    points read it (Imagenette and ImageNet × two nets) and a per-site copy of `0.98` is the
    double-writer disease at its smallest and most plausible.

    ⚠ These are the **reference's** values at the reference's **batch 256**. Anything else is a
    different experiment; the Imagenette callers scale `lr` by batch and say so. -/
structure RmsSchedule where
  /-- `learningRate` — the peak, at batch 256. -/
  lr : Float
  /-- `expLRDecayRate` — the multiplier applied once per `decayEpochs`, after warmup. -/
  decayRate : Float
  /-- `expLRDecayEpochs` — how many epochs one multiplication spans. ⚠ Not 1 on both nets. -/
  decayEpochs : Float := 1.0
  /-- `warmupEpochs` — the linear ramp to `lr`. 5 on both. -/
  warmup : Nat := 5

/-- **MobileNetV2**: 0.045 peak, ×0.98 **per epoch** (`jax/MainMobilenetV2Imagenet.lean`). -/
def mnv2RmsSchedule : RmsSchedule := { lr := 0.045, decayRate := 0.98 }

/-- **EfficientNet-B0**: 0.016 peak, ×0.97 **every 2.4 epochs** — the paper's schedule, and the
    linear scaling of 0.256@4096 down to batch 256 (`jax/MainEfficientNetImagenet.lean`). -/
def enetRmsSchedule : RmsSchedule := { lr := 0.016, decayRate := 0.97, decayEpochs := 2.4 }

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

/-- **FC-width-parametric MNIST CNN** — the Chapter-3 CNN with the two convs held at 32
    channels (so the feature extractor is fixed) and the **dense classifier head** swept:
    `…maxpool → flatten(6272) → dense 6272→d → relu → dense d→d → relu → dense d→10`. The
    canonical `cnnVerified` is `cnnG 512`. The faithful CNN renderer (`cnnTrainStepFaithfulV`)
    takes a single dense width `d` (both hidden FC layers share it), so this is the honest
    den-certified path; `mnist-cnn-grid d` renders `verified_mlir/cnn_{d}_{train_step,fwd}.mlir`
    and trains on it. Isolates the ROI of the classifier head with the conv stack fixed. -/
def cnnG (d : Nat) : VerifiedNetSpec where
  name     := s!"MNIST-CNN-fc{d}"
  slug     := s!"cnn_{d}"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.conv 1 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2, .flatten,
               .dense 6272 d, .relu, .dense d d, .relu, .dense d 10]
  blurb    := s!"MNIST-CNN-fc{d} via the VERIFIED renderer (conv32→conv32→pool→{d}→{d}→10) → IREE FFI → GPU"

-- `cnnG 512` is exactly the canonical `cnnVerified` architecture.
#guard (cnnG 512).toSpecs == cnnVerified.toSpecs

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

/-- **FC-head-parametric cifar8-BN** — the 8-conv per-channel-BN CIFAR net with the conv
    backbone held at `[16,16,32,32]` and only the dense classifier head swept:
    `…flatten(128) → dense 128→d → relu → dense d→d → relu → dense d→10`. The canonical
    `cifar8BnVerified` is `cifar8BnG 64`. `cifar8-bn-grid` trains each width via
    `trainAdamSched "adam"` on the width-slugged renders
    `verified_mlir/cifar8_bn_{d}_{adam_train_step,fwd}.mlir` (emitted by
    `tests/TestCifar8AdamTrain.lean`, D1 parametric). Per-channel BN ⇒ train=eval (no running
    stats, `bnChannels` empty). Slug `cifar8_bn_{d}`. -/
def cifar8BnG (d : Nat) : VerifiedNetSpec where
  name     := s!"CIFAR-CNN8-BN-fc{d}"
  slug     := s!"cifar8_bn_{d}"
  inC      := 3
  imageH   := 32
  imageW   := 32
  nClasses := 10
  data     := .cifar
  layers   := [.conv 3 16 3 1, .bnPerChannel 16, .relu, .conv 16 16 3 1, .bnPerChannel 16, .relu, .maxPool 2 2,
               .conv 16 16 3 1, .bnPerChannel 16, .relu, .conv 16 16 3 1, .bnPerChannel 16, .relu, .maxPool 2 2,
               .conv 16 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2,
               .conv 32 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2, .flatten,
               .dense 128 d, .relu, .dense d d, .relu, .dense d 10]
  blurb    := s!"CIFAR-CNN8-BN-fc{d} via the VERIFIED renderer (8× conv→BN→relu [16,16,32,32] → 128→{d}→{d}→10, AdamW) → IREE FFI → GPU"

-- `cifar8BnG 64` is exactly the canonical `cifar8BnVerified` architecture.
#guard (cifar8BnG 64).toSpecs == cifar8BnVerified.toSpecs

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
    stages 2–4) → GAP → dense. **110 params** (§2l step B: no conv biases). Tied at the FULL spec in `Proofs/SpecVJP.lean`
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
    .convBnNB 3 64 7 2,          -- 7×7-s2 stem → BN → relu       224→112 (no conv bias)
    .maxPool 2 2,                --                                112→56
    .residualStage  64  64 3 1,  -- stage1: 3 identity            @56
    .residualStage  64 128 4 2,  -- stage2: downsample + 3        56→28
    .residualStage 128 256 6 2,  -- stage3: downsample + 5        28→14
    .residualStage 256 512 3 2,  -- stage4: downsample + 2        14→7
    .globalAvgPool,
    .dense 512 10 ]
  blurb := "Real ResNet-34 on Imagenette 224² (7×7-s2 stem→pool→[3,4,6,3] blocks w/ batch-norm, He et al. option-B 1×1 projection shortcuts, no conv biases; 56→28→14→7→GAP→dense) via the VERIFIED renderer → IREE FFI → GPU"
  -- 36 BN layers in forward order (stem; then per basic block 2, per downsample block 3) — the
  -- running-stats layout for trainAdamSched + @resnet34_fwd_eval. Matches TestResnet34Train.bnLayers.
  bnChannels := #[64,
    64,64, 64,64, 64,64,                              -- stage1: 3 id blocks
    128,128,128, 128,128, 128,128, 128,128,           -- d2 + stage2: 3 id blocks
    256,256,256, 256,256, 256,256, 256,256, 256,256, 256,256,  -- d3 + stage3: 5 id blocks
    512,512,512, 512,512, 512,512]                    -- d4 + stage4: 2 id blocks

-- Derived layout (110 params) == the audited hand-list ResNet34Layout.specs. This `#guard` is
-- the one §2k said would have caught the spec/render drift; it fired on the §2l step-B change and
-- is what forced `VLayer` to grow a bias-free conv rather than the layout being edited by hand.
#guard resnet34Verified.toSpecs == ResNet34Layout.specs

/-- **ResNet-34 on full 1000-class ImageNet** — the scale/reference tier (handoff §2k).

    Identical architecture to `resnet34Verified`; only the head width, the class count and the data
    source differ. It exists to be run as a **matched pair** with `jax/MainResnetImagenet.lean`
    (same net, same heavy-ball + coupled-L2 recipe, same tfds augmentation via the generated shim),
    with the JAX side as the external oracle.

    ⚠ **Read the claim ceiling before quoting this.** The proof-carrying tier stops at Imagenette:
    this net has no §1a tie, no `SpecVJP` witness, and no entry in the prefix audit's hand-lists.
    What it has is *provenance* — `pretty(provenGraph)` off the same certified renderer, `nClasses`
    and `B` being ordinary parameters of it — plus whatever the pair agreement shows. The honest
    sentence is **"one architecture, two independent lowerings, agreeing"**, not "proven".

    `slug` is `resnet34in` so its three artifacts cannot collide with the 10-class ones — the
    forwards carry no variant in their path and would otherwise overwrite them. -/
def resnet34ImagenetVerified : VerifiedNetSpec where
  name     := "ResNet-34 (ImageNet-1k)"
  slug     := "resnet34in"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  layers   := [
    .convBnNB 3 64 7 2,          -- 7×7-s2 stem → BN → relu       224→112 (no conv bias)
    .maxPool 2 2,                --                                112→56
    .residualStage  64  64 3 1,  -- stage1: 3 identity            @56
    .residualStage  64 128 4 2,  -- stage2: downsample + 3        56→28
    .residualStage 128 256 6 2,  -- stage3: downsample + 5        28→14
    .residualStage 256 512 3 2,  -- stage4: downsample + 2        14→7
    .globalAvgPool,
    .dense 512 1000 ]
  blurb := "ResNet-34 on full 1000-class ImageNet via the VERIFIED renderer → XLA/PJRT → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
  -- Same 36 BN layers, same order — the architecture is unchanged above the head.
  bnChannels := #[64,
    64,64, 64,64, 64,64,
    128,128,128, 128,128, 128,128, 128,128,
    256,256,256, 256,256, 256,256, 256,256, 256,256, 256,256,
    512,512,512, 512,512, 512,512]

-- The two nets differ in EXACTLY one parameter shape — the head. Anything else moving means the
-- ImageNet spec drifted from the Imagenette one it is supposed to be the 1000-class twin of.
#guard resnet34ImagenetVerified.toSpecs.size == resnet34Verified.toSpecs.size
#guard resnet34ImagenetVerified.toSpecs.pop.pop == resnet34Verified.toSpecs.pop.pop
#guard resnet34ImagenetVerified.toSpecs.back! == (#[1000], 2)

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
    .convBnNB 3 32 3 2,             -- stem (no conv bias — §2m)
    .invertedResidualNB 32  32  16 1,
    .invertedResidualNB 16  96  24 2, .invertedResidualNB 24 144  24 1,
    .invertedResidualNB 24 144  32 2, .invertedResidualNB 32 192  32 1, .invertedResidualNB 32 192  32 1,
    .invertedResidualNB 32 192  64 2, .invertedResidualNB 64 384  64 1, .invertedResidualNB 64 384  64 1, .invertedResidualNB 64 384  64 1,
    .invertedResidualNB 64 384  96 1, .invertedResidualNB 96 576  96 1, .invertedResidualNB 96 576  96 1,
    .invertedResidualNB 96 576 160 2, .invertedResidualNB 160 960 160 1, .invertedResidualNB 160 960 160 1,
    .invertedResidualNB 160 960 320 1,
    .convBnNB 320 1280 1 1,         -- head (no conv bias — §2m)
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

/-- **MobileNetV2 on full 1000-class ImageNet** — the fifth and last scale-tier spec (§2p).
    Identical architecture to `mobilenetv2Verified`; only the head moves (1280→1000), which takes
    the count to the JAX reference's 3,504,872.

    ⚠ A batch-BN net, so it needs `@mnv2in_fwd_eval` with frozen running stats, and its DP evidence
    comes from `shard-check` (which carries the 2×52-tensor stat region) rather than the plain
    duplicated-batch harness.

    ⚠ **§2g's warning applies to this net by name.** `mobilenetv2_fwd` is the artifact that was
    found to be the WRONG BN WORLD — batch-BN against a per-example-BN train step, so the trainer
    scored a different net than it trained (logits rel 1.86). That is why the forward pair here is
    rendered from the same chain the train step differentiates, under its own slug.

    ⚠ **Claim ceiling** (§5): proofs stop at Imagenette. And the recipe does not match — the
    reference uses **RMSProp at LR 0.045**, where this path is AdamW + cosine. -/
def mobilenetv2ImagenetVerified : VerifiedNetSpec where
  name     := "MobileNetV2 (ImageNet-1k)"
  slug     := "mnv2in"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  layers   := [
    .convBnNB 3 32 3 2,
    .invertedResidualNB 32  32  16 1,
    .invertedResidualNB 16  96  24 2, .invertedResidualNB 24 144  24 1,
    .invertedResidualNB 24 144  32 2, .invertedResidualNB 32 192  32 1, .invertedResidualNB 32 192  32 1,
    .invertedResidualNB 32 192  64 2, .invertedResidualNB 64 384  64 1, .invertedResidualNB 64 384  64 1, .invertedResidualNB 64 384  64 1,
    .invertedResidualNB 64 384  96 1, .invertedResidualNB 96 576  96 1, .invertedResidualNB 96 576  96 1,
    .invertedResidualNB 96 576 160 2, .invertedResidualNB 160 960 160 1, .invertedResidualNB 160 960 160 1,
    .invertedResidualNB 160 960 320 1,
    .convBnNB 320 1280 1 1,
    .globalAvgPool,
    .dense 1280 1000 ]
  blurb := "MobileNetV2 on full 1000-class ImageNet via the VERIFIED renderer → XLA/PJRT → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
  bnChannels := #[32,
    32,16,  96,96,24, 144,144,24,  144,144,32, 192,192,32, 192,192,32,
    192,192,64, 384,384,64, 384,384,64, 384,384,64,
    384,384,96, 576,576,96, 576,576,96,
    576,576,160, 960,960,160, 960,960,160,
    960,960,320,
    1280]

-- Exactly one parameter shape may differ (the head), and the BN layout must be IDENTICAL — the
-- running-stat region is positional, so a drift there misaligns every frozen statistic at eval.
#guard mobilenetv2ImagenetVerified.toSpecs.size == mobilenetv2Verified.toSpecs.size
#guard mobilenetv2ImagenetVerified.toSpecs.pop.pop == mobilenetv2Verified.toSpecs.pop.pop
#guard mobilenetv2ImagenetVerified.toSpecs.back! == (#[1000], 2)
#guard mobilenetv2ImagenetVerified.bnChannels == mobilenetv2Verified.bnChannels

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
    .convBnNB 3 32 3 2,            -- stem 3×3-s2
    .mbConvSENB  32   32  16  8 3,  -- s1 t1 (no expand)
    .mbConvSENB  16   96  24  4 3,  -- s2
    .mbConvSENB  24  144  24  6 3,
    .mbConvSENB  24  144  40  6 5,  -- s3
    .mbConvSENB  40  240  40 10 5,
    .mbConvSENB  40  240  80 10 3,  -- s4
    .mbConvSENB  80  480  80 20 3,
    .mbConvSENB  80  480  80 20 3,
    .mbConvSENB  80  480 112 20 5,  -- s5
    .mbConvSENB 112  672 112 28 5,
    .mbConvSENB 112  672 112 28 5,
    .mbConvSENB 112  672 192 28 5,  -- s6
    .mbConvSENB 192 1152 192 48 5,
    .mbConvSENB 192 1152 192 48 5,
    .mbConvSENB 192 1152 192 48 5,
    .mbConvSENB 192 1152 320 48 3,  -- s7
    .convBnNB 320 1280 1 1,         -- head 1×1 (320→1280)
    .globalAvgPool,
    .dense 1280 10 ]
  blurb := "EfficientNet-B0 on Imagenette 224² (stem-s2 → 16 MBConv [t,c,n,s,k], swish + squeeze-excite + batch-norm, 5 downsamples 224→7 → head 320→1280 → GAP → dense) via the VERIFIED renderer → IREE FFI → GPU"
  -- 49 BN layers in forward order (stem; per MBConv: expand-BN [t≠1 only], depthwise-BN, project-BN;
  -- head) — running-stats layout for trainAdamSched + @efficientnet_fwd_eval. Printed by
  -- TestEfficientNetTrain.bnChannelsList; true batch-norm makes batch-BN eval degenerate on sorted val.
  bnChannels := #[32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 40, 240, 240, 40, 240, 240, 80,
    480, 480, 80, 480, 480, 80, 480, 480, 112, 672, 672, 112, 672, 672, 112, 672, 672, 192,
    1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 320, 1280]
  -- ▶ STOCHASTIC DEPTH (`planning/stochastic_depth.md`), used only by the `*sd` variants.
  -- `keep_i = 1 − 0.2·i/(16−1)` at the NINE block indices that carry a skip: 2,4,6,7,9,10,12,13,14.
  --
  -- ⚠⚠ THE INDEX IS THE BLOCK INDEX, NOT THE SITE ORDINAL. The reference advances its ramp counter
  -- on EVERY MBConv block (`dbi := dbi + 1`, unconditional) while the drop fires only inside the
  -- skip guard — so the denominator is 15, not 8, and the nine keeps are UNEVENLY spaced. Deriving
  -- them from the site ordinal instead gives nine evenly-spaced keeps: it compiles, runs, descends
  -- and trains a different objective. §2k's α/K bug in a new place, and no numeric tie can see it,
  -- because every tie compares the render against a peer built from the same constants.
  dropKeeps := (#[2, 4, 6, 7, 9, 10, 12, 13, 14] : Array Nat).map
    (fun i => 1.0 - 0.2 * i.toFloat / 15.0)

/-- **EfficientNet-B0 on full 1000-class ImageNet** — the EfficientNet peer of the R34, ViT and
    ConvNeXt ImageNet specs (§2p). Identical architecture to `efficientnetVerified`; only the head
    moves (1280→1000), which takes the count to the JAX reference's 5,288,548.

    ⚠ **This is the first ImageNet net here with BatchNorm**, and that has two consequences the
    LayerNorm ones did not have: it needs a `_fwd_eval` artifact (frozen running stats — batch-BN
    eval is degenerate on a sorted validation split), and its data-parallel evidence cannot come
    from the plain duplicated-batch harness without the running-stat region, which is 2×49 extra
    tensors on both sides (§5 — omitting it is refused by the shim's G4 guard, not answered wrongly).

    ⚠ **Claim ceiling** (§5): proofs stop at Imagenette; provenance carries. And the recipe does
    not match — `efficientNetB0ImagenetConfig` trains with **RMSProp and exponential LR decay**
    (×0.97 every 2.4 epochs), where the verified path has AdamW + cosine. That is a bigger optimizer
    gap than ConvNeXt's or ViT's, and it is on top of the usual missing mixup/cutmix/EMA. -/
def efficientnetImagenetVerified : VerifiedNetSpec where
  name     := "EfficientNet-B0 (ImageNet-1k)"
  slug     := "enetin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  layers   := [
    .convBnNB 3 32 3 2,
    .mbConvSENB  32   32  16  8 3,
    .mbConvSENB  16   96  24  4 3,
    .mbConvSENB  24  144  24  6 3,
    .mbConvSENB  24  144  40  6 5,
    .mbConvSENB  40  240  40 10 5,
    .mbConvSENB  40  240  80 10 3,
    .mbConvSENB  80  480  80 20 3,
    .mbConvSENB  80  480  80 20 3,
    .mbConvSENB  80  480 112 20 5,
    .mbConvSENB 112  672 112 28 5,
    .mbConvSENB 112  672 112 28 5,
    .mbConvSENB 112  672 192 28 5,
    .mbConvSENB 192 1152 192 48 5,
    .mbConvSENB 192 1152 192 48 5,
    .mbConvSENB 192 1152 192 48 5,
    .mbConvSENB 192 1152 320 48 3,
    .convBnNB 320 1280 1 1,
    .globalAvgPool,
    .dense 1280 1000 ]
  blurb := "EfficientNet-B0 on full 1000-class ImageNet via the VERIFIED renderer → XLA/PJRT → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
  bnChannels := #[32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 40, 240, 240, 40, 240, 240, 80,
    480, 480, 80, 480, 480, 80, 480, 480, 112, 672, 672, 112, 672, 672, 112, 672, 672, 192,
    1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 320, 1280]
  -- ▶ v1.2c: the ImageNet peer of `efficientnetVerified.dropKeeps`. IDENTICAL, and that is the
  -- content: `enetDropIdxs` is a property of the ARCHITECTURE (16 MBConv blocks, 9 with skips), not
  -- of the class count or the batch, so the ramp does not move between scales.
  dropKeeps := (#[2, 4, 6, 7, 9, 10, 12, 13, 14] : Array Nat).map
    (fun i => 1.0 - 0.2 * i.toFloat / 15.0)

-- Exactly one parameter shape may differ — the head. And the BN layout must be IDENTICAL, since
-- the running-stat region is positional: a drift there misaligns every frozen statistic at eval.
#guard efficientnetImagenetVerified.toSpecs.size == efficientnetVerified.toSpecs.size
#guard efficientnetImagenetVerified.toSpecs.pop.pop == efficientnetVerified.toSpecs.pop.pop
#guard efficientnetImagenetVerified.toSpecs.back! == (#[1000], 2)
#guard efficientnetImagenetVerified.bnChannels == efficientnetVerified.bnChannels

-- Derived layout (262 params) == the audited hand-list EfficientNetLayout.specs.
#guard efficientnetVerified.toSpecs == EfficientNetLayout.specs

/-- ch9 **ConvNeXt-T** on Imagenette 224²: 4×4-s4 patchify → [3,3,9,3] ConvNeXt blocks @
    [96,192,384,768] (depthwise 7×7 → scalar-LN → 1×1 expand → GELU → 1×1 project → layerScale)
    with 3 between-stage (LN + 2×2-s2) downsamples (56→28→14→7) → GAP → LN → dense. 180 params.
    Tied at the FULL spec in `Proofs/SpecVJP.lean` (`convnextVerified_denote_eq` →
    `convNextForwardTCh`, the committed channel-LN config, + rung E
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
    .conv 3 96 4 4, .layerNorm 96,                                     -- patchify 4×4/s4 + stem LN
    .convNextBlockCh 96, .convNextBlockCh 96, .convNextBlockCh 96,     -- stage 1 (3) @56
    .layerNorm 96, .conv 96 192 2 2,                                   -- downsample 96→192  56→28
    .convNextBlockCh 192, .convNextBlockCh 192, .convNextBlockCh 192,  -- stage 2 (3) @28
    .layerNorm 192, .conv 192 384 2 2,                                 -- downsample 192→384 28→14
    .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,  -- stage 3 (9) @14
    .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
    .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
    .layerNorm 384, .conv 384 768 2 2,                                 -- downsample 384→768 14→7
    .convNextBlockCh 768, .convNextBlockCh 768, .convNextBlockCh 768,  -- stage 4 (3) @7
    .globalAvgPool, .dense 768 10 ]                                    -- head: GAP → dense
  blurb := "ConvNeXt-T on Imagenette 224² (patchify /4 → stem channel-LN → [3,3,9,3] blocks @ [96,192,384,768] depthwise-7×7 + channel-LN + GELU + layerScale + 3 downsamples 56→7 → GAP → dense) via the VERIFIED renderer → IREE FFI → GPU. LayerNorm is ConvNeXt's REAL channel LN — statistics over the c channels at each spatial position, per-channel [c] affine — on all 22 sites (§2m); the count matches the JAX reference at 28,587,592 for K=1000"

/-- **ConvNeXt-T on full 1000-class ImageNet** — the ConvNeXt peer of `resnet34ImagenetVerified`
    and `vitImagenetVerified` (handoff §2p). Identical architecture to `convnextVerified`; only the
    head moves (768→1000), which is what takes the count to the JAX reference's 28,587,592.

    Data comes from the generated tfds shim, so this side does no augmentation at all.

    ⚠ **Batch is 32 per device**, because `cBS` is still a private constant in the renderer while
    `nClasses` is now a parameter. At four replicas that is global 128 and 10,009 steps/epoch —
    more steps than the reference's 5,004 at batch 256, which §2d.2 says is the axis accuracy
    actually tracks. Threading `cBS` is a separate refactor, not a prerequisite.

    ⚠ **Claim ceiling**: the proof-carrying tier stops at Imagenette; what carries here is
    provenance plus whatever the pair comparison shows (§5). And this is **not** the ConvNeXt paper
    recipe — `convNeXtTinyImagenetConfig` also has mixup 0.8, cutmix 1.0, stochastic depth 0.1,
    EMA 0.9999, grad clip 1.0 and `wdExcludeNormBias`, none of which exist on the verified path.
    The pipeline augs (RandAugment geometric, random erasing) do come across free via the shim. -/
def convnextImagenetVerified : VerifiedNetSpec where
  name     := "ConvNeXt-T (ImageNet-1k)"
  slug     := "cnxin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  layers   := [
    .conv 3 96 4 4, .layerNorm 96,
    .convNextBlockCh 96, .convNextBlockCh 96, .convNextBlockCh 96,
    .layerNorm 96, .conv 96 192 2 2,
    .convNextBlockCh 192, .convNextBlockCh 192, .convNextBlockCh 192,
    .layerNorm 192, .conv 192 384 2 2,
    .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
    .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
    .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
    .layerNorm 384, .conv 384 768 2 2,
    .convNextBlockCh 768, .convNextBlockCh 768, .convNextBlockCh 768,
    .globalAvgPool, .dense 768 1000 ]
  blurb := "ConvNeXt-T on full 1000-class ImageNet via the VERIFIED renderer → XLA/PJRT → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"

-- The two ConvNeXt specs must differ in EXACTLY one parameter shape — the head. Anything else
-- moving means the ImageNet spec drifted from the Imagenette one it is the 1000-class twin of.
#guard convnextImagenetVerified.toSpecs.size == convnextVerified.toSpecs.size
#guard convnextImagenetVerified.toSpecs.pop.pop == convnextVerified.toSpecs.pop.pop
#guard convnextImagenetVerified.toSpecs.back! == (#[1000], 2)

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

/-- **ViT-Tiny on full 1000-class ImageNet** — the ViT peer of `resnet34ImagenetVerified`, and the
    scale tier of handoff §2p. Identical architecture to `vitVerified` above; the head is the only
    thing that moves (192→1000), exactly as the two ResNet-34 specs differ only in theirs.

    Data comes from the generated tfds shim (`VerifiedData.imagenet`), so **this side does no
    augmentation at all** — one definition of the transform, and it is the reference's.

    ⚠ **Claim ceiling, and it is lower here than the name suggests.** The proof-carrying tier stops
    at Imagenette: `vitVerified_denote_eq` / `vitVerified_has_vjp` / rung E are stated about the
    10-class net. What carries to this one is *provenance* — the artifacts are `pretty(provenGraph)`
    off the same renderer, since `nClasses`, `bs` and `replicas` are ordinary parameters of it —
    plus whatever a matched-pair comparison against `jax/MainVitImagenet.lean` shows. Say "one
    architecture, two independent lowerings, agreeing", never "proven" (§5).

    ⚠ It is **not** the DeiT recipe. `vitTinyImagenetConfig` carries mixup, cutmix, stochastic
    depth, EMA and grad clipping; none of those exist on the verified path (mixup/cutmix would need
    soft labels on the shim wire AND a `softLabelCE` cotangent — this render's is smoothed-CE over
    a one-hot). The pipeline-level augs do come across free. Do not compare a number from this to
    DeiT-Ti's 72.0%. -/
def vitImagenetVerified : VerifiedNetSpec where
  name     := "ViT-Tiny (ImageNet-1k)"
  slug     := "vitin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  layers   := [
    .conv 3 192 16 16,            -- patch embed 16×16/s16 (3→192)   224→14×14=196
    .param #[192] 2,              -- CLS token  [192]
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
    .dense 192 1000 ]             -- CLS-head 192→1000
  blurb := "ViT-Tiny on full 1000-class ImageNet via the VERIFIED renderer → XLA/PJRT → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"

-- The two ViT specs must differ in EXACTLY one parameter shape — the head. Anything else moving
-- means the ImageNet spec drifted from the Imagenette one it is supposed to be the 1000-class twin
-- of. This is the guard §2l wished it had had on R34: `resnet34Verified.toSpecs == specs` FIRED on
-- the conv-bias change and forced the hand-list to be fixed properly instead of edited to match.
#guard vitImagenetVerified.toSpecs.size == vitVerified.toSpecs.size
#guard vitImagenetVerified.toSpecs.pop.pop == vitVerified.toSpecs.pop.pop
#guard vitImagenetVerified.toSpecs.back! == (#[1000], 2)
