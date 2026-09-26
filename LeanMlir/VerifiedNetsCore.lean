import LeanMlir.VerifiedSpec
import LeanMlir.ParamLayouts

/-! # Concrete verified architectures — the shared specs

Readable layer-list specs that are referenced by **both** a trainer (`Main*Verified`)
and a proof (`LeanMlir/Proofs/*`). Kept in this light module (no Mathlib) so the proof
side can import the *exact* object the trainer runs — there's then a single source of
truth, and the spec the trainer runs is the object `SpecVJP` states its ties about.

Every verified spec lives here, including the ImageNet and sweep specs that no proof names. -/

/-- **The driver-side half of the MobileNetV2 / EfficientNet RMSProp recipe** — peak LR, the
    exponential decay `VerifiedNet.trainAdamSched` runs, and the warmup length.

    The *emitted* half (ρ, μ, ε, coupled wd) is `Proofs.StableHLO.RmsHyper`, which the renderers
    bake into each graph via `rmsConstsBlock`. These three are not graph constants: `%lr` is a
    runtime `tensor<f32>` argument so that one render serves a whole schedule, and a learning rate
    baked into a graph would be a hyperparameter no log records. Keeping the two halves in two
    modules keeps it that way.

    It lives in this shared-spec module because the Imagenette and ImageNet entry points of both
    nets read it, and one definition keeps their values from drifting apart.

    These are the reference's values at the reference's batch 256. The Imagenette callers scale
    `lr` by batch. -/
structure RmsSchedule where
  /-- `learningRate` — the peak, at batch 256. -/
  lr : Float
  /-- `expLRDecayRate` — the multiplier applied once per `decayEpochs`, after warmup. -/
  decayRate : Float
  /-- `expLRDecayEpochs` — how many epochs one multiplication spans (1 for MobileNetV2, 2.4 for
      EfficientNet-B0). -/
  decayEpochs : Float := 1.0
  /-- `warmupEpochs` — the linear ramp to `lr`. 5 unless a recipe says otherwise. -/
  warmup : Nat := 5
  /-- `expLRStaircase` — the exponent floored (TF's `staircase=True`). -/
  staircase : Bool := false

/-- **MobileNetV2**: 0.045 peak, ×0.98 **per epoch**, 5-epoch warmup, continuous — the schedule the
    Imagenette peers train with. -/
def mnv2RmsSchedule : RmsSchedule := { lr := 0.045, decayRate := 0.98 }

/-- **MobileNetV2 on ImageNet** ([`jax/MainMobilenetV2Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainMobilenetV2Imagenet.lean)): the paper's TF-slim
    schedule, ×0.98 per epoch as a staircase from step 0 with no warmup. -/
def mnv2ImagenetRmsSchedule : RmsSchedule := { mnv2RmsSchedule with warmup := 0, staircase := true }

/-- **EfficientNet-B0**: 0.016 peak, ×0.97 **every 2.4 epochs** — the paper's schedule, and the
    linear scaling of 0.256@4096 down to batch 256 ([`jax/MainEfficientNetImagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainEfficientNetImagenet.lean)). -/
def enetRmsSchedule : RmsSchedule := { lr := 0.016, decayRate := 0.97, decayEpochs := 2.4 }

/-- **EfficientNet-B0 on ImageNet**: TF's schedule, the ×0.97 / 2.4-epoch staircase on the global
    step with the 5-epoch warmup overriding it while it runs. -/
def enetImagenetRmsSchedule : RmsSchedule := { enetRmsSchedule with staircase := true }

/-- The Chapter-1 linear classifier: a single dense 784→10. Trained by
    `MainMnistLinearVerified`; its math VJP is proven in [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean)
    (`linearVerifiedHasVJP`) — both over *this* object. -/
def linearVerified : VerifiedNetSpec where
  name     := "MNIST-Linear"
  slug     := "linear"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 10]
  blurb    := "MNIST-Linear via the VERIFIED renderer (pretty∘emit) → %LOWERER% → GPU"

-- Shape tie: the derived param layout is W:[784,10] (He) + b:[10] (zeros).
#guard linearVerified.toSpecs == #[(#[784, 10], 0), (#[10], 2)]

/-- The Chapter-2 MLP: dense 784→512 → relu → dense 512→512 → relu → dense 512→10.
    Trained by `MainMnistMlpVerified`; its folded VJP is `mlpVerifiedHasVJPAt` in
    [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean),
    at an input where both ReLU pre-activations are nonzero — both over *this* object. -/
def mlpVerified : VerifiedNetSpec where
  name     := "MNIST-MLP"
  slug     := "mlp"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "MNIST-MLP via the VERIFIED renderer (784→512→512→10) → %LOWERER% → GPU"
  -- The chapter-2/3 loss carve-out: this render returns a trailing report-only `%loss`.
  lossSlot := true

-- Shape tie: W₀:[784,512] b₀:[512] | W₁:[512,512] b₁:[512] | W₂:[512,10] b₂:[10].
#guard mlpVerified.toSpecs ==
  #[(#[784, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- **Width-parametric MNIST MLP** `dense 784→d₁ → relu → dense d₁→d₂ → relu → dense d₂→10`.
    The canonical `mlpVerified` is `mlpG 512 512`. Every instance has the shape of
    `Proofs.mlpForward {d₀ d₁ d₂ d₃}`, whose folded VJP `Proofs.mlpHasVJPAt` is polymorphic in all
    four dims, so every `(d₁, d₂)` is an instance of that one definition. `mnist-mlp-grid` renders
    `.lake/build/mlp_{d₁}x{d₂}_{train_step,fwd}.mlir` (`mlirDir`, a build product) from the faithful
    renderer at run time and trains on it. Slug `mlp_{d₁}x{d₂}`. -/
def mlpG (d₁ d₂ : Nat) : VerifiedNetSpec where
  name     := s!"MNIST-MLP-{d₁}x{d₂}"
  slug     := s!"mlp_{d₁}x{d₂}"
  -- ⚠⚠ A BUILD PRODUCT, NOT A COMMITTED RENDER. This spec is instantiated from argv, renders its
  -- artifact at run time and immediately trains on it, so the file is regenerated every
  -- invocation. It wrote into `verified_mlir/` until 2026-08-03 and 74 such files had been checked
  -- in across the three sweep specs — never loaded by anything, and invisible to the writer audit,
  -- which greps for a LITERAL path while these writers interpolate the slug. `verified_mlir/` is
  -- now pinned to exactly the certified corpus (`regen_verified_mlir.sh check`), which it could not
  -- be while transients landed there. See `VerifiedNet.mlirDir`.
  mlirDir  := ".lake/build"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 d₁, .relu, .dense d₁ d₂, .relu, .dense d₂ 10]
  blurb    := s!"MNIST-MLP-{d₁}x{d₂} via the VERIFIED renderer (784→{d₁}→{d₂}→10) → %LOWERER% → GPU"
  -- ⚠ Same carve-out as `mlpVerified`: this renders through `mlpTrainStepFaithfulV`, which
  -- emits a trailing report-only `%loss`. Missing here from ff1ef3d (2026-08-12) until
  -- 2026-09-01, so every `mnist-mlp-grid` invocation died on `G4 VIOLATION: returns 7
  -- outputs, caller supplied 6` — the width sweep behind §2.5 has not been runnable since
  -- the day after its data was taken. The gate refused loudly; nothing else noticed.
  lossSlot := true

-- `mlpG 512 512` is exactly the canonical `mlpVerified` architecture.
#guard (mlpG 512 512).toSpecs == mlpVerified.toSpecs

/-- The Chapter-3 MNIST CNN (no BN): conv 1→32 → relu → conv 32→32 → relu → maxpool
    28→14 → flatten(6272) → dense 6272→512 → relu → dense 512→512 → relu → dense 512→10.
    Trained by `MainMnistCnnVerified`; [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean)
    ties it to `Proofs.mnistCnnNoBnForward` (`cnnVerified_denote_eq`), whose VJP folded through
    conv/maxpool/dense is `Proofs.mnistCnnNoBnHasVJPAt`, at an input satisfying its ReLU and
    max-pool hypotheses (`cnnVerifiedHasVJP` is the canonical witness). -/
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
  blurb    := "MNIST-CNN via the VERIFIED renderer (conv→conv→pool→512→512→10) → %LOWERER% → GPU"
  -- The chapter-2/3 loss carve-out: this render returns a trailing report-only `%loss`.
  lossSlot := true

-- Shape tie: conv0[32,1,3,3]+b | conv1[32,32,3,3]+b | dense 6272→512→512→10 (+biases).
#guard cnnVerified.toSpecs ==
  #[(#[32, 1, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[6272, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- **FC-width-parametric MNIST CNN** — the Chapter-3 CNN with the two convs held at 32
    channels (so the feature extractor is fixed) and the **dense classifier head** swept:
    `…maxpool → flatten(6272) → dense 6272→d → relu → dense d→d → relu → dense d→10`. The
    canonical `cnnVerified` is `cnnG 512`. The faithful CNN renderer (`cnnTrainStepFaithfulV`)
    takes a single dense width `d1` (both hidden FC layers share it), so every width renders
    through that renderer; `mnist-cnn-grid d` renders `.lake/build/cnn_{d}_{train_step,fwd}.mlir`
    (`mlirDir`, a build product) and trains on it. Isolates the ROI of the classifier head with the conv stack fixed. -/
def cnnG (d : Nat) : VerifiedNetSpec where
  name     := s!"MNIST-CNN-fc{d}"
  slug     := s!"cnn_{d}"
  -- ⚠⚠ A BUILD PRODUCT, NOT A COMMITTED RENDER. This spec is instantiated from argv, renders its
  -- artifact at run time and immediately trains on it, so the file is regenerated every
  -- invocation. It wrote into `verified_mlir/` until 2026-08-03 and 74 such files had been checked
  -- in across the three sweep specs — never loaded by anything, and invisible to the writer audit,
  -- which greps for a LITERAL path while these writers interpolate the slug. `verified_mlir/` is
  -- now pinned to exactly the certified corpus (`regen_verified_mlir.sh check`), which it could not
  -- be while transients landed there. See `VerifiedNet.mlirDir`.
  mlirDir  := ".lake/build"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.conv 1 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2, .flatten,
               .dense 6272 d, .relu, .dense d d, .relu, .dense d 10]
  blurb    := s!"MNIST-CNN-fc{d} via the VERIFIED renderer (conv32→conv32→pool→{d}→{d}→10) → %LOWERER% → GPU"
  -- ⚠ As `mlpG` above and `cnnVerified`: this render carries a trailing report-only `%loss`.
  -- Missing since ff1ef3d, so `mnist-cnn-grid` died on `G4 VIOLATION: returns 11 outputs,
  -- caller supplied 10`. Both grid drivers broke in the same commit and neither was run again.
  lossSlot := true

-- `cnnG 512` is exactly the canonical `cnnVerified` architecture.
#guard (cnnG 512).toSpecs == cnnVerified.toSpecs

/-- The Chapter-4 CIFAR-10 CNN (no BN): conv 3→32 → relu → conv 32→32 → relu → maxpool
    → conv 32→64 → relu → conv 64→64 → relu → maxpool → flatten(4096) → dense 4096→512
    → relu → dense 512→512 → relu → dense 512→10. VJP: `Proofs.cifarCnnHasVJPAt` (at a smooth
    point), tied to this spec by `cifarVerified_denote_eq` in `SpecVJP`. -/
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
  blurb    := "CIFAR-10 CNN via the VERIFIED renderer (3→32→32→pool→32→64→64→pool→512→512→10) → %LOWERER% → GPU"

#guard cifarVerified.toSpecs ==
  #[(#[32, 3, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[64, 32, 3, 3], 0), (#[64], 2), (#[64, 64, 3, 3], 0), (#[64], 2),
    (#[4096, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]

/-- The deeper **8-conv CIFAR-10 CNN (no BN)**, the backbone of the BatchNorm comparison: four
    `conv→conv→pool` stages, channels `[16,16,32,32]`, 32→16→8→4→2 spatial, then the
    3-dense head (`d1=64`): flatten 128 → 64 → relu → 64 → relu → 10. VJP:
    `Proofs.cifarCnn8HasVJPAt`, at a point off the ten ReLU kinks (eight conv, two dense) and
    satisfying the four max-pool conditions. -/
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
  blurb    := "Deeper CIFAR-10 CNN (8 convs, [16,16,32,32], 4 pools 32→2 → 128→64→64→10) via the VERIFIED renderer → %LOWERER% → GPU"

-- ⚠ `cifar8Bf16Verified` (slug `cifar8_bf16`) and `cifar8bVerified` (slug `cifar8b`) lived here
-- until 2026-09-20 and were removed with the six narrow-head trainer binaries that were their only
-- consumers. The artifacts they pointed at — `verified_mlir/cifar8_bf16{,_mom,_adam}_train_step.mlir`
-- and `cifar8b{,_bf16,_fp8}_adam_train_step.mlir` — are NOT removed: they are pure-Lean `#eval`
-- renders from `Proofs/Codegen/CnnRender.lean`, gated by .github/workflows/proofs.yml, and they
-- carry the §4.1 batched-op-family and §5.2 precision provenance. The wide-head peers
-- (`cifar8wVerified`, `cifar8wbVerified`, `cifar8w{,b}BnVerified`, below) are what Chapter 4 trains.

#guard cifar8Verified.toSpecs ==
  #[(#[16, 3, 3, 3], 0), (#[16], 2), (#[16, 16, 3, 3], 0), (#[16], 2),
    (#[16, 16, 3, 3], 0), (#[16], 2), (#[16, 16, 3, 3], 0), (#[16], 2),
    (#[32, 16, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[32, 32, 3, 3], 0), (#[32], 2), (#[32, 32, 3, 3], 0), (#[32], 2),
    (#[128, 64], 0), (#[64], 2), (#[64, 64], 0), (#[64], 2), (#[64, 10], 0), (#[10], 2)]

/-- The deeper **8-conv CIFAR-10 CNN with per-channel BatchNorm** — `cifar8Verified` + a
    `.bnPerChannel` after each of the 8 convs (γ=1/β=0 init, before relu). VJP:
    `Proofs.cifarCnnBn8HasVJPAt`, under `0 < εᵢ` ×8, the ten ReLU kinks (eight post-BN, two
    dense) and the four max-pool conditions. Per-channel BN is per-example ⇒ train=eval. -/
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
  blurb    := "Deeper CIFAR-10 CNN + per-channel BatchNorm (8× conv→BN→relu, [16,16,32,32], 4 pools → 128→64→64→10) via the VERIFIED renderer → %LOWERER% → GPU"

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
    `.lake/build/cifar8_bn_{d}_{adam_train_step,fwd}.mlir` (`mlirDir`, a build product, emitted by
    [`tests/TestCifar8AdamTrain.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestCifar8AdamTrain.lean) with `D1` a parameter). Per-channel BN ⇒ train=eval (no running
    stats, `bnChannels` empty). Slug `cifar8_bn_{d}`. -/
def cifar8BnG (d : Nat) : VerifiedNetSpec where
  name     := s!"CIFAR-CNN8-BN-fc{d}"
  slug     := s!"cifar8_bn_{d}"
  -- ⚠⚠ A BUILD PRODUCT, NOT A COMMITTED RENDER. This spec is instantiated from argv, renders its
  -- artifact at run time and immediately trains on it, so the file is regenerated every
  -- invocation. It wrote into `verified_mlir/` until 2026-08-03 and 74 such files had been checked
  -- in across the three sweep specs — never loaded by anything, and invisible to the writer audit,
  -- which greps for a LITERAL path while these writers interpolate the slug. `verified_mlir/` is
  -- now pinned to exactly the certified corpus (`regen_verified_mlir.sh check`), which it could not
  -- be while transients landed there. See `VerifiedNet.mlirDir`.
  mlirDir  := ".lake/build"
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
  blurb    := s!"CIFAR-CNN8-BN-fc{d} via the VERIFIED renderer (8× conv→BN→relu [16,16,32,32] → 128→{d}→{d}→10, AdamW) → %LOWERER% → GPU"

-- `cifar8BnG 64` is exactly the canonical `cifar8BnVerified` architecture.
#guard (cifar8BnG 64).toSpecs == cifar8BnVerified.toSpecs

/-- `cifar8Verified` with the MNIST-style **wide 2×512 dense head** (`d1=512`): flatten 128 →
    512 → relu → 512 → relu → 10. Same 8-conv backbone; the head jumps from 13K to 334K floats
    (whole net 52,858 → 373,626). Same parametric VJP `Proofs.cifarCnn8HasVJPAt` (the dense
    bridge is generic in width). Slug `cifar8w` (render [`LeanMlir/Proofs/Codegen/CnnRender.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Codegen/CnnRender.lean) at `d1 := 512`). -/
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
  blurb    := "Deeper CIFAR-10 CNN, MNIST-style wide head (8 convs, [16,16,32,32], 4 pools 32→2 → 128→512→512→10) via the VERIFIED renderer → %LOWERER% → GPU"


/-- **Wide head (d1=512) on the batched op family** — the net chapter 4's "Lever 3: the
    arithmetic" trains. Same net as `cifar8wVerified` (the one Levers 1–2 measure); only the
    slug differs, so it loads `verified_mlir/cifar8wb_<variant>_train_step.mlir`.

    The f32 and bf16 arms of Lever 3 both come from this slug and one renderer (`c8wbPacked`),
    differing only in the emit, which keeps the comparison controlled. The fp8 arm runs the f32
    graph with host-side E4M3, so it has no artifact of its own. -/
def cifar8wbVerified : VerifiedNetSpec :=
  { cifar8wVerified with
    name  := "CIFAR-CNN8-wide-batched"
    slug  := "cifar8wb"
    blurb := "Wide-head CIFAR-10 CNN (8 convs, BATCHED op family, d1=512) via the VERIFIED renderer → %LOWERER% → GPU" }
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
  blurb    := "Deeper CIFAR-10 CNN + per-channel BatchNorm, MNIST-style wide head (8× conv→BN→relu → 128→512→512→10) via the VERIFIED renderer → %LOWERER% → GPU"

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

/-- `cifar8wBnVerified` on the **BATCHED** op family. Slug `cifar8wb_bn`.

    Same net, same 38 parameters, same spec — the layer list is inherited verbatim, which is the
    point: only the op family the train step is rendered from moves. That is what makes bf16
    reachable on the normalized net (the bf16 ops exist only in the batched family), and what keeps the
    f32-vs-bf16 comparison a controlled one. BatchNorm stays per-example and f32 in both arms,
    so the eval forward is shared with `cifar8w_bn` unchanged. -/
def cifar8wbBnVerified : VerifiedNetSpec :=
  { cifar8wBnVerified with
    name  := "CIFAR-CNN8-wide-BN-batched"
    slug  := "cifar8wb_bn"
    blurb := "Wide-head CIFAR-10 CNN + per-channel BatchNorm (8 convs, BATCHED op family, d1=512) via the VERIFIED renderer → %LOWERER% → GPU" }
#guard cifar8wbBnVerified.toSpecs == cifar8wBnVerified.toSpecs

/-- Chapter 5 **ResNet-34** on Imagenette 224²: 7×7-s2 stem → BN → relu → maxpool →
    [3,4,6,3] basic-block stages (per-channel BN, strided downsample at the first block of
    stages 2–4) → GAP → dense. **110 param tensors** (no conv biases: every conv is BN-followed).
    Tied at the full spec in [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean)
    (`resnet34VerifiedB_denote_eq` → `Proofs.resnet34ForwardBFull` at batch BN, every batch size,
    and the forward-graph tie `resnet34VerifiedB_fwd_faithful`); the pointwise whole-net VJP is
    `Proofs.resnet34ForwardBFullHasVJPAt` (ResNet34FullBVJP.lean). -/
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
    .maxPool 3 2,                -- He et al. 3×3-s2, OVERLAPPING  112→56
    .residualStage  64  64 3 1,  -- stage1: 3 identity            @56
    .residualStage  64 128 4 2,  -- stage2: downsample + 3        56→28
    .residualStage 128 256 6 2,  -- stage3: downsample + 5        28→14
    .residualStage 256 512 3 2,  -- stage4: downsample + 2        14→7
    .globalAvgPool,
    .dense 512 10 ]
  blurb := "Real ResNet-34 on Imagenette 224² (7×7-s2 stem→3×3-s2 overlapping max pool→[3,4,6,3] blocks w/ batch-norm, He et al. option-B 1×1 projection shortcuts, no conv biases; 56→28→14→7→GAP→dense) via the VERIFIED renderer → %LOWERER% → GPU"
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

/-- **ResNet-34 on full 1000-class ImageNet.**

    Identical architecture to `resnet34Verified`; only the head width, the class count and the data
    source differ. It is run as a matched pair with [`jax/MainResnetImagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnetImagenet.lean)
    (same net, same heavy-ball + coupled-L2 recipe, same tfds augmentation via the generated shim).

    What is proved about it: the train-step capstone `Proofs.ResNet34TieB.r34_net_tiedB` binds the
    class count, so it covers this 1000-class head, at one replica, f32 and batch BatchNorm; the
    data-parallel step is `Proofs.ResNet34SyncTieB.r34_net_syncTiedB`. The `SpecVJP` ties
    (`resnet34VerifiedB_denote_eq`, `resnet34VerifiedB_fwd_faithful`) are stated at 10 classes.

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
  -- `resnet34ImagenetConfig` sets only `augment := true` ⇒ RandomResizedCrop + hflip, nothing else.
  -- This is the shim every OTHER net was streaming too, until `shimScript` existed.
  shimScript := "generated_resnet34_imagenet_shim.py"
  layers   := [
    .convBnNB 3 64 7 2,          -- 7×7-s2 stem → BN → relu       224→112 (no conv bias)
    .maxPool 3 2,                -- He et al. 3×3-s2, OVERLAPPING  112→56
    .residualStage  64  64 3 1,  -- stage1: 3 identity            @56
    .residualStage  64 128 4 2,  -- stage2: downsample + 3        56→28
    .residualStage 128 256 6 2,  -- stage3: downsample + 5        28→14
    .residualStage 256 512 3 2,  -- stage4: downsample + 2        14→7
    .globalAvgPool,
    .dense 512 1000 ]
  blurb := "ResNet-34 on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
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

/-! ### ResNet-50 — the bottleneck pair

    Rendered by Proofs/Codegen/ResNet50RenderB.lean (the `resnet50_*` and `resnet50in*` artifacts
    in `verified_mlir/`). The proof chain is Proofs/Nets/ResNet/ResNet50*.lean, with train-step
    capstone `Proofs.ResNet50TieB.r50_net_tiedB`; `SpecVJP` has no ResNet-50 tie, so these specs
    are pinned to the reference by the parameter-count `#guard`s below. -/

/-- Chapter 5 **ResNet-50 on Imagenette 224²** — the bottleneck sibling of `resnet34Verified`:
    7×7-s2 stem → BN → relu → pool → `[3,4,6,3]` bottleneck stages → GAP → dense.

    The stem pool is He et al.'s 3×3/s2 (`Proofs.StableHLO.SHlo.maxPool3s2F` / the `Proofs.StableHLO.BatchableOp.maxPool3s2`
    descriptor, denoting `Proofs.maxPool3s2Flat`), with symmetric padding 1 — the paper's window
    `[2i−1, 2i+1]`, not XLA `'SAME'`'s `[2i, 2i+2]`. A 2×2/s2 pool has the same output shape
    (112→56), so no arity or op-count check can tell the two apart; only the emitted window does. -/
def resnet50Verified : VerifiedNetSpec where
  name     := "ResNet-50 (Imagenette)"
  slug     := "resnet50"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .convBnNB 3 64 7 2,              -- 7×7-s2 stem → BN → relu        224→112 (no conv bias)
    .maxPool 3 2,                    -- He et al. 3×3-s2, OVERLAPPING  112→56
    .bottleneckStage   64  256 3 1,  -- stage1: project + 2 identity   @56  (64→256 at stride 1)
    .bottleneckStage  256  512 4 2,  -- stage2                         56→28
    .bottleneckStage  512 1024 6 2,  -- stage3                         28→14
    .bottleneckStage 1024 2048 3 2,  -- stage4                         14→7
    .globalAvgPool,
    .dense 2048 10 ]
  blurb := "ResNet-50 (bottleneck, v1.5 — stride on the 3×3) on Imagenette 224². LAYOUT SKELETON: no render, no proof chain, no artifact yet."
  -- 53 BN layers, in `conv_bn` call order: per block the three body convs, then the projection.
  bnChannels := #[64,
    -- stage1 @ mid 64, oc 256: proj-block (4) + 2 identity (3 each)
    64,64,256,256,  64,64,256,  64,64,256,
    -- stage2 @ mid 128, oc 512
    128,128,512,512,  128,128,512,  128,128,512,  128,128,512,
    -- stage3 @ mid 256, oc 1024
    256,256,1024,1024,  256,256,1024,  256,256,1024,  256,256,1024,
    256,256,1024,  256,256,1024,
    -- stage4 @ mid 512, oc 2048
    512,512,2048,2048,  512,512,2048,  512,512,2048]

/-- **ResNet-50 on full 1000-class ImageNet** — the verified peer of [`jax/MainResnet50Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnet50Imagenet.lean).
    Same backbone as `resnet50Verified`, head widened to 2048→1000.

    The reference's RSB-A3 `rsb-faithful` recipe runs LAMB at an effective batch of 2048
    (512 × 4 gradient accumulation). The verified driver accumulates in the `acc<k>x<B>` variants
    (`VerifiedVariant.accOn`, `VerifiedVariant.accK`), e.g. `resnet50in_accdp8x64` and
    `resnet50in160_lambaccdp8x64bce`. -/
def resnet50ImagenetVerified : VerifiedNetSpec where
  name     := "ResNet-50 (ImageNet-1k)"
  slug     := "resnet50in"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  shimScript := "generated_resnet50_imagenet_shim.py"   -- ⚠ NOT generated yet; scripts/gen_shims.sh
  layers   := [
    .convBnNB 3 64 7 2,
    .maxPool 3 2,                    -- He et al. 3×3-s2 — see `resnet50Verified`
    .bottleneckStage   64  256 3 1,
    .bottleneckStage  256  512 4 2,
    .bottleneckStage  512 1024 6 2,
    .bottleneckStage 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 1000 ]
  blurb := "ResNet-50 on full 1000-class ImageNet via the VERIFIED renderer. LAYOUT SKELETON: no render, no proof chain, no artifact yet."
  bnChannels := resnet50Verified.bnChannels
  -- ▶▶ **STOCHASTIC DEPTH, RSB-A2/A1's `dropPath := 0.05`** (2026-08-27). Sixteen sites, one per
  -- bottleneck block, on the residual branch — `bottleneck_block` drops `out` and leaves the
  -- shortcut alone.
  --
  -- ⭐ **The index IS the block index here, and unlike EfficientNet that is not a trap** —
  -- `efficientnetVerified.dropKeeps` carries `#[2,4,6,7,9,10,12,13,14]` because its reference
  -- advances `dbi` on every MBConv while the drop fires only inside a skip guard, so its keeps are
  -- UNEVENLY spaced over a denominator of 15. R50 has no such guard: every bottleneck block drops,
  -- `dbi` advances every block, and the ramp is dense. Stated because the two look identical and
  -- one of them is a `#[…]` literal for a reason.
  --
  -- ⚠ The denominator is `totalDrop − 1 = 15`, not 16 — `jax/Jax/Codegen.lean`'s
  -- `denom := Nat.max 1 (totalDrop - 1)`. So block 0 keeps 1.0 exactly and block 15 keeps 0.95;
  -- an off-by-one gives sixteen slightly-wrong keeps that train and descend. Checked against the
  -- regenerated reference's own call sites: `dpkeys[1], 0.996667` and `dpkeys[3], 0.990000`.
  dropKeeps := (Array.range 16).map (fun i => 1.0 - 0.05 * i.toFloat / 15.0)

-- ▶ §2k's precondition, and it is FREE here: the derived layout must total the reference's own
-- reported parameter count. `jax/.lake/build/generated_resnet50_imagenet.py` reports 25,557,032,
-- which is also torchvision's — the two conventions already agree because neither carries conv
-- biases (§2m). A mismatch here means the spec drifted from the net it is paired against, and it
-- is exactly the check whose ABSENCE let two different "ResNet-34"s ship (§2k/§2l).
#guard (resnet50ImagenetVerified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 25557032
#guard (resnet50Verified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 23528522
-- 53 BN layers = 1 stem + 16 blocks × 3 + 4 projections (all four stages project — stage 1
-- changes 64→256 at stride 1, where R34's stage 1 is ic = oc and needs none).
#guard resnet50Verified.bnChannels.size == 53
-- ⚠⚠ AND `bnChannels` IS A HAND-WRITTEN LITERAL WITH NOTHING TYING IT TO `layers`. Measured:
-- deleting a stage-3 block reddens all three counts above and leaves the `.size == 53` check
-- GREEN, because that array is not derived from anything. So pin it to the layout the way §2m
-- says — two independent routes, both gated. Every BN γ is the `(#[c], 1)` entry (initKind 1 =
-- ones), and `toSpecs` emits them in func-arg order, so filtering them out reproduces the BN
-- width list exactly. This is the audit whose ABSENCE let mnv2 ship a 158-param train step
-- against a 160-param eval forward (§2m).
#guard (resnet50Verified.toSpecs.filterMap
          (fun (d, k) => if k == 1 then some d[0]! else none)) == resnet50Verified.bnChannels
#guard (resnet50ImagenetVerified.toSpecs.filterMap
          (fun (d, k) => if k == 1 then some d[0]! else none)) == resnet50ImagenetVerified.bnChannels
-- 161 param tensors = 53 W + 53 γ + 53 β + dense {W, b}.
#guard resnet50ImagenetVerified.toSpecs.size == 161
-- The two differ in EXACTLY the head, the `resnet34in`/`resnet34` contract one net over.
#guard resnet50ImagenetVerified.toSpecs.size == resnet50Verified.toSpecs.size
#guard resnet50ImagenetVerified.toSpecs.pop.pop == resnet50Verified.toSpecs.pop.pop
#guard resnet50ImagenetVerified.toSpecs.back! == (#[1000], 2)

/-- **ResNet-50 on ImageNet-1k at RSB-A3's train resolution, 160²** — the same net as
    `resnet50ImagenetVerified`, fed 160² crops, which makes a 100-epoch A3 run shorter than at 224².

    Everything except `imageH`/`imageW`/`slug`/`shimScript` is the 224 spec's: `layers` is shared
    by construction below, so `toSpecs` — hence the 161 tensors and the 25,557,032 params — is
    derived from the same list. Resolution enters only through `d0 = 3·160·160 = 76,800`, and the
    `#guard`s under this definition pin exactly that.

    The shim is the `short` recipe's, not `default`'s:
      * `short` is timm's A3 ([`jax/MainResnet50Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnet50Imagenet.lean) — `trainRes := 160`,
        `testCropRatio := 0.95`, RandAugment m6, mixup 0.1 / cutmix 1.0). `default` is 224, so it
        cannot feed this net.
      * [`Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean) applies `trainRes` only inside `_imagenet_decode_random_crop_flip`
        (the train path); eval goes through `_imagenet_decode_center_crop` at `_IMG_SIZE = 224`.
        So this shim emits A3's 160/224 split — 76,800 floats on train, 150,528 on val.

    Eval runs at 224² through `resnet50in160_fwd_eval.mlir`, whose `%x` is `tensor<256x150528xf32>`;
    the driver reads the eval width off that artifact and passes it to `loadData` as `evalD0`. -/
def resnet50Imagenet160Verified : VerifiedNetSpec where
  name     := "ResNet-50 (ImageNet-1k, 160² train)"
  slug     := "resnet50in160"
  inC      := 3
  imageH   := 160
  imageW   := 160
  nClasses := 1000
  data     := .imagenet
  -- ⚠ The `short`/A3 recipe's shim (`--shim` writes `<out minus .py>_shim.py`), NOT `default`'s.
  shimScript := "generated_resnet50_imagenet_short_shim.py"
  -- ⭐ SHARED, not re-typed. A copied layer list is the double-writer failure this repo keeps
  -- paying for, and it is exactly what would let a "160 ResNet-50" drift into a different net —
  -- §2k/§2l's two-different-ResNet-34s, one resolution over.
  layers   := resnet50ImagenetVerified.layers
  blurb := "ResNet-50 on full 1000-class ImageNet at RSB-A3's 160² train resolution, via the VERIFIED renderer."
  bnChannels := resnet50Verified.bnChannels

-- ▶ §2.1's check, and it is the whole point: the 160 spec must be the SAME NET. Params, tensor
-- count and elementwise layout are resolution-independent (conv weights do not see spatial size,
-- and the dense head sits after GAP), so all three must match the 224 spec exactly.
#guard resnet50Imagenet160Verified.toSpecs == resnet50ImagenetVerified.toSpecs
#guard (resnet50Imagenet160Verified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 25557032
#guard resnet50Imagenet160Verified.toSpecs.size == 161
#guard (resnet50Imagenet160Verified.toSpecs.filterMap
          (fun (d, k) => if k == 1 then some d[0]! else none)) == resnet50Imagenet160Verified.bnChannels
-- ⭐ AND THE ONE THING THAT MUST DIFFER. Without this the guards above are all satisfied by simply
-- aliasing the 224 spec, which would render the whole definition a no-op. 3·160·160 = 76,800, and
-- it is the exact width `verified_mlir/resnet50in160_fwd.mlir` declares (`tensor<64x76800xf32>`).
#guard resnet50Imagenet160Verified.d0 == 76800
#guard resnet50ImagenetVerified.d0 == 150528

/-- **ResNet-50 at 224², streaming the 2018 recipe's augmentation.** The same net as
    `resnet50ImagenetVerified` — same slug, same renders, same artifacts, same `d0`. The only
    difference is which shim it streams.

    `shimScript` is a field on the net, not on the recipe. The other 224² R50 spec streams
    `generated_resnet50_imagenet_shim.py`, emitted from the `default` recipe, which is RSB-A2 and
    calls `_randaugment(img, 2, 7.0, 0.5)` on every training image. A 2018 run fed that shim would
    train 2018's optimizer and schedule on A2's augmentation, and would not be comparable to the
    JAX 2018 number. This spec streams the 2018 shim (random-resized-crop + hflip).

    `scripts/shim_wiring_gate.py` cannot catch a wrong recipe: it checks that each net streams
    its own shim rather than R34's, and there is no per-recipe slot for it to check. The last
    `#guard` below asserts this spec does not carry A2's shim.

    The shared `slug` is deliberate: `resnet50in_momdp64_train_step` and `resnet50in_fwd_eval` are
    the artifacts a 2018 run executes, and a fresh slug would orphan them. -/
def resnet50Imagenet2018Verified : VerifiedNetSpec :=
  { resnet50ImagenetVerified with
      name       := "ResNet-50 (ImageNet-1k, 2018 recipe)",
      shimScript := "generated_resnet50_imagenet_2018_shim.py",
      blurb      := "ResNet-50 on full 1000-class ImageNet with the 2018 recipe's augmentation \
                     (random-resized-crop + hflip, no RandAugment), via the VERIFIED renderer." }

-- ▶ It must be the SAME NET as the 224 spec in every way a render or an artifact can see.
#guard resnet50Imagenet2018Verified.toSpecs == resnet50ImagenetVerified.toSpecs
#guard resnet50Imagenet2018Verified.d0 == 150528
#guard resnet50Imagenet2018Verified.slug == resnet50ImagenetVerified.slug
-- ⭐ AND THE ONE THING THAT MUST DIFFER — the bug this spec exists to make unrepresentable.
#guard resnet50Imagenet2018Verified.shimScript != resnet50ImagenetVerified.shimScript
#guard resnet50Imagenet2018Verified.shimScript == "generated_resnet50_imagenet_2018_shim.py"

/-- **ResNet-50 at 224² with RSB-A1's augmentation** — the third `shimScript` on the 224 net.

    A1 differs from A2 in three fields ([`jax/MainResnet50Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnet50Imagenet.lean)'s
    `resnet50ImagenetConfigA1`): epochs 300 → 600, weight decay 0.02 → 0.01, and Mixup α
    0.1 → 0.2. The epoch count is a driver knob. The weight decay is a baked
    `stablehlo.constant`, so it is a separate render — the
    `resnet50in_emalambacc4x128wxclipdropbcewd001` and `…accdp4x128…wd001` renders and their bf16
    twins, kept on their own paths by `Proofs.StableHLO.wdVariantMark`. The Mixup α is data-side,
    and this spec carries it: `generated_resnet50_imagenet_a1_shim.py` differs from the `default`/A2 shim in the `_MIX_A`
    default, `0.1` → `0.2`.

    That line reads `float(os.environ.get('SHIM_MIXUP_ALPHA', …))`, so the α is also an
    environment override on the default shim. Setting A1's α that way leaves nothing in the run's
    log recording which α it trained on; a named shim the driver refuses to start without does.

    The shared `slug` is deliberate, for `resnet50Imagenet2018Verified`'s reason: the artifacts an
    A1 run executes are `resnet50in_*`, and a fresh slug would orphan them. -/
def resnet50ImagenetA1Verified : VerifiedNetSpec :=
  { resnet50ImagenetVerified with
      name       := "ResNet-50 (ImageNet-1k, RSB-A1)",
      shimScript := "generated_resnet50_imagenet_a1_shim.py",
      blurb      := "ResNet-50 on full 1000-class ImageNet with RSB-A1's augmentation \
                     (A2's pack at Mixup α 0.2), via the VERIFIED renderer." }

-- ▶ Same net as the 224 spec in every way a render or an artifact can see — A1 changes the DATA
-- and the baked decay, never the architecture.
#guard resnet50ImagenetA1Verified.toSpecs == resnet50ImagenetVerified.toSpecs
#guard resnet50ImagenetA1Verified.d0 == 150528
#guard resnet50ImagenetA1Verified.slug == resnet50ImagenetVerified.slug
-- ⭐ AND THE THINGS THAT MUST DIFFER. All three 224 recipes must name three DISTINCT shims, or one
-- of them is streaming another's augmentation — the §0.9 failure, one level in.
#guard resnet50ImagenetA1Verified.shimScript == "generated_resnet50_imagenet_a1_shim.py"
#guard resnet50ImagenetA1Verified.shimScript != resnet50ImagenetVerified.shimScript
#guard resnet50ImagenetA1Verified.shimScript != resnet50Imagenet2018Verified.shimScript
-- ⚠ Stated as a 3-element distinctness rather than two inequalities, so a FOURTH recipe cannot be
-- added by copying one of these and forgetting to change the shim.
#guard ([resnet50ImagenetVerified.shimScript, resnet50Imagenet2018Verified.shimScript,
         resnet50ImagenetA1Verified.shimScript].eraseDups).length == 3

-- ▶ The cross-net form of this invariant needs every `.imagenet` spec in scope, so it lives at the
-- END of this file (search "trainPix := net.d0").

/-- Chapter 6 **MobileNetV2** on Imagenette 224²: 3×3-s2 stem → BN → relu6 → 17 inverted-residual
    blocks (full-paper `[t,c,n,s]` config, strided depthwise downsamples, per-channel BN,
    relu6, linear bottleneck) → 1×1 head conv (320→1280) → BN → relu6 → GAP → dense.
    Tied at the full paper spec in [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean): `mobilenetv2VerifiedB_denote_eq`
    → `Proofs.mobilenetv2ForwardBFull` at batch BN, every batch size, and the forward-graph tie
    `mobilenetv2VerifiedB_fwd_faithful`. The whole-net VJP is
    `Proofs.mobilenetv2ForwardBFullHasVJPAt` (MobileNetV2FullBVJP.lean): stem, all 17 blocks and
    the head, pointwise — relu6 is kinked, so each of the 35 activation sites carries a
    `≠ 0 ∧ ≠ 6` side condition at every example. -/
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
  blurb := "MobileNetV2 on Imagenette 224² (stem-s2 → 17 inverted-residual blocks, full-paper [t,c,n,s] config, stride-2 depthwise downsamples 224→7 → head conv-BN-relu6 → GAP → LN → dense) via the VERIFIED renderer → %LOWERER% → GPU"
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

/-- **MobileNetV2 on full 1000-class ImageNet.** Identical architecture to
    `mobilenetv2Verified`; only the head moves (1280→1000), which takes the count to the JAX
    reference's 3,504,872.

    A batch-BN net, so it scores through `@mobilenetv2in_fwd_eval` with frozen running stats, and
    its data-parallel check is `shard-check` (which carries the 2×52-tensor stat region) rather
    than the plain duplicated-batch harness.

    Every MobileNetV2 forward, eval included, is rendered from `Proofs.StableHLO.mnv2FwdChainB`, the chain the
    train step differentiates, and both forwards are batch-BN because both train steps are; a
    forward from a different chain would score a different net than the one trained.

    What is proved about it: the train-step capstone `Proofs.MobileNetV2TieB.mnv2_net_tiedB` binds
    the class count, so it covers this 1000-class head, at one replica, f32 and batch BatchNorm;
    the data-parallel step is `Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB`. The `SpecVJP` ties
    are stated at 10 classes. The optimizer follows the variant: the `rms*` renders (the shipping
    `rmsdp64bf16`) are the reference's RMSProp at LR 0.045 with its warmup and ×0.98 exponential
    decay; the `adam*` renders are AdamW. -/
def mobilenetv2ImagenetVerified : VerifiedNetSpec where
  name     := "MobileNetV2 (ImageNet-1k)"
  slug     := "mobilenetv2in"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- ⚠ `mobilenetV2ImagenetConfig` sets `useAutoAugment := false` explicitly ("MNv2 paper used
  -- crop/flip only"), so this shim is R34's pipeline — MEASURED: the two generated files differ in
  -- exactly one line, the banner naming the reference. mnv2 is therefore the one net whose data
  -- stream this whole thread does not change, which is what makes it the inert control.
  shimScript := "generated_mobilenet_v2_imagenet_shim.py"
  -- Classifier dropout 0.2 (`mobilenetV2ImagenetConfig.dropout`), per ELEMENT on the 1280-wide GAP
  -- output. Read only by a variant carrying `do` (`rmsdp64wxdols0bf16`); the older renders have no
  -- mask slot and train exactly as before.
  dropoutKeep := some (0.8, 1280)
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
  blurb := "MobileNetV2 on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
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

/-- Chapter 7 **EfficientNet-B0** on Imagenette 224²: 3×3-s2 stem → 16 MBConv blocks (`[t,c,n,s,k]`
    B0 config; expand 1×1 [skip when t=1] → depthwise k×k → squeeze-excite → project 1×1, all
    BN + swish) → 1×1 head (320→1280) → GAP → dense. 213 param tensors, 4,020,358 scalars (the
    1000-class peer below is 5,288,548, i.e. B0's canonical 5.29M). The 16 `mbConvSE ic mid oc r k`
    args are the B0 generator unrolled (mid=t·ic, r=ic/4, ic threads stage→stage). Tied at the
    full spec in [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean) (`efficientnetVerified_denote_eq` →
    `Proofs.efficientnetForwardBFull`, batched ∀N, and the forward-graph tie
    `efficientnetVerified_fwd_faithful`); the full-depth VJP is
    `Proofs.efficientnetForwardBFullHasVJP` (global; its only hypotheses are the `0 < ε`
    positivities, `Proofs.B0Weights.EpsPos`). -/
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
  blurb := "EfficientNet-B0 on Imagenette 224² (stem-s2 → 16 MBConv [t,c,n,s,k], swish + squeeze-excite + batch-norm, 5 downsamples 224→7 → head 320→1280 → GAP → LN → dense) via the VERIFIED renderer → %LOWERER% → GPU"
  -- 49 BN layers in forward order (stem; per MBConv: expand-BN [t≠1 only], depthwise-BN, project-BN;
  -- head) — running-stats layout for trainAdamSched + @efficientnet_fwd_eval. Printed by
  -- TestEfficientNetTrain.bnChannelsList; true batch-norm makes batch-BN eval degenerate on sorted val.
  bnChannels := #[32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 40, 240, 240, 40, 240, 240, 80,
    480, 480, 80, 480, 480, 80, 480, 480, 112, 672, 672, 112, 672, 672, 112, 672, 672, 192,
    1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 320, 1280]
  -- ▶ STOCHASTIC DEPTH (`planning/archive/stochastic_depth.md`), used only by the `*sd` variants.
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
  -- ▶ CLASSIFIER DROPOUT (`recipe_gaps.md` gap C). `efficientNetB0ImagenetConfig` sets
  -- `dropout := 0.2` (`jax/MainEfficientNetImagenet.lean`), so keep = 0.8, and the width is the
  -- head's 1280 — the GAP output the classifier consumes, NOT `nClasses`.
  --
  -- ⚠⚠ IT IS PER-ELEMENT AND `dropKeeps` ABOVE IS PER-EXAMPLE, which is why this is a separate
  -- field and not another entry in that array. The reference draws `bernoulli(key, keep, x.shape)`
  -- here against `(B, 1, …, 1)` there. Folding this into `dropKeeps` would type-check on the Lean
  -- side and produce a `tensor<Bxf32>` mask for a `tensor<Bx1280xf32>` input — an arity/type
  -- refusal, which is the good outcome; the BAD one is the reverse, drawing B values and repeating
  -- them, which is stochastic depth on the classifier and is silent. See `F32.dropoutMask`.
  --
  -- ⚠ It is on the IMAGENETTE net too, and the ramp comment above says why the SD ramp does not
  -- move between scales; this is the same argument. `efficientNetB0Config` (Imagenette) does not
  -- itself set dropout — this carries the ImageNet reference's value so the `adamdo` render has a
  -- gate vehicle at the cheap scale, exactly as `dropKeeps` carries ImageNet's ramp. ⚠ So do NOT
  -- quote an Imagenette `adamdo` run as a reference comparison; it is a gate vehicle
  -- (`planning/archive/ema.md`'s finding 3, same shape).
  dropoutKeep := some (0.8, 1280)

/-- **EfficientNet-B0 on full 1000-class ImageNet** — the EfficientNet peer of the R34, ViT and
    ConvNeXt ImageNet specs. Identical architecture to `efficientnetVerified`; only the head
    moves (1280→1000), which takes the count to the JAX reference's 5,288,548.

    A BatchNorm net, which has two consequences the LayerNorm nets do not: it needs a `_fwd_eval`
    artifact (frozen running stats — batch-BN eval is degenerate on a sorted validation split),
    and its data-parallel check needs the running-stat region, 2×49 extra tensors on both sides
    (omitting it is refused by the shim's G4 guard).

    ⚠ **Claim ceiling** (§5): proofs stop at Imagenette; provenance carries. The recipe follows
    the variant: the `emarmsdp64dropdo*` renders (the shipping one is `emarmsdp64dropdobf16`) carry
    the reference's RMSProp with ×0.97-every-2.4-epoch decay, EMA, drop-connect and classifier
    dropout, as `efficientNetB0ImagenetConfig` trains; the `adam*` renders are AdamW + cosine. -/
def efficientnetImagenetVerified : VerifiedNetSpec where
  name     := "EfficientNet-B0 (ImageNet-1k)"
  slug     := "efficientnetin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- `efficientNetB0ImagenetConfig` sets `useAutoAugment := true` — the full ImageNet policy,
  -- geometric ops included. Streaming R34's shim here dropped it entirely.
  shimScript := "generated_efficientnet_b0_imagenet_shim.py"
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
  blurb := "EfficientNet-B0 on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
  bnChannels := #[32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 40, 240, 240, 40, 240, 240, 80,
    480, 480, 80, 480, 480, 80, 480, 480, 112, 672, 672, 112, 672, 672, 112, 672, 672, 192,
    1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 192, 1152, 1152, 320, 1280]
  -- ▶ v1.2c: the ImageNet peer of `efficientnetVerified.dropKeeps`. The SITES are identical —
  -- `enetDropIdxs` is a property of the ARCHITECTURE (16 MBConv blocks, 9 with skips) — but since
  -- 2026-09-25 the ramp is TF's `0.2 · i/16` (`efficientNetB0ImagenetConfig.dropPathOverN`), where
  -- the Imagenette peer keeps timm's `i/15`. Checked against the regenerated reference's call
  -- sites: `dpkeys[2], 0.975000` and `dpkeys[14], 0.825000`.
  -- ⚠ Host-fed, so it reaches every `…drop…` variant this driver runs, the older renders included.
  dropKeeps := (#[2, 4, 6, 7, 9, 10, 12, 13, 14] : Array Nat).map
    (fun i => 1.0 - 0.2 * i.toFloat / 16.0)
  -- The ImageNet peer, and IDENTICAL for the reason the drop SITES are: the mask width is the head's
  -- input (1280), which is a property of the ARCHITECTURE. Only the classifier's OUTPUT moves
  -- between scales (10 → 1000), and the dropout site sits before it.
  dropoutKeep := some (0.8, 1280)

-- Exactly one parameter shape may differ — the head. And the BN layout must be IDENTICAL, since
-- the running-stat region is positional: a drift there misaligns every frozen statistic at eval.
#guard efficientnetImagenetVerified.toSpecs.size == efficientnetVerified.toSpecs.size
#guard efficientnetImagenetVerified.toSpecs.pop.pop == efficientnetVerified.toSpecs.pop.pop
#guard efficientnetImagenetVerified.toSpecs.back! == (#[1000], 2)
#guard efficientnetImagenetVerified.bnChannels == efficientnetVerified.bnChannels

-- Derived layout (213 param TENSORS, 4,020,358 scalars) == the audited hand-list
-- EfficientNetLayout.specs. ⚠ The count read 262 here for a long time and no gate saw it, because
-- the `#guard` below compares the two ARRAYS, not either one against a number.
#guard efficientnetVerified.toSpecs == EfficientNetLayout.specs

/-- Chapter 8 **ConvNeXt-T** on Imagenette 224²: 4×4-s4 patchify → [3,3,9,3] ConvNeXt blocks @
    [96,192,384,768] (depthwise 7×7 → channel-LN → 1×1 expand → GELU → 1×1 project → layerScale)
    with 3 between-stage (LN + 2×2-s2) downsamples (56→28→14→7) → GAP → LN → dense.
    **182 param tensors, 27,827,818 scalars** (28,589,128 at K = 1000 —
    `timm.create_model('convnext_tiny')`'s count).
    Tied at the full spec in [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean) (`convnextVerified_denote_eq` →
    `Proofs.convNextForwardTCh`, the channel-LN net, and the forward-graph tie
    `convnextVerified_fwd_faithful`); the full-depth VJP is
    `Proofs.convNextForwardTChHasVJP`, with correctness theorem
    `Proofs.convNextForwardTChHasVJP_correct` (ConvNeXtFullT.lean). It is global rather than
    pointwise, because GELU has no kink. Its only hypotheses are the 23 LN positivities (stem +
    18 blocks + 3 downsamples + the head LN). -/
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
    .globalAvgPool, .layerNorm 768, .dense 768 10 ]                    -- head: GAP → LN → dense
  blurb := "ConvNeXt-T on Imagenette 224² (patchify /4 → stem channel-LN → [3,3,9,3] blocks @ [96,192,384,768] depthwise-7×7 + channel-LN + GELU + layerScale + 3 downsamples 56→7 → GAP → LN → dense) via the VERIFIED renderer → %LOWERER% → GPU. LayerNorm is ConvNeXt's REAL channel LN — statistics over the c channels at each spatial position, per-channel [c] affine — on all 22 of those sites (§2m), plus a 23rd over the [768] GAP output — the paper's head LN, restored 2026-08-30; the count matches timm at 28,589,128 for K=1000"
  -- ▶ STOCHASTIC DEPTH (`planning/archive/stochastic_depth.md`), used only by the `*drop` variants.
  -- `keep_i = 1 − 0.1·i/(18−1)` at EVERY block — ConvNeXt has one site per block and every block
  -- carries a residual, so unlike EfficientNet there is no skip guard and the site list is
  -- `0 … 17` entire.
  --
  -- ⚠⚠ THE DENOMINATOR IS 17, i.e. `totalDrop − 1` OVER THE WHOLE NET, and the index is the GLOBAL
  -- block index. The reference's `dbi` is one counter advanced once per `convnext_block` across all
  -- four stages (`emitForward`'s drop-path ramp in `jax/Jax/Codegen.lean`); re-indexing per stage would give four short ramps
  -- instead of one long one — it compiles, runs, descends and trains a different objective, and no
  -- numeric tie can see it, because every tie compares the render against a peer built from the same
  -- constants. `tests/TestDropPathRamp.lean` is what pins this against the renderer's
  -- `cnxBlockIdx`, from a THIRD reading of the reference.
  --
  -- ⚠ Block 0's keep is exactly 1.0, so `F32.dropScales` hands that site an exact 1.0 and the op is
  -- the identity in IEEE — which is the reference's `keep_prob < 1.0` guard, obtained as data rather
  -- than as a missing site.
  dropKeeps := (Array.range 18).map (fun i => 1.0 - 0.1 * i.toFloat / 17.0)

/-- **ConvNeXt-T on full 1000-class ImageNet** — the ConvNeXt peer of `resnet34ImagenetVerified`
    and `vitImagenetVerified`. Identical architecture to `convnextVerified`; only the head moves
    (768→1000), which is what takes the count to timm's 28,589,128.

    Data comes from the generated tfds shim, so this side does no augmentation at all. The
    committed data-parallel renders take 64 examples per replica (`%x : tensor<64x150528xf32>`),
    global 256 at four replicas, the reference's batch.

    What is proved about it: the train-step capstone `Proofs.CnxTiePoCGB.cnx_net_tiedGB` binds the
    class count, so it covers this 1000-class head, at one replica, in f32, on the chain without
    drop-path. The `SpecVJP` ties are stated at 10 classes.

    `convNeXtTinyImagenetConfig`'s extra knobs — mixup 0.8, cutmix 1.0, stochastic depth 0.1, EMA
    0.9999, grad clip 1.0 and `wdExcludeNormBias` — land as follows:
    * `wdExcludeNormBias`, grad clip and stochastic depth are render variants (`wx`, `clip`,
      `drop`), combined in `convnextin_adamdpwxclipdrop`.
    * EMA is a render variant too (`convnextin_ema`, `convnextin_emadp`), and
      `convnextin_emadpwxclipdropbf16` carries it together with `wx`, `clip` and `drop`.
    * Mixup and CutMix are data-side and ride the producer's `SHIM_MIX`, never the graph.
    These are variants, not layers, so a feature can be present with no constructor for it in the
    spec language: check `verified_mlir/` for the variant marker.
    The pipeline augmentations (geometric RandAugment, random erasing) come from this net's own
    shim (`shimScript`). -/
def convnextImagenetVerified : VerifiedNetSpec where
  name     := "ConvNeXt-T (ImageNet-1k)"
  slug     := "convnextin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- `convNeXtTinyImagenetConfig`: RandAugment m9/mstd0.5/inc1 (geometric) + random erasing p0.25.
  -- Mixup/CutMix are in that config too but ride the PRODUCER's `SHIM_MIX`, not the pipeline.
  shimScript := "generated_convnext_tiny_imagenet_shim.py"
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
    .globalAvgPool, .layerNorm 768, .dense 768 1000 ]
  blurb := "ConvNeXt-T on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
  -- The ImageNet peer of `convnextVerified.dropKeeps`. IDENTICAL, and that is the content: the ramp
  -- is a property of the ARCHITECTURE (18 blocks, one site each) and of `dropPath := 0.1`, neither
  -- of which moves with the class count or the batch. ⚠ Unlike the Imagenette render, this one is a
  -- REFERENCE recipe item — `convNeXtTinyImagenetConfig.dropPath := 0.1`, the ConvNeXt-T paper
  -- value — so `convnextin_adamdpwxclipdrop` is the first ConvNeXt artifact carrying every
  -- optimizer-and-regulariser knob its reference sets.
  dropKeeps := (Array.range 18).map (fun i => 1.0 - 0.1 * i.toFloat / 17.0)

/-- **ConvNeXt-Small on full ImageNet-1k** — ConvNeXt-T deepened: `[3,3,9,3] → [3,3,27,3]`,
    dims unchanged at `[96,192,384,768]`. The renderer takes the depth table as a parameter
    (`Proofs.StableHLO.CnxDims`), and the per-site certificates are generic in `c`/`e`/`h` and not
    indexed by depth, so the 18 extra blocks are further uses of the same theorems.

    **344 parameter tensors, 50,223,688 scalars** (the `#guard`s below), the published ConvNeXt-S
    size of 50.22M.

    ImageNet only: there is no ConvNeXt-S Imagenette peer.

    The stochastic-depth rate is the one recipe knob that moves with size, and it is data. The
    ConvNeXt paper uses 0.4 for S at 300 epochs against T's 0.1, so `dropKeeps` below is a steeper
    ramp over 36 sites, not the Tiny ramp with more entries. The render reads its drop scales from
    the driver's blob, so a rate change costs no artifact: `LEAN_MLIR_DROP_RATE_U` (micro-units,
    `200000` = 0.2), read by the ConvNeXt-S entry point
    (apps/imagenette/MainConvNeXtSImagenet.lean), overrides it per run.

    No accuracy has been measured: the artifacts render, the shapes tie and the count is
    `#guard`ed. -/
def convnextSImagenetVerified : VerifiedNetSpec where
  name     := "ConvNeXt-S (ImageNet-1k)"
  slug     := "convnextsin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- ⚠ ConvNeXt-T's shim, and that is correct rather than lazy: the shim is the DATA pipeline
  -- (RandAugment m9/mstd0.5/inc1 + random erasing p0.25), which the paper does not change between
  -- T and S. Only the model and the drop rate move.
  shimScript := "generated_convnext_tiny_imagenet_shim.py"
  layers   := [
    .conv 3 96 4 4, .layerNorm 96,
    .convNextBlockCh 96, .convNextBlockCh 96, .convNextBlockCh 96,
    .layerNorm 96, .conv 96 192 2 2,
    .convNextBlockCh 192, .convNextBlockCh 192, .convNextBlockCh 192,
    .layerNorm 192, .conv 192 384 2 2 ] ++
    -- stage 3: 27 blocks @384, the only thing that differs from `convnextImagenetVerified`.
    -- Spelled as a `replicate` rather than 27 copy-pasted lines: at this length a hand-list is a
    -- place for an off-by-one that the parameter count would catch only if someone read it.
    List.replicate 27 (VLayer.convNextBlockCh 384) ++
  [ .layerNorm 384, .conv 384 768 2 2,
    .convNextBlockCh 768, .convNextBlockCh 768, .convNextBlockCh 768,
    .globalAvgPool, .layerNorm 768, .dense 768 1000 ]
  blurb := "ConvNeXt-Small on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU (ConvNeXt-T deepened: stage 3 goes 9 → 27 blocks, dims unchanged at [96,192,384,768], 50.2M params)"
  -- ▶ 36 sites, denominator 35, and **rate 0.4** — the ConvNeXt paper's S value at 300 epochs,
  -- against T's 0.1. ⚠ This is the first `dropKeeps` in the repo that is not its Tiny peer's: the
  -- ramp SHAPE is architectural (one site per block, global index) but the RATE is per-size recipe.
  -- Copying T's 0.1 here would have been the silent-hyperparameter shape (§2a-quater) — it renders,
  -- trains and descends, at a regularisation strength the reference does not use for this size.
  dropKeeps := (Array.range 36).map (fun i => 1.0 - 0.4 * i.toFloat / 35.0)

-- 344 parameter tensors and 50,223,688 scalars: ConvNeXt-T's 182/28,589,128 plus 18 stage-3 blocks
-- at 9 tensors and 1,201,920 scalars each. S DEEPENS — it is the first net here added by adding
-- tensors rather than widening the ones that were there, which is the arithmetic ViT-S's
-- "same 200 tensors" claim is the mirror image of.
#guard convnextSImagenetVerified.toSpecs.size == 344
#guard convnextSImagenetVerified.toSpecs.size == convnextImagenetVerified.toSpecs.size + 18 * 9
#guard (convnextSImagenetVerified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 50223688
-- ⚠ The guards that tie this spec to the RENDERER's own depth table — that `cnxAllParams` at
-- `cnxSmall` is these same 342 tensors, and that `dropKeeps` has one entry per rendered site —
-- live in `tests/TestDropPathRamp.lean`, beside ConvNeXt-T's. They cannot live here: this module
-- is BELOW `Proofs/Codegen` in the import graph, which is the same reason `convnextVerified`'s ramp
-- is checked against `cnxBlockIdx` over there rather than next to its own definition.
-- ⚠ The ramp must NOT be Tiny's: 0.4 against 0.1 is a real recipe difference, so the last keep
-- is 0.6 here and 0.9 there. A guard on the SIZE alone would pass on a copied Tiny ramp.
#guard (convnextSImagenetVerified.dropKeeps[35]! - 0.6).abs < 1e-9
#guard (convnextImagenetVerified.dropKeeps[17]! - 0.9).abs < 1e-9

/-- **ConvNeXt-Base on full ImageNet-1k** — ConvNeXt-S's depth at `[128,256,512,1024]`.

    B moves the stem (96 → 128), the head (768 → 1024) and every stage width, so the renderer's
    depths and dims are one `Proofs.StableHLO.CnxDims` record: a net with S's depths and T's dims
    cannot be spelled.

    B shares S's depth table exactly (`[3,3,27,3]`, 36 blocks), so anything keying on block count
    cannot tell them apart.

    **344 parameter tensors, 88,591,464 scalars** (the `#guard`s below) — the same tensor count as
    S (B widens, it does not add), and the published 88.59M. The per-site certificates are generic
    in `c`/`e`/`h`, so B instantiates them at its four widths.

    Stochastic depth is 0.5 — the ConvNeXt paper's B value at 300 epochs, against S's 0.4 and T's
    0.1 — and is data, not a render knob.

    Nothing has been trained: the artifacts render, the shapes tie and the count is `#guard`ed. -/
def convnextBImagenetVerified : VerifiedNetSpec where
  name     := "ConvNeXt-B (ImageNet-1k)"
  slug     := "convnextbin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- ConvNeXt-T's shim: the paper does not change the DATA pipeline across T/S/B, only the model
  -- and the drop rate. Same reasoning as ConvNeXt-S's.
  shimScript := "generated_convnext_tiny_imagenet_shim.py"
  layers   := [
    .conv 3 128 4 4, .layerNorm 128,
    .convNextBlockCh 128, .convNextBlockCh 128, .convNextBlockCh 128,
    .layerNorm 128, .conv 128 256 2 2,
    .convNextBlockCh 256, .convNextBlockCh 256, .convNextBlockCh 256,
    .layerNorm 256, .conv 256 512 2 2 ] ++
    List.replicate 27 (VLayer.convNextBlockCh 512) ++
  [ .layerNorm 512, .conv 512 1024 2 2,
    .convNextBlockCh 1024, .convNextBlockCh 1024, .convNextBlockCh 1024,
    .globalAvgPool, .layerNorm 1024, .dense 1024 1000 ]
  blurb := "ConvNeXt-Base on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU (ConvNeXt-S's [3,3,27,3] depth at [128,256,512,1024], 88.6M params)"
  -- 36 sites, denominator 35, rate **0.5** — the paper's B value. ⚠ Same ramp SHAPE as S, different
  -- rate: the shape is architectural, the rate is per-size recipe. Three sizes, three rates.
  dropKeeps := (Array.range 36).map (fun i => 1.0 - 0.5 * i.toFloat / 35.0)

-- Same 344 tensors as S — B widens every one and adds none, which is the width-generic claim
-- stated as arithmetic. 88,591,464 scalars against S's 50,223,688.
#guard convnextBImagenetVerified.toSpecs.size == convnextSImagenetVerified.toSpecs.size
#guard convnextBImagenetVerified.toSpecs.size == 344
#guard (convnextBImagenetVerified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 88591464
-- ⚠ And every shape must actually MOVE. S and B have identical tensor counts and identical drop
-- ramps in shape, so a spec that accidentally copied S's widths would pass both guards above.
#guard convnextBImagenetVerified.toSpecs != convnextSImagenetVerified.toSpecs
#guard convnextBImagenetVerified.dropKeeps.size == convnextSImagenetVerified.dropKeeps.size
#guard (convnextBImagenetVerified.dropKeeps[35]! - 0.5).abs < 1e-9
#guard (convnextBImagenetVerified.dropKeeps[35]! - convnextSImagenetVerified.dropKeeps[35]!).abs > 1e-3

-- The two ConvNeXt specs must differ in EXACTLY one parameter shape — the head. Anything else
-- moving means the ImageNet spec drifted from the Imagenette one it is the 1000-class twin of.
#guard convnextImagenetVerified.toSpecs.size == convnextVerified.toSpecs.size
#guard convnextImagenetVerified.toSpecs.pop.pop == convnextVerified.toSpecs.pop.pop
#guard convnextImagenetVerified.toSpecs.back! == (#[1000], 2)

-- Derived layout (182 params, head LN included since 2026-08-30) == the audited hand-list.
#guard convnextVerified.toSpecs == ConvNeXtLayout.specs

/-- Chapter 9 **ViT-Tiny** on Imagenette 224² (patch-16): 16×16-s16 conv patch embed (3→192,
    →196 patches), learned CLS token + positional embed (→197 tokens), 12 pre-norm transformer
    blocks (dim 192, 3 heads, MLP 768), final per-channel LayerNorm, CLS-slice dense head 192→10.
    200 params. Tied at the full spec in [`Proofs/SpecVJP.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/SpecVJP.lean) (`vitVerified_denote_eq` →
    `Proofs.vitForwardKV` at depth 12 with distinct per-block parameters and vector LN), with the
    whole-net VJP `vitVerifiedHasVJP` (global, `0 < ε` only) and the forward-graph tie
    `vitVerified_fwd_faithful` (the depth-12 multi-head vector-LN graph
    `Proofs.StableHLO.vitFwdGraphKMHV`). -/
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
  blurb := "ViT-Tiny on Imagenette 224² (patch-16 → CLS+pos → 12 transformer blocks @ dim192/3heads/MLP768 → final LN → CLS-head 10) via the VERIFIED renderer → %LOWERER% → GPU"
  -- ▶ STOCHASTIC DEPTH (`planning/archive/stochastic_depth.md`), used only by the `*drop` variants.
  -- ⚠⚠ **24 ENTRIES FOR 12 KEEPS, AND THE PAIRING IS THE CONTENT.** ViT drops each block's TWO
  -- residual branches INDEPENDENTLY (`ka, km = jax.random.split(drop_key)`) but at the SAME keep
  -- probability, so site `2i` and site `2i+1` share `keep_i = 1 - 0.1*i/11`. The driver needs one
  -- entry per SITE because it draws one Bernoulli stream per mask INPUT; deriving 24 evenly-spaced
  -- keeps from the site ordinal instead would be a different objective, and emitting ONE mask per
  -- block would halve the noise. Neither is visible in any structural check
  -- (`stochastic_depth.md` §6.3); `tests/TestDropPathRamp.lean` is what pins it.
  -- ⚠ The denominator is 11 = 12 blocks - 1, and block 11 keeps exactly `1 - dropPath` — unlike
  -- EfficientNet, whose deepest SITE is one ramp step short because its last block has no skip.
  dropKeeps := (Array.range 24).map (fun sIdx => 1.0 - 0.1 * (sIdx / 2).toFloat / 11.0)

-- Derived layout (200 params) == the audited hand-list ViTLayout.specs.
#guard vitVerified.toSpecs == ViTLayout.specs

/-- **ViT-Tiny on full 1000-class ImageNet** — the ViT peer of `resnet34ImagenetVerified`.
    Identical architecture to `vitVerified` above; the head is the only thing that moves
    (192→1000), exactly as the two ResNet-34 specs differ only in theirs.

    Data comes from the generated tfds shim (`VerifiedData.imagenet`), so this side does no
    augmentation at all — one definition of the transform, and it is the reference's.

    What is proved about it: the train-step capstone `Proofs.ViTTiePoCGB.vit_net_tiedGB` binds the
    class count, so it covers this 1000-class head, at one replica, in f32, on the chain without
    drop-path. `vitVerified_denote_eq`, `vitVerifiedHasVJP` and `vitVerified_fwd_faithful` are
    stated at 10 classes. The matched-pair reference is [`jax/MainVitImagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainVitImagenet.lean).

    The recipe follows the variant. The shipping `emadp128x4wxclipdropbf16` (bf16, with drop-path,
    so outside the capstone's scope) carries the rest of `vitTinyImagenetConfig`: EMA, grad clip
    1.0, drop-path (24 host-drawn masks), weight decay off norm/bias, and mixup/cutmix, which ride
    the producer's `SHIM_MIX` (this shim bakes `both`) as soft targets on the wire; the render's
    cotangent smooths that mixed target (α = 0.1). The pipeline-level augmentations (RandAugment,
    random erasing, repeated aug ×3) come from this net's own shim (`shimScript`). The remaining
    differences from DeiT-Ti (clip, EMA-scored eval, LN eps, tanh GELU) are listed in
    planning/imagenet_parity.md. -/
def vitImagenetVerified : VerifiedNetSpec where
  name     := "ViT-Tiny (ImageNet-1k)"
  slug     := "vitin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- `vitTinyImagenetConfig` is the DeiT suite: RandAugment m9/mstd0.5/inc1 (geometric) + random
  -- erasing p0.25 + **repeated augmentation ×3**. The last one changes the STREAM, not just the
  -- transform — an epoch sees ~1/3 the unique images at 3 views each — so it is the one gap here
  -- that a per-image comparison would not have shown.
  shimScript := "generated_vit_tiny_imagenet_shim.py"
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
  blurb := "ViT-Tiny on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU, with the tfds batch shim supplying the same augmentation the Lean→JAX reference trainer uses"
  -- The ImageNet peer of `vitVerified.dropKeeps`. IDENTICAL — the ramp is a property of the
  -- ARCHITECTURE (12 blocks, 2 sites each) and of `dropPath := 0.1`, neither of which moves with the
  -- class count. ⚠ Here it IS a reference recipe item (`vitTinyImagenetConfig.dropPath`, the DeiT
  -- value), where on Imagenette it is a gate vehicle.
  dropKeeps := (Array.range 24).map (fun sIdx => 1.0 - 0.1 * (sIdx / 2).toFloat / 11.0)

/-- **ViT-Small on full ImageNet-1k** — ViT-Tiny widened.

    `Proofs.vitForwardKVHasVJP` is stated for all `heads d_head mlpDim k`, and it is a global
    `HasVJP` rather than the pointwise `_at` form the ReLU-family nets carry, because
    GELU/softmax/LayerNorm have no kink. So S is covered by the same definition as Tiny, at
    different arguments.

    S is Tiny widened and nothing else: `D = 384 = 6 heads × 64` against Tiny's `192 = 3 × 64`, MLP
    1536 against 768. Same depth (12), same 16×16 patch grid (196 tokens + CLS), same block
    structure. `d_head` stays 64 — ViT widens by adding heads.

    ImageNet only: there is no ViT-S Imagenette peer.

    No accuracy has been measured: the artifacts render and the shapes tie. -/
def vitSImagenetVerified : VerifiedNetSpec where
  name     := "ViT-Small (ImageNet-1k)"
  slug     := "vitsin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  shimScript := "generated_vit_tiny_imagenet_shim.py"
  layers   := [
    .conv 3 384 16 16,            -- patch embed 16×16/s16 (3→384)   224→14×14=196
    .param #[384] 2,              -- CLS token  [384]
    .param #[197, 384] 2,         -- positional embedding  [197,384]
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .transformerBlock 384 1536,
    .layerNorm 384,               -- final LayerNorm (per-channel [384])
    .dense 384 1000 ]             -- CLS-head 384→1000
  blurb := "ViT-Small on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU (Tiny widened: D 384 = 6 heads x 64, MLP 1536, same depth 12)"
  -- Identical to Tiny's: the ramp is a property of the DEPTH (12 blocks, 2 sites each) and of
  -- `dropPath := 0.1`. Width does not enter it, which is why widening needs no new ramp.
  dropKeeps := (Array.range 24).map (fun sIdx => 1.0 - 0.1 * (sIdx / 2).toFloat / 11.0)

-- 200 parameter tensors, the SAME count as ViT-Tiny — S widens every tensor and adds none, which
-- is the whole claim of the depth-k/width-generic renderer stated as arithmetic. 22,050,664
-- parameters against Tiny's 5,717,416; ViT-S/16 is quoted at ~22M.
#guard vitSImagenetVerified.toSpecs.size == vitImagenetVerified.toSpecs.size
#guard vitSImagenetVerified.toSpecs.size == 200
#guard (vitSImagenetVerified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 22050664

/-- **ViT-Base (DeiT-B) on full ImageNet-1k.** `D = 768 = 12 heads × 64`, MLP 3072, still depth 12
    and still 16×16 patches — a third `Proofs.StableHLO.VitDims` for the same renderer.

    The two `vitbin_adamdp128x4wxclipdrop*` renders run on four cards at global 512, DeiT's
    batch. The fp32 render needs the PJRT allocator fraction raised (`LEAN_MLIR_MEM_FRACTION=0.97`,
    15.11 GiB; the default 0.75 gives 11.68 GiB), and `runViTBImagenet` refuses to start the fp32
    render without it; the bf16 twin fits the default arena. There is no smaller-batch data-parallel
    render.

    Neither precision has been trained. -/
def vitBImagenetVerified : VerifiedNetSpec where
  name     := "ViT-Base (ImageNet-1k)"
  slug     := "vitbin"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  shimScript := "generated_vit_tiny_imagenet_shim.py"
  layers   := [
    .conv 3 768 16 16,
    .param #[768] 2,
    .param #[197, 768] 2,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .transformerBlock 768 3072,
    .layerNorm 768,
    .dense 768 1000 ]
  blurb := "ViT-Base on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU"
  dropKeeps := (Array.range 24).map (fun sIdx => 1.0 - 0.1 * (sIdx / 2).toFloat / 11.0)

-- 86,567,656 parameters, DeiT-B's published 86.57M, in the SAME 200 tensors as Ti and S. Three
-- widths, one renderer, one theorem. `jax/MainVitBImagenet.lean` emits the same count from an
-- independent implementation (`planning/archive/vit_convnext_sb_scaleup.md`).
#guard vitBImagenetVerified.toSpecs.size == vitImagenetVerified.toSpecs.size
#guard (vitBImagenetVerified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 86567656

-- The two ViT specs must differ in EXACTLY one parameter shape — the head. Anything else moving
-- means the ImageNet spec drifted from the Imagenette one it is supposed to be the 1000-class twin
-- of. This is the guard §2l wished it had had on R34: `resnet34Verified.toSpecs == specs` FIRED on
-- the conv-bias change and forced the hand-list to be fixed properly instead of edited to match.
#guard vitImagenetVerified.toSpecs.size == vitVerified.toSpecs.size
#guard vitImagenetVerified.toSpecs.pop.pop == vitVerified.toSpecs.pop.pop
#guard vitImagenetVerified.toSpecs.back! == (#[1000], 2)

/-! ### MobileNetV4-Conv-M — the Universal Inverted Bottleneck -/

/-- **MobileNetV4-Conv-M on Imagenette 224²** (the book's MobileNetV4 side quest, chapter 6) —
    a trunk built from one parameterised block. `uib`'s `k = 0` omits a depthwise, so the same
    constructor renders all four MNv4 families — ExtraDW (both DWs), IB / MBConv (post only),
    ConvNeXt-like (pre only) and FFN (neither) — and the fused stage is the only other block form
    in the net. 21 UIB blocks, 233 parameter tensors, 8,447,322 parameters (the `#guard`s below).
    This spec has no Imagenette accuracy run of its own.

    The two MNv4 specs move together: `mnv4ImagenetVerified` takes its `bnChannels` from this one
    and `#guard`s its `toSpecs` against it. [`jax/MainMobilenetV4.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainMobilenetV4.lean)
    is the reference the ties read.

    A pre/post-DW swap is invisible to everything in this file: same `k`, same channels ⇒ same
    `toSpecs`, so the `#guard`s below pass on a spec that swaps them, and at stride 1 both
    positions are shape-preserving so the types pass too. What pins the order is
    `scripts/parity/mnv4_forward_tie.py` against the JAX reference on shared weights, and what pins
    the backward's dispatch is `scripts/parity/grad_tie.py --net mnv4`. R50's stride-on-the-3×3
    is invisible in the same way.

    The net is timm 1.0.28's `mobilenetv4_conv_medium`: stride on the post-DW, a BN-only pre-DW, a
    ReLU stage 0, GAP before `conv_head`, a symmetric stem. Three gates check it:
    `scripts/parity/mnv4_timm_parity.py` (the JAX reference against timm, logits to 1.5e-5
    relative); `scripts/parity/mnv4_forward_tie.py` (this render against the JAX reference,
    `max |Δ| = 1.767e-05` at B = 2); `scripts/parity/grad_tie.py --net mnv4 --nokink` at B = 8
    (0 of 201 live parameters worse than 10× the control; the two precision-limited head
    parameters are exempt there and checked by the default mode). -/
def mobilenetv4Verified : VerifiedNetSpec where
  name     := "MobileNetV4-Conv-M"
  slug     := "mnv4"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 10
  data     := .imagenette
  layers   := [
    .convBnNB 3 32 3 2,                 -- stem, 224→112 (symmetric pad, timm `conv_stem`)
    .fusedMbConvNB 32 48 4 3 2,         -- fused stage, 112→56 (relu, timm `EdgeResidual`)
    .uib  48  80 4 2 3 5,               -- ExtraDW  56→28
    .uib  80  80 2 1 3 3,               -- ExtraDW  28
    .uib  80 160 6 2 3 5,               -- ExtraDW  28→14
    .uib 160 160 4 1 3 3,               -- ExtraDW  14
    .uib 160 160 4 1 3 3,               -- ExtraDW  14
    .uib 160 160 4 1 3 5,               -- ExtraDW  14
    .uib 160 160 4 1 3 3,               -- ExtraDW  14
    .uib 160 160 4 1 3 0,               -- ConvNeXt 14
    .uib 160 160 2 1 0 0,               -- FFN      14
    .uib 160 160 4 1 3 0,               -- ConvNeXt 14
    .uib 160 256 6 2 5 5,               -- ExtraDW  14→7
    .uib 256 256 4 1 5 5,               -- ExtraDW  7
    .uib 256 256 4 1 3 5,               -- ExtraDW  7
    .uib 256 256 4 1 3 5,               -- ExtraDW  7
    .uib 256 256 4 1 0 0,               -- FFN      7
    .uib 256 256 4 1 3 0,               -- ConvNeXt 7
    .uib 256 256 2 1 3 5,               -- ExtraDW  7
    .uib 256 256 4 1 5 5,               -- ExtraDW  7
    .uib 256 256 4 1 0 0,               -- FFN      7
    .uib 256 256 4 1 0 0,               -- FFN      7
    .uib 256 256 2 1 5 0,               -- ConvNeXt 7
    .convBnNB 256 960 1 1,              -- head conv 1 (Conv-M `cn_r1_k1_s1_c960`)
    .globalAvgPool,                     -- timm pools BEFORE conv_head
    .convBnNB 960 1280 1 1,             -- head conv 2 (`conv_head` on the pooled features)
    .dense 1280 10 ]
  blurb := "MobileNetV4-Conv-M on Imagenette 224² (stem-s2 → fused MBConv → 21 Universal Inverted Bottleneck blocks spanning all four families from ONE constructor, 224→7 → head conv 256→960 → GAP → conv_head 960→1280 → dense, timm order) via the VERIFIED renderer → %LOWERER% → GPU"
  -- 52 BN layers in forward order: stem; the fused stage's k×k-BN and project-BN; then per UIB
  -- block pre-DW-BN (if preDWk≠0) / expand-BN / post-DW-BN (if postDWk≠0) / project-BN; head.
  -- ⚠ The `if`s are the `k = 0` family dispatch, so this list's LENGTH varies per block (2, 3 or 4)
  -- — which is exactly why it is `#guard`ed against `toSpecs` below rather than eyeballed.
  bnChannels := #[32,
    128, 48,
    48, 192, 192, 80,        80, 160, 160, 80,        80, 480, 480, 160,
    160, 640, 640, 160,      160, 640, 640, 160,      160, 640, 640, 160,
    160, 640, 640, 160,      160, 640, 160,           320, 160,
    160, 640, 160,
    160, 960, 960, 256,      256, 1024, 1024, 256,    256, 1024, 1024, 256,
    256, 1024, 1024, 256,    1024, 256,               256, 1024, 256,
    256, 512, 512, 256,      256, 1024, 1024, 256,    1024, 256,
    1024, 256,               256, 512, 256,
    960, 1280]

-- 233 parameter tensors — the same count `mnv4-fwd-smoke` ties to `mnv4ShapeList` shape-for-shape,
-- and 8,447,322 parameters. ⭐ `jax/MainMobilenetV4.lean`'s `totalParams` reads 8447322 too, from
-- a wholly independent implementation, which is the cross-check that the Conv-M transcription is
-- the same net on both sides.
#guard mobilenetv4Verified.toSpecs.size == 233
#guard (mobilenetv4Verified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 8447322
-- ⭐ Every conv in this net is BN-followed, so the BN channel list must be, in forward order, the
-- OUTPUT channel count of every conv weight — which is the kernel's first dimension for a regular
-- conv `[oc,ic,kH,kW]` and for a depthwise `[c,1,k,k]` alike. This pins `bnChannels`'s length,
-- widths AND order against the layer list, so the `k = 0` dispatch cannot be written one way in
-- `layers` and another in `bnChannels`. A misaligned stat slot is otherwise SILENT: the arities
-- still match and the wrong layer's statistics simply flow into the wrong `@mnv4_fwd_eval` slot.
#guard mobilenetv4Verified.bnChannels ==
  (mobilenetv4Verified.toSpecs.filterMap (fun (d, _) => if d.size == 4 then some d[0]! else none))
#guard mobilenetv4Verified.bnChannels.size == 77

/-- **MobileNetV4-Conv-M on full 1000-class ImageNet** — identical trunk to
    `mobilenetv4Verified`, only the head moves (1280→1000). `#guard`ed at 9,715,512 parameters,
    the ~9.7M Conv-M is quoted at.

    The chapter's 75.48% top-1 from the 100-epoch JAX reference behind
    [`jax/MainMobilenetV4Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainMobilenetV4Imagenet.lean) was measured on an earlier transcription
    that differed from timm's; it is a target for this spec, not a comparison. This spec has no
    verified ImageNet training run.

    What is proved about it: the train-step capstone `Proofs.Mnv4TieB.mnv4_net_tiedB` binds the
    class count, so it covers this 1000-class head, at one replica, f32 and batch BatchNorm; the
    data-parallel step is `Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB`. Data-parallel renders:
    `mnv4in_adamdp64` (and its bf16 twin) and `mnv4in_emaaccdp8x128wxdowd005bf16`.

    A batch-BN net, so it scores through `@mnv4in_fwd_eval` with frozen running stats. It has
    its Imagenette peer's pre/post-DW-swap invisibility: `toSpecs` cannot see the order, and the
    forward and gradient ties (see `mobilenetv4Verified`) run against the Imagenette render
    (`@mnv4_fwd`, 10 classes). This spec differs from it only in the classifier, which the
    `#guard`s below pin, so what those ties establish about block order carries to it. -/
def mnv4ImagenetVerified : VerifiedNetSpec where
  name     := "MobileNetV4-Conv-M (ImageNet-1k)"
  slug     := "mnv4in"
  inC      := 3
  imageH   := 224
  imageW   := 224
  nClasses := 1000
  data     := .imagenet
  -- ⚠ Generated from the Conv-M reference recipe (`gen_shims.sh`'s `mobilenet-v4-imagenet:default`).
  -- That is correct and deliberate: a shim supplies AUGMENTED BATCHES, not weights, so what it
  -- carries across is the MNv4-family data pipeline (RandAugment, the 224² crop), which is shared
  -- by both sizes. Nothing about Conv-M's block table reaches this net through it.
  shimScript := "generated_mobilenet_v4_imagenet_shim.py"
  layers   := [
    .convBnNB 3 32 3 2,
    .fusedMbConvNB 32 48 4 3 2,
.uib  48  80 4 2 3 5,
    .uib  80  80 2 1 3 3,
    .uib  80 160 6 2 3 5,
    .uib 160 160 4 1 3 3,
    .uib 160 160 4 1 3 3,
    .uib 160 160 4 1 3 5,
    .uib 160 160 4 1 3 3,
    .uib 160 160 4 1 3 0,
    .uib 160 160 2 1 0 0,
    .uib 160 160 4 1 3 0,
    .uib 160 256 6 2 5 5,
    .uib 256 256 4 1 5 5,
    .uib 256 256 4 1 3 5,
    .uib 256 256 4 1 3 5,
    .uib 256 256 4 1 0 0,
    .uib 256 256 4 1 3 0,
    .uib 256 256 2 1 3 5,
    .uib 256 256 4 1 5 5,
    .uib 256 256 4 1 0 0,
    .uib 256 256 4 1 0 0,
    .uib 256 256 2 1 5 0,
    .convBnNB 256 960 1 1,
    .globalAvgPool,                     -- timm pools BEFORE conv_head
    .convBnNB 960 1280 1 1,
    .dense 1280 1000 ]
  blurb := "MobileNetV4-Conv-M on full 1000-class ImageNet via the VERIFIED renderer → %LOWERER% → GPU, with the tfds batch shim supplying the MNv4 reference augmentation"
  bnChannels := mobilenetv4Verified.bnChannels
  -- ▶ classifier dropout at the reference's 0.1 (keep 0.9) on the 1280-wide head, read only by
  -- `do` variants (`VerifiedVariant.cdOn`); `adamdp64*` carry no mask and are untouched.
  dropoutKeep := some (0.9, 1280)

-- Exactly one parameter shape may differ (the head), and the BN layout must be IDENTICAL — the
-- running-stat region is positional, so a drift there misaligns every frozen statistic at eval.
-- Same three-way pin `mobilenetv2ImagenetVerified` carries.
#guard mnv4ImagenetVerified.toSpecs.size == mobilenetv4Verified.toSpecs.size
#guard mnv4ImagenetVerified.toSpecs.pop.pop == mobilenetv4Verified.toSpecs.pop.pop
#guard mnv4ImagenetVerified.toSpecs.back! == (#[1000], 2)
#guard mnv4ImagenetVerified.bnChannels == mobilenetv4Verified.bnChannels
-- 8,447,322 − (1280·10 + 10) + (1280·1000 + 1000). The head is the only term that moves, so this
-- is the Imagenette count with its head swapped and nothing else — which is the whole claim above,
-- stated as arithmetic rather than as a comment. 9,715,512 is the ~9.7M Conv-M is quoted at.
#guard (mnv4ImagenetVerified.toSpecs.foldl
          (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0) == 9715512
-- The same stat-alignment gate the Imagenette spec carries, re-run against the 1000-class layout.
#guard mnv4ImagenetVerified.bnChannels ==
  (mnv4ImagenetVerified.toSpecs.filterMap (fun (d, _) => if d.size == 4 then some d[0]! else none))

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- ⭐⭐ THE INVARIANT THAT LICENSES `loadData`'s `trainPix := net.d0` (`VerifiedTrain.lean`,
-- the `.imagenet` branch). Placed here because it needs EVERY `.imagenet` spec in scope.
--
-- That function drains VAL at a hardcoded `3·224·224` and, until 2026-08-06, returned the very same
-- literal as the TRAIN stream width — so the 160 net asked its shim for 150,528 floats/img while
-- the shim correctly sent 76,800, and the wire guard refused. Switching it to `net.d0` is a
-- behaviour change ONLY for a net whose `d0` differs from the val width, i.e. only the 160 net.
-- ▶ These guards ARE the proof that all six incumbents are untouched, and they are why that switch
-- needed no per-net re-validation.
--
-- ⚠⚠ IF YOU ADD AN `.imagenet` NET AT A NON-224 TRAIN RESOLUTION, ONE OF THESE FIRES — and that is
-- the point, not an obstacle. It means the val drain is still 224 while your train stream is not,
-- so `evalD0` (`planning/archive/next_session_rsb_a3.md` §2.3) must land before the eval loop can be
-- trusted. Do NOT relax the guard to make it pass; add the net to the exempt list below it only
-- once the eval path reads its own width.
#guard resnet34ImagenetVerified.d0     == 3*224*224
#guard vitImagenetVerified.d0          == 3*224*224
#guard mobilenetv2ImagenetVerified.d0  == 3*224*224
#guard efficientnetImagenetVerified.d0 == 3*224*224
#guard convnextImagenetVerified.d0     == 3*224*224
#guard resnet50ImagenetVerified.d0     == 3*224*224
#guard mnv4ImagenetVerified.d0         == 3*224*224
-- The one net that is DELIBERATELY not 224, and the reason `evalD0` is still open.
#guard resnet50Imagenet160Verified.d0  == 3*160*160


