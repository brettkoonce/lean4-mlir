import LeanMlir.Types

/-! # Reference NetSpecs shared by several entry points

The Imagenette ResNet-34 and ConvNeXt-T of the reference codegen path (`MlirCodegen` → `Train`),
each used by a trainer, the ablation table, the Grad-CAM probe and a test or inspector; and the
Oxford-IIIT Pets UNet and its skipless autoencoder baseline, used by the Pets trainers, the Pets
predictor, the UNet forward test and the UNet Bestiary entry. One
definition, so a change to the net reaches every consumer. A consumer that trains under another
name overrides `name`: it is the checkpoint prefix (`NetSpec.buildPrefix`). -/

namespace ReferenceNets

/-- ResNet-34 at 224×224, 10 classes. The stem pool is 2×2 stride 2 rather than the paper's 3×3
    stride 2 (IREE had no `select_and_scatter` for 3/2). -/
def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

/-- ConvNeXt-T (Liu et al. 2022) at 224×224, 10 classes: stages (3, 3, 9, 3) at (96, 192, 384,
    768) channels, depthwise 7×7 + LayerNorm + inverted bottleneck + GELU + LayerScale, a
    LN + 2×2 stride-2 conv between stages. The stem is a 4×4 stride-4 `convBn` (the paper uses
    LN; same parameter count). ~28M parameters. -/
def convNextTinyGelu : NetSpec where
  name := "ConvNeXt-T-GELU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,                    -- patchify stem (4×4 stride 4)
    .convNextStage 96 3 .ln .gelu,             -- stage 1: 3 blocks at 96 ch
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .gelu,            -- stage 2: 3 blocks at 192 ch
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .gelu,            -- stage 3: 9 blocks at 384 ch
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .gelu,            -- stage 4: 3 blocks at 768 ch
    .globalAvgPool,
    .dense 768 10 .identity
  ]

/-- UNet for Oxford-IIIT Pets: 224×224 RGB → 3-class trimap (foreground / background / boundary,
    after the trimap remap 1/2/3 → 0/1/2). Depth 4, base 32 channels, a 512-channel bottleneck;
    each `unetUp` takes the skip of the matching `unetDown` (LIFO). 7.85M parameters. -/
def unetPets : NetSpec where
  name := "UNet (Pets, 224×224 RGB → 3-class trimap)"
  imageH := 224
  imageW := 224
  layers := [
    .unetDown 3   32,                  -- encoder stage 1: 224 → 112
    .unetDown 32  64,                  -- encoder stage 2: 112 → 56
    .unetDown 64  128,                 -- encoder stage 3: 56  → 28
    .unetDown 128 256,                 -- encoder stage 4: 28  → 14
    .convBn 256 512 3 1 .same,         -- bottleneck part 1
    .convBn 512 512 3 1 .same,         -- bottleneck part 2
    .unetUp 512 256,                   -- decoder stage 4 (skip: encoder 4)
    .unetUp 256 128,                   -- decoder stage 3 (skip: encoder 3)
    .unetUp 128 64,                    -- decoder stage 2 (skip: encoder 2)
    .unetUp 64  32,                    -- decoder stage 1 (skip: encoder 1)
    .conv2d 32 3 1 .same .identity     -- output projection (1×1 conv to 3 classes)
  ]

/-- The skipless baseline for `unetPets`: the same 224 → 14 → 224 spatial trip through four
    2×2 max-pools and four 2× bilinear upsamples, no skip connections. ~5.5M parameters. -/
def autoencoderPets : NetSpec where
  name := "Autoencoder (Pets, 224×224 RGB → 3-class trimap, skipless)"
  imageH := 224
  imageW := 224
  layers := [
    -- Encoder: 224 → 14 (four 2× downsamples), 3 → 512 channels
    .convBn 3   64  3 1 .same, .maxPool 2 2,   -- 224 → 112
    .convBn 64  128 3 1 .same, .maxPool 2 2,   -- 112 → 56
    .convBn 128 256 3 1 .same, .maxPool 2 2,   -- 56  → 28
    .convBn 256 512 3 1 .same, .maxPool 2 2,   -- 28  → 14
    .convBn 512 512 3 1 .same,                 -- bottleneck @ 14×14
    -- Decoder: 14 → 224 (four 2× bilinear upsamples), 512 → 64 channels
    .bilinearUpsample 2, .convBn 512 256 3 1 .same,   -- 14  → 28
    .bilinearUpsample 2, .convBn 256 128 3 1 .same,   -- 28  → 56
    .bilinearUpsample 2, .convBn 128 64  3 1 .same,   -- 56  → 112
    .bilinearUpsample 2, .convBn 64  64  3 1 .same,   -- 112 → 224
    .conv2d 64 3 1 .same .identity                    -- output projection (1×1, 3 classes)
  ]

end ReferenceNets
