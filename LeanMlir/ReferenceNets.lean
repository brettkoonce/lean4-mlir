import LeanMlir.Types

/-! # Reference NetSpecs shared by several entry points

The Imagenette ResNet-34 and ConvNeXt-T of the reference codegen path (`MlirCodegen` → `Train`),
each used by a trainer, the ablation table, the Grad-CAM probe and a test or inspector; and the
BraTS UNet, used by its trainer, its predictor, the UNet forward test and the UNet Bestiary
entry. One
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

/-- UNet for BraTS (MSD Task01_BrainTumour): 240×240, 4 MRI modalities → 4-class tumour mask.
    Depth 4, base 32 channels, a 512-channel bottleneck; each `unetUp` takes the skip of the
    matching `unetDown` (LIFO). The trainer and the predictor both use this one definition:
    `buildPrefix` derives from `name`, so two copies that drift by a character point the predictor
    at a checkpoint that does not exist. -/
def unetBrats : NetSpec where
  name := "UNet (BraTS, 240×240 4-modality MRI → 4-class tumour)"
  imageH := 240
  imageW := 240
  layers := [
    .unetDown 4   32,
    .unetDown 32  64,
    .unetDown 64  128,
    .unetDown 128 256,
    .convBn 256 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .unetUp 512 256,
    .unetUp 256 128,
    .unetUp 128 64,
    .unetUp 64  32,
    .conv2d 32 4 1 .same .identity
  ]

end ReferenceNets
