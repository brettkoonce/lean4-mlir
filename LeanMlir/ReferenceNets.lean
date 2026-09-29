import LeanMlir.Types

/-! # Reference NetSpecs shared by several entry points

The Imagenette ResNet-34 and ConvNeXt-T of the reference codegen path (`MlirCodegen` → `Train`),
each used by a trainer, the ablation table, the Grad-CAM probe and a test or inspector; the
BraTS UNet, used by its trainer, its predictor, the UNet forward test and the UNet Bestiary
entry; and the ResNet-34 UNet on BraTS, used by its trainer, the predictor and the per-volume
scorer. One
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

/-- R34 encoder (stride /32, as pretrained) + a UNet decoder, with or without skip
    connections, on BraTS at 224×224 — the segmentation demo's anchor
    ([`demos/MainUnetBratsR34.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/demos/MainUnetBratsR34.lean)),
    rendered by `brats-predict` and scored per volume by `brats-eval`. `ctx` is the 2.5D context: `ctx = 0` is the 4-modality 2D net on
    `data/brats224`; `ctx = k` reads `DatasetKind.brats224Ctx k`'s `4·(2k+1)`-channel records
    and differs from the 2D net in exactly one tensor, the stem `[64, 4·(2k+1), 7, 7]`. The
    name carries the channel count, so each context's artifacts have their own prefix and the
    2D prefix is unchanged.

    **`skips := true` (default).** The decoder concatenates four encoder taps on the way back
    up. This needs no new backward math: the encoder→decoder gradient join is `fpnTapGrad`,
    built for `.fpnDetect`, which adds an externally-supplied gradient to a residual stage's
    output before its skip-add/ReLU backward. `unetUp`'s concat-split backward saves the
    skip-half gradient as `%unet_skip_g{e}`. The stem's skip needs even less: the `.maxPool 2 2`
    sits exactly where a `unetDown`'s internal maxpool sits, so it reuses that path verbatim.
    Taps are matched by **exact shape**, not stack order — which is why stage 4 is correctly
    ignored (nothing upsamples into 7²) without special-casing.

    **`skips := false`.** The skipless decoder, kept reproducible. It has to rebuild every
    boundary from the 7×7 bottleneck alone, and its masks are visibly blobbier for it
    (mIoU 0.633 against 0.741 with skips, same schedule).

    Either way the transfer A/B is internally controlled: both arms of a given variant differ
    only in initialization. -/
def r34UnetBratsOf (skips : Bool) (ctx : Nat := 0) : NetSpec where
  -- Distinct names ⇒ distinct `buildPrefix` ⇒ the variants can never overwrite each other's
  -- checkpoints or race on the same graph artifact.
  name :=
    let input := if ctx == 0 then "4-modality MRI"
                 else s!"{4 * (2 * ctx + 1)}-channel 2.5D±{ctx} MRI"
    if skips
    then s!"ResNet-34 UNet skip (BraTS, 224×224 {input} → 4-class tumour)"
    else s!"ResNet-34 → UNet (BraTS, 224×224 {input} → 4-class tumour)"
  imageH := 224
  imageW := 224
  layers :=
    -- Encoder: R34 exactly as pretrained — stem, maxPool, four stages ⇒ /32.
    -- With `skips`, four of these feed the decoder: the stem's pre-pool output
    -- (112², 64ch) and stages 1–3 (56²/64, 28²/128, 14²/256). Stage 4 is the
    -- bottleneck, not a skip — nothing upsamples into 7², so the codegen's
    -- shape-matched tap lookup excludes it without being told to.
    [ .convBn (4 * (2 * ctx + 1)) 64 7 2 .same,  -- 224 → 112, 64ch   ← skip
      .maxPool 2 2,                 -- 112 →  56
      .residualBlock  64  64 3 1,   --  56, 64ch         ← skip
      .residualBlock  64 128 4 2,   --  28, 128ch        ← skip
      .residualBlock 128 256 6 2,   --  14, 256ch        ← skip
      .residualBlock 256 512 3 2    --   7, 512ch  (bottleneck)
    ] ++
    (if skips then
      -- Each `unetUp` upsamples ×2, concatenates the matching encoder tap,
      -- then runs 2× (conv+BN). R34's channel ladder (64/64/128/256/512) is
      -- already UNet-shaped, so no adapter convs are needed anywhere.
      [ .unetUp 512 256,            --   7 →  14, + stage3 (256)
        .unetUp 256 128,            --  14 →  28, + stage2 (128)
        .unetUp 128 64,             --  28 →  56, + stage1 (64)
        .unetUp 64 64,              --  56 → 112, + stem   (64)
        .bilinearUpsample 2,        -- 112 → 224 (no tap at full res)
        .convBn 64 32 3 1 .same,
        .conv2d 32 4 1 .same .identity
      ]
     else
      -- The no-skip decoder, kept EXACTLY as run so its published numbers stay
      -- reproducible. The decoder must rebuild every boundary from the 7×7
      -- bottleneck alone, which is why its masks come out visibly blobbier.
      [ .bilinearUpsample 2, .convBn 512 256 3 1 .same,
        .bilinearUpsample 2, .convBn 256 128 3 1 .same,
        .bilinearUpsample 2, .convBn 128 64 3 1 .same,
        .bilinearUpsample 2, .convBn 64 32 3 1 .same,
        .bilinearUpsample 2, .convBn 32 32 3 1 .same,
        .conv2d 32 4 1 .same .identity
      ])

/-- The skip-equipped 2D net (the default). -/
def r34UnetBrats : NetSpec := r34UnetBratsOf true

end ReferenceNets
