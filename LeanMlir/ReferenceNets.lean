import LeanMlir.Types

/-! # Reference NetSpecs shared by several entry points

The Imagenette ResNet-34 and ConvNeXt-T of the reference codegen path (`MlirCodegen` → `Train`),
each used by a trainer, the ablation table, the Grad-CAM probe and a test or inspector; the
BraTS UNet, used by its trainer, its predictor, the UNet forward test and the UNet Bestiary
entry; the ResNet-34 UNet on BraTS, used by its trainer, the predictor and the per-volume
scorer; Chapter 4's CIFAR-CNN8-wide-BN at the Chapter 10 classifier demos' shapes; the MNIST
DDPM denoiser; and the R34/R50 FPN detectors. One
definition, so a change to the net reaches every consumer. A consumer that trains under another
name overrides `name`: it is the checkpoint prefix (`NetSpec.buildPrefix`).

## References

- He, Zhang, Ren, Sun 2016, *Deep Residual Learning for Image Recognition*. <https://arxiv.org/abs/1512.03385>
- Ronneberger, Fischer, Brox 2015, *U-Net*. <https://arxiv.org/abs/1505.04597> -/

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
    (lower mIoU than with skips at the same schedule).

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

/-- The BraTS net and record format a `brats-predict` / `brats-eval` invocation names:
    `useR34 = false` is the from-scratch `unetBrats` on `data/brats`; `useR34` is the ResNet-34
    UNet (`noSkip` drops its skips), on `data/brats224`, or with `ctx = k > 0` its 2.5D variant on
    `DatasetKind.brats224Ctx k`. -/
def bratsNetOf (useR34 noSkip : Bool) (ctx : Nat) : NetSpec × DatasetKind :=
  if !useR34 then (unetBrats, .brats)
  else (r34UnetBratsOf (!noSkip) ctx, if ctx == 0 then .brats224 else .brats224Ctx ctx)

/-- Chapter 4's CIFAR-CNN8-wide-BN at an `inC`-channel `H × W` input and an `nOut`-way head:
    eight convolutions in four conv-conv-pool stages at 16, 16, 32, 32 channels into the
    2 × 512 head, flatten 32 · (H/16) · (W/16). The ArASL, gravitational-wave and remote-sensing
    demos train it, each under its own `name`. -/
def cifar8wOf (name : String) (inC H W nOut : Nat) : NetSpec where
  name := name
  imageH := H
  imageW := W
  layers := [
    .convBn inC 16 3 1 .same,
    .convBn 16 16 3 1 .same,
    .maxPool 2 2,
    .convBn 16 16 3 1 .same,
    .convBn 16 16 3 1 .same,
    .maxPool 2 2,
    .convBn 16 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense (32 * (H / 16) * (W / 16)) 512 .relu,
    .dense 512 512 .relu,
    .dense 512 nOut .identity
  ]

/-- The MNIST DDPM denoiser: a tiny time-conditioned UNet on a 2-channel input (the image plus
    a scalar t/T_max timestep encoding tiled to the same spatial dims); the output stays
    single-channel (predicted ε for the image). One definition for the trainer, the sampler and
    the score probe, whose `buildPrefix`es must agree. `centred` names the [-1, 1] arm (the
    default) apart from the uncentred ablation, so their checkpoints never collide. -/
def tinyDdpmUnet (centred : Bool := true) : NetSpec where
  name := if centred then "tiny DDPM UNet T-cond centered (MNIST 28x28x1)"
                     else "tiny DDPM UNet T-cond (MNIST 28x28x1)"
  imageH := 28
  imageW := 28
  layers := [
    .unetDown 2 16,
    .unetDown 16 32,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .unetUp 64 32,
    .unetUp 32 16,
    .conv2d 16 1 1 .same .identity
  ]

/-- ResNet-34 + FPN detector at 448: the R34 backbone tapped at C3/C4/C5 into `.fpnDetect`, with
    `tower` 3×3 convs per pyramid level in the RetinaNet head (0 = the minimal 1×1 head, 4 the
    RetinaNet default). The VisDrone and NEU-DET detectors and the emit probe train it under
    their own names, which keep their checkpoints apart. -/
def r34FpnDet (name : String) (tower : Nat) : NetSpec where
  name := name
  imageH := 448
  imageW := 448
  detStride := 32
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,   -- stride 4
    .residualBlock  64 128 4 2,   -- C3: 128ch, 56×56 (stride 8)
    .residualBlock 128 256 6 2,   -- C4: 256ch, 28×28 (stride 16)
    .residualBlock 256 512 3 2,   -- C5: 512ch, 14×14 (stride 32)
    .fpnDetect 256 128 256 512 14 3 tower
  ]

/-- The ResNet-50 backbone variant of `r34FpnDet`. The first six layers follow
    [`jax/MainResnet50Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnet50Imagenet.lean)'s
    `resnet50Imagenet` — same order, same channels, same
    `convPadStyle` — because `bootstrapBackbone` is a PREFIX copy: it drops the checkpoint's
    leading floats onto the init's leading floats and only verifies that the bytes landed, not
    that they mean the same thing. Any divergence in these six lines silently loads a
    correct-sized, wrong-layout backbone. The classifier (`.globalAvgPool` + `.dense 2048 1000`)
    is what the bootstrap drops.

    At 448 input the stages land on 112 / 56 / 28 / 14, so C3/C4/C5 are 56/28/14, the FPN
    scales of the R34 detector. Only the tap WIDTHS move (512/1024/2048 vs R34's 128/256/512),
    and `.fpnDetect` takes those as arguments, so this is a spec change with no new codegen. -/
def r50FpnDet (name : String) (tower : Nat) : NetSpec where
  name := name
  -- torchvision's Conv2d(3,64,7,stride=2,padding=3), matching the ImageNet render
  -- the A3 checkpoint was trained under. Omitting this mismatches the stem.
  convPadStyle := .symmetric
  imageH := 448
  imageW := 448
  detStride := 32
  layers := [
    .convBn 3 64 7 2 .same,
    -- DELIBERATELY 2×2, where `resnet50Imagenet` has `.maxPool 3 2`.
    -- The train-step emitter's max-pool backward is a tile-compare-select that
    -- is correct ONLY for non-overlapping windows (`emitTrainBackward` in
    -- `LeanMlir/MlirCodegen.lean` says so outright), and size 3 > stride 2
    -- overlaps: one input can be the max of
    -- several windows, so its gradient is a SUM the tiling never forms. It also
    -- happens to fail loudly first — the forward pads 224→225 but the backward
    -- reads `inShape` (224) against the padded SSA, so the graph does not even
    -- parse. Fixing that type error alone would trade a compile failure for a
    -- silently wrong gradient, which is worse.
    -- Safe to change because pooling is PARAMETER-FREE: the bootstrap prefix is
    -- untouched, and both windows take 224→112, so every downstream shape and
    -- the C3/C4/C5 taps are identical. The cost is a one-layer distribution
    -- shift — the backbone was pretrained under 3×3 pooling — which fine-tuning
    -- absorbs. The R34 detector does exactly this.
    .maxPool 2 2,
    .bottleneckBlock   64  256 3 1,   -- C2: stride 4,  112×112
    .bottleneckBlock  256  512 4 2,   -- C3: 512ch,      56×56
    .bottleneckBlock  512 1024 6 2,   -- C4: 1024ch,     28×28
    .bottleneckBlock 1024 2048 3 2,   -- C5: 2048ch,     14×14
    .fpnDetect 256 512 1024 2048 14 3 tower
  ]

end ReferenceNets
