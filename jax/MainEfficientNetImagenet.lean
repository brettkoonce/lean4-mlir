import Jax

/-! EfficientNet-B0 on full 1000-class ImageNet — Lean → JAX reference trainer.
    Same MBConv body as `MainEfficientNet.lean` (Imagenette) but with a
    1000-class head and the `.imagenet` (tfds streaming) dataset.

    bf16 incl. bf16 conv: the Codegen routes the MBConv
    expand/depthwise/project convs through `convdt`, so `bf16Conv` reaches
    the heavy convs in all MBConv blocks (the ~2x MBConv-block win). The
    squeeze-excitation 1x1s are left in fp32 on purpose — they act on 1x1-spatial pooled
    tensors (no bf16 win) and the sigmoid gate is precision-sensitive. -/

def efficientNetB0Imagenet : NetSpec where
  name := "EfficientNet-B0 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  -- ITS ABSENCE IS SILENT. `NetSpec.convBnAct`
  -- defaults to `.relu` (`LeanMlir/Types.lean`), so without this line the two `.convBn` layers
  -- below — the stem and the 1×1 head — emit `jax.nn.relu` while every MBConv interior emits
  -- `swish`. EfficientNet-B0 is SiLU/swish THROUGHOUT, stem and head included.
  -- The Imagenette twin carries this line too; the deviation it prevents is several times the
  -- strided-depthwise padding deviation (0.37 of the logit scale at train resolution, closed as
  -- D3 on 2026-10-12: `mbconv_block` now pads its depthwise `'SAME'` under `convPadStyle`).
  -- The VERIFIED render is swish (194 `stablehlo.logistic`, zero `stablehlo.maximum`), so without
  -- this line the port is MORE paper-faithful than the reference it is scored against, and the
  -- reference ↔ verified accuracy comparison is not like-for-like.
  -- `scripts/parity/enet_forward_tie.py` cannot catch this: it ties the render against
  -- `generated_efficientnet_b0.py`, the Imagenette file.
  convBnAct := .swish
  layers := [
    .convBn 3 32 3 2 .same,                          -- 224→112
    .mbConv  32  16 1 3 1 1 true,                     -- 112
    .mbConv  16  24 6 3 2 2 true,                     -- 112→56
    .mbConv  24  40 6 5 2 2 true,                     -- 56→28
    .mbConv  40  80 6 3 2 3 true,                     -- 28→14
    .mbConv  80 112 6 5 1 3 true,                     -- 14
    .mbConv 112 192 6 5 2 4 true,                     -- 14→7
    .mbConv 192 320 6 3 1 1 true,                     -- 7
    .convBn 320 1280 1 1 .same,                       -- 1x1 head
    .globalAvgPool,
    .dense 1280 1000 .identity                        -- 1000-class head
  ]

/-- EfficientNet-B0 on ImageNet-1k, TF TPU EfficientNet's own code row for row, at the
    code's length: `train_steps` 218949 at batch 2048 ≈ 350 epochs of data. (`full` below
    is the same config under the name the jobs and CI call it by.)
    RMSProp + momentum 0.9 at peak lr 0.016 for batch 256 (= 0.256 @ 4096, TF's value), and TF's
    schedule: ×0.97 every 2.4 epochs as a staircase on the global step, a 5-epoch linear warmup
    overriding it while it runs. Weight decay 1e-5 coupled into the gradient, off BN γ/β and biases
    (TF and timm exclude BN), TF's BN ε 1e-3 (decay 0.99, already TF's), drop-connect ramped over
    i/16 as TF's `drop_rate · idx / len(blocks)`, label smoothing 0.1, classifier dropout 0.2,
    bf16 + bf16Conv.

    EfficientNet's original recipe is RMSProp + AutoAugment + stochastic depth + EMA, and all four
    are here: AutoAugment policy v0 as TF trains it (useAutoAugment + autoAugmentV0, geometric ops via
    ImageProjectiveTransformV3; no RandAugment), drop-connect 0.2, EMA 0.9999 with the BN buffers
    shadowed too. RMSProp knobs: ρ=0.9, μ=0.9, ε=1e-3 (EfficientNet's value, inside the sqrt as
    TF has it, mean-square initialised to 1.0). Mixup/cutmix off. -/
def efficientNetB0ImagenetConfig : TrainConfig where
  learningRate   := 0.016   -- EfficientNet reference base LR 0.016@bs256 (= 0.256@bs4096); paper-faithful, RMSProp matching TF (ε-inside-sqrt + mean-square init 1.0)
  batchSize      := 256
  epochs         := 350     -- train_steps 218949 at batch 2048: 350 epochs of data
  optimizer      := .rmsprop  -- EfficientNet's original optimizer
  momentum       := 0.9       -- μ for the RMSprop momentum buffer
  rmspropDecay   := 0.9       -- ρ, the running mean-square decay
  rmspropEps     := 1e-3      -- EfficientNet uses ε=1e-3
  gradClipNorm   := 0.0       -- OFF (paper uses none): the TF-RMSProp form (ε-inside-sqrt + ms-init-1.0) has no blow-up to compensate for
  weightDecay    := 1e-5
  wdExcludeNormBias := true   -- no decay on BN γ/β or biases (TF, timm; the verified `wx`)
  cosineDecay      := false   -- the paper exp-decay schedule instead
  expLRDecayRate   := 0.97    -- EfficientNet: ×0.97 every 2.4 epochs
  expLRDecayEpochs := 2.4
  expLRStaircase   := true    -- TF: floored, on the global step (warmup overrides while it runs)
  dropout          := 0.2     -- EfficientNet-B0 classifier dropout
  warmupEpochs   := 5
  augment        := true
  useAutoAugment := true     -- full AutoAugment ImageNet policy (incl. geometric)
  autoAugmentV0  := true     -- TF EfficientNet's policy v0 (timm `v0`), img_mean fill
  augBicubic     := false   -- NOT timm's PIL bicubic: TF's geometry is nearest (augNearest below; 2026-10-12)
  cropFallbackCenter := true -- EfficientNet: a whole-image draw falls back to the centre crop
  labelSmoothing := 0.1
  bf16           := true
  bf16Conv       := true    -- reaches the MBConv expand/depthwise/project
  f32StemHead    := true    -- classifier head in fp32, as every verified render keeps it
  useEMA         := true     -- weight averaging (decay 0.9999) — paper-faithful; the emitter EMA-shadows the BN buffers too (eval uses ema_bn); EMA weights with live BN stats blow up at eval
  dropPath       := 0.2      -- stochastic depth, EfficientNet-B0 drop-connect rate
  dropPathOverN  := true     -- TF's and timm's ramp: 0.2 · i/16, not the default i/15
  runningBN      := true     -- paper-faithful eval: running BN stats, not eval-batch stats
  bnEps          := 1e-3     -- TF's BN ε
  -- ── The rest of TF TPU EfficientNet's own code, read off
  --    tensorflow/tpu models/official/efficientnet/{main,utils,efficientnet_model,preprocessing,
  --    autoaugment}.py and set row by row. Each line names its source. ──
  rmspropTf      := true     -- utils.build_optimizer: tf.train.RMSPropOptimizer(lr, decay, momentum, epsilon) — TF1's ApplyRMSProp folds lr into the momentum buffer
  wdDecayBiases  := true     -- main.py: weight_decay · Σ l2_loss(v) over every trainable variable with 'batch_normalization' not in its name — BN γ/β exempt, biases decayed
  initHeadTfUniform := true  -- efficientnet_model.py dense_kernel_initializer: U(±1/√1000) on the classifier, zeros bias
  autoAugmentTf  := true     -- autoaugment.py: the v0 table op for op; 'TranslateY' absolute, fill 128, TF's posterize shift
  augNearest     := true     -- contrib_image.{transform,rotate,translate}: interpolation defaults to NEAREST
  randAugmentTranslateConst := 250.0  -- distort_image_with_autoaugment: HParams(cutout_const=100, translate_const=250) → (m/10)·250 px
  resizeLegacyTf1 := true    -- preprocessing.py _resize_image: tf.image.resize_bicubic (TF1 op: no antialias, half_pixel_centers=False)
  evalCropTf     := true     -- preprocess_for_eval: _decode_and_center_crop — crop first, 224/256 of the shorter side …
  evalCropTfEfficientNet := true  -- … with its arithmetic (truncated side, ((h − side) + 1)//2 offsets), then bicubic to 224
  evalF32        := true     -- the logged number scored through an f32 forward (the verified path's and timm's)

#eval efficientNetB0Imagenet.validate!

/-- `full` is the same config: the name `scripts/jobs/enet-full-jax-4gpu.conf` and the CI
    emission check call the 350-epoch run by. -/
def efficientNetB0ImagenetConfigFull : TrainConfig := efficientNetB0ImagenetConfig

def efficientNetB0ImagenetRecipes : List Recipe := [
  { name := "default", cfg := efficientNetB0ImagenetConfig,
    out := "generated_efficientnet_b0_imagenet.py",
    desc := "the code's 350-epoch run (RMSProp + AutoAugment, peak LR 0.016)" },
  { name := "full",    cfg := efficientNetB0ImagenetConfigFull,
    out := "generated_efficientnet_b0_imagenet_full.py",
    desc := "the same run, under the name the jobs call it by" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "efficientnet-b0-imagenet" efficientNetB0Imagenet .imagenet
    efficientNetB0ImagenetRecipes args
