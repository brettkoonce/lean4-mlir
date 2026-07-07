import Jax

/-! EfficientNet-B0 on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.
    Same MBConv body as `MainEfficientNet.lean` (Imagenette) but with a
    1000-class head and the `.imagenet` (tfds streaming) dataset.

    bf16 incl. bf16 conv: as of the Codegen change that routes the MBConv
    expand/depthwise/project convs through `convdt`, `bf16Conv` now reaches
    the heavy convs in all MBConv blocks (the ~2x MBConv-block win from
    reference_bf16_depthwise_4060ti). The squeeze-excitation 1x1s are left in
    fp32 on purpose — they act on 1x1-spatial pooled tensors (no bf16 win)
    and the sigmoid gate is precision-sensitive. -/

def efficientNetB0Imagenet : NetSpec where
  name := "EfficientNet-B0 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
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

/-- EfficientNet-B0 80-epoch recipe — the validation tier of the 80→300
    ladder (EPOCHS is baked from this spec; bump to 300 + re-emit for the
    real run). RMSProp + momentum 0.9, base lr 0.045 at batch 256, 5-epoch
    warmup + paper exp-LR-decay (×0.97 every 2.4 epochs), weight decay 1e-5 (EfficientNet's small value — protects
    the depthwise/SE params), label smoothing 0.1, random-crop + flip,
    bf16 + bf16Conv.

    EfficientNet's original recipe is RMSProp + AutoAugment + stochastic depth
    + EMA — and as of this config we have ALL FOUR: RMSProp (below) + the full
    AutoAugment ImageNet policy (useAutoAugment, geometric ops included via
    ImageProjectiveTransformV3) + stochastic depth + EMA. This is the faithful
    B0 recipe; the remaining gap to 77% is schedule/length, not missing pieces.
    RMSProp knobs: ρ=0.9, μ=0.9, ε=1e-3 (EfficientNet's value). LR schedule
    is the paper's exponential decay (×0.97 every 2.4 epochs after warmup; cosineDecay off).

    TODO(recipe): this is a CHANGE on two axes (optimizer SGD→RMSProp, aug
    color-RandAugment→full AutoAugment) — prior results no longer apply. Re-run
    80ep + re-eval (eval_enet_full50k.py, supervise script unchanged) for fresh
    numbers. lr 0.045 is the MobileNet-style peak; the paper-faithful
    linear-scaled value is ~0.016 at batch 256 (0.256@4096), so if SE/swish
    make 0.045 unstable early, drop toward ~0.016. AutoAugment is a CPU-side
    tf.data op — watch input throughput isn't the bottleneck on the first run.
    Mixup/cutmix still off (flip those flags for the very full recipe). -/
def efficientNetB0ImagenetConfig : TrainConfig where
  learningRate   := 0.016   -- EfficientNet reference base LR 0.016@bs256 (= 0.256@bs4096); NB our RMSProp ε1e-3 stack erodes here — _full uses the stable 0.01
  batchSize      := 256
  epochs         := 80
  optimizer      := .rmsprop  -- EfficientNet's original optimizer
  momentum       := 0.9       -- μ for the RMSprop momentum buffer
  rmspropDecay   := 0.9       -- ρ, the running mean-square decay
  rmspropEps     := 1e-3      -- EfficientNet uses ε=1e-3
  weightDecay    := 1e-5
  cosineDecay      := false   -- replaced by the paper exp-decay schedule (gap B)
  expLRDecayRate   := 0.97    -- EfficientNet: ×0.97 every 2.4 epochs (after warmup)
  expLRDecayEpochs := 2.4
  dropout          := 0.2     -- EfficientNet-B0 classifier dropout (gap C)
  warmupEpochs   := 5
  augment        := true
  useAutoAugment := true     -- full AutoAugment ImageNet policy (incl. geometric)
  labelSmoothing := 0.1
  bf16           := true
  bf16Conv       := true    -- now reaches the MBConv expand/depthwise/project
  useEMA         := true     -- weight averaging (decay 0.9999) — eval + ckpt use it
  dropPath       := 0.2      -- stochastic depth, EfficientNet-B0 drop-connect rate
  runningBN      := true     -- paper-faithful eval (gap A): running BN stats, not eval-batch stats

#eval efficientNetB0Imagenet.validate!

/-- Paper-faithful full run: identical recipe at the 350-epoch schedule and the
    stable peak LR (the paper-scaled 0.016 erodes in our RMSProp ε1e-3 stack; ~0.01
    trains). Selected with the `full` recipe arg. -/
def efficientNetB0ImagenetConfigFull : TrainConfig :=
  { efficientNetB0ImagenetConfig with epochs := 350, learningRate := 0.01 }

def efficientNetB0ImagenetRecipes : List Recipe := [
  { name := "default", cfg := efficientNetB0ImagenetConfig,
    out := "generated_efficientnet_b0_imagenet.py",
    desc := "80-epoch validation tier (RMSProp + AutoAugment, default LR 0.016)" },
  { name := "full",    cfg := efficientNetB0ImagenetConfigFull,
    out := "generated_efficientnet_b0_imagenet_full.py",
    desc := "paper-faithful 350-epoch run (stable peak LR 0.01)" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "efficientnet-b0-imagenet" efficientNetB0Imagenet .imagenet
    efficientNetB0ImagenetRecipes args
