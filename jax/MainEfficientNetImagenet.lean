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
  -- ⚠⚠ ADDED 2026-08-30, AND ITS ABSENCE WAS SILENT FOR THE WHOLE 76.80% RUN. `NetSpec.convBnAct`
  -- defaults to `.relu` (`LeanMlir/Types.lean:378`), so without this line the two `.convBn` layers
  -- below — the stem and the 1×1 head — emitted `jax.nn.relu` while every MBConv interior emitted
  -- `swish`. EfficientNet-B0 is SiLU/swish THROUGHOUT, stem and head included.
  -- ▶ The Imagenette twin has carried this line since `planning/archive/mnv4_verified.md` §3f measured the
  -- deviation at 51% of logit range — five times the padding deviation — and this spec, written as
  -- "same MBConv body as MainEfficientNet.lean", never received it. The fix was made once and
  -- applied to one of the two files.
  -- ⚠ The VERIFIED render was always swish (194 `stablehlo.logistic`, zero `stablehlo.maximum`), so
  -- until this line the port was MORE paper-faithful than the reference it was scored against, and
  -- B0's phase-2 ↔ phase-4 accuracy comparison was not like-for-like.
  -- ⛔ `scripts/parity/enet_forward_tie.py` cannot catch this: it ties the render against
  -- `generated_efficientnet_b0.py`, the Imagenette file, which is the one that was already right.
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

/-- EfficientNet-B0 80-epoch tier (`default`); the published reference is the 350-epoch
    `full` recipe below, which is this config with only `epochs` changed.
    RMSProp + momentum 0.9 at peak lr 0.016 for batch 256 (= 0.256 @ 4096, TF's value), and TF's
    schedule: ×0.97 every 2.4 epochs as a staircase on the global step, a 5-epoch linear warmup
    overriding it while it runs. Weight decay 1e-5 coupled into the gradient, off BN γ/β and biases
    (TF and timm exclude BN), TF's BN ε 1e-3 (decay 0.99, already TF's), drop-connect ramped over
    i/16 as TF's `drop_rate · idx / len(blocks)`, label smoothing 0.1, classifier dropout 0.2,
    bf16 + bf16Conv (planning/imagenet_parity.md §5.3, 2026-09-25).

    EfficientNet's original recipe is RMSProp + AutoAugment + stochastic depth + EMA, and all four
    are here: the full AutoAugment ImageNet policy (useAutoAugment, geometric ops included via
    ImageProjectiveTransformV3; no RandAugment), drop-connect 0.2, EMA 0.9999 with the BN buffers
    shadowed too. RMSProp knobs: ρ=0.9, μ=0.9, ε=1e-3 (EfficientNet's value, inside the sqrt as
    TF has it, mean-square initialised to 1.0). Mixup/cutmix off. -/
def efficientNetB0ImagenetConfig : TrainConfig where
  learningRate   := 0.016   -- EfficientNet reference base LR 0.016@bs256 (= 0.256@bs4096); paper-faithful now that RMSProp matches TF (ε-inside-sqrt + mean-square init 1.0)
  batchSize      := 256
  epochs         := 80
  optimizer      := .rmsprop  -- EfficientNet's original optimizer
  momentum       := 0.9       -- μ for the RMSprop momentum buffer
  rmspropDecay   := 0.9       -- ρ, the running mean-square decay
  rmspropEps     := 1e-3      -- EfficientNet uses ε=1e-3
  gradClipNorm   := 0.0       -- OFF (paper uses none): the TF-RMSProp fix (ε-inside-sqrt + ms-init-1.0) removes the blow-up this was compensating for
  weightDecay    := 1e-5
  wdExcludeNormBias := true   -- no decay on BN γ/β or biases (TF, timm; the verified `wx`)
  cosineDecay      := false   -- replaced by the paper exp-decay schedule (gap B)
  expLRDecayRate   := 0.97    -- EfficientNet: ×0.97 every 2.4 epochs
  expLRDecayEpochs := 2.4
  expLRStaircase   := true    -- TF: floored, on the global step (warmup overrides while it runs)
  dropout          := 0.2     -- EfficientNet-B0 classifier dropout (gap C)
  warmupEpochs   := 5
  augment        := true
  useAutoAugment := true     -- full AutoAugment ImageNet policy (incl. geometric)
  augBicubic     := true    -- C6: PIL-bicubic geometry, as timm (planning/imagenet_parity.md)
  labelSmoothing := 0.1
  bf16           := true
  bf16Conv       := true    -- now reaches the MBConv expand/depthwise/project
  useEMA         := true     -- weight averaging (decay 0.9999) — paper-faithful; the emitter now EMA-shadows the BN buffers too (eval uses ema_bn), fixing the earlier EMA-weights×live-BN-stats eval blow-up
  dropPath       := 0.2      -- stochastic depth, EfficientNet-B0 drop-connect rate
  dropPathOverN  := true     -- TF's ramp: 0.2 · i/16, not timm's i/15
  runningBN      := true     -- paper-faithful eval (gap A): running BN stats, not eval-batch stats
  bnEps          := 1e-3     -- TF's BN ε

#eval efficientNetB0Imagenet.validate!

/-- Paper-faithful full run: identical recipe at the 350-epoch schedule, at the
    paper's real LR 0.016 (now that the TF-RMSProp fix makes it train stably —
    no lowered LR, no grad clip). Selected with the `full` recipe arg. -/
def efficientNetB0ImagenetConfigFull : TrainConfig :=
  { efficientNetB0ImagenetConfig with epochs := 350 }

def efficientNetB0ImagenetRecipes : List Recipe := [
  { name := "default", cfg := efficientNetB0ImagenetConfig,
    out := "generated_efficientnet_b0_imagenet.py",
    desc := "80-epoch validation tier (RMSProp + AutoAugment, default LR 0.016)" },
  { name := "full",    cfg := efficientNetB0ImagenetConfigFull,
    out := "generated_efficientnet_b0_imagenet_full.py",
    desc := "paper-faithful 350-epoch run (peak LR 0.016)" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "efficientnet-b0-imagenet" efficientNetB0Imagenet .imagenet
    efficientNetB0ImagenetRecipes args
