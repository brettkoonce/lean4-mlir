import Jax

/-! ConvNeXt-Tiny on full 1000-class ImageNet — Lean → JAX trainer.
    Patchify stem, compute ratio (3,3,9,3),
    channels (96,192,384,768), depthwise-7×7 + channel-LN + inverted-
    bottleneck + GELU + LayerScale blocks, dedicated 2×2 stride-2 downsamples.
    ~28.6M params at 224×224, 1000 classes.

    bf16 incl. bf16 conv: the depthwise-7×7 is markedly faster in bf16 and the
    1×1 expand/project are GEMM-like (big bf16 win) — see the per-op
    microbench in `planning/archive/imagenet_sweep.md`. Channel-LN / GELU stay fp32.

    Architecture is canonical ConvNeXt-T with a faithful patchify stem (4×4 s4
    conv → channel-LN, no BN/ReLU — see the NetSpec below); EMA + stochastic depth
    wired. No remaining architectural deviation from the paper.

    The verified peer is tied to this file's generated output on shared weights by
    `scripts/parity/convnext_forward_tie.py` (`--net convnextsin` for the S pair), the ConvNeXt sibling
    of the mnv2/mnv4/enet forward ties: max |Δ| 1.464e-03 over the logits, under its 2e-3 bound.
    A divergence here compares two different networks. -/

def convNeXtTinyImagenet : NetSpec where
  name := "ConvNeXt-T (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .convNextStem 3 96 4,                      -- patchify stem: 4×4 s4 conv → channel-LN (faithful, no BN/ReLU)
    .convNextStage 96 3 .ln .gelu,             -- stage 1: 3 blocks @ 96
    .convNextDownsample 96 192,                -- 56→28
    .convNextStage 192 3 .ln .gelu,            -- stage 2: 3 blocks @ 192
    .convNextDownsample 192 384,               -- 28→14
    .convNextStage 384 9 .ln .gelu,            -- stage 3: 9 blocks @ 384
    .convNextDownsample 384 768,               -- 14→7
    .convNextStage 768 3 .ln .gelu,            -- stage 4: 3 blocks @ 768
    .globalAvgPool,
    -- head LayerNorm. The paper is `GAP → LN → Linear`
    -- (`self.norm(x.mean([-2,-1]))`, eps 1e-6) and timm's head is
    -- `NormMlpClassifierHead(global_pool → LayerNorm2d(768) → flatten → fc)`.
    .layerNorm 768,
    .dense 768 1000 .identity                  -- 1000-class head
  ]

/-- ConvNeXt-T, the official recipe (facebookresearch/ConvNeXt's main.py and its README
    command) at its 300-epoch schedule, the full regularizer stack on (EMA + stochastic
    depth). ConvNeXt needs AdamW, not SGD: decoupled weight
    decay 0.05 (excluding norm/bias/LayerScale — timm no_weight_decay), label smoothing 0.1,
    bf16 + bf16 conv. EMA (decay 0.9999) + stochastic depth
    (dropPath 0.1, the ConvNeXt-T paper value) on. Geometric RandAugment (N=2, M=9,
    the ConvNeXt recipe value) on — the full RandAugment(N,M) sampler over the
    color+geometric op set (shear/rotate via ImageProjectiveTransformV3). Mixup α0.8
    + CutMix α1.0 + Random Erasing p0.25 on — the full DeiT-style pack.

    **The official recipe's batch and schedule, 2026-10-12** (`planning/recipe_prep.md` §4; ch 8's
    marker): the values are facebookresearch/ConvNeXt `main.py`'s own defaults and its README
    command for ConvNeXt-T — `--lr 4e-3` at a total batch of 4096 (`--batch_size 128 --update_freq
    4` on 8 cards), here 256 × `gradAccumSteps 16` (ConvNeXt has no BatchNorm, so the accumulation
    is exact and `LR` targets the effective batch); `--min_lr 1e-6` as the cosine's floor;
    `--warmup_epochs 20` ramped PER ITERATION from `utils.cosine_scheduler`'s `start_warmup_value
    = 0` (the emitter's per-step form, so `schedPerEpoch` / `warmupLRInit` stay off); `--clip_grad
    None` (no clip — the 2.5e-4 / batch-256 tier before today clipped at 1.0 as "cheap
    insurance"); `--mixup_switch_prob 0.5` with `--mixup_mode batch` (one draw per batch decides
    CutMix or Mixup); RandAugment's geometric fill at timm's img_mean; timm's `ModelEma` at
    `--model_ema_decay 0.9999`, a constant decay from step 0 (`emaWarmup := false`); and the
    validation forward in f32 (`evalF32`), as `--model_ema_eval` scores through the fp32 model.
    The verified pair render is `convnextin_emaaccdp16xwxdroperfbf16` (4 × 64 × 16).
    -/
def convNeXtTinyImagenetConfig : TrainConfig where
  learningRate   := 0.004   -- the official `--lr 4e-3` AT ITS BATCH of 4096 (effective, below)
  batchSize      := 256     -- the micro-batch, 4 × 64 — what one forward sees
  gradAccumSteps := 16      -- × 16 = 4096, the official total batch; exact without BatchNorm
  epochs         := 300     -- the official schedule
  optimizer      := .adam
  weightDecay    := 0.05
  wdExcludeNormBias := true  -- timm no_weight_decay: skip norm γ/β, biases, LayerScale γ (1-D params)
  minLR          := 0.000001  -- `--min_lr 1e-6`, the per-step cosine's floor
  mixSwitchProb  := 0.5       -- `--mixup_switch_prob 0.5`: CutMix with p 0.5 per batch, else Mixup
  randAugmentFillMean := true -- the geometric ops fill with timm's img_mean (124, 116, 104), not 128
  evalF32        := true      -- score through an f32 forward, as the verified path and timm do
  emaWarmup      := false     -- timm ModelEma: a constant decay from step 0, no ramp-in
  cnxInit        := true    -- ConvNeXt `_init_weights`: trunc_normal(0.02) on every conv AND the
                            -- head. The generic Xavier path puts the stem at std 0.118 against the
                            -- paper's 0.02 — 5.9× too wide, the same failure `vitInit` fixes for
                            -- ViT. Set on the BASE config, not behind a separate recipe the way
                            -- `deit-init` is.
  cosineDecay    := true
  warmupEpochs   := 20      -- ConvNeXt paper warmup
  augment        := true
  useRandAugment       := true   -- ConvNeXt recipe RandAugment...
  augBicubic     := true    -- PIL-bicubic geometry, as timm
  cropTorchvision := true   -- torchvision RandomResizedCrop, not TF's sampler (bicubic resize, the recipe's)
  geluExact      := true    -- nn.GELU's exact x·Φ(x), not jax.nn.gelu's tanh default
  randAugmentGeometric := true   -- ...the full color+geometric sampler (N=2, M=9)
  randAugmentMstd := 0.5         -- ConvNeXt rand-m9-mstd0.5
  randAugmentInc  := true        -- ...-inc1 increasing-severity mappings
  useMixup       := true     -- ConvNeXt paper aug pack: Mixup α0.8...
  mixupAlpha     := 0.8
  useCutmix      := true     -- ...+ CutMix α1.0 (alternates with Mixup per step)...
  cutmixAlpha    := 1.0
  randomErasing  := true     -- ...+ Random Erasing p0.25 — completes the DeiT-style pack
  randomErasingProb := 0.25
  erasingPixel   := true    -- timm RandomErasing(mode='pixel'), N(0,1) fill
  labelSmoothing := 0.1
  gradClipNorm   := 0.0     -- `--clip_grad None`: the official recipe does not clip
  bf16           := true
  bf16Conv       := true
  f32StemHead    := true    -- classifier head in fp32, as every verified render keeps it
  useEMA         := true     -- weight averaging (decay 0.9999) — eval + ckpt use it
  dropPath       := 0.1      -- stochastic depth, ConvNeXt-T paper value

#eval convNeXtTinyImagenet.validate!

/-- `full` is the same config, kept because `scripts/jobs/cnx-default-jax-4gpu.conf` and the
    CI emission check name its emitted file `generated_convnext_tiny_imagenet_full.py`. -/
def convNeXtTinyImagenetConfigFull : TrainConfig := convNeXtTinyImagenetConfig

def convNeXtTinyImagenetRecipes : List Recipe := [
  { name := "default", cfg := convNeXtTinyImagenetConfig,
    out := "generated_convnext_tiny_imagenet.py",
    desc := "the official 300-epoch run (full aug pack + geometric RandAugment)" },
  { name := "full",    cfg := convNeXtTinyImagenetConfigFull,
    out := "generated_convnext_tiny_imagenet_full.py",
    desc := "the same run, under the name the jobs call it by" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "convnext-tiny-imagenet" convNeXtTinyImagenet .imagenet
    convNeXtTinyImagenetRecipes args
