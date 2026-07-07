import Jax

/-! ResNet-50 (bottleneck) on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.

    Architecture is `MainResnet50.lean`'s bottleneck backbone (3/4/6/3) with the
    head swapped to `dense 2048→1000` and the dataset kind set to `.imagenet`
    (tfds streaming), matching `MainResnetImagenet.lean` (the R34 ImageNet trainer).

    PHASE 5 (RSB-A2 plan): the recipe below is now the LITERAL RSB-A2 — timm's
    "ResNet Strikes Back" A2 300-epoch config (Wightman et al. 2021), the
    canonical modern ResNet-50 baseline → 79.8% top-1. All four ingredients
    (LAMB, BCE, repeated-aug, the DeiT-style aug stack) landed across phases 1-4
    and are wired together here. The phase-1 SGD skeleton lives in git history.
    See `planning/rsb_a2_resnet50.md`. -/

def resnet50Imagenet : NetSpec where
  name := "ResNet-50 (ImageNet)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .bottleneckBlock   64  256 3 1,
    .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 1000 .identity
  ]

/-- Literal RSB-A2 (timm "ResNet Strikes Back" A2, Wightman et al. 2021) → 79.8%
    top-1. LAMB + BCE-with-logits over multi-hot targets, the DeiT-style aug pack
    (Mixup 0.1 + CutMix 1.0 + RandAugment m7-mstd0.5-inc1), Repeated Augmentation
    3×, stochastic depth 0.05, weight decay 0.02, model-EMA, 300 epochs with a
    5-epoch warmup + cosine.

    LR: RSB-A2's reference is lr 5e-3 @ batch 2048; LINEAR-scaled to our batch
    512 (2× 7900 XTX, 256/GPU) → 5e-3 × 512/2048 = 1.25e-3. No label smoothing —
    BCE over the mixup/cutmix soft targets subsumes it (the RSB recipe).

    Conv dtype: `bf16Conv := true` — R50 is conv-bound and its real home is the
    CUDA box (ares), where bf16 conv on cuDNN tensor cores is ~1.6× faster
    (measured: 458→ vs 737 ms/step on 4× 4060 Ti, A2@224). On ROCm/MIOpen bf16
    conv is slower but still correct, so this stays on for both. RandAugment +
    3× repeated-aug are CPU-side tf.data — watch input throughput (the warmup
    ETA check confirms it's not input-bound). -/
def resnet50ImagenetConfig : TrainConfig where
  learningRate   := 0.00125  -- RSB-A2 lr 5e-3 @ batch 2048, linear-scaled to batch 512
  batchSize      := 512
  epochs         := 300
  optimizer      := .lamb
  weightDecay    := 0.02
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true     -- random-resized-crop + hflip (the base aug under RA)
  labelSmoothing := 0.0      -- BCE over mixup/cutmix soft labels subsumes it (RSB)
  lossKind       := .bce     -- BCE-with-logits, multi-hot (timm --bce-loss)
  useMixup       := true     -- RSB aug pack: Mixup α0.1...
  mixupAlpha     := 0.1
  useCutmix      := true     -- ...+ CutMix α1.0 (alternates per step)...
  cutmixAlpha    := 1.0
  useRandAugment := true     -- ...+ full RandAugment (color+geometric)...
  randAugmentGeometric := true
  randAugmentN   := 2
  randAugmentM   := 7.0      -- RSB rand-m7-...
  randAugmentMstd := 0.5     -- ...-mstd0.5-...
  randAugmentInc := true     -- ...-inc1 increasing-severity mappings
  repeatedAug    := 3        -- RSB Repeated Augmentation 3× (phase 2)
  dropPath       := 0.05     -- stochastic depth, RSB-A2 value
  useEMA         := true     -- model EMA; eval + checkpoints use the shadow
  emaDecay       := 0.9999
  bf16           := true
  bf16Conv       := true     -- CUDA/cuDNN: bf16 conv ~1.6× faster (R50 is conv-bound, ares is its home); slower-but-correct on ROCm
  runningBN      := true     -- paper-faithful eval (gap A) + bottleneck running-BN

#eval resnet50Imagenet.validate!

/-- The short / validation tier is the **literal RSB-A3** (timm "ResNet Strikes
    Back" A3, the 100-epoch tier) → **78.1% top-1** — a faithful, far cheaper
    validation than truncating A2. Same LAMB + BCE core, but: 100 epochs, **train
    @160 / test @224 (crop 0.95)** — the resolution split is ~2× faster/step, so A3
    is ~6× cheaper than the 300-ep A2 (~10-11 hr on ares vs ~60-65 hr).

    Deltas vs A2 (decoded from timm's a3 args
    `lamb-cosine-lr0.008-wd0.02-n0-rand-m6-mstd0.5-inc1-m0.1-sd0.0-d0.0-ls0.0-100`):
    lr 8e-3@2048 → 0.002@512, RandAugment m7→m6, NO repeated-aug (n0), NO stochastic
    depth (sd0.0), NO model-EMA, 100 ep, 160/224 split. Mixup 0.1 / CutMix 1.0,
    wd 0.02, BCE, geo-RA mstd0.5-inc1 all carry over from A2.
    The `short` recipe arg; writes a separate `_short.py`. -/
def resnet50ImagenetConfigShort : TrainConfig :=
  { resnet50ImagenetConfig with
      learningRate  := 0.002    -- RSB-A3 lr 8e-3 @ batch 2048, linear-scaled to 512
      epochs        := 100
      randAugmentM  := 6.0      -- A3 rand-m6 (A2 is m7)
      repeatedAug   := 1        -- A3: no repeated augmentation (n0)
      dropPath      := 0.0      -- A3: no stochastic depth (sd0.0)
      useEMA        := false    -- A3: no model EMA
      trainRes      := 160      -- A3: train @160×160
      testCropRatio := 0.95 }   -- A3: eval @224, center-crop ratio 0.95

/-- Optimizer-regime probe (diagnosing the ~41% RSB-A3 result). Same A3 recipe
    but swaps LAMB→AdamW (LAMB is a large-batch optimizer; we run bs512) and adds
    the timm no_weight_decay skip-list (BN γ/β + biases excluded from wd). NB on
    the JAX path `.muon` also degrades to AdamW; we use `.adam` here for clarity.
    Keeps epochs=100 so the cosine LR schedule matches the baseline — the probe
    only RUNS the first ~10 epochs, so val@ep10 is comparable to the LAMB run. -/
def resnet50ImagenetConfigAdamProbe : TrainConfig :=
  { resnet50ImagenetConfigShort with
      optimizer         := .adam   -- AdamW (== Muon's JAX fallback); bs512-appropriate
      wdExcludeNormBias := true }  -- skip BN γ/β + biases from weight decay

/-- **RSB-faithful A3** — reproduces timm's LAMB @ **bs2048** on this 4×16 GB box via
    gradient accumulation (512 micro × 4 = effective 2048), so LAMB finally gets the
    large batch it was designed for. This is the fix for the bs512-starved 40.8%
    result (memory `project_r50_a3_lowval_diagnostic`): LAMB is a large-batch
    optimizer that was run at 1/4 its intended batch. BN stats are per-micro-batch
    (**Ghost-BN**, Hoffer et al. 2017 — benign at micro=512). LR is restored to the
    paper's **8e-3 @ bs2048** (NOT the 512-scaled 2e-3), and the timm no_weight_decay
    skip-list (BN γ/β + biases) — the other faithful-reproduction lever — is on.
    Grad-accum mechanics are GPU-validated (see `planning/grad_accum.md` §Status);
    this config is the accuracy run. The `rsb-faithful` recipe arg; writes a
    separate `_rsbfaithful.py`. -/
def resnet50ImagenetConfigRSBFaithful : TrainConfig :=
  { resnet50ImagenetConfigShort with
      learningRate      := 0.008   -- RSB-A3 lr @ bs2048 (NOT the 512-scaled 0.002)
      gradAccumSteps    := 4       -- 512 micro × 4 = effective bs2048 on 4×16GB
      wdExcludeNormBias := true }  -- timm no_weight_decay: BN γ/β + biases skip wd

/-- **True bs2048** — the `rsb-faithful` recipe with the gradient-accumulation
    crutch removed: one single-forward batch of 2048 (`gradAccumSteps := 1`), which
    needs a big-memory card (**~80 GB**; a 48 GB card is borderline/OOM — see
    `project_r50_a3_lowval_diagnostic`). This eliminates the Ghost-BN approximation
    of the accum path (each accum micro-step normalized over its own 512 rather than
    the full 2048), so LAMB and BN both see a genuine 2048-sample batch — the cleanest
    reproduction of timm's bs2048.

    BN CAVEAT: on a single device this normalizes over the full 2048 (one running-stat
    update/step). That is a LARGER BN batch than timm's *per-GPU* BN (timm trained
    multi-GPU, so its per-GPU BN batch was 2048/n_gpu). So this is "clean big-batch BN,"
    not a literal match of timm's per-GPU BN — a deliberate, documented trade. Same lr
    0.008@2048, wdExcludeNormBias, everything else identical to `rsb-faithful`.
    Selected with recipe arg `true-2048`. -/
def resnet50ImagenetConfigTrue2048 : TrainConfig :=
  { resnet50ImagenetConfigShort with
      learningRate      := 0.008   -- RSB-A3 lr @ bs2048 (native, no scaling)
      batchSize         := 2048    -- true single-forward bs2048 (needs ~80 GB; no grad-accum)
      wdExcludeNormBias := true }  -- timm no_weight_decay: BN γ/β + biases skip wd
      -- gradAccumSteps stays at the default 1: no accumulation.

/-- A named training recipe: a `TrainConfig`, its generated-file name, and a
    one-line description. Recipe selection is a positional CLI arg
    (`resnet50-imagenet <recipe> [data_dir]`), listed by `--help` — replacing the
    old undiscoverable `LEAN_MLIR_*` env flags (still honored as a fallback). -/
def resnet50ImagenetRecipes : List Recipe := [
  { name := "default",      cfg := resnet50ImagenetConfig,
    out := "generated_resnet50_imagenet.py",
    desc := "RSB-A2, 300 epochs, bs512 (the full recipe)" },
  { name := "short",        cfg := resnet50ImagenetConfigShort,
    out := "generated_resnet50_imagenet_short.py",
    desc := "RSB-A3, 100 epochs, bs512, train@160/eval@224" },
  { name := "rsb-faithful", cfg := resnet50ImagenetConfigRSBFaithful,
    out := "generated_resnet50_imagenet_rsbfaithful.py",
    desc := "RSB-A3 + gradient accumulation -> effective bs2048 (LAMB's design batch)" },
  { name := "true-2048",    cfg := resnet50ImagenetConfigTrue2048,
    out := "generated_resnet50_imagenet_true2048.py",
    desc := "RSB-A3 at a real single-forward bs2048 (no grad-accum; needs ~80GB)" },
  { name := "adam-probe",   cfg := resnet50ImagenetConfigAdamProbe,
    out := "generated_resnet50_imagenet_adamprobe.py",
    desc := "A3 optimizer probe: AdamW + no-weight-decay on norm/bias" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "resnet50-imagenet" resnet50Imagenet .imagenet resnet50ImagenetRecipes args
