# resnet50_imagenet.md — 300-epoch ResNet-50 on ImageNet (Lean→JAX)

Plan for a modern long-schedule ResNet-50 ImageNet run. Written 2026-06-19.
Motivation: R50 @ ~79–80% top-1 is *the* canonical ImageNet number — the most
recognizable result in the book — and (the practical part) it needs **almost no
new code**: the bottleneck block and the tfds ImageNet path both already exist.
It would be the book's strongest ImageNet result (vs ConvNeXt-T 75.93%, ENet-B0
72.31%, R34 72.1%) and validate the bottleneck-block codegen at full scale + long
schedule. Pairs naturally with a 300ep **ResNet-34** companion run (even less
work — see the Companion section below) for a basic-block-vs-bottleneck comparison
at a fixed, fair 300-epoch schedule.

## What already exists (the practical part)

- **`.bottleneckBlock` layer type** — works today in `jax/MainResnet50.lean`
  (ResNet-50 on *Imagenette*, ~23.5M params, the canonical 3/4/6/3 stage counts:
  `.bottleneckBlock 64 256 3 1` … `1024 2048 3 2`). Bottleneck codegen is proven.
- **`.imagenet` tfds streaming dataset** — works today in
  `jax/MainResnetImagenet.lean` (R34 @ ImageNet, the 72.1%/90.7% 90ep run).
- **The full modern aug pack + grad-clip + EMA** — Mixup/CutMix/RandAugment(geo)/
  RandomErasing, `gradClipNorm`, `useEMA`, label smoothing all wired (used by
  ConvNeXt/ViT). So the recipe knobs are all available as config flags.

## Code delta (small — config/wiring, NO codegen)

1. **New `jax/MainResnet50Imagenet.lean`** = `MainResnet50.lean`'s bottleneck
   backbone with the head widened to `.dense 2048 1000`, dataset `.imagenet`,
   output `generated_resnet50_imagenet.py`, and the 300ep recipe below. (Merge of
   `MainResnet50.lean` architecture + `MainResnetImagenet.lean` ImageNet pattern.)
2. **`lakefile.lean`**: add a `resnet50-imagenet` exe (mirror the `resnet50` /
   `resnet34-imagenet` entries).
3. **Supervisor** `scripts/supervise_r50_300ep_6gpu.sh` (mirror the convnet 6-GPU
   ones: DEVS 0-5, SPE 5083, AER watchdog + per-epoch auto-resume; EPOCHS 300).
4. **Eval** `scripts/eval_r50_full50k.py` (mirror `eval_mnv2_full50k.py`, EMA + full
   50k, drop_remainder=False).

No `Codegen.lean` changes needed (unlike the ViT stochastic-depth item) — *unless*
we want stochastic depth (see open questions).

## TARGET: literal RSB-A2 (chosen 2026-06-22)

Decision: reproduce the **actual timm RSB-A2 recipe** (Wightman et al. 2021, 300ep → **79.8%**),
not the SGD "Bag of Tricks" approximation below. RSB-A2 is *the* recognizable modern ResNet-50
baseline and an honest one to put in the book — worth the new code. The SGD recipe in the next
section becomes the fallback / fair-comparison point (and is near-free given what's landed).

**RSB-A2 spec:** LAMB optimizer, lr 5e-3 @ batch 2048 (scaled), 300ep, 5ep warmup + cosine, WD 0.02,
**BCE loss** (multi-hot targets, no label smoothing), Mixup 0.1 + CutMix 1.0, RandAugment m7/mstd0.5,
**Repeated Augmentation (3×)**, stochastic depth 0.05, bf16, test crop ratio 0.95.

**Gap to RSB-A2** (vs everything landed through 2026-06-22):

*Config-only / exists:* bottleneck arch (proven in `MainResnet50.lean`), `.imagenet` path, the aug
pack incl. RandAugment **mstd/inc1** (gap D), grad-clip, EMA, cosine+warmup, plus the new
`MainResnet50Imagenet.lean` (config merge, no codegen — see "Code delta" above). RSB knobs (RA m7,
mixup 0.1, WD 0.02) are one-liners.

*Mechanical (mirror the gap-A / stochastic-depth threading just done for basic_block):*
- **Bottleneck running-BN** — gap A is wired for `basic_block` (r34) but NOT `bottleneck_block`
  (the only un-threaded BN block helper left). ~2–3 hr, same pattern.
- **Bottleneck stochastic depth** — `dropPath` exists but isn't threaded into `bottleneck_block`
  (only conv/mbconv/convnext). ~couple hr.

*Genuinely new — the RSB-A2 signature, ~2–3 days total with the above:*
1. **LAMB optimizer** — A2's defining optimizer; repo has only sgd/adam/rmsprop. New
   `OptimizerKind.lamb` + the layer-wise trust-ratio update + state. ~½ day (biggest piece).
2. **BCE loss** — binary cross-entropy over multi-hot mixup/cutmix targets (RSB showed BCE > CE
   for this recipe). New `LossKind.bce` (current: classCE/softLabelCE/perPixelCE). ~2–3 hr.
3. **Repeated Augmentation (3×)** — still unimplemented (the deferred ViT-spike gap); data-pipeline
   change + throughput risk. ~½ day. **Double-counts**: it also unlocks faithful DeiT-Ti (ViT's one
   remaining gap), so build it once for both.

So: literal RSB-A2 = **3 new features (LAMB, BCE, repeated-aug) + bottleneck running-BN/SD threading
+ the new Main**. Everything else rode in on the gap-A–D work. Compute below (~60–65 hr) is unchanged;
LAMB at batch 2048 may shift the per-step cost — measure first 400 steps.

## Recipe (300ep, modern SGD — "Bag of Tricks"/RSB-flavored) — FALLBACK / comparison point

Reference points: RSB-A2 (Wightman et al. 2021, 300ep, LAMB) hits 79.8%; "Bag of
Tricks" (He et al. 2018, 120ep, SGD) hits ~79.3% for ResNet-50-D. We have SGD, not
LAMB, so target the Bag-of-Tricks/SGD lineage — proven to reach ~79%.

```
def resnet50ImagenetConfig : TrainConfig where
  learningRate   := 0.1          -- SGD base @ batch 256 (R34-proven)
  batchSize      := 256          -- 252 (6x42) at run time
  epochs         := 300
  useAdam        := false        -- SGD + momentum (ResNets; see feedback_optimizer_choice)
  momentum       := 0.9
  weightDecay    := 1e-4         -- standard R50; consider 5e-5 for the long schedule
  cosineDecay    := true
  warmupEpochs   := 5
  labelSmoothing := 0.1
  augment        := true         -- random-resized-crop + flip
  useMixup       := true         -- Mixup 0.2 (Bag-of-Tricks value; lighter than ConvNeXt's 0.8)
  mixupAlpha     := 0.2
  useCutmix      := true         -- optional; RSB uses it. start with mixup-only if unsure
  cutmixAlpha    := 1.0
  useRandAugment := true
  randAugmentGeometric := true   -- full RA (impl exists)
  useEMA         := true         -- decay 0.9999, cheap win
  bf16           := true
  bf16Conv       := true         -- CUDA: 1.60x vs fp32 (R34 finding); set false on AMD/MIOpen
```

Note: heavier ConvNeXt-style aug (Mixup 0.8 + RandErase) over-regularizes a plain
ResNet — keep Mixup light (~0.2) for R50. The long schedule is the main lever.

## Compute estimate (6× 4060 Ti, ares, CUDA, bf16)

R50 is ~4.1 GFLOPs / 25.6M params — close to ConvNeXt-T (~4.5 GFLOPs / 28.6M, which
ran ~12.6 min/epoch on this box), and R50's plain convs are friendlier to bf16 than
ConvNeXt's 7×7 depthwise. Estimate **~11–13 min/epoch** → 300ep ≈ **~60–65 hr
(~2.5–3 days)**. Heavy aug adds host-pipeline cost (the ENet ~65%-util bottleneck);
watch throughput — may stretch toward 3+ days. A 2.5–3-day uninterrupted run on
ares is a real gamble (PCIe AER hard-resets + the 2026-06-19 wall-power outage):
the AER-watchdog/auto-resume supervisor is mandatory, and checkpoint hygiene must
be solid before committing days of compute.

## Expected accuracy

| recipe | top-1 |
|--------|-------|
| vanilla R50, 90ep SGD | ~76% |
| Bag of Tricks, 120ep SGD + tricks | ~79.3% |
| RSB-A2, 300ep LAMB + heavy aug | 79.8% |
| **this plan** (300ep SGD + light Mixup/CutMix/RA + EMA) | **~78–79.5%** (est.) |

Would be the book's top ImageNet number and the recognizable "~80%" milestone.

## Open questions

- **Stochastic depth?** `dropPath` is wired for mbconv/convnext/invres blocks but
  (verify) *not* `.bottleneckBlock`/`.residualBlock`. Bag-of-Tricks/SGD R50 reaches
  ~79% **without** it, so the zero-codegen plan skips SD. Adding it (codegen, like
  the ViT item) would be a separate ~half-day if we want the last fraction.
- **Optimizer**: SGD+momentum (proven, zero-code) vs LAMB (RSB's choice, not in the
  codebase → codegen). Recommend SGD for the first run.
- **Aug strength**: start Mixup-only (α0.2); add CutMix if the curve looks
  under-regularized late. Avoid the ConvNeXt-weight (α0.8) pack on a plain ResNet.
- **Box**: ares (6× 4060 Ti, CUDA) per the estimate above; bf16Conv on. mars (ROCm)
  would need bf16Conv off + the RCCL/LD_PRELOAD multi-GPU fix.

## Companion: ResNet-34 @ 300ep (even less work — do both as a pair)

R34 already has a full ImageNet trainer — `jax/MainResnetImagenet.lean`, the 72.1%/90.7%
90ep run. A 300ep modern-recipe R34 is the **lowest-effort run in the whole sweep**: bump
`epochs := 90 → 300`, add the same light Mixup/RandAug flags as the R50 block above,
re-emit. No new file, no codegen, no lakefile change (the `resnet34-imagenet` exe exists).

Running R34 and R50 at the *same* 300ep modern recipe is a clean **basic-block vs
bottleneck** comparison at fixed schedule — a nice pairing for the book.

- **Est:** ~74–75% top-1 (vs 72.1% at 90ep). R34's lower capacity caps the long-schedule
  gain below R50's ~79% — itself an interesting result (depth/width matters more than
  schedule once you're already near a small net's ceiling).
- **Compute:** ~9 min/epoch × 300 ≈ **~45 hr** (~1.9 days) on 6× 4060 Ti.
- **Recipe:** identical to the R50 config above (SGD 0.1 + cosine + light Mixup α0.2 +
  RandAug + EMA + label smoothing + bf16) — just keep R34's existing residual-block
  backbone and `.dense 512 1000` head.

Do them back-to-back (R34 then R50, ~105 hr / ~4.4 days total) for the matched pair, or
R50 alone for just the headline ~80% number.

## Related

- `planning/r34_imagenet.md` — the R34 ImageNet run (timing/recipe baseline).
- `planning/jax_imagenet_sweep.md` — the convnet sweep + paper-length TODO + gap table.
- blueprint Ch.6 §"What's in the production recipe?" — DAWNBench fast-end recipe; the
  long-schedule end (RSB 600ep → 80.4%) is the proposed addition that this run embodies.
- memory `feedback_optimizer_choice` (SGD+momentum for ResNets),
  `project_r34_imagenet_bf16` (bf16 1.60x on CUDA).
