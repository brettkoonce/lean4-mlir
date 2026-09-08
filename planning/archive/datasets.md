# datasets.md — drop-in dataset variants

Goal: add three datasets that **reuse existing loaders / formats** for
free ablation surface. Each is a near-copy of a `DatasetIO` struct
already in `LeanMlir/Train.lean`. No FFI changes, no codegen changes.

The three:

| New | Mirrors | Same on-disk format? | Why we'd want it |
|---|---|---|---|
| KMNIST | mnist | yes (IDX) | "MNIST is too easy" — top-1 ~98% vs MNIST's ~99.5% |
| Fashion-MNIST | mnist | yes (IDX) | Slight difficulty bump (~93% top-1), preserves Ch 3 plumbing |
| Imagewoof | imagenette | yes (our `.bin` format) | Fine-grained variant — 10 dog breeds vs 10 easy classes |

## Why these three specifically

The blueprint's "MNIST is solved, here's CNN→ResNet→ConvNeXt" arc
loses some bite if every CNN gets 99% on MNIST. Fashion-MNIST and
KMNIST give the same chapters real headroom to differentiate
architectures without changing the input pipeline at all.

Imagewoof is the same trick at the ImageNet end of the curve. A model
that hits 90% on Imagenette might only do 70% on Imagewoof — the
difference exposes whether the network actually learned discriminative
features or just the "outdoor scene vs indoor scene" gross gist.

## Implementation pattern

For each, three steps:

1. **Add to the `DatasetKind` enum** in `LeanMlir/Types.lean`.
2. **Add a `DatasetIO` instance** in `LeanMlir/Train.lean`.
   - For KMNIST/FMNIST: copy `mnistIO` verbatim — the IDX file names
     are identical (`train-images-idx3-ubyte`, etc.); only the
     directory differs and that's already an arg.
   - For Imagewoof: copy `imagenetteIO` verbatim. Same `.bin` format,
     same image sizes. Only the data path changes.
3. **Add to the `datasetIO` dispatcher.**

The simplest realization re-uses the existing IO struct and only
distinguishes by name in logging:

```lean
private def datasetIO : DatasetKind → DatasetIO
  | .imagenette => imagenetteIO
  | .imagewoof  => imagenetteIO
  | .mnist      => mnistIO
  | .kmnist     => mnistIO
  | .fmnist     => mnistIO
  | .cifar10    => cifar10IO
```

Plus the dataset-name string in `runTraining`'s log line:

```lean
| .kmnist     => "kmnist"
| .fmnist     => "fashion-mnist"
| .imagewoof  => "imagewoof"
```

That's the entire codebase change. ~10 lines total.

## Data acquisition

| Dataset | Source | Size | Conversion needed |
|---|---|---|---|
| KMNIST | rois-codh/kmnist (CC BY-SA 4.0) | 18 MB | None — already IDX |
| Fashion-MNIST | zalandoresearch/fashion-mnist (MIT) | 30 MB | None — already IDX |
| Imagewoof | fastai/imagenette (Apache 2.0) | ~330 MB | Run our existing imagenette → .bin converter pointed at `imagewoof2-160` |

All three are CC-style licensed, free to ship.

For Imagewoof, the converter probably needs no change — our
`imagenette → .bin` script walks `train/<class>/*.JPEG` and writes
the same 224×224 normalized binary. Imagewoof has the same dir
structure. Verify, but expect zero modification.

## Per-dataset notes

### KMNIST

- 60K train / 10K test, 28×28 grayscale, 10 classes (cursive Japanese
  hiragana characters).
- Top-1 baselines: ~95% (small CNN), ~98% (ResNet-class).
- Class semantics are wholly unfamiliar to most readers — useful as a
  "the network is learning shapes, not your prior" demonstration.
- Pair with the MNIST CNN ablation table for a sibling comparison.

### Fashion-MNIST

- Same shape and counts as MNIST. 10 classes of clothing.
- Top-1 baselines: ~91% (MLP), ~93% (small CNN), ~94.5% (ResNet).
- Cleanest "MNIST replacement" — no exotic format, just harder.
- Useful early in the book to pre-empt "MNIST is too easy" critique
  without having to introduce CIFAR's preprocessing details.

### Imagewoof

- 10 dog breeds (Australian terrier, Border terrier, Samoyed, Beagle,
  Shih-tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden
  retriever, Old English sheepdog).
- Same 224×224 / 256×256 image sizes as Imagenette.
- Top-1 baselines (R34): ~75% vs Imagenette's ~93%. Real headroom.
- Story: "your CNN's 90% on Imagenette doesn't mean it 'sees' dogs."

## Ablation entries to add

Mirror the existing entries 1:1, swapping `.imagenette` → `.imagewoof`
and `.mnist` → `.kmnist` / `.fmnist`. Concrete proposals:

- `mnist-cnn-bare`        → `kmnist-cnn-bare`, `fmnist-cnn-bare`
- `cifar-bn-bare`         → keep CIFAR (orthogonal)
- `r34-imagenette-bare`   → `r34-imagewoof-bare`
- `enet-b0-imagenette-*`  → `enet-b0-imagewoof-*` (same swish/relu cells)
- `convnext-tiny-imagenette-*` → `convnext-tiny-imagewoof-*`

Skip ViT-Tiny on Imagewoof for now — Imagenette ViT runs already take
~2 hr per cell on a single GPU and the Imagewoof variant is harder, so
it'd push ablation runs over the practical limit.

## Out of scope

- TinyImageNet (200 classes, 64×64): different format + different
  pipeline assumptions (non-square preprocessing, separate val).
- ImageNet-100, ImageNet-1k: out — disk + compute too far past where
  the blueprint pitches its complexity ceiling.
- CIFAR-100: same format as CIFAR-10 but different `n_classes` (100).
  Adds the "more classes ≠ harder per se" wrinkle, which is its own
  conversation. Defer.
- DomainNet, ImageNet-R, etc. (distribution-shift evaluation):
  different paper, different chapter.

## Open questions

1. **ImageNet normalization stats for Imagewoof?** Likely identical to
   Imagenette's `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`
   (the existing ImageNet defaults bake into our loader). Confirm by
   running val accuracy with both — if Imagewoof std differs
   meaningfully, we'll see it.

2. **One ablation table per dataset, or merged?** The blueprint
   currently has Imagenette tables in Ch 7+. For Imagewoof either
   add a sibling table or extend with an extra column per cell. Lean
   toward sibling table for visual clarity.

3. **KMNIST + Fashion-MNIST in the same chapter, or one per chapter?**
   Both replace MNIST. Probably one chapter section saying "we re-ran
   the MLP and CNN families on these three (MNIST, F-MNIST, KMNIST)
   and here's the difficulty curve."

## Estimated scope

| Task | Effort |
|---|---|
| Wire 3 enum variants + dispatch | ~30 min |
| Run KMNIST/FMNIST through existing MNIST ablations | ~2 hr (compute) |
| Run Imagewoof through existing Imagenette ablations | ~6–8 hr (compute, 1 GPU) |
| Update blueprint tables | ~1 hr |

Code change is trivial; bulk of effort is GPU-time for re-running
existing ablation cells on the harder variants.
