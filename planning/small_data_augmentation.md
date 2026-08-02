# Data augmentation at small data scale — parked for a home of its own

Pulled out of the blueprint on 2026-08-02. Both chapters 8 and 9 closed with a
"Data Augmentation" section that was really an *Imagenette* ablation: measuring
the marginal effect of each augmentation knob at 9.5K images. The finding is
real and worth keeping, but it does not belong at the end of an architecture
chapter whose next section is the ImageNet recipe — the reader is being handed
a small-data result right where they are about to read a large-data one.

The citation/implementation prose from ConvNeXt's version was kept and folded
into ch8's "ImageNet recipe" section (it explains what the DeiT pack *is*). The
measurements below came out entirely.

**The thread to write:** one section, somewhere of its own, on augmentation as
a function of data scale — "marginal effect on top of what you already have,"
not the "stack everything" framing implicit in published paper recipes. Two
architectures agreeing that CutMix is the single load-bearing knob at ~9.5K
images is the spine of it.

## ConvNeXt-T on Imagenette (was §8.4)

Architecture and base optimizer fixed, each knob layered on the bare config,
same Imagenette data and 80-epoch budget.

| Cell | Val acc | Δ vs bare |
|---|---|---|
| `convnext-tiny-gelu-cutmix` | 87.81% | +2.9 |
| `convnext-tiny-gelu-erase` | 85.63% | +0.7 |
| `convnext-tiny-gelu-randaug` (M=9) | 85.48% | +0.5 |
| `convnext-tiny-gelu` (bare) | 84.94% | — |
| `convnext-tiny-gelu-mixup` | 83.45% | −1.5 |

- **CutMix is the load-bearing knob.** +2.9% over bare, a single config change.
- **Random Erasing and RandAugment at M=9 are in the same tier.** +0.7% and
  +0.5%; both at the edge of seed noise. M=9 is too aggressive for our
  9.5K-image scale (the paper trained on 1.2M images), and erasing 25% of
  pixels is in the same noise band.
- **Mixup actively hurts at this scale.** −1.5% below bare. The blended-label
  gradient signal is too aggressive at ~475 images per class; the model can't
  extract a clean target from a Beta(0.8, 0.8) mix of two images when each
  class has so few exemplars. A Mixup ablation on full ImageNet (1.28M images,
  ~1280 per class) typically lifts +0.5 to +1.0; here we're 100× below that
  data scale.

One seed per cell. The +0.5% and +0.7% deltas are within noise; the +2.9% from
CutMix and the −1.5% from Mixup are well above.

## ViT-Tiny on Imagenette (was §9.5)

Base recipe is bare — random crop + hflip, no Mixup / CutMix / RandAugment /
heavy weight decay. Same data and 80-epoch budget.

| Cell | Val acc | Δ vs bare |
|---|---|---|
| `vit-tiny-cutmix-wd05` | 77.43% | +5.7 |
| `vit-tiny-cutmix` | 77.10% | +5.4 |
| `vit-tiny-mixup` | 74.69% | +3.0 |
| `vit-tiny-recipe2` | 74.41% | +2.7 |
| `vit-tiny-full` | 74.23% | +2.5 |
| `vit-tiny-randaug` (M=9) | 74.03% | +2.3 |
| `vit-tiny-randaug-m4` (M=4) | 71.98% | +0.3 |
| `vit-tiny-bare` | 71.70% | — |
| `vit-tiny-erase` | 70.62% | −1.1 |

(`recipe2` = CutMix + Random Erasing + RandAugment-M9; `full` = Mixup + Random
Erasing.)

- **CutMix is the load-bearing knob.** +5.4% over bare. No close substitute;
  Mixup is a partial second at +3.0%.
- **Stacking on top of CutMix gives near-zero return.** CutMix + heavy WD adds
  +0.3%, well within seed noise. CutMix + RA + RE (`recipe2`) is *negative*
  (−2.7% vs CutMix alone): at 9.5K training images, piling on more aug erodes
  the per-image class signal faster than the diversity helps.
- **Random Erasing alone hurts** (−1.1%) at this scale. Removing 2–33% of
  pixels per image needs more redundant signal elsewhere than a 9.5K-image
  dataset has.
- **RandAugment magnitude matters.** M=9 (paper default) works (+2.3%). M=4 is
  essentially identity (+0.3%); too gentle to bite.

The reading for a small-data practitioner: pick one strong knob (CutMix) and
stop. The reading for a large-data practitioner: the same table at 1.2M images
would lift different rows — the marginal effect of each knob is scale-dependent.

One seed per cell. The 0.3% delta between CutMix and CutMix+WD is below noise;
the 5.4% delta from CutMix is well above.

## Cross-architecture note

Both ablations independently rank CutMix ≫ RandAugment at the same data scale,
which suggests the ranking is data-regime-driven rather than
architecture-driven. That cross-check was stated in §8.4 and is the reason the
two tables belong in one place rather than two.
