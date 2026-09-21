# BN statistic-group, augmented probe — **a bounded negative result**

**2026-09-20/21.** `planning/global_bn_verified.md` §1, second round, after
`runs/2026-09-20-bn-group-sweep/` turned out to have run the no-augmentation probe.
Two BN groupings × two seeds, one per 4060 Ti, 80 epochs, fully annealed.

## ⛔ The finding: Imagenette cannot resolve this question

| | global 192 | group 48 | Δ |
|---|---|---|---|
| seed 0 | 41.38 | 48.05 | **+6.67** |
| seed 1 | 42.27 | 47.85 | **+5.58** |
| mean | 41.83 | 47.95 | **+6.13** |

Consistent across seeds in direction and magnitude. **And not usable for §3.5**, for three
reasons that together are conclusive:

1. **The regime is wrong.** Both arms sit at 41–48 % top-1 with eval loss ≈ 5.0 against
   `ln 10 = 2.30` — heavy overfitting. R50's 23.5 M parameters against 9,469 images. In that
   regime the noise in a small BN group is the main regulariser available, so the gap measures
   regularisation headroom, not BatchNorm.
2. **The sign is opposite to the real pairs.** Here the verified side's grouping (48) looks
   **+6.13 better**. On the committed ImageNet pairs the reference won slightly at convergence:
   ResNet-34 $-0.10$, MobileNetV2 $+0.01$.
3. **Augmentation barely moved it.** The no-aug round gave $+4.60$ at the same 4× point; adding
   the probe's crop gave $+6.13$. If this were the BN-group effect rather than a regularisation
   artefact, closing most of the regularisation gap should have shrunk it.

⚠ **The probe's "augmentation" is much weaker than the reference's**, which is why (1) survived
the switch. `probe_resnet50_imagenette.py` does `np.pad(224→252)` then crops back — translation
only, ±14 px, and **one offset shared by all 192 images in the batch**. The reference
(`generated_*_imagenet.py`) does `tf.image.sample_distorted_bounding_box`: Inception-style
random-resized-crop, per image, with scale and aspect-ratio jitter. These are not the same
recipe, and the probe's does not stop R50 overfitting Imagenette.

## What to do instead

⭐ **The best available evidence on the converged BN-group effect is already in the book** —
ResNet-34's and MobileNetV2's own epoch tables, on the real nets at real scale under the real
recipes, both converging to ≤ 0.10. No Imagenette proxy improves on that.

▶ If a direct number is wanted, it is a **~30-epoch ImageNet ResNet-34 reference at BN64 vs
BN256** — the actual confound, no overfitting, same lowerer in both arms. ~4.5 h per arm on four
cards. That is the only version of §1 worth running.

⛔ Do not spend more on Imagenette for this question.

## Setup (unchanged from the first round except the probe)

Paired by construction: within a seed, identical init, shuffle, crop stream and schedule — the BN
grouping is the only difference, so the per-seed Δ is the estimator. Batch 192, Adam lr 1e-3,
wd 1e-4, 3-epoch warmup, cosine to 0, f32, one card each. Eval pinned to global BN in every arm.
`SEED` shifts init and data order together.

⚠ The seed-to-seed spread on Δ is 1.09 points at n=2. Anything under ~1 point would not have been
resolvable here even in the right regime.

## Provenance

`GHOST_BN_GROUPS={1,4} SEED={0,1} EPOCHS=80 CUDA_VISIBLE_DEVICES=<n> python -u
jax/scripts/bn_sharding_demo.py`. Logs `g{k}_s{seed}.log`. First round (no-aug, 4-point curve):
`runs/2026-09-20-bn-group-sweep/RESULTS.md`.
