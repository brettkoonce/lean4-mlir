# ViT-S (DeiT-S) / ImageNet-1k, VERIFIED path, 300 epochs — mars (4× RTX 3060), 2026-11-10 → 11-14

**✅ 79.95 % top-1 / 94.98 % top-5** (EMA, 39,976 of 50,000), against its JAX reference's 79.95 / 94.96
(39,977) and DeiT-S's published 79.8 / 95.0. One image apart; McNemar p = 1.

* conf `scripts/jobs/vits-default-emabf16-4gpu.conf` (4 × 256 one-shot, effective 1024), render
  `vits_emadp256x4wxdropeps0000001smxerfbf16`, exe `vit-s-imagenet-verified`
* launched with `--setenv=LEAN_MLIR_MEM_FRACTION=0.95` (the render peaks at 10.75 GiB; the JAX
  variable does not reach the verified path)
* one attempt, no faults, no NaN: 14:12:25 UTC 11-10 → ✅ 05:08:55 UTC 11-14 (86.9 h, ~17.4 min an
  epoch with the streamed eval every epoch). The JAX peer took ~72 h on the same box.

## Curve (EMA, in-training eval, all 50,000 val images)

| epoch | 50 | 100 | 150 | 200 | 250 | 290 | 295 | 300 |
|---|---|---|---|---|---|---|---|---|
| verified | 47.34 | 70.77 | 75.07 | 77.39 | 79.17 | 79.88 | 79.90 | 79.95 |
| JAX | 46.89 | 70.66 | 74.88 | 77.53 | 79.18 | 79.85 | 79.90 | 79.95 |

## Paired tests (`scripts/demos/mcnemar.py`, the e300 bitmap against the JAX e300 re-scores)

| against | discordant | McNemar exact p |
|---|---|---|
| JAX EMA (79.95) | 3,543 (1,771 / 1,772) | 1 |
| JAX live (79.92) | 3,621 (1,819 / 1,802) | 0.79 |

`mcnemar_vs_jax_ema.txt`, `mcnemar_vs_jax_live.txt`. The e300 bitmap reproduces the log's count exactly
(39,976). JAX bitmaps: `../2026-11-05-vits-jax-300ep/bitmaps/`.

## Reading

* A dead heat: the two paths' curves track within ~0.2 throughout and finish one image apart.
* With ViT-Ti (72.35 verified / 72.31 JAX vs DeiT-Ti 72.2), both DeiT widths now land on their papers on
  both paths.
