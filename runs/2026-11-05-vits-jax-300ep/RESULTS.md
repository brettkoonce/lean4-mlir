# ViT-S (DeiT-S) / ImageNet-1k, JAX path, 300 epochs — mars (4× RTX 3060), 2026-11-05 → 11-10

**✅ 79.95 % top-1 / 94.96 % top-5** (EMA), against DeiT-S's published 79.8 / 95.0.

* conf `scripts/jobs/vits-accum-jax-4gpu.conf` (2 × 512 accumulation = DeiT's 1024), trainer
  `jax/generated/generated_vit_s_imagenet_accum.py`; lineage `/home/skoonce/vit_s_jax_2611/`
* launched with `--setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`: at the default 0.75 (~8.9 GiB of a 12 GB
  3060) the fresh start OOM'd on a 7.35 GiB block. The conf's "fits default" was measured on ares' 16 GB cards.
* three launches, no faults: e1–20 (11-05 20:34 → 11-06 01:21), paused for the R50 A3 per-chunk run;
  e21–23 (11-06 16:40 → 17:24), paused for the verified A3 per-256 run; e24–300 (11-07 14:22 → 11-10
  09:09). Each resume restored the full train state from the newest `.state.npz`. ~72 h of training,
  ~14.7 min an epoch (863 s train + 18 s eval on every fifth).
* Mixup/CutMix drawn once per 1024-row step, as its verified peer (`vits-default-emabf16-4gpu`) does
  per record; ViT-Ti reached its paper that way, so the per-chunk change was not applied here.

## Curve (EMA, in-training eval, 224 / crop 0.875)

| epoch | 50 | 100 | 150 | 200 | 250 | 290 | 295 | 300 |
|---|---|---|---|---|---|---|---|---|
| top-1 | 46.89 | 70.66 | 74.88 | 77.53 | 79.18 | 79.85 | 79.90 | 79.95 |

## Re-score, all 50,000 val images (`jax/scripts/eval_full50k.py`, f32, PROTOCOL=train, BATCH=200)

| weights | top-1 | top-5 | bitmap |
|---|---|---|---|
| EMA | **79.95** (39,977 — the trainer's own count) | 94.96 | `bitmaps/vits_jax_e300_f32_ema.bin` |
| live | 79.92 (39,959) | 94.90 | `bitmaps/vits_jax_e300_f32_live.bin` |

`rescore_f32_ema.log`, `rescore_f32_live.log`. EMA vs live: 950 discordant, McNemar p = 0.58
(`mcnemar_ema_vs_live.txt`) — EMA is free here, as on ViT-Ti.

## Reading

* +0.15 over DeiT-S's 79.8 (top-5 −0.04), as ViT-Ti's 72.31 is +0.11 over DeiT-Ti's 72.2: both DeiT
  widths land on their papers on this pipeline.
* Its verified peer started 2026-11-10 (`../2026-11-10-vits-verified-300ep/`); the pairing is that run's
  e300 bitmap against `bitmaps/vits_jax_e300_f32_ema.bin`.
