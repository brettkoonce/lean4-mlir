# ResNet-50, RSB-A3 as timm ran it, bf16 — verified path, 100 epochs

**✅ 78.08 % top-1 / 93.84 % top-5** (39,039 / 50,000, 95 % CI 77.71–78.44), epoch 100/100.
* Landed 2026-10-14 04:33:22 UTC.
* **21.9 h** wall clock, on attempt 1.
* Zero restarts, zero thermal rests.

The verified half of the timm-schedule A3 pair (`4d612bb9`, "A3/A2/A1 as timm ran them"). The JAX half
is `runs/2026-10-12-r50-a3-jax/` (77.60 / 93.61), run first on this box.
* Job `scripts/jobs/r50-a3-wxclip4x128-bf16-4gpu.conf` (`CKPT_TAG=2611`), launched 2026-10-13
  06:39:19 UTC on ares (4× 4060 Ti) via `launch.sh`.
* Render `lambaccdp4x128wxclipbcebf16` (4 × 128 per step, accumulated to the A3 global 2048), XLA
  through PJRT, parameters resident.
* Tree at `9861f21c`, unchanged for the whole run.
* `aborted_10-12/` holds a first launch stopped at step 1 on 10-12 so the JAX half could run first;
  no checkpoint was written.
* Per-epoch top-1 bitmaps went to `bitmaps/` (`*.bin`, not committed).

~306 ms/step, 2500 steps/epoch, ~13 min an epoch including eval and checkpoint.

## Curve (top-1 / top-5)

| epoch | 10 | 25 | 50 | 75 | 90 | 95 | **100** |
|---|---|---|---|---|---|---|---|
| top-1 | 34.18 | 52.36 | 61.40 | 74.59 | 77.56 | 77.99 | **78.08** |
| top-5 | 60.47 | 76.47 | 83.32 | 92.04 | 93.66 | 93.80 | **93.84** |

Epochs 91–100 average **77.91** top-1.

## Against the JAX half

| | top-1 e100 | top-5 e100 | top-1 e91–100 mean |
|---|---|---|---|
| verified (this run) | **78.08** | **93.84** | 77.91 |
| JAX (`2026-10-12-r50-a3-jax`) | 77.60 | 93.61 | 77.38 |
| Δ | **+0.48** | +0.23 | +0.53 |

One run's 95 % half-width is ±0.37, so the verified run finished ahead at this seed. The book's pair
was 77.74 JAX / 77.91 verified (+0.17). McNemar on the paired predictions is owed: this run wrote
bitmaps, the JAX trainer does not.

## Against the other A3 numbers (added 2026-10-14, with the ch 5 update)

| | e100 top-1 / top-5 | mean e97–100 top-1 | mean e91–100 top-1 |
|---|---|---|---|
| **verified, this run (timm's schedule)** | **78.078 / 93.844** | **78.04** | **77.91** |
| JAX, same settings (`runs/2026-10-12-r50-a3-jax`) | 77.602 / 93.610 | 77.46 | 77.38 |
| verified 10-09 (per-step cosine from zero, alternating Mixup/CutMix, grey fill) | 77.914 / 93.712 | 77.90 | 77.83 |
| JAX 10-07, same settings as the 10-09 run, f32 rescore | 77.75 / 93.65 | 77.72 | 77.57 |
| verified 09-23 (γ = 1, no threshold, the old schedule) | 78.330 / 94.034 | 78.36 | 78.23 |
| timm RSB A3 (paper) / `resnet50.a3_in1k` under this protocol | 78.1 / 78.05 | | |

* **+0.48 is the first ResNet pair outside the single-run interval** (R34 +0.03, 2018 +0.33, A3 10-07/09
  +0.17, all verified ahead). One seed per arm; the McNemar pairing reads these two checkpoints, and seeds
  per arm would read the paths.
* The three settings moved this path +0.17 (77.91 → 78.08) and the reference −0.15. This run lands 0.02
  under the paper's 78.1 and 0.03 over `timm`'s own weights' 78.05. Against the 09-23 run it is −0.25,
  where the 10-09 run (threshold + zero-γ on the old schedule) was −0.42: timm's schedule gave some back.
* The log's own lines for the three: `▸ LR SCHEDULE: per epoch (timm CosineLRScheduler, t_in_epochs) —
  warmup 0.000100→0.008000 over 5 epochs, then cosine over all 100 epochs to the floor 0.000001`;
  `Epoch 1/100: lr=0.000100` … `Epoch 100/100: lr=0.000003`, the JAX log's rates epoch for epoch; the
  shim is the 4d612bb9 `generated_resnet50_imagenet_short_shim.py` (per-batch draw, `_RA_FILL` = img_mean).
  Step-0 loss 0.720 (the 10-09 run's 0.724); epoch 1 reads 0.102 % (51 images, the JAX half's 51 too),
  against 1.728 % on the 10-09 run — the whole first epoch at 1e-4.
* Top-1 by 20-epoch window, verified minus JAX: +0.58, +3.06, +1.45, +1.42, +0.50 (the 10-09 pair:
  +1.92, +0.24, +1.16, +1.05, +0.29). The lead through the middle has the same shape; the anneal closed
  it to +0.26 over the last ten epochs then, and to +0.53 now.
* The recipe delta, A3 − 2018: verified 78.078 − 77.090 = **+0.99**; JAX 77.602 − 76.796 = **+0.84**.
* Pace: 785 s an epoch at the median over the epochs the log's timestamps resolve (min 750, max 915;
  788 s from the supervisor's wall clock); the 10-09 run on the 3060 box took 20.94 h at 754 s.
* The bitmap file names carry the variant twice (`a3_lambaccdp4x128wxclipbcebf16_lambaccdp4x128wxclipbcebf16_e{N}.bin`):
  the driver appends `_{variant}_e{N}` to the prefix itself (`LeanMlir/Verified/Train.lean`, the
  `writeBinFile` at the bitmap dump) and the conf's prefix already carried it. The conf now passes
  `$RUNDIR/bitmaps/a3`, which gives the 10-09 run's `a3_lambaccdp4x128wxclipbcebf16_e{N}.bin`.
* **The pairing (2026-10-14):** `../2026-10-12-r50-a3-jax/rescore.sh` (the JAX checkpoint through f32 on
  atlas's CPU, 38,803 set) against this run's e100 bitmap (39,039): both right 36,902, both wrong 9,060,
  only JAX 1,901, only verified 2,137 — 4,038 discordant (8.1 %), net +236, McNemar exact p = **0.0002**.
  These two checkpoints are separable; the 10-07/09 pair's were not (p = 0.20).

## The eval resize at timm's integers — `rescore_f6.sh`, 2026-10-20

F6 (`planning/imagenet_parity.md` §9.3), the JAX half's `rescore_f6.sh` section has the protocol: the
short side at `floor(224 / 0.95) = 235`, not 236, and the long side truncated, not rounded.
`rescore_f6.sh` ran on ares (4× 4060 Ti, ~78 s a mode) over the 2611 checkpoint:

| mode | protocol | top-1 | top-5 |
|---|---|---|---|
| `pre` | 236, rounded long side (the control; in-training 39,039) | 78.076 (39,038) / **78.08** | 93.846 / **93.85** |
| `f6`  | 235, truncated long side | 78.050 (39,025) / **78.05** | 93.966 / **93.97** |

−13 images on top-1 (the JAX half moved the same −13), +60 on top-5. **The pairing at F6** —
`scripts/demos/mcnemar.py` over `../2026-10-12-r50-a3-jax/bitmaps/r50-a3_jax_f32_f6.bin` (the headline's
own CPU rescore, 38,790) and this run's `bitmaps/a3_f6_f6.bin` (39,025; copied from ares): both right
36,899, both wrong 9,084, only JAX 1,891, only verified 2,126 — 4,017 discordant (8.0 %), net +235,
McNemar exact p = **0.0002** (pre-F6: 4,038 / 1,901 / 2,137 / +236, p = 0.0002). The ares session's
first pairing used the JAX checkpoint rescored on ares's own GPUs instead (38,794, a 4-image card gap):
1,889 / 2,120, p = 0.00028 — the same answer. Logs: `rescore_f6_pre.log` / `rescore_f6_f6.log`. Both
bitmaps are `runs/**/*.bin`, git-ignored; they live on atlas (and the verified one on ares).
**The book's A3 pair is 78.14 / 93.70 (reference, `runs/2026-11-06-r50-a3mix8-jax`) and 77.99 / 93.69 (verified, `runs/2026-11-06-r50-a3-ver-mix256`)**, Mixup or CutMix drawn per 256-row chunk on both paths, scored at 235.
