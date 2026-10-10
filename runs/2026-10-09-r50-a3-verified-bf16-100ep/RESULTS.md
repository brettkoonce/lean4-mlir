# ResNet-50 RSB-A3, bf16 sync-BN 4 × 128, fp32 head, zero-γ, BN 0.9, BCE threshold 0.2 — verified path, 100 epochs

**✅ 77.914 % top-1 / 93.712 % top-5** (live weights, no EMA), epoch 100/100.
* Landed 2026-10-10 07:13:06 UTC.
* **20.94 h** wall clock (754 s an epoch), on attempt 1.
* Zero restarts, zero thermal rests, no `WATCH_EVENT`, EDAC 0 / 0 throughout.

This is the verified half of the A3 rerun the book's R50 table owes (`content.tex` :6884, "[TODO: rerun
both pairs with the head, the γ initialization and the BN decay matched (fp32, zero, 0.9), and A3 at
timm's BCE target threshold of 0.2, which neither column had.]"). Its JAX partner on the same settings
is `runs/2026-10-07-r50-a3-jax-bf16-100ep/` (77.742 / 93.650).
* Job: `scripts/jobs/r50-a3-wxclip4x128-bf16-4gpu.conf`, `CKPT_TAG=2610`. The checkpoint is
  `.lake/build/resnet50in160_lambaccdp4x128wxclipbcebf16_ckpt_xla_2610.bin`, so the landed 09-23 run's
  checkpoint is untouched.
* Unit `r50a3-ver`. Launched 2026-10-09 10:16:31 UTC by `chain_after.sh`, 38 s after the 2018 JAX
  rerun completed. The verified 2018 rerun (`runs/2026-10-10-r50-2018-verified-bf16-90ep/`) followed it
  at 07:13:43 UTC.
* Tree at `9d7def4d` + the pull (`pc_exe` rebuilt `resnet50-imagenet-verified` at 2026-10-08 14:05
  UTC, after zero-γ landed).

## The run, from its own output

* `attempt 1: resuming at epoch 0/100`: a fresh start.
* `ResNet-50 (ImageNet-1k, 160² train) lambaccdp4x128wxclipbcebf16 (cosine+warmup 5ep, baseLR 0.008000)`,
  4 replicas × 128, 2500 steps an epoch (accum 4 → effective 2048).
* `decay 0.900000 …, new-batch weight 0.025996 = 1 − 0.900000^(1/4), compensated for grad-accum`
* `▸ INIT: zero-γ on 16 residual-closing BNs (the JAX reference's zero_init_last)`
* The BCE threshold shows in the numbers rather than a log line:
  * step-0 loss 0.724, against 0.833 on the landed run (whose log also has no zero-γ line);
  * epoch 1 reads **1.728 %**, against 1.732 % for the JAX rerun (which has the threshold) and 2.924 %
    for the landed verified run (which has not).
* 8 shim producers, one replaced every 10 epochs (`LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10`): 9 respawns.

## Curve (top-1 / top-5)

| epoch | 1 | 10 | 25 | 50 | 75 | 90 | 95 | 99 | **100** |
|---|---|---|---|---|---|---|---|---|---|
| top-1 | 1.73 | 38.56 | 47.25 | 63.28 | 74.61 | 77.52 | 77.85 | 77.89 | **77.91** |
| top-5 | 5.92 | 65.17 | 71.65 | 85.00 | 92.02 | 93.58 | 93.66 | 93.72 | **93.71** |

The tail is flat: e97–100 averages 77.901 / 93.699, and e91–100 averages 77.83 top-1.

## Against the other A3 numbers

| | e100 top-1 / top-5 | mean e97–100 top-1 | mean e91–100 top-1 |
|---|---|---|---|
| **verified, this run** | **77.914 / 93.712** | **77.90** | **77.83** |
| JAX rerun, same settings (`runs/2026-10-07-r50-a3-jax-bf16-100ep`) | 77.742 / 93.650 | 77.72 | 77.57 |
| verified landed 09-23 (γ = 1, no threshold; BN 0.9 already) | 78.330 / 94.034 | 78.36 | 78.23 |
| JAX landed August (book) | 78.26 / 93.79 | 78.25 | 78.16 |

* **The pair on matched settings: verified +0.17 / +0.06 over JAX** at e100, +0.26 over e91–100. That
  is inside the ±0.36 single-run 95 % half-width: the paths tie, as they did in August (+0.07).
* **The rerun's drop is shared by both paths.** Verified falls 0.42 from its landed run, and JAX falls
  0.52. So the JAX rerun's −0.52 (flagged in its RESULTS.md) is not a JAX-side problem. It is the
  settings that changed. On the verified path only two moved: the 0.2 threshold and zero-γ (the
  landed run already had BN 0.9 with the same 0.025996 compensation). JAX moved those plus the fp32
  head, BN 0.99 → 0.9, the torchvision crop and C6 geometry. Threshold or zero-γ, these runs can't
  separate. The book's
  A3 number moves down about 0.4–0.5 either way.
* Top-1 by 20-epoch window, verified minus JAX: +1.92, +0.24, +1.16, +1.05, +0.29. Verified leads
  through the middle of training and the two converge by the end.
* Per image: this run's `bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin` (local only) can be paired by
  McNemar with a re-score of the JAX rerun's final `.state.npz` (template
  `runs/2026-10-01-vit-jax-bf16-300ep/score_arms.py`). Not done.

## Pace and loader memory

* **754 s an epoch** over the run (min 716, max 784). By 20-epoch window: 733, 753, 759, 756 and
  761 s. It crept about 3 % over the first 40 epochs, then held.
* The conf's ETA said ~22 h; this came in at 20.94 h.
* Loader arena per producer: about 3.4 GiB at e1, levelling at about 6 GiB mean (max 8.4–9.2 GiB)
  from e30 on. Respawning one producer every 10 epochs held it there. No fault fired.
* MemAvailable never fell below 91.9 GB, swap ≤ 8.0 GB.

## Artifacts

* Checkpoint, outside git: `.lake/build/resnet50in160_lambaccdp4x128wxclipbcebf16_ckpt_xla_2610.bin`
  (+ `.bn`, `.epoch`).
* Per-image top-1 bitmaps for every epoch, `bitmaps/` (local only; `runs/**/*.bin` is gitignored).
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `loader_rss.tsv`, `edac.tsv`,
  `log_ts.log`, the five watcher scripts, `chain_after.sh` and its `CHAIN_EVENT`.
* `r50a3_verified_curve.csv`, written by `summarize.sh`.

## Reproduce

    bash runs/2026-10-09-r50-a3-verified-bf16-100ep/summarize.sh
