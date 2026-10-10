# ResNet-50, `2018` recipe, bf16 with an fp32 head — JAX reference path, 90 epochs

**✅ 76.796 % top-1 / 93.360 % top-5** (live weights, no EMA), epoch 90/90.
* Landed 2026-10-09 10:15:53 UTC.
* **25.84 h** of trainer time, on attempt 1.
* Zero restarts, zero thermal rests, EDAC 0 / 0 throughout.

This is the JAX half of the 2018 rerun the book's R50 table owes (`content.tex` :6884, "[TODO: rerun
both pairs with the head, the γ initialization and the BN decay matched (fp32, zero, 0.9) …]").
* Job: `scripts/jobs/r50-2018-jax-4gpu.conf`. Its precheck asserts the torchvision crop, no
  RandAugment, 90 epochs, lr 0.1, the fp32 head and BN decay 0.9.
* Unit `r502018-jax`. Launched 2026-10-08 08:25:32 UTC by `chain_after.sh`, 58 s after the A3
  rerun (`runs/2026-10-07-r50-a3-jax-bf16-100ep/`) completed.
* Fresh start: there was no checkpoint under `/home/skoonce/r50_2018_jax_2610/`.
* The verified A3 rerun (`runs/2026-10-09-r50-a3-verified-bf16-100ep/`) followed it 38 s later.

## The run, from its own output

* `lr=0.100000  batch_size=256 (4 devices x 64)  epochs=90  params=25557032`
* `steps_per_epoch=5004  total_steps=450360`
* SGD momentum 0.9, wd 1e-4, cosine with a 5-epoch warmup, label smoothing 0.1, 224 / 224, zero-γ,
  BN decay 0.9, bf16 matmuls with an fp32 head

Trainer: `jax/.lake/build/generated_resnet50_imagenet_2018.py`, byte-equal to the committed copy,
on `/home/skoonce/.venv-cuda` (jax 0.11.1, cuda13). Tree at `9d7def4d`.

## Curve (top-1 / top-5)

| epoch | 1 | 5 | 10 | 30 | 60 | 75 | 85 | 89 | **90** |
|---|---|---|---|---|---|---|---|---|---|
| top-1 | 4.61 | 37.17 | 49.99 | 54.37 | 66.63 | 72.87 | 76.49 | 76.78 | **76.80** |
| top-5 | 13.30 | 64.02 | 76.47 | 78.93 | 88.12 | 91.35 | 93.26 | 93.33 | **93.36** |

The tail is flat: e87–90 averages 76.778 / 93.323.

## Against the earlier 2018 numbers

| | e90 top-1 / top-5 |
|---|---|
| **JAX, this run** | **76.796 / 93.360** |
| JAX, landed August (book) | 76.95 / 93.44 |
| verified sync-BN (`runs/2026-09-24-r50-2018-syncbn-bf16-90ep`) | 77.160 / 93.430 |

* **Against the August reference: −0.15 / −0.08.** That is inside the ±0.37 95 % half-width, so it
  is noise at one seed. Unlike A3, the 2018 recipe held.
* What moved since August:
  * the fp32 head
  * BN decay 0.99 → 0.9
  * torchvision's RandomResizedCrop (`aa974a95`)

  The JAX side already started at zero-γ.
* **Against the landed verified 77.16: −0.36.** That verified run started at γ = 1. The TODO's
  comparison is this run against the verified 2018 rerun at zero-γ, queued as
  `runs/2026-10-10-r50-2018-verified-bf16-90ep/`.
* **The recipe delta on the JAX side is now +0.95** (A3 77.742 − 2018 76.796), against +1.31 in the
  book. Most of the shrink is A3's −0.52; see that run's RESULTS.md.

## Pace

* **1,033 s an epoch, eval included** (min 1,030, max 1,035; ~1,015 s train + ~18 s val).
* That is 25.84 h in all, against the conf's ares ETA of 22.5 h and the handoff's mars range of
  22.5–26 h: the top of the range, and the same as the 2026-08-24 mars sweep's 25.9 h.

## Artifacts

* Checkpoints, outside the repo in `/home/skoonce/r50_2018_jax_2610/`:
  * the weights for every epoch (`resnet50_imagenet_2018_e{1..90}.bin`)
  * the full train state for e88–90
  * the final `resnet50_imagenet_2018.state.npz`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `edac.tsv`, the two watcher
  scripts, `chain_after.sh` and its `CHAIN_EVENT`
* `r502018_jax_curve.csv`, written by `summarize.sh`

## Reproduce

    bash runs/2026-10-07-r50-2018-jax-bf16-90ep/summarize.sh
