# ResNet-50 RSB-A3, `rsb-faithful` recipe, bf16 with an fp32 head — JAX reference path, 100 epochs

**✅ 77.742 % top-1 / 93.650 % top-5** (live weights, no EMA), epoch 100/100.
* Landed 2026-10-08 08:24:34 UTC.
* **14.75 h** of trainer time, on attempt 1.
* Zero restarts, zero thermal rests, EDAC 0 / 0 throughout.

This is the JAX half of the A3 rerun the book's R50 table owes (`content.tex` :6884, "[TODO: rerun
both pairs with the head, the γ initialization and the BN decay matched (fp32, zero, 0.9), and A3
at timm's BCE target threshold of 0.2, which neither column had.]").
* Job: `scripts/jobs/r50-a3-jax-4gpu.conf`. Its precheck asserts:
  * torchvision's crop, and C6's bicubic RandAugment geometry
  * RandAugment m6 mstd0.5
  * the 0.2 BCE threshold
  * 512 × accum 4, lr 8e-3, 100 epochs
  * the fp32 head
* Unit `r50a3-jax`. Launched 2026-10-07 17:39:02 UTC by `chain_after.sh`, 55 s after the R34
  rerun (`runs/2026-10-07-r34-jax-bf16-90ep/`) completed.
* Fresh start: there was no checkpoint under `/home/skoonce/r50_a3_jax_2610/`.
* The R50-2018 JAX rerun followed it 58 s later.

## The run, from its own output

* `lr=0.008000  batch_size=2048 (4 devices x 512)  epochs=100  params=25557032`
* `grad-accum=4x  micro_batch=512 (4 devices x 128) -> effective_batch=2048`
* Train 160 / eval 224, LAMB, BCE with the 0.2 target threshold, Mixup 0.1 / CutMix 1.0, zero-γ, BN
  decay 0.9, bf16 matmuls with an fp32 head.

Trainer: `jax/.lake/build/generated_resnet50_imagenet_rsbfaithful.py`, byte-equal to the committed
copy, on `/home/skoonce/.venv-cuda` (jax 0.11.1, cuda13). Tree at `9d7def4d`.

## Curve (top-1 / top-5)

| epoch | 1 | 10 | 25 | 50 | 75 | 90 | 95 | 99 | **100** |
|---|---|---|---|---|---|---|---|---|---|
| top-1 | 1.73 | 35.82 | 49.10 | 57.68 | 73.34 | 77.25 | 77.68 | 77.64 | **77.74** |
| top-5 | 5.94 | 61.68 | 73.95 | 80.48 | 91.36 | 93.39 | 93.63 | 93.57 | **93.65** |

The tail is flat: e97–100 averages 77.721 / 93.637, and e91–100 averages 77.57 top-1.

## Against the earlier A3 numbers

| | e100 top-1 / top-5 | mean e97–100 top-1 | mean e91–100 top-1 |
|---|---|---|---|
| **JAX, this run** | **77.742 / 93.650** | **77.72** | **77.57** |
| JAX, landed August (book) | 78.26 / 93.79 | 78.25 | 78.16 |
| verified sync-BN 4×128 (`runs/2026-09-23-r50-a3-syncbn-bf16-100ep`) | 78.330 / 94.034 | 78.36 | 78.23 |
| timm RSB A3 (paper) | 78.1 | | |

⚠ **The rerun lands 0.52 below the August reference.** The gap holds as an offset: −0.53 over
e97–100 and −0.59 over e91–100. That is past the ±0.36 single-run 95 % half-width, so it should not
be read as noise without a second look.

The run moved several things at once relative to the August run, so this run alone cannot assign
the drop:
1. **BCE target threshold 0.2 (D14, `de7f7857`).** This is timm's A3 as it actually ran. It changes
   the loss on every Mixup/CutMix target. Epoch 1 reads 1.73 % here, against 2.81–2.92 % on the runs
   without it.
2. **torchvision RandomResizedCrop, and A3 on C6's bicubic RandAugment geometry (`aa974a95`).**
3. **The fp32 classifier head (`559a05b3`), and BN decay 0.99 → 0.9** (August's 0.99 per the
   conf's header). The head alone moved R34 by −0.02 (`runs/2026-10-07-r34-jax-bf16-90ep`).

**What it means for the book's pair.** The verified 78.33 also trained without the threshold, and
with γ = 1. The TODO's comparison is this run against a verified rerun on the same settings (phase 2
of `planning/next_session_resnet_jax_reruns.md`), not against 78.33. Until that lands:
* the old pair (78.26 vs 78.33) stands as the matched-at-the-time comparison;
* this run is −0.59 against the old verified run, on different settings.

Possible follow-ups (not done):
* An ablation of the threshold alone, about 15 h on mars.
* Pairing this run per image against the verified rerun once it exists. The e98–100 train states
  are kept.

## Pace

* **531 s an epoch, eval included** (min 529, max 534; ~513 s train + ~17 s val).
* That is 14.75 h in all, against the conf's ares ETA of 13.5 h and the handoff's mars estimate of
  13.5–15 h.

## Artifacts

* Checkpoints, outside the repo in `/home/skoonce/r50_a3_jax_2610/`:
  * the weights for every epoch (`resnet50_imagenet_rsbfaithful_e{1..100}.bin`)
  * the full train state for e98–100
  * the final `resnet50_imagenet_rsbfaithful.state.npz`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `edac.tsv`, the two watcher
  scripts, `chain_after.sh` and its `CHAIN_EVENT`
* `r50a3_jax_curve.csv`, written by `summarize.sh`

## Reproduce

    bash runs/2026-10-07-r50-a3-jax-bf16-100ep/summarize.sh
