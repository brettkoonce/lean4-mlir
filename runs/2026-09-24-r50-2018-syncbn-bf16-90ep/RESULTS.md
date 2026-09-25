# ResNet-50, 2018 recipe, bf16, **sync-BN** — verified path, 90 epochs

**✅ 77.160 % top-1 / 93.430 % top-5** (38,580 / 46,715 of 50,000), epoch 90/90, 95 % CI 76.79–77.53.
Landed 2026-09-25 22:54:28 UTC. **31.60 h**, `attempt 1`, **zero restarts, zero thermal rests**,
EDAC 0 corrected / 0 uncorrectable throughout (memory: the box's failing DIMM).

`planning/global_bn_verified.md` §3.5, the R50 2018 leg. Job `scripts/jobs/r50-2018-bf16-4gpu.conf`,
unit `r50-2018`, launched 2026-09-24 15:18:29 UTC from `28fc373f` (pulled, exe rebuilt, shims
78/78 in sync), 1,264 s/epoch average.

## What it retires

The committed verified 2018 number **77.07 / 93.48** (2026-08-27, 77.074 / 93.476) trained the
per-replica-BN render — the same variant name, `momdp64bf16`, before `260df19d` re-rendered it as
sync-BN. This run trains the committed sync-BN artifact
`verified_mlir/resnet50in_momdp64bf16_train_step.mlir` (321 `all_reduce`, against the per-replica
render's 162: the 159 extra are the BN statistic reductions), so the BN group is 4 × 64 = **256,
the reference's own** (the 2018 recipe has no accumulation, so against the per-replica run ONE variable moves). The rows
it retires in `blueprint/src/content.tex` at `fbbd03fa` — ⛔ Brett's edits, not made here:

* `:6347` — the verified 2018 row, `64 (per replica)` · ~20.5 min · ~30.7 hr · 77.07 / 93.48
* `:6381`–`:6382` — the pair table's verified 2018 cells, 77.07 / 93.48, Δ +0.12 / +0.04
* `:6384` — `BN group … 256 & 64`
* `:6386` — `wall clock … 30.7 h` (this run: 31.6 h; its A3 cell, 22.8 h, becomes 21.8 h with the pair)

## ⭐ The pair: 2018 | A3, both at the reference's BN group

This run's partner is the A3 sync-BN run, `runs/2026-09-23-r50-a3-syncbn-bf16-100ep/` (78.330 /
94.034, pushed `3af1cc9c`) — together they are the two cells of `content.tex:6376`'s
2018 | A3 table, and for the first time **both verified cells train the reference's own BN group**.

| | 2018 ref | **2018 verified** | Δ | A3 ref | **A3 verified** | Δ |
|---|---|---|---|---|---|---|
| top-1 | 76.95 | **77.160** | +0.21 | 78.26 | **78.330** | +0.07 |
| top-5 | 93.44 | **93.430** | −0.01 | 93.79 | **94.034** | +0.24 |
| precision | bf16 | bf16 | | bf16 | bf16 | |
| BN group | 256 | **256** (was 64) | | 512 | **512** (was 64) | |
| box | 4060 Ti | 3060 | | 4060 Ti | 3060 | |
| wall clock | 24.0 h | 31.6 h | | 15.1 h | 21.8 h | |

⇒ **Both recipes tie their JAX references**, each inside one CI half-width (±0.37 / ±0.36). The
book's A3 −0.28 is gone; 2018's +0.12 becomes +0.21. The recipe gap is preserved: A3 − 2018 is
**+1.17** verified against **+1.31** in the reference.

Per-epoch top-1 at the book's A3-table epochs (`summarize.sh`):

| epoch | 5 | 25 | 50 | 75 | 90 | 100 |
|---|---|---|---|---|---|---|
| 2018 verified | 40.72 | 54.24 | 62.82 | 73.00 | **77.16** | — |
| A3 verified | 21.80 | 56.29 | 62.18 | 74.67 | 77.89 | **78.33** |

## Against the per-replica 2018 run (secondary)

| | sync-BN (this run) | per-replica (08-27) |
|---|---|---|
| top-1, e90 | **77.160** | 77.074 |
| top-5, e90 | **93.430** | 93.476 |

⇒ **A dead heat**: +0.09 top-1 and −0.05 top-5, opposite signs, both a fraction of one CI
half-width. The same finding as R34's pair (`runs/2026-09-22-r34-syncbn-bf16-90ep/`, +0.10): at
global batch 256 the BN statistic group (64 vs 256) is not a variable 90 epochs resolve. Unlike A3's
change from its old run (+0.35, with the accumulation depth k moving too), this pair moves only
the BN group.

⚠ **Unpaired, and the old arm is an endpoint only.** The 08-27 run wrote no bitmaps, and its log
lived in `/tmp` and is gone; the book carries only its endpoint. Its checkpoint survives, renamed out
of this run's slot to `.lake/build/resnet50in_momdp64bf16_PERREPLICA_2026-08-27_e90_ckpt_xla.bin`.
The sync-BN render reused its variant name, so the slot read `90`, which would have made this job
exit as "complete" or resume per-replica weights. That checkpoint has **no `.bn` companion** (it
predates `367bb28b`), so even a `score-checkpoint` that read `.bn` could not re-score it into a
bitmap. This run's 90 bitmaps are local, as are A3's 100.

## Verified from the run's own output, not the conf

`momdp64bf16` · `2018 recipe` · `cosine+warmup 5ep, baseLR 0.100000` · `DATA-PARALLEL: 4 replicas
x bs 64 = global batch 256, 5004 steps/epoch` · `new-batch weight 0.100000` (no accumulation, so no
compensation — correct) · `running-stats BN: 53 layers, 53120 stat floats → eval via
@resnet50in_fwd_eval` · `TRAIN RES: 224×224 … generated_resnet50_imagenet_2018_shim.py` ·
`EVAL SHARDED: 4 replicas x 256` · train step RESIDENT (483 tensors, 292.5 MB). Step-0 loss 7.911;
e1 10.66 / 26.64.

## Pace

1,264 s/epoch mean (1,249–1,303, none slow), 31.60 h against the bench's ~30.7 h and the
per-replica run's 30.72 h. Trainer RssAnon 1.9–3.8 GiB (streamed val).

⚠ **Loader memory crept again, the same shape as A3, without costing pace.** Per-producer
arena-class memory: 3.36 GiB at e1 → 3.85 at e40 (near-flat) → 4.45 at e60 → **6.67 at e89**, total
RSS 4.98 → 7.82 GiB, accelerating after ~e50. Epochs did not slow and the fault watcher never
fired. Two runs of two different R50 shims now show it, so it is the shims' (or tf.data's), not a
per-recipe accident. Harmless at 90–100 epochs on this box; a longer schedule should respawn.

## Artifacts

* checkpoint `.lake/build/resnet50in_momdp64bf16_ckpt_xla.bin{,.bn,.epoch}` (marker 90)
* per-image top-1 bitmaps, all 90 epochs: `bitmaps/r50-2018_momdp64bf16_e{1..90}.bin`
  (50,000 B each) — LOCAL ONLY (`runs/**/*.bin` is gitignored)
* `attempt.log` / `full.log` / `master.log`, `log_ts.log` (per-line UTC), `epoch_clock.tsv`,
  `loader_rss.tsv`, `edac.tsv`, `WATCH_EVENT`, and the five watcher scripts
* `r50_2018_syncbn_verified_curve.csv` — written by `summarize.sh`
* `aborted-e1s1700-for-pull/` — a first launch (2026-09-24 15:08:57) on the pre-pull tree, stopped
  at e1 step ~1700 with no checkpoint written so the run would train from latest main (same render,
  same shim, rebuilt exe). Not an attempt of this run.

## Against the reference, top-1, 15-epoch windows (`summarize.sh`)

| window | sync-BN | JAX ref | Δ |
|---|---|---|---|
| e1–15 | 42.25 | 35.02 | +7.23 |
| e16–30 | 54.50 | 47.56 | +6.94 |
| e31–45 | 58.96 | 52.08 | +6.88 |
| e46–60 | 63.94 | 58.56 | +5.37 |
| e61–75 | 70.09 | 67.05 | +3.04 |
| **e76–90** | **75.87** | **75.20** | **+0.67** |
| **e81–90** | **76.60** | **76.27** | **+0.33** |

⇒ The same convergence A3 shows: the verified path runs ~7 points ahead through the first half and
the gap closes in the anneal, to +0.21 at e90.

## Reproduce

    bash runs/2026-09-24-r50-2018-syncbn-bf16-90ep/summarize.sh

Reads the JAX curve in place from `blueprint/src/content.tex` (top-1) and the A3 partner's curve in
place from `runs/2026-09-23-r50-a3-syncbn-bf16-100ep/` (run its `summarize.sh` first); nothing is copied.
