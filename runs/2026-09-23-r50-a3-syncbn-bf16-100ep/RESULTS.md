# ResNet-50, RSB-A3, bf16, **sync-BN 4 × 128** — verified path, 100 epochs

**✅ 78.330 % top-1 / 94.034 % top-5** (39,165 / 47,017 of 50,000), epoch 100/100, 95 % CI 77.97–78.69.
Landed 2026-09-24 14:47:47 UTC. **21.82 h**, `attempt 1`, **zero restarts, zero thermal rests**,
EDAC 0 corrected / 0 uncorrectable throughout (memory: the box's failing DIMM).

`planning/global_bn_verified.md` §3.5 leg 2. Job `scripts/jobs/r50-a3-wxclip4x128-bf16-4gpu.conf`,
unit `r50-a3`, launched 2026-09-23 16:58:31 UTC, 786 s/epoch average.

## What it retires

The committed verified A3 number **77.98 / 93.76** (`runs/r50-a3-bf16-100ep-verified.log`,
2026-09-01) trained 8 micro-steps × 4 replicas × 64 with **per-replica** BN — a Ghost-BN group of
64 against the JAX reference's 512. This run trains the committed sync-BN render
`verified_mlir/resnet50in160_lambaccdp4x128wxclipbcebf16_train_step.mlir` (321 `all_reduce`), where
BN normalises over all 4 replicas' micro-batches: **group 512 and k = 4, both matching the
reference**. The rows it retires in `blueprint/src/content.tex` — ⛔ Brett's edits, not made here:

* `:6348` — the verified A3 row, `64 (per replica)` · 77.98 / 93.76
* `:6384` — the pair table's `BN group … 512 & 64`
* `:6518` — A3 `shipped, and run … 77.98%`
* `:6540` — "Ghost-BN group 64 against the reference's 512 — an 8× gap, and **only half of it is
  reachable here**". That is now false: 4 × 128 reaches the whole group on this box.

## The comparison

| | sync-BN 4×128 (this run) | per-replica 8×64 (09-01) | JAX reference |
|---|---|---|---|
| top-1, e100 | **78.330** | 77.978 | 78.26 |
| top-5, e100 | **94.034** | 93.758 | 93.79 |
| top-1, mean e97–100 | **78.358** | 77.989 | 78.25 |
| top-5, mean e97–100 | **94.020** | 93.760 | — |
| top-1, mean e91–100 | **78.23** | 77.89 | 78.16 |

⇒ **vs JAX: +0.07 top-1 — a tie.** Inside one CI half-width (±0.36), and the e91–100 window agrees
(+0.07). The −0.28 the book carries for the verified A3 is gone.

⇒ **vs the old verified run: +0.35 top-1 / +0.28 top-5, and it is a consistent offset, not one
epoch.** Both tails are flat (this run 78.33–78.39 over e97–100; the old one 77.96–78.02), and the
offset holds at +0.34 over e91–100 and +0.35 over e81–100. It still sits at the edge of one CI
half-width, so "real but small" is the honest reading, not a measured gain.

⚠ **What is still confounded.** This run moved TWO variables at once relative to the old run: the
BN statistic group (64 → 512) AND the accumulation depth (k = 8 × 64 → 4 × 128; the effective
batch is 2048 either way, as is the 62,500-update cosine). So the +0.35 is "matching the reference's
accumulation shape", not "sync-BN" alone. Nothing left in the A3 comparison against JAX is known to
differ; the remaining difference is a +0.07 tie.

⚠ **Unpaired.** The 09-01 run wrote no per-image bitmaps, and `score-checkpoint` refuses every BN
net (`VerifiedTrain.lean`'s BN guard) although the `<ckpt>.bn` companion it asks for has shipped
(`367bb28b`). This run's bitmaps exist for all 100 epochs, so a McNemar against any future A3 arm
is possible; against the old one it is not.

## Verified from the run's own output, not the conf

`lambaccdp4x128wxclipbcebf16` · `LAMB (per-tensor trust ratio) + BCE-with-logits` ·
`cosine+warmup 5ep, baseLR 0.008000` · `DATA-PARALLEL: 4 replicas x bs 128 = global batch 512, 2500
steps/epoch` · ⭐ `new-batch weight 0.025996 = 1 − 0.900000^(1/4), compensated for grad-accum` —
k = 4, not a stale 1/8 (memory: `batch-spelled-twice`) · `running-stats BN: 53 layers, 53120 stat
floats → eval via @resnet50in160_fwd_eval` · `EVAL RES SPLIT: train d0 76800, eval d0 150528` ·
`EVAL SHARDED: 4 replicas x 256` · train step RESIDENT (644 tensors, 390.0 MB).

Epoch-1 sanity against the old run: step-0 loss 0.8327 (old 0.8354), e1 2.924 % (old 2.812).

## Pace

786 s/epoch mean (737–818, none over 1000 s); the bench (`runs/2026-09-22-syncbn-bench/`, and
`runs/2026-09-23-a3-worker-sweep/`) predicted 294–300 ms/step fed ⇒ 21.5–22 h, landed 21.82 h.
The old 8×64 run took ~22.8 h. Trainer RssAnon 2.3–3.9 GiB (streamed val).

⚠ **Loader memory crept, without ever costing pace.** Per-producer arena-class memory (the signal
that preceded EfficientNet's feed degradation) went 3.77 GiB at e1 → 4.40 at e30 (flat e20–30) →
4.56 at e60 → **6.89 at e99**, total RSS 5.33 → 8.09 GiB, accelerating after ~e70. Epoch times did
not suffer (e81–99 averaged 756 s against e3–20's 787), the fault watcher never fired, and the box kept
>100 GiB free. This conf runs its loaders bare (no respawn), so a longer A3-shaped schedule would
want `LEAN_MLIR_SHIM_RESPAWN_EPOCHS` before trusting the curve past ~100 epochs.

## Artifacts

* checkpoint `.lake/build/resnet50in160_lambaccdp4x128wxclipbcebf16_ckpt_xla.bin{,.bn,.epoch}` (marker 100)
* per-image top-1 bitmaps, all 100 epochs: `bitmaps/a3_lambaccdp4x128wxclipbcebf16_e{1..100}.bin`
  (50,000 B each) — LOCAL ONLY (`runs/**/*.bin` is gitignored)
* `attempt.log` / `full.log` / `master.log`, `log_ts.log` (per-line UTC), `epoch_clock.tsv`,
  `loader_rss.tsv`, `edac.tsv`, `WATCH_EVENT`, and the five watcher scripts
* `r50_a3_syncbn_verified_curve.csv` — written by `summarize.sh`

## Three arms, top-1, 20-epoch windows (`summarize.sh`)

| window | sync-BN 4×128 | per-replica 8×64 | JAX ref | Δ sync − old | Δ sync − ref |
|---|---|---|---|---|---|
| e1–20 | 36.53 | 37.17 | 29.80 | −0.64 | +6.73 |
| e21–40 | 55.46 | 55.02 | 46.39 | +0.44 | +9.07 |
| e41–60 | 63.74 | 64.02 | 57.88 | −0.28 | +5.85 |
| e61–80 | 73.23 | 72.93 | 70.40 | +0.29 | +2.83 |
| **e81–100** | **77.76** | 77.42 | 77.42 | **+0.35** | **+0.34** |
| **e91–100** | **78.23** | 77.89 | 78.16 | **+0.34** | **+0.07** |

⇒ The two verified arms trade sign until ~e60 and separate for good only once the cosine anneals —
the same shape R34's sync-BN pair showed, but here the converged gap is ~3× R34's +0.05–0.10 and
does not close. Both verified arms run far ahead of the reference early (the reference's per-epoch
curve is much noisier through e60); all three meet over the last ten epochs.

## Reproduce

    bash runs/2026-09-23-r50-a3-syncbn-bf16-100ep/summarize.sh

Reads the reference and the old verified curve in place from `blueprint/src/content.tex` (top-1),
and the old run's local log for its top-5 tail when present; nothing is copied.
