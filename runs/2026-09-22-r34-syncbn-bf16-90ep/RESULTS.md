# ResNet-34, 2018 recipe, bf16, **sync-BN** — verified path, 90 epochs

**✅ 74.168 % top-1 / 91.894 % top-5** (37,084 / 45,947 of 50,000), epoch 90/90, 95 % CI 73.78–74.55.
Landed 2026-09-23 14:55:16 UTC. **23.71 h**, `attempt 1`, **zero restarts, zero thermal rests**,
EDAC 0 corrected / 0 uncorrectable throughout (memory: the box's mapped-out DIMM).

`planning/global_bn_verified.md` §3.5 leg 1. Job `scripts/jobs/r34-default-bf16-4gpu.conf`, unit
`r34-verified`, launched 2026-09-22 15:12:48 UTC, 948 s/epoch average.

## What it retires

The committed pair number **74.064 / 91.754** (`runs/2026-09-16-r34-bf16-90ep/`) trained the
per-replica-BN render, a function the tree no longer emits. This run trains the committed sync-BN
artifact (`verified_mlir/resnet34in_momdp64bf16_train_step.mlir`, 219 `all_reduce`), so
`content.tex:5671`'s `BN statistic group … 64 (per replica)` row can go — ⛔ Brett's edit, not mine.

## The comparison, epoch for epoch

| | sync-BN (this run) | per-replica (09-16) | Δ |
|---|---|---|---|
| top-1 | **74.168** | 74.064 | **+0.104** |
| top-5 | **91.894** | 91.754 | **+0.140** |
| mean over e81–90 | — | — | +0.159 / +0.178 |

⇒ **A confirmation, not an overturn** — exactly what §3.5 predicted (converged deltas ≤ 0.10 in the
book's own epoch tables). The sign is positive and stable over the whole annealed tail (9 of the
last 10 epochs ≥ 0), but the size is inside one CI half-width, so this is "no detectable difference,
slight edge to sync-BN", not a measured gain.

⚠ **It is an UNPAIRED comparison.** Two separate obstacles, neither fixable after the fact here:
* the 09-16 run wrote **no bitmaps** (`LEAN_MLIR_DUMP_CORRECT` was not set), so there is no
  per-image record of the per-replica arm to McNemar against;
* `score-checkpoint` **refuses every BN net** (`VerifiedTrain.lean:2809`) — so the old checkpoint
  (kept as `.lake/build/resnet34in_momdp64bf16_ckpt_xla_perreplicaBN_e90.bin`) cannot be re-scored
  into one either. ⭐ **That blocker is now STALE**: its docstring's exit (a) — "append the stat
  floats to the checkpoint format" — effectively shipped as the `<ckpt>.bn` companion
  (`367bb28b`), and BOTH checkpoints have one. Teaching the tool to read `.bn` (refusing only when
  it is absent) would make this pair McNemar-able, and the run's own 74.168 is an exact
  known-answer gate for that change.

Against the book's JAX reference (74.16): **+0.008**, a dead heat. ⚠ That reference's own training
log is gone and the one on disk ends at 73.95 (memory: `r34-reference-curve-provenance`), and no
reference WEIGHTS exist on this box, so that comparison is unpaired too.

## Verified from the run's own output, not the conf

`momdp64bf16` · `cosine+warmup 5ep, baseLR 0.100000` · `DATA-PARALLEL: 4 replicas x bs 64 = global
batch 256, 5004 steps/epoch` · `running-stats BN: 36 layers, 17024 stat floats → eval via
@resnet34in_fwd_eval` · `EVAL SHARDED: 4 replicas x 256` · train step RESIDENT (330 tensors).

## Artifacts

* checkpoint `.lake/build/resnet34in_momdp64bf16_ckpt_xla.bin{,.bn,.epoch}` (epoch marker 90)
* per-image top-1 bitmaps, all 90 epochs: `bitmaps/r34_momdp64bf16_e{1..90}.bin` (50,000 B each) —
  the only per-image record this net can leave, and the input a later McNemar needs
* `attempt.log` / `full.log` / `master.log`, `log_ts.log` (per-line UTC), `epoch_clock.tsv`,
  `loader_rss.tsv`, `edac.tsv`
* `aborted-e1s2600-for-bench/` — a first launch (14:42:50) stopped at e1 step 2600 with no
  checkpoint written, to free the GPUs for the R50 sync-BN benchmark. Not an attempt of this run.

## Pace

908 s/epoch at e1, settling at ~945–980 from e38 (948 s mean). The 09-16 run held 876. Sync-BN's
compute accounts for ~20 s of that; the rest is the data feed at the conf's `SHIM_WORKERS=4`
(fed − synth = 26 ms/step, `runs/2026-09-22-syncbn-bench/`). Eval window 37 s per epoch (streamed,
2 producers). Nothing degraded over 24 h: all four producers stayed balanced at ~230 % CPU.

## Three arms, 15-epoch windows (`summarize.sh`)

| window | sync-BN | per-replica | reference | Δ (sync − per-replica) |
|---|---|---|---|---|
| e1–15 | 35.61 / 61.01 | 35.01 / 60.19 | 36.32 / 61.71 | +0.60 / +0.83 |
| e16–30 | 45.61 / 71.87 | 46.15 / 72.33 | 45.05 / 71.35 | −0.54 / −0.46 |
| e31–45 | 50.28 / 75.89 | 50.75 / 76.21 | 49.75 / 75.55 | −0.47 / −0.32 |
| e46–60 | 56.27 / 80.65 | 56.56 / 80.86 | 55.90 / 80.37 | −0.29 / −0.21 |
| e61–75 | 64.08 / 86.01 | 63.94 / 85.79 | 64.12 / 86.06 | +0.14 / +0.22 |
| **e76–90** | **72.36 / 90.95** | 72.32 / 90.82 | 72.28 / 90.94 | **+0.05 / +0.13** |

⇒ the sign flips window by window and the converged window is +0.05 — i.e. the BN statistic group
is NOT a variable this pair can resolve at 90 epochs, which is the finding §3.5 wanted. All three
arms sit inside 0.1 of each other over e76–90.

## Reproduce

    bash runs/2026-09-22-r34-syncbn-bf16-90ep/summarize.sh

Reads the reference curve and the per-replica arm in place under `runs/2026-09-16-r34-bf16-90ep/`;
nothing is copied.
