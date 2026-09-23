# Sync-BN step-time benchmark, 4× RTX 3060 box — 2026-09-22

`scripts/bf16_probe_3060.sh`, bf16 only, fed + synth arms, clock over steps 200→600, resident,
`CKPT_TAG=probe3060sbn`. Rows overridden (`rows.txt`) to match the **job confs**, not the script's
built-in table: the built-in `r50a3` row is the old `8x64`/G2 5000 shape and its `r50` row has no
`LEAN_MLIR_RECIPE=2018` (wrong shim). Env verified in `/proc/<pid>/environ` on the first arm.
`resnet50-imagenet-verified` was rebuilt first — the exe on disk was from 09-02, older than the
sync-BN renders (`412bbe4a`, `13e4d876`) and without streaming val.

| net (conf) | variant | workers | fed med | synth med | fed − synth | p90 fed | steps/ep |
|---|---|---|---|---|---|---|---|
| R50 2018 (`r50-2018-bf16-4gpu`) | `momdp64bf16` | 8 | **238** | 235 | 3 | 239 | 5004 |
| R50 A3 (`r50-a3-wxclip4x128-bf16-4gpu`) | `lambaccdp4x128wxclipbcebf16` | 8 | **300** | 244 | **56** | 313 | 2500 |
| R34 2018 (`r34-default-bf16-4gpu`) | `momdp64bf16` | 4 | **174** | 148 | **26** | 184 | 5004 |

## Sync-BN cost (synth = compute only), against the per-replica-BN renders on this box

| net | before (`runs/probe3060.tsv`) | sync-BN | Δ |
|---|---|---|---|
| R34 bf16 | 144 | 148 | +4 ms (+2.8 %) |
| R50 bf16 | 227 | 235 | +8 ms (+3.5 %) |
| A3 | 8x64: 138/256 img = 276 per 512 | 4x128: 244 per 512 | shape changed — not a sync-BN Δ; the 4x128 step is 12 % cheaper per image |

## Val producers during these probes: NONE

`spawnValStream` (2 producers, hardcoded `n := 2`) runs only inside the per-epoch eval block
(`VerifiedTrain.lean:2661`), fresh per pass and reaped after. The probe returns at step 600
(`:2598`), mid-epoch 1, before any eval. So the train numbers above carry no val-producer load —
and the eval window is NOT measured here. See the R34 run's epoch-1 eval for the first number.

## Read

* **R50 2018 is compute-bound** (3 ms wait at 8 workers). Sync-BN costs +8 ms/step.
* **A3 at 4x128 is shim-bound**: 56 ms/step (23 %) of data wait at 8 workers ⇒ the shim tops out
  near 512/0.300 ≈ 1,700 img/s while compute could take ≈ 2,100. Worth ~3.9 h over 100 epochs.
  The 8x64 worker sweep (`probe3060_a3.tsv`: 4→142, 8→146, 14→150) was taken when the shim was NOT
  the limit, so it does not settle the worker count for 4x128.
* **R34 at the conf's 4 workers waits 26 ms/step**. The live run paces 176 ms against 163 on the
  09-16 per-replica run; sync-BN explains only +4 of the +13. The remainder is feed — plausibly
  the page cache (the 09-21 DIMM fault mapped out 16 GiB; `vmstat bi` 40–70 MB/s during the run),
  but that is inferred, not measured.
* ⚠ A 400-step window cannot see a producer that degrades after hours (the ENet one did at
  5.6–13 h), nor ares' A3 stall tail. Here mean = median on every arm.

## Eval window, measured on the R34 run's epoch 1 (2026-09-22 15:28)

step 5000 at 15:28:05 → `@resnet34in_fwd_eval` RESIDENT (eval start) 15:28:13 → result 15:28:42:
**37 s** end of train → result, of which ~7 s spawns the two Python val producers and ~29 s scores
50,000 (≈1,720 img/s — producer-bound: two producers measured 1,608 img/s). The 09-16 run's old
held path took 40 s on the same measure, so streaming buys the ~31 GiB, not time.
R34 epoch 1 = **908 s** (steps 15:13:33 → checkpoint 15:28:41). The probe predicts 174 × 5004 + 37
= 911 s — **probe → ETA is good to <1 %** on this net.

## ETAs (probe median × steps + 37 s eval, per epoch)

| run | s/epoch | epochs | wall | note |
|---|---|---|---|---|
| R34 2018 (live) | 908 measured | 90 | **22.7 h** | lands ≈ 13:55 UTC Wed 09-23 |
| R50 2018 | 238 × 5004 + 37 ≈ 1,228 | 90 | **30.7 h** | ≈ the per-replica run's actual 30.72 h (230 ms + ~78 s/epoch overhead then, 238 ms + 37 s now — the overhead split is inferred from wall time) |
| R50 A3 4x128 | 300 × 2500 + 37 ≈ 787 | 100 | **21.9 h** | shim-bound; compute floor 244 × 2500 + 37 ⇒ **18.0 h** if the feed kept up |

⚠ R50's eval is assumed producer-bound like R34's (its sharded fwd should take ~16 s for 50k, under
the producers' ~29 s). Unmeasured until an R50 run's epoch 1.
