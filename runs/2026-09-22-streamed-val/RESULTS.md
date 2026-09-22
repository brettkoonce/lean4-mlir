# Streamed ImageNet val — planning/streaming_val.md, built and gated 2026-09-22 (ares)

Box: ares, 4× 4060 Ti, 251 GB. Checkpoint for the equality gate: the only ImageNet checkpoint on
this box that `score-checkpoint` accepts (BN nets refuse without running stats), the ViT-Tiny
`adamdp128x4wxclipdrop` warm probe of 2026-09-11 — 2.05% top-1, which is fine for an equality.
Deterministic shim + `SHIM_DETERMINISM=1` throughout.

## Golden (TODAY's binary, the 30 GB drain) — `vit/G1.*`, `vit/GN.*`
```
G1  R=1  checkpoint: acc = 1024/50000 = 2.048000%  top5 = 3466/50000  (99 s)
GN  R=4  checkpoint: acc = 1024/50000 = 2.048000%  top5 = 3466/50000  (70 s)
✓ golden  G1 == GN (50000 images, bitmaps bit-identical)
```

## §2 producer throughput — `measure_producers.sh` (val → /dev/null, wall of the slowest producer)
| producers | wall | img/s | note |
|---|---|---|---|
| 1 (unsharded, the old drain's producer) | 42.8 s | 1,168 | the plan's inferred ~1,250 |
| 2 batch-block | 31.1 s | 1,608 | unchanged between the two sharding designs |
| 4 batch-block | 29.4 s | 1,701 | was 1,946 with index slices; each producer now reads the whole raw stream |

Eval consumes ~1,700 img/s, so with 2 producers the pass is producer-bound at ~31 s against ~29 s
of compute: +2 s per pass, as §2 predicted ("2 ⇒ hidden (+0–2 s)"). A third producer would close it.

## The gate — `scripts/streamed_val_gate.sh test` (streamed binary vs golden)
```
T1      R=1                            acc = 1024/50000  top5 = 3466/50000  (51 s)
TN      R=4                            acc = 1024/50000  top5 = 3466/50000  (44 s)
F_order R=1 LEAN_MLIR_VAL_FAULT=order  acc = 1024/50000  top5 = 3466/50000  (51 s)
F_tail  R=1 LEAN_MLIR_VAL_FAULT=tail   acc = 1023/49920  top5 = 3463/49920  (52 s)
✓ test    T1 == G1 (count, top-5 and all 50000 per-image bits)
✓ test    TN == G1 (count, top-5 and all 50000 per-image bits)
✓ control F_order != G1 — same count, 2016 images flipped
✓ control F_tail reads 49920, not 50000
✓ PASS
```
Process wall time 99 → 51 s at R=1 and 70 → 44 s at R=4: the 40 s startup drain is gone.

## Two things the gate caught before any run could

1. **Absolute-index slices do not reproduce the drain's order.** The first emitter cut expressed
   producer k's blocks as a tfds split spec, `validation[0:256]+validation[512:768]+…` — no
   decode-and-discard, the shared iterator untouched. Same 50,000 images, same count, same top-5,
   **2,018 bitmap positions moved**. tfds reads a split as an interleave of its shard files (cycle
   length 16), so the order it yields — the order every historical bitmap is in — is not index
   order. The landed shim walks the same stream (`enumerate().filter()` on the raw records, before
   the decode) and keeps its blocks; the fragment stays byte-identical because the filter is
   installed on the `tfds.load` the fragment calls.
2. **A mutable accumulator that only held the last invoke.** `scored := scored + real` as the eval
   loop's last statement came back as 80 (the tail) on the R=1 path — "delivered 80 of 50000",
   refusal — and as 256 with the tail dropped, while `correct`, mutated inside the inner loop of
   the same body, summed correctly, R=4 summed correctly, and the same binary path summed
   correctly the moment an `eprintln` read the variable after the assignment. A 20-line copy of
   the loop's shape sums correctly standalone (Lean 4.34.0, default codegen). The denominator now
   comes from what the pass delivered — the stream carry's `total`, cross-checked against the
   bitmap's length — and the trainer does not depend on that accumulator.

## §4.4 trainer smoke — `smoke.sh` (R=1, `adam128`, one epoch capped at 40 steps, tag `streamsmoke`)
```
trainer peak VmRSS: 2414 MiB                         (the drain held ~30 GB of it before)
epoch 1: test_acc = 55/50000 = 0.110000%  top5 = 283/50000     ▸ val = all 50000 streamed
checkpoint: acc = 55/50000 = 0.110000%  top5 = 283/50000        (score-checkpoint on that checkpoint)
✓ in-run bitmap == score-checkpoint bitmap (50000 bytes)
```
The eval window is not readable off this log (the trainer's stdout is block-buffered through the
pipe, so every line carries the flush time); measure it on a real run.

## Not done here
* Goldens for the ViT e300 and ConvNeXt-T finals (§4.1): those checkpoints are on the other box.
* Pre-spawn timing (§3.2): producers spawn at eval start; TF startup is inside the 44–51 s above.
