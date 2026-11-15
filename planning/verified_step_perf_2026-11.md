# Verified multi-card step: where the time goes — 2026-11

Why the verified ConvNeXt step runs about 1.8× slower than the JAX reference per image on fast
cards, what was measured, and what to try. Written 2026-11-15 from a rented 4× RTX 5080 box, as the
starting point for a session on a home box. Raw outputs are untracked under
`runs/2026-11-14-vast-3090-japan/` and `runs/2026-11-15-vast-5080-california/` on atlas.

## The gap

Per 256 images, ConvNeXt-S, both paths at their best loader configuration:

| cards | verified | JAX | gap |
|---|---|---|---|
| 4× RTX 4060 Ti (the confs' October smokes) | 323 ms | 247 ms | 76 ms |
| 4× RTX 5080 | 184 ms | 102 ms | 82 ms |
| 4× RTX 3090 | 304 ms | 187 ms | 117 ms |

The gap is close to a constant number of milliseconds, so it is a 1.3× ratio on slow cards and 1.8×
on fast ones. ConvNeXt-B on the 5080s: 253 ms against 138. ResNet-50 A2 (512 rows): 206 against 170.
The loader is not part of it: on synthetic batches the verified step is within a few ms of these.

## What the probes show (ConvNeXt-S, synthetic batches, 4× RTX 5080)

`PJRT_FFI_TIMING` splits a step into upload, compute (`launch` = time inside the Execute call,
`device` = the wait for the cards), and read-back; the probe's median minus that total is the time
the trainer spends between calls.

| render | cards × rows | probe median | in the bridge | launch | device |
|---|---|---|---|---|---|
| `emaaccdp16xwxdroperfbf16` (production) | 4 × 64 | 184 ms | 148.5 | 45.8 | 88.7 |
| `emadpwxclipdroperfbf16` (no accumulation) | 4 × 64 | 171 ms | 142.8 | 42.4 | 86.1 |
| `emawxclipdroperfbf16` (single device) | 1 × 64 | 90 ms | 83.6 | 31.9 | 43.3 |

Operands: 1,765 / 1,419 / 1,419. Cross-replica all-reduces per step: 345 / 345 / 0.

Readings:

- **Cross-card synchronization roughly doubles device time.** Same 64 rows per card, 43 ms on one
  card and 86–89 ms on four. The only graph difference is the 345 all-reduces, one per tensor.
- **`launch` is mostly a fixed per-call cost.** 32 ms with no all-reduce at all, 42–46 ms with them.
  ResNet-50 (773 operands, 321 all-reduces) launches in 41 ms, so it does not track operand count.
- **The time between calls is a multi-card cost.** About 6 ms on one card, 30–35 ms on four.
- **Accumulation regions are cheap.** 184 ms against 171.

Put together, the production step's 184 ms is roughly: 43 one card's compute, 32 fixed launch, 55
cross-card synchronization (device plus launch), 30 multi-card work between calls, 24 upload,
read-back and accumulation. JAX's 102 ms is close to the first two.

On the 5080s the verified ResNet step keeps the cards at about 55 % utilization and 126 W of 342;
JAX with a loader that keeps up reaches 99–100 %.

Not measured: a stack profile of the time between calls (a debugger cannot attach inside a rented
container; it can on a home box), and a 2-card point (the 2-card ConvNeXt-S render failed at start
on 16 GB cards).

## ViT, same box (synthetic unless noted)

| render | cards × rows | probe median | launch | device |
|---|---|---|---|---|
| ViT-B `emaaccdp2x128x4wxdropeps0000001smxerfbf16` (production), real data | 4 × 128 | 379 ms | 28 | 312 |
| ViT-B production | 4 × 128 | 371 ms | 29 | 308 |
| ViT-B `emadp128x4wxclipdroperfbf16` (no accumulation) | 4 × 128 | 352 ms | 27 | 288 |
| ViT-B `adam128wxclipdroperfbf16` (single device) | 1 × 128 | 248 ms | 20 | 211 |
| ViT-S `emadp128x4wxclipdroperfbf16` | 4 × 128 | 148 ms | 22 | 87 |

ViT-B is compute-bound: the cards run at 80 % and 200 W, cross-card synchronization (201 all-reduces)
adds about 77 ms of device time, and launch plus the time between calls is about 13 % of the step.
ViT-S is small enough that the fixed costs are again roughly 40 % of the step. There is no
single-device ViT-S point.

Did not run on 16 GB cards: the ViT-S production render (4 × 256; out of memory on one 10.85 GB
allocation at the conf's default arena) and the JAX ViT-B trainer (one 12.89 GB allocation, at
memory fraction 0.93). The JAX ViT-S trainer read about 412 ms per 1,024-row optimizer step with the
8-process feed, from step-log differences over 100 steps.

## The all_reduce bytes probe on the 5080s (2026-11-15)

`scripts/probes/allreduce_bytes_probe.sh`, ConvNeXt-S production step, 4× RTX 5080, synthetic. The
3060 box read 354 / 354 / 331 / 314 ms for the last four rows.

| row | probe median | launch | device | loss at step 200 |
|---|---|---|---|---|
| as rendered, binary before the `dropScales` fix | 186 ms | 48.0 | 88.3 | 7.022544 |
| as rendered (344 all_reduces, 200 MB) | 178 ms | 47.8 | 92.4 | 7.022544 |
| one concatenated all_reduce | 176 ms | 44.5 | 91.7 | 7.022544 |
| half the gradient exchanged (timing only) | 152 ms | 37.9 | 68.3 | — |
| almost none exchanged (timing only) | 127 ms | 38.2 | 44.7 | — |

- The `dropScales` fix is worth 8 ms here, all outside compute; losses match at steps 0, 1, 2, 100, 200.
- The number of all_reduces is free on this box too. The bytes cost about 0.25 ms per MB: device
  92 → 68 → 45 ms, launch about 10 ms. The last row's device time is the 1-card figure (43 ms).
- ViT-B exchanges about 344 MB, which predicts 86 ms; the 1-card against 4-card probe above read 77.

So the lever is not coalescing but **exchanging once per accumulation cycle**: each card accumulates
its own gradients over the cycle's micro-batches and the apply step does one exchange. The sum is the
same; the order of additions is not, so losses will not be bit-identical to the current step. It
needs a new render (accumulation step without all_reduces, apply step with one exchange), per-card
accumulators in the driver (today they are identical across cards after every exchange), and the
data-parallel statements and render gates extended to it. It should land before a verified
ConvNeXt run starts, or that run stays on the current step.

Estimated 300-epoch verified runs on the 5080 box. Only ConvNeXt-S is measured; the others scale its
saving by bytes exchanged and by (cycle − 1) / cycle:

| net (cycle) | now | `dropScales` fix | plus one exchange per cycle | JAX |
|---|---|---|---|---|
| ConvNeXt-S (16) | 184 ms, ~80 h | 176 ms, ~77 h | ~130 ms, ~57 h | 46 h |
| ConvNeXt-B (16) | 253 ms, ~109 h | ~105 h | ~160 ms, ~69 h | 62 h |
| ResNet-50 A1 (4), 600 epochs | 202 ms, ~89 h | ~88 h | ~81 h | 74 h |
| ResNet-50 A2 (4) | 206 ms, ~47 h | ~46 h | ~43 h | 38 h |
| ViT-B (2) | 379 ms, ~82 h | ~82 h | ~74 h | — |

On the 3060s the same change is about 11 % for ConvNeXt-S (354 → about 316 ms). ViT-S does not
accumulate and gains nothing.

A rented box built from the image is not a git checkout and `supervise.sh` rebuilds the executable
from source at every launch, so a probe that needs a patched driver is built in a copy of the tree
(`rsync` without `runs/` and `.venv`, `.venv` symlinked; `lake build` there is a no-op until the
source changes), not in the tree a paused run will resume from.

## Built: one exchange per cycle (2026-11-15, mars)

The `cyc` renders: `convnextsin_` / `convnextbin_emaaccdp16xcycwxdroperfbf16` (and their tanh twins).
Same recipe, operands and blob layout as `emaaccdp16xwxdroperfbf16`; each replica accumulates its own
gradients and the all_reduce sits inside a `stablehlo.if` on `%aup`, one per parameter. The driver is
unchanged except for a guard.

| 4× RTX 3060, synthetic | per micro-batch | `cyc` |
|---|---|---|
| ConvNeXt-S | 354 ms | 319 ms |
| ConvNeXt-B | 535 ms | 472 ms |

- **Tie.** Two cycles of real data (`SHIM_DETERMINISM=1`), checkpoints compared by region against the
  per-micro-batch render: `m` norm-rel 3.80e-4, against 3.85e-4 for that render run twice. With
  `PJRT_FFI_RESIDENT=0` (the control) `m` is off by 1.08; the driver now refuses that combination.
- **Proof side.** New node `allReduceMeanWhenF` (denotation, printer, parser roundtrip);
  `Proofs.cycleExchange_eq`: the replica mean of the per-replica totals is the total of the replica
  means. Three-axiom clean, in the axiom audit.
- **One conditional for all 344 tensors** instead of one each measured 313 ms against 319 on S
  (hand-rewritten render). Not built: it needs a multi-operand node.
- **Confs.** `cnxs-` and `cnxb-default-emabf16-4gpu` now name the `cyc` variant; one line each to revert.
  On 12 GB cards launch with `LEAN_MLIR_MEM_FRACTION=0.95` in the environment.
- Not done: ResNet-50, ViT-B and MobileNetV4 (`optOne`'s `cyc` reaches only ConvNeXt's writer; the
  clip and LAMB routes hoist the accumulator and need their own wiring), the two-card renders, and a
  5080 measurement of the built render.

## What to try

1. **One gradient exchange per accumulation cycle** (above). Coalescing the all_reduces without
   cutting the bytes gains nothing.
2. **One call per accumulation cycle.** The JAX trainer runs its `GRAD_ACCUM` micro-batches inside
   one compiled call; the verified driver calls the step 16 times. Folding the cycle into the step
   amortizes the 32 ms launch.
3. **Profile the 30 ms between calls** on four cards. Candidates: the host-drawn stochastic-depth
   masks, per-replica slicing of the batch, argument marshalling for 1,765 operands × 4 replicas.
4. **Overlap the image upload with compute** (ResNet: 27 ms of a 206 ms step on the 5080s).

Each is worth checking on a home box first: the 4060 Ti figures above are the confs' October
numbers, not this method.

## Reproducing the probes

From the repo root, cards idle. Production step, 4 cards, synthetic data:

    LEAN_MLIR_BENCH_SYNTH=1 PJRT_FFI_TIMING=40 LEAN_MLIR_MAX_STEPS=240 LEAN_MLIR_PROBE_WARM=40 \
      ONCE=1 RUNDIR=/tmp/cnxs_4card scripts/supervise.sh cnxs-default-emabf16-4gpu

Single device, no cross-card synchronization (the 4-card one-shot is the same with
`LEAN_MLIR_VARIANT=emadpwxclipdroperfbf16`, `PJRT_REPLICAS=4`, `LEAN_MLIR_REPLICAS=4` and all four
devices):

    . scripts/platform/env.sh
    CUDA_VISIBLE_DEVICES=0 PJRT_PLUGIN=$PLATFORM_PLUGIN PJRT_REPLICAS=1 LEAN_MLIR_REPLICAS=1 \
      LEAN_MLIR_VARIANT=emawxclipdroperfbf16 LEAN_MLIR_BATCH=64 LEAN_MLIR_EPOCHS=300 \
      LEAN_MLIR_BASE_LR_U=4000 LEAN_MLIR_G2_STEPS=4992 SHIM_WORKERS=4 \
      LEAN_MLIR_BENCH_SYNTH=1 PJRT_FFI_TIMING=40 LEAN_MLIR_MAX_STEPS=240 LEAN_MLIR_PROBE_WARM=40 \
      .lake/build/bin/convnext-s-imagenet-verified data

Read `PROBE:` (median step) and `COMPUTE … (launch … + device …)`. The probe returns before the
epoch ends and writes no checkpoint. Drop `LEAN_MLIR_BENCH_SYNTH` for real data.

## Loader notes from the same sessions

- **Shim producers must be pinned on a many-core box.** Unpinned, each sizes its TensorFlow pools to
  every visible core: 8 producers gave 1,488 img/s unpinned and 4,678 pinned (10 physical cores
  each) on 2× EPYC 7K62 under a 92-core quota. Pinning is a `SHIM_PYTHON` wrapper that `taskset`s
  producer `i` of `n` from `SHIM_SHARD`; its core layout is per box, so it is not in the repo. Check
  `/sys/fs/cgroup/cpu.max` before choosing the cores.
- **More producers is worse.** The shim shards by record (`ds.shard` after `tfds.load`), so each of
  `n` producers reads the whole dataset. 8 beat 12 and 16 on every box. Sharding by file would lift
  that.
- **The JAX trainer's single tf.data process tops out near 1,500 img/s.** `LEAN_MLIR_FEED_WORKERS=k`
  (this branch, `mpFeedPy` in `jax/Jax/Codegen.lean`) feeds it from `k` shim processes instead:
  R50 A2 1,489 → 678 ms a step, ConvNeXt-S 3,375 → 1,627, ConvNeXt-B 3,210 → 2,212 on the 5080s.
  Home boxes are card-bound and do not need it.
