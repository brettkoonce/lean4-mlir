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

## What to try

1. **Fewer, larger all-reduces.** One exchange for all gradients (or a few), not one per tensor.
   Targets about 55 ms of 184. This changes the rendered step, so the whole-net data-parallel
   statements and the arity / render gates are in scope.
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
