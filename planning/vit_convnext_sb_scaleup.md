# ViT-S/B + ConvNeXt-S/B on ImageNet — what the scale-up actually costs

**Question asked:** to add the S/B variants of the two ImageNet demos, is it
enough to change the model parameters, or is something else needed?

**Answer:** for the *model*, yes — the specs are pure parameter changes and the
phase-2 JAX emitter handles all four without a line of codegen work. Everything
that needed doing was infrastructure around the runs, listed at the bottom.

Measured 2026-07-25 on ares (6× RTX 4060 Ti, 16 GB; JAX 0.10.2 CUDA).

## The specs are parameter-only

| new file | change vs the existing demo |
|---|---|
| `jax/MainVitSImagenet.lean` | `patchEmbed 3 384 16 196`, `transformerEncoder 384 6 1536 12` (Ti: 192/3/768) |
| `jax/MainVitBImagenet.lean` | `patchEmbed 3 768 16 196`, `transformerEncoder 768 12 3072 12` |
| `jax/MainConvNeXtSImagenet.lean` | stage-3 depth 9 → 27; channels unchanged |
| `jax/MainConvNeXtBImagenet.lean` | stage-3 depth 9 → 27; channels 96/192/384/768 → 128/256/512/1024 |

Nothing else moved. `mhsa` derives `head_dim = D // n_heads` at runtime, the
transformer/ConvNeXt block emitters loop over `nBlocks`, and the init / load /
save param walks are all generated from the same spec — so width and depth are
genuinely free parameters. Emitted parameter counts land on the published
figures exactly:

| model | emitted params | reference |
|---|---|---|
| ViT-S/16 | 22,050,664 | 22.05 M (DeiT-S) |
| ViT-B/16 | 86,567,656 | 86.57 M (DeiT-B) |
| ConvNeXt-S | 50,222,152 | 50.22 M |
| ConvNeXt-B | 88,589,416 | 88.59 M |

Recipes are copied from the existing Ti demos unchanged (DeiT applies one recipe
across Ti/S/B), with two deliberate exceptions, both commented in-file:
stochastic depth follows the ConvNeXt paper's per-size values (S 0.4, B 0.5) on
the 300-epoch recipe and is pulled back (0.2 / 0.3) on the 80-epoch tier, where
the paper values underfit.

## Measured cost per step

`jax/scripts/step_probe.py`: real `train_step` + `ema_update` on
synthetic batches, params replicated / batch sharded exactly as the trainer does.
Excludes the tf.data input pipeline, so ms/step is a **compute-only lower bound**
(the ConvNeXt-T run measured 143 ms/step live vs 130 ms here — ~10% pipeline tax).
`peak` is `peak_bytes_in_use` on device 0; the ceiling is 11.68 GiB, not 16, because
XLA preallocates 75% by default.

6 GPUs, no gradient accumulation:

| model | batch | ms/step | peak | min/epoch | hr @80ep | hr @300ep |
|---|---|---|---|---|---|---|
| ViT-Ti (existing) | 510 | 110 | 1.92 GiB | 4.6 | 6.1 | 23.1 |
| **ViT-S** | 510 | 243 | 4.05 GiB | 10.2 | 13.5 | 50.8 |
| **ViT-B** | 510 | 595 | 9.32 GiB | 24.9 | 33.2 | 124.5 |
| ConvNeXt-T (existing) | 252 | 130 | 3.01 GiB | 11.0 | 14.7 | 55.1 |
| **ConvNeXt-S** | 252 | 208 | 4.92 GiB | 17.6 | 23.5 | 88.0 |
| **ConvNeXt-B** | 252 | 303 | 7.23 GiB | 25.7 | 34.2 | 128.4 |

All four run out of the box on 6 GPUs at the Tiny demos' batch sizes, loss finite
and decreasing. ViT-B at 9.32 of 11.68 GiB is the only one without comfortable
headroom.

### 4 GPUs

The historical runs used 4 GPUs (0,2,3,4) to dodge the PCIe AER issue. At 4
devices the per-device batch is 1.5× larger.

| model | batch | ms/step | peak | hr @80ep | hr @300ep |
|---|---|---|---|---|---|
| ViT-S | 512 | 313 | 7.60 GiB | 17.4 | 65.2 |
| ViT-B | 512 | — | **OOM** | — | — |
| ViT-B (4×128 accum) | 512 eff | 712 | 5.65 GiB | 39.6 | 148.5 |
| ConvNeXt-S | 256 | 262 | 6.77 GiB | 29.1 | 109.3 |
| ConvNeXt-B | 256 | 395 | 9.56 GiB | 43.9 | 164.6 |

Three of four run unchanged; **ViT-B OOMs** (`RESOURCE_EXHAUSTED`, peak 11.41 of
11.68 GiB). Going 6→4 devices costs ~30% wall-clock, not 50% — these are
compute-bound enough that fewer, fuller devices lose less than the device count
suggests.

This is why each new file carries an `accum` recipe: same effective batch (so the
LR stays correct), micro-batched. These nets have no BN, so accumulation is
numerically exact. The existing `gradAccumSteps` path needed no changes — it just
had never been exercised on ViT/ConvNeXt before.

There is no `jax.checkpoint`/remat in the phase-2 emitter, but XLA does its own
rematerialization under memory pressure — see the ConvNeXt-B row below.

### ConvNeXt at effective batch 512 on 4× 16 GB

bs512 is not just a memory question: the LR doubles with the batch under the
paper's linear rule (4e-3 @ bs4096 → **5e-4** @ bs512, vs 2.5e-4 @ bs256). The
`bs512` recipes on the S and B files carry that.

| model | recipe | ms/step | peak of 11.68 GiB | min/epoch | hr @80ep |
|---|---|---|---|---|---|
| ConvNeXt-T | bs512 direct | 313 | 7.60 GiB | 13.0 | 17.4 |
| ConvNeXt-S | bs512 direct | 481 | 11.26 GiB | 20.1 | 26.8 |
| ConvNeXt-S | 2×256 accum | 521 | **6.91 GiB** | 21.7 | 29.0 |
| ConvNeXt-B | bs512 direct | 747 | 11.52 GiB | 31.1 | 41.5 |
| ConvNeXt-B | 4×128 accum | **681** | **6.19 GiB** | 28.4 | 37.9 |

All three fit one-shot, but S leaves 0.42 GiB and B leaves 0.16 GiB — and this
probe `device_put`s a pre-made array, so it does not model the tf.data
depth-2 prefetch buffers that also sit on device. Run the accum recipes.

**ConvNeXt-B's accum recipe is strictly better than one-shot**: 681 vs 747
ms/step at 5.3 GiB less. Same mechanism as the 3060 case — under memory pressure
XLA rematerializes, and the recompute costs more than the accumulation loop. For
ConvNeXt-S the trade is real but mild (8% for 4.35 GiB).

Larger batch also wins per epoch for both: S 21.7 min at bs512 vs 21.9 at bs256,
B 28.4 vs 32.9.

Accumulation correctness cross-check: direct and accum losses agree to 3–4
decimals at matched steps (S 6.1816 vs 6.1828, B 5.7601 vs 5.7625), the residue
being drop-path RNG splitting across micro-batches.

### Why ViT-B on 6 GPUs runs at 510, not 512

512 = 2^9 has no factor of 3, so it cannot be split evenly across 6 devices;
`BATCH_SIZE = (512 // n_devices) * n_devices` yields 510. Grad accumulation does
not help — the total is `micro × accum` with `micro` divisible by 6, so every
reachable total is divisible by 6. Exactly 512 on 6 GPUs is unavailable.

Uneven sharding is not an escape hatch — JAX 0.10.2 rejects it three ways
(`IndivisibleError`): `device_put` of a 512 leading dim over 6 devices, a jitted
op over such an array, and even hand-building it via
`make_array_from_single_device_arrays` from explicit 85/85/85/85/86/86 buffers.
`NamedSharding` validates against its computed shard shape before it looks at
your buffers; `shard_map` and `pmap` carry the same constraint.

It does not matter anyway. DeiT's rule is `lr = 5e-4 × batch/512`, so batch 510 at
LR 4.98e-4 is simply a correct config at a slightly smaller batch — not an
approximation of a 512 run. (A literal 512 *is* reachable by padding to 516 and
zero-weighting 4 samples in the loss, but that burns 0.8% of compute to fix a
0.4% discrepancy.)

**Round up, not down.** Measured on ViT-B / 6 GPUs:

| batch | ms/step | peak | min/epoch |
|---|---|---|---|
| 510 (6×85) | 594.7 | 9.32 GiB | 24.9 |
| **516 (6×86)** | **576.8** | 9.40 GiB | **23.9** |

516 is 3% faster per step *and* carries 1.2% more samples — ~4% more throughput.
Step times do not overlap (510: 592–596 ms over five samples; 516: 573–578), so
86/device genuinely tiles better than 85/device.

General note for this box: 6 devices means every batch must be `6 × per_device`,
and no power of two ever is. The clean choices are **384** (6×64) and **768**
(6×128); near the usual literature values, 510/516 stand in for 512 and 252/258
for 256. Scale LR linearly to whatever you pick.

## Would 4× RTX 3060 12 GB work?

**Memory: yes, all four.** XLA preallocates ~73% of a card (11.68 GiB observed of
16380 MiB), so a 12 GB 3060 gets ~8.7 GiB. Emulated by capping these cards at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.548` → an 8.54 GiB limit, slightly *tighter*
than a real 3060, so these results are conservative. 4 devices throughout:

| model | recipe | ms/step | peak of 8.54 GiB |
|---|---|---|---|
| ViT-S | bs512 | 313.5 | 7.60 GiB |
| ConvNeXt-S | bs256 | 262.7 | 6.77 GiB |
| ConvNeXt-B | bs256 | 401.2 | 8.54 GiB — exactly at the ceiling |
| ConvNeXt-B | 4×64 accum | **375.6** | **4.18 GiB** |
| ViT-B | 4×128 accum | 712.5 | 5.66 GiB |

Two things worth knowing:

- **ConvNeXt-B fits at bs256 without accumulation**, contrary to the naive
  read of its 9.56 GiB peak at the 16 GB budget. XLA rematerialized to fit the
  smaller budget and paid only 1.6% (395 → 401 ms/step). But peak landing
  *exactly* on the limit means zero headroom for prefetch buffers or
  fragmentation.
- **The accum recipe is strictly better for ConvNeXt-B here**: 375.6 ms/step at
  4.18 GiB beats 401.2 ms/step at 8.54 GiB on both axes. Micro-batches fit
  natively, so XLA skips the recompute that rematerialization forces. Use
  `accum` for ConvNeXt-B on 12 GB cards — it is faster *and* leaves 4 GiB spare.

**Speed: ~1.7× slower per card, estimated, not benchmarked** (no 3060 on hand).
RTX 3060 12 GB is 12.74 TFLOPS FP32 / ~51 TFLOPS bf16-tensor vs the 4060 Ti's
22.06 / ~88 — a 1.73× compute deficit — but 360 vs 288 GB/s of bandwidth, a 1.25×
surplus. These runs sit at 95–100% GPU util (compute-bound), so compute dominates.
Applying 1.7× to the measured rows:

| model | recipe | est. hr @80ep | est. hr @300ep |
|---|---|---|---|
| ViT-S | bs512 | ~30 | ~111 (4.6 d) |
| ConvNeXt-S | bs256 | ~50 | ~186 (7.8 d) |
| ViT-B | 4×128 accum | ~67 | ~252 (10.5 d) |
| ConvNeXt-B | 4×64 accum | ~71 | ~266 (11.1 d) |

Caveats specific to a 3060 box: Ampere (sm_86) has native bf16 tensor cores, so
`bf16` and `bf16Conv` work unchanged — the sm_89 workaround in the IREE notes is
IREE-specific and does not apply to this JAX path. If one 3060 drives a display,
subtract a few hundred MB and ViT-S's 0.9 GiB margin gets thin; run it with
`accum` there. The **8 GB** 3060 variant would give only ~5.9 GiB, which all four
would need `accum` to clear.

## What actually needed doing (the "something else")

1. **`jax/` did not build at HEAD.** `DatasetKind.brats224` was added for the
   BraTS work and three matches in `Jax/Codegen.lean` were never updated
   (two `.brats` panics, one trace-header name). Fixed.
2. **`MainMobilenetV3.lean` did not build either** — long-standing, since 574b1f9
   changed `mbConvV3`'s last field from `useHSwish : Bool` to `act : Activation`.
   Fixed mechanically (`false` → `.relu`, `true` → `.hSwish`; the per-line RE/HS
   comments confirm each). Unrelated to this work, but it blocks a clean
   all-executables build.
3. **Disk.** The supervise scripts run with `LEAN_MLIR_CKPT_EVERY=1`, and `.bin`
   weight files were never pruned (only `.state.npz` was). At ~340 MB per
   checkpoint, a 300-epoch ViT-B or ConvNeXt-B run writes ~100 GB of weights, and
   the box has ~120 GB free. Added `_prune_bin_ckpts`, gated on
   `LEAN_MLIR_KEEP_BIN` and **defaulting to keep-everything** so existing runs are
   byte-identical; set it to 3 for the big nets.
4. **Eval scripts hardcoded the Tiny trainer path.** `eval_vit_full50k.py` and
   `eval_convnext_full50k.py` now take `GEN` from the environment (same default).
   The rest of both scripts is already architecture-agnostic.
5. **Supervise scripts are per-run copies.** A new variant needs a copy with
   `DEVS` / `PY` / `CKPT_BASE` / `SPE` edited. Also note `supervise_vit_80ep.sh`
   points `VENV_PY` at `/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python`,
   which no longer exists; the working CUDA venv is
   `/home/skoonce/lean/klawd_max_power/lean4-jax/.venv`.
6. **Compile time.** ConvNeXt-S/B spend ~85–105 s per jit compile and there are
   several (train_step, eval_batch, ema_update, mixup, cutmix) — roughly 3–5 min
   of startup per (re)start. With AER auto-resume that is a per-resume tax the
   Tiny runs did not really pay.

## Suggested order if these get run

ConvNeXt-S at the 80-epoch tier is the best value: 23.5 hr, and ConvNeXt-T
already holds the sweep accuracy lead at 75.93%, so S is the most likely to move
the headline number. ViT-S 80ep (13.5 hr) is the cheapest new data point and the
clean apples-to-apples against ViT-Ti's 65.6%. The B variants at 300 epochs are
~5-day runs each and should wait until an S result confirms the recipe.
