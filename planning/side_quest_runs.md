# side_quest_runs.md — the side-quest ImageNet pairs on ares, 24/7

Started 2026-09-28. The six side-quest pairs (R50 RSB-A2/A1, MNv4-Conv-M paper tier, ViT-S/B,
ConvNeXt-S/B) run on this box (4× 4060 Ti, 16-core 5955WX, 4 of 8 memory channels populated)
around the clock, with everything else moved off it. The JAX reference of each batch runs first,
then its verified pass. Running a side quest does not promote it into the Track-4 table.

## 0. Rules

* Nothing launches without the user's word ([[lake_run_job_launches]]); each batch is proposed
  with its commands.
* No commits or pushes without approval; stage, then stop.
* A recipe follows the paper's hyperparameter table; where timm's reference code adds a detail
  the table leaves out (the BCE target threshold), timm's code is the spec.
* Perf work on the feed is measured before it is built (§4), and a change to an emitted
  augmentation op keeps the PIL gate green (`scripts/gates/aug_bicubic_pil_check.py`).

## 1. Schedule

Stall-free hours on this box, i.e. once the DIMM throttle is cooled away (§5). Every job takes
all four cards, so the queue is sequential. Per-epoch evals (a few hours per run) are not included.

| batch | runs (hours) | total | days 24/7 |
|---|---|---|---|
| 1, JAX | R50 A2 73 · MNv4 `full` 500 ep ~70–85 · ViT-S 63 | ~210 | ~9 |
| 2, verified | R50 A2 ~95 · MNv4 500 ep ~90 · ViT-S ~68 | ~255 | ~10.5 |
| 3, JAX | R50 A1 ~146 · ViT-B ~146 · ConvNeXt-S ~110 | ~400 | ~16.7 |
| 4, verified | R50 A1 ~190 · ViT-B ~170 · ConvNeXt-S ~159 | ~520 | ~21.7 |
| 5 | ConvNeXt-B JAX ~167, then verified ~253 | ~420 | ~17.5 |
| total | | ~1,805 | ~75 |

Where each number comes from:

| run | basis |
|---|---|
| A2 JAX 73, ViT-S JAX 63 | measured today, clean windows (§5) |
| A1 JAX | 2× A2's measured rate |
| ViT-B / ConvNeXt-S / ConvNeXt-B JAX | the 4-GPU compute probe (`runs/2026-08-27-jax-sb-tier-step-probe`: 700.8 / 260.8 / 399.3 ms per step); the feed does not bind them |
| MNv4 JAX | between the tf.data ceiling (1.5 s/step) and the 3060 box's 1.9 s/step |
| MNv4 / ViT-S verified | their shim ceilings (§4.2); MNv4's matches the 3060 box's run |
| A2 / A1 verified | the 08-27 device-probe model |
| ViT-B / ConvNeXt verified | the 09-10 bf16 probes, which predate sync-BN, so re-probe before launch |

## 2. Where things stand (2026-09-29)

* **Batch-1 JAX software is done**, committed as `b84da996` (not pushed). Its three trainers pass
  their smokes (§5).
* **A2 JAX can launch now.** It does not stall even without the fan:
  `setsid nohup scripts/supervise.sh r50-a2accum-jax-4gpu >/dev/null 2>&1 &`
* **MNv4 `full` and ViT-S JAX are ready but wait for the DIMM fan** (§5). After the fan goes in,
  rerun the ViT-S thermal smoke (scratchpad `thermal/run.sh`, re-created from §5's description).
  Pass = the hottest DIMM stays below 77 °C and ViT-S runs at its clean ~300 ms/step.
* **Next session:** the software list in §3, in order. When it is done, the software side of the
  whole queue is prepared.

## 3. Software work list

In queue order. Sizes: S = small, M = medium, L = large.

### Batch 1, JAX A2 / MNv4 `full` / ViT-S — done (`b84da996`)

* BCE target threshold, timm's `--bce-target-thresh`: `TrainConfig.bceTargetThresh`
  (`LeanMlir/Types.lean`), emitted after smoothing in `jax/Jax/Codegen.lean`'s BCE branch.
* A2/A1 per RSB Table 2: no EMA, threshold 0.2; A1 adds label smoothing 0.1
  (`jax/MainResnet50Imagenet.lean` `a2-accum`, `a1`).
  * RSB lists no gradient clip; the trainers' clip is timm LAMB's own `max_grad_norm=1.0`, kept.
  * The A3 recipes (`short`, `rsb-faithful`) are untouched: their runs are in the book.
* MNv4 `full` scores live weights, since the paper runs no EMA. It still differs from the paper
  in RandAugment p 0.5 (paper 0.7) and in m15 exceeding the shim's `_AA_MAX` of 10.
* `jax/generated/`: only the a2accum, a1 and MNv4 `full` trainers moved.
* Confs `r50-a2accum-jax-4gpu`, `mnv4-full-jax-4gpu` and `vits-default-jax-4gpu`; the DRY_RUN
  prechecks pass.

### Batch 2, verified A2 / MNv4 / ViT-S — needed in ~9 days, when the batch-1 JAX runs finish

1. **ViT-S**
   * `vitInit := true` in the verified driver (`apps/imagenette/MainViTSImagenet.lean:47-49`;
     Ti has it at `MainViTImagenet.lean:43`).
   * EMA+bf16 render `vitsin_emadp128x4wxclipdropbf16`: one `#eval` after
     `LeanMlir/Proofs/Codegen/ViTRenderB.lean` ~816, on the Ti template at :750, with
     `V := vitSDims`, plus its `#guard`.
   * Pass `emaDecay := 0.99996` in the driver (`MainViTSImagenet.lean:69`; Ti at
     `MainViTImagenet.lean:64-72`).
   * A `vitsin` arm in `tests/TestVitEmaDropRender.lean:75-79`.
   * Conf `vits-default-emabf16-4gpu` on the Ti emabf16 conf's checks, plus its lakefile
     `imagenetRows` row and `script` line.
   * Size: S each.
2. **A2**
   * No-EMA render `resnet50in_lambaccdp4x128wxclipdropbce{,bf16}`
     (`ResNet50RenderB.lean` ~1766-1797). M.
   * The 0.2 target threshold in the verified trainer's target build. M.
   * A verified conf from `r50-a2-*`/`r50-a3-wxclip4x128-bf16-4gpu`, with 300 epochs,
     `BASE_LR_U=5000`, `LEAN_MLIR_BATCH=128`, prechecks and a lakefile row. S.
   * The `resnet50in_fwd_eval_s288` scoring render, since timm scores at 288/1.0. S.
   * An R50 timm parity gate in `scripts/parity/`. M.
3. **MNv4**
   * No-EMA render plus its 1-replica peer for `mnv4-dp-check`
     (`MobileNetV4RenderB.lean:1282-1289`, `ema := false`, `wdStr := "0.1"`). S.
   * Select the m15 `full` shim; `SHIM_SCRIPT` works as a stopgap. S.
   * A classifier dropout 0.2 knob (`NetsCore.lean:1511`; the keep is fixed at 0.9 today). S.
   * UIB drop-path 0.075 on the verified side: render plus tie carve-out. M–L; the one big item.
   * Conf and lakefile row. S.
4. **All three:** new renders go on the `proofs.yml` render-guard list, then regenerate MANIFEST
   and add `TestVariantPredicates.lean` rows.

### Batch 3, JAX A1 / ViT-B / ConvNeXt-S

5. Confs for A1 and ViT-B; their trainers exist. ViT-B uses `accum` (4×128), because a single
   512 micro-batch OOMs. S.
6. ConvNeXt-S: `cnxInit := true` in `jax/MainConvNeXtSImagenet.lean`, then re-emit, plus a JAX
   conf. S.

### Batch 4, verified A1 / ViT-B / ConvNeXt-S

7. **A1**
   * No-EMA wd 0.01 render. M.
   * Label smoothing 0.1 applied before the threshold on the verified side. M.
   * Conf. S.
   * Fix the driver comment naming the deleted 8×64 variant
     (`apps/imagenette/MainResnet50Imagenet.lean:91-94`), and refuse recipe `a1` on a
     non-wd001 variant. S.
8. **ViT-B**
   * Item 1 for `vitbin`: `MainViTBImagenet.lean:93-95` / `136-138`, render after
     `ViTRenderB.lean:889`. S each.
   * Probe EMA+bf16 memory at `LEAN_MLIR_MEM_FRACTION=0.97`. bf16 alone peaks at 12.61 of
     15.11 GiB.
9. **ConvNeXt-S**
   * `cnxInit` in the verified driver. S.
   * Driver defaults: batch 64 and the EMA variant. S.
   * EMA+bf16 render at batch 64, plus `_fwd` / `_drop_fwd` re-rendered at 64
     (`ConvNeXtRenderB.lean:804-810` pattern, `V := cnxSmall`, `bB := 64`). S.
   * `convnextsin_fwd_s288`. S.
   * Layout-gate arm, render-guard entries and conf. S.

### Batch 5, ConvNeXt-B

10. **ConvNeXt-B**
    * Item 9 for `convnextbin`.
    * Memory: it is 9.73 GiB at 32/replica, 64/replica likely overflows 11.68, and 0.97 OOMs this
      net. Probe it first; it probably needs an accumulation render, and none exists for
      ConvNeXt. M.
    * Proof tier only, not a run blocker: the whole-net ties are pinned at T, and B's 1024-wide
      head is uncovered (`ConvNeXtStepTieGB.lean`).

### Optional feed work (§4.3)

* Mixup/cutmix in place in the shims: blocked numpy, bit-identical. S. Worth up to +34% shim
  throughput; only verified ViT-S is shim-bound.
* uint8 wire plus PIL Rotate: −28% ms/step on ViT-S JAX under the stall. M. Matters far less if
  the fan removes the stall.

### Decisions for the user (before the verified ViT / ConvNeXt renders)

* ViT (`imagenet_parity.md` §5.5): clip on or off; LN eps 1e-6 with exact-erf GELU; cooldown and
  min_lr; EMA vs live scoring (DeiT scores live weights).
* ConvNeXt: exact-erf vs tanh GELU; the clip at 300 epochs.

### Probes before each verified launch

* A 15-minute run with the three-number split: graph, synthetic feed, real feed
  ([[host_draw_costed_at_wrong_batch]]).
* Median and mean both ([[imagenet_mean_is_a_memory_leak]]).
* It replaces §1's modelled hours with measured ones.

## 4. Feed performance (measured 2026-09-28, CPU only)

No GPU was used. The harness lives in the session scratchpad (`augbench/`). ⚠ tf.data's AutoGraph
re-reads each mapped function's source from its FILE by line number, so a benchmark that edits the
pipeline source in memory silently runs the original. The edited source must be written to a file
of its own.

### 4.1 The JAX trainers' in-process tf.data

The train iterator alone, one 512-image batch at a time, 40 batches after warm-up. Every run kept
about 28 of the 32 hardware threads busy. The CPU is a 16-core 5955WX, so the pipeline is bound
by physical cores.

| pipeline | as shipped | uint8 out¹ | no RandAugment | ms per 512 as shipped |
|---|---|---|---|---|
| ViT-S / ViT-B (identical) | 2,580 img/s | 3,145 | 4,332 | 200 |
| ConvNeXt-S | 2,551 | 3,067 | 4,503 | 204 |
| MNv4 `full` (m15) | 2,692 | 3,143 | 5,047 | 186 |
| R50 A2 (bilinear warps) | 3,808 | 4,754 | 4,827 | 133 |

¹ normalize, random erasing and the CHW transpose moved off the host (onto the device, as mixup
already is); the batch ships as uint8 HWC, a quarter of the bytes.

ViT-S knock-outs:
- erasing off: 2,756 (+7%)
- Rotate through PIL (`tf.numpy_function`): 2,917 (+13%)
- Rotate off: 3,124 (+21%)
- PIL Rotate and uint8 out together: 3,989 (+55%)
- decode, RandAugment and the float tail all off: ~24,000 (the source, shuffles and batching are not the limit)

Single-threaded costs per image:

| stage | ms/img |
|---|---|
| decode + crop + bicubic resize | 1.9 |
| RandAugment (2 ops at p 0.5) | 2.0 |
| normalize + transpose | 0.8 |
| erase | 0.8 |

By op, when RandAugment picks it:

| op | ms/img |
|---|---|
| Rotate (the emitted 16-tap bicubic) | 7.8–9.7 |
| Sharpness | 3.1 |
| Shear / Translate, each (the 4-tap 1-D warp) | 2.2 |
| Equalize | 1.5 |
| everything else | < 1 |

For comparison, TF's own bilinear warp is 1.6 ms, PIL's bicubic rotate 2.0 and PIL's shear 2.0.

Warp prototypes, all bit-identical to the emitted ops over 40 images and 7 angles or shears:

| prototype | result |
|---|---|
| uint8 gathers | Rotate 9.65 → 8.68 ms; shear unchanged |
| one fused 16-tap gather | 14.0 ms (slower) |

The emitted warp is bound by per-op overhead (about a hundred small float ops), not by gather
bytes. Only a fused kernel beats it, and PIL is one.

Where the JAX feed binds (compute rates from the 08-27 step probes, not re-measured):

| trainer | needs img/s | tf.data ceiling | outcome |
|---|---|---|---|
| R50 A2 | ~1,500 | 3,808 | compute-bound |
| ConvNeXt-S | ~980 | 2,551 | compute-bound |
| ConvNeXt-B | ~640 | ~2,550 | compute-bound |
| ViT-S | ~1,770 | 2,580 | compute-bound on paper |
| ViT-B | ~730 (701 ms/step) | 2,580 | compute-bound |
| MNv4 | ≥2,200 (the 3060 box's measured rate) | 2,692 | at the ceiling |

ViT-Ti's real run went feed-bound at 380 ms/step against a 200 ms feed and 124 ms of compute, so
the combined process loses ~180 ms somewhere: the stall, device transfer, or contention with the
trainer. Only a real run (the §2 smokes) tells.

### 4.2 The verified path's shims

N concurrent shim processes (`SHIM_SHARD=i/N`, batch 128, wire v2 with 1,000-class targets), each
read to the end of the pipe and timed over 40 batches:

| shim | 2 | 3 | 4 | 6 | 8 workers |
|---|---|---|---|---|---|
| ViT-S (mixup + cutmix on the host) | 1,154 | 1,422 | **1,571** | 1,637 | 1,393 |
| ViT-S, `SHIM_MIX=off` | | | 2,107 | | |
| MNv4 (no mixing) | | | **2,024** | | 1,638 |

* More producers do not help. 4 is at or near the peak for both nets and 8 is slower (each
  producer's own AUTOTUNE pool oversubscribes the 16 cores).
* The shims run ~40% below the in-process pipeline (1,571 vs 2,580 for the same augmentation).
  Host-side mixup/cutmix alone costs 34%. It is 25 ms per 128-image batch per producer idle, more
  under contention, and a blocked in-place form is 17 ms and bit-identical.
* ViT-S verified measured 310–323 ms/step median; the ViT-S shim ceiling is 326 ms per 512. That
  job is shim-bound.
* MNv4's shim ceiling is 253 ms per 512. The 3060 box ran the 100-epoch pair at 666 s/epoch
  (~266 ms/step), i.e. at that ceiling. The ~1,516 s/epoch figure this box's 210 h estimate came
  from is 2.3× slower than its own shims can feed. That gap is the stall or the pre-respawn
  producer aging (fixed 09-27), not augmentation cost. If it is gone, MNv4 500 ep verified here is
  ~90 h, not ~210.

### 4.3 Levers, ranked by measured effect (none built)

| # | lever | measured effect | cost | fidelity |
|---|---|---|---|---|
| L1 | Re-probe each job with respawn in place, before trusting any ETA | could halve MNv4's verified estimate | 15–30 min per job | — |
| L2 | Mixup/cutmix off the shim host: blocked in-place numpy (S), or on the device as the JAX trainer does (M) | up to +34% ViT/ConvNeXt shim throughput | S / M | bit-identical (in-place); the device form needs the mixup gates re-pinned |
| L3 | uint8 wire: normalize, erasing and transpose on the device | +22% in-process tf.data; 4× fewer host→device and pipe bytes | M (both paths; the verified trainer then normalizes in a small XLA program) | the same math, different place |
| L4 | Rotate through PIL (`tf.numpy_function`) | +13% on the bicubic nets | S | PIL is the reference, so exact by construction; the GIL cost inside the JAX trainer process is unmeasured |
| L5 | Faster Shear/Translate/Sharpness | ≤ ~8% combined, from the op table | M | — |
| — | More shim workers | negative past 4 | — | — |
| — | tf.data `deterministic=False` in the JAX trainers | none (2,568 vs 2,576) | — | — |
| — | File-level sharding for the shims | small: reading the whole stream costs ~0.2 cores per producer | — | — |

The box has 4 of the 5955WX's 8 memory channels populated. The periodic stall tracks memory
traffic ([[imagenet_periodic_invoke_stall]]), and filling channels 0–3 is the hardware lever for
it.

## 5. Batch-1 smokes on the four cards (2026-09-28/29)

Each trainer ran under `timeout` with no checkpoints. The rates below are 100-step windows taken
from the trainer's own step lines, time-stamped as they printed. Scratch logs are in the session
scratchpad (`smoke/`, `probe/`).

| trainer | clean window | overall | stall | 300/500-ep ETA here |
|---|---|---|---|---|
| R50 A2 (`a2accum`) | 1,405 ms / optimizer step, flat over 5 windows | 1,405 | none in 9 min | **~73 h** |
| ViT-S (`default`) | 300 ms/step (= compute) | 656 ms/step | ~40–85 s pauses every 2–3 min, ~54% of wall | ~137 h (63 h clean) |
| MNv4 `full`, warm cache | — (8 micro-batches per step: one line per ~8 min) | 3.3 → 4.7 → 5.1 s / step, worsening | ~65% of wall against the ~1.6 s tf.data-bound rate | **~210 h** (3060 box ~81 h) |

All three compile and train; A2 shows the BCE threshold and no EMA (loss 0.63 → 0.015 in 500
steps), and MNv4 `full` scores live weights.

The stall is not gone on the JAX path: loader respawn lives in the verified shims, and the JAX
trainers run tf.data in-process. It scales with host augmentation traffic: A2's cheap bilinear
pipeline at ~1,500 img/s never paused, while ViT-S (~1,700 img/s through the bicubic pipeline) and
MNv4 (feed-bound, ~2,500 img/s) pause hard.

Perf-only ViT-S probes (scratch copies; erasing done in pixel space, so not recipe-exact):

| variant | overall ms/step | vs shipped | stall windows |
|---|---|---|---|
| shipped | 656 | — | 5 of 8 after onset |
| uint8 wire, normalize/transpose on device | 569 | −13% | 7 of 15 |
| uint8 wire + PIL Rotate | 474 | −28% | 4 of 17 |

Cutting host bytes and CPU per image thins the stall but does not remove it; the clean rate
(290–300 ms) is compute either way.

### 5.1 The stall is the DIMMs' thermal throttle (2026-09-29)

With `jc42` loaded, the four DIMM sensors show up in `sensors`: i2c 1-001c to 1-001f, high 80.0 °C,
hysteresis 77.0, critical 95. The shipped ViT-S trainer ran 15 minutes with every DIMM and the CPU
(k10temp) logged every 2 s. There was a 30 s idle baseline before and a 2 min cool-down after.

| phase | DIMM 1-001d (hottest) | trainer |
|---|---|---|
| idle | 56 °C | — |
| 0–200 s | climbs to 79.7 °C | clean, 300 ms/step |
| ~200 s | reaches 80 °C | first pause |
| after | oscillates 77–80 °C | pauses at 80, resumes near 77 |

* The oscillation sits exactly between the sensor's high and hysteresis marks.
* The other three DIMMs peak at 74–76 °C. The peak reading was 80.2 °C.
* The CPU sits at 95.7 °C Tctl under load, the 5955WX's limit. That does not pause the trainer,
  but it caps the tf.data rate.

The fix is airflow across the DIMMs. The re-test is the same smoke with the same logging.
Baseline: 54% of wall stalled, 656 ms/step overall, 300 clean.
