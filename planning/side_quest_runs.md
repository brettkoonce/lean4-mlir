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

Clean compute from the 2026-08/09 probes on this box, ±20% until each job's re-probe (§3). Every
job takes all four cards, so the queue is sequential.

| batch | runs | clean hours | days 24/7 |
|---|---|---|---|
| 1, JAX | R50 A2 ~71 · MNv4 `full` 500 ep ~95 · ViT-S ~60–79 | ~225–245 | ~10 |
| 2, verified | R50 A2 ~95 · MNv4 500 ep ~210 · ViT-S ~70 | ~375 | ~15.5 |
| 3, JAX | R50 A1 ~143 · ViT-B ~80 · ConvNeXt-S ~120 | ~345 | ~14.5 |
| 4, verified | R50 A1 ~190 · ViT-B ~175 · ConvNeXt-S ~160 | ~525 | ~22 |
| 5 | ConvNeXt-B JAX ~183, then verified ~253 | ~435 | ~18 |
| total | | ~1,900 | ~80 |

The box's periodic stall ([[imagenet_periodic_invoke_stall]]) took 16–47% of wall per net on
09-22; loader respawn landed 09-27 and may have changed that, so the batch-1 smokes measure it.

## 2. Code per batch

### Batch 1 (JAX) — prepared 2026-09-28, uncommitted

| item | where | state |
|---|---|---|
| BCE target threshold (timm `--bce-target-thresh`), a `TrainConfig` field emitted after smoothing | `LeanMlir/Types.lean` `bceTargetThresh`, `jax/Jax/Codegen.lean` BCE branch | done |
| A2/A1 per RSB Table 2: no EMA, threshold 0.2; A1 adds label smoothing 0.1 | `jax/MainResnet50Imagenet.lean` `a2-accum`, `a1` | done |
| MNv4 `full` scores live weights (paper: no EMA) | `jax/MainMobilenetV4Imagenet.lean` | done |
| `jax/generated/` re-emitted: only the a2accum, a1 and MNv4 full trainers move | `scripts/regen_jax_generated.sh` | done, synced |
| confs `r50-a2accum-jax-4gpu`, `mnv4-full-jax-4gpu`, `vits-default-jax-4gpu`; all three DRY_RUN prechecks pass | `scripts/jobs/` | done |
| 15-min smoke of each trainer on the four cards (compile, first steps, windowed ms/step with the real feed) | — | owed; needs the user |

Notes:
* RSB Table 2 lists no gradient clip; the trainers' clip is timm LAMB's own `max_grad_norm=1.0`,
  kept.
* The A3 recipes (`short`, `rsb-faithful`) are untouched: their runs are in the book.
* MNv4 `full` still differs from the paper in RandAugment p 0.5 (paper 0.7) and m15 over the
  shim's `_AA_MAX` of 10.

### Batch 2 (verified, A2 / MNv4 / ViT-S)

| item | where | size |
|---|---|---|
| A2: no-EMA render `resnet50in_lambaccdp4x128wxclipdropbce{,bf16}`; target threshold in the host label build | `ResNet50RenderB.lean`, trainer label path | M |
| A2: verified conf, lakefile row, `resnet50in_fwd_eval_s288` (timm scores at 288/1.0) | `scripts/jobs/`, `lakefile.lean`, renders | S |
| A2: R50 timm parity gate | `scripts/parity/` | M |
| MNv4: no-EMA render (DP + 1-replica peer), render-guard/MANIFEST/test rows | `MobileNetV4RenderB.lean`, `proofs.yml` | S |
| MNv4: m15 shim selection, classifier dropout 0.2 knob, UIB drop-path 0.075 on the verified side | driver, `NetsCore`, renderer | S, S, M–L |
| ViT-S: `vitInit := true`, EMA+bf16 render, emaDecay 0.99996, layout-gate arm, emabf16 conf | `MainViTSImagenet.lean`, `ViTRenderB.lean`, tests, conf | S each |
| ViT-S: §5.5 decisions (clip, LN eps + erf GELU, cooldown, EMA vs live scoring; DeiT scores live) | `imagenet_parity.md` §5.5 | decide |

### Batch 3 (JAX, A1 / ViT-B / ConvNeXt-S)

| item | where | size |
|---|---|---|
| A1 conf (trainer done in batch 1) | `scripts/jobs/` | S |
| ViT-B JAX conf; 4-GPU memory/speed probe | `scripts/jobs/` | S + probe |
| ConvNeXt-S `cnxInit := true`, JAX conf | `jax/MainConvNeXtSImagenet.lean` | S |

### Batch 4 (verified, A1 / ViT-B / ConvNeXt-S)

| item | where | size |
|---|---|---|
| A1: no-EMA wd 0.01 render, smoothing 0.1 before the threshold, conf; stale 8×64 driver comment | renders, driver | M |
| ViT-B: the ViT-S list, plus EMA+bf16 memory probe at `LEAN_MLIR_MEM_FRACTION=0.97` | as ViT-S | S + probe |
| ConvNeXt-S: `cnxInit`, batch-64 EMA+bf16 render, `_fwd_s288`, conf, layout gate | `ConvNeXtRenderB.lean`, driver | S each |

### Batch 5 (ConvNeXt-B)

| item | where | size |
|---|---|---|
| everything ConvNeXt-S needs, plus memory: 9.73 GiB at 32/replica, 64 likely overflows and 0.97 OOMs this net — may need an accumulation render | `ConvNeXtRenderB.lean` | M |
| proofs: whole-net ties pinned at T, B's 1024-wide head uncovered | `ConvNeXtStepTieGB.lean` | proof tier only |

## 3. Probes owed before each batch

* Batch 1: the three 15-min smokes (§2); windowed ms/step and stall share from their step lines.
* Every batch: graph / synthetic-feed / real-feed split for its verified jobs
  ([[host_draw_costed_at_wrong_batch]]), and median and mean both
  ([[imagenet_mean_is_a_memory_leak]]).

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
| ViT-B | ~2,600–2,800 | 2,580 | at the ceiling |
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
