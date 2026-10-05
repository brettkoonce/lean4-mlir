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

Measured 2026-10-05 on this box with the DIMM fans in (§5.2): every row is the job's own trainer on
all four cards, no stall in any window. JAX = 100-step windows from the trainer's own step lines;
verified = the §3a smoke's MEAN over steps 201–600. Every job takes all four cards, so the queue is
sequential. Per-epoch evals (a few hours per run) are not included.

| batch | runs (hours) | total | days 24/7 |
|---|---|---|---|
| 1, JAX | R50 A2 76 · MNv4 `full` 500 ep ~100–105 · ViT-S 72 | ~250 | ~10.4 |
| 2, verified | R50 A2 93 · MNv4 500 ep 100 · ViT-S 79 | ~272 | ~11.3 |
| 3, JAX | R50 A1 153 · ViT-B 148 · ConvNeXt-S 127 | ~428 | ~17.8 |
| 4, verified | R50 A1 185 · ViT-B 185 · ConvNeXt-S 156 | ~526 | ~21.9 |
| 5 | ConvNeXt-B JAX 167, then verified 216 | ~383 | ~16 |
| total | | ~1,860 | ~77.5 |

The rates behind each row (`runs/2026-10-04-dimm-fan-thermal/`):

| run | step | steps/epoch × epochs | ms/step | bound |
|---|---|---|---|---|
| A2 JAX | optimizer (512 × 4) | 625 × 300 | 1,467 | compute, flat |
| A1 JAX | optimizer (512 × 4) | 625 × 600 | 1,469 | compute, flat |
| ViT-S JAX | 512 | 2,502 × 300 | 344 | tf.data, flat 343–346 |
| ViT-B JAX | 512 | 2,502 × 300 | 708 | compute, flat |
| ConvNeXt-S JAX | 256 | 5,004 × 300 | 304 | compute, flat |
| ConvNeXt-B JAX | 256 | 5,004 × 300 | 401 | compute, flat |
| MNv4 `full` JAX | optimizer (512 × 8) | 312 × 500 | 2,543 → 2,386 → 2,303 | still falling at 20 min |
| ViT-S verified | 512 | 2,502 × 300 | 379 (med 381, min 302); 609 before the shim fix | shim: 79 ms starved (was 305) |
| ViT-B verified | 512 | 2,502 × 300 | 887 (med 884, min 839) | compute |
| A2 verified | micro (4 × 128) | 2,500 × 300 | 446 (med 446, min 415) | compute |
| A1 verified | micro (4 × 128) | 2,500 × 600 | 444 (med 444, min 412) | compute |
| ConvNeXt-S verified | 256 | 5,004 × 300 | 373 (med 372, min 355) | compute |
| ConvNeXt-B verified | 256 | 5,004 × 300 | 518 (med 518, min 507) | compute |
| MNv4 verified | micro (4 × 128) | 2,496 × 500 | 290 (med 290, min 233) on the uint8 wire; 318 / 329 before | shim CPU: 57 ms starved |

Against the 09-29 table: ConvNeXt-B verified is 37 h shorter (253 modelled), ConvNeXt-S JAX 17 h
longer (110 was a compute probe), and MNv4 verified is the one big miss left (100 against ~90, after §4.5). ViT-S
verified was 127 h until the shim fix in §4.4 (in-place mixing, a writer thread); it is 79 now.

## 2. Where things stand (2026-10-05)

* **The DIMM throttle is gone** (§5.2). The DIMMs were re-slotted per the board manual and two fans
  put on them; across 16 runs the hottest DIMM peaked at 57.9 °C (throttle at 80) and no window
  stalled. ViT-S JAX: 344 ms/step flat, against 656 overall on 09-29.
* **Every job has a measured rate** (§1), both paths. The §3a smokes are done for all seven verified
  confs. The `ETA=` strings in the confs carry the §1 numbers.
* **The R50 drop-path renders were broken and are fixed** (c84fad74). All 12
  `resnet50in_…drop…` train steps (A2/A1, EMA and not) returned 16 values short — the masks went in
  and never came out, while the driver copies the whole output blob back (`#out = #in − 2`). The
  PJRT shim refuses that (G4: 755 outputs, 771 destinations), but only on a GPU, and nothing had
  run them; the 08-27 `arity_check.py` encoded the wrong rule and printed green. Fixed in
  `ResNet50RenderB.lean` (masks returned last, as MobileNetV4 does), re-rendered (12 files, two lines
  each), and guarded by the new `scripts/gates/train_step_arity.py` (all 213 train steps; in
  proofs.yml and in the six drop-render confs' PRECHECK). `lake build`, `lake build Certs` and
  `regen_verified_mlir.sh check` are green on the fix.
* **The ViT shim fix landed** (§4.4, f5b6853f): mixup/cutmix in place and the pipe write on its
  own thread. Byte-identical streams (SHIM_HASH and the wire, old vs new, five shims); ViT-S verified
  609 → 379 ms/step (127 → 79 h), MNv4 329 → 318.
* **The uint8 wire landed** (§4.5): a shim whose train images are uint8 before the
  normalize sends them as uint8 HWC and the trainer normalizes in C, bit-identical
  (`tests/u8_wire_tie.sh`). MNv4 verified 318 → 290 ms/step (110 → 100 h). Still producer
  CPU-bound (57 ms starved): what is left is TF's per-image augmentation itself (§4.3 L4/L5).
* All the JAX jobs can launch.

## 3. Software work list

### Batch 1, JAX A2 / MNv4 `full` / ViT-S — done (`b84da996`)

(as committed: `TrainConfig.bceTargetThresh`, A2/A1 per RSB Table 2, MNv4 `full` without EMA,
confs `r50-a2accum-jax-4gpu`, `mnv4-full-jax-4gpu`, `vits-default-jax-4gpu`)

### Batches 2–5 — done (2026-09-29, uncommitted)

| net | verified conf | JAX conf | render / code |
|---|---|---|---|
| ViT-S | `vits-default-emabf16-4gpu` | `vits-default-jax-4gpu` (batch 1) | `vitsin_emadp128x4wxclipdropbf16`; driver `vitInit` + `emaDecay := 0.99996` |
| ViT-B | `vitb-default-emabf16-4gpu` (MEM 0.97) | `vitb-accum-jax-4gpu` | `vitbin_emadp128x4wxclipdropbf16`, peak 13.19 of 15.11 GiB; driver as S |
| R50 A2 | `r50-a2-bf16-4gpu` | `r50-a2accum-jax-4gpu` (batch 1) | `resnet50in_lambaccdp4x128wxclipdropbce{,bf16}` (no EMA); `a2` recipe + `resnet50ImagenetA2Verified` |
| R50 A1 | `r50-a1-bf16-4gpu` | `r50-a1-jax-4gpu` | `…bcewd001{,bf16}` (no EMA); A1 shim now smooths 0.1 |
| MNv4 paper | `mnv4-full-4gpu` | `mnv4-full-jax-4gpu` (batch 1) | `mnv4in_acc{dp,}8x128wxdropdowd01bf16`: UIB drop-path on the 18 skip blocks, `mnv4ImagenetFullVerified` (m15 shim, dropout 0.2, keeps 1 − 0.075·i/20) |
| ConvNeXt-S | `cnxs-default-emabf16-4gpu` | `cnxs-default-jax-4gpu` | `convnextsin_ema{dp,}wxclipdropbf16` at 64, peak 6.80 GiB; `convnextsin_fwd` at 64, `_fwd_s288`; `cnxInit` both paths |
| ConvNeXt-B | `cnxb-default-emabf16-4gpu` | `cnxb-default-jax-4gpu` | as S; peak **9.53 GiB of the 11.68 default**, so no accumulation render |

Shared pieces:
* **The BCE target transform rides the shim.** The BCE renders take `%onehot` as given, so the
  0.2 threshold (and A1's ε 0.1) is applied in the generated shim's `_emit` after mixing
  (`Jax/Codegen.lean`). Only the a1/a2accum shims changed. `scripts/gates/bce_target_gate.py`
  checks it against the default shim at one seed; it is bit-exact and has a control.
* The R50 driver refuses an `a1`/`a2` recipe whose variant has the wrong decay or an EMA. The MNv4
  driver refuses `full` without a `drop` variant and `default` with one.
* `resnet50in_fwd_eval_s288` scores A2/A1 at timm's 288 / 1.0.
* `mnv4-dp-check` feeds the drop masks. `vit-ema-drop-render` has `vitsin`, `vitbin`,
  `convnextsin` and `convnextbin` arms.
* The lakefile rows for ViT-S/B and ConvNeXt-S/B now name the emabf16 jobs, as does the book's
  side-quest job table. The g512 and 4 × 32 confs stay as siblings.

timm parity, every side-quest net, on shared weights (CPU, in CI's jax.yml `timm parity` job):
* `resnet_timm_parity.py`: the JAX references AND the verified forwards against timm, plus the
  drop-path ramp on both paths.
  * `resnet50`: train 224/160, eval 224/288. JAX ≤ 2.2e-5, verified ≤ 5.5e-5 of max |logit|.
  * `resnet34`, the main-track anchor: JAX 4.0e-6, verified 5.6e-6.
  * `--break` (one BN's γ/β swapped) turns all 12 logit checks red at 0.16–0.38.
  * CI runs the JAX half, because the verified half needs IREE.
* `vit_timm_parity.py` covers ViT-S and ViT-B (deit_small/base), and now the verified renders too
  (`vitin/vitsin/vitbin_fwd` through IREE).
  * JAX 6.9e-7 / 1.1e-6 / 1.7e-6.
  * Verified 8.0e-7 / 1.1e-6 / 2.1e-6.
* `cnx_timm_parity.py` (new): ConvNeXt-T/S/B against timm at tanh GELU, 224 and 288, ≤ 1.1e-6 plus
  the ramp. Its controls, erf GELU (1.4e-4) and two LayerScales swapped (1.1e-2), both go red.
  `convnext_forward_tie.py` (verified ↔ JAX) gained `convnextbin` and now runs the reference at f32.
  It had run the trainers' bf16, passing at 1.4e-3 (B at 1.86e-3) against a 2e-3 bar, i.e. bf16
  rounding. At f32: T 4.8e-7, S 7.2e-7, B 8.0e-7, tolerance 1e-5, `--break` 8.8e-1.

Not done:
* The MNv4 drop-path known-answer and misplacement gates (`droppath-tie` gates A/B). They need a
  `mnv4in_drop_fwd` render and a GPU. Structurally, all 18 masks are used once forward on the
  branch and once backward on the branch cotangent, with the skip fan-in unmasked.
* MNv4 training at 256.

### 3a. GPU smokes owed before each launch (the user starts these)

A capped step window with the job's own env, no checkpoint written:

    ONCE=1 LEAN_MLIR_MAX_STEPS=400 LEAN_MLIR_PROBE_WARM=200 scripts/supervise.sh <job>

in this order: `vits-default-emabf16-4gpu`, `vitb-default-emabf16-4gpu`, `r50-a2-bf16-4gpu`,
`mnv4-full-4gpu`, `cnxs-default-emabf16-4gpu`, `cnxb-default-emabf16-4gpu`. Each gives median and
mean ms/step ([[imagenet_mean_is_a_memory_leak]]), replacing §1's modelled hours. Plus:

    DP_BATCH=128 DP_VARIANT=acc8x128wxdropdowd01bf16 DP_VARIANT_DP=accdp8x128wxdropdowd01bf16 \
      PJRT_REPLICAS=4 .lake/build/bin/mnv4-dp-check

### Optional feed work (§4.3)

* Mixup/cutmix in place in the shims: blocked numpy, bit-identical. S. Worth up to +34% shim
  throughput; only verified ViT-S is shim-bound.
* uint8 wire plus PIL Rotate: −28% ms/step on ViT-S JAX under the stall. M. Matters far less if
  the fan removes the stall.

### Decisions for the user

The S/B renders took ViT-Ti's and ConvNeXt-T's answers (clip on, EMA scored, their GELU form and LN ε). A different
answer is a re-render plus a JAX re-emit, not new machinery.

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

### 4.4 The bench at the jobs' shape, and the ViT shim fix (2026-10-05)

`runs/2026-10-05-shim-feed-bench/`: the producers spawned with `spawnShim`'s env and read in the
trainer's round-robin (`rr`) or one reader thread each (`free`). Before the fix:

| config | img/s | ms per 512 |
|---|---|---|
| ViT 512 × 4, rr (the job) | 831 | 616 (job: 609) |
| ViT 512 × 4, free | 1,220 | 420 |
| ViT 512 × 4, rr, mix off | 1,113 | 460 |
| ViT 128 × 4, rr | 822 | 623 |
| MNv4 512 × 4, rr (the job) | 1,758 | 291 (job: 329) |

The bench reproduces the jobs, and batch size does not matter (128 ≈ 512). On ViT, mixing and the
round-robin were each about a third: `_emit` mixed on the producer's main thread (four fresh
308 MB arrays per batch) and then blocked in `write()` on a 308 MB record against a 64 KB pipe.
MNv4 does not mix; it is plain CPU (the box saturated, rr ≈ free). Every config reads 55–87% of
§4.2's 09-28 numbers on the same shim; clocks and memory are normal, and the 09-28 harness is gone.

The fix (`Jax/Codegen.lean`, all 37 shims re-emitted): `_mix_rows` mixes in place a block of row
pairs at a time — each element the same float32 expression as before — and `_writer` puts the
pipe write on its own thread behind a one-batch queue, writing the arrays' buffers (`memoryview`)
instead of `tobytes()` copies. `identity.py`: old vs new SHIM_HASH digests and streamed bytes are
equal on ViT (B = 63 and 64), ConvNeXt-S, R50 A2 (the BCE threshold) and MNv4, and mix on vs off
differs (the control). `mixup_gate.py`'s known-answer checks and `bce_target_gate.py` pass; the
mixup gate's two pinned "inert when off" digests fail on the old shim and the new alike (stale
since an earlier change; the scripts-audit open item).

| after | img/s | ms per 512 |
|---|---|---|
| ViT 512 × 4, rr | 1,410 | 363 |
| ViT 512 × 4, free | 1,381 | 371 |
| MNv4 512 × 4, rr | 1,862 | 275 |

On the GPUs: ViT-S verified 609 → 379 ms/step (starved 305 → 79), MNv4 verified 329 → 318.

### 4.5 The uint8 wire (2026-10-05)

`runs/2026-10-05-uint8-wire/`. Wire v5/v6 = v3/v4 with uint8 HWC images and the shim's six float32
normalize constants after the preamble. The trainer asks for it (`SHIM_U8=1`, on by default,
`LEAN_MLIR_SHIM_U8=0` off) and each shim grants it only where it is exact:
- at generation time, for a recipe with no erasing and no mixing (both act after the normalize);
- at trace time, when the augmented image is uint8 — `_randaugment`/`_autoaugment` end in
  `tf.cast(<uint8>, tf.float32)`, and `_pp` takes that Cast's uint8 input from the graph.
  MNv4/MNv2/EfficientNet-B0 grant it; R34's crop + flip ends in float and stays on v3; ViT and
  ConvNeXt (mixing, erasing) stay on v4. Validation is never uint8.

`emitDataLoading` gains `u8Wire`, set only by the shim, so the JAX reference trainers re-emit
byte-identical. The consumer reads B·flat bytes into the front of the main-thread float buffer and
`lean_mlir_u8_norm` widens it in place, last image first (`((float)u − mean[c]) / std[c]`, the
shim's own two float32 ops, then its transpose), AVX2 behind a runtime check (15 vs 52 ms per 512,
bit-identical to the generic loop over every byte value, `normbench3.c`).

Gates: `tests/u8_wire_tie.sh` — float wire twice (control, 0 bytes) and uint8 vs float (0 bytes
of the 155 MB trained state after 16 micro-steps), with a vacuity check that the uint8 run was
granted the wire; it caught the first version, whose trace-time test looked for a uint8 tensor and
found RandAugment's float32 cast. The float wire is unchanged: `identity_float.txt`.

MNv4 verified: mean 318 → 290 ms/step (starved 109 → 57 ms), ~100 h for 500 epochs.

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

### 5.2 The re-test with the DIMM fans in (2026-10-05)

The DIMMs were moved to the slots the board manual gives for four modules (their sensors now read at
i2c 1-0018/19/1c/1d) and two fans were tied on over them. Same smoke as §5.1: 15 min of the shipped
ViT-S JAX trainer, temps every 2 s, 20 s idle before and 60 s cool-down after.

| | 09-29 | 10-05 |
|---|---|---|
| idle, hottest DIMM | 56 °C | 38 °C |
| under load, hottest DIMM | 80.2 °C, oscillating 77–80 | 56.8 °C |
| CPU Tctl peak | 95.7 °C | 89.0 °C |
| ms/step | 656 overall, 54% stalled | 344, all 24 windows 343–346 |

The two DIMMs nearer the fans run 15–20 °C cooler than the other two. All 16 runs (§1's seven JAX and seven
verified, plus the two R50 smokes that failed before the fix) peaked at 57.9 °C. ViT-S's 344 is
above §5's "300 clean": those clean windows were served from a queue the producers filled during
the pauses, so 344 is the in-process tf.data rate, not a new loss.

