# ImageNet ms/step on 4× RTX 4060 Ti, f32 and bf16 — 2026-09-10

Re-measurement behind every `ETA=` in `scripts/jobs/*-4gpu.conf`, i.e. every estimate
`lake run imagenet` prints. The numbers on file predated three things and were wrong
by up to 1.8× because of them.

## Why they needed re-running

1. **The tf.data determinism default flipped OFF** (`4a0a2781`, 2026-09-08). Every ETA
   on file was measured with it ON. On the box where it was characterised, a live
   300-epoch ViT job went 830 → 577 s/epoch, 69 h → 48 h. Determinism ON keeps the
   *median* within ~5 ms of the compute floor while the *mean* balloons (333 vs 203),
   so this change is nearly invisible to a median-ranked probe and worth ~1.5× in
   wall clock.
2. **The box lost two cards** (2026-09-08). Several confs still quoted a 4× RTX 3060
   box that is not this one.
3. **The probe script itself was wrong.** See below.

## What was wrong with the old probe, and what this run does instead

`scripts/bf16_probe_4gpu.sh` (now marked superseded, kept for provenance) had two
independent defects:

* **It timed a burst, not a steady state.** `LEAN_MLIR_MAX_STEPS=40` with the clock
  starting at step 8 measures a window the producers pre-filled during compile and the
  val drain — `SHIM PREFETCH` holds one read in flight per handle. ViT reads 159 ms/step
  that way against a 375 ms/step steady state. This run uses `LEAN_MLIR_PROBE_WARM=200`
  with `MAX_STEPS=600`, so the clock starts long after the queue is drained.
* **It measured graphs the jobs do not train.** Its `mnv2` rows were `adamdp64`; the job
  trains `rmsdp64` (RMSProp is MobileNetV2's reference optimizer). Its `enet` rows were
  the light `rmsdp64`; the job bakes `emarmsdp64dropdo`, which nearly doubles the step.
  It had no rows at all for r50a3, vitema, ConvNeXt-S/B or ViT-S/B.

**And one defect this run found the hard way.** A first pass here swept every net at a
uniform `SHIM_WORKERS=8`. The job confs do not share a worker count — ConvNeXt-T/S/B,
MobileNetV2/V4, ViT-Ti and RSB-A3 run **4** producers, the ResNets and B0 run 8 — and
ConvNeXt's own precheck measures 8 as *worse* than 4 (212.5 against 191 ms/step on -S,
floor 190). That pass was discarded. Every row below carries its own job's worker count,
listed in the table.

## Reading the table: mean, not median

`ms/step` in the ETA is the **mean**. Shim starvation is bursty — the trainer eats the
prefetch queue, then stalls — so the median can sit at the compute floor while the p90
is five times higher. `runs/probe3060.tsv` has the clean example: EfficientNet bf16 fed
is med 99 / mean 193 / p90 502. On the median that graph looks 1.66× faster than f32;
on the mean it is not faster at all. **Mean is what sets wall clock.** Rows where
mean > 1.15 × med are flagged; on those the ETA is a feed result, not a graph result,
and it moves when `SHIM_WORKERS` or the determinism default moves.

## The ETA formula

    hours = (steps_per_epoch × ms_step / 1000 + 37.5) × epochs / 3600

37.5 s/epoch is eval + checkpoint, measured on ResNet-34, under 2% of every row. The
one-time ~30 GB val drain is not in it: paid once, not per epoch.

⚠ **The old ETAs did not follow from their own ms/step.** `r34-default-4gpu.conf` read
"~37 h … (386 ms/step, 90 ep)", but 386 ms/step over 5004 steps/epoch is 49 h; 37 h
implies 295 ms/step. Two measurements from different places sat in one string. Every
number below is computed from the measurement beside it, so the arithmetic checks.

## Reproducing

    PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
    SHIM_PYTHON="$PWD/.venv/bin/python3" DEVS=0,1,2,3 WARM=200 STEPS=600 ARMS=fed \
    CKPT_TAG=probe4060 ROWS="$(cat runs/2026-09-10-bf16-probe-4060ti/rows.txt)" \
      bash scripts/bf16_probe_3060.sh runs/<date>/fed.tsv

    scripts/probe_to_eta.py runs/<date>/fed.tsv --box "4x 4060 Ti"

`rows.txt` in this directory is the exact row table used, worker counts included. The
second command prints the `ETA=` string for each conf and refuses to write one when the
conf's variant and the probed variant differ.

## Results

| net | variant | prec | workers | med | mean | p90 | steps/ep | note |
|---|---|---|---|---|---|---|---|---|
| cnx | `adamdpwxclipdropbf16` | bf16 | 4 | 120 | **133** | 183 | 10009 |  |
| cnx | `adamdpwxclipdrop` | f32 | 4 | 202 | **201** | 204 | 10009 |  |
| cnxb | `adamdpwxclipdropbf16` | bf16 | 4 | 296 | **295** | 298 | 10009 |  |
| cnxb | `adamdpwxclipdrop` | f32 | 4 | 514 | **513** | 515 | 10009 |  |
| cnxs | `adamdpwxclipdropbf16` | bf16 | 4 | 188 | **188** | 191 | 10009 |  |
| cnxs | `adamdpwxclipdrop` | f32 | 4 | 339 | **339** | 341 | 10009 |  |
| enetema | `emarmsdp64dropdobf16` | bf16 | 8 | 102 | **204** | 893 | 5004 | **starving** |
| enetema | `emarmsdp64dropdo` | f32 | 8 | 186 | **200** | 272 | 5004 |  |
| mnv2 | `rmsdp64bf16` | bf16 | 4 | 113 | **112** | 121 | 5004 |  |
| mnv2 | `rmsdp64` | f32 | 4 | 172 | **172** | 181 | 5004 |  |
| mnv4 | `adamdp64bf16` | bf16 | 4 | 157 | **253** | 546 | 5004 | **starving** |
| mnv4 | `adamdp64` | f32 | 4 | 146 | **253** | 563 | 5004 | **starving** |
| r34 | `momdp64bf16` | bf16 | 8 | 142 | **142** | 145 | 5004 |  |
| r34 | `momdp64` | f32 | 8 | 215 | **214** | 217 | 5004 |  |
| r50 | `momdp64bf16` | bf16 | 8 | 222 | **222** | 223 | 5004 |  |
| r50 | `momdp64` | f32 | 8 | 355 | **355** | 357 | 5004 |  |
| r50a3 | `lambaccdp8x64wxclipbcebf16` | bf16 | 4 | 144 | **175** | 281 | 5000 | **starving** |
| r50a3 | `lambaccdp8x64wxclipbce` | f32 | 4 | 209 | **208** | 210 | 5000 |  |
| vit | `adamdp128x4wxclipdropbf16` | bf16 | 4 | 141 | **876** | 3219 | 2502 | **starving** |
| vit | `adamdp128x4wxclipdrop` | f32 | 4 | 220 | **894** | 2974 | 2502 | **starving** |
| vitb | `adamdp128x4wxclipdropbf16` | bf16 | 8 | 797 | **795** | 803 | 2502 |  |
| vitb | `adamdp128x4wxclipdrop` | f32 | 8 | 1356 | **1353** | 1361 | 2502 |  |
| vits | `adamdp128x4wxclipdropbf16` | bf16 | 8 | 287 | **644** | 1087 | 2502 | **starving** |
| vits | `adamdp128x4wxclipdrop` | f32 | 8 | 501 | **502** | 507 | 2502 |  |

`mean` is the number that schedules. Six rows are **starving** — their mean is well above
their median, so what they measure is the data pipeline, not the graph.

### The starvation is predictable, and it is a supply limit

Sort every row by the images per second it demands divided by its producer count and the
split is clean: above ~300 img/s per producer everything starves, below ~250 nothing does.

| row | global batch | floor | img/s needed | workers | per worker | starving |
|---|---|---|---|---|---|---|
| vit bf16 | 512 | 141 ms | 3631 | 4 | 908 | yes |
| vit f32 | 512 | 220 ms | 2327 | 4 | 582 | yes |
| mnv4 f32 | 256 | 146 ms | 1753 | 4 | 438 | yes |
| B0 bf16 | 256 | 102 ms | 2510 | 8 | 314 | yes |
| mnv2 f32 | 256 | 172 ms | 1488 | 4 | 372 | no |
| r34 f32 | 256 | 215 ms | 1191 | 8 | 149 | no |
| ConvNeXt-B | 128 | 514 ms | 249 | 4 | 62 | no |

One producer sustains roughly 300-400 images/s at 224 px. **This is why bf16 can make a job
slower**: a faster graph raises demand against fixed supply. MobileNetV4's two arms land on an
identical 253 ms mean despite floors of 140 (f32) and 80 (bf16), so bf16 buys nothing there,
and on the median it even reads slower.

### ViT-Tiny: a worker sweep, and the floor

| arm | workers | med | mean | p90 | 300-ep |
|---|---|---|---|---|---|
| f32 fed | 4 (old conf) | 220 | 894 | 2974 | 190 h |
| f32 fed | 8 | 213 | **520** | 1616 | **112 h** |
| f32 fed | 12 | 222 | 593 | 1293 | 127 h |
| f32 fed | 16 | 251 | 570 | 1023 | 122 h |
| f32 **synth** | — | 217 | **217** | 219 | **48 h** |
| bf16 fed | 12 | 149 | 581 | 1930 | 124 h |
| bf16 **synth** | — | 141 | **140** | 143 | **32 h** |

The synth arm has p90 == median: the graph is perfectly steady and every bit of the gap is
the feed. 8 producers is the knee — 12 and 16 do not beat it — and the conf moved 4 → 8 for
a 78 h saving. ⛔ **But fed is still 2.4x the floor at the knee**, so ~64 h of ViT-Ti's 112 is
pipeline overhead that no worker count recovered. The p90 falling (2974 → 1023) while the mean
plateaus says the stalls are periodic, not steady starvation. Open.

⚠ The 2026-08-31 sweep that set `SHIM_WORKERS=4` is superseded: it ranked on the median (which
sits at this net's compute floor at every worker count and cannot see starvation) and ran with
tf.data determinism ON, which capped each producer at ~1/5 throughput so extra producers really
did nothing then. Both premises changed on 2026-09-08.

## What was written back

Fourteen `scripts/jobs/*-4gpu.conf` ETA strings, each computed from the mean beside it, each
stamped with this directory. Two of them (`r50-2018-4gpu`, `r50-a3-wxclip-bf16-4gpu`) had no
ETA line at all and printed "not on file". Estimates over 100 h carry a day figure.
`vit-default-4gpu.conf` also moved `SHIM_WORKERS` 4 → 8, precheck included.

Not written: `r50-a3-4gpu` (trains `lambaccdp8x64bce`, not the `wxclip` variant measured here)
and `vit-default-emabf16-4gpu` (trains `emadp128x4wxclipdropbf16`, not in this row table).
Both still print "not on file" rather than a number from a different graph.

