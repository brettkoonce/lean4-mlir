# ImageNet ms/step on 4× RTX 4060 Ti, re-measured after the two feed bugs — 2026-09-11

Every `ETA=` string in `scripts/jobs/*-4gpu.conf` came from
`runs/2026-09-10-bf16-probe-4060ti`, and two bugs sat under every row of that table:

1. **The trainer retained host memory** — mimalloc lazily freeing 308 MB batch buffers that a
   pool thread allocated and the main thread dropped — until the box ran at the reclaim
   watermark and the mean step time became a memory-pressure measurement.
   Fixed: `VerifiedTrain.readInto`, main-thread buffers. `runs/2026-09-11-vit-leak-ab/README.md`.
2. **The runtime shims were stale.** `jax/.lake/build/*_shim.py` predated `4a0a2781` and still
   defaulted tf.data determinism ON, capping every producer at ~1.7 cores; the last-spawned one
   paced the trainer at 3 s per batch. Fixed on this box: `scripts/regen_jax_generated.sh sync`.

Same window as the 2026-09-10 table (`WARM=200 STEPS=600`, 400 measured steps, fed arm), same
driver (`scripts/bf16_probe_3060.sh`), rows from the confs **as they stand today** (`rows.txt`;
worker counts included, and the TSV's `workers` column now records the count a row actually ran
at — the old table said 8 on rows that ran at 4).

## Results

| net | prec | workers | old med / mean / p90 | **new med / mean / p90** | img/s at new mean | verdict |
|---|---|---|---|---|---|---|
| r34 | f32 | 8 | 215 / 214 / 217 | **225 / 226 / 236** | 1133 | compute-bound |
| r34 | bf16 | 8 | 142 / 142 / 145 | **154 / 154 / 164** | 1662 | feed-limited (mean > min) |
| r50 | f32 | 8 | 355 / 355 / 357 | **358 / 361 / 374** | 709 | compute-bound |
| r50 | bf16 | 8 | 222 / 222 / 223 | **222 / 223 / 230** | 1148 | compute-bound |
| mnv2 | f32 | 4 | 172 / 172 / 181 | **177 / 177 / 188** | 1446 | feed-limited (mean > min) |
| mnv2 | bf16 | 4 | 113 / 112 / 121 | **114 / 114 / 128** | 2246 | feed-limited (mean > min) |
| mnv4 | f32 | 4 | 146 / 253 / 563 | **142 / 288 / 198** | 889 | **tail** (mean ≫ median) |
| mnv4 | bf16 | 4 | 157 / 253 / 546 | **128 / 127 / 143** | 2016 | feed-limited (mean > min) |
| enetema | f32 | 8 | 186 / 200 / 272 | **221 / 220 / 232** | 1164 | feed-limited (mean > min) |
| enetema | bf16 | 8 | 102 / 204 / 893 | **149 / 279 / 170** | 918 | **tail** (mean ≫ median) |
| cnx | f32 | 4 | 202 / 201 / 204 | **205 / 204 / 208** | 627 | compute-bound |
| cnx | bf16 | 4 | 120 / 133 / 183 | **124 / 124 / 126** | 1032 | compute-bound |
| cnxs | f32 | 4 | 339 / 339 / 341 | **339 / 338 / 340** | 379 | compute-bound |
| cnxs | bf16 | 4 | 188 / 188 / 191 | **187 / 187 / 190** | 684 | compute-bound |
| cnxb | f32 | 4 | 514 / 513 / 515 | **513 / 513 / 514** | 250 | compute-bound |
| cnxb | bf16 | 4 | 296 / 295 / 298 | **300 / 299 / 302** | 428 | compute-bound |
| vit | f32 | 8 | 220 / 894 / 2974 | **259 / 539 / 300** | 950 | **tail** (mean ≫ median) |
| vit | bf16 | 8 | 141 / 876 / 3219 | **249 / 544 / 382** | 941 | **tail** (mean ≫ median) |
| vits | f32 | 8 | 501 / 502 / 507 | **509 / 505 / 512** | 1014 | compute-bound |
| vits | bf16 | 8 | 287 / 644 / 1087 | **310 / 694 / 390** | 738 | **tail** (mean ≫ median) |
| vitb | f32 | 8 | 1356 / 1353 / 1361 | **1359 / 1358 / 1360** | 377 | compute-bound |
| vitb | bf16 | 8 | 797 / 795 / 803 | **799 / 797 / 803** | 642 | compute-bound |
| r50a3 | f32 | 4 | 209 / 208 / 210 | **214 / 213 / 218** | 1202 | compute-bound |
| r50a3 | bf16 | 4 | 144 / 175 / 281 | **139 / 139 / 145** | 1842 | compute-bound |
| r50a3w8 | bf16 | 8 | 144 / 175 / 281 | **152 / 270 / 159** | 948 | **tail** (mean ≫ median) |

ms/step; `workers` is the count each row actually ran at (the conf's). `img/s at new mean` is
`4·bs·1000/mean`. Verdicts: **compute-bound** = mean within 8 % of min; **feed-limited** = mean
over min but ≈ median; **tail** = mean > 1.15 × median.

**What moved.** The four rows the two bugs distorted most: ViT-Tiny 894 → 539 and 876 → 544
(mean), MobileNetV4 f32 253 → 288 / bf16 253 → 127, EfficientNet-B0 bf16 204 → 279,
ViT-S bf16 644 → 694, RSB-A3 bf16 175 → 139 at the f32 twin's 4 producers. Every compute-bound
row reproduces to within 1–5 ms. **What did not move is the tail**: seven rows still carry a mean
of roughly twice their median, with a p90 barely above the median — a few enormous stalls, not
jitter — and the rest of this file is about those.

⚠ MobileNetV4's old verdict ("both precisions land on 253, bf16 buys nothing") was the bug: the
bf16 row is 127 clean. The f32 row now carries the stall instead. ⚠ RSB-A3 bf16 at its conf's 8
producers (152 med / 270 mean) is worse than at 4 (139 / 139); the conf's count came from the
bug too.

## The stalls that remain, and what they are not

Reproduced on ViT-Tiny f32 (8 producers, `WARM=200 STEPS=600`) six times with the per-step dump
(`diag_vit_w8*.steps.tsv`, columns `step ms wait issue invoke`) and system sampling beside it.

* **Two episodes per 400 measured steps, each ~60 s**, beginning near step 330–390 and again
  ~190 steps later, in every run. Inside an episode consecutive steps take 1–24 s.
* **The time is in the invoke.** `wait_ms` (blocked on the prefetched batch) is 0 on most slow
  steps and `issue_ms` is single digits; `invoke_ms` is 1–14 s. The producers are not late — the
  trainer is not consuming.
* **The GPUs are idle, not slow.** `nvidia-smi` at 2 s: during an episode utilization is 0 % and
  power ~28 W on all four cards; temperature never exceeds 55 °C, SM clock holds 2,730 MHz under
  load, and the throttle mask is clear whenever the cards are busy. Not thermal, not power.
* **Disk is 0.** `vmstat bi` stays at 0 through the run; the dataset is page-cache resident.
* **The box saturates during an episode**: runnable threads 70–100 on 32 hardware threads, idle
  1–3 %, system time 15–22 %. Producer CPU roughly doubles *during* the episode — a consequence
  (they run ahead while nothing consumes), not a cause, see the next point.
* **Every host-side knob failed**, which is what rules the producers out: producers at `nice 15`
  (mean 541), `TF_NUM_INTEROP_THREADS=3` per producer (571), producers pinned to threads 8–31
  (550), trainer pinned to 0–11 with producers on 12–31 (577), XLA command buffers disabled
  (569). Un-treated: 553–567. The episode step indices barely move across all of these.
* **gdb, attached from an ancestor during three episodes** (`diag_vit_w8_gdb2.gdb.txt`): the
  trainer has **239 threads**, 152 of them `tsl::UnboundedWorkQueue` workers. At each stop
  exactly **2 threads were running**, both in `memcpy` under
  `xla::TransposePlan::ExecuteChunk ← xla::CommonPjRtClient::Linearize ← LinearizeIntoImpl ←
  UnboundedAsyncWorkRunner` — PJRT's host-side staging copy of the batch on the way to the
  device. Everything else waited on futexes: the Lean pool readers in `read()`, NCCL proxies in
  `poll()`, the main thread on PJRT.

So the stall is inside the PJRT host→device path of the invoke, with the GPU waiting, on a
schedule set by step count rather than by anything the producers do. Two suspects fit the
~190-step period and neither is tested:

1. **The device BFC pool.** At the plugin's 0.75 fraction each card's pool is 11.68 GiB; the
   per-card input slice is 77 MB; 11.68 GiB / 77 MB ≈ **156 steps**. If freed input buffers
   return to the pool late (PJRT releases them from a deferred path), the pool fills with dead
   chunks and the allocator's retry/GC path runs. Test: `LEAN_MLIR_MEM_FRACTION=0.97` should
   stretch the period to ~200 steps; `LEAN_MLIR_PREALLOCATE=0` changes the allocator's regime.
2. **The pinned staging pool** the `Linearize` copies into: a slow drain of `MemAvailable`
   between episodes (~1 GB) and a 4–9 GB release during one is visible in `diag_vit_w8.sys.tsv`.
   Test: sample `Mlocked` per step (the sampler now records it) against the episode boundaries.

### Next session: the tests, in order

Each is one six-minute run on all four cards with the diagnosis driver in this directory, which
writes the per-step dump (`step ms wait issue invoke`), `vmstat`, and a sampler with producer
CPU, trainer thread state and locked memory. The untreated baseline to compare against is
`diag_vit_w8` (episodes at steps ~367 and ~550, 25 steps over 1 s, mean 553).

    bash runs/2026-09-11-imagenet-probe-postfix/diag_vit_steady.sh diag_vit_w8_mem97     LEAN_MLIR_MEM_FRACTION=0.97
    bash runs/2026-09-11-imagenet-probe-postfix/diag_vit_steady.sh diag_vit_w8_prealloc0 LEAN_MLIR_PREALLOCATE=0
    awk -F'\t' '$2>1000 {print $1, $2, $5}' runs/2026-09-11-imagenet-probe-postfix/diag_vit_w8_mem97.steps.tsv   # slow steps: index, ms, invoke_ms

1. **`LEAN_MLIR_MEM_FRACTION=0.97`** raises each card's BFC pool from 11.68 to ~15.1 GiB — as a
   DIAGNOSTIC only: 0.75 is the plugin's default for a reason and 0.97 is known to crash
   ConvNeXt's bf16 arms, so it is a lever to move the period with, never a setting. If the
   device pool is the mechanism (11.68 GiB / 77 MB per-card input slice ≈ 156 steps), the gap
   between episodes stretches from ~190 toward ~250 steps and the first one moves later. Fix:
   reuse the four device input buffers across steps in `ffi/pjrt_ffi.c` (allocate once, copy
   into them) instead of a fresh `BufferFromHostBuffer` allocation every step.
2. **`LEAN_MLIR_PREALLOCATE=0`** puts the allocator in its on-demand regime. Stalls gone →
   same conclusion as 1; stalls unchanged → the device pool is exonerated.
3. **Locked memory against the episodes.** Columns 11–13 of `<name>.sys.tsv` are the trainer's
   `VmLck`, its `RssAnon`, and the box's `Mlocked`, every 5 s. A sawtooth that drops at each
   episode means the pinned staging pool is being torn down and re-pinned; the fix is a
   persistent pinned host buffer for the batch (one registration, reused) rather than PJRT's
   per-step staging copy.
4. **If all three are flat**, the remaining lead is the staging copy's dispatch itself:
   `xla::CommonPjRtClient::Linearize` chunks the 308 MB onto an unbounded pool (152 workers
   observed). Re-run `diag_gdb2.sh` and look at *how many* threads are in `ExecuteChunk` per stop
   and what the main thread waits on; then try the PJRT async host→device path
   (`CreateBuffersForAsyncHostToDevice` + `TransferRawDataToBuffer`), which copies from a
   pinned buffer without the transpose plan.

⚠ `--xla_gpu_host_memory_limit_gb` does not exist in this XLA build; `XLA_FLAGS` is honored
(a bogus flag aborts at startup), so other flags can be tried the same way. ⚠ gdb attaches only
from an ancestor here (`ptrace_scope=1`): run the trainer *under* gdb as `diag_gdb2.sh` does.
⚠ The step index of an episode is the discriminator, not the mean — the mean moves with noise,
the period moves only with the mechanism.

Structural fixes, once the suspect is confirmed: pinned, reused host batch buffers so PJRT DMA's
without a staging copy (an `ffi/pjrt_ffi.c` change); or a uint8 wire format with normalisation
in the graph, which cuts the pipe, the copy and the staging by 4× but moves a transform across
the shim boundary.

⚠ The ETA strings are computed from the **median** (`probe_to_eta.py --stat med`, the user's
call on 2026-09-11): the median is what the graph costs, and the stall above is a trainer-side
defect to be fixed, not a property of the job. On the seven tail rows the string names today's
mean beside it ("⚠ today's mean is 539: periodic stalls …"), so nobody schedules on the median
without seeing the gap. The 2026-09-10 table and the first draft of this one used the mean.

### Producer supply, for the record

Once determinism is off the producers' aggregate CPU is ~21 cores whether there are 4, 6 or 8 of
them (5.05 / 3.63 / 2.72 cores each), and the heavy-recipe rows deliver ~2,000 img/s. That is a
recipe-cost ceiling, not a worker-count knob, and `ds.shard` at example level makes every
producer parse the whole record stream — file-level sharding would remove an 8× redundant
parse. Not the cause of the stalls above; a separate follow-up.

## What was written back

* 14 `scripts/jobs/*-4gpu.conf` `ETA=` lines, on the **median**, **bf16 first** —
  `bf16 ~55 h / f32 ~57 h on 4x 4060 Ti (249 / 259 ms/step median, 300 ep)` — whichever
  precision the conf trains (its variant says which); the stall rows carry today's means as an
  annotation (`eta_strings.txt`). The RSB-A3 bf16 conf quotes its own 8-producer bf16 row against
  the f32 conf's 4-producer row. MobileNetV4's conf now runs **500 epochs** (the Conv-M paper's
  schedule, the user's call): `EPOCHS=500` for the supervisor and `LEAN_MLIR_EPOCHS=500` in its
  environment for the trainer's cosine and loop bound; the Lean default stays at the reference's
  tier-2 100. The ETA reads the 500 from the conf.
  `vit-default-emabf16-4gpu` was not probed (no `vitema` row) and is unchanged. **Staged, not
  committed.**
* `scripts/probe_to_eta.py`: reads each row's worker count from the TSV (the driver now records
  the effective value), maps the `r50a3w8` row to `r50-a3-wxclip-bf16-4gpu`, labels the string
  with the statistic it used, and words the mean ≫ median annotation as a stall rather than
  "feed-bound".
* `scripts/bf16_probe_3060.sh`: records the row's effective `SHIM_WORKERS` in the TSV.
* NOT changed: any conf's `SHIM_WORKERS`. The measurements say 6 for ViT and 4 for RSB-A3 bf16;
  that is a decision for daylight.

## Files

`rows.txt` the row table · `fed.tsv` / `fed.log` the 25 rows · `table.md` the markdown table ·
`eta_strings.txt` · `diag_vit_steady.sh` the steady-state diagnosis driver (per-step dump +
vmstat + memory/CPU/locked-memory sampler) · `diag_vit_w8*.{log,steps.tsv,sys.tsv,vmstat.txt}`
the six diagnosis runs (`_nice`, `_tf3`, `_pinned`, `_split`, `_nocmdbuf`, `_gpu` with
`nvsmi.csv`, `_phases`) · `diag_vit_w8_gdb2.gdb.txt` three storms of thread stacks ·
`diag_gdb2.sh` how gdb was made an ancestor · `then_ablation.sh` the chain into the ablation
sweep.

## Reproducing

    bash runs/2026-09-11-imagenet-probe-postfix/run.sh
    scripts/probe_to_eta.py runs/2026-09-11-imagenet-probe-postfix/fed.tsv --box "4x 4060 Ti"

⚠ `scripts/regen_jax_generated.sh box` first, on any box. A stale runtime shim reproduces the old
table, not this one, and nothing in `lake build` will say so.
