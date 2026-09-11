# The ImageNet trainer retains ~136 MB/step of host memory, and it is the reader threads

**2026-09-11, 4× RTX 4060 Ti.** Eight arms, ViT-Tiny/ImageNet, `adamdp128x4wxclipdrop`,
4 replicas × bs 128 = global 512, `LEAN_MLIR_PROBE_WARM=50 LEAN_MLIR_MAX_STEPS=400`.

## Why this exists

`vit-default-4gpu.conf` says `~112 h (4.7 d) … ⚠⚠ FEED-BOUND`. The same network trained
**300 epochs in 48 h 00 m on the four 3060s** (`ea96760c`), a box with older GPUs and less
RAM. A worse box cannot be 2.3× faster, so one of the two numbers describes something other
than the hardware.

It was not the hardware. Both are right about their own box, and this box is sick.

## The measurement

| arm | configuration | min | **med** | p90 | **mean** | RSS slope | peak |
|---|---|---|---|---|---|---|---|
| A | 8 producers, depth 8 (the conf) | 210 | **220** | 1576 | **561** | 136 MB/step | 90.6 GB |
| B | `LEAN_MLIR_PREFETCH=0` | 676 | **721** | 748 | **721** | **0** | 33.2 GB |
| C | 8 producers, depth 4 | 208 | **223** | 914 | **388** | 136 MB/step | 74.5 GB |
| D | 4 producers, depth 4 | 206 | **219** | 2772 | **847** | 87 MB/step | 68.6 GB |
| E | depth 8 + `MALLOC_MMAP_THRESHOLD_` pinned | 210 | **218** | 1697 | **409** | 190 MB/step | 90.8 GB |
| F | depth 8 + `readExact` preallocation | — | — | — | — | 569 MB/step | 89.4 GB |
| G | depth 8 + `MALLOC_ARENA_MAX=1` | 212 | **225** | 1607 | **404** | 154 MB/step | 94.8 GB |
| H | **1 producer, depth 1** | — | (4220) | — | — | **79 MB/step** | 41.2 GB |

ms/step. The synthetic (no-shim) floor for this graph is **217**.

⭐ **Read the median column.** It is 218–225 in every arm that prefetches, against a 217 ms
compute floor, and `med − min` — the probe's own starvation statistic — is 8–15 ms. **Nine out
of ten steps already run at full speed.** Nothing is feed-bound. The mean is wrecked by a
minority of steps that take 1.5–2.8 s, and those track memory pressure, not data supply.

## What is actually happening

The trainer's **anonymous** RSS climbs ~136 MB/step until the box has no free memory, and then
the kernel runs **continuous direct reclaim** — the blocking kind. Measured mid-run at depth 8:

| | |
|---|---|
| `pgscan_direct` | 1,121,727 pages / 24.7 s — **3.6× the background reclaimer** |
| `allocstall_movable` | 1,056 in 24.7 s, i.e. 43 blocking allocator stalls per second |
| GPU 0 idle | **51 %** of samples, in a clean periodic sawtooth |
| free RAM | 2 GB of 251 |
| disk read | **0.0 MB/s** |

It also **degrades as it runs**: ~480 ms/step early, ~2,300 ms/step by step 1300. A short probe
therefore *flatters* the job. Memory is released in full at process exit (91 GB → 7 GB).

## It is reading OFF THE MAIN THREAD, not the buffers

The rate tracks whether the read happens on a pool thread, and then scales mildly with how many:

* 8 producers → 136 MB/step; 4 → 87; **1 → 79**; **prefetch off → exactly 0** (arm B, flat at
  33.16 GB across 300 steps).
* **Depth is irrelevant.** Arms A and C have 8 producers at depth 8 and 4 and both read
  136 MB/step. Arm D has depth 4 but only 4 producers and reads 87. Depth is not thread count.
* ⛔⛔ **One producer leaks 79 of the 136**, so this is not depth-n and not sharding. It is
  inherent to reading off the main thread, i.e. `045b64eb` (2026-08-05, "overlap the batch read
  with compute — 377 → 224 ms/step"), not `6135bb07` (2026-08-11, depth-n). Producer count adds
  to it; it does not cause it. ⚠ At 79 MB/step the ImageNet path fills this box inside **one
  epoch** (2502 steps ≈ 198 GB), at any worker count.
* ▶ Arm H also re-confirms the conf's own `SHIM_WORKERS=1 → 4,348 ms/step` note: it measured
  4,220 ms/step here, which is the independent check that this arm ran what it claimed to.

`readExact` grows its buffer by doubling (`ByteArray.append` → `copySlice … exact := false`),
so a ~300 MB batch walks up through 64 KB … 256 MB, 512 MB. Under `LEAN_MLIR_PREFETCH` that
churn happens on a **pool thread** while the main thread frees the result. Arm B runs the same
code on the main thread and retains nothing.

## Three workarounds, all dead

⛔ `MALLOC_MMAP_THRESHOLD_` pinned (E), `MALLOC_ARENA_MAX=1` (G), and preallocating the buffer
with `ByteArray.emptyWithCapacity n` (F) **all still leak**, and F was *worse* than doing
nothing. F is the informative failure: preallocation changes the allocation *pattern* but not
*which thread allocates*, and arm B already proved the pattern is not the variable. The change
was reverted.

Since neither allocator tuning helps, the memory is **genuinely retained**, not stranded in a
per-thread arena. The next suspect is the `IO.asTask` result path itself — and arm H narrows it
usefully, because a single outstanding task leaks nearly as much as eight.

## Scope

✅ **ImageNet only.** `imgStreams` is populated by `spawnShimSharded` under
`if net.data == .imagenet && !synth`, so CIFAR, MNIST and Imagenette never open a producer pipe
and cannot take this path. Every 4-GPU ImageNet job is affected; nothing else is.

⚠ **No job conf sets `LEAN_MLIR_PREFETCH`**, so every ImageNet job in `scripts/jobs/` runs with
it ON and leaking. There is no ViT conf for the 3060 box in the tree — five confs name that box
(`enet-default`, `enet-default-bf16`, `r50-2018`, `r50-2018-bf16`, `r50-a3-wxclip-bf16`) and
none is ViT — so the 48 h run's configuration was never checked in and cannot be diffed. It was
not prefetch-off, though: arm B's 721 ms/step is 150 h for 300 epochs, and 48 h needs 230.

## What the fix is worth

The median is 220 ms/step. If the mean collapses to it, ViT-Ti's 300 epochs are
750,600 × 0.220 s = **45.9 h**, against the 112 h now printed — and within noise of the 3060
box's measured 48 h 00 m. That is the size of the prize, on this net alone.

## What this invalidates

⛔ **Every `ETA=` string written on 2026-09-10** (`runs/2026-09-10-bf16-probe-4060ti`, 14 confs).
They are computed from the **mean**, and on this box the mean is this bug. That directory's own
"the starvation is predictable, and it is a supply limit" section ranks nets by images/s per
producer; the ranking is real but the mechanism it infers is not.

⛔ **The four bf16 "does not pay" verdicts** in `bf16/job-confs`. All four were read off the mean.
MobileNetV4 is the clearest: both precisions landed on an *identical* 253 ms mean despite
compute floors of 80 (bf16) and 140 (f32). Identical means across different floors is not a
pipeline running dry, it is both arms hitting the same external stall.

⚠ `vit-default-4gpu.conf` moved `SHIM_WORKERS` 4 → 8 on the strength of w4 mean 894 → w8 mean
520. Both numbers are this bug. Arm D measures w4 at mean 847 here, so the direction happens to
hold, but the magnitude is not a feed result.

## The fix, and what was under it (2026-09-11, later the same day)

### It was never a leak

Lean v4.32's runtime allocator is **mimalloc**, statically linked — the `MALLOC_*` knobs in arms
E and G configured glibc, which owns none of these buffers. A huge block (≳ 32 MB) that one
thread allocates and another frees is not released by mimalloc: the freeing thread
`madvise(MADV_FREE)`s it and leaves the real free to the owner, a pool thread that never gets
round to it. The pages stay in RSS, counted as `LazyFree` in `smaps_rollup`, until the kernel is
under pressure — which is why the growth was sublinear, why arm H plateaued at 41 GB with 97 GB
free, and why the box then lived at the reclaim watermark.

`repro/` reproduces it with no GPU, no data and no shim, in ~30 lines: `Handle.read` on a pool
thread, drop on main.

| repro mode (150 MB × 100, /dev/zero) | RssAnon | LazyFree | page faults |
|---|---|---|---|
| `direct` — read on the main thread | 1.4 GB flat | 0 | 686 |
| `task` — the trainer's old path | 9.0 GB | 8.2 GB | 43,416 |
| `dedicated` — fresh OS thread per read | 1.7 GB flat | 0.9 GB | 154,183 |
| `task` + `MIMALLOC_PURGE_DELAY=0`, 200 ms between reads | 5.3 GB @30 | 4.7 GB | — |
| **`intoswap` — the fix** | **1.25 GB flat** | **0** | **619** |

⛔ Two things that look like fixes and are not: `MIMALLOC_PURGE_DELAY=0` holds flat only while the
owning thread keeps allocating (a tight loop), and does nothing at a 200 ms step cadence;
`Task.Priority.dedicated` keeps RSS flat because an exiting thread abandons its heap, but refaults
the whole buffer every step.

### The fix: the main thread allocates and frees, the pool thread only fills

`VerifiedTrain.readInto` (`lean_mlir_read_into` in `ffi/f32_helpers.c`) reads into a
caller-supplied `ByteArray`. The prefetch loop allocates the batch buffer on the main thread,
hands it to the task through an `IO.Ref` (`swap`, so the task holds the only reference — a buffer
captured directly by the closure arrives with rc = −2), the task fills it, main frees it, and
mimalloc recycles the segment in place. `readExact` is now `readExactInto` on a fresh
exact-capacity buffer, which also retires the `append`-doubling copy (a 308 MB batch used to cost
a 616 MB buffer and a memcpy).

⚠ `lean_is_exclusive` is **false for every multi-threaded object** — an object that has crossed a
`Task` keeps a negated rc for life — so the C side checks `m_rc ∈ {1, −1}` by hand and writes the
size field directly (`lean_sarray_set_size` asserts `lean_is_exclusive`, and in a debug build that
assert prompts on stdin, which looks exactly like a hang).

| arm A, 4× bs128, 8 producers | RSS step 100 → 300 | LazyFree | faults/step | min | med | p90 | mean |
|---|---|---|---|---|---|---|---|
| pre-fix | 77.7 → 87.7 GB | — | — | 210 | 220 | 1576 | 561 |
| **post-fix** | **36.7 → 37.1 GB** | **0.00** | **11** | 206 | 214 | 1488 | 389 |

The memory is fixed. The mean is not — and it is not memory: 159 GB was free the whole run.

### Under it: one producer paced the trainer, because the shim on disk was stale

`LEAN_MLIR_PROBE_DUMP` (new) writes the per-step series; every slow step is `s ≡ 7 (mod 8)`, every
one waits 0.84–1.43 s in `IO.wait`, and the other seven slots never wait at all. The per-read
trace (issue / start / end, also new) says which side:

| handle | queue (start − issue) | transfer (end − start) |
|---|---|---|
| 0–6 | 0 ms | 676–1206 ms |
| **7** | 0 ms | **3043 ms** (spread 70 ms) |

Not the task pool. Producer 7 delivers a batch 3 s after the read starts, so the seven others
have slack and block in `write()` while it paces the round. With four producers the transfer
times are 1272 / 1398 / 2126 / 3286 in spawn order: it is always the last spawned, never a
particular shard. All producers had identical CPU, bytes read, bytes written and thread counts;
the pipes are all 64 KB; the box was **82 % idle**. Their writer threads spent half their time
waiting on tf.data's `next()`.

The producers were capped at ~1.7 cores each because they were running with
`enable_op_determinism()` ON. `4a0a2781` (2026-09-10) flipped that default to OFF in the
generator and in the tracked `jax/generated/*.py` — but the trainer resolves its shim from
`jax/.lake/build/`, an untracked build product that `lake build` never regenerates, and every one
of the 39 runtime shims on this box dated from 2026-08-30. **Every probe on this box since the
flip, all fourteen `ETA=` strings included, ran determinism ON.** The 3060 box's 48 h run passed
`SHIM_DETERMINISM=0` by hand.

| post-fix, `SHIM_DETERMINISM=0` | min | med | p90 | mean | max wait | transfer/handle |
|---|---|---|---|---|---|---|
| 8 producers | 209 | 253 | 294 | 269 | 0 | 802–828 |
| 6 producers | 206 | 257 | 288 | 265 | 0 | 813–844 |
| 4 producers | 205 | 257 | 354 | 275 | 144 | 854–908 |
| **6 producers, synced shims, no override** (`A_fix_synced_w6`) | 208 | 257 | 287 | 270 | 0 | — |

The tail is gone at 6 and 8; 4 is at the feed limit. **Six** is the count for this box. The last
row is what a job conf now gets here after `scripts/regen_jax_generated.sh sync`: nothing in the
environment, and the same numbers as the explicit `SHIM_DETERMINISM=0` arm.

### Still open: 40 ms of contention on the step itself

With determinism off the median is 253–257 against the same 205–209 minimum, at every worker
count — so it is not producer CPU. It is interference with the trainer's own step from producers
that now run in bursts (scheduling latency or memory bandwidth against the 308 MB H2D staging
copy). ⚠ The probe's `med − min` line calls this "starvation wait"; the dump's `wait_ms` column
is 0, so it is not. Lowering the producers' priority
(`SHIM_PYTHON` → a wrapper that `exec nice -n 15 python3`, 6 producers, determinism off) measures
min 208 / med 250 / p90 272 / mean 259: 7 ms of median and a tighter p90, so scheduling priority
is a minor part of it, not the mechanism. The remaining candidates are in `NEXT_SESSION.md`.

### What this changes in the plan

* `tests/prefetch_tie.sh` **passes** on the new read path: control 0 differing bytes, prefetch
  OFF vs ON 0 differing bytes over 24 steps across an epoch boundary (`prefetch_tie_postfix.log`).
* ⛔ The ETAs are still invalid, now for two reasons. `scripts/regen_jax_generated.sh box` is the
  precheck and `… sync` the fix; run here on 2026-09-11 it synced **68 of 78** artifacts (the 39
  shims and the emitted JAX trainers). `scripts/gen_shims.sh` covers only the 10 default recipes.
* `vit-default-4gpu.conf`'s `SHIM_WORKERS=8` was chosen on the buggy mean; with determinism off
  and the leak fixed, 6 measures best here and the conf should say so.
* The 3060 box (192 GB, `SHIM_WORKERS=4`, prefetch on by default, `SHIM_DETERMINISM=0` by hand)
  had the same lazily-freed pages and was fine. Why this box hit the reclaim wall and that one did
  not is not proven; the leading guess is the page cache competing for the same 144 GB dataset,
  which is moot now that the pages are recycled.

## Reproducing

```bash
bash runs/2026-09-11-vit-leak-ab/run.sh       # arms A, B
bash runs/2026-09-11-vit-leak-ab/run_cde.sh   # arms C, D, E
bash runs/2026-09-11-vit-leak-ab/run_g.sh     # arm G
bash runs/2026-09-11-vit-leak-ab/run_h.sh     # arm H  (slow: 1 producer is ~4.2 s/step)
```

⚠ `run2.sh`/`run3.sh` are kept only as the record of a mistake: they gate on
`pgrep -f 'vit-leak-ab/run.sh'`, which **matches the spawning shell's own command line**, so
both waited forever instead of starting. `run_cde.sh` is the fixed sequencing.

The per-arm `*.rss.tsv` carry `t_s / step / RssAnon GB / free GB`. The slope that matters is
RSS against **step**, not against wall clock — arm B is 3.3× slower per step, so a per-second
rate would flatter it for the wrong reason.
