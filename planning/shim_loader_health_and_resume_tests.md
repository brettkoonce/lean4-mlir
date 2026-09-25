# shim_loader_health_and_resume_tests.md — the degrading shim loader, a loader-only respawn, and resume hardening

**Opened 2026-09-14**, during the EfficientNet-B0 verified 350-epoch run
(`runs/2026-09-12-enet-verified-350ep/`). Three jobs, all to land **before the ConvNeXt run**:

1. **§2 — the soak.** Find out whether the loader degradation is the allocator, before ConvNeXt
   rides on the same shim architecture.
2. **§3 — the loader-only respawn.** Design for replacing a sick loader without stopping the
   trainer. Build it only if §2 does not remove the cause.
3. **§4 — resume hardening.** The supervisor's periodic restart is now load-bearing, so the resume
   path it rides on needs more than the one live test it has had.

---

## 1. What is known (measured, with where)

**Symptom.** The run held 690–700 s/epoch for epochs 3–64, then degraded in bursts from epoch 65
(~13 h of loader uptime) and settled at ~1,250 s/epoch by epoch 114: steps at 240–250 ms against 134
at launch. `epoch_clock.tsv`, `log_ts.log`.

**Mechanism: one loader, amplified by round-robin.** `readShimBatchRR` takes batch `k` from loader
`k % 4`. In 20 of 20 `wchan` samples three loaders sat in `anon_pipe_write` (full pipe, waiting on
the trainer) and the fourth never did; the trainer waited on it. Loader write rate = 4.33
batches/s = exactly the step rate. One slow loader sets everyone's pace.

**Not the box.** No thread over 18% in any loader, box 58% idle. Zero real disk reads (81% of train
shards in page cache, 0 major faults). No swap activity. Throttle counters zero on all 24 CPUs,
cores 30–34 °C, RAPL limit 4,095 W, no kernel messages. Brett has seen the same on both setups.

**A restart fixes it immediately.** Supervisor restart at epoch 123 → 130–133 ms/step. Planned
restarts (`REST_EPOCHS`) at 125 and 160 were clean; epochs 161–172 held 703–706 s.

**⭐ It is the LAST-SPAWNED loader, and the growth is in malloc arenas.** 7 h 53 m after the
epoch-160 restart (`/proc/<pid>/smaps`, anonymous mappings bucketed by size):

| loader (spawn order) | RSS | 8–68 MiB anon maps | RSS in them | >512 MiB maps |
|---|---|---|---|---|
| 1st | 5.3 GiB | 324 | 3.96 GiB | 1.08 GiB |
| 2nd | 5.5 GiB | 325 | 4.22 GiB | 1.09 GiB |
| 3rd | 5.6 GiB | 321 | 4.26 GiB | 1.17 GiB |
| **4th** | **9.0 GiB** | **396** | **7.54 GiB** | 1.22 GiB |

The 8–68 MiB class is glibc malloc's per-thread arena heaps (64 MiB each on 64-bit); glibc's default
arena cap is 8 × cores = 192, and each loader runs **168 threads**. Large buffers and file-backed
memory are the same in all four. The loader that lagged before the epoch-123 restart was also the
last spawned, and `runs/2026-09-11-imagenet-probe-postfix/README.md` saw "the last-spawned one"
pace the trainer too (different trigger: stale shims).

⚠ **But "always the last-spawned" is too strong, and one generation is the counterexample.** Over
the five loader generations this run went through (`loader_rss.tsv`, last sample per pid), the
last-spawned loader ballooned in FOUR — 9.28 GiB @ 8.0 h, 7.04 @ 7.7 h, 8.44 @ 7.7 h, 9.51 @ 5.6 h,
against ~5.3–6.2 for its three siblings — and in the fifth (epochs 280→320) all four stayed flat at
5.35–5.67 GiB for 7.7 h. Growth rate varies a lot too: the last generation hit 9.5 GiB in 5.6 h and
was already costing 801–860 s epochs at the end of the run. So the trigger is probabilistic, not
positional — which also means **§2's arm A must run long enough, and ideally more than once, before
a "no divergence" result means anything.**

**⭐ The JAX reference never degraded.** Same augmentation, tf.data IN-PROCESS: 464–467 s/epoch
through epoch 150, then 454–458 s for all 200 epochs of a single 26 h process
(`/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet_full.log`). So the fault lives in the
shim arrangement — 4 processes × 168 threads on 24 hardware threads, each with its own TF pools and
its own tf.data autotuner, each parsing the whole record stream under example-level `ds.shard` —
not in tf.data or the dataset.

**Hypotheses, in the order §2 tests them.**
- **H1 — glibc arena fragmentation.** Many threads × per-thread arenas × allocation churn from JPEG
  decode and AutoAugment; free memory stranded in arenas, allocation gets slower as they fragment.
  Fits "growth is in arena-class maps".
- **H2 — tf.data AUTOTUNE divergence.** `map(num_parallel_calls=AUTOTUNE)` and
  `prefetch(AUTOTUNE)` with no options set. The last-spawned loader starts tuning while three
  others already load the CPU, and tunes differently. Fits "always the last one".
- **H3 — oversubscription.** 672 loader threads on 24 hardware threads; amplifies both of the above.

**Tools on the box.** glibc 2.39; `libjemalloc.so.2` installed; no tcmalloc; no `py-spy`.

---

## 2. The soak — BEFORE ConvNeXt

Shim-only: no GPU, no trainer, so it runs on an otherwise idle box and is clean to read.

**Harness, `scripts/shim_soak.py`.** Spawn 4 loaders exactly as `spawnShimSharded` does — the
**ConvNeXt** shim, since that is the next run, with `SHIM_SHARD=i/4`, `SHIM_SEED=seed+i`, spawned in
order — and read them round-robin with one read in flight per handle into a reused buffer (mirror
`issueRead`). Consume unthrottled. Every 60 s log per loader: RSS, arena-class RSS and map count
(the §1 bucketing), p50/p90 read latency, batches/s.

⚠ **Gate the harness on reproducing the fault first.** Arm A must show the §1 signature (one
loader's arena-class RSS pulling away, its latency rising) within ~10 h. If it does not, the harness
is missing what the trainer adds — do not conclude anything from arms B–D.

| arm | change | tests |
|---|---|---|
| A | none | reproduces §1? |
| B | `MALLOC_ARENA_MAX=2` | H1 (glibc arenas) |
| C | `LD_PRELOAD=libjemalloc.so.2` | H1 (a different allocator entirely) |
| D | `num_parallel_calls` + prefetch pinned (e.g. 6 and 2), `options.threading.private_threadpool_size=6` | H2/H3 |

~10 h per arm. **Pass:** at 10 h, max/min loader RSS within 20% and no loader's p50 latency > 1.5×
the others'.

⚠ **If B or C wins, scope it to the LOADERS.** `IO.Process.spawn`'s `env` adds to the inherited
environment, so a `LD_PRELOAD` in the job conf's `ENV_EXTRA` would also preload jemalloc into the
TRAINER — under mimalloc (Lean runtime) and XLA's own allocators. Add a `SHIM_LD_PRELOAD` /
`SHIM_MALLOC_ARENA_MAX` pass-through that `spawnShim` maps onto the child only. (`MALLOC_ARENA_MAX`
is harmless to the trainer, which does not use glibc malloc, but scope it anyway.) D needs a shim
emitter knob.

**If short on time:** launch ConvNeXt with the winning arm guessed (B), `REST_EPOCHS` kept as the
safety net, and the loader-RSS logger from §5 running — a production A/B against this run's table.

---

## 3. Loader-only respawn — design

**Why over the supervisor restart.** The supervisor restart works (§1) and costs only ~1.5 min. The
wins are structural: data health stops depending on checkpoint/resume; no recompile; the trainer
never stops; it can react within minutes instead of on a fixed epoch list; it works for probes and
jobs with no checkpoints.

### 3a. Level 0 — stop the amplification (smallest change)
Consume whichever in-flight read completes first instead of strict round-robin. A slow loader then
costs a quarter of its shortfall instead of pacing all four. Batch order is already
nondeterministic with determinism off. ⚠ Must stay strict round-robin under `SHIM_DETERMINISM=1`:
`tests/prefetch_tie.sh`, `scripts/residency_gate.sh`, `scripts/mixup_gate.py` and
`scripts/shim_wiring_gate.py` replay streams.

### 3b. Level 1 — staggered periodic respawn ✅ BUILT 2026-09-16 · ⚠ STILL UNTESTED AGAINST THE FAULT (2026-09-17)
Every `E` epochs replace ONE loader, cycling through the handles, so none lives longer than `4E`
epochs and at most one is ever cold. Knob: **`LEAN_MLIR_SHIM_RESPAWN_EPOCHS=E`**, default 0 = off,
announced in the banner.

**What landed** (`LeanMlir/VerifiedTrain.lean`): the §3e prerequisites — `ShimProc` (child + pipe),
`spawnShim`/`spawnShimSharded` returning it, `readShimBatchRR` indexing `.h`, and the validation
shim now killed and reaped instead of left as a `<defunct>` python — plus the respawn itself at the
epoch boundary, after the checkpoint is on disk.

⚠ **The BLOCKING form, not §3d's zero-downtime swap.** The replacement is spawned at the boundary
and the trainer waits out its startup. Measured on the smoke test: **~3 s per respawn**, an order
cheaper than the ~30 s budgeted, because `spawnShim` only waits for the 16-byte preamble rather
than the first batch. At `E=10` that is ~0.05%. ▶ §3d is worth building only if that changes.

**Evidence.**
* Inert when off: `tests/prefetch_tie.sh` — control 0 differing bytes, verdict 0 differing bytes
  over 24 steps across an epoch boundary, i.e. the refactor moved no bit of trained state.
  ⛔ That gate hardcoded ares' `xla_cuda12` plugin path and could not run on the 3060 box at all
  (`dlopen … No such file or directory` at step 0); it is box-aware now, like the job confs.
* Works when on: R34 bf16, `LEAN_MLIR_G2_STEPS=12 LEAN_MLIR_MAX_EPOCHS=3
  LEAN_MLIR_SHIM_RESPAWN_EPOCHS=1` — banner announced, slots 0→1→2 cycled after epochs 1/2/3,
  seeds 5/10/15 (`shimSeed + slot + gen×n`, so no two generations of a shard repeat an
  augmentation sequence), stepping continued across each swap, **zero shim children left behind**.
* ⚠ The smoke test also caught the respawn firing after the FINAL epoch — spawn a loader, exit on
  top of it. Fixed with `ep + 1 < nEpochs`.

⭐⭐ **THE R34 bf16 RUN HAPPENED — 2026-09-17, and the fault DID NOT REPRODUCE.**
`runs/2026-09-16-r34-bf16-90ep/` (74.064 / 91.754 in 22 h 10 m, one attempt). It was run as **Arm A**:
respawn OFF for all 90 epochs, `REST_EPOCHS=""`, so the loaders ran **21.9 h continuously** — past the
≥ 16 h bar §5 asked for, and past the ~13 h at which B0's producer diverged.

* **Pace held.** By loader age: 884 s (0–2 h warm-up), then 873/873/872/872/873 through 12 h, and
  882/883/881 from 16 h to the end. Zero respawns, zero rests, zero thermal events.
* **Memory plateaued.** Arena class ~3.2–3.45 GiB at spawn → ~3.9–4.2 GiB by hour 2 → **flat for the
  next twenty hours**, and **uniform across all four loaders**. B0's signature was ONE producer at
  5 → 7–11 GiB while the others sat still.
* ⛔ **The five slow epochs in that run are NOT this fault.** e57/e59/e60/e61/e62 (972–1055 s) track
  host `MemAvailable` falling 126 → 87 GiB for ~2.5 h — another job of Brett's on the same box,
  confirmed. They **recovered on their own**, and loader RSS *fell* through them. This fault has
  never self-cleared; it only ever went away on a kill.

⭐ **And there is a mechanism, not just a null.** The `SHIM_WORKERS` sweep taken immediately before
(`runs/2026-09-16-r34-bf16-sweep/`) measured R34's feed at **19 ms of a 163 ms step (12 %)** against
B0's **37 of 134 (28 %)** — fed-minus-synth, 800 steps per arm. R34's shim is flip-only; B0's does
AutoAugment (the full ImageNet policy, no RandAugment) per image. A shim doing a quarter of the per-image work allocates to
steady state early and stays there. **There is very little feed here to degrade.**

⚠⚠ **What this settles and what it does not.** It **bounds** the fault to heavy-augmentation shims;
it does **not** exonerate the multi-process loader arrangement, and it says **nothing** about whether
the respawn works in production — the respawn was never switched on. The root cause inside the loader
remains unidentified; **§2's allocator soak is still the experiment that would find it.**

### 3c. Level 2 — lag-triggered respawn
Keep a rolling p50 of issue→ready latency per handle; the prefetch already timestamps both
(`issueRead`'s `tIssue`, `IO.wait`). If one handle exceeds `k ×` the median of the others for `W`
steps, respawn that handle. ~20 lines on top of the swap protocol. Knob: `SHIM_RESPAWN_LAG=k`.

### 3d. Swap protocol (zero downtime, no lost or duplicated batch)
1. Spawn the replacement inside `IO.asTask` (`.dedicated`): `spawnShim` blocks on the 16-byte
   preamble, which includes TF import and pipeline build.
2. Keep reading the OLD handle meanwhile.
3. When the replacement is ready AND the slot has nothing in flight — i.e. right after `IO.wait`
   frees it, before the refill loop issues its next read — set `imgStreams[slot] := new`.
   `issueRead` captures `imgStreams` by value at issue time, so a swap at that point cannot change a
   read already in flight.
4. Kill and reap the old child. Its pipe may hold a half-written batch; it is never read again, so
   framing cannot slip.

### 3e. Code changes
- `spawnShim` returns `ShimProc := { child, h, shard, seed, gen }` instead of dropping the
  `IO.Process.Child` — today the trainer **cannot** kill or reap a loader. `spawnShimSharded`
  returns `Array ShimProc`; readers take `.h`. This also lets the validation shim be reaped (a
  `<defunct>` python3 was observed on the live run).
- `imgStreams` becomes `let mut`.
- Seed per generation: `seed + i + 4·gen`, so a replacement draws fresh augmentation. ⚠ And the
  shim's `ds.shuffle(8192, seed=42, …)` is a constant: every fresh loader — and every supervisor
  restart today — replays its shard's example order from the top. Make it `42 + SHIM_SEED` in the
  emitter; the four replay gates above pin their own seeds, re-run them.
- Default OFF, and `tests/prefetch_tie.sh` re-run with it off.

**Recommendation:** run §2 first. If an allocator arm removes the cause, build Level 0 anyway (cheap,
removes the amplification whatever the cause) and keep 1/2 as unbuilt insurance. If nothing in §2
fixes it, build Level 0 + Level 2.

---

## 4. Resume hardening — BEFORE ConvNeXt

Resume carries every planned restart and every thermal rest, and has had one live test (the
epoch-1 SIGTERM on this run, hash-identical companion). Single GPU, Imagenette, all via
`trainAdamSched`:
- **BatchNorm + EMA:** `efficientnet-verified-adam`, `LEAN_MLIR_VARIANT=emarms`.
- **LayerNorm + EMA** (ConvNeXt's path, no companion): `convnext-verified-adam`,
  `LEAN_MLIR_VARIANT=ema`.

Both exes are stale (July): rebuild. The in-memory shuffle is seeded `ep + 42` and dropout masks by
step, so a resume replays the same data. ⚠ Confirm the Imagenette crop/flip FFI is seeded the same
way before trusting R1.

| test | what | pass |
|---|---|---|
| **R1** bit-exact, BN+EMA | 2 epochs straight, twice (the determinism floor), vs 1 epoch → kill → resume → 1. Compare `ckpt.bin` + `.bn` sha256 and the epoch-2 eval. `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. | resumed == straight, or within the straight-vs-straight floor |
| **R2** bit-exact, LN+EMA | R1 on `convnext-verified-adam` / `ema` | same; no `.bn` written |
| **R3** kill fuzz | 20× SIGKILL at a random offset within ±3 s of the epoch-end write | every restart resumes at N or N−1; no size-guard throw; `.tmp` leftovers ignored |
| **R4** fallback | delete `.bn`, resume | ⚠ warning printed; eval after 1 epoch within ~0.5 pt of R1 |
| **R5** wrong companion | a `.bn` of the wrong size | loud refusal, not a resume |
| **R6** supervisor rests | 4-epoch job, `REST_EPOCHS="2"`, `REST_SECS=5` | rests at 2, resumes, reaches `✅ COMPLETE` |
| **R7** (optional) accumulation + EMA | an RSB-A2 variant, `[θ|m|v|G|E]` | same as R1 |

Script: `tests/resume_known_answer.sh`. ⚠ Scope note: the fp8 driver's checkpoint write
(`VerifiedTrain.lean` ~line 4840) is still in-place, and `scoreCheckpoint` still refuses BN nets even
though `.bn` now exists — both are follow-ups, not blockers.

---

## 5. Order of work

**▶ DECIDED (Brett, 2026-09-14):** finish the EfficientNet run; then implement the **staggered
respawn** (§3b, with §3a Level 0 and the §3e prerequisites) and the tweaks judged worthwhile from
§2/§4; then test it **on this box with the R34-2018 bf16 job**, which is wanted anyway. The same
slowdown has shown on ares on that job — R34's shim is flip-only, so the fault is **not**
AutoAugment-specific, which again points at the multi-process loader arrangement.
✅ **DONE 2026-09-17.** The conf was written as R34's counterpart of `r50-2018-bf16` —
`scripts/jobs/r34-default-bf16-4gpu.conf`. ⛔ Named `default`, **not** `2018`: §2c N1 forces `RECIPE`
to equal the second dash-field, R34's recipe slug is `default` (there is no
`generated_resnet34_imagenet_2018_shim.py`), and `supervise.sh` refuses a mismatch, so
`r34-2018-bf16-4gpu` would have been rejected at launch. `r34-default-4gpu.conf` was deliberately
left at fp32 — it is the lakefile's Ch. 5 ImageNet row and the reproduction path for the book's
published 74.14 / 91.86. The run: `runs/2026-09-16-r34-bf16-90ep/`, and the advice below was
followed — `loader_rss.tsv` logged throughout, respawn OFF, loaders given **21.9 h** of life.

⚠⚠ **AND THE RESULT CONTRADICTS THE PARAGRAPH ABOVE.** "The same slowdown has shown on ares on that
job" — it did **not** reproduce here: 90 epochs, 21.9 h of continuous loader uptime, pace flat, arena
plateaued at hour 2, growth uniform across all four loaders (§3b has the numbers). One of these is
true and both are on record, so the difference has to be named before either is quoted:
* the ares observation may have been the same **host-memory** confound this run hit and correctly
  excluded (five epochs at 972–1055 s that tracked `MemAvailable` 126 → 87 GiB and self-recovered —
  a co-tenant job, confirmed). If nobody was logging `MemAvailable` on ares, that is indistinguishable
  from a loader fault, and it is exactly what it looks like.
* or the fault is **box-dependent** (ares is 4× 4060 Ti on 32 threads; this is 4× 3060 on 24), in
  which case the flip-only/heavy-aug framing in §3b is the wrong axis and core count or thread
  oversubscription is the right one.
▶ Neither is established. ⛔ Do not cite "it shows on R34 too" as support for the multi-process
hypothesis until the ares run is re-examined for host memory over its slow window.

**While the EfficientNet run finishes (now → ~2026-09-16 00:00 UTC):**
- Loader-RSS logger running (`enet-loaders` unit → `runs/2026-09-12-enet-verified-350ep/loader_rss.tsv`,
  one row per loader per epoch). It gives the production divergence curve between restarts — the
  target §2's harness must reproduce.
- No builds or soaks: the feed is CPU-bound and shares the box.

**After it lands:**
1. Archive the run (`RESULTS.md`, curve CSV, `epoch_clock.tsv`, `loader_rss.tsv`) and write §7.
2. §4 R1–R6 on the GPUs (~2–3 h including rebuilds).
3. §2 arm A (10 h), then B/C/D as needed — sequentially, on an otherwise idle box.
4. Implement from §2's verdict (`SHIM_*` pass-through and/or §3 Level 0).
5. ConvNeXt: `REST_EPOCHS` kept as the safety net until a full run shows it is not needed.
