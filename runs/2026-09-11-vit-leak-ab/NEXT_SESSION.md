# Next session: land the fix, regenerate the shims, then re-measure

Read `README.md` in this directory first — its last section ("The fix, and what was under it")
is the state of play. This file is only what is left to do.

## What is done (2026-09-11, branch `fix/shim-prefetch-owner-thread`)

* **The retention is understood and fixed.** It was mimalloc lazily freeing huge blocks that a
  pool thread allocated and the main thread dropped. `VerifiedTrain.readInto` + main-thread
  buffers: RSS flat, `LazyFree` 0, ~11 page faults/step. Standalone repro in `repro/`.
* **The p90 tail is understood.** One producer paced the trainer because the runtime shims in
  `jax/.lake/build/` predate `4a0a2781` and still default tf.data determinism ON. With
  `SHIM_DETERMINISM=0` every wait is 0 ms at 6 and 8 producers.
* **Instruments that stay:** `LEAN_MLIR_PROBE_DUMP=<file>` (per-step `step ms wait_ms`, plus a
  per-read `READ h= step= issued= start= end=` trace on stderr), `run_fix.sh` (RSS / LazyFree /
  faults / per-producer CPU against the step counter), `pipe_sizes.py`.

## Step 1 — land it

The branch has two code files (`ffi/f32_helpers.c`, `LeanMlir/VerifiedTrain.lean`) and this
directory. `tests/prefetch_tie.sh` is the gate; its result is recorded in `README.md` and
`prefetch_tie_postfix.log`. Fast-forward onto main, no merge commit.

## Step 2 — the runtime artifacts on this box (DONE 2026-09-11, keep the precheck)

`scripts/regen_jax_generated.sh box` is the precheck (a diff, instant) and `… sync` copies the
committed `jax/generated/` set into `jax/.lake/build/`. Run on 2026-09-11: 68 of 78 artifacts
were stale — the 39 shims AND the emitted JAX trainers (`generated_vit_tiny_imagenet.py` was 4
lines behind). ⚠ `scripts/gen_shims.sh` is NOT the tool: it re-emits only the 10 default-recipe
shims. ⚠ Only `vit-default-emabf16-4gpu.conf` calls the `box` precheck; every other conf will
happily train on a stale file. The README records the confirmation probe taken with no
`SHIM_DETERMINISM` in the environment after the sync.

## Step 3 — the 40 ms that is left (optional, but it is 15 % of every ImageNet job)

Determinism off: median 253–257 against a 205–209 minimum at 4, 6 and 8 producers, all waits 0.
Contention on the trainer's own step, not the feed. Candidates, cheapest first:

1. Lower the producers' scheduling priority (`nice` in `spawnShim`, or `SHIM_PYTHON` pointing at
   a wrapper). The README records what the wrapper measured.
2. Pin the H2D staging: the 308 MB batch goes host→device from pageable memory every step; under
   heavy producer memory traffic that staging copy is the obvious victim.
3. Cap tf.data's parallelism per producer (`private_threadpool_size`) so 6 producers do not
   present 28+ runnable threads in bursts.

Whatever is tried, `min` is the floor (205–209) and the target is `med ≈ min`.

## Step 4 — re-measure everything the two bugs invalidated

In this order:

1. Re-run the 2026-09-10 probe for all 14 confs and rewrite the `ETA=` strings.
   `runs/2026-09-10-bf16-probe-4060ti/README.md` has the invocation; `scripts/probe_to_eta.py`
   refuses a variant mismatch. Use the **mean** now — with the leak gone and determinism off, the
   mean is the job again (mean − median was 12–16 ms in every det-off arm here).
2. Set `SHIM_WORKERS=6` in `vit-default-4gpu.conf` (measured best here; 4 is at the feed limit,
   8 buys nothing) and re-check the other confs' worker counts under determinism off — they were
   all chosen with it on.
3. Re-run the four bf16 "does not pay" verdicts on `bf16/job-confs`, MobileNetV4 first.
4. Only then decide bf16-as-default with an f32 fallback.

⚠ Ask before chaining these: 14 confs × 2 precisions × ~4 min is over an hour of 4-GPU load.

## Traps that cost time today

* ⛔⛔ `pgrep -f <pattern>` matches the shell that is running the `pgrep` when the pattern is in
  that shell's own command line. It bit twice more today (a wait loop that never ended, and a
  phantom "shard-less shim in `pipe_read`" that was my own bash). `pgrep -x <comm>` or a pid file.
* ⛔ The trainer logs steps 0, 1, 2 and then every 100th; a trigger on `step 20/` never fires.
* ⛔ A Lean C helper compiled without `-DNDEBUG` keeps `lean.h`'s asserts, and an assert failure
  prompts `(C)ontinue, (A)bort, (S)top` on stdin — under a pipe that is an indefinite hang with
  no output. Read stderr.
* ⚠ `RssAnon` cannot distinguish "referenced" from "lazily freed"; `LazyFree` in
  `/proc/PID/smaps_rollup` can, in one line.
* ⚠ `jax/.lake/build/*.py` are build products of `jax/`, not of the root `lake build`, and the
  tracked twins in `jax/generated/` being current says nothing about them.
  `scripts/regen_jax_generated.sh box` before any ImageNet run; its header describes the 45 GPU-h
  this class of bug cost once already.
