# IREE CUDA big-model reduction hang — investigation handoff

**Status: SOLVED 2026-06-21.** It was NOT an IREE codegen bug. The FFI runtime
`ffi/libiree_ffi.so` was linked against IREE runtime commit **32b074edda (Apr 20)**,
which is 8 days OLDER than the compiler wheel **af030e43 (rc20260428, Apr 28)**. The
Apr-20 CUDA runtime hangs on the big train-step at the 3rd invocation; Apr-28 fixes it.
The "canonical compiler rc20260428 + runtime 32b074edda" pairing in the rest of this doc
is a **skewed** config — a *matched* runtime was never tested because the only standalone
runner (`iree-run-module`) was ABI-NEWER (`rrIiii`) and wouldn't load the wheel's `rriiii`
vmfbs. Everything below the line is the (now-superseded) isolation log; keep it for the
method, ignore its "IREE codegen" conclusion.

## How it was proven (decisive)

1. Compiled the mobilenetv2 adam train-step to a standalone vmfb with the **same wheel**
   (`iree-compile … --iree-hal-target-backends=cuda --iree-cuda-target=sm_86`).
2. Dumped the EXACT step-3 FFI inputs via an env-gated hook (`IREE_DUMP_STEP=3`) added to
   `lean_iree_mlp_train_step_v` in `ffi/iree_lean_ffi.c` → `/tmp/dump_{x,params,y}.bin`.
   The inputs were numerically **clean** (no NaN/Inf; x∈[-2.1,2.6], params∈[-1.8,13.4]).
3. Replayed those exact bytes + the vmfb through pip
   **`iree-base-runtime==3.12.0rc20260428`** (= af030e43) via python `iree.runtime`:
   ran fine for 8 calls, **even with fresh per-call device uploads** mimicking the C FFI
   alloc/free pattern. Same vmfb + same data + matched runtime ⇒ NO hang. So the vmfb,
   the data, and our C invoke pattern are all innocent; only the runtime *version* differs.

## The fix

Rebuild the FFI runtime to match the compiler:
```bash
cd ~/src/iree && git checkout af030e43d8343263a6c869eae32f958f229ff7af
cd ~/src/iree-build
.venv/bin/ninja iree_runtime_unified flatcc_runtime flatcc_parsing   # ninja = project .venv's
# then relink ffi/libiree_ffi.so per IREE_BUILD.md §4 (new .so is rriiii, matches wheel vmfbs)
```
After the swap the hang cleared on every big trainer (runtime-level, not model-specific):
`mobilenetv2` → step 100/295 loss 2.20 (invoke #112), `resnet34` invoke #7, and
**`vit-verified-adam` invoke #7** (loss 3.51→3.32→2.88) — so ViT was the same stale-runtime
bug, not an attention-codegen bug. efficientnet/convnext share the runtime. Old hang was
always at invoke #3. **Lesson:** pin the from-source FFI runtime commit to the SAME release as
the `iree-base-compiler` wheel — extends the pip-lock rule to the FFI runtime.

Diagnostic harness (parse the 739-arg signature, split the packed
`[θ|m|v|lr|bc1|bc2|bnStats]` blob, replay): `/tmp/{run,loop,replay}_mbv2.py`. Env-gated hooks
left in place: `IREE_DUMP_STEP=N` (dump inputs at invocation N) and `IREE_FFI_TRACE=1`
(log each invoke + output-pop), both no-ops when unset.

**Date opened:** 2026-06-21 (ares, 6× RTX 4060 Ti, CUDA 12.9, driver 575.57.08).

---

<details><summary>Superseded isolation log (concluded "IREE codegen bug" — WRONG; kept for method)</summary>

---

## Symptom

Every **big** verified-IREE Imagenette trainer (`resnet34/mobilenetv2/
efficientnet/convnext-verified-adam`, 224×224) hangs at **step 3** of training:

```
step 0/295: loss=2.506...    ← computes correctly
step 1/295: loss=2.486...
step 2/295: loss=2.407...
step 3: <hang>               ← GPU pegged 100% util, mem-util 0% (compute spin),
                                P2/boost, no throttle = a stuck/non-terminating
                                kernel. Sometimes presents host-side (1% util)
                                depending on codegen flags.
```

Process ignores SIGTERM (busy in native IREE code) — needs `kill -9`.

## Confirmed working (controls)

- **`cifar8-verified` / `cifar8-verified-adam` / `cifar8-bn-verified-adam`** — all
  TRAIN fine (small 32×32 spatial). So config, `.so`, compiler, driver, GPU are
  all healthy.
- **jax 0.10 path trains R34/ImageNet** (big, 224px, heavy GPU) fine → **hardware
  is innocent**; big-model TRAINING is unblocked via jax.

## What it's NOT (ruled out, with evidence)

| ruled out | how |
|---|---|
| Adam optimizer | `cifar8-verified-adam` trains |
| Batch norm presence | `cifar8-bn-verified-adam` trains |
| GPU hardware / state | jax big-model works; rebooted; cifar works |
| GPU "poison" | survived a reboot |
| checkpoint-resume | fresh (no resume) still hangs |
| our FFI config / locks | CUDA+ROCm in sync (jax 0.10, iree-cc rc20260428) |
| process contention | hangs solo, single PID |
| **codegen pipeline** | hangs identically with `reduction-vector-distribution=false`, `prefetch-num-stages=1`, `stream-partitioning-favor=debug` (no concurrency), AND `vectorize-pipeline` (legacy) |
| **compiler/runtime version** | hangs on iree-cc **rc20260420, rc20260428, rc20260530** (matched + skewed). On **3.11.0** it doesn't hang — it *fails to compile* (see below) |
| `[0,2,3]` reduction *shape* | splitting all 364 `reduce[0,2,3]`→`[2,3]∘[0]` did NOT clear the hang (the `[2,3]` over 112×112 is still huge) |

## Leading hypothesis

**IREE's CUDA backend mis-codegens / hangs on large-spatial-dim reductions** (the
BN mean/var over `B×C×112×112`). cifar's tiny reductions dodge it. The hang is in
a kernel, not our renderer (changing kernel codegen moves the symptom 100%↔7%↔1%
but never clears it).

The `[0,2,3]` thread (red herring for the hang, real for compile):
- **3.11.0** can't compile `reduce[0,2,3]` over NCHW — `'func.func' op failed to
  distribute` = **iree#24282** (already in `upstream-issues/`, "fixed" in
  rc20260428).
- **3.12rc** compiles it, but the resulting CUDA kernel **hangs at scale**. So the
  #24282 "fix" traded a compile error for a runtime hang on CUDA at large sizes.
- Splitting to `[2,3]∘[0]` (the forward/d_x already do this) does NOT help → it's
  large reductions generally, not the `[0,2,3]` shape.

## Next diagnostic steps (for the clean session)

1. **Pin the exact stuck dispatch — pure IREE, no FFI.** Compile a big train-step
   to a standalone vmfb and run under `iree-run-module --trace_execution`:
   - `~/src/iree-build/tools/iree-run-module` exists.
   - The train-step takes ~100+ tensor inputs (whole param set) — auto-generate
     `--input="<TxCxHxW>xf32=0.01"` specs by parsing the func signature in
     `verified_mlir/mobilenetv2_adam_train_step.mlir` (regex the `func.func(...)`
     arg list). A python parser was started; the signature regex needs `util.func`
     vs `func.func` care.
   - If iree-run-module **hangs** → confirms pure-IREE bug (clean upstream repro);
     the last traced op = the culprit dispatch. If it **doesn't** hang → the bug
     is in our FFI's HAL/queue setup (different story).
   - Full dispatch-tensor tracing (`--iree-flow-trace-dispatch-tensors`) works but
     is ~30s/dispatch (dumps 14M-float tensors); too slow to reach a late dispatch.
     Filter markers at the source with `grep --line-buffered`.
2. **File upstream** (`iree-org/iree`) with the standalone vmfb/MLIR repro + the
   isolation table above. Mirror it into `upstream-issues/2026-06-iree-cuda-
   bigmodel-trainstep-hang/` (template = any existing folder there; note: this is
   a *runtime hang*, not a compile error like the others).
3. **Interim:** big-model training uses the jax path; IREE-verified stays
   cifar-scale.

## A debug hook worth re-adding (was reverted)

`LeanMlir/Types.lean:ireeCompileArgs` — add an `IREE_EXTRA_FLAGS` env passthrough
(space-separated) so you can iterate iree-compile flags without a Lean rebuild:
```lean
let envFlags ← (IO.getEnv "IREE_EXTRA_FLAGS").map (·.getD "")
let envArgs := ((envFlags.splitOn " ").filter (· ≠ "")).toArray
return baseArgs ++ chipArgs ++ extraArgs ++ envArgs ++ #["-o", outPath]
```

## Key facts for the new session

- **Run recipe (CUDA):** from repo root, `export PATH="$PWD/.venv/bin:$PATH";
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64; export IREE_BACKEND=cuda;
  CUDA_VISIBLE_DEVICES=0 .lake/build/bin/<trainer> data`. (Mask AER cards: use
  0,2,3,4; cards 1/5 = bus02/bus62 hard-reset under load.)
- **Verified trainers read a pre-generated `verified_mlir/*.mlir`** (built by
  `tests/Test*TrainPC.lean` via the proof renderers in `LeanMlir/Proofs/
  StableHLO.lean` + `CnnRender.lean`), then iree-compile it. Editing
  `MlirCodegen.lean` does NOT affect them (that's the non-verified path).
- **Where the `[0,2,3]` reductions are emitted (verified path):**
  `LeanMlir/Proofs/StableHLO.lean` (bulk: forward `smr`/`vsr` mean/var + backward),
  `LeanMlir/Proofs/CnnRender.lean` (5 backward sites). The forward + d_x already
  split; d_gamma/d_beta + the forward mean/var do not.
- **vmfb cache:** `.lake/build/<model>_*.vmfb`. Clear to force recompile after a
  codegen/flag change.
- **Operational gotchas learned this session:**
  - `pkill -f "<pattern>"` **kills its own shell** (the pattern matches the pkill
    command line) — use `kill -9 $(pgrep -f "build/bin/...")`.
  - Foreground `sleep` is blocked in the harness shell — use background tasks /
    poll loops.
  - These trainers **ignore SIGTERM** — always `kill -9`.
  - rc20260530 runtime is a **universal regression** (hangs ALL trainers + can
    wedge GPU state) — do NOT use it. Canonical = compiler rc20260428 + runtime
    built from IREE `32b074edda` (`ffi/libiree_ffi.so`).

## Config state at handoff (all canonical, nothing committed)

- jax **0.10.0** (CUDA + ROCm locks agree); iree-base-compiler **3.12.0rc20260428**
  (CUDA + ROCm agree); runtime `ffi/libiree_ffi.so` built from IREE `32b074edda`.
- Working tree clean (all debug edits reverted). `~/src/iree` checked out to
  `32b074edda`.

See memory `project_verified_imagenette_sweep.md` for the blow-by-blow.

</details>
