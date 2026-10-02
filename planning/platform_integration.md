# Platform integration suite: "this GPU platform works with this build"

Written 2026-10-01. Steps 1-2 of §7 (tiers 0-2, manifest, `PLATFORMS.md`) landed 2026-10-02 with a
CUDA baseline; tier 3 is open. Goal: one script, run by hand about monthly (and later from a
self-hosted runner with no changes), that answers whether a GPU platform (CUDA, ROCm, Intel XPU)
runs a given build of this repo, records the answer in the repo, and separates driver/plugin
breakage from breakage in our code. Near-term target: the Imagenette demos on an Intel Arc Pro
B60 through Intel's OpenXLA PJRT plugin.

## 1. Patterns borrowed

| pattern | source | what we take |
|---|---|---|
| conformance suite + recorded submission | Khronos Vulkan/OpenCL CTS, MLPerf `system_desc.json` | a versioned suite; every run filed with a machine-readable system manifest |
| per-platform expectation files | Mesa `deqp-runner` (`*-fails.txt`, `*-flakes.txt`) | pass = "matches this platform's baseline"; new failure = regression, unexpected pass = update the baseline |
| device-generic tests | PyTorch `instantiate_device_type_tests` / OpInfo, JAX `jtu.skip_on_devices` + dtype tolerances | each test written once; skips, xfails and tolerances are data keyed by (backend, dtype) |
| perf history with system info | Phoronix / OpenBenchmarking, LLVM LNT | ms/step and peak memory reported against prior runs on the same hardware, never gated |
| goldens with a `--check` generator | already ours (`regen_verified_mlir.sh`, the cert generators) | op-level reference outputs committed, regenerated only on purpose |

A new backend's baseline may start mostly red; the signal is the baseline shrinking.

## 2. Layout

```
scripts/platform/check.sh [--tier 0|1|2|3] [--backend cuda|rocm|xpu] [--plan]
scripts/platform/goldens.py [--check]          # JAX-on-CPU reference outputs for tier 2
scripts/platform/tolerances.tsv                # backend | dtype | op family | atol | rtol
scripts/platform/expected/<backend>-<gpu>.txt  # known fails / flakes, Mesa style
runs/platform/<date>-<host>-<backend>/
    manifest.json   # see section 4
    results.tsv     # test | tier | PASS/FAIL/XFAIL/XPASS/SKIP | measured | bound
    logs/           # stays on the box; not committed (scripts/gates/repo_shape.txt)
PLATFORMS.md        # generated: one row per (platform, core)
```

`--plan` prints what would run and the expected wall time and launches nothing (the
`lake run <job> plan` convention). Tier 3 is never the default.

## 3. Tiers

| tier | runs | time | a failure here points at |
|---|---|---|---|
| 0 platform | vendor SMI (`nvidia-smi` / `rocm-smi` / `xpu-smi`), PJRT plugin dlopens, `GetPjrtApi` version vs our compile-options header, client create, device count, one tiny compile + execute via `libpjrt_ffi.so` | seconds | driver, plugin, install |
| 1 shim | `ffi/test_pjrt_guards.c`, `test_pjrt_compile_check.c`, `test_pjrt_allreduce.c`, `test_pjrt_dp.c` (the last two SKIP below 2 devices) | ~1 min | our runtime vs this plugin |
| 2 ops | a fixed set of `verified_mlir/` artifacts, one per op family, f32 and bf16, run on fixed inputs and compared to the JAX-CPU goldens under `tolerances.tsv` | minutes | backend codegen (bf16 conv/dot result-type traps live here) |
| 3 training | MNIST MLP N steps (loss falls), a short R34 Imagenette smoke, `residency_gate_all.sh`, `sharded_eval_gate.sh` (multi-device only), a determinism check under `det_shim.sh` | ~30 min | end to end |
| report | median ms/step and XLA peak memory for 2-3 nets | - | trend only |

Tiers 0-2 need no datasets, so they run on borrowed cloud boxes. Before every run the script
rebuilds `ffi/libpjrt_ffi.so` (the gcc line in `ffi/README.md`) and the gate binaries; a stale
binary has printed green before.

## 4. Driver vs core split

`manifest.json` records two halves:

- core: repo SHA, `libpjrt_ffi.so` hash, Lean toolchain, gate-binary hashes, `verified_mlir`
  manifest hash (`scripts/gates/gen_mlir_manifest.py`).
- platform: GPU model and count, kernel driver version, CUDA / ROCm / oneAPI runtime version,
  PJRT plugin name + version + hash, kernel version.

`PLATFORMS.md` is generated from every `runs/platform/*/manifest.json` + `results.tsv`. When a run
turns red, diff it against the last green row: core equal and platform changed means the driver
or plugin; platform equal and core changed means us. To bisect on purpose, pin one half and swap
the other. The first failing tier narrows the layer further.

## 5. Backends

| backend | state | first step |
|---|---|---|
| CUDA | trains today (4x 4060 Ti) | tiers 0-3 to produce the first baseline |
| ROCm | code pruned; `pjrt_ffi.c` still has the plugin path and the `is_rocm` branch | tiers 0-1 on the future MI300 cloud box |
| Intel XPU | Intel's OpenXLA plugin (reported to target JAX 0.11.2; we pin 0.11.0) | section 6 |

## 6. Intel Arc Pro B60

The shim is plugin-agnostic: `$PJRT_PLUGIN` names any PJRT C API `.so` and the trainers only hand
it StableHLO text. Places that assume CUDA or ROCm:

1. Pinned host memory dlopens `libamdhip64` / `libcudart` directly. Needs a Level Zero path, or
   falls back to unpinned (slower, still correct).
2. Platform-name branches (`PJRT_Client_PlatformName`): add an XPU case or a safe default.
3. `ffi/pjrt_compile_options.h` is generated against our pinned XLA. A plugin built against
   0.11.2 should accept it inside the C API's compatibility window; tier 0 checks it.
4. The JAX reference side needs its own venv + lockfile with Intel's plugin (never the main
   `.venv`), so reference and verified runs can be compared on the same card.

Hardware notes: 24 GB covers every Imagenette demo. Battlemage XMX does bf16, so the bf16 renders
apply, but watch for the kernel-selection trap (a bf16 op correct and slower than f32); tier 2's
report row shows it per op. Dual-B60 boards would exercise the DP and sync-BN collectives.

Pre-purchase check, no Intel GPU needed: install Intel's wheel into a scratch venv, dlopen the
plugin, call `GetPjrtApi`, compare its API version to ours. Client creation then fails for lack of
a device, which is expected.

## 7. Order of work

1. ✅ Tier 0 + 1, the manifest, the `PLATFORMS.md` generator. CUDA baseline on this box
   (3× 4060 Ti, 14/14, 11 s). As built: tier 0 = `scripts/platform/probe.c` (bare plugin:
   API version, client, devices) + `smoke.c` (one compile + execute through the shim); tier 1 =
   the four `ffi/test_pjrt_*.c` against fixtures in `scripts/platform/fixtures/` plus
   `verified_mlir/cifar8_adamdp_train_step.mlir` for the 2-replica compile. The plugin reports
   PJRT API 0.114 against our vendored header's 0.90; same major, so reported, not failed.
2. ✅ Tier 2: 15 artifacts, seven op families, f32 and bf16 (`scripts/platform/tier2_artifacts.tsv`);
   `tier2.py goldens [--check]` on XLA:CPU, `tier2_run.c` through the shim on the device,
   `tolerances.tsv`. ~3 min on one 4060 Ti. What it took to make the comparison mean something,
   all in `tier2.py`'s docstrings: train steps compare `out − in`; the Adam state is m = 0, v = 1,
   lr = 1 (random small v makes the update sign-like and amplifies noise); per-output errors are
   floored (analytically-zero gradients) and gated on median + p90, not max (deep batch-BN nets
   at random init are chaotic — XLA:CPU against itself at 1 + 1e-6 spreads p90 0.6 in bf16).
   Fault injection is recorded in `tolerances.tsv`'s header; the two deep bf16 rows are thin.
3. Tier 3: wrap the existing gate scripts.
4. Intel pre-purchase check (section 6), then the B60 shim items 1-2 behind tier 0.
5. ROCm tiers 0-1 when the MI300 box exists.
6. Later: a `workflow_dispatch` job on a labelled self-hosted runner calling the same script.

## 8. Open

- Tier 2's deep bf16 rows catch a single swapped kernel by only 1.5-1.8×. A better-conditioned
  probe (trained weights rather than random init, or the forward alone for bf16) would widen it.
- Whether tier 3's Imagenette smoke uses a cached subset so the suite has one dataset dependency.
