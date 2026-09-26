# iree_ci.md — IREE checks in CI

PJRT/XLA trains everything; IREE is the second compiler. Its checks ran only by hand and rotted
(three gradchecks hardcoded `--device=hip` after the ROCm box left; `regen_verified_mlir.sh`'s
smokes print FAILED and exit 0 because `.venv/` has no `iree-compile`). `.github/workflows/iree.yml`
is the loop: IREE 3.11.0 from PyPI, `IREE_BACKEND=llvm-cpu` on `local-task`, no GPU, no FFI build.

## Step 0 (measured 2026-09-26, llvm-cpu, PyPI 3.11.0)

| check | local time | result |
|---|---|---|
| TestMHSA | 3.6 s | PASS, rel 1.9e-3 |
| TestViTBlock | 3.9 s | PASS, rel 9.6e-4 |
| TestViTTiny | 4.4 s | PASS, rel 1.7e-4 |
| TestSDPA | 3.5 s | FAIL, rel 0.27 — same on CUDA; a defect in the test, not the platform (fixed below) |
| control: TestMHSA at `IREE_BACKEND=rocm` | — | no PASS line (red, as required) |

## Landed

* `runFn` (GradcheckHelpers) takes its device from `IREE_BACKEND` (`cuda`/`hip`/`local-task`).
* `iree.yml`: the gradchecks, gated on their `✅ PASS` line, plus the control.
  Push on the listed paths, nightly, and on demand.
* Harness precision: `runFn` passes inputs and reads outputs as raw f32 files (`--input=…=@x.bin`,
  `--output=@y.npy`). The text round trip (`Float.toString`'s six decimals in, IREE's six
  significant digits out) had put a ~3e-4 absolute floor under every finite-difference quotient at
  ε = 1e-3; through files it is f32's ~1.5e-5. Rel err after: MHSA 1.5e-4, ViTBlock 9e-6,
  ViTTiny 1.4e-5 (were 1.9e-3 / 9.6e-4 / 1.7e-4).
* TestSDPA (step 1): the backward was right. Seed 0 draws a direction nearly orthogonal to the
  gradient (directional derivative 1.7e-3, median ~0.5 over seeds 0–19), so even the f32 floor is
  1% of it. The test now uses seed 1 (rel 5.6e-5); seeds 1–19 all pass (1.4e-5 to 1.9e-3).

## Next, one at a time (each measured on llvm-cpu before it joins the job)

1. ~~TestSDPA~~ — landed, see above.
2. `regen_verified_mlir.sh` tests loop: fail on a missing compiler instead of printing FAILED.
3. `iree-compile` smoke over a small, fixed subset of `verified_mlir/` (time the 224² train steps
   on CPU first; the full set is ~200 artifacts).
4. Move jax.yml's "Forward ties through IREE" step here.
5. VJP oracle phase 3 (needs a CPU-mode `libiree_ffi.so` source build, cached): last, only if
   the build fits a runner.
