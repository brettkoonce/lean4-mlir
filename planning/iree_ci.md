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
* `regen_verified_mlir.sh tests` (steps 2–3): exits 1 when `iree-compile` is not on PATH, and
  when any of the nine files prints FAILED / skipped or no `compile OK`. Measured on llvm-cpu: all
  nine pass, 26 committed `verified_mlir/` artifacts (every net's fwd + SGD/AdamW train steps,
  ViT-Tiny, the twelve cifar8 optimizer variants), 2 min, 2.9 GB peak. Negative checks: no compiler
  → rc 1; `IREE_EXTRA_FLAGS=--no-such-flag` → all nine red. In `iree.yml` after the gradchecks,
  plus `git diff --exit-code verified_mlir/` (TestCifar8AdamTrain re-renders its artifacts).
* VJP oracle, whole (step 5): `libiree_ffi.so` built on the runner from the IREE v3.11.0 runtime
  (runtime submodules only, four archives — 3.11 moved printf into `libprintf_printf.a` — and
  `-DUSE_CPU`), cached on the version + wrapper hash; `tests/vjp_oracle/run.sh` on
  `make_tiny_mnist.py`'s set via `$VJP_ORACLE_DATA`. 14/14 pass at the no-GPU tolerances, worst
  step-2 Δ 2.0e-6 (uib); the same numbers locally with the box's CUDA-build FFI on local-task.
  Control: dense's phase-3 trace against dense-relu's phase-2 trace fails the differ.

## Next, one at a time (each measured on llvm-cpu before it joins the job)

1. ~~TestSDPA~~ — landed, see above.
2. ~~`regen_verified_mlir.sh` tests loop~~ — landed, see above.
3. ~~`iree-compile` smoke over a subset of `verified_mlir/`~~ — the step-2 loop is that subset
   (26 artifacts incl. the 224² train steps, ~2 min). A wider sweep only if something slips past it.
4. ~~Move jax.yml's "Forward ties through IREE" step here~~ — `iree.yml` job `forward-ties`
   (MNv4 at batch 32, MNv2 ImageNet + its BN-ε control); jax.yml's `timm-parity` no longer
   installs IREE.
5. ~~VJP oracle phase 3~~ — `iree.yml` job `vjp-oracle`, see above.
