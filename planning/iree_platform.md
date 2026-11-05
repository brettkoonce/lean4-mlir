# iree_platform.md — the IREE row up to the bar the GPU rows meet

Started 2026-11-04, on branch `iree-platform` (rebased onto origin/main a437a8b3 2026-11-05).
Nothing here has run.
The row exists (planning/platforms_ladder.md §5, `support.tsv`, Table A-2); what it lacks is
records. Every cell it has is either pre-verified-renderer (April–May, hand-written MLIR on the
7900 XTX), a one-step oracle, or a ten-step CPU smoke. The `-iree` twins have never been run as
tiers on a GPU against the current certified bytes. This plan does that, at the bar the Arc row
met, and fixes the text that still describes the old state.

## 0. Where it stands (origin/main a437a8b3, 2026-11-05)

| rung | what is on record | what it is | gap |
|---|---|---|---|
| suite | n/a: "admitted by the `-iree` twins" | `check.sh` is PJRT-only (plugin load, shim guards, goldens) | no IREE-mode admission; §3 S7 |
| mnist | `historical/traces/` (2026-04-21): MLP + CNN training traces on IREE over ROCm and CUDA, step-1 losses at f32 ULP vs JAX; CI `iree.yml` one-step oracle (linear/MLP/CNN, llvm-cpu, 3.11.0) | hand-written `MlirCodegen` path, not the verified renders; oracle is one step | no `lake run mnist-iree` transcript on current renders |
| cifar | `historical/IREE.md` (2026-04, ROCm, the CIFAR CNN); `runs/ablation_cifar8/` (9 arms, narrow head, cosine+warmup, July); `runs/ablation_cifar8w/` (SGD arm only, wide head, July) | old recipe (cosine), old per-image-BN renders; not the chapter's constant-LR six arms | no `cifar-iree` at the book's recipe; `cifar8wb-bbn-ablation` (the chapter's headline net) is filtered out of the twin |
| imagenette | `runs/2026-05-27-convnext-tiny-ablations` (ROCm); `runs/mobilenetv2_verified_crop_gpu0.log` 86.81 %, `runs/efficientnet_verified_crop_gpu1.log` 87.65 % (June 14, CUDA, 80 ep, June renders); `runs/2026-10-28-imagenette-iree-llvmcpu/` all seven lower and step (10 steps, 3.13.0, CPU) | the 80-epoch ones predate D23/§7 rerenders; the 10-28 run is lowering, not training | no 80-epoch IREE number for any net on its current certified bytes; R34, R50, MNv4, ConvNeXt, ViT never trained 80 epochs through IREE on verified renders |
| imagenet | n/a | IREE shim refuses DP (`mkSessionDp`); no collectives | out of scope, permanently |

Version skew today: the row and Appendix A say **IREE 3.11**; CI pins **3.11.0**; the CPU stack
(`deploy/iree_cpu_setup.sh`) is **3.13.0** because 3.11.0's `local-task` segfaults on every 224²
artifact; ares has `iree-compile` **3.12.0rc20260428** outside the repo
(`~/lean4-mlir/.venv`, `~/lean4-mlir-job/iree-bin`) and an `ffi/libiree_ffi.so` from 2026-09-26.
The repo's own `.venv` has no `iree-compile`, so `runDemoGroup` and `ireeCompilePath` fall through
to `$PATH`.

## 1. The bar

The same one the Arc row met (deploy/ARC_B50.md § Measured), per rung, one seed each:

* **mnist**: the three nets' finals against CUDA/XLA's 30-seed medians (`runs/2026-09-01-mnist-n30`).
* **cifar**: the six arms of `cifar8w-ablation` + `cifar8w-bn-ablation` at the chapter's recipe
  (40 ep, constant LR), each inside CUDA's five-seed range (`runs/2026-09-01-cifar8w-6arm-constlr`);
  plus the batch-BN arms if D4 lands.
* **imagenette**: all seven at 80 epochs, each inside CUDA's five-seed range or within two images
  of it, wall clock and ms/step recorded.

A miss is a finding, not a failure of the plan: a net outside the range gets a step-1 tie
(`LEAN_MLIR_NO_SHUFFLE=1`, shared init, as `historical/traces/` did) before anything else is said.

## 2. Decisions (recommendation first)

* **D1 — one IREE version everywhere: 3.13.0.** It is the oldest release that runs the 224²
  artifacts on the CPU and passes the bbn gradient gate on `llvm-cpu` (`ireeCompileArgs`'
  comment). Move CI (`iree.yml`'s three `3.11.0` pins), the row header, `platforms.tex` and
  Appendix A/B to it; the 3.11 miscompile and segfault become history lines, not row caveats.
  Alternative: keep 3.11.0 in CI and 3.13.0 on boxes — two versions in one row.
* **D2 — where the IREE stack lives on ares: `.venv-iree`, not `.venv`.** `.venv` is the pinned
  JAX env (jax/requirements-cuda-lock.txt); the IREE wheels pull nothing JAX needs but the lock
  discipline says one env per lock. `ireeCompilePath` prefers `.venv/bin/iree-compile`, so either
  teach it `$IREE_COMPILE` / `.venv-iree` or put `.venv-iree/bin` on `$PATH` in `runDemoGroup`.
  Alternative: install into `.venv` as `iree_cpu_setup.sh` does on atlas (simplest; breaks the
  lock-file rule on the training box).
* **D3 — the row stays one row, cells say the target.** "IREE 3.13 · CUDA (4060 Ti, sm_86 PTX)"
  in each cell's tag, CPU records as a second entry in the same cell. Splitting into IREE-CUDA /
  IREE-CPU rows doubles a row that has one box per target.
* **D4 — `cifar8wb-bbn-ablation` rejoins `cifar-iree`.** It was excluded for 3.11's llvm-cpu
  miscompile; on 3.13 that gate passes, and on CUDA it needs
  `--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`, which `ireeCompileArgs`
  applies only for `rocm`. Extend that branch to `cuda`, check the ResNet-34 step does not slow
  (the rocm comment says it did not there), drop the filter. Alternative: leave the twin at four
  arms and say so in the row.
* **D5 — admission stays "by the twins" for now.** An IREE mode for `check.sh` (§3 S7) is the
  right end state but is new code; the twins plus the smoke table are an honest admission today.

## 3. Steps

Each step ends in a run directory or a commit; nothing past S2 launches without the user's word.

* **S1 — the GPU stack on ares (≈ 1 h, no GPU time).** `deploy/iree_cpu_setup.sh` gains a
  `--target cuda` path (or a sibling `deploy/iree_cuda_setup.sh`): wheels at 3.13.0 into the D2
  env, the runtime at the wheel's commit with the CUDA HAL driver, `ffi/libiree_ffi.so` without
  `-DUSE_CPU`. Record `runs/<date>-iree-cuda-setup/manifest.txt` as the CPU setup did. The old
  `.so` is moved aside, not deleted. Note: PTX targets `sm_86` (the default chip; 3.11 had no
  `sm_89`, memory note) — re-check whether 3.13 takes `IREE_CHIP=sm_89` and whether it is faster.
* **S2 — the smoke table on CUDA (≈ 1–2 h, one card).** The 10-28 CPU protocol, on the GPU:
  `LEAN_MLIR_LOWERER=iree LEAN_MLIR_G2_STEPS=10 LEAN_MLIR_MAX_EPOCHS=1 LEAN_MLIR_CKPT_TAG=smoke`
  for the three mnist, three cifar (incl. bbn per D4) and seven imagenette binaries.
  `results.tsv` columns as `runs/2026-10-28-imagenette-iree-llvmcpu/` plus **median ms/step**
  (median, not mean — memory: the mean measures a host leak). That ms/step × steps is the
  per-net ETA for S4, replacing the guesses in §4.
* **S3 — mnist-iree and cifar-iree, full (≈ minutes + a few hours, one card).** `lake run
  mnist-iree`, `lake run cifar-iree`, transcripts teed to `runs/<date>-<tier>-iree-cuda/`, scored
  against §1. Ask before launching cifar if its S2 ETA is over an hour.
* **S4 — imagenette-iree, 80 epochs, seven nets (the long one; ask first).** One net per card,
  four at a time, longest first (ConvNeXt, R34, R50 on the first wave). Fresh `LEAN_MLIR_CKPT_TAG`
  so nothing collides with the XLA checkpoints (the IREE ckpt name already drops `_xla`). One
  `runs/<date>-imagenette-iree-cuda/` with a `results.tsv` (top-1, top-5, wall, ms/step, vs CUDA
  range) and a README in the 10-28 shape.
* **S4b — optional: the same tiers over ROCm on the 7900 XTX in ares.** The card is back
  (ROCm 10.1 row, deploy/ROCM_7900XTX.md, 2026-11-01). `IREE_BACKEND=rocm` (chip `gfx1100`, the
  reduction flag already applied) needs a ROCm-HAL build of the shim beside S1's CUDA one; one
  card, so the seven nets run in series. It fills the row's "over ROCm" with current records
  instead of April's.
* **S5 — records and text (one commit after S3, one after S4).** `support.tsv` cells (mnist,
  cifar, imagenette) → `platforms_md.py` and `--tex` regenerate PLATFORMS.md and
  `figures/platforms.tex`; Appendix A's IREE paragraph (§5 list); the `imagenette-iree` docstring
  if anything it says moved.
* **S6 — optional: re-anchor `lake run benchmark`'s IREE column.** Its Imagenette `refSec` rows
  are the 7900 XTX on hand-written renders (lakefile's own staleness caveat). That card is in
  ares again, so S4b's wall clocks re-anchor the column on the same card and the certified
  bytes, from medians. It moves published per-chapter estimates, so it is its own decision.
* **S7 — optional: `check.sh --lowerer iree`.** Tiers 0–2 for IREE: the shim loads and reports
  `iree`; `mkSessionDp` refuses loudly; the committed tier-2 artifacts compile and match the
  XLA:CPU goldens through IREE. Turns D5's "admitted by the twins" into a suite cell.

## 4. Cost

| step | card-hours | wall | notes |
|---|---|---|---|
| S1 | 0 | ≈ 1 h | runtime submodules ≈ 240 MB; disk is at 98 % on ares, check first |
| S2 | ≈ 1–2 | ≈ 1–2 h | compiles dominate (minutes per arch, CPU-bound) |
| S3 | ≈ 2–4 | same | mnist minutes; cifar's six arms on IREE were 2038 s vs 888 s XLA on the 7900 XTX |
| S4 | ≈ 50–80 (guess) | ≈ 1–1.5 d on 4 cards | June CUDA IREE runs: MNv2 8.7 h, B0 8.5 h at 80 ep on one 4060 Ti; 7900 XTX IREE/XLA ratios 3.8× (MNv2) to 9× (R34); R50/MNv4 never timed; S2 replaces this row |

The cards are shared with the paused MNv4 500-ep JAX half (~65 h left); S4 and that resume
cannot both have the box. The user picks the order.

## 5. Stale text to fix (S5), as of origin/main a437a8b3

* `blueprint/src/content.tex` ≈ l.20035: "the Imagenette twin runs five of the seven nets, since
  ResNet-50 and MobileNetV4 have no IREE driver" — all seven since 2026-10-28, and it was never a
  driver gap (the `imagenette-iree` docstring already says so).
* Same paragraph: "IREE 3.11's `llvm-cpu` miscompiles the pooled train steps" — true of 3.11,
  fixed by 3.13 (D1); say which release.
* `scripts/platform/support.tsv` IREE row and `blueprint/src/figures/platforms.tex` (generated):
  "IREE 3.11" in the platform name; the mnist/cifar cells point at April records.
* `planning/platforms_ladder.md` §5: "five of the seven imagenette nets (ResNet-50 and MobileNetV4
  have no driver)" — superseded; add a line pointing here rather than editing the history.
* `lakefile.lean` `cifar-iree` docstring: the bbn exclusion reason, if D4 lands.
* `.github/workflows/iree.yml`: the `3.11.0` pins (D1), and the vmvx-vs-llvm-cpu note.

## 6. Out of scope

ImageNet on IREE (no collectives; the shim refuses DP by design). 80-epoch training on the CPU (days per net at
the 10-28 rates; the CPU cell stays a lowering record). Any change to proofs or renders beyond
D4's compile flag: IREE consumes the same `verified_mlir/` bytes, and that is the point.
