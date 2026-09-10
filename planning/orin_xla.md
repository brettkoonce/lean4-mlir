# Orin: the verified StableHLO path on the device GPU (XLA / PJRT)

**Opened 2026-09-10.** Companion to `planning/orin_rerun.md` (the TensorRT detector) — this thread is
the OTHER half of "8 GB is enough for Lean + verified ML": the proof-rendered trainers themselves,
run through XLA on the Orin Nano's GPU. Convs were dead there until today.

## §0 Why convs were dead

The stock aarch64 `jax-cuda12-plugin` pulls the **SBSA** cuDNN wheel, which ships sm_50…sm_120 and
**skips sm_87**. Every conv model died at `CUDNN_STATUS_EXECUTION_FAILED`; dense models never
noticed (cuBLAS was fine).

⛔⛔ **CORRECTED 2026-09-10 — see `planning/orin_plugin_rebuild.md`.** This section originally read
"JetPack 6.2's Tegra cuDNN 9.3.0.75 is the only cuDNN with sm_87 conv kernels", and that is wrong.
It generalised from SBSA wheels to all of aarch64. NVIDIA ships a separate **Tegra**
(`linux-aarch64`) cuDNN line reaching **9.20**, with sm_87 throughout — verified on the device at
9.12.0.46. The distinction is `linux-sbsa` (no sm_87) against `linux-aarch64` (sm_87), NOT a
version ceiling. Tegra CUDA likewise reaches 12.9.79; the real cap is the DRIVER (12060 here),
which limits what RUNS, not what builds. Everything measured in §3 below stands — the 9.3 build
works — but 9.3 was never the only option, and the plugin should be rebuilt on current jax.

## §1 The plugin (built 2026-09-10 on the training box)

`~/lean/klawd_max_power/jax-orin-build/dist/xla_cuda_plugin.so` — 197,284,744 B, md5
`22aaa262472275bf1be68787f6b2b7a4`; provenance and the rebuild recipe in the README next to it.
jax `jax-v0.4.38`, hermetic CUDA 12.6.0 / cuDNN **9.3.0** as the **Tegra** redists, `sm_87,compute_87`
(34 cubins + 34 PTX modules), NCCL off. Built natively for arm64 inside `nvcr.io/nvidia/l4t-jetpack:r36.4.0`
under QEMU user-mode on the x86 box (8.9 h); there is no x86→aarch64 CUDA cross-compile in that jax,
and no Jetson-built plugin exists on any index. Loads as PJRT C API 0.58; its only dynamic deps are
libc/libstdc++ — CUDA and cuDNN are dlopen'd, so with no pip `nvidia-*` packages it binds the system
Tegra libraries, which is the intended pairing.

⚠ The ship gate as first written (`cuobjdump --list-ptx | grep compute_`) is a no-op on every binary:
cuobjdump names PTX `*.sm_87.ptx`. Count `PTX file` lines instead.

## §2 The working environment on the Orin

```
export PJRT_PLUGIN=/home/skoonce/ckpt/xla_cuda_plugin.so
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu
export LEAN_MLIR_LOWERER=xla LEAN_MLIR_PREALLOCATE=0 LEAN_MLIR_MEM_FRACTION=0.15
```

`MEM_FRACTION` is a narrow window because the Orin's memory is UNIFIED: the device BFC pool and the
pinned d2h staging draw on the same 7.4 GB. 0.25 starves the host ("failed to alloc 134217728 bytes on
host"), 0.10 starves the device, 0.15 works for conv models; dense models are happy at 0.25.
`PREALLOCATE=0` is required. ⛔ Installing the pip SBSA cuDNN 9.7 wheel re-breaks convs.

## §3 Results on the Orin (clocks pinned, proof-rendered StableHLO, GPU)

| demo | epoch | result |
|---|---|---|
| mnist-linear-verified | ~2.4 s | 92.10% (12 ep) |
| mnist-mlp-verified | 3.65 s | 97.87% (12 ep) |
| **mnist-cnn-verified** | 9.2 s | **98.68%** (10 ep) |
| **cifar-verified** | 18.5–24 s | **64.45%** (10 ep) |
| cifar8-bn-verified | — | OOM-killed, §4 |

One-time compiles: 44.8 s (cnn_train_step, 11 outputs), 29.5 s (cifar8_bn_train_step, 38 outputs).
Board: Orin Nano 8 GB, L4T R36.4.7, 25 W. A shipped trainer is a ~5 MB binary linking 8 objects and
zero Mathlib; checking a proof module costs ~550 MB anonymous (2.7 GB RSS, 2.2 GB of it mmap'd
Mathlib oleans, evictable); the full `lake build` completes 2251/2251 on the board.

## §4 The cifar8-bn footprint — measured on both boxes

The Orin's kernel OOM-killed `cifar8-bn-verified` ~80 s after compile at **5.93 GB anonymous RSS**
(total-vm 16.7 GB). The same binaries on the training box (RTX 4060 Ti, jax 0.11.0 plugin), RSS
sampled once a second for the whole run:

| demo | train step | steady anon RSS (x86) | peak total RSS | Orin |
|---|---|---|---|---|
| mnist-cnn-verified | 11 outputs | 1.50 GB, flat | 1.94 GB | fits |
| cifar-verified | 14 outputs | 2.26 GB, flat | 2.68 GB | fits |
| cifar8-bn-verified | 38 outputs | **2.12 GB, flat** | 2.52 GB | **5.93 GB anon → killed** |

So the 5.93 GB is NOT the executable's own footprint — on x86 the 38-output step is no heavier than
the 14-output one, and nothing grows after the first ten seconds. The extra ~3.8 GB is Orin-specific:
either unified-memory accounting (the BFC pool and pinned buffers counted as the process's anon
pages, which `MEM_FRACTION` moves but does not shrink), or the Dec-2024 XLA in jax 0.4.38 holding
more than the 2026 one does. The discriminating probe is on the device: sample `RssAnon` of
`mnist-cnn-verified` (which fits) and compare with 1.50 GB here — if the Orin reads ~2.8× on that
too, it is accounting; if it reads ~1.5 GB, the excess is specific to the 38-output step.

## §5 Side finding, training box, 2026-09-10 — cifar-verified collapses at epoch 15

Running `cifar-verified` for its default 40 epochs here: 42.3 → 68.31% by epoch 14, then **10.00%
(chance) from epoch 15 to 40**. Fixed LR, no BN, batch 128. The Orin's 64.45% is a 10-epoch run and
matches epoch 10 here (65.39%). Not documented anywhere I could find; not investigated. If the demo
is quoted at 40 epochs anywhere, it is quoting a number the run does not reach.
