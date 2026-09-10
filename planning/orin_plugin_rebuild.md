# Orin: rebuild the PJRT plugin on current jax — the cuDNN 9.3 ceiling was wrong

**Opened 2026-09-10, from the device session.** Supersedes the constraint that shaped
`planning/orin_xla.md` §0-§1. Companion to `planning/orin_rerun.md` (the TensorRT detector).
⛔ Nothing here is built yet: this is the brief for a fresh session.

## §0 The correction, and it inverts the whole premise

The 2026-09-09 build pinned `jax-v0.4.38` with hermetic cuDNN **9.3.0**, on the reasoning that
"JetPack 6.2's Tegra cuDNN is 9.3.0.75 and that's the only cuDNN with sm_87 conv kernels — NVIDIA
caps Tegra there, and every SBSA cuDNN ≥ 9.7 has sm_50…sm_120 but skips sm_87."

**The second half is right and the first half is not.** It generalised from `linux-sbsa` wheels
and JetPack's apt feed to all of aarch64. NVIDIA publishes a separate **Tegra** (`linux-aarch64`)
cuDNN line that runs to **9.20**, and it carries sm_87 throughout. Verified on the device with
cuDNN **9.12.0.46**, `linux-aarch64`:

| library | architectures |
|---|---|
| `libcudnn_cnn.so.9.12.0` | sm_75 80 86 **87** 90 100 101 |
| `libcudnn_engines_precompiled.so.9.12.0` | sm_53 61 62 70 72 75 80 86 **87** 90 100 101 |
| `libcudnn_engines_runtime_compiled.so` | sm_75 80 86 **87** 90 100 101 |
| `libcudnn_ops.so.9.12.0` | sm_75 80 86 **87** 90 100 101 |

The two engine libraries are exactly what threw `CUDNN_STATUS_EXECUTION_FAILED` on the SBSA 9.7
attempt. ⭐ **`linux-sbsa` lacks sm_87; `linux-aarch64` has it.** That one distinction was the
entire blocker, and it was never a version ceiling.

⚠ Also wrong: "no CUDA 12.8 for Orin." Tegra CUDA redists reach **12.9.79**. The real ceiling is
the **driver** — this board reports 12060, i.e. CUDA 12.6 — and a driver caps what can RUN, not
what can build.

## §1 Build

jax ≥ 0.6 declares cuDNN ≥ 9.8, which is a jax policy and is satisfied by the Tegra redists.
jax 0.11 states "all versions of CUDA 12.1 or newer remain supported", so pin CUDA to the driver
rather than gambling on minor-version compatibility on Tegra.

⚠ Two of the version strings first written here are not redist keys and would have aborted at
the fetch phase. `rules_ml_toolchain` keys its `json_dict` on the **toolkit** version, and
`12.6.77` is the NVCC that ships inside toolkit **12.6.2** (`redistrib_12.6.77.json` is a 404).
The cuDNN JSONs are three-part: `redistrib_9.12.0.json` carries `"version": "9.12.0.46"` and
resolves to `cudnn-linux-aarch64-9.12.0.46_cuda12-archive.tar.xz`, the artifact §0 measured. The
corrected recipe, and the one actually run:

```bash
git clone --depth 1 --branch jax-v0.11.1 https://github.com/jax-ml/jax.git jax011
cd jax011
python3 build/build.py build --wheels=jax-cuda-pjrt \
    --cuda_version=12.6.2 \
    --cudnn_version=9.12.0 \
    --cuda_compute_capabilities=sm_87,compute_87 \
    --clang_path=/usr/lib/llvm-18/bin/clang \
    --python_version=3.12 \
    --disable_nccl --disable_mkl_dnn \
    --bazel_path=/work/bin/bazel-7.7.1
```

⭐ The fetch log settles it: toolkit **12.6.2** hands back archives named
`cuda_nvcc-linux-aarch64-**12.6.77**-archive.tar.xz`, so `12.6.77` was the component version all
along and the two numbers name the same toolkit. Zero `linux-sbsa` in the log, and cuDNN arrives
as `cudnn-linux-aarch64-9.12.0.46_cuda12-archive.tar.xz` — the §0 artifact, byte for byte.

Two more things 0.11.1 changed out from under the 0.4.38 recipe. Its `.bazelversion` is **7.7.1**,
so the 6.5.0 binary at `/work/bin/bazel` will not do; `bazel-7.7.1-linux-arm64` is staged beside
it. And `setup.py` declares `python_requires>=3.12` while the JetPack container ships 3.10, so
`--python_version=3.12` is required — without it `build.py` defaults the hermetic Python to
whatever ran the CLI. `build.py` itself still runs fine on the container's 3.10.

▶ Same container trick as the 2026-09-09 build, and it is what keeps the hermetic rules on Tegra
redists: `nvcr.io/nvidia/l4t-jetpack:r36.4.0` under QEMU with `--hostname tegra-build`. XLA's
`_get_platform_architecture` returns the Tegra flavour only when `uname -a` contains "tegra".
**Confirm `linux-aarch64` and zero `linux-sbsa` in the fetch log** — that check is the one that
proved the last build was on the right redists.

✅ Settled: `--wheels=jax-cuda-pjrt` is still the one. In 0.11.1 it resolves to
`//jaxlib/tools:jax_cuda12_pjrt_wheel`, and `jax-cuda-plugin` is the separate Python kernels
wheel as before.

✅ Also settled: the Tegra selection survived the move of hermetic CUDA out of XLA and into
`rules_ml_toolchain`. `gpu/nvidia_common_rules.bzl` still shells out to `uname -a` and looks for
the substring `tegra`, and its `_REDIST_ARCH_DICT` still maps `linux-aarch64` to the Tegra triple
and `linux-sbsa` to the plain aarch64 one.

Ship gate unchanged, and it is the one that matters:

```bash
cuobjdump --list-elf xla_cuda_plugin.so | grep -oE 'sm_[0-9]+' | sort -u   # -> sm_87
cuobjdump --list-ptx xla_cuda_plugin.so | grep -c 'PTX file'               # -> > 0
```

⛔ Do NOT use `--list-ptx | grep compute_`: cuobjdump names PTX entries `*.sm_87.ptx` and the PTX
text says `.target sm_87`, so that predicate is a no-op on every binary ever built.

## §2 Runtime side, already staged on the device

JetPack ships only cuDNN 9.3, so the 9.12 libraries travel with the plugin. Downloaded and staged
at `/home/skoonce/pjrt/cudnn912/lib` (796 MB) and it goes **first** on `LD_LIBRARY_PATH`. Unlike
the SBSA 9.7 attempt this is the correct pairing rather than a workaround, because these libraries
actually contain sm_87.

## §3 The CPU plugin — ✅ ALREADY BUILT, do not rebuild blind

The device session asked for this too. It was built on the training box 2026-09-10 from the
`jax-v0.4.38` tree's pinned XLA, target `//xla/pjrt/c:pjrt_c_api_cpu_plugin.so`:

| | |
|---|---|
| artifact | `~/lean/klawd_max_power/jax-orin-build/dist/pjrt_c_api_cpu_plugin.so` |
| size / md5 | 150,203,168 B / `55495cc91ab1f6303c26af58d75f8d08` |
| built | 4,464 actions, QEMU arm64, `--config=linux --config=linux_arm64` (`-march=armv8-a -mtune=generic`) |
| verified | `GetPjrtApi` exported, dlopens under emulation as **PJRT C API 0.58** |
| deps | libc, libm, libstdc++, libgcc_s — **zero CUDA linkage** |

⭐ The baseline `-march=armv8-a` is deliberate: the Orin is Cortex-A78AE (armv8.2) and a Pi 4 is
A72 (armv8-a), so building for the Orin's own ISA would produce something that runs there and
SIGILLs on a Pi. ⚠ One loose end: a grep for non-baseline opcodes returned 2 unidentified hits.
Resolve before shipping to a Pi 4; a Pi 5 (A76, armv8.2) is unaffected either way.
⚠ Container glibc is 2.35 (Ubuntu 22.04): fine for Raspberry Pi OS bookworm (2.36), **not** for
bullseye (2.31).

Rebuild recipe if 0.11.1's XLA is wanted instead: `jax-orin-build/build_cpu_plugin.sh`, and note
standalone XLA ships only a **3.11** requirements lock (3.10 fails at analysis).

The device measured XLA-CPU running `mlp_train_step` at **14.5 ms/step**, about 42% of the Orin
CPU's peak FLOPs, against the 192 ms/step IREE-CPU row in `historical/BENCHMARK.md` on a much
larger Xeon. It also avoids the ~5 GB CUDA resident footprint that currently limits the device to
one detector frame at a time.

## §4 What this supersedes

* `planning/orin_xla.md` §0 and §1 — the "cuDNN 9.3 is the only cuDNN with sm_87" claim. The
  measured results in that doc (mnist-cnn 98.68%, cifar 64.45%) stand; only the explanation of
  why 9.3 was required is wrong. The 9.3 build works, it simply was not the only option.
* Any note saying Tegra CUDA stops before 12.8.

## §5 Open, in order

1. Build on `jax-v0.11.1` per §1; confirm Tegra redists in the log; run the ship gate.
   `deploy/build_orin_pjrt_plugin.sh` does all of it from scratch on **any** x86 Linux box with
   docker: registers the qemu-aarch64 binfmt handler, pulls the JetPack image, installs clang-18
   and bazel 7.7.1 inside it, clones the source and launches the build detached. It takes a
   workdir argument and needs roughly 80 GB. ⭐ No GPU of any kind is required — the CUDA
   toolkit and cuDNN arrive as redist tarballs and the plugin links against CUDA **stubs**
   (`--config=cuda_libraries_from_stubs`), dlopening the real libraries only on the Orin. An
   AMD-only box builds this CUDA plugin fine, which is what makes farming it out possible.
   ⚠ The whole build is emulated (aarch64 under qemu-user), so budget 9-11 h on a 32-thread
   host and expect it to saturate every core it is given.
2. Re-run `planning/orin_xla.md` §3's demos on it and compare — same accuracy, and whether the
   newer XLA moves ms/step.
3. `cifar8-bn-verified` OOM'd at 5.93 GB anon on the 8 GB board. Re-test: a newer XLA may hold
   less, and §3's note about the CUDA resident footprint suggests the CPU plugin may fit where
   the CUDA one does not.
4. Ship the CPU plugin to the device and to a Pi; settle the §3 ISA question first.
