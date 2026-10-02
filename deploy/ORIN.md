# The Jetson Orin: building for it, setting it up, running on it

One page for the Orin Nano 8 GB, from a fresh JetPack install to a result checked back in.
Three things run on the board, and they share the setup below:

| what | route | detail |
|---|---|---|
| the platform suite | PJRT plugin + `scripts/platform/check.sh` | §3 |
| the verified trainers (proof-rendered StableHLO on the GPU) | PJRT plugin + Lean | §4 |
| the VisDrone detector | ONNX → TensorRT | §5, `ORIN_SMOKE_TEST.md` |

The PJRT plugin is the part that does not come off a shelf: no aarch64 jax wheel carries
sm_87 kernels for a Jetson, so it is built on the training box (§1).

## §1 The plugin — built on x86, for sm_87

`deploy/build_orin_pjrt_plugin.sh` does the whole build on any x86 Linux box with docker
and about 80 GB of disk. No GPU is involved: CUDA and cuDNN arrive as redist tarballs, the
plugin links CUDA stubs, and the real libraries are dlopened on the Orin.

```bash
bash deploy/build_orin_pjrt_plugin.sh ~/orin-plugin-build     # starts detached
tail -f ~/orin-plugin-build/build_011.log
```

How it works: jax has no x86→aarch64 CUDA cross-compile, so the build runs natively for
arm64 inside `nvcr.io/nvidia/l4t-jetpack:r36.4.0` under QEMU user-mode. The container's
hostname is `tegra-build` because `rules_ml_toolchain` picks the Tegra (`linux-aarch64`)
redists only when `uname -a` contains "tegra"; the alternative, `linux-sbsa`, has no sm_87
in cuDNN and every convolution fails at run time with `CUDNN_STATUS_EXECUTION_FAILED`.

### The current plugin (the 0.11.1 stack)

| | |
|---|---|
| source | `jax-v0.11.1`, bazel 7.7.1, `--wheels=jax-cuda-pjrt` |
| CUDA / cuDNN | toolkit 12.6.2 (component 12.6.77) / cuDNN 9.12.0.46, all `linux-aarch64`; zero `linux-sbsa` fetched |
| target | `sm_87,compute_87`; NCCL and oneDNN off; clang 18 host, hermetic NVCC |
| artifact | `pjrt_c_api_gpu_plugin.so`, 308,376,976 B, md5 `0881a0e28c3c63cd1919e3a44cf55ec6` |
| wheel | `jax_cuda12_pjrt-0.11.1.dev0+selfbuilt-…-manylinux_2_27_aarch64.whl`, md5 `394103cd6615d599c4dd67a982573925`; its `jax_plugins/xla_cuda12/xla_cuda_plugin.so` is the artifact byte for byte |
| ship gate | 47 sm_87 cubins, 47 PTX modules, no other architecture |
| where | training box, `~/lean/klawd_max_power/jax-orin-build/dist011/` |
| built | finished 2026-09-13, 95,308 s of bazel (26.5 h) for 18,690 actions |

Ship gate, on any box with a CUDA toolkit:

```bash
cuobjdump --list-elf pjrt_c_api_gpu_plugin.so | grep -oE 'sm_[0-9]+' | sort | uniq -c   # 47 sm_87
cuobjdump --list-ptx pjrt_c_api_gpu_plugin.so | grep -c 'PTX file'                      # 47
```

⛔ `--list-ptx | grep compute_` prints nothing for every binary ever built. cuobjdump names
PTX entries `*.sm_87.ptx`, so count `PTX file` lines.

The first plugin, `jax-v0.4.38` with cuDNN 9.3, is `dist/xla_cuda_plugin.so` beside it. It is
the one `planning/orin_xla.md` §3 measured. It is superseded, and kept only because its
numbers are on record.

### Build traps, each of which cost a run

- **`--experimental_worker_for_repo_fetching=off` is required under QEMU.** Without it, bazel 7
  deadlocks right after the last redist is fetched. The tell: about 3% CPU, nothing written to
  the bazel root for ten minutes, and a JVM thread dump with zero RUNNABLE threads. The
  script sets the flag. `planning/orin_plugin_rebuild.md` §5 has the diagnosis.
- **A reused container keeps its CPU cap.** `docker start jaxbuild` inherits whatever
  `--cpus` the container was created with, and `--jobs` cannot exceed it. Check
  `docker inspect jaxbuild --format '{{.HostConfig.NanoCpus}}'` before launching. It
  should be 0 or the core count, and `docker update --cpus=N jaxbuild` raises it live. The
  2026-09-13 build ran its first nine hours on 8 cores, so its 26.5 h is an upper bound
  and not the expected time.
- **The python driver can die after bazel succeeds.** In that build `build.py` was killed
  (`EXIT=137`) while the bazel server ran on and finished the wheel. If the log ends
  in `Build completed successfully` but `dist011/` has only the wheel, take the `.so` out of
  the wheel with `unzip -p`.
- **Prove the redists are Tegra.** Count `linux-sbsa` (must be 0) and `linux-aarch64`
  (must not be 0) in the log of the run that *fetched* them. A rerun on a warm bazel root
  fetches nothing, so it proves nothing. Use awk or `command grep`, since a `.gitignore`-aware
  grep skips `*.log`.

### Rebuilding as a burn-in test

The rebuild is a deterministic, all-cores, many-hour job with an exact acceptance check.
On a new box, run the script into a fresh workdir. The build passes when:

1. the log fetched only `linux-aarch64` archives, `cudnn-linux-aarch64-9.12.0.46_cuda12` among them;
2. the ship gate reads 47 sm_87 cubins and 47 PTX modules;
3. the plugin passes §3 on the Orin.

A matching md5 would be a bonus and is not a requirement, since bazel does not promise
byte-identical output across hosts. Record the wall clock with the core count beside it.

## §2 Device setup, from a fresh JetPack

Board as last measured: Orin Nano 8 GB, L4T R36.4.7 (JetPack 6.2), 25 W mode, CUDA driver
12.6. The training box must be reachable as `trainingbox` (an `~/.ssh/config` entry). The
plugin comes from there.

```bash
sudo apt install -y git build-essential python3-numpy
git clone <repo> ~/lean4-jax-mlir && cd ~/lean4-jax-mlir
bash deploy/orin_setup.sh            # → ~/pjrt: plugin, cuDNN 9.12 (Tegra), orin_env.sh
. ~/pjrt/orin_env.sh                 # every shell that runs XLA
sudo jetson_clocks                   # before any timing; record `sudo nvpmodel -q`
```

`orin_setup.sh` copies the plugin from the training box and checks its md5. It downloads
cuDNN 9.12.0.46 for `linux-aarch64` from NVIDIA's redist and checks its sha256, runs the
ship gate when `cuobjdump` is present, and writes the env file. It is safe to rerun.

The env file is the whole contract:

| variable | value | why |
|---|---|---|
| `PJRT_PLUGIN` | `~/pjrt/pjrt_c_api_gpu_plugin.so` | the shim dlopens this |
| `LD_LIBRARY_PATH` | `~/pjrt/cudnn912/lib` first | the plugin was compiled against cuDNN 9.12; JetPack ships 9.3, and XLA refuses an older runtime cuDNN |
| `LEAN_MLIR_LOWERER` | `xla` | the trainers' backend |
| `LEAN_MLIR_PREALLOCATE` | `0` | required: device and host share one DRAM |
| `LEAN_MLIR_MEM_FRACTION` | `0.15` conv models, `0.25` dense | 0.25 starves the host on conv models and 0.10 starves the device (`planning/orin_xla.md` §2, measured on the 0.4.38 plugin) |

⛔ Do not `pip install` jax's CUDA plugin or any `nvidia-cudnn-*` wheel on the board. Those
are SBSA builds, and with one on the path the convolutions break again.

For the TensorRT route (§5), also set up `pycuda` in a venv (`~/orinvenv`). `trtexec` is at
`/usr/src/tensorrt/bin/trtexec`, not on `PATH`.

## §3 The platform suite — "does this board run this build"

The suite needs only gcc and numpy, no Lean. Run it first on any new install or new plugin:

```bash
. ~/pjrt/orin_env.sh
scripts/platform/check.sh --plan     # what will run; launches nothing
scripts/platform/check.sh            # tiers 0-2 → runs/platform/<date>-<host>-cuda/
```

- Tier 0 covers the plugin loading, its PJRT API version, the device list and one compile +
  execute. Tier 1 covers the shim's guards. Tier 2 runs 15 `verified_mlir` artifacts
  against XLA:CPU goldens that are committed in the repo.
- With one GPU, tier 1's `compile_dp`, `allreduce` and `dp` report SKIP. That is the
  expected result.
- On a Jetson, `check.sh` defaults `LEAN_MLIR_PREALLOCATE=0` and `LEAN_MLIR_MEM_FRACTION=0.15`
  when they are unset.
- A test the Orin is known to fail goes in `scripts/platform/expected/cuda-<gpu>.txt`, so it
  reports XFAIL instead of FAIL. The `<gpu>` part is the slug the run prints. Never loosen a
  tolerance in `tolerances.tsv` to get the Orin green.
- The run directory's `manifest.json` and `results.tsv` are the record. Bring them back
  (§6), and `PLATFORMS.md` regenerates from them.

## §4 The verified trainers on the GPU

The full `lake build` completes on the board, and a trainer is a ~5 MB binary. Fetch
Mathlib's cache rather than building it:

```bash
lake exe cache get
lake build mnist-cnn-verified && ./run.sh mnist-cnn-verified
lake build cifar-verified     && ./run.sh cifar-verified
lake run mnist                       # linear, MLP and CNN in sequence
```

On record from the 0.4.38 plugin, clocks pinned (`planning/orin_xla.md` §3): mnist-cnn
98.68% and cifar 64.45% at 10 epochs. `cifar8-bn-verified` was OOM-killed at 5.93 GB
anonymous RSS on the 8 GB board, against 2.12 GB on x86. Re-test it on the 0.11.1
plugin (`planning/orin_plugin_rebuild.md` §5).

⚠ Run `cifar-verified` for 10 epochs, not its default 40. On the training box it reaches
68.31% at epoch 14 and drops to chance from epoch 15 (`planning/orin_xla.md` §5).

## §5 The VisDrone detector — TensorRT

The detector does not use the plugin. Its route is ONNX exported on the training box, then
`trtexec --fp16` on the device. Result on record: 55.9 fps end to end on the u8 engine with
clocks pinned. `ORIN_SMOKE_TEST.md` is the step-by-step brief with its acceptance numbers,
and `README.md` in this directory covers the export.

## §6 Bringing work back

The Orin has no git credentials. Commit on the device and ship the commits as patches:

```bash
git format-patch origin/main..HEAD -o ~/orin-patches/
scp -r ~/orin-patches trainingbox:lean/klawd_max_power/
```

They are reviewed and rebased onto `main` on the training box. A run on the device goes
under `runs/<date>-orin-<what>/` with a README naming the plugin md5, the env file's
values, `nvpmodel -q`, whether `jetson_clocks` was applied, and the wall clock. Logs stay
out of git, as everywhere else in `runs/`.

A device-side change must not alter what any other platform runs. A memory-saving op or
a smaller batch for the 8 GB board is an Orin option or job, opt-in, so that the default
renders stay byte-identical. The render guard checks that.

## §7 Next: the demos on the board, from a live sensor

The second level is the chapter demos running on the Orin from a generic camera instead of
a test frame. In order:

1. **A generic input source in `orin_detect.py`.** Replace the `--camera` stub with one
   `--source` that takes an image file, a directory, a video file, a V4L2 index (any USB
   camera) or a GStreamer string (`nvarguscamerasrc …` for a CSI sensor), all through
   `cv2.VideoCapture`. Preprocessing is unchanged: squash to 448 and feed the u8 engine.
   Report per-stage times in a rolling window, so a live run gives the same three-stage
   split as `--bench`.
2. **NEU-DET.** The steel-defect detector is the VisDrone R34+FPN graph verbatim: 448 px,
   three anchors per level, the same ten-slot class one-hot with six slots live. The ONNX
   signature, `trtexec` and the engine runner carry over unchanged. Three things in
   `orin_detect.py` are VisDrone-specific today: `CLASS_NAMES`, `ANCHORS` (duplicated from
   `demos/MainYolov1VisdroneFpn.lean`) and the golden frame. Make them a `--profile
   visdrone|neudet` table. Take NEU's anchors from `data/neu_det/anchors_fpn_{p3,p4,p5}.txt`
   (what `demos/MainYolov1NeuDetFpn.lean` reads), and regenerate a NEU golden with
   `export_onnx.py --regen-golden` before `--verify-frame`. The acceptance check is the same
   as VisDrone's: the engine reproduces the Lean stack's detection count and top score on
   the golden frame.
3. **Further demos**, each the same pattern of export → gate on a golden → engine → `--source`.
   Classifiers are the simpler case: one logit vector, no decode.
