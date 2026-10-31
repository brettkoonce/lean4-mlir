# ROCm 7.2.4 against ROCm 10.1 on minerva's 7900 XTX — step time, and what broke — 2026-10-25

One Radeon RX 7900 XTX (gfx1100), kernel 7.0.0-38, BIOS 5601, the slot at Gen3 through a riser —
the configuration the card later dropped off the bus in (`../../deploy/ROCM_7900XTX.md` §2; the
drop cut this bench short at ConvNeXt). `bench.sh` is the run; `results.tsv` its rows.

## The arms

| arm | libraries | plugin |
|---|---|---|
| `724` | the pinned apt ROCm 7.2.4 in `/opt/rocm` | `jax-rocm7-pjrt` 0.11.0 in the pinned `.venv` |
| `pip101` | ROCm 10.1 as wheels, `rocm[libraries,device-gfx1100]==10.1.0` in `.venv-rocm10` | `jax-rocm10-pjrt` 0.11.1+rocm10.1.0, same venv |
| `apt101` | ROCm 10.1 as apt debs (`amdrocm10.1-gfx1100`, `stable.repo.amd.com`), `LD_LIBRARY_PATH=/opt/rocm-7.2.4/core-10.1/lib` | the `pip101` plugin |
| `…sw` | the same, with the four MIOpen switches below | |

ROCm 10 is TheRock's build of the userspace — HIP, hipBLASLt, MIOpen, the compiler — shipped three
ways (wheels, tarball, debs) from one build: both 10.1 arms carry TheRock commit `b324869f`, but
their `libMIOpen.so.1` differ (the wheel is built for gfx1100 only, the deb for every target).
Nothing of it touches the kernel driver, so the arms swap per process; `/proc/<pid>/maps` of an
`apt101` run shows HIP, hipBLASLt and MIOpen loaded from `core-10.1`. The `.venv-rocm10` recipe:

```bash
python3 -m venv --without-pip .venv-rocm10
python3 -m pip --python .venv-rocm10/bin/python3 install \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple \
  "rocm[libraries,device-gfx1100]==10.1.0" "jax-rocm10-pjrt==0.11.1+rocm10.1.0"
PJRT_PLUGIN=$PWD/.venv-rocm10/lib/python3.12/site-packages/jax_plugins/xla_rocm10/xla_rocm_plugin.so ./run.sh …
```

The MIOpen switches (`sw`):

| variable | what it turns off | why |
|---|---|---|
| `MIOPEN_DEBUG_CONV_GEMM=0` | the GEMM convolution solvers (im2col + matrix multiply) | ViT's patch-embed weight gradient falls through `GemmFwdRest` (no workspace) to `MIOpenIm2d2Col`'s `EXTREME_LARGE` branch, which does not compile under HIPRTC (`get_global_id` undeclared) — `historical/upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`. Also R34 bf16's `GemmWrw: Not enough workspace` |
| `MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS=0` | the grouped implicit-GEMM forward kernels | bf16 garbage on gfx1100, which has no XDLOPS |
| `MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS=0` | their weight-gradient pass | the same |
| `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS=0` | their data-gradient pass | the same |

## Step time, ms/step (`LEAN_MLIR_MAX_STEPS=100`, warm-up 8; every arm from an empty MIOpen database)

| net | 7.2.4 | 10.1 pip | 10.1 pip + sw | 10.1 apt | 10.1 apt + sw |
|---|---|---|---|---|---|
| ResNet-34 | **78** | 88 | 89 | 88 | 89 |
| MobileNetV2 | 114 | 110 | 119 | **109** | 118 |
| ViT-Tiny | fails | fails | **71** | fails | **71** |
| EfficientNet-B0 | 126 | **121** | 130 | **121** | 129 |
| ConvNeXt-T | — card off the bus at step 3 (13:58:35), §2 of the runbook | | | | |

ResNet-50 and MobileNetV4 were not reached. pip and apt 10.1 agree to 1 ms everywhere. 10.1 is
13 % slower than 7.2.4 on ResNet-34 and 3–4 % faster on the depthwise nets; the switches cost
7–8 % where they change anything. A run the night before (40-step probes, the switches on, a warm
7.2.4 database) had ConvNeXt at 161 / 244 / 177 ms and ResNet-50 at 134 / 161 / 151 ms
(7.2.4 / 10.1 + sw / 10.1) — single probes on a card that was dropping, so indicative only.

## The finding that matters: an empty MIOpen database breaks ViT on 7.2.4 too

Each arm here got its own empty `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR`, and **7.2.4's
ViT fails exactly as 10.1's does** (`724/vit-verified-adam.log`). Every earlier 7.2.4 ViT run on
minerva — the suites, the 80-epoch runs — used the default database in `~/.config/miopen`, whose
stored find results keep MIOpen off the broken solver. 10.1's MIOpen is a different version and
keys a different database, so 10.1 always started cold; "10.1 breaks ViT" (the night before's
reading) was "an empty database breaks ViT". A fresh install of either version hits it.
`cold724/` (2026-10-26) runs the 7.2.4 suite from an empty database with and without
`MIOPEN_DEBUG_CONV_GEMM=0`.

## Files

`bench.sh` (the arms), `results.tsv`, each arm's logs and MIOpen database under `<arm>/`,
`sensors.log` (rocm-smi at 1 Hz; the drop at 13:58:35 is where it goes to zeros), `suites.sh`
(the four suites it would have run next — stopped by the drop), `cap275/` (ConvNeXt at a 275 W
power cap after a reboot: dropped again after ~110 s at a steady 275 W, 62 °C),
`suite-k68.log` + `sensors-k68-suite.log` (the suite on kernel 6.8 without the riser, 21 pass, 3 skips).
