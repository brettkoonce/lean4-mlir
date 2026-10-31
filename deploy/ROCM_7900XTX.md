# AMD Radeon RX 7900 XTX — the verified MNIST, CIFAR and Imagenette tiers on ROCm 7.2.4

**Status: the software works; minerva's power does not, yet.** On minerva's 7900 XTX
(2026-10-25 → 31), ROCm 7.2.4: the suite (21 pass, 3 two-device skips), MNIST, CIFAR's six arms, the batch-BN gradient
check, five seeds and width grid, and five of the seven Imagenette nets at 80 epochs, every one
inside or at the edge of CUDA's range (§ Measured). **ResNet-50 and MobileNetV4 are not done:**
from the evening of 10-25 the box resets itself under full load — seven times, on kernel
6.8 and 7.0, at the stock 303 W and capped at 275 W, with no error in any log (§2). One MIOpen trap besides: from an **empty MIOpen database** ViT-Tiny does
not train on any current ROCm, and the fix is one environment variable (§3).

The trainers reach the GPU through AMD's PJRT plugin, `jax-rocm7-pjrt`, which the shim
(`ffi/pjrt_ffi.c`) `dlopen`s from the pinned `.venv`; the plugin finds ROCm's libraries — HIP,
hipBLASLt, rocBLAS, MIOpen — in `/opt/rocm`. ROCm here is userspace only: the kernel driver is
the distribution's in-tree `amdgpu`, no DKMS.

The card: Sapphire Pulse RX 7900 XTX 24 GB (1002:744c, gfx1100, RDNA3; the driver's default power
cap 303 W, range 272–350 W), behind its own internal PCIe switch, x16.

## Requirements

| | |
|---|---|
| box | Linux x86_64. minerva: ASUS Pro WS X570-ACE (BIOS 5601), Ryzen 9 3950X, Ubuntu 24.04 |
| kernel | the distribution's, its in-tree `amdgpu`: 6.8.0-147 and 7.0.0-38 both run everything here (§2) |
| power | each 8-pin socket on its own cable from the supply, no pigtails; minerva resets under load without that (§2) |
| ROCm | **7.2.4 from `repo.radeon.com`, apt-pinned**, the `rocm-ml-libraries` set: `sudo bash deploy/rocm724_install.sh apply` (`dryrun` first) |
| plugin | `jax-rocm7-pjrt` 0.11.0 in the repo's `.venv` — the plugin only, no jax (the pin `jax/requirements-rocm-lock.txt` carries) |
| PATH | `/opt/rocm/bin` on it: `scripts/platform/env.sh` picks the `rocm` backend by finding `rocm-smi` |
| slot | the card in the board's slot, no riser (§2). minerva's slot is still forced to Gen3 from the B50 (8 GT/s under load); Gen4 is untested since |
| tools, disk | gcc, Python 3.12+, elan; ~25 GB with ROCm's libraries |

## Run it

```bash
sudo bash deploy/rocm724_install.sh dryrun && sudo bash deploy/rocm724_install.sh apply
python3 -m venv .venv && .venv/bin/pip install jax-rocm7-pjrt==0.11.0   # the plugin only, as in the README
export PATH=/opt/rocm/bin:$PATH
scripts/platform/check.sh                     # the suite, tiers 0–2 (§ Measured)
lake run mnist && lake run cifar              # the MNIST and CIFAR tiers
lake run imagenette                           # or one net: ./run.sh <net>-verified-adam 0 rocm
```

`HIP_VISIBLE_DEVICES` picks the card, as `CUDA_VISIBLE_DEVICES` does on NVIDIA; `run.sh`'s GPU
index becomes it.

## Measured

minerva, one 7900 XTX, kernel 6.8.0-147 (10-25) and 7.0.0-38 (10-26 on), ROCm 7.2.4 (`hip-runtime-amd` 7.2.53211, MIOpen 3.5.1),
`jax-rocm7-pjrt` 0.11.0, MIOpen's default (warm) database. Every run's 1 Hz sensor log (power,
edge / junction / memory temperature, load, link) is beside it.

| | result | B50 | CUDA |
|---|---|---|---|
| suite, tiers 0–2 | 21 pass, 3 skips (two-device tests) on 6.8 ([`runs/platform/2026-10-25-minerva-rocm-k68/`](../runs/platform/2026-10-25-minerva-rocm-k68/)) and on 7.0 ([`…-rocm/`](../runs/platform/2026-10-25-minerva-rocm/)) | 21 + 3 | — |
| MNIST linear / MLP / CNN, 12 ep | **92.10 / 97.82 / 98.76 %** | 92.10 / 97.83 / 98.66 | 92.10 / 97.77 / 98.78 (medians of 30) |
| CIFAR-10 `lake run cifar`, 6 × 40 ep | no BN 71.11 / 73.94 / 73.36 %, BN 75.27 / **76.54** / 74.59 % (SGD / Nesterov / AdamW) | 76.24 best | 76.3 best |
| batch-BN gradient check | passes both tiers | passes | passes |
| batch-BN `cifar8wb-bbn-ablation`, 5 seeds × 6 arms | per-arm means 75.65–77.83 %, each range overlapping CUDA's | — | 75.29–78.01 |
| batch-BN head-width grid, 10 widths × 25 ep | 74.97–78.16 %, each width within one interval of CUDA's | — | 75.98–77.85 |

Run directories: [`runs/2026-10-25-mnist-*`](../runs/), [`runs/2026-10-25-cifar8w*-ablation/`](../runs/),
[`runs/2026-10-25-xtx-bbn-seeds/`](../runs/2026-10-25-xtx-bbn-seeds/RESULTS.md),
[`runs/2026-10-25-cifar8wb-bbn-grid-xla/`](../runs/2026-10-25-cifar8wb-bbn-grid-xla/README.md); the
queue that ran them, its block logs and sensor logs are
[`runs/2026-10-25-xtx-ladder/`](../runs/2026-10-25-xtx-ladder/). CIFAR's no-BN SGD arm (71.11 %)
is 0.4 over CUDA's five-seed range, as the B50's was — the least stable arm in the set.

Imagenette, 80 epochs, `LEAN_MLIR_CKPT_TAG=xtx80`, each run in ≤ 55-minute blocks resumed from
the per-epoch checkpoint; CUDA is five seeds on a 4060 Ti
([`runs/2026-08-31-imagenette-n3/`](../runs/2026-08-31-imagenette-n3/)):

| net | top-1 (95 % CI) | B50 | CUDA, 5 seeds | loaded wall | mean W | run |
|---|---|---|---|---|---|---|
| ResNet-34 | **90.50 %** (89.54–91.38) | 89.68 | 89.68–90.37 | 34 min | 286 | [`2026-10-25-xtx-resnet34-…`](../runs/2026-10-25-xtx-resnet34-verified-adam/) |
| MobileNetV2 | **88.92 %** (87.90–89.86) | 88.71 | 88.74–89.32 | 49 min | 285 | [`2026-10-25-xtx-mobilenetv2-…`](../runs/2026-10-25-xtx-mobilenetv2-verified-adam/) |
| ViT-Tiny | **67.64 %** (66.16–69.09) | 68.46 | 67.82–69.17 | 38 min | 260 | [`2026-10-25-xtx-vit-…`](../runs/2026-10-25-xtx-vit-verified-adam/) |
| EfficientNet-B0 | **90.22 %** (89.25–91.11) | 90.55 | 89.45–90.50 | 56 min | 283 | [`2026-10-25-xtx-efficientnet-…`](../runs/2026-10-25-xtx-efficientnet-verified-adam/) |
| ConvNeXt-T | **81.96 %** (80.73–83.13) | 82.65 | 82.27 (one run) | ~75 min¹ | 294 | [`2026-10-31-xtx-convnext-…`](../runs/2026-10-31-xtx-convnext-verified-adam/) |
| ResNet-50 | — reset 1 min in (10-31) | 89.61 | 89.58–89.96 | | | [`2026-10-31-xtx-resnet50-…`](../runs/2026-10-31-xtx-resnet50-verified-adam/) |
| MobileNetV4-Conv-M | — not reached | 86.06 | 85.15–86.90 | | | |

¹ In four pieces across three resets: epochs 1–44 (10-26, its log lost to a rotation in the
queue script, the checkpoint kept), 45–50 (10-26), 51–53 (10-30, at 275 W), 54–80 (10-31, after
the recabling — `runs/2026-10-{26,30,31}-xtx-convnext-verified-adam/`). ViT is the one net under
CUDA's range, by 7 images of 3,925; MobileNetV2, EfficientNet and ResNet-34 sit at or above its top.

Junction temperature reached 96–100 °C on every Imagenette net, memory 84–90 °C.

## What to expect

- **An empty MIOpen database breaks ViT** (§3): set `MIOPEN_DEBUG_CONV_GEMM=0` on a fresh install.
- **bf16 is 3× slower than f32 on CIFAR's small convolutions.** One bbn seed (six arms, 40
  epochs): each f32 arm ~2 min 20 s (8.6 ms/step, a 4060 Ti's pace), each bf16 arm ~7 min — three
  quarters of the seed's 29 min. Not command buffers (`PJRT_COMMAND_BUFFERS=1`: 8.8 against 8.6
  ms/step). The Imagenette nets, where bf16 helps on 7.2.4, keep pace with a 4060 Ti; 32×32 bf16
  convs in MIOpen do not. Unfiled.
- **`rocm-smi` prints `Fail to open libdrm_amdgpu.so`** on minerva and works anyway — the
  unversioned symlink lives in `libdrm-dev`, which ROCm's packages do not pull.
- **A card that drops off the bus stays off it**: `amdgpu … device lost from bus!`, then
  `SMU: bus error … response:0xFFFFFFFF` every second, `rocm-smi` hangs, and a process that had
  the card becomes a zombie. Only a reboot — on minerva a power-off — brings it back (§2).
- **Turn off automatic upgrades on a box that trains overnight**, as on the B50
  (`ARC_B50.md`, What to expect).
- **ROCm 10.1 runs too, beside 7.2.4**, and is no faster (§1).

## §1 ROCm 10.1 beside 7.2.4

ROCm 10 is AMD's TheRock build of the same userspace, shipped as wheels, a tarball or debs from
`stable.repo.amd.com/rocm/` (the old `repo.radeon.com/rocm/apt` stops at 7.2.4). It never
touches the kernel driver, so it lives beside 7.2.4 and is chosen per process by
`PJRT_PLUGIN`: the wheels in a venv of their own (`.venv-rocm10`, recipe in
[`runs/2026-10-25-rocm10-bench/README.md`](../runs/2026-10-25-rocm10-bench/README.md)), or the
debs (`amdrocm10.1-gfx1100`, installed into `/opt/rocm/core-10.1`) by `LD_LIBRARY_PATH`. The two
are the same build and measure the same to 1 ms.

On the suite, 10.1 is 17 of 21 from an empty database (the three ViT rows and ResNet-34 bf16 —
§3) and 21 of 21 with §3's four switches
([`runs/platform/2026-10-25-minerva-rocm10/`](../runs/platform/2026-10-25-minerva-rocm10/),
[`…-rocm10-nogemm/`](../runs/platform/2026-10-25-minerva-rocm10-nogemm/)). Step time against
7.2.4: ResNet-34 13 % slower (88 against 78 ms), MobileNetV2 and EfficientNet-B0 3–4 % faster,
ViT 71 ms with the switches; the switches cost 7–8 % where they change anything. Nothing here
argues for moving off 7.2.4.

## §2 The bus drops, then the resets

The 7900 XTX ran on minerva from 2026-08-24 to 10-10 on BIOS 3601 and kernel 6.8 with no bus
drop in the journal — one boot lasted 155 hours across the suites and the training runs, two
cards in. On 10-11 the box changed for the Arc B50: BIOS 5601 (Resizable BAR), kernel 7.0, the
card on a riser, the slot later forced to Gen3. Nothing loaded the 7900 XTX on that setup until
10-25, when it fell off the bus three times, each in ConvNeXt-T's first minutes:

| time | run | before the drop |
|---|---|---|
| 01:06:16 | ConvNeXt, ROCm 10.1, first step | ~30 min after boot |
| 13:58:35 | ConvNeXt, 7.2.4, step 3 | 266–306 W, one 1-s reading of 431 W; 64 / 88 / 78 °C (edge / junction / memory) |
| 14:43:07 | ConvNeXt, 7.2.4, ~step 110, **power capped at 275 W** | a steady 273–278 W, 62 / 75–87 / 78–80 °C |

No AER, no PCIe error: `device lost from bus!` and then nothing answers. The card was not hot,
and capping its power changed nothing. Between the first and second drops it vanished from the
bus entirely across two reboots — the root port itself not enumerated — until the card was
replugged.

Kernel 6.8 and the card straight in the slot, both at once, then ran about eight hours of
training on 10-25 (the suite, four Imagenette nets, MNIST, CIFAR, two batch-BN seeds) with no drop.
At 23:12:57 the whole box reset instead, and it kept doing so:

| reset | kernel | run | last sensor line |
|---|---|---|---|
| 10-25 23:12:57 | 6.8 | width grid, fc4096, 2 min in | 301 W, 64 / 87 / 78 °C |
| 10-26 10:17:07 | 7.0 | ConvNeXt, 41 min in | 303 W, 71 / 87 / 88 °C |
| 10-26 14:17:13 | 7.0 | ConvNeXt, 4 min in | 302 W, 69 / 91 / 86 °C |
| 10-26 14:42:20 | 7.0, power cables reseated | ConvNeXt, 3 min in | 303 W, 70 / 94 / 86 °C |
| 10-30 23:41:20 | 7.0, **capped at 275 W** | ConvNeXt, 4 min in | 276 W, 70 / 92 / 88 °C |
| 10-31 ~14:04 | 7.0, recabled: one cable per input | ResNet-50, 1 min in | 307 W, 71 / 95 / 84 °C |

(A sixth, 10-30 ~23:15, came within seconds of ResNet-50's start; its logs are empty.) After the
10-31 recabling ConvNeXt's last 27 epochs ran clean for 24 minutes at 294 W mean, 344 W peak — then
ResNet-50 reset the box a minute in.

These are not the bus drops. The kernel journal has nothing — no `amdgpu` error, no machine check,
no lockup, no panic — and the sensor logs end in NUL bytes: the last seconds of writes never
reached the disk, so the machine stopped without the kernel getting a word out. The box restarts
on its own. The CPU's reset-reason register reads the same on every boot, a software reset through
port 0xCF9, so it does not say why. The slot's root port (00:03.1) has no AER, so a link error
there would go unreported.

What it is not: the kernel (6.8 and 7.0 both), the riser (out since 10-25), the GPU's temperature
(the card ran hotter on 10-25 without a reset), the supply's capacity (1000 W, this card alone)
and the power cap (275 W reset too). What fits: the 7900 XTX's millisecond load spikes — up to
~1.5–2× its rated power, invisible to a 1 Hz log (one reading of 431 W on 10-25) — tripping the
supply's protection or the board's power delivery. Untested: the card in another box, the B50's
BIOS 5601 rolled back to 3601, an ATX 3.x supply.

**Load-test every card after a BIOS, kernel or riser change.** The B50's changes were checked on
the B50 and nothing else; the cost was a day.

## §3 MIOpen from an empty database

ViT-Tiny's patch-embed weight gradient is a 16-dilated convolution that MIOpen, searching from an
empty database, sends to `GemmFwdRest` with no workspace and from there to `MIOpenIm2d2Col`'s
`EXTREME_LARGE` branch, which does not compile under HIPRTC (`get_global_id` undeclared). The
diagnosis, a 20-line reproducer and the failing compile line are
[`historical/upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`](../historical/upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/README.md).
It is in 7.2.0, 7.2.4 and 10.1. Every ViT run on minerva passed only because `~/.config/miopen`
already held find results that keep MIOpen off that solver; with
`MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` pointed at an empty directory, 7.2.4 fails on
the first step: the suite from an empty database is 18 of 21, the three ViT rows failing in
`MIOpenIm2d2Col`; with `MIOPEN_DEBUG_CONV_GEMM=0` it is 21 of 21
([`runs/platform/2026-10-26-minerva-rocm-coldmiopen/`](../runs/platform/2026-10-26-minerva-rocm-coldmiopen/),
[`…-nogemm/`](../runs/platform/2026-10-26-minerva-rocm-coldmiopen-nogemm/)). ViT-Tiny's step with
the switch from cold is 84 ms against 78 ms from the warm database
(`runs/2026-10-25-rocm10-bench/cold724/`) — the switch costs 8 %, so set it only on a fresh
install.

| variable | turns off | needed for |
|---|---|---|
| `MIOPEN_DEBUG_CONV_GEMM=0` | the GEMM convolution solvers | ViT from an empty database; R34 bf16's `GemmWrw: Not enough workspace` on 10.1 |
| `MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS=0` | grouped implicit-GEMM forward | bf16 on 10.1: these kernels give garbage on gfx1100, which has no XDLOPS |
| `MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS=0` | their weight gradient | the same |
| `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS=0` | their data gradient | the same |

They are per-process environment variables; nothing to install or undo.

## After it runs

Send back `runs/platform/<date>-<host>-rocm/` and the `runs/<date>-mnist-*/`, `runs/<date>-cifar8w*-*/`
and Imagenette run directories, as for every platform.
