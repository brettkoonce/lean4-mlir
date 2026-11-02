# Intel Arc Pro B50 — the verified MNIST, CIFAR and Imagenette tiers on oneAPI

**Status: works — every `arc_setup.sh` rung green on minerva's Arc Pro B50 at Gen3 (2026-10-22),
and all seven Imagenette nets' full 80-epoch runs on record (2026-10-22 to 24), each inside
CUDA's five-seed range or within two images of its edge.** MNIST and CIFAR-10 train end to end
on the GPU and match CUDA; the seven Imagenette nets take ~9 h in all at 62–69 W; ConvNeXt-T and
ViT-Tiny reproduce CUDA's training loss to 4–5 decimals. That took **the slot at PCIe Gen3**: at
Gen4 the card fell off the bus under load every time — two minutes into sustained 70 W, and once
three minutes into the suite (§4).

The trainers reach a GPU through a PJRT plugin that the shim (`ffi/pjrt_ffi.c`) `dlopen`s, so an
Intel card needs Intel's plugin and nothing else from this repo's side. The plugin is
`jax_oneapi_pjrt` (Intel's OpenXLA plugin, platform `oneapi`, PJRT API 0.115 against our header's
0.90), with its oneAPI runtime — SYCL, oneDNN, MKL — as pip wheels from
`jax_oneapi_plugin[with-oneapi]`. Getting the card visible to that runtime was most of the work
(§1); one shim workaround for a race in Intel's SYCL runtime was the rest (§3).

The card: Arc Pro B50 16 GB (8086:e212, Battlemage, 70 W, slot-powered — no auxiliary
connector), behind its own internal PCIe switch, on x8 Gen4.

## Requirements

| | |
|---|---|
| box | Linux x86_64. minerva: ASUS Pro WS X570-ACE, Ubuntu 24.04 |
| kernel | **7.0** (Ubuntu 24.04: `linux-generic-hwe-24.04`). 6.8's `xe` has no B50 ID |
| BIOS | **Resizable BAR on, BAR at full VRAM (16 GB).** At 256 MB Intel's runtime sees no device at all (§1). On minerva also **the slot forced to Gen3**: at Gen4 the card drops off the bus under load (§4) |
| runtime | Intel's Level Zero GPU runtime from `ppa:kobuk-team/intel-graphics`: `libze1`, `libze-intel-gpu1`, `intel-opencl-icd`, `xpu-smi` |
| tools | gcc, Python 3.12+, elan |
| disk | ~20 GB: the oneAPI wheels (~3 GB), Mathlib's prebuilt cache, the build |

## Run it

```bash
git clone https://github.com/brettkoonce/lean4-mlir && cd lean4-mlir
bash deploy/arc_setup.sh                 # every rung, in order
bash deploy/arc_setup.sh smoke           # resume from a rung after fixing it
```

| rung | proves | when it fails, the problem is in |
|---|---|---|
| `check` | an Intel GPU bound to `xe`, its BAR at full VRAM, the Level Zero runtime, gcc, Python, elan, disk | the box: kernel, BIOS, Intel's packages |
| `venv` | `.venv-xpu` with `jax_oneapi_pjrt` + `jax_oneapi_plugin[with-oneapi]` + `jax`, all 0.11.2 — never the pinned `.venv` | pip / the wheels |
| `jax` | JAX computes on the `oneapi` device, with `JAX_PLATFORMS=oneapi` so it cannot fall back to the CPU quietly | the plugin itself, independent of this repo |
| `smoke` | `ffi/test_pjrt_smoke.c` through the real shim: create the client, run `verified_mlir/linear_fwd.mlir` against the host's answer, compile the other five MNIST renders | the shim ↔ plugin seam |
| `suite` | `scripts/platform/check.sh --backend xpu`, tiers 0–2: the shim's guards and 16 train steps against XLA:CPU goldens (§2) | the plugin's numerics, op by op |
| `data` | MNIST into `data/mnist/`, CIFAR-10 into `data/` | the network |
| `lean` | elan installs the toolchain, `lake exe cache get` fetches Mathlib, and the five MNIST and CIFAR trainers build | the Lean build |
| `mnist` | `lake run mnist`: linear, MLP and CNN, on the GPU | the trainers on oneAPI |
| `cifar` | `lake run cifar`: the six optimizer arms, ~10½ min | sustained conv training (where §3 showed up) |

Everything is teed to `runs/<date>-arc-setup/setup.log`, with the box's kernel, GPU, BAR, link,
runtime and package versions in `manifest.txt` beside it. Each trainer's own log lands under
`runs/<date>-<trainer>/`, as on the other platforms. Imagenette is not a rung: run it by hand
(`lake run imagenette`, or one net with `lake run train <net> 0 xpu`) once §4 is settled on your box.

## What is Arc-specific

- **`scripts/platform/env.sh`**'s `xpu` arm points `PJRT_PLUGIN` at
  `.venv-xpu/…/jax_plugins/xla_oneapi/xla_oneapi_plugin.so` and puts `.venv-xpu/lib` on
  `LD_LIBRARY_PATH`: the plugin finds its oneAPI runtime only through that path. The backend is
  picked by `xpu-smi` on PATH and an Intel display device on the bus; with an AMD card also in the
  box it picks `rocm`, and `PLATFORM_BACKEND=xpu` selects the Arc (`arc_setup.sh` sets it).
- **`ffi/pjrt_ffi.c`** serializes host-to-device uploads and device-to-host readbacks when the
  plugin's platform name is oneAPI/SYCL/XPU — `h2d_serial()` and `d2h_serial()`, §3. Every other
  platform is unchanged. Its startup lines say which: `[pjrt_ffi] h2d: SERIAL — one upload in
  flight (oneAPI default …)` and `[pjrt_ffi] d2h: SERIAL — one readback in flight (…)`.
- **`run.sh`** accepts the `xpu` backend; the GPU index becomes `ZE_AFFINITY_MASK`.
- **bf16 works.** Tier 2's bf16 rows pass, so the bf16 renders apply. Unlike the Mac there is no
  float32-only restriction.

## Measured

Arc Pro B50 16 GB in minerva (Ryzen, X570, x8 Gen4 through a riser), kernel 7.0.0-38,
`jax_oneapi_pjrt` 0.11.2 (`intel-sycl-rt` 2025.1.1), the shim's serial uploads on.

The rungs below are the 2026-10-22 `arc_setup.sh suite` run at Gen3
([`runs/2026-10-22-arc-setup/`](../runs/2026-10-22-arc-setup/), its 1 Hz sensor logs beside it;
link at 8 GT/s throughout), with CIFAR retrained from scratch under a fresh checkpoint tag.

| | per epoch | result | CUDA (README) |
|---|---|---|---|
| suite, tiers 0–2 | — | 21 pass, 0 fail, 3 skips (two-device tests, one card; same 24 tests as 10-11) — [`runs/platform/2026-10-22-minerva-xpu/`](../runs/platform/2026-10-22-minerva-xpu/) | — |
| `smoke`: `linear_fwd` vs the host | — | max error 1.7e-06; all six MNIST renders compile | — |
| MNIST linear, 12 ep | ~0.25 s | **92.10 %** | 92.10 % |
| MNIST MLP, 12 ep | ~0.55 s | **97.83 %** | 97.77 % (median of 30; 97.49–97.93) |
| MNIST CNN, 12 ep | ~1.5 s | **98.66 %** (98.74 % on 10-12) | 98.78 % (median of 30; 97.39–98.94) |
| CIFAR-10 `lake run cifar`, 6 × 40 ep | ~2–3 s | **76.24 %** best arm (BN, Nesterov); 11 min for all six | 76.3 % |
| ResNet-34, Imagenette 224², 80 ep (Gen3) | ~64 s (218 ms/step with eval) | **89.68 %** (top-5 98.62 %, CI 88.69–90.59); 85.8 min | 89.99 % |

CIFAR's six arms, 40 epochs each, against CUDA's five seeds per arm
([`runs/2026-09-01-cifar8w-6arm-constlr/`](../runs/2026-09-01-cifar8w-6arm-constlr/), a 4060 Ti).
Seed 1 is `lake run cifar` ([`runs/2026-10-22-cifar8w-ablation/`](../runs/2026-10-22-cifar8w-ablation/),
[`…-bn-ablation/`](../runs/2026-10-22-cifar8w-bn-ablation/)); seeds 2 and 3 of the no-BN net are
[`runs/2026-10-22-cifar8w-seeds-gen3/`](../runs/2026-10-22-cifar8w-seeds-gen3/)
(`LEAN_MLIR_SEED=<s> LEAN_MLIR_CKPT_TAG=gen3s<s>`).

| net | optimizer | B50, seed 1 | B50, seeds 2 / 3 | CUDA, 5 seeds |
|---|---|---|---|---|
| no BN | SGD | 73.37 % | 72.36 / 67.34 % | 68.03–70.73 % |
| no BN | Nesterov | 73.52 % | 74.17 / 72.73 % | 71.05–74.36 % |
| no BN | AdamW | 73.50 % | 73.32 / 72.62 % | 71.04–73.82 % |
| BN | SGD | 74.87 % | — | 74.14–75.21 % |
| BN | Nesterov | 76.24 % | — | 75.75–76.73 % |
| BN | AdamW | 74.33 % | — | 74.24–74.46 % |

Five of six arms land inside CUDA's range. The sixth, no-BN SGD (lr 0.1, no normalization), is the
least stable arm in the set — the no-BN net is the one whose momentum arm diverged to 10.00 % in
3 of 5 CUDA seeds under the old cosine schedule
([`runs/2026-08-12-cifar8w-6arm-xla-cuda/`](../runs/2026-08-12-cifar8w-6arm-xla-cuda/)) — and
the B50's three seeds straddle CUDA's range (67.34–73.37 %) rather than sitting off one side of
it. Epoch-1 loss says the same: within 0.015 of CUDA's seed 1 on the four stable arms, apart only
on the two no-BN arms with a large step. The 2026-10-12 record
([`runs/2026-10-12-cifar8w-ablation/`](../runs/2026-10-12-cifar8w-ablation/), 69.69 % on this
arm) is not a clean baseline: both SGD arms resumed from checkpoints an earlier crashed attempt
left behind (epoch 10 without BN, epoch 4 with), so 10-22 is the first clean set.

The seven Imagenette nets (2026-10-13), each 60 steps at batch 32 in probe mode
(`LEAN_MLIR_MAX_STEPS=60 LEAN_MLIR_PROBE_WARM=20 LEAN_MLIR_SKIP_EVAL=1`), the step time the
median over steps 21–60. Spreads were within 3 ms and data starvation 2 ms or less; all seven ran
clean. "Full run" is 80 epochs × 295 steps, its eval share an estimate (about +14%: the probes ran
with eval off).

| net | ms/step | training only | full run (est.) |
|---|---|---|---|
| ViT-Tiny | 80 | 32 min | ~36 min |
| MobileNetV2 | 110 | 43 min | ~50 min |
| MobileNetV4-Conv-M | 125 | 49 min | ~56 min |
| EfficientNet-B0 | 128 | 50 min | ~57 min |
| ResNet-34 | 187 | 74 min | ~1.4 h |
| ConvNeXt-T | 241 | 95 min | ~1.8 h |
| ResNet-50 | 287 | 113 min | ~2.1 h |
| **`lake run imagenette`** | | **~7.6 h** | **~8½–9 h** |

The ResNet-34 run (2026-10-22, [`runs/2026-10-22-resnet34-verified-adam/`](../runs/2026-10-22-resnet34-verified-adam/),
the card's 1 Hz sensors beside the log) is the first full Imagenette run on the card: Gen3, 68.6 W
mean over 85.8 min, GPU at most 69 °C, no kernel errors. Its full-run time is the probe's 187
ms/step plus about 17 % for the per-epoch eval and compiles, so the column above, at +14 %, is about right;
Gen3 itself cost nothing measurable, since the weights stay on the card and only the batch
crosses PCIe.

All seven nets then ran the first five epochs of their real 80-epoch schedule
(2026-10-22, Gen3, [`runs/2026-10-22-imagenette-e5-gen3/`](../runs/2026-10-22-imagenette-e5-gen3/)
with its sensor log; `LEAN_MLIR_MAX_EPOCHS=5 LEAN_MLIR_CKPT_TAG=e5`, which caps the epochs without
changing the schedule, so epoch 5 here is epoch 5 of a full run). The reference is CUDA's five
seeds per net on a 4060 Ti ([`runs/2026-08-31-imagenette-n3/`](../runs/2026-08-31-imagenette-n3/)),
same seed 1 as the B50. 35 min for all seven, 64 W mean, GPU at most 69 °C, no kernel errors.

| net | 5 ep wall | probe + eval predicts | epoch-5 val | training loss vs CUDA |
|---|---|---|---|---|
| ResNet-34 | 329 s | ~335 s | 60.66 % (CUDA 48.5–68.3) | at or just under the seeds' low edge (≤ 0.015) |
| ResNet-50 | 499 s | ~513 s | 60.61 % (55.4–64.3) | inside, or ≤ 0.02 under the low edge |
| MobileNetV2 | 203 s | ~196 s | 59.90 % (51.3–62.2) | inside; seed 1 to ±0.01 |
| MobileNetV4-Conv-M | 231 s | ~223 s | 25.55 % — a one-epoch dip, see below | below all five seeds (the net changed since; see below) |
| EfficientNet-B0 | 238 s | ~229 s | 58.06 % (58.7–67.7) | inside; seed 1 to ±0.01 |
| ConvNeXt-T | 436 s | ~431 s | 62.68 % (63.2–65.7, older recipe) | **= the 09-13 CUDA run to 5 decimals** |
| ViT-Tiny | 173 s | ~143 s | 50.75 % (50.4–51.2) | **= CUDA seed 1 to 4 decimals** |

Wall-clock includes startup and the per-epoch eval; every net lands within a few percent of the
13th's probe plus eval (ViT's extra half-minute is startup, a large share of so short a run), so
no net is running a slow schedule here. ConvNeXt-T and ViT-Tiny are the strongest evidence that
the card computes what CUDA computes: epoch-1 loss 2.321477 against 2.321471
([`runs/2026-09-13-convnext-imagenette-ls1e-6-resident/`](../runs/2026-09-13-convnext-imagenette-ls1e-6-resident/),
the LayerScale-1e-6 recipe — the five n3 seeds predate it) and 2.127477 against 2.127563. The BN
nets wander more between backends and can only be held to the seed spread; the ResNets sit at its
low edge (lower loss, by at most 0.02), which is the direction CUDA's TF32 convolutions would push
a reference if the card runs full fp32 — not checked; `NVIDIA_TF32_OVERRIDE=0` on the reference
box would say.

Two rows need a sentence. **MobileNetV4**'s reference seeds predate 90e4af7e (2026-09-25, the net
rebuilt to match timm's `mobilenetv4_conv_medium`), and no CUDA run of the current net on
Imagenette is on record — its lower loss (1.575 against 1.70–1.72 at epoch 3) is the newer net,
not the card. Its epoch-5 validation fell from 52.97 % to 25.55 % while training loss kept falling;
resuming a copy of the checkpoint for two more epochs gave 49.48 % and then 65.68 %
([`mnv4-resume/`](../runs/2026-10-22-imagenette-e5-gen3/mnv4-resume/)), so it was a warm-up wobble
in the eval, not a fault (CUDA's seeds wobble too, by up to 16 points). **EfficientNet-B0**'s 58.06 %
is just under its seeds' 58.68 % floor at an epoch where they span nine points; its loss is inside.

**All seven, 80 epochs (2026-10-22 to 24, Gen3).** One process at a time, 15 minutes idle between
runs, `LEAN_MLIR_CKPT_TAG=full`, each run's 1 Hz sensor log beside it; no kernel errors in any of
them, the link at 8 GT/s throughout. CUDA is the same five seeds per net on a 4060 Ti
([`runs/2026-08-31-imagenette-n3/`](../runs/2026-08-31-imagenette-n3/)).

| net | top-1 | CUDA, 5 seeds | wall | mean W | run |
|---|---|---|---|---|---|
| ViT-Tiny | 68.46 % | 67.82–69.17 | 39 min | 67.2 | [`2026-10-23-vit-…`](../runs/2026-10-23-vit-verified-adam/) |
| MobileNetV2 | 88.71 % | 88.74–89.32 (one image under) | ~50 min¹ | 61.8 | [`2026-10-23-mobilenetv2-…`](../runs/2026-10-23-mobilenetv2-verified-adam/) |
| MobileNetV4-Conv-M | 86.06 % | 85.15–86.90² | 57 min | 63.3 | [`2026-10-23-mobilenetv4-…`](../runs/2026-10-23-mobilenetv4-verified-adam/) |
| EfficientNet-B0 | 90.55 % | 89.45–90.50 (two images over) | 58 min | 64.3 | [`2026-10-24-efficientnet-…`](../runs/2026-10-24-efficientnet-verified-adam/) |
| ResNet-34 | 89.68 % | 89.68–90.37 | 86 min | 68.6 | [`2026-10-22-resnet34-…`](../runs/2026-10-22-resnet34-verified-adam/) |
| ConvNeXt-T | 82.65 % | 82.27 (one run)³ | 1 h 52 | 68.7 | [`2026-10-24-convnext-…`](../runs/2026-10-24-convnext-verified-adam/) |
| ResNet-50 | 89.61 % | 89.58–89.96 | 2 h 06 | 68.1 | [`2026-10-24-resnet50-…`](../runs/2026-10-24-resnet50-verified-adam/) |

¹ In two pieces: epochs 1–46 until the card dropped off the bus at 06:34:38 on 10-23 — one second
after unattended-upgrades installed liblzma5 and needrestart restarted systemd-udevd, journald and
others (`kernel-`, `systemd-`, `apt-06-34.txt` beside the run; "What to expect" below) — then
epochs 47–80 from the `full` checkpoint (21 min), the first piece's log kept as
`*.until-2239.log`. ² CUDA's seeds predate 90e4af7e (the timm `conv_medium` rebuild); no
post-rebuild 80-epoch CUDA run is on record. ³ The LayerScale 1e-6 recipe's one CUDA run,
[`runs/2026-09-13-convnext-imagenette-ls1e-6-resident/`](../runs/2026-09-13-convnext-imagenette-ls1e-6-resident/);
the August seeds (85.35–85.78) are the older recipe. The 95 % interval on any one of these is
about ±1.1 points, wider than every gap above.

**TODO:** a second card for the suite's two-device tests and the DP path.

## What to expect

- **The plugin is experimental.** JAX prints `Platform 'oneapi' is experimental`. One runtime bug
  has been found and worked around in the shim (§3); expect the next one to show up under
  sustained training, not in the suite.
- **The BAR is the whole bring-up.** If `clinfo` shows no platform or `zeInit` fails, look at
  the BAR size before anything else (§1).
- **A card that drops off the bus stays off it.** `CT write: non-zero status: 4294967295` in
  dmesg, then every read returns `0xFFFFFFFF`: only a power cycle at the wall brings it back. On
  minerva the cure was the slot at Gen3 (§4); try that first on a box that shows it.
- **A second `arc_setup.sh` on the same box does not retrain CIFAR.** `lake run cifar` scores
  each arm's finished checkpoint in `.lake/build/` instead of training it again (the 10-22 suite
  pass printed six `checkpoint:` lines and no epochs). To retest training — after a hardware
  change, say — give it a fresh tag: `LEAN_MLIR_CKPT_TAG=<name> bash deploy/arc_setup.sh cifar`.
  MNIST keeps no checkpoint and always trains.
- **`run.sh` logs to `runs/<date>-<trainer>/`, so a second run of a trainer on the same day
  overwrites the first one's log.** Set `RUN_LOG_DIR` for a short or side run; the 10-22 five-epoch
  ResNet-34 replaced the full run's log until it was restored.
- **One process on the card at a time.** Two `muzero-ttt` processes sharing the B50 (2026-10-23)
  both hung in SYCL — a host task in `DispatchHostTask::waitForEvents → urEventWait`, the card at
  full clock, no kernel message until they were killed (`guc_exec_queue_timedout_job`, GT0 reset,
  recovered without a power cycle). A three-process smoke had passed first, and sharing bought
  ~11 % throughput anyway. The shim's serial copies (§3) order one process's transfers; nothing
  orders two processes'. Evidence on the `muzero-ttt` branch, `runs/2026-10-23-muzero-ttt-b50/hang-18-24/`.
- **Turn off automatic upgrades on a box that trains overnight.** needrestart's service restarts
  after an unattended upgrade took the card off the bus mid-run (MobileNetV2, above). On minerva:
  `systemctl disable --now unattended-upgrades apt-daily-upgrade.timer apt-daily.timer` and
  `$nrconf{restart} = 'l'` in `/etc/needrestart/needrestart.conf`.

## After it runs

Send back `runs/<date>-arc-setup/` (the log and manifest), the suite's `runs/platform/<date>-<host>-xpu/`
and the `runs/<date>-mnist-*/` and `runs/<date>-cifar8w-*/` logs. A green `cifar` rung on a new
card earns it a row in `scripts/platform/support.tsv`, and with it `PLATFORMS.md`.

## §1 Bringing it up, in the order it cost time

1. **Kernel 7.0.** The 6.8 kernel's `xe` has no B50 ID, so install `linux-generic-hwe-24.04`
   (7.0.0-38 here). minerva stays on 24.04 because ROCm 7.2.4's apt repo stops at noble. The
   in-tree `amdgpu` kept working on 7.0.
2. **A 16 GB BAR is mandatory.** At 256 MB, Intel's compute runtime reports no device at all:
   `clinfo` shows 0 platforms, and `zeInit` fails with UNINITIALIZED. The only hint is a "Resizable
   BAR not detected" warning. BIOS 3601 (2021) on the ASUS Pro WS X570-ACE gave a 7900 XTX its
   32 GB but left the B50 at 256 MB in either slot, because the B50 sits behind its own internal
   PCIe switch. `pci=realloc` did not help. BIOS 5601 (2026-08) fixed it. Check with
   `lspci -vv -s 0d:00.0 | grep -A2 Resizable`; `arc_setup.sh check` reads it from sysfs.
3. **Userspace from `ppa:kobuk-team/intel-graphics`:** `libze1`, `libze-intel-gpu1`,
   `intel-opencl-icd`, `xpu-smi`. Noble's own 23.43 runtime predates Battlemage.
4. **`.venv-xpu`:** the two Intel packages and `jax`, 0.11.2. Keep this venv separate; never
   install into the pinned `.venv`. minerva has no `python3.12-venv`, so `arc_setup.sh` makes it
   with `python3 -m venv --without-pip` and installs with the system pip's `--python`.

Topology and noise you can ignore:

- The card's own link reads x1 2.5 GT/s. That is the internal switch's downstream port. The real
  link is the upstream bridge: root port `00:03.1` → `0b:00.0` (switch) → `0c:01.0` →
  `0d:00.0` (GPU), plus `0c:02.0` → `0e:00.0` (HDA audio), at x8 Gen4.
- `xe … PCODE Mailbox failed: -6` in dmesg has no visible effect.

## §2 The suite

```bash
scripts/platform/check.sh --backend xpu
```

Tiers 0–2 pass: 21 pass, 0 fail, plus 3 skips for the tests that need two devices
([`runs/platform/2026-10-11-minerva-xpu`](../runs/platform/2026-10-11-minerva-xpu/results.tsv)).
Every one of tier 2's 16 train-step graphs matches XLA:CPU within tolerance. That covers f32 and
bf16 across dense, conv, BN, depthwise, swish-SE, attention, ResNet-34 and ConvNeXt. The
expected-failures file,
[`scripts/platform/expected/xpu-arc-pro-b50.txt`](../scripts/platform/expected/xpu-arc-pro-b50.txt),
is empty.

The suite runs short graphs, so it caught neither §3 nor §4. Both showed up only under
sustained training.

## §3 The SYCL race, and the shim's workaround

With `jax_oneapi_pjrt` 0.11.2, which bundles `intel-sycl-rt` 2025.1.1, conv nets crashed
intermittently: about 8% of 12-epoch MNIST-CNN runs, and both CIFAR nets on their first try. The
crash is a use-after-free in libsycl's scheduler (`cleanupDependencyEvents`, or the abort
`urEventWait must not be called for an internal event`). It happens when several
`BufferFromHostBuffer` uploads are in flight at once: each one completes through a host callback
on an XLA thread, and SYCL's cleanup frees an event another host task is still waiting on. Under
stress it also caused GPU page faults, and once took the card off the bus.

Things that do not help:

- Waiting on each input's ready event before Execute.
- `SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP=1`.
- sycl-rt 2025.3.3, which does not start with the 0.11.2 plugin.

The fix is to keep only one upload in flight. `ffi/pjrt_ffi.c`'s `h2d_serial()` waits for each
upload's completion and ready events before issuing the next. It is on by default when the
plugin's platform name is oneAPI/SYCL/XPU, and every other backend stays batched.
`PJRT_FFI_H2D_SERIAL=0|1` overrides the default, and the shim prints its choice at startup
(`[pjrt_ffi] h2d: SERIAL …`). With the fix, the stress loop ran 50,000 steps with 0 crashes,
against 5 crashes in 5 runs without it. It costs 8.5 → 10.3 ms per step in the stress loop. On
the real nets it disappears: parameters stay resident on the device, so only the batch is
uploaded.

The repro and the upstream report are in
[`runs/2026-10-11-xpu-sycl-race/`](../runs/2026-10-11-xpu-sycl-race/). `repro.sh` sets
`PJRT_FFI_H2D_SERIAL=0` so that it still reproduces the bug. Drop the default once Intel ships a
plugin with a fixed runtime and the repro runs clean.

**The same race on readback (2026-10-23).** With uploads serialized, a net that reads many small
outputs back every step still corrupted the host heap: `cifar8wb_bbn`, the batch-statistics BN
net, returns its 16 batch statistics, the loss and Adam's two bias corrections — 19 readbacks a
step — and aborted with `malloc(): … corrupted` about once per 160k steps (3 aborts in the
2026-10-22 seed sweep, [`runs/2026-10-22-bbn-seeds/crashes/`](../runs/2026-10-22-bbn-seeds/crashes/)).
The backtrace caught under gdb
([`stress-old/run21.log`](../runs/2026-10-22-bbn-node-b50/stress-old/run21.log)) is a plugin worker
thread inside `sycl::queue::submit`, under XLA's `CopyRawDeviceToHostAndReturnEvent`: the shim
issued the step's `ToHostBuffer` calls back to back, the plugin handed each to its own thread, and
they submitted to one SYCL queue at once. `d2h_serial()` awaits each readback before issuing the
next, on the same platforms as `h2d_serial()`; `PJRT_FFI_D2H_SERIAL=0|1` overrides it. The f32
AdamW arm repeated with fresh seeds: 24 runs (~367k steps) without it, 1 abort; 40 runs (~624k
steps) with it, 0 — about 4 were expected at the old rate. Per-run time 3.1 → 3.0 min: no cost.
The BN Imagenette nets are exposed too — ResNet-34 reads back 75 tensors a step (its 72 batch
statistics and three scalars) — but its full run was 23.6k steps, where ~0.15 aborts were due at
this rate; the full runs from 2026-10-23 on have `d2h_serial()`.


## §4 The bus drop at Gen4, and Gen3 as the fix

**2026-10-22: with the slot forced to Gen3 in the BIOS (8 GT/s x8 at the root port and the card's
upstream switch, from 16 GT/s), ResNet-34 ran all 80 epochs** — 85.8 min at 68.6 W mean, past
every earlier failure point by 40×, no kernel errors. Nothing else changed: same riser, same 70 W
cap, runtime PM back at `auto`. So the drop was the Gen4 link, signal integrity through the riser
or the card's own switch, not the slot's power. One run is one run; the record below is kept for
whoever sees it next. The host's own unexplained reset (10-14) is a separate question.

The record at Gen4. ResNet-34 Imagenette has died three times, every time about 2:00–2:20 into full 70 W load:
the 2026-10-11 stress loop, 2026-10-12 (~2 min), and 2026-10-13 (2:18, then 2:02 under the
sensor log). The sequence in dmesg is always:

```
xe 0000:0d:00.0: [drm] *ERROR* GT0: CT write: non-zero status: 4294967295
```

Every MMIO read after that returns `0xFFFFFFFF`, and the device is wedged. No AER is logged.
The trainer dies on a Level Zero abort (`direct_submission_hw.inl`). **Only a full power cycle,
off at the wall, recovers it.** A PCI remove/rescan does not.

Ruled out:

- **The shim and the SYCL race:** §3's fix is in, and the failure is a link drop, not a libsycl
  fault.
- **Runtime PM:** it dropped again with `power/control=on` on `0b:00.0` through `0e:00.0`. The
  earlier `Unable to change power state from D3hot` line on `0c:02.0` was a symptom.
- **Thermal:** `hwmon` sampled at 1 Hz showed 70 W flat at the cap, GPU at 69–70 °C, VRAM at
  66–68 °C and the fan at 3070 rpm, all steady for a minute. Then it dropped at 2:02 with no
  precursor.

**2026-10-18: it dropped under the suite, not just under sustained load.** About three minutes into
`check.sh`'s tier 2 — short bursts of train steps, the load that passed in full on 10-11 — at
01:24:42, the same `CT write` line, the switch port `0c:02.0` stuck in D3hot, then wedged. And the
host itself had reset without a shutdown on 10-14 at 02:35, idle (the journal's last lines are
cron). So it is not a matter of minutes at 70 W: the card can fall off under a light load too, and
the box's own power is suspect along with the card's.

That leaves power or signal. **The leading suspect is slot power through the riser**, with the PSU
behind it. The B50 has
no auxiliary power connector and draws all 70 W through the slot, and on minerva that path runs
through a riser. A Gen4 signal-integrity problem on the same riser is the other candidate.

### The pass/fail test

A logged 4-minute ResNet-34 load. Pass means more than 5 minutes at 70 W. It has failed at
2:02–2:18 every time.

```bash
ulimit -c unlimited
LEAN_MLIR_MAX_STEPS=1300 LEAN_MLIR_PROBE_WARM=20 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_CKPT_TAG=hwtest \
  PLATFORM_BACKEND=xpu lake run train r34 0 xpu
```

Alongside it, log the sensors from `/sys/bus/pci/devices/0000:0d:00.0/hwmon/hwmon*/` once a
second:

- `energy1_input`: the card's energy counter. Take the delta for watts. The package power column
  is unreliable.
- `temp2`: the package.
- `temp3`: VRAM.
- `fan1_input`.

### What was next, had Gen3 not fixed it

1. **Card directly in a motherboard slot, no riser** — and, given the host resets, a look at the
   PSU (its age and rating, and whether the resets stop with the B50 out). If it passes, the riser is the cause.
   Replace it with a powered riser, or with an OCuLink or MCIO dock that powers the slot from
   the PSU.
2. **Cap the power at 45 W.** `xpu-smi config --powerlimit` fails ("Power level 1 not found":
   the card has no sustain level), so use sysfs. The setting does not survive a reboot.
   ```bash
   echo 45000000 | sudo tee /sys/bus/pci/devices/0000:0d:00.0/hwmon/hwmon*/power1_cap
   ```
   If it survives at 45 W, the 70 W slot draw is implicated.
3. **Force Gen3 in the BIOS.** If that fixes it, the cause is signal integrity, not power. —
   It did (2026-10-22, above). Back at Gen4, a better riser (or none) is what to test.
