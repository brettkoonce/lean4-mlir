# A second Mac: what to run, what to commit, how to send it back

For whoever runs this repo's verified trainers on an Apple Silicon Mac the author does not have.
Everything below is scripted; each step says what green looks like and what to commit. A strong
programmer with an AI assistant should get through steps 1–3 in an afternoon, most of it waiting
on builds. Questions: open an issue or a draft PR with the log attached.

**What this is.** Lean 4 neural-network trainers whose train steps are proof-rendered StableHLO,
run on the GPU through a PJRT plugin. On a Mac the plugin is [jax-graft](https://github.com/dfm/jax-graft)
(XLA with a Metal backend). The only Mac on record is a 2020 M1 Air ([`MAC.md`](MAC.md)); it is
slow, and a newer chip is wanted for the platforms table the book prints (`PLATFORMS.md`,
Appendix A). Nothing here changes a proof or a render — it is the same committed artifacts,
on a new GPU, with the numbers written down.

## Requirements

Apple Silicon, **macOS 26 or later** (the plugin's wheel is `macosx_26_0_arm64`), Xcode
command-line tools, Python 3.12+ (Homebrew or uv), [elan](https://github.com/leanprover/elan),
~15 GB free. 16 GB of memory is enough; 24+ is more comfortable for the 224² nets. Plug it in.

```bash
git clone https://github.com/brettkoonce/lean4-mlir && cd lean4-mlir
git checkout mac-ladder                      # the branch with the Mac scripts
git checkout -b mac-<your-chip>              # your work goes on its own branch, off that one
```

## 1. Smoke test — `deploy/mac_setup.sh` (~30–60 min, mostly the Lean build)

```bash
bash deploy/mac_setup.sh
```

Seven rungs, each proving one layer before the next leans on it; it stops at the first failure
with the layer named, and `bash deploy/mac_setup.sh <rung>` resumes there after a fix
([`MAC.md`](MAC.md) § Run it has the rung table). Green is the last rung, `mnist`, printing three
accuracies close to these:

| | M1 Air | CUDA (median of 30 seeds) |
|---|---|---|
| MNIST linear / MLP / CNN, 12 epochs | 92.10 / 97.79 / 98.71 % | 92.10 / 97.78 / 98.78 % |

The linear one should match exactly; the other two within a few tenths. Everything is logged to
`runs/<date>-mac-setup/setup.log`, with the box's chip, memory, GPU cores, macOS and package
versions in `manifest.txt` beside it.

**Commit:** `runs/<date>-mac-setup/` (both files) and the three `runs/<date>-mnist-*/` log
directories. If a rung fails and you cannot fix it, commit the setup log anyway and open a draft
PR — the log is the finding.

If `uname -m` prints `x86_64`, your terminal runs under Rosetta. The scripts re-exec themselves
as arm64, but a bare `lake build` typed in that terminal compiles the C helpers as x86_64 and the
link then fails (`undefined symbol: lean_f32_*`). Open a native shell (`arch -arm64 zsh`) for
anything by hand.

## 2. Benchmark — `deploy/mac_bench.sh` (~10–20 min)

```bash
bash deploy/mac_bench.sh
```

Builds the CIFAR batch-BN trainer and the seven Imagenette trainers (the first build of the big
nets is the long part), times each one's train step on the GPU for 70 steps of synthetic data,
and prints `runs/<date>-mac-bench/estimate.txt`: the step time per net, and from it how long
the CIFAR ladder (step 4) and each 80-epoch Imagenette run (step 5) would take on this Mac.
Those estimates decide what you run next.

**Commit:** `runs/<date>-mac-bench/` (`bench.tsv`, `estimate.txt`, `commit.txt`, the logs).

## 3. The platform suite — `scripts/platform/check.sh` (~10 min)

```bash
scripts/platform/check.sh
```

Tiers 0–2 of the platform integration suite: the plugin loads, the shim's client and compile
options work, and 16 committed StableHLO artifacts (one per op family, f32 and bf16) run on the
GPU and match XLA:CPU goldens. Expect three `SKIP`s (the two-device tests; a Mac has one GPU).
A `FAIL` is a finding, not a blocker: commit it as it is and name it in the PR. The suite writes
`runs/platform/<date>-<host>-metal/` and regenerates `PLATFORMS.md` from `scripts/platform/support.tsv`.

**Commit:** `runs/platform/<date>-<host>-metal/manifest.json` and `results.tsv` (the `logs/`
directory is untracked by design).

## 4. The CIFAR ladder — `deploy/mac_ladder.sh` (hours; the bench says how many)

```bash
bash deploy/mac_ladder.sh
```

What the 7900 XTX and the Arc B50 ran at CIFAR and below: the suite again, MNIST, the batch-BN
gradient check, `lake run cifar` (twelve arms, 40 epochs), five seeds of the batch-BN net and a
ten-width head sweep. One job at a time on the one GPU; resumable (`bash deploy/mac_ladder.sh
<stage>`); `caffeinate` keeps the Mac awake; `sensors.tsv` records the thermal speed limit once
a minute, which shows whether a fanless machine throttled. [`MAC.md`](MAC.md) § The ladder has
the stage table and the 7900 XTX's numbers to compare against. Run it overnight if the bench
puts it under ~8 h; otherwise run `bash deploy/mac_ladder.sh build data suite mnist gradcheck cifar`
and skip the seeds and grid.

**Commit:** `runs/<date>-mac-ladder/` and the `runs/<date>-{mnist,cifar8w}*/` trainer logs it
names at the end.

## 5. Imagenette (hours per net; the bench says how many)

```bash
.venv/bin/pip install Pillow && PATH=$PWD/.venv/bin:$PATH bash scripts/datasets/download_imagenette.sh
LEAN_MLIR_IMAGENETTE_STREAM=1 LEAN_MLIR_CKPT_TAG=<chip>80 lake run train r34
```

ResNet-34 first: 80 epochs of AdamW at 224², batch 32, checkpointed every epoch (Ctrl-C and the
same command resume it, but a resumed run is a valid run and not the same run — the shuffle
state is not in the checkpoint, so run each net in one piece where you can; never `kill -9` —
the GPU also drives the display). CUDA lands at
89.68–90.37 % over five seeds, the 7900 XTX at 90.47 %. Then, if the bench's estimates allow,
`vit`, `mnv2`, `enet`, `mnv4`, `r50`, `cnx` the same way (`lake run train` with no argument lists
them). `LEAN_MLIR_IMAGENETTE_STREAM=1` streams the train split instead of keeping 7.4 GB of it
resident; leave it on at 16 GB, optional above.

**Commit:** each `runs/<date>-<net>-verified-adam/` log directory.

## 6. The platforms table

Add a row for your Mac to `scripts/platform/support.tsv` — copy the `Apple M1 MacBook Air` row,
change the name, the `env` column's host (`hostname -s`), the suite cell to `auto:<host>` (it is
then filled from your suite run), and each tier cell to the date, a link to the run directory and
the result, in the row's existing style. Then:

```bash
.venv/bin/python3 scripts/platform/platforms_md.py        # regenerates PLATFORMS.md and the book's table
.venv/bin/python3 scripts/platform/platforms_md.py --check
```

**Commit:** `support.tsv`, `PLATFORMS.md`, `blueprint/src/figures/platforms.tex`. Add a
"§ Measured on <chip>" to [`MAC.md`](MAC.md) in the shape of § Measured, with the numbers.

## 7. The PR

First `python3 scripts/gates/gen_mlir_manifest.py` and commit `verified_mlir/MANIFEST.md`: its
runs column counts the run directories that used each artifact, yours included, and CI fails
when it is stale. Then open the PR against `main`, from your branch. Title it
`runs(mac <chip>): …` and say in the body which
steps ran, which did not and why, and anything that had to be fixed to get there — a fix to a
script is welcome in the same PR. Do not commit `data/`, `.venv/`, `.lake/`, the
`runs/platform/*/logs/` directories, or checkpoints (`.lake/build/*_ckpt*`); `.gitignore`
already covers them.

## If something breaks

- **`mac_setup.sh` stops at `jax`:** the plugin itself does not run on this macOS/Python; nothing
  in this repo is involved yet. The Python error is the report.
- **It stops at `smoke`:** the shim ↔ plugin seam (`ffi/pjrt_ffi.c`); `PJRT_FFI_TRACE=1` names
  the call that failed and that is the report.
- **A trainer dies with an unsupported op:** jax-graft refuses an op at compile time, by name,
  rather than computing it wrongly. The name is the report.
- **Out of memory on a 224² net:** `LEAN_MLIR_IMAGENETTE_STREAM=1` first; then
  `METAL_PJRT_MEMORY_FRACTION=0.7` (the plugin's budget, default half of RAM).
- **Slow and getting slower:** `runs/<date>-mac-ladder/sensors.tsv` — a `cpu_speed_limit`
  under 100 is thermal throttling. Say so in the PR; a throttled time is still a time.
