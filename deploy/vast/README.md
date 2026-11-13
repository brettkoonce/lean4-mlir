# The bench box as a vast.ai image

An image that boots on a rented multi-card CUDA instance and measures it, with no setup on the
instance: `vastai/base-image` plus this tree built (the pinned CUDA 12.9 venv, the PJRT shim, the
Lean toolchain, Mathlib's cache, the verified ImageNet trainers, the JAX trainers), and a
supervisor program that runs [`bench.sh`](bench.sh) once at boot.

| file | what it is |
|---|---|
| `Dockerfile`, `Dockerfile.dockerignore` | the image; built from the repo root |
| `build.sh` | `docker build` with the revision as the tag; pushes nothing |
| `bench.sh` | the boot run: box facts, JAX on every card, shim smoke, bare-card rows, fed rows if ImageNet is there |
| `supervisor.conf`, `supervisor.sh` | the base image's hook that starts `bench.sh` |

## 1. Build and push (any box with docker and ~70 GB free; no GPU needed)

    git clone https://github.com/brettkoonce/lean4-mlir.git && cd lean4-mlir
    git checkout bench/a100-probe
    docker login                                         # the Docker Hub account the image goes to
    IMAGE=<hub user>/lean4-mlir deploy/vast/build.sh
    docker push <hub user>/lean4-mlir:<rev>         # build.sh prints this line

The Lean build is the long step (Mathlib's cache downloads; the trainers and their renders
compile). The image is ~45 GB unpacked. A source edit rebuilds from `COPY . .` down; the venv
layer is keyed on `jax/requirements-cuda-lock.txt` alone.

Try it before pushing, on a box with NVIDIA cards and the container toolkit:

    docker run --rm --gpus all --entrypoint bash <hub user>/lean4-mlir:<rev> \
      -c 'BENCH_OUT=/tmp/bench bash /opt/lean4-mlir/deploy/vast/bench.sh'

## 2. Rent

A vast.ai template with:

- **Image** `<hub user>/lean4-mlir:<rev>`
- **Launch mode** docker ENTRYPOINT. In SSH or Jupyter mode put `entrypoint.sh` in the on-start
  script so the base image's boot (and with it supervisor) still runs.
- **Disk** 80 GB without ImageNet, 250 GB with it
- **Environment**, all optional:

| variable | default | |
|---|---|---|
| `BENCH_PRICE` | – | the instance's $/hr; adds a cost column to the summary |
| `BENCH_NETS` | `a1 vitb cnxs cnxb a2 vits` | the bare-card rows, in order |
| `BENCH_FED_NETS` | `a1 vitb cnxs cnxb` | the fed rows |
| `BENCH_IMAGES` | `150000` | images per fed row |
| `BENCH_AUTORUN` | `1` | `0` boots without running |
| `TFDS_DATA_DIR` | `/workspace/tensorflow_datasets` | where `imagenet2012/` is looked for |

The rows are the four-card jobs' own renders, so the instance wants four cards of 16 GB or more
(two cards fall back to the `*-2card` renders).

## 3. Read it

The instance log (and `/var/log/portal/lean4-mlir-bench.log`) follows the run and ends on

    ════ BENCH DONE … — results in /workspace/lean4-mlir-bench/<date>-<host>/

with `summary.txt` (step time, peak memory per card, and the hours a verified pass takes at that
step), `device-<net>.txt`, `box.txt` and `bench.log` beside it. Copy that directory off, then
**destroy the instance: nothing here stops the rental.** By hand, or again:

    BENCH_FORCE=1 bash /opt/lean4-mlir/deploy/vast/bench.sh

## 4. With ImageNet

Without `imagenet2012` under `$TFDS_DATA_DIR` the run stops after the bare-card rows. With it,
the tree is read once into page cache, the loader is measured alone at 8-64 workers, and
`lake run bench` runs each fed net on both paths. The tree is the tfds one the trainers read,
144 GB:

    rsync -a --info=progress2 ~/tensorflow_datasets/imagenet2012 <host>:tensorflow_datasets/

⚠ The job confs' `SHIM_WORKERS` (4 or 8) are ares's; on cards several times faster the fed rows
will read loader-bound until those are raised. The loader sweep above is what says to what.
