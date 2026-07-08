# RunPod probe v0 — imagenette benchmarks (no Lean toolchain needed on the pod)

Committed copies of the Lean-generated JAX trainers (normally gitignored in
`jax/.lake/build/`) so a fresh pod can run them with nothing but git + pip. Generated
2026-07-08 from `jax/MainResnet50.lean` / `MainResnet.lean` / `MainVit.lean` — do not
edit by hand; regenerate via `cd jax && lake exe {resnet50,resnet34,vit-tiny}` and re-copy.

| script | net | params | batch | note |
|---|---|---|---|---|
| `probe_resnet50_imagenette.py` | ResNet-50 | 23.5M | see header print | closest to the A3/R50 program runs |
| `probe_resnet34_imagenette.py` | ResNet-34 | 21.3M | 192 | matches the local klawd/mars reference chapters |
| `probe_vit_tiny_imagenette.py` | ViT-Tiny | 5.5M | see header print | the transformer/GEMM probe |

## Pod runbook (run from the repo root)

```bash
git clone -b runpod-probe --depth 1 git@github.com:brettkoonce/lean4-mlir.git lean4-jax
cd lean4-jax
pip install -r jax/requirements-cuda-lock.txt   # NVIDIA pod; rocm lock on AMD
python -c "import jax; print(jax.devices(), jax.devices()[0].device_kind)"
./download_imagenette.sh                        # ~1.5 GB; needs `pip install Pillow`
nvidia-smi                                      # note the card + free VRAM

python jax/probe/probe_resnet50_imagenette.py   # data_dir baked as data/imagenette
python jax/probe/probe_resnet34_imagenette.py   #   (relative → run from repo root)
python jax/probe/probe_vit_tiny_imagenette.py
```

## Reading the numbers

- Trainers print `[Epoch N] ... elapsed` each epoch — ms/epoch is the benchmark.
  **Epoch 1 includes XLA compile + conv autotune — throw it away**; steady state is
  epoch 2+. Imagenette epochs are short; a handful of epochs is enough signal.
  Ctrl-C freely — nothing on this pod is precious, no checkpoints matter.
- Capture per run: card, steady ms/epoch (and its batch/steps: 9469 train imgs),
  peak VRAM (`nvidia-smi` mid-run or `jax.local_devices()[0].memory_stats()`),
  and any crash traceback in full.
- Reference points to beat/compare (local): klawd A3 step ~250 ms @160px bs512 on
  4× 4060 Ti (=2048 img/s); klawd A2 step ~457 ms @224px.
- If OOM: halve batch by editing the `BATCH_SIZE` line in the script copy, note it.

The full probe plan (A3-shaped R50 + ViT-S DeiT-shaped configs, ImageNet-scale
extrapolations) is `planning/mi300x_rental_program.md` §Run-0a — this directory is
its v0 so pod-side patterns can be tested before that coding session happens.
