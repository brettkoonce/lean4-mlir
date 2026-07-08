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
git clone -b runpod-probe --depth 1 https://github.com/brettkoonce/lean4-mlir.git lean4-jax
cd lean4-jax
python -m venv /root/venv && source /root/venv/bin/activate   # dodge the image's stack
# the iree-*rc pins live on iree.dev, not PyPI, and the JAX probes don't need them:
grep -v "^iree" jax/requirements-cuda-lock.txt > /tmp/req-jax.txt   # rocm lock on AMD
python -m pip install -r /tmp/req-jax.txt Pillow
# (only for `lake run benchmark` pods: python -m pip install "iree-base-compiler==<pin>" \
#    "iree-base-runtime==<pin>" -f https://iree.dev/pip-release-links.html)
python -c "import jax; print(jax.devices(), jax.devices()[0].device_kind)"
./download_imagenette.sh                        # ~1.5 GB; run on LOCAL disk, not /workspace
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

## `benchmark.py` — the one-command `lake run benchmark` for the JAX path

Probe → estimate in one shot, no hand-feeding estimate.py. Reuses the committed
generated models, loads real pixels from `data/imagenette`, times the effective
train step, and prints per-run wall-clock estimates:

```bash
python jax/probe/benchmark.py                       # R50-A3, all visible GPUs
python jax/probe/benchmark.py --net vit-tiny        # ViT-Ti @DeiT bs1024 + S/B rows
CUDA_VISIBLE_DEVICES=0 python jax/probe/benchmark.py --eff-steps 5
```

- `--net r50-a3` (default): the R50-A3 demo, rsb-faithful tier — R50 @160px, LAMB
  lr 8e-3 @ eff-bs2048 via grad-accum (Ghost-BN per micro), BCE, wd skip-list,
  bf16 conv on CUDA. Model from `probe_resnet50_imagenette_noaug.py`.
- `--net vit-tiny`: the committed vit demo step (`probe_vit_tiny_imagenette.py`,
  AdamW) at DeiT batch 1024 @224, bf16 matmul on CUDA; prints DeiT-300ep rows for
  ViT-Ti plus FLOP-scaled ViT-S/16 and ViT-B/16 (estimate.py's scaling model).

Per-device micro-batch defaults per net (r50: 128 = the demo's 512/4; vit: 256 =
DeiT's 1024/4) with accum derived to keep the effective batch, so numbers compare
across device counts. Measured 2026-07-08 on ares 4× 4060 Ti (0,2,3,4), bf16:
R50-A3 eff-step 848 ms → 2,417 img/s (ImageNet A3 100ep ≈ 14.7 h; 1 GPU 771 img/s
≈ 46 h); ViT-Ti 262 ms @1024 → 3,902 img/s (DeiT 300ep: Ti ≈ 27 h, S ≈ 100 h,
B ≈ 382 h). Train step only — eval, host aug (mixup/cutmix/RA), and checkpointing
excluded, same philosophy as LEAN_MLIR_BENCH_SYNTH.
