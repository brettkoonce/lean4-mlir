# 2026-10-03/04 — BraTS: the 20k-step 3D UNet against two 2D seeds

`planning/brats_25d_3d.md` §4d has the table and the reading; `tail_table.txt` is
`brats_tail.py`'s output at min-ET 0 / 200 / 500.

```bash
# the 2D anchor's second seed (LEAN_MLIR_SEED moves init, shuffle and augmentation), then its per-patient CSV
runs/2026-10-03-brats-3d/seed2_2d.sh 0
# the seed-0 anchor re-scored for the false-alarm columns (it reproduces 09-29's CSV to a pixel)
CUDA_VISIBLE_DEVICES=2 lake exe brats-eval net=r34 arm=r34 best out=runs/2026-10-03-brats-3d/pervol_2d_r34_s0.csv
# the 09-29 4k-step 3D checkpoint re-scored the same way
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 .venv/bin/python -u jax/scripts/unet3d_brats.py \
    --init runs/2026-09-29-brats-25d/unet3d_params.npz --out runs/2026-10-03-brats-3d/unet3d_1h_rescore
# the 3D UNet at 20k steps, eval + checkpoint every 2k (resume: append --resume runs/2026-10-03-brats-3d/unet3d_20k_ckpt.npz)
runs/2026-10-03-brats-3d/unet3d_20k.sh 3
# the second 3D seed
runs/2026-10-03-brats-3d/unet3d_20k_s1.sh 0
# the comparison
python3 scripts/probes/brats_tail.py pervol_2d_r34_s0.csv pervol_2d_r34_s2.csv unet3d_1h_rescore_pervol.csv \
    unet3d_20k_pervol.csv --names "2D s0,2D s1,3D 1h,3D 20k" --min-et 0,200,500 --paired
```

`unet3d_20k.out` is the trainer's stdout (XLA chatter stripped), with every 2k-step eval;
`unet3d_20k_pervol_s<step>.csv` the per-patient CSV at each. The params (94 MB) and the
resumable checkpoint (282 MB, with both Adam moments) stay unstaged.
