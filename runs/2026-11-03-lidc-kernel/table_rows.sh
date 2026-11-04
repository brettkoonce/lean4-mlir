#!/usr/bin/env bash
# The rows of the book's tab:ct_synthesis, every cell from a score file (plan §12.4). The LIDC block of
# the §9/§11 rows reads the range run's texture / nodule passes (the same dumps re-scored there); the
# Mayo block reads this run's per-slice csv (all Mayo arms side by side) and each arm's fidelity report.
# The kernel rows: home = scored on the b50f val (data/lidc10b50), cross = the b50f nets on the
# Shepp-Logan val — both printed; the section picks (§12.4).
set -euo pipefail
cd "$(dirname "$0")/../.."
K=runs/2026-11-03-lidc-kernel/score
R=runs/2026-11-03-lidc-range/score
PS=$K/perslice_mayo.csv
T=$R/texture_lidc10.json; N=$R/nodules_lidc10.json
TK=$K/texture_lidc10b50.json; NK=$K/nodules_lidc10b50.json
TX=$K/texture_lidc10.json; NX=$K/nodules_lidc10.json
.venv-lidc/bin/python scripts/demos/lidc_synthesis_table.py \
 --row "low dose, the input | texture=$T#low dose | nodules=$N#quarter dose | fidelity=$R/fidelity_mayo_mse.mayo_val.txt#floor | perslice=$PS#floor" \
 --row "Gaussian | texture=$T#gauss | nodules=$N#Gaussian | fidelity=$R/fidelity_mayo_gauss.val.txt | perslice=$PS#gauss" \
 --row "total variation | texture=$T#tv | nodules=$N#TV | fidelity=$R/fidelity_mayo_tv.val.txt" \
 --row "BM3D | texture=$T#bm3d | nodules=$N#BM3D | fidelity=$R/fidelity_mayo_bm3d.val.txt" \
 --row "\\textbf{MSE UNet} | texture=$T#mse | nodules=$N#MSE UNet | fidelity=$K/fidelity_mayo_mse.mayo_val.txt | perslice=$PS#mse" \
 --row "\\quad the same, trained at doses 3--25\\,\\% | texture=$T#msemix | nodules=$N#MSE UNet, range | fidelity=$K/fidelity_mayo_msemix.mayo_val.txt | perslice=$PS#msemix" \
 --row "\\textbf{diffusion}, one sample | texture=$T#diff.k0 | nodules=$N#diffusion k0 | fidelity=$K/fidelity_mayo_diff.mayo_val.k0.txt | perslice=$PS#diff.k0" \
 --row "\\quad the same, at doses 3--25\\,\\%, told the level | texture=$T#diffmixsig.k0 | nodules=$N#diffusion, range + σ̂ k0 | fidelity=$K/fidelity_mayo_diffmixsig.mayo_val.k0.txt | perslice=$PS#diffmixsig.k0" \
 --row "\\quad the same, trained on Mayo's kernel (home: b50f val) | texture=$TK#diff.k0 | nodules=$NK#diffusion k0 | fidelity=$K/fidelity_mayo_diff_b50.mayo_val.k0.txt | perslice=$PS#diff_b50.k0" \
 --row "\\quad the same, trained on Mayo's kernel (cross: SL val) | texture=$TX#diff_b50.k0 | nodules=$NX#diffusion, b50f k0 | fidelity=$K/fidelity_mayo_diff_b50.mayo_val.k0.txt | perslice=$PS#diff_b50.k0" \
 --row "\\quad MSE UNet, trained on Mayo's kernel (home: b50f val) | texture=$TK#mse | nodules=$NK#MSE UNet | fidelity=$K/fidelity_mayo_mse_b50.mayo_val.txt | perslice=$PS#mse_b50" \
 --row "\\quad MSE UNet, trained on Mayo's kernel (cross: SL val) | texture=$TX#mse_b50 | nodules=$NX#MSE UNet, b50f | fidelity=$K/fidelity_mayo_mse_b50.mayo_val.txt | perslice=$PS#mse_b50" \
 --row "the Shepp-Logan nets on the b50f val: diffusion | texture=$TK#diff_sl.k0 | nodules=$NK#diffusion, Shepp-Logan k0" \
 --row "the Shepp-Logan nets on the b50f val: MSE UNet | texture=$TK#mse_sl | nodules=$NK#MSE UNet, Shepp-Logan"
