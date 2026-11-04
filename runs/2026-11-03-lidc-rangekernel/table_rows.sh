#!/usr/bin/env bash
# The two "both at once" rows of the book's tab:ct_synthesis (plan §13.3): the dose range (3–25 %, log-uniform
# per slice) rebuilt under Mayo's kernel, the MSE UNet and the generator told σ̂. Home = the b50f 10 % val
# (data/lidc10b50, the kernel rows' LIDC block); Mayo = the same 1,000 slices as every row. The generator's
# row diverges (pooled 10^21 at home, 10^29 on Mayo): the script prints the order of magnitude and dashes
# the texture cells; its medians (52 / 168 HU) and share worse (6 % / 49 %) are in score/perslice_*.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=runs/2026-11-03-lidc-rangekernel/score
.venv-lidc/bin/python scripts/demos/lidc_synthesis_table.py \
 --row "\\quad the same, both at once (MSE UNet) | texture=$S/texture_lidc10b50.json#msemix | nodules=$S/nodules_lidc10b50.json#MSE UNet, range | fidelity=$S/fidelity_mayo_msemix_b50.mayo_val.txt | perslice=$S/perslice_mayo.csv#msemix_b50" \
 --row "\\quad the same, both at once (diffusion) | texture=$S/texture_lidc10b50.json#diffmixsig.k0 | nodules=$S/nodules_lidc10b50.json#diffusion, range + σ̂ k0 | fidelity=$S/fidelity_mayo_diffmixsig_b50.mayo_val.k0.txt | perslice=$S/perslice_mayo.csv#diffmixsig_b50.k0"
