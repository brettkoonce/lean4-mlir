#!/usr/bin/env bash
# The x₀-target rows of tab:ct_synthesis (plan §13.5): the §12 and §13 generators retrained to predict the clean
# residual. The LIDC block is the b50f 10 % val (texture, nodules), the Mayo block the 1,000 held-out slices.
cd "$(dirname "$0")/../.."
S=runs/2026-11-04-lidc-x0pred/score
.venv-lidc/bin/python scripts/demos/lidc_synthesis_table.py \
  --row "\\quad\\quad the same, predicting the clean slice | texture=$S/texture_lidc10b50.json#diffx0.k0 | nodules=$S/nodules_lidc10b50.json#diffusion, x₀-target k0 | fidelity=$S/fidelity_mayo_diffx0_b50.txt | perslice=$S/perslice_mayo.csv#diffx0_b50.k0" \
  --row "\\quad\\quad the same, predicting the clean slice | texture=$S/texture_lidc10b50.json#diffmixsigx0.k0 | nodules=$S/nodules_lidc10b50.json#diffusion, range + σ̂, x₀-target k0 | fidelity=$S/fidelity_mayo_diffmixsigx0_b50.txt | perslice=$S/perslice_mayo.csv#diffmixsigx0_b50.k0"
