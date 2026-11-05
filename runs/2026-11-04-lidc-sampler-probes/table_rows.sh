#!/usr/bin/env bash
# The clipped-x̂₀ row of tab:ct_synthesis (plan §13.4): the §13 both-at-once generator sampled with --clamp-x0.
# The LIDC block is the b50f 10 % val (texture, nodules), the Mayo block the 1,000 held-out slices.
cd "$(dirname "$0")/../.."
S=runs/2026-11-04-lidc-sampler-probes/score
.venv-lidc/bin/python scripts/demos/lidc_synthesis_table.py \
  --row "\\quad\\quad the same, \$\\hat x_0\$ clipped to the HU range | texture=$S/texture_lidc10b50.json#diffmixsig_clampx0.k0 | nodules=$S/nodules_lidc10b50.json#diffmixsig_clampx0 k0 | fidelity=$S/fidelity_mayo_diffmixsig_b50_clampx0.txt | perslice=$S/perslice_mayo.csv#diffmixsig_b50_clampx0.k0"
