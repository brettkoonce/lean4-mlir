#!/usr/bin/env bash
# Does the toy's sampler ranking survive at image scale?
#
# The 2-D demo found the reverse SDE beating DDIM at every evaluation budget and
# saturating by NFE 50. That was one target, 2 dimensions, an 18K-param MLP. This
# runs the same arms against the 50-epoch MNIST model, scored by Chapter 3's
# verified CNN — a completely different metric family on a 784-dimensional target.
#
# ⚠ All arms share a uniform-in-t grid, which the toy measured as the SDE's best
# and the explicit ODE solvers' worst. `euler` here is therefore a reference
# point, not a fair test of Euler.
set -u
NGEN=${NGEN:-1024}
NFES=${NFES:-"10 20 50 200"}
ARMS=${ARMS:-"ddim sde euler"}
printf "%-8s %6s %8s %9s %10s %8s %8s\n" sampler NFE cover conf energy pxmean pxsd
printf -- "-----------------------------------------------------------------\n"
for s in $ARMS; do
  for n in $NFES; do
    lake exe mnist-ddpm-score "$NGEN" "$n" "$s" >/dev/null 2>&1 || {
      printf "%-8s %6s %8s\n" "$s" "$n" ERR; continue; }
    python3 scripts/mnist_ddpm_score.py 2>&1 \
      | awk -v s="$s" -v n="$n" '/^  DDPM samples/ {
          printf "%-8s %6s %8s %9s %10s %8s %8s\n", s, n, $3, $4, $5, $7, $8 }'
  done
done
