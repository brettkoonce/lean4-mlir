#!/usr/bin/env bash
# Sampler x NFE scaling on the 2-D toy — planning/diffusion_2d_demo.md.
#
# ⭐ The question: at a fixed budget of network evaluations, which reverse-process
# solver gets closest to the data? All arms run on the SAME weights (`reuse`), so
# the only thing varying is the solver and its step count.
#
# ⚠ The x-axis is NFE, not steps. Heun is second-order and spends two evaluations
# per step, so at NFE=20 it takes 10 steps. Comparing at equal STEPS would hand it
# a 2x compute advantage and call the result a win.
#
# Usage: scripts/toy2d_sampler_sweep.sh            # 8-gaussians, 512 samples
#        NGEN=2048 NFES="20 50" scripts/toy2d_sampler_sweep.sh
set -u
TARGET=${TARGET:-eight_gaussians}
NGEN=${NGEN:-512}
STEPS=${STEPS:-20000}
NFES=${NFES:-"10 20 50 100 200"}
# arm = sampler:eta%. eta only means anything to ddim; the others are eta-free.
ARMS=${ARMS:-"ddim:0 ddim:25 euler:0 heun:0 sde:0"}
# Extra flags passed to every arm, e.g. EXTRA=logsnr to change the grid spacing.
EXTRA=${EXTRA:-}

printf "%-12s %6s %8s %10s %9s   (%s)\n" sampler NFE recall energy off-supp "${EXTRA:-uniform-t}"
printf -- "----------------------------------------------------------\n"
for arm in $ARMS; do
  s=${arm%%:*}; e=${arm##*:}
  for n in $NFES; do
    if ! lake exe diffusion-2d "$TARGET" "$STEPS" "$n" "$e" "$NGEN" reuse "$s" $EXTRA \
         >/dev/null 2>&1; then
      printf "%-12s %6s %8s %10s %9s\n" "$s(e$e)" "$n" ERR ERR ERR; continue
    fi
    m=$(python3 scripts/toy2d_metrics.py --target="$TARGET" 2>&1)
    rec=$(sed -n 's/.*cell recall *\([0-9]*\/[0-9]*\).*/\1/p' <<<"$m" | head -1)
    en=$(sed -n 's/.*energy distance *\([0-9.]*\).*/\1/p'      <<<"$m" | head -1)
    off=$(sed -n 's/.*off-support *\([0-9.]*\)%.*/\1/p'        <<<"$m" | head -1)
    printf "%-12s %6s %8s %10s %8s%%\n" "$s(e$e)" "$n" "$rec" "$en" "$off"
  done
done
