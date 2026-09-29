#!/bin/bash
# the Phase 1 ladder at the 80-epoch schedule: five arms × three seeds, one GPU
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
for seed in 1 2 3; do for arm in rgb rgbn ms10 all ir; do
  CUDA_VISIBLE_DEVICES=0 .lake/build/bin/rs-bands arm=$arm epochs=80 seed=$seed tag=s$seed out=runs/2026-09-29-rs-phase1 > runs/2026-09-29-rs-phase1/${arm}_s${seed}.log 2>&1 || echo "FAILED $arm s$seed"
  grep "on eurosat_test" runs/2026-09-29-rs-phase1/${arm}_s${seed}.log | sed "s/^/s$seed /"
done; done
echo LADDER80_DONE
