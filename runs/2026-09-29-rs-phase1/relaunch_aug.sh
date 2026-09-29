#!/bin/bash
# wait for the no-augmentation ladder's seed 1 to finish, stop it, rebuild with the dihedral gather, run the augmented ladder
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
while ! grep -q "on eurosat_test" runs/2026-09-29-rs-phase1/ir_s1.log 2>/dev/null; do sleep 10; done
pkill -f "[f]or seed in 1 2 3; do for arm in rgb" && echo "no-aug ladder stopped after seed 1"
sleep 2; pkill -f "[r]s-bands arm=" || true
mkdir -p runs/2026-09-29-rs-phase1/noaug && mv runs/2026-09-29-rs-phase1/*_s1.log runs/2026-09-29-rs-phase1/rs_cifar8w_*_s1_* runs/2026-09-29-rs-phase1/noaug/ 2>/dev/null
rm -f runs/2026-09-29-rs-phase1/*_s2.log runs/2026-09-29-rs-phase1/rs_cifar8w_*_s2_* 2>/dev/null
lake build rs-bands > runs/2026-09-29-rs-phase1/build_aug.log 2>&1 || { echo "BUILD FAILED"; tail -30 runs/2026-09-29-rs-phase1/build_aug.log; exit 1; }
echo "built"
for seed in 1 2 3; do for arm in rgb rgbn ms10 all ir; do
  CUDA_VISIBLE_DEVICES=0 .lake/build/bin/rs-bands arm=$arm epochs=20 seed=$seed tag=s$seed out=runs/2026-09-29-rs-phase1 > runs/2026-09-29-rs-phase1/${arm}_s${seed}.log 2>&1 || echo "FAILED $arm s$seed"
  grep "on eurosat_test" runs/2026-09-29-rs-phase1/${arm}_s${seed}.log | sed "s/^/s$seed /"
done; done
echo LADDER_DONE
