#!/bin/bash
# Phase 3a: every 80-epoch EuroSAT arm × seed scored zero-shot on the three Brazil parts; one JSON per (arm, seed, part)
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P1=runs/2026-09-29-rs-phase1; P3=runs/2026-09-29-rs-phase3; GPU=${GPU:-1}
for seed in ${SEEDS:-1 2 3}; do for arm in rgb rgbn ms10 all ir; do
  [ -f $P1/rs_cifar8w_${arm}_s${seed}_params.bin ] || { echo "skip $arm s$seed (no checkpoint yet)"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU .lake/build/bin/rs-bands arm=$arm eval tag=s$seed out=$P1 score=eurosat_test,amazon_dry,cerrado_dry,cerrado_wet > $P3/eval_${arm}_s${seed}.log 2>&1 || { echo "FAILED eval $arm s$seed"; continue; }
  for part in eurosat_test amazon_dry cerrado_dry cerrado_wet; do
    .venv-rs/bin/python scripts/demos/rs_score.py $P1/rs_cifar8w_${arm}_s${seed}_logits_${part}.bin --part $part --json > $P3/score_${arm}_s${seed}_${part}.json 2>&1 || echo "FAILED score $arm s$seed $part"
  done
  .venv-rs/bin/python scripts/demos/rs_score.py $P1/rs_cifar8w_${arm}_s${seed}_logits_cerrado_dry.bin --part cerrado_dry --pair $P1/rs_cifar8w_${arm}_s${seed}_logits_cerrado_wet.bin --pair-part cerrado_wet --json > $P3/pair_${arm}_s${seed}.json 2>&1 || echo "FAILED pair $arm s$seed"
  echo "scored $arm s$seed: $(grep -h '^amazon_dry\|^cerrado_dry\|^cerrado_wet' $P3/score_${arm}_s${seed}_{amazon_dry,cerrado_dry,cerrado_wet}.json | sed 's/ (7-way.*acc / /; s/ \[.*macro-F1/ F1/' | tr '\n' ';')"
done; done
echo ZERO_SHOT_DONE
