#!/bin/bash
# Phase 4 / Table 3: fine-tune the seed-1 80-epoch EuroSAT arms on Brazil labels — N = 300 chips (200 epochs, 1,000 steps)
# and the whole training side (80 epochs) — five folds by chip id, lr 1e-4, the 10-wide head kept (outputs 0–6 = the seven classes)
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P1=runs/2026-09-29-rs-phase1; P3=runs/2026-09-29-rs-phase3; GPU=${GPU:-3}
for arm in ${ARMS:-rgb all ir}; do for n in 300 0; do for k in 0 1 2 3 4; do
  if [ $n -gt 0 ]; then ep=200; tag=ft; else ep=80; tag=ft; fi
  CUDA_VISIBLE_DEVICES=$GPU .lake/build/bin/rs-bands arm=$arm train=brazil_all val=brazil_all score=brazil_all classes=10 fold=$k labels=$n epochs=$ep lr=0.0001 seed=1 init=$P1/rs_cifar8w_${arm}_s1 tag=$tag out=$P3 > $P3/ft_${arm}_n${n}_fold${k}.log 2>&1 || { echo "FAILED finetune $arm n$n fold $k"; continue; }
  sfx=$([ $n -gt 0 ] && echo "_n$n" || echo "")
  .venv-rs/bin/python scripts/demos/rs_score.py $P3/rs_cifar8w_${arm}_ft_fold${k}${sfx}_logits_brazil_all.bin --part brazil_all --fold $k --head brazil --json > $P3/score_ft_${arm}_n${n}_fold${k}.json 2>&1 || echo "FAILED score finetune $arm n$n fold $k"
  echo "finetune $arm n=$n fold $k: $(grep '^brazil_all' $P3/score_ft_${arm}_n${n}_fold${k}.json | sed 's/ (7-way.*acc / /; s/ \[.*macro-F1/ F1/')"
done; done; done
echo FINETUNE_DONE
