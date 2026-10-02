#!/usr/bin/env bash
# Queue 14 (20:15, Brett: "use the GPUs; maybe not the 20 h run out of the gate") — ESM-2 3B, plan §11a
# item 3. The pool (27,690 chains, 6.12 M residues) embedded in fp16 as four shards, one per card, and
# the EUs on the CPU meanwhile (fp32; a 1,693-residue target's 36 × 40 attention maps do not fit a
# card); pack targets + valsub; then two 30-epoch arms, each finished (predict / assemble / fold /
# score): 64 ch on GPU 0 — the ladder's next rung after 35M 0.585 → 150M 0.735 → 650M 0.838 — and
# 128 ch × crop 96 on GPU 1, the best 650M config (0.858). GPUs 2, 3 stay free for the pair-input op.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
free=$(df --output=avail -BG . | tail -n 1 | tr -dc 0-9)
echo "[$(date +%H:%M)] 3B chain; ${free} GB free"
[ "$free" -ge 40 ] || { echo "not enough disk for the 31 GB pool"; exit 1; }
echo "[$(date +%H:%M)] embed 3B -> pool in four shards; the EUs on the CPU"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $P -u scripts/datasets/casp16_embed.py --model esm2_t36_3B_UR50D --out emb3b \
    --pool-out data/casp16/packed/pool_esm3b_feat.bin --device cuda --half --shard $k/4 --threads 4 --max-tokens 8192 \
    > $A/embed3b_$k.log 2>&1 &
done
$P -u scripts/datasets/casp16_targets.py --model esm2_t36_3B_UR50D --out targets_esm3b --device cpu > $A/targets3b.log 2>&1 &
wait
for k in 0 1 2 3; do echo "  shard $k: $(grep -v Warning $A/embed3b_$k.log | tail -n 1)"; done
echo "  targets: $(grep -v Warning $A/targets3b.log | tail -n 1)"
n_done=$(cat data/casp16/packed/pool_esm3b_feat.bin.done* | sort -u | wc -l); n_pool=$(wc -l < data/casp16/packed/pool_order.txt)
[ "$n_done" -eq "$n_pool" ] || { echo "pool incomplete: $n_done of $n_pool chains"; exit 1; }
echo "[$(date +%H:%M)] pack esm3b targets + valsub"
$P -u scripts/datasets/casp16_pack.py --features esm3b --sets targets,valsub 2>&1 | tail -n 3
echo "[$(date +%H:%M)] train esm3b: 64 ch on GPU 0, 128 ch × crop 96 on GPU 1"
(
  export CUDA_VISIBLE_DEVICES=0
  d=runs/$(date +%F)-distogram-r16x64-e30-esm3b; mkdir -p $d
  lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-esm3b > $d/train.log 2>&1
  echo "[$(date +%H:%M)] trained 64 ch: $(grep trained $d/train.log)"
  MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm3b dim=2569 ch=64 units=16 tag=e30-esm3b > $A/finish_esm3b.log 2>&1
  tail -n 5 $A/finish_esm3b.log
) &
(
  export CUDA_VISIBLE_DEVICES=1
  d=runs/$(date +%F)-distogram-r16x128-e30-esm3b-crop96; mkdir -p $d
  lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=30 batch=16 ch=128 units=16 crop=96 seed=1 tag=e30-esm3b-crop96 > $d/train.log 2>&1
  echo "[$(date +%H:%M)] trained 128 ch × crop 96: $(grep trained $d/train.log)"
  MEMFRAC=0.5 $A/finish_run.sh 1 list=train_full fs=esm3b dim=2569 crop=96 ch=128 units=16 tag=e30-esm3b-crop96 > $A/finish_esm3b_crop96.log 2>&1
  tail -n 5 $A/finish_esm3b_crop96.log
) &
wait
echo "[$(date +%H:%M)] queue14 done"
