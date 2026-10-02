#!/usr/bin/env bash
# Queue 15 (20:30) — the pair-input op, plan §11a item 1, tier A: ESM-2 650M's own contact head as one
# host pair plane on the pair tile (`pair=1`, Layer.pairTile's pairIn). The pool's planes from the
# 650M contact head in two shards on GPUs 2 and 3 (after queue14's four-shard embed, if that queue is
# running), the EUs' and val subset's planes packed, then 650M × 64 ch × 30 ep with pair=1 on GPU 2 and
# finish — against the 64-ch 650M baseline (0.838, fold 0.569 / 0.572).
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
if [ -f $A/queue14.log ]; then
  while ! grep -q "pack esm3b\|pool incomplete" $A/queue14.log; do sleep 60; done
fi
echo "[$(date +%H:%M)] 650M contact-head planes -> pool_pair_esm650.bin, two shards on GPUs 2, 3"
for k in 0 1; do
  CUDA_VISIBLE_DEVICES=$((k + 2)) $P -u scripts/datasets/casp16_embed.py --model esm2_t33_650M_UR50D --out emb650 \
    --pair-out data/casp16/packed/pool_pair_esm650.bin --device cuda --shard $k/2 --threads 4 --max-tokens 6144 --max-pairs 1500000 \
    > $A/pair650_$k.log 2>&1 &
done
wait
for k in 0 1; do echo "  shard $k: $(grep -v Warning $A/pair650_$k.log | tail -n 1)"; done
n_done=$(cat data/casp16/packed/pool_pair_esm650.bin.done* | sort -u | wc -l); n_pool=$(wc -l < data/casp16/packed/pool_order.txt)
[ "$n_done" -eq "$n_pool" ] || { echo "pair pool incomplete: $n_done of $n_pool chains"; exit 1; }
echo "[$(date +%H:%M)] pack the EUs' and valsub's planes"
$P -u scripts/datasets/casp16_pack.py --features esm650 --pair-only 2>&1 | tail -n 2
echo "[$(date +%H:%M)] train 650M × 64 ch × 30 ep, pair=1, on GPU 2"
export CUDA_VISIBLE_DEVICES=2
d=runs/$(date +%F)-distogram-r16x64-e30-esm650-pair1; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=32 ch=64 units=16 pair=1 seed=1 tag=e30-esm650-pair1 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full fs=esm650 dim=1289 ch=64 units=16 pair=1 tag=e30-esm650-pair1 > $A/finish_esm650_pair1.log 2>&1
tail -n 5 $A/finish_esm650_pair1.log
echo "[$(date +%H:%M)] queue15 done"
