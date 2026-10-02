#!/usr/bin/env bash
# Queue 16 (21:35) — the two levers stacked: ESM-2 3B features + the 3B contact head's logit plane
# (`pair=1`), on GPU 3 (idle since queue14's embed phase). The pool's 3B planes in fp16 (the 3B model
# is 11 GB in fp32; L ≤ 512 in the pool, 400k-pair batches), the EUs' planes from the fp32 CPU pass
# already in targets_esm3b/<EU>.npz (the 1,693-residue target's maps fit no card) and the val subset's
# out of the pool file, then 3B × 64 ch × 30 ep pair=1 and finish — against 3B × 64 ch (queue14) and
# 650M × 64 ch pair=1 (queue15), which each read ~+2 pt on val at epoch 15 over the 650M baseline.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue16_esm3b_pair.sh > runs/2026-10-02-distogram-ablations/queue16.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=3
echo "[$(date +%H:%M)] 3B contact-head planes -> pool_pair_esm3b.bin (fp16, GPU 3)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P -u scripts/datasets/casp16_embed.py --model esm2_t36_3B_UR50D --out emb3b \
  --pair-out data/casp16/packed/pool_pair_esm3b.bin --device cuda --half --threads 4 --max-tokens 6144 --max-pairs 400000 \
  > $A/pair3b.log 2>&1
echo "  $(grep -v Warning $A/pair3b.log | tail -n 1)"
n_done=$(cat data/casp16/packed/pool_pair_esm3b.bin.done* | sort -u | wc -l); n_pool=$(wc -l < data/casp16/packed/pool_order.txt)
[ "$n_done" -eq "$n_pool" ] || { echo "pair pool incomplete: $n_done of $n_pool chains"; exit 1; }
echo "[$(date +%H:%M)] pack the EUs' (npz, fp32) and valsub's (pool, fp16) planes"
$P -u scripts/datasets/casp16_pack.py --features esm3b --pair-only 2>&1 | tail -n 2
[ -s data/casp16/packed/targets_pair_esm3b.bin ] || { echo "no targets_pair_esm3b.bin"; exit 1; }
echo "[$(date +%H:%M)] train 3B × 64 ch × 30 ep, pair=1, on GPU 3"
d=runs/$(date +%F)-distogram-r16x64-e30-esm3b-pair1; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=30 batch=32 ch=64 units=16 pair=1 seed=1 tag=e30-esm3b-pair1 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 3 list=train_full fs=esm3b dim=2569 ch=64 units=16 pair=1 tag=e30-esm3b-pair1 > $A/finish_esm3b_pair1.log 2>&1
tail -n 5 $A/finish_esm3b_pair1.log
echo "[$(date +%H:%M)] queue16 done"
