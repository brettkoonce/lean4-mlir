#!/bin/bash
# The Kneser-Ney ladder beside each job (runs/2026-10-21-tinygpt-ngram-ladder/README.md's
# protocol): the KN per-position arrays were fit once in this directory (`ngram_ladder.py fit`,
# then `score` under chunked:64, chunked:128 and streaming at T=64 → kn_T64/ and T=128 →
# kn_T128/); each job directory gets symlinks to the arrays at its own window — the `tiny`
# keys train and score at T=128, the `nano` keys at T=64 — and `ngram_ladder.py table` runs
# there, writing table.md / table.csv / context_curve.csv beside the net's score bins.
# Usage: kn_table.sh [<run dir>]   (default: this directory; the dropout run passes its own)
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
L=runs/2026-10-27-tinygpt-ladder-xla; R=${1:-$L}
for d in $R/*/; do
  d=${d%/}
  b=$(ls $d/tinygpt_*_score_chunked.bin 2>/dev/null | head -1)
  [ -n "$b" ] || continue
  key=$(basename "$b" .bin); key=${key#tinygpt_}; key=${key%_score_chunked}
  case $key in tiny*) T=128 ;; *) T=64 ;; esac
  for n in $(seq 1 10); do
    ln -sfr $L/ngram_chunked${T}_n$n.npy $d/ngram_chunked${T}_n$n.npy
    ln -sfr $L/kn_T$T/ngram_streaming_n$n.npy $d/ngram_streaming_n$n.npy
  done
  python3 scripts/demos/ngram_ladder.py table --run $d --nets "$key" --T $T > /dev/null \
    && echo "$(basename $d) ($key, T=$T): $(grep -o 'gap, TinyGPT.*(P2).*' $d/table.md)" \
    || echo "$(basename $d): table FAILED"
done
