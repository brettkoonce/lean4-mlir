#!/usr/bin/env bash
# Extract the JAX reference's 300-epoch curve into reference_curve.tsv, WITH ASSERTIONS.
#
# ⛔ Run this FIRST, before anything else can be confused with it — the R34 lesson: the plausible
# reference log on disk was a DIFFERENT run (ended 73.95, not the 74.16 the book quotes), and
# nobody noticed because nothing ever asserted its endpoint.
set -eu
cd /home/skoonce/lean/proof_verify_demo/verify-v2
SRC=/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet_full.log
OUT=runs/2026-09-18-cnx-verified-300ep/reference_curve.tsv

printf 'epoch\tlr\ttrain_loss\ttop1_n\ttop1\ttop5\tval_loss\ttrain_s\tval_s\n' > "$OUT"
grep 'val_top1=' "$SRC" | sed -E \
  's/^\[Epoch ([0-9]+)\] lr=([0-9.e+-]+) loss\(train_avg\)=([0-9.]+) val_top1=([0-9]+)\/50000 \(([0-9.]+)\) val_top5=([0-9.]+) val_loss=([0-9.]+) +\[([0-9.]+)s train, ([0-9.]+)s val\].*$/\1\t\2\t\3\t\4\t\5\t\6\t\7\t\8\t\9/' \
  >> "$OUT"

rows=$(( $(wc -l < "$OUT") - 1 ))
[ "$rows" -eq 300 ] || { echo "⛔ expected 300 rows, got $rows"; exit 1; }
# every row must have parsed into 9 fields — an unparsed line would pass the row count
bad=$(awk -F'\t' 'NR>1 && NF!=9' "$OUT" | wc -l)
[ "$bad" -eq 0 ] || { echo "⛔ $bad rows did not parse into 9 fields"; exit 1; }
# the C4 denominator: every eval must be over 50,000, not 49,920
den=$(grep -c '/50000' "$SRC")
[ "$den" -eq 300 ] || { echo "⛔ only $den/300 evals use the 50000 denominator (pre-C4 log?)"; exit 1; }
last1=$(tail -1 "$OUT" | cut -f5); last5=$(tail -1 "$OUT" | cut -f6)
[ "$last1" = "0.8153" ] || { echo "⛔ final top-1 is $last1, expected 0.8153"; exit 1; }
[ "$last5" = "0.9551" ] || { echo "⛔ final top-5 is $last5, expected 0.9551"; exit 1; }

echo "✅ reference_curve.tsv — 300 rows, denominator 50000, final $last1 / $last5"
echo "   ⚠ the book prints 81.53 / 95.50; this log's last epoch is 95.51. Unresolved — see RESULTS.md."
echo "   ⚠ no EMA line in this log: confirm which weights 81.53 is before §8 quotes it."
