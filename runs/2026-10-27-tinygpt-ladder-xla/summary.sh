#!/bin/bash
# One line per job: the trainer-logged val (score.log's last line, 12 fixed batches), the
# net's P1 chunked and P2 streaming means and its P2 gap to the best KN order (table.md, from
# kn_table.sh), and the final train loss in bits. Usage: summary.sh [<run dir>]
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
R=${1:-runs/2026-10-27-tinygpt-ladder-xla}
printf '%-18s %8s %8s %8s %8s  %s\n' job train_b logged P1 P2 'gap to KN (P2), 95% CI'
for d in $R/*/; do
  d=${d%/}; [ -f $d/table.md ] || continue
  tr=$(grep -o 'final loss: [0-9.]*' $d/train.log | awk '{printf "%.3f", $3/log(2)}')
  lg=$(grep -o '= [0-9.]* bits/char' $d/score.log | tail -1 | awk '{print $2}')
  p1=$(grep '^| TinyGPT' $d/table.md | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  p2=$(grep '^| TinyGPT' $d/table.md | awk -F'|' '{gsub(/ /,"",$4); print $4}')
  gap=$(grep -o '(P2): [-+0-9.]* bits/char, 95% block-bootstrap CI \[[^]]*\]' $d/table.md | sed 's/(P2): //; s/ bits\/char, 95% block-bootstrap CI//')
  printf '%-18s %8s %8s %8s %8s  %s\n' "$(basename $d)" "$tr" "$lg" "$p1" "$p2" "$gap"
done
