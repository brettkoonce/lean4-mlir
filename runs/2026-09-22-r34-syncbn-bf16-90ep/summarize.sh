#!/usr/bin/env bash
# Archive helper for the R34 bf16 SYNC-BN verified 90-epoch run. Rebuilds every number RESULTS.md
# quotes, from the logs. Read-only on the run; safe to re-run at any time.
#
# ⛔ NOTHING IS COPIED INTO THIS DIRECTORY. The reference curve and the per-replica arm are read
# where they already live, under runs/2026-09-16-r34-bf16-90ep/:
#   * reference_curve.tsv          — extracted ONCE from blueprint/src/content.tex (R34's own JAX
#     log is GONE from this box; the one on disk is a pre-C4 run ending 73.95, not 74.16). That
#     file's header explains it; do not re-derive it from a log.
#   * r34_bf16_verified_curve.csv  — the PER-REPLICA-BN arm this run exists to replace.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-22-r34-syncbn-bf16-90ep
CSV=$R/r34_syncbn_verified_curve.csv
PREV=runs/2026-09-16-r34-bf16-90ep/r34_bf16_verified_curve.csv
RTSV=runs/2026-09-16-r34-bf16-90ep/reference_curve.tsv

[ -s "$RTSV" ] || { echo "⛔ $RTSV missing — the reference curve is not re-derivable from a log."; exit 1; }

# Pairs `Epoch N/90: loss= lr=` with `epoch N: test_acc = C/50000 = P%  top5 = C/50000 = P%`, taking
# the LAST of each (a re-run epoch must not leave two rows). The two PERCENT fields are the only
# `NN.NN%` tokens on the eval line — index off those, not off the `=` signs.
awk '
  /^Epoch [0-9]+\/90: loss=/ { e=$2; sub(/\/90:/,"",e); l=$3; sub(/loss=/,"",l); r=$4; sub(/lr=/,"",r)
                               loss[e+0]=l; lr[e+0]=r; next }
  /^  epoch [0-9]+: .*_acc = / { e=$2; sub(/:/,"",e); c=0
                                 for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+%$/) { c++; v=$i; gsub(/%/,"",v)
                                   if (c==1) t1=v; else if (c==2) t5=v }
                                 if (c>=2) { top1[e+0]=t1; top5[e+0]=t5; if (e+0>mx) mx=e+0 } next }
  END { print "epoch,train_loss,lr,top1,top5"
        for (e=1; e<=mx; e++) if (e in top1) printf "%d,%s,%s,%s,%s\n", e, (e in loss?loss[e]:""), (e in lr?lr[e]:""), top1[e], top5[e] }
' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) epochs)"

echo "--- final"
tail -3 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n", $1, $4, $5}'
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{printf "best sync-BN: top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"

echo "--- launches / rests"
echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   supervisor starts: $(grep -c ' START job=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'ended \(|COMPLETE' "$R/master.log"

echo "--- wall clock"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
if [ -n "$f" ]; then
  awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" -v s="$s" -v f="$f" \
    'BEGIN{d=b-a; printf "launched %s UTC  finished %s UTC  = %d h %d m %d s (%.2f h, %.0f s/epoch)\n", s, f, d/3600, (d%3600)/60, d%60, d/3600, d/90}'
else echo "not COMPLETE yet (launched $s UTC)"; fi

echo "--- pace (sync-BN probe: 174 ms/step at w4 => ~871 s of steps + 37 s eval; > 1000 s is the fault)"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if (d>1000) slow++} {pt=$2}
  END{if(n) printf "%d epoch gaps, mean %.0f s; > 1000 s: %d\n", n, s/n, slow+0}'

echo "--- loader memory, last sample per producer (arena_class is the signal, not total rss)"
awk -F'\t' 'NR>1 && $8>10 {last[$3]=$0} END{for (k in last) print last[k]}' "$R/loader_rss.tsv" 2>/dev/null \
  | sort -t$'\t' -k3 -n | awk -F'\t' '{printf "pid %s  age %5.1f h  rss %5.2f GiB  arena %5.2f GiB  maps %s\n", $3, $4/3600, $5, $6, $7}'

echo "--- ⭐ sync-BN vs per-replica vs reference, 15-epoch windows"
awk -F'[,\t]' '
  FILENAME==ARGV[1] && FNR>1 { s1[$1+0]=$4; s5[$1+0]=$5; next }
  FILENAME==ARGV[2] && FNR>1 { p1[$1+0]=$4; p5[$1+0]=$5; next }
  { r1[$1+0]=$2; r5[$1+0]=$3 }
  END { for (w=0; w<6; w++) { a=w*15+1; b=(w+1)*15; n=0; x1=x5=y1=y5=z1=z5=0
          for (e=a; e<=b; e++) if ((e in s1) && (e in p1) && (e in r1)) {
            x1+=s1[e]; x5+=s5[e]; y1+=p1[e]; y5+=p5[e]; z1+=r1[e]; z5+=r5[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): sync %.2f/%.2f  per-replica %.2f/%.2f  ref %.2f/%.2f   Δ(sync-perrep) %+.2f/%+.2f\n",
                        a, b, n, x1/n, x5/n, y1/n, y5/n, z1/n, z5/n, (x1-y1)/n, (x5-y5)/n } }' "$CSV" "$PREV" "$RTSV"

echo "--- per-image bitmaps (LOCAL ONLY — runs/**/*.bin is gitignored)"
echo "$(ls "$R"/bitmaps/*.bin 2>/dev/null | wc -l) epochs dumped; e90 = $R/bitmaps/r34_momdp64bf16_e90.bin"
echo "⚠ the per-replica arm has NO bitmap and score-checkpoint refuses BN nets, so the +0.10 is UNPAIRED (RESULTS.md)."
