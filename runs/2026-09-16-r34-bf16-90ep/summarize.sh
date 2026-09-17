#!/usr/bin/env bash
# Archive helper for the R34 bf16 verified 90-epoch run. Rebuilds every number RESULTS.md needs
# from the logs. Read-only on the run; safe to re-run at any time.
#
# ⛔⛔ THE REFERENCE IS NOT A LOG HERE, AND THAT IS NOT AN OVERSIGHT. Every other net's summarize.sh
# greps a JAX reference log under /home/skoonce/. R34's is GONE from this box — /home/skoonce/
# r34_2018_90ep/ (the path imagenet_rerun_sweep.md and inflight_r50_queue.md both name) does not
# exist, and /home/skoonce/r34_90ep_eu_logs/r34_90ep.log is a DIFFERENT run: it evaluates against a
# 49,152 denominator (pre-C4) and ends at 73.95, not 74.16. Pairing against THAT file would quietly
# shift the reference by -0.21 and nothing would say so.
# ▶ So reference_curve.tsv was extracted ONCE from blueprint/src/content.tex (the published
#   pgfplots curve, "2018 recipe, 90 epochs --- reference"), asserted to be 90 points ending at
#   exactly 74.16 / 91.92, and committed beside this script. It is the canonical curve.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-16-r34-bf16-90ep
CSV=$R/r34_bf16_verified_curve.csv
RTSV=$R/reference_curve.tsv

[ -s "$RTSV" ] || { echo "⛔ $RTSV missing — the reference curve is not re-derivable from a log."; exit 1; }

# ⚠ Pairs `Epoch N/90: loss= lr=` with `epoch N: test_acc = C/50000 = P%  top5 = C/50000 = P%`,
# taking the LAST of each: a re-run epoch (restart before its checkpoint landed) must not leave two
# rows. The two PERCENT fields are the only `NN.NN%` tokens on the eval line — index off those, not
# off the `=` signs, of which there are three.
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
echo "reference -> $RTSV ($(wc -l < "$RTSV") epochs, from content.tex)"

echo "--- final / best"
tail -3 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n", $1, $4, $5}'
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{printf "best verified : top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"
awk -F'\t' '$2+0>b {b=$2+0; be=$1; b5=$3} END{printf "best reference: top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$RTSV"

echo "--- launches / rests"
echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   supervisor starts: $(grep -c ' START job=' "$R/master.log")   planned rests: $(grep -c 'planned cooldown' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'ended \(|COMPLETE' "$R/master.log"

echo "--- wall clock"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
if [ -n "$f" ]; then
  awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" -v s="$s" -v f="$f" \
    'BEGIN{d=b-a; printf "launched %s UTC  finished %s UTC  = %d h %d m %d s (%.2f h)\n", s, f, d/3600, (d%3600)/60, d%60, d/3600}'
else echo "not COMPLETE yet (launched $s UTC)"; fi

echo "--- pace (probe says 163 ms/step at w4 => ~816 s/epoch of steps; > 950 s is the fault, matching fault_watch.sh)"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if (d>950) {slow++; if (slow<=8) ss=ss" e"$1"("d"s)"}} {pt=$2}
  END{if(n) printf "%d epochs timed, mean %.0f s; > 1000 s: %d%s\n", n, s/n, slow+0, (slow?" (first:"ss" …)":"")}'

echo "--- ⭐ THE VERDICT: per-epoch pace against LOADER AGE (lifetime, not epoch count)"
# EfficientNet degraded after ~13 h of producer uptime, but anywhere from 5.6 h to never across
# five generations. Bucket the epoch times by the OLDEST loader's age at that epoch.
awk -F'\t' 'NR==FNR { if (FNR>1 && $4+0>age[$2+0]) age[$2+0]=$4+0; next }
  FNR>2 { d=$2-pt; a=age[$1+0]; if (a>0) { b=int(a/7200); n[b]++; s[b]+=d } }
  { pt=$2 }
  END { printf "%-14s %8s %10s\n", "loader age", "epochs", "mean s/ep"
        for (b=0; b<24; b++) if (b in n) printf "%3d-%3d h      %8d %10.0f\n", b*2, b*2+2, n[b], s[b]/n[b] }' \
  "$R/loader_rss.tsv" "$R/epoch_clock.tsv" 2>/dev/null

echo "--- loader memory, last sample per process (arena_class is the signal, not total rss)"
awk -F'\t' 'NR>1 && $8>10 {last[$3]=$0} END{for (k in last) print last[k]}' "$R/loader_rss.tsv" 2>/dev/null \
  | sort -t$'\t' -k3 -n | awk -F'\t' '{printf "pid %s  age %5.1f h  rss %5.2f GiB  arena %5.2f GiB  maps %s\n", $3, $4/3600, $5, $6, $7}'

echo "--- shim respawns (empty = Arm A, respawn was OFF)"
grep -cE 'SHIM RESPAWN|respawn' "$R/log_ts.log" 2>/dev/null || true
grep -E 'SHIM RESPAWN' "$R/log_ts.log" 2>/dev/null | head -12

echo "--- verified vs reference, 15-epoch windows (mean of per-epoch evals)"
awk -F'[,\t]' 'NR==FNR { if (FNR>1) { v1[$1+0]=$4; v5[$1+0]=$5 }; next }
  { r1[$1+0]=$2; r5[$1+0]=$3 }
  END { for (w=0; w<6; w++) { a=w*15+1; b=(w+1)*15; n=0; s1=s2=s3=s4=0
          for (e=a; e<=b; e++) if ((e in v1) && (e in r1)) { s1+=v1[e]; s2+=v5[e]; s3+=r1[e]; s4+=r5[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): verified %.2f / %.2f   ref %.2f / %.2f   Δ %+.2f / %+.2f\n", a, b, n, s1/n, s2/n, s3/n, s4/n, (s1-s3)/n, (s2-s4)/n } }' "$CSV" "$RTSV"
