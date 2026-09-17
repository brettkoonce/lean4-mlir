#!/usr/bin/env bash
# Archive helper for the ConvNeXt-T bf16 verified 300-epoch run. Rebuilds every number RESULTS.md
# needs from the logs. Read-only on the run; safe to re-run at any time.
#
# ⭐ UNLIKE R34's, THE REFERENCE HERE IS A REAL LOG AND IT IS ON THIS BOX:
#   /home/skoonce/convnext_t300_3060/convnext_tiny_imagenet_full.log — 300 per-epoch evals, all
#   over the 50,000 denominator (post-C4), ending 0.8153 / 0.9551. `extract_reference.sh` lifts it
#   into reference_curve.tsv and ASSERTS row count, field count, denominator and endpoint, because
#   R34's lesson was that the plausible reference log on disk was a different run.
# ⚠ reference_curve.tsv stores FRACTIONS (0.8153); the verified log prints PERCENTAGES (81.53).
#   The window table below scales the reference by 100. Getting that wrong reads as a 80-point gap.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-17-cnx-verified-300ep
CSV=$R/cnx_verified_curve.csv
RTSV=$R/reference_curve.tsv

[ -s "$RTSV" ] || { echo "⛔ $RTSV missing — run $R/extract_reference.sh first."; exit 1; }

# ⚠ Pairs `Epoch N/300: loss= lr=` with `epoch N: test_acc = C/50000 = P%  top5 = C/50000 = P%`,
# taking the LAST of each: a re-run epoch (restart before its checkpoint landed) must not leave two
# rows. The two PERCENT fields are the only `NN.NN%` tokens on the eval line — index off those.
# ⚠⚠ `train_loss` IS CARRIED BUT MUST NOT BE QUOTED AS A CURVE. ConvNeXt's initial loss is 10.42
# where every other net starts near ln(1000)=6.91; §5 lists this net's `%loss` as a REPORT-ONLY
# carve-out outside every faithfulness theorem, and R34 shipped a wrong `%loss` once.
awk '
  /^Epoch [0-9]+\/300: loss=/ { e=$2; sub(/\/300:/,"",e); l=$3; sub(/loss=/,"",l); r=$4; sub(/lr=/,"",r)
                                loss[e+0]=l; lr[e+0]=r; next }
  /^  epoch [0-9]+: .*_acc = / { e=$2; sub(/:/,"",e); c=0
                                 for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+%$/) { c++; v=$i; gsub(/%/,"",v)
                                   if (c==1) t1=v; else if (c==2) t5=v }
                                 if (c>=2) { top1[e+0]=t1; top5[e+0]=t5; if (e+0>mx) mx=e+0 } next }
  END { print "epoch,train_loss,lr,top1,top5"
        for (e=1; e<=mx; e++) if (e in top1) printf "%d,%s,%s,%s,%s\n", e, (e in loss?loss[e]:""), (e in lr?lr[e]:""), top1[e], top5[e] }
' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) epochs)"
echo "reference -> $RTSV ($(($(wc -l < "$RTSV")-1)) epochs, from the 3060-box JAX log)"

echo "--- final / best"
tail -3 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n", $1, $4, $5}'
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{if(b)printf "best verified : top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"
awk -F'\t' 'NR>1 && $5+0>b {b=$5+0; be=$1; b5=$6} END{printf "best reference: top1 %.3f @ e%s (top5 %.3f)\n", b*100, be, b5*100}' "$RTSV"

echo "--- launches / rests"
echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log" 2>/dev/null)   supervisor starts: $(grep -c ' START job=' "$R/master.log" 2>/dev/null)   planned rests: $(grep -c 'planned cooldown' "$R/master.log" 2>/dev/null)   thermal: $(grep -c '🌡' "$R/master.log" 2>/dev/null)"
grep -E 'ended \(|COMPLETE' "$R/master.log" 2>/dev/null

echo "--- resumes (⭐ THE LAYERNORM RESUME PATH — never exercised at ImageNet scale before this run)"
grep -c 'resuming from checkpoint at epoch' "$R/full.log" 2>/dev/null
grep -h 'resuming from checkpoint at epoch' "$R/full.log" 2>/dev/null | head -12
[ -f .lake/build/convnextin_adamdpwxclipdropbf16_ckpt_xla.bin.bn ] \
  && echo "⛔ a .bn companion exists for a LayerNorm net — investigate" \
  || echo "✅ no .bn companion (correct: no BatchNorm in this net)"

echo "--- wall clock"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" 2>/dev/null | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" 2>/dev/null | tail -1 | awk '{print $2" "$3}')
if [ -n "${f:-}" ]; then
  awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" -v s="$s" -v f="$f" \
    'BEGIN{d=b-a; printf "launched %s UTC  finished %s UTC  = %d h %d m %d s (%.2f h)\n", s, f, d/3600, (d%3600)/60, d%60, d/3600}'
else echo "not COMPLETE yet (launched ${s:-?} UTC)"; fi

echo "--- pace (probe 2026-09-17: 220 ms/step fed at w4 => ~1101 s/epoch of steps; fault_watch.sh"
echo "    self-calibrates its threshold off e4-e12 rather than trusting this forecast)"
tail -n +2 "$R/epoch_clock.tsv" 2>/dev/null | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if (d>1400) {slow++; if (slow<=8) ss=ss" e"$1"("d"s)"}} {pt=$2}
  END{if(n) printf "%d epochs timed, mean %.0f s; > 1400 s: %d%s\n", n, s/n, slow+0, (slow?" (first:"ss" …)":"")}'

echo "--- ⭐ THE VERDICT: per-epoch pace against LOADER AGE (lifetime, not epoch count)"
# EfficientNet degraded after ~5.6-13 h of producer uptime and sometimes never. With the respawn ON
# at E=10 no loader should exceed ~40 epochs (~12 h), so these buckets should stay FLAT and shallow.
awk -F'\t' 'NR==FNR { if (FNR>1 && $4+0>age[$2+0]) age[$2+0]=$4+0; next }
  FNR>2 { d=$2-pt; a=age[$1+0]; if (a>0) { b=int(a/7200); n[b]++; s[b]+=d } }
  { pt=$2 }
  END { printf "%-14s %8s %10s\n", "loader age", "epochs", "mean s/ep"
        for (b=0; b<24; b++) if (b in n) printf "%3d-%3d h      %8d %10.0f\n", b*2, b*2+2, n[b], s[b]/n[b] }' \
  "$R/loader_rss.tsv" "$R/epoch_clock.tsv" 2>/dev/null

echo "--- loader memory, last sample per process (arena_class is the signal, not total rss)"
awk -F'\t' 'NR>1 && $8>10 {last[$3]=$0} END{for (k in last) print last[k]}' "$R/loader_rss.tsv" 2>/dev/null \
  | sort -t$'\t' -k3 -n | awk -F'\t' '{printf "pid %s  age %5.1f h  rss %5.2f GiB  arena %5.2f GiB  maps %s\n", $3, $4/3600, $5, $6, $7}'

echo "--- ⭐⭐ shim respawns (this run is the mitigation's FIRST PRODUCTION TEST — 63b21d84)"
echo "count: $(grep -cE 'SHIM RESPAWN|respawn' "$R/log_ts.log" 2>/dev/null || echo 0)"
grep -E 'SHIM RESPAWN' "$R/log_ts.log" 2>/dev/null | head -12

echo "--- verified vs reference, 50-epoch windows (mean of per-epoch evals; reference scaled x100)"
awk -F'[,\t]' 'NR==FNR { if (FNR>1) { v1[$1+0]=$4; v5[$1+0]=$5 }; next }
  FNR>1 { r1[$1+0]=$5*100; r5[$1+0]=$6*100 }
  END { for (w=0; w<6; w++) { a=w*50+1; b=(w+1)*50; n=0; s1=s2=s3=s4=0
          for (e=a; e<=b; e++) if ((e in v1) && (e in r1)) { s1+=v1[e]; s2+=v5[e]; s3+=r1[e]; s4+=r5[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): verified %.2f / %.2f   ref %.2f / %.2f   Δ %+.2f / %+.2f\n", a, b, n, s1/n, s2/n, s3/n, s4/n, (s1-s3)/n, (s2-s4)/n } }' "$CSV" "$RTSV"

echo "--- ⭐ epochs ABOVE the reference (EfficientNet's tell was 0 of 300)"
awk -F'[,\t]' 'NR==FNR { if (FNR>1) v1[$1+0]=$4; next }
  FNR>1 { r=$5*100; e=$1+0; if (e in v1) { n++; if (v1[e] > r) up++ } }
  END { if (n) printf "%d of %d paired epochs have verified > reference\n", up+0, n }' "$CSV" "$RTSV"
