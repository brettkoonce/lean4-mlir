#!/usr/bin/env bash
# Archive helper for the EfficientNet-B0 verified 350-epoch run: the curve CSV in the shape
# runs/2026-09-10-mnv2-verified-350ep established, plus the facts RESULTS.md needs. Read-only on the
# run; safe to re-run at any time.
# ⚠ The reference log contains a NUL byte: every grep of it needs -a or it reports "binary file
#   matches" and returns nothing — silently, which is how a comparison table turns into zeros.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-12-enet-verified-350ep
REF=/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet_full.log
CSV=$R/enet_verified_curve.csv
RTSV=$R/reference_curve.tsv

# ⚠ Pairs `Epoch N/350: loss= lr=` with `epoch N: test_acc = C/50000 = P%  top5 = C/50000 = P%`,
# taking the LAST of each: a re-run epoch (restart before its checkpoint landed) must not leave two
# rows. The two PERCENT fields are the only `NN.NN%` tokens on the eval line — index off those, not
# off the `=` signs, of which there are three.
awk '
  /^Epoch [0-9]+\/350: loss=/ { e=$2; sub(/\/350:/,"",e); l=$3; sub(/loss=/,"",l); r=$4; sub(/lr=/,"",r)
                                loss[e+0]=l; lr[e+0]=r; next }
  /^  epoch [0-9]+: .*_acc = / { e=$2; sub(/:/,"",e); c=0
                                 for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+%$/) { c++; v=$i; gsub(/%/,"",v)
                                   if (c==1) t1=v; else if (c==2) t5=v }
                                 if (c>=2) { top1[e+0]=t1; top5[e+0]=t5; if (e+0>mx) mx=e+0 } next }
  END { print "epoch,train_loss,lr,top1,top5"
        for (e=1; e<=mx; e++) if (e in top1) printf "%d,%s,%s,%s,%s\n", e, (e in loss?loss[e]:""), (e in lr?lr[e]:""), top1[e], top5[e] }
' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) epochs)"

grep -a -E '^\[Epoch [0-9]+\] ' "$REF" \
  | awk '{ e=$2; gsub(/[^0-9]/,"",e)
           if (match($0, /val_top1=[0-9]+\/50000 \(([0-9.]+)\) val_top5=([0-9.]+)/, m))
             printf "%d\t%.3f\t%.3f\n", e, 100*m[1], 100*m[2] }' | sort -n > "$RTSV"
echo "reference -> $RTSV ($(wc -l < "$RTSV") epochs)"

echo "--- final / best"
tail -3 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n", $1, $4, $5}'
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{printf "best verified : top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"
awk -F'\t' '$2+0>b {b=$2+0; be=$1; b5=$3} END{printf "best reference: top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$RTSV"

echo "--- launches / rests"
echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   supervisor starts: $(grep -c ' START job=' "$R/master.log")   planned rests: $(grep -c 'planned cooldown' "$R/master.log")"
grep -E 'ended \(|COMPLETE' "$R/master.log"

echo "--- wall clock"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
if [ -n "$f" ]; then
  awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" -v s="$s" -v f="$f" \
    'BEGIN{d=b-a; printf "launched %s UTC  finished %s UTC  = %d h %d m %d s (%.2f h)\n", s, f, d/3600, (d%3600)/60, d%60, d/3600}'
else echo "not COMPLETE yet (launched $s UTC)"; fi

echo "--- pace"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if (d>800) {slow++; if (slow<=8) ss=ss" e"$1"("d")"}} {pt=$2}
  END{printf "%d epochs timed, mean %.0f s; > 800 s: %d (first:%s …)\n", n, s/n, slow, ss}'
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; if ($1<=122) {a+=d; an++} else {b+=d; bn++}} {pt=$2}
  END{printf "before the feed fix (e<=122): mean %.0f s over %d epochs; after: mean %.0f s over %d\n", a/an, an, b/bn, bn}'

echo "--- verified vs reference, 50-epoch windows (mean of per-epoch evals)"
awk -F'[,\t]' 'NR==FNR { if (FNR>1) { v1[$1+0]=$4; v5[$1+0]=$5 }; next }
  { r1[$1+0]=$2; r5[$1+0]=$3 }
  END { for (w=0; w<7; w++) { a=w*50+1; b=(w+1)*50; n=0; s1=s2=s3=s4=0
          for (e=a; e<=b; e++) if ((e in v1) && (e in r1)) { s1+=v1[e]; s2+=v5[e]; s3+=r1[e]; s4+=r5[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): verified %.2f / %.2f   ref %.2f / %.2f   Δ %+.2f / %+.2f\n", a, b, n, s1/n, s2/n, s3/n, s4/n, (s1-s3)/n, (s2-s4)/n } }' "$CSV" "$RTSV"

echo "--- loader memory, last sample per process"
awk -F'\t' 'NR>1 && $8>10 {last[$3]=$0} END{for (k in last) print last[k]}' "$R/loader_rss.tsv" \
  | sort -t$'\t' -k3 -n | awk -F'\t' '{printf "pid %s  age %4.1f h  rss %5.2f GiB  arena %5.2f GiB  maps %s\n", $3, $4/3600, $5, $6, $7}'
