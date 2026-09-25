#!/usr/bin/env bash
# Archive helper for the R50 2018-recipe bf16 SYNC-BN verified 90-epoch run. Rebuilds every number
# RESULTS.md quotes, from the logs. Read-only on the run; safe to re-run at any time.
#
# ⛔ NOTHING IS COPIED INTO THIS DIRECTORY. The JAX reference is read in place from
# blueprint/src/content.tex — the `\addplot` coordinate list ending `(90,76.95)` (top-1 only).
# The per-replica verified arm (2026-08-27, 77.074 / 93.476) has NO curve anywhere: its log lived in
# /tmp and is gone, the book carries only its endpoint, and its checkpoint survives as
# .lake/build/resnet50in_momdp64bf16_PERREPLICA_2026-08-27_e90_ckpt_xla.bin (no .bn — it predates
# 367bb28b). So that arm is compared at e90 only.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-24-r50-2018-syncbn-bf16-90ep
CSV=$R/r50_2018_syncbn_verified_curve.csv
TEX=blueprint/src/content.tex
# ⭐ THE PAIR: this run's partner is the A3 sync-BN run — the book's 2018 | A3 table, both cells now
# at the reference's own BN group. Read in place from its own summarize.sh output (run that first).
A3=runs/2026-09-23-r50-a3-syncbn-bf16-100ep/r50_a3_syncbn_verified_curve.csv
OLD1=77.074; OLD5=93.476   # per-replica e90 (memory + content.tex:\mathbf{77.07\%} row)

curve() { grep -m1 -F "(90,$1)" "$TEX" | grep -oE '\([0-9]+,[0-9.]+\)' | tr -d '()' | tr ',' ' '; }
REF=$(curve 76.95)
[ "$(wc -l <<<"$REF")" -eq 90 ] || { echo "⛔ could not read the 90-point JAX 2018 curve from $TEX (anchor (90,76.95))"; exit 1; }

# Same parser as R34's summarize.sh: LAST `Epoch N/90: loss= lr=` and LAST eval line per epoch.
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
tail -4 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n", $1, $4, $5}'
awk -F, 'NR>1 && $1>=87 {a+=$4; b+=$5; n++} END{printf "e87-90 mean: top1 %.3f top5 %.3f\n", a/n, b/n}' "$CSV"
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{printf "best: top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"
awk -F, -v o1=$OLD1 -v o5=$OLD5 'NR>1 && $1==90 {printf "vs per-replica e90 %s / %s: %+.3f / %+.3f\n", o1, o5, $4-o1, $5-o5}' "$CSV"
awk -F, 'NR>1 && $1==90 {printf "vs JAX 76.95 / 93.44: %+.3f / %+.3f\n", $4-76.95, $5-93.44}' "$CSV"
echo "--- ⭐ THE PAIR (book's 2018 | A3 table): verified sync-BN vs JAX, both recipes"
[ -s "$A3" ] || { echo "⛔ $A3 missing — run runs/2026-09-23-r50-a3-syncbn-bf16-100ep/summarize.sh first"; exit 1; }
t18=$(awk -F, '$1==90{print $4" "$5}' "$CSV"); ta3=$(awk -F, '$1==100{print $4" "$5}' "$A3")
awk -v a="$t18" -v b="$ta3" 'BEGIN{split(a,x," "); split(b,y," ")
  printf "2018 (90 ep):  ref 76.95 / 93.44   verified %.3f / %.3f   Δ %+.2f / %+.2f   BN group 256 / 256\n", x[1], x[2], x[1]-76.95, x[2]-93.44
  printf "A3  (100 ep):  ref 78.26 / 93.79   verified %.3f / %.3f   Δ %+.2f / %+.2f   BN group 512 / 512\n", y[1], y[2], y[1]-78.26, y[2]-93.79
  printf "A3 − 2018:     ref %+.2f           verified %+.2f\n", 78.26-76.95, y[1]-x[1] }'
echo "per-epoch top-1 (the book's A3 table epochs; 2018 capped at 90):"
for e in 5 25 50 75 90; do printf "  e%-3s 2018 %s   A3 %s\n" $e "$(awk -F, -v e=$e '$1==e{printf "%.2f",$4}' "$CSV")" "$(awk -F, -v e=$e '$1==e{printf "%.2f",$4}' "$A3")"; done
printf "  e100 A3 %s\n" "$(awk -F, '$1==100{printf "%.2f",$4}' "$A3")"

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

echo "--- pace (bench: ~30.7 h => ~1230 s/epoch; the conf ETA said 222 ms/step)"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if(!mn||d<mn)mn=d; if(d>mx)mx=d; if (d>1500) slow++} {pt=$2}
  END{if(n) printf "%d epoch gaps, mean %.0f s, min %d, max %d; > 1500 s: %d\n", n, s/n, mn, mx, slow+0}'
awk -F'\t' 'NR>2 && NR<=90 && $4!="" {r=$4/1048576; if(!mn||r<mn)mn=r; if(r>mx)mx=r} END{printf "trainer RssAnon e2-89: %.2f-%.2f GiB (e1 includes compile; e90 is sampled mid-teardown)\n", mn, mx}' "$R/epoch_clock.tsv"

echo "--- loader memory: per-epoch mean over the 8 train producers (arena_class is the signal)"
awk -F'\t' 'NR>1{a[$2]+=$6; r[$2]+=$5; n[$2]++} END{for(e in a) printf "%d %.2f %.2f\n", e, r[e]/n[e], a[e]/n[e]}' "$R/loader_rss.tsv" \
  | sort -n | awk '$1==1 || $1%10==0 || $1==89 {printf "e%-3d rss %.2f GiB  arena %.2f GiB\n", $1, $2, $3}'

echo "--- ⭐ sync-BN vs JAX reference, top-1, 15-epoch windows (per-replica arm: endpoint only)"
awk -F'[, ]' '
  FILENAME==ARGV[1] && FNR>1 { s[$1+0]=$4; next }
  { r[$1+0]=$2 }
  END { for (w=0; w<6; w++) { a=w*15+1; b=(w+1)*15; n=0; x=z=0
          for (e=a; e<=b; e++) if ((e in s) && (e in r)) { x+=s[e]; z+=r[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): sync %.2f  ref %.2f   Δ %+.2f\n", a, b, n, x/n, z/n, (x-z)/n }
        n=0; x=z=0; for (e=81; e<=90; e++) { x+=s[e]; z+=r[e]; n++ }
        printf "e81-90  (n=%2d): sync %.2f  ref %.2f   Δ %+.2f\n", n, x/n, z/n, (x-z)/n }' "$CSV" <(echo "$REF")

echo "--- EDAC"
tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'

echo "--- per-image bitmaps (LOCAL ONLY — runs/**/*.bin is gitignored)"
echo "$(ls "$R"/bitmaps/*.bin 2>/dev/null | wc -l) epochs dumped; e90 = $R/bitmaps/r50-2018_momdp64bf16_e90.bin"
