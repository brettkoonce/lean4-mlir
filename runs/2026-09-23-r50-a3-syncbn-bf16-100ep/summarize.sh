#!/usr/bin/env bash
# Archive helper for the R50 RSB-A3 bf16 SYNC-BN 4×128 verified 100-epoch run. Rebuilds every
# number RESULTS.md quotes, from the logs. Read-only on the run; safe to re-run at any time.
#
# ⛔ NOTHING IS COPIED INTO THIS DIRECTORY. The two other arms are read where they already live,
# in blueprint/src/content.tex's A3 figure (top-1 only — the book plots no top-5 curve):
#   * the JAX reference     — the `\addplot` coordinate list ending `(100,78.26)`
#   * the old verified run  — the one ending `(100,77.98)`: 8 × 4 × 64, per-replica BN (Ghost-BN 64).
#     Its full log, runs/r50-a3-bf16-100ep-verified.log, is local-only (untracked); it is read for
#     the top-5 tail when present and skipped when not.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-23-r50-a3-syncbn-bf16-100ep
CSV=$R/r50_a3_syncbn_verified_curve.csv
TEX=blueprint/src/content.tex
OLDLOG=runs/r50-a3-bf16-100ep-verified.log

# `(e,v) (e,v) …` → "e v" rows, from the FIRST \addplot coordinate line ending in the given point.
curve() { grep -m1 -F "(100,$1)" "$TEX" | grep -oE '\([0-9]+,[0-9.]+\)' | tr -d '()' | tr ',' ' '; }
REF=$(curve 78.26); OLD=$(curve 77.98)
[ "$(wc -l <<<"$REF")" -eq 100 ] && [ "$(wc -l <<<"$OLD")" -eq 100 ] || {
  echo "⛔ could not read both 100-point A3 curves from $TEX (anchors (100,78.26) / (100,77.98))"; exit 1; }

# Same parser as R34's summarize.sh: LAST `Epoch N/100: loss= lr=` and LAST eval line per epoch.
awk '
  /^Epoch [0-9]+\/100: loss=/ { e=$2; sub(/\/100:/,"",e); l=$3; sub(/loss=/,"",l); r=$4; sub(/lr=/,"",r)
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
awk -F, 'NR>1 && $1>=97 {a+=$4; b+=$5; n++} END{printf "e97-100 mean: top1 %.3f top5 %.3f\n", a/n, b/n}' "$CSV"
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{printf "best: top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"
if [ -s "$OLDLOG" ]; then
  grep -aE '^  epoch (9[7-9]|100): ' "$OLDLOG" | grep -oE '= [0-9.]+%' | tr -d '=% ' | awk 'NR%2==1{a+=$1} NR%2==0{b+=$1; n++}
    END{printf "old 8x64 e97-100 mean (from %s): top1 %.3f top5 %.3f\n", "'"$OLDLOG"'", a/n, b/n}'
  grep -aE '^  epoch 100: ' "$OLDLOG" | sed 's/^/old 8x64 final: /'
else echo "(old 8x64 log absent — top-5 tail not available; top-1 from $TEX only)"; fi

echo "--- launches / rests"
echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   supervisor starts: $(grep -c ' START job=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'ended \(|COMPLETE' "$R/master.log"

echo "--- wall clock"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
if [ -n "$f" ]; then
  awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" -v s="$s" -v f="$f" \
    'BEGIN{d=b-a; printf "launched %s UTC  finished %s UTC  = %d h %d m %d s (%.2f h, %.0f s/epoch)\n", s, f, d/3600, (d%3600)/60, d%60, d/3600, d/100}'
else echo "not COMPLETE yet (launched $s UTC)"; fi

echo "--- pace (bench: 294 ms/step fed at w8 => ~735 s of steps + eval; > 1000 s is the fault)"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if(!mn||d<mn)mn=d; if(d>mx)mx=d; if (d>1000) slow++} {pt=$2}
  END{if(n) printf "%d epoch gaps, mean %.0f s, min %d, max %d; > 1000 s: %d\n", n, s/n, mn, mx, slow+0}'
awk -F'\t' 'NR>2 && NR<=100 && $4!="" {r=$4/1048576; if(!mn||r<mn)mn=r; if(r>mx)mx=r} END{printf "trainer RssAnon e2-99: %.2f-%.2f GiB (e1 includes compile; e100 is sampled mid-teardown)\n", mn, mx}' "$R/epoch_clock.tsv"

echo "--- loader memory: per-epoch mean over the 8 train producers (arena_class is the signal)"
awk -F'\t' 'NR>1{a[$2]+=$6; r[$2]+=$5; n[$2]++} END{for(e in a) printf "%d %.2f %.2f\n", e, r[e]/n[e], a[e]/n[e]}' "$R/loader_rss.tsv" \
  | sort -n | awk '$1==1 || $1%10==0 || $1==99 {printf "e%-3d rss %.2f GiB  arena %.2f GiB\n", $1, $2, $3}'

echo "--- ⭐ sync-BN 4x128 vs old per-replica 8x64 vs JAX reference, top-1, 20-epoch windows"
awk -F'[, ]' '
  FILENAME==ARGV[1] && FNR>1 { s[$1+0]=$4; next }
  FILENAME==ARGV[2] { o[$1+0]=$2; next }
  { r[$1+0]=$2 }
  END { for (w=0; w<5; w++) { a=w*20+1; b=(w+1)*20; n=0; x=y=z=0
          for (e=a; e<=b; e++) if ((e in s) && (e in o) && (e in r)) { x+=s[e]; y+=o[e]; z+=r[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): sync %.2f  old %.2f  ref %.2f   Δsync-old %+.2f  Δsync-ref %+.2f\n",
                        a, b, n, x/n, y/n, z/n, (x-y)/n, (x-z)/n }
        n=0; x=y=z=0; for (e=91; e<=100; e++) { x+=s[e]; y+=o[e]; z+=r[e]; n++ }
        printf "e91-100 (n=%2d): sync %.2f  old %.2f  ref %.2f   Δsync-old %+.2f  Δsync-ref %+.2f\n", n, x/n, y/n, z/n, (x-y)/n, (x-z)/n }' \
  "$CSV" <(echo "$OLD") <(echo "$REF")

echo "--- EDAC"
tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'

echo "--- per-image bitmaps (LOCAL ONLY — runs/**/*.bin is gitignored)"
echo "$(ls "$R"/bitmaps/*.bin 2>/dev/null | wc -l) epochs dumped; e100 = $R/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin"
