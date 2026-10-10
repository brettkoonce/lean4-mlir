#!/usr/bin/env bash
# Archive helper for the R50 RSB-A3 bf16 sync-BN 4×128 verified 100-epoch RERUN (fp32 head, zero-γ, BN 0.9,
# BCE threshold 0.2), on the 3060 box. Rebuilds every number RESULTS.md quotes, from the logs. Read-only on
# the run; safe to re-run at any time.
#
# ⛔ NOTHING IS COPIED INTO THIS DIRECTORY. The two other arms are read where they already live:
#   * the JAX rerun on the same settings  — runs/2026-10-07-r50-a3-jax-bf16-100ep/r50a3_jax_curve.csv
#   * the landed verified run (γ = 1, no threshold) — runs/2026-09-23-r50-a3-syncbn-bf16-100ep/r50_a3_syncbn_verified_curve.csv
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-10-09-r50-a3-verified-bf16-100ep
CSV=$R/r50a3_verified_curve.csv
JAX=runs/2026-10-07-r50-a3-jax-bf16-100ep/r50a3_jax_curve.csv
OLD=runs/2026-09-23-r50-a3-syncbn-bf16-100ep/r50_a3_syncbn_verified_curve.csv

# Same parser as the 09-23 run: LAST `Epoch N/100: loss= lr=` and LAST eval line per epoch.
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

echo "--- settings, from the trainer's own log"
grep -m1 -aF 'resuming at epoch' "$R/master.log"
grep -m1 -aF 'new-batch weight' "$R/full.log" | sed 's/^ *//'
grep -m1 -aF 'zero-γ' "$R/full.log" | sed 's/^ *//'

echo "--- final"
tail -4 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n", $1, $4, $5}'
awk -F, 'NR>1 && $1>=97 {a+=$4; b+=$5; n++} END{printf "e97-100 mean: top1 %.3f top5 %.3f\n", a/n, b/n}' "$CSV"
awk -F, 'NR>1 && $1>=91 {a+=$4; n++} END{printf "e91-100 mean: top1 %.3f\n", a/n}' "$CSV"
awk -F, 'NR>1 && $4+0>b {b=$4+0; be=$1; b5=$5} END{printf "best: top1 %.3f @ e%s (top5 %s)\n", b, be, b5}' "$CSV"

echo "--- launches / rests"
echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   supervisor starts: $(grep -c ' START job=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'ended \(|COMPLETE' "$R/master.log"
echo "shim respawns: $(grep -c 'shim respawn' "$R/attempt.log")"

echo "--- wall clock"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
if [ -n "$f" ]; then
  awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" -v s="$s" -v f="$f" \
    'BEGIN{d=b-a; printf "launched %s UTC  finished %s UTC  = %d h %d m %d s (%.2f h, %.0f s/epoch)\n", s, f, d/3600, (d%3600)/60, d%60, d/3600, d/100}'
else echo "not COMPLETE yet (launched $s UTC)"; fi

echo "--- pace"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-pt; n++; s+=d; if(!mn||d<mn)mn=d; if(d>mx)mx=d; w=int(($1-1)/20); ws[w]+=d; wn[w]++} {pt=$2}
  END{if(n) printf "%d epoch gaps, mean %.0f s, min %d, max %d\n", n, s/n, mn, mx
      for (w=0; w<5; w++) if (wn[w]) printf "  e%d-%d mean %.0f s\n", w*20+1, w*20+20, ws[w]/wn[w]}'

echo "--- loader memory: per-epoch mean over the 8 train producers (arena_class is the signal)"
awk -F'\t' 'NR>1{a[$2]+=$6; r[$2]+=$5; n[$2]++; if($6>m[$2])m[$2]=$6} END{for(e in a) printf "%d %.2f %.2f %.2f\n", e, r[e]/n[e], a[e]/n[e], m[e]}' "$R/loader_rss.tsv" \
  | sort -n | awk '$1==1 || $1%10==0 || $1==99 {printf "e%-3d rss %.2f GiB  arena %.2f GiB (max %.2f)\n", $1, $2, $3, $4}'

echo "--- ⭐ this run vs the JAX rerun (same settings) vs the landed verified run, top-1, 20-epoch windows"
awk -F, '
  FILENAME==ARGV[1] && FNR>1 { s[$1+0]=$4; next }
  FILENAME==ARGV[2] && FNR>1 { j[$1+0]=$3; next }
  FILENAME==ARGV[3] && FNR>1 { o[$1+0]=$4 }
  END { for (w=0; w<5; w++) { a=w*20+1; b=(w+1)*20; n=0; x=y=z=0
          for (e=a; e<=b; e++) if ((e in s) && (e in j) && (e in o)) { x+=s[e]; y+=j[e]; z+=o[e]; n++ }
          if (n) printf "e%-3d-%3d (n=%2d): new %.2f  jax %.2f  old %.2f   Δnew-jax %+.2f  Δnew-old %+.2f\n",
                        a, b, n, x/n, y/n, z/n, (x-y)/n, (x-z)/n }
        n=0; x=y=z=0; for (e=91; e<=100; e++) if ((e in s) && (e in j) && (e in o)) { x+=s[e]; y+=j[e]; z+=o[e]; n++ }
        if (n) printf "e91-100 (n=%2d): new %.2f  jax %.2f  old %.2f   Δnew-jax %+.2f  Δnew-old %+.2f\n", n, x/n, y/n, z/n, (x-y)/n, (x-z)/n
        printf "e1: new %s  jax %s  old %s\n", s[1], j[1], o[1] }' "$CSV" "$JAX" "$OLD"

echo "--- EDAC"
tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'

echo "--- per-image bitmaps (LOCAL ONLY — runs/**/*.bin is gitignored)"
echo "$(ls "$R"/bitmaps/*.bin 2>/dev/null | wc -l) epochs dumped; e100 = $R/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin"
