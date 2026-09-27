#!/usr/bin/env bash
# Archive helper for the MNv4-Conv-M VERIFIED run, `emaaccdp8x128wxdowd005bf16`, 100 epochs (3060
# box) — and the PAIR with its JAX reference, read IN PLACE from runs/2026-09-26-mnv4-jax-bf16-100ep/
# (run that directory's summarize.sh first; nothing is copied here). Both paths score the EMA shadow.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-26-mnv4-verified-bf16-100ep
CSV=$R/mnv4_verified_curve.csv
JCSV=runs/2026-09-26-mnv4-jax-bf16-100ep/mnv4_jax_curve.csv
[ -s "$JCSV" ] || { echo "⛔ $JCSV missing — run the JAX run's summarize.sh first"; exit 1; }
awk '/^  epoch [0-9]+: test_acc = / { e=$2; sub(/:/,"",e); c=0
       for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+%$/) { c++; v=$i; gsub(/%/,"",v); if (c==1) t1=v; else if (c==2) t5=v }
       if (c>=2) { top1[e+0]=t1; top5[e+0]=t5; if (e+0>mx) mx=e+0 } }
     END { print "epoch,top1,top5"; for (e=1;e<=mx;e++) if (e in top1) printf "%d,%s,%s\n", e, top1[e], top5[e] }' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) scored epochs)"
echo "--- ⭐ the pair, every scored epoch (verified − JAX)"
awk -F, 'FILENAME==ARGV[1] && FNR>1 {j1[$1]=$3; j5[$1]=$4; next} FNR>1 && ($1 in j1) {
  printf "e%-3s verified %6.2f / %6.2f   JAX %6.2f / %6.2f   Δ %+5.2f / %+5.2f\n", $1, $2, $3, j1[$1], j5[$1], $2-j1[$1], $3-j5[$1];
  if ($1>=90) {a+=$2-j1[$1]; b+=$3-j5[$1]; n++} } END{printf "mean Δ over e90/95/100: %+.2f / %+.2f\n", a/n, b/n}' "$JCSV" "$CSV"
echo "--- launches / restarts"
echo "supervisor starts: $(grep -c ' START job=' "$R/master.log")   trainer launches: $(grep -c 'launched PID=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
cat "$R/RESTARTS"; grep -E 'COMPLETE' "$R/master.log"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" 'BEGIN{d=b-a; printf "wall %s → %s = %.2f h (restart included)\n", "'"$s"'", "'"$f"'", d/3600}'
echo "--- pace (epoch-clock gaps; e85 includes the restart)"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-p; e=$1+0; if(e<=76){a+=d;na++} else if(e<=84){b+=d;nb++; printf "  e%d %d s\n", e, d} else if (e>=86){c+=d;nc++}} {p=$2}
  END{printf "e2-76 mean %.0f s   e77-84 mean %.0f s (the fault)   e86-100 mean %.0f s (fresh loaders)\n", a/na, b/nb, c/nc}'
echo "--- loader memory at the fault (from WATCH_EVENT.loader-fault-e83)"; sed -n '/loaders at detection/,$p' "$R/WATCH_EVENT.loader-fault-e83"
echo "--- EDAC"; tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'
echo "--- bitmaps (LOCAL ONLY): $(ls "$R"/bitmaps/*.bin 2>/dev/null | wc -l) scored epochs"
