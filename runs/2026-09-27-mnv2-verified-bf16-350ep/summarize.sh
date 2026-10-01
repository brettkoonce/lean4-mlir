#!/usr/bin/env bash
# Archive helper for the MobileNetV2 VERIFIED run, `rmsdp64wxdols0eps0001bf16`, 350 epochs (3060 box)
# — and the PAIR with its JAX reference, read IN PLACE from runs/2026-09-27-mnv2-jax-bf16-350ep/
# (run that directory's summarize.sh first; nothing is copied here). Both score the RAW weights.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-27-mnv2-verified-bf16-350ep
CSV=$R/mnv2_verified_curve.csv
JCSV=runs/2026-09-27-mnv2-jax-bf16-350ep/mnv2_jax_curve.csv
[ -s "$JCSV" ] || { echo "⛔ $JCSV missing — run the JAX run's summarize.sh first"; exit 1; }
awk '/^Epoch [0-9]+\/350: loss=/ { split($2,a,"/"); e=a[1]+0; l=$3; sub(/loss=/,"",l); loss[e]=l }
     /^  epoch [0-9]+: test_acc = / { e=$2; sub(/:/,"",e); c=0
       for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+%$/) { c++; v=$i; gsub(/%/,"",v); if (c==1) t1=v; else if (c==2) t5=v }
       if (c>=2) { top1[e+0]=t1; top5[e+0]=t5; if (e+0>mx) mx=e+0 } }
     END { print "epoch,train_loss,top1,top5"; for (e=1;e<=mx;e++) if (e in top1) printf "%d,%s,%s,%s\n", e, loss[e], top1[e], top5[e] }' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) scored epochs)"
echo "--- final"; grep -aE '^  epoch 350: ' "$R/full.log" | tail -1
echo "--- ⭐ the pair, 25-epoch windows (verified − JAX): top-1, top-5, TRAIN loss"
awk -F, 'FILENAME==ARGV[1] && FNR>1 {jl[$1]=$3; j1[$1]=$4; j5[$1]=$5; next}
  FNR>1 && ($1 in j1) { w=int(($1-1)/25); a[w]+=$3-j1[$1]; b[w]+=$4-j5[$1]; c[w]+=$2-jl[$1]; v[w]+=$3; j[w]+=j1[$1]; n[w]++; if (w>mw) mw=w }
  END { for (w=0; w<=mw; w++) printf "e%3d-%3d  verified %6.2f  JAX %6.2f   Δtop1 %+5.2f  Δtop5 %+5.2f  Δtrain-loss %+.4f\n", w*25+1, w*25+n[w], v[w]/n[w], j[w]/n[w], a[w]/n[w], b[w]/n[w], c[w]/n[w] }' "$JCSV" "$CSV"
echo "--- endpoint"; awk -F, 'FILENAME==ARGV[1] && FNR>1 {j1[$1]=$4; j5[$1]=$5; next} FNR>1 && $1==350 {printf "e350 verified %s / %s   JAX %s / %s   Δ %+.3f / %+.3f\n", $3, $4, j1[350], j5[350], $3-j1[350], $4-j5[350]}' "$JCSV" "$CSV"
echo "--- launches / restarts"
echo "supervisor starts: $(grep -c ' START job=' "$R/master.log")   trainer launches: $(grep -c 'launched PID=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'COMPLETE' "$R/master.log"
s=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ START' "$R/master.log" | head -1 | awk '{print $2" "$3}')
f=$(grep -oE '^\[sup\] [0-9-]+ [0-9:]+ ✅ COMPLETE' "$R/master.log" | tail -1 | awk '{print $2" "$3}')
awk -v a="$(date -u -d "$s" +%s)" -v b="$(date -u -d "$f" +%s)" 'BEGIN{printf "wall %s → %s = %.2f h\n", "'"$s"'", "'"$f"'", (b-a)/3600}'
echo "--- pace (epoch-clock gaps)"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-p; n++; s+=d; if(d>650) printf "  slow e%d %d s\n", $1, d} {p=$2} END{printf "%d gaps, mean %.0f s\n", n, s/n}'
echo "--- loaders: any producer whose arena-class memory passed 7 GiB"
awk -F'\t' 'NR>1 && $6>7 {last[$3]=$2" "$4" "$6} END{for(k in last){split(last[k],a," "); printf "  pid %s last seen e%s, age %.1f h, arena %s GiB\n", k, a[1], a[2]/3600, a[3]}}' "$R/loader_rss.tsv"
grep -aoE 'shim respawn: producer [0-9] of 4 replaced after epoch [0-9]+' "$R/full.log" | wc -l | awk '{print "  producer respawns: "$1}'
echo "--- EDAC"; tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'
echo "--- bitmaps (LOCAL ONLY): $(ls "$R"/bitmaps/*.bin 2>/dev/null | wc -l) scored epochs"
