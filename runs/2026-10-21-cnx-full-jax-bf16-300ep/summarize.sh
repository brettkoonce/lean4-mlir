#!/usr/bin/env bash
# Archive helper for the ConvNeXt-T JAX reference, `default` recipe = facebookresearch/ConvNeXt's own
# (batch 4096 as 16 accumulated micro-batches of 256, AdamW 4e-3, per-step warmup 20 ep + cosine to
# 1e-6, no clip, ModelEma 0.9999 constant, scored), bf16, 300 epochs (3060 box).
# ▶ 2026-10-24 copy of runs/2026-10-16-enet-full-jax-bf16-350ep/summarize.sh, retargeted to this run.
# Rebuilds every number RESULTS.md quotes from the logs. Read-only on the run; safe to re-run.
# The trainer scores the EMA shadow on all 50,000 EVERY epoch (f32 eval forward). Checkpoints live
# OUTSIDE the repo, in /home/skoonce/cnx_t_jax_2611/.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-10-21-cnx-full-jax-bf16-300ep
CSV=${OUT:-$R}/cnx_jax_curve.csv
# `[Epoch N] lr=… loss(train_avg)=… val_top1=C/50000 (0.xxxx) val_top5=0.xxxx …` — LAST line per epoch wins
awk '/^\[Epoch [0-9]+\] lr=/ {
       e=$2; sub(/\]/,"",e); e+=0; l=$4; sub(/loss\(train_avg\)=/,"",l); lr=$3; sub(/lr=/,"",lr)
       loss[e]=l; lrs[e]=lr; if (e>mx) mx=e
       if (match($0,/val_top1=[0-9]+/)) { c=substr($0,RSTART+9,RLENGTH-9); match($0,/val_top5=[0-9.]+/)
         t5=substr($0,RSTART+9,RLENGTH-9); top1[e]=sprintf("%.3f",c/500); top5[e]=sprintf("%.2f",t5*100) } }
     END { print "epoch,lr,train_loss,top1,top5"
           for (e=1;e<=mx;e++) if (e in loss) printf "%d,%s,%s,%s,%s\n", e, lrs[e], loss[e], top1[e], top5[e] }' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) epochs, $(awk -F, 'NR>1 && $4!=""' "$CSV" | wc -l) scored)"
echo "--- final"; tail -1 "$CSV" | awk -F, '{printf "e%s top1 %s top5 %s (EMA; train loss %s)\n",$1,$4,$5,$3}'
grep -aoE '\[Epoch 300\].*val_top1=[0-9]+/50000' "$R/full.log" | tail -1 | grep -oE '[0-9]+/50000' | sed 's/^/e300 top-1 count /'
echo "--- best"; awk -F, 'NR>1 && $4!="" && $4>b {b=$4; e=$1} END{printf "best top1 %.3f at e%d\n", b, e}' "$CSV"
echo "--- tail, e276-300"
awk -F, 'NR>1 && $1>=276 && $4!="" {a+=$4; b+=$5; n++; if(!mn||$4<mn)mn=$4; if($4>mx)mx=$4} END{printf "mean %.3f / %.3f over %d   top-1 range %.3f..%.3f\n", a/n, b/n, n, mn, mx}' "$CSV"
echo "--- curve"
awk -F, 'NR>1 {e=$1+0; if (e==10||e==25||e==50||e==60||e==75||e==100||e==150||e==200||e==250||e==300) printf "e%-3d top1 %s top5 %s loss %s\n", e, $4, $5, $3}' "$CSV"
echo "--- launches"; echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'COMPLETE' "$R/master.log"; grep -aoE 'Total time: [0-9.]+s' "$R/full.log" | tail -1 | awk '{sub(/s$/,"",$3); printf "trainer total %.0f s = %.2f h\n", $3, $3/3600}'
echo "--- pace"
grep -aoE '\[[0-9.]+s train, [0-9.]+s val' "$R/full.log" | tr -d '[s,' | awk '{n++; s+=$1; v+=$3; if(!mn||$1<mn)mn=$1; if($1>mx)mx=$1} END{printf "train %.1f s/epoch mean over %d (min %.1f, max %.1f), val %.1f s\n", s/n, n, mn, mx, v/n}'
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-p; n++; s+=d; if(!mn||d<mn)mn=d; if(d>mx)mx=d} {p=$2}
  END{printf "%d epoch-clock gaps, mean %.1f s, min %d, max %d (eval every epoch included)\n", n, s/n, mn, mx}'
echo "--- EDAC"; tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'
echo "--- ⭐ against the 09-04 81.53 JAX run (global-256 recipe; its per-epoch EMA curve is runs/2026-09-18-cnx-verified-300ep/reference_curve.tsv, read in place)"
awk -F'[,\t]' 'FILENAME==ARGV[1] {if (FNR>1) {o[$1+0]=$5*100; o5[$1+0]=$6*100}; next}
         FNR>1 && $4!="" && (($1+0) in o) {e=$1+0; if (e==10||e==25||e==50||e==75||e==100||e==150||e==200||e==250||e==300)
           printf "e%-3d top1 new %.3f  old %.3f  Δ %+.2f   top5 new %.2f old %.2f\n", e, $4, o[e], $4-o[e], $5, o5[e]
           w=int((e-1)/50); d[w]+=$4-o[e]; k[w]++
           if (e>=276) {a+=$4; b+=o[e]; m++} }
         END{for (w=0;w<6;w++) if (k[w]) printf "e%d-%d mean Δtop1 %+.3f\n", w*50+1, w*50+50, d[w]/k[w]
             if(m) printf "e276-300 mean: new %.3f  old %.3f  Δ %+.3f (n=%d)\n", a/m, b/m, (a-b)/m, m}' \
  runs/2026-09-18-cnx-verified-300ep/reference_curve.tsv "$CSV"
