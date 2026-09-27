#!/usr/bin/env bash
# Archive helper for the MNv4-Conv-M JAX reference, `default` recipe, bf16, 100 epochs (3060 box).
# Rebuilds every number RESULTS.md quotes from the logs. Read-only on the run; safe to re-run.
# The JAX trainer scores the EMA shadow every 5 epochs (and at 100); checkpoints live OUTSIDE the
# repo, in /home/skoonce/mnv4_timm_100ep/.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-26-mnv4-jax-bf16-100ep
CSV=$R/mnv4_jax_curve.csv
# `[Epoch N] lr=… loss(train_avg)=… val_top1=C/50000 (0.xxxx) val_top5=0.xxxx …` — LAST per epoch
awk '/^\[Epoch [0-9]+\] .*val_top1=/ {
       e=$2; sub(/\]/,"",e); l=$4; sub(/loss\(train_avg\)=/,"",l)
       match($0,/val_top1=[0-9]+/); c=substr($0,RSTART+9,RLENGTH-9)
       match($0,/val_top5=[0-9.]+/); t5=substr($0,RSTART+9,RLENGTH-9)
       top1[e+0]=c/500; top5[e+0]=t5*100; loss[e+0]=l; if (e+0>mx) mx=e+0 }
     END { print "epoch,train_loss,top1,top5"
           for (e=1;e<=mx;e++) if (e in top1) printf "%d,%s,%.3f,%.3f\n", e, loss[e], top1[e], top5[e] }' "$R/full.log" > "$CSV"
echo "curve -> $CSV ($(($(wc -l < "$CSV")-1)) scored epochs)"
echo "--- final (EMA shadow)"; tail -3 "$CSV" | awk -F, '{printf "e%-3s top1 %s top5 %s\n",$1,$3,$4}'
echo "--- launches"; echo "trainer launches: $(grep -c 'launched PID=' "$R/master.log")   thermal: $(grep -c '🌡' "$R/master.log")"
grep -E 'COMPLETE' "$R/master.log"; grep -aoE 'Total time: [0-9.]+s' "$R/full.log" | tail -1 | awk '{sub(/s$/,"",$3); printf "trainer total %.0f s = %.2f h\n", $3, $3/3600}'
echo "--- pace"
tail -n +2 "$R/epoch_clock.tsv" | awk -F'\t' 'NR>1{d=$2-p; n++; s+=d; if(!mn||d<mn)mn=d; if(d>mx)mx=d} {p=$2}
  END{printf "%d epoch gaps, mean %.0f s, min %d, max %d\n", n, s/n, mn, mx}'
echo "--- EDAC"; tail -n +2 "$R/edac.tsv" | awk -F'\t' '{ce+=$4; ue+=$5} END{printf "rows %d, corrected %d, uncorrectable %d\n", NR, ce, ue}'
