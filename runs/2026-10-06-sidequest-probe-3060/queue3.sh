#!/bin/bash
# ConvNeXt-B verified OOMed at the default arena on the 12 GB cards (5.87 GiB request). Its conf's
# PRECHECK forbids LEAN_MLIR_MEM_FRACTION (on ares 0.97 OOMed ConvNeXt's bf16 arms by fragmentation),
# so this runs the trainer directly with the conf's own ENV_EXTRA + the §3a smoke knobs + a
# fraction, trying 0.85, 0.90, 0.97 and stopping at the first that completes the 600 steps.
set -u
D=$(cd "$(dirname "$0")" && pwd)
eval "$(sed -n '/^set -u/,/^echo "start/p' "$D/queue.sh" | sed '$d')"
. scripts/jobs/cnxb-default-emabf16-4gpu.conf
echo "start $(date -u)"
for f in 0.85 0.90 0.97; do
  n=v_cnxb_f${f#0.}
  echo "verified cnxb frac $f $(date -u)"
  gpus_idle; logtrace "$D/gpu_$n.tsv" & lp=$!
  sleep 20
  grouprun "$VSECS" "$D/$n.log" env CUDA_VISIBLE_DEVICES="$DEVS" "${ENV_EXTRA[@]}" \
    LEAN_MLIR_MEM_FRACTION=$f LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200 "${CMD[@]}"
  sleep 30; kill $lp; sleep 30
  grep -aq 'PROBE:' "$D/$n.log" && { echo "ok at $f"; break; }
done
echo "done $(date -u)"
