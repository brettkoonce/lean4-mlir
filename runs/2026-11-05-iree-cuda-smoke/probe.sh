#!/bin/bash
# Steady-state ms/step through IREE on one card — the Arc row's protocol (deploy/ARC_B50.md § Measured):
# LEAN_MLIR_MAX_STEPS=60 LEAN_MLIR_PROBE_WARM=20 LEAN_MLIR_SKIP_EVAL=1, the median over steps 21–60
# (the median, not the mean — the mean measures the host; the spread is the diagnostic). One line per
# net in probe.tsv: net, median, min, p90, mean ms/step, starvation wait, and the 80-epoch ETA at
# 295 steps per epoch plus the eval share.
#   runs/2026-11-05-iree-cuda-smoke/probe.sh <gpu> <net>...
cd "$(dirname "$0")/../.."
OUT=runs/2026-11-05-iree-cuda-smoke/probe; mkdir -p $OUT
gpu=$1; shift
export PATH="$PWD/.venv-iree/bin:$PATH"
export LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda LEAN_MLIR_MAX_STEPS=60 LEAN_MLIR_PROBE_WARM=20 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_CKPT_TAG=probe
export RUN_LOG_DIR=$OUT
[ -f $OUT/probe.tsv ] || printf '# IREE 3.12.0 cuda sm_86, 4060 Ti, ares, %s: LEAN_MLIR_MAX_STEPS=60 LEAN_MLIR_PROBE_WARM=20 LEAN_MLIR_SKIP_EVAL=1; ms/step over steps 21-60 at batch 32\nnet\tgpu\tmedian_ms\tmin_ms\tp90_ms\tmean_ms\twait_ms\ttrain_80ep_min\n' > $OUT/probe.tsv
for n in "$@"; do
  echo "━━━ probe $n (gpu $gpu) $(date '+%T')"
  nice -n 19 ./run.sh $n $gpu cuda > /dev/null 2> $OUT/$n.stderr || echo "  ✗ $n exit $? — $OUT/$n.stderr"
  L=$OUT/$n.log
  med=$(grep -m1 -oP 'PROBE: \K[0-9]+' $L); sp=$(grep -m1 'PROBE-SPREAD' $L)
  mn=$(echo "$sp" | grep -oP 'min=\K[0-9]+'); p90=$(echo "$sp" | grep -oP 'p90=\K[0-9]+'); mean=$(echo "$sp" | grep -oP 'mean=\K[0-9]+'); wait_=$(echo "$sp" | grep -oP 'wait = med-min = \K[0-9]+')
  if [ -n "$med" ]; then
    eta=$(echo "scale=1; $med * 295 * 80 / 60000" | bc)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$gpu" "$med" "$mn" "$p90" "$mean" "$wait_" "$eta" >> $OUT/probe.tsv
    echo "  ✓ $n: $med ms/step (min $mn, p90 $p90, mean $mean; wait $wait_) → 80 ep ≈ $eta min training"
  else echo "  ✗ $n: no PROBE line in $L"; fi
done
