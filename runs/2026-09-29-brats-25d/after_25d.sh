#!/usr/bin/env bash
# Once both 2.5D arms have exited: per-patient scoring of their best checkpoints (one GPU each,
# staggered — two PJRT sessions starting in the same second has failed twice today), then the
# four A/B comparisons, then the logs move into the run directory stripped of XLA chatter.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/2026-09-29-brats-25d
run() { local dev=$1; shift; echo "=== $*"; CUDA_VISIBLE_DEVICES=$dev ./.lake/build/bin/brats-eval "$@" 2>&1 | grep -v 'ptxas\|^W0\|^I0\|pjrt_ffi\|^    batch'; }
run 0 net=r34 ctx=1 arm=r34 best out=$OUT/pervol_25d_r34.csv
run 1 net=r34 ctx=1 arm=scratch best out=$OUT/pervol_25d_scratch.csv
for f in runs/brats_skip_c1_r34_gpu0.log runs/brats_skip_c1_scratch_gpu1.log; do
  grep -v 'ptxas\|^W0\|^I0\|^E0' "$f" > "$OUT/$(basename $f)"
done
echo "=== A/B: 2D r34 vs 2.5D r34"
python3 scripts/probes/brats_r34_ab.py runs/2026-09-25-brats-r34-xla/brats_skip_r34_gpu0.log $OUT/brats_skip_c1_r34_gpu0.log --names "2D r34,2.5D r34" --arm r34
echo "=== A/B: 2D scratch vs 2.5D scratch"
python3 scripts/probes/brats_r34_ab.py runs/2026-09-25-brats-r34-xla/brats_skip_scratch_gpu1.log $OUT/brats_skip_c1_scratch_gpu1.log --names "2D scratch,2.5D scratch" --arm scratch
echo "=== A/B: 2.5D r34 vs 2.5D scratch"
python3 scripts/probes/brats_r34_ab.py $OUT/brats_skip_c1_r34_gpu0.log $OUT/brats_skip_c1_scratch_gpu1.log
echo "=== render: 2.5D arms on the same slices"
CUDA_VISIBLE_DEVICES=0 ./.lake/build/bin/brats-predict net=r34 ctx=1 arm=scratch,r34 best $OUT/brats_25d_pred.ppm 2>&1 | grep -v 'ptxas\|^W0\|^I0\|pjrt_ffi'
echo ALL-DONE
