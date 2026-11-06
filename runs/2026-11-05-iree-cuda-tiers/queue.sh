#!/bin/bash
# One card's queue of `-iree` tier runs, in order: the same binaries `lake run mnist-iree`,
# `cifar-iree` and `imagenette-iree` run, through run.sh with the environment runDemoGroup sets
# (LEAN_MLIR_LOWERER=iree, .venv-iree/bin on PATH, IREE_BACKEND=cuda), one card per queue so the
# three tiers and the seven Imagenette nets share the four 4060 Ti. The checkpoint tag `iree` keeps
# them off the XLA runs' paths. Each run's log lands in runs/2026-11-05-<tier>-iree-cuda/<net>.log
# (run.sh's tee), its wall clock and max RSS beside it in <net>.time.
#   setsid -f nohup runs/2026-11-05-iree-cuda-tiers/queue.sh <gpu> <tier>:<net>... > runs/2026-11-05-iree-cuda-tiers/queue-<gpu>.log 2>&1
cd "$(dirname "$0")/../.."
gpu=$1; shift
export PATH="$PWD/.venv-iree/bin:$PATH"
export LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda LEAN_MLIR_CKPT_TAG=iree
for item in "$@"; do
  tier=${item%%:*}; n=${item#*:}
  OUT=runs/2026-11-05-$tier-iree-cuda; mkdir -p $OUT
  export RUN_LOG_DIR=$OUT
  # The bbn net's bf16 arms need IREE's GPU vector-distribution pipeline off (lakefile.lean, `cifar-iree`).
  case $n in cifar8wb-bbn-ablation) export IREE_EXTRA_FLAGS=--iree-codegen-llvmgpu-use-vector-distribution=false ;; *) unset IREE_EXTRA_FLAGS ;; esac
  echo "━━━ $(date '+%F %T') gpu $gpu: $tier $n"
  /usr/bin/time -f "$n wall %e s, maxrss %M KB" -o $OUT/$n.time nice -n 19 ./run.sh $n $gpu cuda > $OUT/$n.out 2>&1 \
    && echo "  ✓ $(date '+%T') $(cat $OUT/$n.time)" || echo "  ✗ $(date '+%T') $n exit $? — $OUT/$n.out"
done
echo "━━━ $(date '+%F %T') gpu $gpu: queue done"
