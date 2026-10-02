#!/usr/bin/env bash
# fold + score an ensemble arm (plain at the finish_run.sh defaults, orient at restarts 0 like the orient arms)
# usage: ens_fold.sh <gpu> <dir>
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=$1; d=$2
echo "[$(date +%H:%M)] fold plain"; $P -u scripts/demos/casp16_fold.py $d --device cuda --max-len 512 --threads 4 > $d/fold_all.log 2>&1
echo "[$(date +%H:%M)] score plain"; $P -u scripts/demos/casp16_fold_score.py $d > $d/fold_scores.log 2>&1; tail -n 5 $d/fold_scores.log
if ls $d/*.pred.npz >/dev/null 2>&1 && $P -c "import numpy,sys,glob; sys.exit(0 if 'omega' in numpy.load(sorted(glob.glob('$d/*.pred.npz'))[0]) else 1)"; then
  echo "[$(date +%H:%M)] fold orient"; $P -u scripts/demos/casp16_fold.py $d --device cuda --max-len 512 --threads 4 --orient --restarts 0 > $d/fold_all_orient.log 2>&1
  echo "[$(date +%H:%M)] score orient"; $P -u scripts/demos/casp16_fold_score.py $d --suffix orient > $d/fold_scores_orient.log 2>&1; tail -n 5 $d/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] done $d"
