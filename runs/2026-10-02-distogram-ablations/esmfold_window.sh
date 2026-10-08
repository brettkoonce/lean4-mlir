#!/usr/bin/env bash
# ESMFold on the two Phase-1 targets over 1,000 residues (T1218, T1269), per evaluation unit from an
# 800-residue window — the five units the 78-unit ESMFold row needs (plan §9, §10). The 54 targets
# already folded are skipped; every unit is rescored and data/casp16/esmfold/fold_scores.csv rewritten
# with a `window` column. One 16 GB card, ~10 min.
#   setsid -f nohup runs/2026-10-02-distogram-ablations/esmfold_window.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
A=runs/2026-10-02-distogram-ablations
CUDA_VISIBLE_DEVICES=${GPU:-0} .venv-casp/bin/python -u scripts/demos/casp16_esmfold.py --window 800 > $A/esmfold_window.log 2>&1
echo "[$(date +%H:%M)] exit $?" >> $A/esmfold_window.log
