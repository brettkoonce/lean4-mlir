#!/usr/bin/env bash
# CASP16 field data for the distogram demo (scripts/demos/casp16_score.py reads all of it):
# Phase-1 target sequences, the domain-trimmed experimental structures, the official per-model
# score table, the evaluation-unit table, the group list, and the field's predictions for the
# featured targets. Everything is public at predictioncenter.org; ~60 MB. Idempotent.
set -euo pipefail
ROOT=${CASP16_DIR:-"$(cd "$(dirname "$0")/../.." && pwd)/data/casp16"}
RAW=$ROOT/raw
B=https://predictioncenter.org/download_area/CASP16
mkdir -p "$RAW/dom" "$RAW/predictions" "$ROOT/tools"
cd "$RAW"
fetch() { [ -s "$2" ] || curl -sfL --retry 3 --max-time 600 -o "$2" "$1"; }
fetch $B/sequences/casp16.T1.seq.txt                        casp16.T1.seq.txt
fetch $B/targets/casp16.targets_monomer_trimmed2domains.tgz casp16.targets_monomer_trimmed2domains.tgz
fetch $B/results/tables/CASP16_prot_domains.scores.csv      CASP16_prot_domains.scores.csv
fetch "https://predictioncenter.org/casp16/domains_summary.cgi"       domains_summary.html
fetch "https://predictioncenter.org/casp16/docs.cgi?view=groupsbyname" groups.html
[ -s dom/T1235-D1.pdb ] || tar xzf casp16.targets_monomer_trimmed2domains.tgz -C dom
# the field's models for the featured targets (one tarball per target, all groups, 5 models each)
for t in ${CASP16_TARGETS:-T1235 T1267s1 T1226}; do
  fetch $B/predictions/regular/$t.tar.gz predictions/$t.tar.gz
  [ -d predictions/$t ] || tar xzf predictions/$t.tar.gz -C predictions
done
# scorers: US-align (static build from source) and OpenStructure (docker)
if [ ! -x "$ROOT/tools/USalign" ]; then
  git clone -q --depth 1 https://github.com/pylelab/USalign "$ROOT/tools/usalign-src"
  g++ -static -O3 -ffast-math -lm -o "$ROOT/tools/USalign" "$ROOT/tools/usalign-src/USalign.cpp"
fi
docker image inspect registry.scicore.unibas.ch/schwede/openstructure:latest >/dev/null 2>&1 \
  || docker pull registry.scicore.unibas.ch/schwede/openstructure:latest
echo "casp16: $(ls dom | wc -l) domain structures, $(grep -c '^>' casp16.T1.seq.txt) T1 sequences, $(ls -d predictions/*/ | wc -l) prediction sets"
