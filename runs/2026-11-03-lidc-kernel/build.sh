#!/usr/bin/env bash
# Plan §7 (b) / §12: the 10 %-dose LIDC build under the `b50f` kernel — the same 300 patients, seed-0
# split, slice selection and per-slice I0 calibration as data/lidc10, one realisation, with the FBP
# apodisation fitted to Mayo's B50f-reconstructed noise (preprocess_lidc.py B50F_GAIN; the pre-build
# probe is runs/2026-11-03-lidc-kernel/probe_b50f/). Gates, in order: the split and the (patient,
# slice) record order equal data/lidc10's (so records_val.txt / records_spread.txt apply unchanged
# and every per-record comparison with the Shepp-Logan build is slice for slice), the loader tests at
# N = 1, the calibration (achieved/target within 2 %), and phase 4a's spectrum check on the BUILT val
# against Mayo (peak within 10 %) — the gate that failed under Shepp-Logan (ratio 1.29).
#
#   runs/2026-11-03-lidc-kernel/build.sh [workers=24]     # ≈ 1.3 h at 24 workers on an idle box; resumable
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${1:-24}
PY=.venv-lidc/bin/python
RUN=runs/2026-11-03-lidc-kernel
OUT=data/lidc10b50
[ -d data/lidc/dicom ] && [ -d data/lidc/xml ] || { echo "⛔ data/lidc/{dicom,xml} missing"; exit 1; }
mkdir -p $OUT $RUN
if [ ! -e $OUT/val.bin ]; then
  nice -n 19 $PY scripts/datasets/preprocess_lidc.py data/lidc/dicom data/lidc/xml $OUT \
    --dose 0.10 --filter b50f --realisations 1 --workers "$W" --resume 2>&1 | tee -a $OUT/preprocess.log
fi
echo "== gates =="
cmp <(python3 -c "import json;print(json.load(open('data/lidc10/split.json')))") \
    <(python3 -c "import json;print(json.load(open('$OUT/split.json')))") \
  && echo "split identical to data/lidc10" || { echo "⛔ split differs from data/lidc10"; exit 1; }
$PY - <<PY
import json
for s in ("val", "train"):
    a = [(r["patient"], r["slice"]) for r in json.load(open(f"data/lidc10/{s}.json"))]
    b = [(r["patient"], r["slice"]) for r in json.load(open("$OUT/{}.json".format(s)))]
    assert a == b, f"⛔ {s}: record order differs from data/lidc10"
    print(f"{s}: {len(a)} records, the (patient, slice) order of data/lidc10")
m = json.load(open("$OUT/manifest_lidc.json"))
r = m["achieved_over_target"]["median"]
assert m["filters"] == ["b50f"], m["filters"]
assert abs(r - 1) <= 0.02, f"⛔ calibration achieved/target {r}"
print(f"calibration achieved/target median {r:.3f}; filters {m['filters']}; i0_add median {m['i0_add']['median']:.3g}")
PY
lake build test-lidc-loader test-dataset-record-sizes 2>&1 | tail -1
for b in $OUT/val.bin $OUT/train.bin; do $PY scripts/probes/lidc_loader_check.py $b | tail -1 | grep PASS || { echo "⛔ loader gate failed on $b"; exit 1; }; done
.lake/build/bin/test-dataset-record-sizes 2>&1 | grep -i "lidc10 " || true
echo "== phase 4a: the built val's noise spectrum against Mayo's =="
nice -n 10 $PY scripts/probes/lidc_noise_nps_check.py $OUT/val.bin data/mayo/val.bin --out-dir $RUN/nps_check \
  --lidc-records 1000 --mayo-records 0 --lidc-label "LIDC (simulated, 10%, b50f)" | tee $RUN/nps_check.txt
grep -q "gate (peak within 10%): PASS" $RUN/nps_check.txt || { echo "⛔ spectrum gate failed"; exit 1; }
echo "build + gates done $(date -u +%H:%M)"
