#!/usr/bin/env bash
# Plan §13: the dose-range build under Mayo's kernel — §11's range (dose log-uniform in [0.03, 0.25] per
# slice, one realisation) reconstructed through the `b50f` window of §12 instead of §11's five textbook
# filters; the same 300 patients, seed-0 split and slice selection as every other build. Then the σ̂
# sidecars (ct_noise_level.py) for the new files and for the b50f 10 % val the generator is sampled on,
# and the gates: split and (patient, slice) order equal data/lidc10's; filters == ['b50f']; calibration
# achieved/target within 2 %; the loader tests at N = 1; phase 4a's spectrum check on the built val
# against Mayo; the record list of the new val by the §10.5 rule.
#
#   runs/2026-11-03-lidc-rangekernel/build.sh [workers=24]   # ≈ 3 h at 24 workers (the b50f path, §12.3); resumable
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${1:-24}
PY=.venv-lidc/bin/python
RUN=runs/2026-11-03-lidc-rangekernel
OUT=data/lidcmixb50
[ -d data/lidc/dicom ] && [ -d data/lidc/xml ] || { echo "⛔ data/lidc/{dicom,xml} missing"; exit 1; }
free=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
[ -e $OUT/val.bin ] || [ "$free" -ge 25 ] || { echo "⛔ only ${free} GB free; the build needs ≈ 18"; exit 1; }
mkdir -p $OUT $RUN
if [ ! -e $OUT/val.bin ]; then
  nice -n 19 $PY scripts/datasets/preprocess_lidc.py data/lidc/dicom data/lidc/xml $OUT \
    --dose-range 0.03 0.25 --filter b50f --realisations 1 --workers "$W" --resume 2>&1 | tee -a $OUT/preprocess.log
fi
echo "== gates =="
cmp <(python3 -c "import json;print(json.load(open('data/lidc10/split.json')))") \
    <(python3 -c "import json;print(json.load(open('$OUT/split.json')))") \
  && echo "split identical to data/lidc10" || { echo "⛔ split differs from data/lidc10"; exit 1; }
$PY - <<PY
import json
for s in ("val", "train"):
    a = [(r["patient"], r["slice"]) for r in json.load(open(f"data/lidc10/{s}.json"))]
    rows = json.load(open("$OUT/{}.json".format(s)))
    b = [(r["patient"], r["slice"]) for r in rows]
    assert a == b, f"⛔ {s}: record order differs from data/lidc10"
    d = sorted(r["dose"] for r in rows)
    print(f"{s}: {len(a)} records, the (patient, slice) order of data/lidc10; dose min/quartiles/max "
          + " / ".join(f"{d[int(q * (len(d) - 1))]:.3f}" for q in (0, .25, .5, .75, 1)))
m = json.load(open("$OUT/manifest_lidc.json"))
r = m["achieved_over_target"]["median"]
assert m["filters"] == ["b50f"], m["filters"]
assert abs(r - 1) <= 0.02, f"⛔ calibration achieved/target {r}"
print(f"calibration achieved/target median {r:.3f}; filters {m['filters']}; i0_add median {m['i0_add']['median']:.3g}")
for k, v in m.items():
    if "quartile" in k or "dose" in k:
        print(f"  manifest {k}: {json.dumps(v)[:300]}")
PY
lake build test-lidc-loader test-dataset-record-sizes 2>&1 | tail -1
for b in $OUT/val.bin $OUT/train.bin; do $PY scripts/probes/lidc_loader_check.py $b | tail -1 | grep PASS || { echo "⛔ loader gate failed on $b"; exit 1; }; done
echo "== σ̂ sidecars =="
nice -n 19 $PY scripts/datasets/ct_noise_level.py $OUT/train.bin $OUT/val.bin data/lidc10b50/val.bin --workers 8 2>&1 | tee $RUN/sigma.txt
ls -la $OUT/*.sigma.f32 data/lidc10b50/val.bin.sigma.f32
echo "== phase 4a: the built val's noise spectrum against Mayo's =="
nice -n 10 $PY scripts/probes/lidc_noise_nps_check.py $OUT/val.bin data/mayo/val.bin --out-dir $RUN/nps_check \
  --lidc-records 1000 --mayo-records 0 --lidc-label "LIDC (simulated, 3-25 %, b50f)" | tee $RUN/nps_check.txt
grep -q "gate (peak within 10%): PASS" $RUN/nps_check.txt || { echo "⛔ spectrum gate failed"; exit 1; }
$PY scripts/demos/lidc_select_records.py --data=$OUT --out=$RUN/records_mixb50.txt | tail -1
echo "build + gates done $(date -u +%H:%M)"
