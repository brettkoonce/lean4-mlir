#!/usr/bin/env bash
# Platform integration suite: does THIS GPU platform run THIS build?
# Design and the tier table: planning/platform_integration.md.
#
#   scripts/platform/check.sh [--tier 0|1] [--backend cuda|rocm|xpu] [--plan] [--out DIR]
#
#   --tier N     run tiers 0..N (default 2). Tier 3 (training) is not written yet.
#   --backend    default: whichever vendor SMI is on PATH.
#   --plan       print what would run and the expected wall time; launch nothing.
#   --out DIR    default runs/platform/<date>-<host>-<backend>[-k].
#
# Which devices it uses is CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES, as for every
# trainer; set them to keep the suite off cards that are busy. The two-device tests SKIP
# below two visible devices.
#
# Everything is built fresh into <out>/build — the shim included — and the tests link
# that copy, never ffi/libpjrt_ffi.so, so a run cannot pick up a stale binary and cannot
# replace one a running trainer has loaded.
#
# Writes <out>/manifest.json (core vs platform, section 4 of the plan), <out>/results.tsv,
# <out>/logs/ (untracked: a committed run is its manifest and results), then regenerates PLATFORMS.md. Exit 1 on any FAIL; XPASS (a listed
# failure that passed) is reported and means scripts/platform/expected/ needs an edit.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

TIER=2; BACKEND=""; PLAN=0; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    --plan) PLAN=1; shift ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$TIER" in 0|1|2) ;; *) echo "--tier $TIER: only tiers 0-2 exist so far" >&2; exit 2 ;; esac

if [ -z "$BACKEND" ]; then
  if command -v nvidia-smi >/dev/null; then BACKEND=cuda
  elif command -v rocm-smi >/dev/null; then BACKEND=rocm
  elif command -v xpu-smi >/dev/null; then BACKEND=xpu
  else echo "no vendor SMI on PATH; pass --backend" >&2; exit 2; fi
fi
case "$BACKEND" in
  cuda) SMI=(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader)
        DEFAULT_PLUGIN=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so ;;
  rocm) SMI=(rocm-smi --showproductname --showdriverversion)
        DEFAULT_PLUGIN=.venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so ;;
  xpu)  SMI=(xpu-smi discovery)
        DEFAULT_PLUGIN="" ;;
  *) echo "unknown backend $BACKEND" >&2; exit 2 ;;
esac
PLUGIN=${PJRT_PLUGIN:-$DEFAULT_PLUGIN}

if [ "$PLAN" = 1 ]; then
  echo "platform suite — backend $BACKEND, tiers 0..$TIER, plugin ${PLUGIN:-<unset: export PJRT_PLUGIN>}"
  echo "  visible devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<all>} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-<all>}"
  echo "  build   shim + probe + smoke + 4 ffi tests (gcc)             ~10 s"
  echo "  tier 0  smi, probe (plugin, API version, devices), smoke     ~15 s"
  [ "$TIER" -ge 1 ] && \
  echo "  tier 1  guards, compile-dp, allreduce, dp (last 3 need 2 devices)  ~1 min"
  [ "$TIER" -ge 2 ] && \
  echo "  tier 2  $(grep -c '^[a-z]' scripts/platform/tier2_artifacts.tsv) verified_mlir artifacts vs XLA:CPU goldens, one device   ~3 min"
  exit 0
fi

slug() { tr 'A-Z' 'a-z' | sed -E 's/nvidia geforce |nvidia |amd radeon |intel\(r\) //; s/[^a-z0-9]+/-/g; s/^-|-$//g'; }
if [ -z "$OUT" ]; then
  base="runs/platform/$(date -u +%F)-$(hostname -s)-$BACKEND"; OUT=$base; k=2
  while [ -e "$OUT" ]; do OUT="$base-$k"; k=$((k+1)); done
fi
mkdir -p "$OUT/build" "$OUT/logs"
BUILD=$(cd "$OUT/build" && pwd); LOGS=$OUT/logs
RESULTS=$OUT/results.tsv
printf 'test\ttier\tstatus\tdetail\n' > "$RESULTS"

# expected/<backend>-<gpu>.txt: `<test> FAIL|FLAKE  # why`, Mesa deqp-runner style.
GPU_SLUG=unknown
[ "$BACKEND" = cuda ] && GPU_SLUG=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1 | slug)
[ -z "$GPU_SLUG" ] && GPU_SLUG=unknown
EXPECTED=scripts/platform/expected/$BACKEND-$GPU_SLUG.txt
expected_status() { [ -f "$EXPECTED" ] && awk -v t="$1" '$1==t {print $2; exit}' "$EXPECTED"; }

NFAIL=0; NXPASS=0
record() {  # record <test> <tier> <rc> <detail>   (rc: 0 pass, 77 skip, else fail)
  local t=$1 tier=$2 rc=$3 detail=$4 exp st
  exp=$(expected_status "$t")
  if [ "$rc" = 77 ]; then st=SKIP
  elif [ "$rc" = 0 ]; then
    if [ "$exp" = FAIL ]; then st=XPASS; NXPASS=$((NXPASS+1)); else st=PASS; fi
  else
    if [ "$exp" = FAIL ] || [ "$exp" = FLAKE ]; then st=XFAIL; else st=FAIL; NFAIL=$((NFAIL+1)); fi
  fi
  printf '%s\t%s\t%s\t%s\n' "$t" "$tier" "$st" "$detail" >> "$RESULTS"
  printf '  %-16s %-5s %s\n' "$t" "$st" "$detail"
}
# run <test> <tier> <ok-regex> <cmd...>: pass = exit 0 AND the known-answer line printed.
run() {
  local t=$1 tier=$2 ok=$3; shift 3
  timeout 300 "$@" > "$LOGS/$t.log" 2>&1; local rc=$?
  if [ $rc = 0 ] && ! grep -qE "$ok" "$LOGS/$t.log"; then rc=1; fi
  local detail
  if [ $rc = 0 ]; then detail=$(grep -E "$ok" "$LOGS/$t.log" | tail -1)
  elif [ $rc = 124 ]; then detail="timed out after 300 s"
  else detail=$(grep -E 'FAIL|error|Error|✗|WRONG' "$LOGS/$t.log" | head -1); [ -z "$detail" ] && detail="exit $rc"; fi
  record "$t" "$tier" "$rc" "${detail:0:160}"
}
finish() {
  python3 scripts/platform/manifest.py "$OUT" --backend "$BACKEND" --plugin "$PLUGIN" --tier "$TIER" \
    || echo "  (manifest.py failed)"
  python3 scripts/platform/platforms_md.py || echo "  (platforms_md.py failed)"
  echo
  echo "$(grep -c $'\tPASS\t' "$RESULTS") pass, $(grep -c $'\tXFAIL\t' "$RESULTS") xfail, $NFAIL fail, $NXPASS xpass, $(grep -c $'\tSKIP\t' "$RESULTS") skip — $OUT"
  [ "$NXPASS" -gt 0 ] && echo "XPASS: a listed failure now passes — drop it from $EXPECTED"
  [ "$NFAIL" -gt 0 ] && exit 1
  exit 0
}

echo "platform suite — $BACKEND, tiers 0..$TIER → $OUT"
echo "build:"
CC=${CC:-gcc}
b() { local t=$1; shift; "$CC" "$@" >> "$LOGS/build.log" 2>&1; record "build:$t" 0 $? ""; }
b shim     -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o "$BUILD/libpjrt_ffi.so"
b probe    -O2 -Iffi scripts/platform/probe.c -ldl -o "$BUILD/probe"
b smoke    -O2 -Iffi scripts/platform/smoke.c -L"$BUILD" -lpjrt_ffi -ldl -Wl,-rpath,"$BUILD" -o "$BUILD/smoke"
if [ "$TIER" -ge 1 ]; then
  b guards        -O2 -Iffi ffi/test_pjrt_guards.c -L"$BUILD" -lpjrt_ffi -ldl -Wl,-rpath,"$BUILD" -o "$BUILD/test_pjrt_guards"
  b dp            -O2 -Iffi ffi/test_pjrt_dp.c -L"$BUILD" -lpjrt_ffi -ldl -Wl,-rpath,"$BUILD" -o "$BUILD/test_pjrt_dp"
  b allreduce     -O2 -Iffi ffi/test_pjrt_allreduce.c -ldl -o "$BUILD/test_pjrt_allreduce"
  b compile_check -O2 -Iffi ffi/test_pjrt_compile_check.c -ldl -o "$BUILD/test_pjrt_compile_check"
fi
[ "$TIER" -ge 2 ] && \
  b tier2_run   -O2 -Iffi scripts/platform/tier2_run.c -L"$BUILD" -lpjrt_ffi -ldl -Wl,-rpath,"$BUILD" -o "$BUILD/tier2_run"
[ "$NFAIL" -gt 0 ] && { echo "build failed — see $LOGS/build.log"; finish; }

echo "tier 0 — platform:"
"${SMI[@]}" > "$LOGS/smi.log" 2>&1; record smi 0 $? "$(head -1 "$LOGS/smi.log" | cut -c1-120)"
if [ -z "$PLUGIN" ] || [ ! -f "$PLUGIN" ]; then
  record probe 0 1 "no plugin at '${PLUGIN}' — export PJRT_PLUGIN"; finish
fi
run probe 0 '^devices=[1-9]' "$BUILD/probe" "$PLUGIN"
NDEV=$(sed -n 's/^devices=//p' "$LOGS/probe.log"); NDEV=${NDEV:-0}
# Off CUDA the baseline is named from the plugin's own device kind (ROCm, XPU).
if [ "$GPU_SLUG" = unknown ]; then
  GPU_SLUG=$(sed -n 's/^device_0=//p' "$LOGS/probe.log" | slug); GPU_SLUG=${GPU_SLUG:-unknown}
  EXPECTED=scripts/platform/expected/$BACKEND-$GPU_SLUG.txt
fi
run smoke 0 'compile \+ execute OK' env PJRT_PLUGIN="$PLUGIN" "$BUILD/smoke" scripts/platform/fixtures/add.mlir
[ "$TIER" -lt 1 ] && finish

echo "tier 1 — shim:"
"$BUILD/probe" --dump-options 2 "$BUILD/options_r2.pb" >> "$LOGS/build.log" 2>&1
run guards 1 'all guards fire' env PJRT_PLUGIN="$PLUGIN" "$BUILD/test_pjrt_guards"
if [ "$NDEV" -lt 2 ]; then
  for t in compile_dp allreduce dp; do record $t 1 77 "needs 2 devices, $NDEV visible"; done
else
  # a real Lean-emitted 2-replica train step, compiled with the shim's own r=2 options
  run compile_dp 1 'COMPILED OK' "$BUILD/test_pjrt_compile_check" "$PLUGIN" \
      verified_mlir/cifar8_adamdp_train_step.mlir "$BUILD/options_r2.pb"
  run allreduce 1 'ALL-REDUCE CORRECT' "$BUILD/test_pjrt_allreduce" "$PLUGIN" \
      scripts/platform/fixtures/allreduce.mlir "$BUILD/options_r2.pb"
  run dp 1 'DP INVOKE CORRECT' env PJRT_PLUGIN="$PLUGIN" PJRT_REPLICAS=2 \
      "$BUILD/test_pjrt_dp" scripts/platform/fixtures/dp_shard.mlir
fi
[ "$TIER" -lt 2 ] && finish

echo "tier 2 — ops vs XLA:CPU goldens:"
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python   # numpy only; no JAX needed here
while IFS=$'\t' read -r name family dtype; do
  case "$name" in ''|'#'*) continue ;; esac
  run "$name" 2 '^OK ' env PJRT_PLUGIN="$PLUGIN" "$PY" scripts/platform/tier2.py run "$name" \
      --runner "$BUILD/tier2_run" --work "$BUILD/tier2" --backend "$BACKEND"
done < scripts/platform/tier2_artifacts.tsv
finish
