#!/bin/bash
# deploy/mac_bench.sh — how long will this Mac take? The train step of each net the Mac could run,
# timed on the GPU, and the ladder and the Imagenette runs estimated from it. ~10–20 minutes.
#
#   bash deploy/mac_bench.sh             # the CIFAR batch-BN net and the seven Imagenette nets
#   bash deploy/mac_bench.sh r34 vit     # some of them (the short names below, or lean_exe names)
#
# Each net runs its real trainer binary on SYNTHETIC data (LEAN_MLIR_BENCH_SYNTH=1: one batch,
# repeated, so no dataset is needed) for a few dozen steps and prints the trainer's own
# `PROBE:` line — the median ms/step over steps 21..70, the first 20 being warm-up — then
# exits before it writes anything. An estimate for a full run follows from that: the steps per
# epoch (read from the trainer's `step N/M` line), the epochs the tier trains, and a share for
# the per-epoch eval pass (+15 % at 224², +10 % on CIFAR), which the probe does not time.
#
# The memory side is NOT benchmarked: synthetic data is one batch, where a real Imagenette run
# keeps the 7.4 GB train split resident unless LEAN_MLIR_IMAGENETTE_STREAM=1 (deploy/MAC.md).
#
# Results: runs/<date>-mac-bench/bench.tsv (one row per net), estimate.txt (the summary below),
# and each net's log. Needs deploy/mac_setup.sh green first (the venv, the plugin, the toolchain).
#
# Written for macOS's /bin/bash 3.2 and BSD awk/sed.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" = Darwin ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ]; then
  if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
    echo "mac_bench: this shell runs under Rosetta (x86_64); re-running natively as arm64"
    exec arch -arm64 /bin/bash "$0" "$@"
  fi
fi

WARM=${BENCH_WARM:-20}; STEPS=${BENCH_STEPS:-70}   # the probe window: steps WARM+1..STEPS

# short name | lean_exe | extra argv | epochs the tier trains | eval share
NETS="bbn|cifar8wb-bbn-ablation|data sgd|40|1.10
r34|resnet34-verified-adam||80|1.15
r50|resnet50-verified-adam||80|1.15
mnv2|mobilenetv2-verified-adam||80|1.15
mnv4|mobilenetv4-verified-adam||80|1.15
enet|efficientnet-verified-adam||80|1.15
cnx|convnext-verified-adam||80|1.15
vit|vit-verified-adam||80|1.15"

pick () {   # pick <name>: the NETS line whose short or exe name is <name>
  echo "$NETS" | awk -F'|' -v n="$1" '$1 == n || $2 == n'
}
if [ $# -eq 0 ]; then ROWS=$NETS
else
  ROWS=""
  for n in "$@"; do
    r=$(pick "$n"); [ -n "$r" ] || { echo "unknown net '$n' — one of: $(echo "$NETS" | cut -d'|' -f1 | tr '\n' ' ')" >&2; exit 2; }
    ROWS="$ROWS$r
"
  done
fi

[ "$(uname -s)" = Darwin ] || { echo "not macOS — this is the Mac's bench" >&2; exit 1; }
[ -x .venv/bin/python3 ] || { echo "no .venv — run deploy/mac_setup.sh first" >&2; exit 1; }
[ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
command -v lake > /dev/null || { echo "no lake on PATH — run deploy/mac_setup.sh first" >&2; exit 1; }
. scripts/platform/env.sh
[ "$PLATFORM_BACKEND" = metal ] || { echo "env.sh resolved backend '$PLATFORM_BACKEND', not metal" >&2; exit 1; }
[ -n "${PJRT_PLUGIN:-}" ] && [ -f "$PJRT_PLUGIN" ] || { echo "no Metal plugin in .venv — run: bash deploy/mac_setup.sh venv" >&2; exit 1; }
export PJRT_PLUGIN LEAN_MLIR_LOWERER=xla

L=runs/$(date +%F)-mac-bench
mkdir -p "$L"
TSV=$L/bench.tsv
printf 'net\texe\tbs\tsteps_per_epoch\tstep_ms_med\tstep_ms_min\tstep_ms_p90\tepoch_est_s\tepochs\trun_est_h\tstatus\n' > "$TSV"
chip=$(sysctl -n machdep.cpu.brand_string); mem=$(( $(sysctl -n hw.memsize) / 1073741824 ))
gpu=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ {print $2; exit}')
echo "════ mac_bench $(date '+%F %T') — $chip, ${gpu:-?} GPU cores, $mem GB, macOS $(sw_vers -productVersion), host $(hostname -s)"
echo "  plugin $PJRT_PLUGIN; probe window steps $((WARM + 1))..$STEPS on synthetic data"
{ git log --oneline -1; git status --short | grep -v '^??'; } > "$L/commit.txt" 2>/dev/null

# The shim the trainers dlopen, fresh from its source (what lake run does before a tier).
if [ ! -f ffi/libpjrt_ffi.so ] || [ ffi/pjrt_ffi.c -nt ffi/libpjrt_ffi.so ]; then
  cc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so || { echo "the shim did not compile" >&2; exit 1; }
fi

caffeinate -dims -w $$ &

echo "$ROWS" | while IFS='|' read -r name exe argv epochs share; do
  [ -n "$name" ] || continue
  printf '\n━━━ %s  %s (%s)\n' "$(date '+%T')" "$name" "$exe"
  log=$L/$name.log
  if ! lake build "$exe" > "$L/build-$name.out" 2>&1; then
    tail -20 "$L/build-$name.out"
    printf '%s\t%s\t-\t-\t-\t-\t-\t-\t%s\t-\tBUILD FAILED\n' "$name" "$exe" "$epochs" >> "$TSV"
    echo "  ✗ lake build $exe failed — $L/build-$name.out"; continue
  fi
  t0=$SECONDS
  # shellcheck disable=SC2086
  LEAN_MLIR_BENCH_SYNTH=1 LEAN_MLIR_MAX_STEPS=$STEPS LEAN_MLIR_PROBE_WARM=$WARM \
  LEAN_MLIR_CKPT_TAG=bench$$ \
    .lake/build/bin/"$exe" $argv > "$log" 2>&1
  rc=$?; el=$((SECONDS - t0))
  rm -f .lake/build/*_ckpt_xla_bench$$.bin*   # the probe returns before a checkpoint; belt and braces
  probe=$(grep -m1 '^  PROBE:' "$log")
  spread=$(grep -m1 '^  PROBE-SPREAD:' "$log")
  bs=$(sed -nE 's/.*; bs ([0-9]+),.*/\1/p' "$log" | head -1)
  spe=$(sed -nE 's/^  step [0-9]+\/([0-9]+):.*/\1/p' "$log" | head -1)
  if [ -z "$probe" ]; then
    grep -iE 'error|fail|abort|killed|out of memory|unsupported' "$log" | head -3
    printf '%s\t%s\t%s\t%s\t-\t-\t-\t-\t%s\t-\tFAILED rc=%s\n' "$name" "$exe" "${bs:--}" "${spe:--}" "$epochs" "$rc" >> "$TSV"
    echo "  ✗ no PROBE line after ${el}s (rc=$rc) — $log"; continue
  fi
  med=$(echo "$probe" | sed -nE 's/^  PROBE: ([0-9]+) ms\/step.*/\1/p')
  min=$(echo "$spread" | sed -nE 's/.*min=([0-9]+).*/\1/p'); p90=$(echo "$spread" | sed -nE 's/.*p90=([0-9]+).*/\1/p')
  est=$(awk -v s="$spe" -v m="$med" -v sh="$share" -v e="$epochs" 'BEGIN {
    ep = s * m / 1000 * sh; printf "%.0f\t%.2f", ep, ep * e / 3600 }')
  ep_s=${est%	*}; run_h=${est#*	}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tok\n' "$name" "$exe" "$bs" "$spe" "$med" "${min:--}" "${p90:--}" "$ep_s" "$epochs" "$run_h" >> "$TSV"
  echo "  $probe"
  echo "  $spread"
  echo "  → $spe steps/epoch at bs $bs: ~${ep_s} s/epoch, ~${run_h} h for $epochs epochs  (${el}s incl. compile)"
done

# ─── the estimate ────────────────────────────────────────────────────────────
{
  echo "mac_bench $(date '+%F %T') — $chip, ${gpu:-?} GPU cores, $mem GB, macOS $(sw_vers -productVersion), $(hostname -s)"
  echo "jax-graft $(.venv/bin/python3 -c 'import importlib.metadata as m; print(m.version("jax-graft"))' 2>/dev/null), commit $(git rev-parse --short HEAD)"
  echo
  echo "train step, median ms (synthetic data, steps $((WARM + 1))..$STEPS):"
  awk -F'\t' 'NR > 1 { printf "  %-6s %-28s bs %-4s %7s ms/step  (min %s, p90 %s)  %s\n", $1, $2, $3, $5, $6, $7, ($11 == "ok" ? "" : $11) }' "$TSV"
  echo
  bbn=$(awk -F'\t' '$1 == "bbn" && $11 == "ok" { print $8 }' "$TSV")
  if [ -n "$bbn" ]; then
    # deploy/mac_ladder.sh: lake run cifar (12 arms × 40 ep), 5 seeds × 6 arms × 40, 10 widths × 25;
    # all three at the batch-BN rate (the no-BN and per-image-BN arms are cheaper, so this is high),
    # plus ~15 min for the suite, MNIST and the gradient check.
    awk -v ep="$bbn" 'BEGIN {
      c = 480 * ep / 3600; s = 1200 * ep / 3600; g = 250 * ep / 3600
      printf "deploy/mac_ladder.sh (CIFAR and below), at %.0f s per CIFAR epoch:\n", ep
      printf "  cifar  12 arms x 40 ep     %5.1f h\n  seeds  5 x 6 arms x 40 ep  %5.1f h\n  grid   10 widths x 25 ep   %5.1f h\n", c, s, g
      printf "  suite + mnist + gradcheck   ~0.3 h\n  total                      %5.1f h\n", c + s + g + 0.25 }'
  else
    echo "deploy/mac_ladder.sh: no estimate — the CIFAR batch-BN net did not bench"
  fi
  echo
  echo "Imagenette, 80 epochs each (lake run train <name>; +15 % per epoch for the val pass):"
  awk -F'\t' 'NR > 1 && $1 != "bbn" { if ($11 == "ok") { printf "  %-6s %5.1f h  (%s s/epoch)\n", $1, $10, $8; t += $10 } else printf "  %-6s   —    %s\n", $1, $11 }
    END { if (t > 0) printf "  all seven: %.1f h\n", t }' "$TSV"
  echo
  echo "Reference points: a 7900 XTX runs the ladder in ~1.5 h and all seven Imagenette nets in ~6.7 h;"
  echo "an M1 Air (8 GPU cores) does ~12 s per CIFAR epoch (~7 h ladder) and ~5.7 min per ResNet-34 epoch."
} | tee "$L/estimate.txt"
echo
echo "━━━ bench.tsv, estimate.txt and the logs: $L/"
