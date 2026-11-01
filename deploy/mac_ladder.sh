#!/bin/bash
# deploy/mac_ladder.sh — the 7900 XTX's ladder, CIFAR and below, on an Apple Silicon Mac: after
# deploy/mac_setup.sh is green, the platform suite, MNIST, the batch-BN gradient check, CIFAR's
# arms, the batch-BN seeds and the head-width grid, one GPU, one job at a time.
# deploy/MAC.md § The ladder is the runbook; runs/2026-11-01-ares-xtx-ladder/sweep.sh is the
# 7900 XTX's run of the same rungs (on the `rocm10-7900xtx` branch).
#
#   bash deploy/mac_ladder.sh               # every stage, in order
#   bash deploy/mac_ladder.sh cifar         # start at a stage (build data suite mnist gradcheck cifar seeds grid)
#   bash deploy/mac_ladder.sh seeds grid    # only the stages named
#
#   build      lake build the eight binaries the ladder runs
#   data       MNIST into data/mnist/, CIFAR-10 into data/cifar-10/
#   suite      scripts/platform/check.sh: tiers 0–2, into runs/platform/<date>-<host>-metal/
#   mnist      `lake run mnist`: linear, MLP, CNN, 12 epochs
#   gradcheck  cifar8-bbn-gradcheck: the batch-BN step's backward, both tiers
#   cifar      `lake run cifar`: no BN, per-image BN, batch BN (twelve arms, 40 epochs)
#   seeds      cifar8wb-bbn-ablation at seeds 1–5, six arms each (scripts/sweeps/seed_sweep.sh's
#              SUITE=bbn, same log names and done-markers)
#   grid       cifar8wb-bbn-grid at the ten head widths, 25 epochs
#              (scripts/sweeps/run_cifar8wb_bbn_grid_sweep.sh, same results.tsv columns)
#
# Why not those two sweep scripts themselves: they are bash 4 with gawk, flock and a journalctl
# PCIe watchdog, none of which a stock Mac has. Here the seeds and widths run one after another
# on the one GPU, with the same environment, logs and result columns.
#
# Resumable: each stage leaves runs/<date>-mac-ladder/done/<stage>, and seeds and grid leave one
# marker per job, so running it again finishes it (MAC_LADDER_DIR pins the directory across a
# midnight). The Mac is held awake with caffeinate while it runs; plug it in.
#
# Written for macOS's /bin/bash 3.2 and BSD awk/sed.
set -uo pipefail
cd "$(dirname "$0")/.."

# Under Rosetta, re-run natively, as mac_setup.sh does: the plugin is arm64-only.
if [ "$(uname -s)" = Darwin ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ]; then
  if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
    echo "mac_ladder: this shell runs under Rosetta (x86_64); re-running natively as arm64"
    exec arch -arm64 /bin/bash "$0" "$@"
  fi
fi

ALL="build data suite mnist gradcheck cifar seeds grid"
SEEDS=${SEEDS:-"1 2 3 4 5"}
GRID_EPOCHS=${GRID_EPOCHS:-25}
WIDTHS="8 4096 16 2048 32 1024 64 512 128 256"   # the grid sweep's order, small and large interleaved

if [ $# -eq 0 ]; then STAGES=$ALL
elif [ $# -eq 1 ]; then
  case " $ALL " in *" $1 "*) ;; *) echo "unknown stage '$1' (one of: $ALL)" >&2; exit 2 ;; esac
  STAGES=${ALL#*"$1"}; STAGES="$1$STAGES"
else
  for s in "$@"; do case " $ALL " in *" $s "*) ;; *) echo "unknown stage '$s' (one of: $ALL)" >&2; exit 2 ;; esac; done
  STAGES="$*"
fi

L=${MAC_LADDER_DIR:-$(ls -d runs/*-mac-ladder 2>/dev/null | tail -1)}
L=${L:-runs/$(date +%F)-mac-ladder}
mkdir -p "$L/done"
LOG=$L/ladder.log
exec > >(tee -a "$LOG") 2>&1
STAGE=${STAGES%% *}

say ()  { printf '\n━━━ %s  %s\n' "$(date '+%F %T')" "$*"; }
ok ()   { printf '  ✓ %s\n' "$*"; }
die ()  { printf '\n  ✗ %s\n\n  The log is %s. Fix the above, then resume with:\n    bash deploy/mac_ladder.sh %s\n' "$1" "$LOG" "$STAGE" >&2; exit 1; }
want () { case " $STAGES " in *" $1 "*) STAGE=$1; return 0 ;; *) return 1 ;; esac; }
done_already () { [ -f "$L/done/$1" ] && { ok "$1 already done ($(cat "$L/done/$1")) — rm $L/done/$1 to redo it"; return 0; }; return 1; }
mark () { echo "$(date '+%F %T') $*" > "$L/done/$STAGE"; }

[ "$(uname -s)" = Darwin ] || die "not macOS — this is the Mac's ladder; the Linux boxes run the sweep scripts"
[ -x .venv/bin/python3 ] || die "no .venv — run deploy/mac_setup.sh first"
[ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
command -v lake > /dev/null || die "no lake on PATH — run deploy/mac_setup.sh first"

. scripts/platform/env.sh
[ "$PLATFORM_BACKEND" = metal ] || die "scripts/platform/env.sh resolved backend '$PLATFORM_BACKEND', not metal"
[ -n "${PJRT_PLUGIN:-}" ] && [ -f "$PJRT_PLUGIN" ] || die "no Metal plugin in .venv — run: bash deploy/mac_setup.sh venv"
export PJRT_PLUGIN LEAN_MLIR_LOWERER=xla
# Its own checkpoints, as the 7900 XTX's ladder had (ares-xtx80): an earlier partial run's
# untagged checkpoint would otherwise be resumed, or read as finished and not trained at all.
# The seeds override it per seed (s1…s5), as seed_sweep.sh does.
export LEAN_MLIR_CKPT_TAG=${LEAN_MLIR_CKPT_TAG:-mac-ladder}

echo "════ mac_ladder $(date '+%F %T') — stages: $STAGES → $L"
echo "  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion), host $(hostname -s)"
echo "  plugin $PJRT_PLUGIN"
{ git log --oneline -1; git status --short | grep -v '^??'; } > "$L/commit.txt" 2>/dev/null

# Awake for as long as this shell lives (-s holds only on AC power, so say so on battery).
caffeinate -dims -w $$ &
pmset -g batt 2>/dev/null | grep -q "AC Power" || echo "  ⚠ on battery: the ladder is hours long and the Mac may sleep or throttle — plug it in"

# Once a minute: the thermal and memory state, unprivileged (powermetrics needs root). A fanless
# Air throttles under sustained load; this is what says when.
( printf 'time\tcpu_speed_limit\tthermal_warning\tmem_free_pct\tbattery\n'
  while kill -0 $$ 2>/dev/null; do
    th=$(pmset -g therm 2>/dev/null)
    lim=$(echo "$th" | sed -n 's/.*CPU_Speed_Limit *= *\([0-9]*\).*/\1/p' | head -1)
    warn=$(echo "$th" | grep -qi 'No thermal warning' && echo none || echo "$th" | grep -i 'warning' | head -1 | tr '\t' ' ')
    mem=$(memory_pressure -Q 2>/dev/null | sed -n 's/.*percentage: *\([0-9]*\)%.*/\1/p')
    bat=$(pmset -g batt 2>/dev/null | grep -o '[0-9]*%' | head -1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${lim:--}" "${warn:--}" "${mem:--}" "${bat:--}"
    sleep 60
  done ) >> "$L/sensors.tsv" 2>/dev/null &

# The final-epoch accuracies in a trainer's log, " / "-joined (an ablation prints one per arm).
finals () {  # finals <log> <epochs>
  awk -v E="$2" '$0 ~ "^  epoch +" E ":" && match($0, /= [0-9.]+%/) {
      a[++n] = substr($0, RSTART + 2, RLENGTH - 2) }
    END { for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? " / " : ""), a[i] }' "$1"
}

# ─── build ───────────────────────────────────────────────────────────────────
if want build; then
  say "build: the ladder's eight binaries"
  if ! done_already build; then
    for t in mnist-linear-verified mnist-mlp-verified mnist-cnn-verified \
             cifar8w-ablation cifar8w-bn-ablation cifar8wb-bbn-ablation \
             cifar8-bbn-gradcheck cifar8wb-bbn-grid; do
      echo "  ▸ lake build $t"
      lake build "$t" > "$L/build-$t.out" 2>&1 || { tail -30 "$L/build-$t.out"; die "lake build $t failed (all of it: $L/build-$t.out)"; }
    done
    ok "built"; mark built
  fi
fi

# ─── data ────────────────────────────────────────────────────────────────────
if want data; then
  say "data: MNIST and CIFAR-10"
  bash scripts/datasets/download_mnist.sh > /dev/null || die "the MNIST download failed"
  bash scripts/datasets/download_cifar.sh > /dev/null || die "the CIFAR-10 download failed"
  ok "$(ls data/mnist/*-ubyte | wc -l | tr -d ' ') MNIST files, $(ls data/cifar-10/*.bin | wc -l | tr -d ' ') CIFAR-10 batches"
fi

# ─── suite ───────────────────────────────────────────────────────────────────
if want suite; then
  say "suite: scripts/platform/check.sh, tiers 0–2"
  if ! done_already suite; then
    scripts/platform/check.sh 2>&1 | tee "$L/suite.out"
    rc=${PIPESTATUS[0]}
    out=$(sed -n 's/^.* skip — \(runs\/platform\/[^ ]*\)$/\1/p' "$L/suite.out" | tail -1)
    # A FAIL is a finding to send back, not a reason to stop: the trainers may never touch the op.
    [ "$rc" = 0 ] && ok "suite green: $out" || echo "  ⚠ suite rc=$rc — $out/results.tsv names the failures; carrying on"
    mark "rc=$rc $out"
  fi
fi

# ─── mnist ───────────────────────────────────────────────────────────────────
if want mnist; then
  say "mnist: lake run mnist"
  if ! done_already mnist; then
    lake run mnist 2>&1 | tee "$L/mnist.out"
    [ "${PIPESTATUS[0]}" = 0 ] || die "lake run mnist failed (each trainer's log is under runs/$(date +%F)-mnist-*/)"
    mark ok
  fi
fi

# ─── gradcheck ───────────────────────────────────────────────────────────────
if want gradcheck; then
  say "gradcheck: the batch-BN cifar8 step's backward"
  if ! done_already gradcheck; then
    .lake/build/bin/cifar8-bbn-gradcheck 2>&1 | tee "$L/gradcheck.out"
    [ "${PIPESTATUS[0]}" = 0 ] && grep -q 'GRADCHECK PASSES' "$L/gradcheck.out" \
      || die "the batch-BN gradient check failed on Metal — that is the result to send back ($L/gradcheck.out)"
    mark pass
  fi
fi

# ─── cifar ───────────────────────────────────────────────────────────────────
if want cifar; then
  say "cifar: lake run cifar (no BN, per-image BN, batch BN; 40 epochs)"
  if ! done_already cifar; then
    lake run cifar 2>&1 | tee "$L/cifar.out"
    rc=${PIPESTATUS[0]}
    for t in cifar8w-ablation cifar8w-bn-ablation cifar8wb-bbn-ablation; do
      f=$(ls -t runs/*-"$t"/"$t".log 2>/dev/null | head -1)
      [ -n "$f" ] && echo "  $t: $(finals "$f" 40)   ($f)"
    done
    [ "$rc" = 0 ] || die "lake run cifar: a trainer failed (named above)"
    mark ok
  fi
fi

# ─── seeds ───────────────────────────────────────────────────────────────────
if want seeds; then
  D=$L/bbn-seeds
  say "seeds: cifar8wb-bbn-ablation at seeds $SEEDS, six arms × 40 epochs each → $D"
  mkdir -p "$D/done" "$D/bitmaps"
  for s in $SEEDS; do
    tag=cifar8wb_bbn_s$s
    [ -f "$D/done/$tag" ] && { ok "$tag already done"; continue; }
    echo "  ▸ $(date '+%T') $tag"
    t0=$SECONDS
    LEAN_MLIR_SEED=$s LEAN_MLIR_CKPT_TAG=s$s LEAN_MLIR_DUMP_CORRECT="$D/bitmaps/$tag" \
      .lake/build/bin/cifar8wb-bbn-ablation data > "$D/$tag.log" 2>&1
    rc=$?; el=$((SECONDS - t0))
    # rc alone is not enough: a finished checkpoint prints `done` with no epoch line (seed_sweep.sh)
    if [ $rc = 0 ] && grep -q '^done' "$D/$tag.log" && grep -qE 'epoch +40:.*(val_acc|test_acc)' "$D/$tag.log"; then
      echo "$tag ${el}s" > "$D/done/$tag"
      echo "    OK $((el / 60))m$((el % 60))s  $(finals "$D/$tag.log" 40)" | tee -a "$D/sweep.log"
    else
      echo "    FAIL rc=$rc after $((el / 60))m — $D/$tag.log" | tee -a "$D/sweep.log"
    fi
  done
  n=$(ls "$D/done" | wc -l | tr -d ' ')
  [ "$n" = "$(echo $SEEDS | wc -w | tr -d ' ')" ] || die "$n seeds done of [$SEEDS] — see $D/sweep.log; rerun to retry the rest"
  mark "$n seeds"
fi

# ─── grid ────────────────────────────────────────────────────────────────────
if want grid; then
  D=$L/cifar8wb-bbn-grid-xla
  say "grid: cifar8wb-bbn-grid at ten head widths × $GRID_EPOCHS epochs → $D"
  mkdir -p "$D/done"
  for d in $WIDTHS; do
    [ -f "$D/done/fc$d" ] && { ok "fc$d already done"; continue; }
    log=$D/cifar8wbbbn_fc$d.log
    rm -f .lake/build/cifar8wb_bbn_${d}_adam_ckpt*   # never resume an old width (and its .bn stats)
    echo "  ▸ $(date '+%T') fc$d"
    .lake/build/bin/cifar8wb-bbn-grid "$d" "$GRID_EPOCHS" data > "$log" 2>&1
    rc=$?
    last=$(awk '/test_acc = /{l=$0} END{print l}' "$log")
    acc=$(echo "$last"   | sed -nE 's/.*test_acc = [0-9]+\/[0-9]+ = ([0-9.]+)%.*/\1/p')
    n=$(echo "$last"     | sed -nE 's/.*test_acc = [0-9]+\/([0-9]+).*/\1/p')
    lo=$(echo "$last"    | sed -nE 's/.*95% CI ([0-9.]+)[^0-9.]+[0-9.]+\].*/\1/p')
    hi=$(echo "$last"    | sed -nE 's/.*95% CI [0-9.]+[^0-9.]+([0-9.]+)\].*/\1/p')
    floats=$(sed -nE 's/.*params, ([0-9]+) floats.*/\1/p' "$log" | head -1)
    msep=$(sed -nE 's/.*\(([0-9]+)ms\).*/\1/p' "$log" | tail -1)
    if [ $rc = 0 ] && [ -n "$acc" ]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$acc" "${n:-NA}" "${lo:-NA}" "${hi:-NA}" "${floats:-NA}" "${msep:-NA}" > "$D/done/fc$d"
      echo "    fc$d acc=$acc% n=${n:-NA} CI=[${lo:-NA},${hi:-NA}]"
    else
      echo "    fc$d FAIL rc=$rc — $log"
    fi
  done
  { printf 'd\tacc\tn_eval\tci_lo\tci_hi\tfloats\tms_per_ep\n'; cat "$D"/done/fc* 2>/dev/null | sort -n; } > "$D/results.tsv"
  cat "$D/results.tsv"
  n=$(ls "$D/done" | wc -l | tr -d ' ')
  [ "$n" = 10 ] || die "$n of 10 widths done — rerun to retry the rest"
  mark "10 widths"
fi

say "done: $STAGES — send back $L/ and the runs/<date>-{mnist,cifar8w}*/ trainer logs it names"
