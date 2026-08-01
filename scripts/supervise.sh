#!/usr/bin/env bash
# supervise.sh — ONE supervisor engine for long training runs on this box.
#
# Replaces the copy-paste family in jax/scripts/supervise_*.sh (10+ near-identical
# scripts differing only in device list, checkpoint path, epoch count and rest schedule).
# A job is a small config file in scripts/jobs/<name>.conf; this file is the engine and
# should not need editing to add a job.
#
#     scripts/supervise.sh <job>            # scripts/jobs/<job>.conf
#     DRY_RUN=1 scripts/supervise.sh <job>  # print the plan and exit, run nothing
#
# WHY A SUPERVISOR AT ALL, on this box specifically:
#   * PCIe AER. ares logs BadTLP-under-load; one 80-epoch ViT run took 5 hits. A run
#     without a watchdog dies on the first one, hours in, silently.
#   * Heat. The box runs hot without fans, so long runs need a duty cycle.
#   * Neither is a correctness problem, so neither belongs in the trainer.
#
# WHAT A JOB CONFIG MUST SET (see scripts/jobs/*.conf for worked examples):
#   CMD          — the command line to run, as a bash array
#   DEVS         — CUDA_VISIBLE_DEVICES value
#   EPOCHS       — total epochs; the run is COMPLETE when last_epoch >= EPOCHS
#   epoch_now()  — echo the last COMPLETED epoch (0 if none). This is the one piece
#                  that genuinely differs between paths, so it is a function:
#                    verified path -> cat the trainer's <ckpt>.epoch file
#                    jax path      -> newest <base>_e<N>.state.npz
# OPTIONAL:
#   ENV_EXTRA    — array of KEY=VAL passed to the run
#   REST_EPOCHS  — space-separated epochs to rest after (default: none)
#   REST_SECS    — cooldown length (default 1800)
#   TEMP_MAX     — °C; rest when any watched GPU exceeds this (default 0 = disabled)
#   TEMP_RESUME  — °C to cool back to before resuming (default TEMP_MAX-12)
#   STALL_SECS   — kill+restart if the log is silent this long (default 1800, 0 = off)
#   MAX_ATTEMPTS — default 60
#   PRECHECK     — a function; non-zero exit aborts before the first launch
set -u
cd "$(dirname "$0")/.." || exit 1

JOB="${1:-}"
[ -n "$JOB" ] || { echo "usage: $0 <job>   (configs in scripts/jobs/)"; exit 2; }
CONF="scripts/jobs/${JOB}.conf"
[ -f "$CONF" ] || { echo "no such job config: $CONF"; ls scripts/jobs/ 2>/dev/null; exit 2; }

# ── defaults, then the job overrides them ──────────────────────────────────────
REST_EPOCHS=""; REST_SECS=1800; TEMP_MAX=0; TEMP_RESUME=""; STALL_SECS=1800
MAX_ATTEMPTS=60; ENV_EXTRA=(); PRECHECK=""
# shellcheck disable=SC1090
. "$CONF"
: "${TEMP_RESUME:=$(( TEMP_MAX > 12 ? TEMP_MAX - 12 : TEMP_MAX ))}"

RUNDIR="${RUNDIR:-/tmp/supervise_${JOB}}"
mkdir -p "$RUNDIR"
RUNLOG="$RUNDIR/attempt.log"      # current attempt only (stall + done detection)
MASTER="$RUNDIR/master.log"       # supervisor's own narration, survives attempts
FULLLOG="$RUNDIR/full.log"        # every attempt concatenated

say() { echo "[sup] $(date '+%F %T') $*" | tee -a "$MASTER"; }

# Kill the run as a process TREE, by its own process group.
#
# ⚠ This replaced `pkill -9 -f "${CMD[0]}"`, which was actively dangerous: CMD[0] is
# whatever the job config says, so a job whose command starts with a common interpreter
# (`bash -c ...`, `python ...`) made this a pattern-kill of EVERY such process on the box.
# It killed the operator's tmux the first time the self-test ran. The JAX originals were
# safe only by accident — each hardcoded one unique `.py` filename. A generic engine must
# never pattern-match on process names.
kill_run() {
  local sig="${2:--9}"
  if [ -n "${PGID:-}" ] && [ "$PGID" != "$$" ] && [ "$PGID" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
    kill "$sig" -- "-$PGID" 2>/dev/null
  fi
  [ -n "${1:-}" ] && kill "$sig" "$1" 2>/dev/null
  return 0
}


# Hottest of the watched GPUs. Empty if nvidia-smi is unavailable, which disables
# thermal resting rather than pretending the box is cold.
hottest() {
  nvidia-smi -i "$DEVS" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
    | sort -rn | head -1
}

# Kernel-log AER scan since the attempt started. `no action required` lines are the
# benign corrected-error chatter and must not trigger a restart.
aer_since() {
  journalctl -k --since "$1" 2>/dev/null \
    | grep -iE "BadTLP|AER:|Uncorrected|Fatal" | grep -qivE "no action required"
}

if [ -n "$PRECHECK" ]; then
  if ! $PRECHECK; then say "⛔ PRECHECK failed — refusing to start"; exit 1; fi
fi

say "START job=$JOB devs=$DEVS epochs=$EPOCHS rest_after=[${REST_EPOCHS:-none}] \
temp_max=${TEMP_MAX:-off} stall=${STALL_SECS}s logs=$RUNDIR"
say "cmd: ${CMD[*]}"
if [ "${DRY_RUN:-0}" != "0" ]; then say "DRY_RUN — nothing launched"; exit 0; fi

# Take the run down with us. `setsid` gives the run its own SESSION, which is what makes
# the process-group kill precise — but it also means the run does NOT die when this script
# does. Measured: kill the supervisor and the trainer keeps going, still holding ~12 GB on
# all four GPUs, where it would then fight the next attempt for memory. So Ctrl-C, SIGTERM
# and any exit path must explicitly take the group with them.
cleanup() {
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    say "supervisor exiting — taking the run down (PID=$PID PGID=${PGID:-?})"
    kill_run "$PID"
  fi
}
trap cleanup EXIT
# Disarm EXIT first, or `exit` here re-enters `cleanup` and logs the takedown twice.
trap 'trap - EXIT; cleanup; exit 130' INT TERM

next_rest() {  # first rest epoch strictly ahead of $1; 999 = none left
  for e in $REST_EPOCHS; do [ "$1" -lt "$e" ] && { echo "$e"; return; }; done
  echo 999
}

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))
  EP="$(epoch_now)"; EP="${EP:-0}"

  if [ "$EP" -ge "$EPOCHS" ]; then
    say "✅ COMPLETE — $EP/$EPOCHS epochs"; exit 0
  fi

  REST_AT="$(next_rest "$EP")"
  say "attempt $attempt: resuming at epoch $EP/$EPOCHS (next rest: $REST_AT)"

  : > "$RUNLOG"
  echo "===== $(date '+%F %T') attempt $attempt (from epoch $EP) =====" >> "$FULLLOG"
  START="$(date '+%Y-%m-%d %H:%M:%S')"

  # `setsid` puts the run in its OWN process group so it can be killed as a tree —
  # trainer, the tee subshell, and grandchildren like the tfds shim's python — without
  # pattern-matching on the command name. See `kill_run`.
  setsid env CUDA_VISIBLE_DEVICES="$DEVS" "${ENV_EXTRA[@]}" \
      "${CMD[@]}" > >(tee -a "$FULLLOG" > "$RUNLOG") 2>&1 &
  PID=$!
  PGID="$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')"
  say "  launched PID=$PID PGID=${PGID:-?}"

  result="unknown"
  while kill -0 "$PID" 2>/dev/null; do
    if aer_since "$START"; then
      say "  !!! PCIe AER — killing PID=$PID"; kill_run "$PID"; sleep 2
      result="aer"; break
    fi
    NOW="$(epoch_now)"; NOW="${NOW:-0}"
    if [ "$NOW" -ge "$EPOCHS" ]; then result="done"; kill_run "$PID"; break; fi
    if [ "$REST_AT" -ne 999 ] && [ "$NOW" -ge "$REST_AT" ]; then
      say "  💤 epoch $NOW reached — planned cooldown ${REST_SECS}s"
      kill_run "$PID"; sleep 2; result="rest"; break
    fi
    if [ "$TEMP_MAX" -gt 0 ]; then
      T="$(hottest)"
      if [ -n "$T" ] && [ "$T" -ge "$TEMP_MAX" ]; then
        say "  🌡 ${T}°C ≥ ${TEMP_MAX}°C — thermal cooldown"
        kill_run "$PID"; sleep 2; result="hot"; break
      fi
    fi
    # Stall guard: a hung process stays alive forever, so MAX_ATTEMPTS never fires
    # and the run silently stops progressing. The JAX supervisors have no such check.
    if [ "$STALL_SECS" -gt 0 ] && [ -s "$RUNLOG" ]; then
      AGE=$(( $(date +%s) - $(stat -c %Y "$RUNLOG") ))
      if [ "$AGE" -ge "$STALL_SECS" ]; then
        say "  ⏳ no output for ${AGE}s — assuming hung, killing PID=$PID"
        kill_run "$PID"; sleep 2; result="stall"; break
      fi
    fi
    sleep 10
  done
  wait "$PID" 2>/dev/null
  [ "$result" = "unknown" ] && result="exited"

  tail -1 "$RUNLOG" 2>/dev/null | sed 's/^/[sup]   last line: /' | tee -a "$MASTER" >/dev/null
  kill_run "$PID"   # ensure nothing survives into the next attempt

  EP2="$(epoch_now)"; EP2="${EP2:-0}"
  if [ "$EP2" -ge "$EPOCHS" ]; then say "✅ COMPLETE — $EP2/$EPOCHS epochs"; exit 0; fi

  case "$result" in
    rest) sleep "$REST_SECS"; say "  cooldown over" ;;
    hot)
      say "  waiting for ≤${TEMP_RESUME}°C…"
      for _ in $(seq 1 240); do
        T="$(hottest)"; [ -z "$T" ] && break
        [ "$T" -le "$TEMP_RESUME" ] && break
        sleep 15
      done
      say "  cooled to $(hottest)°C" ;;
    *) say "  ended ($result) at epoch $EP2; 15s then retry"; sleep 15 ;;
  esac
done

say "⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS at epoch $(epoch_now) — giving up"
exit 1
