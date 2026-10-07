#!/usr/bin/env bash
# Chapter 5's ResNet reruns (planning/imagenet_parity.md §9.5; memory r50-reruns-ready-2026-10-06),
# in sequence on the 3060 box (mars), each through its own job conf — the whole ResNet milestone
# on one queue so ares stays free:
#
#   r34-default-jax-4gpu              90 ep  ~13.3 h on 4x 4060 Ti   JAX; fp32 head (X3): the landed 74.16 scored through a bf16 head
#   r50-2018-jax-4gpu                 90 ep  ~22.5 h                  JAX; + zero-γ, BN decay 0.9 (the landed 76.95 trained 0.99)
#   r50-a3-jax-4gpu                  100 ep  ~13.5 h                  JAX; + the BCE target threshold 0.2 (D14); RSB-A3 `rsb-faithful`
#   r50-2018-bf16-4gpu                90 ep  ~29 h                    verified peer; zero-γ (X4), CKPT_TAG 2610
#   r50-a3-wxclip4x128-bf16-4gpu     100 ep  ~22 h                    verified peer; zero-γ + the threshold via the `short` shim, CKPT_TAG 2610
#
# The JAX three go first (they are the ones the book's §5.7/§5.8 tables wait on); the verified
# two follow because the landed verified runs already have fp32 heads and 0.9, so only zero-γ and
# the threshold move them. All five carry torchvision's RandomResizedCrop (aa974a95). None has
# run on 3060s: the hours above are ares' numbers (~100 h in all), so expect more here.
#
# On mars, from the checkout, after `git pull` and `scripts/regen_jax_generated.sh sync` (the JAX
# trainers and the verified shims run from jax/.lake/build, which is gitignored), and
# `lake build resnet50-imagenet-verified` (the two verified prechecks refuse a binary older than
# the driver — zero-γ is host-side):
#
#     for j in r34-default-jax-4gpu r50-2018-jax-4gpu r50-a3-jax-4gpu r50-2018-bf16-4gpu r50-a3-wxclip4x128-bf16-4gpu; do
#       DRY_RUN=1 scripts/supervise.sh $j || break; done      # every plan must end "DRY_RUN — nothing launched"
#     systemd-run --user --unit=resnet-queue --working-directory="$PWD" --setenv=PATH="$PATH" \
#       bash runs/2026-10-07-resnet-jax-mars/queue.sh
#     journalctl --user -u resnet-queue -f                     # or tail runs/2026-10-07-resnet-jax-mars/queue.log
#
# Each job is a `scripts/supervise.sh` run with its own RUNDIR (runs/2026-10-07-<job>/master.log is
# the record), and the queue goes on to the next job only on `✅ COMPLETE`; anything else stops
# the queue and writes QUEUE_EVENT, so a failed run is never followed by a fresh one on a busy box.
# JAX checkpoints go to the fresh `/home/skoonce/{r34_2018,r50_2018,r50_a3}_jax_2610/` dirs the confs
# name (`pc_jax_trainer` creates them); the verified ones to `.lake/build/resnet50in*_ckpt_xla_2610.bin`
# — the tag keeps them off the landed 09-23 / 09-24 runs' checkpoints, which sit at their last epoch
# in this checkout and would otherwise make each job ✅ COMPLETE at once. A re-launch of this script
# resumes each job from its own checkpoint, so a box reset costs at most one epoch per job.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=runs/2026-10-07-resnet-jax-mars
LOG=$D/queue.log
JOBS=(r34-default-jax-4gpu r50-2018-jax-4gpu r50-a3-jax-4gpu r50-2018-bf16-4gpu r50-a3-wxclip4x128-bf16-4gpu)
say() { echo "[queue] $(date -u '+%F %T') $*" | tee -a "$LOG"; }

gpus_idle() {  # no compute process on any card, polled up to 10 min
  for _ in $(seq 60); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || return 0; sleep 10
  done; return 1
}

# Per-epoch clock for a job: one row per completed epoch, by the conf's own `epoch_now` (JAX: the
# newest `<base>_e<N>.state.npz`, of which the trainer keeps only 3; verified: the `.epoch` file),
# plus host memory. Runs while $2 (the supervisor PID) lives.
job_epoch() { ( . "scripts/jobs/$1.conf" >/dev/null 2>&1; epoch_now ) 2>/dev/null; }
epoch_clock() {  # $1 = job, $2 = supervisor pid
  local out n last
  out="runs/2026-10-07-$1/epoch_clock.tsv"
  [ -s "$out" ] || printf 'epoch\tunix\tutc\tmem_available_kb\n' > "$out"
  last="$(tail -1 "$out" | cut -f1)"
  while kill -0 "$2" 2>/dev/null; do
    n="$(job_epoch "$1")"; n="${n:-0}"
    if [ "$n" -gt 0 ] 2>/dev/null && [ "$n" != "$last" ]; then
      printf '%s\t%s\t%s\t%s\n' "$n" "$(date +%s)" "$(date -u '+%F %T')" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$out"
      last=$n
    fi
    sleep 30
  done
}

say "START on $(hostname) at $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)'); jobs: ${JOBS[*]}"
for j in "${JOBS[@]}"; do
  rd="runs/2026-10-07-$j"; mkdir -p "$rd"
  if grep -q '✅ COMPLETE' "$rd/master.log" 2>/dev/null; then say "skip $j — already ✅ COMPLETE ($rd/master.log)"; continue; fi
  gpus_idle || { say "⛔ GPUs busy 10 min before $j — stopping the queue"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tee -a "$LOG"; echo "⛔ GPUs busy before $j $(date -u '+%F %T')" > "$D/QUEUE_EVENT"; exit 1; }
  say "▶ $j → $rd"
  RUNDIR="$rd" scripts/supervise.sh "$j" & sp=$!
  epoch_clock "$j" "$sp" & ec=$!
  wait "$sp"; rc=$?; kill "$ec" 2>/dev/null; wait "$ec" 2>/dev/null
  if grep -q '✅ COMPLETE' "$rd/master.log" 2>/dev/null; then say "✅ $j COMPLETE (exit $rc)"
  else say "⛔ $j ended without ✅ COMPLETE (exit $rc) — stopping the queue"; tail -5 "$rd/master.log" 2>/dev/null | tee -a "$LOG"
       echo "⛔ $j ended without ✅ COMPLETE (exit $rc) $(date -u '+%F %T')" > "$D/QUEUE_EVENT"; exit 1; fi
done
say "✅ queue done — all of: ${JOBS[*]}"
