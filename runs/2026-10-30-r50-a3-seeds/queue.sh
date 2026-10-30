#!/usr/bin/env bash
# R50 A3, seeds per arm — planning/runs_queue_2026-10-29.md Tier 2. Brett, 2026-10-30: "do the A3
# seeds after [Tier 1] / can you prep/queue that? … i don't want to run a2 if there's an issue / we
# should chase that first". So: after tier1-sweeps, before A2 resumes.
#
# Four runs, one at a time on the four cards, each a fresh lineage (RUN_SEED in both confs):
#   jax s2 (~15 h) → verified s2 (~21 h) → jax s3 (~15 h) → verified s3 (~21 h)   ≈ 72 h
# Seed 1 of each arm is the landed pair (runs/2026-10-12-r50-a3-{jax,rsb-bf16}, on ares).
#
# Launches NOTHING unless the step before it ended ✅ COMPLETE (tier1-sweeps: imagenette 35/35);
# every decision is a line in QUEUE_EVENT.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
Q=runs/2026-10-30-r50-a3-seeds
ev() { echo "$(date -u '+%F %T') UTC  $*" >> "$Q/QUEUE_EVENT"; }

gpus_idle() {   # up to 10 min for the previous job's processes to leave the cards
  for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || return 0; sleep 10; done
  return 1
}

ev "▶ queue armed; waiting for tier1-sweeps"
while systemctl --user is-active -q tier1-sweeps; do sleep 120; done
got=$(ls runs/2026-10-29-imagenette-seeds/done 2>/dev/null | wc -l)
[ "$got" -eq 35 ] || { ev "⛔ NOT LAUNCHED — tier1-sweeps ended with imagenette $got/35 done"; exit 0; }
ev "✅ tier1-sweeps done (imagenette 35/35)"

prev=""
for step in "jax 2" "ver 2" "jax 3" "ver 3"; do
  set -- $step; arm=$1; seed=$2
  if [ "$arm" = jax ]; then job=r50-a3-jax-4gpu; else job=r50-a3-wxclip4x128-bf16-4gpu; fi
  unit=r50a3-$arm-s$seed
  rd=runs/2026-10-30-r50-a3-$arm-s$seed
  if [ -n "$prev" ]; then
    while systemctl --user is-active -q "r50a3-${prev// /-s}"; do sleep 60; done
    prd=runs/2026-10-30-r50-a3-${prev// /-s}
    grep -q '✅ COMPLETE' "$prd/master.log" 2>/dev/null || {
      ev "⛔ STOPPED — r50a3-${prev// /-s} ended without ✅ COMPLETE:"; tail -5 "$prd/master.log" >> "$Q/QUEUE_EVENT"; exit 0; }
    ev "✅ r50a3-${prev// /-s} COMPLETE"
  fi
  gpus_idle || { ev "⛔ STOPPED — GPUs still busy 10 min before $unit"; exit 0; }
  mkdir -p "$rd"
  # ⛔ --setenv=PATH: the confs' prechecks run `lake build` / find the venv.
  systemd-run --user --unit="$unit" --working-directory="$PWD" --setenv=PATH="$PATH" \
    --setenv=RUNDIR="$rd" --setenv=RUN_SEED="$seed" \
    scripts/supervise.sh "$job"
  sleep 20
  systemctl --user is-active -q "$unit" || { ev "⛔ $unit not active 20 s after launch:"; tail -5 "$rd/master.log" >> "$Q/QUEUE_EVENT"; exit 0; }
  ev "▶ launched $unit ($job, RUN_SEED=$seed, RUNDIR=$rd)"
  prev="$arm $seed"
done
while systemctl --user is-active -q "r50a3-${prev// /-s}"; do sleep 60; done
grep -q '✅ COMPLETE' "runs/2026-10-30-r50-a3-${prev// /-s}/master.log" 2>/dev/null \
  && ev "✅ r50a3-${prev// /-s} COMPLETE — queue finished; A2 may resume (paused at e77)" \
  || ev "⛔ r50a3-${prev// /-s} ended without ✅ COMPLETE"
