#!/usr/bin/env bash
# queue_r50_a3_pair.sh — run the RSB-A3 pair after the in-flight 2018 run, in order.
#
#     setsid nohup scripts/queue_r50_a3_pair.sh > /tmp/queue_r50_a3_pair.out 2>&1 &
#
#   leg 0 (already running)  R50 `2018`, JAX, 90 ep        — waits for it, launches nothing
#   leg 1                    R50 RSB-A3, JAX reference     — ~17 h
#   leg 2                    R50 RSB-A3, VERIFIED wx+clip  — ~31–34 h
#
# ⭐ WHY THE ORDER IS NOT NEGOTIABLE. `planning/imagenet_rerun_sweep.md` §5: "Do not start (2)
# before (3)'s reference is scheduled. The whole value of the verified number is that it sits
# beside a reference measured under the same rules." The 77.43% verified result has no live peer —
# the 77.22% JAX reference it was quoted against predates C2/C3/C4/C6 and re-scores to 74.62%
# (§6), which is not a comparison anyone can use. Leg 1 is what makes leg 2 quotable.
#
# ⚠ ONLY ADVANCES ON CLEAN COMPLETION. Each leg is gated on its supervisor printing its own
# completion marker. If a leg instead exhausts MAX_ATTEMPTS it never prints one, so this waits
# forever rather than launching the next leg onto a sick box — a stalled queue is recoverable, a
# 34 h run started into a thermal fault is not.
#
# ⚠⚠ THE TWO SUPERVISORS PRINT DIFFERENT MARKERS, and grepping the wrong one is a silent hang:
#     jax/scripts/supervise_r50_a3_100ep.sh  ->  "TRAINING COMPLETE"
#     scripts/supervise.sh (the engine)      ->  "COMPLETE — N/M epochs"
# `wait_for` takes the marker per leg for exactly that reason.
#
# ⚠ Both legs want the same four GPUs, so this is strictly sequential. There is no concurrency to
# recover here: the box has six cards and idx 1 and 5 throw BadTLP under load.
set -u
cd "$(dirname "$0")/.." || exit 1

SETTLE="${SETTLE:-60}"          # seconds between legs, for GPU memory release
POLL="${POLL:-60}"

say() { echo "[queue] $(date '+%F %T') $*"; }

# Wait for $2 to appear in $1. Refuses to treat a MISSING log as "not yet done" forever without
# saying so — a typo'd path would otherwise look identical to a long-running job.
wait_for() {
  local log="$1" marker="$2" name="$3" waited=0
  say "waiting for $name: '$marker' in $log"
  until grep -q "$marker" "$log" 2>/dev/null; do
    if [ ! -f "$log" ] && [ $(( waited % 1800 )) -eq 0 ]; then
      say "  note: $log does not exist yet (waited ${waited}s)"
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
  say "$name complete after ${waited}s."
}

# ── leg 0: the in-flight 2018 run ─────────────────────────────────────────────────────────────
wait_for /tmp/r50_2018_90ep_master.log "TRAINING COMPLETE" "leg 0 (R50 2018, JAX, 90 ep)"
sleep "$SETTLE"

# ── leg 1: the RSB-A3 JAX reference ───────────────────────────────────────────────────────────
# Reuses the supervisor currently driving the 2018 run — full-state lossless resume, proven on
# this exact box tonight. The generic engine (scripts/supervise.sh) is the sanctioned path for
# the VERIFIED leg, but its jax-path `epoch_now()` branch is exercised by no existing job config,
# and a 52 h chain is the wrong place to debut it.
#
# ⚠ `rsb-faithful`, NOT `short`: the 77.22% reference is the accumulate-to-2048 recipe, and
# `short` is the bs512-starved one that produced 40.8%.
say "launching leg 1 — R50 RSB-A3, JAX reference (100 ep, ~17 h)"
mkdir -p /home/skoonce/r50_a3_jax_100ep
( cd jax && PY=.lake/build/generated_resnet50_imagenet_rsbfaithful.py \
    TAG=r50_a3_jax_100ep \
    CKPT_BASE=/home/skoonce/r50_a3_jax_100ep/r50_a3_jax \
    VENV_PY=/home/skoonce/lean/klawd_max_power/lean4-jax-mlir/.venv/bin/python \
    CKPT_EVERY=5 COOLDOWN_AT="25 50 75" COOLDOWN_SECS=1800 CUDA_DEVS=0,2,3,4 \
    setsid nohup bash scripts/supervise_r50_a3_100ep.sh \
      > /home/skoonce/r50_a3_jax_100ep/supervisor.out 2>&1 & )
sleep 10
wait_for /tmp/r50_a3_jax_100ep_master.log "TRAINING COMPLETE" "leg 1 (R50 A3, JAX reference)"
sleep "$SETTLE"

# ── leg 2: the VERIFIED wx+clip artifact ──────────────────────────────────────────────────────
# ⚠ Regenerate the shims first. They are build products, gitignored, and went stale twice in the
# week of 2026-08-14 — once silently. The job's PRECHECK checks the shim EXISTS, which a stale one
# also does; this is the half that check cannot make.
say "regenerating shims before leg 2"
scripts/gen_shims.sh || { say "⛔ gen_shims.sh failed — refusing to start leg 2"; exit 1; }

say "launching leg 2 — R50 RSB-A3, VERIFIED wx+clip (100 ep, ~31–34 h)"
setsid nohup bash scripts/supervise.sh r50-a3-wxclip-4gpu \
  > /tmp/supervise_r50-a3-wxclip-4gpu.out 2>&1 &
sleep 10
wait_for /tmp/supervise_r50-a3-wxclip-4gpu/master.log "COMPLETE —" "leg 2 (R50 A3, verified)"

say "✅ queue drained — both A3 legs complete."
say "next: score both, then update planning/imagenet_rerun_sweep.md §5 items 2 and 3."
