#!/usr/bin/env bash
# Queued 2026-10-07 on Brett's word ("run the 3x jax runs"): launch the R50 2018 JAX reference (90 ep) once r50a3-jax has ended with
# ✅ COMPLETE and the GPUs are idle. Launches NOTHING otherwise — writes CHAIN_EVENT and exits.
# (Template: runs/2026-09-26-mnv4-verified-bf16-100ep/chain_after_jax.sh.)
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
RD=runs/2026-10-07-r50-2018-jax-bf16-90ep
JRD=runs/2026-10-07-r50-a3-jax-bf16-100ep
# Wait for the A3 chain unit (r50a3-chain) to hand off first: r50a3-jax is not active until it does.
while systemctl --user is-active -q r50a3-chain; do sleep 60; done
while systemctl --user is-active -q r50a3-jax; do sleep 60; done
if ! grep -q '✅ COMPLETE' "$JRD/master.log"; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — r50a3-jax ended without ✅ COMPLETE:"; tail -5 "$JRD/master.log"; } > "$RD/CHAIN_EVENT"; exit 0; fi
sleep 30; for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 10; done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — GPUs still busy 10 min after r50a3-jax ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RD/CHAIN_EVENT"; exit 0; fi
echo "▶ launching $(date -u '+%F %T') UTC" > "$RD/CHAIN_EVENT"
systemd-run --user --unit=r502018-jax --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=RUNDIR=$RD scripts/supervise.sh r50-2018-jax-4gpu
sleep 10
systemctl --user is-active -q r502018-jax || { echo "⛔ r502018-jax not active 10 s after launch" >> "$RD/CHAIN_EVENT"; exit 0; }
for w in epoch_clock edac_watch; do
  systemd-run --user --unit=r502018-jax-${w//_/-} --working-directory="$PWD" bash $RD/$w.sh; done
echo "✅ launched r502018-jax + 2 watchers" >> "$RD/CHAIN_EVENT"
