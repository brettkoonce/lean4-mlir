#!/usr/bin/env bash
# Queued 2026-09-26 on Brett's word ("do it after"): launch the MNv4 verified 100-ep run once the
# MNv4 JAX 100-ep run (unit mnv4-jax) has ended with ✅ COMPLETE and the GPUs are idle. Launches
# NOTHING if the JAX run ended any other way — it writes CHAIN_EVENT and exits.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
RD=runs/2026-09-26-mnv4-verified-bf16-100ep
JRD=runs/2026-09-26-mnv4-jax-bf16-100ep
while systemctl --user is-active -q mnv4-jax; do sleep 60; done
if ! grep -q '✅ COMPLETE' "$JRD/master.log"; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — mnv4-jax ended without ✅ COMPLETE:"; tail -5 "$JRD/master.log"; } > "$RD/CHAIN_EVENT"; exit 0; fi
sleep 30; for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 10; done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — GPUs still busy 10 min after mnv4-jax ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RD/CHAIN_EVENT"; exit 0; fi
echo "▶ launching $(date -u '+%F %T') UTC" > "$RD/CHAIN_EVENT"
systemd-run --user --unit=mnv4v-log-ts --working-directory="$PWD" bash $RD/log_ts.sh
systemd-run --user --unit=mnv4-verified --working-directory="$PWD" \
  --setenv=RUNDIR=$RD --setenv=LEAN_MLIR_DUMP_CORRECT=$RD/bitmaps/mnv4 \
  scripts/supervise.sh mnv4-default-4gpu
sleep 10
systemctl --user is-active -q mnv4-verified || { echo "⛔ mnv4-verified not active 10 s after launch" >> "$RD/CHAIN_EVENT"; exit 0; }
for w in epoch_clock loader_rss fault_watch edac_watch; do
  systemd-run --user --unit=mnv4v-${w//_/-} --working-directory="$PWD" bash $RD/$w.sh; done
echo "✅ launched mnv4-verified + 5 watchers" >> "$RD/CHAIN_EVENT"
