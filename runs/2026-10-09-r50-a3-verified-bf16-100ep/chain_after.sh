#!/usr/bin/env bash
# Queued 2026-10-08 on Brett's word ("queue the pjrt runs after"): launch the R50 A3 verified rerun (100 ep) once r502018-jax has ended
# with ✅ COMPLETE and the GPUs are idle. Launches NOTHING otherwise — writes CHAIN_EVENT and exits.
# Template: runs/2026-09-26-mnv4-verified-bf16-100ep/chain_after_jax.sh, plus the PATH the
# verified precheck's pc_exe needs and a wait on the previous chain unit (r502018-chain) first.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
RD=runs/2026-10-09-r50-a3-verified-bf16-100ep
JRD=runs/2026-10-07-r50-2018-jax-bf16-90ep
while systemctl --user is-active -q r502018-chain; do sleep 60; done
while systemctl --user is-active -q r502018-jax; do sleep 60; done
if ! grep -q '✅ COMPLETE' "$JRD/master.log" 2>/dev/null; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — r502018-jax ended without ✅ COMPLETE:"; tail -5 "$JRD/master.log"; } > "$RD/CHAIN_EVENT"; exit 0; fi
sleep 30; for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 10; done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — GPUs still busy 10 min after r502018-jax ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RD/CHAIN_EVENT"; exit 0; fi
echo "▶ launching $(date -u '+%F %T') UTC" > "$RD/CHAIN_EVENT"
systemd-run --user --unit=r50a3-ver-log-ts --working-directory="$PWD" bash $RD/log_ts.sh
systemd-run --user --unit=r50a3-ver --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=RUNDIR=$RD --setenv=LEAN_MLIR_DUMP_CORRECT=$RD/bitmaps/a3 \
  scripts/supervise.sh r50-a3-wxclip4x128-bf16-4gpu
sleep 10
systemctl --user is-active -q r50a3-ver || { echo "⛔ r50a3-ver not active 10 s after launch" >> "$RD/CHAIN_EVENT"; exit 0; }
for w in epoch_clock loader_rss fault_watch edac_watch; do
  systemd-run --user --unit=r50a3-ver-${w//_/-} --working-directory="$PWD" bash $RD/$w.sh; done
echo "✅ launched r50a3-ver + 5 watchers" >> "$RD/CHAIN_EVENT"
