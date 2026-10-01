#!/usr/bin/env bash
# ▶ MNv2 350ep copy (2026-09-27) of the MNv4 chain; Brett, 2026-09-27: "yeah run it please".
# Launch the MNv2 verified 350-ep run once the MNv2 JAX 350-ep run (unit mnv2-jax) has ended with ✅ COMPLETE and the GPUs are idle. Launches
# NOTHING if the JAX run ended any other way — it writes CHAIN_EVENT and exits.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
RD=runs/2026-09-27-mnv2-verified-bf16-350ep
JRD=runs/2026-09-27-mnv2-jax-bf16-350ep
while systemctl --user is-active -q mnv2-jax; do sleep 60; done
if ! grep -q '✅ COMPLETE' "$JRD/master.log"; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — mnv2-jax ended without ✅ COMPLETE:"; tail -5 "$JRD/master.log"; } > "$RD/CHAIN_EVENT"; exit 0; fi
sleep 30; for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 10; done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — GPUs still busy 10 min after mnv2-jax ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RD/CHAIN_EVENT"; exit 0; fi
echo "▶ launching $(date -u '+%F %T') UTC" > "$RD/CHAIN_EVENT"
systemd-run --user --unit=mnv2v-log-ts --working-directory="$PWD" bash $RD/log_ts.sh
# ⛔ --setenv=PATH: the conf's precheck runs `lake build` (pc_exe); the MNv4 chain lacked it.
systemd-run --user --unit=mnv2-verified --working-directory="$PWD" --setenv=PATH="$PATH" \
  --setenv=RUNDIR=$RD --setenv=LEAN_MLIR_DUMP_CORRECT=$RD/bitmaps/mnv2 \
  scripts/supervise.sh mnv2-default-4gpu
sleep 10
systemctl --user is-active -q mnv2-verified || { echo "⛔ mnv2-verified not active 10 s after launch" >> "$RD/CHAIN_EVENT"; exit 0; }
for w in epoch_clock loader_rss fault_watch edac_watch; do
  systemd-run --user --unit=mnv2v-${w//_/-} --working-directory="$PWD" bash $RD/$w.sh; done
echo "✅ launched mnv2-verified + 5 watchers" >> "$RD/CHAIN_EVENT"
