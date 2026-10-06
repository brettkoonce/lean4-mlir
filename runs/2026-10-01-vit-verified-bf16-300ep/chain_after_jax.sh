#!/usr/bin/env bash
# ▶ ViT-Ti 300ep copy (2026-10-01) of the MNv2 run, itself a copy of the MNv4 chain; Brett, 2026-09-27: "great work start the run" (2026-10-01).
# Launch the ViT-Ti verified 300-ep run once the ViT-Ti JAX 300-ep run (unit vit-jax) has ended with ✅ COMPLETE and the GPUs are idle. Launches
# NOTHING if the JAX run ended any other way — it writes CHAIN_EVENT and exits.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
RD=runs/2026-10-01-vit-verified-bf16-300ep
JRD=runs/2026-10-01-vit-jax-bf16-300ep
while systemctl --user is-active -q vit-jax; do sleep 60; done
if ! grep -q '✅ COMPLETE' "$JRD/master.log"; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — vit-jax ended without ✅ COMPLETE:"; tail -5 "$JRD/master.log"; } > "$RD/CHAIN_EVENT"; exit 0; fi
sleep 30; for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 10; done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — GPUs still busy 10 min after vit-jax ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RD/CHAIN_EVENT"; exit 0; fi
echo "▶ launching $(date -u '+%F %T') UTC" > "$RD/CHAIN_EVENT"
systemd-run --user --unit=vitv-log-ts --working-directory="$PWD" bash $RD/log_ts.sh
# ⛔ --setenv=PATH: the conf's precheck runs `lake build` (pc_exe); the MNv4 chain lacked it.
systemd-run --user --unit=vit-verified --working-directory="$PWD" --setenv=PATH="$PATH" \
  --setenv=RUNDIR=$RD --setenv=LEAN_MLIR_DUMP_CORRECT=$RD/bitmaps/vit \
  scripts/supervise.sh vit-default-emabf16-4gpu
sleep 10
systemctl --user is-active -q vit-verified || { echo "⛔ vit-verified not active 10 s after launch" >> "$RD/CHAIN_EVENT"; exit 0; }
for w in epoch_clock loader_rss fault_watch edac_watch; do
  systemd-run --user --unit=vitv-${w//_/-} --working-directory="$PWD" bash $RD/$w.sh; done
echo "✅ launched vit-verified + 5 watchers" >> "$RD/CHAIN_EVENT"
