#!/usr/bin/env bash
# One-shot feed refresh (2026-09-14): restart the run right after the next epoch checkpoint so the
# shim producers respawn, and pick up the conf's new REST_EPOCHS. Runs as its OWN systemd unit so an
# agent-harness kill cannot strand the run between the stop and the relaunch.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-12-enet-verified-350ep
E=.lake/build/efficientnetin_emarmsdp64dropdobf16_ckpt_xla.bin.epoch
LOG=$R/restart_refresh.log
say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
start=$(tr -cd 0-9 < "$E"); say "waiting for the epoch marker to move past $start"
until [ "$(tr -cd 0-9 < "$E")" != "$start" ]; do sleep 2; done
say "epoch $(tr -cd 0-9 < "$E") checkpoint landed; stopping enet-verified and enet-clock"
systemctl --user stop enet-clock 2>/dev/null
systemctl --user stop enet-verified
for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$left" ]; then say "⛔ GPU processes survived the stop: $left — NOT relaunching"; exit 1; fi
if pgrep -f generated_efficientnet_b0_imagenet_shim >/dev/null; then say "⚠ shim producers outlived the stop; killing"; pkill -f generated_efficientnet_b0_imagenet_shim; sleep 3; fi
say "GPUs idle, producers gone; relaunching"
systemctl --user reset-failed enet-verified enet-clock 2>/dev/null
systemd-run --user --unit=enet-verified --working-directory="$PWD" "$PWD/run_enet_verified.sh" >> "$LOG" 2>&1
sleep 5
say "enet-verified: $(systemctl --user is-active enet-verified)"
systemd-run --user --unit=enet-clock --working-directory="$PWD" "$PWD/$R/epoch_clock.sh" >> "$LOG" 2>&1
say "enet-clock: $(systemctl --user is-active enet-clock); supervisor pid: $(pgrep -f 'supervise.sh enet-default-4gpu' | tr '\n' ' ')"
say "DONE"
