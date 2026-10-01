#!/usr/bin/env bash
# ▶ MNv2 350ep copy (2026-09-27) of the MNv4 run's restart script — NOT run; kept ready in case
# fault_watch fires. B0's cure: restart right after a checkpoint so every train producer is fresh.
# Waits for the epoch marker to move past the current epoch, stops the run, relaunches the same conf (auto-resume).
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
V=runs/2026-09-27-mnv2-verified-bf16-350ep
E=.lake/build/mobilenetv2in_rmsdp64wxdols0eps0001bf16_ckpt_xla.bin.epoch
start=$(tr -cd 0-9 < $E)
while [ "$(tr -cd 0-9 < $E)" = "$start" ]; do systemctl --user is-active -q mnv2-verified || exit 0; sleep 5; done
n=$(tr -cd 0-9 < $E); sleep 5   # the .bin/.bn writes are atomic renames; the marker is last
echo "$(date -u '+%F %T') restart: checkpoint e$n written; stopping for fresh loaders" >> $V/RESTARTS
for u in mnv2-verified mnv2v-epoch-clock mnv2v-loader-rss mnv2v-fault-watch mnv2v-edac-watch; do systemctl --user stop $u; done
sleep 20; while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 5; done
pgrep -f generated_mobilenet_v2_imagenet_shim >/dev/null && { echo "⛔ stale shim producers still alive; not relaunching" >> $V/RESTARTS; exit 1; }
[ -s $V/WATCH_EVENT ] && mv $V/WATCH_EVENT $V/WATCH_EVENT.run-stopped-e$n
for u in mnv2-verified mnv2v-epoch-clock mnv2v-loader-rss mnv2v-fault-watch mnv2v-edac-watch; do systemctl --user reset-failed $u 2>/dev/null; done
systemd-run --user --unit=mnv2-verified --working-directory="$PWD" --setenv=PATH="$PATH" \
  --setenv=RUNDIR=$V --setenv=LEAN_MLIR_DUMP_CORRECT=$V/bitmaps/mnv2 scripts/supervise.sh mnv2-default-4gpu
sleep 15; systemctl --user is-active -q mnv2-verified || { echo "⛔ relaunch not active" >> $V/RESTARTS; exit 1; }
# fault_watch recalibrates from the epochs in its window; start it after the resume so the new baseline is fresh
for w in epoch_clock loader_rss edac_watch; do systemd-run --user --unit=mnv2v-${w//_/-} --working-directory="$PWD" bash $V/$w.sh; done
# BASE_TO=$n: its scan starts after BASE_TO, so the faulted epochs are not re-flagged
systemd-run --user --unit=mnv2v-fault-watch --working-directory="$PWD" --setenv=BASE_TO=$n bash $V/fault_watch.sh
echo "$(date -u '+%F %T') relaunched from e$n with fresh loaders" >> $V/RESTARTS
