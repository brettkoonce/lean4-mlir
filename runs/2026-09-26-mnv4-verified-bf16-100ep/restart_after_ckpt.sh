#!/usr/bin/env bash
# 2026-09-27: fault_watch fired a SUSTAINED LOADER FAULT at e83 (1095 s vs 629 base; one producer's
# arena 8.19 GiB). B0's cure: restart right after a checkpoint so every train producer is fresh.
# Waits for the epoch marker to move past 83, stops the run, relaunches the same conf (auto-resume).
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
V=runs/2026-09-26-mnv4-verified-bf16-100ep
E=.lake/build/mnv4in_emaaccdp8x128wxdowd005bf16_ckpt_xla_e100.bin.epoch
start=$(tr -cd 0-9 < $E)
while [ "$(tr -cd 0-9 < $E)" = "$start" ]; do systemctl --user is-active -q mnv4-verified || exit 0; sleep 5; done
n=$(tr -cd 0-9 < $E); sleep 5   # the .bin/.bn writes are atomic renames; the marker is last
echo "$(date -u '+%F %T') restart: checkpoint e$n written; stopping for fresh loaders" >> $V/RESTARTS
for u in mnv4-verified mnv4v-epoch-clock mnv4v-loader-rss mnv4v-fault-watch mnv4v-edac-watch; do systemctl --user stop $u; done
sleep 20; while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 5; done
pgrep -f generated_mobilenet_v4_imagenet_shim >/dev/null && { echo "⛔ stale shim producers still alive; not relaunching" >> $V/RESTARTS; exit 1; }
[ -s $V/WATCH_EVENT ] && mv $V/WATCH_EVENT $V/WATCH_EVENT.run-stopped-e$n
for u in mnv4-verified mnv4v-epoch-clock mnv4v-loader-rss mnv4v-fault-watch mnv4v-edac-watch; do systemctl --user reset-failed $u 2>/dev/null; done
systemd-run --user --unit=mnv4-verified --working-directory="$PWD" --setenv=PATH="$PATH" \
  --setenv=RUNDIR=$V --setenv=LEAN_MLIR_DUMP_CORRECT=$V/bitmaps/mnv4 scripts/supervise.sh mnv4-default-4gpu
sleep 15; systemctl --user is-active -q mnv4-verified || { echo "⛔ relaunch not active" >> $V/RESTARTS; exit 1; }
# fault_watch recalibrates from the epochs in its window; start it after the resume so the new baseline is fresh
for w in epoch_clock loader_rss edac_watch; do systemd-run --user --unit=mnv4v-${w//_/-} --working-directory="$PWD" bash $V/$w.sh; done
# BASE_TO=$n: its scan starts after BASE_TO, so e81-83 (the fault itself) are not re-flagged
systemd-run --user --unit=mnv4v-fault-watch --working-directory="$PWD" --setenv=BASE_TO=$n bash $V/fault_watch.sh
echo "$(date -u '+%F %T') relaunched from e$n with fresh loaders" >> $V/RESTARTS
