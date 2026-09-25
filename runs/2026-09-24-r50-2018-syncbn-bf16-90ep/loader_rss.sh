#!/usr/bin/env bash
# ▶ R50 2018 sync-BN copy (2026-09-24) of the R34/A3 sync-BN watchers; only paths/names/patterns retargeted.
# ⭐⭐ THE WATCHER THIS RUN EXISTS FOR. Per-epoch shim-loader memory: which loader grows, and in
# which memory class. On EfficientNet the degrading producer showed as RSS 5 -> 7-11 GiB with the
# growth in glibc malloc ARENA-CLASS mappings (anonymous, 8-68 MiB) — that is the `arena_class_gib`
# column, and it is the signal, not total RSS.
#
# ▶ R34 COPY (2026-09-22, from runs/2026-09-18-cnx-verified-300ep/). R34's shim is flip-only and
# stayed flat for 21.9 h on the 09-16 run, and this conf runs it BARE: no respawn
# (`LEAN_MLIR_SHIM_RESPAWN_EPOCHS` deliberately absent). So a loader here lives for the whole run,
# and a monotone `arena_class_gib` would be the fault itself, not a mitigation failing.
#
# ⚠ VAL PRODUCERS ARE SKIPPED. Since streaming val (8182b6e1) each eval pass spawns two short-lived
# producers of the SAME shim script with SHIM_SPLIT=validation. They are reaped every epoch and
# would otherwise add a fresh near-zero-age row per epoch that reads like a respawn.
#
# Reads /proc only. One row per TRAIN loader per epoch marker change; `etime_s` (loader age) is the
# x-axis — LIFETIME is what matters, not epoch count.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
E=.lake/build/resnet50in_momdp64bf16_ckpt_xla.bin.epoch
OUT=runs/2026-09-24-r50-2018-syncbn-bf16-90ep/loader_rss.tsv
[ -s "$OUT" ] || printf 'utc\tepoch\tpid\tetime_s\trss_gib\tarena_class_gib\tarena_class_maps\tthreads\n' > "$OUT"
last=""
while systemctl --user is-active -q r50-2018; do
  n=$(tr -cd 0-9 < "$E" 2>/dev/null)
  if [ -n "$n" ] && [ "$n" != "$last" ]; then
    for p in $(pgrep -f generated_resnet50_imagenet_2018_shim | sort -n); do
      [ -r "/proc/$p/smaps" ] || continue
      # train loaders only — see the header on the per-pass val producers
      tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx 'SHIM_SPLIT=validation' && continue
      et=$(ps -o etimes= -p "$p" | tr -d ' ')
      th=$(awk '/^Threads/{print $2}' "/proc/$p/status" 2>/dev/null)
      awk -v utc="$(date -u '+%F %T')" -v ep="$n" -v pid="$p" -v et="$et" -v th="$th" \
        '/^[0-9a-f]+-[0-9a-f]+ /{split($1,a,"-"); sz=(strtonum("0x"a[2])-strtonum("0x"a[1]))/1048576; path=$6}
         /^Rss:/{tot+=$2; if (path=="" && sz>=8 && sz<=68) {ar+=$2; nm++}}
         END{if (tot>0) printf "%s\t%s\t%s\t%s\t%.2f\t%.2f\t%d\t%s\n", utc, ep, pid, et, tot/1048576, ar/1048576, nm, th}' \
        "/proc/$p/smaps" >> "$OUT" 2>/dev/null
    done
    last=$n
  fi
  sleep 30
done
