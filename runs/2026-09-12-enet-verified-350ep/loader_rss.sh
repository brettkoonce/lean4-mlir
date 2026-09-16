#!/usr/bin/env bash
# Per-epoch shim-loader memory log (2026-09-14): which loader grows, and in which memory class.
# Reads /proc only. One row per loader per epoch marker change; pids change at every restart, so the
# `etime_s` column (loader age) is the x-axis, and spawn order is the pid order within one epoch.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
E=.lake/build/efficientnetin_emarmsdp64dropdobf16_ckpt_xla.bin.epoch
OUT=runs/2026-09-12-enet-verified-350ep/loader_rss.tsv
[ -s "$OUT" ] || printf 'utc\tepoch\tpid\tetime_s\trss_gib\tarena_class_gib\tarena_class_maps\tthreads\n' > "$OUT"
last=""
while systemctl --user is-active -q enet-verified; do
  n=$(tr -cd 0-9 < "$E" 2>/dev/null)
  if [ -n "$n" ] && [ "$n" != "$last" ]; then
    for p in $(pgrep -f generated_efficientnet_b0_imagenet_shim | sort -n); do
      [ -r "/proc/$p/smaps" ] || continue
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
