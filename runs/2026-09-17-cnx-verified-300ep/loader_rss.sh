#!/usr/bin/env bash
# ⭐⭐ THE WATCHER THIS RUN EXISTS FOR. Per-epoch shim-loader memory: which loader grows, and in
# which memory class. On EfficientNet the degrading producer showed as RSS 5 -> 7-11 GiB with the
# growth in glibc malloc ARENA-CLASS mappings (anonymous, 8-68 MiB) — that is the `arena_class_gib`
# column, and it is the signal, not total RSS.
#
# ⚠ ConvNeXt's shim is the HEAVY kind — AutoAugment + RandAugment + random erasing per image,
# EfficientNet's class. R34's flip-only shim stayed flat for 21.9 h; if these rows stay flat HERE
# too, augmentation weight is exonerated. ⚠⚠ But the respawn is ON in this run
# (LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10), so a flat curve is EXPECTED and is evidence about the
# MITIGATION, not about the fault. The fault question is answered by `etime_s`: no loader should
# ever exceed ~40 epochs of age, and if one does, the respawn is not firing.
#
# Reads /proc only. One row per loader per epoch marker change; pids change at every respawn, so the
# `etime_s` column (loader age) is the x-axis — LIFETIME is what matters, not epoch count.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
E=.lake/build/convnextin_adamdpwxclipdropbf16_ckpt_xla.bin.epoch
OUT=runs/2026-09-17-cnx-verified-300ep/loader_rss.tsv
[ -s "$OUT" ] || printf 'utc\tepoch\tpid\tetime_s\trss_gib\tarena_class_gib\tarena_class_maps\tthreads\n' > "$OUT"
last=""
while systemctl --user is-active -q cnx-verified; do
  n=$(tr -cd 0-9 < "$E" 2>/dev/null)
  if [ -n "$n" ] && [ "$n" != "$last" ]; then
    for p in $(pgrep -f generated_convnext_tiny_imagenet_shim | sort -n); do
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
