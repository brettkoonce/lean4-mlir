#!/usr/bin/env bash
# ▶ 2026-10-21 ConvNeXt-T official-recipe pair copy of runs/2026-10-16-enet-*/<this file>: RUNDIRs, units cnx-off-jax / cnx-off-ver, ckpts and patterns retargeted. Brett, 2026-10-21: "sure give it a go".
# ▶ 2026-10-16 EfficientNet-B0 TF-recipe pair copy of runs/2026-10-14-vit-deit-verified-smx-bf16-300ep/epoch_clock.sh, retargeted: RUNDIR, the emarmstfdp64dropdowneps0001bf16 checkpoint, unit cnx-off-ver / watchers enetv-*, the efficientnet shim, 5004 steps an epoch, 350 epochs. Brett, 2026-10-16: "i'll have you do efficient net next, same idea".
# ▶ 2026-10-14 smx relaunch copy of runs/2026-10-11-vit-deit-verified-bf16-300ep/epoch_clock.sh (checkpoint → the smx variant, unit cnx-off-ver, watchers vitsx-*). Brett, 2026-10-14: "commit please and then yeah run it". Earlier: RUNDIR, the emadp256x4wxdropeps0000001erfbf16 checkpoint, units cnx-off-ver / vitdv-* / vit-deit-jax and 1251 steps an epoch retargeted. Brett, 2026-10-11: "and then after that i'd have you do vit-default-emabf16-4gpu basically".
# ▶ ViT-Ti verified 300ep copy (2026-10-01) of the MNv2 run, itself a copy of the R34/A3/2018 sync-BN watchers; only paths/names/patterns retargeted.
# Per-epoch clock + host-memory guard for the R34 sync-BN bf16 verified run (copied 2026-09-22 from
# the ConvNeXt run's): one row per checkpoint-marker change. The checkpoint is ONE overwritten file,
# so its mtime history exists only if logged here.
# RssAnon is the leak guard (13d90e68's mimalloc bug showed as monotone anon growth). ⭐ Streaming
# val (8182b6e1) removed the ~28 GiB preloaded val split, so expect ~2–3 GiB here, not ~30. A
# ~30 GiB baseline would mean the run is NOT on the streamed path.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
E=.lake/build/convnextin_emaaccdp16xwxdroperfbf16_ckpt_xla.bin.epoch
OUT=runs/2026-10-21-cnx-verified-bf16-300ep/epoch_clock.tsv
[ -s "$OUT" ] || printf 'epoch\tunix\tutc\ttrainer_rss_anon_kb\tmem_available_kb\tswap_used_kb\n' > "$OUT"
last="$(tail -1 "$OUT" | cut -f1)"
while systemctl --user is-active -q cnx-off-ver; do
  n=$(tr -cd 0-9 < "$E" 2>/dev/null)
  if [ -n "$n" ] && [ "$n" != "$last" ]; then
    t=$(stat -c %Y "$E")
    P=$(pgrep -f '^\.lake/build/bin/convnext-imagenet-verified data' | head -1)
    rss=$( [ -n "$P" ] && awk '/RssAnon/{print $2}' "/proc/$P/status" 2>/dev/null )
    av=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    sw=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$t" "$(date -u -d @"$t" '+%F %T')" "${rss:-}" "$av" "$sw" >> "$OUT"
    last=$n
  fi
  sleep 20
done
