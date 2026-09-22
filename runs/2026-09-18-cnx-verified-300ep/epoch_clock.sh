#!/usr/bin/env bash
# Per-epoch clock + host-memory guard for the ConvNeXt-T bf16 verified run: one row per
# checkpoint-marker change. The checkpoint is ONE overwritten file, so its mtime history exists
# only if logged here.
# RssAnon is the leak guard (13d90e68's mimalloc bug showed as monotone anon growth); the large
# baseline is the preloaded 50k val split and is expected.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
E=.lake/build/convnextin_adamdpwxclipdropbf16_ckpt_xla.bin.epoch
OUT=runs/2026-09-18-cnx-verified-300ep/epoch_clock.tsv
[ -s "$OUT" ] || printf 'epoch\tunix\tutc\ttrainer_rss_anon_kb\tmem_available_kb\tswap_used_kb\n' > "$OUT"
last="$(tail -1 "$OUT" | cut -f1)"
while systemctl --user is-active -q cnx-verified; do
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
