#!/usr/bin/env bash
# Per-epoch clock for the ViT-Ti JAX 300-epoch run: one row per new `<base>_e<N>.state.npz` (the trainer
# keeps only the 3 newest, so their mtimes exist only if logged here), plus host memory.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
B=/home/skoonce/vit_tiny_default_300ep/vit_tiny_imagenet
OUT=runs/2026-10-01-vit-jax-bf16-300ep/epoch_clock.tsv
[ -s "$OUT" ] || printf 'epoch\tunix\tutc\tmem_available_kb\n' > "$OUT"
last="$(tail -1 "$OUT" | cut -f1)"
while systemctl --user is-active -q vit-jax; do
  n=0; for f in "$B"_e*.state.npz; do [ -e "$f" ] || continue; k="${f##*_e}"; k="${k%.state.npz}"; [ "$k" -gt "$n" ] && n=$k; done
  if [ "$n" -gt 0 ] && [ "$n" != "$last" ]; then
    t=$(stat -c %Y "${B}_e${n}.state.npz")
    printf '%s\t%s\t%s\t%s\n' "$n" "$t" "$(date -u -d @"$t" '+%F %T')" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$OUT"
    last=$n
  fi
  sleep 30
done
