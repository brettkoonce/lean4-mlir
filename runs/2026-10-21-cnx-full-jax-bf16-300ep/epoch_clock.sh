#!/usr/bin/env bash
# ▶ 2026-10-21 ConvNeXt-T official-recipe pair copy of runs/2026-10-16-enet-*/<this file>: RUNDIRs, units cnx-off-jax / cnx-off-ver, ckpts and patterns retargeted. Brett, 2026-10-21: "sure give it a go".
# ▶ 2026-10-16 EfficientNet-B0 TF-recipe pair copy of runs/2026-10-11-vit-deit-jax-bf16-300ep/epoch_clock.sh: RUNDIR, ckpt base enet_tf350_2612/efficientnet_b0_imagenet and unit cnx-off-jax retargeted. Brett, 2026-10-16: "i'll have you do efficient net next, same idea".
# ▶ 2026-10-11 DeiT-recipe rerun copy of runs/2026-10-01-vit-jax-bf16-300ep/epoch_clock.sh: RUNDIR, ckpt base vit_tiny_jax_2611 and unit cnx-off-jax retargeted.
# Per-epoch clock for the ConvNeXt-T JAX 300-epoch run: one row per new `<base>_e<N>.state.npz` (the trainer
# keeps only the 3 newest, so their mtimes exist only if logged here), plus host memory.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
B=/home/skoonce/cnx_t_jax_2611/convnext_tiny_imagenet_full
OUT=runs/2026-10-21-cnx-full-jax-bf16-300ep/epoch_clock.tsv
[ -s "$OUT" ] || printf 'epoch\tunix\tutc\tmem_available_kb\n' > "$OUT"
last="$(tail -1 "$OUT" | cut -f1)"
while systemctl --user is-active -q cnx-off-jax; do
  n=0; for f in "$B"_e*.state.npz; do [ -e "$f" ] || continue; k="${f##*_e}"; k="${k%.state.npz}"; [ "$k" -gt "$n" ] && n=$k; done
  if [ "$n" -gt 0 ] && [ "$n" != "$last" ]; then
    t=$(stat -c %Y "${B}_e${n}.state.npz")
    printf '%s\t%s\t%s\t%s\n' "$n" "$t" "$(date -u -d @"$t" '+%F %T')" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$OUT"
    last=$n
  fi
  sleep 30
done
