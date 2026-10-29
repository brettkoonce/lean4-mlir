#!/usr/bin/env bash
# ▶ 2026-10-21 ConvNeXt-T official-recipe pair copy of runs/2026-10-16-enet-*/<this file>: RUNDIRs, units cnx-off-jax / cnx-off-ver, ckpts and patterns retargeted. Brett, 2026-10-21: "sure give it a go".
# ▶ 2026-10-16 EfficientNet-B0 TF-recipe pair copy of runs/2026-10-14-vit-deit-verified-smx-bf16-300ep/log_ts.sh, retargeted: RUNDIR, the emarmstfdp64dropdowneps0001bf16 checkpoint, unit cnx-off-ver / watchers enetv-*, the efficientnet shim, 5004 steps an epoch, 350 epochs. Brett, 2026-10-16: "i'll have you do efficient net next, same idea".
# ▶ 2026-10-14 smx relaunch copy of runs/2026-10-11-vit-deit-verified-bf16-300ep/log_ts.sh (checkpoint → the smx variant, unit cnx-off-ver, watchers vitsx-*). Brett, 2026-10-14: "commit please and then yeah run it". Earlier: RUNDIR, the emadp256x4wxdropeps0000001erfbf16 checkpoint, units cnx-off-ver / vitdv-* / vit-deit-jax and 1251 steps an epoch retargeted. Brett, 2026-10-11: "and then after that i'd have you do vit-default-emabf16-4gpu basically".
# ▶ ViT-Ti verified 300ep copy (2026-10-01) of the MNv2 run, itself a copy of the R34/A3/2018 sync-BN watchers; only paths/names/patterns retargeted.
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read
# per window — the degradation on EfficientNet was visible here as 134 -> 240-250 ms/step long
# before it showed in the epoch clock.
# ▶ ViT: this conf RESPAWNS one producer every 10 epochs (LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10), so
# 'SHIM RESPAWN' matches the banner and 'respawn' each replacement. 'RESIDENT' lines mark the eval
# window start; the window is (that line) -> (the 'epoch N:' line).
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-10-21-cnx-verified-bf16-300ep
tail -n0 -F "$R/attempt.log" 2>/dev/null \
  | grep --line-buffered -E 'step [0-9]+/4992|Epoch [0-9]+/300|epoch [0-9]+: .*_acc|resum|SHIM RESPAWN|respawn|RESIDENT' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
