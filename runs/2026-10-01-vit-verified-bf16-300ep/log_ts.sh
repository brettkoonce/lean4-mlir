#!/usr/bin/env bash
# ▶ ViT-Ti verified 300ep copy (2026-10-01) of the MNv2 run, itself a copy of the R34/A3/2018 sync-BN watchers; only paths/names/patterns retargeted.
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read
# per window — the degradation on EfficientNet was visible here as 134 -> 240-250 ms/step long
# before it showed in the epoch clock.
# ▶ ViT: this conf RESPAWNS one producer every 10 epochs (LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10), so
# 'SHIM RESPAWN' matches the banner and 'respawn' each replacement. 'RESIDENT' lines mark the eval
# window start; the window is (that line) -> (the 'epoch N:' line).
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-10-01-vit-verified-bf16-300ep
tail -n0 -F "$R/attempt.log" 2>/dev/null \
  | grep --line-buffered -E 'step [0-9]+/2502|Epoch [0-9]+/300|epoch [0-9]+: .*_acc|resum|SHIM RESPAWN|respawn|RESIDENT' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
