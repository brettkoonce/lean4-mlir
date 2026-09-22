#!/usr/bin/env bash
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read
# per window — the degradation on EfficientNet was visible here as 134 -> 240-250 ms/step long
# before it showed in the epoch clock.
# ⚠ `SHIM RESPAWN` is matched deliberately: the respawn is ON in this run, so every swap must appear
# here with its slot and seed, or the respawn evidence has no timeline at all.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-18-cnx-verified-300ep
tail -n0 -F "$R/attempt.log" 2>/dev/null \
  | grep --line-buffered -E 'step [0-9]+/5004|Epoch [0-9]+/300|epoch [0-9]+: .*_acc|resum|SHIM RESPAWN|respawn|RESIDENT' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
