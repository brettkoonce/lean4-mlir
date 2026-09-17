#!/usr/bin/env bash
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read
# per window — the degradation on EfficientNet was visible here as 134 -> 240-250 ms/step long
# before it showed in the epoch clock.
# ⚠ `SHIM RESPAWN` is matched deliberately: when Arm B is running, every swap must appear here with
# its slot and seed, or the respawn evidence has no timeline.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-16-r34-bf16-90ep
tail -n0 -F "$R/attempt.log" 2>/dev/null \
  | grep --line-buffered -E 'step [0-9]+/5004|Epoch [0-9]+/90|epoch [0-9]+: .*_acc|BN companion|resum|SHIM RESPAWN|respawn' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
