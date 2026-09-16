#!/usr/bin/env bash
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read per window.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-12-enet-verified-350ep
tail -n0 -F "$R/attempt.log" 2>/dev/null | grep --line-buffered -E 'step [0-9]+/5004|Epoch [0-9]+/350|epoch [0-9]+: .*_acc|BN companion|resum' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
