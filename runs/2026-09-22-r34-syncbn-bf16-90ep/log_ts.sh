#!/usr/bin/env bash
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read
# per window — the degradation on EfficientNet was visible here as 134 -> 240-250 ms/step long
# before it showed in the epoch clock.
# ▶ R34 copy (2026-09-22). Respawn is OFF on this conf, so `SHIM RESPAWN` should match only the
# startup banner. `RESIDENT` matters more here: its `@resnet34in_fwd_eval` line at each pass is the
# eval-window START, so the window is (that line) → (the `epoch N:` line). Streaming val's
# commit leaves that number open for a real run.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-22-r34-syncbn-bf16-90ep
tail -n0 -F "$R/attempt.log" 2>/dev/null \
  | grep --line-buffered -E 'step [0-9]+/5004|Epoch [0-9]+/90|epoch [0-9]+: .*_acc|resum|SHIM RESPAWN|respawn|RESIDENT' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
