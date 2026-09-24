#!/usr/bin/env bash
# ▶ R50 A3 4x128 sync-BN copy (2026-09-23) of the R34 sync-BN run's watcher; only paths/names/patterns retargeted.
# Timestamps every line of the live attempt log (UTC), so step pace and eval duration can be read
# per window — the degradation on EfficientNet was visible here as 134 -> 240-250 ms/step long
# before it showed in the epoch clock.
# ▶ R34 copy (2026-09-22). Respawn is OFF on this conf, so `SHIM RESPAWN` should match only the
# startup banner. `RESIDENT` matters more here: its `@resnet50in160_fwd_eval` line at each pass is the
# eval-window START, so the window is (that line) → (the `epoch N:` line). Streaming val's
# commit leaves that number open for a real run.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-23-r50-a3-syncbn-bf16-100ep
tail -n0 -F "$R/attempt.log" 2>/dev/null \
  | grep --line-buffered -E 'step [0-9]+/2500|Epoch [0-9]+/100|epoch [0-9]+: .*_acc|resum|SHIM RESPAWN|respawn|RESIDENT' \
  | while IFS= read -r l; do printf '%s %s\n' "$(date -u '+%F %T')" "$l"; done >> "$R/log_ts.log"
