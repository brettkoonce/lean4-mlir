#!/usr/bin/env bash
# ⛔⛔ THE TEST §4 EXISTS FOR: resume has NEVER been exercised on a LayerNorm net at ImageNet scale.
#
# The 367bb28b resume fix and all eight of its live resumes were on EfficientNet — a BatchNorm net
# with a `.bn` companion file. ConvNeXt writes NO companion (no BN, and `ema := false` in this
# variant), so its resume surface is just the `[θ|m|v]` blob plus the epoch marker. Every ViT and
# ConvNeXt verified run so far was ONE attempt with ZERO restarts, so that path has never run here.
# ⚠ Every thermal rest, every REST_EPOCHS fallback and every respawn-failure recovery in §3 rides
# on it. Cost of finding out now: ~1 epoch. Cost of finding out at epoch 250: the run.
#
# Method: wait for epoch 1's checkpoint, hash it, SIGTERM the TRAINER (not the supervisor), then
# check the restarted attempt (a) announces a resume AT epoch 1, (b) read the same bytes it wrote,
# (c) actually steps, and (d) lands an epoch-2 accuracy CONTINUOUS with epoch 1's.
#
# ⛔ If it does not resume cleanly: STOP. That is R2 of the shim plan §4 — run it on Imagenette
#    first (`convnext-verified-adam`, variant `ema`, rebuild the stale exe) and relaunch only after.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-18-cnx-verified-300ep
C=.lake/build/convnextin_adamdpwxclipdropbf16_ckpt_xla.bin
E=$C.epoch
LOG=$R/resume_test.log
exec > >(tee -a "$LOG") 2>&1
echo "══ resume known-answer test, ConvNeXt-T bf16 — started $(date -u '+%F %T') UTC ══"

# ⚠ There must be NO .bn companion: if one appears, this net is not the net we think it is.
if [ -f "$C.bn" ]; then
  echo "⛔ a .bn companion EXISTS for a LayerNorm net ($C.bn) — investigate before trusting anything."
fi

# ⚠ GUARD THE READ. `tr -cd 0-9 < "$E"` writes a redirect error to stderr on every poll until the
# marker exists, which at a 5 s cadence buried the first run of this test in ~200 identical lines
# before epoch 1 even finished. The test still worked; the log was useless. Check the file first.
marker() { [ -f "$E" ] && tr -cd 0-9 < "$E" 2>/dev/null; }
deadline=$((SECONDS + 7200))
until [ "$(marker)" = "1" ]; do
  [ $SECONDS -gt $deadline ] && { echo "TIMEOUT waiting for epoch-1 marker"; exit 1; }
  sleep 15
done
sleep 3   # let the blob's own write settle behind the marker
SZ1=$(stat -c %s "$C"); SHA1=$(sha256sum "$C" | cut -c1-32); MT1=$(stat -c %Y "$C")
ACC1=$(grep -hE '^  epoch 1: ' "$R/full.log" | tail -1)
echo "epoch-1 marker seen $(date -u +%T)"
echo "  ckpt: $SZ1 bytes  sha256=$SHA1  mtime=$MT1"
echo "  $ACC1"

PID=$(pgrep -f '^\.lake/build/bin/convnext-imagenet-verified data' | head -1)
[ -n "$PID" ] || { echo "⛔ trainer pid not found — is the unit running?"; exit 1; }
kill -TERM "$PID"; echo "SIGTERM -> trainer pid=$PID (NOT the supervisor)"

# (a) the resume must be announced, and AT EPOCH 1
deadline=$((SECONDS + 2400))
until grep -q 'resuming from checkpoint at epoch' "$R/full.log" 2>/dev/null; do
  [ $SECONDS -gt $deadline ] && { echo "⛔ TIMEOUT: no resume announcement"; tail -30 "$R/attempt.log"; exit 1; }
  sleep 5
done
RES=$(grep -h 'resuming from checkpoint at epoch' "$R/full.log" | tail -1)
echo "resume line: $RES   ($(date -u +%T))"
case "$RES" in
  *"at epoch 1") echo "  ✅ resumed AT EPOCH 1" ;;
  *) echo "  ⛔ resumed at the WRONG epoch — expected 1: $RES" ;;
esac

# (b) it must have read back exactly the bytes it wrote
SZ2=$(stat -c %s "$C"); SHA2=$(sha256sum "$C" | cut -c1-32)
if [ "$SHA1" = "$SHA2" ] && [ "$SZ1" = "$SZ2" ]; then
  echo "  ✅ checkpoint BYTE-IDENTICAL across the restart ($SZ2 bytes, $SHA2)"
else
  echo "  ⛔ checkpoint CHANGED across the restart: $SZ1/$SHA1 -> $SZ2/$SHA2"
fi
# ⚠ and no BN companion should have appeared
[ -f "$C.bn" ] && echo "  ⛔ a .bn companion appeared during the restart" || echo "  ✅ still no .bn companion (correct for LayerNorm)"

# (c) the resumed attempt must actually step
deadline=$((SECONDS + 2400))
until grep -qE 'Epoch 2/300|step [0-9]+/5004' "$R/attempt.log" 2>/dev/null; do
  [ $SECONDS -gt $deadline ] && { echo "⛔ TIMEOUT waiting for resumed steps"; tail -30 "$R/attempt.log"; exit 1; }
  sleep 10
done
echo "  ✅ stepping again: $(grep -hE 'Epoch 2/300|step [0-9]+/5004' "$R/attempt.log" | tail -1)"

# (d) epoch 2 must be CONTINUOUS with epoch 1 — the actual known-answer. A resume that silently
#     restarted from init would show epoch 2 at ~epoch-1-from-scratch accuracy, not above it.
deadline=$((SECONDS + 5400))
until grep -qhE '^  epoch 2: ' "$R/full.log" 2>/dev/null; do
  [ $SECONDS -gt $deadline ] && { echo "⚠ TIMEOUT waiting for epoch 2 — check continuity by hand"; exit 1; }
  sleep 20
done
ACC2=$(grep -hE '^  epoch 2: ' "$R/full.log" | tail -1)
echo "  $ACC2"
# line format: "  epoch 1: test_acc = 4352/50000 = 8.704000%  top5 = ..."
A1=$(sed -E 's@.*test_acc = [0-9]+/50000 = ([0-9.]+)%.*@\1@' <<< "$ACC1")
A2=$(sed -E 's@.*test_acc = [0-9]+/50000 = ([0-9.]+)%.*@\1@' <<< "$ACC2")
echo "  epoch1 top1=${A1}%  epoch2 top1=${A2}%"
if awk -v a="$A1" -v b="$A2" 'BEGIN{exit !(b+0 > a+0)}' 2>/dev/null; then
  echo "  ✅ CONTINUOUS — epoch 2 is above epoch 1; the resume kept the weights and the optimizer state"
else
  echo "  ⚠ epoch 2 is NOT above epoch 1 — at 20-epoch warmup this CAN happen legitimately, so read"
  echo "    it against the reference curve's e1/e2 (0.0064 -> 0.0068) before calling it a defect."
fi
echo "── the reference's own first two epochs, for comparison ──"
head -3 "$R/reference_curve.tsv" | column -t
echo "══ done $(date -u '+%F %T') UTC ══"
