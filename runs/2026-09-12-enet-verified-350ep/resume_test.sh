#!/usr/bin/env bash
# ImageNet resume known-answer test: wait for epoch 1's checkpoint, SIGTERM the trainer (not the
# supervisor), and check the restarted attempt reads back the exact BN companion it wrote.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-12-enet-verified-350ep
E=.lake/build/efficientnetin_emarmsdp64dropdobf16_ckpt_xla.bin.epoch
B=.lake/build/efficientnetin_emarmsdp64dropdobf16_ckpt_xla.bin.bn

deadline=$((SECONDS + 3600))
until [ "$(tr -cd 0-9 < "$E" 2>/dev/null)" = "1" ]; do
  [ $SECONDS -gt $deadline ] && { echo "TIMEOUT waiting for epoch-1 marker"; exit 1; }
  sleep 3
done
T1=$(date -u +%T)
PID=$(pgrep -f '^\.lake/build/bin/efficientnet-imagenet-verified data' | head -1)
kill -TERM "$PID"
echo "epoch-1 marker seen $T1; SIGTERM trainer pid=$PID"
sleep 2
H1=$(grep -oE 'BN companion -> [^ ]+ \([0-9]+ floats x 2, hash [0-9]+\)' "$R/full.log" | tail -1 | grep -oE 'hash [0-9]+' | grep -oE '[0-9]+')
echo "save line hash=$H1   file: $(stat -c '%s bytes %y' "$B") sha256=$(sha256sum "$B" | cut -c1-16)"
grep -E 'epoch 1: .*_acc' "$R/full.log" | tail -1

deadline=$((SECONDS + 1800))
until grep -qE 'resumed BN running stats|no BN companion|BN companion .* bytes but' "$R/full.log"; do
  [ $SECONDS -gt $deadline ] && { echo "TIMEOUT waiting for the resume"; exit 1; }
  sleep 5
done
H2=$(grep -oE 'resumed BN running stats \+ ema_bn from [^ ]+ \(hash [0-9]+\)' "$R/full.log" | tail -1 | grep -oE 'hash [0-9]+' | grep -oE '[0-9]+')
echo "load line hash=$H2   $(date -u +%T)"
if [ -n "$H1" ] && [ "$H1" = "$H2" ]; then echo "✅ HASH-MATCH"; else echo "⛔ HASH-MISMATCH or missing (save=$H1 load=$H2)"; fi
grep -nE 'resuming from checkpoint|resumed BN|no BN companion|attempt [0-9]' "$R/full.log" | tail -6
tail -4 "$R/master.log"
# and the resumed attempt must actually step
deadline=$((SECONDS + 1200))
until [ "$(grep -cE 'Epoch 2/350|step [0-9]+/5004' "$R/attempt.log" 2>/dev/null)" -ge 1 ]; do
  [ $SECONDS -gt $deadline ] && { echo "TIMEOUT waiting for resumed steps"; tail -20 "$R/attempt.log"; exit 1; }
  sleep 10
done
echo "resumed attempt is stepping: $(grep -E 'Epoch 2/350|step [0-9]+/5004' "$R/attempt.log" | tail -1)"
