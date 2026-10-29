#!/usr/bin/env bash
# ▶ 2026-10-21 ConvNeXt-T official-recipe pair copy of runs/2026-10-16-enet-verified-bf16-350ep/chain_after_jax.sh: RUNDIRs, units cnx-off-jax / cnx-off-ver / cnxv-* and the job cnx-default-emabf16-4gpu retargeted. Brett, 2026-10-21: "sure give it a go".
# ▶ PLUS A FORCED RESUME TEST (Brett, 2026-10-21: the box may have to be moved mid-run, "would the
#   suspend resume logic work"). This variant (`emaaccdp16xwxdroperfbf16`) is the first 5-region
#   accumulating ConvNeXt blob [θ|m|v|ema|G] and had never been resumed. After the epoch-2 checkpoint
#   and ~1000 micro-steps into epoch 3, this script does what a move would: stops the watchers and
#   the trainer unit (SIGTERM mid-epoch), relaunches the same supervise + watchers, and checks the
#   relaunch printed `resuming from checkpoint at epoch 2`. Epoch 3 is trained twice (the partial
#   one is lost), exactly as in a real stop.
# Launches NOTHING if the JAX run ended any other way than ✅ COMPLETE — it writes CHAIN_EVENT and exits.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
RD=runs/2026-10-21-cnx-verified-bf16-300ep
JRD=runs/2026-10-21-cnx-full-jax-bf16-300ep
E=.lake/build/convnextin_emaaccdp16xwxdroperfbf16_ckpt_xla.bin.epoch
WATCHERS="epoch_clock loader_rss fault_watch edac_watch"
while systemctl --user is-active -q cnx-off-jax; do sleep 60; done
if ! grep -q '✅ COMPLETE' "$JRD/master.log"; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — cnx-off-jax ended without ✅ COMPLETE:"; tail -5 "$JRD/master.log"; } > "$RD/CHAIN_EVENT"; exit 0; fi
sleep 30; for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 10; done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  { echo "⛔ NOT LAUNCHED $(date -u '+%F %T') UTC — GPUs still busy 10 min after cnx-off-jax ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > "$RD/CHAIN_EVENT"; exit 0; fi

launch() {   # $1 = label for CHAIN_EVENT
  systemd-run --user --unit=cnxv-log-ts --working-directory="$PWD" bash $RD/log_ts.sh
  # ⛔ --setenv=PATH: the conf's precheck runs `lake build` (pc_exe); the MNv4 chain lacked it.
  systemd-run --user --unit=cnx-off-ver --working-directory="$PWD" --setenv=PATH="$PATH" \
    --setenv=RUNDIR=$RD \
    scripts/supervise.sh cnx-default-emabf16-4gpu
  sleep 10
  systemctl --user is-active -q cnx-off-ver || { echo "⛔ cnx-off-ver not active 10 s after $1" >> "$RD/CHAIN_EVENT"; exit 0; }
  for w in $WATCHERS; do
    systemd-run --user --unit=cnxv-${w//_/-} --working-directory="$PWD" bash $RD/$w.sh; done
  echo "✅ $1 $(date -u '+%F %T') UTC: cnx-off-ver + 5 watchers" >> "$RD/CHAIN_EVENT"
}
echo "▶ launching $(date -u '+%F %T') UTC" > "$RD/CHAIN_EVENT"
launch "launched"

# ── the forced resume test ──────────────────────────────────────────────────────────────────────
until [ "$(tr -cd 0-9 < "$E" 2>/dev/null)" = "2" ]; do
  systemctl --user is-active -q cnx-off-ver || { echo "⛔ resume test SKIPPED: cnx-off-ver ended before the epoch-2 checkpoint" >> "$RD/CHAIN_EVENT"; exit 0; }
  sleep 30; done
until grep -qaE 'step 1000/4992' <(sed -n '/epoch 2: test_acc/,$p' "$RD/attempt.log"); do
  systemctl --user is-active -q cnx-off-ver || { echo "⛔ resume test SKIPPED: cnx-off-ver ended inside epoch 3" >> "$RD/CHAIN_EVENT"; exit 0; }
  sleep 15; done
echo "▶ resume test: stopping mid-epoch 3 at $(date -u '+%F %T') UTC (marker $(tr -cd 0-9 < "$E"))" >> "$RD/CHAIN_EVENT"
for w in $WATCHERS log_ts; do systemctl --user stop cnxv-${w//_/-}; done
systemctl --user stop cnx-off-ver
for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
pgrep -f 'convnext-imagenet-verified|generated_convnext_tiny_imagenet' >/dev/null && \
  echo "⚠ stray trainer/loader processes after stop: $(pgrep -f 'convnext-imagenet-verified|generated_convnext_tiny_imagenet' | tr '\n' ' ')" >> "$RD/CHAIN_EVENT"
echo "  stopped; marker still $(tr -cd 0-9 < "$E"); relaunching" >> "$RD/CHAIN_EVENT"
# A unit killed by SIGTERM ends `failed` and stays loaded; systemd-run refuses to reuse its name until reset.
systemctl --user reset-failed cnx-off-ver cnxv-log-ts $(for w in $WATCHERS; do printf 'cnxv-%s ' "${w//_/-}"; done) 2>/dev/null
n0=$(grep -ac 'resuming from checkpoint at epoch 2' "$RD/full.log")
launch "relaunched"
for i in $(seq 1 60); do
  [ "$(grep -ac 'resuming from checkpoint at epoch 2' "$RD/full.log")" -gt "$n0" ] && break; sleep 10; done
if [ "$(grep -ac 'resuming from checkpoint at epoch 2' "$RD/full.log")" -gt "$n0" ]; then
  echo "✅ resume test: the relaunch resumed from the epoch-2 checkpoint" >> "$RD/CHAIN_EVENT"
else
  { echo "⛔ resume test: no 'resuming from checkpoint at epoch 2' within 10 min of the relaunch:"; tail -5 "$RD/attempt.log"; } >> "$RD/CHAIN_EVENT"
fi
