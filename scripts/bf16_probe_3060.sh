#!/usr/bin/env bash
# bf16_probe_3060.sh — steady-state ms/step on THIS box (4× RTX 3060), f32 and bf16, fed and synth.
#
#     scripts/bf16_probe_3060.sh /tmp/probe.tsv            # every net, both precisions, fed+synth
#     scripts/bf16_probe_3060.sh /tmp/probe.tsv r34 vit    # just those nets
#     WORKERS=12 ARMS=fed NETS=vit scripts/bf16_probe_3060.sh /tmp/sweep.tsv    # the §1c sweep
#
# ⛔ WHY THIS EXISTS AND WHY IT IS NOT `bf16_probe_4gpu.sh`. That script is ares': it pins
# `CUDA_VISIBLE_DEVICES=0,2,3,4` (cards this box does not have) and the `xla_cuda12` plugin, so it
# cannot run here at all. It is left untouched so ares' committed numbers keep their provenance.
#
# ⛔⛔ AND ITS NUMBERS ARE WRONG EVEN ON ARES. `LEAN_MLIR_MAX_STEPS=40` with the probe clock
# starting at step 8 measures a window that is roughly half BURST: `SHIM PREFETCH` holds one read
# in flight per handle (depth = SHIM_WORKERS = 8) and the producers fill those during compile and
# during the one-time val drain, so the early steps are served from a queue nobody waited for.
# §9.6's ViT row reads 159 ms/step where the steady state is 375 — 2.4×. This script starts the
# clock at step 200 (`LEAN_MLIR_PROBE_WARM`, added for exactly this) and runs to 600.
#
# ⭐ THE SYNTH ARM IS THE POINT, not a bonus. `LEAN_MLIR_BENCH_SYNTH` removes the shim read and
# nothing else, so it is the COMPUTE-ONLY FLOOR. fed-minus-synth IS the shim starvation, in ms.
# Without both arms a slow kernel and a slow producer are the same number. (Precedent: 783 fed vs
# 249 synth at 4×bs64 resident fp32 on ares.) The driver also now prints PROBE-SPREAD, so
# `med - min` gives a second, single-run read on the same wait.
#
# ⚠ `LEAN_MLIR_CKPT_TAG` is NOT optional — without it a finished run's checkpoint makes the probe
#   exit instantly at "resuming from checkpoint" and report nothing. ⚠ It is qualified PER VARIANT
#   here, not one tag for the whole sweep: checkpoints are per-variant and outlive their artifact,
#   and a shared tag is an invitation for r50 to resume r34's (different region count).
# ⚠ `PJRT_FFI_RESIDENT=1` is OFF BY DEFAULT and worth ~2×; `SHIM_WORKERS` ~6.5×. Both silent.
# ⚠ `.venv/bin/python` DOES NOT EXIST on this box. `SHIM_PYTHON` is not optional either.
# ⚠⚠ Every number here is a SYSTEM result — shim feed and f32 all-reduce included. For a statement
#   about the RENDERER alone use scripts/bf16_device_step.py.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT="${1:?usage: scripts/bf16_probe_3060.sh <out.tsv> [net ...]}"; shift || true

PLUG="${PJRT_PLUGIN:-/home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so}"
PY="${SHIM_PYTHON:-/home/skoonce/.venv-cuda/bin/python3}"
DEVS="${DEVS:-0,1,2,3}"
WORKERS="${WORKERS:-8}"
WARM="${WARM:-200}"
STEPS="${STEPS:-600}"
TAG="${CKPT_TAG:-probe3060}"
ARMS="${ARMS:-fed synth}"
PRECS="${PRECS:-f32 bf16}"   # the §1c sweep wants ONE arm, not the cross product
TIMEOUT="${TIMEOUT:-2400}"

[ -f "$PLUG" ] || { echo "⛔ plugin not found: $PLUG"; exit 1; }
[ -x "$PY" ]   || { echo "⛔ shim python not found: $PY"; exit 1; }

# ⚠⚠ A stale GPU process fakes OOM and NCCL errors. Refuse to benchmark next to one.
LEFTOVER="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
if [ -n "$LEFTOVER" ]; then
  echo "⛔ GPUs are not idle — benchmark refused. Compute apps present:"; echo "$LEFTOVER"
  echo "   (clear them first; a leftover process fakes OOM and NCCL failures)"; exit 1
fi

# net | exe | f32 variant | bf16 variant | per-replica batch | extra env
#
# vitema / enetema ARE THE TWO PRODUCTION GRAPHS -- what the job confs actually run. The other
# rows are the LIGHT variants. enet-default-4gpu.conf's own header is why both are here: EMA +
# drop-path + classifier dropout nearly double B0's step, and it warns to verify against what the
# artifact BAKES rather than a number carried forward. A shim verdict read off the light graph is
# exactly such a carried-forward number.
# vitema has NO f32 render (vitin_emadp128x4wxclipdrop is absent) -- run its bf16 arm only.
ROWS="
r34|resnet34-imagenet-verified|momdp64|momdp64bf16|64|LEAN_MLIR_BASE_LR_U=100000
r50|resnet50-imagenet-verified|momdp64|momdp64bf16|64|
mnv2|mobilenetv2-imagenet-verified|adamdp64|adamdp64bf16|64|
mnv4|mobilenetv4-imagenet-verified|adamdp64|adamdp64bf16|64|
enet|efficientnet-imagenet-verified|rmsdp64|rmsdp64bf16|64|
cnx|convnext-imagenet-verified|adamdpwxclipdrop|adamdpwxclipdropbf16|32|
vit|vit-imagenet-verified|adamdp128x4wxclipdrop|adamdp128x4wxclipdropbf16|128|
vitema|vit-imagenet-verified|adamdp128x4wxclipdrop|emadp128x4wxclipdropbf16|128|
enetema|efficientnet-imagenet-verified|emarmsdp64dropdo|emarmsdp64dropdobf16|64|
r50a3|resnet50-imagenet-verified|lambaccdp8x64wxclipbce|lambaccdp8x64wxclipbcebf16|64|LEAN_MLIR_RES=160 LEAN_MLIR_G2_STEPS=5000
"

WANT="${NETS:-$*}"

if [ ! -s "$OUT" ]; then
  printf 'net\tvariant\tprec\tarm\tbs\tworkers\tmed_ms\tmin_ms\tp90_ms\tmean_ms\twait_ms\tsteps_ep\telapsed_s\tnote\n' > "$OUT"
fi

probe () {  # net exe variant prec bs arm extra
  local net="$1" exe="$2" var="$3" prec="$4" bs="$5" arm="$6" extra="$7"
  local log; log="$(mktemp)"
  local t0=$SECONDS
  local synthenv=()
  [ "$arm" = "synth" ] && synthenv=(LEAN_MLIR_BENCH_SYNTH=1)
  # `extra` may carry SEVERAL env assignments, space separated (r50a3 needs RES and G2_STEPS).
  local extraenv=()
  [ -n "$extra" ] && read -r -a extraenv <<< "$extra"

  echo "  ▶ $net/$prec/$arm (variant $var, bs $bs, workers $WORKERS) ..."
  env CUDA_VISIBLE_DEVICES="$DEVS" PJRT_PLUGIN="$PLUG" SHIM_PYTHON="$PY" \
      PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 SHIM_WORKERS="$WORKERS" \
      LEAN_MLIR_VARIANT="$var" LEAN_MLIR_BATCH="$bs" \
      LEAN_MLIR_PROBE_WARM="$WARM" LEAN_MLIR_MAX_STEPS="$STEPS" \
      LEAN_MLIR_CKPT_TAG="${TAG}-${net}-${var}" "${synthenv[@]}" "${extraenv[@]}" \
      timeout "$TIMEOUT" ".lake/build/bin/$exe" data > "$log" 2>&1

  local med min p90 mean wait spe note
  med="$(grep -oE 'PROBE: [0-9]+ ms/step' "$log" | grep -oE '[0-9]+' | head -1)"
  min="$(sed -n 's/.*PROBE-SPREAD: min=\([0-9]*\).*/\1/p' "$log" | head -1)"
  p90="$(sed -n 's/.*PROBE-SPREAD:.*p90=\([0-9]*\).*/\1/p' "$log" | head -1)"
  mean="$(sed -n 's/.*PROBE-SPREAD:.*mean=\([0-9]*\).*/\1/p' "$log" | head -1)"
  wait="$(sed -n 's/.*starvation wait = med-min = \([0-9]*\).*/\1/p' "$log" | head -1)"
  spe="$(grep -oE 'global batch [0-9]+, [0-9]+ steps/epoch' "$log" | grep -oE '[0-9]+ steps' | grep -oE '[0-9]+' | head -1)"
  [ -z "$spe" ] && spe="$(grep -oE 'step 0/[0-9]+' "$log" | grep -oE '[0-9]+$' | head -1)"

  if [ -z "$med" ]; then
    # ⚠ grep the REAL failure. `uncaught exception` first: XLA's benign cuda_timer warning contains
    # the word "missing" ("missing warmup") and used to win this race, hiding the actual cause.
    note="$(grep -iE 'uncaught exception' "$log" | head -1 | cut -c1-200)"
    [ -z "$note" ] && note="$(grep -iE 'error|refus|mismatch|no such|Cannot|assert' "$log" | grep -viE 'cuda_timer|sub-optimal|absl::InitializeLog' | head -1 | cut -c1-200)"
    [ -z "$note" ] && note="no PROBE line (timeout? see saved log)"
    cp "$log" "runs/probe3060-FAIL-$net-$prec-$arm.log"
    med=FAIL
  else
    note=""
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$net" "$var" "$prec" "$arm" "$bs" "$WORKERS" \
    "$med" "${min:-}" "${p90:-}" "${mean:-}" "${wait:-}" "${spe:-}" "$((SECONDS-t0))" "$note" >> "$OUT"
  echo "    → ${med} ms/step (min ${min:-?}, wait ${wait:-?} ms, ${spe:-?} steps/ep, $((SECONDS-t0))s)"
  rm -f "$log"
}

echo "$ROWS" | while IFS='|' read -r net exe v32 vbf bs extra; do
  [ -z "$net" ] && continue
  if [ -n "$WANT" ]; then case " $WANT " in *" $net "*) ;; *) continue ;; esac; fi
  for arm in $ARMS; do
    for prec in $PRECS; do
      case "$prec" in
        f32)  probe "$net" "$exe" "$v32" f32  "$bs" "$arm" "$extra" ;;
        bf16) probe "$net" "$exe" "$vbf" bf16 "$bs" "$arm" "$extra" ;;
      esac
    done
  done
done

echo "ALL DONE -> $OUT"
column -t -s $'\t' "$OUT"
