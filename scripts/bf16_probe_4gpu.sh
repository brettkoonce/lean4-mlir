#!/usr/bin/env bash
# 4-GPU steady-state ms/step probe for EVERY ImageNet trainer, f32 and bf16 arm.
#
#     scripts/bf16_probe_4gpu.sh /tmp/out.tsv
#
# Produces the table in `planning/bf16_renderer.md` §21 — per-net ms/step at 4x, from which the
# end-to-end cost of a full run is `steps/epoch * epochs * ms/step + 37.5 s/epoch` (eval+ckpt,
# measured on R34; the 30 GB val drain is ONE-TIME, not per-epoch).
#
# DEVS per scripts/jobs/*-4gpu.conf: the AER-clean four (idx 1 and 5 excluded, reference_ares_pcie_aer).
#
# ⚠ `LEAN_MLIR_CKPT_TAG` is NOT optional. Without it a finished run's checkpoint makes the probe
#   exit instantly at "resuming from checkpoint at epoch 90" and report nothing.
# ⚠ `PJRT_FFI_RESIDENT=1` is OFF BY DEFAULT and worth ~1.9x on a big-parameter net (§16.2).
# ⚠ `SHIM_WORKERS=8` matters more than the emit on the light nets: MNv2 reads 1.51x here against a
#   committed 1.37x (§12.1) purely because that probe left it unset (§21.3).
# ⚠⚠ Every number this prints is a SYSTEM result — shim feed and f32 all-reduce included (§13.2).
#   For a statement about the RENDERER use scripts/bf16_device_step.py instead.
#
# ✅ mnv4's DP pair was TIED on 2026-08-27 (`mnv4-dp-check` + `shard-check mnv4in`, both green,
#   both controls red — runs/2026-08-27-mnv4-dp-shard-gates/). §21.2's "cost it, don't train it"
#   caveat is lifted; scripts/jobs/mnv4-default-4gpu.conf trains off exactly this artifact.
cd "$(dirname "$0")/.."
OUT="${1:?usage: scripts/bf16_probe_4gpu.sh <out.tsv>}"
: > "$OUT"
PLUG=".venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"

probe () {  # name exe variant batch extra...
  local name="$1" exe="$2" var="$3" bs="$4"; shift 4
  local log; log="$(mktemp)"
  local t0=$SECONDS
  env CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_PLUGIN="$PLUG" \
      PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
      LEAN_MLIR_VARIANT="$var" LEAN_MLIR_BATCH="$bs" \
      LEAN_MLIR_MAX_STEPS=40 LEAN_MLIR_CKPT_TAG=probe4gpu "$@" \
      timeout 1500 ".lake/build/bin/$exe" data > "$log" 2>&1
  local ms spe
  ms="$(grep -oE 'PROBE: [0-9]+ ms/step' "$log" | grep -oE '[0-9]+' | head -1)"
  spe="$(grep -oE 'step 0/[0-9]+' "$log" | grep -oE '[0-9]+$' | head -1)"
  if [ -z "$ms" ]; then
    echo "$name|$var|$bs|FAIL|$spe|$((SECONDS-t0))|$(grep -iE 'error|refus|mismatch|missing' "$log" | head -1 | cut -c1-120)" >> "$OUT"
  else
    echo "$name|$var|$bs|$ms|$spe|$((SECONDS-t0))|" >> "$OUT"
  fi
  rm -f "$log"
  echo "  done: $name $var -> ${ms:-FAIL} ms/step (${spe:-?} steps/epoch)"
}

probe r34  resnet34-imagenet-verified      momdp64                     64  LEAN_MLIR_BASE_LR_U=100000
probe r34  resnet34-imagenet-verified      momdp64bf16                 64  LEAN_MLIR_BASE_LR_U=100000
probe r50  resnet50-imagenet-verified      momdp64                     64
probe r50  resnet50-imagenet-verified      momdp64bf16                 64
probe mnv2 mobilenetv2-imagenet-verified   adamdp64                    64
probe mnv2 mobilenetv2-imagenet-verified   adamdp64bf16                64
probe cnx  convnext-imagenet-verified      adamdpwxclipdrop            32
probe cnx  convnext-imagenet-verified      adamdpwxclipdropbf16        32
probe enet efficientnet-imagenet-verified  rmsdp64                     64
probe enet efficientnet-imagenet-verified  rmsdp64bf16                 64
probe mnv4 mobilenetv4-imagenet-verified          adamdp64                    64
probe mnv4 mobilenetv4-imagenet-verified          adamdp64bf16                64
probe vit  vit-imagenet-verified           adamdp128x4wxclipdrop      128
probe vit  vit-imagenet-verified           adamdp128x4wxclipdropbf16  128
echo "ALL DONE"
