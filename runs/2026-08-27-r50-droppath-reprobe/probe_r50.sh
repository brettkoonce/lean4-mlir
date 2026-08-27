#!/usr/bin/env bash
# Compute-only ms/step probe for the RSB tiers' phase-2 JAX trainers, 4× RTX 4060 Ti,
# bf16, effective batch 2048 as 4×512 grad-accum. A3 (train@160) is the control the
# book already has a verified wall clock for; A2 is the unmeasured tier, and A1 is
# A2's three constants at 2× the epochs, so one probe prices both.
set -u
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
PY=.venv/bin/python
OUT="$1"; : > "$OUT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STEPS=10

probe () {
  local label="$1" gen="jax/.lake/build/$2" batch="$3" accum="$4"
  local log="${OUT%.tsv}.$label.log" t0=$SECONDS
  echo "── probing $label (batch=$batch accum=$accum)"
  GEN="$gen" BATCH="$batch" ACCUM="$accum" $PY -u jax/scripts/step_probe.py > "$log" 2>&1
  local rc=$? line
  line=$(grep -m1 '^RESULT' "$log")
  echo "$label|${line:-FAIL rc=$rc}|$((SECONDS-t0))s" >> "$OUT"
  sleep 5
}

probe r50-a3 generated_resnet50_imagenet_rsbfaithful.py 512 4
probe r50-a2 generated_resnet50_imagenet_a2accum.py     512 4
echo "ALL DONE" >> "$OUT"
