#!/usr/bin/env bash
# Compute-only ms/step probe for the ConvNeXt-T/S/B and ViT-Ti/S/B phase-2 JAX
# trainers, 4× RTX 4060 Ti, bf16 as generated. Peer of the 2026-07-25 table in
# planning/vit_convnext_sb_scaleup.md, re-run on jax 0.11.0 with fresh emits.
set -u
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
PY=.venv/bin/python
OUT="$1"
: > "$OUT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STEPS=10

probe () {  # label gen batch accum
  local label="$1" gen="jax/.lake/build/$2" batch="$3" accum="$4"
  local log="${OUT%.tsv}.$label.log" t0=$SECONDS
  echo "── probing $label (batch=$batch accum=$accum) ──"
  GEN="$gen" BATCH="$batch" ACCUM="$accum" $PY jax/scripts/step_probe.py > "$log" 2>&1
  local rc=$?
  local line
  line=$(grep -m1 '^RESULT' "$log")
  echo "$label|${line:-FAIL rc=$rc}|compile+run ${SECONDS}s" >> "$OUT"
  echo "   ${line:-FAIL rc=$rc}  [$((SECONDS-t0))s]"
  sleep 5
}

probe cnx-t  generated_convnext_tiny_imagenet.py 256 1
probe cnx-s  generated_convnext_s_imagenet.py    256 1
probe cnx-b  generated_convnext_b_imagenet.py    256 1
probe vit-ti generated_vit_tiny_imagenet.py      512 1
probe vit-s  generated_vit_s_imagenet.py         512 1
probe vit-b  generated_vit_b_imagenet_accum.py   128 4
echo "ALL DONE"
