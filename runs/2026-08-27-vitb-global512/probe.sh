#!/usr/bin/env bash
# ViT-B at global 512 — does DeiT's batch fit without §4d's accumulation loop?
#
# ⚠ Run from the repo root with the PINNED venv (jax/requirements-cuda-lock.txt). Every figure
# below is FOUR REPLICAS: ViT-B has no single-device recipe, and a single-device peak would not
# contain the all-reduce this asks about.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/2026-08-27-vitb-global512
DP=verified_mlir/vitbin_adamdp128x4wxclipdrop_train_step.mlir
DPB=verified_mlir/vitbin_adamdp128x4wxclipdropbf16_train_step.mlir
S1=verified_mlir/vitbin_adam128wxclipdrop_train_step.mlir
S1B=verified_mlir/vitbin_adam128wxclipdropbf16_train_step.mlir

{
  echo "── A. peak, RAISED budget (XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 → 15.11 GiB), 4 replicas ──"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 .venv/bin/python scripts/bf16_peak_memory.py \
    --budget-gib 15.11 --replicas 4 "$DP" "$DPB"
  echo
  echo "── B. the SAME two at the DEFAULT budget (fraction unset → 11.68 GiB) ──"
  echo "   ⚠ XLA's own rematerialisation line is the interesting output here, not the table."
  .venv/bin/python scripts/bf16_peak_memory.py --replicas 4 "$DP" "$DPB"
  echo
  echo "── C. CONTROL — the all-reduce, priced by DIFFERENCE against renders that have none ──"
  echo "   The 1-replica artifacts are the same net, batch and flags with no collective in the"
  echo "   graph. If the DP peak exceeds them, that excess IS the collective's buffer."
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 .venv/bin/python scripts/bf16_peak_memory.py \
    --budget-gib 15.11 --replicas 1 "$DP" "$DPB" "$S1" "$S1B"
} 2>&1 | tee "$D/peak_memory.log"

{
  echo "── D. EXECUTION at the raised budget — compiling is not fitting ──"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 .venv/bin/python scripts/bf16_device_step.py \
    --replicas 4 --reps 10 "$DP" "$DPB"
  echo
  echo "── E. BASIS — ⚠ THIS STEP NO LONGER RUNS. It timed the 32×4 pair in the same session so"
  echo "      the wall clock would be like-for-like (386.04 / 273.79 against §4c's committed"
  echo "      383.8 / 273.7 — a 0.6 % / 0.03 % control). Those artifacts were DELETED once the"
  echo "      128×4 pair beat them on every axis, so the control survives only as this log."
  echo "      ▶ The live control is now ViT-Tiny, in F below."
} 2>&1 | tee "$D/device_step.log"

{
  echo "── F2. TRAINER ms/step, 40 steps on real ImageNet, both precisions, and ViT-Tiny as the"
  echo "      session control. This is what the book's schedule table quotes. ⚠ Needs the dataset."
  for V in adamdp128x4wxclipdrop adamdp128x4wxclipdropbf16; do
    for EXE in vit-b-imagenet-verified vit-imagenet-verified; do
      echo "-- $EXE $V"
      CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
        LEAN_MLIR_MEM_FRACTION=0.97 LEAN_MLIR_VARIANT=$V LEAN_MLIR_BATCH=128 \
        PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
        LEAN_MLIR_MAX_STEPS=40 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_CKPT_TAG=_probe \
        .lake/build/bin/$EXE data 2>&1 | grep -E "PROBE:"
    done
  done
  rm -f .lake/build/vit*_ckpt_xla_probe.bin*
} 2>&1 | tee "$D/trainer_probe.log"

{
  echo "── F. THE CONTROL THAT MAKES THIS A FINDING — the same bytes, the same four cards, and"
  echo "      the ONLY difference is that XLA_PYTHON_CLIENT_MEM_FRACTION is not set. ──"
  echo
  echo "\$ .venv/bin/python scripts/bf16_device_step.py --replicas 4 --reps 3 $DP"
  .venv/bin/python scripts/bf16_device_step.py --replicas 4 --reps 3 "$DP"
  echo
  echo "── G. and bf16 at the DEFAULT budget, which is the OTHER half of the answer: it fits there"
  echo "      on its own (10.88 of 11.68), so bf16 alone would have bought global 512 in bf16 ONLY."
  echo "      §4c's rule — a tier without its precision peer reads as a decision — needs the fp32"
  echo "      twin, and the fp32 twin is what the allocator fix buys. ──"
  .venv/bin/python scripts/bf16_device_step.py --replicas 4 --reps 5 "$DPB"
} 2>&1 | tee "$D/oom_control.log"
