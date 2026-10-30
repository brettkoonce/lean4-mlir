#!/bin/bash
# flags_next.sh — on the instance: the compiler-flag sweep warmed on atlas 2026-10-30
# (/var/tmp/neuron-flags-merged.tar.gz, 18 graphs). Two lanes, one per NeuronCore: ViT on core 0, ResNet-34 on core 1.
# Each run: clear that net's checkpoint (trainers resume regardless of flags), then smoke.sh (300-step epoch + 160-step probe).
# Setup first:  tar -C /var/tmp -xzf neuron-flags-merged.tar.gz && cp -a /var/tmp/neuron-flags-merged/. /var/tmp/neuron-compile-cache/
# Flag strings must match exactly (they are hashed into the cache key).
cd ~/lean4-mlir
run () { local core=$1 ck=$2; shift 2; mkdir -p ~/old-ckpt; mv .lake/build/${ck}*_ckpt_xla.bin ~/old-ckpt/ 2>/dev/null; ~/smoke.sh $core "$@"; }
( run 0 vit_adam vit-O1     vit-verified-adam "NEURON_CC_FLAGS=--auto-cast=none -O1"
  run 0 vit_adam vit-O3     vit-verified-adam "NEURON_CC_FLAGS=--auto-cast=none -O3"
  run 0 vit_adam vit-xf     vit-verified-adam "NEURON_CC_FLAGS=--auto-cast=none --model-type=transformer"
  run 0 vit_adam vit-xfbf16 vit-verified-adam "NEURON_CC_FLAGS=--auto-cast=matmult --auto-cast-type=bf16 --model-type=transformer"
  run 0 vit_adam vit-bf16O3 vit-verified-adam "NEURON_CC_FLAGS=--auto-cast=matmult --auto-cast-type=bf16 -O3" ) &
( run 1 resnet34_adam16 r34-b16-O1     resnet34-verified-adam LEAN_MLIR_VARIANT=adam16 LEAN_MLIR_BATCH=16 "NEURON_CC_FLAGS=--auto-cast=none -O1"
  run 1 resnet34_adam16 r34-b16-O3     resnet34-verified-adam LEAN_MLIR_VARIANT=adam16 LEAN_MLIR_BATCH=16 "NEURON_CC_FLAGS=--auto-cast=none -O3"
  run 1 resnet34_adam16 r34-b16-bf16O3 resnet34-verified-adam LEAN_MLIR_VARIANT=adam16 LEAN_MLIR_BATCH=16 "NEURON_CC_FLAGS=--auto-cast=matmult --auto-cast-type=bf16 -O3"
  # the book's batch-32 f32 graph: fails neuronx-cc at -O2 (IBIR243), compiles at -O1 — does it load under the DMA cap?
  run 1 resnet34_adam r34-b32-O1       resnet34-verified-adam "NEURON_CC_FLAGS=--auto-cast=none -O1" ) &
wait; echo ALL DONE
