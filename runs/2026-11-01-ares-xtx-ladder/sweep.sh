#!/bin/bash
# The rest of minerva's 7900 XTX ladder, replayed on ares (ROCm 10.1 wheels): MNIST, gradcheck,
# CIFAR, the five Imagenette nets minerva finished, the bbn seeds and the width grid. Straight
# through (no minerva-style blocks); stops on a GPU drop. Waits for the R50 run already going.
cd /home/skoonce/lean/lean4-mlir
while pgrep -f 'bin/resnet50-verified-adam' >/dev/null; do sleep 30; done
. deploy/rocm10_env.sh
export HIP_VISIBLE_DEVICES=0 LEAN_MLIR_CKPT_TAG=ares-xtx80
L=runs/2026-11-01-ares-xtx-ladder; D=$(date -u +%F)
say() { echo "=== $(date '+%F %H:%M:%S') $*"; }
since=$(date '+%Y-%m-%d %H:%M:%S')
dropped() { journalctl -k --since "$since" --no-pager 2>/dev/null | grep -qE 'device lost from bus|amdgpu.*(ring .* timeout|GPU reset)|AER: .*(Uncorrectable|Fatal)'; }
step() {  # name, cmd
  say "START $1"; bash -c "$2" > $L/sweep-$1.out 2>&1; local rc=$?; sync
  say "END $1 rc=$rc"
  if dropped; then journalctl -k --since "$since" --no-pager > $L/kernel-drop.txt; say "GPU DROP — stopping"; exit 2; fi
}
git log --oneline -1 > $L/commit.txt; git status --short | grep -v '^??' >> $L/commit.txt
step build "lake build mnist-linear-verified mnist-mlp-verified mnist-cnn-verified cifar8w-ablation cifar8w-bn-ablation cifar8wb-bbn-ablation cifar8-bbn-gradcheck fwd-tie cifar8wb-bbn-grid resnet34-verified-adam mobilenetv2-verified-adam vit-verified-adam efficientnet-verified-adam convnext-verified-adam"
step mnist "lake run mnist"
step gradcheck ".lake/build/bin/cifar8-bbn-gradcheck"
step cifar "lake run cifar"
for net in resnet34 mobilenetv2 vit efficientnet convnext; do
  step $net "RUN_LOG_DIR=runs/$D-ares-xtx-$net-verified-adam ./run.sh $net-verified-adam 0 rocm"
done
step seeds "SWEEP_DIR=runs/$D-ares-xtx-bbn-seeds SUITE=bbn scripts/sweeps/seed_sweep.sh 0 '1 2 3 4 5'"
step grid "GPUS=0 scripts/sweeps/run_cifar8wb_bbn_grid_sweep.sh 25 runs/$D-ares-xtx-cifar8wb-bbn-grid-xla"
say "ALLDONE"
