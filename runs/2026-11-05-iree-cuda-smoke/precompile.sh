#!/bin/bash
# Pre-compile the Imagenette tier's 19 artifacts for IREE cuda (sm_86, ireeCompileArgs' default) into
# the trainers' cache names (.lake/build/<stem>_cuda.vmfb; compileVmfb reuses any cache newer than its
# .mlir), four at a time, so the smokes below pay the GPU steps and not a serial compile each.
# One line per artifact in compile.tsv: stem, status, wall seconds, vmfb bytes.
cd "$(dirname "$0")/../.."
OUT=runs/2026-11-05-iree-cuda-smoke
STEMS="resnet34_adam_train_step resnet34_fwd resnet34_fwd_eval resnet50_adam_train_step resnet50_fwd resnet50_fwd_eval
mobilenetv2_adam_train_step mobilenetv2_fwd mobilenetv2_fwd_eval mnv4_adam_train_step mnv4_fwd mnv4_fwd_eval
efficientnet_adam_train_step efficientnet_fwd efficientnet_fwd_eval convnext_adam_train_step convnext_fwd vit_adam_train_step vit_fwd"
one () {
  s=$1; t0=$(date +%s.%N)
  if nice -n 19 .venv-iree/bin/iree-compile verified_mlir/$s.mlir --iree-hal-target-backends=cuda --iree-cuda-target=sm_86 \
       -o .lake/build/${s}_cuda.vmfb 2> $OUT/compile_$s.err; then st=OK; else st=FAIL; fi
  t1=$(date +%s.%N)
  printf '%s\t%s\t%.1f\t%s\n' "$s" "$st" "$(echo "$t1 - $t0" | bc)" "$(stat -c %s .lake/build/${s}_cuda.vmfb 2>/dev/null || echo 0)" >> $OUT/compile.tsv
  [ $st = OK ] && rm -f $OUT/compile_$s.err
}
printf '# iree-compile 3.13.0 (.venv-iree), --iree-hal-target-backends=cuda --iree-cuda-target=sm_86, ares, %s; four compiles at a time, nice 19\nstem\tstatus\twall_s\tvmfb_bytes\n' "$(date +%F)" > $OUT/compile.tsv
export -f one; export OUT
echo $STEMS | tr ' ' '\n' | xargs -P 4 -I{} bash -c 'one {}'
echo "precompile done: $(grep -c OK $OUT/compile.tsv) OK, $(grep -c FAIL $OUT/compile.tsv) FAIL"
