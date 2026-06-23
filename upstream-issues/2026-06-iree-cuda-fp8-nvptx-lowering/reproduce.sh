#!/bin/bash
# f8 fails on CUDA, works on CPU; fp32 works on CUDA.
# sm_86 reproduces on any recent iree-base-compiler. (sm_89 also fails the same
# way on rc20260623+, which is the first to accept --iree-cuda-target=sm_89.)
set -x
# FAIL — f8 convert on CUDA (unrealized_conversion_cast in LLVM translation):
iree-compile convert_f8_to_f32.mlir --iree-hal-target-backends=cuda --iree-cuda-target=sm_86 -o /dev/null
# FAIL — identical even with the small-float emulation flag:
iree-compile convert_f8_to_f32.mlir --iree-hal-target-backends=cuda --iree-cuda-target=sm_86 \
  --iree-llvmgpu-enable-small-float-emulation -o /dev/null
# FAIL — the f8 GEMM:
iree-compile gemm_f8_f32acc.mlir --iree-hal-target-backends=cuda --iree-cuda-target=sm_86 -o /dev/null
# OK — same f8 convert on llvm-cpu:
iree-compile convert_f8_to_f32.mlir --iree-hal-target-backends=llvm-cpu -o /dev/null
# OK — fp32 GEMM on CUDA (target + backend are fine; only f8 is broken):
sed 's/f8E4M3FN/f32/g' gemm_f8_f32acc.mlir | iree-compile - --iree-hal-target-backends=cuda --iree-cuda-target=sm_86 -o /dev/null
