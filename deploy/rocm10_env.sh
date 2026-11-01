# deploy/rocm10_env.sh — source from the repo root to train on ROCm 10.1 (the wheels
# deploy/rocm10_setup.sh puts in .venv-rocm10) instead of the pinned 7.2.4:
#
#   . deploy/rocm10_env.sh && ./run.sh resnet50-verified-adam 0 rocm
#
# Per process: nothing here touches /opt/rocm or the 7.2.4 .venv, so both stacks live on one box.

_r10=$PWD/.venv-rocm10
PJRT_PLUGIN=$(ls "$_r10"/lib/python3*/site-packages/jax_plugins/xla_rocm10/xla_rocm_plugin.so 2>/dev/null | head -1)
if [ -z "$PJRT_PLUGIN" ]; then
  echo "rocm10_env.sh: no plugin under $_r10 — run deploy/rocm10_setup.sh first" >&2
else
  export PJRT_PLUGIN
  # The backend by name: a wheels-only box has no rocm-smi for scripts/platform/env.sh to find.
  export PLATFORM_BACKEND=rocm
  export PATH=$_r10/bin:$PATH                       # rocminfo, amd-smi from the wheels

  # MIOpen on gfx1100 under 10.1 — required, not tuning (deploy/ROCM_7900XTX.md §1, §3):
  export MIOPEN_DEBUG_CONV_GEMM=0                   # im2col GEMM solvers: ViT's patch-embed wgrad
                                                    #   hits MIOpenIm2d2Col, which fails under HIPRTC
  export MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS=0   # grouped implicit-GEMM fwd:
  export MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS=0   #   bf16 garbage on gfx1100,
  export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS=0   #   which has no XDLOPS

  # Keep 10.1's tuned-kernel database apart from 7.2.4's (MIOpen 3.6 vs 3.5).
  export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen-rocm10.1}
  export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}
  mkdir -p "$MIOPEN_USER_DB_PATH"
  echo "ROCm 10.1: $PJRT_PLUGIN (MIOpen switches on, db $MIOPEN_USER_DB_PATH)"
fi
unset _r10
