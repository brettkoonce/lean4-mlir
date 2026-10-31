#!/usr/bin/env bash
# Full platform suite (tiers 0-2) per ROCm arm, after bench.sh.
cd /home/skoonce/lean/lean4-mlir
while pgrep -f rocm10-bench/bench.sh >/dev/null; do sleep 10; done
sleep 900   # cool-down: no GPU stretch over an hour on this box
export PATH=/opt/rocm/bin:$PATH
P10=$PWD/.venv-rocm10/lib/python3.12/site-packages/jax_plugins/xla_rocm10/xla_rocm_plugin.so
SW="MIOPEN_DEBUG_CONV_GEMM=0 MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS=0 MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS=0"
O=runs/platform/2026-10-25-minerva
for arm in 724 pip101 apt101 pip101sw; do
  M=$PWD/runs/2026-10-25-rocm10-bench/suite-miopen-$arm; mkdir -p $M
  env=(MIOPEN_USER_DB_PATH=$M MIOPEN_CUSTOM_CACHE_DIR=$M)
  case $arm in
    724)      out=$O-rocm-replug ;;
    pip101)   out=$O-rocm10pip; env+=(PJRT_PLUGIN=$P10) ;;
    apt101)   out=$O-rocm10apt; env+=(PJRT_PLUGIN=$P10 LD_LIBRARY_PATH=/opt/rocm-7.2.4/core-10.1/lib) ;;
    pip101sw) out=$O-rocm10pip-nogemm; env+=(PJRT_PLUGIN=$P10 $SW) ;;
  esac
  rm -rf $out
  env "${env[@]}" timeout 3600 scripts/platform/check.sh --backend rocm --out $out > $PWD/runs/2026-10-25-rocm10-bench/suite-$arm.log 2>&1
  echo "SUITE $arm: $(grep -E "[0-9]+ pass," runs/2026-10-25-rocm10-bench/suite-$arm.log | tail -1)"
  sleep 60
done
