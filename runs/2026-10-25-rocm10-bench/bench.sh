#!/usr/bin/env bash
# 7.2.4 vs ROCm 10.1 (pip wheels vs apt debs), ms/step on one 7900 XTX.
# LEAN_MLIR_MAX_STEPS=100 (warm-up 8), bench checkpoint tag, own MIOpen db per arm.
cd /home/skoonce/lean/lean4-mlir
R=$PWD/runs/2026-10-25-rocm10-bench
P10=$PWD/.venv-rocm10/lib/python3.12/site-packages/jax_plugins/xla_rocm10/xla_rocm_plugin.so
APT10=/opt/rocm-7.2.4/core-10.1/lib
SW="MIOPEN_DEBUG_CONV_GEMM=0 MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS=0 MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS=0 MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS=0"
export PATH=/opt/rocm/bin:$PATH
( while :; do echo "$(date +%T) $(rocm-smi --showtemp --showpower --showuse --csv 2>/dev/null | sed -n 2p)"; sleep 1; done ) > $R/sensors.log 2>&1 &
SP=$!; trap "kill $SP" EXIT
for net in resnet34 mobilenetv2 vit efficientnet convnext resnet50 mobilenetv4; do
  for arm in 724 pip101 pip101sw apt101 apt101sw; do
    mkdir -p $R/$arm/miopen
    env=(LEAN_MLIR_MAX_STEPS=100 LEAN_MLIR_CKPT_TAG=bench RUN_LOG_DIR=$R/$arm
         MIOPEN_USER_DB_PATH=$R/$arm/miopen MIOPEN_CUSTOM_CACHE_DIR=$R/$arm/miopen)
    case $arm in
      pip101)   env+=(PJRT_PLUGIN=$P10) ;;
      pip101sw) env+=(PJRT_PLUGIN=$P10 $SW) ;;
      apt101)   env+=(PJRT_PLUGIN=$P10 LD_LIBRARY_PATH=$APT10) ;;
      apt101sw) env+=(PJRT_PLUGIN=$P10 LD_LIBRARY_PATH=$APT10 $SW) ;;
    esac
    t0=$(date +%s)
    env "${env[@]}" timeout 1200 ./run.sh $net-verified-adam 0 rocm > /dev/null 2>&1; rc=$?
    r=$(grep -oE 'PROBE: [0-9.]+ ms/step' $R/$arm/$net-verified-adam.log | head -1)
    echo -e "$net\t$arm\t${r:-FAIL rc=$rc}\t$(( $(date +%s)-t0 ))s\t$(date +%T)" | tee -a $R/results.tsv
  done
done
