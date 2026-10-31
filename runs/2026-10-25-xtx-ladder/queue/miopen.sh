#!/bin/bash
# After ladder.sh: ROCm 7.2.4 from an EMPTY MIOpen database/cache, as a fresh install has it —
# the suite as is, the suite with MIOPEN_DEBUG_CONV_GEMM=0, and ViT's step time with it.
cd /home/skoonce/lean/lean4-mlir
while pgrep -f 'xtx-queue/(imagenette|ladder).sh' >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') break 15 min"; sleep 900
export PATH=/opt/rocm/bin:$PATH PLATFORM_BACKEND=rocm
R=runs/2026-10-25-rocm10-bench/cold724; mkdir -p $R
since=$(date '+%Y-%m-%d %H:%M:%S')
dropped() { journalctl -k --since "$since" --no-pager 2>/dev/null | grep -qE 'device lost from bus|amdgpu.*(ring .* timeout|GPU reset)'; }
fresh() { rm -rf $R/miopen-$1; mkdir -p $R/miopen-$1; export MIOPEN_USER_DB_PATH=$PWD/$R/miopen-$1 MIOPEN_CUSTOM_CACHE_DIR=$PWD/$R/miopen-$1; }
( fresh default; timeout 1500 scripts/platform/check.sh --backend rocm --out runs/platform/2026-10-26-minerva-rocm-coldmiopen > $R/suite-default.log 2>&1 )
echo "=== $(date '+%F %T') suite cold default: $(grep -E '[0-9]+ pass,' $R/suite-default.log | tail -1)"; dropped && { echo DROP; exit 2; }
( fresh nogemm; export MIOPEN_DEBUG_CONV_GEMM=0; timeout 1500 scripts/platform/check.sh --backend rocm --out runs/platform/2026-10-26-minerva-rocm-coldmiopen-nogemm > $R/suite-nogemm.log 2>&1 )
echo "=== $(date '+%F %T') suite cold CONV_GEMM=0: $(grep -E '[0-9]+ pass,' $R/suite-nogemm.log | tail -1)"; dropped && { echo DROP; exit 2; }
for v in nogemm warm; do
  ( [ $v = nogemm ] && { fresh vit; export MIOPEN_DEBUG_CONV_GEMM=0; }
    export LEAN_MLIR_MAX_STEPS=100 LEAN_MLIR_CKPT_TAG=bench RUN_LOG_DIR=$PWD/$R/vit-$v
    timeout 900 ./run.sh vit-verified-adam 0 rocm > /dev/null 2>&1 )
  echo "=== $(date '+%F %T') vit $v: $(grep -ohE 'PROBE: [0-9]+ ms/step|get_global_id' $R/vit-$v/*.log | sort -u | tr '\n' ' ')"
done
echo "=== $(date '+%F %T') ALLDONE"
