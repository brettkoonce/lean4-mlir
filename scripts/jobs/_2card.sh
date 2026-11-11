# _2card.sh — turn a four-card job conf into the same job on two large cards. Sourced after the base
# conf by each `*-2card.conf`:
#
#     . scripts/jobs/<base>.conf
#     . scripts/jobs/_2card.sh
#     two_card <slug> <variant> <per-replica batch> <all-reduce count>   # verified
#     two_card_jax                                                       # JAX reference
#
# The micro-batch, the effective batch and the sync-BN group are the base job's; only the split
# across cards changes. SHIM_WORKERS can be set from outside once `lake run bench feed` has a number.
two_card_jax() { DEVS="0,1"; }

two_card() {
  TWO_SLUG=$1; TWO_VAR=$2; TWO_BATCH=$3; TWO_AR=$4
  DEVS="0,1"; VARIANT=$TWO_VAR
  local out=() kv
  for kv in "${ENV_EXTRA[@]}"; do
    case "$kv" in
      PJRT_REPLICAS=*)      out+=(PJRT_REPLICAS=2) ;;
      LEAN_MLIR_REPLICAS=*) out+=(LEAN_MLIR_REPLICAS=2) ;;
      LEAN_MLIR_VARIANT=*)  out+=("LEAN_MLIR_VARIANT=$TWO_VAR") ;;
      LEAN_MLIR_BATCH=*)    out+=("LEAN_MLIR_BATCH=$TWO_BATCH") ;;
      LEAN_MLIR_MEM_FRACTION=*) ;;                          # a 16 GB card's setting
      SHIM_WORKERS=*)       out+=("SHIM_WORKERS=${SHIM_WORKERS:-${kv#*=}}") ;;
      *)                    out+=("$kv") ;;
    esac
  done
  ENV_EXTRA=("${out[@]}")
  CKPT_EPOCH_FILE=".lake/build/${TWO_SLUG}_${TWO_VAR}_ckpt_xla.bin.epoch"
  PRECHECK=precheck_two_card
}

precheck_two_card() {
  . scripts/lib/precheck.sh
  local ok=0
  pc_render "$TWO_SLUG" "$TWO_VAR" 2 "$TWO_AR" || ok=1
  pc_pjrt_so || ok=1
  pc_gpu_idle || ok=1
  return $ok
}
