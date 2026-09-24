# Sourced by the GPU gates and launchers, from the repo root.

# det_shim_ensure DIR LOG [INDENT] — make DIR hold a current deterministic PJRT shim
# (scripts/det_shim.sh: autotuning off, deterministic ops), building it only when it is missing
# or older than the shim source or its builder. A stale shim reused silently would compare a
# different executable than the one the gate claims to test. Returns 2 on a failed build.
det_shim_ensure() {
  local dir="$1" log="$2" ind="${3:-}"
  if [ -f "$dir/libpjrt_ffi.so" ] && [ ! ffi/pjrt_ffi.c -nt "$dir/libpjrt_ffi.so" ] \
     && [ ! scripts/det_shim.sh -nt "$dir/libpjrt_ffi.so" ]; then
    return 0
  fi
  echo "${ind}building the deterministic shim in $dir ..."
  scripts/det_shim.sh "$dir" > "$log" 2>&1 || {
    echo "${ind}✗ det_shim.sh failed:"; cat "$log"; return 2; }
}

# gpu_busy [GPU] — true when any compute process holds GPU (every GPU when none is given).
# A leftover process fakes OOM and NCCL failures, and a shared card halves a timing.
gpu_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader ${1:+-i "$1"} 2>/dev/null | grep -q .
}
