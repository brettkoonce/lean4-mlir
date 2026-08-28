#!/bin/bash
# Build the VisDrone detector for a Jetson Orin (or CPU) with IREE.
#
# Run this on the BUILD box, not the device — iree-compile is a host tool and
# cross-compiles. Then copy deploy/ + build/ to the Orin and run orin_detect.py.
#
#   ./build_orin.sh                 # cuda sm_87 (Orin)
#   ./build_orin.sh cpu             # llvm-cpu aarch64 (Orin CPU, Pi5, ...)
#
# ⚠ sm_87 is Orin. Do not copy the sm_86 workaround from the RTX-40 notes here —
# that exists because IREE 3.11 mis-handles sm_89, and is unrelated.
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-cuda}"
REPO="$(cd .. && pwd)"
OUT="build"
mkdir -p "$OUT"

# The trained arm to ship. FPN_TAG must match the arm you actually want —
# getting it wrong silently ships a different arm's weights, which is the trap
# that cost this thread a day. The emit prints the prefix it used; read it.
ARM="${FPN_TAG:-ctrl12}"
PREFIX="$REPO/.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__${ARM}"

echo "== 1. emit the batch-1 forward graph =="
# Training artifacts are emitted at batch 8; a camera runs one frame at a time.
( cd "$REPO" && CUDA_VISIBLE_DEVICES="" FPN_BACKBONE=r34 FPN_TAG="$ARM" \
    ./.lake/build/bin/yolov1-visdrone-fpn emit-deploy deploy/build )

echo "== 2. stage the weights =="
for f in params bn_stats; do
  src="${PREFIX}_${f}.bin"
  [ -f "$src" ] || { echo "MISSING $src — is FPN_TAG=$ARM right, and did that arm finish?"; exit 1; }
  cp "$src" "$OUT/${f}.bin"
  echo "  $(basename "$src") -> $OUT/${f}.bin  ($(stat -c%s "$src") bytes)"
done

echo "== 3. compile =="
MLIR="$OUT/detector_fwd_eval_b1.mlir"
# iree-compile is often not on PATH on this box; override with $IREE_COMPILE.
IREE_COMPILE="${IREE_COMPILE:-$(command -v iree-compile || true)}"
if [ -z "$IREE_COMPILE" ]; then
  for c in "$HOME/lean4-mlir-job/iree-bin/iree-compile" \
           "$HOME/lean4-mlir/.venv/bin/iree-compile"; do
    [ -x "$c" ] && IREE_COMPILE="$c" && break
  done
fi
[ -n "$IREE_COMPILE" ] || { echo "no iree-compile found; set \$IREE_COMPILE"; exit 1; }
echo "  using $IREE_COMPILE"
case "$TARGET" in
  cuda)
    "$IREE_COMPILE" "$MLIR" \
      --iree-hal-target-backends=cuda \
      --iree-cuda-target=sm_87 \
      -o "$OUT/detector.vmfb"
    ;;
  cpu)
    "$IREE_COMPILE" "$MLIR" \
      --iree-hal-target-backends=llvm-cpu \
      --iree-llvmcpu-target-triple=aarch64-none-linux-gnu \
      -o "$OUT/detector.vmfb"
    ;;
  *) echo "unknown target '$TARGET' (want: cuda | cpu)"; exit 1;;
esac

echo
echo "built $OUT/detector.vmfb ($(stat -c%s "$OUT/detector.vmfb") bytes)"
echo
echo "Copy to the device and run:"
echo "  scp -r deploy/ orin:~/detector/"
echo "  ssh orin 'cd ~/detector && python3 orin_detect.py --image frame.jpg --bench 50'"
