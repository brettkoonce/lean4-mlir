#!/usr/bin/env bash
# score_timm.sh — score a VERIFIED checkpoint under timm's validation protocol.
#
#     LEAN_MLIR_VARIANT=<v> [LEAN_MLIR_CKPT=… | LEAN_MLIR_CKPT_TAG=…] [LEAN_MLIR_REPLICAS=4 …] \
#       scripts/score_timm.sh <net> [dataDir]
#
# <net> is a `score-checkpoint` name (mnv4-in, convnext-in, mobilenetv2-in, …). The test size and
# crop come from jax/timm_eval_protocols.json — the table `jax/scripts/eval_full50k.py
# PROTOCOL=timm` reads for the JAX path, generated from the pinned timm by
# scripts/timm_eval_protocols.py — and are handed to score-checkpoint as LEAN_MLIR_EVAL_SIZE /
# LEAN_MLIR_EVAL_CROP. A size other than the net's own needs its eval graph rendered at that size
# (`<slug>_fwd_eval_s<S>.mlir` / `<slug>_fwd_s<S>.mlir`); score-checkpoint refuses when it is absent.
set -euo pipefail
cd "$(dirname "$0")/.."
net="${1:?usage: LEAN_MLIR_VARIANT=<v> scripts/score_timm.sh <net> [dataDir]}"
data="${2:-data}"
: "${LEAN_MLIR_VARIANT:?LEAN_MLIR_VARIANT names the render that wrote the checkpoint}"

# score-checkpoint net -> the generated-trainer stem its recipe is (jax/timm_eval_protocols.json keys)
case "$net" in
  mnv4-in)         key=mobilenet_v4_imagenet ;;
  mobilenetv2-in)  key=mobilenet_v2_imagenet ;;
  efficientnet-in) key=efficientnet_b0_imagenet ;;
  convnext-in)     key=convnext_tiny_imagenet ;;
  convnexts-in)    key=convnext_s_imagenet ;;
  convnextb-in)    key=convnext_b_imagenet ;;
  vit-in)          key=vit_tiny_imagenet ;;
  vits-in)         key=vit_s_imagenet ;;
  vitb-in)         key=vit_b_imagenet ;;
  resnet34-in)     key=resnet34_imagenet ;;
  resnet50-in160)  key=resnet50_imagenet_short ;;          # RSB-A3
  resnet50-in)     case "$LEAN_MLIR_VARIANT" in             # RSB-A2 renders are the BCE ones
                     *bce*) key=resnet50_imagenet ;;
                     *)     key=resnet50_imagenet_2018 ;;
                   esac ;;
  *) echo "⛔ no timm protocol mapping for '$net'"; exit 1 ;;
esac

read -r size crop tag < <(python3 - "$key" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
from timm_eval_protocols import lookup
t = lookup(json.load(open("jax/timm_eval_protocols.json")), sys.argv[1])
if t is None: sys.exit(f"no protocol for {sys.argv[1]}")
print(t["test_size"], t["test_crop_pct"], t["timm"])
EOF
)
echo "▸ timm protocol for $net: $tag → ${size}px, crop_pct $crop"
exec env LEAN_MLIR_EVAL_SIZE="$size" LEAN_MLIR_EVAL_CROP="$crop" \
  .lake/build/bin/score-checkpoint "$net" "$data"
