#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  gen_shims.sh — generate the per-net ImageNet batch shims (handoff §0.2 ▶1, the SHIM WIRING).
#
#  One shim per verified ImageNet net, each from ITS OWN reference `TrainConfig`, so the
#  augmentation the verified trainer streams is the augmentation that net's JAX reference trains
#  on. `VerifiedNet.shimScript` names the file; `spawnShim` REFUSES if it is absent, rather than
#  falling back — the fallback used to be ResNet-34's and it silently gave every net RRC+hflip.
#
#  This is the ONE writer of these files (the double-writer rule, applied to the data path).
#  Run it after touching `JaxCodegen.generateShim` or any `Main*Imagenet.lean` config.
#
#  Gate it with `scripts/shim_wiring_gate.py`, which reads the configs and the generated Python
#  independently and requires the augmentation PARTITION to agree net for net.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

# exe (in jax/) → the file it writes under jax/.lake/build/. The right-hand column is what
# `VerifiedNet.shimScript` must spell; `shim_wiring_gate.py` checks that pairing both ways.
EXES=(
  "resnet34-imagenet:generated_resnet34_imagenet_shim.py"
  "vit-tiny-imagenet:generated_vit_tiny_imagenet_shim.py"
  "mobilenet-v2-imagenet:generated_mobilenet_v2_imagenet_shim.py"
  "efficientnet-b0-imagenet:generated_efficientnet_b0_imagenet_shim.py"
  "convnext-tiny-imagenet:generated_convnext_tiny_imagenet_shim.py"
)

echo "── generating $(( ${#EXES[@]} )) per-net ImageNet shims (recipe 'default') ──"
for e in "${EXES[@]}"; do
  exe="${e%%:*}"; want="${e##*:}"
  # stderr is swallowed unless the build FAILS: `lake exe` replays a dozen unrelated linter
  # warnings per invocation, and five of those buries the md5 table this script exists to print.
  err="$(mktemp)"
  if ! ( cd jax && lake exe "$exe" default --shim >/dev/null 2>"$err" ); then
    cat "$err" >&2; rm -f "$err"; echo "FAIL: (cd jax && lake exe $exe default --shim)" >&2; exit 1
  fi
  rm -f "$err"
  # The shim's name comes from the recipe's `out`, not from the exe name — so verify rather than
  # assume. A recipe rename would otherwise write a file nothing looks for, and the failure would
  # surface later as "imagenet shim not found" on a net that was just regenerated.
  if [ ! -f "jax/.lake/build/$want" ]; then
    echo "FAIL: $exe did not write jax/.lake/build/$want" >&2
    echo "      (its default recipe's \`out\` was renamed — update EXES here AND the matching" >&2
    echo "       \`shimScript\` in LeanMlir/VerifiedNets.lean, which is what the driver spawns)" >&2
    exit 1
  fi
  printf '  %-40s %s\n' "$want" "$(md5sum "jax/.lake/build/$want" | cut -c1-12)"
done
echo "── done. \`scripts/shim_wiring_gate.py\` gates the partition. ──"
