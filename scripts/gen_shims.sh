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

# exe (in jax/) → RECIPE → the file it writes under jax/.lake/build/. The right-hand column is what
# `VerifiedNet.shimScript` must spell; `shim_wiring_gate.py` checks that pairing both ways.
#
# ⚠ The RECIPE column was added 2026-08-06 and every pre-existing row keeps `default`, which is what
# this script hardcoded before. It exists because a net's resolution lives in its recipe's
# `TrainConfig`, not in the exe — see the `short` row below.
EXES=(
  "resnet34-imagenet:default:generated_resnet34_imagenet_shim.py"
  "vit-tiny-imagenet:default:generated_vit_tiny_imagenet_shim.py"
  "mobilenet-v2-imagenet:default:generated_mobilenet_v2_imagenet_shim.py"
  "efficientnet-b0-imagenet:default:generated_efficientnet_b0_imagenet_shim.py"
  "convnext-tiny-imagenet:default:generated_convnext_tiny_imagenet_shim.py"
  # R50 joined 2026-08-05 with the R50 scoping work: `resnet50ImagenetVerified` sets
  # `shimScript` to this name, so without a row here the spec points at a file nothing
  # writes — `spawnShim` REFUSES rather than falling back, which is the whole §0.9 design.
  "resnet50-imagenet:default:generated_resnet50_imagenet_shim.py"
  # ⭐ MNv4 joined 2026-08-12 with `mnv4ImagenetVerified`. ⚠ The exe here is the faithful
  # **Conv-M** reference, while the verified spec that names this shim is **Conv-S** — deliberate,
  # and safe, because a shim carries AUGMENTED BATCHES and not weights. What crosses over is the
  # MNv4-family data pipeline (RandAugment, the 224² crop), which both sizes share. Nothing about
  # Conv-M's block table reaches the verified net through it.
  "mobilenet-v4-imagenet:default:generated_mobilenet_v4_imagenet_shim.py"
  # ⭐ RSB-A3's 160²-train shim, for `resnet50Imagenet160Verified` (2026-08-06).
  # `short` IS timm's A3: `trainRes := 160`, `testCropRatio := 0.95`, RandAugment m6, mixup/cutmix.
  # ⚠ ONE shim, TWO widths — `Jax/Codegen.lean` applies `trainRes` only on the TRAIN path, while
  # eval keeps the hardcoded `_IMG_SIZE = 224`. So this file emits 76,800 floats/img on train and
  # 150,528 on val, which is A3's split and is why the driver needs its own `evalD0` (§2.3).
  "resnet50-imagenet:short:generated_resnet50_imagenet_short_shim.py"
  # ⭐ The 2018 recipe's shim, for `resnet50Imagenet2018Verified` (2026-08-24).
  # ⛔ WHY THIS ROW EXISTS. `shimScript` is a field on the NET, and until this row the only 224²
  # R50 net pointed at the `default` shim — which is RSB-A2's, and calls
  # `_randaugment(img, 2, 7.0, 0.5)` UNCONDITIONALLY on the train path. A verified 2018 run
  # therefore trained 2018's optimizer and schedule on A2's augmentation: neither recipe, and not
  # comparable to the JAX 2018 number it exists to sit beside. Caught 2026-08-24 by a mean −4.90
  # top-1 gap against the JAX per-epoch curve over epochs 1–10; the run was killed at epoch 13.
  # ⚠ `shim_wiring_gate.py` cannot catch this class: it checks each NET streams its own shim
  # rather than R34's, and there is no per-RECIPE slot for it to check.
  # This shim is random-resized-crop + bicubic/antialias + hflip and nothing else, which is what
  # `resnet50ImagenetConfig2018` (`useRandAugment := false`, CE+LS) specifies.
  "resnet50-imagenet:2018:generated_resnet50_imagenet_2018_shim.py"
)

echo "── generating $(( ${#EXES[@]} )) per-net ImageNet shims ──"
for e in "${EXES[@]}"; do
  exe="${e%%:*}"; want="${e##*:}"
  rest="${e#*:}"; recipe="${rest%%:*}"
  # stderr is swallowed unless the build FAILS: `lake exe` replays a dozen unrelated linter
  # warnings per invocation, and five of those buries the md5 table this script exists to print.
  err="$(mktemp)"
  if ! ( cd jax && lake exe "$exe" "$recipe" --shim >/dev/null 2>"$err" ); then
    cat "$err" >&2; rm -f "$err"; echo "FAIL: (cd jax && lake exe $exe $recipe --shim)" >&2; exit 1
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
