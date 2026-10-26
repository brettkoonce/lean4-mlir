#!/usr/bin/env bash
# scripts/platform/neuron_warm.sh — fill Neuron's compile cache on an x86 box with no Trainium, so
# the instance starts with every graph already compiled. deploy/TRAINIUM.md §3.
#
#   scripts/platform/neuron_warm.sh                    # the MNIST and CIFAR tiers + the suite's tier 2
#   scripts/platform/neuron_warm.sh vit_adam_train_step resnet34_adambf16_train_step
#   NEURON_WARM_JOBS=8 scripts/platform/neuron_warm.sh
#
# Neuron's runtime has a fake device (NEURON_RT_FAKE_INSTANCE_TYPE): with it the plugin creates a
# client on a box with no Neuron hardware, compiles each graph with neuronx-cc into its compile
# cache, and fails only when it loads the NEFF onto the fake NeuronCore. scripts/platform/neuron_warm.c
# drives that through the shim's iree_ffi_session_create, the trainers' own entry point, under
# scripts/platform/env.sh's neuron profile (--auto-cast=none, PJRT C-API 0.90), so each entry's key
# (MODULE_<graph hash>+<flags hash>) is the one a trainer computes on the instance. Copy the cache
# there and the plugin logs "Using a cached neff" instead of compiling.
#
# The runtime (libnrt) and collectives (libnccom) libraries it needs come from Neuron's apt
# packages, extracted — not installed — under ~/.cache/neuron-warm/. A fake trn1.32xlarge stands
# in for the trn1.2xlarge (the runtime does not fake that size); both compile for --target=trn1.
#
# Writes runs/platform/<date>-<host>-neuron-warm/results.tsv (render, CACHED | FAIL, seconds) and
# one log per render under logs/, merges every new entry into $NEURON_WARM_CACHE (default
# /var/tmp/neuron-compile-cache, the plugin's own default) and packs that as
# $NEURON_WARM_CACHE.tar.gz for the instance. Exit 1 if any render failed to compile.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

JOBS=${NEURON_WARM_JOBS:-$(( $(nproc) / 8 > 0 ? $(nproc) / 8 : 1 ))}
CACHE=${NEURON_WARM_CACHE:-/var/tmp/neuron-compile-cache}
# The packages the fake device was found to work with (Neuron SDK 2.33.0's apt repository).
RT_DEB=pool/main/a/aws-neuronx-runtime-lib/aws-neuronx-runtime-lib_2.35.30.0-d2725ccb8_amd64.deb
CC_DEB=pool/main/a/aws-neuronx-collectives/aws-neuronx-collectives_2.35.43.0-354ad3064_amd64.deb
LIBS=${NEURON_WARM_LIBS:-$HOME/.cache/neuron-warm/2.35}

export PLATFORM_BACKEND=neuron
. scripts/platform/env.sh
[ -n "${PJRT_PLUGIN:-}" ] || { echo "no Neuron plugin: run deploy/trainium_setup.sh venv" >&2; exit 2; }

if [ ! -f "$LIBS/libnrt.so.1" ] || [ ! -f "$LIBS/libnccom.so" ]; then
  echo "fetching libnrt and libnccom from apt.repos.neuron.amazonaws.com into $LIBS"
  tmp=$(mktemp -d); mkdir -p "$LIBS"
  for d in "$RT_DEB" "$CC_DEB"; do
    curl -fsS -o "$tmp/p.deb" "https://apt.repos.neuron.amazonaws.com/$d" && dpkg-deb -x "$tmp/p.deb" "$tmp/x" \
      || { echo "could not fetch $d" >&2; rm -rf "$tmp"; exit 2; }
  done
  cp -a "$tmp"/x/opt/aws/neuron/lib/. "$LIBS"/; rm -rf "$tmp"
fi
export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NEURON_RT_FAKE_INSTANCE_TYPE=trn1.32xlarge NEURON_RT_VISIBLE_CORES=0

if [ $# -gt 0 ]; then RENDERS=("$@")
else
  # What the setup's rungs compile: the MNIST and CIFAR trainers' graphs and the suite's tier 2,
  # less what scripts/platform/expected/neuron-trn1-2xlarge.txt marks SKIP.
  mapfile -t RENDERS < <( {
    printf '%s\n' linear_fwd linear_train_step mlp_fwd mlp_train_step cnn_fwd cnn_train_step \
      cifar8w_fwd cifar8w_sgd_train_step cifar8w_mom_train_step cifar8w_adam_train_step \
      cifar8w_bn_fwd cifar8w_bn_sgd_train_step cifar8w_bn_mom_train_step cifar8w_bn_adam_train_step
    grep -v '^#' scripts/platform/tier2_artifacts.tsv | cut -f1
  } | grep -v -x -F -f <(awk '$2 == "SKIP" { print $1 }' scripts/platform/expected/neuron-trn1-2xlarge.txt) | sort -u )
fi

base=runs/platform/$(date +%F)-$(hostname -s)-neuron-warm; OUT=$base; k=2
while [ -e "$OUT" ]; do OUT=$base-$k; k=$((k+1)); done
mkdir -p "$OUT/logs" "$CACHE"
OUT=$(cd "$OUT" && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/neuron-warm.XXXXXX"); trap 'rm -rf "$WORK"' EXIT
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o "$WORK/libpjrt_ffi.so" || exit 2
gcc -O2 scripts/platform/neuron_warm.c -ldl -o "$WORK/neuron_warm" || exit 2
echo "warming $CACHE: ${#RENDERS[@]} renders, $JOBS at a time, plugin $PJRT_PLUGIN → $OUT"

# One render, in its own cache directory, so whether its NEFF appeared is unambiguous.
warm_one () {
  local name=$1 c=$WORK/cache/$1 log=$OUT/logs/$1.log t0 secs
  [ -f "verified_mlir/$name.mlir" ] || { printf '%s\tMISSING\t-\n' "$name"; return; }
  mkdir -p "$c"; t0=$(date +%s.%N)
  NEURON_COMPILE_CACHE_URL=$c "$WORK/neuron_warm" "$WORK/libpjrt_ffi.so" "verified_mlir/$name.mlir" > "$log" 2>&1
  secs=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
  if find "$c" -name model.neff -size +0 | grep -q .; then printf '%s\tCACHED\t%s\n' "$name" "$secs"
  else printf '%s\tFAIL\t%s\n' "$name" "$secs"
  fi
}
export -f warm_one
export OUT WORK

{
  printf 'render\tstatus\tseconds\n'
  printf '%s\n' "${RENDERS[@]}" | xargs -P "$JOBS" -I{} bash -c 'warm_one "$1"' _ {} | sort
} | tee "$OUT/results.tsv" | column -t -s $'\t'

# Merge: each entry is MODULE_<hash>+<flags> under neuronxcc-<version>/, unique by name.
for d in "$WORK"/cache/*/neuronxcc-*; do
  [ -d "$d" ] || continue
  mkdir -p "$CACHE/$(basename "$d")" && cp -a "$d"/. "$CACHE/$(basename "$d")"/
done
tar -C "$(dirname "$CACHE")" -czf "$CACHE.tar.gz" "$(basename "$CACHE")"
printf 'neuronx-cc\t%s\nflags\t%s\nplugin\t%s\ncache\t%s (%s entries)\n' \
  "$(neuronx-cc --version 2>&1 | sed -n 's/.*version //p' | head -1)" "$NEURON_CC_FLAGS" \
  "$(md5sum "$PJRT_PLUGIN" | cut -d' ' -f1)" "$CACHE" "$(find "$CACHE" -name model.neff | wc -l)" > "$OUT/manifest.tsv"
cached=$(grep -c $'\tCACHED\t' "$OUT/results.tsv"); total=$(( $(wc -l < "$OUT/results.tsv") - 1 ))
echo "$cached of $total compiled into $CACHE; for the instance: $CACHE.tar.gz ($(du -h "$CACHE.tar.gz" | cut -f1))"
echo "  scp it there, then: tar -C /var/tmp -xzf $(basename "$CACHE").tar.gz"
[ "$cached" -eq "$total" ]
