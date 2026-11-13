#!/bin/bash
# deploy/vast/build.sh — build the bench image from this tree. Pushes nothing.
#
#     deploy/vast/build.sh                   # → lean4-jax-bench:<rev>
#     IMAGE=<hub user>/lean4-jax-bench deploy/vast/build.sh
#
# The image is ~45 GB unpacked (base ~18, venv ~8, toolchain ~2, .lake ~15+) and docker needs that
# much free under its root while it builds; the Lean build is the long step.
set -euo pipefail
cd "$(dirname "$0")/../.."
REV="$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
IMAGE=${IMAGE:-lean4-jax-bench}
NEED_GB=${NEED_GB:-70}
root=$(docker info -f '{{.DockerRootDir}}')
free=$(df -BG --output=avail "$root" | tail -1 | tr -dc 0-9)
[ "$free" -ge "$NEED_GB" ] || { echo "⛔ ${free} GB free under $root; the build wants ~${NEED_GB} (NEED_GB=… to override)"; exit 1; }
docker build -f deploy/vast/Dockerfile --build-arg GIT_REV="$REV" ${BASE:+--build-arg BASE="$BASE"} \
  -t "$IMAGE:$REV" -t "$IMAGE:latest" .
echo
echo "built $IMAGE:$REV ($(docker image inspect -f '{{.Size}}' "$IMAGE:$REV" | awk '{printf "%.1f GB", $1/1e9}'))"
echo "push:  docker push $IMAGE:$REV"
