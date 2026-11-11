#!/usr/bin/env bash
# apt-get update + install that survives the runner's default Ubuntu mirrors being unreachable.
#
#   scripts/lib/apt_install.sh <package> [<package> ...]
#
# GitHub's ubuntu-24.04 runners read their mirrors from /etc/apt/apt-mirrors.txt:
# azure.archive.ubuntu.com first, then archive.ubuntu.com and security.ubuntu.com. On 2026-11-11
# none of the three resolved from the runner for hours ("Temporary failure resolving
# 'azure.archive.ubuntu.com'") while every other host did, and `apt-get update` spent eighteen
# minutes timing out before the install failed. So: ask DNS first, and if the default mirrors do
# not answer (or the install fails anyway), put backup mirrors that carry the same signed archive
# (all four noble pockets) at the head of the list and try once more.
set -uo pipefail

APT=(sudo apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
     -o DPkg::Lock::Timeout=120)
MIRRORS=/etc/apt/apt-mirrors.txt
BACKUPS=(https://mirrors.edge.kernel.org/ubuntu/ http://mirror.math.princeton.edu/pub/ubuntu/
         http://mirrors.mit.edu/ubuntu/)

resolves() { timeout 15 getent hosts "$1" >/dev/null 2>&1; }

use_backups() {
  echo "::warning::apt: the runner's default Ubuntu mirrors are unreachable; using backup mirrors"
  for h in azure.archive.ubuntu.com archive.ubuntu.com security.ubuntu.com mirrors.edge.kernel.org github.com; do
    if resolves "$h"; then echo "  DNS ok:   $h"; else echo "  DNS FAIL: $h"; fi
  done
  if [ ! -f "$MIRRORS" ]; then
    echo "::error::apt: $MIRRORS is missing, so there is no mirror list to extend"; return 1
  fi
  { n=1
    for m in "${BACKUPS[@]}"; do printf '%s\tpriority:%d\n' "$m" "$n"; n=$((n + 1)); done
    # the runner's own entries stay, behind the backups
    sed -E 's/\tpriority:[0-9]+$//' "$MIRRORS" | while read -r m; do
      [ -n "$m" ] && { printf '%s\tpriority:%d\n' "$m" "$n"; n=$((n + 1)); }
    done
  } | sudo tee "$MIRRORS.new" >/dev/null
  sudo mv "$MIRRORS.new" "$MIRRORS"
  cat "$MIRRORS"
}

install() { "${APT[@]}" update && "${APT[@]}" install -y "$@"; }

if resolves azure.archive.ubuntu.com || resolves archive.ubuntu.com; then
  install "$@" && exit 0
fi
use_backups || exit 1
install "$@"
