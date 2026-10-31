#!/usr/bin/env bash
# Install ROCm 7.2.4 userspace on Ubuntu 24.04 (noble), pinned.
#
# Uses the IN-TREE amdgpu kernel driver (kernel 6.8) -- no DKMS. gfx1100
# (7900 XTX) is fully supported by it. deploy/ROCM_7900XTX.md is the runbook.
#
#   sudo bash deploy/rocm724_install.sh dryrun   # show what would change
#   sudo bash deploy/rocm724_install.sh apply    # do it
set -euo pipefail

MODE="${1:-dryrun}"
ROCM_VER=7.2.4
ROCM_DIR=/opt/rocm-${ROCM_VER}
APT_URL="https://repo.radeon.com/rocm/apt/${ROCM_VER}"
KEYRING=/etc/apt/keyrings/rocm.gpg
# rocm-ml-libraries pulls hip runtime + rocblas/hipblaslt/hipfft/miopen/rccl/
# rocm-smi-lib/rocm-llvm. rocprofiler-sdk is a NEEDED of the PJRT plugin but is
# not in that meta. Deliberately NOT installing the `rocm` meta: it drags in
# rocdecode/rocjpeg, which hard-require mesa-amdgpu-va-drivers from the amdgpu
# repo -- and the amdgpu repo publishes no noble tree for this release.
# hsa-amd-aqlprofile: NOT pulled by any of the above, but the PJRT plugin's
# rocprofiler-sdk dep needs libhsa-amd-aqlprofile64.so.1 at dlopen time.
INSTALL_PKGS=(rocm-ml-libraries rocprofiler-sdk rocm-device-libs rocminfo hsa-amd-aqlprofile)

if [ "$(id -u)" -ne 0 ]; then echo "must run as root (sudo)"; exit 1; fi
case "$MODE" in dryrun|apply) ;; *) echo "usage: $0 dryrun|apply"; exit 1 ;; esac

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. old stack
# Every ROCm-6.3-era package carries "60300" in its version. That pattern also
# catches the amdgpu-repo userspace cluster (mesa-amdgpu-*, libdrm-amdgpu-*,
# libllvm19.1-amdgpu, amdgpu-dkms) while leaving the DISTRO libdrm-amdgpu1
# (2.4.125-1ubuntu0.1~24.04.2) alone -- 7.2.4's hsa-rocr accepts that one.
mapfile -t OLD < <(dpkg-query -W -f='${Package} ${Version}\n' 2>/dev/null \
                   | awk '$2 ~ /60300/ {print $1}' | sort -u)

say "ROCm 6.3 / amdgpu packages to purge (${#OLD[@]})"
printf '  %s\n' "${OLD[@]:-<none>}"

# --------------------------------------------------------------- 2. apt config
say "apt repo to add"
echo "  ${APT_URL} noble main   (pinned 600, keyring ${KEYRING})"

say "packages to install"
printf '  %s\n' "${INSTALL_PKGS[@]}"

if [ "$MODE" = dryrun ]; then
  say "DRY RUN -- nothing changed."
  echo "Re-run with 'apply' to execute."
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

if [ "${#OLD[@]}" -gt 0 ]; then
  say "purging old stack"
  apt-get purge -y "${OLD[@]}"
  apt-get autoremove -y
fi

say "adding ROCm ${ROCM_VER} repo"
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o "$KEYRING"
chmod 0644 "$KEYRING"

cat > /etc/apt/sources.list.d/rocm.sources <<SRC
Types: deb
URIs: ${APT_URL}
Suites: noble
Components: main
Architectures: amd64
Signed-By: ${KEYRING}
SRC

# Pin so a later `apt upgrade` cannot silently walk us onto 7.3/8.0. AMD has
# rearranged this stack before; this box stays on ${ROCM_VER} until told otherwise.
cat > /etc/apt/preferences.d/rocm-pin-600 <<PIN
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 600
PIN

# Old jammy repo files, already disabled by the noble dist-upgrade -- drop them
# so nothing re-enables a 6.3 source later.
rm -f /etc/apt/sources.list.d/amdgpu.list.distUpgrade \
      /etc/apt/sources.list.d/amdgpu.list.dpkg-old \
      /etc/apt/sources.list.d/amdgpu-proprietary.list.distUpgrade \
      /etc/apt/sources.list.d/amdgpu-proprietary.sources \
      /etc/apt/sources.list.d/rocm.list.distUpgrade \
      /etc/apt/sources.list.d/rocm.list.dpkg-old

say "apt update"
apt-get update

say "installing ROCm ${ROCM_VER}"
apt-get install -y "${INSTALL_PKGS[@]}"

# ------------------------------------------------------------- 3. paths/loader
say "wiring /opt/rocm and the loader"
[ -d "$ROCM_DIR" ] || { echo "expected $ROCM_DIR after install"; exit 1; }
ln -sfn "$ROCM_DIR" /opt/rocm
echo "$ROCM_DIR/lib"     >  /etc/ld.so.conf.d/rocm.conf
echo "$ROCM_DIR/lib64"   >> /etc/ld.so.conf.d/rocm.conf
ldconfig

say "done"
echo "installed: $(cat ${ROCM_DIR}/.info/version 2>/dev/null || echo "$ROCM_VER")"
echo "/opt/rocm -> $(readlink -f /opt/rocm)"
