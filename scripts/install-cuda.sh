#!/usr/bin/env bash
# install-cuda.sh — install CUDA Toolkit 12.8 from NVIDIA's repo on Ubuntu 24.04.
#
# Why 12.8: sm_120 (RTX 5090 / RTX PRO 6000 Blackwell) is only supported by
# CUDA 12.8+. Ubuntu 24.04's distro `nvidia-cuda-toolkit` package ships older
# 12.0–12.4 and cannot build PTX that targets Blackwell. For a mixed
# NVIDIA fleet, 12.8 is the floor.
#
# Usage (on the machine being prepped — WSL dev box or real DUT):
#     sudo bash install-cuda.sh
#
# Auto-detects WSL vs native Ubuntu 24.04 and picks the right NVIDIA repo target.
# Does NOT modify ~/.bashrc (some WSL setups have a permission quirk there).
# Prints the PATH export line at the end for the operator to place where they want.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must run as root. Try: sudo bash $0" >&2
  exit 1
fi

# --- Detect repo target ------------------------------------------------------

if grep -qi microsoft /proc/version 2>/dev/null; then
  REPO_TARGET="wsl-ubuntu"
  ENV_LABEL="WSL2 Ubuntu"
else
  REPO_TARGET="ubuntu2404"
  ENV_LABEL="native Ubuntu 24.04"
fi

echo "==> Detected environment: ${ENV_LABEL}"
echo "==> Using NVIDIA repo target: ${REPO_TARGET}"
echo ""

# --- Sanity checks -----------------------------------------------------------

if ! command -v wget >/dev/null 2>&1; then
  echo "==> Installing wget (required to fetch the keyring)..."
  apt-get update -qq
  apt-get install -y wget
fi

# Warn if an older distro toolkit is already installed — it won't conflict with
# /usr/local/cuda-12.8, but the operator should know which one is on PATH.
if dpkg -l | grep -q '^ii  nvidia-cuda-toolkit '; then
  echo "==> WARNING: distro nvidia-cuda-toolkit is installed. It will coexist with"
  echo "    /usr/local/cuda-12.8 but you MUST use the 12.8 nvcc for Blackwell work."
  echo ""
fi

# --- Fetch + install keyring -------------------------------------------------

KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${REPO_TARGET}/x86_64/cuda-keyring_1.1-1_all.deb"
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_LOCAL}"' EXIT

echo "==> Fetching cuda-keyring from ${KEYRING_URL}"
wget -q --show-progress -O "${TMPDIR_LOCAL}/cuda-keyring.deb" "${KEYRING_URL}"

echo "==> Installing cuda-keyring"
dpkg -i "${TMPDIR_LOCAL}/cuda-keyring.deb"

# --- Install CUDA 12.8 toolkit -----------------------------------------------

echo "==> apt-get update"
apt-get update

echo "==> Installing cuda-toolkit-12-8 (this pulls ~3GB, can take a while)"
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8

# --- Verify ------------------------------------------------------------------

NVCC="/usr/local/cuda-12.8/bin/nvcc"
if [[ ! -x "${NVCC}" ]]; then
  echo "ERROR: ${NVCC} missing after install. Something went wrong." >&2
  exit 1
fi

echo ""
echo "==> Verifying nvcc..."
"${NVCC}" --version
echo ""

# --- Operator instructions ---------------------------------------------------

cat <<'EOF'
==> DONE.

nvcc is at /usr/local/cuda-12.8/bin/nvcc but NOT on the default PATH.

To use it in the CURRENT shell:

    export PATH=/usr/local/cuda-12.8/bin:$PATH

To make it permanent for your user, add that line to ONE of:

    ~/.bashrc            (if writable)
    ~/.profile
    ~/.bash_profile
    /etc/profile.d/cuda-12-8.sh    (system-wide, needs sudo)

PSFU's ensure_gpu_burn() will find nvcc on PATH and pick it up automatically.

Next step for PSFU:

    cd /path/to/psfu
    PATH=/usr/local/cuda-12.8/bin:$PATH python3 psfu.py bootstrap

That will build gpu-burn against the local GPU fleet, smoke-test it, try the
board-sensor fallback chain, and print a readiness summary — without starting
a stress run.
EOF
