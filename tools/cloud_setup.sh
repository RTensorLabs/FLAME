#!/usr/bin/env bash
# One-shot setup of a rented cloud GPU machine (vast.ai, RunPod, …) for FLAME + Lintless.
#
#   curl -sSL https://raw.githubusercontent.com/RTensorLabs/FLAME/main/tools/cloud_setup.sh | bash
#
# Expects an Ubuntu image WITH the CUDA toolkit (nvcc), e.g. nvidia/cuda:12.4.1-devel-ubuntu22.04.
# Installs build tools + Rust, clones the three repos into ~/code, builds them with CUDA and
# runs the tests. Takes ~10 minutes.
set -euo pipefail

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v nvcc >/dev/null || { echo "nvcc not found: choose a *-devel* CUDA image (e.g. nvidia/cuda:12.4.1-devel-ubuntu22.04)"; exit 1; }
nvcc --version | tail -1

echo "== system packages"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential pkg-config git curl tmux rsync python3-pip >/dev/null

echo "== Rust"
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"

echo "== repositories"
mkdir -p ~/code && cd ~/code
[ -d FLAME ] || git clone -q https://github.com/RTensorLabs/FLAME
[ -d flame-research ] || git clone -q https://github.com/RTensorLabs/FLAME-Research flame-research
[ -d lintless ] || git clone -q https://github.com/RTensorLabs/lintless

# compute capability of GPU 0, e.g. "8.9" → 89 (Candle compiles its kernels for it)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
export CUDA_COMPUTE_CAP="$CAP"
echo "export CUDA_COMPUTE_CAP=$CAP" >> ~/.bashrc
echo "== building with CUDA (compute capability $CAP) — a few minutes"
for r in FLAME flame-research lintless; do (cd ~/code/$r && cargo build --release --features cuda -q); done
echo 'export PATH="$HOME/code/FLAME/target/release:$HOME/code/flame-research/target/release:$HOME/code/lintless/target/release:$PATH"' >> ~/.bashrc

echo "== tests"
(cd ~/code/FLAME && cargo test --release -q 2>&1 | grep "test result")

echo
echo "Ready. Next (see FLAME/docs/cloud-gpu-guide.md):"
echo "  source ~/.bashrc && cd ~/code/FLAME"
echo "  flame train --config configs/overfit.toml --run runs/overfit   # must say 'running on CUDA GPU 0'"
