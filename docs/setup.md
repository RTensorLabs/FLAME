# Setup (Windows laptop with an NVIDIA GPU)

## Option A (recommended): WSL2 + Ubuntu

Candle's CUDA build is simplest on Linux, and WSL2 gives you Linux with full GPU access.

1. **NVIDIA Windows driver**: install the latest Game Ready/Studio driver on Windows. Do *not*
   install a Linux driver inside WSL.
2. **WSL2**: in an admin PowerShell run `wsl --install -d Ubuntu`, then reboot.
3. **Inside Ubuntu**:
   ```bash
   sudo apt update && sudo apt install -y build-essential pkg-config git curl
   # CUDA toolkit for WSL (gives nvcc); pick the current version from
   # https://developer.nvidia.com/cuda-downloads → Linux → x86_64 → WSL-Ubuntu
   sudo apt install -y cuda-toolkit
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   nvidia-smi && nvcc --version        # both must work
   # Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source ~/.cargo/env
   ```
4. **Clone the three repos side by side** (keep them inside the Linux filesystem, e.g. `~/code`,
   not `/mnt/c`, which is much slower):
   ```bash
   mkdir -p ~/code && cd ~/code
   git clone https://github.com/RTensorLabs/FLAME
   git clone https://github.com/RTensorLabs/FLAME-Research flame-research
   git clone https://github.com/RTensorLabs/lintless
   ```
5. **Build with CUDA**:
   ```bash
   cd ~/code/FLAME && cargo build --release --features cuda
   ./target/release/flame --help
   ```
   The first CUDA build compiles kernels and takes a few minutes. If it fails with a
   compute-capability error, run `export CUDA_COMPUTE_CAP=86` (Ampere laptop GPUs are 8.6) and rebuild.

## Option B: native Windows

Possible, but you need Visual Studio Build Tools (C++ workload), the CUDA toolkit for Windows
and Rust (MSVC toolchain), all on the same PATH. Build from the *x64 Native Tools* prompt:
`cargo build --release --features cuda`. If the CUDA build gives you trouble, use WSL2.

## CPU-only

Everything also runs on the CPU (`cargo build --release`, then add `--cpu` to commands).
That's fine for the tests, the overfit gate and Tiny Shakespeare, but 10–50× slower than the GPU.

## Python (optional)

Only `tools/reference_check.py` (the PyTorch cross-check of FLAME's forward pass) uses Python:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch safetensors          # the CPU build is enough for this check
```

## Check everything

```bash
cd ~/code/FLAME && cargo test --release
cd ~/code/flame-research && cargo test --release
cd ~/code/lintless && cargo test --release
```
