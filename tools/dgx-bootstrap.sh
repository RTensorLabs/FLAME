#!/usr/bin/env bash
set -euo pipefail
cd /home/kenpath/flame-training/FLAME
export CARGO_HOME=/home/kenpath/flame-training/.cargo
export RUSTUP_HOME=/home/kenpath/flame-training/.rustup
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"
export CUDA_COMPUTE_CAP=121
export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CARGO_BUILD_JOBS=3
export RAYON_NUM_THREADS=4
mkdir -p ops
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 --fail -sSf https://sh.rustup.rs -o ops/rustup-init.sh
  sh ops/rustup-init.sh -y --profile minimal --no-modify-path
fi
rustc --version
cargo build --release --locked --features cuda > ops/build.log 2>&1 &
BUILD_PID=$!
/home/kenpath/.local/bin/uv venv .venv
/home/kenpath/.local/bin/uv pip install --python .venv/bin/python pyarrow requests
export PATH="$PWD/.venv/bin:$PATH"
bash tools/fetch_corpus.sh all > ops/corpus.log 2>&1 &
DATA_PID=$!
wait "$BUILD_PID"
cargo test --release --locked > ops/tests.log 2>&1
wait "$DATA_PID"
printf 'bootstrap complete\n'
