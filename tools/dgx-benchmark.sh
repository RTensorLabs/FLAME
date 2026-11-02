#!/usr/bin/env bash
set -euo pipefail
cd /home/kenpath/flame-training/FLAME
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RAYON_NUM_THREADS=4
curl --fail -sSL --create-dirs -o data/raw/dgx-benchmark.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
target/release/flame tokenizer byte --out data/tokenizers/byte.json
target/release/flame prepare --input data/raw/dgx-benchmark.txt --tokenizer data/tokenizers/byte.json --out data/dgx-benchmark --separator '\n\n' --val-fraction 0.1
cat > ops/benchmark.toml <<'EOF'
[model]
vocab_size = 8192
d_model = 512
n_layers = 8
n_heads = 8
max_seq_len = 1024
[train]
micro_batch_size = 8
grad_accum_steps = 16
seq_len = 1024
max_steps = 4
lr = 0.0006
warmup_steps = 300
eval_every = 1000
eval_batches = 4
log_every = 1
checkpoint_every_minutes = 5
[data]
tokenizer = "data/tokenizers/byte.json"
dataset_dir = "data/dgx-benchmark"
EOF
target/release/flame train --config ops/benchmark.toml --run runs/dgx-speed-benchmark
target/release/flame status --run runs/dgx-speed-benchmark
