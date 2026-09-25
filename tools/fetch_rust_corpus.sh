#!/usr/bin/env bash
# Downloads Rust source code for pre-training the FLAME Rust base model.
#
#   tools/fetch_rust_corpus.sh [OUT_DIR]        (default: data/raw/rust)
#
# Two sources, both public and permissively licensed (MIT/Apache-2.0 for almost all):
#   1. shallow git clones of large, well-written Rust projects (the Rust compiler + std, tokio, …)
#   2. crates.io sources: popular crates are fetched through cargo (with their dependencies),
#      which lands them in ~/.cargo/registry/src; that folder is then used as input too.
# Expect roughly 1–3 GB of .rs files ⇒ ≈ 300 M – 1 B tokens with an 8k BPE.
# Takes 10–40 minutes depending on your connection. Safe to re-run (skips what exists).
# Keep the corpus for training only; don't redistribute it.
set -euo pipefail
OUT="${1:-data/raw/rust}"
mkdir -p "$OUT/repos"

REPOS=(
  rust-lang/rust rust-lang/cargo rust-lang/rust-analyzer rust-lang/rust-clippy rust-lang/rustlings
  rust-lang/rust-by-example rust-lang/book tokio-rs/tokio tokio-rs/axum serde-rs/serde serde-rs/json
  clap-rs/clap BurntSushi/ripgrep sharkdp/bat sharkdp/fd astral-sh/ruff astral-sh/uv
  huggingface/candle huggingface/tokenizers bevyengine/bevy denoland/deno alacritty/alacritty
  helix-editor/helix nushell/nushell zed-industries/zed rayon-rs/rayon hyperium/hyper
  rust-random/rand dtolnay/anyhow dtolnay/thiserror dtolnay/syn diesel-rs/diesel launchbadge/sqlx
  gfx-rs/wgpu servo/servo tauri-apps/tauri rustls/rustls image-rs/image
)
for r in "${REPOS[@]}"; do
  dir="$OUT/repos/${r//\//__}"
  if [ -d "$dir" ]; then echo "have $r"; continue; fi
  echo "clone $r"
  git clone --depth 1 --quiet "https://github.com/$r" "$dir" || echo "  (failed, skipping $r)"
  # keep only Rust sources and markdown docs to save disk
  find "$dir" -type f ! -name '*.rs' ! -name '*.md' -delete 2>/dev/null || true
  rm -rf "$dir/.git"
done

CRATES=(
  regex itertools once_cell log env_logger tracing tracing-subscriber chrono time uuid bytes
  futures async-trait reqwest url base64 sha2 hex flate2 walkdir tempfile toml serde_yaml
  csv indexmap hashbrown smallvec parking_lot crossbeam dashmap num-traits num-bigint
  ndarray nalgebra glam petgraph nom pest logos lalrpop-util proc-macro2 quote bitflags
  memchr aho-corasick unicode-segmentation criterion proptest quickcheck insta
  mio socket2 tonic prost warp actix-web rocket tower tower-http http
  ratatui crossterm indicatif console dialoguer colored
  wasm-bindgen js-sys web-sys winit egui eframe
  rusqlite redis tokio-postgres sled rocksdb
  zip tar lz4_flex zstd bincode postcard rmp-serde ron
  ring aes-gcm ed25519-dalek argon2 jsonwebtoken
  heapless embedded-hal cortex-m defmt
)
tmp="$(mktemp -d)"
cargo new --quiet --lib "$tmp/fetch"
for c in "${CRATES[@]}"; do
  echo "fetch crate $c"
  (cd "$tmp/fetch" && cargo add --quiet "$c" >/dev/null 2>&1 && cargo fetch --quiet >/dev/null 2>&1) || true
  # reset between crates so incompatible version requirements never block the next one
  (cd "$tmp/fetch" && git checkout -q Cargo.toml 2>/dev/null || sed -i '/^\[dependencies\]/q' Cargo.toml)
done
rm -rf "$tmp"

echo
echo "done. Rust sources:"
du -sh "$OUT" ~/.cargo/registry/src 2>/dev/null || true
echo "next:"
echo "  flame tokenizer train --input $OUT ~/.cargo/registry/src --ext rs --vocab-size 8192 --out data/tokenizers/rust-8k.json --sample-mb 200"
echo "  flame prepare --input $OUT ~/.cargo/registry/src --ext rs --tokenizer data/tokenizers/rust-8k.json --out data/rust"
