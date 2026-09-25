#!/usr/bin/env bash
# Downloads / extracts the pre-training data for the FLAME Rust base model.
#
#   tools/fetch_corpus.sh all        everything below (≈ 1–2 h, ≈ 5–8 GB of disk while downloading)
#   tools/fetch_corpus.sh code       Rust source code        → data/raw/rust-code  (+ ~/.cargo/registry/src)
#   tools/fetch_corpus.sh docs       English docs about Rust → data/raw/rust-docs
#   tools/fetch_corpus.sh errors     rustc error explanations → data/raw/rustc-errors
#   tools/fetch_corpus.sh english    general English (FineWeb-Edu sample) → data/raw/english   (needs Python)
#
# Sources and licences (all public; keep the corpus for training, don't redistribute it):
#   code    shallow git clones of large open-source Rust projects (MIT/Apache-2.0) and popular
#           crates.io crates fetched through cargo (they land in ~/.cargo/registry/src)
#   docs    The Rust Programming Language book, Rust by Example, the Rustonomicon, the Rust
#           Reference, the Cargo book, the async book, the edition guide, API guidelines,
#           Rust design patterns, all Rust RFCs, plus the Markdown files of the code repos
#   errors  `rustc --explain E0xxx` for every error code of YOUR installed compiler
#   english HuggingFaceFW/fineweb-edu (ODC-By), "sample-10BT": educational web pages
# Safe to re-run: finished parts are skipped.
set -euo pipefail
PART="${1:-all}"
RAW="data/raw"
mkdir -p "$RAW"

clone() {  # clone <owner/repo> <dir>
  if [ -d "$2" ]; then echo "  have $1"; return; fi
  echo "  clone $1"
  git clone --depth 1 --quiet "https://github.com/$1" "$2" || { echo "  (failed, skipping $1)"; rm -rf "$2"; }
}

fetch_code() {
  echo "== Rust code"
  local out="$RAW/rust-code"; mkdir -p "$out"
  local repos=(
    rust-lang/rust rust-lang/cargo rust-lang/rust-analyzer rust-lang/rust-clippy rust-lang/rustlings
    tokio-rs/tokio tokio-rs/axum serde-rs/serde serde-rs/json clap-rs/clap BurntSushi/ripgrep
    sharkdp/bat sharkdp/fd astral-sh/ruff astral-sh/uv huggingface/candle huggingface/tokenizers
    bevyengine/bevy denoland/deno alacritty/alacritty helix-editor/helix nushell/nushell
    zed-industries/zed rayon-rs/rayon hyperium/hyper rust-random/rand dtolnay/anyhow
    dtolnay/thiserror dtolnay/syn diesel-rs/diesel launchbadge/sqlx gfx-rs/wgpu servo/servo
    tauri-apps/tauri rustls/rustls image-rs/image
  )
  for r in "${repos[@]}"; do
    local dir="$out/${r//\//__}"
    clone "$r" "$dir"
    [ -d "$dir" ] || continue
    # Markdown goes to the docs corpus, only .rs stays here
    if [ -d "$dir/.git" ]; then
      local md="$PWD/$RAW/rust-docs/repo-md/${r//\//__}"
      mkdir -p "$md"
      (cd "$dir" && find . -name '*.md' -type f -exec cp --parents -t "$md" {} + 2>/dev/null) || true
      find "$dir" -type f ! -name '*.rs' -delete 2>/dev/null || true
      rm -rf "$dir/.git"
    fi
  done
  echo "  fetching popular crates through cargo (→ ~/.cargo/registry/src)"
  local crates=(
    regex itertools once_cell log env_logger tracing tracing-subscriber chrono time uuid bytes
    futures async-trait reqwest url base64 sha2 hex flate2 walkdir tempfile toml serde_yaml
    csv indexmap hashbrown smallvec parking_lot crossbeam dashmap num-traits num-bigint
    ndarray nalgebra glam petgraph nom pest logos proc-macro2 quote bitflags memchr aho-corasick
    unicode-segmentation criterion proptest quickcheck insta mio socket2 tonic prost warp
    actix-web rocket tower tower-http http ratatui crossterm indicatif console dialoguer colored
    wasm-bindgen js-sys web-sys winit egui eframe rusqlite redis tokio-postgres sled
    zip tar lz4_flex zstd bincode postcard rmp-serde ron ring aes-gcm ed25519-dalek argon2
    jsonwebtoken heapless embedded-hal cortex-m defmt
  )
  local tmp; tmp="$(mktemp -d)"
  cargo new --quiet --lib --vcs none "$tmp/fetch"
  cp "$tmp/fetch/Cargo.toml" "$tmp/Cargo.toml.orig"
  for c in "${crates[@]}"; do
    (cd "$tmp/fetch" && cargo add --quiet "$c" >/dev/null 2>&1 && cargo fetch --quiet >/dev/null 2>&1) || echo "  (crate $c failed, skipping)"
    cp "$tmp/Cargo.toml.orig" "$tmp/fetch/Cargo.toml"   # reset so version conflicts never pile up
  done
  rm -rf "$tmp"
  du -sh "$out" ~/.cargo/registry/src 2>/dev/null || true
}

fetch_docs() {
  echo "== English documentation about Rust"
  local out="$RAW/rust-docs"; mkdir -p "$out/books"
  local repos=(
    rust-lang/book rust-lang/rust-by-example rust-lang/nomicon rust-lang/reference
    rust-lang/async-book rust-lang/edition-guide rust-lang/api-guidelines rust-lang/rfcs
    rust-lang/rustc-dev-guide rust-lang/cargo rust-unofficial/patterns rust-lang/rust-clippy
  )
  for r in "${repos[@]}"; do
    local dir="$out/books/${r//\//__}"
    clone "$r" "$dir"
    [ -d "$dir" ] || continue
    if [ -d "$dir/.git" ]; then
      find "$dir" -type f ! -name '*.md' -delete 2>/dev/null || true
      rm -rf "$dir/.git"
    fi
  done
  du -sh "$out" 2>/dev/null || true
}

fetch_errors() {
  echo "== rustc error explanations"
  local out="$RAW/rustc-errors"; mkdir -p "$out"
  local f="$out/explanations.txt"; : > "$f"
  local n=0
  for i in $(seq -w 1 999); do
    if t=$(rustc --explain "E0$i" 2>/dev/null) && [ -n "$t" ]; then
      printf 'Rust compiler error E0%s\n\n%s\n<|endoftext|>\n' "$i" "$t" >> "$f"
      n=$((n + 1))
    fi
  done
  echo "  $n explanations → $f ($(du -h "$f" | cut -f1))"
}

fetch_english() {
  echo "== General English (FineWeb-Edu)"
  python3 "$(dirname "$0")/fetch_english.py" --out "$RAW/english/fineweb-edu.txt" --target-mb "${ENGLISH_MB:-1500}"
}

case "$PART" in
  code) fetch_code ;;
  docs) fetch_docs ;;
  errors) fetch_errors ;;
  english) fetch_english ;;
  all) fetch_code; fetch_docs; fetch_errors; fetch_english ;;
  *) echo "usage: $0 all|code|docs|errors|english"; exit 1 ;;
esac
echo "done. Next: docs/end-to-end-guide.md step ⑤ (tokenizer + prepare)."
