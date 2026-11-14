#!/usr/bin/env bash
# Build the in-browser FLAME demo as a static site (e.g. a free static Hugging Face Space).
#
#   deploy/web/build.sh <complete-model.tar.gz> <out-dir> [--f16]
#
# Needs: rustup target add wasm32-unknown-unknown
#        cargo install wasm-bindgen-cli --version <same as wasm-bindgen in Cargo.lock>
#        uv (only for --f16)
# --f16 stores the weights in half precision: half the download, converted back to f32 on load.
#
# Test locally:  python3 -m http.server -d <out-dir> 8000   then open http://localhost:8000
# Upload:        hf upload <user>/<space> <out-dir> . --repo-type space
set -euo pipefail

ARCHIVE=${1:?usage: build.sh <complete-model.tar.gz> <out-dir> [--f16]}
OUT=${2:?usage: build.sh <complete-model.tar.gz> <out-dir> [--f16]}
F16=${3:-}
RUN=${FLAME_RUN:-runs/rust-30m-dgx-20261102-stage2}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if [ -f "$ARCHIVE.sha256" ]; then
  (cd "$(dirname "$ARCHIVE")" && sha256sum -c "$(basename "$ARCHIVE").sha256")
fi

(cd "$ROOT" && cargo build --release -p flame-wasm --target wasm32-unknown-unknown)

rm -rf "$OUT"
mkdir -p "$OUT/model"
wasm-bindgen --target web --no-typescript --out-dir "$OUT/pkg" \
  "$ROOT/target/wasm32-unknown-unknown/release/flame_wasm.wasm"
cp "$HERE/index.html" "$HERE/worker.js" "$OUT/"
cp "$HERE/SPACE_README.md" "$OUT/README.md"

# Model: frozen config, best weights and the tokenizer the config points to.
TOK=$(tar xzf "$ARCHIVE" -O "./$RUN/config.toml" | sed -n 's/^tokenizer *= *"\(.*\)"/\1/p')
tar xzf "$ARCHIVE" -O "./$RUN/config.toml" > "$OUT/model/config.toml"
tar xzf "$ARCHIVE" -O "./$TOK" > "$OUT/model/tokenizer.json"
tar xzf "$ARCHIVE" -O "./$RUN/best/model.safetensors" > "$OUT/model/model.safetensors"

if [ "$F16" = "--f16" ]; then
  uv run --no-project --with safetensors --with numpy python -I "$HERE/to_f16.py" "$OUT/model/model.safetensors"
fi

echo "Built static site in $OUT:"
du -sh "$OUT"/pkg/*.wasm "$OUT"/model/*
