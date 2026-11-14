#!/usr/bin/env bash
# Assemble a ready-to-upload Hugging Face Space folder.
#
#   deploy/hf-space/stage.sh <complete-model.tar.gz> <out-dir>
#
# The output contains: Dockerfile, app.py, requirements.txt, README.md (Space card),
# flame-src/ (the Rust source needed to build the CLI) and model/ (config, best weights,
# tokenizer). Upload it with:  hf upload <user>/<space> <out-dir> . --repo-type space
set -euo pipefail

ARCHIVE=${1:?usage: stage.sh <complete-model.tar.gz> <out-dir>}
OUT=${2:?usage: stage.sh <complete-model.tar.gz> <out-dir>}
RUN=${FLAME_RUN:-runs/rust-30m-dgx-20261102-stage2}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

if [ -f "$ARCHIVE.sha256" ]; then
  (cd "$(dirname "$ARCHIVE")" && sha256sum -c "$(basename "$ARCHIVE").sha256")
fi

rm -rf "$OUT"
mkdir -p "$OUT/flame-src" "$OUT/model"

cp "$HERE/Dockerfile" "$HERE/app.py" "$HERE/requirements.txt" "$OUT/"
cp "$HERE/SPACE_README.md" "$OUT/README.md"

# Rust source: only what `cargo build -p flame-cli` needs.
cp "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" "$ROOT/LICENSE" "$OUT/flame-src/"
mkdir -p "$OUT/flame-src/crates"
for c in flame flame-cli flame-wasm; do
  mkdir -p "$OUT/flame-src/crates/$c"
  cp "$ROOT/crates/$c/Cargo.toml" "$OUT/flame-src/crates/$c/"
  cp -r "$ROOT/crates/$c/src" "$OUT/flame-src/crates/$c/"
  # Cargo reads these target dirs when they exist; copy them so the workspace stays valid.
  for d in examples tests; do
    [ -d "$ROOT/crates/$c/$d" ] && cp -r "$ROOT/crates/$c/$d" "$OUT/flame-src/crates/$c/"
  done
done

# Model: frozen config, best weights and the tokenizer the config points to.
TOK=$(tar xzf "$ARCHIVE" -O "./$RUN/config.toml" | sed -n 's/^tokenizer *= *"\(.*\)"/\1/p')
tar xzf "$ARCHIVE" -C "$OUT/model" \
  "./$RUN/config.toml" "./$RUN/best/model.safetensors" "./$TOK"

echo "Staged Space in $OUT:"
du -sh "$OUT"/*
