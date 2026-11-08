#!/usr/bin/env bash
# Simple prompt loop for a trained FLAME model.
# Type a prompt, press Enter, read the reply. Type "exit" (or press Ctrl+D) to quit.
# Each prompt is answered on its own: the model does not remember earlier turns.
#
# Settings (environment variables):
#   RUN          run directory to load      (default: runs/rust-30m-dgx-20261102-stage2)
#   FLAME        path to the flame binary   (default: target/release/flame)
#   MAX_TOKENS   tokens per reply           (default: 120)
#   TEMPERATURE  sampling temperature       (default: 0.7)
#   DEVICE       "cpu" forces CPU; empty uses the GPU when built with --features cuda
#                (default: cpu)

cd "$(dirname "$0")" || exit 1
RUN=${RUN:-runs/rust-30m-dgx-20261102-stage2}
FLAME=${FLAME:-target/release/flame}
MAX_TOKENS=${MAX_TOKENS:-120}
TEMPERATURE=${TEMPERATURE:-0.7}
DEVICE=${DEVICE-cpu}

if [ ! -x "$FLAME" ]; then
  echo "error: $FLAME not found. Build it first: cargo build --release --locked" >&2
  exit 1
fi
if [ ! -f "$RUN/config.toml" ]; then
  echo "error: no trained run at $RUN" >&2
  echo "Download and unpack a model package (see docs/dgx-training.md), or set RUN=<run dir>." >&2
  exit 1
fi

errlog=$(mktemp)
trap 'rm -f "$errlog"' EXIT

cpu_flag=()
[ "$DEVICE" = "cpu" ] && cpu_flag=(--cpu)

echo "FLAME ($RUN) — type a prompt (code prompts work best). Type 'exit' to quit."
while true; do
  printf '\nyou> '
  IFS= read -r prompt || break
  [ "$prompt" = "exit" ] && break
  [ -z "$prompt" ] && continue
  echo "flame>"
  # Hide the "[flame] loaded ..." line, but show real errors (e.g. a missing tokenizer).
  if ! "$FLAME" generate "${cpu_flag[@]}" --run "$RUN" \
    --prompt "$prompt" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --seed "$RANDOM" 2>"$errlog"; then
    cat "$errlog" >&2
  fi
done
echo "bye"
