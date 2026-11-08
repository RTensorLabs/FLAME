# FLAME — Framework for Lightweight AI Model Engineering

FLAME is a small, readable Rust + [Candle](https://github.com/huggingface/candle) framework
for training language models from scratch on **one laptop**. It is the engineering base of
a research program on efficient, edge-friendly AI (see [`docs/research-notes.md`](docs/research-notes.md)).

It is one of three repositories:

| repo | what it is | status |
|---|---|---|
| **FLAME** (this repo) | framework: tokenizer, Transformer, pausable trainer, CLI | Phase 1–2 done |
| [flame-research](https://github.com/RTensorLabs/FLAME-Research) | the research track: dynamically routed modular models vs dense/MoE baselines | ready to run experiments |
| [lintless](https://github.com/RTensorLabs/lintless) | the first application: the FLAME Rust base model **fine-tuned** to explain and fix Rust compiler errors | ready; needs the base model |

Clone them **next to each other** (flame-research and lintless use FLAME via a relative path):

```text
code/
  FLAME/
  flame-research/
  lintless/
```

---

## What's in Phase 1 & 2

| component | file |
|---|---|
| Byte-level + BPE tokenizer (training, encode, decode) | `crates/flame/src/tokenizer.rs` |
| Embedding, RMSNorm, RoPE, causal self-attention, SwiGLU, residual blocks, LM head | `crates/flame/src/nn.rs`, `model.rs` |
| Cross-entropy (with optional loss mask) | `crates/flame/src/model.rs` |
| Memory-mapped token dataset, deterministic batching | `crates/flame/src/data.rs` |
| Hand-written AdamW (state is saved), warmup + cosine LR, grad clipping | `crates/flame/src/optim.rs` |
| Training loop, validation, best-model tracking, ETA | `crates/flame/src/trainer.rs` |
| **Pause / resume** (Ctrl+C, time budget, PAUSE file, periodic crash-safe checkpoints) | `control.rs`, `checkpoint.rs` |
| Sampling (temperature, top-k) | `crates/flame/src/generate.rs` |
| Training-time estimator | `crates/flame/src/estimate.rs` |
| `flame` CLI | `crates/flame-cli/src/main.rs` |
| Optional DGX BF16/fused trainer, native batching, recovery publisher | `tools/torch_train.py`, `torch_backend.py`, `dgx_train.py`; [DGX guide](docs/dgx-training.md) |

Verified by tests (`cargo test`):

- **Causality**: changing future tokens never changes past logits.
- **Overfit gate**: a tiny model memorises a tiny text (loss < 0.15).
- **Native CPU pause/resume is exact**: training 30 steps straight gives **bit-identical** weights to
  12 + 11 + 7 steps across three separate "sessions".
- **Ctrl+C and the PAUSE file** save a checkpoint and stop.
- **Cross-check with PyTorch**: `tools/reference_check.py` re-implements the model in PyTorch
  from the paper descriptions and compares logits (measured max difference: 5e-7).

The optional DGX CUDA backend has separate numerical resume and checkpoint-compatibility
checks; fused GPU updates are not promised bit-identical. See [DGX verification](docs/dgx-training.md).

---

## 1. Setup

You need Rust (https://rustup.rs). For the GPU you also need the NVIDIA driver and the
CUDA toolkit. On Windows, **WSL2 + Ubuntu is strongly recommended** (Candle's CUDA build is
much easier there). Step-by-step: [`docs/setup.md`](docs/setup.md).

```bash
git clone https://github.com/RTensorLabs/FLAME && cd FLAME

# CPU build (works everywhere, fine for Phase 1 tests)
cargo build --release

# GPU build (RTX 3070 Ti etc.) — needs the CUDA toolkit (nvcc) on PATH
cargo build --release --features cuda

cargo test --release          # all correctness tests (~30 s)
```

The binary is `target/release/flame`. The examples below write it as `flame`; use
`./target/release/flame` or put it on your PATH.

## 2. Phase 1 gate: overfit a tiny dataset (~1 minute)

```bash
flame tokenizer byte --out data/tokenizers/byte.json
flame prepare --input examples/overfit.txt --tokenizer data/tokenizers/byte.json \
              --out data/overfit --val-fraction 0.25
flame train --config configs/overfit.toml --run runs/overfit
flame generate --run runs/overfit --prompt "FLAME is" --temperature 0
```

Expected: loss falls to ≈ 0.05 and the sample repeats the paragraph. **If this fails, don't
scale up.** Something is broken.

## 3. First real run: Tiny Shakespeare (≈ 5–20 min on GPU)

```bash
curl -L -o data/raw/shakespeare.txt --create-dirs \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
flame prepare --input data/raw/shakespeare.txt --tokenizer data/tokenizers/byte.json \
              --out data/shakespeare --separator "\n\n" --val-fraction 0.1
flame train --config configs/tiny-shakespeare.toml --run runs/shakespeare
flame generate --run runs/shakespeare --prompt "ROMEO:" --max-tokens 300
```

Expected validation loss is about 1.5 nats/char.

## 4. A small language model: TinyStories (≈ 0.5–1.5 h on GPU)

```bash
# ~2 GB download; the validation file (~20 MB) is enough for a first try
curl -L -o data/raw/tinystories-train.txt --create-dirs \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
flame tokenizer train --input data/raw/tinystories-train.txt --vocab-size 4096 \
              --out data/tokenizers/tinystories-4k.json --sample-mb 50
flame prepare --input data/raw/tinystories-train.txt \
              --tokenizer data/tokenizers/tinystories-4k.json --out data/tinystories
flame params   --config configs/tinystories-5m.toml
flame estimate --config configs/tinystories-5m.toml
flame train    --config configs/tinystories-5m.toml --run runs/tinystories-5m
```

## 5. Pausing and resuming (important on a laptop)

Every training command can be stopped at any time and **resumed by running the same
command again**:

| you want to… | do this |
|---|---|
| stop now | **Ctrl+C** once. It finishes the current step, saves, and exits. (A second Ctrl+C quits without saving.) |
| train for exactly the time you have | `--max-minutes 50` stops cleanly before 50 minutes are up |
| stop from another window / your phone over SSH | create an empty file `runs/<name>/PAUSE` |
| survive crashes, sleep, power loss | automatic checkpoint every 10 min (`checkpoint_every_minutes`) |
| see progress / time left | `flame status --run runs/<name>` |
| start over | `--fresh` (the old run is renamed to a backup, never deleted) |

Typical week: weekday evening `flame train … --max-minutes 45`, weekend
`flame train … --max-minutes 200`. Details: [`docs/pause-and-resume.md`](docs/pause-and-resume.md).

## 6. How long will training take on my laptop?

The full analysis for an **RTX 3070 Ti Laptop (8 GB)** with ~6 h of training per week is in
[`docs/hardware-and-time-estimates.md`](docs/hardware-and-time-estimates.md). Summary:

| run | params | GPU time | calendar time at ~6 h/week |
|---|---:|---:|---|
| overfit test | 0.1 M | 1 min | now |
| Tiny Shakespeare | 0.9 M | 5–20 min | one evening |
| TinyStories | 5.9 M | 0.5–1.5 h | one evening / one weekend |
| research experiments (5 synthetic + 5 TinyStories runs) | ~5 M each | ≈ 5–12 h total | 1–2 weeks |
| Rust + English base model, 2 stages (context 1024) | 30 M | 13–50 h | 2–8 weeks |
| Lintless = fine-tune of that base model | 30 M | 10–40 min | one evening |
| Rust base model | ~100 M | 60–230 h | 3–9 months (not recommended yet) |

Measure your real speed after 10 minutes of training with `flame status`. It prints your
achieved TFLOP/s, which you then pass to `flame estimate --tflops X`.

**DGX Spark / GB10, measured 2026-11-02:** the 29.89 M model at context 1024 and
131,072 tokens/step reached about **5,400 tokens/s** with native fp32 training and
**52,000–54,000 tokens/s** with the optional BF16/fused backend. For both stages
(1.31 B tokens), that projects to **6.7–7.0 h of optimizer updates**, with an initial
**8–10 h wall-clock allowance** for evaluation, backups and shared-machine load.
These are early-run projections, not a completed-run measurement. The laptop table
above does not predict this machine/backend. See [DGX training and recovery](docs/dgx-training.md)
for commands, precision, live progress, GitHub releases, and restore instructions.

## 7. Commands

```text
flame tokenizer byte|train     create a tokenizer
flame prepare                  tokenise text files → train.bin / val.bin
flame train                    train or resume (pause options above)
flame status                   progress, hours trained, time left
flame generate                 sample text from a run
flame eval                     validation loss / perplexity / bits per token
flame params                   parameter count and memory of a config
flame estimate                 training-time estimate for a config
flame export-reference         bundle for tools/reference_check.py (PyTorch cross-check)
```

Use `flame <command> --help` for all options.

### Prompt loop: `chat.sh`

`flame generate` answers one prompt and exits. `./chat.sh` (repo root) keeps asking for
prompts until you type `exit` or press Ctrl+D:

```bash
cargo build --release --locked
./chat.sh                                         # loads runs/rust-30m-dgx-20261102-stage2 on the CPU
RUN=runs/shakespeare ./chat.sh                    # any other trained run
MAX_TOKENS=200 TEMPERATURE=0.4 ./chat.sh          # longer, less random replies
```

Each reply uses a random seed. The run directory must hold a trained model; for the 30M
Rust model, unpack the complete-model package from Releases into the repo root
([DGX guide](docs/dgx-training.md)). The script is a convenience loop, **not a chatbot**:
FLAME base models only continue text, and each prompt is answered on its own with no
memory of earlier turns. Code-shaped prompts (`fn main() {`, `struct User {`) work best.

## 8. Config files

Included: `overfit`, `tiny-shakespeare`, `tinystories-5m`; `rust-30m-stage1` + `rust-30m-stage2`
(the Rust + English base model that Lintless fine-tunes, trained in two stages on a data mix,
context 1024); `rust-100m-stage1/2` (only if 30 M isn't enough); and small smoke-test configs
(`rust-smoke`, `staged-smoke-stage1/2`) that run on a CPU in minutes.

Configs are TOML with three sections, `[model]`, `[train]` and `[data]`. See `configs/`; every
field is documented in `crates/flame/src/config.rs`. When a run starts, its config is frozen
into `runs/<name>/config.toml`. Changing `[model]` or `[data]` later is refused, since that
would load weights into a different model. You *may* change `[train]`, for example raising
`max_steps` to train longer.

## 9. Documentation

> **📘 Start with the universal guide: [`docs/universal/01-complete-guide.md`](docs/universal/01-complete-guide.md).**
> It explains the whole project (FLAME, flame-research, Lintless) from zero: the basics, the datasets,
> every command, and a step-by-step plan. Then [`docs/universal/02-getting-the-most.md`](docs/universal/02-getting-the-most.md).
> How the code is built, with diagrams: [`docs/architecture.md`](docs/architecture.md).
> Training on a rented GPU (vast.ai etc., hours instead of weeks): [`docs/cloud-gpu-guide.md`](docs/cloud-gpu-guide.md).
> DGX Spark training, measured speed, model publication and checkpoint recovery: [`docs/dgx-training.md`](docs/dgx-training.md).

**New here?** Read [`docs/beginners-guide.md`](docs/beginners-guide.md) (the ideas, in plain words), then
follow [`docs/end-to-end-guide.md`](docs/end-to-end-guide.md): every step from a fresh laptop to a
trained Rust + English base model, with expected output and times. All datasets, with download
links and extraction steps: [`docs/datasets.md`](docs/datasets.md).

To understand the code, start with [`docs/developer-guide.md`](docs/developer-guide.md) (design decisions
and reasons, practices, how to extend) and [`docs/build-log.md`](docs/build-log.md) (how it was built
and verified).

Everything important lives in [`docs/`](docs/README.md): research notes, roadmap, the design
review, setup, the pause/resume guide, time estimates, the concepts behind each component,
troubleshooting, and a lab-notebook template.
