# FLAME end-to-end guide

Everything from a fresh laptop to a trained **Rust + English base model**, in order, written
for beginners. Every step says **what to run, what you should see, and how long it takes**
on an RTX 3070 Ti Laptop (8 GB) with about 6 hours of training time per week.

> New to the vocabulary (token, loss, epoch, stage, fine-tuning…)? Read
> [beginners-guide.md](beginners-guide.md) first (10 minutes).

Commands are run from the `FLAME/` folder unless stated otherwise. Lines starting with `#`
are comments; you don't type them.

## The whole plan at a glance

```text
FLAME            ① setup → ② build → ③ test → ④ sanity runs
                 → ⑤ download data → ⑥ tokenizer + prepare
                 → ⑦ pre-train stage 1 (≈ 75% of the time) → ⑧ pre-train stage 2 → ⑨ check the base model
lintless         collect fine-tuning data on the CPU while ⑦–⑧ run ──────► fine-tune → evaluate → use
flame-research   independent experiments whenever the GPU is free
decision         ⑩ is Lintless on the 30 M model good enough? if not → 100 M
```

| step | your time | computer time |
|---|---|---|
| ①–③ setup, build, test | ~1.5 h | – |
| ④ sanity runs | 15 min | ~30 min |
| ⑤–⑥ data | 15 min | 2–3 h (downloads + tokenising; unattended) |
| ⑦–⑧ pre-training | a few minutes per session | 13–50 GPU hours → **2–8 weeks** of sessions |
| ⑨ check | 15 min | 10 min |

---

## ① Setup (once, ~1 hour)

You need Windows with WSL2 (Linux inside Windows), the NVIDIA driver, the CUDA toolkit and
Rust. The full step-by-step is in [setup.md](setup.md). After it:

```bash
nvidia-smi          # shows "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
nvcc --version      # shows a CUDA version
cargo --version     # shows a Rust version
```

Clone the three repositories **into the same folder** (they refer to each other as `../FLAME`):

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/RTensorLabs/FLAME
git clone https://github.com/RTensorLabs/FLAME-Research flame-research
git clone https://github.com/RTensorLabs/lintless
cd FLAME
```

Keep the code inside Linux (`~/code`), **not** under `/mnt/c/…`, which is several times slower.

## ② Build (~5 min the first time)

```bash
cargo build --release --features cuda
echo 'export PATH="$HOME/code/FLAME/target/release:$PATH"' >> ~/.bashrc && source ~/.bashrc
flame --help         # lists the commands
```

- `--release` builds optimised code (10–50× faster than a debug build). `--features cuda`
  enables the GPU.
- The first CUDA build compiles GPU kernels and takes a few minutes; later builds take seconds.
- Error mentioning "compute capability": run `export CUDA_COMPUTE_CAP=86`, then build again.
- No NVIDIA GPU? `cargo build --release` gives a CPU-only build (fine for ②–④, far too slow for ⑦).

## ③ Test (~1 min)

```bash
cargo test --release
```

Every line should say `test result: ok`. These tests check that the model can't see future
words, that it can memorise a small text, that pausing and resuming gives *exactly* the same
result, that Ctrl+C and the PAUSE file save properly, that data mixing follows its weights,
and that the tokenizer round-trips. If any test fails, stop and fix that first.

## ④ Sanity runs: prove the setup works (~30 min)

**4a. Overfit gate (1 min).** A tiny model must memorise one paragraph. If it can't, something is broken.

```bash
flame tokenizer byte --out data/tokenizers/byte.json
flame prepare --input examples/overfit.txt --tokenizer data/tokenizers/byte.json --out data/overfit --val-fraction 0.25
flame train --config configs/overfit.toml --run runs/overfit
```

You should see `running on CUDA GPU 0` (if it says CPU, you built without `--features cuda`),
the loss falling, and at the end `best val_loss` ≈ **0.05**.

**4b. Tiny Shakespeare (5–20 min).** Your first real training run, and a measurement of your GPU speed.

```bash
curl -L --create-dirs -o data/raw/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
flame prepare --input data/raw/shakespeare.txt --tokenizer data/tokenizers/byte.json --out data/shakespeare --separator "\n\n" --val-fraction 0.1
flame train --config configs/tiny-shakespeare.toml --run runs/shakespeare
```

**Practise pausing now:** press **Ctrl+C** once in the middle. It saves and stops. Run the same
`flame train …` command again: it prints `resumed from step …` and continues.

```bash
flame generate --run runs/shakespeare --prompt "ROMEO:" --max-tokens 300
flame status --run runs/shakespeare
```

Expect val loss ≈ 1.5 and Shakespeare-looking text. **Write down** the
`throughput … TFLOP/s achieved` number from `flame status`; all time estimates use it.

**4c (optional, 3 min). Staged training smoke test.** Try the two-stage mechanism on toy data:

```bash
REG=$(ls -d ~/.cargo/registry/src/*/ | head -1)
flame tokenizer train --input $REG data/raw/shakespeare.txt --ext rs,txt --vocab-size 2048 --out data/tokenizers/smoke-mix-2k.json --sample-mb 4
flame prepare --input $REG --ext rs --tokenizer data/tokenizers/smoke-mix-2k.json --out data/smoke-code
flame prepare --input data/raw/shakespeare.txt --separator "\n\n" --tokenizer data/tokenizers/smoke-mix-2k.json --out data/smoke-english
flame train --config configs/staged-smoke-stage1.toml --run runs/staged-smoke-stage1
flame train --config configs/staged-smoke-stage2.toml --run runs/staged-smoke-stage2   # prints "stage start: weights from …"
flame eval  --run runs/staged-smoke-stage2                                              # loss per source
```

## ⑤ Download the pre-training data (2–3 h unattended, ~5–8 GB disk)

The base model learns from four sources. What they are, their licences, and exactly how they
are downloaded and extracted: [datasets.md](datasets.md).

```bash
tools/fetch_corpus.sh code       # Rust source code (git clones + cargo fetch)
tools/fetch_corpus.sh docs       # English docs about Rust (the Book, RFCs, …)
tools/fetch_corpus.sh errors     # the rustc explanation of every error code
pip install pyarrow requests
tools/fetch_corpus.sh english    # general English (FineWeb-Edu), 1.5 GB by default
du -sh data/raw/* ~/.cargo/registry/src
```

Everything can be re-run; finished parts are skipped. If a clone fails (network), just run
the same command again.

## ⑥ Tokenizer + prepare (~30–60 min unattended)

One tokenizer for everything (it can't change later), then each source is converted to
token files separately so the training config can mix them:

```bash
flame tokenizer train \
  --input data/raw/rust-code ~/.cargo/registry/src data/raw/rust-docs data/raw/rustc-errors data/raw/english \
  --ext rs,md,txt --vocab-size 8192 --out data/tokenizers/rust-en-8k.json --sample-mb 250

T=data/tokenizers/rust-en-8k.json
flame prepare --input data/raw/rust-code ~/.cargo/registry/src --ext rs  --tokenizer $T --out data/rust-code
flame prepare --input data/raw/rust-docs                       --ext md  --tokenizer $T --out data/rust-docs
flame prepare --input data/raw/rustc-errors                    --ext txt --tokenizer $T --out data/rustc-errors --val-fraction 0.05
flame prepare --input data/raw/english                         --ext txt --tokenizer $T --out data/english
```

Check: each `prepare` prints its token count; `bytes/token` should be about 3–4.

## ⑦ Pre-train stage 1 (≈ 75% of the training time)

Stage 1 learns Rust and English together: 59% code, 35% general English, 6% Rust docs.

```bash
flame params   --config configs/rust-30m-stage1.toml           # ≈ 30 M parameters
flame estimate --config configs/rust-30m-stage1.toml --tflops <your number from 4b>
flame train    --config configs/rust-30m-stage1.toml --run runs/rust-30m-stage1 --max-minutes 45
```

At the start, `flame train` prints the **data plan**: how many tokens it takes from each
source and how often each is repeated. A ⚠ means a source would be repeated more than ~16
times (memorised). Fix it by lowering that source's `weight` in the config, or by downloading
more of it (for English: `ENGLISH_MB=3000 tools/fetch_corpus.sh english` + prepare again),
**before** training for long.

**Every session:** run the same `flame train …` line (`--max-minutes 45` on weekdays,
`--max-minutes 210` at the weekend). It resumes automatically. You can shut the laptop down
between sessions.

**Between sessions**, check progress:

```bash
flame status   --run runs/rust-30m-stage1                         # % done, hours, time left
flame generate --run runs/rust-30m-stage1 --prompt "fn main() {" --temperature 0.7
flame generate --run runs/rust-30m-stage1 --prompt "The borrow checker" --temperature 0.7
```

**What healthy training looks like**
- The loss drops quickly for the first few hundred steps, then slowly.
- `val_loss` follows `loss` closely.
- `gnorm` (gradient size) settles below ~1–2 after warm-up.
- Samples improve from symbol soup, to Rust/English-looking words, to plausible code and sentences.

**Problems:** loss becomes NaN → lower `lr` (e.g. 4e-4) and restart with `--fresh`; "out of
memory" → set `micro_batch_size = 1` and `grad_accum_steps = 128`; much slower than the
estimate → check temperatures with `nvidia-smi -l 5` (throttling) and the `/mnt/c` note in ①.
More: [troubleshooting.md](troubleshooting.md).

**Meanwhile:** collect Lintless data on the CPU (Lintless guide, step 4).

## ⑧ Pre-train stage 2 (≈ 25% of the training time)

Stage 2 starts from stage 1's final weights (`init_from = "runs/rust-30m-stage1"` in the
config) and focuses: 50% code, 20% Rust docs, 0.5% rustc error explanations, and ~30% English
**replayed** so it isn't forgotten. The learning rate is lower and decays to zero.

```bash
flame train --config configs/rust-30m-stage2.toml --run runs/rust-30m-stage2 --max-minutes 45
```

The first session prints `stage start: weights from runs/rust-30m-stage1/…`. Later sessions
resume stage 2's own checkpoint. Same routine as stage 1.

## ⑨ Check the base model (~10 min)

```bash
flame eval --run runs/rust-30m-stage2          # overall + per-source loss
flame eval --run runs/rust-30m-stage1          # compare: stage 2 should be better on rust-docs
flame generate --run runs/rust-30m-stage2 --prompt "impl Iterator for" --temperature 0.5
flame generate --run runs/rust-30m-stage2 --prompt "error[E0382]: borrow of moved value" --temperature 0.5
```

`flame eval` prints the loss **for each source**. Stage 2 should improve on `rust-docs`
without a big loss increase on `english` (that would mean forgetting). Record the numbers in
[lab-notebook.md](lab-notebook.md).

**Now fine-tune Lintless** on `runs/rust-30m-stage2`: see `lintless/docs/end-to-end-guide.md`.

## ⑩ Deciding to scale up (30 M → 100 M)

Judge by **Lintless** (the real task), not by the base model's loss:

| Lintless result | action |
|---|---|
| fixes compile at least as often as rustc's suggestions overall, and clearly more on errors rustc can't fix | **good enough**: keep 30 M, keep improving the data |
| below rustc, but replies are well-formed code | **needs data**: double the Lintless data, re-evaluate |
| still below after doubling the data, or broken / missing code | **needs capacity**: go to 100 M |

To go to 100 M: first the planned FLAME speed work (without it 100 M takes months), then
`configs/rust-100m-stage1.toml` and `rust-100m-stage2.toml` (same data and tokenizer), then
set Lintless's `base_run = "../FLAME/runs/rust-100m-stage2"` and fine-tune again.

---

## Command reference

| task | command |
|---|---|
| build | `cargo build --release --features cuda` |
| test / lint | `cargo test --release` · `cargo clippy --workspace --all-targets` |
| download data | `tools/fetch_corpus.sh all\|code\|docs\|errors\|english` |
| tokenizer | `flame tokenizer train --input … --vocab-size N --out F` · `flame tokenizer byte --out F` |
| tokenise data | `flame prepare --input … --ext … --tokenizer T --out D` |
| train / resume | `flame train --config C --run R [--max-minutes N]` |
| pause | Ctrl+C · create the file `R/PAUSE` · `--max-minutes` |
| progress | `flame status --run R` |
| evaluate | `flame eval --run R` (per-source loss for mixes) |
| sample text | `flame generate --run R --prompt "…" [--temperature T]` |
| sizes / time | `flame params --config C` · `flame estimate --config C --tflops X` |
| start over | add `--fresh` (the old run is renamed to a backup, never deleted) |

## Folder layout after everything

```text
FLAME/
  data/raw/…            downloaded text (git-ignored)
  data/tokenizers/…     tokenizers
  data/<source>/        train.bin, val.bin, meta.json per source
  runs/<run>/           config.toml, checkpoints/, best/, metrics.jsonl, sessions.log
```
