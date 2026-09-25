# Datasets: what, where from, and how to get them

Every dataset used by FLAME, flame-research and Lintless: what it is, its licence, its size,
the exact commands to download and **extract** it, where it ends up, and how to turn it into
training data.

The pipeline is the same for every dataset:

```text
download / extract  →  data/raw/<name>/…     (plain text: .txt / .md / .rs files)
tokenizer train     →  data/tokenizers/<name>.json            (once per project)
flame prepare       →  data/<name>/train.bin, val.bin, meta.json
config [data]       →  points at data/<name> (one dataset) or several (a mix)
```

`data/` is git-ignored: datasets never go into the repository.

---

## Overview

| dataset | used for | size (raw) | licence | how to get it |
|---|---|---|---|---|
| overfit paragraph | Phase 1 overfit gate | 1 KB | ours | in the repo: `examples/overfit.txt` |
| Tiny Shakespeare | first real run | 1.1 MB | public domain | `curl` (below) |
| TinyStories | small LM, research experiments | 2.2 GB (train) + 22 MB (valid) | CDLA-Sharing-1.0 | `curl` (below) |
| Rust code | base model (stage 1+2) | 1–3 GB | mostly MIT/Apache-2.0 | `tools/fetch_corpus.sh code` |
| Rust docs (English) | base model (stage 1+2) | ≈ 40–120 MB | mostly MIT/Apache-2.0, CC-BY | `tools/fetch_corpus.sh docs` |
| rustc error explanations | base model (stage 2) | ≈ 0.4 MB (518 codes) | MIT/Apache-2.0 (Rust project) | `tools/fetch_corpus.sh errors` |
| FineWeb-Edu (English) | base model (stage 1+2) | 1.5 GB of text (configurable) | ODC-By 1.0 | `tools/fetch_corpus.sh english` (Python) |
| Lintless samples | Lintless fine-tuning | generated | from your own sources | `lintless collect-*` (Lintless repo) |

Use these datasets for training only. Don't redistribute the downloaded data.

---

## Tiny Shakespeare

All of Shakespeare's plays in one text file; the classic first character-level dataset.

```bash
curl -L --create-dirs -o data/raw/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
flame tokenizer byte --out data/tokenizers/byte.json
flame prepare --input data/raw/shakespeare.txt --tokenizer data/tokenizers/byte.json \
      --out data/shakespeare --separator "\n\n" --val-fraction 0.1
```

`--separator "\n\n"` makes every paragraph a "document", so 10% of the paragraphs become validation data.

## TinyStories

Short, simple stories written by GPT-4 for training tiny models (Eldan & Li, 2023). Hosted on
Hugging Face as plain text, with stories separated by `<|endoftext|>`.

```bash
# full training set, 2.2 GB (~470 M tokens with a 4k BPE)
curl -L --create-dirs -o data/raw/tinystories-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
# optional: the 22 MB validation file, enough for a quick first try
curl -L -o data/raw/tinystories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

flame tokenizer train --input data/raw/tinystories-train.txt --vocab-size 4096 \
      --out data/tokenizers/tinystories-4k.json --sample-mb 50
flame prepare --input data/raw/tinystories-train.txt \
      --tokenizer data/tokenizers/tinystories-4k.json --out data/tinystories
```

Used by `configs/tinystories-5m.toml` and flame-research's `configs/stories-*.toml`.

---

## The Rust + English base-model corpus

Four sources, prepared **separately** so the training config can mix them by weight (see
[end-to-end-guide.md ⑤–⑦](end-to-end-guide.md)):

| source | folder after download | prepared into | what's in it |
|---|---|---|---|
| Rust code | `data/raw/rust-code/` + `~/.cargo/registry/src/` | `data/rust-code` | `.rs` files of ~37 large projects and ~100 popular crates (with their dependencies) |
| Rust docs | `data/raw/rust-docs/` | `data/rust-docs` | Markdown: the Book, Rust by Example, Nomicon, Reference, Async book, Edition guide, API guidelines, all RFCs, rustc dev guide, Cargo book, Rust design patterns, Clippy docs, plus READMEs and docs of the code projects |
| rustc errors | `data/raw/rustc-errors/explanations.txt` | `data/rustc-errors` | `rustc --explain` text for every error code, one document per code |
| English | `data/raw/english/fineweb-edu.txt` | `data/english` | educational web pages (FineWeb-Edu, sample-10BT) |

### Download and extract

```bash
tools/fetch_corpus.sh code       # ~30–60 min; git clones + cargo fetch; keeps only .rs (Markdown moves to rust-docs)
tools/fetch_corpus.sh docs       # ~5 min;  git clones; keeps only .md
tools/fetch_corpus.sh errors     # ~20 s;   runs `rustc --explain` for E0001…E0999
pip install pyarrow requests
tools/fetch_corpus.sh english    # ~20–60 min; downloads parquet files, extracts the text column
# or all at once:  tools/fetch_corpus.sh all
```

**How the extraction works:**
- **code / docs:** each project is shallow-cloned (`git clone --depth 1`, latest version only,
  no history). Files that aren't `.rs` (code) or `.md` (docs) are deleted, and so is `.git`.
  Popular crates are fetched with `cargo add <crate> && cargo fetch` in a throw-away project;
  cargo unpacks them (and their dependencies) into `~/.cargo/registry/src`.
- **errors:** `rustc --explain E0xxx` prints the long explanation of each error code. Codes
  without one are skipped. Each explanation becomes one document.
- **english:** `tools/fetch_english.py` lists the parquet files of
  `HuggingFaceFW/fineweb-edu/sample/10BT` through the Hugging Face API, downloads one at a time
  (~2 GB each), reads the `text` column with pyarrow, writes the pages separated by
  `<|endoftext|>`, and deletes each parquet file after use. Pages under 200 characters are
  skipped. It stops at `ENGLISH_MB` megabytes of text (default 1500 ≈ 375 M tokens).
  Re-running continues where it stopped.

Change the English amount with, e.g., `ENGLISH_MB=800 tools/fetch_corpus.sh english`.

### Tokenizer (one for all four sources)

The vocabulary is fixed before training and shared by every stage and by Lintless, so it is
trained once on a sample of **all** sources (each `--input` gets an equal share of the sample):

```bash
flame tokenizer train \
  --input data/raw/rust-code ~/.cargo/registry/src data/raw/rust-docs data/raw/rustc-errors data/raw/english \
  --ext rs,md,txt --vocab-size 8192 --out data/tokenizers/rust-en-8k.json --sample-mb 250
```

### Prepare (tokenise) each source

```bash
T=data/tokenizers/rust-en-8k.json
flame prepare --input data/raw/rust-code ~/.cargo/registry/src --ext rs  --tokenizer $T --out data/rust-code
flame prepare --input data/raw/rust-docs                       --ext md  --tokenizer $T --out data/rust-docs
flame prepare --input data/raw/rustc-errors                    --ext txt --tokenizer $T --out data/rustc-errors --val-fraction 0.05
flame prepare --input data/raw/english                         --ext txt --tokenizer $T --out data/english
cat data/*/meta.json | grep -E '"train_tokens"|"bytes_per_token"'
```

**Expected sizes** (they vary with what was downloaded):

| source | train tokens | bytes/token |
|---|---|---|
| rust-code | 300 M – 1 B | ≈ 3–4 |
| rust-docs | 10 – 30 M | ≈ 3.5–4 |
| rustc-errors | ≈ 0.1 M | ≈ 3.5 |
| english | ≈ 350 M (at 1500 MB) | ≈ 4 |

### Check the mix before training

At the start of every run, `flame train` prints how many tokens it will draw from each source
and how many times that source's data will be **repeated** (epochs):

```text
[flame] this run draws 983.0 M tokens:
  rust-code      weight  59.0%  data    612.4 M tokens  drawn    580.0 M  ≈   0.9 epochs
  english        weight  35.0%  data    351.0 M tokens  drawn    344.1 M  ≈   1.0 epochs
  rust-docs      weight   6.0%  data     14.2 M tokens  drawn     59.0 M  ≈   4.2 epochs
```

A ⚠ appears above ~16 epochs: that data would be memorised. Lower its weight, or download
more of it, and restart the run with `--fresh`.

---

## Lintless samples

Generated from compiling Rust code, not downloaded. See `lintless/docs/data-pipeline.md` and
the Lintless end-to-end guide. Good sources: the bundled `seeds/`, your own projects, and
any crate you clone normally with git. Not the copies in `data/raw/rust-code/`: those keep only
`.rs` files, so they can't be built.
