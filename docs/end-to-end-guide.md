# FLAME end-to-end guide

Everything from a fresh laptop to a trained Rust base model, in order. Every step lists
**what to run, what you should see, and how long it takes** on an RTX 3070 Ti Laptop (8 GB).
Commands are run from the `FLAME/` directory unless stated otherwise.

The overall plan across the three repos:

```text
FLAME          ① setup → ② build → ③ test → ④ sanity runs → ⑤ Rust corpus → ⑥ train rust-30m (weeks)
lintless       collect data (CPU, during ⑥) ─────────────────────────────► ⑦ fine-tune → evaluate
flame-research independent experiments (TinyStories / synthetic), any time the GPU is free
               ⑧ judge Lintless → good: use it · weak: more data → still weak: scale to 100 M
```

---

## ① Setup (once, ~1 hour)

Follow [setup.md](setup.md): WSL2 + Ubuntu, NVIDIA driver on Windows, CUDA toolkit in WSL,
Rust, then clone the three repos **side by side**:

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/RTensorLabs/FLAME
git clone https://github.com/RTensorLabs/flame-research
git clone https://github.com/RTensorLabs/lintless
```

Check: `nvidia-smi` shows the GPU, `nvcc --version` works, `cargo --version` works.

## ② Build (~5 min the first time)

```bash
cd ~/code/FLAME
cargo build --release --features cuda
echo 'export PATH="$HOME/code/FLAME/target/release:$PATH"' >> ~/.bashrc && source ~/.bashrc
flame --help
```

- The CUDA build compiles GPU kernels the first time; later builds are fast.
- If it fails with a compute-capability error: `export CUDA_COMPUTE_CAP=86`, then rebuild.
- CPU-only build (works anywhere, much slower): `cargo build --release`.

## ③ Test (~1 min)

```bash
cargo test --release
```

Expect every `test result: ok`. What the tests prove is listed in
[developer-guide.md](developer-guide.md#4-practices): causality, overfitting, bit-exact
pause/resume, Ctrl+C/PAUSE handling, and tokenizer round-trips.

Optional, if you ever change `nn.rs` or `model.rs` (needs `pip install torch safetensors`):

```bash
flame export-reference --config configs/tiny-shakespeare.toml --out runs/refcheck
python3 tools/reference_check.py runs/refcheck      # expect "OK", max diff ≈ 1e-6
```

## ④ Sanity runs: prove the GPU setup works (~30 min)

**4a. Overfit gate (1 min).** A tiny model must memorise a paragraph.

```bash
flame tokenizer byte --out data/tokenizers/byte.json
flame prepare --input examples/overfit.txt --tokenizer data/tokenizers/byte.json --out data/overfit --val-fraction 0.25
flame train --config configs/overfit.toml --run runs/overfit
```

Expect the first line `[flame] running on CUDA GPU 0` (if it says CPU, the build lacks
`--features cuda`) and a final `val_loss` around **0.05**.

**4b. Tiny Shakespeare (5–20 min).** Your first real training run, and your throughput measurement.

```bash
curl -L -o data/raw/shakespeare.txt --create-dirs https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
flame prepare --input data/raw/shakespeare.txt --tokenizer data/tokenizers/byte.json --out data/shakespeare --separator "\n\n" --val-fraction 0.1
flame train --config configs/tiny-shakespeare.toml --run runs/shakespeare
flame generate --run runs/shakespeare --prompt "ROMEO:" --max-tokens 300
flame status --run runs/shakespeare
```

Expect val loss ≈ 1.5 and Shakespeare-looking text. Try one pause: press Ctrl+C in the
middle, rerun the same command, and check that it says `resumed from step …`.

**Write down** the `throughput … TFLOP/s achieved` line from `flame status`. Use it for every
estimate from now on: `flame estimate --config configs/rust-30m.toml --tflops <your number>`.

## ⑤ Build the Rust corpus (~1 hour, mostly downloading)

```bash
tools/fetch_rust_corpus.sh                   # → data/raw/rust + ~/.cargo/registry/src (1–3 GB of .rs)
flame tokenizer train --input data/raw/rust ~/.cargo/registry/src --ext rs \
      --vocab-size 8192 --out data/tokenizers/rust-8k.json --sample-mb 200
flame prepare --input data/raw/rust ~/.cargo/registry/src --ext rs \
      --tokenizer data/tokenizers/rust-8k.json --out data/rust
cat data/rust/meta.json
```

Check in `meta.json`:
- `bytes_per_token` ≈ 3–4 (a good code tokenizer);
- `train_tokens`: how much unique data you have.

**Match `max_steps` to your data.** One step is 2 × 64 × 1024 = 131 072 tokens. Repeating
data up to about 4 epochs is nearly as good as fresh data (Muennighoff et al., 2023); more
than that wastes GPU time. So:

```text
max_steps = min(10 000, 4 × train_tokens / 131 072)
```

For example, 300 M train tokens → `max_steps = 9 150`. Edit `configs/rust-30m.toml` **before** starting.

## ⑥ Train the 30 M Rust base model (13–50 GPU hours → 2–8 weeks)

```bash
flame params   --config configs/rust-30m.toml
flame estimate --config configs/rust-30m.toml --tflops <your number from 4b>
flame train    --config configs/rust-30m.toml --run runs/rust-30m --max-minutes 45    # weekday
flame train    --config configs/rust-30m.toml --run runs/rust-30m --max-minutes 210   # weekend
```

Run the same command each session; it resumes. Between sessions:

```bash
flame status   --run runs/rust-30m                              # progress, hours, time left
flame generate --run runs/rust-30m --prompt "fn main() {" --temperature 0.7
tail -n 5 runs/rust-30m/metrics.jsonl
```

**What healthy training looks like:**
- the loss drops fast in the first ~500 steps, then slowly;
- `val_loss` follows `loss` (a gap that keeps growing = over-fitting, too many epochs);
- `gnorm` mostly below ~1–2 after warm-up;
- samples go from symbol soup → Rust-looking tokens → plausible functions with matching braces.

**Problems:** NaN → lower `lr`; out of memory → `micro_batch_size = 1`, `grad_accum_steps = 128`;
slower than estimated → check thermals (`nvidia-smi -l 5`) and that you're not working in `/mnt/c`.
See [troubleshooting.md](troubleshooting.md).

**Meanwhile (CPU):** collect Lintless data, following `lintless/docs/end-to-end-guide.md` steps 3–4.

## ⑦ Evaluate the base model (~10 min)

```bash
flame eval --run runs/rust-30m                      # val loss, perplexity, bits/token
flame generate --run runs/rust-30m --prompt "impl Iterator for" --temperature 0.5
flame generate --run runs/rust-30m --prompt "fn parse_config(path: &Path) -> Result<" --temperature 0.5
```

Record the numbers in [lab-notebook.md](lab-notebook.md). Then go to Lintless (fine-tune,
then evaluate).

## ⑧ Deciding to scale up (30 M → 100 M)

Judge by **Lintless**, not by the base model's loss: the goal is fixing Rust errors. After
`lintless eval` (see the Lintless guide):

| Lintless result on the test set | action |
|---|---|
| "compiles" ≥ rustc-suggestion baseline overall, and clearly above it on codes rustc can't fix | **satisfactory**: keep 30 M, improve data |
| below the baseline, but valid-looking code blocks almost always | more / more varied data (more seeds, more projects, clippy), 1–2 more epochs, then re-evaluate |
| still below after doubling the data, **or** broken code / no code blocks | capacity-limited: **scale to 100 M** |

Scaling up:
1. Do the planned FLAME speed work first (TF32/bf16, fused attention). Without it 100 M takes months.
2. `flame train --config configs/rust-100m.toml --run runs/rust-100m` (same tokenizer and data).
3. In Lintless, set `base_run = "../FLAME/runs/rust-100m"` and fine-tune again; the data is reused.

## Everyday command reference

| task | command |
|---|---|
| build | `cargo build --release --features cuda` |
| test | `cargo test --release` |
| lint | `cargo clippy --workspace --all-targets` |
| train / resume | `flame train --config C --run R [--max-minutes N]` |
| pause | Ctrl+C · `touch R/PAUSE` · `--max-minutes` |
| progress | `flame status --run R` |
| sample | `flame generate --run R --prompt "…"` |
| evaluate | `flame eval --run R` |
| sizes / time | `flame params --config C` · `flame estimate --config C --tflops X` |
| start over | `flame train … --fresh` (the old run is backed up, never deleted) |
