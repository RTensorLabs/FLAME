# FLAME developer guide

How the code is organised, how data flows through it, **why** each design decision was made
(and what was rejected), the practices the code follows, and how to extend it. Read this
before changing anything. For the chronological story of how it was built, including the
bugs found along the way, see [build-log.md](build-log.md).

---

## 1. Repository tour

```text
FLAME/
├── Cargo.toml                 workspace: two crates, shared dependency versions, build profiles
├── crates/
│   ├── flame/                 the library (everything reusable)
│   │   ├── src/
│   │   │   ├── lib.rs         module list, select_device(), build_transformer()
│   │   │   ├── config.rs      TOML config structs: ModelConfig, TrainConfig, DataConfig, RunConfig
│   │   │   ├── tokenizer.rs   byte + BPE tokenizers, pre-tokeniser, BPE trainer
│   │   │   ├── data.rs        token files (mmap), Batch, BatchSource trait, deterministic batching
│   │   │   ├── nn.rs          RmsNorm, Rope, causal_mask, CausalSelfAttention, SwiGlu, Block
│   │   │   ├── model.rs       LanguageModel trait, ModelOutput, Transformer, lm_loss
│   │   │   ├── init.rs        seeded, device-independent weight initialisation
│   │   │   ├── optim.rs       AdamW (with saveable state), grad_norm, lr_at (warmup+cosine)
│   │   │   ├── trainer.rs     Trainer::run: the training loop; evaluate(); format_eta()
│   │   │   ├── control.rs     StopControl: Ctrl+C / --max-minutes / PAUSE file
│   │   │   ├── checkpoint.rs  atomic checkpoints, TrainState, best-model weights
│   │   │   ├── generate.rs    sampling (temperature, top-k, greedy)
│   │   │   └── estimate.rs    6·N·D training-time estimator
│   │   └── tests/training.rs  end-to-end tests (causality, overfit, pause/resume)
│   └── flame-cli/src/main.rs  the `flame` binary: tokenizer, prepare, train, status, generate, …
├── configs/                   example run configs (overfit → Shakespeare → TinyStories → Rust 30M)
├── examples/overfit.txt       the tiny text for the Phase 1 overfit gate
├── tools/reference_check.py   PyTorch re-implementation used to cross-check the forward pass
└── docs/                      you are here
```

Size: about 1 900 lines of library code, 500 lines of CLI, 140 lines of integration tests.

### Where to start reading

1. `nn.rs` → `model.rs`: the model, top to bottom.
2. `trainer.rs`: one loop that uses everything else.
3. `data.rs`, `optim.rs`, `checkpoint.rs`, `control.rs`: the pieces the trainer calls.
4. `flame-cli/src/main.rs`: how the pieces are wired to commands.

---

## 2. How data flows

### Preparing data (`flame tokenizer` + `flame prepare`)

```text
text files ──► pretokenize() ──► BPE merges learned ──► tokenizer.json
text files ──► 64 MB blocks ──► split into documents ──► encode (parallel, rayon)
           ──► every Nth document → val.bin, the rest → train.bin   (u16 little-endian ids)
           ──► meta.json (token counts, bytes/token)
```

### Training (`flame train`)

```text
config.toml ─► RunConfig ─► build_transformer() (seeded init) ─► Trainer::run()
                                                                   │
  ┌────────────────────────────────────────────────────────────────┘
  │ load checkpoint if one exists (weights, AdamW m/v, step, tokens, timings)
  │ loop until max_steps or a stop request:
  │   StopControl::should_stop()?  → save + exit ("paused")
  │   lr = lr_at(step)
  │   for each micro-batch: batch = source.train_batch(step, micro)   ← pure function of (seed, step, micro)
  │                         out = model.forward(input); loss = lm_loss(+aux); backward; accumulate grads
  │   clip by global norm; AdamW.step()
  │   log every log_every; evaluate every eval_every (+ save best/); checkpoint every N minutes
  └ save final checkpoint
```

### A forward pass (`Transformer::forward`)

```text
ids (B,T) ─► tok_emb ─► [Block × n_layers] ─► RmsNorm ─► x · Eᵀ (tied head) ─► logits (B,T,V)
Block:  x = x + Attn(RmsNorm(x))      Attn: q,k,v = Wx; RoPE(q,k); softmax(qkᵀ/√d + mask) v; Wo
        x = x + SwiGLU(RmsNorm(x))    SwiGLU: W_down(SiLU(W_gate x) ⊙ W_up x)
```

---

## 3. Design decisions and why

Each entry: **decision** → why → alternatives considered.

### Language and framework

**Rust + Candle.** This was your chosen direction: one language from research to the edge
runtime, no Python in the deployment path, and Candle's autograd is small enough to understand.
→ *Rejected for FLAME itself:* PyTorch (faster to iterate, but it is the thing we want to
eventually not depend on). PyTorch *is* used where it's the right tool: the reference check
and Lintless fine-tuning.

**Two crates, not eight.** Your notes listed `flame-core`, `flame-nn`, … `flame-monitor`.
Splitting before the boundaries are known creates churn (moving code between crates, feature
flags, versioning). There is one library crate (`flame`) whose modules mirror those names, plus
the CLI. → Extract a crate when a boundary actually hurts, e.g. when the runtime needs
`nn` + `model` without the trainer.

**Separate repos for the research and Lintless.** They have different lifecycles: the
research is experimental and changes daily; Lintless is an application (data pipeline +
fine-tuning + a `fix` tool). FLAME stays a stable base. Both depend on FLAME by relative
path (`../FLAME/crates/flame`), so everything can be edited together on one laptop without
publishing versions.

**Lintless is a fine-tune of a FLAME model, not of an external one.** That is the plan in
the research notes (base model → diagnostic data → instruction tuning). It keeps the whole
stack in Rust and owned end to end, and it doubles as the test of whether a small
from-scratch domain model is useful. It reuses FLAME's `Transformer`, tokenizer, `Trainer`
(pause/resume included) and `generate` unchanged; the only new FLAME-facing code is a
`BatchSource` for masked instruction data.

**Context length 1024 for the Rust base model.** Lintless needs a diagnostic + a code window
+ the fix in one sequence (≈ 500–700 tokens). FLAME's attention stores the full T×T score
matrix several times per layer for the backward pass, so memory grows with T²: 1024 fits in
8 GB at micro-batch 2, while 2048 leaves no headroom and 8192 would need ~100 GB. Candle's
fused softmax has no backward pass, so longer contexts need a memory-efficient
(recompute-based) attention in FLAME first. That is on the roadmap.

### Model

**LLaMA-style decoder (RMSNorm, RoPE, SwiGLU, pre-norm, no biases).** This is the modern
default that your reading (Attention Is All You Need + LLaMA) prepared you for, and it is what
the baselines in current papers use. A GPT-2-style model (LayerNorm, learned positions, GELU)
would be a weaker baseline.

**Everything written from primitive ops.** `silu`, `RmsNorm`, `softmax` and RoPE are
composed from `exp`, `mean`, `mul`, … instead of Candle's fused kernels. Reasons:
(1) each concept is readable in a few lines; (2) every op is guaranteed to have a backward pass
(some fused Candle ops are forward-only); (3) it matches the PyTorch reference exactly.
→ *Cost:* slower. Fused kernels / flash-attention are on the roadmap as a deliberate speed step.

**Tied embeddings by default.** For small models the embedding matrix is a large share of
the parameters (4096 × 256 = 1 M of 5.9 M). Tying saves it and usually helps at this scale.
Configurable with `tie_embeddings`.

**Init N(0, 0.02), residual output projections scaled by 1/√(2·layers)** (GPT-2). This keeps
the residual stream's variance from growing with depth. Sanity check: the initial loss must
be ≈ ln(vocab).

**Causal mask uses −1e9, not −∞.** Both give 0 after softmax, but −∞ can produce NaN
(−∞ − (−∞)) in edge cases and in reduced precision. −1e9 is safe in f32 and bf16.

**No KV cache yet.** Generation re-runs the whole context for each new token. That's fine for
the tiny models of Phases 1–2 and keeps the model free of caching state. It's the natural
first "runtime" exercise.

### Tokenizer

**Own byte-level BPE instead of the `tokenizers` crate.** It's a core concept to understand,
it's about 250 lines, and it avoids a heavy dependency. The layout is: ids 0–255 are raw
bytes, so **any** input is encodable and nothing is ever "unknown"; 256 = `<|endoftext|>`,
257 = `<|pad|>`; merges start at 258.

**Pre-tokenisation tuned for prose *and* code.** Identifiers (`[A-Za-z_][A-Za-z0-9_]*`) stay
whole, numbers are split into chunks of at most 3 digits (so the vocabulary isn't wasted on
numbers), punctuation runs stay together, a single space attaches to the following word
(" fn", " let"), and a newline + indentation is one chunk (important for code). It's lossless:
concatenating the chunks gives back the input, which is tested.

**Incremental BPE training with a lazy max-heap.** Naively recounting all pairs after every
merge is O(merges × corpus). Instead, pair counts are updated only for words containing the
merged pair, and stale heap entries are skipped. Ties break on the smallest pair, so
training is deterministic.

**u16 token files.** This caps the vocabulary at 65 536, which is plenty for our models, and
halves disk and RAM use compared with u32. It's the same format as nanoGPT.

### Data

**Memory-mapped files.** A 2 GB dataset costs almost no RAM; the OS pages it in.

**Random windows, not epochs.** Each sequence is a random contiguous slice. This is simple,
works for any dataset size, and is standard for LM pre-training.

**Batches are a pure function of `(seed, step, micro)`** (`step_rng` = splitmix64 mixing).
This is the key to exact pause/resume: there is no data-loader state to save. It also means a
run is reproducible from its config alone.

**Validation uses a fixed seed independent of the training seed**, so val losses are
comparable across runs with different seeds.

**Val split by document** (every Nth document). Splitting by position could put half a
document in train and half in val.

### Optimisation

**Hand-written AdamW** instead of `candle_nn::AdamW`. Candle's optimiser keeps its moment
buffers private, and a pausable trainer *must* save them: resuming Adam with zeroed moments
causes a loss spike and is a different run. The implementation follows the paper
(decoupled weight decay, bias correction) and is about 60 lines.

**No weight decay on vectors (rank < 2)**: norm gains shouldn't be pulled toward 0. This is
standard practice.

**Warmup + cosine to `min_lr`.** This is the LLaMA/GPT-3 schedule. Warmup prevents early
instability when Adam's second-moment estimates are still noisy.

**Global-norm gradient clipping at 1.0**: a cheap guard against the occasional huge gradient.

**Gradient accumulation** is summed manually over micro-batches, with the loss divided by the
number of micro-batches. It lets an 8 GB GPU train with large effective batches.

**Parameters sorted by name** in the optimiser, so the checkpoint layout is stable
regardless of HashMap iteration order.

### Reproducibility

**Seeded host-side initialisation (`init.rs`).** Candle can't seed its CPU RNG, and GPU RNG
streams differ from CPU ones. A custom `VarBuilder` backend draws every initial weight from
a seeded `StdRng` on the host, then uploads it. The same seed gives the same model on CPU and
GPU, and the reference check can rebuild the exact weights.

**`LanguageModel::set_mode(training, step, micro)`**: a hook so that models with
training-only randomness (the research model's routing exploration) derive it from the step
and stay deterministic across pauses.

### Pause / resume and checkpoints

The use case drove this: one laptop, 30-minute to 4-hour sessions, shutdowns in between.

- **Three stop mechanisms** (`control.rs`): Ctrl+C / SIGTERM / SIGHUP via the `ctrlc` crate
  with its `termination` feature; a time budget that stops *before* overrunning (it predicts
  the next step's duration and keeps ~30 s of headroom); and a `PAUSE` file, which works from
  any other terminal or file manager. A second Ctrl+C force-quits, because a hung save must
  never trap you.
- **Periodic checkpoints by wall-clock minutes**, not steps, because what you lose in a crash
  is *time*.
- **Atomic save**: write `latest.tmp/`, rename `latest` → `latest.old`, `latest.tmp` →
  `latest`, delete `latest.old`. At any instant at least one complete checkpoint exists.
  `state.json` is written last and marks a directory as complete.
- **Frozen config**: a run copies its config to `runs/<name>/config.toml`. Resuming with a
  different `[model]` or `[data]` is refused (the weights wouldn't match). `[train]` may
  change, so `max_steps` can be extended.
- **`--fresh` renames, never deletes**: training time is too expensive to lose to a typo.

### CLI

**One binary with subcommands (clap derive)**: discoverable (`flame --help`) and each
command small. Commands print what to do next (e.g. "run the same command again to resume").
`status` and `estimate` exist because of the limited training time: you should always know
how many sessions are left.

---

## 4. Practices

**Tests prove the properties that matter, not just "it runs."**

| test | property |
|---|---|
| `model_is_causal` | future tokens cannot change past logits (information leak = fake results) |
| `overfits_tiny_dataset` | the model + optimiser + data path can learn (the Phase 1 gate) |
| `pause_and_resume_is_bit_identical` | 30 steps straight == 12 + 11 + 7 steps, bit for bit |
| `ctrl_c_pauses_and_saves` | Ctrl+C and the PAUSE file stop cleanly with a checkpoint |
| tokenizer tests | pre-tokenisation is lossless; BPE round-trips (incl. unseen Unicode); save/load |
| `schedule_shape` | warmup and cosine hit the expected values |
| `tools/reference_check.py` | the forward pass equals an independent PyTorch implementation (5e-7) |

**Verify against an independent implementation.** Bugs in RoPE layout, masking or norm
placement don't crash; they silently make the model worse. The PyTorch check is written from
the paper descriptions, not translated from the Rust.

**Determinism by construction.** No hidden global RNG. Every random choice is derived from
the config seed and the step.

**Fail loudly with a helpful message.** Config validation (`d_model % n_heads`, even head
dim), NaN loss aborts with "lower the learning rate", missing files suggest the command to
create them.

**Comments explain *why*, not *what*.** Every module starts with a doc comment giving its
purpose and the key idea; tricky lines (atomic rename order, `-1e9`, stale heap entries)
are commented inline.

**Clippy-clean** (`cargo clippy --workspace --all-targets` gives no warnings) and rustfmt
style. Tests run with `opt-level = 2` because unoptimised tensor math is painfully slow.

**Never destroy training time.** `--fresh` backs up; checkpoints are atomic; Ctrl+C saves.

### Commands to run before committing

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --release
python tools/reference_check.py runs/refcheck   # after `flame export-reference`, if you touched nn.rs/model.rs
```

---

## 5. How to extend

**A new layer / model**: add it in `nn.rs` / `model.rs` and implement `LanguageModel`
(`forward`, `max_seq_len`, `n_params`, `flops_per_token`, `describe`). The trainer,
evaluator and generator work unchanged. Add it to the causality test. Use
`get_with_hints(..., Init::Randn{..})` for weights (the seeded backend doesn't support
Kaiming init; it errors on purpose).

**A new data source**: implement `BatchSource` (`train_batch(step, micro)` must be
deterministic; `val_batches` fixed). Use `Batch::mask` to exclude positions from the loss.
Examples: `TokenDataset`, `InMemoryDataset`, and `SyntheticTasks` in flame-research.

**An auxiliary loss** (e.g. routing): return it in `ModelOutput::aux_loss`; the trainer adds
it. Scalar diagnostics go in `ModelOutput::stats` and are logged automatically.

**A new CLI command**: add a variant to `Cmd` in `flame-cli/src/main.rs` and a match arm.

**A new config field**: add it with `#[serde(default = …)]` so old configs keep loading.
The structs use `deny_unknown_fields`, so typos in configs are errors, not silent defaults.

---

## 6. Known limitations / next steps

- FP32 only, no fused attention: roughly 2–4× slower than possible (see roadmap).
- No KV cache: generation is O(T²) per token.
- `metrics.jsonl` may repeat a few steps after a crash (those steps were redone after resuming from the last checkpoint).
- `flame prepare` tokenises one 64 MB block at a time in parallel; very large corpora take a while.
- Vocabulary ≤ 65 536 (u16 files).
