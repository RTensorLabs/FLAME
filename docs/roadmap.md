# Roadmap

Status legend: ✅ done · 🟡 ready to run (code exists, needs your GPU time) · ⬜ not started

## FLAME (framework)

| phase | item | status |
|---|---|---|
| 1 | byte + BPE tokenizer | ✅ |
| 1 | embedding, RMSNorm, RoPE, causal attention, SwiGLU, residuals, LM head, cross-entropy | ✅ |
| 1 | reference check against PyTorch | ✅ (5e-7) |
| 2 | dataset, batching, training loop, AdamW, LR schedule, validation, checkpointing, generation | ✅ |
| 2 | **overfit a tiny dataset** | ✅ (loss 0.05) |
| 2 | pause/resume (Ctrl+C, time budget, PAUSE file, auto-checkpoint), bit-exact | ✅ |
| 2 | train Tiny Shakespeare on your GPU | 🟡 |
| 2 | train TinyStories 5 M on your GPU | 🟡 |
| 3 | infrastructure: split crates only when needed (currently `flame` + `flame-cli`) | ✅ (deliberately minimal) |
| 3 | KV cache for generation | ⬜ |
| 3 | bf16/TF32 + fused attention (speed) | ⬜ |
| 3 | memory-efficient attention (recompute in backward) → contexts of 2k–8k | ⬜ |
| 4 | Rust code corpus + 8k BPE | 🟡 (see below) |
| 4 | 30 M Rust base model (`configs/rust-30m.toml`) | 🟡 (2–8 weeks of evenings) |
| 4 | ~100 M model (`configs/rust-100m.toml`) | ⬜ only if Lintless on 30 M isn't good enough (end-to-end guide ⑧), after the speed work |

### Building the Rust corpus (Phase 4)

`tools/fetch_rust_corpus.sh` downloads ~1–3 GB of Rust source (shallow clones of large Rust
projects + popular crates via cargo), then tokenizer training and `flame prepare` follow.
Step by step, including how to match `max_steps` to the amount of data:
[end-to-end-guide.md ⑤](end-to-end-guide.md#-build-the-rust-corpus-1-hour-mostly-downloading).

## flame-research

See `flame-research/docs/hypothesis-and-protocol.md`. Order: synthetic tasks (dense-total,
dense-active, MoE, modular-token, modular-chunk) → TinyStories versions → routing and
specialisation analysis → residency simulation → only then a real streaming runtime.

## Lintless

Lintless is the FLAME Rust base model fine-tuned on compiler-diagnostic data (all Rust).
v1: collect data (seeds + your crates, CPU, while the base model trains) → fine-tune the
30 M base (one evening) → evaluate against rustc's own suggestions → `lintless fix`.
v2: more mutation operators and clippy data, a KV cache for faster generation, longer context.
