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
| 4 | Rust code corpus + 8k BPE | 🟡 (see below) |
| 4 | 30 M Rust base model (`configs/rust-30m.toml`) | 🟡 (2–8 weeks of evenings) |
| 4 | ~100 M model | ⬜ (after the speed work) |

### Building the Rust corpus (Phase 4)

Easy, license-aware sources of Rust code already on your machine:

```bash
# every crate you've ever downloaded with cargo (often hundreds of MB of real Rust)
flame tokenizer train --input ~/.cargo/registry/src --ext rs --vocab-size 8192 \
      --out data/tokenizers/rust-8k.json --sample-mb 100
flame prepare --input ~/.cargo/registry/src --ext rs \
      --tokenizer data/tokenizers/rust-8k.json --out data/rust
# optional: add your own projects and the standard library source
#   rustup component add rust-src   →   $(rustc --print sysroot)/lib/rustlib/src/rust/library
```

Each file becomes one document. Most crates are MIT/Apache-2.0; keep the corpus for training,
don't redistribute it.

## flame-research

See `flame-research/docs/hypothesis-and-protocol.md`. Order: synthetic tasks (dense-total,
dense-active, MoE, modular-token, modular-chunk) → TinyStories versions → routing and
specialisation analysis → residency simulation → only then a real streaming runtime.

## Lintless

v1: seed + project mutations → 0.5 B LoRA → evaluate vs the base model → `lintless fix`.
v2: clippy data, a 1.5 B QLoRA, more mutation operators. v3: fine-tune the FLAME Rust base
model on the same data and compare.
