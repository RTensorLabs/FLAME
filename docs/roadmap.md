# Roadmap

Status legend: ✅ done · 🔵 running · 🟡 ready to run (code exists, needs your GPU time) · ⬜ not started

## FLAME (framework)

| phase | item | status |
|---|---|---|
| 1 | byte + BPE tokenizer | ✅ |
| 1 | embedding, RMSNorm, RoPE, causal attention, SwiGLU, residuals, LM head, cross-entropy | ✅ |
| 1 | reference check against PyTorch | ✅ (5e-7) |
| 2 | dataset, batching, training loop, AdamW, LR schedule, validation, checkpointing, generation | ✅ |
| 2 | **overfit a tiny dataset** | ✅ (loss 0.05) |
| 2 | native pause/resume (Ctrl+C, time budget, PAUSE file, auto-checkpoint), CPU bit-exact test | ✅ |
| 2 | train Tiny Shakespeare on your GPU | 🟡 |
| 2 | train TinyStories 5 M on your GPU | 🟡 |
| 3 | infrastructure: split crates only when needed (currently `flame` + `flame-cli`) | ✅ (deliberately minimal) |
| 3 | KV cache for generation | ⬜ |
| 3 | native TF32 matmuls | ✅ |
| 3 | optional dense DGX BF16/fused-attention + AdamW trainer, native batch/checkpoint bridge | ✅ (30M validated; [measured results](dgx-training.md)) |
| 3 | fused training directly in the Candle core | ⬜ |
| 3 | memory-efficient attention (recompute in backward) → contexts of 2k–8k | ⬜ |
| 4 | Rust code/docs, compiler errors, English corpus + 8k BPE | ✅ prepared for DGX run (2026-11-02) |
| 4 | data mixing (weighted sources) + staged pre-training (`init_from`) | ✅ |
| 4 | 29.89 M Rust + English base model, two stages | 🔵 DGX run `rust-30m-dgx-20261102`; initial wall-clock projection 8–10 h; final model/evaluation pending |
| 4 | versioned DGX recovery archives, GitHub publisher, PC copies and restore gate | ✅ configured and verified; [recovery guide](dgx-training.md) |
| 4 | ~100 M model (`configs/rust-100m-stage1/2.toml`) | ⬜ only if Lintless on 30 M isn't good enough (end-to-end guide ⑧), after the speed work |

### Building the corpus (Phase 4)

`tools/fetch_corpus.sh` downloads and extracts the four sources (Rust code, Rust docs, rustc
error explanations, English). Details and licences: [datasets.md](datasets.md). Steps:
[end-to-end-guide.md ⑤–⑧](end-to-end-guide.md).

## flame-research

See `flame-research/docs/hypothesis-and-protocol.md`. Order: synthetic tasks (dense-total,
dense-active, MoE, modular-token, modular-chunk) → TinyStories versions → routing and
specialisation analysis → residency simulation → only then a real streaming runtime.

## Lintless

Lintless is the FLAME Rust base model fine-tuned on compiler-diagnostic data (all Rust).
v1: collect data (seeds + your crates, CPU, while the base model trains) → fine-tune the
30 M base (one evening) → evaluate against rustc's own suggestions → `lintless fix`.
v2: more mutation operators and clippy data, a KV cache for faster generation, longer context.
