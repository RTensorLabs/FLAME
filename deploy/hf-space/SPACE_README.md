---
title: FLAME 30M Rust
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
license: mit
short_description: 30M-parameter Rust code model trained from scratch in Rust
---

# FLAME 30M

A 30-million-parameter decoder-only Transformer (8 layers, d_model 512, 8k BPE vocabulary),
trained from scratch with the FLAME framework on Rust code, Rust docs, English text and rustc
error messages. Training took 2500 steps (328M tokens). Final validation loss: 1.70 overall,
0.97 on Rust code.

It is a base model: it continues text and does not follow instructions.

The Space builds the `flame` Rust CLI (CPU only) and wraps `flame generate` in a Gradio page.
