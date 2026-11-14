---
title: FLAME 30M Rust
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
license: mit
short_description: 30M Rust code model, trained from scratch, runs in-browser
---

# FLAME 30M

A 30-million-parameter decoder-only Transformer (8 layers, d_model 512, 8k BPE vocabulary,
1024-token context), trained from scratch with the FLAME framework on Rust code, Rust docs,
English text and rustc error messages. Training took 2500 steps (328M tokens). Final validation
loss: 1.70 overall, 0.97 on Rust code.

It is a base model: it continues text and does not follow instructions.

The model code (Rust + [Candle](https://github.com/huggingface/candle)) is compiled to
WebAssembly and runs in the visitor's browser. No server does any work.
