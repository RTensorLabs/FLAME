# Beginner's guide: the ideas behind everything

Read this once before the end-to-end guide. It explains, in plain words, every idea the
guides and logs use. No maths is required.

## The big picture

A **language model** reads a sequence of pieces of text and predicts the next piece. That's
all it ever does. Writing a paragraph, a function or a bug fix is just predicting the next
piece over and over.

We build ours in three phases:

```text
1. PRE-TRAINING  (weeks)   read lots of Rust code + English → learn the "language"
     stage 1: a broad mix                    (general knowledge)
     stage 2: focused on Rust docs/errors    (specialise, without forgetting)
2. FINE-TUNING   (an hour) show it thousands of "error + code → explanation + fix" examples
3. USE           (seconds) give it a new error, it writes an explanation and a fix
```

Phase 1 happens in FLAME, phases 2–3 in Lintless.

## Words you'll see

**Token.** The pieces a model reads: parts of words or code. `fn main() {` might become
`fn`, ` main`, `()`, ` {`. The model sees numbers (token ids), never letters.

**Tokenizer / vocabulary.** The fixed list of pieces (for us, 8 192 of them) and the rules for
cutting text into them. It is created **once**, before training, from a sample of all the data.
It never changes afterwards, because the model has one learned vector per token.

**Parameters (weights).** The numbers inside the model that training adjusts. "30 M" = 30
million of them. More parameters can hold more knowledge but need more data and time.

**Training step.** The model reads a batch of text, measures how wrong its predictions were,
and nudges every parameter a little to be less wrong. Our base model takes ~10 000 steps.

**Batch.** How much text one step reads. Ours: 2 sequences × 64 accumulation rounds × 1 024
tokens ≈ 131 000 tokens per step. **Gradient accumulation** reads a big batch in small
pieces, so it fits in 8 GB of GPU memory.

**Context length (1 024).** How many tokens the model can look at at once.

**Loss.** The number that says how wrong the predictions are; lower is better.
**Validation loss** is measured on text the model never trains on: if it stops falling
while the training loss keeps falling, the model is memorising (**over-fitting**).

**Perplexity.** `e^loss`: roughly "how many tokens the model is hesitating between". A
perplexity of 5 means it has narrowed 8 192 options down to about 5.

**Epoch.** One full pass over a dataset. Reading the same data 2–4 times is fine; 20 times
means memorising.

**Learning rate (lr).** How big each nudge is. It starts at 0, rises during **warm-up**, and
slowly falls again (a cosine schedule). Too high gives NaN or exploding loss; too low, nothing learns.

**Checkpoint.** A saved copy of the weights + optimizer state + step number. Lets you stop
and resume: FLAME saves one every 10 minutes and whenever you pause.

**Pre-training.** Learning from raw text with no labels. The "labels" are simply the next token.

**Data mix / weights.** Pre-training reads several sources at once. A weight of 0.35 for
English means 35% of the sequences come from English text.

**Stages.** Pre-training split in two: stage 1 on a broad mix, stage 2 starting from stage 1's
weights with a more focused mix and a lower learning rate. Earlier data keeps a share in
stage 2 (**replay**), otherwise the model forgets it.

**Fine-tuning.** Continuing to train a pre-trained model on a small, specific task dataset.
Lintless only computes the loss on the *answer* part of each example (the explanation +
fixed code), not on the question.

**Evaluation.** Measuring how good the model is at the real task. For Lintless: does the fix
compile? Compared against a **baseline** (rustc's own suggestions), because a number alone
tells you nothing.

**GPU / VRAM / CUDA.** The graphics card does the maths ~20–100× faster than the CPU. VRAM
is its memory (8 GB), which limits model and batch size. CUDA is NVIDIA's software that lets
programs use the GPU.

**TFLOP/s.** Trillions of arithmetic operations per second; how fast training runs. Your
GPU's real figure is shown by `flame status`.

## How the three repositories relate

```text
FLAME            the engine: tokenizer, model, trainer, pausing, generation, data tools
  ├─ flame-research   experiments on new architectures (uses FLAME's trainer and layers)
  └─ lintless         the application: fine-tunes a FLAME base model on Rust errors
```

All three are Rust and are cloned next to each other in one folder (`~/code/`).

## What "done" looks like

1. `cargo test --release` passes in every repo.
2. The overfit run reaches loss ≈ 0.05, and Tiny Shakespeare produces Shakespeare-like text.
3. The base model (stage 2) writes plausible Rust and English when prompted.
4. `lintless eval` shows fixes that compile, and more often than rustc's own suggestions on some error types.
5. `lintless fix my_file.rs` helps you with a real error.

Next: [end-to-end-guide.md](end-to-end-guide.md).
