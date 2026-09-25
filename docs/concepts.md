# Concepts → code

Each component FLAME implements, what it does, why, and where it lives. Read this alongside
the code; every item is small enough to read in a few minutes.

| concept | one-line idea | source | code |
|---|---|---|---|
| Tokenizer (byte) | each UTF-8 byte is a token (vocab 256 + 2 specials) | – | `tokenizer.rs` `Tokenizer::Byte` |
| Tokenizer (BPE) | repeatedly merge the most frequent adjacent pair; ≈ 3–4× fewer tokens than bytes | Sennrich et al. 2016; GPT-2 | `tokenizer.rs` `BpeTokenizer::train` |
| Pre-tokenisation | split text into words/numbers/punctuation first so merges don't cross them | GPT-2 | `tokenizer.rs` `pretokenize` |
| Embedding | token id → learned vector of size `d_model` | – | `model.rs` `tok_emb` |
| RMSNorm | `x / sqrt(mean(x²)+ε) · g`: normalise the scale without centring | Zhang & Sennrich 2019 | `nn.rs` `RmsNorm` |
| Pre-norm residual | `x + f(norm(x))`: stable deep training | GPT-2, LLaMA | `nn.rs` `Block` |
| Causal self-attention | each position mixes the values of earlier positions, weighted by `softmax(q·k/√d)` | Vaswani et al. 2017 | `nn.rs` `CausalSelfAttention` |
| Causal mask | `-∞` above the diagonal so position *t* never sees *t+1…* | Vaswani et al. 2017 | `nn.rs` `causal_mask` |
| RoPE | rotate q and k by position-dependent angles, so q·k depends on relative distance | Su et al. 2021 | `nn.rs` `Rope` |
| SwiGLU MLP | `W2(SiLU(W1 x) ⊙ W3 x)`, a gated MLP; hidden ≈ 8/3·d | Shazeer 2020; LLaMA | `nn.rs` `SwiGlu` |
| Tied embeddings | reuse the embedding matrix as the output projection; saves V·d params | Press & Wolf 2017 | `model.rs` |
| Init | N(0, 0.02); residual output projections scaled by 1/√(2·layers) | GPT-2 | `nn.rs` `Block::new` |
| Cross-entropy | `-log p(correct next token)`, averaged; optional mask to ignore padding/prompt | – | `model.rs` `lm_loss` |
| AdamW | Adam with *decoupled* weight decay (decay applied to weights, not gradients) | Loshchilov & Hutter 2019 | `optim.rs` |
| Warmup + cosine LR | ramp LR up linearly, then decay along a half cosine to `min_lr` | GPT-3, LLaMA | `optim.rs` `lr_at` |
| Grad clipping | rescale all gradients if their global L2 norm exceeds a threshold | – | `trainer.rs` |
| Gradient accumulation | sum gradients of several micro-batches before one step: big batches on small GPUs | – | `trainer.rs` |
| Perplexity | `exp(loss)`: "how many tokens the model is choosing between" | – | `flame eval` |
| Sampling | temperature scales logits; top-k keeps the k most likely tokens | – | `generate.rs` |

## Quick sanity numbers

- A random model's initial loss ≈ `ln(vocab)`: 5.55 for bytes (258), 8.3 for a 4096-token BPE.
  A very different starting loss points to an init bug.
- Byte-level Shakespeare reaches ≈ 1.5 validation loss; TinyStories with a 4k BPE at 5 M params ≈ 1.5–1.8.
- Training FLOPs ≈ 6 × params × tokens (see [hardware-and-time-estimates.md](hardware-and-time-estimates.md)).

## Where to go next (the "encounter → read → implement → test → measure → document" loop)

- **KV cache** for generation (a runtime exercise; generation currently recomputes the full context).
- **bf16 / TF32 matmuls** and **fused attention** (flash-attention): the main speedups for FLAME.
- **Grouped-query attention** (LLaMA-2/3): fewer K/V heads, smaller KV cache.
- **Weight-tying ablation**, **learning-rate sweeps**, **μP** for transferring hyper-parameters across sizes.
