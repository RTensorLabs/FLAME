# Hardware and training-time estimates (RTX 3070 Ti Laptop, 8 GB)

These are **estimates**. Your real numbers depend on the laptop's power limit (the 3070 Ti
Laptop ships at 80–150 W), cooling, driver and whether you're on WSL2. After 10 minutes of
any run, `flame status --run runs/<name>` prints your *measured* throughput and time left.
Use that instead of this page from then on.

## Measured DGX Spark result (2026-11-02)

The laptop estimates below apply to native Candle training. The same 29.89 M
Transformer was also measured on a shared NVIDIA GB10 DGX Spark, at context 1024,
micro-batch 8 × accumulation 16 (131,072 tokens/step):

| training backend | measured tokens/s | projected optimizer time for 1,310,720,000 tokens |
|---|---:|---:|
| native Candle fp32, TF32 matmuls enabled | about 5,400 | about 67 h |
| optional PyTorch BF16 autocast, fused attention + AdamW | 52,000–54,000 | 6.7–7.0 h |

The native path materializes full attention matrices and uses unfused fp32
operations. The accelerated path uses lower-precision activations and fused
kernels, with fp32 master weights and optimizer moments. Both consume the full
7500 + 2500 step budget; the Rust batch bridge preserves native sampling.
The laptop's estimated 13–50 h range is not a DGX measurement.

The initial wall-clock allowance for the accelerated run is **8–10 h**, including
evaluation, recovery archives, uploads and load variation. These are projections
from a short benchmark and early real-data steps; final elapsed time and model
quality remain to be measured. See [dgx-training.md](dgx-training.md) for the run,
verification results and storage locations. The optional backend has only been
validated here for the dense 30M configuration; do not extrapolate its speed to
research models, Lintless or 100M without measuring them.

## Your time budget

You said: **3–4 h on weekends, 0–1 h on weekdays.** This page assumes:

| | hours |
|---|---:|
| weekend | ≈ 3.5 h |
| weekdays (5 × ~0.5 h on average) | ≈ 2.5 h |
| **total** | **≈ 6 h of training per week** |

If the 3–4 h is per weekend *day*, you have about 10 h/week; divide the "weeks" columns
below by ~1.7.

## The GPU

| spec | RTX 3070 Ti Laptop |
|---|---|
| VRAM | 8 GB GDDR6, ≈ 448 GB/s |
| FP32 peak | ≈ 15–20 TFLOP/s (depends on power limit) |
| BF16 tensor-core peak | ≈ 60–80 TFLOP/s dense |
| architecture | Ampere (supports bf16 and TF32) |

**Achieved** throughput is much lower than peak:

- **FLAME (Candle, FP32, un-fused kernels, no flash-attention):** assume **1.5–6 TFLOP/s**,
  about 3 on average. Models under ~2 M params are limited by kernel-launch overhead, not FLOPs,
  so they are slower than the formula says.
- **Lintless** is a FLAME fine-tune, so it has the same throughput as FLAME.

## The formula

```text
training FLOPs ≈ 6 × N × D        N = parameters used per token, D = training tokens
               (+ attention: 12 × layers × seq_len × d_model per token, matters for small models)
hours          = FLOPs / (achieved TFLOP/s × 10¹²) / 3600
```

`flame estimate --config <file> --tflops <measured>` does this for any config.

## Estimates

| run | config | params | tokens | FLOPs | GPU hours | calendar at 6 h/week |
|---|---|---:|---:|---:|---:|---|
| overfit gate | `configs/overfit.toml` | 0.12 M | 0.4 M | – | ~1 min (30 s on CPU) | now |
| Tiny Shakespeare | `configs/tiny-shakespeare.toml` | 0.89 M | 25 M | 1.5e14 | 5–20 min | one evening |
| TinyStories | `configs/tinystories-5m.toml` | 5.9 M | 98 M | 3.7e15 | 0.5–1.5 h | one evening/weekend |
| research: 5 synthetic-task runs | `flame-research/configs/tasks-*.toml` | ~5 M each | 20 M each | – | 15–45 min each ⇒ **2–4 h** | one weekend |
| research: 5 TinyStories runs | `flame-research/configs/stories-*.toml` | ~5–6 M each | 98 M each | ~4e15 each | 0.5–2 h each ⇒ **3–8 h** | 1–2 weeks |
| Lintless data (seeds) | – | – | – | – | seconds (CPU) | – |
| Lintless data (cargo projects) | – | – | – | – | 1–3 h per 2 000 samples (CPU only; can run while you do other things) | – |
| Rust + English base model (stage 1 + 2, context 1024) | `configs/rust-30m-stage1.toml` + `stage2` | 30 M | 1.3 B | 2.7e17 | **13–50 h** (≈ 25) | **2–8 weeks** (≈ 4) |
| Lintless fine-tune of the base model | `lintless/configs/finetune-30m.toml` | 30 M (all weights) | ≈ 7.5 M (5 000 samples × ~500 tokens × 3 epochs) | ≈ 1.5e15 | **10–40 min** | one evening |
| Lintless evaluation (500 samples) | – | 30 M | – | – | 30–80 min (no KV cache yet) | one evening |
| ~100 M base model | `configs/rust-100m-stage1/2.toml` | ~100 M | 2 B | 1.3e18 | 60–230 h (≈ 115) | 3–9 months |

Modular research models do extra work per step: a host sync for routing decisions and
index-select/scatter for sparse dispatch. Expect **1.5–3× the wall-clock of a dense model with
the same FLOPs** at this scale. That overhead is itself a finding; record it.

### What this means in practice

- Phases 1–2 (FLAME) and the whole first research round fit in **the first 2–3 weeks**.
- The 30 M Rust base model is a **1–2 month background project**: train it on weekday
  evenings with `--max-minutes 45`. Lintless depends on it.
- Once the base model exists, the Lintless fine-tune takes **one evening**. Collect the
  Lintless data (CPU only) while the base model trains.
- **Skip ~100 M for now.** It needs about 4 months of your GPU time. If you need it later,
  benchmark the optional BF16/fused backend first; it is validated for the dense 30M
  run, with native Candle fused training still future work.

## Memory (8 GB VRAM)

Training memory ≈ **16 bytes per parameter** (fp32 weights 4 + grads 4 + AdamW m, v 8) **+ activations**.

| params | weights+grads+optimizer | fits in 8 GB? |
|---:|---:|---|
| 6 M | 0.1 GB | easily |
| 30 M | 0.5 GB | yes; activations dominate, keep micro-batch 8 at seq 512 |
| 100 M | 1.6 GB | yes, with micro-batch 4 × seq 1024 and gradient accumulation |
| 30 M at context 1024 | 0.5 GB + ≈ 1.6 GB attention per 2 sequences | yes, micro-batch 2 × accumulation 64 |
| 30 M at context 2048 | 0.5 GB + ≈ 6.4 GB per sequence | no headroom: needs memory-efficient attention first |

Attention activations grow with `micro_batch × heads × seq_len²`. **If you run out of memory,
halve `micro_batch_size` and double `grad_accum_steps`.** The effective batch, the results
and the time stay about the same.

## Laptop care for long training

- Train **plugged in**, on a hard surface or a cooling stand.
- Disable sleep while training (Windows: Settings → System → Power → "When plugged in, put
  my device to sleep after: Never"). The 10-minute auto-checkpoint protects you if you forget.
- Watch temperatures with `nvidia-smi -l 5`. Sustained 85 °C+ means throttling. Lowering the
  power limit costs a little speed but lets the laptop run cooler and quieter.
- You can shut down any time: Ctrl+C (or just close the lid after `--max-minutes` ends).
  Rerunning the same command resumes.
