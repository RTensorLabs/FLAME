# Build log: how FLAME was built

A chronological record of how the code was written, in what order, what went wrong, and how
it was verified. It's useful as a model for the "encounter → implement → test → measure →
document" loop, and for knowing which parts have been battle-tested.

## Order of work (and why this order)

1. **Workspace skeleton and dependency check.** Candle 0.11 was the current release. Before
   writing any code, Candle's source was inspected for three things the design depends on:
   - `softmax`, `log_softmax`, `cross_entropy` are composed from primitive ops, so they are
     differentiable;
   - `gather`, `index_select`, `index_add` and `max` all have backward passes;
   - `VarMap::save/load` exist, but `candle_nn::AdamW` keeps its moments private.

   That last finding is why FLAME has its own AdamW.
2. **Config → tokenizer → data.** These are the foundations with no ML risk, and they can be
   tested in isolation (tokenizer round-trip tests).
3. **`nn.rs` → `model.rs`.** The components in the order your notes listed them.
4. **Optimiser, checkpoint, control, trainer.** Pause/resume was designed in from the start,
   not added later. It shapes the data layer (deterministic batches) and the optimiser
   (saveable state).
5. **Integration tests** for causality, overfitting and pause/resume, before the CLI.
6. **CLI**, then end-to-end runs through the CLI.
7. **Reference check against PyTorch.**
8. **A real dataset** (Tiny Shakespeare) with a real pause and resume.
9. **Docs**, including the time estimates computed from the actual configs.

## Problems found and how they were fixed

| # | problem | how it showed up | fix |
|---|---|---|---|
| 1 | `-(2 * i)` on a `usize` in RoPE | compile error | cast to `f64` before negating |
| 2 | **Candle cannot seed the CPU RNG** (`cannot seed the CPU rng with set_seed`) | all four integration tests failed at model construction | wrote `init.rs`: a `VarBuilder` backend that draws weights from a seeded host RNG, which also makes CPU and GPU initialisation identical |
| 3 | log line printed `step 276` when logging "every 25 steps" | reading the first training output | log condition moved to the post-increment step |
| 4 | `--separator "\n\n"` arrived as a literal backslash-n | Shakespeare became 1 document and the val split was empty | CLI un-escapes `\n` and `\t`; `prepare` errors clearly if the val split is empty |
| 5 | pausing via `--steps` was reported as "time budget reached" | the session log | added `StopReason::StepLimit` |
| 6 | clippy warnings (`is_multiple_of`, a boolean precedence in the pre-tokeniser, doc indentation) | `cargo clippy` | fixed; the pre-tokeniser change also made the intent explicit |

None of these would have been caught by "it compiles". #2 and #4 were caught by tests or
by checking the output of a real run.

## Verification results

| check | result |
|---|---|
| unit tests (tokenizer ×3, LR schedule) | pass |
| `model_is_causal` | max past-logit change < 1e-5 when the future changes |
| `overfits_tiny_dataset` | final loss < 0.15 (the CLI overfit run reaches 0.053) |
| `pause_and_resume_is_bit_identical` | identical weights (difference exactly 0.0) |
| `ctrl_c_pauses_and_saves` | pauses at step 0 with a checkpoint; PAUSE file consumed |
| PyTorch reference check | max |Δlogits| = **4.8e-7** |
| CLI overfit run (twice) | val_loss 0.0527 both times: deterministic |
| Tiny Shakespeare on a 4-core CPU | loss 4.65 → 2.75 in 77 steps; paused (`--steps 60`), resumed (`--max-minutes 1.5`), `status` correct |
| clippy | 0 warnings |

**Not yet verified:** the CUDA build (the development machine had no GPU). The code uses
only `Device::cuda_if_available` and no CUDA-specific code paths, so the risk is limited to
the build toolchain. See `setup.md`.

## Measured throughput (CPU, for reference)

0.89 M-param model, batch 32 × 256, on 4 CPU cores: ~1 800 tokens/s, i.e. ~3.8 h for the full
Tiny Shakespeare config. The GPU should be 20–100× faster on this model. Record your real
number in the lab notebook.

## Later changes

- **Lintless rebuilt in Rust on top of FLAME** (it was first prototyped in Python on a
  pretrained model, which did not match the plan). It now fine-tunes a FLAME run with FLAME's
  trainer. Tested end to end on CPU: a tiny Rust base model (`configs/rust-smoke.toml`,
  1.5 min) → Lintless data (112 samples) → fine-tune paused at step 5 and resumed to 24 →
  eval (incl. the rustc-suggestion baseline) → `fix`.
- **Data mixing and staged pre-training**: `[data] sources` mixes prepared datasets by weight
  (`MixedDataset`, tested for proportions and determinism); `[train] init_from` starts a run
  from another run's final weights; `flame eval` reports per-source loss; the tokenizer
  trainer samples every `--input` equally. Tested with `configs/staged-smoke-stage1/2.toml`:
  stage 2 loaded stage 1 and started at val loss 5.23 (random init ≈ 7.6).
- **Corpus tools**: `tools/fetch_corpus.sh` (code, docs, errors, english) and
  `tools/fetch_english.py`. Tested here: `errors` (518 explanations) and `docs` (2 279
  Markdown files, 38 MB). `code` and `english` weren't run (download size, and Hugging Face is
  blocked in the development sandbox).
- **`rust-30m` context raised from 512 to 1024** (micro-batch 2 × accumulation 64, same
  tokens per step) so Lintless samples fit. See the context-length decision in the developer guide.
- **TF32 on GPUs** (default, `FLAME_TF32=0` to disable): fp32 matmuls use the tensor cores via
  `candle_core::cuda::set_gemm_reduced_precision_f32`. Applies to flame-research and Lintless
  too (shared `select_device`). Not measured yet (no GPU in the development sandbox).
- **Cloud training**: `tools/cloud_setup.sh` (one-shot setup of a rented GPU machine) and
  `docs/cloud-gpu-guide.md`.
