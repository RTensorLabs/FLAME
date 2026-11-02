# Pausing and resuming training

Training on one laptop means sessions of 30 minutes to 4 hours, shutdowns in between,
and sometimes crashes. Everything in FLAME, flame-research and Lintless is built so you can
stop at any moment and continue later with no loss of quality.

This page describes the native trainer. The optional DGX BF16/fused trainer uses
the same model/AdamW/state checkpoint layout and native Rust batching, with
different signal handling and numerical guarantees. See the DGX-specific pause,
service and restore commands in [dgx-training.md](dgx-training.md).

## The rule

> **To resume, run exactly the same command again.**

The trainer finds the newest checkpoint in the run directory and continues from there with the
same weights, optimiser state, learning-rate schedule position and data order.

## Ways to stop cleanly (a checkpoint is saved)

| method | how | when to use |
|---|---|---|
| Ctrl+C | press once in the training terminal | you're at the laptop |
| time budget | `--max-minutes 45` | "I have 45 minutes tonight", the most useful one |
| step budget | `--steps 500` (FLAME / research) | reproducible short sessions |
| PAUSE file | `touch runs/<name>/PAUSE` (Windows: create an empty file named `PAUSE`) | from another terminal / SSH / file explorer |
| shutdown / logout | the OS sends SIGTERM/SIGHUP, which is treated like Ctrl+C | best effort only; see below |

Pressing **Ctrl+C twice** quits immediately **without** saving. You then lose only the
work since the last automatic checkpoint.

## Automatic checkpoints

`checkpoint_every_minutes` (default 10) saves even if nobody asks. If the laptop crashes,
sleeps or loses power, you lose at most ~10 minutes.

FLAME checkpoints are **atomic**: they are written to `latest.tmp/`, then swapped in. A power
cut in the middle of a save leaves the previous checkpoint intact (`latest` or `latest.old`).

## What is saved

FLAME / flame-research (`runs/<name>/`):

```text
config.toml               frozen config ([model]/[data] may not change; [train] may)
checkpoints/latest/       model.safetensors, optim.safetensors (AdamW m & v), state.json
best/model.safetensors    lowest validation loss so far
metrics.jsonl             every log/eval line (may contain a few repeated steps after a crash)
sessions.log              when each session started and why it stopped
```

Lintless fine-tunes use FLAME's trainer, so `runs/lintless-*/` has exactly the same layout
(plus `tokenizer.json` and `finetune.toml`). Everything on this page applies to them too.

## Native resume guarantees

- The data order is a pure function of `(seed, step)`, so no data-loader state needs saving.
- AdamW's moment buffers are saved. FLAME implements AdamW itself precisely because Candle's
  built-in optimiser doesn't expose them.
- The LR schedule is a function of the step.
- Weight initialisation is seeded on the host, independent of the device.

The CPU test `pause_and_resume_is_bit_identical` trains 30 steps straight and
12 + 11 + 7 steps over three sessions, and checks the final weights are
**bit-for-bit equal**. This is not a bitwise guarantee for arbitrary CUDA kernels.

For the optional DGX backend, saved weights and moments reload in fp32 and the
Rust bridge supplies the same `(seed, step, micro)` batches. Fused CUDA backward
kernels can produce small numerical differences in subsequent updates. The
two-update versus one + reload + one gate agreed within 1.13e-6 for model weights
and 4.58e-6 for optimizer moments. Switching between fp32 and BF16 also changes
arithmetic; it is not the same numerical trajectory.

## Recovery after accidental deletion

An atomic rotating `latest` checkpoint protects against interrupted writes; it
does not replace independent backups. The managed DGX run saves locally every
five minutes and creates versioned recovery archives about every 30 minutes,
including optimizer state, tokenizer, config, source, metrics and checksums.
Archives upload to [GitHub Releases](https://github.com/RTensorLabs/FLAME/releases/tag/rust-30m-dgx-20261102-checkpoints)
and are copied to the PC every 15 minutes while it is available. Old archives
are retained and deletion is not mirrored. The complete-model release is
published after both stages and evaluation finish. Full recovery commands and
the separate prepared-data backup are in [dgx-training.md](dgx-training.md).

## A typical week

```bash
# Monday evening (45 min)
flame train --config configs/rust-30m-stage1.toml --run runs/rust-30m-stage1 --max-minutes 45
# Saturday (3.5 h)
flame train --config configs/rust-30m-stage1.toml --run runs/rust-30m-stage1 --max-minutes 210
# any time
flame status --run runs/rust-30m-stage1
```

## Extending or restarting a run

- **Train longer**: raise `max_steps` in the config and rerun. The cosine schedule is
  recomputed for the new length, so the LR jumps up a little; that's fine for extending.
- **Start over**: `--fresh` renames the old run directory to `<run>.bak-<time>`, never deleting it.
- **Different model**: use a new `--run` directory.
