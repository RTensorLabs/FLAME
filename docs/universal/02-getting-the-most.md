# Getting the most out of the project

How to get the **best models**, the **most learning**, and the **least wasted time** out of
FLAME, flame-research and Lintless with one laptop and ~6 hours of GPU time per week. Read
[01-complete-guide.md](01-complete-guide.md) first.

---

## 1. The five rules

1. **The GPU should always be busy while you sleep or work, never while you wait.** Plan
   sessions so you start a run and walk away. Do reading, data collection and analysis while
   the GPU trains.
2. **Never train long on something you haven't tested short.** Every new config gets a
   5-minute run first (`--steps 50`). Bugs found in minute 5 cost 5 minutes; found in week 3, they cost weeks.
3. **Write the prediction before the result.** One line in the lab notebook: "I expect X
   because Y." That turns runs into learning.
4. **Change one thing at a time.** Two changes at once and you can't tell which one mattered.
5. **Compare against a baseline.** "Loss 1.9" means nothing; "loss 1.9 vs 2.1 for the dense
   model" means something.

---

## 2. Planning your weekly GPU time

| slot | best use |
|---|---|
| weekday evening (45 min) | `flame train … --max-minutes 45` on the long run (base model stage 1/2) |
| weekend block (3–4 h) | long run **or** a complete short experiment (research, Lintless fine-tune + eval) |
| overnight (if the laptop can stay on safely) | the long run with `--max-minutes 480`; the biggest free win |
| while the GPU is busy | CPU work: `lintless collect-*`, downloads, `flame prepare`, reading code/docs, writing notes |
| while the GPU is free but you're busy | nothing. Better: always leave the long run going when possible |

**Order of priority:** the base model is on the critical path (Lintless waits for it), so it
gets most GPU time. Research runs are short and fit in the gaps (a weekend).

---

## 3. Getting the best base model

### Data matters more than anything else

- **More, cleaner data beats more steps on the same data.** If the data plan shows code at > 4 epochs,
  download more code (add repos to `tools/fetch_corpus.sh`) rather than training longer.
- **Look at your data.** Open random files in `data/raw/rust-code/` and `data/raw/english/`.
  Generated files, huge data tables and minified code are noise. Delete such folders before `flame prepare`.
- **Keep the mix sensible.** Code ≥ 50% (it's the main skill), English 25–35% (for explanations),
  docs 5–20% (bridges both), errors tiny (it's small).
- **Watch the ⚠** in the data plan. Anything above ~16 epochs is being memorised, not learned.

### Settings

- The provided configs are sensible defaults. The most sensitive knob is `lr`: if the loss
  is noisy or spikes, lower it by 30–50%; if it falls very slowly from the start, you *may*
  raise it by 30%.
- **Don't change the model shape halfway.** It needs a new run.
- **Extending training:** if stage 1 finished and validation loss is still falling steadily,
  raise `max_steps` in the config and continue. The learning rate briefly rises again, which is fine.

### Checking quality during training

- Every few sessions: `flame generate` with 3 fixed prompts (one Rust, one English, one Rust
  error), temperature 0.5. Save the outputs in the lab notebook, so you can *see* progress.
- `flame eval --run R` shows the loss **per dataset**. Stage 2 should improve `rust-docs`
  without hurting `english` much.

---

## 4. Getting the best Lintless

In order of impact:
1. **More varied examples.** 5 000 examples from 50 projects beat 10 000 from 3 projects. Add
   seeds (single-file programs) and run `collect-project` on many different crates.
2. **Balance.** If one error code is > 30% of the data, reduce it (`--per-operator 3`).
3. **Look at failures.** `runs/lintless-30m/eval/eval.jsonl` shows every reply. Group the
   failures by error code; add data for the worst codes.
4. **Epochs.** Watch `val_loss` during fine-tuning; if it rises after epoch 2, use `epochs = 2`.
5. **Bigger base model** only after 1–4 (see the scale-up table in the FLAME end-to-end guide).

Beat the baseline first on the error types where rustc gives **no** suggestion. That's where
Lintless adds real value.

---

## 5. Getting the most from the research

- Follow the protocol: `flame-research/docs/hypothesis-and-protocol.md` lists the success
  criteria. Decide *before* looking at results.
- Run all 5 synthetic-task configs in one weekend. They're short and give the first answer
  to "does routing help?".
- Always run `flame-research compare` after a batch of runs, and paste the table into the experiment log.
- A clear **negative** result ("the modular model is worse because the router collapses") is
  valuable. Write down *why*.
- Good first ablation: exploration (`explore = 0 / 0.1 / 0.2`), because the first smoke test
  showed STOP collapse without it.
- Idea worth testing (it's cheap at 5 M): **staged vs single-mix pre-training** on TinyStories
  + code. It checks whether our two-stage design is worth it.

---

## 6. Getting the most learning

The project is also your way to learn how models work. A good order:

1. **Read FLAME in this order:** `nn.rs` (the layers) → `model.rs` → `trainer.rs` →
   `data.rs` → `optim.rs`. Each is 100–300 lines with comments explaining *why*.
2. **Change something small and watch the effect** on Tiny Shakespeare (15 minutes each):
   - turn off RoPE (return `x` unchanged in `Rope::apply`): loss gets worse; why?
   - set `tie_embeddings = false`: more parameters; better or worse?
   - double `lr`, halve `lr`: what happens to the loss curve?
3. **Implement one missing feature yourself.** Great first exercises, in increasing difficulty:
   - print the top-5 next-token probabilities in `generate`;
   - a **KV cache** for generation (makes Lintless 10× faster);
   - gradient-norm logging per layer;
   - TF32 / bf16 matmuls (speed).
4. **Read the paper behind each component** when you touch it: the table in
   [../concepts.md](../concepts.md) lists them.

---

## 7. Protecting your work

- **Back up finished models**: copy `runs/NAME/best/` and `runs/NAME/config.toml` to an external
  drive or cloud folder after each stage. Weeks of GPU time live in a few hundred MB.
- **Never delete a run folder**: `--fresh` renames it to a backup anyway. Delete backups only
  when you're sure.
- **Disk space**: raw data is 5–8 GB, prepared data 2–4 GB, and each run up to ~1 GB. Check with `du -sh ~/code/*/data ~/code/*/runs`.
- **Code changes**: commit them to git with a clear message; push to GitHub.

---

## 8. Laptop care

- Plugged in, on a hard surface or cooling stand, lid open.
- Disable sleep while plugged in (Windows power settings). If it sleeps anyway, the
  10-minute auto-checkpoint limits the damage.
- Watch temperatures: `nvidia-smi -l 5` (in another terminal). Constant 85 °C+ means it is
  throttling. Cooling helps more than any setting.
- Close heavy programs (games, many browser tabs) while training.

---

## 9. Checklists

**Before a long run:**
- [ ] `cargo test --release` passes
- [ ] 5-minute test of this exact config (`--steps 50`) worked
- [ ] data plan printed: no ⚠
- [ ] `flame estimate` time fits your plan
- [ ] prediction written in the lab notebook

**Every session:**
- [ ] same `flame train …` command, with `--max-minutes`
- [ ] glance at loss / val_loss / gnorm
- [ ] `flame status` once a week; save 3 fixed-prompt samples

**After a stage / fine-tune:**
- [ ] `flame eval` (per-source) / `lintless eval`
- [ ] results + samples in the lab notebook
- [ ] back up `best/`
- [ ] decide the next step using the tables in the end-to-end guides

---

## 10. What to do after the first version works

In rough order of value:
1. **FLAME speed work** (TF32/bf16, fused attention): 2–4× faster training; makes 100 M practical.
2. **KV cache**: 10×+ faster generation (Lintless evaluation and `fix`).
3. **More Lintless error types** (lifetimes, borrow conflicts): the hard, valuable cases.
4. **Research round 2**, based on round-1 results.
5. **100 M base model**, if Lintless at 30 M plateaus.
6. **Longer context** (memory-efficient attention) so Lintless can see whole files.
