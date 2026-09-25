# Lab notebook

One entry per experiment. Copy the template. Write the **prediction before running**; that
is what turns a run into an experiment.

## Template

```markdown
### YYYY-MM-DD — <short name>
- **Question:** what do I want to know?
- **Prediction:** what I expect and why (numbers if possible)
- **Change vs baseline:** exactly one variable
- **Command / config / commit:** …
- **Result:** val loss, accuracy, avg steps, tokens/s, GPU hours
- **Surprise?** …
- **Conclusion / next step:** …
```

## Entries

### Phase 1 — overfit gate (CPU)
- **Question:** can the Transformer + trainer memorise a 267-byte paragraph?
- **Prediction:** loss → < 0.1 within 400 steps.
- **Result:** val loss 0.053 at step 400; samples reproduce the paragraph. 30 s on CPU.
- **Conclusion:** model + optimiser + data path are sound. PyTorch reference agrees to 5e-7.

### Phase 1 — pause/resume
- **Result:** 30 steps straight vs 12 + 11 + 7 across three sessions → bit-identical weights.
