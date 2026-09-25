# Troubleshooting

| symptom | cause / fix |
|---|---|
| `running on CPU (build with --features cuda …)` | you built without `--features cuda`, or CUDA isn't visible (`nvidia-smi` in WSL must work) |
| CUDA build fails with compute-capability errors | `export CUDA_COMPUTE_CAP=86` and rebuild |
| `out of memory` | halve `micro_batch_size`, double `grad_accum_steps` (same effective batch) |
| loss is `NaN` / "loss became NaN" | LR too high: divide `lr` by 3; keep `grad_clip = 1.0` |
| loss stuck at ≈ ln(vocab) | the model isn't learning: check the data (`meta.json` token counts), try the overfit config |
| val loss ≪ train loss / suspiciously good | information leak (future tokens) or val data inside train data; run `cargo test` (causality tests) |
| `[model]/[data] differ from the run's frozen config` | you edited the model after starting; use a new `--run` or `--fresh` |
| `validation split is empty` | too few documents: use `--separator "\n\n"` or raise `--val-fraction` |
| training slower than the estimate | tiny models are launch-overhead bound; laptop throttling (`nvidia-smi -l 5`); running from `/mnt/c` in WSL |
| Ctrl+C doesn't stop immediately | by design it finishes the step and saves; press twice to force-quit (no save) |
| resumed run's metrics have repeated steps | the run crashed after logging but before the next checkpoint; harmless |
