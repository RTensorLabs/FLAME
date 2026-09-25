# Review of the research plan and recommendations

A review of [research-notes.md](research-notes.md), written before any code existed.
The recommendations have been turned into code in `flame-research` (marked ✅).

## Overall

The reasoning and research discipline are strong: recognising that partial loading needs a
modular *architecture*, insisting on baselines, and the build order (dense baseline first,
residency last) are all right. The main risks are prior art, a missing baseline, and routing
granularity.

## 1. The core hypothesis has close prior art (read it first)

- **Shared workspace between modules**: Goyal et al., *Recurrent Independent Mechanisms*
  (2019); *Coordination Among Neural Modules Through a Shared Global Workspace* (2021). The
  latter is very close to the IR idea.
- **Latent arrays ("32 × 128")**: Perceiver / Perceiver IO; **Perceiver AR** for the causal case.
- **STOP / adaptive depth**: Graves, *Adaptive Computation Time*; Universal Transformers;
  PonderNet; Mixture-of-Depths; Mixture-of-Recursions.
- **Routing to reused modules**: Routing Networks (Rosenbaum et al.), Neural Module Networks,
  MoE (Switch, Mixtral, DeepSeekMoE).
- **Residency side**: PowerInfer (hot/cold neurons), *LLM in a Flash*, MoE offloading
  (Eliseev & Mazur; Pre-gated MoE), Colibri.

The specific angle, *architecture designed to be residency-friendly*, is less explored than
either half alone. See `flame-research/docs/related-work.md`.

## 2. Add two baselines ✅

| baseline | answers |
|---|---|
| dense, same **total** params | does modular lose quality? |
| dense, same **active** params/FLOPs | does routing beat simply using a smaller model? (the real bar) |
| **standard MoE**, same total + active | do the workspace and recurrent routing add anything beyond MoE? |

Pre-register what "comparable" means (e.g. validation loss within 2% of dense-total, and
clearly better than dense-active). See `flame-research/docs/hypothesis-and-protocol.md`.

## 3. Routing granularity decides whether residency is possible ✅

Per-token routing uses nearly every module within a few hundred tokens, so everything must
stay resident. In practice MoE experts specialise by token type or syntax, not by domain.
Measure **routing locality** (distinct modules per 16/64/256-token window) from the start,
and try **chunk-level routing** decided from the previous chunk. That design is causal and
known one chunk ahead, so a runtime can prefetch.

**Cheap trick ✅**: at 5 M params, measuring a real runtime is meaningless because everything
fits in cache. Record routing traces and replay them through a simulated memory hierarchy
(LRU cache, SSD/PCIe bandwidth, `--scale` to pretend modules are bigger).

## 4. Technical traps

- **Causality leaks through the workspace ✅**: a global latent workspace lets tokens see the
  future. The model keeps a per-position workspace, and tests assert that future tokens cannot
  change past logits.
- **Discrete routing is hard to train ✅**: it needs load-balancing and z-losses, gradients
  through the router (gate scaling), a compute penalty for STOP, and **exploration**, because
  STOP otherwise collapses. The first smoke test showed exactly this collapse.
- **FLOPs ≠ latency**: dynamic paths batch poorly on GPUs, so report both.
- **Tiny scale can give false negatives ✅**: add synthetic tasks with controllable
  difficulty (addition, multi-hop lookup, brackets) where the right amount of computation is known.

## 5. Engineering

- Rust + Candle is fine, but **verify against a reference ✅** (`tools/reference_check.py`
  matches PyTorch to 5e-7) and keep the "overfit a tiny dataset" gate ✅.
- **Don't split into 8 crates up front ✅**: FLAME has one library crate + one CLI crate;
  extract more when boundaries hurt.
- **Lintless and the research are separate projects ✅**: Lintless fine-tunes an existing
  small code model (hours), rather than waiting for a from-scratch FLAME base model (weeks).
