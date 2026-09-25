# AI Architecture Research Notes

> The original research notes that started FLAME, lightly formatted. The review of these notes
> is in [review-and-recommendations.md](review-and-recommendations.md); the concrete plan is in
> [roadmap.md](roadmap.md).

## Long-term goal

Build AI systems that:

- run efficiently on edge devices such as phones, laptops, robots and embedded systems;
- achieve substantially stronger reasoning and knowledge capabilities at much smaller parameter counts;
- use significantly less memory, power and computation;
- treat **model architecture, memory and runtime as one systems problem** rather than optimising each independently;
- explore alternatives to the assumption that more parameters and dense computation is the main path to better intelligence.

The long-term objective is **not simply to build a smaller Transformer**. The goal is to
understand whether intelligence can be represented and executed in a fundamentally more
efficient way.

## Why this started

The laptop has limited VRAM, so running large models entirely in GPU memory is hard or
impossible. Current systems rely on a hierarchy `SSD → RAM → VRAM`. That lets models larger
than VRAM run, but moving weights between levels becomes a major bottleneck.

> Instead of only optimising computation, can we optimise **how models are represented,
> loaded and executed**?

## The original hypothesis

If a model only needs certain knowledge for a problem, perhaps only the relevant parts need to
be loaded. For "Explain Rust ownership": Rust + programming + ownership + reasoning.

```text
Question → determine required computation → load/activate relevant components → reason → answer
```

## Problem: Transformer knowledge is distributed

Dense Transformers have no clean "Rust module" or "math module". Knowledge is spread across
layers and parameters. **Partial loading cannot simply be bolted onto a dense Transformer.**
If the architecture does not produce localised or composable computation, a runtime cannot
know which weights are safe to leave out.

## Three different research problems

1. **Systems**: VRAM management, the SSD/RAM/VRAM hierarchy, weight streaming, CPU/GPU scheduling,
   memory mapping, residency, latency (llama.cpp, vLLM, Colibri). *How are existing models executed?*
2. **Architecture**: Transformer, MoE, Mamba, RWKV, liquid networks, recurrent and modular
   architectures, routing, adaptive computation, latent memory. *How should computation be structured?*
3. **Learning / representation**: why knowledge is distributed; whether it can become localised;
   whether modules specialise naturally; semantic vs token-level representations; reasoning through
   compressed intermediate states; small models reaching large-model capability. *What should the model learn and store?*

## Colibri

Colibri treats VRAM/RAM/disk as one memory hierarchy and dynamically loads the needed parts
of large MoE models. So **partial loading / weight streaming alone is not the novel research
idea**. The question has to move one level deeper:

> **Can we design a model whose computation is inherently modular, dynamically selected, and
> therefore naturally compatible with partial residency?**

That shifts the work from *runtime optimisation* to *architecture + runtime co-design*.

## Current research hypothesis

```text
Input → Encoder → Router → Selected modules → Shared latent workspace (IR) → Router → More modules
      → Shared workspace → … → Decoder → Output
```

The model builds a computation path per input (for example `Module 2 → 7 → 2 → 5 → STOP`)
instead of a fixed layer stack. The core idea is **conditional computation**.

- **Shared latent workspace**: modules communicate through a learned intermediate
  representation, not natural language. Initially something simple, e.g. 32 latent tokens × 128 dims.
- **Modules are not assigned domains**: they are generic learned functions. Afterwards we
  inspect what specialisation emerges, if any; "all modules learn the same function" is also a
  useful result.
- **Router**: `Router(IR)` → a distribution over modules + STOP; top-k chosen; repeated. This is a
  dynamic computation graph.
- **STOP / adaptive computation**: easy inputs take fewer steps, hard ones more, so extra
  computation is spent only when the problem requires it.

## First experimental architecture

2–5 M parameters, 4–8 modules: tokenizer → embedding → router → modules → shared IR → router
→ … → STOP → tiny decoder. The objective is **architectural validation, not language quality**.

## Critical baseline (mandatory)

| metric | dense Transformer | modular model |
|---|---|---|
| parameters | same | same |
| train / validation loss | measure | measure |
| task accuracy | measure | measure |
| active params per token | ~100% | measure |
| FLOPs per token, memory, latency | measure | measure |
| modules activated | n/a | measure |
| computation path | fixed | dynamic |

Otherwise a model that simply computes less may look "efficient" while actually being worse.

## Research progression

```text
understand existing architectures → implement a Transformer → train a tiny dense baseline
→ understand its bottlenecks → implement the modular architecture → train it
→ compare at the same parameter budget → study specialisation → study routing
→ study adaptive computation → only then partial residency → weight streaming / memory hierarchy
```

Partial loading comes **last** because it only makes sense once the architecture lets us
identify computation that is unnecessary for an input:
modular computation → learned routing → localised computation → predictable module usage →
partial residency → weight streaming. The runtime optimisation should become a
**consequence of the architecture**.

## FLAME direction

**FLAME = Framework for Lightweight AI Model Engineering**, in Rust + Candle: architectures,
training, datasets, tokenisation, evaluation, checkpointing, inference, experiment tracking and
benchmarking. The first architecture is a conventional decoder-only Transformer; the modular
architecture comes later as an experimental architecture, so that we never debug a new
framework and a new architecture at the same time.

## Learning status (at the start)

Read directly: *Attention Is All You Need*, *LLaMA*. Studied through summaries: RoPE,
RMSNorm, SwiGLU, AdamW. From here the loop is:
**encounter unknown concept → read the source → implement → test → measure → document.**

## Plan (as originally written)

1. **Phase 1, tiny Transformer**: tokenizer, embedding, RMSNorm, RoPE, causal self-attention,
   SwiGLU, residuals, LM head, cross-entropy.
2. **Phase 2, training**: dataset, batching, training loop, AdamW, LR scheduler, validation,
   checkpointing, generation. First goal: **overfit a tiny dataset**.
3. **Phase 3, infrastructure**: extract reusable components (core, nn, models, data, train, runtime, eval, monitor).
4. **Phase 4, real small model**: 10 M → 30 M → ~100 M; a Rust-oriented base model; no specialisation yet.
5. **Phase 5, Lintless**: base model → Rust/compiler diagnostic data → instruction tuning.

## Open research questions

- **Architecture**: can computation become localised enough for modular execution? Can
  modules communicate through a shared workspace? Does specialisation emerge unassigned?
  Can a router learn useful paths? Can a module be reused at different points? Can the model
  decide how much computation it needs?
- **Efficiency**: can active params/token be far below total params? Can dynamic
  computation reduce FLOPs without hurting quality? Does modularity reduce memory residency?
  Can unused modules stay outside VRAM?
- **Representation**: what should the workspace contain? Can latent state preserve
  information across modules? Is semantic compression useful for long context? Could
  semantic memory replace part of the KV cache?
- **Systems**: can architecture and runtime be co-designed? Can the router predict future
  modules early enough to prefetch them? Can modules stream SSD → RAM → VRAM? Can
  computation overlap with loading? What is the real bottleneck: compute, bandwidth, storage
  latency or synchronisation?

## Research rules

For every new idea:

1. **What exact bottleneck does it solve?** Not "more efficient", but "reduces VRAM residency from X to Y".
2. **Has someone already tried it?** Search papers, GitHub, architectures, inference engines, systems research.
3. **What trade-off does it introduce?** Less VRAM vs more latency; less compute vs lower
   quality; more modularity vs communication overhead; compression vs information loss.
4. **Can it be tested at 1–10 M parameters?**
5. **What is the baseline?** Never evaluate an architecture in isolation.

## Current position

The goal is narrow: **build a correct tiny Transformer baseline, then experimentally determine
whether dynamically routed modular computation can reach comparable quality with
substantially less active computation.** If that works, investigate partial residency,
weight streaming and memory-aware scheduling. If it fails, find out why. Candidate failure
modes: insufficient inter-module bandwidth, inadequate workspace, poor routing, no
specialisation, optimisation instability, routing overhead, insufficient capacity,
information loss between modules.

**Core hypothesis:** *Can a small language model dynamically construct a computation path
through independently learned modules, using a shared latent workspace, while retaining
comparable quality to a dense model at the same parameter budget?*

**Long-term question:** *Can architecture and runtime be co-designed so that an AI system
stores and executes only the computation necessary for the current problem?*

## Final principle

```text
identify an assumption → build the smallest possible experiment → establish a baseline
→ measure → find the failure mode → change one variable → measure again → next hypothesis
```

The goal is not to prove the original idea correct. The goal is to **find out what is actually true**.
