# FLAME architecture and data flow

How FLAME is built: the parts of the code, how the model is put together, and how data
flows from a text file to a trained model and back out as generated text. The diagrams are
drawn by GitHub automatically (Mermaid). For the *why* behind each decision, see
[developer-guide.md](developer-guide.md).

---

## 1. The big picture

```mermaid
flowchart LR
    subgraph IN["Input"]
        RAW["text files<br/>.txt .md .rs"]
    end
    subgraph PREP["Preparation (once)"]
        TOK["tokenizer.rs<br/>learn 8,192 pieces"]
        PREPARE["flame prepare<br/>text → token ids"]
    end
    subgraph TRAIN["Training (weeks, pausable)"]
        DATA["data.rs<br/>random windows · mixing"]
        MODEL["model.rs + nn.rs<br/>Transformer"]
        OPT["optim.rs<br/>AdamW + LR schedule"]
        LOOP["trainer.rs<br/>the loop"]
        CKPT["checkpoint.rs + control.rs<br/>save · pause · resume"]
    end
    subgraph OUT["Output"]
        RUN["runs/NAME/<br/>weights + logs"]
        GEN["generate.rs<br/>write text"]
    end
    RAW --> TOK --> PREPARE
    RAW --> PREPARE --> DATA --> LOOP
    MODEL <--> LOOP
    OPT <--> LOOP
    LOOP <--> CKPT --> RUN --> GEN
```

---

## 2. Code structure

FLAME is a Cargo *workspace* with two crates: a library (all the logic) and a command-line
program that calls it. flame-research and Lintless use the library directly.

```mermaid
flowchart TD
    CLI["flame-cli<br/>(the 'flame' command)"]
    RESEARCH["flame-research<br/>(other repo)"]
    LINTLESS["lintless<br/>(other repo)"]
    subgraph LIB["crates/flame (library)"]
        CONFIG["config.rs<br/>TOML settings"]
        TOKENIZER["tokenizer.rs<br/>byte + BPE"]
        DATARS["data.rs<br/>token files · batches · mixing"]
        NN["nn.rs<br/>RMSNorm · RoPE · attention · SwiGLU · Block"]
        MODELRS["model.rs<br/>LanguageModel · Transformer · loss"]
        INIT["init.rs<br/>seeded weights"]
        OPTIM["optim.rs<br/>AdamW · schedule · clipping"]
        TRAINER["trainer.rs<br/>training loop · eval · ETA"]
        CONTROL["control.rs<br/>Ctrl+C · time budget · PAUSE"]
        CHECKPOINT["checkpoint.rs<br/>atomic saves"]
        GENERATE["generate.rs<br/>sampling"]
        ESTIMATE["estimate.rs<br/>time estimates"]
    end
    CLI --> LIB
    RESEARCH --> LIB
    LINTLESS --> LIB
    MODELRS --> NN
    MODELRS --> INIT
    TRAINER --> MODELRS
    TRAINER --> DATARS
    TRAINER --> OPTIM
    TRAINER --> CONTROL
    TRAINER --> CHECKPOINT
    GENERATE --> MODELRS
    DATARS --> TOKENIZER
```

| file | job | size |
|---|---|---|
| `config.rs` | reads the `.toml` config: `[model]`, `[train]`, `[data]` | ~230 lines |
| `tokenizer.rs` | splits text into tokens; trains the BPE dictionary | ~430 |
| `data.rs` | memory-maps token files; builds batches; mixes several datasets by weight | ~330 |
| `nn.rs` | the building blocks of the Transformer | ~190 |
| `model.rs` | the full model; the `LanguageModel` interface; the loss | ~150 |
| `init.rs` | reproducible random starting weights | ~70 |
| `optim.rs` | AdamW optimizer (with saveable state), learning-rate schedule | ~130 |
| `trainer.rs` | the training loop: steps, logging, evaluation, checkpoints, pausing | ~290 |
| `control.rs` | detects Ctrl+C, the time budget and the PAUSE file | ~110 |
| `checkpoint.rs` | crash-safe saving and loading | ~110 |
| `generate.rs` | writes text with temperature / top-k | ~80 |
| `flame-cli/src/main.rs` | the `flame` command and all its sub-commands | ~560 |

---

## 3. The model (a LLaMA-style Transformer)

```mermaid
flowchart TD
    IDS["token ids<br/>(batch × length)"] --> EMB["Embedding<br/>id → vector of 512 numbers"]
    EMB --> B1
    subgraph BLOCK["Block (× 8 layers for the 30 M model)"]
        B1["RMSNorm"] --> ATT["Causal self-attention<br/>8 heads · RoPE positions<br/>each token looks only at earlier tokens"]
        ATT --> ADD1(("+"))
        ADD1 --> B2["RMSNorm"] --> MLP["SwiGLU feed-forward<br/>512 → 1408 → 512"]
        MLP --> ADD2(("+"))
    end
    EMB -. "residual (skip)" .-> ADD1
    ADD1 -. "residual (skip)" .-> ADD2
    ADD2 --> NORM["final RMSNorm"]
    NORM --> HEAD["LM head<br/>(shares the embedding matrix)"]
    HEAD --> LOGITS["scores for each of the 8,192 tokens<br/>at every position"]
```

What each part does, in plain words:
- **Embedding**: turns each token id into a list of numbers the model can compute with.
- **Causal self-attention**: each position gathers information from *earlier* positions (never
  later ones, which would be cheating). **RoPE** encodes *how far apart* two tokens are.
- **SwiGLU**: a small network applied to each position separately; it stores much of the "knowledge".
- **RMSNorm**: keeps the numbers at a stable size so training doesn't blow up.
- **Residual (skip) connections**: each block *adds* its result to its input, which makes deep models trainable.
- **LM head**: turns the final vectors into a score for every possible next token.

Sizes of the provided models:

| config | d_model | layers | heads | context | parameters |
|---|---|---|---|---|---|
| overfit | 64 | 2 | 4 | 64 | 0.12 M |
| tiny-shakespeare | 128 | 4 | 4 | 256 | 0.9 M |
| tinystories-5m | 256 | 6 | 8 | 256 | 5.9 M |
| rust-30m-stage1/2 | 512 | 8 | 8 | 1024 | ≈ 30 M |
| rust-100m-stage1/2 | 768 | 12 | 12 | 1024 | ≈ 91 M |

---

## 4. Data flow: text → training batches

```mermaid
flowchart LR
    subgraph RAWD["data/raw/ (text)"]
        R1["rust-code/*.rs"]
        R2["rust-docs/*.md"]
        R3["rustc-errors/*.txt"]
        R4["english/*.txt"]
    end
    R1 & R2 & R3 & R4 -->|"flame tokenizer train<br/>(equal sample of each)"| TK["data/tokenizers/<br/>rust-en-8k.json"]
    R1 -->|prepare| P1["data/rust-code/<br/>train.bin · val.bin"]
    R2 -->|prepare| P2["data/rust-docs/"]
    R3 -->|prepare| P3["data/rustc-errors/"]
    R4 -->|prepare| P4["data/english/"]
    TK -.-> P1 & P2 & P3 & P4
    P1 & P2 & P3 & P4 --> MIX["MixedDataset<br/>pick a source by weight,<br/>then a random 1024-token window"]
    MIX --> BATCH["Batch<br/>input = tokens 0..1023<br/>target = tokens 1..1024"]
```

Key facts:
- **Token files** are plain arrays of 16-bit numbers (`u16`), read by memory-mapping, so a
  2 GB file costs almost no RAM.
- **`flame prepare`** splits text into documents (by file, or by a separator such as
  `<|endoftext|>`), tokenises them in parallel, and sends every 50th document to `val.bin`.
- **Batches are deterministic:** which windows are read at step *s* depends only on the seed
  and *s*. That's why resuming after a pause gives exactly the same training.
- **Target = input shifted by one:** at every position, the model learns to predict the next token.

---

## 5. One training step

```mermaid
sequenceDiagram
    participant T as Trainer
    participant C as StopControl
    participant D as Data
    participant M as Model
    participant O as AdamW
    participant K as Checkpoint
    T->>C: should I stop? (Ctrl+C / time / PAUSE)
    alt stop requested
        T->>K: save everything, then exit ("paused")
    else continue
        loop grad_accum_steps times (e.g. 64)
            T->>D: batch for (step, micro)
            D-->>T: input + target tokens
            T->>M: forward: predict next tokens
            M-->>T: scores
            T->>T: loss = cross-entropy (+ extra loss if any)
            T->>T: backward: compute gradients, add them up
        end
        T->>T: clip gradients if too large
        T->>O: update all weights (learning rate for this step)
        opt every eval_every steps
            T->>M: validation loss on held-out data
            T->>K: save best/ if improved
        end
        opt every 10 minutes
            T->>K: save a checkpoint
        end
    end
```

---

## 6. Saving, pausing and resuming

```mermaid
stateDiagram-v2
    [*] --> Starting: flame train
    Starting --> Training: no checkpoint (fresh, or init_from a previous stage)
    Starting --> Training: checkpoint found → load weights, optimizer, step
    Training --> Training: step · log · eval · auto-save every 10 min
    Training --> Saving: Ctrl+C / --max-minutes / PAUSE file / --steps
    Training --> Saving: max_steps reached
    Saving --> Paused: step < max_steps
    Saving --> Finished: step = max_steps
    Paused --> Starting: run the same command again
    Finished --> [*]
```

**Crash-safe saving:** a checkpoint is written to `latest.tmp/`, then quickly swapped in
(`latest` → `latest.old`, `latest.tmp` → `latest`, delete `latest.old`). If the power goes out
at any moment, a complete checkpoint still exists.

**What's in a run folder:**

```text
runs/rust-30m-stage1/
├── config.toml               the config, frozen at the start
├── checkpoints/latest/
│   ├── model.safetensors     all weights
│   ├── optim.safetensors     the optimizer's memory (AdamW m and v)
│   └── state.json            step, tokens seen, hours trained, best val loss
├── best/model.safetensors    weights with the best validation loss
├── metrics.jsonl             one line per log/eval
└── sessions.log              when each session started and why it stopped
```

---

## 7. Staged pre-training

```mermaid
flowchart LR
    RND["random weights<br/>(seeded)"] --> S1["Stage 1<br/>rust-30m-stage1<br/>code 59% · English 35% · docs 6%<br/>lr 6e-4 → 6e-5"]
    S1 -->|"init_from<br/>(final weights, fresh optimizer)"| S2["Stage 2<br/>rust-30m-stage2<br/>code 50% · docs 20% · English 29.5% · errors 0.5%<br/>lr 3e-4 → 0"]
    S2 --> BASE(["base model<br/>→ Lintless fine-tuning"])
```

Both stages use the **same model shape and the same tokenizer** (checked at start).
`init_from` is used only in the first session of stage 2; after that, stage 2 resumes from its own checkpoints.

---

## 8. Generation (writing text)

```mermaid
flowchart LR
    P["prompt text"] --> ENC["tokenizer.encode"] --> CTX["context<br/>(last ≤ 1024 tokens)"]
    CTX --> FWD["model forward"] --> LAST["scores for the next token"]
    LAST --> SAMPLE["temperature + top-k<br/>→ pick one token"]
    SAMPLE --> APPEND["append to context"]
    APPEND -->|"repeat until max tokens<br/>or end-of-text"| CTX
    APPEND --> DEC["tokenizer.decode → text"]
```

Currently the whole context is re-read for every new token (there is no KV cache yet). That's
simple but slow for long outputs; a KV cache is on the roadmap.

---

## 9. Tokenizer format

| ids | meaning |
|---|---|
| 0–255 | the 256 possible bytes, so *any* text can be encoded, even unseen characters |
| 256 | `<|endoftext|>`: the document separator |
| 257 | `<|pad|>`: padding (ignored by the loss) |
| 258 … 8191 | learned pieces (BPE merges), e.g. ` fn`, `impl`, ` the`, `ing` |

Saved as JSON (the list of merges), e.g. `data/tokenizers/rust-en-8k.json`.
