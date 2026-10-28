# Training on a rented cloud GPU (vast.ai)

How to use a rented GPU (e.g. $100 of vast.ai credit) to train the base model in hours
instead of weeks, without wasting money. Every step, in order. Prices and speeds are
**estimates**: check the current vast.ai listing, and measure the real speed with the
10-minute benchmark in step 5.

## Summary

| | recommendation |
|---|---|
| **GPU** | **1× H100** (fastest), or **1× A100 80 GB** / **1× RTX 4090** (best value). One GPU only: FLAME doesn't use multiple GPUs. Avoid RTX 5090 for now (its new architecture is untested with Candle). |
| **image** | a CUDA **devel** image: `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| **disk** | 100 GB |
| **instance type** | on-demand (interruptible is cheaper, but you can lose the machine mid-run) |
| **cost** | 30 M model ≈ **$2–8**, 100 M model ≈ **$8–25**; everything incl. data prep and Lintless ≈ **$20–40** |

## 1. Which GPU, and what it costs

FLAME computes in fp32. With **TF32** (on by default since this version), matrix
multiplications run on the tensor cores, which is a big win on data-centre GPUs. The other
operations depend on memory bandwidth, where data-centre GPUs are also strongest.

| GPU (1×) | vast.ai price (approx.) | FLAME speed (est.) | 30 M model | 100 M model |
|---|---|---|---|---|
| RTX 3070 Ti Laptop (yours) | – | 1.5–6 TFLOP/s | 12–50 h | 60–230 h |
| RTX 4090 24 GB | $0.30–0.60/h | 15–30 | 2.5–5 h (≈ $1–3) | 12–23 h (≈ $5–14) |
| A100 80 GB | $0.80–1.40/h | 20–40 | 2–4 h (≈ $2–5) | 9–18 h (≈ $9–25) |
| **H100 80 GB** | $1.70–2.80/h | 30–60 | **1.3–2.5 h** (≈ $3–7) | **6–12 h** (≈ $12–33) |

Training cost = 2.7 × 10¹⁷ operations for 30 M and 1.3 × 10¹⁸ for 100 M (both stages), divided by
the speed. **Pick the H100 if you want results today; the 4090 if you want the most runs per dollar.**

Don't rent multi-GPU machines (2×, 8×): FLAME trains on one GPU, so you'd pay for idle GPUs.

## 2. Plan the money (example with $100)

| step | GPU time | cost on 1× H100 (~$2.2/h) |
|---|---|---|
| setup + tests + benchmark | 0.5 h | $1 |
| download + prepare data (on the rented machine: fast network, many CPUs) | 1.5–2.5 h | $3–6 |
| 30 M base model (stage 1 + 2) | 1.5–2.5 h | $3–6 |
| Lintless data collection, fine-tune + eval on the 30 M model | 1–1.5 h | $2–3 |
| **if 30 M isn't good enough:** 100 M base model (stage 1 + 2) | 6–12 h | $13–26 |
| Lintless on 100 M | 1 h | $2 |
| **total** | | **≈ $25–45**, leaving a reserve |

Money-savers:
- **Collect Lintless data on your laptop** (it's CPU-only) and upload the small `data/*.jsonl` files.
- **Stop or destroy the instance** the moment training ends. Billing continues while it's
  running, even if idle. A *stopped* instance still costs a little for its disk; *destroy* it once
  your files are copied home.
- Set a low "auto-destroy / max duration" in vast.ai if available, as a safety net.

## 3. Rent the machine (vast.ai website)

1. **Search** → filter: *GPU* = H100 (or A100 / RTX 4090), *#GPUs* = 1, *disk* ≥ 100 GB,
   *reliability* ≥ 98%, decent *internet down* speed (≥ 500 Mbps helps with the data download).
2. **Template / image:** choose "Edit image" and enter `nvidia/cuda:12.4.1-devel-ubuntu22.04`
   (the **devel** variant contains `nvcc`, which Candle needs to compile its GPU code).
   Launch mode: **SSH**.
3. **Disk:** 100 GB. **Rent** (on-demand).
4. When it's running, copy the **SSH command** from the instance card, e.g.
   `ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080`, and run it in your Ubuntu terminal.
   (First time: add your SSH public key in vast.ai *Account → SSH keys*; create one with
   `ssh-keygen -t ed25519` and paste `~/.ssh/id_ed25519.pub`.)

## 4. Set it up (~10 min)

On the rented machine:

```bash
tmux new -s flame        # a session that keeps running if your connection drops
curl -sSL https://raw.githubusercontent.com/RTensorLabs/FLAME/main/tools/cloud_setup.sh | bash
source ~/.bashrc && cd ~/code/FLAME
flame train --config configs/overfit.toml --run runs/overfit
```

The last command must print `running on CUDA GPU 0 (TF32 matmuls on…)` and end with
`best val_loss` ≈ 0.05.

**tmux basics:** detach (leave it running) with **Ctrl+B, then D**; reconnect later with
`ssh …` then `tmux attach -t flame`. Always train inside tmux.

## 5. Benchmark first (10 min, ~$0.40)

```bash
# data for the benchmark: the small Shakespeare set is enough to measure speed
curl -sL --create-dirs -o data/raw/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
flame tokenizer byte --out data/tokenizers/byte.json
flame prepare --input data/raw/shakespeare.txt --tokenizer data/tokenizers/byte.json --out data/shakespeare --separator "\n\n" --val-fraction 0.1
flame train --config configs/tiny-shakespeare.toml --run runs/bench --steps 200
flame status --run runs/bench          # "throughput … TFLOP/s achieved"
```

That number is for a tiny model; the 30 M model will reach a higher number. The real check is
the first ~50 steps of step 7: `flame status` shows the speed and the time left for the actual run.
If it's far below the table above, try `FLAME_TF32=0` once to compare, and check `nvidia-smi`
(GPU utilisation should be high).

## 6. Data (~1.5–2.5 h, unattended in tmux)

Follow [end-to-end-guide.md](end-to-end-guide.md) steps ⑤–⑥ on the rented machine (the
datacentre network makes downloads fast):

```bash
pip install pyarrow requests
tools/fetch_corpus.sh all
# then the `flame tokenizer train …` and the 4 × `flame prepare …` commands from step ⑥
```

**Already prepared the data at home?** Upload it instead (from your laptop):
`rsync -avP -e "ssh -p PORT" ~/code/FLAME/data/ root@HOST:~/code/FLAME/data/`. Only the
`data/tokenizers/` folder and the prepared folders (`data/rust-code`, `data/english`, …) are needed.

## 7. Train

The rented GPU has much more memory than your laptop, so read each step in bigger pieces:
same tokens per step (identical training), fewer, larger micro-batches. In
`configs/rust-30m-stage1.toml` and `rust-30m-stage2.toml` set:

| GPU memory | micro_batch_size | grad_accum_steps |
|---|---|---|
| 24 GB (4090) | 8 | 16 |
| 40–80 GB (A100, H100) | 16 | 8 |

(For `rust-100m-*`: 24 GB → 4 × 32; 80 GB → 16 × 8.) These are `[train]` settings, so you may
switch them back to the laptop values later when resuming at home.

```bash
flame train --config configs/rust-30m-stage1.toml --run runs/rust-30m-stage1
flame train --config configs/rust-30m-stage2.toml --run runs/rust-30m-stage2
```

No `--max-minutes` is needed here: let it run. Check from another tmux window
(**Ctrl+B, then C**) with `flame status --run runs/rust-30m-stage1` and `nvidia-smi`.

**Save your work home regularly** (from your laptop, e.g. every hour):

```bash
rsync -avP -e "ssh -p PORT" root@HOST:~/code/FLAME/runs/ ~/code/FLAME/runs/
```

If the machine dies, you continue from the last copied checkpoint, at home or on a new rented machine.

## 8. Lintless on the same machine (~1 h)

```bash
cd ~/code/lintless
# upload your collected data/raw/*.jsonl from the laptop (rsync as above), or collect here:
lintless collect-seeds && lintless split data/raw/*.jsonl --out data
lintless train --config configs/finetune-30m.toml --run runs/lintless-30m
lintless eval  --run runs/lintless-30m
```

Then decide (Lintless guide step 8). If you go to 100 M: `configs/rust-100m-stage1/2.toml`,
then set `base_run = "../FLAME/runs/rust-100m-stage2"` in Lintless and fine-tune again.

## 9. Finish

1. Copy everything home: `rsync … root@HOST:~/code/FLAME/runs/ ~/code/FLAME/runs/` and the
   same for `~/code/lintless/runs/`. You don't need to copy the raw data back.
2. Check at home: `flame status --run runs/rust-30m-stage2`, `lintless fix some_file.rs --run runs/lintless-30m`.
3. **Destroy the instance** in vast.ai.

## Getting even faster later

FLAME still computes everything else in fp32 without fused kernels. The next speed steps
(bf16 mixed precision, fused attention) would roughly double or triple the speed again,
making the 100 M model a 2–4 h job on an H100. See [roadmap.md](roadmap.md).

## Alternative: Modal (monthly free credits)

[Modal](https://modal.com) bills per second and gives monthly free credits, which can cover
the 30 M model. [`tools/modal_train.py`](../tools/modal_train.py) builds FLAME with CUDA into a
Modal image, keeps data and checkpoints in one Volume (`flame-vol`), and runs each stage as a
function. Rough budget: the 30 M model is ≈ 2–4 GPU-hours on an A100/H100 plus ≈ 2–3 CPU-hours
for data preparation; check Modal's pricing page for current per-hour rates and your free credit.

```bash
pip install modal && modal setup
modal run tools/modal_train.py::smoke                  # GPU check; must say "running on CUDA GPU 0"
modal run --detach tools/modal_train.py::prepare_data  # CPU only
modal run --detach tools/modal_train.py::train --stage 1
modal run --detach tools/modal_train.py::train --stage 2
modal volume get flame-vol work/runs/rust-30m-stage2 ./runs/
```

`train` resumes from the last checkpoint, so re-running the same command continues after a
preemption or the time limit. Set `FLAME_MODAL_GPU` (`H100`, `A100-80GB`, `L40S`, `A10G`, `L4`)
to choose the GPU. Set a spend limit in the Modal dashboard as a safety net.
