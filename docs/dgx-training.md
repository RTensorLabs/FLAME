# DGX 30M run and recovery

Run ID: `rust-30m-dgx-20261102`. The two-stage Rust + English recipe uses an
8-layer, width-512 Transformer, 8 heads, context 1024, and an 8192-token BPE.
The exact parameter count is recorded in each checkpoint's `MODEL_CARD.json`.
The 7500 + 2500 optimizer steps consume 1,310,720,000 tokens in total.
DGX configs use micro-batch 8, accumulation 16 (131,072 tokens per step).

The managed DGX runner uses PyTorch BF16 autocast, fused causal attention and
fused AdamW to avoid the native fp32/unfused implementation's bandwidth cost.
Master weights and optimizer moments stay fp32 and use FLAME's safetensors
names/layout. A Rust `batch-server` uses the exact native mixture/RNG, including
the same `(step, micro)` data order. Native FLAME still initializes stages and
does final evaluation/generation; the final model works in the Rust CLI.
Each checkpoint records its backend and precision in `backend.json`.

Measured on this shared GB10 on 2026-11-02: native fp32 about 5,400 tokens/s;
BF16/fused benchmark about 54,000 tokens/s. The initial 70–80 h estimate was
for the native path; the accelerated path projects roughly 7 h of training
plus evaluation, backups and load variation. Actual run metrics supersede this
short benchmark. FP32 cross-backend max logit error: 2.84e-6; BF16 intentionally
uses lower activation precision, so it does not reproduce fp32 training bitwise.
Fused CUDA backward kernels can also vary slightly between sessions; checkpoint
reload preserves saved tensors, optimizer and batch order, but subsequent GPU
updates are checked numerically rather than promised bitwise identical.
The real-data eight-update check achieved 51,936–54,225 tokens/s after warmup;
loss decreased from 8.9966 to 8.5547. The split-resume gate (2 updates versus
1 + reload + 1) agreed within 1.13e-6 for weights and 4.58e-6 for optimizer
moments. Native Rust CLI generation loaded the resulting weights successfully.

## Location and progress

On the DGX (`kenpath`, Tailscale `100.122.210.21`, LAN `192.168.88.69`):

- Source/build/data: `/home/kenpath/flame-training/FLAME`
- Runs: `runs/rust-30m-dgx-20261102-stage1` and `...-stage2`
- Logs: `ops/runner.log`, `ops/corpus.log`, `ops/build.log`, `ops/publisher.log`
- Independent recovery tree: `/home/kenpath/flame-recovery/rust-30m-dgx-20261102`
- Services: `flame-30m.service`, `flame-checkpoint-publisher.timer`
- Local recovery copies: `D:\ModelBackups\FLAME\rust-30m-dgx-20261102`
- Windows task `FLAME 30M recovery pull`: every 15 minutes while the PC is awake;
  missed runs catch up when the PC is available.

```bash
cd /home/kenpath/flame-training/FLAME
tail -20 ops/runner.log
target/release/flame status --run runs/rust-30m-dgx-20261102-stage1
systemctl --user status flame-30m flame-checkpoint-publisher.timer
```

Rust/CUDA and Python dependencies are isolated under `flame-training`.
Existing DGX services are not stopped. The training service has a 20 GiB memory
limit, a CPU quota, and a low CPU priority; it restarts on failure and across reboots.

## Checkpoints and publication

The trainer saves crash-safe model + AdamW + scheduler/data-order state every
five minutes (after a complete optimizer step). The runner pauses about every
30 minutes and copies a stable checkpoint into a new recovery directory, with
tokenizer, frozen config, code, metrics, and SHA256 checksums. The first training
step is also archived. There is no automatic pruning or overwriting of old archives.

The publisher checks every five minutes, uploading uniquely named archives and
checksums to this private repository's recovery release:

<https://github.com/RTensorLabs/FLAME/releases/tag/rust-30m-dgx-20261102-checkpoints>

It verifies the digest and size returned by GitHub. Authentication lives in a
mode-600 file outside the checkout; it is never included in archives or commits.
Publication failures remain visible in the service status/log and retry on the
next timer run. Local training and snapshots continue if GitHub is unavailable.

After both stages finish, the runner evaluates the model, generates a sample,
and produces a complete model package containing both stages, best/latest weights,
optimizer state, tokenizer, source, configs, and validation results. The publisher
uploads that package to:

<https://github.com/RTensorLabs/FLAME/releases/tag/rust-30m-dgx-20261102>

That final release does not exist until training and evaluation have completed.
Intermediate checkpoint assets are not a finished trained model. Training corpus
files are excluded from GitHub; an independent prepared-data copy and its hashes
remain in the DGX recovery tree. PC copies are additional independent storage,
not deletion mirrors. Losing one copy does not remove the others. This does not
protect against someone deliberately deleting every independent copy.

## Restore a model

Download an archive and its `.sha256` from Releases, or copy them from either
recovery location. On Linux:

```bash
sha256sum -c rust-30m-dgx-20261102-stage1-step-0000001.tar.gz.sha256
mkdir restored-flame && cd restored-flame
tar xzf ../rust-30m-dgx-20261102-stage1-step-0000001.tar.gz
sha256sum -c SHA256SUMS
cargo build --release --locked # add --features cuda with a compatible CUDA toolkit
target/release/flame generate --cpu \
  --run runs/rust-30m-dgx-20261102-stage1 --prompt 'fn main() {'
```

Replace the example filename and run path with the checkpoint/stage you chose.
Generation needs the model, frozen config and tokenizer; it does not need the
training corpus. The final package also has `VALIDATION.txt`.

To resume training, restore the corresponding `data/{rust-code,rust-docs,english,
rustc-errors}` folders from `prepared-data` into the restored checkout. Compare
their hashes to `prepared-manifest.json`. Run the same `flame train --config ...
--run ...` command; it loads model, optimizer, step count and deterministic data
order. A second training process must never write to the same run folder.
For continuation with the accelerated backend, install the dependencies listed
in `tools/dgx-bootstrap.sh`, build `cargo build --release --locked --example
batch-server`, and run `.venv/bin/python tools/torch_train.py --config ... --run
...` from the restored root. Returning to the native trainer preserves the model,
optimizer and batching state but changes arithmetic precision/kernel order.

To pause the managed runner, create `ops/PAUSE` plus the active run's `PAUSE` file.
It finishes the current step, saves and archives, then stops. Remove `ops/PAUSE`
and start `flame-30m` to continue. To resume from a recovery archive after deleting
the entire training folder, restore the source/model and prepared data as above,
then reinstall the services or invoke the training command directly.
