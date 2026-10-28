"""Train FLAME on Modal (https://modal.com) using its monthly free credits.

    pip install modal && modal setup                      # once

    modal run tools/modal_train.py::smoke                 # ~5 min: build, CUDA check, overfit test
    modal run --detach tools/modal_train.py::prepare_data # CPU only: download + tokenise (~2-3 h)
    modal run --detach tools/modal_train.py::train --stage 1
    modal run --detach tools/modal_train.py::train --stage 2

    modal volume ls  flame-vol work/runs                  # what is saved
    modal volume get flame-vol work/runs/rust-30m-stage2 ./runs/   # copy the result home

Everything lives in one Modal Volume (`flame-vol`): data in work/data, checkpoints in work/runs.
`train` resumes automatically from the last checkpoint, so if it is stopped, preempted or hits
the time limit, run the same command again.

Choose the GPU with an environment variable (it also sets the CUDA architecture the binary is
compiled for, so use the same value every time or the image is rebuilt):

    FLAME_MODAL_GPU=A100-80GB (default) | H100 | L40S | A10G | L4

This script has not been run against a live Modal account by its author; if the image build
fails, see the notes next to `image` below.
"""
import os
import subprocess
import threading

import modal

GPU = os.environ.get("FLAME_MODAL_GPU", "A100-80GB")

# GPU -> (CUDA compute capability, micro_batch_size, grad_accum_steps). Same tokens per step
# as the laptop config (2 x 64), just fewer, larger micro-batches. See docs/cloud-gpu-guide.md.
GPUS = {
    "H100": ("90", 16, 8),
    "A100-80GB": ("80", 16, 8),
    "A100-40GB": ("80", 16, 8),
    "L40S": ("89", 16, 8),
    "A10G": ("86", 8, 16),
    "L4": ("89", 8, 16),
}
CAP, MICRO, ACCUM = GPUS[GPU]

STAGES = {1: "rust-30m-stage1", 2: "rust-30m-stage2"}
SRC = "/root/FLAME"
CPU_BIN = "/opt/flame-cpu/release/flame"    # data preparation needs no GPU libraries
CUDA_BIN = "/opt/flame-cuda/release/flame"  # training
WORK = "/vol/work"                          # configs use paths relative to the working directory

vol = modal.Volume.from_name("flame-vol", create_if_missing=True)

# If the CUDA build below fails to link libcuda, attach the GPU to that build step instead:
# `.run_commands(<the cuda cargo command>, gpu=GPU)` (costs a few minutes of GPU time once).
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "pkg-config", "git", "curl", "rsync")
    .pip_install("pyarrow", "requests")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin",
        "CUDA_COMPUTE_CAP": CAP,
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",  # libcuda stub: no GPU during the build
    })
    .add_local_dir(
        ".", SRC, copy=True,
        ignore=["target", "data", "runs", ".git", "**/__pycache__"],
    )
    .run_commands(
        f"cd {SRC} && CARGO_TARGET_DIR=/opt/flame-cpu cargo build --release -p flame-cli",
        f"cd {SRC} && CARGO_TARGET_DIR=/opt/flame-cuda cargo build --release -p flame-cli --features cuda",
    )
)

app = modal.App("flame", image=image)


def sh(cmd, cwd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", cmd], cwd=cwd, check=True)


class AutoCommit:
    """Persist the volume every few minutes so checkpoints survive preemption."""

    def __init__(self, every=180):
        self.stop, self.every = threading.Event(), every

    def __enter__(self):
        def loop():
            while not self.stop.wait(self.every):
                vol.commit()
        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join()
        vol.commit()


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=1800)
def smoke():
    """Is the CUDA build working? Must print 'running on CUDA GPU 0' and val_loss ~0.05."""
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv", "/tmp")
    work = "/tmp/smoke"
    os.makedirs(work, exist_ok=True)
    sh(f"{CUDA_BIN} tokenizer byte --out data/tokenizers/byte.json", work)
    sh(f"{CUDA_BIN} prepare --input {SRC}/examples/overfit.txt --tokenizer data/tokenizers/byte.json "
       f"--out data/overfit --val-fraction 0.25", work)
    sh(f"{CUDA_BIN} train --config {SRC}/configs/overfit.toml --run runs/overfit", work)


@app.function(cpu=8, memory=32768, volumes={"/vol": vol}, timeout=6 * 3600)
def prepare_data():
    """Steps 5-6 of docs/end-to-end-guide.md on a CPU container; results go to the volume."""
    work = "/tmp/prep"  # raw downloads stay on the container's local disk
    os.makedirs(work, exist_ok=True)
    sh(f"bash {SRC}/tools/fetch_corpus.sh all", work)
    t = "data/tokenizers/rust-en-8k.json"
    reg = "~/.cargo/registry/src"
    sh(f"{CPU_BIN} tokenizer train --input data/raw/rust-code {reg} data/raw/rust-docs "
       f"data/raw/rustc-errors data/raw/english --ext rs,md,txt --vocab-size 8192 "
       f"--out {t} --sample-mb 250", work)
    sh(f"{CPU_BIN} prepare --input data/raw/rust-code {reg} --ext rs --tokenizer {t} --out data/rust-code", work)
    sh(f"{CPU_BIN} prepare --input data/raw/rust-docs --ext md --tokenizer {t} --out data/rust-docs", work)
    sh(f"{CPU_BIN} prepare --input data/raw/rustc-errors --ext txt --tokenizer {t} "
       f"--out data/rustc-errors --val-fraction 0.05", work)
    sh(f"{CPU_BIN} prepare --input data/raw/english --ext txt --tokenizer {t} --out data/english", work)
    # copy only the prepared data (not data/raw) to the volume
    os.makedirs(f"{WORK}/data", exist_ok=True)
    sh(f"rsync -a --exclude raw data/ {WORK}/data/", work)
    vol.commit()


def _config_for_gpu(stage: int) -> str:
    """Copy the stage config, raising micro-batch size for the GPU (same tokens per step)."""
    text = open(f"{SRC}/configs/{STAGES[stage]}.toml").read()
    lines = []
    for line in text.splitlines():
        if line.startswith("micro_batch_size"):
            line = f"micro_batch_size = {MICRO}"
        elif line.startswith("grad_accum_steps"):
            line = f"grad_accum_steps = {ACCUM}"
        lines.append(line)
    path = f"{WORK}/{STAGES[stage]}.toml"
    open(path, "w").write("\n".join(lines) + "\n")
    return path


TRAIN_HOURS = 8


@app.function(
    gpu=GPU, volumes={"/vol": vol}, timeout=TRAIN_HOURS * 3600,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
def train(stage: int = 1):
    """Train (or resume) stage 1 or 2 of the 30M model. Re-run the same command to resume."""
    vol.reload()
    os.makedirs(WORK, exist_ok=True)
    if not os.path.isdir(f"{WORK}/data/rust-code"):
        raise SystemExit("No prepared data: run `modal run tools/modal_train.py::prepare_data` first.")
    if stage == 2 and not os.path.isdir(f"{WORK}/runs/{STAGES[1]}"):
        raise SystemExit("Stage 2 starts from stage 1's weights: finish `--stage 1` first.")
    cfg = _config_for_gpu(stage)
    # stop cleanly with a checkpoint 10 min before Modal's time limit
    minutes = TRAIN_HOURS * 60 - 10
    with AutoCommit():
        sh(f"{CUDA_BIN} train --config {cfg} --run runs/{STAGES[stage]} --max-minutes {minutes}", WORK)
        sh(f"{CUDA_BIN} status --run runs/{STAGES[stage]}", WORK)
