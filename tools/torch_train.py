"""Optional BF16/fused-attention trainer; saves native FLAME safetensors state.

The Rust batch-server supplies exactly the same seeded batches as flame train.
Inference, evaluation, initialization, tokenizer and checkpoint layout stay native.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time
import tomllib

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch_backend import Model, optimizer


class Batches:
    def __init__(self, config, cfg):
        self.process = subprocess.Popen(["target/release/examples/batch-server", str(config)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.b = cfg["micro_batch_size"]
        self.t = cfg["seq_len"]
        self.accum = cfg.get("grad_accum_steps", 1)

    def get(self, request, n):
        self.process.stdin.write((request + "\n").encode())
        self.process.stdin.flush()
        size = n * 2 * self.b * self.t * 4
        raw = self.process.stdout.read(size)
        if len(raw) != size:
            raise RuntimeError(f"Rust batch-server returned {len(raw)} bytes; expected {size}")
        values = np.frombuffer(raw, dtype="<u4").reshape(n, 2, self.b, self.t)
        return [(torch.from_numpy(pair[0].astype(np.int64)).cuda(),
                 torch.from_numpy(pair[1].astype(np.int64)).cuda()) for pair in values]

    def train(self, step):
        return self.get(f"t {step}", self.accum)

    def val(self, n):
        return self.get(f"v {n}", n)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


def lr_at(step, cfg):
    warmup, maximum = cfg.get("warmup_steps", 0), cfg["max_steps"]
    peak, floor = cfg["lr"], cfg.get("min_lr", 0.0)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(maximum - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def load_optimizer(opt, model, path, step):
    moments = load_file(path, device="cuda")
    for name, p in zip(model.names, model.params):
        opt.state[p] = {"step": torch.tensor(float(step), device=p.device),
                        "exp_avg": moments[f"m.{name}"], "exp_avg_sq": moments[f"v.{name}"]}


def save(run, model, opt, state, best=False):
    if best:
        folder = run / "best"
        folder.mkdir(exist_ok=True)
        save_file(model.weights(), folder / "model.safetensors.tmp")
        (folder / "model.safetensors.tmp").replace(folder / "model.safetensors")
        return
    base = run / "checkpoints"
    temp, latest, old = (base / name for name in ("latest.tmp", "latest", "latest.old"))
    temp.mkdir(parents=True, exist_ok=True)
    save_file(model.weights(), temp / "model.safetensors")
    moments = {}
    for name, p in zip(model.names, model.params):
        s = opt.state[p]
        moments[f"m.{name}"] = s["exp_avg"].detach().float().cpu().contiguous()
        moments[f"v.{name}"] = s["exp_avg_sq"].detach().float().cpu().contiguous()
    save_file(moments, temp / "optim.safetensors")
    (temp / "backend.json").write_text(json.dumps({"backend": "pytorch-bf16-sdpa", "torch": torch.__version__,
        "cuda": torch.version.cuda, "master_weights": "fp32", "optimizer_moments": "fp32",
        "batch_order": "native FLAME Rust batch-server"}, indent=2), encoding="utf-8")
    # Completion marker last, matching native crash recovery conventions.
    (temp / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if old.exists():
        shutil.rmtree(old)
    if latest.exists():
        latest.rename(old)
    temp.rename(latest)
    if old.exists():
        shutil.rmtree(old)


@torch.no_grad()
def evaluate(model, batches):
    losses = []
    for ids, target in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(ids).flatten(0, 1).float(), target.flatten())
        losses.append(loss.float())
    return torch.stack(losses).mean().item()


def train(config, run, steps=None, max_minutes=None):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = Path(run)
    latest = run / "checkpoints/latest"
    if not (latest / "state.json").exists() and (run / "checkpoints/latest.old/state.json").exists():
        latest = run / "checkpoints/latest.old"
    # Delegate initial weights, frozen config checks and staged initialization to FLAME.
    subprocess.run(["target/release/flame", "train", "--config", str(config), "--run", str(run), "--steps", "0"], check=True)
    latest = run / "checkpoints/latest"
    cfg = tomllib.loads((run / "config.toml").read_text())
    tc = cfg["train"]
    state = json.loads((latest / "state.json").read_text())
    model = Model(load_file(latest / "model.safetensors", device="cuda"), cfg["model"])
    opt = optimizer(model, tc)
    load_optimizer(opt, model, latest / "optim.safetensors", state["step"])
    source = Batches(config, tc)
    validation = source.val(tc.get("eval_batches", 20))
    interrupted = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append("shutdown signal"))
    signal.signal(signal.SIGINT, lambda *_: interrupted.append("Ctrl+C"))
    start_step = state["step"]
    session_time = last_save = time.monotonic()
    tps = tc["micro_batch_size"] * tc["seq_len"] * tc.get("grad_accum_steps", 1)
    state["sessions"] += 1
    reason = "finished"
    print(f"[torch] BF16 fused attention; fp32 master weights/moments; native batches; resumed step {start_step}", flush=True)
    with (run / "sessions.log").open("a", encoding="utf-8") as sessions, (run / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        sessions.write(f"unix={int(time.time())} accelerated session {state['sessions']} started at step {start_step}\n")
        sessions.flush()
        try:
            while state["step"] < tc["max_steps"]:
                if interrupted:
                    reason = interrupted[0]
                    break
                if (run / "PAUSE").exists():
                    (run / "PAUSE").unlink()
                    reason = "PAUSE file found"
                    break
                if steps is not None and state["step"] - start_step >= steps:
                    reason = "session step limit"
                    break
                if max_minutes and time.monotonic() - session_time > max_minutes * 60 - 15:
                    reason = "time budget reached"
                    break
                torch.cuda.synchronize()
                started = time.monotonic()
                lr = lr_at(state["step"], tc)
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss_total = torch.zeros((), device="cuda")
                for ids, target in source.train(state["step"]):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(model(ids).flatten(0, 1).float(), target.flatten())
                        scaled = loss / tc.get("grad_accum_steps", 1)
                    scaled.backward()
                    loss_total += loss.detach()
                loss_value = loss_total.item() / tc.get("grad_accum_steps", 1)
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Nonfinite loss at step {state['step']}")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0), error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                state["step"] += 1
                state["tokens_seen"] += tps
                state["train_seconds"] += elapsed
                tok_s = tps / elapsed
                remaining = (tc["max_steps"] - state["step"]) * elapsed / 3600
                print(f"[torch] step {state['step']}/{tc['max_steps']} | loss {loss_value:.4f} | lr {lr:.2e} | {tok_s:.0f} tok/s | ETA {remaining:.2f} h", flush=True)
                metrics.write(json.dumps({"step": state["step"], "tokens": state["tokens_seen"],
                    "loss": loss_value, "lr": lr, "grad_norm": norm.item(), "tok_per_sec": tok_s,
                    "train_hours": state["train_seconds"] / 3600, "backend": "pytorch-bf16-sdpa"}) + "\n")
                metrics.flush()
                if state["step"] % tc.get("eval_every", 200) == 0 or state["step"] == tc["max_steps"]:
                    val = evaluate(model, validation)
                    if not math.isfinite(val):
                        raise RuntimeError("Nonfinite validation loss")
                    if state["best_val_loss"] is None or val < state["best_val_loss"]:
                        state["best_val_loss"] = val
                        save(run, model, opt, state, best=True)
                    metrics.write(json.dumps({"step": state["step"], "tokens": state["tokens_seen"], "val_loss": val}) + "\n")
                    metrics.flush()
                    print(f"[torch] eval @ {state['step']} | val_loss {val:.4f} | perplexity {math.exp(val):.2f}", flush=True)
                if time.monotonic() - last_save >= tc.get("checkpoint_every_minutes", 5) * 60:
                    save(run, model, opt, state)
                    last_save = time.monotonic()
                    print(f"[torch] checkpoint saved at step {state['step']}", flush=True)
            save(run, model, opt, state)
            sessions.write(f"unix={int(time.time())} accelerated session {state['sessions']} ended at step {state['step']}: {reason}\n")
            print(f"[torch] stopped at step {state['step']}: {reason}; checkpoint saved", flush=True)
        finally:
            source.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--max-minutes", type=float)
    args = ap.parse_args()
    train(args.config, args.run, args.steps, args.max_minutes)
