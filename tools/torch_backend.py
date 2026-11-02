"""Accelerated PyTorch kernels for the same FLAME Transformer and checkpoint names.

Master parameters and optimizer moments remain fp32. BF16 autocast and fused
causal attention reduce activation memory and bandwidth on the DGX GB10.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class Model(torch.nn.Module):
    def __init__(self, weights, cfg):
        super().__init__()
        self.cfg = cfg
        self.names = sorted(weights)
        self.params = torch.nn.ParameterList([torch.nn.Parameter(weights[n]) for n in self.names])
        self.lookup = dict(zip(self.names, self.params))
        hd = cfg["d_model"] // cfg["n_heads"]
        inv = cfg["rope_theta"] ** (-torch.arange(0, hd, 2, dtype=torch.float32,
                                                 device=self.params[0].device) / hd)
        freqs = torch.outer(torch.arange(cfg["max_seq_len"], dtype=torch.float32,
                                        device=inv.device), inv)
        emb = torch.cat((freqs, freqs), -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def weights(self):
        return {n: p.detach().float().cpu().contiguous() for n, p in zip(self.names, self.params)}

    def norm(self, x, name):
        return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.cfg["norm_eps"]) * self.lookup[name]

    def rope(self, x):
        dtype = x.dtype
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
        return (x.float() * self.cos[:x.shape[-2]] + rotated.float() * self.sin[:x.shape[-2]]).to(dtype)

    def forward(self, ids):
        w, c = self.lookup, self.cfg
        b, t = ids.shape
        h, d = c["n_heads"], c["d_model"]
        hd = d // h
        x = F.embedding(ids, w["tok_emb.weight"])
        for i in range(c["n_layers"]):
            p = f"layers.{i}."
            a = self.norm(x, p + "attn_norm.weight")
            q, k, v = [F.linear(a, w[p + f"attn.{n}_proj.weight"]).view(b, t, h, hd).transpose(1, 2)
                       for n in ("q", "k", "v")]
            q, k = self.rope(q), self.rope(k)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            x = x + F.linear(y.transpose(1, 2).reshape(b, t, d), w[p + "attn.o_proj.weight"])
            m = self.norm(x, p + "mlp_norm.weight")
            g = F.silu(F.linear(m, w[p + "mlp.gate.weight"]))
            u = F.linear(m, w[p + "mlp.up.weight"])
            x = x + F.linear(g * u, w[p + "mlp.down.weight"])
        x = self.norm(x, "norm.weight")
        return F.linear(x, w["tok_emb.weight"] if c["tie_embeddings"] else w["lm_head.weight"])


def optimizer(model, cfg):
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": cfg.get("weight_decay", 0.1)},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        eps=cfg.get("eps", 1e-8), fused=True)


def benchmark(reference, steps=4):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(reference)
    cfg = json.loads((path / "config.json").read_text())
    w = load_file(path / "model.safetensors", device="cuda")
    io = load_file(path / "io.safetensors", device="cuda")
    model = Model(w, cfg).cuda()
    with torch.no_grad():
        logits = model(io["input"].long())
        diff = (logits - io["logits"]).abs().max().item()
        print(f"FP32 FLAME compatibility max_logit_error={diff:.8g}", flush=True)
        assert diff < 1e-3
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf_logits = model(io["input"].long()).float()
        print(f"BF16 relative logit RMS error={((bf_logits-logits).square().mean()/logits.square().mean()).sqrt().item():.6f}", flush=True)
    opt = optimizer(model, {"lr": 6e-4})
    torch.backends.cuda.matmul.allow_tf32 = True
    ids = torch.randint(0, cfg["vocab_size"], (8, 1024), device="cuda")
    target = torch.randint(0, cfg["vocab_size"], (8, 1024), device="cuda")
    for step in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(ids).flatten(0, 1).float(), target.flatten()) / 16
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(json.dumps({"step": step + 1, "seconds": elapsed, "tokens_per_second": 131072/elapsed,
                          "full_run_hours": elapsed * 10000 / 3600, "loss": loss.item()*16,
                          "gradient_norm": norm.item(), "peak_memory_GB": torch.cuda.max_memory_allocated()/1e9}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    benchmark(args.reference, args.steps)
