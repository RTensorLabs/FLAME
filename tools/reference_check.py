"""Cross-checks FLAME's Candle Transformer against an independent PyTorch implementation.

Usage (from the repo root):
    cargo run --release -p flame-cli -- export-reference --config configs/tiny-shakespeare.toml --out runs/refcheck
    python tools/reference_check.py runs/refcheck

The script re-implements the architecture from the paper descriptions (RMSNorm, RoPE with the
rotate-half layout, causal multi-head attention, SwiGLU, tied embeddings), loads the *same*
weights, runs the *same* input, and compares logits. A max abs difference around 1e-5 means
both implementations compute the same function; anything above ~1e-3 means a bug in one of them.

Requires: pip install torch safetensors
"""
import json
import math
import sys

import torch
from safetensors.torch import load_file


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, theta):
    # x: (B, H, T, Hd)
    _, _, t, hd = x.shape
    inv_freq = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    freqs = torch.outer(torch.arange(t, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * emb.cos() + rotated * emb.sin()


def forward(w, cfg, ids):
    d, h = cfg["d_model"], cfg["n_heads"]
    hd = d // h
    b, t = ids.shape
    x = w["tok_emb.weight"][ids]
    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
    for i in range(cfg["n_layers"]):
        p = f"layers.{i}."
        a = rms_norm(x, w[p + "attn_norm.weight"], cfg["norm_eps"])
        q, k, v = (
            (a @ w[p + f"attn.{n}_proj.weight"].T).view(b, t, h, hd).transpose(1, 2)
            for n in ("q", "k", "v")
        )
        q, k = rope(q, cfg["rope_theta"]), rope(k, cfg["rope_theta"])
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask
        y = att.softmax(-1) @ v
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + y @ w[p + "attn.o_proj.weight"].T
        m = rms_norm(x, w[p + "mlp_norm.weight"], cfg["norm_eps"])
        g = torch.nn.functional.silu(m @ w[p + "mlp.gate.weight"].T)
        u = m @ w[p + "mlp.up.weight"].T
        x = x + (g * u) @ w[p + "mlp.down.weight"].T
    x = rms_norm(x, w["norm.weight"], cfg["norm_eps"])
    head = w["tok_emb.weight"] if cfg["tie_embeddings"] else w["lm_head.weight"]
    return x @ head.T


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "runs/refcheck"
    cfg = json.load(open(f"{d}/config.json"))
    w = load_file(f"{d}/model.safetensors")
    io = load_file(f"{d}/io.safetensors")
    with torch.no_grad():
        ref = forward(w, cfg, io["input"].long())
    diff = (ref - io["logits"]).abs().max().item()
    print(f"max |logits_torch - logits_flame| = {diff:.3e}")
    if diff < 1e-3:
        print("OK: FLAME and the PyTorch reference agree.")
    else:
        print("MISMATCH: the implementations differ — investigate before training!")
        sys.exit(1)


if __name__ == "__main__":
    main()
