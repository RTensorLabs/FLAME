"""Functional resume gate: two updates vs one + reload + one, with real batches."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    runs = [args.out / n for n in ("straight", "split")]
    for run in runs:
        shutil.copytree(args.source / "checkpoints/latest", run / "checkpoints/latest")
        shutil.copy2(args.source / "config.toml", run / "config.toml")
    for run, steps in ((runs[0], 2), (runs[1], 1), (runs[1], 1)):
        subprocess.run([str(Path(__import__("sys").executable)), "tools/torch_train.py", "--config", args.config,
                        "--run", str(run), "--steps", str(steps)], check=True)
    result = {}
    for name in ("model.safetensors", "optim.safetensors"):
        a, b = [load_file(run / "checkpoints/latest" / name) for run in runs]
        error = max((a[k] - b[k]).abs().max().item() for k in a)
        result[name] = error
        # Fused attention backward is not bitwise deterministic. Check numerical
        # agreement while separately checking that the exact saved state loads.
        assert error < 1e-5, result
    states = [json.loads((run / "checkpoints/latest/state.json").read_text()) for run in runs]
    assert states[0]["step"] == states[1]["step"] and states[0]["tokens_seen"] == states[1]["tokens_seen"]
    print("RESUME VERIFIED", json.dumps(result), flush=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
