#!/usr/bin/env python3
"""Task evaluation for a trained FLAME Rust model: does its output actually work?

Perplexity (`flame eval`) measures how well a model predicts text. This script measures
two tasks, checked automatically with the Rust compiler:

  complete  The prompt is a doc comment + function signature. The model writes the body.
            Score: does the function compile, and does it pass the task's asserts?
  fix       The prompt is a real rustc error for a broken program, cut off at `help:`.
            Score: does the model's help text contain the known fix (e.g. `.clone()`)?

Every task is sampled with several seeds. The report gives, per task and overall:
  rate    share of all samples that passed       (how often the model is right)
  any@k   share of tasks where at least one of k seeds passed

Usage (from a folder that holds the model package: runs/, data/tokenizers/, target/):
  python3 tools/task_eval.py
  python3 tools/task_eval.py --root ~/flame-model --seeds 5 --out report.json

Needs `rustc` on PATH (or --rustc). Generation runs on the CPU unless --gpu is given.
The "fix" score is a substring match: it can miss a correct fix worded differently,
and a match is not proof the model understood. Read the saved samples too.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_RUN = "runs/rust-30m-dgx-20261102-stage2"
DEFAULT_TASKS = Path(__file__).resolve().parent / "eval" / "rust_tasks.json"


# ---------------------------------------------------------------- generation

def generate(args, prompt: str, seed: int) -> str:
    """Run `flame generate` once and return only the continuation after the prompt."""
    cmd = [args.flame, "generate", "--run", args.run, "--prompt", prompt,
           "--max-tokens", str(args.max_tokens), "--temperature", str(args.temperature),
           "--seed", str(seed)]
    if not args.gpu:
        cmd.append("--cpu")
    res = subprocess.run(cmd, cwd=args.root, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"flame generate failed:\n{res.stderr.strip()}")
    out = res.stdout
    if out.endswith("\n"):
        out = out[:-1]  # flame prints a final newline after the sample
    # flame echoes the prompt; strip it. Tokenizer round trips can alter whitespace,
    # so fall back to the whole output if the echo does not match exactly.
    return out[len(prompt):] if out.startswith(prompt) else out


# ---------------------------------------------------------------- rust helpers

def run_rustc(args, source: str, workdir: Path, binary: bool) -> subprocess.CompletedProcess:
    src = workdir / "src" / "main.rs"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(source)
    out = workdir / "prog"
    cmd = [args.rustc, "--edition", "2021", "--color", "never", "--crate-name", "eval",
           "-o", str(out) if binary else os.devnull, "src/main.rs"]
    return subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=120)


def close_function(continuation: str) -> str | None:
    """Cut the continuation where the function body (opened by the prompt's `{`) closes.

    Tracks braces outside strings and `//` comments. Returns None if it never closes.
    """
    depth, i, n = 1, 0, len(continuation)
    while i < n:
        c = continuation[i]
        if c == "/" and continuation.startswith("//", i):
            j = continuation.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == '"':
            i += 1
            while i < n and continuation[i] != '"':
                i += 2 if continuation[i] == "\\" else 1
        elif c == "'" and i + 2 < n and continuation[i + 2] == "'":
            i += 2  # char literal like '{'
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return continuation[: i + 1]
        i += 1
    return None


def error_prompt(stderr: str) -> str:
    """First rustc error block, cut right after its first `help:` (or with `help:` added)."""
    lines = stderr.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("error"))
    block = []
    for line in lines[start:]:
        if block and not line.strip():
            break
        block.append(line)
    text = "\n".join(block)
    cut = text.find("help:")
    if cut >= 0:
        return text[: cut + len("help:")]
    return text.rstrip() + "\nhelp:"


# ---------------------------------------------------------------- tasks

def eval_complete(args, task, seed, workdir):
    prompt = task["prompt"]
    cont = generate(args, prompt, seed)
    body = close_function(cont)
    rec = {"output": cont}
    if body is None:
        rec.update(passed=False, stage="unclosed", detail="function body never closed")
        return rec
    program = "#![allow(warnings)]\n" + prompt + body + "\n\n" + task["test"] + "\n"
    rec["program"] = program
    comp = run_rustc(args, program, workdir, binary=True)
    if comp.returncode != 0:
        first = next((l for l in comp.stderr.splitlines() if l.startswith("error")), "")
        rec.update(passed=False, stage="compile", detail=first)
        return rec
    try:
        run = subprocess.run([str(workdir / "prog")], capture_output=True, text=True,
                             timeout=10)
    except subprocess.TimeoutExpired:
        rec.update(passed=False, stage="test", detail="timed out")
        return rec
    ok = run.returncode == 0
    err = run.stderr.strip().splitlines()
    why = next((l for l in err if "assert" in l or "left:" in l), err[0] if err else "")
    rec.update(passed=ok, stage="pass" if ok else "test", detail="" if ok else why.strip())
    return rec


def eval_fix(args, task, seed, prompt):
    cont = generate(args, prompt, seed)
    # Judge only the help text right after `help:`, not later drift.
    head = "\n".join(cont.splitlines()[: args.fix_lines])
    hit = next((s for s in task["accept"] if s in head), None)
    return {"output": cont, "judged": head, "passed": hit is not None,
            "stage": "pass" if hit else "wrong", "detail": hit or ""}


def build_fix_prompt(args, task, workdir) -> str:
    comp = run_rustc(args, task["source"], workdir, binary=False)
    code = task.get("code")
    if comp.returncode == 0 or (code and f"error[{code}]" not in comp.stderr):
        raise RuntimeError(f"task {task['id']}: broken program did not give {code}:\n"
                           f"{comp.stderr}")
    return error_prompt(comp.stderr)


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=".", help="folder holding runs/ and data/ (default: .)")
    p.add_argument("--run", default=DEFAULT_RUN, help=f"run directory (default: {DEFAULT_RUN})")
    p.add_argument("--flame", help="flame binary (default: <root>/target/release/flame)")
    p.add_argument("--rustc", default=shutil.which("rustc") or
                   str(Path.home() / ".cargo/bin/rustc"))
    p.add_argument("--tasks", default=str(DEFAULT_TASKS))
    p.add_argument("--only", choices=["complete", "fix"], help="run one task type")
    p.add_argument("--seeds", type=int, default=3, help="samples per task (default: 3)")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--fix-lines", type=int, default=4,
                   help="lines after `help:` that are judged (default: 4)")
    p.add_argument("--gpu", action="store_true", help="do not pass --cpu to flame")
    p.add_argument("--out", help="save every sample and the scores as JSON")
    args = p.parse_args()

    args.root = str(Path(args.root).expanduser().resolve())
    args.flame = str(Path(args.flame).expanduser().resolve()) if args.flame else \
        str(Path(args.root) / "target/release/flame")
    for path, what in [(args.flame, "flame binary"), (args.rustc, "rustc"),
                       (Path(args.root) / args.run / "config.toml", "trained run")]:
        if not Path(path).exists():
            sys.exit(f"error: {what} not found at {path}")

    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    if args.only:
        tasks = [t for t in tasks if t["kind"] == args.only]
    seeds = list(range(1, args.seeds + 1))
    print(f"FLAME task eval: {len(tasks)} tasks x {len(seeds)} seeds, run {args.run}")

    results, started = [], time.time()
    with tempfile.TemporaryDirectory(prefix="flame-eval-") as tmp:
        workdir = Path(tmp)
        for task in tasks:
            prompt = build_fix_prompt(args, task, workdir) if task["kind"] == "fix" else None
            samples = []
            for seed in seeds:
                if task["kind"] == "complete":
                    rec = eval_complete(args, task, seed, workdir)
                else:
                    rec = eval_fix(args, task, seed, prompt)
                rec["seed"] = seed
                samples.append(rec)
            passed = sum(s["passed"] for s in samples)
            stages = ", ".join(s["stage"] for s in samples)
            print(f"  [{task['kind']:8}] {task['id']:22} {passed}/{len(samples)}  ({stages})", flush=True)
            results.append({"id": task["id"], "kind": task["kind"],
                            "prompt": prompt or task["prompt"], "samples": samples})

    print(f"\nScores ({time.time() - started:.0f} s):")
    summary = {}
    for kind in ["complete", "fix"]:
        rs = [r for r in results if r["kind"] == kind]
        if not rs:
            continue
        total = sum(len(r["samples"]) for r in rs)
        good = sum(s["passed"] for r in rs for s in r["samples"])
        anyk = sum(any(s["passed"] for s in r["samples"]) for r in rs)
        summary[kind] = {"tasks": len(rs), "samples": total, "passed": good,
                         "rate": good / total, f"any@{len(seeds)}": anyk / len(rs)}
        label = "code completion (compiles + tests pass)" if kind == "complete" \
            else "error fixing (known fix in help text)"
        print(f"  {label}: {good}/{total} samples = {good / total:.0%}, "
              f"any@{len(seeds)} = {anyk}/{len(rs)} tasks")
        if kind == "complete":
            comp = sum(s["stage"] in ("pass", "test") for r in rs for s in r["samples"])
            print(f"    compiled at all: {comp}/{total} = {comp / total:.0%}")

    if args.out:
        report = {"run": args.run, "seeds": seeds, "max_tokens": args.max_tokens,
                  "temperature": args.temperature, "summary": summary, "tasks": results}
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nAll samples saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
