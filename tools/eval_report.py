#!/usr/bin/env python3
"""Turn task_eval.py JSON results into a Markdown report.

Usage:
  python3 tools/eval_report.py --model "stage 1=eval-stage1.json" \
      --model "stage 2=eval-stage2.json" --title "FLAME Rust 30M" --out report.md

With two models it adds a significance check (Fisher's exact test) for each rate.
Rates come with 95% Wilson intervals, so small samples show their uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a rate k/n (Wilson score)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher's exact test p-value for k1/n1 vs k2/n2."""
    total, hits = n1 + n2, k1 + k2

    def prob(x: int) -> float:
        return math.comb(hits, x) * math.comb(total - hits, n1 - x) / math.comb(total, n1)

    observed = prob(k1)
    lo, hi = max(0, hits - n2), min(hits, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = **{k / n:.0%}** ({lo:.0%}–{hi:.0%})"


def counts(report: dict, kind: str) -> dict:
    tasks = [t for t in report["tasks"] if t["kind"] == kind]
    samples = [s for t in tasks for s in t["samples"]]
    return {
        "tasks": len(tasks),
        "n": len(samples),
        "passed": sum(s["passed"] for s in samples),
        "compiled": sum(s["stage"] in ("pass", "test") for s in samples),
        "any": sum(any(s["passed"] for s in t["samples"]) for t in tasks),
        "stages": Counter(s["stage"] for s in samples),
    }


def code_block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", action="append", required=True, metavar="LABEL=JSON",
                   help="a model label and its task_eval.py --out file (repeatable)")
    p.add_argument("--title", default="FLAME task evaluation")
    p.add_argument("--tasks", default=str(Path(__file__).resolve().parent / "eval" / "rust_tasks.json"),
                   help="task file used for the run (for the expected fixes)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    models = []
    for spec in args.model:
        label, _, path = spec.partition("=")
        models.append((label, json.loads(Path(path).expanduser().read_text())))
    first = models[0][1]
    k = len(first["seeds"])
    try:
        rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        rustc = "rustc"

    L = [f"# {args.title}", ""]
    L += [f"Generated {date.today().isoformat()} by `tools/eval_report.py` from "
          f"`tools/task_eval.py` results.", ""]
    L += ["| Setting | Value |", "|---|---|"]
    for label, r in models:
        L.append(f"| Run ({label}) | `{r['run']}` |")
    n_complete = counts(first, "complete")["tasks"]
    n_fix = counts(first, "fix")["tasks"]
    L += [f"| Tasks | {n_complete} code completion + {n_fix} error fixing "
          f"(`tools/eval/rust_tasks.json`) |",
          f"| Samples | {k} seeds per task ({', '.join(map(str, first['seeds']))}) |",
          f"| Sampling | temperature {first['temperature']}, top-k 40, "
          f"max {first['max_tokens']} new tokens |",
          f"| Checker | {rustc}, edition 2021 |", ""]

    # ---------------- summary
    L += ["## Summary", "",
          "Percentages in brackets are 95% confidence intervals (Wilson): with this many "
          "samples, the true rate is very likely inside that range.", ""]
    header = "| Metric | " + " | ".join(label for label, _ in models) + " |"
    L += [header, "|---|" + "---|" * len(models)]
    rows = [
        ("Code: compiles", "complete", "compiled", "n"),
        ("Code: compiles **and** passes tests", "complete", "passed", "n"),
        (f"Code: tasks solved by any of {k} seeds", "complete", "any", "tasks"),
        ("Errors: `help:` names the known fix", "fix", "passed", "n"),
        (f"Errors: tasks fixed by any of {k} seeds", "fix", "any", "tasks"),
    ]
    for name, kind, key, total in rows:
        cells = []
        for _, r in models:
            c = counts(r, kind)
            cells.append(rate(c[key], c[total]) if key != "any"
                         else f"{c[key]}/{c[total]}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    if len(models) == 2:
        (la, ra), (lb, rb) = models
        L += [f"### Is the difference between {la} and {lb} real?", "",
              "Fisher's exact test: the p-value is the chance of seeing a gap at least this "
              "large if both models were actually equally good. Below 0.05 is the usual bar "
              "for \"real\".", "",
              "| Metric | " + la + " | " + lb + " | p-value | Verdict |", "|---|---|---|---|---|"]
        for name, kind, key in [("Code compiles", "complete", "compiled"),
                                ("Code passes tests", "complete", "passed"),
                                ("Errors fixed", "fix", "passed")]:
            a, b = counts(ra, kind), counts(rb, kind)
            pv = fisher(a[key], a["n"], b[key], b["n"])
            verdict = "real difference" if pv < 0.05 else "could be chance"
            L.append(f"| {name} | {a[key]}/{a['n']} | {b[key]}/{b['n']} | {pv:.3f} | {verdict} |")
        L.append("")

    # ---------------- failure breakdown
    L += ["## Why code samples failed", "",
          "| Outcome | " + " | ".join(label for label, _ in models) + " |",
          "|---|" + "---|" * len(models)]
    for stage, meaning in [("pass", "passed the tests"),
                           ("test", "compiled, wrong answer"),
                           ("compile", "did not compile"),
                           ("unclosed", f"function not finished within {first['max_tokens']} tokens")]:
        cells = [str(counts(r, "complete")["stages"].get(stage, 0)) for _, r in models]
        L.append(f"| {meaning} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------------- per task
    L += ["## Per task", "", f"Passing samples out of {k}.", "",
          "| Task | Kind | " + " | ".join(label for label, _ in models) + " |",
          "|---|---|" + "---|" * len(models)]
    for i, t in enumerate(first["tasks"]):
        cells = [str(sum(s["passed"] for s in r["tasks"][i]["samples"])) for _, r in models]
        L.append(f"| `{t['id']}` | {t['kind']} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------------- examples
    L += ["## Examples", ""]
    for label, r in models:
        seen, shown = set(), []
        for t in r["tasks"]:
            for s in t["samples"]:
                if t["kind"] == "complete" and s["passed"]:
                    body = s["program"].split("\n", 1)[1].split("\n\nfn main()")[0]
                    if body not in seen:
                        seen.add(body)
                        shown.append(body)
        L += [f"### Correct functions ({label})", ""]
        L += [code_block("\n\n".join(shown), "rust") if shown else "_None._", ""]

    label, r = models[-1]
    wrong = []
    for t in r["tasks"]:
        if t["kind"] == "complete":
            s = next((s for s in t["samples"] if s["stage"] == "test"), None)
            if s:
                body = s["program"].split("\n", 1)[1].split("\n\nfn main()")[0]
                why = s["detail"] if "assert" in s["detail"] or "left" in s["detail"] \
                    else "compiles, but the tests fail"
                wrong.append(f"// {t['id']}: {why}\n{body}")
    if wrong:
        L += [f"### Compiles but wrong ({label}, first such sample per task)", "",
              code_block("\n\n".join(wrong[:5]), "rust"), ""]

    L += [f"### Error fixing ({label}, seed {r['seeds'][0]} for each task)", "",
          "| Task | Expected fix contains | Model's `help:` text (first line) | Correct? |",
          "|---|---|---|---|"]
    accept = {t["id"]: t.get("accept", []) for t in json.loads(
        Path(args.tasks).read_text())["tasks"]}
    for t in r["tasks"]:
        if t["kind"] != "fix":
            continue
        s = t["samples"][0]
        said = (s["judged"].strip().splitlines() or [""])[0].replace("|", "\\|")[:80]
        exp = " or ".join(f"`{a}`" for a in accept.get(t["id"], []))
        said = f"`` {said} ``" if said else "_(nothing)_"  # double ticks: text may hold `
        L.append(f"| `{t['id']}` | {exp} | {said} | {'✅' if s['passed'] else '❌'} |")
    L.append("")
    passes = [(t["id"], s) for t in r["tasks"] if t["kind"] == "fix"
              for s in t["samples"] if s["passed"]]
    if passes:
        L += [f"Samples ({label}) that did name the fix:", ""]
        for tid, s in passes:
            L += [f"- `{tid}`, seed {s['seed']}:", "", code_block("help:" + s["judged"]), ""]

    # ---------------- method
    L += ["## Method", "",
          "- **Code completion:** the prompt is a doc comment plus a function signature ending "
          "in `{`. The output is cut where that function's braces close, compiled with the "
          "task's `main()` of asserts, and run. Samples that never close the function count "
          "as failures.",
          "- **Error fixing:** each broken program is compiled with `rustc` to get the real "
          "error. The prompt is the first error block cut right after `help:`. A sample "
          "passes if the first 4 lines it writes contain the known fix (a fixed substring "
          "such as `.clone()` or `let mut x`).",
          "- Each seed gives the same output every time, so results are reproducible with "
          "`python3 tools/task_eval.py --root <model folder> --run <run> --seeds "
          f"{k} --out <file>`.", "",
          "## Limitations", "",
          f"- Small task set ({n_complete + n_fix} tasks), all short and simple. The intervals "
          "above show how uncertain the rates are.",
          "- The fix check is a substring match: a correct fix worded differently counts as "
          "wrong, and a match does not prove understanding.",
          "- Passing tests does not measure code quality or style.",
          "- Perplexity is not compared here: each stage's validation loss was measured on "
          "its own data mix, so those numbers are not comparable.", ""]

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
