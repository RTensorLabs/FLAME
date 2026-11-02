"""DGX two-stage runner, append-only recovery archives, and GitHub publisher.

Run from the FLAME repository. Credentials stay outside the repository.
The runner never deletes checkpoints, release assets, or old recovery archives.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "rust-30m-dgx-20261102"
RECOVERY = ROOT.parent.parent / "flame-recovery" / RUN_ID
REPO = "RTensorLabs/FLAME"
TAG = f"{RUN_ID}-checkpoints"
TOKEN = Path.home() / ".config/flame/github-token"


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def command(*args, capture=False):
    print("+", *map(str, args), flush=True)
    return subprocess.run(list(map(str, args)), cwd=ROOT, check=True,
                          text=True, capture_output=capture)


def archive_directory(folder, name):
    dest = RECOVERY / (name + ".tar.gz")
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
            tf.add(folder, arcname=".")
        tmp.replace(dest)
    dest.with_suffix(dest.suffix + ".sha256").write_text(
        f"{digest(dest)}  {dest.name}\n", encoding="utf-8")
    dest.chmod(0o444)
    return dest


def snapshot(stage):
    """Only called after the trainer has exited: no checkpoint rotation race."""
    run = ROOT / "runs" / f"{RUN_ID}-{stage}"
    state = json.loads((run / "checkpoints/latest/state.json").read_text())
    name = f"{RUN_ID}-{stage}-step-{state['step']:07d}"
    dest = RECOVERY / "snapshots" / name
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run, tmp / "runs" / run.name, dirs_exist_ok=True)
        tok = ROOT / "data/tokenizers/rust-en-8k.json"
        (tmp / tok.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tok, tmp / tok.relative_to(ROOT))
        for source in ("configs", "tools", "crates"):
            shutil.copytree(ROOT / source, tmp / source, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for source in ("Cargo.toml", "Cargo.lock", "LICENSE"):
            shutil.copy2(ROOT / source, tmp / source)
        for source in ("prepared-manifest.json", "source-commit.txt", "validation.txt", "verification.json"):
            if (ROOT / "ops" / source).exists():
                shutil.copy2(ROOT / "ops" / source, tmp / source)
        model = tmp / "runs" / run.name / "checkpoints/latest/model.safetensors"
        import struct
        with model.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        params = sum(__import__("math").prod(t["shape"]) for k, t in header.items()
                     if k != "__metadata__")
        atomic_json(tmp / "MODEL_CARD.json", {
            "run_id": RUN_ID, "stage": stage, "parameters": params,
            "status": "trained" if stage == "stage2" and state["step"] == 2500 else "training",
            "checkpoint": state, "format": "FLAME Candle safetensors; fp32",
            "backend": json.loads((run / "checkpoints/latest/backend.json").read_text())
                if (run / "checkpoints/latest/backend.json").exists() else {"backend": "native-candle-fp32"},
            "tokenizer": "data/tokenizers/rust-en-8k.json",
            "source_commit": (ROOT / "ops/source-commit.txt").read_text().strip(),
        })
        files = sorted(p for p in tmp.rglob("*") if p.is_file())
        (tmp / "SHA256SUMS").write_text("".join(
            f"{digest(p)}  {p.relative_to(tmp).as_posix()}\n" for p in files), encoding="utf-8")
        tmp.rename(dest)
    archive = archive_directory(dest, name)
    atomic_json(ROOT / "ops/latest-backup.json", {
        "archive": str(archive), "sha256": digest(archive), "stage": stage,
        "step": state["step"], "timestamp": time.time(),
    })
    print("Recovery snapshot:", archive, flush=True)
    return archive


def prepare():
    if (ROOT / "ops/prepared-manifest.json").exists():
        return
    command("target/release/flame", "tokenizer", "byte", "--out", "data/tokenizers/byte.json")
    command("target/release/flame", "prepare", "--input", "examples/overfit.txt",
            "--tokenizer", "data/tokenizers/byte.json", "--out", "data/overfit", "--val-fraction", "0.25")
    command("target/release/flame", "train", "--config", "configs/overfit.toml", "--run", "runs/dgx-overfit-gate")
    gate = json.loads((ROOT / "runs/dgx-overfit-gate/checkpoints/latest/state.json").read_text())
    if gate["best_val_loss"] is None or gate["best_val_loss"] > 0.15:
        raise RuntimeError(f"GPU overfit gate failed: {gate['best_val_loss']}")
    cargo = Path(os.environ["CARGO_HOME"]) / "registry/src"
    tok = "data/tokenizers/rust-en-8k.json"
    if not (ROOT / tok).exists():
        command("target/release/flame", "tokenizer", "train", "--input", "data/raw/rust-code",
                cargo, "data/raw/rust-docs", "data/raw/rustc-errors", "data/raw/english",
                "--ext", "rs,md,txt", "--vocab-size", "8192", "--out", tok, "--sample-mb", "250")
    for name, ext, inputs in (
        ("rust-code", "rs", ["data/raw/rust-code", cargo]),
        ("rust-docs", "md", ["data/raw/rust-docs"]),
        ("rustc-errors", "txt", ["data/raw/rustc-errors"]),
        ("english", "txt", ["data/raw/english"]),
    ):
        if not (ROOT / "data" / name / "meta.json").exists():
            command("target/release/flame", "prepare", "--input", *inputs, "--ext", ext,
                    "--tokenizer", tok, "--out", f"data/{name}", "--val-fraction",
                    "0.05" if name == "rustc-errors" else "0.02")
    manifest = {}
    for name in ("rust-code", "rust-docs", "rustc-errors", "english"):
        meta = json.loads((ROOT / "data" / name / "meta.json").read_text())
        assert meta["train_tokens"] > 1025 and meta["val_tokens"] > 1025, meta
        manifest[name] = {"meta": meta, "files": {
            part: digest(ROOT / "data" / name / part) for part in ("train.bin", "val.bin")}}
    # Independent, private, prepared corpus copy; never publish downloaded training data.
    for name in ("rust-code", "rust-docs", "rustc-errors", "english", "tokenizers"):
        shutil.copytree(ROOT / "data" / name, RECOVERY / "prepared-data" / name, dirs_exist_ok=True)
    atomic_json(ROOT / "ops/prepared-manifest.json", manifest)
    shutil.copy2(ROOT / "ops/prepared-manifest.json", RECOVERY / "prepared-manifest.json")


def run():
    os.chdir(ROOT)
    (ROOT / "ops/source-commit.txt").write_text(
        command("git", "rev-parse", "HEAD", capture=True).stdout, encoding="utf-8")
    prepare()
    for stage in ("stage1", "stage2"):
        config = f"configs/dgx-30m-{stage}.toml"
        cfg = tomllib.loads((ROOT / config).read_text())
        run = ROOT / "runs" / f"{RUN_ID}-{stage}"
        if (run / "PAUSE").exists() or (ROOT / "ops/PAUSE").exists():
            print("Operator pause is set; refusing to restart training", flush=True)
            return
        first = not (run / "checkpoints/latest/state.json").exists()
        while True:
            statefile = run / "checkpoints/latest/state.json"
            step = json.loads(statefile.read_text())["step"] if statefile.exists() else 0
            if step >= cfg["train"]["max_steps"]:
                snapshot(stage)
                break
            if (ROOT / "ops/PAUSE").exists():
                return
            if shutil.disk_usage(ROOT).free < 50 * 1024**3:
                raise RuntimeError("Less than 50 GiB free; stopped without deleting backups")
            started = time.time()
            args = [".venv/bin/python", "tools/torch_train.py", "--config", config, "--run", str(run)]
            args += ["--steps", "1"] if first else ["--max-minutes", "30"]
            atomic_json(ROOT / "ops/status.json", {"status": "training", "stage": stage,
                        "last_saved_step": step, "session_started": started})
            command(*args)
            snapshot(stage)
            first = False
            sessions = (run / "sessions.log").read_text()
            if "PAUSE file" in sessions.splitlines()[-1]:
                (ROOT / "ops/PAUSE").touch()
                return
    final_run = f"runs/{RUN_ID}-stage2"
    evaluation = command("target/release/flame", "eval", "--run", final_run, "--batches", "50", capture=True)
    sample = command("target/release/flame", "generate", "--run", final_run,
                     "--prompt", "fn main() {", "--max-tokens", "160", capture=True)
    (ROOT / "ops/validation.txt").write_text(evaluation.stdout + "\n" + sample.stdout, encoding="utf-8")
    # Include evaluation in a distinct final package instead of modifying an old archive.
    final_dest = RECOVERY / "final-package"
    if not final_dest.exists():
        final = RECOVERY / "final-package.tmp"
        final.mkdir(exist_ok=True)
        for stage in ("stage1", "stage2"):
            shutil.copytree(ROOT / "runs" / f"{RUN_ID}-{stage}", final / "runs" / f"{RUN_ID}-{stage}", dirs_exist_ok=True)
        shutil.copytree(ROOT / "data/tokenizers", final / "data/tokenizers", dirs_exist_ok=True)
        for source in ("configs", "crates", "tools"):
            shutil.copytree(ROOT / source, final / source, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for source in ("Cargo.toml", "Cargo.lock", "LICENSE"):
            shutil.copy2(ROOT / source, final / source)
        shutil.copy2(ROOT / "ops/validation.txt", final / "VALIDATION.txt")
        shutil.copy2(ROOT / "ops/prepared-manifest.json", final / "prepared-manifest.json")
        shutil.copy2(ROOT / "ops/source-commit.txt", final / "source-commit.txt")
        shutil.copy2(ROOT / "docs/dgx-training.md", final / "README.md")
        last_snapshot = RECOVERY / "snapshots" / f"{RUN_ID}-stage2-step-0002500"
        shutil.copy2(last_snapshot / "MODEL_CARD.json", final / "MODEL_CARD.json")
        (final / "SHA256SUMS").write_text("".join(
            f"{digest(p)}  {p.relative_to(final).as_posix()}\n"
            for p in sorted(final.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"), encoding="utf-8")
        final.rename(final_dest)
    package = archive_directory(final_dest, f"{RUN_ID}-complete-model")
    atomic_json(ROOT / "ops/status.json", {"status": "training_complete", "final_archive": str(package)})
    (RECOVERY / "TRAINING_COMPLETE").touch()


def api_session():
    import requests
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + TOKEN.read_text().strip(),
                            "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28"})
    return session


def release(session, tag, final=False):
    url = f"https://api.github.com/repos/{REPO}/releases"
    r = session.get(f"{url}/tags/{tag}", timeout=60)
    if r.status_code == 404:
        commit = (ROOT / "ops/source-commit.txt").read_text().strip()
        r = session.post(url, json={"tag_name": tag, "target_commitish": commit,
            "name": f"FLAME 30M DGX {'complete model' if final else 'recovery checkpoints'}",
            "prerelease": not final, "make_latest": "false",
            "body": "Two-stage Rust + English base model. " +
            ("Training complete; see VALIDATION.txt in the model package. " if final else
             "Training in progress. These are resumable checkpoints, not a finished model. ") +
            "Archives contain weights, optimizer state, tokenizer, configuration, source, and SHA256 checksums. "
            "Downloaded training corpora are excluded. Recovery instructions: docs/dgx-training.md. "
            "Existing assets are never overwritten or pruned."}, timeout=60)
    r.raise_for_status()
    return r.json()


def upload(session, rel, path):
    size = path.stat().st_size
    if size >= 2 * 1024**3:
        raise RuntimeError(f"Asset exceeds GitHub 2 GiB limit: {path}")
    expected = digest(path)
    url = f"https://api.github.com/repos/{REPO}/releases/{rel['id']}/assets"
    assets = []
    page = 1
    while True:
        r = session.get(url, params={"per_page": 100, "page": page}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    existing = next((x for x in assets if x["name"] == path.name), None)
    if existing:
        if existing["size"] != size or existing.get("digest") != "sha256:" + expected:
            raise RuntimeError(f"Existing asset fails checksum verification: {path.name}; refusing to overwrite")
        return existing
    with path.open("rb") as f:
        r = session.post(rel["upload_url"].split("{")[0], params={"name": path.name},
                         data=f, headers={"Content-Type": "application/octet-stream"}, timeout=(60, 1800))
    r.raise_for_status()
    result = r.json()
    if result["size"] != size or result.get("digest") != "sha256:" + expected:
        raise RuntimeError(f"Uploaded asset fails GitHub digest verification: {path.name}")
    print("Published and SHA256 verified:", result["browser_download_url"], flush=True)
    return result


def publish():
    if (ROOT / "ops/published.json").exists():
        return
    session = api_session()
    for path in sorted(RECOVERY.glob("*.tar.gz")):
        marker = path.with_suffix(path.suffix + ".github.json")
        if marker.exists():
            continue
        final = path.name.endswith("-complete-model.tar.gz")
        rel = release(session, RUN_ID if final else TAG, final)
        asset = upload(session, rel, path)
        upload(session, rel, path.with_suffix(path.suffix + ".sha256"))
        atomic_json(marker, {"asset": asset, "verified_at": time.time()})
        if final:
            atomic_json(ROOT / "ops/published.json", {"url": rel["html_url"], "asset": asset["browser_download_url"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "publish"))
    args = parser.parse_args()
    os.umask(0o077)
    RECOVERY.mkdir(parents=True, exist_ok=True)
    (ROOT / "ops").mkdir(exist_ok=True)
    with (ROOT / "ops" / f"{args.mode}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run() if args.mode == "run" else publish()
