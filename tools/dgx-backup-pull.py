"""Windows scheduled task: download and verify new DGX recovery archives.

Uses the owner's existing private DGX login file; never mirrors deletions.
Requires paramiko. Run via uv run --with paramiko python this-file.py.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

import paramiko

sys.path.insert(0, str(Path("D:/Projects/gr-advisory/scripts/dgx")))
from creds import login

DEST = Path("D:/ModelBackups/FLAME/rust-30m-dgx-20261102")
REMOTE = "/home/kenpath/flame-recovery/rust-30m-dgx-20261102"


def download(client, remote, target):
    temp = target.with_suffix(target.suffix + ".part")
    _, out, err = client.exec_command("cat -- " + shlex.quote(remote), timeout=1800)
    with temp.open("wb") as f:
        while data := out.read(4 * 1024 * 1024):
            f.write(data)
    if out.channel.recv_exit_status() != 0:
        raise RuntimeError(err.read().decode("utf-8", "replace")[-500:])
    temp.replace(target)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    host, user, password = login()
    client = None
    for address in dict.fromkeys([host, "100.122.210.21", "192.168.88.69"]):
        try:
            candidate = paramiko.SSHClient()
            candidate.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            candidate.connect(address, username=user, password=password, timeout=10,
                              look_for_keys=False, allow_agent=False)
            client = candidate
            break
        except (OSError, paramiko.SSHException):
            candidate.close()
    if client is None:
        raise RuntimeError("DGX not reachable on configured host, Tailscale, or LAN")
    try:
        _, out, err = client.exec_command(
            f"find {shlex.quote(REMOTE)} -maxdepth 1 -type f -name '*.tar.gz' -printf '%f\\n'", timeout=30)
        names = sorted(out.read().decode().splitlines())
        if out.channel.recv_exit_status():
            raise RuntimeError(err.read().decode()[-500:])
        for name in names:
            if Path(name).name != name or not name.startswith("rust-30m-dgx-20261102-"):
                raise RuntimeError("Unexpected backup name")
            target = DEST / name
            checksum = DEST / (name + ".sha256")
            if (DEST / (name + ".verified.json")).exists() and target.exists():
                continue
            download(client, f"{REMOTE}/{name}.sha256", checksum)
            expected = checksum.read_text(encoding="utf-8").split()[0]
            if not target.exists():
                if shutil.disk_usage(DEST).free < 12 * 1024**3:
                    raise RuntimeError("Less than 12 GiB free: retaining old backups, no pruning")
                download(client, f"{REMOTE}/{name}", target)
            with target.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch: {target}")
            (DEST / (name + ".verified.json")).write_text(
                json.dumps({"sha256": actual, "bytes": target.stat().st_size, "verified_at": time.time()}),
                encoding="utf-8")
            print("Downloaded and SHA256 verified:", target, flush=True)
        (DEST / "last-sync.json").write_text(json.dumps({"time": time.time(), "archives": names}), encoding="utf-8")
    finally:
        client.close()


if __name__ == "__main__":
    main()
