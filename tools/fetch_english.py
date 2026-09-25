"""Downloads general English text for pre-training: FineWeb-Edu (educational web pages,
licence ODC-By), from its public "sample-10BT" subset on Hugging Face. No account needed.

    pip install pyarrow requests
    python tools/fetch_english.py --out data/raw/english/fineweb-edu.txt --target-mb 1500

The parquet files (~2 GB each) are downloaded one at a time, their `text` column is written
out as plain text (documents separated by <|endoftext|>, the separator `flame prepare`
expects), and each parquet file is deleted once used. Stops when --target-mb of text has
been written. Re-running continues where it stopped (already-written text is kept).
"""
import argparse
import os
import sys

import pyarrow.parquet as pq
import requests

REPO = "HuggingFaceFW/fineweb-edu"
SUBDIR = "sample/10BT"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{SUBDIR}"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"
SEP = "\n<|endoftext|>\n"


def list_files():
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    return sorted(e["path"] for e in r.json() if e["path"].endswith(".parquet"))


def download(url, dest):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest + ".part", "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  downloading {os.path.basename(dest)}: {100 * done / total:5.1f}%", end="", flush=True)
    os.replace(dest + ".part", dest)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/english/fineweb-edu.txt")
    ap.add_argument("--target-mb", type=float, default=1500, help="MB of text to produce (1 MB ≈ 250k tokens)")
    ap.add_argument("--min-chars", type=int, default=200, help="skip very short pages")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    target = int(a.target_mb * 1e6)
    written = os.path.getsize(a.out) if os.path.exists(a.out) else 0
    progress_file = a.out + ".files_done"
    done_files = set(open(progress_file).read().split()) if os.path.exists(progress_file) else set()
    if written >= target:
        print(f"already have {written / 1e6:.0f} MB in {a.out}")
        return
    files = list_files()
    print(f"{len(files)} parquet files available; target {target / 1e6:.0f} MB, have {written / 1e6:.0f} MB")
    with open(a.out, "a", encoding="utf-8") as out:
        for path in files:
            if written >= target:
                break
            if path in done_files:
                continue
            local = os.path.join(os.path.dirname(a.out), os.path.basename(path))
            if not os.path.exists(local):
                download(FILE_URL.format(path=path), local)
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(columns=["text"], batch_size=2048):
                for text in batch.column(0).to_pylist():
                    if text and len(text) >= a.min_chars:
                        out.write(text.strip())
                        out.write(SEP)
                        written += len(text.encode("utf-8")) + len(SEP)
                if written >= target:
                    break
            out.flush()
            os.remove(local)
            with open(progress_file, "a") as p:
                p.write(path + "\n")
            print(f"  {written / 1e6:.0f} / {target / 1e6:.0f} MB")
    print(f"done: {written / 1e6:.0f} MB of English in {a.out}")


if __name__ == "__main__":
    sys.exit(main())
