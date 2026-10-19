"""Test every dataset, one at a time: 1 recording to the destination, then check it.

For each dataset in catalog order:
  1. dry run: the size of 1 recording; above --max-mb, or sizes not reported: deferred
  2. download 1 recording to --dest
  3. `check --latest --checksums`: read the files back and compare SHA-256
  4. record the result in the checklist, also stored at <dest>/_checklist.md

Datasets that need a person (credentials, download by hand, restricted) and datasets whose
source holds no recordings are recorded and the run goes on. Any other failure stops the
run, so that it can be debugged. Run again to continue: recorded datasets are not run again.

    python scripts/check_each.py --dest s3://bucket/_test --max-mb 500
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

from biosignal_data.config import load_config
from biosignal_data.storage.stores import S3Settings, Store, open_store

UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
WOULD = re.compile(r"^  would copy\s+.*?, ([\d,.]+) (B|KB|MB|GB|TB)( \((\d+) sizes unknown\))?$")
CHECKLIST_KEY = "_checklist.md"


def cli(*args: str, timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["biosignal-data", *args], check=False, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return -1, f"{out}\nTIMEOUT after {timeout} s"


def open_dest(dest: str) -> Store:
    if not dest.startswith("s3://"):
        return open_store(dest)
    config, _ = load_config()
    if config.s3 is None:
        raise SystemExit("the config has no s3 section")
    s3 = config.s3
    return open_store(dest, S3Settings.from_env(s3.endpoint, s3.region, s3.addressing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--max-mb", type=float, required=True)
    args = parser.parse_args()
    limit = args.max_mb * 1e6
    store = open_dest(args.dest)
    lines = (store.get_bytes(CHECKLIST_KEY) or b"").decode().splitlines()
    done = {m.group(1) for line in lines if (m := re.search(r"`([^`]+)`", line))}
    _, listing = cli("list", timeout=120)
    ids = [line.split()[0] for line in listing.splitlines() if "/" in line.split(" ", 1)[0]]
    print(f"{len(ids)} datasets, {len(done)} already in the checklist", flush=True)

    def record(line: str) -> None:
        lines.append(line)
        store.put_bytes(CHECKLIST_KEY, ("\n".join(lines) + "\n").encode(), "text/markdown")
        print(line, flush=True)

    def stop(dataset_id: str, step: str, code: int, out: str) -> int:
        record(f"- [ ] `{dataset_id}` FAILED in {step} (exit {code})")
        print(f"----- output of {step} for {dataset_id} -----\n{out}\n-----", flush=True)
        return 1

    for dataset_id in ids:
        if dataset_id in done:
            continue
        start = time.time()
        code, out = cli("download", dataset_id, "-n", "1", "--dry-run", timeout=900)
        skip = re.search(rf"^[!✗] {re.escape(dataset_id)}: (.*)$", out, re.M)
        if skip and skip.group(0).startswith("!"):
            record(f"- [ ] `{dataset_id}` needs action: {skip.group(1)[:200]}")
            continue
        would = next((m for line in out.splitlines() if (m := WOULD.match(line))), None)
        if code != 0 or would is None:
            if "no recordings" in out:
                record(f"- [ ] `{dataset_id}` no recordings at the source (only metadata)")
                continue
            return stop(dataset_id, "dry run", code, out)
        size = float(would.group(1).replace(",", "")) * UNITS[would.group(2)]
        if would.group(4):
            record(f"- [ ] `{dataset_id}` deferred: {would.group(4)} file sizes unknown")
            continue
        if size > limit:
            record(f"- [ ] `{dataset_id}` deferred: 1 recording = {size / 1e9:.2f} GB")
            continue
        code, out = cli("download", dataset_id, "-n", "1", "--dest", args.dest, timeout=7200)
        if "no recordings found" in out:
            record(f"- [ ] `{dataset_id}` no recordings at the source (only metadata)")
            continue
        if code != 0:
            return stop(dataset_id, "download", code, out)
        code, out = cli(
            "check", dataset_id, "--latest", "--checksums", "--dest", args.dest, timeout=3600
        )
        if code != 0:
            return stop(dataset_id, "check", code, out)
        record(
            f"- [x] `{dataset_id}` ok: {size / 1e6:.1f} MB copied and verified "
            f"({time.time() - start:.0f} s)"
        )
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
