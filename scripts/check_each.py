"""Test downloads, one dataset at a time, in one process.

For each dataset in catalog order:
  1. plan it against --dest: the version and file list, read once from the source
  2. bigger than --max-mb, or the source does not report sizes: deferred, nothing copied
  3. apply the plan: the files are copied in parallel (download.workers at a time)
  4. check the copy: every file is read back from --dest and its SHA-256 compared
  5. no file that can hold recordings (only metadata): recorded as such, not as ok

Datasets that need a person are left out before anything runs: downloads by hand,
unavailable ones, and those whose access tier (in docs/datasets.yaml) is not open. The
checklist is kept at <dest>/_checklist.md. The run stops at the first failure, so that it
can be debugged; run it again to continue (failed datasets run again).

    python scripts/check_each.py --dest s3://bucket/_test --max-mb 500
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from collections.abc import Callable

from biosignal_data.catalog import AccessTier, Catalog, Dataset, Infos
from biosignal_data.config import load_config, need
from biosignal_data.console import setup_logging
from biosignal_data.fetchers import FetchError, FetchOptions
from biosignal_data.ingest import AlreadyComplete, apply, planned, verify
from biosignal_data.read.raw import complete_versions
from biosignal_data.storage.stores import S3Settings, Store, open_store

CHECKLIST_KEY = "_checklist.md"
ROOT = "raw"


def test_one(
    dataset: Dataset,
    store: Store,
    options: FetchOptions,
    workers: int,
    limit: float,
    record: Callable[[str], None],
) -> bool:
    """Plan, copy and check one dataset; False at a failure (details printed)."""
    start = time.monotonic()
    try:
        with planned(dataset, store, options=options, root=ROOT) as plan:
            if plan.unknown_sizes:
                record(f"- [ ] `{dataset.id}` deferred: {plan.unknown_sizes} file sizes unknown")
                return True
            if plan.total_bytes > limit:
                record(f"- [ ] `{dataset.id}` deferred: {plan.total_bytes / 1e9:.2f} GB")
                return True
            report = apply(plan, store, workers=workers)
    except AlreadyComplete:
        version = complete_versions(store, dataset.id, ROOT)[-1]
        problems = verify(store, dataset.id, version, ROOT, deep=True, workers=workers)
        if problems:
            record(f"- [ ] `{dataset.id}` FAILED check of the stored version {version}")
            print("\n".join(f"  {p}: {e}" for p, e in sorted(problems.items())), flush=True)
            return False
        record(f"- [x] `{dataset.id}` ok: already complete ({version}), checked")
        return True
    except FetchError as exc:  # access, embargo, source refused: recorded, go on
        record(f"- [ ] `{dataset.id}` needs action: {str(exc)[:200]}")
        return True
    except Exception:  # noqa: BLE001 - stop here so that it can be debugged
        record(f"- [ ] `{dataset.id}` FAILED while copying")
        print(traceback.format_exc(), flush=True)
        return False
    if report.failed:
        record(f"- [ ] `{dataset.id}` FAILED: {len(report.failed)} files failed")
        print("\n".join(f"  {p}: {e}" for p, e in report.failed.items()), flush=True)
        return False
    copied = time.monotonic() - start
    problems = verify(store, dataset.id, report.version, ROOT, deep=True, workers=workers)
    if problems:
        record(f"- [ ] `{dataset.id}` FAILED check: {len(problems)} problems")
        print("\n".join(f"  {p}: {e}" for p, e in sorted(problems.items())), flush=True)
        return False
    if not report.recording_files:
        record(f"- [ ] `{dataset.id}` no recordings: copied and checked, but only metadata")
        return True
    mb, checked = report.bytes_copied / 1e6, time.monotonic() - start - copied
    record(
        f"- [x] `{dataset.id}` ok: {report.files} files, {mb:.1f} MB in {copied:.0f} s "
        f"({mb / max(copied, 0.001):.1f} MB/s), checked in {checked:.0f} s"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--max-mb", type=float, required=True)
    args = parser.parse_args()
    limit = args.max_mb * 1e6
    setup_logging(verbose=False)

    config, config_path = load_config()
    settings = need(config.download, "download", config_path)
    options = FetchOptions(
        staging_dir=config.staging.dir if config.staging else None,
        keep_staging=config.staging.keep if config.staging else None,
        connections=settings.workers,
    )
    store: Store
    if args.dest.startswith("s3://"):
        s3 = need(config.s3, "s3", config_path)
        s3_settings = S3Settings.from_env(s3.endpoint, s3.region, s3.addressing)
        store = open_store(args.dest, s3_settings, settings.workers)
    else:
        store = open_store(args.dest)

    # Failures and datasets that needed a person are dropped from the checklist, so that
    # they run again: after a fix, or once there is access.
    lines = [
        line
        for line in (store.get_bytes(CHECKLIST_KEY) or b"").decode().splitlines()
        if "needs action" not in line and "FAILED" not in line
    ]
    done = {m.group(1) for line in lines if (m := re.search(r"`([^`]+)`", line))}
    catalog, infos = Catalog.load(), Infos.load()
    person = {
        d.id
        for d in catalog
        if d.source.type in ("manual", "unavailable")
        or infos.get(d.id).access.tier != AccessTier.OPEN
    }
    todo = [d for d in catalog if d.id not in person and d.id not in done]
    print(
        f"{len(catalog)} datasets: {len(person)} need a person and are left out; "
        f"{len(todo)} to test, {len(catalog) - len(person) - len(todo)} already in the checklist",
        flush=True,
    )

    def record(line: str) -> None:
        lines.append(line)
        store.put_bytes(CHECKLIST_KEY, ("\n".join(lines) + "\n").encode(), "text/markdown")
        print(line, flush=True)

    for dataset in todo:
        if not test_one(dataset, store, options, settings.workers, limit, record):
            return 1
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
