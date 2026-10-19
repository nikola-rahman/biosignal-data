"""Make the standard version of one raw version, in two explicit steps.

    plan = plan_standard(store, "physionet/mitdb", "1.0.0")   # recordings, from the file list
    plan.units, plan.to_convert, plan.discovery.problems       # all known before any work
    report = convert(plan, store, workers=4, work_dir=...)     # converts exactly plan.to_convert

Guarantees:
- Every recording is converted from raw files that are checked against the SHA-256 of the
  raw manifest, and every shard written is read back and compared with the source.
- A run can be stopped and run again: each recording finished is recorded under `_parts/`
  (its rows and a `done.json` written last), and is not converted again.
- The tables, `_manifest.json` and then `_SUCCESS` are written only when every recording
  is done. Every raw file is then either used or listed as not converted, with the reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import shutil
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from biosignal_data import __version__
from biosignal_data.read.raw import RawDataset
from biosignal_data.standard import arrays, tables
from biosignal_data.standard.discover import Discovery, Unit, discover, may_be_annotations
from biosignal_data.standard.layout import DONE, TABLES, StandardLocation
from biosignal_data.standard.sources import (
    EdfSource,
    Source,
    UnsupportedError,
    WfdbSource,
    wfdb_annotations,
)
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import Manifest, StoppedError, utcnow
from biosignal_data.storage.stores import Store
from biosignal_data.text import plural

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


class AlreadyStandard(RuntimeError):  # noqa: N818 - reads naturally at the call site
    pass


class NotConverted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str


class StandardManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    dataset_id: str
    raw_version: str
    raw_manifest_sha256: str
    license: str | None
    tool: str
    created_at: datetime
    completed_at: datetime
    layout: dict[str, Any]
    recordings: int
    channels: int
    events: int
    files_used: int
    problems: list[str]  # e.g. RECORDS entries without files
    not_converted: list[NotConverted]


class StandardProgress(Protocol):
    """`unit_started` and `unit_done` are called from worker threads."""

    def start(self, units: int, done: int) -> None: ...
    def unit_started(self, unit_id: str) -> None: ...
    def unit_done(self, unit_id: str, ok: bool) -> None: ...


@dataclass
class StandardPlan:
    raw: RawDataset
    raw_manifest_sha256: str
    location: StandardLocation
    discovery: Discovery
    done: set[str]  # units finished by an earlier run
    replace: bool = False  # the version is complete and is made again

    @property
    def units(self) -> list[Unit]:
        return self.discovery.units

    @property
    def to_convert(self) -> list[Unit]:
        return [u for u in self.units if u.id not in self.done]


@dataclass
class StandardReport:
    dataset_id: str
    version: str
    location: str
    units: int
    converted: int = 0
    skipped: int = 0  # done by an earlier run
    failed: dict[str, str] = field(default_factory=dict)
    not_converted: int = 0  # raw files, known when complete
    complete: bool = False


def plan_standard(
    store: Store, dataset_id: str, version: str, *, raw_root: str = "raw", replace: bool = False
) -> StandardPlan:
    """Find the recordings of a complete raw version, and those an earlier run finished.
    `replace`: a complete version is made again (by `convert`, which first removes it)
    instead of raising `AlreadyStandard`."""
    raw_location = VersionLocation(dataset_id, version, raw_root)
    data = store.get_bytes(raw_location.manifest_key)
    if data is None or not store.exists(raw_location.success_key):
        raise FileNotFoundError(f"{store.uri}/{raw_location.prefix} is not a complete version")
    location = StandardLocation(dataset_id, version)
    complete = store.exists(location.success_key)
    if complete and not replace:
        raise AlreadyStandard(
            f"{store.uri}/{location.prefix} is already complete; use --replace to make it again"
        )
    raw = RawDataset(store, raw_location, Manifest.from_json(data), Path())

    def read_text(path: str) -> str:
        with store.open(raw_location.data_key(path)) as f:
            return f.read().decode("utf-8", errors="replace")

    discovery = discover((f.path for f in raw.files), read_text)
    done = (
        set()
        if complete
        else {
            key[len(location.parts_prefix) : -len("/" + DONE)]
            for key in store.list(location.parts_prefix)
            if key.endswith("/" + DONE)
        }
    )
    plan = StandardPlan(
        raw, hashlib.sha256(data).hexdigest(), location, discovery, done, replace=complete
    )
    log.info(
        "%s %s: %s, %s to convert",
        dataset_id,
        version,
        plural(len(plan.units), "recording"),
        plural(len(plan.to_convert), "recording"),
    )
    for problem in discovery.problems:
        log.warning("%s: %s", dataset_id, problem)
    return plan


def convert(
    plan: StandardPlan,
    store: Store,
    *,
    workers: int,
    work_dir: Path,
    progress: StandardProgress | None = None,
    stop: threading.Event | None = None,
) -> StandardReport:
    """Convert exactly `plan.to_convert`, `workers` at a time; complete the version when
    every recording is done. Raw files are fetched to `work_dir`, one folder per
    recording, removed when it is done."""
    report = StandardReport(
        plan.raw.location.dataset_id,
        plan.raw.location.version,
        f"{store.uri}/{plan.location.prefix}",
        units=len(plan.units),
        skipped=len(plan.done),
    )
    if plan.replace:  # made again from the raw version: the old one goes first
        log.warning("%s: removing the complete version to make it again", report.location)
        store.delete_many(list(store.list(plan.location.prefix + "/")))
    if progress:
        progress.start(len(plan.units), len(plan.done))
    work_dir.mkdir(parents=True, exist_ok=True)

    def run(unit: Unit) -> None:
        if stop is not None and stop.is_set():
            raise StoppedError("stopped")
        if progress:
            progress.unit_started(unit.id)
        convert_unit(plan, store, unit, work_dir)

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(run, u): u for u in plan.to_convert}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - recorded per recording, reported at the end
                log.error("%s: %s", unit.id, exc)
                report.failed[unit.id] = f"{type(exc).__name__}: {exc}"
                if progress:
                    progress.unit_done(unit.id, ok=False)
                continue
            report.converted += 1
            if progress:
                progress.unit_done(unit.id, ok=True)
    finally:
        # On Ctrl+C: recordings not started yet are dropped; those running finish.
        pool.shutdown(cancel_futures=True)
    if report.failed:
        log.error("%s failed; run again to retry them", plural(len(report.failed), "recording"))
        return report
    manifest = finish(plan, store, workers)
    report.not_converted = len(manifest.not_converted)
    report.complete = True
    return report


def convert_unit(plan: StandardPlan, store: Store, unit: Unit, work_dir: Path) -> None:
    """Convert one recording and record it under `_parts/`."""
    folder = Path(tempfile.mkdtemp(prefix="unit-", dir=work_dir))
    try:
        raw = plan.raw

        def fetch(path: str) -> Path:
            return raw.fetch_to(path, folder)

        location = plan.location
        try:
            source: Source = (
                WfdbSource(unit.path, fetch, raw.has)
                if unit.format == "wfdb"
                else EdfSource(unit.path, fetch)
            )
        except UnsupportedError as exc:
            main = unit.path + ".hea" if unit.format == "wfdb" else unit.path
            skipped = [NotConverted(path=p, reason=str(exc)) for p in (main, *unit.annotations)]
            _record(store, location, unit, used=[], not_converted=skipped, parts={})
            return
        try:
            events = list(source.events)
            used, skipped = list(source.files), []
            for path in unit.annotations:
                try:
                    events += wfdb_annotations(path, fetch, source.frame_rate)
                    used.append(path)
                except Exception as exc:
                    if unit.listed_annotators:
                        raise
                    reason = f"tried as a WFDB annotation file: {type(exc).__name__}: {exc}"
                    skipped.append(NotConverted(path=path, reason=reason))
            signals = None
            if source.groups:
                key = location.signals_key(unit.id)
                attributes = {
                    "recording_id": unit.id,
                    "dataset_id": location.dataset_id,
                    "raw_version": location.version,
                    "format": source.format,
                }
                arrays.write_signals(source, store.zarr_store(key, read_only=False), attributes)
                signals = key[len(location.prefix) + 1 :]
            files = {p: raw.entry(p).sha256 for p in used}
            parts = {
                "recordings": tables.recording_table(unit.id, source, signals, files),
                "channels": tables.channel_table(unit.id, source),
                "events": tables.event_table(unit.id, events),
            }
            _record(
                store,
                location,
                unit,
                used=used,
                not_converted=skipped,
                parts=parts,
                problems=source.problems,
            )
        finally:
            source.close()
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _record(
    store: Store,
    location: StandardLocation,
    unit: Unit,
    *,
    used: list[str],
    not_converted: list[NotConverted],
    parts: dict[str, pa.Table],
    problems: list[str] | None = None,
) -> None:
    for name, table in parts.items():
        store.put_bytes(
            location.part_key(unit.id, f"{name}.parquet"),
            tables.to_parquet(table),
            "application/vnd.apache.parquet",
        )
    done = {
        "unit": unit.id,
        "tables": sorted(parts),
        "used": used,
        "not_converted": [n.model_dump() for n in not_converted],
        "problems": problems or [],
    }
    store.put_bytes(location.part_key(unit.id, DONE), json.dumps(done).encode())


def finish(plan: StandardPlan, store: Store, workers: int) -> StandardManifest:
    """Join the parts into the tables, write the manifest and `_SUCCESS`, remove `_parts/`."""
    location = plan.location
    created = utcnow()

    def load(unit: Unit) -> tuple[dict[str, Any], dict[str, pa.Table]]:
        data = store.get_bytes(location.part_key(unit.id, DONE))
        if data is None:
            raise RuntimeError(f"{unit.id} is not done")
        done = json.loads(data)
        parts = {}
        for name in done["tables"]:
            blob = store.get_bytes(location.part_key(unit.id, f"{name}.parquet"))
            if blob is None:
                raise RuntimeError(f"{unit.id}: the {name} part is missing")
            parts[name] = tables.from_parquet(blob, name)
        return done, parts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load, plan.units))
    joined = {
        name: pa.concat_tables(
            [parts[name] for _, parts in loaded if name in parts],
            promote_options="none",
        )
        if any(name in parts for _, parts in loaded)
        else tables.SCHEMAS[name].empty_table()
        for name in TABLES
    }
    for name, table in joined.items():
        store.put_bytes(
            location.table_key(name), tables.to_parquet(table), "application/vnd.apache.parquet"
        )

    used = set(plan.discovery.lists) | {p for done, _ in loaded for p in done["used"]}
    skipped = {n["path"]: n["reason"] for done, _ in loaded for n in done["not_converted"]}
    has_records = any(posixpath.basename(p) == "RECORDS" for p in plan.discovery.lists)
    records = {u.path for u in plan.units}
    for entry in plan.raw.files:
        if entry.path not in used and entry.path not in skipped:
            skipped[entry.path] = _leftover_reason(
                entry.path, has_records, records, plan.discovery.annotators
            )
    manifest = StandardManifest(
        dataset_id=location.dataset_id,
        raw_version=location.version,
        raw_manifest_sha256=plan.raw_manifest_sha256,
        license=plan.raw.manifest.license,
        tool=f"biosignal-data {__version__}",
        created_at=created,
        completed_at=utcnow(),
        layout=arrays.settings(),
        recordings=joined["recordings"].num_rows,
        channels=joined["channels"].num_rows,
        events=joined["events"].num_rows,
        files_used=len(used),
        problems=plan.discovery.problems + [p for done, _ in loaded for p in done["problems"]],
        not_converted=[NotConverted(path=p, reason=r) for p, r in sorted(skipped.items())],
    )
    store.put_bytes(location.manifest_key, manifest.model_dump_json(indent=2).encode())
    store.put_bytes(location.success_key, b"", content_type="text/plain")
    store.delete_many(list(store.list(location.parts_prefix)))
    return manifest


def _leftover_reason(
    path: str, has_records: bool, records: set[str], annotators: frozenset[str]
) -> str:
    name = posixpath.basename(path)
    stem, _, ext = path.rpartition(".") if "." in name else (path, "", "")
    if annotators and stem in records and may_be_annotations(ext):
        return f"WFDB annotator {ext!r} is not listed in ANNOTATORS"
    if ext in annotators:
        return "WFDB annotation file of no converted record"
    ext = ext.lower()
    if ext == "hea":
        return "WFDB header of no converted record (e.g. a segment of a multi-segment record)"
    if ext in ("dat", "mat") and has_records:
        return f"not named by the header of a converted WFDB record (.{ext})"
    if ext in ("edf", "bdf") and has_records:
        return "not listed in RECORDS"
    return f"no reader for .{ext} files" if ext else "no reader for files without an extension"


def iter_not_converted(manifest: StandardManifest) -> Iterator[tuple[str, int]]:
    """Reasons with their file counts, most files first."""
    counts: dict[str, int] = {}
    for n in manifest.not_converted:
        counts[n.reason] = counts.get(n.reason, 0) + 1
    yield from sorted(counts.items(), key=lambda item: -item[1])
