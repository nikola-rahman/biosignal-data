"""Make the standard version of one raw version, in two explicit steps.

    plan = plan_standard(store, "physionet/mitdb", "1.0.0")   # recordings, from the file list
    plan.units, plan.to_convert, plan.discovery.problems       # all known before any work
    report = convert(plan, store, workers=4, work_dir=...)     # converts exactly plan.to_convert

Guarantees:
- Every recording is converted from raw files that are checked against the SHA-256 of the
  raw manifest, by two independent readers that must agree; every block is decoded again
  and compared with the source before it is kept, and every object is uploaded with its
  MD5, which the storage checks.
- A run can be stopped and run again: each batch of recordings finished is recorded under
  `_parts/` (its objects first, then its rows and a `done.json` written last), and is not
  converted again.
- The tables, `_manifest.json` and then `_SUCCESS` are written only when every recording
  is done. Every raw file is then either used or listed as not converted, with the reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import posixpath
import shutil
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from biosignal_data import __version__
from biosignal_data.read.raw import RawDataset
from biosignal_data.standard import blocks, tables
from biosignal_data.standard.discover import Discovery, Unit, discover, may_be_annotations
from biosignal_data.standard.layout import DONE, TABLES, StandardLocation
from biosignal_data.standard.manifest import NotConverted, StandardManifest
from biosignal_data.standard.sources import (
    EdfSource,
    Source,
    UnsupportedError,
    WfdbSource,
    wfdb_annotations,
)
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import Manifest, utcnow
from biosignal_data.storage.stores import Store
from biosignal_data.text import plural

log = logging.getLogger(__name__)
HEADER_READS = 16  # WFDB headers read at the same time while planning (small files)


class AlreadyStandard(RuntimeError):  # noqa: N818 - reads naturally at the call site
    pass


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
    raw_manifest: bytes = b""  # as stored: kept with the standard version, for its lineage

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
    store: Store,
    dataset_id: str,
    version: str,
    *,
    raw_root: str = "raw",
    replace: bool = False,
    raw_store: Store | None = None,
) -> StandardPlan:
    """Find the recordings of a complete raw version, and those an earlier run finished.
    `replace`: a complete version is made again (by `convert`, which first removes it)
    instead of raising `AlreadyStandard`. `raw_store`: where the raw version is, when not in
    `store` (e.g. staged on local disk from the source's archive)."""
    raw_store = store if raw_store is None else raw_store
    raw_location = VersionLocation(dataset_id, version, raw_root)
    data = raw_store.get_bytes(raw_location.manifest_key)
    if data is None or not raw_store.exists(raw_location.success_key):
        raise FileNotFoundError(f"{raw_store.uri}/{raw_location.prefix} is not a complete version")
    location = StandardLocation(dataset_id, version)
    complete = store.exists(location.success_key)
    if complete and not replace:
        raise AlreadyStandard(
            f"{store.uri}/{location.prefix} is already complete; use --replace to make it again"
        )
    raw = RawDataset(raw_store, raw_location, Manifest.from_json(data), Path())

    def read_text(path: str) -> str:
        with raw_store.open(raw_location.data_key(path)) as f:
            return f.read().decode("utf-8", errors="replace")

    def read_texts(paths: list[str]) -> dict[str, str]:
        with ThreadPoolExecutor(max_workers=HEADER_READS) as pool:
            return dict(zip(paths, pool.map(read_text, paths), strict=True))

    discovery = discover((f.path for f in raw.files), read_text, read_texts)
    done = set() if complete else set(_done_parts(store, location))
    plan = StandardPlan(
        raw, hashlib.sha256(data).hexdigest(), location, discovery, done, replace=complete,
        raw_manifest=data,
    )  # fmt: skip
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


def _done_parts(store: Store, location: StandardLocation) -> dict[str, str]:
    """Each finished recording -> the batch part that holds it (from its done.json)."""
    keys = [k for k in store.list(location.parts_prefix) if k.endswith("/" + DONE)]
    folders = [k[len(location.parts_prefix) : -len("/" + DONE)] for k in keys]
    found: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=HEADER_READS) as pool:
        for folder, data in zip(folders, pool.map(store.get_bytes, keys), strict=True):
            if data is None:
                raise RuntimeError(f"{location.parts_prefix}{folder}: done.json vanished")
            for unit_id in json.loads(data)["units"]:
                found[unit_id] = folder
    return found


def convert(
    plan: StandardPlan,
    store: Store,
    *,
    workers: int,
    part_bytes: int,
    work_dir: Path,
    progress: StandardProgress | None = None,
    stop: threading.Event | None = None,
) -> StandardReport:
    """Convert exactly `plan.to_convert`, `workers` batches at a time (see `_batches`);
    complete the version when every recording is done. Raw files are fetched to
    `work_dir`, one folder per recording, removed when it is done."""
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

    lock = threading.Lock()

    def run(batch: list[Unit]) -> None:
        _convert_batch(plan, store, batch, work_dir, report, lock, progress, stop)

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        batches = _batches(plan, plan.to_convert, part_bytes, workers)
        futures = [pool.submit(run, b) for b in batches]
        for future in as_completed(futures):
            future.result()
    finally:
        # On Ctrl+C: batches not started yet are dropped; those running finish.
        pool.shutdown(cancel_futures=True)
    if report.failed:
        log.error("%s failed; run again to retry them", plural(len(report.failed), "recording"))
        return report
    manifest = finish(plan, store, workers)
    report.not_converted = len(manifest.not_converted)
    report.complete = True
    return report


def _convert_batch(
    plan: StandardPlan,
    store: Store,
    batch: list[Unit],
    work_dir: Path,
    report: StandardReport,
    lock: threading.Lock,
    progress: StandardProgress | None,
    stop: threading.Event | None,
) -> None:
    """Convert a batch, recording by recording, into the batch's own objects; upload them
    and record the recordings that succeeded (called from worker threads)."""
    name = "batch-" + hashlib.sha256("\n".join(u.id for u in batch).encode()).hexdigest()[:20]
    folder = Path(tempfile.mkdtemp(prefix="objects-", dir=work_dir))
    try:
        writer = blocks.ObjectWriter(folder, name)
        results = []
        for unit in batch:
            if stop is not None and stop.is_set():
                with lock:
                    report.failed[unit.id] = "StoppedError: stopped"
                continue
            if progress:
                progress.unit_started(unit.id)
            try:
                result = convert_unit(plan, store, unit, work_dir)
            except Exception as exc:  # noqa: BLE001 - recorded per recording, reported at the end
                log.error("%s: %s", unit.id, exc)
                with lock:
                    report.failed[unit.id] = f"{type(exc).__name__}: {exc}"
                if progress:
                    progress.unit_done(unit.id, ok=False)
                continue
            _keep(writer, result)
            results.append(result)
            if progress:
                progress.unit_done(unit.id, ok=True)
        if results:
            for key, path in writer.objects:
                store.put_bytes(
                    plan.location.object_key(key), path.read_bytes(), "application/octet-stream"
                )
            _record_batch(store, plan.location, name, results)
            with lock:
                report.converted += len(results)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _keep(writer: blocks.ObjectWriter, result: UnitResult) -> None:
    """Put a converted recording's blocks into the batch's objects (only now: a recording
    that failed adds none), and its rows into its `blocks` table."""
    rows = []
    for row, blob in result.blocks:
        obj, offset = writer.add(blob)
        rows.append(row | {"object": obj, "offset": offset})
    result.blocks = []
    if rows:
        result.parts["blocks"] = pa.Table.from_pylist(rows, schema=blocks.BLOCKS)


@dataclass
class UnitResult:
    """One converted recording: its table rows and what it used, for its batch's part."""

    unit: Unit
    parts: dict[str, pa.Table]
    used: list[str]
    not_converted: list[NotConverted]
    problems: list[str]
    blocks: list[tuple[dict[str, Any], bytes]] = field(default_factory=list)  # until kept


def convert_unit(plan: StandardPlan, store: Store, unit: Unit, work_dir: Path) -> UnitResult:
    """Convert one recording: its blocks (checked, compressed) and its table rows are
    returned, to be stored with the other recordings of its batch."""
    folder = Path(tempfile.mkdtemp(prefix="unit-", dir=work_dir))
    try:
        raw = plan.raw

        def fetch(path: str) -> Path:
            return raw.fetch_to(path, folder)

        def open_source() -> Source:
            if unit.format == "wfdb":
                return WfdbSource(unit.path, fetch, raw.has)
            return EdfSource(unit.path, fetch)

        try:
            source = open_source()
        except UnsupportedError as exc:
            main = unit.path + ".hea" if unit.format == "wfdb" else unit.path
            skipped = [NotConverted(path=p, reason=str(exc)) for p in (main, *unit.annotations)]
            return UnitResult(unit, {}, [], skipped, [])
        try:
            events = list(source.events)
            used, skipped, problems = list(source.files), [], list(source.problems)
            for path in unit.annotations:
                try:
                    found, odd = wfdb_annotations(
                        path, fetch, source.frame_rate, listed=unit.listed_annotators
                    )
                    events += found
                    problems += odd
                    used.append(path)
                except Exception as exc:
                    if unit.listed_annotators:
                        raise
                    reason = f"tried as a WFDB annotation file: {type(exc).__name__}: {exc}"
                    skipped.append(NotConverted(path=path, reason=reason))
            encoded: list[tuple[dict[str, Any], bytes]] = []
            sha256: dict[tuple[str, int], str] = {}
            if source.groups:
                check = open_source()  # a second, independent reader of the same files
                try:
                    encoded, sha256 = blocks.encode_source(unit.id, source, check)
                finally:
                    check.close()
            files = {p: raw.entry(p).sha256 for p in used}
            parts = {
                "recordings": tables.recording_table(unit.id, source, bool(source.groups), files),
                "channels": tables.channel_table(unit.id, source, sha256),
                "events": tables.event_table(unit.id, events),
            }
            return UnitResult(unit, parts, used, skipped, problems, encoded)
        finally:
            source.close()
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _record_batch(
    store: Store, location: StandardLocation, batch: str, results: list[UnitResult]
) -> None:
    """One part for a batch of converted recordings (its objects are already stored): each
    table's rows of all of them, then `done.json`, written last (a part without it is not
    done). One part per batch, not per recording: many small recordings made tens of
    thousands of objects, and the S3 storage answered 429 Too Many Requests (PTB-XL,
    2026-10-20)."""
    ids = [r.unit.id for r in results]
    names = sorted({name for r in results for name in r.parts})
    for name in names:
        table = pa.concat_tables(
            [r.parts[name] for r in results if name in r.parts], promote_options="none"
        )
        store.put_bytes(
            location.part_key(batch, f"{name}.parquet"),
            tables.to_parquet(table),
            "application/vnd.apache.parquet",
        )
    done = {
        "units": ids,
        "tables": names,
        "used": sorted({p for r in results for p in r.used}),
        "not_converted": [n.model_dump() for r in results for n in r.not_converted],
        "problems": [p for r in results for p in r.problems],
    }
    store.put_bytes(location.part_key(batch, DONE), json.dumps(done).encode())


def _batches(
    plan: StandardPlan, units: list[Unit], part_bytes: int, workers: int
) -> list[list[Unit]]:
    """`units` in batches, in order, each closed when its raw files reach `part_bytes` (many
    small recordings share a part; a large one has its own) or when it holds a `workers`-th
    of the units, so that every worker has a batch: challenge-2017 (0.17 GB) made only 3
    batches of 64 MiB, and 3 of 16 workers did all of it (2026-10-20)."""
    most = max(1, math.ceil(len(units) / max(1, workers)))
    by_stem: dict[str, int] = {}
    for entry in plan.raw.files:
        stem = posixpath.splitext(entry.path)[0]
        by_stem[stem] = by_stem.get(stem, 0) + entry.size
    batches: list[list[Unit]] = []
    current: list[Unit] = []
    size = 0
    for unit in units:
        current.append(unit)
        size += by_stem.get(
            posixpath.splitext(unit.path)[0] if unit.format == "edf" else unit.path, 0
        )
        if size >= part_bytes or len(current) >= most:
            batches.append(current)
            current, size = [], 0
    if current:
        batches.append(current)
    return batches


def finish(plan: StandardPlan, store: Store, workers: int) -> StandardManifest:
    """Join the parts into the tables, write the manifest and `_SUCCESS`, remove `_parts/`."""
    location = plan.location
    created = utcnow()

    def load(part: str) -> tuple[dict[str, Any], dict[str, pa.Table]]:
        data = store.get_bytes(location.part_key(part, DONE))
        if data is None:
            raise RuntimeError(f"{part} is not done")
        done = json.loads(data)
        parts = {}
        for name in done["tables"]:
            blob = store.get_bytes(location.part_key(part, f"{name}.parquet"))
            if blob is None:
                raise RuntimeError(f"{part}: the {name} part is missing")
            parts[name] = tables.from_parquet(blob, name)
        return done, parts

    holders = _done_parts(store, location)
    missing = [u.id for u in plan.units if u.id not in holders]
    if missing:
        raise RuntimeError(f"{len(missing)} recordings are not done, e.g. {missing[0]}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load, sorted({holders[u.id] for u in plan.units})))
    order = {u.id: i for i, u in enumerate(plan.units)}
    joined = {
        name: _in_unit_order(
            pa.concat_tables(
                [parts[name] for _, parts in loaded if name in parts], promote_options="none"
            ),
            order,
        )
        if any(name in parts for _, parts in loaded)
        else tables.SCHEMAS[name].empty_table()
        for name in TABLES
    }
    for name, table in joined.items():
        store.put_bytes(
            location.table_key(name), tables.to_parquet(table), "application/vnd.apache.parquet"
        )
    # Objects of batches that stopped before their done.json are in no table: removed.
    kept = {location.object_key(o) for o in joined["blocks"].column("object").to_pylist()}
    stale = [k for k in store.list(location.object_key(blocks.DATA) + "/") if k not in kept]
    if stale:
        log.warning("%s: removing %d objects of unfinished batches", location.prefix, len(stale))
        store.delete_many(stale)

    used = set(plan.discovery.lists) | {p for done, _ in loaded for p in done["used"]}
    skipped = {n["path"]: n["reason"] for done, _ in loaded for n in done["not_converted"]}
    for view, owners in plan.discovery.views.values():
        reason = (
            f"a view of {', '.join(owners)}: its header shows their signal files side by "
            "side; the signals are converted with those records"
        )
        for path in (view.path + ".hea", *view.annotations):
            skipped.setdefault(path, reason)
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
        layout=blocks.settings(),
        recordings=joined["recordings"].num_rows,
        channels=joined["channels"].num_rows,
        events=joined["events"].num_rows,
        files_used=len(used),
        problems=plan.discovery.problems + [p for done, _ in loaded for p in done["problems"]],
        not_converted=[NotConverted(path=p, reason=r) for p, r in sorted(skipped.items())],
    )
    _keep_files(store, plan, location, skipped)
    store.put_bytes(location.manifest_key, manifest.model_dump_json(indent=2).encode())
    # The raw version is a build input and may be deleted later; its manifest (every file
    # with its SHA-256; its SHA-256 is raw_manifest_sha256) stays with the standard version.
    store.put_bytes(location.raw_manifest_key, plan.raw_manifest)
    store.put_bytes(location.success_key, b"", content_type="text/plain")
    store.delete_many(list(store.list(location.parts_prefix)))
    return manifest


KEEP_BYTES = 16 * 2**20  # raw files not converted and at most this size are kept as files/


def _keep_files(
    store: Store, plan: StandardPlan, location: StandardLocation, not_converted: dict[str, str]
) -> None:
    """Copy the raw files that hold no signals (tables, documentation) into files/, so that
    the standard version does not need the raw version: the index reads subject tables
    from them, a benchmark its labels (PTB-XL: ptbxl_database.csv). Larger files stay
    listed in the manifest as not converted only."""
    for path in sorted(not_converted):
        entry = plan.raw.entry(path)
        if entry.size > KEEP_BYTES:
            continue
        with plan.raw.store.open(plan.raw.location.data_key(path)) as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise RuntimeError(f"{path}: differs from the raw manifest")
        store.put_bytes(location.file_key(path), data, "application/octet-stream")


def _in_unit_order(table: pa.Table, order: dict[str, int]) -> pa.Table:
    """The rows sorted by their recording's place in the plan, stably: the parts come in
    any order, the tables always in the same one."""
    keys = [(order[r], i) for i, r in enumerate(table.column("recording_id").to_pylist())]
    return table.take(pa.array(sorted(range(table.num_rows), key=keys.__getitem__), pa.int64()))


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
