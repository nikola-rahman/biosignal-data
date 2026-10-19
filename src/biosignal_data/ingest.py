"""Copy one dataset version from its source into the raw layer.

Guarantees:
- Bytes are copied unchanged, to `data/<path as published>`.
- Every file is hashed in flight (SHA-256 and MD5) and checked against the size and
  checksum the source publishes, when it publishes them. A mismatch deletes the object
  and fails the file.
- A run can be interrupted and re-run: progress is checkpointed to
  `_manifest.partial.json`, and files already recorded there with a matching object size
  are not copied again.
- `_manifest.json` and then `_SUCCESS` are written only when every file succeeded. A
  version with `_SUCCESS` is never written to again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol

from biosignal_data import __version__
from biosignal_data.catalog.models import Dataset
from biosignal_data.fetchers import FetchOptions, RemoteFile, fetcher_for
from biosignal_data.fetchers.base import Fetcher, ResolvedVersion
from biosignal_data.read.raw import stored_versions
from biosignal_data.selection import Selection, select
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import FileEntry, HashingReader, Manifest, utcnow
from biosignal_data.storage.stores import Store

log = logging.getLogger(__name__)


class IntegrityError(RuntimeError):
    pass


class IngestProgress(Protocol):
    """Receives the progress of one ingest. `file_started` and `advance` are called from
    worker threads; the others from the calling thread."""

    def start(self, version: str, recordings: int, done: int, total_bytes: int | None) -> None:
        """The work is known: `recordings` in total, of which `done` are already stored from an
        earlier run. `total_bytes` (still to copy) is None if the source omits sizes."""

    def file_started(self, path: str) -> None:
        """A file starts to download."""

    def advance(self, n: int) -> None:
        """`n` bytes were read; negative when the bytes of a failed file are taken back."""

    def file_done(self, path: str, ok: bool) -> None:
        """One file is finished, with or without success."""

    def recording_done(self) -> None:
        """All files of one more recording are stored."""


class _RecordingTracker:
    """Turns finished files into finished recordings for the progress display."""

    def __init__(
        self, recordings: list[list[str]], todo: list[RemoteFile], progress: IngestProgress
    ) -> None:
        self.progress = progress
        self.recording_of = {p: i for i, files in enumerate(recordings) for p in files}
        self.remaining: dict[int, int] = {}
        for remote in todo:
            index = self.recording_of.get(remote.path)
            if index is not None:
                self.remaining[index] = self.remaining.get(index, 0) + 1
        self.already_done = len(recordings) - len(self.remaining)

    def file_done(self, path: str, ok: bool) -> None:
        self.progress.file_done(path, ok)
        index = self.recording_of.get(path)
        if ok and index is not None:
            self.remaining[index] -= 1
            if self.remaining[index] == 0:
                self.progress.recording_done()


class AlreadyComplete(RuntimeError):  # noqa: N818 - reads naturally at the call site
    pass


@dataclass
class IngestReport:
    dataset_id: str
    version: str
    location: str
    copied: int = 0
    skipped: int = 0
    bytes_copied: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    complete: bool = False
    subset: bool = False  # only part of the version was selected; it stays incomplete
    recordings: int = 0  # recordings among the selected files


def _check(remote: RemoteFile, reader: HashingReader) -> None:
    if remote.size is not None and reader.bytes_read != remote.size:
        raise IntegrityError(f"size {reader.bytes_read} != published {remote.size}")
    if remote.checksum:
        algorithm, _, expected = remote.checksum.partition(":")
        actual = {"sha256": reader.sha256, "md5": reader.md5}.get(algorithm.lower())
        if actual is None:
            log.warning("%s: cannot verify %s checksum; recorded only", remote.path, algorithm)
        elif actual != expected.lower():
            raise IntegrityError(f"{algorithm} {actual} != published {expected}")


def copy_file(
    store: Store,
    location: VersionLocation,
    remote: RemoteFile,
    on_read: Callable[[int], None] | None = None,
) -> FileEntry:
    key = location.data_key(remote.path)
    reader: HashingReader | None = None
    try:
        with remote.open() as stream:
            reader = HashingReader(stream, on_read)
            store.put_stream(key, reader)  # type: ignore[arg-type]
        try:
            _check(remote, reader)
        except IntegrityError:
            store.delete(key)
            raise
    except Exception:
        if on_read and reader is not None:
            on_read(-reader.bytes_read)  # progress counts only files that succeed
        raise
    return FileEntry(
        path=remote.path,
        size=reader.bytes_read,
        sha256=reader.sha256,
        md5=reader.md5,
        source_checksum=remote.checksum,
        source_url=remote.url,
    )


def ingest(
    dataset: Dataset,
    store: Store,
    *,
    options: FetchOptions | None = None,
    root: str = "raw",
    workers: int = 8,
    checkpoint_every: float = 30.0,
    dry_run: bool = False,
    fetcher: Fetcher | None = None,
    include: Sequence[str] = (),
    limit: int | None = None,
    progress: IngestProgress | None = None,
    executor: Executor | None = None,
) -> IngestReport:
    """Copy `dataset` into `store`.

    Files are copied by `workers` threads, or by `executor` if given (a pool shared by
    several datasets downloading at the same time).

    `include` (glob patterns; `*` also matches `/`) and `limit` (first N recordings in path
    order, with their metadata; see `selection`) select a subset, for tests. A subset is
    copied and checkpointed, but the version is not marked complete; a later run without
    them continues from it.
    """
    fetcher = fetcher or fetcher_for(dataset, options)
    fetcher.check_access()
    fetcher.prepare()
    try:
        return _ingest(
            dataset,
            store,
            fetcher,
            root=root,
            workers=workers,
            checkpoint_every=checkpoint_every,
            dry_run=dry_run,
            include=include,
            limit=limit,
            progress=progress,
            executor=executor,
        )
    finally:
        fetcher.cleanup()


def _start_manifest(
    dataset: Dataset, store: Store, location: VersionLocation, version: ResolvedVersion
) -> Manifest:
    """Resume from `_manifest.partial.json` if a previous run left one, else start fresh."""
    previous = store.get_bytes(location.partial_manifest_key)
    if previous:
        manifest = Manifest.from_json(previous)
        if manifest.version != version.value:
            raise RuntimeError(f"partial manifest at {location.prefix} is for another version")
        return manifest
    return Manifest(
        dataset_id=dataset.id,
        version=version.value,
        version_origin=version.origin,
        source=dataset.source.model_dump(mode="json"),
        license=dataset.access.license,
        retrieved_at=utcnow(),
        tool=f"biosignal-data {__version__}",
    )


def _plan(
    files: list[RemoteFile], done: dict[str, FileEntry], store: Store, location: VersionLocation
) -> tuple[list[RemoteFile], int]:
    """Files still to copy, and how many are already stored from an earlier run."""
    todo: list[RemoteFile] = []
    skipped = 0
    for remote in files:
        prior = done.get(remote.path)
        if prior and store.size(location.data_key(remote.path)) == prior.size:
            skipped += 1
        else:
            done.pop(remote.path, None)
            todo.append(remote)
    return todo, skipped


class _Checkpointer:
    def __init__(
        self,
        store: Store,
        location: VersionLocation,
        manifest: Manifest,
        done: dict[str, FileEntry],
        every: float,
    ) -> None:
        self.store, self.location, self.manifest, self.done = store, location, manifest, done
        self.every = every
        self.last = time.monotonic()

    def __call__(self, force: bool = False) -> None:
        if force or time.monotonic() - self.last > self.every:
            self.manifest.files = sorted(self.done.values(), key=lambda f: f.path)
            self.store.put_bytes(self.location.partial_manifest_key, self.manifest.to_json())
            self.last = time.monotonic()


def _copy_all(
    todo: list[RemoteFile],
    store: Store,
    location: VersionLocation,
    done: dict[str, FileEntry],
    report: IngestReport,
    checkpoint: _Checkpointer,
    workers: int,
    progress: IngestProgress | None,
    tracker: _RecordingTracker | None,
    executor: Executor | None,
) -> None:
    # Results are handled in this thread only, so `done` and `report` need no lock.
    on_read = progress.advance if progress else None

    def copy(remote: RemoteFile) -> FileEntry:
        if progress:
            progress.file_started(remote.path)
        return copy_file(store, location, remote, on_read)

    pool = executor or ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = {pool.submit(copy, r): r for r in todo}
        for future in as_completed(futures):
            remote = futures[future]
            try:
                entry = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded per file, reported at the end
                log.error("%s: %s", remote.path, exc)
                report.failed[remote.path] = str(exc)
                if tracker:
                    tracker.file_done(remote.path, ok=False)
                continue
            done[entry.path] = entry
            report.copied += 1
            report.bytes_copied += entry.size
            checkpoint()
            if tracker:
                tracker.file_done(remote.path, ok=True)
    finally:
        if pool is not executor:
            pool.shutdown(cancel_futures=True)
    checkpoint(force=True)


def _continue_dated_version(
    dataset: Dataset, store: Store, root: str, version: ResolvedVersion
) -> ResolvedVersion:
    """A source without versions gets today's date as its label. If an earlier run of it
    stopped on another day, continue that version instead of starting a new one."""
    if version.origin != "retrieval-date":
        return version
    unfinished = [
        v for v, complete in stored_versions(store, dataset.id, root).items() if not complete
    ]
    if not unfinished or unfinished[-1] == version.value:
        return version
    log.warning(
        "%s: continuing the incomplete download labelled %s (from an earlier day) "
        "instead of starting a new version %s",
        dataset.id,
        unfinished[-1],
        version.value,
    )
    return ResolvedVersion(unfinished[-1], "retrieval-date")


def _warn_implicit(
    dataset: Dataset,
    version: ResolvedVersion,
    selection: Selection,
    report: IngestReport,
    todo: list[RemoteFile],
) -> None:
    """Say every choice made without the user asking for it, and everything skipped."""
    d = dataset.id
    if version.origin == "retrieval-date":
        log.warning("%s: the source has no versions; using today's date %s", d, version.value)
    elif version.origin == "source" and not dataset.pinned_version:
        log.warning("%s: no version pinned; using the latest, %s", d, version.value)
    if report.skipped:
        log.warning(
            "%s: %d files are already stored from an earlier run; not downloading them again",
            d,
            report.skipped,
        )
    if report.subset:
        log.warning(
            "%s: subset only (%d recordings%s); version %s stays incomplete",
            d,
            len(selection.recordings),
            f" + {selection.metadata_added} metadata files they need"
            if selection.metadata_added
            else "",
            version.value,
        )
    if not selection.signal_files_found:
        log.warning(
            "%s: no file has a known signal format; each data file counts as one recording", d
        )
    unchecked = sum(1 for f in todo if f.checksum is None)
    if unchecked:
        log.warning(
            "%s: %d files have no published checksum; they are hashed but cannot be verified",
            d,
            unchecked,
        )


def _ingest(
    dataset: Dataset,
    store: Store,
    fetcher: Fetcher,
    *,
    root: str,
    workers: int,
    checkpoint_every: float,
    dry_run: bool,
    include: Sequence[str],
    limit: int | None,
    progress: IngestProgress | None,
    executor: Executor | None,
) -> IngestReport:
    version = _continue_dated_version(dataset, store, root, fetcher.resolve_version())
    location = VersionLocation(dataset.id, version.value, root)
    report = IngestReport(dataset.id, version.value, f"{store.uri}/{location.prefix}")
    if store.exists(location.success_key):
        raise AlreadyComplete(f"{report.location} is already complete; versions are immutable")

    manifest = _start_manifest(dataset, store, location, version)
    done = {f.path: f for f in manifest.files}
    files = list(fetcher.list_files())
    if len({f.path for f in files}) != len(files):
        raise RuntimeError("the source listed the same path twice")
    by_path = {f.path: f for f in files}
    selection = select(list(by_path), include, limit)
    selected = [by_path[p] for p in selection.paths]
    report.subset = len(selected) < len(files)
    report.recordings = len(selection.recordings)
    todo, report.skipped = _plan(selected, done, store, location)
    log.info(
        "%s %s: %d recordings (%d files) selected of %d files; %d files to copy",
        dataset.id,
        version.value,
        len(selection.recordings),
        len(selected),
        len(files),
        len(todo),
    )
    _warn_implicit(dataset, version, selection, report, todo)
    if dry_run:
        return report

    tracker = None
    if progress:
        tracker = _RecordingTracker(selection.recordings, todo, progress)
        sizes = [f.size for f in todo]
        total = sum(s for s in sizes if s is not None) if None not in sizes else None
        progress.start(version.value, len(selection.recordings), tracker.already_done, total)
    checkpoint = _Checkpointer(store, location, manifest, done, checkpoint_every)
    _copy_all(todo, store, location, done, report, checkpoint, workers, progress, tracker, executor)
    if report.failed:
        log.error("%d files failed; re-run to retry them", len(report.failed))
        return report
    if report.subset:
        return report  # checkpointed; a run without include/limit completes the version

    manifest.files = sorted(done.values(), key=lambda f: f.path)
    manifest.completed_at = utcnow()
    store.put_bytes(location.manifest_key, manifest.to_json())
    store.put_bytes(location.success_key, b"", content_type="text/plain")
    store.delete(location.partial_manifest_key)
    report.complete = True
    return report


def verify(
    store: Store, dataset_id: str, version: str, root: str = "raw", deep: bool = False
) -> dict[str, str]:
    """Check stored objects against the manifest. Returns {path: problem}.

    An incomplete version (a subset or an interrupted run) is checked against its partial
    manifest, i.e. the files copied so far.
    `deep` re-reads every object and recomputes SHA-256 (slow, reads all bytes)."""
    location = VersionLocation(dataset_id, version, root)
    data = store.get_bytes(location.manifest_key) or store.get_bytes(location.partial_manifest_key)
    if data is None:
        return {"_manifest.json": "missing"}
    problems: dict[str, str] = {}
    for entry in Manifest.from_json(data).files:
        key = location.data_key(entry.path)
        size = store.size(key)
        if size is None:
            problems[entry.path] = "missing"
        elif size != entry.size:
            problems[entry.path] = f"size {size} != manifest {entry.size}"
        elif deep:
            with store.open(key) as stream:
                reader = HashingReader(stream)
                while reader.read(8 * 1024 * 1024):
                    pass
            if reader.sha256 != entry.sha256:
                problems[entry.path] = "sha256 mismatch"
    return problems
