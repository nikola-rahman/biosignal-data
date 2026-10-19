"""Copy one dataset version from its source into the raw layer, in two explicit steps.

    with planned(dataset, store, options=...) as plan:   # access, version, file list, once
        plan.files, plan.to_copy, plan.total_bytes          # all known before any byte moves
        report = apply(plan, store, workers=16)            # copies exactly plan.to_copy

`ingest()` does both for callers that do not need to look at the plan in between.

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
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from biosignal_data import __version__
from biosignal_data.catalog.models import Dataset
from biosignal_data.fetchers import FetchError, FetchOptions, RemoteFile, fetcher_for
from biosignal_data.fetchers.base import Fetcher, ResolvedVersion
from biosignal_data.paths import holds_recordings
from biosignal_data.read.raw import stored_versions
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import FileEntry, HashingReader, Manifest, utcnow
from biosignal_data.storage.stores import Store
from biosignal_data.text import plural

log = logging.getLogger(__name__)


class IntegrityError(RuntimeError):
    pass


class AlreadyComplete(RuntimeError):  # noqa: N818 - reads naturally at the call site
    pass


class IngestProgress(Protocol):
    """Receives the progress of one copy. `file_started` and `advance` are called from
    worker threads; the others from the calling thread."""

    def start(self, version: str, files: int, done: int, total_bytes: int | None) -> None:
        """The work is known: `files` in the version, of which `done` are already stored
        from an earlier run. `total_bytes` (still to copy) is None if the source omits
        sizes."""

    def file_started(self, path: str) -> None:
        """A file starts to download."""

    def advance(self, n: int) -> None:
        """`n` bytes were read; negative when the bytes of a failed file are taken back."""

    def file_done(self, path: str, ok: bool) -> None:
        """One file is finished, with or without success."""


@dataclass
class Plan:
    """What a copy of one dataset version will do, decided before any byte moves."""

    dataset: Dataset
    version: ResolvedVersion
    location: VersionLocation
    files: list[RemoteFile]  # every file of the version, as the source lists it
    to_copy: list[RemoteFile]  # those not stored yet by an earlier run
    manifest: Manifest  # new, or the partial one of an earlier run
    fetcher: Fetcher = field(repr=False)
    complete: bool = False  # set by apply() when the version is complete

    @property
    def already_stored(self) -> int:
        return len(self.files) - len(self.to_copy)

    @property
    def recording_files(self) -> int:
        """Files that can hold recorded signals (see paths.holds_recordings); 0 means the
        source holds no recordings this tool recognizes (e.g. only BIDS sidecars)."""
        return sum(1 for f in self.files if holds_recordings(f.path))

    @property
    def unknown_sizes(self) -> int:
        return sum(1 for f in self.to_copy if f.size is None)

    @property
    def total_bytes(self) -> int:
        """Bytes still to copy, as far as the source reports sizes (see unknown_sizes)."""
        return sum(f.size for f in self.to_copy if f.size is not None)


@dataclass
class IngestReport:
    dataset_id: str
    version: str
    location: str
    files: int = 0  # files in the version
    recording_files: int = 0  # of them, files that can hold recordings (paths.py)
    copied: int = 0
    skipped: int = 0  # already stored by an earlier run
    bytes_copied: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    complete: bool = False


@contextmanager
def planned(
    dataset: Dataset,
    store: Store,
    *,
    options: FetchOptions | None = None,
    fetcher: Fetcher | None = None,
    root: str = "raw",
) -> Iterator[Plan]:
    """Check access, prepare the source, and plan the copy. On exit, the source cleans up
    what it prepared; staged files are removed only if `apply` completed the version."""
    fetcher = fetcher or fetcher_for(dataset, options)
    fetcher.check_access()
    fetcher.prepare()
    plan: Plan | None = None
    try:
        plan = _plan(dataset, store, fetcher, root)
        yield plan
    finally:
        fetcher.cleanup(plan.complete if plan else False)


def ingest(
    dataset: Dataset,
    store: Store,
    *,
    options: FetchOptions | None = None,
    fetcher: Fetcher | None = None,
    root: str = "raw",
    workers: int = 8,
    checkpoint_every: float = 30.0,
    progress: IngestProgress | None = None,
    executor: Executor | None = None,
    stop: threading.Event | None = None,
) -> IngestReport:
    """Plan and copy `dataset` into `store`. Files are copied by `workers` threads, or by
    `executor` if given (a pool shared by several datasets copying at the same time)."""
    with planned(dataset, store, options=options, fetcher=fetcher, root=root) as plan:
        return apply(
            plan,
            store,
            workers=workers,
            checkpoint_every=checkpoint_every,
            progress=progress,
            executor=executor,
            stop=stop,
        )


def _plan(dataset: Dataset, store: Store, fetcher: Fetcher, root: str) -> Plan:
    version = fetcher.resolve_version()
    version = _continue_dated_version(dataset, store, root, version)
    location = VersionLocation(dataset.id, version.value, root)
    if store.exists(location.success_key):
        raise AlreadyComplete(
            f"{store.uri}/{location.prefix} is already complete; versions are immutable"
        )
    manifest = _start_manifest(dataset, store, location, version)
    files = list(fetcher.list_files())
    if not files:
        raise FetchError(f"{dataset.id} {version.value}: the source lists no files")
    if len({f.path for f in files}) != len(files):
        raise FetchError(f"{dataset.id} {version.value}: the source listed a path twice")
    done = {f.path: f for f in manifest.files}
    stored = store.sizes(location.data_prefix) if done else {}
    to_copy = []
    for remote in files:
        prior = done.get(remote.path)
        if not (prior and stored.get(location.data_key(remote.path)) == prior.size):
            done.pop(remote.path, None)
            to_copy.append(remote)
    manifest.files = sorted(done.values(), key=lambda f: f.path)
    plan = Plan(dataset, version, location, files, to_copy, manifest, fetcher)
    log.info(
        "%s %s: %s, %s to copy",
        dataset.id,
        version.value,
        plural(len(files), "file"),
        plural(len(to_copy), "file"),
    )
    _warn_implicit(plan)
    return plan


def apply(
    plan: Plan,
    store: Store,
    *,
    workers: int = 8,
    checkpoint_every: float = 30.0,
    progress: IngestProgress | None = None,
    executor: Executor | None = None,
    stop: threading.Event | None = None,
) -> IngestReport:
    """Copy exactly `plan.to_copy`; complete the version when every file is stored."""
    location = plan.location
    report = IngestReport(
        plan.dataset.id,
        plan.version.value,
        f"{store.uri}/{location.prefix}",
        files=len(plan.files),
        recording_files=plan.recording_files,
        skipped=plan.already_stored,
    )
    done = {f.path: f for f in plan.manifest.files}
    if progress:
        total = plan.total_bytes if not plan.unknown_sizes else None
        progress.start(plan.version.value, len(plan.files), plan.already_stored, total)
    checkpoint = _Checkpointer(store, location, plan.manifest, done, checkpoint_every)
    _copy_all(
        plan.to_copy, store, location, done, report, checkpoint, workers, progress, executor, stop
    )
    if report.failed:
        log.error("%d files failed; re-run to retry them", len(report.failed))
        return report
    plan.manifest.files = sorted(done.values(), key=lambda f: f.path)
    plan.manifest.completed_at = utcnow()
    store.put_bytes(location.manifest_key, plan.manifest.to_json())
    store.put_bytes(location.success_key, b"", content_type="text/plain")
    store.delete(location.partial_manifest_key)
    report.complete = plan.complete = True
    return report


# --- copying -------------------------------------------------------------------------------


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
    stop: threading.Event | None = None,
) -> FileEntry:
    key = location.data_key(remote.path)
    reader: HashingReader | None = None
    try:
        with remote.open() as stream:
            reader = HashingReader(stream, on_read, stop)
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
    executor: Executor | None,
    stop: threading.Event | None,
) -> None:
    # Results are handled in this thread only, so `done` and `report` need no lock.
    on_read = progress.advance if progress else None

    def copy(remote: RemoteFile) -> FileEntry:
        if progress:
            progress.file_started(remote.path)
        return copy_file(store, location, remote, on_read, stop)

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
                if progress:
                    progress.file_done(remote.path, ok=False)
                continue
            done[entry.path] = entry
            report.copied += 1
            report.bytes_copied += entry.size
            checkpoint()
            if progress:
                progress.file_done(remote.path, ok=True)
    finally:
        if pool is not executor:
            pool.shutdown(cancel_futures=True)
    checkpoint(force=True)


# --- planning helpers ----------------------------------------------------------------------


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
        license=dataset.license,
        retrieved_at=utcnow(),
        tool=f"biosignal-data {__version__}",
    )


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


def _warn_implicit(plan: Plan) -> None:
    """Say every choice made without the user asking for it, everything skipped, and what
    makes a copy less useful than it looks."""
    d, version = plan.dataset, plan.version
    if version.origin == "retrieval-date":
        log.warning(
            "%s: the source has no versions; labelling this download with today's date %s",
            d.id,
            version.value,
        )
    elif version.origin == "source" and not d.pinned_version:
        log.warning(
            "%s: no version pinned in the catalog; using the latest, %s", d.id, version.value
        )
    if plan.fetcher.version_is_label and d.pinned_version and version.origin != "user":
        log.warning(
            "%s: version %s is only the catalog's label: this source cannot be asked for a "
            "version, so the files are what it serves now",
            d.id,
            version.value,
        )
    if plan.already_stored:
        log.warning(
            "%s: %s already stored by an earlier run; not copied again",
            d.id,
            plural(plan.already_stored, "file"),
        )
    if not plan.recording_files:
        log.warning(
            "%s: none of its %s can hold recordings (no signal format, no archive); the "
            "copy holds only metadata",
            d.id,
            plural(len(plan.files), "file"),
        )
    unchecked = sum(1 for f in plan.to_copy if f.checksum is None)
    if unchecked:
        log.warning(
            "%s: %s without a published checksum; hashed, but cannot be verified",
            d.id,
            plural(unchecked, "file"),
        )


# --- checking ------------------------------------------------------------------------------


def verify(
    store: Store,
    dataset_id: str,
    version: str,
    root: str = "raw",
    deep: bool = False,
    workers: int = 8,
) -> dict[str, str]:
    """Check stored objects against the manifest. Returns {path: problem}.

    An incomplete version (an interrupted run) is checked against its partial manifest,
    i.e. the files copied so far. Sizes come from one listing of the version. `deep` also
    re-reads every object, `workers` at a time, and recomputes its SHA-256."""
    location = VersionLocation(dataset_id, version, root)
    data = store.get_bytes(location.manifest_key) or store.get_bytes(location.partial_manifest_key)
    if data is None:
        return {"_manifest.json": "missing"}
    entries = Manifest.from_json(data).files
    stored = store.sizes(location.data_prefix)
    problems: dict[str, str] = {}
    to_read: list[FileEntry] = []
    for entry in entries:
        size = stored.get(location.data_key(entry.path))
        if size is None:
            problems[entry.path] = "missing"
        elif size != entry.size:
            problems[entry.path] = f"size {size} != manifest {entry.size}"
        elif deep:
            to_read.append(entry)

    def sha256_of(entry: FileEntry) -> str:
        with store.open(location.data_key(entry.path)) as stream:
            reader = HashingReader(stream)
            while reader.read(8 * 1024 * 1024):
                pass
        return reader.sha256

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for entry, digest in zip(to_read, pool.map(sha256_of, to_read), strict=True):
            if digest != entry.sha256:
                problems[entry.path] = "sha256 mismatch"
    return problems
