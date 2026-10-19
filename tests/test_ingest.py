from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from biosignal_data.ingest import AlreadyComplete, apply, ingest, planned, verify
from biosignal_data.read.raw import RawDataset, complete_versions
from biosignal_data.storage.manifest import Manifest
from biosignal_data.storage.stores import LocalStore
from tests.conftest import FakeFetcher, make_dataset

FILES = {"README": b"hello", "sub-01/eeg/data.edf": b"\x00" * 1000, "a/b/c.csv": b"x,y\n1,2\n"}


def test_ingest_copies_bytes_and_writes_manifest(store: LocalStore) -> None:
    dataset = make_dataset()
    report = ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES), workers=2)

    assert report.complete and report.copied == 3 and not report.failed
    prefix = "raw/test/example/1.0.0"
    for path, body in FILES.items():
        assert store.get_bytes(f"{prefix}/data/{path}") == body
    assert store.exists(f"{prefix}/_SUCCESS")
    assert not store.exists(f"{prefix}/_manifest.partial.json")

    manifest = Manifest.from_json(store.get_bytes(f"{prefix}/_manifest.json") or b"")
    by_path = {f.path: f for f in manifest.files}
    assert by_path["README"].sha256 == hashlib.sha256(b"hello").hexdigest()
    assert by_path["README"].md5 == hashlib.md5(b"hello", usedforsecurity=False).hexdigest()
    assert manifest.total_bytes == sum(len(b) for b in FILES.values())
    assert manifest.license == "CC0"
    assert manifest.source == {"type": "manual", "url": None, "version": "1.0.0"}
    assert json.loads(store.get_bytes(f"{prefix}/_manifest.json") or b"")["file_count"] == 3


def test_completed_version_is_immutable(store: LocalStore) -> None:
    dataset = make_dataset()
    ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES))
    with pytest.raises(AlreadyComplete):
        ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES))


def test_checksum_mismatch_from_a_damaged_transfer_deletes_object_and_fails(
    store: LocalStore,
) -> None:
    dataset = make_dataset()
    good = {"README": "sha256:" + hashlib.sha256(FILES["README"]).hexdigest()}
    fetcher = FakeFetcher(dataset, FILES, checksums=good, damaged={"README"})
    report = ingest(dataset, store, fetcher=fetcher)
    assert fetcher.opened["README"] == 2  # downloaded again before it fails
    assert not report.complete
    assert "README" in report.failed
    assert not store.exists("raw/test/example/1.0.0/data/README")
    assert not store.exists("raw/test/example/1.0.0/_SUCCESS")


def test_matching_published_checksums_pass(store: LocalStore) -> None:
    dataset = make_dataset()
    good = {p: "md5:" + hashlib.md5(b, usedforsecurity=False).hexdigest() for p, b in FILES.items()}
    assert ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES, checksums=good)).complete


def test_size_mismatch_served_the_same_twice_is_kept_with_the_difference(
    store: LocalStore,
) -> None:
    dataset = make_dataset()
    report = ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES, sizes={"README": 99}))
    assert report.complete and "size 5 != published 99" in report.differs["README"]


def test_rerun_resumes_and_skips_finished_files(store: LocalStore) -> None:
    dataset = make_dataset()
    first = FakeFetcher(dataset, FILES, fail={"a/b/c.csv"})
    report = ingest(dataset, store, fetcher=first, checkpoint_every=0)
    assert not report.complete and set(report.failed) == {"a/b/c.csv"}
    assert store.exists("raw/test/example/1.0.0/_manifest.partial.json")

    second = FakeFetcher(dataset, FILES)
    report = ingest(dataset, store, fetcher=second)
    assert report.complete and report.skipped == 2 and report.copied == 1
    assert second.opened == {"a/b/c.csv": 1}


def test_plan_is_complete_before_copying_and_apply_copies_exactly_it(store: LocalStore) -> None:
    dataset = make_dataset()
    fetcher = FakeFetcher(dataset, FILES)
    with planned(dataset, store, fetcher=fetcher) as plan:
        assert plan.version.value == "1.0.0" and len(plan.files) == 3
        assert plan.total_bytes == sum(len(b) for b in FILES.values()) and not plan.unknown_sizes
        assert not list(store.list(""))  # planning copies nothing
        report = apply(plan, store)
    assert report.complete and report.copied == 3
    assert fetcher.listed == 1  # the source was asked for its files once


def test_verify_and_read_back(store: LocalStore, tmp_path: Path) -> None:
    dataset = make_dataset()
    ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES))
    assert verify(store, "test/example", "1.0.0", deep=True) == {}

    assert complete_versions(store, "test/example") == ["1.0.0"]
    ds = RawDataset.open(store, "test/example", "latest", cache_dir=tmp_path / "cache")
    assert [f.path for f in ds.glob("sub-*/eeg/*.edf")] == ["sub-01/eeg/data.edf"]
    assert ds.local_path("README").read_bytes() == b"hello"

    store.put_bytes("raw/test/example/1.0.0/data/README", b"tampered")
    assert verify(store, "test/example", "1.0.0") == {"README": "size 8 != manifest 5"}


def test_latest_complete_version_is_chosen(store: LocalStore, tmp_path: Path) -> None:
    dataset = make_dataset()
    for v in ("1.0.10", "1.0.2"):
        ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES, version=v))
    store.put_bytes("raw/test/example/2.0.0/_manifest.partial.json", b"{}")  # incomplete
    assert complete_versions(store, "test/example") == ["1.0.2", "1.0.10"]
    latest = RawDataset.open(store, "test/example", "latest", cache_dir=tmp_path)
    assert latest.location.version == "1.0.10"


class _Recorder:
    def __init__(self) -> None:
        self.started: tuple[int, int, int | None] | None = None
        self.bytes = 0
        self.files: list[tuple[str, bool]] = []
        self.begun: list[str] = []

    def start(self, version: str, files: int, done: int, total_bytes: int | None) -> None:
        self.started = (files, done, total_bytes)

    def file_started(self, path: str) -> None:
        self.begun.append(path)

    def advance(self, n: int) -> None:
        self.bytes += n

    def file_done(self, path: str, ok: bool) -> None:
        self.files.append((path, ok))


def test_progress_counts_only_successful_bytes(store: LocalStore) -> None:
    dataset = make_dataset()
    good = {"README": "sha256:" + hashlib.sha256(FILES["README"]).hexdigest()}
    recorder = _Recorder()
    fetcher = FakeFetcher(dataset, FILES, checksums=good, damaged={"README"})
    ingest(dataset, store, fetcher=fetcher, progress=recorder)
    assert recorder.bytes == sum(len(b) for p, b in FILES.items() if p != "README")
    assert sorted(recorder.begun) == sorted(FILES)
    assert sorted(recorder.files) == [
        ("README", False),
        ("a/b/c.csv", True),
        ("sub-01/eeg/data.edf", True),
    ]


def test_progress_counts_files_already_stored(store: LocalStore) -> None:
    dataset = make_dataset()
    ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES, fail={"a/b/c.csv"}))
    recorder = _Recorder()
    ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES), progress=recorder)
    assert recorder.started == (3, 2, len(FILES["a/b/c.csv"]))  # 3 files, 2 stored before
    assert recorder.files == [("a/b/c.csv", True)]


def test_undated_source_continues_an_earlier_days_download(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biosignal_data.fetchers import base  # noqa: PLC0415

    dataset = make_dataset(source={"type": "manual"})  # no version

    class Undated(FakeFetcher):
        def resolve_version(self) -> base.ResolvedVersion:
            return base.Fetcher.resolve_version(self)

    monkeypatch.setattr(base, "retrieval_version", lambda: "2026-10-01")
    ingest(dataset, store, fetcher=Undated(dataset, FILES, fail={"a/b/c.csv"}))
    monkeypatch.setattr(base, "retrieval_version", lambda: "2026-10-02")
    report = ingest(dataset, store, fetcher=Undated(dataset, FILES))
    assert report.version == "2026-10-01" and report.complete and report.skipped >= 1


def test_empty_listing_is_an_error_not_a_complete_version(store: LocalStore) -> None:
    from biosignal_data.fetchers import FetchError  # noqa: PLC0415

    dataset = make_dataset()
    with pytest.raises(FetchError, match="lists no files"):
        ingest(dataset, store, fetcher=FakeFetcher(dataset, {}))
    assert not store.exists("raw/test/example/1.0.0/_SUCCESS")


def test_stop_ends_copies_and_keeps_what_is_finished(store: LocalStore) -> None:
    import threading  # noqa: PLC0415

    dataset = make_dataset()
    stop = threading.Event()
    with planned(dataset, store, fetcher=FakeFetcher(dataset, FILES), stop=stop) as plan:
        stop.set()  # Ctrl+C after planning: no file may be read any more
        report = apply(plan, store, stop=stop)
    assert not report.complete and set(report.failed) == set(FILES)
    assert all("stopped" in reason for reason in report.failed.values())
    assert store.unfinished("raw") == ()  # no temporary files are left behind


def test_a_copy_without_recordings_is_reported(store: LocalStore) -> None:
    dataset = make_dataset()
    only_metadata = {"README": b"r", "sub-1/eeg/sub-1_channels.tsv": b"t", ".datalad/config": b"c"}
    report = ingest(dataset, store, fetcher=FakeFetcher(dataset, only_metadata))
    assert report.complete and report.recording_files == 0  # copied faithfully, but flagged
    report = ingest(make_dataset(id="test/other"), store, fetcher=FakeFetcher(dataset, FILES))
    assert report.recording_files == 1  # sub-01/eeg/data.edf


def test_a_file_the_source_lists_but_does_not_have_is_recorded_not_failed(
    store: LocalStore,
) -> None:
    dataset = make_dataset()
    fetcher = FakeFetcher(dataset, FILES, missing={"a/b/c.csv"})
    report = ingest(dataset, store, fetcher=fetcher, workers=2)

    assert report.complete and not report.failed and list(report.missing) == ["a/b/c.csv"]
    manifest = Manifest.from_json(store.get_bytes("raw/test/example/1.0.0/_manifest.json") or b"")
    assert [f.path for f in manifest.files] == ["README", "sub-01/eeg/data.edf"]
    assert [m.path for m in manifest.missing] == ["a/b/c.csv"]
    assert "404" in manifest.missing[0].reason
    assert verify(store, "test/example", "1.0.0", "raw", deep=True) == {}


def test_replace_copies_a_complete_version_again_and_drops_files_no_longer_listed(
    store: LocalStore,
) -> None:
    dataset = make_dataset()
    ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES), workers=2)
    with pytest.raises(AlreadyComplete, match="--replace"):
        ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES), workers=2)

    changed = {"README": b"hello again", "sub-01/eeg/data.edf": b"\x01" * 1000}  # c.csv gone
    report = ingest(dataset, store, fetcher=FakeFetcher(dataset, changed), replace=True)

    assert report.complete and report.copied == 2 and report.skipped == 0
    prefix = "raw/test/example/1.0.0"
    assert store.get_bytes(f"{prefix}/data/README") == b"hello again"
    assert not store.exists(f"{prefix}/data/a/b/c.csv")
    manifest = Manifest.from_json(store.get_bytes(f"{prefix}/_manifest.json") or b"")
    assert sorted(f.path for f in manifest.files) == sorted(changed)
    assert verify(store, "test/example", "1.0.0", "raw", deep=True) == {}


def test_a_stopped_replace_leaves_the_version_incomplete_and_a_normal_run_finishes_it(
    store: LocalStore,
) -> None:
    dataset = make_dataset()
    ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES), workers=2)
    broken = FakeFetcher(dataset, FILES, fail={"README"})
    report = ingest(dataset, store, fetcher=broken, workers=2, replace=True)

    assert not report.complete
    assert not store.exists("raw/test/example/1.0.0/_SUCCESS")
    assert not store.exists("raw/test/example/1.0.0/_manifest.json")
    fetcher = FakeFetcher(dataset, FILES)
    report = ingest(dataset, store, fetcher=fetcher, workers=2)
    assert report.complete and report.copied == 1 and fetcher.opened == {"README": 1}


def test_a_file_served_other_than_listed_twice_is_kept_with_the_difference(
    store: LocalStore,
) -> None:
    dataset = make_dataset()
    wrong = {"README": "sha256:" + "0" * 64}
    report = ingest(dataset, store, fetcher=FakeFetcher(dataset, FILES, checksums=wrong))

    assert report.complete and not report.failed and list(report.differs) == ["README"]
    manifest = Manifest.from_json(store.get_bytes("raw/test/example/1.0.0/_manifest.json") or b"")
    entry = next(f for f in manifest.files if f.path == "README")
    assert entry.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert entry.differs_from_source and "the same bytes in two downloads" in (
        entry.differs_from_source
    )
    assert store.get_bytes("raw/test/example/1.0.0/data/README") == b"hello"


def test_a_stop_ends_a_long_listing(store: LocalStore) -> None:
    import threading  # noqa: PLC0415
    from collections.abc import Iterator  # noqa: PLC0415

    from biosignal_data.fetchers import RemoteFile  # noqa: PLC0415
    from biosignal_data.storage.manifest import StoppedError  # noqa: PLC0415

    dataset = make_dataset()
    stop = threading.Event()

    class Endless(FakeFetcher):
        def list_files(self) -> Iterator[RemoteFile]:
            n = 0
            while True:  # a source with very many files; Ctrl+C after the first few
                n += 1
                if n == 5:
                    stop.set()
                yield RemoteFile(path=f"f{n}", size=1, checksum=None, opener=lambda: None)  # type: ignore[arg-type,return-value]

    with pytest.raises(StoppedError, match="stopped while listing"):
        ingest(dataset, store, fetcher=Endless(dataset, {}), stop=stop)
