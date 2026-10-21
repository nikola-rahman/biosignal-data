from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyedflib
import pytest
import wfdb

from biosignal_data.ingest import ingest
from biosignal_data.read.standard import StandardDataset
from biosignal_data.standard import blocks
from biosignal_data.standard.convert import (
    AlreadyStandard,
    convert,
    plan_standard,
)
from biosignal_data.standard.discover import discover
from biosignal_data.standard.rules import DatasetRules, EdfAnnotationFiles, StandardRules
from biosignal_data.storage.stores import LocalStore
from tests.conftest import FakeFetcher, make_dataset

RNG = np.random.default_rng(0)
N = 3000  # frames of the WFDB record


def wfdb_record(tmp: Path) -> tuple[dict[str, bytes], np.ndarray, np.ndarray]:
    """Record `r1`: ECG at 2 samples per frame (200 Hz), X at 100 Hz and not calibrated."""
    ecg = RNG.integers(-2000, 2000, N * 2).astype(np.int16)
    x = RNG.integers(-2000, 2000, N).astype(np.int16)
    frames = np.empty((N, 3), np.int16)
    frames[:, 0:2], frames[:, 2] = ecg.reshape(N, 2), x
    header = (
        f"r1 2 100 {N} 10:20:30 01/02/2003\n"
        f"r1.dat 16x2 200(5)/mV 16 0 {ecg[0]} 0 0 ECG\n"
        f"r1.dat 16 0/mV 16 0 {x[0]} 0 0 X\n"
        "# age: 40\n"
    )
    (tmp / "r1.hea").write_text(header)
    ann = wfdb.Annotation(
        "r1", "atr", sample=np.array([10, 500]), symbol=["N", "+"], aux_note=["", "(AFIB"], fs=100
    )
    ann.wrann(write_dir=str(tmp))
    files = {
        "r1.hea": header.encode(),
        "r1.dat": frames.astype("<i2").tobytes(),
        "r1.atr": (tmp / "r1.atr").read_bytes(),
    }
    return files, ecg, x


def edf_file(tmp: Path) -> tuple[bytes, np.ndarray, np.ndarray]:
    """EDF+ with EEG at 256 Hz, a 1 Hz channel and one annotation; 60 s."""
    path = tmp / "s1.edf"
    eeg = RNG.integers(-3000, 3000, 256 * 60).astype(np.int16)
    slow = RNG.integers(-100, 100, 60).astype(np.int16)
    w = pyedflib.EdfWriter(str(path), 2, file_type=pyedflib.FILETYPE_EDFPLUS)
    w.setSignalHeaders(
        [
            {"label": "EEG Fpz", "dimension": "uV", "sample_frequency": 256,
             "physical_min": -3276.8, "physical_max": 3276.7,
             "digital_min": -32768, "digital_max": 32767},
            {"label": "Temp", "dimension": "degC", "sample_frequency": 1,
             "physical_min": -100.0, "physical_max": 100.0,
             "digital_min": -32768, "digital_max": 32767},
        ]
    )  # fmt: skip
    w.writeSamples([eeg.astype(np.int32), slow.astype(np.int32)], digital=True)
    w.writeAnnotation(12.5, 2.0, "arousal")
    w.close()
    return path.read_bytes(), eeg, slow


@pytest.fixture
def raw(store: LocalStore, tmp_path: Path) -> dict[str, np.ndarray]:
    """A raw version with a WFDB record, an EDF+ file, a multi-segment record and a CSV."""
    files, ecg, x = wfdb_record(tmp_path)
    edf, eeg, slow = edf_file(tmp_path)
    files |= {
        "RECORDS": b"r1\nr2\nr3\nm1\nsub/s1.edf\nmissing\n",
        "r3.hea": b"r3 1 100 1000\nr3.dat 16 200 16 0 0 0 0 I\n",  # r3.dat not published
        "r2.hea": b"r2 0 100 1000\n",  # signals not published, only annotations
        "r2.atr": files["r1.atr"],
        "ANNOTATORS": b"atr\treference annotations\n",
        "sub/s1.edf": edf,
        "m1.hea": b"m1/2 1 100 200\nm1_1 100\nm1_2 100\n",
        "m1_1.hea": b"m1_1 1 100 100\nm1_1.dat 16 200 16 0 0 0 0 I\n",
        "m1_1.dat": b"\x00" * 200,
        "notes.csv": b"a,b\n1,2\n",
        "r1.at_": files["r1.atr"],  # an annotator not in ANNOTATORS
        "r0.atr": files["r1.atr"],  # annotations of a record that is not there
        "r1.xws": b"wave settings",
    }
    dataset = make_dataset(id="physionet/mitdb")
    ingest(dataset, store, fetcher=FakeFetcher(dataset, files), workers=2)
    return {"ecg": ecg, "x": x, "eeg": eeg, "slow": slow}


def test_objects_of_unfinished_batches_are_removed_at_the_end(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    stale = "standard-blocks/physionet/mitdb/1.0.0/data/batch-earlier-0000.bin"
    store.put_bytes(stale, b"from a batch that stopped before its done.json")
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    report = convert(plan, store, workers=1, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete
    assert not store.exists(stale)
    assert list((tmp_path / "work").iterdir()) == []  # the local objects are removed


def test_recordings_converted_in_processes_are_the_same_as_in_threads(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    report = convert(plan, store, workers=2, part_bytes=1, work_dir=tmp_path / "work", processes=2)
    assert report.complete and not report.failed and report.converted == 5
    ds = StandardDataset.open(store, "physionet/mitdb", "1.0.0")
    np.testing.assert_array_equal(ds.read("r1", "200Hz"), raw["ecg"][None, :])
    np.testing.assert_array_equal(ds.read("sub/s1", "256Hz"), raw["eeg"][None, :])
    in_processes = sorted(ds.channels.column("sha256").to_pylist(), key=str)
    events = ds.events.num_rows
    again = plan_standard(store, "physionet/mitdb", "1.0.0", replace=True)
    convert(again, store, workers=2, part_bytes=1, work_dir=tmp_path / "work2")
    threads = StandardDataset.open(store, "physionet/mitdb", "1.0.0")
    assert sorted(threads.channels.column("sha256").to_pylist(), key=str) == in_processes
    assert threads.events.num_rows == events


def test_standard_version_is_exact_and_accounts_for_every_file(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert [u.id for u in plan.units] == ["m1", "r1", "r2", "r3", "sub/s1"]
    assert plan.discovery.problems == ["missing: listed in RECORDS, but no such header or EDF file"]

    report = convert(plan, store, workers=2, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete and not report.failed and report.converted == 5

    prefix = "standard-blocks/physionet/mitdb/1.0.0"
    ds = StandardDataset.open(store, "physionet/mitdb", "1.0.0")
    np.testing.assert_array_equal(ds.read("r1", "200Hz"), raw["ecg"][None, :])
    np.testing.assert_array_equal(ds.read("r1", "100Hz"), raw["x"][None, :])
    assert ds.read("r1", "200Hz").dtype == np.int16
    np.testing.assert_array_equal(ds.read("sub/s1", "256Hz"), raw["eeg"][None, :])
    np.testing.assert_array_equal(ds.read("sub/s1", "1Hz"), raw["slow"][None, :])
    np.testing.assert_array_equal(ds.read("r1", "200Hz", 17, 61), raw["ecg"][None, 17:61])
    many = ds.read_many([("r1", "200Hz"), ("sub/s1", "256Hz")], threads=2)
    np.testing.assert_array_equal(many["sub/s1", "256Hz"], raw["eeg"][None, :])
    # small recordings share objects: every block is in one of a few objects
    objects = set(ds._table("blocks").column("object").to_pylist())
    assert objects == {k[len(prefix) + 1 :] for k in store.list(f"{prefix}/data/")}

    recordings = pq.read_table(store._path(f"{prefix}/recordings.parquet")).to_pylist()
    assert [r["recording_id"] for r in recordings] == ["r1", "r2", "r3", "sub/s1"]
    assert (recordings[2]["n_channels"], recordings[2]["signals"]) == (0, False)
    r1, r2 = recordings[0], recordings[1]
    assert (r2["n_channels"], r2["signals"], r2["arrays"]) == (0, False, [])
    assert r2["source_files"] == ["r2.hea", "r2.atr"] and r2["duration_s"] == 10.0
    assert r1["source_files"] == ["r1.hea", "r1.dat", "r1.atr"]
    assert r1["start_time"].isoformat() == "2003-02-01T10:20:30"
    assert r1["source_header"].startswith("r1 2 100")

    channels = {
        (c["recording_id"], c["name"]): c
        for c in pq.read_table(store._path(f"{prefix}/channels.parquet")).to_pylist()
    }
    ecg = channels[("r1", "ECG")]
    assert (ecg["scale"], ecg["offset"], ecg["sampling_rate"]) == (1 / 200, -5 / 200, 200.0)
    assert channels[("r1", "X")]["scale"] is None  # gain 0: not calibrated
    fpz = channels[("sub/s1", "EEG Fpz")]
    assert fpz["scale"] == pytest.approx(0.1) and fpz["offset"] == pytest.approx(0.0, abs=0.06)

    events = pq.read_table(store._path(f"{prefix}/events.parquet")).to_pylist()
    assert len([e for e in events if e["recording_id"] == "r2"]) == 2
    wfdb_events = [e for e in events if e["annotator"] == "atr" and e["recording_id"] == "r1"]
    assert [(e["sample"], e["label"], e["text"]) for e in wfdb_events] == [
        (10, "N", None),
        (500, "+", "(AFIB"),
    ]
    assert wfdb_events[1]["onset_s"] == 5.0
    edf_events = [e for e in events if e["annotator"] == "edf+"]
    assert [(e["onset_s"], e["duration_s"], e["label"]) for e in edf_events] == [
        (12.5, 2.0, "arousal")
    ]

    manifest = json.loads(store.get_bytes(f"{prefix}/_manifest.json") or b"")
    reasons = {n["path"]: n["reason"] for n in manifest["not_converted"]}
    assert reasons["m1.hea"].startswith("multi-segment WFDB record")
    assert reasons["m1_1.hea"].startswith("WFDB header of no converted record")
    assert reasons["notes.csv"] == "no reader for .csv files"
    assert reasons["r1.at_"] == "WFDB annotator 'at_' is not listed in ANNOTATORS"
    assert reasons["r0.atr"] == "WFDB annotation file of no converted record"
    assert reasons["r1.xws"] == "no reader for .xws files"
    assert "r1.dat" not in reasons and "RECORDS" not in reasons
    assert manifest["problems"][-1].startswith("r3: the header names r3.dat, which the source")
    assert store.exists(f"{prefix}/_SUCCESS")
    assert not list(store.list(f"{prefix}/_parts/"))
    with pytest.raises(AlreadyStandard):
        plan_standard(store, "physionet/mitdb", "1.0.0")


def test_a_failed_recording_keeps_the_others_and_runs_again(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    from biosignal_data.standard import convert as convert_module  # noqa: PLC0415

    original = convert_module.EdfSource

    def broken(*args: object) -> None:
        raise OSError("simulated read failure")

    monkeypatch.setattr(convert_module, "EdfSource", broken)
    report = convert(
        plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2, part_bytes=1,
        work_dir=tmp_path / "work",
    )  # fmt: skip
    assert not report.complete and list(report.failed) == ["sub/s1"]
    assert not store.exists("standard-blocks/physionet/mitdb/1.0.0/_SUCCESS")

    monkeypatch.setattr(convert_module, "EdfSource", original)
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert [u.id for u in plan.to_convert] == ["sub/s1"]
    report = convert(plan, store, workers=2, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete and report.converted == 1 and report.skipped == 4


def test_discovery_without_lists_tries_unknown_extensions_as_annotations() -> None:
    paths = ["a/100.hea", "a/100.dat", "a/100.qrs", "a/100.txt", "b.edf", "b.edf.event"]
    found = discover(paths, read_text=lambda p: "", read_texts=lambda ps: dict.fromkeys(ps, ""))
    by_id = {u.id: u for u in found.units}
    assert by_id["a/100"].annotations == ("a/100.qrs",)
    assert not by_id["a/100"].listed_annotators
    assert by_id["b"].format == "edf" and by_id["b"].annotations == ("b.edf.event",)


def test_two_records_joined_on_one_records_line_are_both_found() -> None:
    """PTB-XL 1.0.3 lists `records100/21000/21837_lrrecords500/00000/00001_hr`."""
    paths = [
        "RECORDS", "r100/1_lr.hea", "r100/1_lr.dat", "r100/2_lr.hea", "r100/2_lr.dat",
        "r500/1_hr.hea", "r500/1_hr.dat", "x/ab.hea", "x/a.hea", "x/b.hea", "x/ab2.hea",
    ]  # fmt: skip
    records = "r100/1_lr\nr100/2_lrr500/1_hr\nx/ab\nmissing\n"
    found = discover(paths, lambda p: records, lambda ps: dict.fromkeys(ps, "r 0\n"))
    assert [u.id for u in found.units] == ["r100/1_lr", "r100/2_lr", "r500/1_hr", "x/ab"]
    assert found.problems == [
        "r100/2_lrr500/1_hr: two RECORDS entries without a line break between them; "
        "read as r100/2_lr and r500/1_hr",
        "missing: listed in RECORDS, but no such header or EDF file",
    ]


def test_small_files_that_hold_no_signals_are_kept_with_the_version(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert convert(plan, store, workers=1, part_bytes=1, work_dir=tmp_path / "work").complete
    ds = StandardDataset.open(store, "physionet/mitdb", "1.0.0")
    assert ds.file("r1.xws") == b"wave settings"  # not converted
    assert ds.file("notes.csv") == b"a,b\n1,2\n"
    with pytest.raises(FileNotFoundError):
        ds.file("r1.dat")  # signals are in blocks, not files/
    keys = list(store.list(plan.location.prefix + "/files/"))
    assert keys == [plan.location.object_key("files/pack-0000.bin")]  # one pack, not a file each


def test_kept_files_fill_packs_of_at_most_the_object_size(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biosignal_data.standard import kept_files  # noqa: PLC0415
    from biosignal_data.standard.layout import StandardLocation  # noqa: PLC0415

    monkeypatch.setattr(kept_files, "OBJECT_BYTES", 10)
    location = StandardLocation("physionet/x", "1.0.0")
    files = [("a.txt", b"12345"), ("b.txt", b"6789"), ("c.txt", b""), ("d.txt", b"abcdefgh")]
    table = kept_files.pack(store, location, files)
    assert table.column("object").to_pylist() == [
        "files/pack-0000.bin",
        "files/pack-0000.bin",
        "files/pack-0000.bin",
        "files/pack-0001.bin",
    ]
    rows = kept_files.index(store, location)
    assert [kept_files.read(store, location, p, rows) for p, _ in files] == [d for _, d in files]
    key = location.object_key("files/pack-0000.bin")
    store.put_bytes(key, b"12345X789")  # one byte changed
    with pytest.raises(ValueError, match="differs from its SHA-256"):
        kept_files.read(store, location, "b.txt")


def test_kept_files_are_read_ahead_only_up_to_a_bound(store: LocalStore) -> None:
    """ltstdb 1.0.0: 64 readers from AWS against one uploader held all kept files in memory
    (14 GB); the read-ahead is bounded in bytes, and many packs upload in order."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from biosignal_data.standard import kept_files  # noqa: PLC0415
    from biosignal_data.standard.convert import _read_ahead  # noqa: PLC0415
    from biosignal_data.standard.layout import StandardLocation  # noqa: PLC0415

    read: list[str] = []
    taken: list[str] = []
    ahead: list[int] = []

    def fetch(item: str) -> str:
        read.append(item)
        return item

    items = [f"f{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in _read_ahead(pool, fetch, items, [4] * 8, limit=10):
            ahead.append(len(read) - len(taken))
            taken.append(item)
    assert taken == items  # in order
    assert max(ahead) <= 2  # 2 files of 4 bytes fit in 10 bytes, a third does not
    location = StandardLocation("physionet/x", "1.0.0")
    files = [(f"f{i}", bytes([i]) * 7) for i in range(12)]  # 12 packs at 10 bytes each
    with pytest.MonkeyPatch.context() as m:
        m.setattr(kept_files, "OBJECT_BYTES", 10)
        table = kept_files.pack(store, location, files)
    assert table.column("object").to_pylist() == [f"files/pack-{i:04d}.bin" for i in range(12)]
    rows = kept_files.index(store, location)
    assert [kept_files.read(store, location, p, rows) for p, _ in files] == [d for _, d in files]


def test_every_worker_gets_a_batch_of_a_small_dataset(
    store: LocalStore, raw: dict[str, np.ndarray]
) -> None:
    from biosignal_data.standard.convert import _batches  # noqa: PLC0415

    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert len(_batches(plan, plan.units, 10**12, 1)) == 1  # one worker: one batch
    batches = _batches(plan, plan.units, 10**12, 3)  # far below part_bytes
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [u for b in batches for u in b] == plan.units  # in order, none left out


@pytest.mark.parametrize(
    ("channels", "length", "itemsize", "step", "per_block"),
    [
        (23, 2_000_000, 2, 1, 182_361),  # EEG: 8 MiB / (23 x 2 bytes)
        (12, 1000, 2, 1, 1000),  # a 10 s ECG: one block
        (2, 10**8, 4, 3, 1_048_575),  # whole WFDB frames of 3 samples
    ],
)
def test_blocks_hold_at_most_8_mib_in_whole_frames(
    channels: int, length: int, itemsize: int, step: int, per_block: int
) -> None:
    cuts = blocks.spans(length, channels, itemsize, step)
    assert cuts[0] == (0, min(per_block, length)) and cuts[-1][1] == length
    assert all(b - a <= per_block and a % step == 0 for a, b in cuts)
    assert all(cuts[i][1] == cuts[i + 1][0] for i in range(len(cuts) - 1))  # no gaps
    assert per_block * channels * itemsize <= blocks.BLOCK_BYTES


def test_objects_are_closed_before_64_mib_and_ranges_of_one_object_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blocks, "OBJECT_BYTES", 100)
    writer = blocks.ObjectWriter(tmp_path, "b")
    placed = [writer.add(bytes([i]) * 40) for i in range(5)]  # 40, 80 | 40, 80 | 40
    assert [p[0] for p in placed] == ["data/b-0000.bin"] * 2 + ["data/b-0001.bin"] * 2 + [
        "data/b-0002.bin"
    ]
    assert [p[1] for p in placed] == [0, 40, 0, 40, 0]
    files = dict(writer.objects)
    requests: list[tuple[str, int, int]] = []

    def get_range(name: str, offset: int, length: int) -> bytes:
        requests.append((name, offset, length))
        return files[name].read_bytes()[offset : offset + length]

    refs = [
        blocks.BlockRef(name, offset, 40, 0, 40, 1, "|u1", blocks.checksum(bytes([i]) * 40))
        for i, (name, offset) in enumerate(placed)
    ]
    found = blocks.fetch(get_range, refs, threads=2)
    assert all(found[r] == bytes([i]) * 40 for i, r in enumerate(refs))
    assert sorted(requests) == [  # one request per object, not per block
        ("data/b-0000.bin", 0, 80), ("data/b-0001.bin", 0, 80), ("data/b-0002.bin", 0, 40)
    ]  # fmt: skip


def test_a_damaged_block_is_found_by_its_checksum() -> None:
    blob = blocks.encode(np.arange(100, dtype=np.int16).reshape(2, 50))
    ref = blocks.BlockRef("o", 0, len(blob), 0, 50, 2, "<i2", blocks.checksum(blob))
    np.testing.assert_array_equal(blocks.unpack(ref, blob), np.arange(100).reshape(2, 50))
    with pytest.raises(blocks.CorruptBlockError):
        blocks.unpack(ref, blob[:-1] + bytes([blob[-1] ^ 1]))


def test_cli_dry_run_then_convert_then_already(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    from typer.testing import CliRunner  # noqa: PLC0415

    from biosignal_data.cli import app  # noqa: PLC0415

    config = tmp_path / "biosignal-data.yaml"
    config.write_text(
        f"standardize:\n  workers: 2\n  part_bytes: 1\n  processes: 0\n"
        f"staging:\n  dir: {tmp_path / 'staging'}\n  keep: false\n"
    )
    args = ["standardize", "physionet/mitdb", "--dest", str(store.root), "--config", str(config)]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "5 recordings (1 EDF, 4 WFDB), 2 annotation files; 5 to convert" in dry.output
    assert "newest complete raw version 1.0.0" in dry.output
    assert not store.exists("standard-blocks/physionet/mitdb/1.0.0/_SUCCESS")

    done = runner.invoke(app, args)
    assert done.exit_code == 0, done.output
    assert "4 recordings, 4 channels, 5 events" in done.output
    assert "not converted: 1 file: no reader for .csv files" in done.output
    assert "ZarrUserWarning" not in done.output
    assert "already complete (--replace makes it again)" in runner.invoke(app, args).output
    again = runner.invoke(app, [*args, "--replace", "--dry-run"])
    assert "(complete; --replace makes it again)" in again.output
    # a dry run changes nothing
    assert store.exists("standard-blocks/physionet/mitdb/1.0.0/_SUCCESS")
    replaced = runner.invoke(app, [*args, "--replace"])
    assert replaced.exit_code == 0, replaced.output
    assert "4 recordings, 4 channels, 5 events" in replaced.output
    assert store.exists("standard-blocks/physionet/mitdb/1.0.0/_SUCCESS")


def test_reader_opens_tables_and_reads_windows_in_physical_units(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    from biosignal_data.read import StandardDataset  # noqa: PLC0415

    convert(plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2, part_bytes=1,
            work_dir=tmp_path / "work")  # fmt: skip
    ds = StandardDataset.open(store, "physionet/mitdb", "latest")
    assert ds.location.version == "1.0.0"
    assert ds.recordings.num_rows == 4 and ds.manifest.recordings == 4
    stored = ds.window("r1", "200Hz", 1.0, 0.5, physical=False)
    np.testing.assert_array_equal(stored, raw["ecg"][None, 200:300])
    physical = ds.window("r1", "200Hz", 1.0, 0.5)
    np.testing.assert_allclose(physical, (raw["ecg"][None, 200:300] - 5) / 200)
    assert np.isnan(ds.window("r1", "100Hz", 0.0, 1.0)).all()  # not calibrated


def _write_edf(path: Path, values: list[np.ndarray], rate: int, bdf: bool = False) -> None:
    kind = pyedflib.FILETYPE_BDF if bdf else pyedflib.FILETYPE_EDF
    w = pyedflib.EdfWriter(str(path), len(values), file_type=kind)
    lo, hi = (-8388608, 8388607) if bdf else (-32768, 32767)
    w.setSignalHeaders(
        [
            {"label": f"C{i}", "dimension": "uV", "sample_frequency": rate,
             "physical_min": -3276.8, "physical_max": 3276.7,
             "digital_min": lo, "digital_max": hi}
            for i in range(len(values))
        ]
    )  # fmt: skip
    w.writeSamples([v.astype(np.int32) for v in values], digital=True)
    w.close()


def test_a_file_that_fails_to_open_does_not_change_another_open_file(tmp_path: Path) -> None:
    # pyedflib closed the handle of another open file here; that file then read a third one.
    from biosignal_data.standard.sources import EdfSource  # noqa: PLC0415

    a, c = np.full(1000, 111), np.full(1000, 333)
    _write_edf(tmp_path / "a.edf", [a], 100)
    _write_edf(tmp_path / "c.edf", [c], 100)
    (tmp_path / "broken.edf").write_bytes(b"0       not an EDF header")
    first = EdfSource("a.edf", lambda p: tmp_path / p)
    with pytest.raises(Exception):  # noqa: B017 - any error: the point is what follows
        EdfSource("broken.edf", lambda p: tmp_path / p)
    EdfSource("c.edf", lambda p: tmp_path / p)
    np.testing.assert_array_equal(first.read(0, 0, 1000)[0], a)


def test_bdf_values_are_exact_int32(tmp_path: Path) -> None:
    from biosignal_data.standard.sources import EdfSource  # noqa: PLC0415

    values = [RNG.integers(-8_000_000, 8_000_000, 2048 * 10) for _ in range(2)]
    _write_edf(tmp_path / "x.bdf", values, 2048, bdf=True)
    source = EdfSource("x.bdf", lambda p: tmp_path / p)
    assert source.format == "bdf" and source.groups[0].dtype == np.int32
    got = source.read(0, 2048, 2048 * 3)
    np.testing.assert_array_equal(got, np.stack([v[2048 : 2048 * 3] for v in values]))


def test_a_truncated_file_reads_the_records_it_holds_and_says_so(tmp_path: Path) -> None:
    from biosignal_data.standard.sources import EdfSource  # noqa: PLC0415

    values = RNG.integers(-30000, 30000, 100 * 60)
    _write_edf(tmp_path / "t.edf", [values], 100)  # 60 records of 1 s
    data = (tmp_path / "t.edf").read_bytes()
    (tmp_path / "t.edf").write_bytes(data[: len(data) - 100 * 2 * 20 - 50])  # 20.25 records cut
    source = EdfSource("t.edf", lambda p: tmp_path / p)
    assert source.groups[0].length == 100 * 39  # the whole records the file holds
    np.testing.assert_array_equal(source.read(0, 0, 100 * 39)[0], values[: 100 * 39])
    assert any("39 records" in problem for problem in source.problems)


def test_a_header_with_non_ascii_characters_is_read(tmp_path: Path) -> None:
    from biosignal_data.standard.sources import EdfSource  # noqa: PLC0415

    values = RNG.integers(-30000, 30000, 1000)
    _write_edf(tmp_path / "u.edf", [values], 100)
    data = bytearray((tmp_path / "u.edf").read_bytes())
    unit_at = 256 + 16 + 80  # first signal's physical dimension, after label and transducer
    data[unit_at : unit_at + 8] = "µV      ".encode("latin-1")
    (tmp_path / "u.edf").write_bytes(bytes(data))
    source = EdfSource("u.edf", lambda p: tmp_path / p)
    assert source.groups[0].channels[0].unit == "µV"
    np.testing.assert_array_equal(source.read(0, 0, 1000)[0], values)


def test_a_recording_fails_when_two_readers_of_its_file_disagree(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    # What pyedflib did: one reader returned another file's samples, without an error.
    from biosignal_data.standard import convert as convert_module  # noqa: PLC0415
    from biosignal_data.standard.sources import EdfSource  # noqa: PLC0415

    opened: list[int] = []

    class OneReaderIsWrong(EdfSource):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)  # type: ignore[arg-type]
            opened.append(1)
            self._wrong = len(opened) % 2 == 0

        def read(self, group: int, start: int, stop: int) -> np.ndarray:
            data = super().read(group, start, stop)
            return data + 1 if self._wrong else data

    monkeypatch.setattr(convert_module, "EdfSource", OneReaderIsWrong)
    report = convert(
        plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2, part_bytes=1,
        work_dir=tmp_path / "work",
    )  # fmt: skip
    assert not report.complete
    assert list(report.failed) == ["sub/s1"]
    assert "two readers of the file differ" in report.failed["sub/s1"]


@pytest.mark.parametrize("listed", [True, False])
def test_odd_annotation_sample_numbers_are_kept_if_listed_and_rejected_if_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listed: bool
) -> None:
    # capslpdb nfle10.edf.st: every sample number is negative (from -21,887,999).
    import types  # noqa: PLC0415

    from biosignal_data.standard import sources  # noqa: PLC0415

    fake = types.SimpleNamespace(
        sample=np.array([-512, -256, 100, 50]), symbol=['"'] * 4, aux_note=["S0"] * 4,
        subtype=[0] * 4, chan=[0] * 4, num=[0] * 4, fs=256,
    )  # fmt: skip
    monkeypatch.setattr(wfdb, "rdann", lambda *_: fake)
    (tmp_path / "r.edf.st").write_bytes(b"")
    fetch = lambda p: tmp_path / p  # noqa: E731
    if not listed:
        with pytest.raises(sources.UnsupportedError, match="2 of 4 sample numbers are negative"):
            sources.wfdb_annotations("r.edf.st", fetch, 256.0, listed=False)
        return
    events, problems = sources.wfdb_annotations("r.edf.st", fetch, 256.0, listed=True)
    assert [e.sample for e in events] == [-512, -256, 100, 50]
    assert [e.onset_s for e in events] == [-2.0, -1.0, 100 / 256, 50 / 256]
    assert problems == [
        "r.edf.st: 2 of 4 sample numbers are negative (kept as stored)",
        "r.edf.st: 1 sample numbers are lower than the one before (kept as stored)",
    ]


def test_a_record_that_shows_other_records_files_side_by_side_is_a_view() -> None:
    # apnea-ecg: a01er.hea shows a01.dat (record a01) and a01r.dat (record a01r).
    headers = {
        "a01.hea": "a01 1 100 10\na01.dat 16 200 12 0 0 0 0 ECG\n",
        "a01r.hea": "a01r 2 100 10\na01r.dat 16 1 16 0 0 0 0 R1\na01r.dat 16 1 16 0 0 0 0 R2\n",
        "a01er.hea": "a01er 3 100 10\na01.dat 16 200 12 0 0 0 0 ECG\n"
        "a01r.dat 16 1 16 0 0 0 0 R1\na01r.dat 16 1 16 0 0 0 0 R2\n",
        "x.hea": "x 2 100 10\nx1.dat 16 1 16 0 0 0 0 A\nx2.dat 16 1 16 0 0 0 0 B\n",
    }
    paths = [*headers, "a01.dat", "a01r.dat", "a01er.apn", "x1.dat", "x2.dat"]
    found = discover(paths, lambda p: "", lambda ps: {p: headers[p] for p in ps})
    assert sorted(u.id for u in found.units) == ["a01", "a01r", "x"]  # x: own two files
    view, owners = found.views["a01er"]
    assert owners == ("a01", "a01r") and view.annotations == ("a01er.apn",)


def test_a_wfdb_record_shorter_than_its_header_is_read_as_far_as_it_goes(tmp_path: Path) -> None:
    from biosignal_data.standard.sources import WfdbSource  # noqa: PLC0415

    held = RNG.integers(-2000, 2000, (900, 2)).astype(np.int16)  # the header says 1,000
    held.astype("<i2").tofile(tmp_path / "t.dat")
    (tmp_path / "t.hea").write_text(
        "t 2 100 1000\nt.dat 16 200 16 0 0 0 0 A\nt.dat 16 200 16 0 0 0 0 B\n"
    )
    source = WfdbSource("t", lambda p: tmp_path / p, lambda p: True)
    assert source.groups[0].length == 900 and source.duration_s == 9.0
    np.testing.assert_array_equal(source.read(0, 0, 900), held.T)
    assert source.problems == [
        "t: the header says 1,000 samples per signal, but t.dat holds 900; "
        "read as far as the files go (900)"
    ]


def test_a_signal_file_the_header_names_is_never_tried_as_annotations() -> None:
    """ptbdb 1.0.0: `s0010_re.xyz` holds the Frank leads; `.xyz` looks like an annotator."""
    header = "s 2 1000 100\ns.dat 16 2000 16 0 0 0 0 i\ns.xyz 16 2000 16 0 0 0 0 vx\n"
    paths = ["s.hea", "s.dat", "s.xyz", "s.atr"]
    found = discover(paths, lambda p: "", lambda ps: dict.fromkeys(ps, header))
    assert [u.annotations for u in found.units] == [("s.atr",)]


def test_a_record_name_with_a_folder_the_source_lacks_is_read_next_to_the_header(
    tmp_path: Path,
) -> None:
    """ptb-xl-plus 1.0.1 median beats: `ge_median_beats_wfdb/00001_medians` in the header, the
    files next to it."""
    from biosignal_data.standard.sources import WfdbSource  # noqa: PLC0415

    folder = tmp_path / "median_beats" / "12sl"
    folder.mkdir(parents=True)
    held = RNG.integers(-2000, 2000, (600, 2)).astype(np.int32)
    held.astype("<i4").tofile(folder / "00001_medians.dat")
    header = (
        "ge/00001_medians 2 500 600\n"
        "ge/00001_medians.dat 32 1000.5(-7)/mV 32 0 0 0 0 I\n"
        "ge/00001_medians.dat 32 1000.5(-7)/mV 32 0 0 0 0 II\n"
    )
    (folder / "00001_medians.hea").write_text(header)
    record = "median_beats/12sl/00001_medians"
    paths = [f"{record}.hea", f"{record}.dat"]
    found = discover(paths, lambda p: "", lambda ps: dict.fromkeys(ps, header))
    assert [u.id for u in found.units] == [record]
    source = WfdbSource(record, lambda p: tmp_path / p, lambda p: True)
    assert source.header == header  # kept as written
    assert source.files == [f"{record}.hea", f"{record}.dat"]
    assert source.groups[0].channels[0].offset == pytest.approx(7 / 1000.5)
    np.testing.assert_array_equal(source.read(0, 0, 600), held.T)
    assert source.problems == [
        f"{record}.hea: the record line names 'ge/00001_medians', in a folder 'ge' the source "
        "does not have; read as the record and signal files next to the header"
    ]


def test_a_view_is_listed_with_its_records_not_stored_twice(
    store: LocalStore, tmp_path: Path
) -> None:
    ecg = RNG.integers(-2000, 2000, 500).astype(np.int16)
    resp = RNG.integers(-2000, 2000, (500, 2)).astype(np.int16)
    files = {
        "a01.hea": b"a01 1 100 500\na01.dat 16 200 16 0 0 0 0 ECG\n",
        "a01.dat": ecg.astype("<i2").tobytes(),
        "a01r.hea": b"a01r 2 100 500\na01r.dat 16 1 16 0 0 0 0 R1\na01r.dat 16 1 16 0 0 0 0 R2\n",
        "a01r.dat": resp.astype("<i2").tobytes(),
        "a01er.hea": b"a01er 3 100 500\na01.dat 16 200 16 0 0 0 0 ECG\n"
        b"a01r.dat 16 1 16 0 0 0 0 R1\na01r.dat 16 1 16 0 0 0 0 R2\n",
    }
    dataset = make_dataset(id="physionet/apnea-ecg")
    ingest(dataset, store, fetcher=FakeFetcher(dataset, files), workers=2)
    plan = plan_standard(store, "physionet/apnea-ecg", "1.0.0")
    assert sorted(u.id for u in plan.units) == ["a01", "a01r"]
    report = convert(plan, store, workers=2, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete and report.converted == 2
    manifest = json.loads(
        store.get_bytes("standard-blocks/physionet/apnea-ecg/1.0.0/_manifest.json") or b""
    )
    reasons = {n["path"]: n["reason"] for n in manifest["not_converted"]}
    assert reasons["a01er.hea"].startswith("a view of a01, a01r: its header shows")


def test_small_recordings_share_one_part_and_a_stopped_run_continues(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    from biosignal_data.standard import convert as convert_module  # noqa: PLC0415

    original = convert_module.EdfSource

    def broken(*args: object) -> None:
        raise OSError("simulated read failure")

    convert_module.EdfSource = broken  # type: ignore[assignment,misc]
    try:  # the EDF recording fails; the WFDB ones are done
        first = convert(plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2,
                        part_bytes=10**9, work_dir=tmp_path / "work")  # fmt: skip
    finally:
        convert_module.EdfSource = original  # type: ignore[misc]
    assert not first.complete and list(first.failed) == ["sub/s1"]

    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert [u.id for u in plan.to_convert] == ["sub/s1"]
    report = convert(plan, store, workers=2, part_bytes=10**9, work_dir=tmp_path / "work")
    assert report.complete and report.converted == 1 and report.skipped == 4
    location = plan.location
    recordings = pq.read_table(store._path(location.table_key("recordings"))).to_pylist()
    # m1 (multi-segment) is not converted yet, as in the main test: no row
    assert [r["recording_id"] for r in recordings] == ["r1", "r2", "r3", "sub/s1"]
    assert not list(store.list(location.parts_prefix))
    ds = StandardDataset.open(store, "physionet/mitdb", "1.0.0")
    np.testing.assert_array_equal(ds.read("sub/s1", "256Hz"), raw["eeg"][None, :])


@pytest.mark.parametrize(
    "module",
    ["biosignal_data.standard.convert", "biosignal_data.read", "biosignal_data.index",
     "biosignal_data.training", "biosignal_data.quality"],
)  # fmt: skip
def test_each_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    # importing biosignal_data.standard.convert first failed on an import cycle (2026-10-20)
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"], check=False, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def _psg(path: Path, start: datetime) -> None:
    w = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDF)
    w.setSignalHeaders(
        [{"label": "EEG Fpz-Cz", "dimension": "uV", "sample_frequency": 100,
          "physical_min": -3276.8, "physical_max": 3276.7,
          "digital_min": -32768, "digital_max": 32767}]
    )  # fmt: skip
    w.setStartdatetime(start)
    w.writeSamples([RNG.integers(-1000, 1000, 100 * 90).astype(np.int32)], digital=True)
    w.close()


def _hypnogram(path: Path, start: datetime) -> None:
    import edfio  # noqa: PLC0415

    stages = [edfio.EdfAnnotation(0, 30, "Sleep stage W"),
              edfio.EdfAnnotation(30, 60, "Sleep stage 1")]  # fmt: skip
    edf = edfio.Edf([], annotations=stages, starttime=start.time())
    edf.startdate = start.date()
    edf.write(path)


EDFX_RULES = StandardRules(
    datasets=[
        DatasetRules(
            id="physionet/sleep-edfx",
            edf_annotation_files=EdfAnnotationFiles(
                recording_suffix="-PSG.edf", annotation_suffix="-Hypnogram.edf",
                same_first_characters=7, source="https://physionet.org/content/sleep-edfx/1.0.0/",
                evidence="Files are named in the form SC4ssNEO-PSG.edf",
            ),
        )
    ]
)  # fmt: skip


def _sleep_edfx(store: LocalStore, tmp_path: Path, hypnogram_start: datetime) -> None:
    night = datetime(1989, 4, 24, 16, 13)
    _psg(tmp_path / "psg.edf", night)
    _hypnogram(tmp_path / "hyp.edf", hypnogram_start)
    files = {
        "RECORDS": b"SC4001E0-PSG.edf\n",
        "SC4001E0-PSG.edf": (tmp_path / "psg.edf").read_bytes(),
        "SC4001EC-Hypnogram.edf": (tmp_path / "hyp.edf").read_bytes(),
        "SC4003EC-Hypnogram.edf": (tmp_path / "hyp.edf").read_bytes(),  # no such PSG
    }
    dataset = make_dataset(id="physionet/sleep-edfx")
    ingest(dataset, store, fetcher=FakeFetcher(dataset, files), workers=2)


def test_hypnograms_are_the_events_of_the_psg_the_dataset_rule_pairs_them_with(
    store: LocalStore, tmp_path: Path
) -> None:
    """sleep-edfx: SC4001EC-Hypnogram.edf (annotations only) belongs to SC4001E0-PSG.edf."""
    _sleep_edfx(store, tmp_path, datetime(1989, 4, 24, 16, 13))
    plan = plan_standard(store, "physionet/sleep-edfx", "1.0.0", rules=EDFX_RULES)
    assert {u.id: u.edf_annotations for u in plan.units} == {
        "SC4001E0-PSG": ("SC4001EC-Hypnogram.edf",)
    }
    assert plan.discovery.problems == [
        "SC4003EC-Hypnogram.edf: the standard.yaml rule pairs it with 0 recordings (none); left out"
    ]
    report = convert(plan, store, workers=1, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete
    ds = StandardDataset.open(store, "physionet/sleep-edfx", "1.0.0")
    events = [e for e in ds.events.to_pylist() if e["file"] == "SC4001EC-Hypnogram.edf"]
    assert [(e["recording_id"], e["onset_s"], e["duration_s"], e["label"]) for e in events] == [
        ("SC4001E0-PSG", 0.0, 30.0, "Sleep stage W"),
        ("SC4001E0-PSG", 30.0, 60.0, "Sleep stage 1"),
    ]
    recording = ds.recordings.to_pylist()[0]
    assert "SC4001EC-Hypnogram.edf" in recording["source_files"]


def test_a_paired_annotation_file_that_starts_at_another_time_fails_its_recording(
    store: LocalStore, tmp_path: Path
) -> None:
    _sleep_edfx(store, tmp_path, datetime(1989, 4, 24, 16, 14))
    plan = plan_standard(store, "physionet/sleep-edfx", "1.0.0", rules=EDFX_RULES)
    report = convert(plan, store, workers=1, part_bytes=1, work_dir=tmp_path / "work")
    assert list(report.failed) == ["SC4001E0-PSG"]
    assert (
        "starts at 1989-04-24 16:14:00, its recording at 1989-04-24 16:13:00"
        in (report.failed["SC4001E0-PSG"])
    )


def test_standard_yaml_is_valid() -> None:
    rules = StandardRules.load()
    assert rules.get("physionet/sleep-edfx") is not None
