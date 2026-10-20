from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyedflib
import pytest
import wfdb
import zarr

from biosignal_data.ingest import ingest
from biosignal_data.standard import arrays, tables
from biosignal_data.standard.convert import (
    AlreadyStandard,
    _batches,
    convert,
    convert_unit,
    plan_standard,
)
from biosignal_data.standard.discover import discover
from biosignal_data.standard.sources import Group
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


def test_a_new_attempt_removes_what_an_earlier_one_left_in_a_group(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    stale = "standard/physionet/mitdb/1.0.0/signals/r1.zarr/200Hz/c/0/99"
    store.put_bytes(stale, b"from an earlier attempt")
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    report = convert(plan, store, workers=1, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete
    assert not store.exists(stale)
    assert list((tmp_path / "work").iterdir()) == []  # the local groups are removed


def test_standard_version_is_exact_and_accounts_for_every_file(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert [u.id for u in plan.units] == ["m1", "r1", "r2", "r3", "sub/s1"]
    assert plan.discovery.problems == ["missing: listed in RECORDS, but no such header or EDF file"]

    report = convert(plan, store, workers=2, part_bytes=1, work_dir=tmp_path / "work")
    assert report.complete and not report.failed and report.converted == 5

    prefix = "standard/physionet/mitdb/1.0.0"
    root = zarr.open_group(
        store=store.zarr_store(f"{prefix}/signals/r1.zarr", read_only=True), mode="r"
    )
    assert sorted(root.array_keys()) == ["100Hz", "200Hz"]
    np.testing.assert_array_equal(root["200Hz"][:], raw["ecg"][None, :])
    np.testing.assert_array_equal(root["100Hz"][:], raw["x"][None, :])
    assert root["200Hz"].dtype == np.int16
    assert root.attrs["recording_id"] == "r1"

    edf = zarr.open_group(
        store=store.zarr_store(f"{prefix}/signals/sub/s1.zarr", read_only=True), mode="r"
    )
    np.testing.assert_array_equal(edf["256Hz"][:], raw["eeg"][None, :])
    np.testing.assert_array_equal(edf["1Hz"][:], raw["slow"][None, :])

    recordings = pq.read_table(store._path(f"{prefix}/recordings.parquet")).to_pylist()
    assert [r["recording_id"] for r in recordings] == ["r1", "r2", "r3", "sub/s1"]
    assert (recordings[2]["n_channels"], recordings[2]["signals"]) == (0, None)
    r1, r2 = recordings[0], recordings[1]
    assert (r2["n_channels"], r2["signals"], r2["arrays"]) == (0, None, [])
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
    assert not store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")

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


@pytest.mark.parametrize(
    ("rate", "channels", "length", "step", "chunk"),
    [
        (256.0, 23, 921_600, 1, 7680),  # one 30 s epoch is already 353 KB
        (1.0, 3, 28_800, 1, 30 * 365),  # 30 samples per epoch: grown to 64 KiB
        (200.0, 1, 100, 2, 100),  # shorter than one chunk
    ],
)
def test_chunks_are_whole_epochs_of_at_least_64_kib(
    rate: float, channels: int, length: int, step: int, chunk: int
) -> None:
    group = Group(rate, np.dtype("int16"), length, tuple([None] * channels), step)  # type: ignore[list-item]
    shape = arrays.layout(group)
    assert shape.chunk == chunk
    assert shape.shard % shape.chunk == 0
    assert shape.shard * channels * 2 <= max(arrays.MAX_SHARD_BYTES, shape.chunk * channels * 2)


def test_cli_dry_run_then_convert_then_already(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    from typer.testing import CliRunner  # noqa: PLC0415

    from biosignal_data.cli import app  # noqa: PLC0415

    config = tmp_path / "biosignal-data.yaml"
    config.write_text(
        f"standardize:\n  workers: 2\n  part_bytes: 1\n"
        f"staging:\n  dir: {tmp_path / 'staging'}\n  keep: false\n"
    )
    args = ["standardize", "physionet/mitdb", "--dest", str(store.root), "--config", str(config)]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "5 recordings (1 EDF, 4 WFDB), 2 annotation files; 5 to convert" in dry.output
    assert "newest complete raw version 1.0.0" in dry.output
    assert not store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")

    done = runner.invoke(app, args)
    assert done.exit_code == 0, done.output
    assert "4 recordings, 4 channels, 5 events" in done.output
    assert "not converted: 1 file: no reader for .csv files" in done.output
    assert "ZarrUserWarning" not in done.output
    assert "already complete (--replace makes it again)" in runner.invoke(app, args).output
    again = runner.invoke(app, [*args, "--replace", "--dry-run"])
    assert "(complete; --replace makes it again)" in again.output
    assert store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")  # a dry run changes nothing
    replaced = runner.invoke(app, [*args, "--replace"])
    assert replaced.exit_code == 0, replaced.output
    assert "4 recordings, 4 channels, 5 events" in replaced.output
    assert store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")


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
        store.get_bytes("standard/physionet/apnea-ecg/1.0.0/_manifest.json") or b""
    )
    reasons = {n["path"]: n["reason"] for n in manifest["not_converted"]}
    assert reasons["a01er.hea"].startswith("a view of a01, a01r: its header shows")


def test_parts_written_before_problems_were_recorded_still_finish(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    # autonomic-aging-cardiovascular: 1,120 parts from earlier code had no "problems" key.
    from biosignal_data.standard import convert as convert_module  # noqa: PLC0415

    original = convert_module.EdfSource

    def broken(*args: object) -> None:
        raise OSError("simulated read failure")

    convert_module.EdfSource = broken  # type: ignore[assignment,misc]
    try:
        convert(plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2, part_bytes=1,
                work_dir=tmp_path / "work")  # fmt: skip
    finally:
        convert_module.EdfSource = original  # type: ignore[misc]
    parts = "standard/physionet/mitdb/1.0.0/_parts/"
    done_key = next(
        k for k in store.list(parts)
        if k.endswith("/done.json") and "r1" in json.loads(store.get_bytes(k) or b"")["units"]
    )  # fmt: skip
    done = json.loads(store.get_bytes(done_key) or b"")
    del done["problems"]  # as the earlier code wrote it
    store.put_bytes(done_key, json.dumps(done).encode())
    report = convert(
        plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2, part_bytes=1,
        work_dir=tmp_path / "work",
    )  # fmt: skip
    assert report.complete


def test_small_recordings_share_one_part_and_old_parts_still_finish(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    first = plan.units[0]
    # an earlier run left the first record as a per-recording part (before 2026-10-20)
    (tmp_path / "work").mkdir()
    result = convert_unit(plan, store, first, tmp_path / "work")
    location = plan.location
    for name, table in result.parts.items():
        store.put_bytes(location.part_key(first.id, f"{name}.parquet"), tables.to_parquet(table))
    store.put_bytes(location.part_key(first.id, "done.json"), json.dumps(
        {"unit": first.id, "tables": sorted(result.parts), "used": result.used,
         "not_converted": [], "problems": []}).encode())  # fmt: skip

    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert first.id not in [u.id for u in plan.to_convert] and len(plan.to_convert) == 4
    assert len(_batches(plan, plan.to_convert, 10**9)) == 1  # the other four share one part
    report = convert(plan, store, workers=2, part_bytes=10**9, work_dir=tmp_path / "work")

    assert report.complete and report.converted == 4 and report.skipped == 1
    recordings = pq.read_table(store._path(location.table_key("recordings"))).to_pylist()
    # m1 (multi-segment) is not converted yet, as in the main test: no row
    assert [r["recording_id"] for r in recordings] == ["r1", "r2", "r3", "sub/s1"]
    assert not list(store.list(location.parts_prefix))


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
