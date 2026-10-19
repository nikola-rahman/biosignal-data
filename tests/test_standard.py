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
from biosignal_data.standard import arrays
from biosignal_data.standard.convert import AlreadyStandard, convert, plan_standard
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
        "RECORDS": b"r1\nr2\nm1\nsub/s1.edf\nmissing\n",
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


def test_standard_version_is_exact_and_accounts_for_every_file(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert [u.id for u in plan.units] == ["m1", "r1", "r2", "sub/s1"]
    assert plan.discovery.problems == ["missing: listed in RECORDS, but no such header or EDF file"]

    report = convert(plan, store, workers=2, work_dir=tmp_path / "work")
    assert report.complete and not report.failed and report.converted == 4

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
    assert [r["recording_id"] for r in recordings] == ["r1", "r2", "sub/s1"]
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
    report = convert(plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2,
                     work_dir=tmp_path / "work")  # fmt: skip
    assert not report.complete and list(report.failed) == ["sub/s1"]
    assert not store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")

    monkeypatch.setattr(convert_module, "EdfSource", original)
    plan = plan_standard(store, "physionet/mitdb", "1.0.0")
    assert [u.id for u in plan.to_convert] == ["sub/s1"]
    report = convert(plan, store, workers=2, work_dir=tmp_path / "work")
    assert report.complete and report.converted == 1 and report.skipped == 3


def test_discovery_without_lists_tries_unknown_extensions_as_annotations() -> None:
    paths = ["a/100.hea", "a/100.dat", "a/100.qrs", "a/100.txt", "b.edf", "b.edf.event"]
    found = discover(paths, read_text=lambda p: "")
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
        f"standardize:\n  workers: 2\nstaging:\n  dir: {tmp_path / 'staging'}\n  keep: false\n"
    )
    args = ["standardize", "physionet/mitdb", "--dest", str(store.root), "--config", str(config)]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "4 recordings (1 EDF, 3 WFDB), 2 annotation files; 4 to convert" in dry.output
    assert "newest complete raw version 1.0.0" in dry.output
    assert not store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")

    done = runner.invoke(app, args)
    assert done.exit_code == 0, done.output
    assert "3 recordings, 4 channels, 5 events" in done.output
    assert "not converted: 1 file: no reader for .csv files" in done.output
    assert "ZarrUserWarning" not in done.output
    assert "already complete (--replace makes it again)" in runner.invoke(app, args).output
    again = runner.invoke(app, [*args, "--replace", "--dry-run"])
    assert "(complete; --replace makes it again)" in again.output
    assert store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")  # a dry run changes nothing
    replaced = runner.invoke(app, [*args, "--replace"])
    assert replaced.exit_code == 0, replaced.output
    assert "3 recordings, 4 channels, 5 events" in replaced.output
    assert store.exists("standard/physionet/mitdb/1.0.0/_SUCCESS")


def test_reader_opens_tables_and_reads_windows_in_physical_units(
    store: LocalStore, tmp_path: Path, raw: dict[str, np.ndarray]
) -> None:
    from biosignal_data.read import StandardDataset  # noqa: PLC0415

    convert(plan_standard(store, "physionet/mitdb", "1.0.0"), store, workers=2,
            work_dir=tmp_path / "work")  # fmt: skip
    ds = StandardDataset.open(store, "physionet/mitdb", "latest")
    assert ds.location.version == "1.0.0"
    assert ds.recordings.num_rows == 3 and ds.manifest.recordings == 3
    stored = ds.window("r1", "200Hz", 1.0, 0.5, physical=False)
    np.testing.assert_array_equal(stored, raw["ecg"][None, 200:300])
    physical = ds.window("r1", "200Hz", 1.0, 0.5)
    np.testing.assert_allclose(physical, (raw["ecg"][None, 200:300] - 5) / 200)
    assert np.isnan(ds.window("r1", "100Hz", 0.0, 1.0)).all()  # not calibrated
