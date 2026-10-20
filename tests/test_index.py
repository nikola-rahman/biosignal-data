from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biosignal_data.index import AlreadyIndexed, IndexRules, build_index, signal_type, split_of
from biosignal_data.index.rules import RulesError, SubjectRule
from biosignal_data.index.signals import SignalType
from biosignal_data.index.subjects import subject_of
from biosignal_data.standard import tables
from biosignal_data.storage.stores import LocalStore


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        # Names as they appear in the converted PhysioNet datasets.
        ("Fp2-F4", "eeg"), ("C3A2", "eeg"), ("O2-A1", "eeg"), ("EEG Fpz-Cz", "eeg"), ("A1", "eeg"),
        ("LOC / A2", "eog"), ("ROC-LOC", "eog"), ("EOG DX", "eog"),
        ("EMG1-EMG2", "emg"), ("CHIN1", "emg"), ("DX1-DX2", "emg"), ("TIB SIN", "emg"),
        ("ECG1-ECG2", "ecg"), ("EKG", "ecg"), ("MLII", "ecg"), ("aVR", "ecg"), ("V5", "ecg"),
        ("II", "ecg"), ("Resp C", "resp"), ("TORACE", "resp"), ("FLUSSO", "resp"),
        ("SAO2", "spo2"), ("SpO2", "spo2"), ("OX", "spo2"), ("OX STATUS", "other"),
        ("PLETH", "ppg"), ("NIBP", "bp"), ("HR", "hr"), ("HEART RATE VARIA", "hr"),
        ("Position", "position"), ("POSIZIONE", "position"), ("ACCx", "accel"), ("SCG", "scg"),
        ("MIC", "sound"), ("STAT", "other"), ("THE", "unknown"),
    ],
)  # fmt: skip
def test_signal_types_of_real_channel_names(name: str, kind: str) -> None:
    found, decided_by = signal_type(name)
    assert found == SignalType(kind)
    assert decided_by == ("none" if kind == "unknown" else "name")


def test_a_dataset_rule_goes_before_the_general_rules() -> None:
    # adfecgdb: Abdomen_1 is abdominal (fetal) ECG; the general rule would say breathing.
    assert signal_type("Abdomen_1") == (SignalType.RESP, "name")
    override = [(re.compile(r"^Abdomen_\d$"), SignalType.ECG)]
    assert signal_type("Abdomen_1", override) == (SignalType.ECG, "dataset")


def _rule(**fields: object) -> SubjectRule:
    base: dict[str, object] = {"certain": True, "source": "https://physionet.org/", "evidence": "q"}
    return SubjectRule.model_validate(base | fields)


def test_subjects_from_a_pattern_with_one_person_under_two_names() -> None:
    rule = _rule(rule="pattern", pattern=r"^(chb\d+)/", same_person=[["chb01", "chb21"]])
    assert subject_of("physionet/chbmit", "chb04/chb04_02", rule) == (
        "physionet/chbmit:chb04",
        True,
    )
    assert subject_of("physionet/chbmit", "chb21/chb21_01", rule)[0] == "physionet/chbmit:chb01"


def test_without_a_rule_each_recording_is_its_own_subject_and_not_certain() -> None:
    assert subject_of("x/y", "r1", None) == ("x/y:r1", False)
    assert subject_of("x/y", "r1", _rule(rule="one_per_recording")) == ("x/y:r1", True)


def test_splits_are_stable_and_close_to_80_10_10() -> None:
    assert split_of("physionet/mitdb:100") == split_of("physionet/mitdb:100")
    counts = Counter(split_of(f"d/x:{i}") for i in range(20_000))
    assert abs(counts["train"] / 20_000 - 0.8) < 0.02
    assert abs(counts["validation"] / 20_000 - 0.1) < 0.02
    assert abs(counts["test"] / 20_000 - 0.1) < 0.02


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"rule": "pattern", "pattern": r"^chb\d+"}, "needs a group"),
        ({"rule": "pattern"}, "exactly when rule is pattern"),
        ({"rule": "unknown"}, "cannot be certain"),
        ({"rule": "one_per_recording", "evidence": None}, "needs `evidence`"),
    ],
)
def test_invalid_subject_rules_are_rejected(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _rule(**fields)


def _parquet(table: pa.Table) -> bytes:
    out = io.BytesIO()
    pq.write_table(table, out)
    return out.getvalue()


def test_the_index_points_at_the_standard_layer_with_subjects_splits_and_types(
    store: LocalStore, tmp_path: Path
) -> None:
    prefix = "standard/physionet/chbmit/1.0.0"
    recordings = pa.Table.from_pylist(
        [
            {"recording_id": "chb01/chb01_01", "format": "edf", "duration_s": 3600.0,
             "arrays": ["256Hz"], "n_channels": 2, "signals": "signals/chb01/chb01_01.zarr",
             "source_files": [], "source_sha256": []},
            {"recording_id": "chb21/chb21_01", "format": "edf", "duration_s": 3600.0,
             "arrays": ["256Hz"], "n_channels": 2, "signals": "signals/chb21/chb21_01.zarr",
             "source_files": [], "source_sha256": []},
            {"recording_id": "chb02/notes", "format": "edf", "duration_s": None, "arrays": [],
             "n_channels": 0, "signals": None, "source_files": [], "source_sha256": []},
        ],
        schema=tables.RECORDINGS,
    )  # fmt: skip
    channels = pa.Table.from_pylist(
        [
            {"recording_id": rec, "array": "256Hz", "index": i, "name": name, "unit": "uV",
             "sampling_rate": 256.0, "dtype": "int16", "scale": 0.1, "offset": 0.0}
            for rec in ("chb01/chb01_01", "chb21/chb21_01")
            for i, name in enumerate(["FP1-F7", "VNS"])
        ],
        schema=tables.CHANNELS,
    )  # fmt: skip
    store.put_bytes(f"{prefix}/recordings.parquet", _parquet(recordings))
    store.put_bytes(f"{prefix}/channels.parquet", _parquet(channels))
    store.put_bytes(f"{prefix}/_manifest.json", b'{"raw_version": "1.0.0"}')
    store.put_bytes(f"{prefix}/_SUCCESS", b"")
    rules_path = tmp_path / "index.yaml"
    rules_path.write_text(
        "datasets:\n"
        "  - id: physionet/chbmit\n"
        "    subject:\n"
        "      rule: pattern\n"
        "      pattern: '^(chb\\d+)/'\n"
        "      same_person: [[chb01, chb21]]\n"
        "      certain: true\n"
        "      source: https://physionet.org/content/chbmit/1.0.0/\n"
        "      evidence: same female subject\n"
    )

    report = build_index(
        store, ["physionet/chbmit", "physionet/mitdb"], IndexRules.load(rules_path),
        version="2026-10-20",
    )  # fmt: skip

    m = report.manifest
    assert m.datasets == {"physionet/chbmit": "1.0.0"}
    assert m.left_out == {"physionet/mitdb": "no complete standard version"}
    assert (m.recordings, m.recordings_without_signals, m.subjects) == (2, 1, 1)
    assert m.unknown_channels == {"physionet/chbmit: VNS": 2}
    rows = pq.read_table(store._path("index/2026-10-20/recordings.parquet")).to_pylist()
    assert {r["subject_id"] for r in rows} == {"physionet/chbmit:chb01"}  # one person
    assert len({r["split"] for r in rows}) == 1  # a person is in one split only
    assert rows[0]["signals"] == f"{prefix}/signals/chb01/chb01_01.zarr"
    types = pq.read_table(store._path("index/2026-10-20/channels.parquet")).to_pylist()
    assert [(c["name"], c["signal_type"]) for c in types][:2] == [
        ("FP1-F7", "eeg"),
        ("VNS", "unknown"),
    ]
    assert json.loads(store.get_bytes("index/2026-10-20/_manifest.json") or b"")["subjects"] == 1
    with pytest.raises(AlreadyIndexed, match="--replace"):
        build_index(store, ["physionet/chbmit"], IndexRules.load(rules_path), version="2026-10-20")


def test_index_rules_file_errors_name_the_entry(tmp_path: Path) -> None:
    path = tmp_path / "index.yaml"
    path.write_text(
        "datasets:\n  - id: a/b\n    subject: {rule: pattern, certain: false, source: x}\n"
    )
    with pytest.raises(RulesError, match="a/b"):
        IndexRules.load(path)


def test_a_table_rule_takes_the_subject_from_a_table() -> None:
    rule = _rule(
        rule="table", pattern=r"/(ECGPCG\d{4})$",
        table={"file": "s.csv", "delimiter": ",", "recording_column": "Record Name",
               "subject_column": "Subject ID"},
    )  # fmt: skip
    table = {"ECGPCG0013": "S002", "ECGPCG0049": "S002"}
    assert subject_of("p/e", "WFDB/ECGPCG0049", rule, table) == ("p/e:S002", True)
    with pytest.raises(ValueError, match="not in the subject table"):
        subject_of("p/e", "WFDB/ECGPCG0001", rule, table)
    with pytest.raises(ValueError, match="exactly when rule is table"):
        _rule(rule="pattern", pattern="(x)", table=rule.table)


def test_an_exclude_rule_needs_evidence_or_a_check(tmp_path: Path) -> None:
    path = tmp_path / "index.yaml"
    path.write_text(
        "datasets:\n  - id: a/b\n    exclude:\n"
        "      - {pattern: '^x_', reason: copies, source: 'https://physionet.org/'}\n"
    )
    with pytest.raises(RulesError, match="needs `evidence` or `checked`"):
        IndexRules.load(path)


def test_excluded_recordings_and_subjects_from_a_raw_table(
    store: LocalStore, tmp_path: Path
) -> None:
    prefix = "standard/physionet/ephnogram/1.0.0"
    rows = [
        {"recording_id": rec, "format": "wfdb", "duration_s": 60.0, "arrays": ["8000Hz"],
         "n_channels": 1, "signals": f"signals/{rec}.zarr", "source_files": [],
         "source_sha256": []}
        for rec in ("WFDB/ECGPCG0001", "WFDB/ECGPCG0002", "WFDB/x_ECGPCG0002")
    ]  # fmt: skip
    store.put_bytes(
        f"{prefix}/recordings.parquet",
        _parquet(pa.Table.from_pylist(rows, schema=tables.RECORDINGS)),
    )
    store.put_bytes(
        f"{prefix}/channels.parquet",
        _parquet(pa.Table.from_pylist([], schema=tables.CHANNELS)),
    )
    store.put_bytes(f"{prefix}/_manifest.json", b'{"raw_version": "1.0.0"}')
    store.put_bytes(f"{prefix}/_SUCCESS", b"")
    store.put_bytes(
        "raw/physionet/ephnogram/1.0.0/data/t.csv",
        "﻿Record Name,Subject ID\nECGPCG0001,S001\nECGPCG0002,S001\n".encode(),
    )
    rules_path = tmp_path / "index.yaml"
    rules_path.write_text(
        "datasets:\n"
        "  - id: physionet/ephnogram\n"
        "    subject:\n"
        "      rule: table\n"
        "      pattern: '^WFDB/(ECGPCG\\d{4})$'\n"
        "      table: {file: t.csv, delimiter: ',', recording_column: Record Name,\n"
        "              subject_column: Subject ID}\n"
        "      certain: true\n"
        "      source: https://physionet.org/content/ephnogram/1.0.0/\n"
        "      evidence: the unique subject IDs are provided in the spreadsheet\n"
        "    exclude:\n"
        "      - pattern: '/x_'\n"
        "        reason: an excerpt of another recording\n"
        "        source: https://physionet.org/content/ephnogram/1.0.0/\n"
        "        checked: samples equal\n"
    )

    m = build_index(
        store, ["physionet/ephnogram"], IndexRules.load(rules_path), version="2026-10-20"
    ).manifest

    assert (m.recordings, m.subjects, m.subjects_assumed) == (2, 1, 0)
    assert m.excluded == {"physionet/ephnogram: an excerpt of another recording": 1}


def test_benchmark_test_records_are_held_out_of_every_split(
    store: LocalStore, tmp_path: Path
) -> None:
    prefix = "standard/physionet/apnea-ecg/1.0.0"
    rows = [
        {"recording_id": rec, "format": "wfdb", "duration_s": 60.0, "arrays": ["100Hz"],
         "n_channels": 1, "signals": f"signals/{rec}.zarr", "source_files": [],
         "source_sha256": []}
        for rec in ("a01", "a01r", "x01", "x01r")
    ]  # fmt: skip
    store.put_bytes(
        f"{prefix}/recordings.parquet",
        _parquet(pa.Table.from_pylist(rows, schema=tables.RECORDINGS)),
    )
    store.put_bytes(
        f"{prefix}/channels.parquet", _parquet(pa.Table.from_pylist([], schema=tables.CHANNELS))
    )
    store.put_bytes(f"{prefix}/_manifest.json", b'{"raw_version": "1.0.0"}')
    store.put_bytes(f"{prefix}/_SUCCESS", b"")
    rules_path = tmp_path / "index.yaml"
    rules_path.write_text(
        "datasets:\n"
        "  - id: physionet/apnea-ecg\n"
        "    subject: {rule: pattern, pattern: '^([abcx]\\d{2})r?$', certain: false,\n"
        "              source: 'https://physionet.org/content/apnea-ecg/1.0.0/'}\n"
        "    held_out:\n"
        "      - pattern: '^x\\d{2}r?$'\n"
        "        benchmark: apnea\n"
        "        source: https://physionet.org/content/apnea-ecg/1.0.0/\n"
        "        evidence: a test set of 35 records (x01 through x35)\n"
    )

    m = build_index(
        store, ["physionet/apnea-ecg"], IndexRules.load(rules_path), version="v"
    ).manifest

    assert m.held_out == {"apnea": 2}
    rows_out = pq.read_table(store._path("index/v/recordings.parquet")).to_pylist()
    split = {r["recording_id"]: (r["split"], r["held_out_for"]) for r in rows_out}
    assert split["x01"] == split["x01r"] == ("held_out", "apnea")
    assert split["a01"][0] in {"train", "validation", "test"} and split["a01"][1] is None
