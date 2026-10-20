"""Build the index: the small, stored part of the training layer.

    index/<date>/
        recordings.parquet   every recording with signals: dataset, subject, split, duration
        channels.parquet     every channel: name as written, signal type and what decided it
        _manifest.json       the standard versions used, the rules, the counts, unknown names
        _SUCCESS

The signals themselves are not copied: the index points at the Zarr arrays of the standard
layer, which a data loader reads (see training/). An index is immutable once complete.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from biosignal_data import __version__
from biosignal_data.index.rules import (
    DatasetRules,
    IndexRules,
    SubjectTable,
    channel_overrides,
)
from biosignal_data.index.signals import signal_type
from biosignal_data.index.subjects import split_of, subject_of
from biosignal_data.read.raw import complete_versions
from biosignal_data.standard.layout import ROOT as STANDARD_ROOT
from biosignal_data.standard.layout import StandardLocation
from biosignal_data.storage.layout import SUCCESS, VersionLocation, retrieval_version
from biosignal_data.storage.manifest import utcnow
from biosignal_data.storage.stores import Store

log = logging.getLogger(__name__)
ROOT = "index"
SCHEMA_VERSION = 1

RECORDINGS = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("version", pa.string()),  # the standard (= raw) version
        ("recording_id", pa.string()),
        ("subject_id", pa.string()),  # qualified by the dataset
        ("subject_certain", pa.bool_()),  # the source states the mapping
        ("split", pa.string()),  # train, validation, test, or held_out (for a benchmark)
        ("held_out_for", pa.string()),  # the benchmark, when split is held_out
        ("duration_s", pa.float64()),
        ("start_time", pa.timestamp("us")),
        ("arrays", pa.list_(pa.string())),
        ("signals", pa.string()),  # the Zarr group, from the store root
    ]
)
CHANNELS = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("recording_id", pa.string()),
        ("array", pa.string()),
        ("index", pa.int32()),
        ("name", pa.string()),
        ("signal_type", pa.string()),
        ("signal_type_from", pa.string()),  # dataset | name | none
        ("sampling_rate", pa.float64()),
        ("unit", pa.string()),
        ("scale", pa.float64()),
        ("offset", pa.float64()),
        ("invalid_value", pa.int64()),
    ]
)


class AlreadyIndexed(RuntimeError):  # noqa: N818 - reads naturally at the call site
    pass


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    version: str
    tool: str
    built_at: datetime
    rules_sha256: str
    datasets: dict[str, str]  # dataset id -> standard version used
    standard_manifests_sha256: dict[str, str]
    left_out: dict[str, str]  # dataset id -> why (e.g. no complete standard version)
    excluded: dict[str, int]  # "dataset: reason" -> recordings left out by an exclude rule
    held_out: dict[str, int]  # benchmark -> recordings kept out of pretraining for it
    recordings: int
    recordings_without_signals: int
    subjects: int
    subjects_assumed: int  # the source does not state the mapping
    splits: dict[str, int]  # recordings per split
    signal_types: dict[str, int]  # channels per type
    unknown_channels: dict[str, int]  # "dataset: name" -> channels


@dataclass
class IndexReport:
    location: str
    manifest: IndexManifest


def build_index(
    store: Store,
    dataset_ids: list[str],
    rules: IndexRules,
    *,
    version: str | None = None,
    replace: bool = False,
) -> IndexReport:
    """Index the newest complete standard version of each of `dataset_ids`. `version`: the
    label of the index, by default today's UTC date."""
    version = version or retrieval_version()
    prefix = f"{ROOT}/{version}"
    if store.exists(f"{prefix}/{SUCCESS}") and not replace:
        raise AlreadyIndexed(f"{store.uri}/{prefix} is already complete; use --replace")
    found = _Found()
    used: dict[str, str] = {}
    manifests: dict[str, str] = {}
    left_out: dict[str, str] = {}
    for dataset_id in dataset_ids:
        versions = complete_versions(store, dataset_id, STANDARD_ROOT)
        if not versions:
            left_out[dataset_id] = "no complete standard version"
            continue
        location = StandardLocation(dataset_id, versions[-1])
        data = store.get_bytes(location.manifest_key)
        if data is None:  # _SUCCESS is written after the manifest, so this is damage
            raise FileNotFoundError(f"{location.manifest_key} is missing but _SUCCESS exists")
        used[dataset_id] = location.version
        manifests[dataset_id] = hashlib.sha256(data).hexdigest()
        raw = VersionLocation(dataset_id, json.loads(data)["raw_version"])
        indexed = _index_dataset(store, location, raw, rules.get(dataset_id), found)
        log.info("%s %s: %d recordings indexed", dataset_id, location.version, indexed)
    recordings, channels = found.recordings, found.channels
    subjects = {r["subject_id"]: r["subject_certain"] for r in recordings}
    manifest = IndexManifest(
        version=version,
        tool=f"biosignal-data {__version__}",
        built_at=utcnow(),
        rules_sha256=hashlib.sha256(rules.path.read_bytes()).hexdigest(),
        datasets=used,
        standard_manifests_sha256=manifests,
        left_out=left_out,
        excluded=dict(found.excluded),
        held_out=dict(found.held_out),
        recordings=len(recordings),
        recordings_without_signals=found.without_signals,
        subjects=len(subjects),
        subjects_assumed=sum(1 for certain in subjects.values() if not certain),
        splits=dict(Counter(r["split"] for r in recordings)),
        signal_types=dict(Counter(c["signal_type"] for c in channels)),
        unknown_channels=dict(found.unknown.most_common()),
    )
    store.put_bytes(f"{prefix}/recordings.parquet", _parquet(recordings, RECORDINGS))
    store.put_bytes(f"{prefix}/channels.parquet", _parquet(channels, CHANNELS))
    store.put_bytes(f"{prefix}/_manifest.json", manifest.model_dump_json(indent=2).encode())
    store.put_bytes(f"{prefix}/{SUCCESS}", b"", content_type="text/plain")
    return IndexReport(f"{store.uri}/{prefix}", manifest)


@dataclass
class _Found:
    """The rows of the index, and what was left out, over all datasets."""

    recordings: list[dict[str, Any]] = field(default_factory=list)
    channels: list[dict[str, Any]] = field(default_factory=list)
    without_signals: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    held_out: Counter[str] = field(default_factory=Counter)
    unknown: Counter[str] = field(default_factory=Counter)


def _index_dataset(
    store: Store,
    location: StandardLocation,
    raw: VersionLocation,
    rules: DatasetRules,
    found: _Found,
) -> int:
    """Add the recordings and channels of one standard version; returns the recordings."""
    dataset_id = location.dataset_id
    overrides = channel_overrides(rules)
    table = None
    if rules.subject is not None and rules.subject.table is not None:
        table = _subject_table(store, raw, rules.subject.table)
    indexed = set()
    for r in _read(store, location.table_key("recordings")).to_pylist():
        if not r["signals"]:
            found.without_signals += 1
            continue
        exclude = rules.excluded(r["recording_id"])
        if exclude is not None:
            found.excluded[f"{dataset_id}: {exclude.reason}"] += 1
            continue
        subject, certain = subject_of(dataset_id, r["recording_id"], rules.subject, table)
        held_out = rules.held_out_for(r["recording_id"])
        if held_out is not None:
            found.held_out[held_out.benchmark] += 1
        indexed.add(r["recording_id"])
        found.recordings.append(
            {
                "dataset_id": dataset_id,
                "version": location.version,
                "recording_id": r["recording_id"],
                "subject_id": subject,
                "subject_certain": certain,
                "split": "held_out" if held_out is not None else split_of(subject),
                "held_out_for": held_out.benchmark if held_out is not None else None,
                "duration_s": r["duration_s"],
                "start_time": r["start_time"],
                "arrays": r["arrays"],
                "signals": f"{location.prefix}/{r['signals']}",
            }
        )
    for c in _read(store, location.table_key("channels")).to_pylist():
        if c["recording_id"] not in indexed:
            continue
        kind, decided_by = signal_type(c["name"], overrides)
        if decided_by == "none":
            found.unknown[f"{dataset_id}: {c['name']}"] += 1
        found.channels.append(
            {
                "dataset_id": dataset_id,
                "recording_id": c["recording_id"],
                "array": c["array"],
                "index": c["index"],
                "name": c["name"],
                "signal_type": kind.value,
                "signal_type_from": decided_by,
                "sampling_rate": c["sampling_rate"],
                "unit": c["unit"],
                "scale": c["scale"],
                "offset": c["offset"],
                "invalid_value": c["invalid_value"],
            }
        )
    return len(indexed)


def _subject_table(store: Store, raw: VersionLocation, table: SubjectTable) -> dict[str, str]:
    """{recording: subject} from a table of the raw dataset."""
    key = raw.data_key(table.file)
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter=table.delimiter)
    mapping: dict[str, str] = {}
    for row in rows:
        recording, subject = row[table.recording_column], row[table.subject_column]
        if mapping.setdefault(recording, subject) != subject:
            raise ValueError(f"{key}: {recording!r} has two subjects")
    return mapping


def _read(store: Store, key: str) -> pa.Table:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    return pq.read_table(io.BytesIO(data))


def _parquet(rows: list[dict[str, Any]], schema: pa.Schema) -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out, compression="zstd")
    return out.getvalue()
