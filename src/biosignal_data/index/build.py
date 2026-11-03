"""Build the index: the small, stored part of the training layer.

    index/<date>/
        recordings.parquet   every recording with signals: dataset, subject, split, duration
        channels.parquet     every channel: name as written, signal type and what decided it
        _manifest.json       the standard versions used, the rules, the counts, unknown names
        _SUCCESS

The samples themselves are not copied: the index points at the standard version of each
recording, whose blocks a data loader reads (see training/). An index is immutable once
complete.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from biosignal_data import __version__
from biosignal_data.index.rules import (
    DatasetRules,
    IndexRules,
    SubjectTable,
    channel_rules_for,
)
from biosignal_data.index.signals import signal_type
from biosignal_data.index.subjects import split_of, subject_of
from biosignal_data.read.raw import complete_versions
from biosignal_data.standard import kept_files
from biosignal_data.standard.layout import ROOT as STANDARD_ROOT
from biosignal_data.standard.layout import StandardLocation
from biosignal_data.storage.layout import SUCCESS, retrieval_version
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
        ("standard", pa.string()),  # the standard version folder, from the store root
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
        indexed = _index_dataset(store, location, rules.get(dataset_id), found)
        log.info("%s %s: %d recordings indexed", dataset_id, location.version, indexed)
    recordings = found.recordings
    channels = pa.concat_tables(found.channels) if found.channels else CHANNELS.empty_table()
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
        signal_types={
            v["values"].as_py(): v["counts"].as_py()
            for v in pc.value_counts(channels["signal_type"])
        },
        unknown_channels=dict(found.unknown.most_common()),
    )
    store.put_bytes(f"{prefix}/recordings.parquet", _parquet(recordings, RECORDINGS))
    store.put_bytes(f"{prefix}/channels.parquet", _table_parquet(channels))
    store.put_bytes(f"{prefix}/_manifest.json", manifest.model_dump_json(indent=2).encode())
    store.put_bytes(f"{prefix}/{SUCCESS}", b"", content_type="text/plain")
    return IndexReport(f"{store.uri}/{prefix}", manifest)


@dataclass
class _Found:
    """The rows of the index, and what was left out, over all datasets."""

    recordings: list[dict[str, Any]] = field(default_factory=list)
    channels: list[pa.Table] = field(default_factory=list)  # CHANNELS, one table a dataset
    without_signals: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    held_out: Counter[str] = field(default_factory=Counter)
    unknown: Counter[str] = field(default_factory=Counter)


def _index_dataset(
    store: Store,
    location: StandardLocation,
    rules: DatasetRules,
    found: _Found,
) -> int:
    """Add the recordings and channels of one standard version; returns the recordings."""
    dataset_id = location.dataset_id
    table = None
    if rules.subject is not None and rules.subject.table is not None:
        table = _subject_table(store, location, rules.subject.table)
    held_out_tables = {
        i: _column(store, location, h.table.file, h.table.delimiter, h.table.recording_column,
                   h.table.column)
        for i, h in enumerate(rules.held_out) if h.table is not None
    }  # fmt: skip
    indexed = set()
    recordings = _read(store, location.table_key("recordings"), RECORDING_COLUMNS)
    for r in recordings.to_pylist():
        if not r["signals"]:
            found.without_signals += 1
            continue
        exclude = rules.excluded(r["recording_id"])
        if exclude is not None:
            found.excluded[f"{dataset_id}: {exclude.reason}"] += 1
            continue
        subject, certain = subject_of(dataset_id, r["recording_id"], rules.subject, table)
        held_out = rules.held_out_for(r["recording_id"], held_out_tables)
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
                "standard": location.prefix,
            }
        )
    found.channels.append(_channels(store, location, rules, indexed, found.unknown))
    return len(indexed)


# The columns of the standard tables that the index reads (not `source`: the header text)
RECORDING_COLUMNS = ["recording_id", "signals", "duration_s", "start_time", "arrays"]
CHANNEL_COLUMNS = ["recording_id", "array", "index", "name", "sampling_rate", "unit", "scale",
                   "offset", "invalid_value"]  # fmt: skip


def _channels(
    store: Store, location: StandardLocation, rules: DatasetRules, indexed: set[str],
    unknown: Counter[str],
) -> pa.Table:  # fmt: skip
    """The CHANNELS rows of the recordings `indexed`, in the order of the standard table, column
    by column. The signal type is decided once for each distinct (channel rules that hold for
    the recording, channel name) and taken from there for every channel; `unknown` counts the
    channels of each name that no rule types."""
    dataset_id = location.dataset_id
    table = _read(store, location.table_key("channels"), CHANNEL_COLUMNS)
    table = table.filter(pc.is_in(table["recording_id"], value_set=pa.array(sorted(indexed))))
    recordings = pc.dictionary_encode(table["recording_id"].combine_chunks())
    names = pc.dictionary_encode(table["name"].combine_chunks())
    # which of the dataset's channel rules hold for each recording: few distinct sets
    rule_sets: dict[tuple[int, ...], int] = {}
    set_of_recording = np.array([
        rule_sets.setdefault(channel_rules_for(rules, recording), len(rule_sets))
        for recording in recordings.dictionary.to_pylist()
    ], dtype=np.int64)  # fmt: skip
    overrides = [
        [(re.compile(rules.channels[k].name), rules.channels[k].type) for k in kept]
        for kept in rule_sets
    ]
    name_list = names.dictionary.to_pylist()
    rule_set = set_of_recording[recordings.indices.to_numpy()]  # of each channel
    pairs, inverse = np.unique(
        rule_set * len(name_list) + names.indices.to_numpy(), return_inverse=True
    )
    decided = [
        signal_type(name_list[p % len(name_list)], overrides[p // len(name_list)]) for p in pairs
    ]
    counts = np.bincount(inverse, minlength=len(pairs))
    for p, (_, decided_by), n in zip(pairs, decided, counts, strict=True):
        if decided_by == "none":
            unknown[f"{dataset_id}: {name_list[p % len(name_list)]}"] += int(n)
    kinds = pa.array([kind.value for kind, _ in decided], pa.string())
    by = pa.array([decided_by for _, decided_by in decided], pa.string())
    columns = {
        "dataset_id": pa.repeat(pa.scalar(dataset_id, pa.string()), len(table)),
        "recording_id": table["recording_id"],
        "array": table["array"],
        "index": table["index"],
        "name": table["name"],
        "signal_type": kinds.take(pa.array(inverse)),
        "signal_type_from": by.take(pa.array(inverse)),
        "sampling_rate": table["sampling_rate"],
        "unit": table["unit"],
        "scale": table["scale"],
        "offset": table["offset"],
        "invalid_value": table["invalid_value"],
    }
    return pa.table(columns).cast(CHANNELS)


def _subject_table(store: Store, location: StandardLocation, table: SubjectTable) -> dict[str, str]:
    """{recording: subject} from a table of the dataset (kept with its standard version)."""
    return _column(
        store, location, table.file, table.delimiter, table.recording_column,
        table.subject_column,
    )  # fmt: skip


def _column(
    store: Store, location: StandardLocation, file: str, delimiter: str, key_column: str,
    column: str,
) -> dict[str, str]:  # fmt: skip
    """{key: value} of two columns of a delimited table of the dataset (in the standard
    version's files/); a key with two different values is an error."""
    key = f"{location.prefix}: {file}"
    data = kept_files.read(store, location, file)
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter=delimiter)
    mapping: dict[str, str] = {}
    for row in rows:
        if key_column not in row or column not in row:
            raise ValueError(f"{key}: no column {key_column!r} or {column!r}")
        recording, value = row[key_column], row[column]
        if mapping.setdefault(recording, value) != value:
            raise ValueError(f"{key}: {recording!r} has two values in {column!r}")
    return mapping


def _read(store: Store, key: str, columns: list[str]) -> pa.Table:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    return pq.read_table(io.BytesIO(data), columns=columns)


def _parquet(rows: list[dict[str, Any]], schema: pa.Schema) -> bytes:
    return _table_parquet(pa.Table.from_pylist(rows, schema=schema))


def _table_parquet(table: pa.Table) -> bytes:
    out = io.BytesIO()
    pq.write_table(table, out, compression="zstd")
    return out.getvalue()
