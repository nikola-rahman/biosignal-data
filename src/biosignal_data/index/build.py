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

import hashlib
import io
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from biosignal_data import __version__
from biosignal_data.index.rules import IndexRules, channel_overrides
from biosignal_data.index.signals import signal_type
from biosignal_data.index.subjects import split_of, subject_of
from biosignal_data.read.raw import complete_versions
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
        ("split", pa.string()),
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
    recordings: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    used: dict[str, str] = {}
    manifests: dict[str, str] = {}
    left_out: dict[str, str] = {}
    without_signals = 0
    unknown: Counter[str] = Counter()
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
        dataset_rules = rules.get(dataset_id)
        overrides = channel_overrides(dataset_rules)
        recording_table = _read(store, location.table_key("recordings")).to_pylist()
        with_signals = set()
        for r in recording_table:
            if not r["signals"]:
                without_signals += 1
                continue
            subject, certain = subject_of(dataset_id, r["recording_id"], dataset_rules.subject)
            with_signals.add(r["recording_id"])
            recordings.append(
                {
                    "dataset_id": dataset_id,
                    "version": location.version,
                    "recording_id": r["recording_id"],
                    "subject_id": subject,
                    "subject_certain": certain,
                    "split": split_of(subject),
                    "duration_s": r["duration_s"],
                    "start_time": r["start_time"],
                    "arrays": r["arrays"],
                    "signals": f"{location.prefix}/{r['signals']}",
                }
            )
        for c in _read(store, location.table_key("channels")).to_pylist():
            if c["recording_id"] not in with_signals:
                continue
            kind, decided_by = signal_type(c["name"], overrides)
            if decided_by == "none":
                unknown[f"{dataset_id}: {c['name']}"] += 1
            channels.append(
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
        log.info("%s %s: %d recordings indexed", dataset_id, location.version, len(with_signals))
    subjects = {r["subject_id"]: r["subject_certain"] for r in recordings}
    manifest = IndexManifest(
        version=version,
        tool=f"biosignal-data {__version__}",
        built_at=utcnow(),
        rules_sha256=hashlib.sha256(rules.path.read_bytes()).hexdigest(),
        datasets=used,
        standard_manifests_sha256=manifests,
        left_out=left_out,
        recordings=len(recordings),
        recordings_without_signals=without_signals,
        subjects=len(subjects),
        subjects_assumed=sum(1 for certain in subjects.values() if not certain),
        splits=dict(Counter(r["split"] for r in recordings)),
        signal_types=dict(Counter(c["signal_type"] for c in channels)),
        unknown_channels=dict(unknown.most_common()),
    )
    store.put_bytes(f"{prefix}/recordings.parquet", _parquet(recordings, RECORDINGS))
    store.put_bytes(f"{prefix}/channels.parquet", _parquet(channels, CHANNELS))
    store.put_bytes(f"{prefix}/_manifest.json", manifest.model_dump_json(indent=2).encode())
    store.put_bytes(f"{prefix}/{SUCCESS}", b"", content_type="text/plain")
    return IndexReport(f"{store.uri}/{prefix}", manifest)


def _read(store: Store, key: str) -> pa.Table:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    return pq.read_table(io.BytesIO(data))


def _parquet(rows: list[dict[str, Any]], schema: pa.Schema) -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out, compression="zstd")
    return out.getvalue()
