"""The quality layer: what each channel of the standard layer holds, measured once.

    quality/<repository>/<dataset>/<standard version>/
        channels.parquet   one row per channel (below)
        _manifest.json     the standard version measured, the settings, the counts
        _parts/<recording>.parquet   progress of an unfinished run (removed on success)
        _SUCCESS

It measures and never deletes: which channels go into training is a setting of each run.
A row per channel:

- `samples`, `invalid` (stored as the channel's invalid value), `nonfinite` (a physical
  value that is not a finite float32: a broken calibration in the source, e.g.
  autonomic-aging-cardiovascular 0065 NIBP with a gain of 2.78e-152)
- `constant`: every valid stored value is the same
- `longest_flat_s`, `flat_fraction`: runs of one stored value; a run counts as flat from
  `flat_s` seconds on (at least 2 samples)
- `at_low`, `at_high`: the share of samples at the recorder's limits, from the EDF
  header's digital minimum and maximum, else from the stored integer type
  (`limits_from`: `header` or `dtype`)
- `minimum`, `maximum`, `mean`, `std` of the valid physical values (exact), and `p01`,
  `p50`, `p99` from a seeded random sample of at most `percentile_sample` valid values
  per channel (exact percentiles would need a whole channel in memory)
"""

from __future__ import annotations

import io
import json
import logging
import math
import multiprocessing
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from pydantic import BaseModel, ConfigDict

from biosignal_data import __version__
from biosignal_data.read.raw import complete_versions
from biosignal_data.standard.layout import ROOT as STANDARD_ROOT
from biosignal_data.standard.layout import StandardLocation
from biosignal_data.storage.layout import MANIFEST, SUCCESS, VersionLocation, safe_relpath
from biosignal_data.storage.manifest import utcnow
from biosignal_data.storage.stores import Store, StoreFactory

log = logging.getLogger(__name__)
ROOT = "quality"
SCHEMA_VERSION = 1

CHANNELS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("array", pa.string()),
        ("index", pa.int32()),
        ("name", pa.string()),
        ("samples", pa.int64()),
        ("invalid", pa.int64()),
        ("nonfinite", pa.int64()),
        ("constant", pa.bool_()),
        ("longest_flat_s", pa.float64()),
        ("flat_fraction", pa.float64()),
        ("at_low", pa.float64()),
        ("at_high", pa.float64()),
        ("limits_from", pa.string()),
        ("minimum", pa.float64()),
        ("maximum", pa.float64()),
        ("mean", pa.float64()),
        ("std", pa.float64()),
        ("p01", pa.float64()),
        ("p50", pa.float64()),
        ("p99", pa.float64()),
    ]
)


class AlreadyMeasured(RuntimeError):  # noqa: N818 - reads naturally at the call site
    pass


class QualityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    dataset_id: str
    standard_version: str
    standard_manifest_sha256: str
    tool: str
    flat_s: float
    percentile_sample: int
    seed: int
    channels: int
    recordings: int
    completed_at: datetime


@dataclass(frozen=True)
class QualityLocation:
    dataset_id: str
    version: str  # the standard version measured

    @property
    def prefix(self) -> str:
        return VersionLocation(self.dataset_id, self.version, ROOT).prefix

    def part_key(self, recording_id: str) -> str:
        return f"{self.prefix}/_parts/{safe_relpath(recording_id)}.parquet"

    @property
    def table_key(self) -> str:
        return f"{self.prefix}/channels.parquet"


@dataclass
class _Channel:
    """Running counts of one channel, block by block."""

    scale: float | None
    offset: float | None
    invalid_value: int | None
    low: int
    high: int
    flat_samples: int
    sample_size: int
    rng: np.random.Generator
    samples: int = 0
    invalid: int = 0
    nonfinite: int = 0
    at_low: int = 0
    at_high: int = 0
    longest: int = 0
    flat: int = 0
    run_value: int | None = None
    run_length: int = 0
    first_valid: int | None = None
    varies: bool = False
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    keys: np.ndarray = field(default_factory=lambda: np.empty(0))

    def add(self, stored: np.ndarray) -> None:
        self.samples += stored.size
        self.at_low += int((stored == self.low).sum())
        self.at_high += int((stored == self.high).sum())
        self._runs(stored)
        valid = stored if self.invalid_value is None else stored[stored != self.invalid_value]
        self.invalid += stored.size - valid.size
        if valid.size:
            if self.first_valid is None:
                self.first_valid = int(valid[0])
            self.varies = self.varies or bool((valid != self.first_valid).any())
        if self.scale is None:
            physical = valid.astype(np.float64)
        else:
            physical = valid * np.float64(self.scale) + np.float64(self.offset or 0.0)
        finite = np.isfinite(physical) & (np.abs(physical) <= np.finfo(np.float32).max)
        self.nonfinite += int((~finite).sum())
        physical = physical[finite]
        if physical.size:
            self.count += physical.size
            self.total += float(physical.sum())
            self.total_sq += float(np.square(physical).sum())
            self.minimum = min(self.minimum, float(np.min(physical)))
            self.maximum = max(self.maximum, float(np.max(physical)))
            self._keep(physical)

    def _runs(self, stored: np.ndarray) -> None:
        """Runs of one stored value, carried across blocks."""
        if not stored.size:
            return
        cuts = np.flatnonzero(np.diff(stored)) + 1
        lengths = np.diff(np.concatenate([[0], cuts, [stored.size]]))
        if self.run_value is not None and int(stored[0]) == self.run_value:
            lengths[0] += self.run_length
        elif self.run_length:
            self._end_run(self.run_length)
        for length in lengths[:-1]:
            self._end_run(int(length))
        self.run_value, self.run_length = int(stored[-1]), int(lengths[-1])

    def _end_run(self, length: int) -> None:
        self.longest = max(self.longest, length)
        if length >= self.flat_samples:
            self.flat += length

    def _keep(self, physical: np.ndarray) -> None:
        """A uniform random sample of all valid values so far, without replacement: each
        value gets a random key, and the `sample_size` values with the smallest keys stay
        (bottom-k sampling; exact, block by block)."""
        keys = np.concatenate([self.keys, self.rng.random(physical.size)])
        values = np.concatenate([self.sample, physical])
        if values.size > self.sample_size:
            keep = np.argpartition(keys, self.sample_size)[: self.sample_size]
            keys, values = keys[keep], values[keep]
        self.keys, self.sample = keys, values

    def row(self, rate: float) -> dict[str, Any]:
        if self.run_length:
            self._end_run(self.run_length)
            self.run_length = 0
        mean = self.total / self.count if self.count else None
        variance = self.total_sq / self.count - mean**2 if mean is not None else None
        p01, p50, p99 = (
            [float(v) for v in np.percentile(self.sample, [1, 50, 99])]
            if self.sample.size else [None, None, None]
        )  # fmt: skip
        return {
            "samples": self.samples,
            "invalid": self.invalid,
            "nonfinite": self.nonfinite,
            "constant": not self.varies,
            "longest_flat_s": self.longest / rate,
            "flat_fraction": self.flat / self.samples if self.samples else 0.0,
            "at_low": self.at_low / self.samples if self.samples else 0.0,
            "at_high": self.at_high / self.samples if self.samples else 0.0,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
            "mean": mean,
            "std": math.sqrt(max(0.0, variance)) if variance is not None else None,
            "p01": p01,
            "p50": p50,
            "p99": p99,
        }


def _stable(recording_id: str, array: str, index: int) -> int:
    """A seed part that is the same in every process (str hash() is not)."""
    return zlib.crc32(f"{recording_id}/{array}/{index}".encode())


def _limits(channel: dict[str, Any]) -> tuple[int, int, str]:
    source = json.loads(channel["source"] or "{}")
    if "digital_min" in source and "digital_max" in source:
        return int(source["digital_min"]), int(source["digital_max"]), "header"
    info = np.iinfo(np.dtype(channel["dtype"]))
    return int(info.min), int(info.max), "dtype"


def measure_recording(
    group: zarr.Group,
    recording_id: str,
    channels: list[dict[str, Any]],
    *,
    flat_s: float,
    percentile_sample: int,
    seed: int,
) -> list[dict[str, Any]]:
    """The rows of one recording: each array read once, a shard at a time."""
    rows = []
    for array_name in sorted({c["array"] for c in channels}):
        array = group[array_name]
        if not isinstance(array, zarr.Array):
            raise TypeError(f"{recording_id}/{array_name} is not an array")
        mine = sorted((c for c in channels if c["array"] == array_name), key=lambda c: c["index"])
        rate = float(mine[0]["sampling_rate"])
        counters = []
        for c in mine:
            low, high, limits_from = _limits(c)
            counters.append(
                (c, limits_from, _Channel(
                    c["scale"], c["offset"], c["invalid_value"], low, high,
                    max(2, round(flat_s * rate)), percentile_sample,
                    np.random.default_rng([seed, _stable(recording_id, array_name, c["index"])]),
                ))
            )  # fmt: skip
        total = int(array.shape[1])
        step = int(array.shards[1]) if array.shards is not None else total
        for start in range(0, total, max(1, step)):
            block = np.asarray(array[:, start : start + step])
            for c, _, counter in counters:
                counter.add(block[c["index"]])
        for c, limits_from, counter in counters:
            rows.append(
                {"recording_id": recording_id, "array": array_name, "index": c["index"],
                 "name": c["name"], "limits_from": limits_from} | counter.row(rate)
            )  # fmt: skip
    return rows


def _measure_part(
    factory: StoreFactory,
    location: QualityLocation,
    signals_key: str,
    recording_id: str,
    channels: list[dict[str, Any]],
    settings: tuple[float, int, int],
) -> None:
    """One recording, in a worker process: measure it and write its part."""
    store = factory()
    flat_s, percentile_sample, seed = settings
    group = zarr.open_group(store=store.zarr_store(signals_key, read_only=True), mode="r")
    rows = measure_recording(
        group, recording_id, channels, flat_s=flat_s, percentile_sample=percentile_sample,
        seed=seed,
    )  # fmt: skip
    store.put_bytes(location.part_key(recording_id), _parquet(rows))


def measure(
    store: Store,
    factory: StoreFactory,
    dataset_id: str,
    *,
    flat_s: float,
    percentile_sample: int,
    seed: int,
    workers: int,
    replace: bool,
    progress: Callable[[int, int], None] | None = None,
) -> QualityManifest:
    """Measure the newest complete standard version of `dataset_id`; resumes from parts.
    `workers` processes measure one recording each (the counting is Python and NumPy, so
    threads would share one core); `factory` opens the store in each of them."""
    versions = complete_versions(store, dataset_id, STANDARD_ROOT)
    if not versions:
        raise FileNotFoundError(f"{dataset_id}: no complete standard version")
    standard = StandardLocation(dataset_id, versions[-1])
    location = QualityLocation(dataset_id, standard.version)
    if store.exists(f"{location.prefix}/{SUCCESS}"):
        if not replace:
            raise AlreadyMeasured(f"{store.uri}/{location.prefix} is complete; use --replace")
        store.delete(f"{location.prefix}/{SUCCESS}")
        store.delete_many(list(store.list(f"{location.prefix}/")))
    manifest_bytes = store.get_bytes(standard.manifest_key)
    if manifest_bytes is None:
        raise FileNotFoundError(standard.manifest_key)
    recordings = _table(store, standard.table_key("recordings"))
    channels = _table(store, standard.table_key("channels"))
    by_recording: dict[str, list[dict[str, Any]]] = {}
    for c in channels:
        by_recording.setdefault(c["recording_id"], []).append(c)
    # One listing, not a request per recording: thousands of small requests at once made
    # the storage answer 429 Too Many Requests (challenge-2017, 2026-10-20).
    stored = set(store.list(f"{location.prefix}/_parts/"))
    todo = [
        r["recording_id"] for r in recordings
        if r["signals"] and location.part_key(r["recording_id"]) not in stored
    ]  # fmt: skip
    done = sum(1 for r in recordings if r["signals"]) - len(todo)

    settings = (flat_s, percentile_sample, seed)
    context = multiprocessing.get_context("spawn")  # a forked copy of zarr's event loop hangs
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        futures = [
            pool.submit(_measure_part, factory, location, standard.signals_key(r), r,
                        by_recording.get(r, []), settings)
            for r in todo
        ]  # fmt: skip
        for future in as_completed(futures):
            future.result()
            done += 1
            if progress:
                progress(done, done + sum(1 for f in futures if not f.done()))
    parts = sorted(k for k in store.list(f"{location.prefix}/_parts/") if k.endswith(".parquet"))
    tables = [pq.read_table(io.BytesIO(store.get_bytes(k) or b"")) for k in parts]
    table = pa.concat_tables(tables).cast(CHANNELS) if tables else CHANNELS.empty_table()
    store.put_bytes(location.table_key, _parquet(table.to_pylist()))
    manifest = QualityManifest(
        dataset_id=dataset_id, standard_version=standard.version,
        standard_manifest_sha256=_sha256(manifest_bytes), tool=f"biosignal-data {__version__}",
        flat_s=flat_s, percentile_sample=percentile_sample, seed=seed,
        channels=table.num_rows, recordings=len(parts), completed_at=utcnow(),
    )  # fmt: skip
    store.put_bytes(f"{location.prefix}/{MANIFEST}", manifest.model_dump_json(indent=2).encode())
    store.put_bytes(f"{location.prefix}/{SUCCESS}", b"", content_type="text/plain")
    store.delete_many(parts)
    return manifest


def _sha256(data: bytes) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data).hexdigest()


def _table(store: Store, key: str) -> list[dict[str, Any]]:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    rows: list[dict[str, Any]] = pq.read_table(io.BytesIO(data)).to_pylist()
    return rows


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=CHANNELS), out, compression="zstd")
    return out.getvalue()


def iter_rows(store: Store, dataset_id: str, version: str) -> Iterator[dict[str, Any]]:
    """The rows of a complete quality version."""
    location = QualityLocation(dataset_id, version)
    if not store.exists(f"{location.prefix}/{SUCCESS}"):
        raise FileNotFoundError(f"{store.uri}/{location.prefix} is not complete")
    yield from _table(store, location.table_key)
