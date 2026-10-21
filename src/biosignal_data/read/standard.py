"""Read the standard layer: the tables, and the samples straight from the store.

from biosignal_data.read import StandardDataset
ds = StandardDataset.open(store, "physionet/mitdb", "latest")
ds.recordings                                    # pyarrow Table, one row per recording
x = ds.read("100", "360Hz")                       # (channels, samples), stored integers
window = ds.window("100", "360Hz", 3600.0, 30.0)  # (channels, samples), physical units
many = ds.read_many([("100", "360Hz"), ("101", "360Hz")], threads=8)  # whole recordings

Samples are in blocks (see standard/blocks.py). `read` and `window` read only the blocks
they need, one ranged request each; `read_many` reads many recordings with about one
request per object, for a whole dataset at once. `physical=True` applies the scale and
offset of each channel; samples stored as the channel's invalid value, and channels without
a scale, come back as NaN in physical units.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from biosignal_data.read.raw import complete_versions
from biosignal_data.standard import blocks, kept_files
from biosignal_data.standard.layout import ROOT, StandardLocation
from biosignal_data.standard.manifest import StandardManifest
from biosignal_data.storage.stores import S3Settings, Store, open_store


@dataclass
class StandardDataset:
    store: Store
    location: StandardLocation
    manifest: StandardManifest

    @classmethod
    def open(
        cls,
        store: Store | str,
        dataset_id: str,
        version: str,
        *,
        s3: S3Settings | None = None,
    ) -> StandardDataset:
        """Open a complete standard version. `version`: the raw version it was made from,
        or "latest" for the newest complete one. `store` as for `RawDataset.open`."""
        store = open_store(store, s3) if isinstance(store, str) else store
        if version == "latest":
            versions = complete_versions(store, dataset_id, ROOT)
            if not versions:
                raise FileNotFoundError(f"no complete standard version of {dataset_id}")
            version = versions[-1]
        location = StandardLocation(dataset_id, version)
        data = store.get_bytes(location.manifest_key)
        if data is None or not store.exists(location.success_key):
            raise FileNotFoundError(f"{store.uri}/{location.prefix} is not a complete version")
        return cls(store, location, StandardManifest.model_validate_json(data))

    def _table(self, name: str) -> pa.Table:
        data = self.store.get_bytes(self.location.table_key(name))
        if data is None:
            raise FileNotFoundError(f"{self.location.table_key(name)} is missing")
        return pq.read_table(io.BytesIO(data))

    @cached_property
    def recordings(self) -> pa.Table:
        return self._table("recordings")

    @cached_property
    def channels(self) -> pa.Table:
        return self._table("channels")

    @cached_property
    def events(self) -> pa.Table:
        return self._table("events")

    @cached_property
    def _kept_files(self) -> dict[str, dict[str, Any]]:
        return kept_files.index(self.store, self.location)

    def file(self, path: str) -> bytes:
        """A raw file that holds no signals, kept with the version (a table such as
        ptbxl_database.csv, documentation), as published; checked against its SHA-256."""
        return kept_files.read(self.store, self.location, path, self._kept_files)

    @cached_property
    def block_index(self) -> dict[tuple[str, str], blocks.ArrayBlocks]:
        """(recording, array) -> its blocks, from blocks.parquet (read once)."""
        return blocks.index(self._table("blocks"))

    def array(self, recording_id: str, array: str) -> blocks.ArrayBlocks:
        found = self.block_index.get((recording_id, array))
        if found is None:
            raise KeyError(f"{recording_id}: no array {array!r}")
        return found

    def _get_range(self, name: str, offset: int, length: int) -> bytes:
        return self.store.get_range(self.location.object_key(name), offset, length)

    def read(
        self,
        recording_id: str,
        array: str,
        first: int = 0,
        last: int | None = None,
        *,
        physical: bool = False,
    ) -> np.ndarray:
        """Samples [first, last) of every channel of one array; all of it by default."""
        found = self.array(recording_id, array)
        stop = found.length if last is None else last
        refs = found.overlapping(first, stop)
        data = blocks.fetch(self._get_range, refs, threads=1)
        values = blocks.span(found, first, stop, data.__getitem__)
        return self.physical(recording_id, array, values) if physical else values

    def window(
        self,
        recording_id: str,
        array: str,
        start_s: float,
        duration_s: float,
        *,
        physical: bool = True,
    ) -> np.ndarray:
        """`duration_s` of one array from `start_s`; only the blocks it needs are read."""
        rate = self._rate(recording_id, array)
        start = round(start_s * rate)
        return self.read(
            recording_id, array, start, start + round(duration_s * rate), physical=physical
        )

    def read_many(
        self, items: list[tuple[str, str]], *, threads: int, physical: bool = False
    ) -> dict[tuple[str, str], np.ndarray]:
        """Whole arrays of many recordings, (recording, array) -> samples: the blocks of one
        object are read together, so a whole dataset costs about one request per object."""
        arrays = {item: self.array(*item) for item in items}
        data = blocks.fetch(
            self._get_range, (r for a in arrays.values() for r in a.blocks), threads=threads
        )
        found = {}
        for item, a in arrays.items():
            values = blocks.span(a, 0, a.length, data.__getitem__)
            found[item] = self.physical(*item, values) if physical else values
        return found

    @cached_property
    def _calibration(self) -> dict[tuple[str, str], list[dict[str, object]]]:
        by: dict[tuple[str, str], list[dict[str, object]]] = {}
        for c in self.channels.select(
            ["recording_id", "array", "index", "sampling_rate", "scale", "offset", "invalid_value"]
        ).to_pylist():
            by.setdefault((str(c["recording_id"]), str(c["array"])), []).append(c)
        for rows in by.values():
            rows.sort(key=lambda c: int(c["index"]))  # type: ignore[call-overload]
        return by

    def _rate(self, recording_id: str, array: str) -> float:
        rows = self._calibration.get((recording_id, array))
        if not rows:
            raise KeyError(f"{recording_id}: no array {array!r}")
        return float(rows[0]["sampling_rate"])  # type: ignore[arg-type]

    def physical(self, recording_id: str, array: str, values: np.ndarray) -> np.ndarray:
        """Stored integers -> physical values (float64); NaN where a sample is the channel's
        invalid value, and for a channel without a scale."""
        rows = self._calibration[(recording_id, array)]
        scale = np.array([np.nan if c["scale"] is None else c["scale"] for c in rows])
        offset = np.array([np.nan if c["offset"] is None else c["offset"] for c in rows])
        result: np.ndarray = values * scale[:, None] + offset[:, None]
        for i, c in enumerate(rows):
            if c["invalid_value"] is not None:
                result[i, values[i] == c["invalid_value"]] = np.nan
        return result
