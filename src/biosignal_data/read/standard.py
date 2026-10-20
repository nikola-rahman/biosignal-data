"""Read the standard layer: the tables, and the signals straight from the store.

from biosignal_data.read import StandardDataset
ds = StandardDataset.open(store, "physionet/mitdb", "latest")
ds.recordings                                    # pyarrow Table, one row per recording
group = ds.signals("100")                        # zarr.Group: one array per sampling rate
window = ds.window("100", "360Hz", 3600.0, 30.0)  # (channels, samples), physical units

Arrays hold the stored integers; `window(..., physical=True)` applies the scale and offset
of each channel. Samples stored as the channel's invalid value, and channels without a
scale, come back as NaN in physical units.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from biosignal_data.read.raw import complete_versions
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

    def signals(self, recording_id: str) -> zarr.Group:
        """The Zarr group of one recording, read from the store as needed."""
        store = self.store.zarr_store(self.location.signals_key(recording_id), read_only=True)
        return zarr.open_group(store=store, mode="r")

    def window(
        self,
        recording_id: str,
        array: str,
        start_s: float,
        duration_s: float,
        *,
        physical: bool = True,
    ) -> np.ndarray:
        """`duration_s` of one array from `start_s`; only the chunks it needs are read."""
        data = self.signals(recording_id)[array]
        if not isinstance(data, zarr.Array):
            raise KeyError(f"{recording_id}: {array} is not an array")
        rate = data.attrs["sampling_rate"]
        if not isinstance(rate, int | float):
            raise TypeError(f"{recording_id}/{array}: sampling_rate is {rate!r}")
        start = round(start_s * rate)
        values = np.asarray(data[:, start : start + round(duration_s * rate)])
        if not physical:
            return values
        channels: list[dict[str, Any]] = data.attrs["channels"]  # type: ignore[assignment]
        scale = np.array([np.nan if c["scale"] is None else c["scale"] for c in channels])
        offset = np.array([np.nan if c["offset"] is None else c["offset"] for c in channels])
        result: np.ndarray = values * scale[:, None] + offset[:, None]
        for i, c in enumerate(channels):
            if c["invalid_value"] is not None:
                result[i, values[i] == c["invalid_value"]] = np.nan
        return result
