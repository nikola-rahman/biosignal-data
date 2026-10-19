"""Key layout of the standard layer (see docs/standard-layer.md).

<root>/<repository>/<dataset>/<raw version>/
    recordings.parquet, channels.parquet, events.parquet
    signals/<recording>.zarr/
    _manifest.json
    _parts/<unit>/...     progress of an unfinished run (removed on success)
    _SUCCESS
"""

from __future__ import annotations

from dataclasses import dataclass

from biosignal_data.storage.layout import MANIFEST, SUCCESS, VersionLocation, safe_relpath

ROOT = "standard"
TABLES = ("recordings", "channels", "events")
PARTS = "_parts"
DONE = "done.json"


@dataclass(frozen=True)
class StandardLocation:
    dataset_id: str
    version: str  # the raw version the data is made from
    root: str = ROOT

    @property
    def prefix(self) -> str:
        return VersionLocation(self.dataset_id, self.version, self.root).prefix

    def table_key(self, table: str) -> str:
        return f"{self.prefix}/{table}.parquet"

    def signals_key(self, recording_id: str) -> str:
        """The Zarr group of one recording."""
        return f"{self.prefix}/signals/{safe_relpath(recording_id)}.zarr"

    @property
    def parts_prefix(self) -> str:
        return f"{self.prefix}/{PARTS}/"

    def part_key(self, unit_id: str, name: str) -> str:
        """`name`: a table (`channels.parquet`, ...) or `done.json`, written last."""
        return f"{self.parts_prefix}{safe_relpath(unit_id)}/{name}"

    @property
    def manifest_key(self) -> str:
        return f"{self.prefix}/{MANIFEST}"

    @property
    def success_key(self) -> str:
        return f"{self.prefix}/{SUCCESS}"
