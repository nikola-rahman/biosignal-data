"""Key layout of the standard layer (see docs/standard-layer.md).

<root>/<repository>/<dataset>/<raw version>/
    recordings.parquet, channels.parquet, events.parquet
    blocks.parquet        where each block of samples is (see standard/blocks.py)
    data/<batch>-<n>.bin  the blocks, in objects of at most 64 MiB
    _manifest.json
    raw_manifest.json     the raw version's manifest (every file with its SHA-256), for lineage
    _parts/<batch>/...    progress of an unfinished run (removed on success)
    _SUCCESS
"""

from __future__ import annotations

from dataclasses import dataclass

from biosignal_data.storage.layout import MANIFEST, SUCCESS, VersionLocation, safe_relpath

# A new root for the block layout (2026-10-20); "standard" holds the earlier one, a Zarr
# group per recording.
ROOT = "standard-blocks"
TABLES = ("recordings", "channels", "events", "blocks")
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

    def object_key(self, name: str) -> str:
        """An object of blocks, by its name in blocks.parquet (`data/...`)."""
        return f"{self.prefix}/{safe_relpath(name)}"

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
    def raw_manifest_key(self) -> str:
        return f"{self.prefix}/raw_manifest.json"

    @property
    def success_key(self) -> str:
        return f"{self.prefix}/{SUCCESS}"
