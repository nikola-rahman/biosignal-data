"""A local disk cache for the Zarr arrays of the standard layer, by whole objects.

The first read of an object (a shard, 64 MiB at most) downloads all of it and writes it
to `<directory>/<namespace>/<key>` (a temporary file renamed into place, so a crash never
leaves part of an object); every byte range after that is served from the complete local
copy. A standard version never changes, so a cached object never goes stale.

Why not an existing cache: tested on 2026-10-20 with sharded arrays, zarr's
`experimental.cache_store.CacheStore` stores a requested byte range under the key of the
whole object and then returns wrong samples for other ranges, without an error; fsspec's
`simplecache` and `filecache` fail on ranges counted from the end of an object (a shard's
index); `blockcache` is correct but did not speed up repeated reads.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterable
from pathlib import Path

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import WrapperStore

from biosignal_data.storage.layout import safe_relpath

PART = ".part"


class WholeObjectCache(WrapperStore[Store]):
    """Wraps a read-only store. `namespace` keeps the objects of one Zarr group apart
    from those of another (the key of the group in the standard layer). When the cache
    holds more than `max_bytes`, the objects used longest ago are deleted."""

    def __init__(self, store: Store, directory: Path, namespace: str, max_bytes: int) -> None:
        super().__init__(store)
        self._directory = directory
        self._root = directory / safe_relpath(namespace)
        self._max_bytes = max_bytes

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        path = self._root / safe_relpath(key)
        if path.is_file():
            os.utime(path)  # used now: evicted last
        else:
            whole = await self._store.get(key, prototype)
            if whole is None:
                return None
            await asyncio.to_thread(_write_atomic, path, whole.to_bytes())
            await asyncio.to_thread(evict, self._directory, self._max_bytes)
        data = await asyncio.to_thread(read_range, path, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, byte_range) for key, byte_range in key_ranges]


def read_range(path: Path, byte_range: ByteRequest | None) -> bytes:
    with path.open("rb") as f:
        if byte_range is None:
            return f.read()
        if isinstance(byte_range, RangeByteRequest):
            f.seek(byte_range.start)
            return f.read(byte_range.end - byte_range.start)
        if isinstance(byte_range, OffsetByteRequest):
            f.seek(byte_range.offset)
            return f.read()
        if isinstance(byte_range, SuffixByteRequest):
            size = os.fstat(f.fileno()).st_size
            f.seek(max(0, size - byte_range.suffix))
            return f.read()
    raise TypeError(f"unknown byte range {byte_range!r}")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}{PART}")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def evict(directory: Path, max_bytes: int) -> None:
    """Delete the objects used longest ago until the cache holds at most `max_bytes`.
    Shared by the processes of a data loader: each sees the same files."""
    files = []
    for root, _, names in os.walk(directory):
        for name in names:
            if name.endswith(PART):
                continue
            path = Path(root) / name
            try:
                stat = path.stat()
            except FileNotFoundError:  # removed by another process meanwhile
                continue
            files.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _, size, _ in files)
    for _, size, path in sorted(files):
        if total <= max_bytes:
            break
        path.unlink(missing_ok=True)
        total -= size
