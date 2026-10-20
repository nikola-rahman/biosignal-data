"""A local disk cache of the blocks of the standard layer.

The first read of a block downloads its compressed bytes (one ranged request) and writes
them to `<directory>/<object key>.<offset>` (a temporary file renamed into place, so a
crash never leaves part of a block); later reads come from the local copy. A standard
version never changes, so a cached block never goes stale.
"""

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from biosignal_data.storage.layout import safe_relpath

PART = ".part"
_WRITTEN: dict[Path, int] = {}  # bytes this process wrote to each cache since its last check
_LOCK = threading.Lock()


def _written(directory: Path, size: int) -> int:
    with _LOCK:
        _WRITTEN[directory] = _WRITTEN.get(directory, 0) + size
        return _WRITTEN[directory]


class BlockCache:
    """Blocks by (object key, offset). When the cache holds more than `max_bytes`, the
    blocks used longest ago are deleted.

    Finding what to delete means looking at every file of the cache, and the cache holds
    many, so it is done only after this process has written 1/100 of `max_bytes` since the
    last time; the count is kept per process, in `_WRITTEN`, for all the caches it opens.
    With N processes the cache can go over `max_bytes` by at most N/100 of it before old
    blocks are deleted."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self._directory = directory
        self._max_bytes = max_bytes

    def get(self, key: str, offset: int, download: Callable[[], bytes]) -> bytes:
        """The block at `offset` of object `key`; `download` reads it when not cached."""
        path = self._directory / f"{safe_relpath(key)}.{offset}"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = download()
            _write_atomic(path, data)
            if _written(self._directory, len(data)) > self._max_bytes // 100:
                _WRITTEN[self._directory] = 0
                evict(self._directory, self._max_bytes)
            return data
        with contextlib.suppress(FileNotFoundError):  # evicted by another process meanwhile
            os.utime(path)  # used now: evicted last
        return data


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
