from __future__ import annotations

import os
from pathlib import Path

import pytest

from biosignal_data.training.cache import BlockCache, evict


def test_a_block_is_downloaded_once_then_read_from_disk(tmp_path: Path) -> None:
    calls: list[int] = []

    def download() -> bytes:
        calls.append(1)
        return b"block bytes"

    cache = BlockCache(tmp_path / "cache", max_bytes=10**9)
    assert cache.get("standard-blocks/x/1.0/data/b-0000.bin", 128, download) == b"block bytes"
    assert cache.get("standard-blocks/x/1.0/data/b-0000.bin", 128, download) == b"block bytes"
    assert len(calls) == 1
    other = cache.get("standard-blocks/x/1.0/data/b-0000.bin", 256, lambda: b"other")
    assert other == b"other"  # another offset of the same object is another block


def test_eviction_deletes_the_blocks_used_longest_ago(tmp_path: Path) -> None:
    for i, name in enumerate(["a", "b", "c"]):
        path = tmp_path / name
        path.write_bytes(b"x" * 100)
        os.utime(path, (1000 + i, 1000 + i))
    (tmp_path / ".d.123.part").write_bytes(b"x" * 1000)  # a write in progress: not counted
    evict(tmp_path, max_bytes=200)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".d.123.part", "b", "c"]


def test_the_cache_looks_for_old_blocks_only_after_writing_a_share_of_its_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biosignal_data.training import cache as module  # noqa: PLC0415

    scans: list[int] = []
    monkeypatch.setattr(module, "evict", lambda directory, max_bytes: scans.append(max_bytes))
    big = BlockCache(tmp_path / "c1", max_bytes=10**12)
    small = BlockCache(tmp_path / "c2", max_bytes=10**4)
    for i in range(20):
        big.get("o", i, lambda: b"x" * 1000)
        small.get("o", i, lambda: b"x" * 1000)
    assert scans == [10**4] * 20  # 1000 bytes > 1/100 of 10**4 every time; never for big
