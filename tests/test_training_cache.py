from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from biosignal_data.training.cache import WholeObjectCache, evict, read_range

RNG = np.random.default_rng(0)


def _sharded(path: Path) -> np.ndarray:
    data = RNG.integers(-30000, 30000, (4, 200_000)).astype(np.int16)
    root = zarr.open_group(store=LocalStore(path), mode="w", zarr_format=3)
    array = root.create_array(
        "256Hz", shape=data.shape, dtype=data.dtype, chunks=(4, 7680), shards=(4, 76800),
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", typesize=2)],
    )  # fmt: skip
    array[:] = data
    return data


def test_windows_read_through_the_cache_equal_a_direct_read(tmp_path: Path) -> None:
    # zarr's own CacheStore returned other samples here, without an error.
    data = _sharded(tmp_path / "source.zarr")
    cache = WholeObjectCache(
        LocalStore(tmp_path / "source.zarr", read_only=True), tmp_path / "cache",
        "standard/x/1.0.0/signals/r1.zarr", max_bytes=10**9,
    )  # fmt: skip
    array = zarr.open_group(store=cache, mode="r")["256Hz"]
    for start in RNG.integers(0, 200_000 - 7680, 40):  # twice over the same shards
        np.testing.assert_array_equal(array[:, start : start + 7680], data[:, start : start + 7680])
    for cached in (tmp_path / "cache").rglob("*"):
        if cached.is_file():  # whole objects only, each the size of its source
            source = (
                tmp_path
                / "source.zarr"
                / cached.relative_to(tmp_path / "cache/standard/x/1.0.0/signals/r1.zarr")
            )
            assert cached.stat().st_size == source.stat().st_size


def test_every_kind_of_byte_range_is_served_from_the_local_copy(tmp_path: Path) -> None:
    path = tmp_path / "object"
    path.write_bytes(bytes(range(100)))
    assert read_range(path, None) == bytes(range(100))
    assert read_range(path, RangeByteRequest(10, 20)) == bytes(range(10, 20))
    assert read_range(path, OffsetByteRequest(95)) == bytes(range(95, 100))
    assert read_range(path, SuffixByteRequest(3)) == bytes(range(97, 100))
    assert read_range(path, SuffixByteRequest(500)) == bytes(range(100))


def test_eviction_deletes_the_objects_used_longest_ago(tmp_path: Path) -> None:
    import os  # noqa: PLC0415

    for i, name in enumerate(["a", "b", "c"]):
        path = tmp_path / name
        path.write_bytes(b"x" * 100)
        os.utime(path, (1000 + i, 1000 + i))
    (tmp_path / ".d.123.part").write_bytes(b"x" * 1000)  # a write in progress: not counted
    evict(tmp_path, max_bytes=200)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".d.123.part", "b", "c"]
