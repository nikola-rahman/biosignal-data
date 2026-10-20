"""Write the signal groups of one recording as a Zarr v3 group, and check them.

Each group is one array `(channels, samples)` in the source's integer type, compressed
with Blosc (zstd and bit shuffle, lossless), a codec of the Zarr v3 specification. A chunk
is a whole number of 30-second epochs, at least 64 KiB; a shard is as many chunks as fit in
64 MiB. Arrays are written one shard at a time. Every shard is read by two independent
readers of the same raw files, which must agree.

The group is written to the local disk first, where every shard is decoded again and
compared with the source; then its files are uploaded, each with its MD5 (Content-MD5),
which the storage checks against what it received. Zarr writing straight to S3 took about
2 s of metadata requests per recording (listing, deleting, existence checks), most of the
time of a 10-second PTB-XL record (measured 2026-10-20).
"""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zarr.abc.store
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from biosignal_data.standard.sources import Group, Source
from biosignal_data.storage.stores import Store

EPOCH_S = 30
MIN_CHUNK_BYTES = 64 * 1024
MAX_SHARD_BYTES = 64 * 1024 * 1024
# Measured on mitdb, chbmit and sleep-edfx: 4-17 % smaller than delta + zstd on EEG and
# PSG (2 % larger on ECG); level 9 writes about 6 times slower than 3, reads as fast.
ZSTD_LEVEL = 9


class MismatchError(RuntimeError):
    """The samples written differ from the source, or the two readers of the source differ."""


@dataclass(frozen=True)
class ArrayLayout:
    chunk: int  # samples per chunk
    shard: int  # samples per shard (a whole number of chunks)


def layout(group: Group) -> ArrayLayout:
    """The chunk and shard length of a group, in samples."""
    bytes_per_sample = len(group.channels) * group.dtype.itemsize
    epoch = _whole_steps(max(1, round(EPOCH_S * group.rate)), group.step)
    chunk = epoch * max(1, math.ceil(MIN_CHUNK_BYTES / (epoch * bytes_per_sample)))
    chunk = max(1, min(chunk, group.length))
    shard = chunk * max(1, MAX_SHARD_BYTES // (chunk * bytes_per_sample))
    return ArrayLayout(chunk, min(shard, chunk * math.ceil(group.length / chunk) or chunk))


def settings() -> dict[str, Any]:
    """The layout rules, for the manifest."""
    return {
        "codecs": [f"blosc zstd level {ZSTD_LEVEL}, bitshuffle"],
        "chunk": f"whole {EPOCH_S} s epochs, at least {MIN_CHUNK_BYTES} bytes",
        "shard": f"whole chunks, at most {MAX_SHARD_BYTES} bytes",
    }


def write_signals(
    source: Source,
    check: Source,
    store: Store,
    key: str,
    work_dir: Path,
    attributes: dict[str, Any],
) -> None:
    """Write every group of `source` as the Zarr group `key` of `store`; objects of an
    earlier attempt that are not written again are removed. `check` is a second reader of
    the same files, opened on its own: a reader that returns another file's samples (as
    pyedflib did) disagrees with it, and the recording fails. The group is made in a new
    folder in `work_dir`, removed at the end."""
    folder = Path(tempfile.mkdtemp(prefix="zarr-", dir=work_dir))
    try:
        local = folder / "group.zarr"
        _write_local(source, check, LocalStore(local), attributes)
        files = {f"{key}/{p.relative_to(local).as_posix()}": p for p in local.rglob("*")}
        files = {k: p for k, p in files.items() if p.is_file()}
        stale = [k for k in store.list(key + "/") if k not in files]
        if stale:
            store.delete_many(stale)
        # The shards first, the metadata last: a group with metadata has all its data.
        for name in sorted(files, key=lambda k: (k.endswith("zarr.json"), -k.count("/"), k)):
            content = "application/json" if name.endswith(".json") else "application/octet-stream"
            store.put_bytes(name, files[name].read_bytes(), content)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _write_local(
    source: Source, check: Source, store: zarr.abc.store.Store, attributes: dict[str, Any]
) -> None:
    root = zarr.open_group(store=store, mode="w", zarr_format=3, attributes=attributes)
    for index, group in enumerate(source.groups):
        shape = layout(group)
        array = root.create_array(
            group.name,
            shape=(len(group.channels), group.length),
            dtype=group.dtype,
            chunks=(len(group.channels), shape.chunk),
            shards=(len(group.channels), shape.shard),
            compressors=[
                BloscCodec(
                    cname="zstd",
                    clevel=ZSTD_LEVEL,
                    shuffle="bitshuffle",
                    typesize=group.dtype.itemsize,
                )
            ],
            fill_value=0,
            dimension_names=("channel", "time"),
            attributes={
                "sampling_rate": group.rate,
                "channels": [
                    {k: v for k, v in asdict(c).items() if k != "source"} for c in group.channels
                ],
            },
        )
        for start in range(0, group.length, shape.shard):
            stop = min(start + shape.shard, group.length)
            data = source.read(index, start, stop)
            if data.shape != (len(group.channels), stop - start):
                raise MismatchError(
                    f"{group.name}: the source gave {data.shape} for {start}:{stop}"
                )
            if not np.array_equal(data, check.read(index, start, stop)):
                raise MismatchError(f"{group.name} {start}:{stop}: two readers of the file differ")
            array[:, start:stop] = data
            if not np.array_equal(array[:, start:stop], data):
                raise MismatchError(f"{group.name} {start}:{stop}: decoded differs from source")


def _whole_steps(n: int, step: int) -> int:
    return step * math.ceil(n / step)
