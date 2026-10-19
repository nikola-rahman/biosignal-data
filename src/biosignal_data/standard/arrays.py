"""Write the signal groups of one recording as a Zarr v3 group, and check them.

Each group is one array `(channels, samples)` in the source's integer type, compressed
with Blosc (zstd and bit shuffle, lossless), a codec of the Zarr v3 specification. A chunk
is a whole number of 30-second epochs, at least 64 KiB; a shard is as many chunks as fit in
64 MiB. Arrays are written one shard at a time, and every shard is read back and compared
with the source before the next one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import zarr
import zarr.abc.store
from zarr.codecs import BloscCodec

from biosignal_data.standard.sources import Group, Source

EPOCH_S = 30
MIN_CHUNK_BYTES = 64 * 1024
MAX_SHARD_BYTES = 64 * 1024 * 1024
# Measured on mitdb, chbmit and sleep-edfx: 4-17 % smaller than delta + zstd on EEG and
# PSG (2 % larger on ECG); level 9 writes about 6 times slower than 3, reads as fast.
ZSTD_LEVEL = 9


class MismatchError(RuntimeError):
    """What was read back from the store differs from the source."""


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


def write_signals(source: Source, store: zarr.abc.store.Store, attributes: dict[str, Any]) -> None:
    """Write every group of `source` to a new Zarr group (an earlier attempt is replaced)."""
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
            array[:, start:stop] = data
            if not np.array_equal(array[:, start:stop], data):
                raise MismatchError(f"{group.name} {start}:{stop}: read back differs from source")


def _whole_steps(n: int, step: int) -> int:
    return step * math.ceil(n / step)
