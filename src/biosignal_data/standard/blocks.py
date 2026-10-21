"""The samples of the standard layer: compressed blocks in large objects, found through one
table.

    <version>/data/<batch>-<n>.bin   objects of at most OBJECT_BYTES: blocks one after another
    <version>/blocks.parquet         one row per block: where it is and what it holds

A block is one span of time of one recording's channels at one sampling rate: an array
(channels, samples) of the integers the source stored, compressed with Blosc (zstd, bit
shuffle; lossless). A block holds at most BLOCK_BYTES before compression, so a short
recording is one block, and many short recordings share an object.

Why this layout (2026-10-20): each request to the S3 storage costs 80-240 ms, whatever its
size. One Zarr group per recording made three objects of a few KB for a 10-second ECG, and
a reader paid four requests before the first sample (PTB-XL: 87,000 requests, 30 minutes,
to read 0.5 GB). Large objects with an offset index are how large training corpora are
stored (Megatron-LM indexed datasets, MosaicML Streaming: 64 MiB shards); 8 MB is the
usual size of an S3 byte-range read (AWS). A reader loads the index once, then reads a
span with one ranged GET per block, and a whole dataset with about one GET per object.
"""

from __future__ import annotations

import hashlib
import io
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numcodecs import Blosc

if TYPE_CHECKING:
    from biosignal_data.standard.sources import Source
    from biosignal_data.storage.stores import Store

BLOCK_BYTES = 8 * 2**20  # samples of one block, before compression
OBJECT_BYTES = 64 * 2**20  # an object is closed before it grows past this
# Measured on mitdb, chbmit and sleep-edfx: 4-17 % smaller than delta + zstd on EEG and
# PSG (2 % larger on ECG); level 9 writes about 6 times slower than 3, reads as fast.
ZSTD_LEVEL = 9
MERGE_GAP = 2**20  # blocks of one object closer than this are read with one request
DATA = "data"
TABLE = "blocks"

BLOCKS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("array", pa.string()),  # the channels of one sampling rate, e.g. "250Hz"
        ("first_sample", pa.int64()),  # of the array
        ("samples", pa.int64()),
        ("channels", pa.int32()),
        ("dtype", pa.string()),
        ("object", pa.string()),  # relative to the version folder
        ("offset", pa.int64()),  # in the object
        ("nbytes", pa.int64()),
        ("crc32", pa.int64()),  # of the compressed bytes
    ]
)


class CorruptBlockError(RuntimeError):
    """A block read back does not match its checksum, or does not decode to its shape."""


class MismatchError(RuntimeError):
    """The two readers of a source differ, or a block does not decode to its samples."""


def settings() -> dict[str, Any]:
    """The layout rules, for the manifest."""
    return {
        "codec": f"blosc zstd level {ZSTD_LEVEL}, bitshuffle",
        "block": f"one recording, one sampling rate, at most {BLOCK_BYTES} bytes before "
        "compression",
        "object": f"blocks one after another, at most {OBJECT_BYTES} bytes",
        "index": f"{TABLE}.parquet",
    }


def spans(length: int, channels: int, itemsize: int, step: int) -> list[tuple[int, int]]:
    """The [start, stop) sample spans of the blocks of an array: as long as BLOCK_BYTES
    allows, a whole number of `step` (samples per WFDB frame)."""
    per_block = BLOCK_BYTES // max(1, channels * itemsize)
    per_block = max(step, per_block - per_block % step)
    return [(s, min(s + per_block, length)) for s in range(0, length, per_block)]


def encode(data: np.ndarray) -> bytes:
    data = np.ascontiguousarray(data)
    codec = Blosc(cname="zstd", clevel=ZSTD_LEVEL, shuffle=Blosc.BITSHUFFLE)
    blob: bytes = codec.encode(data)
    return blob


def decode(blob: bytes, channels: int, samples: int, dtype: str) -> np.ndarray:
    raw = Blosc().decode(blob)
    data = np.frombuffer(raw, dtype=np.dtype(dtype))
    if data.size != channels * samples:
        raise CorruptBlockError(f"decoded {data.size} values, expected {channels}x{samples}")
    return data.reshape(channels, samples)


def checksum(blob: bytes) -> int:
    return zlib.crc32(blob)


def _identical(a: np.ndarray, b: np.ndarray, dtype: np.dtype[Any]) -> bool:
    """The same shape and the same stored bytes in `dtype`: NaN equals NaN (vitaldb's float
    tracks have NaN gaps; np.array_equal calls them different), any other difference does
    not."""
    little = dtype.newbyteorder("<")
    return a.shape == b.shape and (
        np.ascontiguousarray(a, dtype=little).tobytes()
        == np.ascontiguousarray(b, dtype=little).tobytes()
    )


def encode_source(
    recording_id: str, source: Source, check: Source
) -> tuple[list[tuple[dict[str, Any], bytes]], dict[tuple[str, int], str]]:
    """Every group of `source` as blocks: (row of blocks.parquet without `object` and
    `offset`, compressed bytes), and the SHA-256 of each channel's stored samples, by
    (array, index). `check` is a second reader of the same files, opened on its own: a
    reader that returns another file's samples (as pyedflib did) disagrees with it. Each
    block is decoded again and compared with the source before it is kept."""
    found: list[tuple[dict[str, Any], bytes]] = []
    hashes: dict[tuple[str, int], Any] = {}
    for index, group in enumerate(source.groups):
        n = len(group.channels)
        for start, stop in spans(group.length, n, group.dtype.itemsize, group.step):
            data = source.read(index, start, stop)
            if data.shape != (n, stop - start):
                raise MismatchError(
                    f"{group.name}: the source gave {data.shape} for {start}:{stop}"
                )
            if not _identical(data, check.read(index, start, stop), group.dtype):
                raise MismatchError(f"{group.name} {start}:{stop}: two readers of the file differ")
            blob = encode(data)
            if not _identical(decode(blob, n, stop - start, group.dtype.str), data, group.dtype):
                raise MismatchError(f"{group.name} {start}:{stop}: decoded differs from source")
            little = data.astype(group.dtype.newbyteorder("<"), copy=False)
            for i in range(n):
                hashes.setdefault((group.name, i), hashlib.sha256()).update(little[i].tobytes())
            found.append(
                (
                    {
                        "recording_id": recording_id, "array": group.name,
                        "first_sample": start, "samples": stop - start, "channels": n,
                        "dtype": group.dtype.str, "nbytes": len(blob), "crc32": checksum(blob),
                    },
                    blob,
                )
            )  # fmt: skip
    return found, {key: h.hexdigest() for key, h in hashes.items()}


class ObjectWriter:
    """Appends blocks to local files of at most OBJECT_BYTES, named `<name>-<n>.bin`.
    `add` returns where the block is; `objects` gives the files to upload."""

    def __init__(self, folder: Path, name: str) -> None:
        self._folder = folder
        self._name = name
        self._objects: list[tuple[str, Path]] = []
        self._size = OBJECT_BYTES  # no object open yet

    def add(self, blob: bytes) -> tuple[str, int]:
        if self._size + len(blob) > OBJECT_BYTES and self._size > 0:
            key = f"{DATA}/{self._name}-{len(self._objects):04d}.bin"
            path = self._folder / f"{self._name}-{len(self._objects):04d}.bin"
            path.write_bytes(b"")
            self._objects.append((key, path))
            self._size = 0
        key, path = self._objects[-1]
        with path.open("ab") as f:
            f.write(blob)
        offset = self._size
        self._size += len(blob)
        return key, offset

    @property
    def objects(self) -> list[tuple[str, Path]]:
        return list(self._objects)


@dataclass(frozen=True)
class BlockRef:
    """One row of blocks.parquet."""

    object: str
    offset: int
    nbytes: int
    first_sample: int
    samples: int
    channels: int
    dtype: str
    crc32: int


@dataclass(frozen=True)
class ArrayBlocks:
    """The blocks of one array of one recording, in time order."""

    blocks: tuple[BlockRef, ...]

    @property
    def length(self) -> int:
        """Samples of each channel."""
        last = self.blocks[-1]
        return last.first_sample + last.samples

    @property
    def channels(self) -> int:
        return self.blocks[0].channels

    def overlapping(self, first: int, last: int) -> list[BlockRef]:
        """The blocks with samples in [first, last)."""
        return [
            b for b in self.blocks if b.first_sample < last and first < b.first_sample + b.samples
        ]


def index(table: pa.Table) -> dict[tuple[str, str], ArrayBlocks]:
    """(recording, array) -> its blocks, from blocks.parquet."""
    found: dict[tuple[str, str], list[BlockRef]] = {}
    columns = table.to_pydict()
    for i in range(table.num_rows):
        found.setdefault((columns["recording_id"][i], columns["array"][i]), []).append(
            BlockRef(
                columns["object"][i], columns["offset"][i], columns["nbytes"][i],
                columns["first_sample"][i], columns["samples"][i], columns["channels"][i],
                columns["dtype"][i], columns["crc32"][i],
            )
        )  # fmt: skip
    return {
        key: ArrayBlocks(tuple(sorted(refs, key=lambda b: b.first_sample)))
        for key, refs in found.items()
    }


def unpack(ref: BlockRef, blob: bytes) -> np.ndarray:
    """The samples of a block from its compressed bytes, checked against its checksum."""
    if len(blob) != ref.nbytes or checksum(blob) != ref.crc32:
        raise CorruptBlockError(f"{ref.object} at {ref.offset}: checksum or size differs")
    return decode(blob, ref.channels, ref.samples, ref.dtype)


def span(array: ArrayBlocks, first: int, last: int, get: Callable[[BlockRef], bytes]) -> np.ndarray:
    """Samples [first, last) of every channel; `get` gives a block's compressed bytes."""
    last = min(last, array.length)
    if last <= first:
        return np.empty((array.channels, 0), dtype=np.dtype(array.blocks[0].dtype))
    parts = []
    for ref in array.overlapping(first, last):
        data = unpack(ref, get(ref))
        lo, hi = max(first, ref.first_sample), min(last, ref.first_sample + ref.samples)
        parts.append(data[:, lo - ref.first_sample : hi - ref.first_sample])
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def fetch(
    get_range: Callable[[str, int, int], bytes],
    refs: Iterable[BlockRef],
    *,
    threads: int,
) -> dict[BlockRef, bytes]:
    """The compressed bytes of many blocks, with as few requests as possible: blocks of one
    object closer than MERGE_GAP are read with one ranged GET. `get_range(object, offset,
    length)` reads from the version folder."""
    by_object: dict[str, list[BlockRef]] = {}
    for ref in set(refs):
        by_object.setdefault(ref.object, []).append(ref)
    reads: list[tuple[str, int, int, list[BlockRef]]] = []
    for obj, mine in by_object.items():
        mine.sort(key=lambda b: b.offset)
        start, end, group = mine[0].offset, mine[0].offset + mine[0].nbytes, [mine[0]]
        for ref in mine[1:]:
            if ref.offset - end <= MERGE_GAP:
                end = max(end, ref.offset + ref.nbytes)
                group.append(ref)
            else:
                reads.append((obj, start, end, group))
                start, end, group = ref.offset, ref.offset + ref.nbytes, [ref]
        reads.append((obj, start, end, group))

    def read(item: tuple[str, int, int, list[BlockRef]]) -> dict[BlockRef, bytes]:
        obj, start, end, group = item
        data = get_range(obj, start, end - start)
        if len(data) != end - start:
            raise CorruptBlockError(f"{obj}: got {len(data)} bytes of {end - start}")
        return {r: data[r.offset - start : r.offset - start + r.nbytes] for r in group}

    found: dict[BlockRef, bytes] = {}
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for part in pool.map(read, reads):
            found |= part
    return found


def store_arrays(
    store: Store,
    prefix: str,
    arrays: dict[tuple[str, str], np.ndarray],
    folder: Path,
    *,
    block_samples: dict[str, int] | None = None,
) -> pa.Table:
    """Write arrays that do not come from a source (tests, tools): (recording, array) ->
    (channels, samples), as blocks of `block_samples[array]` samples (default: as `spans`
    gives), into objects under `prefix`, and `prefix`/blocks.parquet. `folder`: for the
    local objects."""
    writer = ObjectWriter(folder, "arrays")
    rows = []
    for (recording_id, array), data in arrays.items():
        n, length = data.shape
        size = (block_samples or {}).get(array)
        cuts = (
            spans(length, n, data.dtype.itemsize, 1)
            if size is None
            else [(s, min(s + size, length)) for s in range(0, length, size)]
        )
        for start, stop in cuts:
            blob = encode(data[:, start:stop])
            obj, offset = writer.add(blob)
            rows.append(
                {"recording_id": recording_id, "array": array, "first_sample": start,
                 "samples": stop - start, "channels": n, "dtype": data.dtype.str,
                 "object": obj, "offset": offset, "nbytes": len(blob), "crc32": checksum(blob)}
            )  # fmt: skip
    for key, path in writer.objects:
        store.put_bytes(f"{prefix}/{key}", path.read_bytes(), "application/octet-stream")
    table = pa.Table.from_pylist(rows, schema=BLOCKS)
    out = io.BytesIO()
    pq.write_table(table, out, compression="zstd")
    store.put_bytes(f"{prefix}/{TABLE}.parquet", out.getvalue(), "application/vnd.apache.parquet")
    return table
