"""The raw files that hold no signals, kept with a standard version: tables (subjects,
labels such as ptbxl_database.csv), documentation, annotation files of other formats. The
standard version then does not need the raw version.

    <version>/files.parquet             one row per file: path, object, offset, nbytes, sha256
    <version>/files/pack-<n>.bin        the files one after another, as published, in objects
                                        of at most blocks.OBJECT_BYTES
    <version>/files/large-<n>.bin       one file larger than that, as published

The same pattern as the blocks: a dataset can keep hundreds of thousands of small files
(ptb-xl-plus 1.0.1: about 280,000), and one object per file costs one request per file to
write and to read (80-240 ms each on our S3 storage). A file is read with one ranged
request, and checked against its SHA-256.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from biosignal_data.standard.blocks import OBJECT_BYTES
from biosignal_data.standard.layout import StandardLocation
from biosignal_data.storage.stores import Store

FILES = pa.schema(
    [
        ("path", pa.string()),  # in the raw version, as published
        ("object", pa.string()),  # files/pack-<n>.bin, relative to the version
        ("offset", pa.int64()),
        ("nbytes", pa.int64()),
        ("sha256", pa.string()),
    ]
)
PACK_PREFIX = "files/pack-"
UPLOADS = 4  # packs uploaded at the same time (one at a time: ~1 min for 1 GB, 2026-10-21)
log = logging.getLogger(__name__)


LARGE_PREFIX = "files/large-"


class _Hashing(io.RawIOBase):
    """A stream that computes the SHA-256 of what is read from it, in order."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream, self.sha256, self.nbytes = stream, hashlib.sha256(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._stream.read(len(buffer))
        buffer[: len(data)] = data
        self.sha256.update(data)
        self.nbytes += len(data)
        return len(data)


def put_large(
    store: Store, location: StandardLocation, n: int, path: str, stream: BinaryIO,
    size: int, sha256: str,
) -> dict[str, Any]:  # fmt: skip
    """Stream one large file to files/large-<n>.bin, checked against `size` and `sha256`
    (the object is removed if it differs); return its row of files.parquet."""
    obj = f"{LARGE_PREFIX}{n:04d}.bin"
    key = location.object_key(obj)
    hashing = _Hashing(stream)
    store.put_stream(key, io.BufferedReader(hashing, 8 * 2**20), size)
    if hashing.nbytes != size or hashing.sha256.hexdigest() != sha256:
        store.delete(key)
        raise RuntimeError(f"{path}: differs from the raw manifest")
    log.info("%s: %s (%s, %.0f MiB) written", location.dataset_id, obj, path, size / 2**20)
    return {"path": path, "object": obj, "offset": 0, "nbytes": size, "sha256": sha256}


def pack(
    store: Store, location: StandardLocation, files: Iterable[tuple[str, bytes]],
    written: Iterable[dict[str, Any]] = (),
) -> pa.Table:  # fmt: skip
    """Write `files` ((path, data), in the order given) as packs, and files.parquet with
    their rows and the rows `written` (files already stored, put_large); return the table. Every
    upload carries its MD5, which the storage checks. Packs are uploaded UPLOADS at a time;
    at most UPLOADS packs wait in memory."""
    rows: list[dict[str, Any]] = []
    buffer = bytearray()
    n = 0
    uploads: deque[Future[None]] = deque()

    def put(key: str, data: bytes) -> None:
        store.put_bytes(key, data, "application/octet-stream")
        log.info("%s: %s (%.0f MiB) written", location.dataset_id, key.rsplit("/", 1)[-1],
                 len(data) / 2**20)  # fmt: skip

    def flush(pool: ThreadPoolExecutor) -> None:
        nonlocal buffer, n
        if buffer:
            while len(uploads) >= UPLOADS:
                uploads.popleft().result()
            key = location.object_key(f"{PACK_PREFIX}{n:04d}.bin")
            uploads.append(pool.submit(put, key, bytes(buffer)))
            buffer, n = bytearray(), n + 1

    with ThreadPoolExecutor(max_workers=UPLOADS) as pool:
        for path, data in files:
            if buffer and len(buffer) + len(data) > OBJECT_BYTES:
                flush(pool)
            rows.append(
                {"path": path, "object": f"{PACK_PREFIX}{n:04d}.bin", "offset": len(buffer),
                 "nbytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )  # fmt: skip
            buffer += data
        flush(pool)
        while uploads:
            uploads.popleft().result()
    table = pa.Table.from_pylist(rows + list(written), schema=FILES)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    store.put_bytes(location.table_key("files"), sink.getvalue(), "application/vnd.apache.parquet")
    return table


def index(store: Store, location: StandardLocation) -> dict[str, dict[str, Any]]:
    """path -> its row of files.parquet."""
    data = store.get_bytes(location.table_key("files"))
    if data is None:
        raise FileNotFoundError(location.table_key("files"))
    return {r["path"]: r for r in pq.read_table(io.BytesIO(data)).to_pylist()}


def read(
    store: Store, location: StandardLocation, path: str,
    rows: dict[str, dict[str, Any]] | None = None,
) -> bytes:  # fmt: skip
    """One kept file, checked against its SHA-256. `rows`: `index(...)`, when many files
    of one version are read."""
    rows = index(store, location) if rows is None else rows
    if path not in rows:
        raise FileNotFoundError(f"{location.prefix}: {path} is not kept (files.parquet)")
    row = rows[path]
    key = location.object_key(row["object"])
    data = store.get_range(key, row["offset"], row["nbytes"]) if row["nbytes"] else b""
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"{key}: {path} differs from its SHA-256 in files.parquet")
    return data
