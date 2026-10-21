"""The raw files that hold no signals, kept with a standard version: tables (subjects,
labels such as ptbxl_database.csv), documentation, annotation files of other formats. The
standard version then does not need the raw version.

    <version>/files.parquet             one row per file: path, object, offset, nbytes, sha256
    <version>/files/pack-<n>.bin        the files one after another, as published, in objects
                                        of at most blocks.OBJECT_BYTES

The same pattern as the blocks: a dataset can keep hundreds of thousands of small files
(ptb-xl-plus 1.0.1: about 280,000), and one object per file costs one request per file to
write and to read (80-240 ms each on our S3 storage). A file is read with one ranged
request, and checked against its SHA-256.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from typing import Any

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


def pack(store: Store, location: StandardLocation, files: Iterable[tuple[str, bytes]]) -> pa.Table:
    """Write `files` ((path, data), in the order given) as packs and files.parquet; return
    the table. Every upload carries its MD5, which the storage checks."""
    rows: list[dict[str, Any]] = []
    buffer = bytearray()
    n = 0

    def flush() -> None:
        nonlocal buffer, n
        if buffer:
            store.put_bytes(
                location.object_key(f"{PACK_PREFIX}{n:04d}.bin"), bytes(buffer),
                "application/octet-stream",
            )  # fmt: skip
            buffer, n = bytearray(), n + 1

    for path, data in files:
        if buffer and len(buffer) + len(data) > OBJECT_BYTES:
            flush()
        rows.append(
            {"path": path, "object": f"{PACK_PREFIX}{n:04d}.bin", "offset": len(buffer),
             "nbytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )  # fmt: skip
        buffer += data
    flush()
    table = pa.Table.from_pylist(rows, schema=FILES)
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
