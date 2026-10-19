"""Object stores the raw layer can live in: S3 (or S3-compatible) and a local folder.

A store only knows keys and bytes. Layout lives in `layout.py`, integrity in `manifest.py`.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

MiB = 1024 * 1024


class Store(Protocol):
    """Minimal key-value interface over an object store."""

    uri: str

    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int | None: ...
    def put_stream(self, key: str, stream: BinaryIO) -> None: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = ...) -> None: ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def open(self, key: str) -> BinaryIO: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str) -> Iterator[str]: ...


class S3Store:
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: S3Client | None = None,
        multipart_chunksize: int = 64 * MiB,
        max_concurrency: int = 8,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client: S3Client = client or boto3.client("s3")
        self.uri = f"s3://{bucket}/{self.prefix}".rstrip("/")
        self._transfer = TransferConfig(
            multipart_threshold=multipart_chunksize,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=max_concurrency,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        return self.size(key) is not None

    def size(self, key: str) -> int | None:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return int(head["ContentLength"])

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        # boto3 reads non-seekable streams sequentially, so a hashing wrapper sees every byte
        # once and in order while parts upload in parallel.
        self.client.upload_fileobj(stream, self.bucket, self._key(key), Config=self._transfer)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/json") -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type
        )

    def get_bytes(self, key: str) -> bytes | None:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise
        return obj["Body"].read()

    def open(self, key: str) -> BinaryIO:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj["Body"]  # type: ignore[return-value]

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list(self, prefix: str) -> Iterator[str]:
        full = self._key(prefix)
        strip = len(self.prefix) + 1 if self.prefix else 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for item in page.get("Contents", []):
                yield item["Key"][strip:]


class LocalStore:
    """A folder on disk with the same layout. Useful for tests and for staging small sets."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.uri = self.root.as_uri()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"key {key!r} escapes the store root")
        return path

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int | None:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
        try:
            with tmp.open("wb") as out:
                shutil.copyfileobj(stream, out, length=8 * MiB)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/json") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def open(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list(self, prefix: str) -> Iterator[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            yield prefix
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".part"):
                yield path.relative_to(self.root).as_posix()


def open_store(uri: str) -> Store:
    """`s3://bucket/prefix` or a local path / `file://` URI."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return S3Store(parsed.netloc, parsed.path)
    if parsed.scheme in ("", "file"):
        return LocalStore(Path(parsed.path if parsed.scheme == "file" else uri))
    raise ValueError(f"unsupported store URI {uri!r}; use s3://bucket/prefix or a local path")
