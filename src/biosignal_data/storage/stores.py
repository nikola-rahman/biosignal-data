"""Object stores the raw layer can live in: S3 (or S3-compatible) and a local folder.

A store only knows keys and bytes. Layout lives in `layout.py`, integrity in `manifest.py`.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, ClassVar, Literal, Protocol
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from s3transfer.manager import TransferConfig, TransferManager
from s3transfer.subscribers import BaseSubscriber

if TYPE_CHECKING:
    import zarr.abc.store
    from mypy_boto3_s3 import S3Client

MiB = 1024 * 1024
# Uploads read a stream, so every part is held in memory until S3 has it. The part size is
# the smallest that S3's limit of 10,000 parts per object allows for the file, and at
# least PART_SIZE; a file whose size the source does not give gets UNKNOWN_SIZE_PART
# (objects up to 625 GiB).
PART_SIZE = 8 * MiB
UNKNOWN_SIZE_PART = 64 * MiB
MAX_PARTS = 10_000
# Per file uploading at the same time: parts read but not yet sent (one more is being read),
# and parts sent at the same time.
PARTS_IN_MEMORY = 2
PARTS_IN_FLIGHT = 4


@dataclass(frozen=True)
class Unfinished:
    """An upload that a stopped run left behind: it uses storage, but is not a listed file.
    On S3, an unfinished multipart upload; in a local folder, a temporary `.part` file."""

    key: str
    size: int
    started: datetime | None
    upload_id: str | None = None  # S3 only


class Store(Protocol):
    """Minimal key-value interface over an object store."""

    uri: str

    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int | None: ...
    def put_stream(self, key: str, stream: BinaryIO, size: int | None) -> None: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = ...) -> None: ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def open(self, key: str) -> BinaryIO: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str) -> Iterator[str]: ...
    def sizes(self, prefix: str) -> dict[str, int]: ...
    def unfinished(self, prefix: str) -> tuple[Unfinished, ...]: ...
    def discard(self, item: Unfinished) -> None: ...
    def zarr_store(self, key: str, *, read_only: bool) -> zarr.abc.store.Store: ...


@dataclass(frozen=True)
class S3Settings:
    """Everything needed to reach an S3-compatible storage. Nothing has a default: every
    provider differs (e.g. Hippius: endpoint https://s3.hippius.com, region
    `decentralized`, path addressing)."""

    endpoint: str
    region: str
    addressing: Literal["path", "virtual"]
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)

    KEY_ID_VAR: ClassVar[str] = "BIOSIGNAL_ACCESS_KEY_ID"
    SECRET_VAR: ClassVar[str] = "BIOSIGNAL_SECRET_ACCESS_KEY"  # noqa: S105 - a variable name

    @classmethod
    def from_env(
        cls, endpoint: str, region: str, addressing: Literal["path", "virtual"]
    ) -> S3Settings:
        """Read the keys from BIOSIGNAL_ACCESS_KEY_ID / BIOSIGNAL_SECRET_ACCESS_KEY only."""
        key_id = os.environ.get(cls.KEY_ID_VAR)
        secret = os.environ.get(cls.SECRET_VAR)
        if not key_id or not secret:
            raise ValueError(f"set {cls.KEY_ID_VAR} and {cls.SECRET_VAR} for the S3 storage")
        return cls(endpoint, region, addressing, key_id, secret)

    def client(self, connections: int) -> S3Client:
        """`connections`: the most requests this client runs at the same time."""
        # An explicit endpoint, region and keys: nothing is read from AWS config files or
        # AWS_* environment variables.
        return boto3.session.Session().client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": self.addressing},
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=connections,
            ),
        )


class S3Store:
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        settings: S3Settings | None = None,
        parallel_uploads: int = 1,
        client: S3Client | None = None,
        part_size: int = PART_SIZE,
    ) -> None:
        """Give `settings` (or, in tests, a ready `client`). `parallel_uploads`: files uploaded
        at the same time.

        All uploads share two transfer managers, so the memory they use is bounded:
        `parallel_uploads * (PARTS_IN_MEMORY + 1) * part size` for files with a known size,
        plus `(PARTS_IN_MEMORY + 1) * UNKNOWN_SIZE_PART` for files without one, which upload
        one at a time. The part size grows above `part_size` only for files bigger than
        10,000 parts of it (78 GiB at 8 MiB)."""
        if client is None:
            if settings is None:
                raise ValueError("S3Store needs settings (endpoint, region, addressing, keys)")
            client = settings.client(connections=(parallel_uploads + 1) * PARTS_IN_FLIGHT)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.settings = settings
        self.client: S3Client = client
        self.uri = f"s3://{bucket}/{self.prefix}".rstrip("/")
        self._part_size = part_size
        self._known_size = _transfer_manager(client, parallel_uploads, part_size)
        self._unknown_size = _transfer_manager(client, 1, max(part_size, UNKNOWN_SIZE_PART))

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

    def put_stream(self, key: str, stream: BinaryIO, size: int | None) -> None:
        """`size`: the size the source gives, or None. It sets the part size only; the
        stream is always read to its end, and one longer than 10,000 parts fails."""
        # s3transfer reads non-seekable streams sequentially, so a hashing wrapper sees
        # every byte once and in order while parts upload in parallel.
        manager = self._unknown_size if size is None else self._known_size
        # Given a size below one part, s3transfer reads the whole stream in one call; so the
        # size is given only where it makes the parts bigger, and a wrong small size from a
        # source cannot fill the memory.
        bigger_parts = size is not None and size > MAX_PARTS * self._part_size
        subscribers = [_KnownSize(size)] if bigger_parts and size is not None else []
        manager.upload(stream, self.bucket, self._key(key), subscribers=subscribers).result()

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

    def sizes(self, prefix: str) -> dict[str, int]:
        """Every key under `prefix` with its size: one listing instead of one HEAD per key."""
        strip = len(self.prefix) + 1 if self.prefix else 0
        paginator = self.client.get_paginator("list_objects_v2")
        return {
            item["Key"][strip:]: int(item["Size"])
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix))
            for item in page.get("Contents", [])
        }

    def unfinished(self, prefix: str) -> tuple[Unfinished, ...]:
        strip = len(self.prefix) + 1 if self.prefix else 0
        found = []
        paginator = self.client.get_paginator("list_multipart_uploads")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for upload in page.get("Uploads", []):
                key, upload_id = upload["Key"], upload["UploadId"]
                parts = self.client.get_paginator("list_parts").paginate(
                    Bucket=self.bucket, Key=key, UploadId=upload_id
                )
                size = sum(p["Size"] for page_ in parts for p in page_.get("Parts", []))
                found.append(Unfinished(key[strip:], size, upload.get("Initiated"), upload_id))
        return tuple(found)

    def discard(self, item: Unfinished) -> None:
        if item.upload_id is None:
            raise ValueError(f"{item.key}: not an S3 multipart upload")
        self.client.abort_multipart_upload(
            Bucket=self.bucket, Key=self._key(item.key), UploadId=item.upload_id
        )

    def zarr_store(self, key: str, *, read_only: bool) -> zarr.abc.store.Store:
        """A Zarr store for the Zarr hierarchy at `key`, reached with the same settings."""
        if self.settings is None:
            raise ValueError("S3Store.zarr_store needs the store's settings, not only a client")
        from zarr.storage import FsspecStore  # noqa: PLC0415 - Zarr loads only when used

        s3 = self.settings
        return FsspecStore.from_url(
            f"s3://{self.bucket}/{self._key(key)}",
            read_only=read_only,
            storage_options={
                "key": s3.access_key_id,
                "secret": s3.secret_access_key,
                "endpoint_url": s3.endpoint,
                "client_kwargs": {"region_name": s3.region},
                "config_kwargs": {
                    "s3": {"addressing_style": s3.addressing},
                    "signature_version": "s3v4",
                },
            },
        )


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

    def put_stream(self, key: str, stream: BinaryIO, size: int | None) -> None:
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

    def sizes(self, prefix: str) -> dict[str, int]:
        return {key: self._path(key).stat().st_size for key in self.list(prefix)}

    def unfinished(self, prefix: str) -> tuple[Unfinished, ...]:
        base = self._path(prefix) if prefix else self.root
        if not base.is_dir():
            return ()
        return tuple(
            Unfinished(
                path.relative_to(self.root).as_posix(),
                path.stat().st_size,
                datetime.fromtimestamp(path.stat().st_mtime, UTC),
            )
            for path in sorted(base.rglob(".*.part"))
            if path.is_file()
        )

    def discard(self, item: Unfinished) -> None:
        self._path(item.key).unlink(missing_ok=True)

    def zarr_store(self, key: str, *, read_only: bool) -> zarr.abc.store.Store:
        from zarr.storage import LocalStore as ZarrLocalStore  # noqa: PLC0415

        return ZarrLocalStore(self._path(key), read_only=read_only)


class _KnownSize(BaseSubscriber):
    """Tells s3transfer the size of a stream, which it cannot find out by itself."""

    def __init__(self, size: int) -> None:
        self.size = size

    def on_queued(self, future: Any, **kwargs: Any) -> None:
        future.meta.provide_transfer_size(self.size)


def _transfer_manager(client: S3Client, files: int, part_size: int) -> TransferManager:
    config = TransferConfig(
        multipart_threshold=part_size,
        multipart_chunksize=part_size,
        max_submission_concurrency=files,  # each reads one file's stream
        max_request_concurrency=files * PARTS_IN_FLIGHT,
        max_in_memory_upload_chunks=files * PARTS_IN_MEMORY,
    )
    return TransferManager(client, config)


def open_store(uri: str, s3: S3Settings | None = None, parallel_uploads: int = 1) -> Store:
    """`s3://bucket/prefix` with its `S3Settings`, or a local path / `file://` URI.
    `parallel_uploads`: files written at the same time (sizes the S3 connection pool)."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        if s3 is None:
            raise ValueError(
                "an s3:// destination needs S3Settings (endpoint, region, addressing, keys)"
            )
        return S3Store(parsed.netloc, parsed.path, settings=s3, parallel_uploads=parallel_uploads)
    if s3 is not None:
        raise ValueError("S3Settings apply to s3:// destinations only")
    if parsed.scheme in ("", "file"):
        return LocalStore(Path(parsed.path if parsed.scheme == "file" else uri))
    raise ValueError(f"unsupported store URI {uri!r}; use s3://bucket/prefix or a local path")
