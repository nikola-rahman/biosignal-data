"""The manifest written next to every ingested version, and stream hashing."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, BinaryIO

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = 1


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path under data/, exactly as published by the source.")
    size: int
    sha256: str
    md5: str
    source_checksum: str | None = Field(
        None, description="Checksum the source published, as '<algorithm>:<hex>', if any."
    )
    source_url: str | None = None


class MissingFile(BaseModel):
    """A file the source lists but does not serve: not in the version, kept on record."""

    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str
    source_url: str | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = MANIFEST_SCHEMA_VERSION
    dataset_id: str
    version: str
    version_origin: str = Field(
        description="'source' when the source published this version, 'retrieval-date' "
        "when the source has no releases and the version is the download date."
    )
    source: dict[str, Any] = Field(description="The catalog `source` block used for this ingest.")
    license: str | None = None
    retrieved_at: datetime
    completed_at: datetime | None = None
    tool: str
    files: list[FileEntry] = Field(default_factory=list)
    missing: list[MissingFile] = Field(
        default_factory=list,
        description="Files the source lists but answers do not exist; the version is "
        "complete without them.",
    )

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def to_json(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload["file_count"] = len(self.files)
        payload["total_bytes"] = self.total_bytes
        return json.dumps(payload, indent=2, sort_keys=False).encode()

    @classmethod
    def from_json(cls, data: bytes) -> Manifest:
        payload = json.loads(data)
        payload.pop("file_count", None)
        payload.pop("total_bytes", None)
        return cls.model_validate(payload)


def utcnow() -> datetime:
    return datetime.now(UTC)


class StoppedError(RuntimeError):
    """The run was stopped (Ctrl+C) while this file was being copied."""


class HashingReader:
    """Wraps a binary stream; hashes and counts bytes as they are read.

    `on_read` receives the size of each chunk (used for progress display)."""

    def __init__(
        self,
        stream: BinaryIO,
        on_read: Callable[[int], None] | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        self._stream = stream
        self._on_read = on_read
        self._stop = stop
        self._sha256 = hashlib.sha256()
        self._md5 = hashlib.md5(usedforsecurity=False)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if self._stop is not None and self._stop.is_set():
            raise StoppedError("the run was stopped")
        chunk = self._stream.read(size)
        if chunk:
            self._sha256.update(chunk)
            self._md5.update(chunk)
            self.bytes_read += len(chunk)
            if self._on_read:
                self._on_read(len(chunk))
        return chunk

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    @property
    def sha256(self) -> str:
        return self._sha256.hexdigest()

    @property
    def md5(self) -> str:
        return self._md5.hexdigest()
