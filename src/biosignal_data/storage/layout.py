"""Key layout of the raw layer.

    <root>/<repository>/<dataset>/<version>/
        data/<path exactly as published by the source>
        _manifest.json      every file: size, sha256, md5, source checksum; plus provenance
        _manifest.partial.json   progress of an unfinished ingest (removed on success)
        _SUCCESS            written last; its presence means the version is complete

`<root>` is the store prefix, `raw` by convention. A version folder is immutable once
`_SUCCESS` exists: a new upstream release is ingested into a new version folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from biosignal_data.catalog.models import VERSION_RE

DATA_DIR = "data"
MANIFEST = "_manifest.json"
PARTIAL_MANIFEST = "_manifest.partial.json"
SUCCESS = "_SUCCESS"


class UnsafePathError(ValueError):
    """A source path would escape its version folder or is otherwise unusable as a key."""


def safe_relpath(path: str) -> str:
    """Normalise a source-provided relative path and refuse anything that could escape."""
    cleaned = path.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(cleaned).parts
    if not parts or any(p in ("..", ".", "") for p in parts):
        raise UnsafePathError(f"refusing unsafe path {path!r}")
    return "/".join(parts)


def retrieval_version(now: datetime | None = None) -> str:
    """Version label for sources without releases: the UTC retrieval date."""
    return (now or datetime.now(UTC)).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class VersionLocation:
    dataset_id: str
    version: str
    root: str = "raw"

    def __post_init__(self) -> None:
        if not VERSION_RE.fullmatch(self.version):
            raise ValueError(f"version {self.version!r} is not usable as a folder name")

    @property
    def prefix(self) -> str:
        return f"{self.root.strip('/')}/{self.dataset_id}/{self.version}".lstrip("/")

    def data_key(self, relpath: str) -> str:
        return f"{self.prefix}/{DATA_DIR}/{safe_relpath(relpath)}"

    @property
    def data_prefix(self) -> str:
        """The folder of the copied files, for listing them all at once."""
        return f"{self.prefix}/{DATA_DIR}/"

    @property
    def manifest_key(self) -> str:
        return f"{self.prefix}/{MANIFEST}"

    @property
    def partial_manifest_key(self) -> str:
        return f"{self.prefix}/{PARTIAL_MANIFEST}"

    @property
    def success_key(self) -> str:
        return f"{self.prefix}/{SUCCESS}"


def dataset_prefix(dataset_id: str, root: str = "raw") -> str:
    return f"{root.strip('/')}/{dataset_id}".lstrip("/")
