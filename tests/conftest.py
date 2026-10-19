from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from pathlib import Path
from typing import BinaryIO

import pytest

from biosignal_data.catalog.models import Dataset
from biosignal_data.fetchers.base import Fetcher, FetchOptions, RemoteFile, ResolvedVersion
from biosignal_data.storage.stores import LocalStore


def make_dataset(**overrides: object) -> Dataset:
    data: dict[str, object] = {
        "id": "test/example",
        "name": "Example",
        "signals": ["EEG"],
        "source": {"type": "manual", "version": "1.0.0"},
        "access": {"tier": "open", "license": "CC0"},
    }
    data.update(overrides)
    return Dataset.model_validate(data)


class FakeFetcher(Fetcher):
    """Serves in-memory files; counts how often each file is opened."""

    def __init__(
        self,
        dataset: Dataset,
        files: dict[str, bytes],
        *,
        version: str = "1.0.0",
        checksums: dict[str, str] | None = None,
        sizes: dict[str, int] | None = None,
        fail: set[str] | None = None,
        options: FetchOptions | None = None,
    ) -> None:
        super().__init__(dataset, options)
        self.files = files
        self.version = version
        self.checksums = checksums or {}
        self.sizes = sizes or {}
        self.fail = fail or set()
        self.opened: dict[str, int] = {}

    def resolve_version(self) -> ResolvedVersion:
        return ResolvedVersion(self.version, "source")

    def _open(self, path: str) -> AbstractContextManager[BinaryIO]:
        self.opened[path] = self.opened.get(path, 0) + 1
        if path in self.fail:
            raise OSError(f"simulated failure for {path}")
        return nullcontext(io.BytesIO(self.files[path]))

    def list_files(self) -> Iterator[RemoteFile]:
        for path, body in self.files.items():
            yield RemoteFile(
                path=path,
                size=self.sizes.get(path, len(body)),
                checksum=self.checksums.get(path),
                opener=partial(self._open, path),
            )


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "bucket")
