"""The raw version's files for the conversion processes: path -> (size, SHA-256) in an
SQLite file that each process opens read-only, instead of a copy of the whole plan in
each process (icentia11k 1.0: 1.6 million files, 4 GB a process). A process looks up only
the files of the recording it converts, so its memory does not grow with the dataset."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from biosignal_data.read.raw import fetch_file
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import FileEntry
from biosignal_data.storage.stores import Store


class Sized(Protocol):
    """What a recording's conversion reads of a file's entry (FileEntry, or Entry)."""

    @property
    def size(self) -> int: ...
    @property
    def sha256(self) -> str: ...


@dataclass(frozen=True)
class Entry:
    size: int
    sha256: str


def write(folder: Path, files: Iterable[FileEntry]) -> Path:
    """An index of `files` in a new file in `folder`; the caller removes it."""
    path = folder / f"files-{uuid.uuid4().hex}.sqlite"
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE files (path TEXT PRIMARY KEY, size INTEGER, sha256 TEXT)")
        rows = ((f.path, f.size, f.sha256) for f in files)
        db.executemany("INSERT INTO files VALUES (?, ?, ?)", rows)
        db.commit()
    finally:
        db.close()
    return path


class IndexedRaw:
    """has / entry / fetch_to of a raw version (as RawDataset), from an index (write)."""

    def __init__(self, store: Store, location: VersionLocation, index: Path) -> None:
        self.store, self.location = store, location
        self._db = sqlite3.connect(f"file:{index}?mode=ro", uri=True, check_same_thread=False)

    def _row(self, path: str) -> tuple[int, str] | None:
        query = "SELECT size, sha256 FROM files WHERE path = ?"
        row = self._db.execute(query, (path,)).fetchone()
        return None if row is None else (int(row[0]), str(row[1]))

    def has(self, path: str) -> bool:
        return self._row(path) is not None

    def entry(self, path: str) -> Entry:
        row = self._row(path)
        if row is None:
            raise FileNotFoundError(f"{path} is not in the manifest of {self.location.prefix}")
        return Entry(*row)

    def fetch_to(self, path: str, folder: Path) -> Path:
        entry = self.entry(path)
        return fetch_file(self.store, self.location, path, entry.size, entry.sha256, folder)
