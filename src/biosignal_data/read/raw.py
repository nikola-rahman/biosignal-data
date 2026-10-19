"""Browse and fetch files from the raw layer.

from biosignal_data.read import RawDataset
ds = RawDataset.open("s3://my-bucket", "physionet/mitdb")    # latest complete version
ds.files[:3]                                                 # manifest entries
path = ds.local_path("100.hea")                              # downloaded once, cached
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from biosignal_data.storage.layout import SUCCESS, VersionLocation, dataset_prefix
from biosignal_data.storage.manifest import FileEntry, Manifest
from biosignal_data.storage.stores import Store, open_store


def default_cache_dir() -> Path:
    return Path(os.environ.get("BIOSIGNAL_DATA_CACHE", "~/.cache/biosignal-data")).expanduser()


def complete_versions(store: Store, dataset_id: str, root: str = "raw") -> list[str]:
    """Versions with a `_SUCCESS` marker, sorted (dates and dotted releases sort sensibly)."""
    prefix = dataset_prefix(dataset_id, root) + "/"
    found = {
        key[len(prefix) :].split("/", 1)[0]
        for key in store.list(prefix)
        if key.endswith("/" + SUCCESS) and key[len(prefix) :].count("/") == 1
    }
    return sorted(found, key=_version_key)


def stored_versions(store: Store, dataset_id: str, root: str = "raw") -> dict[str, bool]:
    """Every version folder of a dataset in the store -> complete (has `_SUCCESS`), sorted."""
    prefix = dataset_prefix(dataset_id, root) + "/"
    found: dict[str, bool] = {}
    for key in store.list(prefix):
        version, _, rest = key[len(prefix) :].partition("/")
        found[version] = found.get(version, False) or rest == SUCCESS
    return dict(sorted(found.items(), key=lambda item: _version_key(item[0])))


def _version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    parts = version.replace("-", ".").split(".")
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


@dataclass
class RawDataset:
    store: Store
    location: VersionLocation
    manifest: Manifest
    cache_dir: Path

    @classmethod
    def open(
        cls,
        store: Store | str,
        dataset_id: str,
        version: str | None = None,
        *,
        root: str = "raw",
        cache_dir: Path | None = None,
    ) -> RawDataset:
        store = open_store(store) if isinstance(store, str) else store
        if version is None:
            versions = complete_versions(store, dataset_id, root)
            if not versions:
                raise FileNotFoundError(f"no complete version of {dataset_id} in {store.uri}")
            version = versions[-1]
        location = VersionLocation(dataset_id, version, root)
        data = store.get_bytes(location.manifest_key)
        if data is None or not store.exists(location.success_key):
            raise FileNotFoundError(f"{store.uri}/{location.prefix} is not a complete version")
        return cls(store, location, Manifest.from_json(data), cache_dir or default_cache_dir())

    @property
    def files(self) -> list[FileEntry]:
        return self.manifest.files

    def glob(self, pattern: str) -> list[FileEntry]:
        return [f for f in self.files if fnmatch(f.path, pattern)]

    @property
    def local_root(self) -> Path:
        """The cache folder mirroring `data/`: point BIDS or other folder-based readers here
        after fetching the files they need with `local_paths`."""
        return self.cache_dir / self.location.prefix / "data"

    def uri(self, path: str) -> str:
        return f"{self.store.uri}/{self.location.data_key(path)}"

    def local_path(self, path: str, *, verify: bool = True) -> Path:
        """Download `path` into the cache (once) and return its local path."""
        entry = next((f for f in self.files if f.path == path), None)
        if entry is None:
            raise FileNotFoundError(f"{path} is not in the manifest of {self.location.prefix}")
        target = self.local_root / path
        if target.is_file() and target.stat().st_size == entry.size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        with self.store.open(self.location.data_key(path)) as src, tmp.open("wb") as out:
            while chunk := src.read(8 * 1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if verify and digest.hexdigest() != entry.sha256:
            tmp.unlink(missing_ok=True)
            raise OSError(f"{path}: downloaded bytes do not match the manifest SHA-256")
        shutil.move(tmp, target)
        return target

    def local_paths(self, pattern: str) -> list[Path]:
        return [self.local_path(f.path) for f in self.glob(pattern)]
