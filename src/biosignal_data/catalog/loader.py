"""Load and validate the two dataset files: catalog.yaml and docs/datasets.yaml."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from biosignal_data.catalog.models import AccessTier, Dataset, DatasetInfo

M = TypeVar("M", bound=BaseModel)
REPO_ROOT = Path(__file__).resolve().parents[3]


class CatalogError(ValueError):
    """A dataset file is invalid. The message names the file and the problem."""


def default_catalog_path() -> Path:
    return REPO_ROOT / "catalog.yaml"


def default_info_path() -> Path:
    return REPO_ROOT / "docs" / "datasets.yaml"


def _load(path: Path, model: type[M]) -> dict[str, M]:
    """`datasets:` entries of one file, by id, in file order."""
    try:
        # libyaml (C) parses the same YAML ~7x faster than PyYAML's pure-Python parser.
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), list):
        raise CatalogError(f"{path}: expected a top-level `datasets:` list")
    entries: dict[str, M] = {}
    for i, item in enumerate(raw["datasets"]):
        try:
            entry = model.model_validate(item)
        except ValidationError as exc:
            name = item.get("id", f"entry {i}") if isinstance(item, dict) else f"entry {i}"
            raise CatalogError(f"{path}: {name}: {exc}") from exc
        entry_id = str(getattr(entry, "id"))  # noqa: B009 - both models have an id
        if entry_id in entries:
            raise CatalogError(f"{path}: duplicate id {entry_id!r}")
        entries[entry_id] = entry
    return entries


@dataclass(frozen=True)
class _Entries(Generic[M]):
    entries: dict[str, M]
    path: Path

    def __iter__(self) -> Iterator[M]:
        return iter(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, dataset_id: object) -> bool:
        return dataset_id in self.entries

    def get(self, dataset_id: str) -> M:
        try:
            return self.entries[dataset_id]
        except KeyError:
            close = [i for i in self.entries if dataset_id.lower() in i.lower()][:5]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise KeyError(f"no dataset {dataset_id!r} in {self.path.name}.{hint}") from None


class Catalog(_Entries[Dataset]):
    """catalog.yaml: what a download needs, per dataset."""

    @classmethod
    def load(cls, path: Path | None = None) -> Catalog:
        path = path or default_catalog_path()
        return cls(_load(path, Dataset), path)

    def filter(self, *, repositories: Iterable[str] = ()) -> list[Dataset]:
        want = set(repositories)
        return [d for d in self if not want or d.repository in want]


class Infos(_Entries[DatasetInfo]):
    """docs/datasets.yaml: the descriptions of the datasets, for people."""

    @classmethod
    def load(cls, path: Path | None = None) -> Infos:
        path = path or default_info_path()
        infos = cls(_load(path, DatasetInfo), path)
        unknown = sorted((d.id, r) for d in infos for r in d.related if r not in infos)
        if unknown:
            listed = ", ".join(f"{d} -> {r}" for d, r in unknown)
            raise CatalogError(f"{path}: `related` points at ids not in the file: {listed}")
        return infos

    def filter(
        self,
        *,
        signals: Iterable[str] = (),
        tiers: Iterable[AccessTier] = (),
        repositories: Iterable[str] = (),
    ) -> list[DatasetInfo]:
        want_signals = {s.lower() for s in signals}
        want_tiers = set(tiers)
        want_repos = set(repositories)
        return [
            d
            for d in self
            if (not want_signals or want_signals & set(d.signals))
            and (not want_tiers or d.access.tier in want_tiers)
            and (not want_repos or d.repository in want_repos)
        ]


def check_pair(catalog: Catalog, infos: Infos) -> None:
    """Both files describe the same datasets."""
    only_catalog = sorted(set(catalog.entries) - set(infos.entries))
    only_infos = sorted(set(infos.entries) - set(catalog.entries))
    if only_catalog or only_infos:
        raise CatalogError(
            f"{catalog.path.name} and {infos.path.name} differ: only in the catalog: "
            f"{only_catalog[:10]}; only in the descriptions: {only_infos[:10]}"
        )
