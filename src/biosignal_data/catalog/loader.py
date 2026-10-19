"""Load and validate the YAML catalog."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from biosignal_data.catalog.models import AccessTier, Dataset


class CatalogError(ValueError):
    """A catalog file is invalid. The message names the file and the problem."""


def default_catalog_dir() -> Path:
    """The `catalog/` folder at the repository root."""
    return Path(__file__).resolve().parents[3] / "catalog"


def path_for(dataset_id: str, root: Path) -> Path:
    """Where the entry for `dataset_id` lives: the id is the path, plus `.yaml`."""
    return root / f"{dataset_id}.yaml"


def load_file(path: Path) -> Dataset:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = Dataset.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise CatalogError(f"{path}: {exc}") from exc
    expected = Path(f"{dataset.id}.yaml")
    if not path.as_posix().endswith("/" + expected.as_posix()):
        raise CatalogError(f"{path}: id {dataset.id!r} must live at catalog/{expected}")
    return dataset


@dataclass(frozen=True)
class Catalog:
    datasets: dict[str, Dataset]

    @classmethod
    def load(cls, root: Path | None = None) -> Catalog:
        root = root or default_catalog_dir()
        datasets: dict[str, Dataset] = {}
        for path in sorted(root.rglob("*.yaml")):
            dataset = load_file(path)
            if dataset.id in datasets:
                raise CatalogError(f"{path}: duplicate id {dataset.id!r}")
            datasets[dataset.id] = dataset
        unknown = {(d.id, r) for d in datasets.values() for r in d.related if r not in datasets}
        if unknown:
            listed = ", ".join(f"{d} -> {r}" for d, r in sorted(unknown))
            raise CatalogError(f"`related` points at ids not in the catalog: {listed}")
        return cls(datasets)

    def __iter__(self) -> Iterator[Dataset]:
        return iter(self.datasets.values())

    def __len__(self) -> int:
        return len(self.datasets)

    def get(self, dataset_id: str) -> Dataset:
        try:
            return self.datasets[dataset_id]
        except KeyError:
            close = [i for i in self.datasets if dataset_id.lower() in i.lower()][:5]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise KeyError(f"no dataset {dataset_id!r} in the catalog.{hint}") from None

    def filter(
        self,
        *,
        signals: Iterable[str] = (),
        tiers: Iterable[AccessTier] = (),
        repositories: Iterable[str] = (),
    ) -> list[Dataset]:
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
