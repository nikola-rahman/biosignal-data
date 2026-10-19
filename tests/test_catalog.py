from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from biosignal_data.catalog import AccessTier, Catalog, CatalogError, Infos, check_pair
from biosignal_data.fetchers import FETCHERS
from tests.conftest import make_dataset, make_info


def test_repository_files_are_valid_and_match() -> None:
    catalog, infos = Catalog.load(), Infos.load()
    assert len(catalog) > 800
    assert {d.source.type for d in catalog} <= set(FETCHERS)
    check_pair(catalog, infos)


def test_every_gated_dataset_says_how_to_get_it() -> None:
    catalog, infos = Catalog.load(), Infos.load()
    for d in infos:
        source = catalog.get(d.id).source.type
        if d.access.tier is not AccessTier.OPEN and source not in ("openneuro", "nemar"):
            assert d.access.steps, f"{d.id} is {d.access.tier} but lists no access steps"
        assert d.homepage or d.references, f"{d.id}: no link to the dataset"
        if d.access.tier is AccessTier.UNAVAILABLE:
            assert source == "unavailable", f"{d.id}: unavailable, but source type {source}"


@pytest.mark.parametrize("bad_id", ["noslash", "Upper/x", "a/b/c/d", "a/../b", "/x"])
def test_bad_ids_are_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        make_dataset(id=bad_id)
    with pytest.raises(ValidationError):
        make_info(id=bad_id)


def test_owner_namespaced_id_is_allowed() -> None:
    assert make_dataset(id="huggingface/Owner/Name").repository == "huggingface"


def test_the_catalog_holds_only_what_a_download_needs() -> None:
    with pytest.raises(ValidationError):
        make_dataset(name="Example")  # descriptions belong in docs/datasets.yaml
    with pytest.raises(ValidationError):
        make_info(source={"type": "manual"})  # download settings belong in catalog.yaml


def test_signals_are_lowercased() -> None:
    assert make_info(signals=["EEG", " ECG "]).signals == ["eeg", "ecg"]


def _write(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(yaml.safe_dump({"datasets": entries}))
    return path


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    entry = make_dataset().model_dump(mode="json")
    with pytest.raises(CatalogError, match="duplicate"):
        Catalog.load(_write(tmp_path / "catalog.yaml", [entry, entry]))


def test_related_must_exist(tmp_path: Path) -> None:
    info = make_info(related=["test/missing"]).model_dump(mode="json")
    with pytest.raises(CatalogError, match="test/missing"):
        Infos.load(_write(tmp_path / "datasets.yaml", [info]))


def test_both_files_must_list_the_same_datasets(tmp_path: Path) -> None:
    catalog = Catalog.load(_write(tmp_path / "c.yaml", [make_dataset().model_dump(mode="json")]))
    infos = Infos.load(
        _write(tmp_path / "d.yaml", [make_info(id="test/other").model_dump(mode="json")])
    )
    with pytest.raises(CatalogError, match="test/example"):
        check_pair(catalog, infos)


def test_filters() -> None:
    infos = Infos.load()
    ecg_open = infos.filter(signals=["ecg"], tiers=[AccessTier.OPEN])
    assert ecg_open and all(
        "ecg" in d.signals and d.access.tier is AccessTier.OPEN for d in ecg_open
    )
    physionet = Catalog.load().filter(repositories=["physionet"])
    assert physionet and {d.repository for d in physionet} == {"physionet"}


def test_get_suggests_close_ids() -> None:
    with pytest.raises(KeyError, match="physionet/mitdb"):
        Catalog.load().get("mitdb")
