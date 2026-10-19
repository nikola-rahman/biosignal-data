from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from biosignal_data.catalog import AccessTier, Catalog, CatalogError
from biosignal_data.fetchers import FETCHERS
from tests.conftest import make_dataset


def test_repository_catalog_is_valid() -> None:
    catalog = Catalog.load()
    assert len(catalog) > 800
    assert {d.source.type for d in catalog} <= set(FETCHERS)


def test_every_entry_documents_how_to_get_gated_data() -> None:
    for d in Catalog.load():
        if d.access.tier is not AccessTier.OPEN and d.source.type not in ("openneuro", "nemar"):
            assert d.access.steps, f"{d.id} is {d.access.tier} but lists no access steps"
        if d.source.type == "manual" and d.access.tier is not AccessTier.UNAVAILABLE:
            assert d.homepage or d.access.steps, f"{d.id}: manual source needs a pointer"


@pytest.mark.parametrize("bad_id", ["noslash", "Upper/x", "a/b/c/d", "a/../b", "/x"])
def test_bad_ids_are_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        make_dataset(id=bad_id)


def test_owner_namespaced_id_is_allowed() -> None:
    assert make_dataset(id="huggingface/Owner/Name").repository == "huggingface"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_dataset(colour="blue")


def test_signals_are_lowercased() -> None:
    assert make_dataset(signals=["EEG", " ECG "]).signals == ["eeg", "ecg"]


def test_file_must_match_id(tmp_path: Path) -> None:
    (tmp_path / "test").mkdir()
    entry = make_dataset().model_dump(mode="json")
    (tmp_path / "test" / "wrong-name.yaml").write_text(yaml.safe_dump(entry))
    with pytest.raises(CatalogError, match="must live at"):
        Catalog.load(tmp_path)


def test_related_must_exist(tmp_path: Path) -> None:
    (tmp_path / "test").mkdir()
    entry = make_dataset(related=["test/missing"]).model_dump(mode="json")
    (tmp_path / "test" / "example.yaml").write_text(yaml.safe_dump(entry))
    with pytest.raises(CatalogError, match="test/missing"):
        Catalog.load(tmp_path)


def test_filter() -> None:
    catalog = Catalog.load()
    ecg_open = catalog.filter(signals=["ecg"], tiers=[AccessTier.OPEN])
    assert ecg_open and all(
        "ecg" in d.signals and d.access.tier is AccessTier.OPEN for d in ecg_open
    )
    assert {d.repository for d in catalog.filter(repositories=["physionet"])} == {"physionet"}


def test_get_suggests_close_ids() -> None:
    with pytest.raises(KeyError, match="physionet/mitdb"):
        Catalog.load().get("mitdb")
