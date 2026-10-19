from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from biosignal_data.catalog import Catalog
from biosignal_data.cli import app
from biosignal_data.docsgen import render

runner = CliRunner()


def test_list_filters() -> None:
    result = runner.invoke(app, ["list", "--signal", "emg", "--access", "open"])
    assert result.exit_code == 0
    assert "meta/emg2pose" in result.stdout


def test_info_and_validate() -> None:
    assert "MIT-BIH" in runner.invoke(app, ["info", "physionet/mitdb"]).stdout
    assert '"slug": "mitdb"' in runner.invoke(app, ["info", "physionet/mitdb", "--json"]).stdout
    assert runner.invoke(app, ["info", "nope/nope"]).exit_code == 1
    assert runner.invoke(app, ["dev", "validate"]).exit_code == 0


def test_download_from_dir_and_check(tmp_path: Path) -> None:
    src = tmp_path / "download"
    src.mkdir()
    (src / "x.txt").write_text("x")
    dest = str(tmp_path / "lake")
    args = [
        "download",
        "mni/open-ieeg-atlas",
        "--dest",
        dest,
        "--from-dir",
        str(src),
        "--version",
        "2026-10-18",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "complete" in result.output
    check = ["check", "mni/open-ieeg-atlas", "--dest", dest, "--checksums"]
    assert runner.invoke(app, check).exit_code == 1  # neither --version nor --latest
    checked = runner.invoke(app, [*check, "--latest"])
    assert checked.exit_code == 0 and "2026-10-18: ok" in checked.output
    info = runner.invoke(app, ["info", "mni/open-ieeg-atlas", "--dest", dest]).stdout
    assert "2026-10-18" in info and "complete" in info
    assert "already complete" in runner.invoke(app, args).output
    assert runner.invoke(app, ["download", "mni/open-ieeg-atlas"]).exit_code == 1  # no --dest


def test_generated_docs_are_up_to_date() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "datasets"
    pages = render(Catalog.load())
    for name, text in pages.items():
        assert (out / name).read_text(encoding="utf-8") == text, (
            f"docs/datasets/{name} is stale; run `biosignal-data dev docs`"
        )
    assert {p.name for p in out.glob("*.md")} == set(pages)


def test_eegdash_import_maps_rows() -> None:
    import importlib.util  # noqa: PLC0415

    script = Path(__file__).resolve().parents[1] / "scripts" / "import_eegdash.py"
    spec = importlib.util.spec_from_file_location("import_eegdash", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        "dataset": "ds000001",
        "dataset_title": "T",
        "record_modality": "eeg, emg",
        "n_subjects": "3.0",
        "n_records": "",
        "duration_hours_total": "1.234",
        "size_bytes": "10.0",
        "license": "CC0",
        "doi": "doi:10.1/x",
    }
    path, data = module.entry(row, "abc")
    assert path.name == "ds000001.yaml"
    assert data["signals"] == ["eeg", "emg"]
    assert data["stats"] == {"subjects": 3, "hours": 1.23, "size_bytes": 10}
    assert data["references"] == ["https://doi.org/10.1/x"]
    assert module.entry({**row, "dataset": "EEG2025r1"}, "abc") is None


def test_s3_destination_reads_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIOSIGNAL_ACCESS_KEY_ID", "hip_key")
    monkeypatch.setenv("BIOSIGNAL_SECRET_ACCESS_KEY", "secret")
    config = tmp_path / "biosignal-data.yaml"
    args = ["check", "x/y", "--dest", "s3://bucket", "--latest", "--config", str(config)]
    assert "no config file" in runner.invoke(app, args).output
    config.write_text("s3:\n  endpoint: https://s3.example\n  region: r1\n")
    assert "addressing" in runner.invoke(app, args).output  # missing value: an error
