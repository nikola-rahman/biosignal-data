from __future__ import annotations

import posixpath
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from biosignal_data.catalog import Catalog, Infos
from biosignal_data.cli import app
from biosignal_data.docsgen import existing_pages, render, write_docs

runner = CliRunner()


def test_list_filters() -> None:
    result = runner.invoke(app, ["list", "--signal", "emg", "--access", "open"])
    assert result.exit_code == 0
    assert "meta/emg2pose" in result.stdout


def test_info_and_validate() -> None:
    assert "MIT-BIH" in runner.invoke(app, ["info", "physionet/mitdb"]).stdout
    as_json = runner.invoke(app, ["info", "physionet/mitdb", "--json"]).stdout
    assert '"slug": "mitdb"' in as_json and '"name": "MIT-BIH' in as_json
    assert runner.invoke(app, ["info", "nope/nope"]).exit_code == 1
    assert runner.invoke(app, ["dev", "validate"]).exit_code == 0


CONFIG = """
download:
  workers: 2
  parallel_datasets: 2
"""


def test_import_status_check(tmp_path: Path) -> None:
    config = tmp_path / "biosignal-data.yaml"
    config.write_text(CONFIG)
    src = tmp_path / "download"
    src.mkdir()
    (src / "x.edf").write_text("x")
    dest = str(tmp_path / "lake")
    args = ["import", "mni/open-ieeg-atlas", str(src), "--dest", dest, "--config", str(config)]
    assert runner.invoke(app, args).exit_code == 2  # --version is required
    args += ["--version", "2026-10-18"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "complete" in result.output

    check = ["check", "mni/open-ieeg-atlas", "--dest", dest, "--config", str(config)]
    assert runner.invoke(app, [*check, "--checksums"]).exit_code == 1  # no --version/--latest
    checked = runner.invoke(app, [*check, "--latest", "--checksums"])
    assert checked.exit_code == 0 and "2026-10-18: sizes and SHA-256 ok" in checked.output

    status = runner.invoke(
        app, ["status", "mni/open-ieeg-atlas", "--dest", dest, "--config", str(config)]
    )
    assert "2026-10-18" in status.output and "complete" in status.output
    assert "already complete" in runner.invoke(app, args).output


def test_selection_is_one_way_and_dest_is_required(tmp_path: Path) -> None:
    config = tmp_path / "biosignal-data.yaml"
    config.write_text(CONFIG)
    base = ["download", "--config", str(config), "--dest", str(tmp_path / "lake")]
    assert "select datasets one way" in runner.invoke(app, base).output
    mixed = runner.invoke(app, [*base, "physionet/mitdb", "--signal", "ecg"])
    assert "select datasets one way" in mixed.output
    no_dest = runner.invoke(app, ["download", "physionet/mitdb", "--config", str(config)])
    assert "missing --dest" in no_dest.output
    assert runner.invoke(app, ["status", "physionet/mitdb"]).exit_code == 2  # --dest required


def test_config_values_have_no_defaults(tmp_path: Path) -> None:
    config = tmp_path / "biosignal-data.yaml"
    config.write_text("download:\n  workers: 2\n")
    result = runner.invoke(
        app, ["download", "physionet/mitdb", "--dry-run", "--config", str(config)]
    )
    assert result.exit_code == 2 and "parallel_datasets" in result.output


def test_generated_docs_are_up_to_date() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "datasets"
    pages = render(Catalog.load(), Infos.load())
    for name, text in pages.items():
        assert (out / name).read_text(encoding="utf-8") == text, (
            f"docs/datasets/{name} is stale; run `biosignal-data dev docs`"
        )
    assert existing_pages(out) == set(pages), "docs/datasets/ has stale or missing pages"


def test_generated_docs_links_resolve() -> None:
    catalog, infos = Catalog.load(), Infos.load()
    pages = render(catalog, infos)
    assert all(f"{d.id}.md" in pages for d in catalog)
    assert [d.id for d in infos if not d.homepage and not d.references] == []
    for name, text in pages.items():
        for target in re.findall(r"\]\(([^)<>]+\.md)\)", text):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            assert resolved in pages, f"{name} links to missing page {target}"


def test_write_docs_removes_stale_pages(tmp_path: Path) -> None:
    catalog, infos = Catalog.load(), Infos.load()
    out = tmp_path / "datasets"
    write_docs(catalog, infos, out)
    (out / "old" / "deep").mkdir(parents=True)
    (out / "old" / "deep" / "gone.md").write_text("x")
    assert write_docs(catalog, infos, out, check=True) == ["old/deep/gone.md"]
    write_docs(catalog, infos, out)
    assert existing_pages(out) == set(render(catalog, infos))
    assert not (out / "old").exists()


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
    catalog_entry, info = module.entry(row, "abc")
    assert catalog_entry == {
        "id": "openneuro/ds000001",
        "source": {"type": "openneuro", "accession": "ds000001"},
        "license": "CC0",
    }
    assert info["signals"] == ["eeg", "emg"]
    assert info["stats"] == {"subjects": 3, "hours": 1.23, "size_bytes": 10}
    assert info["references"] == ["https://doi.org/10.1/x"]
    assert module.entry({**row, "dataset": "EEG2025r1"}, "abc") is None


def test_s3_destination_reads_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIOSIGNAL_ACCESS_KEY_ID", "hip_key")
    monkeypatch.setenv("BIOSIGNAL_SECRET_ACCESS_KEY", "secret")
    config = tmp_path / "biosignal-data.yaml"
    args = [
        "check",
        "physionet/mitdb",
        "--dest",
        "s3://bucket",
        "--latest",
        "--config",
        str(config),
    ]
    assert "no config file" in runner.invoke(app, args).output
    config.write_text("s3:\n  endpoint: https://s3.example\n  region: r1\n")
    assert "addressing" in runner.invoke(app, args).output  # missing value: an error


@pytest.mark.parametrize(
    "command", ["list", "info", "download", "import", "status", "check", "cleanup", "dev"]
)
def test_help_renders(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
