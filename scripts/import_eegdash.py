"""Generate the OpenNeuro and NEMAR entries of catalog.yaml and docs/datasets.yaml from the
EEGDash catalog.

EEGDash (https://github.com/eegdash/EEGDash) maintains `eegdash/dataset/dataset_summary.csv`,
a curated list of every EEG, iEEG, MEG, fNIRS and EMG dataset on OpenNeuro and NEMAR with
subject counts, hours and licenses. Rather than retyping ~720 entries, we import it.

Usage:
    uv run python scripts/import_eegdash.py <path to dataset_summary.csv> --commit <sha>

Only entries previously written by this script (their description carries the marker note)
are replaced, so hand edits to other entries are never lost. Delete an entry's marker note
in docs/datasets.yaml to take it over by hand.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog.yaml"
DESCRIPTIONS = ROOT / "docs" / "datasets.yaml"
MARKER = "Generated from the EEGDash catalog"
SOURCE_URL = "https://github.com/eegdash/EEGDash/blob/{commit}/eegdash/dataset/dataset_summary.csv"


def _int(value: str) -> int | None:
    try:
        return int(float(value)) if value.strip() else None
    except ValueError:
        return None


def _float(value: str) -> float | None:
    try:
        return round(float(value), 2) if value.strip() else None
    except ValueError:
        return None


def _doi(value: str) -> str | None:
    value = value.strip().removeprefix("https://doi.org/").removeprefix("doi:").strip()
    return value or None


def entry(row: dict[str, str], commit: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The catalog entry and the description of one EEGDash row (None for rows to skip)."""
    accession = row["dataset"].strip()
    doi = _doi(row.get("doi") or "")
    if re.fullmatch(r"ds\d{6}", accession):
        repository = "openneuro"
        source: dict[str, Any] = {"type": "openneuro", "accession": accession}
        homepage = f"https://openneuro.org/datasets/{accession}"
    elif re.fullmatch(r"nm\d{6}", accession):
        repository = "nemar"
        source = {"type": "nemar", "accession": accession}
        homepage = f"https://doi.org/{doi}" if doi else "https://nemar.org"
    else:
        return None  # EEGDash-converted derivatives (e.g. EEG2025r*); the originals are listed
    signals = [s.strip() for s in row["record_modality"].split(",") if s.strip()] or ["eeg"]
    stats = {
        "subjects": _int(row["n_subjects"]),
        "recordings": _int(row["n_records"]),
        "hours": _float(row["duration_hours_total"]),
        "size_bytes": _int(row["size_bytes"]),
    }
    dataset_id = f"{repository}/{accession}"
    catalog_entry: dict[str, Any] = {"id": dataset_id, "source": source}
    if row["license"].strip():
        catalog_entry["license"] = row["license"].strip()
    info: dict[str, Any] = {
        "id": dataset_id,
        "name": row["dataset_title"].strip() or accession,
        "signals": signals,
        "access": {"tier": "open"},
        "homepage": homepage,
        "stats": {k: v for k, v in stats.items() if v is not None},
        "notes": [f"{MARKER} ({SOURCE_URL.format(commit=commit)})."],
    }
    if doi:
        info["references"] = [f"https://doi.org/{doi}"]
    return catalog_entry, info


def _read(path: Path, header: str) -> tuple[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    comments = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    return comments or header, yaml.safe_load(text)["datasets"]


def _write(path: Path, comments: str, entries: list[dict[str, Any]]) -> None:
    entries = sorted(entries, key=lambda e: e["id"])
    body = yaml.safe_dump({"datasets": entries}, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(comments + body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--commit", required=True, help="EEGDash commit the CSV came from")
    args = parser.parse_args(argv)

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    catalog_comments, catalog = _read(CATALOG, "")
    info_comments, infos = _read(DESCRIPTIONS, "")
    hand_made = {i["id"] for i in infos if not any(MARKER in n for n in i.get("notes", []))}
    catalog_by_id = {e["id"]: e for e in catalog}
    info_by_id = {i["id"]: i for i in infos}
    written = skipped = 0
    for row in rows:
        result = entry(row, args.commit)
        if result is None:
            continue
        catalog_entry, info = result
        if catalog_entry["id"] in hand_made:
            skipped += 1  # curated by hand; leave it alone
            continue
        catalog_by_id[catalog_entry["id"]] = catalog_entry
        info_by_id[info["id"]] = info
        written += 1
    _write(CATALOG, catalog_comments, list(catalog_by_id.values()))
    _write(DESCRIPTIONS, info_comments, list(info_by_id.values()))
    print(f"wrote {written} entries, left {skipped} hand-curated entries untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
