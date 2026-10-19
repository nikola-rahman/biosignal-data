"""Generate catalog entries for OpenNeuro and NEMAR datasets from the EEGDash catalog.

EEGDash (https://github.com/eegdash/EEGDash) maintains `eegdash/dataset/dataset_summary.csv`,
a curated list of every EEG, iEEG, MEG, fNIRS and EMG dataset on OpenNeuro and NEMAR with
subject counts, hours and licenses. Rather than retyping ~720 entries, we import it.

Usage:
    uv run python scripts/import_eegdash.py <path to dataset_summary.csv> --commit <sha>

Only files previously written by this script (they carry the marker note) are overwritten,
so hand edits to other entries are never lost. Delete a generated file's marker note to take
it over by hand.
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
CATALOG = ROOT / "catalog"
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


def entry(row: dict[str, str], commit: str) -> tuple[Path, dict[str, Any]] | None:
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
    data: dict[str, Any] = {
        "id": f"{repository}/{accession}",
        "name": row["dataset_title"].strip() or accession,
        "signals": signals,
        "source": source,
        "access": {"tier": "open", "license": row["license"].strip() or None},
        "homepage": homepage,
        "stats": {k: v for k, v in stats.items() if v is not None},
        "notes": [f"{MARKER} ({SOURCE_URL.format(commit=commit)})."],
    }
    if doi:
        data["references"] = [f"https://doi.org/{doi}"]
    return CATALOG / repository / f"{accession}.yaml", data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--commit", required=True, help="EEGDash commit the CSV came from")
    args = parser.parse_args(argv)

    written = skipped = 0
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        result = entry(row, args.commit)
        if result is None:
            continue
        path, data = result
        if path.exists() and MARKER not in path.read_text(encoding="utf-8"):
            skipped += 1  # curated by hand; leave it alone
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
        )
        written += 1
    print(f"wrote {written} entries, left {skipped} hand-curated entries untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
