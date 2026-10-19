"""Find the recordings of a raw version from its file list.

WFDB (PhysioNet's own convention): the records are the entries of the `RECORDS` files; a
dataset without `RECORDS` has one record per `.hea` file. Annotation files are
`<record>.<annotator>`, with the annotators listed in `ANNOTATORS`. Without `ANNOTATORS`,
any `<record>.<extension>` that is not a known kind of file is tried as an annotation file,
and reported if it does not read as one.

EDF and BDF: every `.edf` / `.bdf` file is a recording, unless `RECORDS` lists the records
(then only those listed).
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

EDF_SUFFIXES = (".edf", ".bdf")
LISTS = ("RECORDS", "ANNOTATORS")
# Extensions that are never WFDB annotation files: signals, documents, tables, images, code.
NOT_ANNOTATIONS = frozenset(
    {
        "hea", "dat", "mat", "edf", "bdf", "rec", "txt", "csv", "tsv", "json", "xml", "html",
        "htm", "pdf", "md", "rst", "png", "jpg", "jpeg", "gif", "svg", "tif", "tiff", "bmp",
        "ps", "eps", "zip", "gz", "tar", "tgz", "bz2", "xz", "zst", "7z", "rar", "npy", "npz",
        "h5", "hdf5", "nwb", "xls", "xlsx", "doc", "docx", "ppt", "pptx", "wav", "mp3", "mp4",
        "avi", "m", "py", "c", "h", "sh", "r", "ipynb", "log", "ini", "cfg", "yaml", "yml",
        "sha256", "md5", "sig", "asc", "dcm", "fif", "set", "fdt", "vhdr", "vmrk", "eeg",
        "cnt", "gdf", "xdf",
    }
)  # fmt: skip
_EXTENSION = re.compile(r"^[A-Za-z0-9_]{1,16}$")


@dataclass(frozen=True)
class Unit:
    """One recording to convert."""

    id: str  # the record name or file path without its extension
    format: Literal["wfdb", "edf"]
    path: str  # WFDB: the record name (header without .hea); EDF: the file
    annotations: tuple[str, ...]  # WFDB annotation files of the record
    listed_annotators: bool  # annotators from ANNOTATORS: a file that does not read fails


@dataclass(frozen=True)
class Discovery:
    units: list[Unit]
    lists: list[str]  # the RECORDS and ANNOTATORS files read
    problems: list[str]  # e.g. a RECORDS entry without files


def discover(paths: Iterable[str], read_text: Callable[[str], str]) -> Discovery:
    """`paths`: every file of the raw version. `read_text`: the text of one of them."""
    paths = set(paths)
    lists = sorted(p for p in paths if posixpath.basename(p) in LISTS)
    annotators = {
        line.split()[0]
        for p in lists
        if p.endswith("ANNOTATORS")
        for line in read_text(p).splitlines()
        if line.strip() and not line.startswith("#")
    }
    records = [p for p in lists if p.endswith("RECORDS")]
    problems = []
    if records:
        names = []
        for list_path in records:
            folder = posixpath.dirname(list_path)
            for line in read_text(list_path).splitlines():
                entry = line.strip()
                if entry and not entry.startswith("#") and not entry.endswith("/"):
                    names.append(posixpath.normpath(posixpath.join(folder, entry)))
    else:
        names = [p[: -len(".hea")] for p in paths if p.endswith(".hea")]
        names += [p for p in paths if p.lower().endswith(EDF_SUFFIXES)]

    by_stem = _files_by_stem(paths)
    units: dict[str, Unit] = {}
    for name in sorted(set(names)):
        if name + ".hea" in paths:
            unit = Unit(name, "wfdb", name, (), bool(annotators))
        elif name.lower().endswith(EDF_SUFFIXES) and name in paths:
            unit = Unit(posixpath.splitext(name)[0], "edf", name, (), bool(annotators))
        else:
            problems.append(f"{name}: listed in RECORDS, but no such header or EDF file")
            continue
        is_annotator = annotators.__contains__ if annotators else _may_be_annotations
        found = tuple(sorted(f"{name}.{ext}" for ext in by_stem.get(name, ()) if is_annotator(ext)))
        unit = Unit(unit.id, unit.format, unit.path, found, unit.listed_annotators)
        if unit.id in units:
            raise ValueError(f"two recordings would get the id {unit.id!r}: rename one source")
        units[unit.id] = unit
    return Discovery(list(units.values()), lists, problems)


def _files_by_stem(paths: set[str]) -> dict[str, set[str]]:
    """`a/b.atr` -> {"a/b": {"atr"}}; also `a/b.edf.event` -> {"a/b.edf": {"event"}}."""
    found: dict[str, set[str]] = {}
    for p in paths:
        stem, dot, ext = p.rpartition(".")
        if dot and "/" not in ext:
            found.setdefault(stem, set()).add(ext)
    return found


def _may_be_annotations(ext: str) -> bool:
    return bool(_EXTENSION.match(ext)) and ext.lower() not in NOT_ANNOTATIONS
