"""Find the recordings of a raw version from its file list.

WFDB (PhysioNet's own convention): the records are the entries of the `RECORDS` files; a
dataset without `RECORDS` has one record per `.hea` file. Annotation files are
`<record>.<annotator>`, with the annotators listed in `ANNOTATORS`. Without `ANNOTATORS`,
any `<record>.<extension>` that is not a known kind of file is tried as an annotation file,
and reported if it does not read as one.

EDF and BDF: every `.edf` / `.bdf` file is a recording, unless `RECORDS` lists the records
(then only those listed).

A `RECORDS` line that names no files but is exactly one pair of existing records written
without a line break between them is read as those two, and reported as a problem.

A WFDB record that only shows the signal files of other records side by side (it uses two
or more signal files, and each is the only signal file of another record; apnea-ecg a01er
shows a01.dat and a01r.dat) is a view, not a recording: its samples are converted with
those records, so it is listed with them instead of being stored twice.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

EDF_SUFFIXES = (".edf", ".bdf")
LISTS = ("RECORDS", "ANNOTATORS")
# Extensions that are never WFDB annotation files: signals, documents, tables, images, code,
# WAVE settings (.xws).
NOT_ANNOTATIONS = frozenset(
    {
        "hea", "dat", "mat", "edf", "bdf", "rec", "txt", "csv", "tsv", "json", "xml", "html",
        "htm", "pdf", "md", "rst", "png", "jpg", "jpeg", "gif", "svg", "tif", "tiff", "bmp",
        "ps", "eps", "zip", "gz", "tar", "tgz", "bz2", "xz", "zst", "7z", "rar", "npy", "npz",
        "h5", "hdf5", "nwb", "xls", "xlsx", "doc", "docx", "ppt", "pptx", "wav", "mp3", "mp4",
        "avi", "m", "py", "c", "h", "sh", "r", "ipynb", "log", "ini", "cfg", "yaml", "yml",
        "sha256", "md5", "sig", "asc", "dcm", "fif", "set", "fdt", "vhdr", "vmrk", "eeg",
        "cnt", "gdf", "xdf", "xws",
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
    annotators: frozenset[str]  # from ANNOTATORS; empty if there is none
    problems: list[str]  # e.g. a RECORDS entry without files
    views: dict[str, tuple[Unit, tuple[str, ...]]] = field(default_factory=dict)
    # view record id -> (its unit, the records whose signal files it shows)


def discover(
    paths: Iterable[str],
    read_text: Callable[[str], str],
    read_texts: Callable[[list[str]], dict[str, str]],
) -> Discovery:
    """`paths`: every file of the raw version. `read_text`: the text of one of them;
    `read_texts`: the texts of many (the WFDB headers), read at the same time."""
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
                    name = posixpath.normpath(posixpath.join(folder, entry))
                    joined = None if _has_files(name, paths) else _joined(entry, folder, paths)
                    if joined:
                        problems.append(
                            f"{name}: two RECORDS entries without a line break between them; "
                            f"read as {joined[0]} and {joined[1]}"
                        )
                    names += joined or [name]
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
        is_annotator = annotators.__contains__ if annotators else may_be_annotations
        found = tuple(sorted(f"{name}.{ext}" for ext in by_stem.get(name, ()) if is_annotator(ext)))
        unit = Unit(unit.id, unit.format, unit.path, found, unit.listed_annotators)
        if unit.id in units:
            raise ValueError(f"two recordings would get the id {unit.id!r}: rename one source")
        units[unit.id] = unit
    views = _views(units, read_texts)
    kept = [u for u in units.values() if u.id not in views]
    return Discovery(kept, lists, frozenset(annotators), problems, views)


def _has_files(name: str, paths: set[str]) -> bool:
    return name + ".hea" in paths or (name.lower().endswith(EDF_SUFFIXES) and name in paths)


def _joined(entry: str, folder: str, paths: set[str]) -> list[str] | None:
    """The two records of a RECORDS line that is two entries without a line break between
    them (PTB-XL 1.0.3: `records100/21000/21837_lrrecords500/00000/00001_hr`), when exactly
    one split gives two records that exist; otherwise None."""
    splits = [
        [posixpath.normpath(posixpath.join(folder, e)) for e in (entry[:i], entry[i:])]
        for i in range(1, len(entry))
    ]
    found = [pair for pair in splits if all(_has_files(n, paths) for n in pair)]
    return found[0] if len(found) == 1 else None


def _views(
    units: dict[str, Unit], read_texts: Callable[[list[str]], dict[str, str]]
) -> dict[str, tuple[Unit, tuple[str, ...]]]:
    wfdb = {u.path: u for u in units.values() if u.format == "wfdb"}
    headers = read_texts([path + ".hea" for path in wfdb]) if wfdb else {}
    files = {
        path: signal_files(header_as_read(headers[path + ".hea"], path)[0], posixpath.dirname(path))
        for path in wfdb
    }
    # A signal file "belongs" to a record that uses it and no other file.
    owner = {fs[0]: path for path, fs in files.items() if fs is not None and len(fs) == 1}
    views = {}
    for path, fs in files.items():
        if fs is not None and len(fs) > 1 and all(f in owner and owner[f] != path for f in fs):
            views[wfdb[path].id] = (wfdb[path], tuple(sorted(wfdb[owner[f]].id for f in fs)))
    return views


def header_as_read(header: str, record: str) -> tuple[str, str | None]:
    """The header with a folder removed from its record name and signal file names, when the
    record line names `<folder>/<record name>` (ptb-xl-plus 1.0.1 median beats:
    `ge_median_beats_wfdb/00001_medians`, a folder the source does not have; the signal
    files are next to the header). WFDB reads `a/b` as record `a` with segments `b`, so such
    a header does not parse. Returns the text to read and a problem to report (None when
    the header is used as written)."""
    lines = header.splitlines(keepends=True)
    used = [i for i, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith("#")]
    first = used[0] if used else None
    if first is None:
        return header, None
    name = lines[first].split()[0]
    base = posixpath.basename(record)
    folder, _, last = name.rpartition("/")
    if not folder or last != base:
        return header, None
    prefix = folder + "/"
    lines[first] = lines[first].replace(name, base, 1)
    n_sig = int(lines[first].split()[1]) if len(lines[first].split()) > 1 else 0
    signal_line = 0
    for i in range(first + 1, len(lines)):
        if signal_line == n_sig:
            break
        if not lines[i].strip() or lines[i].lstrip().startswith("#"):
            continue
        file = lines[i].split()[0]
        if file.startswith(prefix):
            lines[i] = lines[i].replace(file, file[len(prefix) :], 1)
        signal_line += 1
    return "".join(lines), (
        f"{record}.hea: the record line names {name!r}, in a folder {folder!r} the source "
        "does not have; read as the record and signal files next to the header"
    )


def signal_files(header: str, folder: str) -> list[str] | None:
    """The distinct signal files a WFDB header names, in order; None for a multi-segment
    record (its first line names segments, not files)."""
    lines = [
        ln.split() for ln in header.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines or "/" in lines[0][0]:
        return None
    n_sig = int(lines[0][1]) if len(lines[0]) > 1 else 0
    found = [posixpath.join(folder, ln[0]) for ln in lines[1 : 1 + n_sig]]
    return list(dict.fromkeys(found))


def _files_by_stem(paths: set[str]) -> dict[str, set[str]]:
    """`a/b.atr` -> {"a/b": {"atr"}}; also `a/b.edf.event` -> {"a/b.edf": {"event"}}."""
    found: dict[str, set[str]] = {}
    for p in paths:
        stem, dot, ext = p.rpartition(".")
        if dot and "/" not in ext:
            found.setdefault(stem, set()).add(ext)
    return found


def may_be_annotations(ext: str) -> bool:
    return bool(_EXTENSION.match(ext)) and ext.lower() not in NOT_ANNOTATIONS
