"""Find the recordings of a raw version from its file list.

WFDB (PhysioNet's own convention): the records are the entries of the `RECORDS` files; a
dataset without `RECORDS` has one record per `.hea` file. Annotation files are
`<record>.<annotator>`, with the annotators listed in `ANNOTATORS`. Without `ANNOTATORS`,
any `<record>.<extension>` that is not a known kind of file is tried as an annotation file,
and reported if it does not read as one.

EDF and BDF: every `.edf` / `.bdf` file is a recording, unless `RECORDS` lists the records
(then only those listed). VitalDB: every `.vital` file is a recording.

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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from biosignal_data.standard.rules import EdfAnnotationFiles, UnitCorrection

EDF_SUFFIXES = (".edf", ".bdf")
VITAL_SUFFIX = ".vital"  # VitalDB (VitalRecorder): every such file is a recording
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
        "cnt", "gdf", "xdf", "xws", "vital",
    }
)  # fmt: skip
_EXTENSION = re.compile(r"^[A-Za-z0-9_]{1,16}$")


@dataclass(frozen=True)
class Unit:
    """One recording to convert."""

    id: str  # the record name or file path without its extension
    format: Literal["wfdb", "edf", "vital"]
    path: str  # WFDB: the record name (header without .hea); EDF: the file
    annotations: tuple[str, ...]  # WFDB annotation files of the record
    listed_annotators: bool  # annotators from ANNOTATORS: a file that does not read fails
    edf_annotations: tuple[str, ...] = ()  # EDF+ annotation files (standard.yaml rule)
    # A segment of a multi-segment WFDB record, whose own header gives no start: its start,
    # from its record's header (the record's start and the frames of the segments before it).
    start_time: datetime | None = None
    unit_corrections: tuple[UnitCorrection, ...] = ()  # standard.yaml rules of the dataset


@dataclass(frozen=True)
class Discovery:
    units: list[Unit]
    lists: list[str]  # the RECORDS and ANNOTATORS files read
    annotators: frozenset[str]  # from ANNOTATORS; empty if there is none
    problems: list[str]  # e.g. a RECORDS entry without files
    views: dict[str, tuple[Unit, tuple[str, ...]]] = field(default_factory=dict)
    # view record id -> (its unit, the records whose signal files it shows)
    segmented: tuple[str, ...] = ()  # headers of multi-segment records and their layouts


def discover(
    paths: Iterable[str],
    read_text: Callable[[str], str],
    read_texts: Callable[[list[str]], dict[str, str]],
    edf_annotation_files: EdfAnnotationFiles | None = None,
) -> Discovery:
    """`paths`: every file of the raw version. `read_text`: the text of one of them;
    `read_texts`: the texts of many (the WFDB headers), read at the same time.
    `edf_annotation_files`: the dataset's rule (standard.yaml) for EDF+ files that hold
    only the annotations of an EDF recording."""
    paths = set(paths)
    lists = sorted(p for p in paths if posixpath.basename(p) in LISTS)
    # All at the same time: icentia11k 1.0 has a RECORDS list in each of 11,000 folders.
    texts = read_texts(lists) if lists else {}
    annotators = {
        line.split()[0]
        for p in lists
        if p.endswith("ANNOTATORS")
        for line in texts[p].splitlines()
        if line.strip() and not line.startswith("#")
    }
    records = [p for p in lists if p.endswith("RECORDS")]
    problems = []
    if records:
        names = []
        for list_path in records:
            folder = posixpath.dirname(list_path)
            for line in texts[list_path].splitlines():
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

    names += [p for p in paths if p.lower().endswith(VITAL_SUFFIX)]  # never in RECORDS
    by_stem = _files_by_stem(paths)
    units: dict[str, Unit] = {}
    for name in sorted(set(names)):
        unit = _unit(name, paths, bool(annotators))
        if unit is None:
            problems.append(f"{name}: listed in RECORDS, but no such header or EDF file")
            continue
        is_annotator = annotators.__contains__ if annotators else may_be_annotations
        found = tuple(sorted(f"{name}.{ext}" for ext in by_stem.get(name, ()) if is_annotator(ext)))
        unit = Unit(unit.id, unit.format, unit.path, found, unit.listed_annotators)
        if unit.id in units:
            raise ValueError(f"two recordings would get the id {unit.id!r}: rename one source")
        units[unit.id] = unit
    files, headers = _signal_files(units, read_texts)
    is_annotator = annotators.__contains__ if annotators else may_be_annotations
    units, segmented = _segments_as_recordings(
        units, files, headers, paths, by_stem, is_annotator, read_texts, problems
    )
    units = _without_signal_files(units, files)
    if edf_annotation_files is not None:
        units = _with_edf_annotations(units, paths, edf_annotation_files, problems)
    views = _views(units, files)
    kept = [u for u in units.values() if u.id not in views]
    return Discovery(kept, lists, frozenset(annotators), problems, views, segmented)


def _unit(name: str, paths: set[str], listed_annotators: bool) -> Unit | None:
    """The recording `name` (a WFDB record name, an EDF or a .vital file), if it is there."""
    if name + ".hea" in paths:
        return Unit(name, "wfdb", name, (), listed_annotators)
    if name in paths and name.lower().endswith(EDF_SUFFIXES):
        return Unit(posixpath.splitext(name)[0], "edf", name, (), listed_annotators)
    if name in paths and name.lower().endswith(VITAL_SUFFIX):
        return Unit(posixpath.splitext(name)[0], "vital", name, (), listed_annotators)
    return None


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


def _signal_files(
    units: dict[str, Unit], read_texts: Callable[[list[str]], dict[str, str]]
) -> tuple[dict[str, list[str] | None], dict[str, str]]:
    """WFDB record path -> the signal files its header names (None: multi-segment), and
    the headers read (path -> text)."""
    wfdb = [u.path for u in units.values() if u.format == "wfdb"]
    headers = read_texts([path + ".hea" for path in wfdb]) if wfdb else {}
    files = {
        path: signal_files(header_as_read(headers[path + ".hea"], path)[0], posixpath.dirname(path))
        for path in wfdb
    }
    return files, headers


def _segments_as_recordings(
    units: dict[str, Unit], files: dict[str, list[str] | None], headers: dict[str, str],
    paths: set[str], by_stem: dict[str, set[str]], is_annotator: Callable[[str], bool],
    read_texts: Callable[[list[str]], dict[str, str]], problems: list[str],
) -> tuple[dict[str, Unit], tuple[str, ...]]:  # fmt: skip
    """Each multi-segment WFDB record as its segments: every segment with samples is a
    single-segment record (its own header and signal files) and becomes a recording, with
    the start its record's header gives it (mimic4wdb 0.1.0: 200 records, 3,859 segments,
    238 channel sets: a monitor adds and removes signals). Gaps (`~`) take time and hold no
    data; the layout segment (0 samples) holds no data. `files` gets the segments' signal
    files. Returns the units and the headers of the records and their layouts."""

    def multi_segment(unit: Unit) -> bool:
        if unit.format != "wfdb" or files.get(unit.path) is not None:
            return False
        line = _record_line(header_as_read(headers[unit.path + ".hea"], unit.path)[0])
        return line is not None and "/" in line[0]

    records = [u for u in units.values() if multi_segment(u)]
    if not records:
        return units, ()
    kept = {k: u for k, u in units.items() if u not in records}
    segments: list[Unit] = []
    segmented: list[str] = []
    for record in records:
        header = header_as_read(headers[record.path + ".hea"], record.path)[0]
        found, layouts = _segments(header, record.path)
        segmented += [record.path + ".hea", *(p for p in layouts if p in paths)]
        for path, start in found:
            if path + ".hea" not in paths:
                problems.append(f"{path}: a segment of {record.path}, but no such header")
                continue
            annotations = tuple(
                sorted(f"{path}.{ext}" for ext in by_stem.get(path, ()) if is_annotator(ext))
            )
            unit = Unit(path, "wfdb", path, annotations, record.listed_annotators, (), start)
            if unit.id in kept:
                raise ValueError(f"two recordings would get the id {unit.id!r}: rename one source")
            kept[unit.id] = unit
            segments.append(unit)
    texts = read_texts([u.path + ".hea" for u in segments]) if segments else {}
    for unit in segments:
        header = header_as_read(texts[unit.path + ".hea"], unit.path)[0]
        named = signal_files(header, posixpath.dirname(unit.path))
        if named is None:  # a segment that is itself multi-segment: not in WFDB
            problems.append(f"{unit.path}: a segment that is multi-segment itself; left out")
            del kept[unit.id]
            continue
        files[unit.path] = named
    for record in records:
        del files[record.path]
    return dict(sorted(kept.items())), tuple(sorted(segmented))


def _segments(header: str, record: str) -> tuple[list[tuple[str, datetime | None]], list[str]]:
    """A multi-segment header's segments with samples (path, start) and its layout headers.
    The start of a segment is the record's start plus the frames before it at the frame
    rate; None when the header gives no date and time."""
    lines = _lines(header)
    fields = lines[0].split()
    rate = float(fields[2].split("/")[0]) if len(fields) > 2 else 250.0  # WFDB's default
    start = _wfdb_start(
        fields[4] if len(fields) > 4 else None, fields[5] if len(fields) > 5 else None
    )
    folder = posixpath.dirname(record)
    found: list[tuple[str, datetime | None]] = []
    layouts: list[str] = []
    frames = 0
    for line in lines[1:]:
        name, length = line.split()[0], int(line.split()[1])
        if name != "~":
            path = posixpath.join(folder, name)
            if length == 0:
                layouts.append(path + ".hea")
            else:
                offset = timedelta(seconds=frames / rate)
                found.append((path, None if start is None else start + offset))
        frames += length
    return found, layouts


def _lines(header: str) -> list[str]:
    """A WFDB header's lines without comments and blank lines."""
    return [s.strip() for s in header.splitlines() if s.strip() and not s.lstrip().startswith("#")]


def _record_line(header: str) -> list[str] | None:
    """The fields of a WFDB header's record line (the first line), or None if it is empty."""
    lines = _lines(header)
    return lines[0].split() if lines else None


def _wfdb_start(time: str | None, date: str | None) -> datetime | None:
    """A WFDB header's start (`HH:MM:SS[.sss]` and `DD/MM/YYYY`), or None."""
    if time is None or date is None:
        return None
    try:
        parts = [float(p) for p in time.split(":")]
        hours, minutes, seconds = [0.0] * (3 - len(parts)) + parts
        day, month, year = (int(p) for p in date.split("/"))
        return datetime(year, month, day) + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except ValueError:
        return None


def _with_edf_annotations(
    units: dict[str, Unit], paths: set[str], rule: EdfAnnotationFiles, problems: list[str]
) -> dict[str, Unit]:
    """Each EDF+ annotation file to the one recording the dataset's rule pairs it with; a
    file the rule pairs with no or several recordings is reported and left out."""
    n = rule.same_first_characters

    def key(path: str) -> tuple[str, str]:
        return posixpath.dirname(path), posixpath.basename(path)[:n]

    recordings: dict[tuple[str, str], list[str]] = {}
    for unit_id, unit in units.items():
        path = unit.path
        if (
            unit.format == "edf"
            and path.endswith(rule.recording_suffix)
            and not path.endswith(rule.annotation_suffix)  # hmc: .edf and _sleepscoring.edf
        ):
            recordings.setdefault(key(unit.path), []).append(unit_id)
    found: dict[str, list[str]] = {}
    for path in sorted(p for p in paths if p.endswith(rule.annotation_suffix)):
        owners = recordings.get(key(path), [])
        if len(owners) == 1:
            found.setdefault(owners[0], []).append(path)
        else:
            problems.append(
                f"{path}: the standard.yaml rule pairs it with {len(owners)} recordings "
                f"({', '.join(owners) or 'none'}); left out"
            )
    annotation_files = {p for ps in found.values() for p in ps}
    kept = {}
    for unit_id, unit in units.items():
        if unit.path in annotation_files:  # without RECORDS, such a file is no recording
            continue
        kept[unit_id] = replace(unit, edf_annotations=tuple(found.get(unit_id, ())))
    return kept


def _without_signal_files(
    units: dict[str, Unit], files: dict[str, list[str] | None]
) -> dict[str, Unit]:
    """A file a header names as a signal file is never an annotation file (ptbdb 1.0.0:
    `s0010_re.xyz` holds the Frank leads vx, vy, vz; read as annotations, its samples gave
    labels)."""
    signals = {f for fs in files.values() if fs for f in fs}
    kept = {}
    for unit_id, unit in units.items():
        annotations = tuple(a for a in unit.annotations if a not in signals)
        kept[unit_id] = replace(unit, annotations=annotations)
    return kept


def _views(
    units: dict[str, Unit], files: dict[str, list[str] | None]
) -> dict[str, tuple[Unit, tuple[str, ...]]]:
    wfdb = {u.path: u for u in units.values() if u.format == "wfdb"}
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
