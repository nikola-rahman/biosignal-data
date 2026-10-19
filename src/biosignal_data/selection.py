"""Choose a subset of a dataset's files for test ingests: `--include` and `--limit`.

`--limit N` counts recordings, not files. A recording is a signal file together with the files
that belong to it, found by name:

- a folder-format recording (CTF `.ds`, MEF3 `.mefd`, EGI `.mff`) with all files in it;
- the same folder and the same name without extension: `100.hea`, `100.dat`, `100.atr` (WFDB);
  `x.vhdr`, `x.vmrk`, `x.eeg` (BrainVision); `x.set`, `x.fdt` (EEGLAB);
- in BIDS, the same entity prefix: `sub-01_task-rest_eeg.edf`, `sub-01_task-rest_channels.tsv`,
  `sub-01_task-rest_events.tsv`, ...

Metadata files (JSON, TSV, text, extensionless names such as README) in the folder of a
chosen recording or in any folder above it are added, so that the subset can be opened like
the full dataset (e.g. BIDS sidecars inherited from the dataset root).

If no file has a known signal format (e.g. a dataset of archives), each non-metadata file
counts as one recording.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

# Files that hold signals and identify a recording. Companion files (.dat, .eeg, .vmrk,
# .fdt, ...) are found through these by name.
SIGNAL_SUFFIXES = frozenset(
    {
        ".bdf", ".cnt", ".edf", ".fif", ".gdf", ".h5", ".hdf5", ".hea", ".mat", ".npy",
        ".npz", ".nwb", ".set", ".snirf", ".vhdr", ".xdf",
    }
)  # fmt: skip
# Recordings stored as a folder of files: CTF MEG, MEF3, EGI.
DIRECTORY_SUFFIXES = frozenset({".ds", ".mefd", ".mff"})
METADATA_SUFFIXES = frozenset({".json", ".tsv", ".txt", ".csv", ".md", ".rst", ".pdf", ".xml"})
COMPRESSION_SUFFIXES = frozenset({".gz", ".bz2", ".xz", ".zst"})


@dataclass(frozen=True)
class Selection:
    paths: list[str]  # every file to copy, sorted
    recordings: list[list[str]]  # the recordings among `paths`, each as its files
    # False: no file has a known signal format, so every data file counts as a recording.
    signal_files_found: bool = True
    metadata_added: int = 0  # metadata files added to the chosen recordings (with a limit)


@lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and (end := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1 : end].replace("\\", "\\\\")
            out.append("[^" + body[1:] + "]" if body.startswith("!") else "[" + body + "]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(path: str, pattern: str) -> bool:
    """Standard path glob: `*` and `?` stay inside one folder, `**` crosses folders
    (`**/x` also matches `x` at the top), `[abc]` and `[!abc]` are character sets."""
    return _glob_regex(pattern).match(path) is not None


def _suffix_and_stem(name: str) -> tuple[str, str]:
    base, ext = posixpath.splitext(name)
    if ext.lower() in COMPRESSION_SUFFIXES:
        base, ext = posixpath.splitext(base)
    return ext.lower(), base


def _unit(path: str) -> str:
    """The folder of a folder-format recording that contains `path`, else `path`."""
    parts = path.split("/")
    for i, part in enumerate(parts[:-1]):
        if posixpath.splitext(part)[1].lower() in DIRECTORY_SUFFIXES:
            return "/".join(parts[: i + 1])
    return path


def _group_key(path: str) -> tuple[str, str]:
    folder, name = posixpath.split(_unit(path))
    _, stem = _suffix_and_stem(name)
    if name.startswith("sub-") and "_" in stem:
        stem = stem.rsplit("_", 1)[0]  # BIDS: drop the suffix (eeg, channels, events, ...)
    return folder, stem


def _is_signal(path: str) -> bool:
    if _unit(path) != path:
        return True
    return _suffix_and_stem(posixpath.basename(path))[0] in SIGNAL_SUFFIXES


def _is_metadata(path: str) -> bool:
    suffix, _ = _suffix_and_stem(posixpath.basename(path))
    return suffix in METADATA_SUFFIXES or suffix == ""


def _is_at_or_above(folder: str, below: str) -> bool:
    return folder in ("", below) or below.startswith(folder + "/")


def recordings_of(paths: Sequence[str]) -> list[list[str]]:
    """Group sorted `paths` into recordings, ordered by their first signal file."""
    groups: dict[tuple[str, str], list[str]] = {}
    for path in paths:
        groups.setdefault(_group_key(path), []).append(path)
    recordings = sorted(
        (members for members in groups.values() if any(_is_signal(p) for p in members)),
        key=lambda members: min(p for p in members if _is_signal(p)),
    )
    return recordings or [[p] for p in paths if not _is_metadata(p)] or [[p] for p in paths]


def select(
    paths: Sequence[str], include: Sequence[str] = (), limit: int | None = None
) -> Selection:
    """Paths to copy: those matching `include` (all if empty), then the first `limit`
    recordings in path order with their metadata (all if `limit` is None)."""
    chosen = sorted(p for p in paths if not include or any(glob_match(p, g) for g in include))
    recordings = recordings_of(chosen)
    found = any(_is_signal(p) for p in chosen)
    if limit is None:
        return Selection(chosen, recordings, found)

    picked = recordings[:limit]
    selected = {p for members in picked for p in members}
    in_recording = {p for members in recordings for p in members}
    folders = {posixpath.dirname(_unit(p)) for p in selected}
    for path in chosen:
        if path in in_recording or not _is_metadata(path):
            continue
        if any(_is_at_or_above(posixpath.dirname(path), f) for f in folders):
            selected.add(path)
    added = len(selected) - sum(len(members) for members in picked)
    return Selection(sorted(selected), picked, found, added)
