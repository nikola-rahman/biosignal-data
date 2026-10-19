"""Path rules shared by the tool: globs, and which files can hold recordings.

`holds_recordings` is used for one check only: a copied version in which no file can hold
a recording (for example a BIDS dataset whose source has only the sidecar files) is
reported instead of looking finished. It never decides what gets copied.
"""

from __future__ import annotations

import posixpath
import re
from functools import lru_cache

# Signal file formats.
SIGNAL_SUFFIXES = frozenset(
    {
        ".bdf", ".cnt", ".edf", ".fif", ".gdf", ".h5", ".hdf5", ".hea", ".mat", ".npy",
        ".npz", ".nwb", ".set", ".snirf", ".vhdr", ".xdf", ".dat", ".eeg", ".rec",
    }
)  # fmt: skip
# Recordings stored as a folder of files: CTF MEG, MEF3, EGI.
FOLDER_SUFFIXES = frozenset({".ds", ".mefd", ".mff"})
# Archives: their contents are not inspected, so they count as possibly holding recordings.
ARCHIVE_SUFFIXES = frozenset({".zip", ".tar", ".tgz", ".7z", ".rar"})
COMPRESSION_SUFFIXES = frozenset({".gz", ".bz2", ".xz", ".zst"})
# Compressed tables hold continuous data in BIDS (x_physio.tsv.gz) and elsewhere.
TABLE_SUFFIXES = frozenset({".tsv", ".csv"})


def holds_recordings(path: str) -> bool:
    """Whether a file can hold recorded signals: a signal format, a file inside a folder
    format, a compressed table, or an archive. Hidden files (.git/, .datalad/) never do."""
    parts = path.split("/")
    if any(part.startswith(".") for part in parts):
        return False
    if any(posixpath.splitext(part)[1].lower() in FOLDER_SUFFIXES for part in parts[:-1]):
        return True
    base, ext = posixpath.splitext(parts[-1].lower())
    if ext in COMPRESSION_SUFFIXES:
        inner = posixpath.splitext(base)[1]
        return inner in TABLE_SUFFIXES or inner in SIGNAL_SUFFIXES or inner == ".tar"
    return ext in SIGNAL_SUFFIXES or ext in ARCHIVE_SUFFIXES


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
