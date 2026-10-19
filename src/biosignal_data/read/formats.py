"""Thin openers for the native file formats found in the raw layer.

Each function returns the object of the established library for that format (MNE `Raw`,
`wfdb.Record`, `h5py.File`, ...). Nothing is resampled, renamed or converted: these are
conveniences for looking at raw files, not a standard representation.
"""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

MNE_SUFFIXES = {
    ".edf",
    ".bdf",
    ".gdf",
    ".vhdr",
    ".set",
    ".fif",
    ".cnt",
    ".mff",
    ".nxe",
    ".con",
    ".sqd",
    ".ds",
    ".snirf",
}


def _require(module: str) -> Any:
    """Import a reader library on first use, so `import biosignal_data` stays fast."""
    return importlib.import_module(module)


def read_mne(path: str | Path, **kwargs: Any) -> Any:
    """EDF/EDF+, BDF, GDF, BrainVision, EEGLAB, FIF, Curry, ... -> `mne.io.Raw`.
    `preload=False` by default, so only the header is read until data is requested."""
    mne = _require("mne")
    kwargs.setdefault("preload", False)
    return mne.io.read_raw(str(path), **kwargs)


def read_bids(root: str | Path, **entities: Any) -> Any:
    """One recording of a BIDS dataset (OpenNeuro, NEMAR) -> `mne.io.Raw`, with BIDS sidecars
    applied by mne-bids. `entities` are BIDSPath fields: subject, session, task, run, datatype."""
    mne_bids = _require("mne_bids")
    return mne_bids.read_raw_bids(mne_bids.BIDSPath(root=str(root), **entities), verbose=False)


def read_wfdb(record: str | Path, **kwargs: Any) -> Any:
    """PhysioNet WFDB record (`100` for `100.hea` + `100.dat`) -> `wfdb.Record`."""
    wfdb = _require("wfdb")
    return wfdb.rdrecord(str(Path(record).with_suffix("")), **kwargs)


def read_wfdb_annotation(record: str | Path, extension: str = "atr", **kwargs: Any) -> Any:
    wfdb = _require("wfdb")
    return wfdb.rdann(str(Path(record).with_suffix("")), extension, **kwargs)


def read_nwb(path: str | Path) -> Any:
    """NWB (DANDI) -> `pynwb.NWBHDF5IO`; use as a context manager, then `.read()`."""
    pynwb = _require("pynwb")
    return pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True)


def read_hdf5(path: str | Path) -> Any:
    """HDF5 -> `h5py.File` (read-only)."""
    h5py = _require("h5py")
    return h5py.File(str(path), "r")


def read_mat(path: str | Path, **kwargs: Any) -> Any:
    """MATLAB .mat: v4-v7 via scipy (dict), v7.3 (HDF5-based) via h5py (`h5py.File`)."""
    if _is_hdf5(Path(path)):
        return read_hdf5(path)
    scipy_io = _require("scipy.io")
    return scipy_io.loadmat(str(path), **kwargs)


def _is_hdf5(path: Path) -> bool:
    """MAT v7.3 files are HDF5 with the signature after a 512-byte user block."""
    with path.open("rb") as f:
        f.seek(512)
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def read_table(path: str | Path, **kwargs: Any) -> Any:
    """CSV, TSV, Parquet, Excel -> `pandas.DataFrame`."""
    pd = _require("pandas")
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path, **kwargs)
    kwargs.setdefault("sep", "\t" if suffix == ".tsv" else ",")
    return pd.read_csv(path, **kwargs)


def read_xml(path: str | Path) -> ET.Element:
    """XML (e.g. NSRR or Compumedics annotation exports) -> root element, as-is."""
    return ET.parse(path).getroot()  # noqa: S314 - local files from the raw layer


def read_any(path: str | Path, **kwargs: Any) -> Any:
    """Pick an opener from the file suffix. Raises for formats without an opener."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in MNE_SUFFIXES:
        return read_mne(p, **kwargs)
    if suffix in (".hea", ".dat") and p.with_suffix(".hea").exists():
        return read_wfdb(p, **kwargs)
    if suffix == ".nwb":
        return read_nwb(p)
    if suffix in (".h5", ".hdf5", ".hdf"):
        return read_hdf5(p)
    if suffix == ".mat":
        return read_mat(p, **kwargs)
    if suffix in (".csv", ".tsv", ".parquet", ".xls", ".xlsx"):
        return read_table(p, **kwargs)
    if suffix == ".xml":
        return read_xml(p)
    raise ValueError(f"no opener for {p.name}; open it with the library of your choice")
