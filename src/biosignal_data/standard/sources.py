"""Readers for the source formats: the exact stored samples, grouped by sampling rate.

A source is opened from local copies of its raw files (`fetch` gives the path of a raw
file, fetched and checked against the raw manifest). It reports its channel groups, its
events and its header, and reads any slice of a group as the integers the file stores.
Nothing is converted to physical units: each channel has `scale` and `offset` instead.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from biosignal_data.standard.discover import header_as_read

Fetch = Callable[[str], Path]
Available = Callable[[str], bool]  # whether a raw file is in the raw version


class UnsupportedError(Exception):
    """A known kind of input that this version does not convert. The raw files are listed
    in the manifest as not converted, with this message as the reason."""


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    scale: float | None  # physical = stored * scale + offset; None: the source gives none
    offset: float | None
    invalid_value: int | None  # the stored value that marks a missing sample, if any
    source: dict[str, Any]  # the format's own fields for this channel, as written


@dataclass(frozen=True)
class Group:
    """Channels with the same sampling rate: one array of shape (channels, length)."""

    rate: float
    dtype: np.dtype[Any]
    length: int
    channels: tuple[Channel, ...]
    step: int = 1  # slices start and end at multiples of this (samples per WFDB frame)

    @property
    def name(self) -> str:
        return f"{self.rate:g}Hz"


@dataclass(frozen=True)
class Event:
    file: str  # the raw file it comes from
    annotator: str  # WFDB annotator (file extension) or "edf+"
    onset_s: float  # from the start of the recording
    label: str  # WFDB symbol, EDF+ annotation text
    sample: int | None = None  # the source's own sample number, where it has one
    duration_s: float | None = None
    text: str | None = None  # WFDB aux_note
    channel: str | None = None
    source: dict[str, Any] = field(default_factory=dict)


class Source(Protocol):
    format: str
    files: list[str]  # the raw files read
    groups: list[Group]
    events: list[Event]  # stored in the signal file itself (EDF+)
    frame_rate: float  # the rate WFDB annotation files of this record count samples at
    problems: list[str]  # e.g. signal files the source does not publish
    start_time: datetime | None
    duration_s: float | None  # None: the header does not say
    header: str  # the source header, as written

    def read(self, group: int, start: int, stop: int) -> np.ndarray: ...
    def close(self) -> None: ...


# --- WFDB ----------------------------------------------------------------------------------

# Stored integer type per WFDB signal format (every value of the format fits).
_WFDB_DTYPES = {
    **dict.fromkeys(("8", "16", "61", "80", "160", "212", "310", "311", "508", "516"), "int16"),
    **dict.fromkeys(("24", "32", "524"), "int32"),
}
_GAIN = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


class WfdbSource:
    """A single-segment WFDB record: `<record>.hea` and the signal files it names."""

    format = "wfdb"

    def __init__(
        self, record: str, fetch: Fetch, available: Available, start_time: datetime | None = None
    ) -> None:
        """`start_time`: the start when the header gives none (a segment of a multi-segment
        record: from its record's header)."""
        import wfdb  # noqa: PLC0415 - readers load only when used
        from wfdb.io._signal import _digi_nan  # noqa: PLC0415

        self._wfdb = wfdb
        header_path = fetch(record + ".hea")
        self.header = header_path.read_text(encoding="latin-1")
        self.problems = []
        as_read, problem = header_as_read(self.header, record)
        if problem is not None:
            # The header as read goes next to the signal files; the source's header is kept
            # as written (source_header) and the change reported.
            header_path = header_path.with_name(header_path.stem + "_as_read.hea")
            header_path.write_text(as_read, encoding="latin-1")
            self.problems.append(problem)
        head = wfdb.rdheader(str(header_path.with_suffix("")))
        if getattr(head, "seg_name", None):
            raise UnsupportedError("multi-segment WFDB record (not converted yet)")
        # A record without signals (only its annotations are published) has no groups; so
        # has one whose signal files the source does not publish (only its header).
        folder = posixpath.dirname(record)
        signal_files = sorted({posixpath.join(folder, f) for f in head.file_name or []})
        unpublished = [p for p in signal_files if not available(p)]
        if unpublished:
            self.problems.append(
                f"{record}: the header names {', '.join(unpublished)}, which the source does "
                "not publish; the recording is kept without signals"
            )
            signal_files = []
        local_files = {path: fetch(path) for path in signal_files}
        self.files = [record + ".hea", *signal_files]
        self._local = str(header_path.with_suffix(""))
        sig_len = head.sig_len
        if signal_files and sig_len is not None:
            held = _frames_held(head, folder, local_files)
            short = {f: n for f, n in held.items() if n < sig_len}
            if short:
                sig_len = min(short.values())
                self.problems.append(
                    f"{record}: the header says {head.sig_len:,} samples per signal, but "
                    + ", ".join(f"{f} holds {n:,}" for f, n in sorted(short.items()))
                    + f"; read as far as the files go ({sig_len:,})"
                )
        self.start_time = (
            datetime.combine(head.base_date, head.base_time)
            if head.base_date is not None and head.base_time is not None
            else start_time
        )
        self.duration_s = sig_len / head.fs if sig_len is not None else None
        lines = _signal_lines(self.header, head.n_sig)
        by_rate: dict[int, list[int]] = {}
        for i, spf in enumerate(head.samps_per_frame if signal_files else []):
            by_rate.setdefault(spf, []).append(i)
        self.groups, self._indices = [], []
        for spf, indices in sorted(by_rate.items(), reverse=True):
            dtypes = {_wfdb_dtype(head.fmt[i]) for i in indices}
            channels = []
            for i in indices:
                calibrated = _gain_written(lines[i])
                gain, baseline = float(head.adc_gain[i]), int(head.baseline[i])
                channels.append(
                    Channel(
                        name=head.sig_name[i],
                        unit=head.units[i],
                        scale=1 / gain if calibrated else None,
                        offset=-baseline / gain if calibrated else None,
                        invalid_value=int(_digi_nan(head.fmt[i])),
                        source={"signal_line": lines[i], "fmt": head.fmt[i]},
                    )
                )
            self.groups.append(
                Group(
                    rate=head.fs * spf,
                    dtype=np.dtype("int32" if "int32" in dtypes else "int16"),
                    length=sig_len * spf,
                    channels=tuple(channels),
                    step=spf,
                )
            )
            self._indices.append(indices)
        self.frame_rate = float(head.fs)
        self.events: list[Event] = []

    def read(self, group: int, start: int, stop: int) -> np.ndarray:
        g = self.groups[group]
        if start % g.step or (stop % g.step and stop != g.length):
            raise ValueError(f"slice {start}:{stop} is not on whole frames of {g.step}")
        record = self._wfdb.rdrecord(
            self._local,
            sampfrom=start // g.step,
            sampto=stop // g.step,
            channels=self._indices[group],
            physical=False,
            smooth_frames=False,
            return_res=g.dtype.itemsize * 8,
        )
        return np.stack([np.asarray(s, dtype=g.dtype) for s in record.e_d_signal])

    def close(self) -> None:
        pass


def wfdb_annotations(
    path: str, fetch: Fetch, record_fs: float, *, listed: bool
) -> tuple[list[Event], list[str]]:
    """A WFDB annotation file `<record>.<annotator>`, and its problems. Sample numbers are
    counted at the file's own rate where it stores one, else at the record's frame rate.

    `listed`: the annotator is in the dataset's ANNOTATORS, so the file is an annotation
    file: its events are kept as stored, and negative or out-of-order sample numbers are
    reported as problems (capslpdb nfle10.edf.st has only negative ones). A file only
    guessed to be one (`listed=False`) with such numbers is rejected instead."""
    import wfdb  # noqa: PLC0415

    local = fetch(path)
    annotator = path.rsplit(".", 1)[1]
    ann = wfdb.rdann(str(local.with_name(local.name[: -len(annotator) - 1])), annotator)
    samples = np.asarray(ann.sample)
    oddities = []
    if (negative := int((samples < 0).sum())) > 0:
        oddities.append(f"{negative} of {len(samples)} sample numbers are negative")
    if (back := int((np.diff(samples) < 0).sum())) > 0:
        oddities.append(f"{back} sample numbers are lower than the one before")
    if oddities and not listed:
        raise UnsupportedError(f"does not read as WFDB annotations ({'; '.join(oddities)})")
    problems = [f"{path}: {o} (kept as stored)" for o in oddities]
    fs = float(ann.fs or record_fs)
    events = [
        Event(
            file=path,
            annotator=annotator,
            sample=int(sample),
            onset_s=int(sample) / fs,
            label=symbol,
            text=aux or None,
            source={"subtype": int(sub), "chan": int(chan), "num": int(num), "fs": fs},
        )
        for sample, symbol, aux, sub, chan, num in zip(
            samples, ann.symbol, ann.aux_note, ann.subtype, ann.chan, ann.num, strict=True
        )
    ]
    return events, problems


# Bits per sample of the WFDB formats with a fixed size (310 and 311 pack 3 samples in 32
# bits). The FLAC formats (508, 516, 524) compress, so their length cannot be computed.
_WFDB_BITS = {
    "8": 8, "80": 8, "16": 16, "61": 16, "160": 16, "212": 12,
    "310": 32 / 3, "311": 32 / 3, "24": 24, "32": 32,
}  # fmt: skip


def _frames_held(head: Any, folder: str, local: dict[str, Path]) -> dict[str, int]:
    """Whole frames each signal file holds, from its size; files in a FLAC format are left
    out (their size says nothing about their length)."""
    held = {}
    for path, file in local.items():
        signals = [
            i for i, name in enumerate(head.file_name) if posixpath.join(folder, name) == path
        ]
        if any(head.fmt[i] not in _WFDB_BITS for i in signals):
            continue
        bits = sum(head.samps_per_frame[i] * _WFDB_BITS[head.fmt[i]] for i in signals)
        offset = (head.byte_offset or [None] * head.n_sig)[signals[0]] or 0
        held[path] = math.floor((file.stat().st_size - offset) * 8 / bits + 1e-9)
    return held


def _wfdb_dtype(fmt: str) -> str:
    if fmt not in _WFDB_DTYPES:
        raise UnsupportedError(f"WFDB signal format {fmt} (not converted yet)")
    return _WFDB_DTYPES[fmt]


def _signal_lines(header: str, n_sig: int) -> list[str]:
    """The signal specification lines of a header, as written (comments left out)."""
    lines = [ln.strip() for ln in header.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return lines[1 : 1 + n_sig]


def _gain_written(line: str) -> bool:
    """WFDB: a gain that is missing or 0 means the signal is not calibrated (the library
    then uses 200 in its place, which is not a measured value)."""
    fields = line.split()
    if len(fields) < 3:
        return False
    match = _GAIN.match(fields[2])
    return match is not None and float(match.group(0)) != 0


# --- EDF, EDF+, BDF ------------------------------------------------------------------------


class EdfSource:
    """An EDF, EDF+ or BDF file, read with edfio. Discontinuous EDF+ (EDF+D) is not
    converted yet.

    edfio reads the data records through a memory map: a slice reads only the records it
    needs, and no state is shared between files. A file shorter than its header says is
    read as far as its whole records go; edfio's warning about it is kept as a problem.
    (pyedflib was used before. A file it refused to open closed the handle of another open
    file, which then read a third file's samples: capslpdb, 2026-10-19.)
    """

    format = "edf"

    def __init__(self, path: str, fetch: Fetch) -> None:
        import edfio  # noqa: PLC0415 - readers load only when used

        local = fetch(path)
        with local.open("rb") as f:
            fixed = f.read(256)
            n_signals = int(fixed[252:256].decode("ascii").strip() or 0)
            self.header = (fixed + f.read(256 * n_signals)).decode("latin-1")
        if fixed[192:197] in (b"EDF+D", b"BDF+D"):
            raise UnsupportedError("discontinuous EDF+ (EDF+D) (not converted yet)")
        self.format = "bdf" if fixed[:1] == b"\xff" else "edf"
        self.files = [path]
        # Headers are read as latin-1: every byte is one character, so nothing is lost; EDF
        # asks for ASCII, but real files have e.g. "µV".
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self.format == "bdf":  # edfio reads BDF only whole, into memory
                # Edf or Bdf: edfio types their signals apart, with no public common base.
                edf: Any = edfio.read_bdf(local, header_encoding="latin-1")
            else:
                edf = edfio.read_edf(local, lazy_load_data=True, header_encoding="latin-1")
        self._edf = edf
        self.problems = [f"{path}: {w.message}" for w in caught]
        self.start_time = _edf_start(edf)
        self.duration_s: float | None = edf.num_data_records * edf.data_record_duration
        dtype = np.dtype("int32" if self.format == "bdf" else "int16")
        by_rate: dict[float, list[int]] = {}
        for i, signal in enumerate(edf.signals):
            by_rate.setdefault(signal.sampling_frequency, []).append(i)
        self.groups, self._indices = [], []
        for rate, indices in sorted(by_rate.items(), reverse=True):
            lengths = {
                edf.signals[i].samples_per_data_record * edf.num_data_records for i in indices
            }
            if len(lengths) != 1:
                raise ValueError(f"channels at {rate} Hz have different lengths {lengths}")
            self.groups.append(
                Group(
                    rate=rate,
                    dtype=dtype,
                    length=int(lengths.pop()),
                    channels=tuple(_edf_channel(edf.signals[i]) for i in indices),
                )
            )
            self._indices.append(indices)
        # WFDB annotation files of an EDF record count samples at its highest rate.
        self.frame_rate = max((g.rate for g in self.groups), default=0.0)
        self.events = [
            Event(
                file=path,
                annotator="edf+",
                onset_s=float(a.onset),
                duration_s=float(a.duration) if a.duration is not None else None,
                label=str(a.text),
            )
            for a in edf.annotations
        ]

    def read(self, group: int, start: int, stop: int) -> np.ndarray:
        g = self.groups[group]
        rows = []
        for i in self._indices[group]:
            # edfio slices by seconds and rounds back to samples, so start / rate is start.
            row = self._edf.signals[i].get_digital_slice(start / g.rate, stop / g.rate)
            if len(row) != stop - start:
                raise ValueError(f"signal {i}: {len(row)} samples read, {stop - start} asked")
            rows.append(np.asarray(row, dtype=g.dtype))
        return np.stack(rows)

    def close(self) -> None:
        pass


# --- VitalDB (.vital) ---------------------------------------------------------------------

# Stored type of each .vital sample format (vitaldb.utils.FMT_TYPE_LEN).
_VITAL_DTYPES = {1: "<f4", 2: "<f8", 3: "i1", 4: "u1", 5: "<i2", 6: "<u2", 7: "<i4", 8: "<u4"}
_VITAL_FLOAT = (1, 2)  # float32, float64: physical values; gain and offset are not applied
_WIDER = {1: np.int16, 2: np.int32, 4: np.int64}  # an integer type one size wider
VITAL_WAVE, VITAL_NUMBER, VITAL_TEXT = 1, 2, 5  # vitaldb.utils.TYPE_WAV, TYPE_NUM, TYPE_STR


class VitalSource:
    """A VitalDB file (VitalRecorder, .vital), read with the vitaldb package (the VitalDB
    team's reader). Each wave track is a channel at its own rate; tracks at one rate share an
    array of ceil((end - start) * rate) samples from the file's start, and each record is
    placed where vitaldb's get_track_samples places it: at ceil((dt - start) * rate), a later
    record over an earlier one where their times overlap (the overwritten samples are counted
    in `problems`). Samples are stored as the file stores them; for the integer formats the
    track's gain and offset are the scale (physical = stored * gain + offset, as vitaldb
    computes it); the float formats are physical already (vitaldb ignores their gain). Tracks
    of one rate with different formats share the smallest type that holds every stored value
    exactly (numpy's promotion: uint16 and float32 give float32). Times no record covers are
    missing (see _vital_group). vitaldb also reads physical values above 4e9 as missing;
    they are stored as written here.

    Text tracks (event markers) are events. Numeric tracks (heart rate, SpO2, drug rates: one
    value every one or two seconds) are not converted yet: about 4 billion values in vitaldb
    1.0.0, too many for the events table; `problems` counts them.
    """

    format = "vital"

    def __init__(self, path: str, fetch: Fetch) -> None:
        import vitaldb  # noqa: PLC0415 - readers load only when used

        vf = vitaldb.VitalFile(str(fetch(path)))
        self.files = [path]
        self.problems: list[str] = []
        start, end = float(vf.dtstart), float(vf.dtend)
        self.start_time = datetime.fromtimestamp(start, UTC).replace(tzinfo=None) if start else None
        self.duration_s = end - start if end > start else None
        tracks = list(vf.trks.items())
        self.header = json.dumps(
            [{"track": name, "type": t.type, "fmt": t.fmt, "srate": t.srate, "unit": t.unit,
              "gain": t.gain, "offset": t.offset, "mindisp": t.mindisp, "maxdisp": t.maxdisp,
              "device": t.dname, "records": len(t.recs)} for name, t in tracks]
        )  # fmt: skip
        waves = [(n, t) for n, t in tracks if t.type == VITAL_WAVE and t.recs and t.srate > 0]
        by_rate: dict[float, list[tuple[str, Any]]] = {}
        for name, track in waves:
            by_rate.setdefault(float(track.srate), []).append((name, track))
        self.groups, self._arrays = [], []
        for rate, members in sorted(by_rate.items(), reverse=True):
            group, array = _vital_group(path, rate, members, start, end, self.problems)
            self.groups.append(group)
            self._arrays.append(array)
        self.frame_rate = max((g.rate for g in self.groups), default=0.0)
        self.events = [
            Event(file=path, annotator="vital", onset_s=float(r["dt"]) - start,
                  label=str(r["val"]), text=name)
            for name, t in tracks if t.type == VITAL_TEXT for r in t.recs
        ]  # fmt: skip
        numbers = [(n, len(t.recs)) for n, t in tracks if t.type == VITAL_NUMBER and t.recs]
        if numbers:
            self.problems.append(
                f"{path}: {len(numbers)} numeric tracks, {sum(n for _, n in numbers):,} values: "
                "not converted yet"
            )

    def read(self, group: int, start: int, stop: int) -> np.ndarray:
        return self._arrays[group][:, start:stop]

    def close(self) -> None:
        self._arrays = []


def _vital_group(
    path: str, rate: float, members: list[tuple[str, Any]], start: float, end: float,
    problems: list[str],
) -> tuple[Group, np.ndarray]:  # fmt: skip
    """The wave tracks of one rate as one array. A sample of a record is data whatever its
    value (vitaldb reads the formats' missing codes inside records as values: 65535 is a
    saturated 16-bit sample); only times no record covers are missing. With such gaps, an
    integer array is one size wider and its gaps hold the type's minimum, a value no track
    can store (uint16 -> int32, gaps -2**31); a float array holds NaN there."""
    length = math.ceil((end - start) * rate)
    rows, covered = [], []
    for name, track in members:
        row = np.zeros(length, dtype=_VITAL_DTYPES[track.fmt])
        written, overwritten = _place(row, track.recs, start, rate)
        if overwritten:
            problems.append(
                f"{path}: {name}: {overwritten:,} samples overwritten by later records at the "
                "same times (as vitaldb places them)"
            )
        rows.append(row)
        covered.append(written)
    dtype = np.result_type(*(row.dtype for row in rows))
    gaps = [not written.all() for written in covered]
    integer = dtype.kind in "iu"
    if any(gaps) and integer and dtype.itemsize < 8:
        dtype = np.result_type(dtype, _WIDER[dtype.itemsize])
    missing = int(np.iinfo(dtype).min) if integer else None
    array = np.empty((len(rows), length), dtype=dtype)
    channels = []
    for i, ((name, track), row, written) in enumerate(zip(members, rows, covered, strict=True)):
        array[i] = row
        array[i][~written] = missing if missing is not None else np.nan
        scaled = track.fmt not in _VITAL_FLOAT  # vitaldb applies gain and offset to these
        channels.append(
            Channel(
                name=name, unit=track.unit or "",
                scale=float(track.gain) if scaled else 1.0,
                offset=float(track.offset) if scaled else 0.0,
                invalid_value=missing if gaps[i] else None,
                source={"fmt": track.fmt, "gain": track.gain, "offset": track.offset,
                        "device": track.dname, "mindisp": track.mindisp,
                        "maxdisp": track.maxdisp},
            )
        )  # fmt: skip
    return Group(rate, dtype, length, tuple(channels)), array


def _place(
    row: np.ndarray, records: list[dict[str, Any]], start: float, rate: float
) -> tuple[np.ndarray, int]:
    """Copy each record into `row` as vitaldb's get_track_samples does; return the samples
    a record covers and how many a later record overwrote."""
    written = np.zeros(len(row), dtype=bool)
    overwritten = 0
    for record in records:
        values = np.asarray(record["val"])
        first = math.ceil((float(record["dt"]) - start) * rate)
        lo, hi = max(first, 0), min(first + len(values), len(row))
        if hi <= lo:
            continue
        overwritten += int(written[lo:hi].sum())
        row[lo:hi] = values[lo - first : hi - first]
        written[lo:hi] = True
    return written, overwritten


# --- PPG-DaLiA (.pkl) ----------------------------------------------------------------------

# What an unpickler may build from a dataset's pickles: NumPy arrays, scalars and dtypes (by
# the names NumPy 1 and 2 pickle them under), and the bytes of Python 2 strings. Nothing
# else: a pickle that names any other callable is refused, so reading one runs no code.
_PICKLE_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
}  # fmt: skip


def load_pickle(path: Path) -> Any:
    """A pickle of NumPy arrays and Python containers (also from Python 2: its strings as
    latin-1), built without running any other code (_PICKLE_ALLOWED)."""
    import pickle  # noqa: PLC0415

    class _Restricted(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) not in _PICKLE_ALLOWED:
                raise pickle.UnpicklingError(f"refused: {module}.{name}")
            return super().find_class(module, name)

    with path.open("rb") as f:
        return _Restricted(f, encoding="latin1").load()  # noqa: S301 - restricted above


# PPG-DaLiA's signals in S<n>.pkl (its readme, section III): name -> (rate, channel names, unit).
# Wrist: Empatica E4; chest: RespiBAN. ACC of the E4 is in 1/64 g; the readme gives no unit
# for BVP, chest ACC, ECG and RESP, so those are left empty. The chest's EDA, EMG and Temp
# "only include dummy data and should thus be ignored" (readme); they are kept as published.
_DALIA_SIGNALS = {
    ("wrist", "BVP"): (64.0, ("BVP",), ""),
    ("wrist", "ACC"): (32.0, ("ACC_x", "ACC_y", "ACC_z"), "1/64 g"),
    ("wrist", "EDA"): (4.0, ("EDA",), "uS"),
    ("wrist", "TEMP"): (4.0, ("TEMP",), "degC"),
    ("chest", "ACC"): (700.0, ("chest_ACC_x", "chest_ACC_y", "chest_ACC_z"), ""),
    ("chest", "ECG"): (700.0, ("chest_ECG",), ""),
    ("chest", "Resp"): (700.0, ("chest_Resp",), ""),
    ("chest", "EDA"): (700.0, ("chest_EDA",), ""),
    ("chest", "EMG"): (700.0, ("chest_EMG",), ""),
    ("chest", "Temp"): (700.0, ("chest_Temp",), ""),
}
DALIA_WINDOW_S, DALIA_SHIFT_S = 8.0, 2.0  # of 'label' (readme, section III.3)


class DaliaSource:
    """A subject's S<n>.pkl of PPG-DaLiA: "synchronised raw sensor data and labels" (readme),
    read with load_pickle. The signals are stored as the pickle holds them (float64); scale
    1, offset 0. Signals at one rate share an array; a signal of another length than the
    others at its rate is refused. Events: 'label' (the ECG-based heart rate of each 8 s
    window, shifted by 2 s: annotator `label`, label `heart_rate`, the value as `text`),
    'rpeaks' (annotator `rpeaks`, the sample of the chest ECG), and 'activity' (4 Hz, one
    event per run of the same ID: annotator `activity`, label the ID). 'questionnaire' and
    'subject' go into the header."""

    format = "ppg-dalia"

    def __init__(self, path: str, fetch: Fetch) -> None:
        data = load_pickle(fetch(path))
        self.files = [path]
        self.problems: list[str] = []
        self.start_time = None
        signals = data["signal"]
        by_rate: dict[float, list[tuple[str, np.ndarray, str]]] = {}
        for (device, key), (rate, names, unit) in _DALIA_SIGNALS.items():
            values = np.asarray(signals[device][key])
            values = values.reshape(len(values), -1).T  # (samples, n) -> (n, samples)
            if values.shape[0] != len(names):
                raise ValueError(f"{path}: {device}/{key} has {values.shape[0]} channels")
            for name, row in zip(names, values, strict=True):
                by_rate.setdefault(rate, []).append((name, row, unit))
        unknown = {(d, k) for d in signals for k in signals[d]} - set(_DALIA_SIGNALS)
        if unknown:
            raise ValueError(f"{path}: signals not in the readme: {sorted(unknown)}")
        self.groups, self._arrays = [], []
        for rate, members in sorted(by_rate.items(), reverse=True):
            lengths = {len(row) for _, row, _ in members}
            if len(lengths) != 1:
                raise ValueError(f"{path}: signals at {rate:g} Hz have lengths {sorted(lengths)}")
            array = np.stack([row for _, row, _ in members])
            channels = tuple(
                Channel(name=name, unit=unit, scale=1.0, offset=0.0, invalid_value=None,
                        source={"dtype": str(row.dtype)})
                for name, row, unit in members
            )  # fmt: skip
            self.groups.append(Group(rate, array.dtype, array.shape[1], channels))
            self._arrays.append(array)
        self.duration_s = max(g.length / g.rate for g in self.groups)
        self.frame_rate = max(g.rate for g in self.groups)
        hr = np.asarray(data["label"], dtype=np.float64).ravel()
        self.events = [
            Event(file=path, annotator="label", onset_s=i * DALIA_SHIFT_S, label="heart_rate",
                  duration_s=DALIA_WINDOW_S, text=repr(float(v)))
            for i, v in enumerate(hr)
        ]  # fmt: skip
        self.events += [
            Event(file=path, annotator="rpeaks", onset_s=int(k) / 700.0, label="R", sample=int(k))
            for k in np.asarray(data["rpeaks"]).ravel()
        ]
        activity = np.asarray(data["activity"]).ravel()
        changes = np.flatnonzero(np.diff(activity)) + 1
        starts = np.concatenate([[0], changes])
        stops = np.concatenate([changes, [len(activity)]])
        self.events += [
            Event(file=path, annotator="activity", onset_s=a / 4.0, label=str(int(activity[a])),
                  duration_s=(b - a) / 4.0)
            for a, b in zip(starts, stops, strict=True)
        ]  # fmt: skip
        quest = dict(data.get("questionnaire", {}))
        self.header = json.dumps(
            {"subject": str(data.get("subject")), "keys": sorted(map(str, data)),
             "questionnaire": {str(k): str(v) for k, v in quest.items()}}
        )  # fmt: skip

    def read(self, group: int, start: int, stop: int) -> np.ndarray:
        return self._arrays[group][:, start:stop]

    def close(self) -> None:
        self._arrays = []


def _edf_start(edf: Any) -> datetime | None:
    """The start as written; None where the file anonymizes the date (EDF+ "X")."""
    import edfio  # noqa: PLC0415

    try:
        return datetime.combine(edf.startdate, edf.starttime)
    except edfio.AnonymizedDateError:
        return None


def _edf_channel(signal: Any) -> Channel:
    pmin, pmax = signal.physical_min, signal.physical_max
    dmin, dmax = signal.digital_min, signal.digital_max
    scale = (pmax - pmin) / (dmax - dmin) if dmax != dmin else None
    return Channel(
        name=signal.label,
        unit=signal.physical_dimension,
        scale=scale,
        offset=pmin - dmin * scale if scale is not None and math.isfinite(scale) else None,
        invalid_value=None,
        source={
            "physical_min": pmin,
            "physical_max": pmax,
            "digital_min": dmin,
            "digital_max": dmax,
            "transducer": signal.transducer_type,
            "prefilter": signal.prefiltering,
        },
    )
