"""Readers for the source formats: the exact stored samples, grouped by sampling rate.

A source is opened from local copies of its raw files (`fetch` gives the path of a raw
file, fetched and checked against the raw manifest). It reports its channel groups, its
events and its header, and reads any slice of a group as the integers the file stores.
Nothing is converted to physical units: each channel has `scale` and `offset` instead.
"""

from __future__ import annotations

import math
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

Fetch = Callable[[str], Path]


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

    def __init__(self, record: str, fetch: Fetch) -> None:
        import wfdb  # noqa: PLC0415 - readers load only when used
        from wfdb.io._signal import _digi_nan  # noqa: PLC0415

        self._wfdb = wfdb
        header_path = fetch(record + ".hea")
        self.header = header_path.read_text(encoding="latin-1")
        head = wfdb.rdheader(str(header_path.with_suffix("")))
        if getattr(head, "seg_name", None):
            raise UnsupportedError("multi-segment WFDB record (not converted yet)")
        # A record without signals (only its annotations are published) has no groups.
        folder = posixpath.dirname(record)
        signal_files = sorted({posixpath.join(folder, f) for f in head.file_name or []})
        for path in signal_files:
            fetch(path)
        self.files = [record + ".hea", *signal_files]
        self._local = str(header_path.with_suffix(""))
        self.start_time = (
            datetime.combine(head.base_date, head.base_time)
            if head.base_date is not None and head.base_time is not None
            else None
        )
        self.duration_s = head.sig_len / head.fs if head.sig_len is not None else None
        lines = _signal_lines(self.header, head.n_sig)
        by_rate: dict[int, list[int]] = {}
        for i, spf in enumerate(head.samps_per_frame or []):
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
                    length=head.sig_len * spf,
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


def wfdb_annotations(path: str, fetch: Fetch, record_fs: float) -> list[Event]:
    """A WFDB annotation file `<record>.<annotator>`. Sample numbers are counted at the
    file's own rate where it stores one, else at the record's frame rate."""
    import wfdb  # noqa: PLC0415

    local = fetch(path)
    annotator = path.rsplit(".", 1)[1]
    ann = wfdb.rdann(str(local.with_name(local.name[: -len(annotator) - 1])), annotator)
    samples = np.asarray(ann.sample)
    if samples.size and (samples[0] < 0 or np.any(np.diff(samples) < 0)):
        raise UnsupportedError("does not read as WFDB annotations (sample numbers out of order)")
    fs = float(ann.fs or record_fs)
    return [
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
    """An EDF, EDF+ or BDF file. Discontinuous EDF+ (EDF+D) is not converted yet."""

    format = "edf"

    def __init__(self, path: str, fetch: Fetch) -> None:
        import pyedflib  # noqa: PLC0415

        local = fetch(path)
        with local.open("rb") as f:
            fixed = f.read(256)
            n_signals = int(fixed[252:256].decode("ascii").strip() or 0)
            self.header = (fixed + f.read(256 * n_signals)).decode("latin-1")
        if fixed[192:197] in (b"EDF+D", b"BDF+D"):
            raise UnsupportedError("discontinuous EDF+ (EDF+D) (not converted yet)")
        self.format = "bdf" if fixed[:1] == b"\xff" else "edf"
        self.files = [path]
        self._reader = reader = pyedflib.EdfReader(str(local))
        try:
            self.start_time: datetime | None = reader.getStartdatetime()
            self.duration_s: float | None = reader.datarecords_in_file * reader.datarecord_duration
            dtype = np.dtype("int32" if self.format == "bdf" else "int16")
            by_rate: dict[float, list[int]] = {}
            for i in range(reader.signals_in_file):
                by_rate.setdefault(reader.getSampleFrequency(i), []).append(i)
            self.groups, self._indices = [], []
            for rate, indices in sorted(by_rate.items(), reverse=True):
                lengths = {int(reader.getNSamples()[i]) for i in indices}
                if len(lengths) != 1:
                    raise ValueError(f"channels at {rate} Hz have different lengths {lengths}")
                self.groups.append(
                    Group(
                        rate=rate,
                        dtype=dtype,
                        length=lengths.pop(),
                        channels=tuple(_edf_channel(reader, i) for i in indices),
                    )
                )
                self._indices.append(indices)
            onsets, durations, texts = reader.readAnnotations()
            # WFDB annotation files of an EDF record count samples at its highest rate.
            self.frame_rate = max((g.rate for g in self.groups), default=0.0)
            self.events = [
                Event(
                    file=path,
                    annotator="edf+",
                    onset_s=float(onset),
                    duration_s=float(duration) if duration >= 0 else None,
                    label=str(text),
                )
                for onset, duration, text in zip(onsets, durations, texts, strict=True)
            ]
        except Exception:
            reader.close()
            raise

    def read(self, group: int, start: int, stop: int) -> np.ndarray:
        g = self.groups[group]
        return np.stack(
            [
                np.asarray(
                    self._reader.readSignal(i, start, stop - start, digital=True), dtype=g.dtype
                )
                for i in self._indices[group]
            ]
        )

    def close(self) -> None:
        self._reader.close()


def _edf_channel(reader: Any, i: int) -> Channel:
    pmin, pmax = reader.getPhysicalMinimum(i), reader.getPhysicalMaximum(i)
    dmin, dmax = reader.getDigitalMinimum(i), reader.getDigitalMaximum(i)
    scale = (pmax - pmin) / (dmax - dmin) if dmax != dmin else None
    return Channel(
        name=reader.getLabel(i),
        unit=reader.getPhysicalDimension(i),
        scale=scale,
        offset=pmin - dmin * scale if scale is not None and math.isfinite(scale) else None,
        invalid_value=None,
        source={
            "physical_min": pmin,
            "physical_max": pmax,
            "digital_min": dmin,
            "digital_max": dmax,
            "transducer": reader.getTransducer(i),
            "prefilter": reader.getPrefilter(i),
        },
    )
