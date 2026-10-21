from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

vitaldb = pytest.importorskip("vitaldb")

from vitaldb.utils import Track  # noqa: E402

from biosignal_data.standard import sources  # noqa: E402
from biosignal_data.standard.discover import discover  # noqa: E402

RNG = np.random.default_rng(7)
T0 = 4102444800.0  # 2100-01-01, as VitalDB's anonymized files start


def _file() -> vitaldb.VitalFile:
    """Integer (uint16, gain and offset) and float tracks at 100 Hz, with a gap and an
    overlap; a 25 Hz track; a numeric and a text track."""
    art = RNG.integers(0, 60000, 300).astype(np.uint16)
    art[5] = 65535  # the format's missing value inside a record
    eeg = RNG.normal(size=250).astype(np.float32)
    co2 = RNG.integers(0, 1000, 50).astype(np.uint16)
    tracks = [
        Track("ART", type=1, fmt=6, srate=100.0, gain=0.01, offset=-50.0, unit="mmHg",
              dname="SNUADC", recs=[{"dt": T0, "val": art[:100]},
                                    {"dt": T0 + 1.5, "val": art[100:200]},  # 0.5 s gap
                                    {"dt": T0 + 2.4, "val": art[200:300]}]),  # overlap: 9 samples
        Track("EEG1_WAV", type=1, fmt=1, srate=100.0, unit="uV", dname="BIS",
              recs=[{"dt": T0 + 0.2, "val": eeg}]),
        Track("CO2", type=1, fmt=6, srate=25.0, gain=0.075, unit="mmHg", dname="Primus",
              recs=[{"dt": T0, "val": co2}]),
        Track("HR", type=2, fmt=1, unit="/min", dname="Solar8000",
              recs=[{"dt": T0 + 1.0, "val": 71.0}, {"dt": T0 + 3.0, "val": 72.0}]),
        Track("EVENT", type=5, recs=[{"dt": T0 + 0.5, "val": "Case started"}]),
    ]  # fmt: skip
    vf = vitaldb.VitalFile()
    vf.trks = {t.dtname: t for t in tracks}
    vf.dtstart, vf.dtend = T0, T0 + 3.4
    return vf


def test_a_vital_file_reads_as_vitaldb_places_its_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vf = _file()
    (tmp_path / "0001.vital").write_bytes(b"")
    monkeypatch.setattr(vitaldb, "VitalFile", lambda path: vf)
    source = sources.VitalSource("vital_files/0001.vital", lambda p: tmp_path / "0001.vital")

    rates = (("SNUADC/ART", 100), ("Primus/CO2", 25))
    lengths = [len(vf.get_track_samples(name, 1 / rate)) for name, rate in rates]
    assert [(g.name, g.dtype.str, g.length) for g in source.groups] == [
        ("100Hz", "<f4", lengths[0]),  # uint16 + float32: float32, every stored value exact
        ("25Hz", "<i4", lengths[1]),  # uint16 with a gap at the end: int32, gaps -2**31
    ]
    for g, group in enumerate(source.groups):
        stored = source.read(g, 0, group.length)
        for row, channel in enumerate(group.channels):
            values = stored[row].astype(np.float64)
            if channel.invalid_value is not None:
                values[values == channel.invalid_value] = np.nan
            physical = values * channel.scale + channel.offset
            expected = vf.get_track_samples(channel.name, 1 / group.rate).astype(np.float64)
            expected[np.isinf(expected) | (expected > 4e9)] = np.nan
            np.testing.assert_allclose(physical, expected, rtol=1e-6, equal_nan=True)
    art = source.groups[0].channels[0]
    assert (art.name, art.scale, art.offset, art.invalid_value) == ("SNUADC/ART", 0.01, -50.0, None)
    assert source.read(0, 0, 10)[0][5] == 65535  # a saturated sample: data, as vitaldb reads it
    assert np.isnan(source.read(0, 100, 150)[0]).all()  # the 0.5 s gap
    co2 = source.groups[1].channels[0]
    assert co2.invalid_value == -(2**31) and source.read(1, 84, 85)[0][0] == -(2**31)
    assert source.duration_s == pytest.approx(3.4)
    assert [(e.label, e.text, e.onset_s) for e in source.events] == [("Case started", "EVENT", 0.5)]
    assert source.problems == [
        "vital_files/0001.vital: SNUADC/ART: 9 samples overwritten by later records at the "
        "same times (as vitaldb places them)",
        "vital_files/0001.vital: 1 numeric tracks, 2 values: not converted yet",
    ]


def test_every_vital_file_is_a_recording() -> None:
    paths = ["vital_files/0001.vital", "vital_files/0002.vital", "clinical_data.csv"]
    found = discover(paths, lambda p: "", lambda ps: {})
    assert [(u.id, u.format, u.annotations) for u in found.units] == [
        ("vital_files/0001", "vital", ()), ("vital_files/0002", "vital", ()),
    ]  # fmt: skip


def test_a_vital_file_with_nan_gaps_converts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Float tracks hold NaN where no record is: the checks (two readers, decoded blocks)
    compare stored bytes, so NaN gaps are not a difference."""
    from biosignal_data.standard import blocks  # noqa: PLC0415

    vf = _file()
    (tmp_path / "0001.vital").write_bytes(b"")
    monkeypatch.setattr(vitaldb, "VitalFile", lambda path: vf)

    def source() -> sources.VitalSource:
        return sources.VitalSource("vital_files/0001.vital", lambda p: tmp_path / "0001.vital")

    found, sha256 = blocks.encode_source("vital_files/0001", source(), source())
    assert len(sha256) == 3 and found
    other = source()
    other._arrays[0][1, 50] = 1.5  # one sample of the float track differs
    with pytest.raises(blocks.MismatchError, match="two readers of the file differ"):
        blocks.encode_source("vital_files/0001", source(), other)
