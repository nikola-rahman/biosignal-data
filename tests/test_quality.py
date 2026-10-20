from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biosignal_data.quality import AlreadyMeasured, iter_rows, measure
from biosignal_data.quality.compute import _Channel, measure_recording
from biosignal_data.standard import blocks, tables
from biosignal_data.storage.stores import LocalStore, StoreFactory

RATE = 10.0


def _channel(name: str, index: int, scale: float | None, offset: float | None,
             invalid: int | None, source: dict[str, object]) -> dict[str, object]:  # fmt: skip
    return {"recording_id": "r1", "array": "10Hz", "index": index, "name": name,
            "standard_name": None, "unit": "uV", "sampling_rate": RATE, "dtype": "int16",
            "scale": scale, "offset": offset, "invalid_value": invalid,
            "source": json.dumps(source)}  # fmt: skip


def _signals() -> tuple[np.ndarray, list[dict[str, object]]]:
    """Three channels, 100 s at 10 Hz, written in shards of 30 s (runs cross shards)."""
    data = np.zeros((3, 1000), dtype=np.int16)
    data[0, :500] = 7  # flat for 50 s, then varying
    data[0, 500:] = np.arange(500) % 13
    data[0, 600:610] = -32768  # 10 invalid samples
    data[1, :] = -8400  # a flat line with a broken calibration (gain 2.78e-152)
    data[2, :] = np.arange(1000) % 2048 - 1024
    data[2, :5] = 2047  # at the EDF digital maximum
    channels = [
        _channel("C3", 0, 0.5, 1.0, -32768, {}),
        _channel("NIBP", 1, 3.5951824555096175e151, 3.0199532626280783e155, None, {}),
        _channel("F4", 2, 0.03, 0.0, None, {"digital_min": -2048, "digital_max": 2047}),
    ]
    return data, channels


def _blocks(data: np.ndarray) -> list[np.ndarray]:
    """The array in blocks of 30 s (runs cross blocks)."""
    return [data[:, s : s + 300] for s in range(0, data.shape[1], 300)]


def test_the_rows_count_flat_runs_invalid_samples_limits_and_broken_calibrations() -> None:
    data, channels = _signals()
    rows = {r["name"]: r for r in measure_recording(
        lambda _: iter(_blocks(data)), "r1", channels, flat_s=2.0, percentile_sample=100,
        seed=0)}  # fmt: skip
    c3, nibp, f4 = rows["C3"], rows["NIBP"], rows["F4"]
    assert (c3["samples"], c3["invalid"], c3["nonfinite"]) == (1000, 10, 0)
    assert c3["longest_flat_s"] == 50.0 and not c3["constant"]
    assert c3["flat_fraction"] == pytest.approx(0.5)  # the 10 invalid: 1 s, below 2 s
    valid = data[0][data[0] != -32768] * 0.5 + 1.0
    assert c3["minimum"] == valid.min() and c3["maximum"] == valid.max()
    assert c3["mean"] == pytest.approx(valid.mean()) and c3["std"] == pytest.approx(valid.std())
    assert nibp["constant"] and nibp["nonfinite"] == 1000 and nibp["mean"] is None
    assert nibp["flat_fraction"] == 1.0 and nibp["limits_from"] == "dtype"
    assert f4["limits_from"] == "header" and f4["at_high"] == pytest.approx(5 / 1000)


def test_the_percentile_sample_is_uniform_and_reproducible() -> None:
    def counter() -> _Channel:
        return _Channel(None, None, None, -32768, 32767, 2, 1000, np.random.default_rng(3))

    values = np.arange(100_000, dtype=np.int32)
    a, b = counter(), counter()
    for c in (a, b):
        for start in range(0, values.size, 7_000):  # blocks of uneven size
            c.add(values[start : start + 7_000])
    assert np.array_equal(np.sort(a.sample), np.sort(b.sample))
    assert a.sample.size == 1000 and len(set(a.sample.tolist())) == 1000  # no repeats
    assert abs(float(np.median(a.sample)) - 50_000) < 5_000  # not biased to early blocks


def _parquet(rows: list[dict[str, object]], schema: pa.Schema) -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)
    return out.getvalue()


def test_measure_writes_a_complete_version_and_does_not_measure_twice(
    store: LocalStore, tmp_path: Path
) -> None:
    data, channels = _signals()
    prefix = "standard-blocks/d/x/1.0"
    blocks.store_arrays(
        store, prefix, {("r1", "10Hz"): data}, tmp_path, block_samples={"10Hz": 300}
    )
    recordings = [{"recording_id": "r1", "format": "edf", "duration_s": 100.0,
                   "arrays": ["10Hz"], "n_channels": 3, "signals": True,
                   "source_files": [], "source_sha256": []}]  # fmt: skip
    store.put_bytes(f"{prefix}/recordings.parquet", _parquet(recordings, tables.RECORDINGS))
    store.put_bytes(f"{prefix}/channels.parquet", _parquet(channels, tables.CHANNELS))
    store.put_bytes(f"{prefix}/_manifest.json", b'{"raw_version": "1.0"}')
    store.put_bytes(f"{prefix}/_SUCCESS", b"")

    factory = StoreFactory(str(store.root), None)
    m = measure(store, factory, "d/x", flat_s=2.0, percentile_sample=100, seed=0, workers=2,
                replace=False)  # fmt: skip

    assert (m.recordings, m.channels, m.standard_version) == (1, 3, "1.0")
    rows = list(iter_rows(store, "d/x", "1.0"))
    assert {r["name"] for r in rows} == {"C3", "NIBP", "F4"}
    assert not list(store.list("quality/d/x/1.0/_parts/"))
    with pytest.raises(AlreadyMeasured, match="--replace"):
        measure(store, factory, "d/x", flat_s=2.0, percentile_sample=100, seed=0, workers=2,
                replace=False)  # fmt: skip
