from __future__ import annotations

import io
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from zarr.codecs import BloscCodec

from biosignal_data.index.build import CHANNELS, RECORDINGS
from biosignal_data.storage.stores import LocalStore
from biosignal_data.training import Example, ExampleSource, ReadError, Selection

INVALID = -32768
# (array, rate, channels as (name, type, scale)); scale None: no calibration
ARRAYS = [
    ("100Hz", 100.0, [("C3", "eeg", 0.5), ("C4", "eeg", 0.5), ("ECG", "ecg", 0.002)]),
    ("10Hz", 10.0, [("RESP", "resp", 1.0), ("POS", "position", None)]),
]


def _parquet(rows: list[dict[str, object]], schema: pa.Schema) -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)
    return out.getvalue()


def _index(store: LocalStore, seconds: dict[str, float]) -> dict[tuple[str, str], np.ndarray]:
    """Recordings with two arrays (100 Hz and 10 Hz), in shards of 10 s. Returns the
    expected values of each (recording, channel name)."""
    rng = np.random.default_rng(1)
    expected, recordings, channels = {}, [], []
    for i, (recording, length_s) in enumerate(seconds.items()):
        key = f"standard/d/x/1.0/signals/{recording}.zarr"
        group = zarr.open_group(store=store.zarr_store(key, read_only=False), mode="w")
        for array_name, rate, chans in ARRAYS:
            n = round(length_s * rate)
            stored = rng.integers(-1000, 1000, (len(chans), n)).astype(np.int16)
            stored[:, 3] = INVALID
            per_shard = round(10 * rate)
            array = group.create_array(
                array_name, shape=stored.shape, dtype=stored.dtype,
                chunks=(len(chans), per_shard // 4), shards=(len(chans), per_shard),
                compressors=[BloscCodec(cname="zstd", clevel=1)],
            )  # fmt: skip
            array[:] = stored
            for index, (name, kind, scale) in enumerate(chans):
                values = stored[index].astype(np.float32)
                if scale is not None:
                    values = (stored[index] * scale + 1.0).astype(np.float32)
                values[stored[index] == INVALID] = np.nan
                expected[recording, name] = values
                channels.append(
                    {"dataset_id": "d/x", "recording_id": recording, "array": array_name,
                     "index": index, "name": name, "signal_type": kind,
                     "signal_type_from": "name", "sampling_rate": rate, "unit": "uV",
                     "scale": scale, "offset": None if scale is None else 1.0,
                     "invalid_value": INVALID}
                )  # fmt: skip
        recordings.append(
            {"dataset_id": "d/x", "version": "1.0", "recording_id": recording,
             "subject_id": f"d/x:s{i}", "subject_certain": True,
             "split": "test" if recording == "held-out" else "train",
             "duration_s": length_s, "start_time": None, "arrays": ["100Hz", "10Hz"],
             "signals": key}
        )  # fmt: skip
    store.put_bytes("index/i1/recordings.parquet", _parquet(recordings, RECORDINGS))
    store.put_bytes("index/i1/channels.parquet", _parquet(channels, CHANNELS))
    store.put_bytes("index/i1/_SUCCESS", b"")
    return expected


def _source(store: LocalStore, tmp_path: Path, selection: Selection, **kw: object) -> ExampleSource:
    fields: dict[str, object] = {
        "cache_directory": tmp_path / "cache", "cache_bytes": 10**9, "max_tokens": 40,
        "samples_per_token": 16, "max_channels": 4, "block_s": 10.0, "shuffle_buffer": 8,
        "reads_in_flight": 3, "seed": 7,
    }  # fmt: skip
    return ExampleSource(selection, lambda: store, **(fields | kw))  # type: ignore[arg-type]


def test_a_selection_keeps_one_split_and_selects_channels_by_type(store: LocalStore) -> None:
    _index(store, {"r1": 25.0, "held-out": 25.0})
    every = Selection.load(store, "i1", split="train", signal_types=None)
    assert [len(r.channels) for r in every.recordings] == [5]
    ecg = Selection.load(store, "i1", split="train", signal_types={"ecg", "resp"})
    assert [[c.name for c in r.channels] for r in ecg.recordings] == [["ECG", "RESP"]]
    with pytest.raises(FileNotFoundError, match="not a complete index"):
        Selection.load(store, "i2", split="train", signal_types=None)


def _place(examples: list[Example]) -> dict[tuple[str, str], dict[int, float]]:
    """Every sample of every example, by (recording, channel) and sample index."""
    placed: dict[tuple[str, str], dict[int, float]] = {}
    for e in examples:
        for signal, channel in zip(e.signals, e.origin.channels, strict=True):
            first = round((e.origin.start_s + signal.start_s) * signal.sampling_rate)
            samples = placed.setdefault((e.origin.recording_id, channel.name), {})
            for i, value in enumerate(signal.values):
                assert first + i not in samples, "a sample in two examples"
                samples[first + i] = value
    return placed


def test_every_sample_is_in_one_example_with_its_value_and_time(
    store: LocalStore, tmp_path: Path
) -> None:
    # 47.3 s: not a whole number of spans, nor of shards (10 s).
    expected = _index(store, {"r1": 47.3, "r2": 3.0})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    source = _source(store, tmp_path, selection)
    examples = list(source.examples(epoch=0, part=0, parts=1))
    for e in examples:
        assert 1 <= len(e.signals) <= 4
        tokens = sum(math.ceil(s.values.size / 16) for s in e.signals)
        assert tokens <= 40
        for s in e.signals:
            assert s.values.dtype == np.float32
            assert 0 <= s.start_s < 1 / s.sampling_rate
    placed = _place(examples)
    assert set(placed) == set(expected)
    for key, values in expected.items():
        samples = placed[key]
        assert sorted(samples) == list(range(values.size)), key  # all, each once
        np.testing.assert_array_equal(np.array([samples[i] for i in range(values.size)]), values)


def test_examples_mix_rates_over_one_time_span(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {"r1": 60.0})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    examples = list(_source(store, tmp_path, selection).examples(0, 0, 1))
    mixed = [e for e in examples if len({s.sampling_rate for s in e.signals}) == 2]
    assert mixed  # 100 Hz and 10 Hz channels together, aligned in time
    sizes = Counter(len(e.signals) for e in examples)
    assert len(sizes) > 1  # groups of random size


def test_parts_split_the_recordings_and_an_epoch_fixes_the_examples(
    store: LocalStore, tmp_path: Path
) -> None:
    _index(store, {f"r{i}": 12.0 for i in range(7)})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    source = _source(store, tmp_path, selection)
    parts = [source.blocks_of(epoch=0, part=p, parts=3) for p in range(3)]
    blocks = sorted((b.recording.recording_id, b.index) for part in parts for b in part)
    assert blocks == [(f"r{i}", k) for i in range(7) for k in (0, 1)]  # 12 s in blocks of 10

    def cuts(epoch: int) -> Counter[tuple[str, float, tuple[str, ...]]]:
        return Counter(
            (e.origin.recording_id, e.origin.start_s, tuple(c.name for c in e.origin.channels))
            for e in source.examples(epoch, 0, 1)
        )

    assert cuts(0) == cuts(0)
    assert cuts(0) != cuts(1)  # another epoch cuts at other places


def test_stopping_early_and_a_failed_read(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {f"r{i}": 30.0 for i in range(4)})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    examples = _source(store, tmp_path, selection, shuffle_buffer=2).examples(0, 0, 1)
    next(examples)
    examples.close()  # the reading threads stop; this does not hang
    store.put_bytes("standard/d/x/1.0/signals/r2.zarr/100Hz/c/0/1", b"damaged")
    with pytest.raises(ReadError, match=re.escape("r2.zarr")):
        list(_source(store, tmp_path, selection, cache_directory=tmp_path / "c2").examples(0, 0, 1))


def test_settings_that_cannot_work_are_refused(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {"r1": 5.0})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    with pytest.raises(ValueError, match="twice max_channels"):
        _source(store, tmp_path, selection, max_tokens=8)


def test_a_broken_calibration_gives_nan_never_inf() -> None:
    from biosignal_data.training.loader import Channel, _values  # noqa: PLC0415

    # autonomic-aging-cardiovascular 0065 NIBP: gain 2.7815e-152 in the source header
    broken = Channel("1000Hz", 1, 1000.0, 3.5951824555096175e151, 3.0199532626280783e155,
                     None, "NIBP", "bp", "mmHg")  # fmt: skip
    values = _values(np.full(10, -8400, dtype=np.int16), broken)
    assert values.dtype == np.float32 and not np.isinf(values).any()
    assert np.isnan(values).all()


def test_the_blocks_of_long_recordings_are_spread_over_the_epoch(
    store: LocalStore, tmp_path: Path
) -> None:
    _index(store, {"long": 300.0} | {f"s{i}": 10.0 for i in range(30)})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    order = [
        b.recording.recording_id for b in _source(store, tmp_path, selection).blocks_of(0, 0, 1)
    ]
    assert len(order) == 60  # 30 blocks of the long recording, 30 short recordings
    first_half = order[:30].count("long")
    assert 8 <= first_half <= 22  # about half of the long recording comes in each half
