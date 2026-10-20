from __future__ import annotations

import io
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
from biosignal_data.training import ReadError, Selection, WindowSource

RATE = 100.0
INVALID = -32768


def _parquet(rows: list[dict[str, object]], schema: pa.Schema) -> bytes:
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)
    return out.getvalue()


def _index(store: LocalStore, lengths: dict[str, int]) -> dict[str, np.ndarray]:
    """Recordings of 3 channels (eeg, eeg, ecg) at 100 Hz, shards of 1,000 samples."""
    rng = np.random.default_rng(1)
    data, recordings, channels = {}, [], []
    for i, (recording, length) in enumerate(lengths.items()):
        stored = rng.integers(-1000, 1000, (3, length)).astype(np.int16)
        stored[0, 5] = INVALID
        key = f"standard/d/x/1.0/signals/{recording}.zarr"
        group = zarr.open_group(store=store.zarr_store(key, read_only=False), mode="w")
        array = group.create_array(
            "100Hz", shape=stored.shape, dtype=stored.dtype, chunks=(3, 250),
            shards=(3, 1000), compressors=[BloscCodec(cname="zstd", clevel=1)],
        )  # fmt: skip
        array[:] = stored
        data[recording] = stored
        recordings.append(
            {"dataset_id": "d/x", "version": "1.0", "recording_id": recording,
             "subject_id": f"d/x:s{i}", "subject_certain": True,
             "split": "train" if recording != "held-out" else "test",
             "duration_s": length / RATE, "start_time": None, "arrays": ["100Hz"],
             "signals": key}
        )  # fmt: skip
        for index, (name, kind) in enumerate([("C3", "eeg"), ("C4", "eeg"), ("ECG", "ecg")]):
            channels.append(
                {"dataset_id": "d/x", "recording_id": recording, "array": "100Hz",
                 "index": index, "name": name, "signal_type": kind, "signal_type_from": "name",
                 "sampling_rate": RATE, "unit": "uV", "scale": 0.5, "offset": 1.0,
                 "invalid_value": INVALID}
            )  # fmt: skip
    store.put_bytes("index/i1/recordings.parquet", _parquet(recordings, RECORDINGS))
    store.put_bytes("index/i1/channels.parquet", _parquet(channels, CHANNELS))
    store.put_bytes("index/i1/_SUCCESS", b"")
    return data


def _source(store: LocalStore, tmp_path: Path, selection: Selection, **kw: object) -> WindowSource:
    fields: dict[str, object] = {
        "cache_directory": tmp_path / "cache", "cache_bytes": 10**9, "window_s": 3.0,
        "min_window_s": 1.0, "shuffle_buffer": 8, "reads_in_flight": 3, "seed": 7,
    }  # fmt: skip
    return WindowSource(selection, lambda: store, **(fields | kw))  # type: ignore[arg-type]


def test_a_selection_keeps_one_split_and_the_chosen_signal_types(store: LocalStore) -> None:
    _index(store, {"r1": 2500, "held-out": 2500})
    selection = Selection.load(store, "i1", split="train", signal_types={"ecg"})
    assert [(u.recording_id, [c.name for c in u.channels]) for u in selection.units] == [
        ("r1", ["ECG"])
    ]
    with pytest.raises(FileNotFoundError, match="not a complete index"):
        Selection.load(store, "i2", split="train", signal_types={"ecg"})


def test_windows_are_the_stored_samples_in_physical_units(
    store: LocalStore, tmp_path: Path
) -> None:
    # 2,950 samples cross two shard borders; windows of 300 samples from a random cut.
    data = _index(store, {"r1": 2950, "r2": 120, "r3": 450})
    selection = Selection.load(store, "i1", split="train", signal_types={"eeg", "ecg"})
    windows = list(_source(store, tmp_path, selection).windows(epoch=0, part=0, parts=1))
    by_recording: dict[str, list[tuple[int, np.ndarray]]] = {}
    for w in windows:
        assert w.signals.dtype == np.float32
        assert w.channel_names == ("C3", "C4", "ECG")
        assert w.signal_types == ("eeg", "eeg", "ecg")
        assert 100 <= w.signals.shape[1] <= 300  # min_window_s .. window_s
        by_recording.setdefault(w.recording_id, []).append((round(w.start_s * RATE), w.signals))
    for recording, pieces in by_recording.items():
        stored = data[recording]
        expected = stored * 0.5 + 1.0
        expected[stored == INVALID] = np.nan
        pieces.sort(key=lambda p: p[0])
        for (start, signals), after in zip(pieces, [*pieces[1:], None], strict=True):
            np.testing.assert_array_equal(signals, expected[:, start : start + signals.shape[1]])
            if after is not None:  # windows do not overlap
                assert start + signals.shape[1] <= after[0]
    assert "r2" in by_recording  # shorter than a window: one window of all of it
    assert sum(p[1].shape[1] for p in by_recording["r1"]) >= 2950 - 2 * 100


def test_parts_split_the_units_and_an_epoch_fixes_which_windows_come(
    store: LocalStore, tmp_path: Path
) -> None:
    _index(store, {f"r{i}": 700 for i in range(7)})
    selection = Selection.load(store, "i1", split="train", signal_types={"eeg"})
    source = _source(store, tmp_path, selection)
    parts = [source.units_of(epoch=0, part=p, parts=3) for p in range(3)]
    assert sorted(u.recording_id for part in parts for u in part) == [f"r{i}" for i in range(7)]

    def starts(epoch: int) -> Counter[tuple[str, float]]:
        return Counter((w.recording_id, w.start_s) for w in source.windows(epoch, 0, 1))

    assert starts(0) == starts(0)
    assert starts(0) != starts(1)  # another epoch cuts at other places


def test_stopping_early_and_a_failed_read(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {f"r{i}": 3000 for i in range(4)})
    selection = Selection.load(store, "i1", split="train", signal_types={"eeg"})
    windows = _source(store, tmp_path, selection, shuffle_buffer=2).windows(0, 0, 1)
    next(windows)
    windows.close()  # the reading threads stop; this does not hang
    store.put_bytes("standard/d/x/1.0/signals/r2.zarr/100Hz/c/0/1", b"damaged")
    with pytest.raises(ReadError, match=re.escape("r2.zarr/100Hz")):
        list(_source(store, tmp_path, selection, cache_directory=tmp_path / "c2").windows(0, 0, 1))
