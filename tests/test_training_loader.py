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

from biosignal_data.index.build import CHANNELS, RECORDINGS
from biosignal_data.standard.blocks import store_arrays
from biosignal_data.storage.stores import LocalStore
from biosignal_data.training import (
    Example,
    ExampleSource,
    ReadError,
    SampleAndTokenLimit,
    SampleTokenLimit,
    Selection,
)
from biosignal_data.training.loader import Block

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


FOLDER = "standard-blocks/d/x/1.0"


def _index(store: LocalStore, seconds: dict[str, float]) -> dict[tuple[str, str], np.ndarray]:
    """Recordings with two arrays (100 Hz and 10 Hz), in blocks of 10 s. Returns the
    expected values of each (recording, channel name)."""
    rng = np.random.default_rng(1)
    expected, recordings, channels, arrays = {}, [], [], {}
    for i, (recording, length_s) in enumerate(seconds.items()):
        for array_name, rate, chans in ARRAYS:
            n = round(length_s * rate)
            stored = rng.integers(-1000, 1000, (len(chans), n)).astype(np.int16)
            stored[:, 3] = INVALID
            arrays[recording, array_name] = stored
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
             "standard": FOLDER}
        )  # fmt: skip
    store_arrays(
        store, FOLDER, arrays, store.root.parent, block_samples={"100Hz": 1000, "10Hz": 100}
    )
    store.put_bytes("index/i1/recordings.parquet", _parquet(recordings, RECORDINGS))
    store.put_bytes("index/i1/channels.parquet", _parquet(channels, CHANNELS))
    store.put_bytes("index/i1/_SUCCESS", b"")
    return expected


def _source(store: LocalStore, tmp_path: Path, selection: Selection, **kw: object) -> ExampleSource:
    fields: dict[str, object] = {
        "cache_directory": tmp_path / "cache", "cache_bytes": 10**9,
        "limit": SampleTokenLimit(max_tokens=40, samples_per_token=16), "spans": "shared",
        "subject_hours_per_epoch": None, "max_channels": 4, "block_s": 10.0,
        "shuffle_buffer": 8, "reads_in_flight": 3, "seed": 7, "passes": None,
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
    r2 = selection.recordings[2].array_blocks("100Hz").blocks[1]  # another block's bytes
    data = bytearray(store.get_bytes(f"{FOLDER}/{r2.object}") or b"")
    data[r2.offset : r2.offset + 8] = b"damaged!"
    store.put_bytes(f"{FOLDER}/{r2.object}", bytes(data))
    with pytest.raises(ReadError, match=re.escape("d/x/r2")):
        list(_source(store, tmp_path, selection, cache_directory=tmp_path / "c2").examples(0, 0, 1))


def test_settings_that_cannot_work_are_refused(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {"r1": 5.0})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    with pytest.raises(ValueError, match="twice max_channels"):
        _source(store, tmp_path, selection, limit=SampleTokenLimit(8, 16))


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


def test_a_read_that_times_out_or_is_cut_off_is_tried_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from http.client import IncompleteRead  # noqa: PLC0415

    from botocore.exceptions import ClientError, ResponseStreamingError  # noqa: PLC0415

    from biosignal_data.training import loader  # noqa: PLC0415

    monkeypatch.setattr(loader, "RETRY_WAIT_S", 0.0)

    class Flaky:
        def __init__(self, errors: list[Exception]) -> None:
            self.errors = errors

        def __call__(self) -> bytes:
            if self.errors:
                raise self.errors.pop(0)
            return b"block"

    cut = IncompleteRead(b"x" * 10, 90)
    transient = [
        TimeoutError("no answer"),
        ResponseStreamingError(error=cut),
        cut,
        OSError("reset"),
    ]
    assert loader._read_with_retries(Flaky(list(transient)), "x") == b"block"
    with pytest.raises(TimeoutError):
        loader._read_with_retries(Flaky([TimeoutError("no answer")] * loader.READ_ATTEMPTS), "x")
    missing = ClientError({"Error": {"Code": "NoSuchKey", "Message": "gone"}}, "GetObject")
    calls = Flaky([missing])
    with pytest.raises(ClientError):  # a missing object is not tried again
        loader._read_with_retries(calls, "x")


def test_unimax_shares_the_budget_equally_with_a_cap_on_passes() -> None:
    from biosignal_data.training.mixture import passes, unimax  # noqa: PLC0415

    # a: equal share 20, but 4 passes of 1 = 4; the 16 left over go to b and c (28 each).
    assert unimax({"a": 1.0, "b": 10.0, "c": 100.0}, 60.0, 4) == {"a": 4.0, "b": 28.0, "c": 28.0}
    # Two clusters of 50 tokens each; inside "big", x (10) gets 4 passes, y the rest.
    got = passes({"x": 10.0, "y": 90.0, "z": 5.0}, {"big": ["x", "y"], "small": ["z", "w"]},
                 budget=100.0, max_epochs=4)  # fmt: skip
    assert got["z"] == 4.0  # capped: 20 of its 50
    assert got["x"] == 4.0 and abs(got["y"] - 40.0 / 90.0) < 1e-12  # 80 for "big": 40 + 40
    with pytest.raises(ValueError, match="in no cluster: z"):
        passes({"z": 1.0}, {"big": ["x"]}, budget=10.0, max_epochs=4)


def test_a_mixture_repeats_blocks_and_cuts_each_reading_afresh(
    store: LocalStore, tmp_path: Path
) -> None:
    _index(store, {f"r{i}": 30.0 for i in range(4)})  # 3 blocks each: 12 blocks
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    source = _source(store, tmp_path, selection, passes={"d/x": 2.5})
    order = source.blocks_of(0, 0, 1)
    assert len(order) == 30  # every block twice, half of them a third time
    keys = [(b.recording.recording_id, b.index, b.copy) for b in order]
    assert len(set(keys)) == 30 and {b.copy for b in order} == {0, 1, 2}
    zero, one = (next(b for b in order if b.recording.recording_id == "r0" and b.index == 0
                      and b.copy == c) for c in (0, 1))  # fmt: skip
    starts = [[e.origin.start_s for e in source.cut(b, 0)] for b in (zero, one)]
    assert starts[0] != starts[1]  # a repeated block is cut at other points
    fewer = _source(store, tmp_path, selection, passes={"d/x": 0.5}).blocks_of(0, 0, 1)
    assert len(fewer) == 6


def test_the_pieces_of_an_epoch_read_every_example_once(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {f"r{i}": 30.0 for i in range(5)})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    source = _source(store, tmp_path, selection)

    def key(e: object) -> tuple[str, float]:
        return (e.origin.recording_id, e.origin.start_s)  # type: ignore[attr-defined]

    whole = sorted(key(e) for e in source.examples(0, 0, 1))
    pieces = sorted(key(e) for k in range(3) for e in source.examples(0, 0, 1, k, 3))
    assert whole == pieces and len(whole) > 3


def _global_tokens(e: Example, token_s: float) -> int:
    """Grid cells of `token_s` from the example's start that hold a sample, over its channels
    (biosignal_model.channel_batch.token_cells)."""
    total = 0
    for s in e.signals:
        times = s.start_s + np.arange(s.values.size) / s.sampling_rate
        total += np.unique(np.floor(np.round(times / token_s, 9))).size
    return total


def test_per_group_spans_keep_both_limits_and_every_sample_once(
    store: LocalStore, tmp_path: Path
) -> None:
    expected = _index(store, {"r1": 47.3, "r2": 3.0})
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    limit = SampleAndTokenLimit(max_samples=600, max_tokens=60, token_s=0.05)
    source = _source(store, tmp_path, selection, limit=limit, spans="per_group")
    examples = list(source.examples(epoch=0, part=0, parts=1))
    for e in examples:
        assert sum(s.values.size for s in e.signals) <= 600
        assert _global_tokens(e, 0.05) <= 60
        assert all(0 <= s.start_s < 1 / s.sampling_rate for s in e.signals)
    placed = _place(examples)
    assert set(placed) == set(expected)
    for key, values in expected.items():
        samples = placed[key]
        assert sorted(samples) == list(range(values.size)), key
        np.testing.assert_array_equal(np.array([samples[i] for i in range(values.size)]), values)

    # A group of 10 Hz channels gets its own, longer span: (60 - 1) / 10 tokens a second
    # = 5.9 s for one channel, where any group with a 100 Hz channel has at most
    # (60 - 1) / 20 = 2.95 s; with shared spans every step has a 100 Hz group.
    def seconds(e: Example) -> float:  # to the last sample (< the span)
        return max(s.start_s + (s.values.size - 1) / s.sampling_rate for s in e.signals)

    slow = [seconds(e) for e in examples if {s.sampling_rate for s in e.signals} == {10.0}]
    fast = [seconds(e) for e in examples if 100.0 in {s.sampling_rate for s in e.signals}]
    assert max(fast) <= 2.95 + 1e-9 < max(slow)
    shared = _source(store, tmp_path, selection, limit=limit, spans="shared")
    assert max(seconds(e) for e in shared.examples(0, 0, 1)) <= 2.95 + 1e-9


def test_a_subject_gives_at_most_its_cap_of_hours_to_one_pass(
    store: LocalStore, tmp_path: Path
) -> None:
    _index(store, {f"r{i}": 45.0 for i in range(4)})  # 5 blocks of 10 s (the last 5 s) each
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    cap_h = 25.0 / 3600  # 25 s: two whole blocks, or two and the 5 s one
    source = _source(store, tmp_path, selection, subject_hours_per_epoch=cap_h)

    def per_subject(blocks: list[Block]) -> dict[str, list[tuple[int, int]]]:
        out: dict[str, list[tuple[int, int]]] = {}
        for b in blocks:
            out.setdefault(b.recording.subject_id, []).append((b.index, b.copy))
        return out

    def seconds(index: int) -> float:
        return min(10.0, 45.0 - 10.0 * index)

    first, second = (per_subject(source.blocks_of(epoch, 0, 1)) for epoch in (0, 1))
    assert len(first) == 4
    for chosen in first.values():
        assert 20.0 <= sum(seconds(i) for i, _ in chosen) <= 25.0
    assert first != second  # other random blocks in another epoch
    for e in source.examples(0, 0, 1):  # examples come from the chosen blocks only
        assert (int(e.origin.start_s // 10.0), 0) in first[e.origin.subject_id]
    repeated = _source(store, tmp_path, selection, subject_hours_per_epoch=cap_h,
                       passes={"d/x": 2.0})  # fmt: skip
    for chosen in per_subject(repeated.blocks_of(0, 0, 1)).values():
        for copy in (0, 1):  # the cap holds in each pass
            assert sum(seconds(i) for i, c in chosen if c == copy) <= 25.0
    with pytest.raises(ValueError, match="at least one block"):
        _source(store, tmp_path, selection, subject_hours_per_epoch=5.0 / 3600)


def test_the_mixture_counts_global_tokens_after_the_subject_cap(store: LocalStore) -> None:
    from biosignal_data.training.mixture import dataset_tokens  # noqa: PLC0415

    _index(store, {"r1": 100.0, "r2": 20.0})  # subjects s0 (100 s) and s1 (20 s)
    selection = Selection.load(store, "i1", split="train", signal_types=None)
    # 3 channels at 100 Hz and 2 at 10 Hz; cells of 0.05 s: 20 a second at 100 Hz, and one a
    # sample (10 a second) at 10 Hz
    v4 = SampleAndTokenLimit(max_samples=10**6, max_tokens=10**6, token_s=0.05)
    per_second = 3 * 20 + 2 * 10
    assert dataset_tokens(selection, v4, None) == {"d/x": pytest.approx(per_second * 120.0)}
    capped = dataset_tokens(selection, v4, 50.0 / 3600)  # s0 gives 50 s, s1 its 20 s
    assert capped == {"d/x": pytest.approx(per_second * 70.0)}
    old = dataset_tokens(selection, SampleTokenLimit(max_tokens=10, samples_per_token=16), None)
    assert old == {"d/x": pytest.approx((3 * 100 + 2 * 10) * 120.0 / 16)}
